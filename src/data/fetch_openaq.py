"""OpenAQ access: v3 REST discovery and unsigned S3 archive download.

Two access paths, used for different jobs:

* **v3 REST API** (``https://api.openaq.org/v3``) requires a free API key in the
  ``X-API-Key`` header. Used only for *discovery* -- enumerating candidate
  monitors with their identifiers, providers, parameters and date ranges.
* **S3 archive** (``s3://openaq-data-archive``) needs no credentials and is the
  bulk-download path. One gzipped CSV per location per day, partitioned
  ``locationid=<id>/year=<yyyy>/month=<mm>/``.

Location IDs are never hardcoded. Discovery is run first, the candidate table is
reviewed, and the chosen IDs are written into ``config.yaml``.

Response shapes here were verified against the live API rather than taken from
documentation. Notably ``datetimeFirst``/``datetimeLast`` are nested objects with
``utc`` and ``local`` keys, and are ``null`` for some locations -- so the S3
partition listing is the authoritative source of what data actually exists.
"""

from __future__ import annotations

import gzip
import io
import os
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from src.utils import Config

API_KEY_MISSING_MSG = (
    "No OpenAQ API key found. Discovery needs one (bulk S3 download does not).\n"
    "  1. Get a free key at https://explore.openaq.org/ -> account -> API keys\n"
    "  2. Copy .env.example to .env and paste the key into it\n"
    "Do not put the key in .env.example -- that file is tracked by git."
)


class OpenAQError(RuntimeError):
    """Raised when the OpenAQ API returns an unusable response."""


@dataclass
class LocationCandidate:
    """A candidate monitoring location returned by v3 discovery.

    Attributes:
        location_id: OpenAQ location identifier, used as the S3 partition key.
        name: Human-readable site name.
        provider: Data provider name.
        owner: Owning organisation name.
        country: ISO country code.
        latitude: Site latitude in decimal degrees.
        longitude: Site longitude in decimal degrees.
        distance_m: Distance from the configured city centre, metres, if the
            location came from a radius query.
        is_monitor: True for reference monitors, False for low-cost sensors.
        is_mobile: True for mobile platforms.
        timezone: IANA timezone reported by OpenAQ.
        parameters: Measured parameter names, e.g. ``["pm25", "pm10"]``.
        pm25_units: Units reported for the PM2.5 sensor, verbatim from the API.
        datetime_first: First measurement timestamp (UTC, ISO-8601) or None.
        datetime_last: Last measurement timestamp (UTC, ISO-8601) or None.
        s3_years: Years present in the S3 archive, populated by
            :func:`probe_s3_coverage`.
        s3_partitions: Number of ``year=/month=`` partitions found in S3.
    """

    location_id: int
    name: str
    provider: str
    owner: str
    country: str
    latitude: float | None
    longitude: float | None
    distance_m: float | None
    is_monitor: bool
    is_mobile: bool
    timezone: str | None
    parameters: list[str] = field(default_factory=list)
    pm25_units: str | None = None
    datetime_first: str | None = None
    datetime_last: str | None = None
    s3_years: list[int] = field(default_factory=list)
    s3_partitions: int = 0

    @property
    def has_pm25(self) -> bool:
        """True if this location reports a PM2.5 sensor."""
        return "pm25" in self.parameters

    @property
    def s3_year_span(self) -> str:
        """Compact string describing the S3 year range, e.g. ``"2017-2025"``."""
        if not self.s3_years:
            return "-"
        lo, hi = min(self.s3_years), max(self.s3_years)
        return f"{lo}" if lo == hi else f"{lo}-{hi}"


def get_api_key(cfg: Config) -> str:
    """Load the OpenAQ API key from the environment or ``.env``.

    Args:
        cfg: Loaded configuration, for the ``data.openaq.api_key_env`` name.

    Returns:
        The API key.

    Raises:
        OpenAQError: If no key is set.
    """
    from dotenv import load_dotenv

    load_dotenv()
    var = str(cfg.get("data.openaq.api_key_env", "OPENAQ_API_KEY"))
    key = os.environ.get(var, "").strip()
    if not key:
        raise OpenAQError(API_KEY_MISSING_MSG)
    return key


def _request_with_backoff(
    url: str,
    *,
    params: dict[str, Any],
    headers: dict[str, str],
    max_retries: int,
    backoff_base_s: float,
    timeout_s: float,
) -> dict[str, Any]:
    """GET a JSON endpoint, retrying with exponential backoff.

    Retries on 429 and 5xx. Honours a ``Retry-After`` header when present.

    Args:
        url: Endpoint URL.
        params: Query parameters.
        headers: Request headers, including ``X-API-Key``.
        max_retries: Maximum attempts before giving up.
        backoff_base_s: Base for exponential backoff.
        timeout_s: Per-request timeout.

    Returns:
        The decoded JSON body.

    Raises:
        OpenAQError: If every attempt fails.
    """
    last_error = ""
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=timeout_s)
        except requests.RequestException as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        else:
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code == 401:
                raise OpenAQError(f"401 Unauthorized -- the API key was rejected.\n{API_KEY_MISSING_MSG}")
            if resp.status_code not in (429, 500, 502, 503, 504):
                raise OpenAQError(f"HTTP {resp.status_code} from {url}: {resp.text[:400]}")
            retry_after = resp.headers.get("Retry-After")
            last_error = f"HTTP {resp.status_code}"
            if retry_after and retry_after.isdigit():
                time.sleep(min(float(retry_after), 60.0))
                continue
        time.sleep(backoff_base_s ** (attempt + 1))
    raise OpenAQError(f"giving up on {url} after {max_retries} attempts ({last_error})")


def _iter_locations(
    cfg: Config, key: str, params: dict[str, Any], *, page_limit: int = 1000
) -> Iterator[dict[str, Any]]:
    """Yield every location matching ``params``, following pagination.

    Args:
        cfg: Loaded configuration.
        key: OpenAQ API key.
        params: Query parameters other than ``page``/``limit``.
        page_limit: Safety ceiling on pages fetched.

    Yields:
        Raw location dictionaries from the API.
    """
    base = str(cfg.get("data.openaq.api_base")).rstrip("/")
    headers = {"X-API-Key": key}
    req = cfg.get("data.openaq.request", {})
    limit = 1000
    page = 1
    while page <= page_limit:
        body = _request_with_backoff(
            f"{base}/locations",
            params={**params, "limit": limit, "page": page},
            headers=headers,
            max_retries=int(req.get("max_retries", 5)),
            backoff_base_s=float(req.get("backoff_base_s", 1.5)),
            timeout_s=float(req.get("timeout_s", 60)),
        )
        results = body.get("results", [])
        if not results:
            return
        yield from results
        if len(results) < limit:
            return
        page += 1
        time.sleep(float(req.get("sleep_between_calls_s", 0.2)))


def _utc_of(node: Any) -> str | None:
    """Extract the ``utc`` field from an OpenAQ datetime object.

    The API returns ``{"utc": ..., "local": ...}`` rather than a bare string, and
    the whole object is ``null`` for some locations.

    Args:
        node: The ``datetimeFirst``/``datetimeLast`` value from the API.

    Returns:
        The UTC timestamp string, or None.
    """
    if isinstance(node, dict):
        return node.get("utc")
    return node if isinstance(node, str) else None


def _to_candidate(raw: dict[str, Any]) -> LocationCandidate:
    """Convert a raw v3 location payload into a :class:`LocationCandidate`.

    Args:
        raw: One entry from the ``results`` array.

    Returns:
        The parsed candidate.
    """
    sensors = raw.get("sensors") or []
    parameters: list[str] = []
    pm25_units: str | None = None
    for sensor in sensors:
        param = (sensor or {}).get("parameter") or {}
        name = param.get("name")
        if not name:
            continue
        parameters.append(name)
        if name == "pm25":
            pm25_units = param.get("units")

    coords = raw.get("coordinates") or {}
    return LocationCandidate(
        location_id=int(raw["id"]),
        name=str(raw.get("name") or "(unnamed)"),
        provider=str((raw.get("provider") or {}).get("name") or "-"),
        owner=str((raw.get("owner") or {}).get("name") or "-"),
        country=str((raw.get("country") or {}).get("code") or "-"),
        latitude=coords.get("latitude"),
        longitude=coords.get("longitude"),
        distance_m=raw.get("distance"),
        is_monitor=bool(raw.get("isMonitor", False)),
        is_mobile=bool(raw.get("isMobile", False)),
        timezone=raw.get("timezone"),
        parameters=sorted(set(parameters)),
        pm25_units=pm25_units,
        datetime_first=_utc_of(raw.get("datetimeFirst")),
        datetime_last=_utc_of(raw.get("datetimeLast")),
    )


def discover_locations(cfg: Config, key: str) -> list[LocationCandidate]:
    """Enumerate candidate monitors by country and by radius, then merge.

    Both strategies are run because they can disagree: the radius query catches
    sites near Dhaka whose country tagging is unexpected, and the country query
    catches Dhaka-area sites whose coordinates place them just outside the
    radius.

    Args:
        cfg: Loaded configuration.
        key: OpenAQ API key.

    Returns:
        Candidates sorted by distance from the configured city centre, with
        unknown distances last.
    """
    site = cfg.get("data.site")
    lat = float(site["latitude"])
    lon = float(site["longitude"])
    radius = int(site["discovery_radius_m"])
    iso = str(site["country_iso"])

    merged: dict[int, LocationCandidate] = {}

    for raw in _iter_locations(cfg, key, {"iso": iso}):
        cand = _to_candidate(raw)
        merged[cand.location_id] = cand

    for raw in _iter_locations(
        cfg, key, {"coordinates": f"{lat},{lon}", "radius": radius}
    ):
        cand = _to_candidate(raw)
        if cand.location_id in merged:
            # The radius query carries the distance field; the country query does not.
            merged[cand.location_id].distance_m = cand.distance_m
        else:
            merged[cand.location_id] = cand

    return sorted(
        merged.values(),
        key=lambda c: (c.distance_m is None, c.distance_m or 0.0),
    )


# ---------------------------------------------------------------------------
# S3 archive
# ---------------------------------------------------------------------------


def _s3_client(cfg: Config):  # noqa: ANN202 - botocore client type is not public
    """Build an unsigned S3 client for the public OpenAQ archive.

    Returns:
        A boto3 S3 client configured for anonymous access.
    """
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config as BotoConfig

    return boto3.client(
        "s3",
        region_name=str(cfg.get("data.openaq.s3_region", "us-east-1")),
        config=BotoConfig(signature_version=UNSIGNED),
    )


def probe_s3_coverage(cfg: Config, candidates: list[LocationCandidate]) -> None:
    """Populate S3 year/partition coverage on each candidate, in place.

    The API's ``datetimeFirst``/``datetimeLast`` are null for some locations and
    can disagree with what is actually archived, so the S3 partition listing is
    treated as ground truth for what can be downloaded.

    Args:
        cfg: Loaded configuration.
        candidates: Candidates to probe; mutated in place.
    """
    client = _s3_client(cfg)
    bucket = str(cfg.get("data.openaq.s3_bucket"))
    paginator = client.get_paginator("list_objects_v2")

    for cand in candidates:
        prefix = f"records/csv.gz/locationid={cand.location_id}/"
        years: set[int] = set()
        partitions = 0
        try:
            for page in paginator.paginate(
                Bucket=bucket, Prefix=prefix, Delimiter="/", PaginationConfig={"MaxItems": 200}
            ):
                for common in page.get("CommonPrefixes", []):
                    token = common["Prefix"].rstrip("/").rsplit("year=", 1)
                    if len(token) == 2 and token[1].isdigit():
                        years.add(int(token[1]))
                        partitions += 1
        except Exception:  # noqa: BLE001 - a probe failure must not abort discovery
            pass
        cand.s3_years = sorted(years)
        cand.s3_partitions = partitions


def candidates_to_frame(candidates: list[LocationCandidate]) -> pd.DataFrame:
    """Render candidates as a DataFrame for display and CSV export.

    Args:
        candidates: Discovered candidates.

    Returns:
        One row per candidate, ordered as discovered.
    """
    return pd.DataFrame(
        [
            {
                "location_id": c.location_id,
                "name": c.name,
                "provider": c.provider,
                "owner": c.owner,
                "country": c.country,
                "lat": c.latitude,
                "lon": c.longitude,
                "distance_km": None if c.distance_m is None else round(c.distance_m / 1000.0, 2),
                "is_monitor": c.is_monitor,
                "is_mobile": c.is_mobile,
                "has_pm25": c.has_pm25,
                "pm25_units": c.pm25_units,
                "n_parameters": len(c.parameters),
                "parameters": ",".join(c.parameters),
                "api_first_utc": c.datetime_first,
                "api_last_utc": c.datetime_last,
                "s3_years": c.s3_year_span,
                "s3_n_year_partitions": c.s3_partitions,
            }
            for c in candidates
        ]
    )


def iter_s3_keys(cfg: Config, location_id: int, year: int, month: int) -> Iterator[str]:
    """Yield S3 object keys for one location-month partition.

    Args:
        cfg: Loaded configuration.
        location_id: OpenAQ location identifier.
        year: Four-digit year.
        month: Month number, 1-12.

    Yields:
        Object keys under the partition prefix.
    """
    client = _s3_client(cfg)
    bucket = str(cfg.get("data.openaq.s3_bucket"))
    template = str(cfg.get("data.openaq.s3_prefix_template"))
    prefix = template.format(location_id=location_id, year=year, month=month)
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            yield obj["Key"]


def download_month(
    cfg: Config, location_id: int, year: int, month: int, dest_dir: Path
) -> tuple[int, int]:
    """Download and concatenate one location-month partition to a Parquet file.

    Skips work entirely if the destination already exists, so an interrupted
    fetch resumes cheaply.

    Args:
        cfg: Loaded configuration.
        location_id: OpenAQ location identifier.
        year: Four-digit year.
        month: Month number, 1-12.
        dest_dir: Directory to write ``loc<id>_<yyyy>-<mm>.parquet`` into.

    Returns:
        ``(n_objects, n_rows)`` retrieved.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"loc{location_id}_{year}-{month:02d}.parquet"
    if dest.exists():
        try:
            return 0, len(pd.read_parquet(dest))
        except Exception:  # noqa: BLE001 - a corrupt cache file is re-fetched
            dest.unlink(missing_ok=True)

    client = _s3_client(cfg)
    bucket = str(cfg.get("data.openaq.s3_bucket"))
    frames: list[pd.DataFrame] = []
    n_objects = 0
    for key in iter_s3_keys(cfg, location_id, year, month):
        body = client.get_object(Bucket=bucket, Key=key)["Body"].read()
        with gzip.GzipFile(fileobj=io.BytesIO(body)) as fh:
            frames.append(pd.read_csv(fh))
        n_objects += 1

    if not frames:
        return 0, 0
    df = pd.concat(frames, ignore_index=True)
    df.to_parquet(dest, index=False)
    return n_objects, len(df)


def list_partitions(cfg: Config, location_id: int) -> list[tuple[int, int]]:
    """List every ``(year, month)`` partition present for a location.

    Args:
        cfg: Loaded configuration.
        location_id: OpenAQ location identifier.

    Returns:
        Sorted ``(year, month)`` pairs.
    """
    client = _s3_client(cfg)
    bucket = str(cfg.get("data.openaq.s3_bucket"))
    paginator = client.get_paginator("list_objects_v2")

    partitions: set[tuple[int, int]] = set()
    year_prefix = f"records/csv.gz/locationid={location_id}/"
    for year_page in paginator.paginate(Bucket=bucket, Prefix=year_prefix, Delimiter="/"):
        for year_common in year_page.get("CommonPrefixes", []):
            year_token = year_common["Prefix"].rstrip("/").rsplit("year=", 1)[-1]
            if not year_token.isdigit():
                continue
            year = int(year_token)
            for month_page in paginator.paginate(
                Bucket=bucket, Prefix=year_common["Prefix"], Delimiter="/"
            ):
                for month_common in month_page.get("CommonPrefixes", []):
                    month_token = month_common["Prefix"].rstrip("/").rsplit("month=", 1)[-1]
                    if month_token.isdigit():
                        partitions.add((year, int(month_token)))
    return sorted(partitions)


def download_location(cfg: Config, location_id: int, logger: Any) -> pd.DataFrame:
    """Download every archived record for one location.

    Daily objects are fetched concurrently and cached per location-month, so an
    interrupted run resumes without re-downloading completed months.

    Args:
        cfg: Loaded configuration.
        location_id: OpenAQ location identifier.
        logger: Logger for progress reporting.

    Returns:
        Concatenated raw records, exactly as archived.
    """
    from concurrent.futures import ThreadPoolExecutor

    raw_dir = cfg.path_for("data_raw") / "openaq"
    raw_dir.mkdir(parents=True, exist_ok=True)

    partitions = list_partitions(cfg, location_id)
    logger.info("location %d: %d month partitions in S3", location_id, len(partitions))

    def _one(part: tuple[int, int]) -> pd.DataFrame:
        year, month = part
        dest = raw_dir / f"loc{location_id}_{year}-{month:02d}.parquet"
        if dest.exists():
            return pd.read_parquet(dest)
        download_month(cfg, location_id, year, month, raw_dir)
        return pd.read_parquet(dest) if dest.exists() else pd.DataFrame()

    frames: list[pd.DataFrame] = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        for i, frame in enumerate(pool.map(_one, partitions), start=1):
            if len(frame):
                frames.append(frame)
            if i % 12 == 0 or i == len(partitions):
                logger.info("location %d: %d/%d months", location_id, i, len(partitions))

    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    logger.info("location %d: %d raw records", location_id, len(combined))
    return combined


def load_openaq_raw(cfg: Config, logger: Any) -> pd.DataFrame:
    """Download and concatenate every configured OpenAQ location.

    Args:
        cfg: Loaded configuration.
        logger: Logger for progress and overlap accounting.

    Returns:
        Raw records for all configured locations with a parsed UTC ``datetime``.

    Raises:
        OpenAQError: If no location IDs are configured, or nothing was retrieved.
    """
    location_ids = list(cfg.get("data.openaq.location_ids", []))
    if not location_ids:
        raise OpenAQError(
            "data.openaq.location_ids is empty. Run scripts/01_discover_openaq.py, "
            "review the candidate table, and set the chosen IDs in config.yaml."
        )

    frames = []
    for location_id in location_ids:
        frame = download_location(cfg, int(location_id), logger)
        if len(frame):
            frames.append(frame)

    if not frames:
        raise OpenAQError(f"no records retrieved for locations {location_ids}")

    combined = pd.concat(frames, ignore_index=True)
    # The archive stores ISO-8601 with an explicit local offset (+06:00 for
    # Dhaka), so utc=True yields correct instants without assuming a timezone.
    combined["datetime"] = pd.to_datetime(combined["datetime"], utc=True, format="ISO8601")
    return combined.sort_values("datetime").reset_index(drop=True)


def apply_qc(cfg: Config, df: pd.DataFrame, logger: Any) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Apply the configured quality filters, logging the rows each one removes.

    Every filtering decision is recorded so the data audit can report exactly how
    many observations each rule discarded.

    Args:
        cfg: Loaded configuration.
        df: Raw records from :func:`load_openaq_raw`.
        logger: Logger for per-filter accounting.

    Returns:
        ``(filtered, ledger)`` where ``ledger`` has one entry per filter with the
        rule name, rows removed, and rows remaining.

    Raises:
        OpenAQError: If units cannot be reconciled to the configured target.
    """
    qc = cfg.get("qc.pm25")
    target_units = str(cfg.get("data.openaq.target_units", "ug/m3"))
    parameter = str(cfg.get("data.openaq.parameter", "pm25"))
    ledger: list[dict[str, Any]] = []
    n = len(df)

    def _record(rule: str, before: int, after: int, note: str = "") -> None:
        ledger.append(
            {"rule": rule, "removed": before - after, "remaining": after, "note": note}
        )
        logger.info("QC %-28s removed %7d  remaining %7d %s", rule, before - after, after, note)

    ledger.append({"rule": "raw records", "removed": 0, "remaining": n, "note": ""})
    logger.info("QC %-28s %26d", "raw records", n)

    before = len(df)
    df = df[df["parameter"] == parameter]
    _record(f"parameter != {parameter}", before, len(df))

    # Units: µg/m³ and ug/m3 are the same unit written differently. Anything that
    # is not micrograms per cubic metre is rejected rather than guessed at.
    observed = sorted(set(df["units"].dropna().astype(str).unique()))
    micrograms = {"µg/m³", "ug/m3", "ugm3", "µg/m3", "µg/m³"}
    unknown = [u for u in observed if u not in micrograms]
    if unknown:
        raise OpenAQError(
            f"unhandled PM2.5 units {unknown} (target {target_units!r}). "
            "Add an explicit conversion rather than assuming a scale factor."
        )
    logger.info("QC units observed: %s -> all micrograms per cubic metre", observed)

    before = len(df)
    df = df[df["value"].notna()]
    _record("value is NaN", before, len(df))

    if bool(qc.get("drop_negative", True)):
        before = len(df)
        df = df[df["value"] >= 0]
        _record("negative value", before, len(df))

    if bool(qc.get("drop_exact_zero", True)):
        before = len(df)
        df = df[df["value"] != 0]
        _record("exact zero (sensor fault)", before, len(df))

    cap = float(qc.get("sanity_cap_ugm3", 1000.0))
    before = len(df)
    df = df[df["value"] <= cap]
    _record(f"above sanity cap {cap:g}", before, len(df))

    max_repeats = int(qc.get("flatline_max_repeats", 12))
    if max_repeats > 0 and len(df):
        df = df.sort_values("datetime")
        run_id = (df["value"] != df["value"].shift()).cumsum()
        run_len = run_id.map(run_id.value_counts())
        before = len(df)
        df = df[run_len <= max_repeats]
        _record(
            f"flatline run > {max_repeats}",
            before,
            len(df),
            "(consecutive identical values = stuck sensor)",
        )

    before = len(df)
    df = df.drop_duplicates(subset=["datetime"], keep="first")
    _record(
        "duplicate timestamps",
        before,
        len(df),
        "(overlap between concatenated location IDs)",
    )

    return df.reset_index(drop=True), ledger


def to_hourly(cfg: Config, df: pd.DataFrame, logger: Any) -> pd.DataFrame:
    """Resample quality-controlled records onto a regular hourly UTC grid.

    Gaps are left as NaN rather than filled: contiguity matters for gap-aware
    windowing later, so missing hours must remain visible.

    Args:
        cfg: Loaded configuration.
        df: Quality-controlled records.
        logger: Logger for coverage reporting.

    Returns:
        Frame indexed by hourly UTC timestamp with a single ``pm25`` column.
    """
    freq = str(cfg.get("qc.resample.freq", "1h"))
    how = str(cfg.get("qc.resample.aggregation", "mean"))

    series = df.set_index("datetime")["value"].sort_index()
    hourly = series.resample(freq).agg(how)

    full = pd.date_range(hourly.index.min(), hourly.index.max(), freq=freq, tz="UTC")
    hourly = hourly.reindex(full)
    hourly.index.name = "datetime_utc"

    n_obs = int(hourly.notna().sum())
    n_total = len(hourly)
    logger.info(
        "hourly grid: %d of %d hours observed (%.1f%%), %s to %s",
        n_obs,
        n_total,
        100.0 * n_obs / n_total,
        hourly.index.min(),
        hourly.index.max(),
    )
    return hourly.to_frame(name="pm25")


def write_openaq(cfg: Config, frame: pd.DataFrame, logger: Any) -> Path:
    """Persist the hourly PM2.5 series to ``data/interim``.

    Args:
        cfg: Loaded configuration.
        frame: Hourly UTC PM2.5 frame.
        logger: Logger for the write confirmation.

    Returns:
        Path of the written Parquet file.
    """
    out_dir = cfg.path_for("data_interim")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "openaq_pm25_hourly.parquet"
    frame.to_parquet(out)
    logger.info("wrote %s (%d rows)", out, len(frame))
    return out
