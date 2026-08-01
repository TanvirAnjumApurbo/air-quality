"""NASA POWER hourly meteorology for a single point.

Free, no key, no registration. Requests are chunked by calendar year and each
chunk is cached to ``data/raw`` so an interrupted fetch resumes cheaply.

Two properties of this API were verified against live responses rather than
assumed, because getting either wrong corrupts the fused dataset silently:

* **Time standard.** The hourly endpoint defaults to ``LST`` (Local Solar Time),
  *not* UTC. ``time-standard=UTC`` is sent explicitly so that timestamps align
  with the OpenAQ target series. Confirmed for Dhaka on 2024-01-01: under LST the
  daily T2M minimum falls at 06h and the maximum at 13h; under UTC they fall at
  23h and 07h, i.e. 05:00 and 13:00 Asia/Dhaka.
* **Units.** Taken verbatim from the response ``parameters`` block. Notably
  ``PRECTOTCORR`` is reported in mm/day and ``ALLSKY_SFC_SW_DWN`` in Wh/m^2,
  which differ from the mm/hour and W/m^2 often assumed. Values are stored as
  returned; nothing is rescaled on an assumption.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from src.utils import Config


class PowerError(RuntimeError):
    """Raised when the NASA POWER API returns an unusable response."""


def _request_year(cfg: Config, year: int, start: str, end: str) -> dict[str, Any]:
    """Fetch one calendar-year chunk from the POWER hourly point endpoint.

    Args:
        cfg: Loaded configuration.
        year: Year being fetched, used only for error messages.
        start: Inclusive start date, ``YYYYMMDD``.
        end: Inclusive end date, ``YYYYMMDD``.

    Returns:
        The decoded JSON body.

    Raises:
        PowerError: If every retry fails.
    """
    site = cfg.get("data.site")
    power = cfg.get("data.power")
    req = power.get("request", {})
    params = {
        "parameters": ",".join(power["parameters"]),
        "community": power["community"],
        "longitude": site["longitude"],
        "latitude": site["latitude"],
        "start": start,
        "end": end,
        "format": "JSON",
        "time-standard": power.get("time_standard", "UTC"),
    }

    max_retries = int(req.get("max_retries", 5))
    last_error = ""
    for attempt in range(max_retries):
        try:
            resp = requests.get(
                power["endpoint"], params=params, timeout=float(req.get("timeout_s", 180))
            )
        except requests.RequestException as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        else:
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code < 500 and resp.status_code != 429:
                raise PowerError(f"HTTP {resp.status_code} for {year}: {resp.text[:400]}")
            last_error = f"HTTP {resp.status_code}"
        time.sleep(float(req.get("backoff_base_s", 2.0)) ** (attempt + 1))
    raise PowerError(f"giving up on POWER year {year} after {max_retries} attempts ({last_error})")


def _validate_response(cfg: Config, body: dict[str, Any]) -> dict[str, str]:
    """Check the response contains every requested parameter and report its units.

    Args:
        cfg: Loaded configuration.
        body: Decoded POWER JSON response.

    Returns:
        Mapping of parameter name to the units string the API reported.

    Raises:
        PowerError: If a requested parameter is absent, or the API did not honour
            the requested time standard.
    """
    requested = list(cfg.get("data.power.parameters"))
    returned = body.get("properties", {}).get("parameter", {})
    missing = [p for p in requested if p not in returned]
    if missing:
        raise PowerError(
            f"POWER did not return requested parameters: {missing}. "
            f"Returned: {sorted(returned)}. Substitute a documented equivalent "
            "rather than silently dropping the driver."
        )

    wanted_ts = str(cfg.get("data.power.time_standard", "UTC")).upper()
    actual_ts = str(body.get("header", {}).get("time_standard", "")).upper()
    if actual_ts and actual_ts != wanted_ts:
        raise PowerError(
            f"POWER returned time_standard={actual_ts!r} but {wanted_ts!r} was requested. "
            "Refusing to continue: a time-standard mismatch offsets every "
            "meteorological driver relative to the target series."
        )

    return {name: str(meta.get("units", "")) for name, meta in (body.get("parameters") or {}).items()}


def _body_to_frame(cfg: Config, body: dict[str, Any]) -> pd.DataFrame:
    """Convert a POWER JSON body into a tidy hourly DataFrame.

    Timestamps arrive as ``YYYYMMDDHH`` strings and are parsed to a
    timezone-aware UTC index. Fill values are converted to NaN.

    Args:
        cfg: Loaded configuration.
        body: Decoded POWER JSON response.

    Returns:
        DataFrame indexed by UTC timestamp with one column per parameter.
    """
    params = body["properties"]["parameter"]
    frame = pd.DataFrame(params)
    frame.index = pd.to_datetime(frame.index, format="%Y%m%d%H", utc=True)
    frame.index.name = "datetime_utc"

    fill = float(cfg.get("data.power.fill_value", -999.0))
    # Compare with tolerance: the sentinel round-trips through JSON as a float.
    frame = frame.mask((frame - fill).abs() < 1e-6)
    return frame.sort_index()


def fetch_power(cfg: Config, logger: Any) -> tuple[pd.DataFrame, dict[str, str]]:
    """Fetch the full configured date range, one calendar year per request.

    Each year is cached as JSON under ``data/raw/power``; a cached year is reused
    and not re-requested.

    Args:
        cfg: Loaded configuration.
        logger: Logger for progress and fill-value accounting.

    Returns:
        ``(frame, units)`` where ``frame`` is hourly UTC meteorology and ``units``
        maps parameter name to the units string reported by the API.
    """
    raw_dir = cfg.path_for("data_raw") / "power"
    raw_dir.mkdir(parents=True, exist_ok=True)

    start = pd.Timestamp(str(cfg.get("data.range.start")))
    end = pd.Timestamp(str(cfg.get("data.range.end")))
    sleep_s = float(cfg.get("data.power.request.sleep_between_calls_s", 3.0))

    frames: list[pd.DataFrame] = []
    units: dict[str, str] = {}

    for year in range(start.year, end.year + 1):
        y_start = max(start, pd.Timestamp(f"{year}-01-01"))
        y_end = min(end, pd.Timestamp(f"{year}-12-31"))
        if y_start > y_end:
            continue

        cache = raw_dir / f"power_{year}.json"
        if cache.exists():
            body = json.loads(cache.read_text(encoding="utf-8"))
            logger.info("POWER %d: cached", year)
        else:
            logger.info(
                "POWER %d: requesting %s to %s",
                year,
                y_start.date(),
                y_end.date(),
            )
            body = _request_year(
                cfg, year, y_start.strftime("%Y%m%d"), y_end.strftime("%Y%m%d")
            )
            cache.write_text(json.dumps(body), encoding="utf-8")
            time.sleep(sleep_s)

        units.update(_validate_response(cfg, body))
        frame = _body_to_frame(cfg, body)
        n_fill = int(frame.isna().sum().sum())
        if n_fill:
            logger.info(
                "POWER %d: %d fill values (-999) converted to NaN across %d cells",
                year,
                n_fill,
                frame.size,
            )
        frames.append(frame)

    if not frames:
        raise PowerError("no POWER data fetched for the configured range")

    combined = pd.concat(frames).sort_index()
    duplicated = int(combined.index.duplicated().sum())
    if duplicated:
        logger.info("POWER: dropping %d duplicated timestamps at year boundaries", duplicated)
        combined = combined[~combined.index.duplicated(keep="first")]

    logger.info(
        "POWER: %d hourly rows, %s to %s, time standard %s",
        len(combined),
        combined.index.min(),
        combined.index.max(),
        cfg.get("data.power.time_standard", "UTC"),
    )
    return combined, units


def write_power(cfg: Config, frame: pd.DataFrame, units: dict[str, str], logger: Any) -> Path:
    """Persist the meteorology frame and its units to ``data/interim``.

    Args:
        cfg: Loaded configuration.
        frame: Hourly UTC meteorology.
        units: Parameter-to-units mapping from the API.
        logger: Logger for the write confirmation.

    Returns:
        Path of the written Parquet file.
    """
    out_dir = cfg.path_for("data_interim")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "power_hourly.parquet"
    frame.to_parquet(out)
    (out_dir / "power_units.json").write_text(json.dumps(units, indent=2), encoding="utf-8")
    logger.info("wrote %s (%d rows, %d cols)", out, len(frame), frame.shape[1])
    return out
