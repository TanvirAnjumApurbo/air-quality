"""UCI Beijing Multi-Site Air-Quality data (repo id 501).

Two roles in this study:

1. **Fallback.** If Dhaka coverage turns out too sparse to model, Beijing becomes
   the primary dataset and Dhaka is demoted to a case study.
2. **Cross-city generalisation.** Even when Dhaka works, running the identical
   pipeline on Beijing is a cheap extra results table that materially strengthens
   the paper.

12 stations, hourly, Mar 2013 - Feb 2017. Source timestamps are given as
``year``/``month``/``day``/``hour`` columns in Beijing local time
(``Asia/Shanghai``, UTC+8, no DST); they are converted to a UTC index here so the
downstream pipeline treats both cities identically.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.utils import Config


class UCIError(RuntimeError):
    """Raised when the UCI dataset cannot be fetched or has an unexpected shape."""


def fetch_uci_beijing(cfg: Config, logger: Any) -> pd.DataFrame:
    """Download the Beijing Multi-Site dataset, caching the raw pull.

    Args:
        cfg: Loaded configuration.
        logger: Logger for progress reporting.

    Returns:
        The raw concatenated dataframe exactly as distributed, with its original
        columns.

    Raises:
        UCIError: If the download fails or returns no features.
    """
    raw_dir = cfg.path_for("data_raw") / "uci"
    raw_dir.mkdir(parents=True, exist_ok=True)
    cache = raw_dir / "beijing_multisite_raw.parquet"

    if cache.exists():
        logger.info("UCI Beijing: cached at %s", cache)
        return pd.read_parquet(cache)

    repo_id = int(cfg.get("data.uci.repo_id", 501))

    # Primary path: the ucimlrepo package. As of this build it does NOT work for
    # id=501 -- the API reports the dataset "exists in the repository, but is not
    # available for import". The static archive zip is still served, so the
    # direct download below is the working path. Both are attempted so the
    # package path resumes automatically if UCI re-enables it.
    df: pd.DataFrame | None = None
    logger.info("UCI Beijing: trying ucimlrepo for repo id %d", repo_id)
    try:
        from ucimlrepo import fetch_ucirepo

        dataset = fetch_ucirepo(id=repo_id)
        features = dataset.data.features
        targets = dataset.data.targets
        if features is not None and len(features):
            df = features if targets is None else pd.concat([features, targets], axis=1)
            logger.info("UCI Beijing: ucimlrepo succeeded")
    except Exception as exc:  # noqa: BLE001 - fall through to the archive download
        logger.warning(
            "UCI Beijing: ucimlrepo unavailable (%s: %s); falling back to the static archive zip",
            type(exc).__name__,
            str(exc).split(".")[0],
        )

    if df is None:
        df = _fetch_from_archive_zip(cfg, raw_dir, logger)

    df.to_parquet(cache, index=False)
    logger.info("UCI Beijing: %d rows, %d columns -> %s", len(df), df.shape[1], cache)
    return df


def _fetch_from_archive_zip(cfg: Config, raw_dir: Path, logger: Any) -> pd.DataFrame:
    """Download and parse the UCI static archive zip for the Beijing dataset.

    The archive contains one CSV per monitoring station; all are concatenated and
    a ``station`` column is preserved so a single site can be selected later.

    Args:
        cfg: Loaded configuration.
        raw_dir: Directory to cache the downloaded zip in.
        logger: Logger for progress reporting.

    Returns:
        Concatenated per-station records.

    Raises:
        UCIError: If the download fails or the archive holds no station CSVs.
    """
    import io
    import zipfile

    import requests

    repo_id = int(cfg.get("data.uci.repo_id", 501))
    url = str(
        cfg.get(
            "data.uci.archive_zip_url",
            f"https://archive.ics.uci.edu/static/public/{repo_id}/"
            "beijing+multi+site+air+quality+data.zip",
        )
    )
    zip_path = raw_dir / f"uci_{repo_id}_archive.zip"

    if not zip_path.exists():
        logger.info("UCI Beijing: downloading %s", url)
        try:
            resp = requests.get(url, timeout=300)
            resp.raise_for_status()
        except requests.RequestException as exc:
            raise UCIError(f"archive download failed for {url}: {exc}") from exc
        zip_path.write_bytes(resp.content)
    logger.info("UCI Beijing: archive at %s (%.1f MB)", zip_path, zip_path.stat().st_size / 1024**2)

    # Archive layout, verified by listing it rather than assumed:
    #   PRSA2017_Data_20130301-20170228.zip   <- the actual data, 12 station CSVs
    #   <uuid>.JPG                            <- a photo
    #   data.csv, test.csv                    <- small unrelated files
    # The nested zip must be preferred: taking outer-level CSVs yields data.csv
    # and test.csv, which parse cleanly to ~1000 rows of the wrong thing.
    station_prefix = "PRSA_Data_"
    frames: list[pd.DataFrame] = []

    with zipfile.ZipFile(zip_path) as outer:
        inner_names = [n for n in outer.namelist() if n.lower().endswith(".zip")]
        if inner_names:
            with outer.open(inner_names[0]) as fh, zipfile.ZipFile(io.BytesIO(fh.read())) as inner:
                members = [
                    n
                    for n in inner.namelist()
                    if n.lower().endswith(".csv") and Path(n).name.startswith(station_prefix)
                ]
                if not members:
                    raise UCIError(
                        f"nested archive {inner_names[0]} holds no {station_prefix}*.csv files: "
                        f"{inner.namelist()[:10]}"
                    )
                for name in sorted(members):
                    with inner.open(name) as csv_fh:
                        frames.append(pd.read_csv(csv_fh))
        else:
            members = [
                n
                for n in outer.namelist()
                if n.lower().endswith(".csv") and Path(n).name.startswith(station_prefix)
            ]
            if not members:
                raise UCIError(
                    f"no nested zip and no {station_prefix}*.csv in {zip_path}: "
                    f"{outer.namelist()[:10]}"
                )
            for name in sorted(members):
                with outer.open(name) as csv_fh:
                    frames.append(pd.read_csv(csv_fh))

    logger.info("UCI Beijing: parsed %d station CSVs from the archive", len(frames))
    combined = pd.concat(frames, ignore_index=True)

    # Sanity-check against the documented shape: 12 stations, hourly,
    # Mar 2013 - Feb 2017, roughly 420k rows.
    expected_stations = 12
    if len(frames) != expected_stations:
        logger.warning(
            "UCI Beijing: expected %d station files, parsed %d", expected_stations, len(frames)
        )
    if len(combined) < 400_000:
        raise UCIError(
            f"UCI Beijing archive parsed to only {len(combined):,} rows; the documented "
            "dataset has roughly 420,000. The wrong members were almost certainly read."
        )
    return combined


def to_hourly_utc(cfg: Config, df: pd.DataFrame, logger: Any) -> pd.DataFrame:
    """Build a UTC-indexed hourly frame for one Beijing station.

    Args:
        cfg: Loaded configuration. ``data.uci.station`` selects the station; when
            null, the station with the most non-missing PM2.5 observations is
            chosen and logged.
        df: Raw dataframe from :func:`fetch_uci_beijing`.
        logger: Logger for the station choice and row accounting.

    Returns:
        Hourly frame indexed by UTC timestamp for the selected station.

    Raises:
        UCIError: If expected columns are absent.
    """
    target = str(cfg.get("data.uci.target_column", "PM2.5"))
    needed = {"year", "month", "day", "hour"}
    missing = needed - set(df.columns)
    if missing:
        raise UCIError(f"UCI frame is missing time columns: {sorted(missing)}")
    if target not in df.columns:
        raise UCIError(f"target column {target!r} not in UCI frame; columns: {list(df.columns)}")

    station_col = "station" if "station" in df.columns else None
    if station_col is None:
        raise UCIError("UCI frame has no 'station' column; cannot select a single site")

    requested = cfg.get("data.uci.station", None)
    if requested:
        station = str(requested)
        if station not in set(df[station_col].unique()):
            raise UCIError(
                f"station {station!r} not present; available: {sorted(df[station_col].unique())}"
            )
    else:
        counts = df.groupby(station_col)[target].count().sort_values(ascending=False)
        station = str(counts.index[0])
        logger.info(
            "UCI Beijing: station not configured; selected %r with %d non-missing %s values "
            "(most complete of %d stations)",
            station,
            int(counts.iloc[0]),
            target,
            len(counts),
        )

    sub = df[df[station_col] == station].copy()

    local_tz = str(cfg.get("data.uci.timezone_local", "Asia/Shanghai"))
    naive = pd.to_datetime(sub[["year", "month", "day", "hour"]])
    # Beijing observes no DST, so localisation is unambiguous.
    sub.index = naive.dt.tz_localize(local_tz).dt.tz_convert("UTC")
    sub.index.name = "datetime_utc"

    drop_cols = [c for c in ["year", "month", "day", "hour", "No", station_col] if c in sub.columns]
    sub = sub.drop(columns=drop_cols).sort_index()

    duplicated = int(sub.index.duplicated().sum())
    if duplicated:
        logger.info("UCI Beijing: dropping %d duplicated timestamps", duplicated)
        sub = sub[~sub.index.duplicated(keep="first")]

    logger.info(
        "UCI Beijing station %s: %d hourly rows, %s to %s",
        station,
        len(sub),
        sub.index.min(),
        sub.index.max(),
    )
    return sub


def write_uci(cfg: Config, frame: pd.DataFrame, logger: Any) -> Path:
    """Persist the Beijing hourly frame to ``data/interim``.

    Args:
        cfg: Loaded configuration.
        frame: Hourly UTC frame for the selected station.
        logger: Logger for the write confirmation.

    Returns:
        Path of the written Parquet file.
    """
    out_dir = cfg.path_for("data_interim")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "uci_beijing_hourly.parquet"
    frame.to_parquet(out)
    logger.info("wrote %s (%d rows, %d cols)", out, len(frame), frame.shape[1])
    return out
