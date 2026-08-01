"""Reshape the UCI Beijing record into the schema the Dhaka pipeline expects.

The point of the cross-city run is to test the *method*, not to build the best
possible Beijing model, so the feature set is deliberately restricted to match
Dhaka: PM2.5 history plus meteorology. Beijing also carries co-located PM10,
SO2, NO2, CO and O3, and including them would make it a structurally different
and considerably easier problem — a difference in the data, not in the method,
which is exactly what a generalisation check must avoid.

Variable mapping, chosen so the two cities present the same columns to the same
feature builder:

===========  ==========  ==========================================
Beijing      Dhaka role  Note
===========  ==========  ==========================================
TEMP         T2M         degrees C, direct
DEWP         T2MDEW      degrees C, direct
PRES         PS          hPa converted to kPa to match POWER's units
WSPM         WS10M       m/s, direct
wd           WD10M       16-point compass string converted to degrees
RAIN         PRECTOTCORR mm; POWER reports mm/day, this is per hour
(derived)    RH2M        computed from TEMP and DEWP (Magnus)
(none)       —           no shortwave irradiance equivalent exists
===========  ==========  ==========================================

Beijing therefore has seven meteorological drivers against Dhaka's eight. That
asymmetry is real and is reported rather than papered over.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from src.utils import Config

# 16-point compass to degrees clockwise from north, the direction the wind blows
# FROM -- the same meteorological convention NASA POWER uses for WD10M.
COMPASS_TO_DEGREES = {
    "N": 0.0, "NNE": 22.5, "NE": 45.0, "ENE": 67.5,
    "E": 90.0, "ESE": 112.5, "SE": 135.0, "SSE": 157.5,
    "S": 180.0, "SSW": 202.5, "SW": 225.0, "WSW": 247.5,
    "W": 270.0, "WNW": 292.5, "NW": 315.0, "NNW": 337.5,
}

# Magnus-Tetens coefficients over water, as used for dew-point/RH conversion.
MAGNUS_A = 17.625
MAGNUS_B = 243.04


def relative_humidity_from_dewpoint(temp_c: np.ndarray, dewpoint_c: np.ndarray) -> np.ndarray:
    """Derive relative humidity from temperature and dew point.

    Uses the Magnus-Tetens approximation. Beijing does not report RH directly,
    but Dhaka's POWER extract does, so deriving it keeps the two feature sets
    aligned rather than leaving a hole in one city.

    Args:
        temp_c: Air temperature in degrees Celsius.
        dewpoint_c: Dew-point temperature in degrees Celsius.

    Returns:
        Relative humidity in percent, clipped to a physical 0-100.
    """
    numerator = MAGNUS_A * dewpoint_c / (MAGNUS_B + dewpoint_c)
    denominator = MAGNUS_A * temp_c / (MAGNUS_B + temp_c)
    return np.clip(100.0 * np.exp(numerator - denominator), 0.0, 100.0)


def prepare(cfg: Config, raw: pd.DataFrame, logger: Any) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    """Split the Beijing frame into target and meteorology, in Dhaka's schema.

    Args:
        cfg: Loaded Beijing configuration.
        raw: Hourly UTC frame for one Beijing station.
        logger: Logger for the QC ledger.

    Returns:
        ``(pm25_frame, met_frame, qc_ledger)``.
    """
    qc = cfg.get("qc.pm25")
    ledger: list[dict[str, Any]] = []

    def _record(rule: str, before: int, after: int, note: str = "") -> None:
        ledger.append({"rule": rule, "removed": before - after, "remaining": after, "note": note})
        logger.info("QC %-28s removed %7d  remaining %7d %s", rule, before - after, after, note)

    target = raw["PM2.5"].astype(float)
    n_start = int(target.notna().sum())
    ledger.append({"rule": "raw records", "removed": 0, "remaining": n_start, "note": ""})
    logger.info("QC %-28s %26d", "raw records", n_start)

    # Same filters, same order, same thresholds as the Dhaka path.
    before = int(target.notna().sum())
    if bool(qc.get("drop_negative", True)):
        target = target.where(target >= 0)
        _record("negative value", before, int(target.notna().sum()))

    before = int(target.notna().sum())
    if bool(qc.get("drop_exact_zero", True)):
        target = target.where(target != 0)
        _record("exact zero (sensor fault)", before, int(target.notna().sum()))

    before = int(target.notna().sum())
    cap = float(qc.get("sanity_cap_ugm3", 1000.0))
    target = target.where(target <= cap)
    _record(f"above sanity cap {cap:g}", before, int(target.notna().sum()))

    max_repeats = int(qc.get("flatline_max_repeats", 12))
    if max_repeats > 0:
        before = int(target.notna().sum())
        observed = target.dropna()
        run_id = (observed != observed.shift()).cumsum()
        run_len = run_id.map(run_id.value_counts())
        stuck = observed.index[run_len > max_repeats]
        target = target.copy()
        target.loc[stuck] = np.nan
        _record(
            f"flatline run > {max_repeats}",
            before,
            int(target.notna().sum()),
            "(consecutive identical values = stuck sensor)",
        )

    pm25 = target.to_frame(name="pm25")
    pm25.index.name = "datetime_utc"

    # ---- meteorology, renamed and unit-matched to the Dhaka schema ---------
    met = pd.DataFrame(index=raw.index)
    met.index.name = "datetime_utc"
    met["T2M"] = raw["TEMP"].astype(float)
    met["T2MDEW"] = raw["DEWP"].astype(float)
    met["RH2M"] = relative_humidity_from_dewpoint(
        raw["TEMP"].to_numpy(dtype=float), raw["DEWP"].to_numpy(dtype=float)
    )
    met["WS10M"] = raw["WSPM"].astype(float)
    met["WD10M"] = raw["wd"].map(COMPASS_TO_DEGREES).astype(float)
    # hPa -> kPa so the column carries the same units as NASA POWER's PS.
    met["PS"] = raw["PRES"].astype(float) / 10.0
    met["PRECTOTCORR"] = raw["RAIN"].astype(float)

    unmapped = set(raw["wd"].dropna().unique()) - set(COMPASS_TO_DEGREES)
    if unmapped:
        logger.warning("unmapped wind-direction codes ignored: %s", sorted(unmapped))

    logger.info(
        "meteorology: %d drivers (%s). No shortwave-irradiance equivalent exists in "
        "this record, so Beijing carries one driver fewer than Dhaka.",
        met.shape[1],
        ", ".join(met.columns),
    )
    logger.info(
        "PM2.5 observed %d of %d hours (%.1f%%)",
        int(pm25["pm25"].notna().sum()),
        len(pm25),
        100.0 * pm25["pm25"].notna().mean(),
    )
    return pm25, met, ledger


def write_prepared(
    cfg: Config, pm25: pd.DataFrame, met: pd.DataFrame, ledger: list[dict[str, Any]], logger: Any
) -> None:
    """Write the prepared frames where the shared pipeline stages expect them.

    Args:
        cfg: Loaded Beijing configuration.
        pm25: Hourly target frame.
        met: Hourly meteorology frame.
        ledger: QC accounting.
        logger: Logger.
    """
    out_dir = cfg.path_for("data_interim")
    out_dir.mkdir(parents=True, exist_ok=True)

    pm25.to_parquet(out_dir / str(cfg.get("data.files.target")))
    met.to_parquet(out_dir / str(cfg.get("data.files.meteorology")))
    (out_dir / str(cfg.get("data.files.qc_ledger"))).write_text(
        json.dumps(ledger, indent=2), encoding="utf-8"
    )

    units = {
        "T2M": "C", "T2MDEW": "C", "RH2M": "% (derived from TEMP and DEWP, Magnus)",
        "WS10M": "m/s", "WD10M": "Degrees (converted from 16-point compass)",
        "PS": "kPa (converted from hPa)", "PRECTOTCORR": "mm/hour",
    }
    (out_dir / str(cfg.get("data.files.met_units"))).write_text(
        json.dumps(units, indent=2), encoding="utf-8"
    )
    logger.info("wrote prepared Beijing frames to %s", out_dir)
