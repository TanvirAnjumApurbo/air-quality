"""Data audit: coverage, gaps, missingness and distributional profiles.

This is the mandatory gate before any modelling. It answers one question --
*is there enough contiguous, physically plausible data to support the study?* --
and produces the evidence for it rather than asserting it.

Calendar-dependent summaries (diurnal, seasonal) are computed in the local
timezone from ``features.calendar_tz`` so that "hour of day" means what a reader
in Dhaka expects, while the underlying index stays UTC.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.utils import Config


@dataclass
class GapReport:
    """Contiguity summary for one variable.

    Attributes:
        longest_gap_hours: Longest run of consecutive missing hours.
        longest_gap_start: UTC timestamp where that run begins.
        longest_gap_end: UTC timestamp where that run ends.
        n_gaps: Number of distinct missing runs.
        longest_run_hours: Longest run of consecutive observed hours.
        longest_run_start: UTC timestamp where the longest observed run begins.
        longest_run_end: UTC timestamp where the longest observed run ends.
    """

    longest_gap_hours: int
    longest_gap_start: pd.Timestamp | None
    longest_gap_end: pd.Timestamp | None
    n_gaps: int
    longest_run_hours: int
    longest_run_start: pd.Timestamp | None
    longest_run_end: pd.Timestamp | None


def _runs(mask: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp, int]]:
    """Find maximal runs where ``mask`` is True.

    Args:
        mask: Boolean series indexed by timestamp.

    Returns:
        ``(start, end, length)`` for each run, in chronological order.
    """
    if not mask.any():
        return []
    group = (mask != mask.shift()).cumsum()
    out: list[tuple[pd.Timestamp, pd.Timestamp, int]] = []
    for _, block in mask[mask].groupby(group[mask]):
        out.append((block.index[0], block.index[-1], len(block)))
    return out


def gap_report(series: pd.Series) -> GapReport:
    """Summarise contiguity of observed and missing stretches.

    Args:
        series: Values on a regular time grid, missing entries as NaN.

    Returns:
        The contiguity summary.
    """
    missing = series.isna()
    gaps = _runs(missing)
    runs = _runs(~missing)

    longest_gap = max(gaps, key=lambda r: r[2]) if gaps else None
    longest_run = max(runs, key=lambda r: r[2]) if runs else None

    return GapReport(
        longest_gap_hours=longest_gap[2] if longest_gap else 0,
        longest_gap_start=longest_gap[0] if longest_gap else None,
        longest_gap_end=longest_gap[1] if longest_gap else None,
        n_gaps=len(gaps),
        longest_run_hours=longest_run[2] if longest_run else 0,
        longest_run_start=longest_run[0] if longest_run else None,
        longest_run_end=longest_run[1] if longest_run else None,
    )


def gap_size_histogram(series: pd.Series) -> pd.DataFrame:
    """Bucket missing runs by length.

    Distinguishes brief dropouts, which limited forward-fill can bridge safely,
    from multi-day outages, which cannot be bridged and instead fragment the
    series into separate usable stretches.

    Args:
        series: Values on a regular hourly grid.

    Returns:
        One row per length bucket with the run count and total hours lost.
    """
    gaps = _runs(series.isna())
    lengths = np.array([g[2] for g in gaps]) if gaps else np.array([], dtype=int)
    buckets = [
        ("1 h", lambda x: x == 1),
        ("2-3 h", lambda x: (x >= 2) & (x <= 3)),
        ("4-24 h", lambda x: (x > 3) & (x <= 24)),
        ("1-7 d", lambda x: (x > 24) & (x <= 168)),
        ("> 7 d", lambda x: x > 168),
    ]
    rows = []
    for label, predicate in buckets:
        mask = predicate(lengths) if lengths.size else np.array([], dtype=bool)
        rows.append(
            {
                "gap_length": label,
                "n_gaps": int(mask.sum()) if lengths.size else 0,
                "hours_lost": int(lengths[mask].sum()) if lengths.size else 0,
            }
        )
    return pd.DataFrame(rows)


def window_yield(
    series: pd.Series, windows: list[int], horizons: list[int], max_ffill: int
) -> pd.DataFrame:
    """Estimate how many gap-free training windows survive for each configuration.

    A sequence window spanning a data gap is fabricated, so it must be rejected.
    This projects that cost before any modelling code is written, because it --
    not raw coverage -- decides whether a 168-hour input window is affordable.

    Args:
        series: Target on a regular hourly grid.
        windows: Candidate input window lengths in hours.
        horizons: Forecast horizons in hours.
        max_ffill: Limit for backward-looking forward-fill, from
            ``impute.max_ffill_hours``.

    Returns:
        One row per (window, horizon) with the count and percentage retained.
    """
    filled = series.ffill(limit=max_ffill)
    runs = _runs(filled.notna())
    total_slots = len(series)

    rows = []
    for window in windows:
        for horizon in horizons:
            need = window + horizon
            valid = sum(length - need + 1 for _, _, length in runs if length >= need)
            available = max(total_slots - need + 1, 1)
            rows.append(
                {
                    "input_window_h": window,
                    "horizon_h": horizon,
                    "slots_in_span": available,
                    "gap_free_windows": valid,
                    "pct_retained": round(100.0 * valid / available, 1),
                }
            )
    return pd.DataFrame(rows)


def coverage_summary(frame: pd.DataFrame, freq: str = "1h") -> dict[str, Any]:
    """Compute overall coverage of a time-indexed frame.

    Args:
        frame: Frame indexed by a regular timestamp grid.
        freq: Expected sampling frequency.

    Returns:
        Coverage statistics including expected vs present hours.
    """
    index = frame.index
    expected = pd.date_range(index.min(), index.max(), freq=freq, tz=index.tz)
    return {
        "first_utc": str(index.min()),
        "last_utc": str(index.max()),
        "span_days": round((index.max() - index.min()).total_seconds() / 86400.0, 1),
        "span_years": round((index.max() - index.min()).total_seconds() / (86400.0 * 365.25), 2),
        "expected_hours": len(expected),
        "rows_present": len(frame),
        "index_is_complete": bool(len(expected) == len(frame)),
    }


def missingness_by_variable(frame: pd.DataFrame) -> pd.DataFrame:
    """Report missing counts and percentages for every column.

    Args:
        frame: Frame indexed by timestamp.

    Returns:
        One row per variable, sorted by percentage missing, descending.
    """
    total = len(frame)
    rows = [
        {
            "variable": col,
            "observed": int(frame[col].notna().sum()),
            "missing": int(frame[col].isna().sum()),
            "pct_missing": round(100.0 * frame[col].isna().sum() / total, 3) if total else np.nan,
        }
        for col in frame.columns
    ]
    return pd.DataFrame(rows).sort_values("pct_missing", ascending=False).reset_index(drop=True)


def monthly_missingness(series: pd.Series, tz: str) -> pd.DataFrame:
    """Build a year x month matrix of percentage missing.

    Args:
        series: Values on a regular hourly grid, missing entries as NaN.
        tz: Local timezone used to assign observations to calendar months.

    Returns:
        Matrix indexed by year with month columns, values in percent missing.
    """
    local = series.copy()
    local.index = local.index.tz_convert(tz)
    frame = pd.DataFrame(
        {
            "year": local.index.year,
            "month": local.index.month,
            "missing": local.isna().to_numpy(),
        }
    )
    pivot = frame.pivot_table(index="year", columns="month", values="missing", aggfunc="mean")
    return (pivot * 100.0).round(2)


def distribution_stats(series: pd.Series, cfg: Config) -> dict[str, Any]:
    """Summarise the PM2.5 distribution and exceedance of national standards.

    Args:
        series: Observed values in ug/m3.
        cfg: Loaded configuration, for the verified exceedance thresholds.

    Returns:
        Summary statistics and threshold-exceedance fractions.
    """
    values = series.dropna()
    quantiles = [0.01, 0.05, 0.25, 0.5, 0.75, 0.90, 0.95, 0.99]
    stats: dict[str, Any] = {
        "n": int(values.size),
        "mean": round(float(values.mean()), 2),
        "std": round(float(values.std()), 2),
        "min": round(float(values.min()), 2),
        "max": round(float(values.max()), 2),
        "skew": round(float(values.skew()), 3),
        "kurtosis": round(float(values.kurtosis()), 3),
    }
    for q in quantiles:
        stats[f"p{int(q * 100):02d}"] = round(float(values.quantile(q)), 2)

    strat = cfg.get("evaluation.stratify.by_pollution_level")
    thresholds = [float(strat["threshold_ugm3"])] + [
        float(t) for t in strat.get("secondary_thresholds_ugm3", [])
    ]
    stats["exceedance"] = {
        f"above_{t:g}_ugm3_pct": round(100.0 * float((values > t).mean()), 2) for t in thresholds
    }
    return stats


def diurnal_profile(series: pd.Series, tz: str) -> pd.DataFrame:
    """Mean, median and spread of the target by local hour of day.

    Args:
        series: Observed values on an hourly grid.
        tz: Local timezone in which "hour of day" is meaningful.

    Returns:
        One row per hour 0-23.
    """
    local = series.copy()
    local.index = local.index.tz_convert(tz)
    grouped = local.groupby(local.index.hour)
    out = grouped.agg(["count", "mean", "median", "std"])
    out.index.name = "hour_local"
    return out.round(2)


def seasonal_profile(series: pd.Series, tz: str, cfg: Config) -> pd.DataFrame:
    """Mean, median and spread of the target by local calendar month.

    Args:
        series: Observed values on an hourly grid.
        tz: Local timezone.
        cfg: Loaded configuration, for the verified season month boundaries.

    Returns:
        One row per month 1-12, with the season label attached.
    """
    local = series.copy()
    local.index = local.index.tz_convert(tz)
    grouped = local.groupby(local.index.month)
    out = grouped.agg(["count", "mean", "median", "std"])
    out.index.name = "month_local"

    season = cfg.get("features.season")
    monsoon = set(season["monsoon_months"])
    out["season"] = [
        "monsoon (wet)" if m in monsoon else "dry" for m in out.index
    ]
    return out.round(2)


def season_summary(series: pd.Series, tz: str, cfg: Config) -> pd.DataFrame:
    """Aggregate the target by season.

    Args:
        series: Observed values on an hourly grid.
        tz: Local timezone.
        cfg: Loaded configuration, for the verified season month boundaries.

    Returns:
        One row per season with count, mean, median and spread.
    """
    local = series.copy()
    local.index = local.index.tz_convert(tz)
    monsoon = set(cfg.get("features.season.monsoon_months"))
    labels = pd.Series(
        ["monsoon (wet)" if m in monsoon else "dry" for m in local.index.month],
        index=local.index,
        name="season",
    )
    out = local.groupby(labels).agg(["count", "mean", "median", "std", "max"])
    return out.round(2)
