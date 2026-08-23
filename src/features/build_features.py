"""Feature construction with the leakage rules enforced structurally.

The design principle throughout: a feature row stamped at time *t* may contain
**only** information observable at or before *t*. Targets are the sole
forward-looking quantity, and they are produced by an explicit negative shift so
that the direction is visible in the code rather than implied.

Contiguity is tracked explicitly. After a limited, backward-looking forward-fill,
the series decomposes into maximal gap-free *runs*; every lag, rolling window and
target is then required to lie inside a single run. A window straddling a run
boundary would be fabricated data, so it is rejected and counted rather than
quietly used.

Calendar features are derived in ``features.calendar_tz`` (Asia/Dhaka) while the
index stays UTC, so "hour of day" is physically meaningful without the merge ever
leaving UTC.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.utils import Config, ConfigError


@dataclass
class FeatureBuildReport:
    """Accounting for what feature construction produced and discarded.

    Attributes:
        n_rows_grid: Rows on the raw hourly grid.
        n_observed_target: Hours with a genuinely observed PM2.5 value.
        n_ffilled: Hours whose PM2.5 came from limited forward-fill.
        n_runs: Number of maximal gap-free runs after forward-fill.
        longest_run_h: Length of the longest run, hours.
        max_lag_h: Longest backward dependency in the feature set.
        history_floor_h: Hours of unbroken history a row must carry to be
            scored. Equals ``max_lag_h`` unless ``features.history_floor_h``
            decouples the two for the lookback frontier's control arm.
        lookback_cap_h: The configured ``features.lookback_h``, or None when
            the lists are used as written.
        n_features: Number of predictor columns produced.
        rows_valid_per_horizon: Usable supervised rows per horizon.
        rows_rejected_per_horizon: Rows dropped per horizon, by reason.
    """

    n_rows_grid: int
    n_observed_target: int
    n_ffilled: int
    n_runs: int
    longest_run_h: int
    max_lag_h: int
    history_floor_h: int
    lookback_cap_h: int | None
    n_features: int
    rows_valid_per_horizon: dict[int, int] = field(default_factory=dict)
    rows_rejected_per_horizon: dict[int, dict[str, int]] = field(default_factory=dict)


def assign_runs(present: pd.Series) -> pd.Series:
    """Label maximal contiguous stretches where data is present.

    Args:
        present: Boolean series on a regular grid; True where usable.

    Returns:
        Integer run identifier per timestamp; -1 where ``present`` is False.
    """
    run_id = (~present).cumsum()
    return run_id.where(present, -1).astype("int64")


def position_in_run(run_id: pd.Series) -> pd.Series:
    """Index of each timestamp within its run, zero-based.

    Args:
        run_id: Run labels from :func:`assign_runs`.

    Returns:
        Position within the run; -1 outside any run.
    """
    valid = run_id >= 0
    pos = valid.groupby(run_id).cumsum() - 1
    return pos.where(valid, -1).astype("int64")


def run_lengths(run_id: pd.Series) -> pd.Series:
    """Length of the run each timestamp belongs to.

    Args:
        run_id: Run labels from :func:`assign_runs`.

    Returns:
        Run length in hours; 0 outside any run.
    """
    counts = run_id[run_id >= 0].value_counts()
    return run_id.map(counts).fillna(0).astype("int64")


def add_wind_components(frame: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Decompose wind speed and direction into u and v components.

    Wind direction from NASA POWER follows the meteorological convention: degrees
    clockwise from north indicating the direction the wind blows *from*. The
    standard decomposition therefore carries a negative sign, so that ``u`` is
    positive for a westerly (eastward-blowing) wind.

    Args:
        frame: Frame containing the configured speed and direction columns.
        cfg: Loaded configuration (``features.wind_uv_from``).

    Returns:
        The frame with ``wind_u`` and ``wind_v`` added.
    """
    spec = cfg.get("features.wind_uv_from")
    speed_col, dir_col = str(spec["speed"]), str(spec["direction"])
    if speed_col not in frame.columns or dir_col not in frame.columns:
        return frame

    radians = np.deg2rad(frame[dir_col].to_numpy())
    speed = frame[speed_col].to_numpy()
    out = frame.copy()
    out["wind_u"] = -speed * np.sin(radians)
    out["wind_v"] = -speed * np.cos(radians)
    return out


def add_calendar_features(frame: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Add cyclical calendar encodings and the season indicator.

    Derived in the configured local timezone so the diurnal cycle is physical,
    while the index itself remains UTC.

    Args:
        frame: UTC-indexed frame.
        cfg: Loaded configuration (``features.calendar_tz``, ``features.cyclical``,
            ``features.season``).

    Returns:
        The frame with calendar columns added.
    """
    tz = str(cfg.get("features.calendar_tz"))
    local = frame.index.tz_convert(tz)
    out = frame.copy()

    cyclical = cfg.get("features.cyclical")
    sources = {
        "hour_of_day": local.hour.to_numpy().astype(float),
        "day_of_week": local.dayofweek.to_numpy().astype(float),
        "day_of_year": local.dayofyear.to_numpy().astype(float),
    }
    for name, spec in cyclical.items():
        if name not in sources:
            continue
        period = float(spec["period"])
        angle = 2.0 * np.pi * sources[name] / period
        out[f"{name}_sin"] = np.sin(angle)
        out[f"{name}_cos"] = np.cos(angle)

    monsoon = sorted(set(cfg.get("features.season.monsoon_months")))
    # np.isin, not Index.isin: under pandas 3 the latter returns a bare ndarray,
    # so chaining a Series/Index method onto it fails.
    out["is_monsoon"] = np.isin(local.month, monsoon).astype(int)
    # Retained unencoded for stratified evaluation, not used as a predictor.
    out["month_local"] = local.month.to_numpy()
    out["hour_local"] = local.hour.to_numpy()
    return out


#: Config keys holding a backward reach, in the order they are reported.
LOOKBACK_KEYS = (
    "features.pm25_lags_h",
    "features.rolling_windows_h",
    "features.diff_horizons_h",
    "features.met_lags_h",
)


def capped_lookbacks(cfg: Config) -> dict[str, list[int]]:
    """The backward-dependency lists after ``features.lookback_h`` is applied.

    :func:`add_target_history`, :func:`add_met_history`,
    :func:`derived_history_columns` and :func:`max_backward_dependency` all read
    the same four config lists. If a cap were applied in some of them and not
    others, the columns actually built and the names generated to exclude them
    from the sequence channels would disagree -- which is precisely the drift
    :func:`derived_history_columns` was written to prevent. So the cap is applied
    here, once, and those four read this.

    Args:
        cfg: Loaded configuration.

    Returns:
        Dotted config key to its list of hours, ascending, cap applied.

    Raises:
        ConfigError: If the cap is below one, or empties any of the lists.
    """
    cap = cfg.get("features.lookback_h", None)
    lists = {key: sorted(int(x) for x in cfg.get(key)) for key in LOOKBACK_KEYS}
    if cap is None:
        return lists

    cap = int(cap)
    if cap < 1:
        raise ConfigError(f"features.lookback_h must be >= 1, got {cap}")

    capped = {key: [v for v in values if v <= cap] for key, values in lists.items()}
    empty = [key for key, values in capped.items() if not values]
    if empty:
        raise ConfigError(
            f"features.lookback_h={cap} leaves {', '.join(empty)} empty; "
            f"the cap must admit at least the shortest configured lag of each"
        )
    return capped


def history_floor(cfg: Config) -> int:
    """Hours of unbroken history a row must carry to be scored.

    Defaults to :func:`max_backward_dependency`, which is the status quo: the
    supervision floor and the deepest feature are welded together. Setting
    ``features.history_floor_h`` **above** the cap decouples them, which is the
    identical-row-count control arm of the lookback frontier -- the same device
    the gap-injection experiment uses when it holds the removed hour count
    constant across arms.

    A floor **below** the deepest feature is a leakage-adjacent error, not an
    option. The row's deepest lag would be NaN because it reaches outside the
    run, and ``models.data.get_split_arrays`` replaces NaN with 0.0 *after*
    scaling -- that is, with the training mean, silently and without a warning.

    Args:
        cfg: Loaded configuration.

    Returns:
        The floor in hours.

    Raises:
        ConfigError: If the configured floor is below the deepest feature.
    """
    deepest = max_backward_dependency(cfg)
    floor = cfg.get("features.history_floor_h", None)
    if floor is None:
        return deepest
    floor = int(floor)
    if floor < deepest:
        raise ConfigError(
            f"features.history_floor_h={floor} is below the deepest configured feature "
            f"({deepest} h). Every backward dependency must resolve inside the run, or the "
            f"deepest lag arrives at the model as the training mean rather than as data."
        )
    return floor


def scoreable_mask(frame: pd.DataFrame, horizon: int) -> pd.Series:
    """Rows a forecast could be scored on at all, at any history floor.

    This is ``valid_h{h}`` minus the history condition: the row sits in a run,
    the target lies inside that same run ``horizon`` hours ahead, and the target
    is a genuine observation. It is the one set every arm of the lookback sweep
    is scored over, so RMSE stays comparable across arms for exactly the reason
    the gap-injection experiment protects its test period -- a denominator that
    moves with the treatment is not a denominator.

    Args:
        frame: Built feature frame.
        horizon: Forecast horizon in hours.

    Returns:
        Boolean mask over the frame's index.
    """
    in_run = frame["run_id"] >= 0
    horizon_ok = (frame["run_len"] - frame["pos_in_run"]) > horizon
    target_observed = frame["is_observed"].shift(-horizon).fillna(False).astype(bool)
    return in_run & horizon_ok & target_observed


def add_target_history(frame: pd.DataFrame, cfg: Config, target: str) -> pd.DataFrame:
    """Add lags, rolling statistics and differences of the target.

    All rolling windows are right-aligned and inclusive of the current row, and
    use ``min_periods`` equal to the window so that a partially-filled window is
    NaN rather than a differently-scaled statistic.

    Args:
        frame: Frame containing the target column.
        cfg: Loaded configuration (``features.pm25_lags_h``,
            ``features.rolling_windows_h``, ``features.rolling_stats``,
            ``features.diff_horizons_h``).
        target: Target column name.

    Returns:
        The frame with history features added.
    """
    out = frame.copy()
    series = out[target]
    caps = capped_lookbacks(cfg)

    for lag in caps["features.pm25_lags_h"]:
        out[f"{target}_lag_{lag}"] = series.shift(int(lag))

    stats = list(cfg.get("features.rolling_stats"))
    for window in caps["features.rolling_windows_h"]:
        w = int(window)
        roller = series.rolling(window=w, min_periods=w)
        for stat in stats:
            out[f"{target}_roll{w}_{stat}"] = getattr(roller, stat)()

    for horizon in caps["features.diff_horizons_h"]:
        d = int(horizon)
        out[f"{target}_diff_{d}"] = series.diff(d)
        # Rate of change per hour, guarded against division by a near-zero base.
        base = series.shift(d)
        out[f"{target}_roc_{d}"] = (series - base) / base.where(base.abs() > 1e-6) / d

    return out


def add_met_history(frame: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Add lagged meteorology.

    Only backward shifts are applied. There is deliberately no code path here
    that can produce a forward-shifted meteorological feature; the oracle variant
    is built separately and labelled.

    Args:
        frame: Frame containing the meteorological columns.
        cfg: Loaded configuration (``features.met_vars``, ``features.met_lags_h``).

    Returns:
        The frame with lagged meteorology added.
    """
    out = frame.copy()
    met_lags = capped_lookbacks(cfg)["features.met_lags_h"]
    met_vars = [v for v in cfg.get("features.met_vars") if v in out.columns]
    extra = [c for c in ("wind_u", "wind_v") if c in out.columns]
    for var in met_vars + extra:
        for lag in met_lags:
            out[f"{var}_lag_{int(lag)}"] = out[var].shift(int(lag))
    return out


def add_oracle_met(frame: pd.DataFrame, cfg: Config, horizons: list[int]) -> pd.DataFrame:
    """Add perfect-forecast meteorology at *t+h*, for the upper-bound variant only.

    These columns are an oracle: they use weather observed at the target time,
    which no real forecast has. They exist so the study can report an explicit
    upper bound, and are prefixed ``oracle_`` so they can never be swept into the
    honest feature set by a wildcard.

    Args:
        frame: Frame containing the meteorological columns.
        cfg: Loaded configuration.
        horizons: Horizons to build oracle columns for.

    Returns:
        The frame with ``oracle_<var>_h<h>`` columns added.
    """
    if not bool(cfg.get("task.oracle_met_variant.enabled", False)):
        return frame
    out = frame.copy()
    met_vars = [v for v in cfg.get("features.met_vars") if v in out.columns]
    extra = [c for c in ("wind_u", "wind_v") if c in out.columns]
    for var in met_vars + extra:
        for h in horizons:
            out[f"oracle_{var}_h{int(h)}"] = out[var].shift(-int(h))
    return out


def feature_columns(frame: pd.DataFrame, cfg: Config, *, include_oracle: bool = False) -> list[str]:
    """Select predictor columns, excluding targets, bookkeeping and oracles.

    Args:
        frame: Built feature frame.
        cfg: Loaded configuration.
        include_oracle: Whether to include the perfect-forecast oracle columns.

    Returns:
        Predictor column names, in stable order.
    """
    target = str(cfg.get("features.target"))
    bookkeeping = {
        "run_id",
        "pos_in_run",
        "run_len",
        "is_observed",
        "split",
        "month_local",
        "hour_local",
    }
    cols: list[str] = []
    for col in frame.columns:
        if col in bookkeeping or col.startswith("target_") or col.startswith("valid_h"):
            continue
        if col.startswith("oracle_") and not include_oracle:
            continue
        cols.append(col)
    # The contemporaneous target value is a legitimate predictor: it is the
    # observation at t, which persistence itself uses. Its future values are not.
    if target not in cols and target in frame.columns:
        cols.insert(0, target)
    return cols


def derived_history_columns(cfg: Config, frame: pd.DataFrame | None = None) -> set[str]:
    """Names of every column produced by lagging, rolling or differencing.

    Generated from the same configuration keys that :func:`add_target_history`
    and :func:`add_met_history` read, so the two cannot drift apart: adding a
    rolling statistic to the config changes what is built and what is listed
    here in one step.

    Args:
        cfg: Loaded configuration.
        frame: Optional built frame. When given, meteorological names are
            restricted to columns actually present, matching what
            :func:`add_met_history` does.

    Returns:
        The set of derived-history column names.
    """
    target = str(cfg.get("features.target"))
    caps = capped_lookbacks(cfg)
    names: set[str] = set()

    for lag in caps["features.pm25_lags_h"]:
        names.add(f"{target}_lag_{int(lag)}")
    for window in caps["features.rolling_windows_h"]:
        for stat in cfg.get("features.rolling_stats"):
            names.add(f"{target}_roll{int(window)}_{stat}")
    for horizon in caps["features.diff_horizons_h"]:
        names.add(f"{target}_diff_{int(horizon)}")
        names.add(f"{target}_roc_{int(horizon)}")

    met_vars = list(cfg.get("features.met_vars"))
    if frame is not None:
        met_vars = [v for v in met_vars if v in frame.columns]
    for var in [*met_vars, "wind_u", "wind_v"]:
        for lag in caps["features.met_lags_h"]:
            names.add(f"{var}_lag_{int(lag)}")

    return names


def sequence_channel_columns(
    frame: pd.DataFrame, cfg: Config, *, include_oracle: bool = False
) -> list[str]:
    """Select the per-timestep channels a sequence model receives.

    Tabular models are given engineered history because each row must stand
    alone: a tree sees one timestamp and needs the past folded into it. A
    recurrent model consumes the window itself, so the same columns are largely
    redundant with what it already reads -- ``pm25_lag_24`` at step *t* is the
    ``pm25`` channel at step *t-24*, which is inside a 48-hour window. Supplying
    both inflates a 48x18 input to 48x102 with heavy collinearity and no added
    information, which handicaps the sequence tier in a comparison against the
    tabular tier that the same columns help.

    ``features.sequence_channels.mode`` selects between:

    - ``contemporaneous`` -- observation channels only, letting the recurrence
      extract temporal structure. The honest comparison.
    - ``engineered`` -- the full tabular predictor set, reproducing the earlier
      behaviour so the two representations can be reported side by side.

    Lags reaching further back than the window (``pm25_lag_168`` under a 24-hour
    window) are the one genuinely additional signal dropped in
    ``contemporaneous`` mode. That is a deliberate trade and it is what the
    ``engineered`` arm exists to quantify.

    Args:
        frame: Built feature frame.
        cfg: Loaded configuration.
        include_oracle: Whether to include the perfect-forecast oracle columns.

    Returns:
        Channel names, in the same stable order as :func:`feature_columns`.

    Raises:
        ValueError: If the configured mode is unknown.
    """
    columns = feature_columns(frame, cfg, include_oracle=include_oracle)
    mode = str(cfg.get("features.sequence_channels.mode", "contemporaneous"))

    if mode == "engineered":
        return columns
    if mode != "contemporaneous":
        raise ValueError(
            f"features.sequence_channels.mode must be 'contemporaneous' or 'engineered', "
            f"got {mode!r}"
        )

    derived = derived_history_columns(cfg, frame)
    return [c for c in columns if c not in derived]


def max_backward_dependency(cfg: Config) -> int:
    """Longest backward lookback any feature requires, in hours.

    Args:
        cfg: Loaded configuration.

    Returns:
        The maximum of all configured lags and rolling windows, after
        ``features.lookback_h`` is applied.
    """
    return max(max(values) for values in capped_lookbacks(cfg).values())


def build_features(
    pm: pd.DataFrame, met: pd.DataFrame, cfg: Config, logger: Any
) -> tuple[pd.DataFrame, FeatureBuildReport]:
    """Fuse PM2.5 and meteorology and construct the full feature matrix.

    Args:
        pm: Hourly PM2.5 frame indexed by UTC timestamp.
        met: Hourly meteorology frame indexed by UTC timestamp.
        cfg: Loaded configuration.
        logger: Logger for row accounting.

    Returns:
        ``(frame, report)`` where ``frame`` carries features, per-horizon targets
        and per-horizon validity masks.
    """
    target = str(cfg.get("features.target"))
    horizons = [int(h) for h in cfg.get("task.horizons_h")]
    max_ffill = int(cfg.get("impute.max_ffill_hours", 3))

    fused = pm.join(met, how="left")
    n_grid = len(fused)

    # Mark genuine observations before any filling, so a target is never an
    # imputed value pretending to be one.
    fused["is_observed"] = fused[target].notna()
    n_observed = int(fused["is_observed"].sum())

    # Leakage rule 3: forward-fill only. It carries a PAST value forward, which
    # is available at prediction time. Linear interpolation would blend a future
    # observation into the present and is therefore never used.
    fused[target] = fused[target].ffill(limit=max_ffill)
    met_cols = [c for c in met.columns if c in fused.columns]
    fused[met_cols] = fused[met_cols].ffill(limit=max_ffill)
    n_ffilled = int(fused[target].notna().sum()) - n_observed

    usable = fused[target].notna() & fused[met_cols].notna().all(axis=1)
    fused["run_id"] = assign_runs(usable)
    fused["pos_in_run"] = position_in_run(fused["run_id"])
    fused["run_len"] = run_lengths(fused["run_id"])

    n_runs = int((fused["run_id"] >= 0).groupby(fused["run_id"]).ngroups)
    longest = int(fused["run_len"].max())
    logger.info("contiguity: %d runs after %dh ffill, longest %d h", n_runs, max_ffill, longest)

    fused = add_wind_components(fused, cfg)
    fused = add_calendar_features(fused, cfg)
    fused = add_target_history(fused, cfg, target)
    fused = add_met_history(fused, cfg)
    fused = add_oracle_met(fused, cfg, horizons)

    max_lag = max_backward_dependency(cfg)
    # The floor, not the deepest feature. They are the same number unless the
    # lookback frontier's control arm has deliberately decoupled them.
    floor_h = history_floor(cfg)
    cap = cfg.get("features.lookback_h", None)

    report = FeatureBuildReport(
        n_rows_grid=n_grid,
        n_observed_target=n_observed,
        n_ffilled=n_ffilled,
        n_runs=n_runs,
        longest_run_h=longest,
        max_lag_h=max_lag,
        history_floor_h=floor_h,
        lookback_cap_h=None if cap is None else int(cap),
        n_features=len(feature_columns(fused, cfg)),
    )

    # ---- targets and validity, per horizon -------------------------------
    for h in horizons:
        fused[f"target_h{h}"] = fused[target].shift(-h)
        target_observed = fused["is_observed"].shift(-h).fillna(False).astype(bool)

        # Every backward dependency must resolve inside this run...
        history_ok = fused["pos_in_run"] >= floor_h
        # ...and the target must lie inside the same run, h steps ahead.
        horizon_ok = (fused["run_len"] - fused["pos_in_run"]) > h
        in_run = fused["run_id"] >= 0

        valid = in_run & history_ok & horizon_ok & target_observed
        fused[f"valid_h{h}"] = valid

        rejected = {
            "outside_any_run": int((~in_run).sum()),
            "insufficient_history": int((in_run & ~history_ok).sum()),
            "target_beyond_run_end": int((in_run & history_ok & ~horizon_ok).sum()),
            "target_not_observed": int((in_run & history_ok & horizon_ok & ~target_observed).sum()),
        }
        report.rows_valid_per_horizon[h] = int(valid.sum())
        report.rows_rejected_per_horizon[h] = rejected
        logger.info(
            "h=%3d: %6d valid rows  (rejected: no-run %d, short-history %d, "
            "target-past-run-end %d, target-imputed %d)",
            h,
            int(valid.sum()),
            rejected["outside_any_run"],
            rejected["insufficient_history"],
            rejected["target_beyond_run_end"],
            rejected["target_not_observed"],
        )

    return fused, report
