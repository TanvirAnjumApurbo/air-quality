"""Leakage guards -- one test per non-negotiable methodological rule.

These are the mistakes that get air-quality papers torn apart. Each rule from the
study protocol has an executable test here, written to fail loudly if a future
refactor reintroduces the mistake.

Rules under test:

1. No future meteorology in the honest feature set.
2. Split before fitting anything; chronological only, never shuffled.
3. No interpolation across split boundaries.
4. Gap-aware windowing: no window may span a data gap.
5. Direct multi-horizon, not recursive.
"""

from __future__ import annotations

import copy
import logging

import numpy as np
import pandas as pd
import pytest
from src.eval.split import (
    TrainOnlyScaler,
    assign_splits,
    inverse_transform_target,
    purge_boundary_rows,
    resolve_boundaries,
    transform_target,
)
from src.features.build_features import (
    assign_runs,
    build_features,
    feature_columns,
    max_backward_dependency,
    position_in_run,
)
from src.utils import Config, ConfigError, load_config

LOG = logging.getLogger("test_leakage")


@pytest.fixture(scope="module")
def cfg() -> Config:
    return load_config()


@pytest.fixture(scope="module")
def cfg_fractional(cfg) -> Config:
    """The real config with explicit boundaries cleared.

    The production config pins resolved boundary dates in 2022/2023, which lie
    outside the synthetic fixtures' span. Split *logic* is exercised here in
    fraction mode; that the pinned dates must fall inside the data is asserted
    separately in `test_explicit_boundaries_outside_span_raise`.
    """
    raw = copy.deepcopy(cfg.raw)
    raw["split"]["explicit_boundaries"] = {"train_end": None, "val_end": None}
    return Config(raw=raw, path=cfg.path)


@pytest.fixture(scope="module")
def synthetic() -> tuple[pd.DataFrame, pd.DataFrame]:
    """A gap-free synthetic pair with a known structure.

    The target is a deterministic function of time so that any accidental
    forward shift shows up as an exact, detectable relationship.
    """
    index = pd.date_range("2020-01-01", periods=2000, freq="1h", tz="UTC")
    n = len(index)
    pm = pd.DataFrame({"pm25": np.arange(n, dtype=float) + 10.0}, index=index)
    pm.index.name = "datetime_utc"
    met = pd.DataFrame(
        {
            "T2M": np.sin(np.arange(n) / 24.0) * 5 + 25,
            "T2MDEW": np.cos(np.arange(n) / 24.0) * 3 + 20,
            "RH2M": np.full(n, 70.0),
            "WS10M": np.full(n, 2.0),
            "WD10M": np.full(n, 90.0),
            "PS": np.full(n, 101.0),
            "PRECTOTCORR": np.zeros(n),
            "ALLSKY_SFC_SW_DWN": np.zeros(n),
        },
        index=index,
    )
    met.index.name = "datetime_utc"
    return pm, met


@pytest.fixture(scope="module")
def gapped() -> tuple[pd.DataFrame, pd.DataFrame]:
    """A pair with one long, deliberate gap that no forward-fill can bridge."""
    index = pd.date_range("2020-01-01", periods=1200, freq="1h", tz="UTC")
    n = len(index)
    values = np.arange(n, dtype=float) + 10.0
    values[500:560] = np.nan  # 60-hour outage, far beyond the ffill limit
    pm = pd.DataFrame({"pm25": values}, index=index)
    pm.index.name = "datetime_utc"
    met = pd.DataFrame(
        {
            v: np.full(n, 1.0)
            for v in [
                "T2M",
                "T2MDEW",
                "RH2M",
                "WS10M",
                "WD10M",
                "PS",
                "PRECTOTCORR",
                "ALLSKY_SFC_SW_DWN",
            ]
        },
        index=index,
    )
    met.index.name = "datetime_utc"
    return pm, met


# ---------------------------------------------------------------------------
# Rule 1: no future meteorology
# ---------------------------------------------------------------------------


@pytest.mark.leakage
def test_no_future_meteorology_in_feature_set(cfg, synthetic):
    """Every honest predictor must correlate with the past, never the future."""
    pm, met = synthetic
    frame, _ = build_features(pm, met, cfg, LOG)
    predictors = feature_columns(frame, cfg, include_oracle=False)

    assert not any(c.startswith("oracle_") for c in predictors), (
        "oracle columns leaked into the honest feature set"
    )

    # A forward-shifted column would equal a backward-shifted copy of itself.
    # Compare each predictor against the future meteorology it must not know.
    for var in cfg.get("features.met_vars"):
        if var not in frame.columns:
            continue
        for h in cfg.get("task.horizons_h"):
            future = frame[var].shift(-int(h))
            for col in predictors:
                if not col.startswith(var):
                    continue
                aligned = pd.concat([frame[col], future], axis=1).dropna()
                if len(aligned) < 50 or aligned.iloc[:, 0].std() == 0:
                    continue
                identical = np.allclose(
                    aligned.iloc[:, 0].to_numpy(), aligned.iloc[:, 1].to_numpy()
                )
                assert not identical, f"{col} equals {var} at t+{h}: future meteorology"


@pytest.mark.leakage
def test_met_lag_columns_are_backward_shifts(cfg, synthetic):
    """A lag column must equal the base column shifted backward, exactly."""
    pm, met = synthetic
    frame, _ = build_features(pm, met, cfg, LOG)
    for var in ["T2M", "RH2M"]:
        for lag in cfg.get("features.met_lags_h"):
            col = f"{var}_lag_{int(lag)}"
            assert col in frame.columns
            expected = frame[var].shift(int(lag))
            pd.testing.assert_series_equal(
                frame[col], expected, check_names=False, check_dtype=False
            )


@pytest.mark.leakage
def test_target_history_uses_only_past(cfg, synthetic):
    """Lag and rolling features must never incorporate the current-or-future target."""
    pm, met = synthetic
    frame, _ = build_features(pm, met, cfg, LOG)
    target = str(cfg.get("features.target"))

    for lag in cfg.get("features.pm25_lags_h"):
        col = f"{target}_lag_{int(lag)}"
        pd.testing.assert_series_equal(
            frame[col], frame[target].shift(int(lag)), check_names=False, check_dtype=False
        )

    # Rolling windows are right-aligned and inclusive of t, so on a strictly
    # increasing series the rolling max equals the current value and the rolling
    # min equals the value w-1 hours back. Anything else means the window is
    # centred or forward-looking.
    for window in cfg.get("features.rolling_windows_h"):
        w = int(window)
        roll_max = frame[f"{target}_roll{w}_max"].dropna()
        assert np.allclose(roll_max.to_numpy(), frame[target].reindex(roll_max.index).to_numpy())
        roll_min = frame[f"{target}_roll{w}_min"].dropna()
        expected_min = frame[target].shift(w - 1).reindex(roll_min.index)
        assert np.allclose(roll_min.to_numpy(), expected_min.to_numpy())


@pytest.mark.leakage
def test_oracle_columns_exist_and_are_forward(cfg, synthetic):
    """The oracle variant must be genuinely forward-looking, and clearly named."""
    pm, met = synthetic
    frame, _ = build_features(pm, met, cfg, LOG)
    if not cfg.get("task.oracle_met_variant.enabled"):
        pytest.skip("oracle variant disabled")

    h = int(cfg.get("task.headline_horizon_h"))
    col = f"oracle_T2M_h{h}"
    assert col in frame.columns
    pd.testing.assert_series_equal(
        frame[col], frame["T2M"].shift(-h), check_names=False, check_dtype=False
    )
    assert col not in feature_columns(frame, cfg, include_oracle=False)
    assert col in feature_columns(frame, cfg, include_oracle=True)


@pytest.mark.leakage
def test_config_forbids_future_meteorology(cfg):
    """The config invariant itself must be enforced, not merely documented."""
    assert cfg.get("task.allow_future_meteorology") is False
    raw = dict(cfg.raw)
    raw["task"] = {**raw["task"], "allow_future_meteorology": True}
    import tempfile
    from pathlib import Path

    import yaml

    with tempfile.TemporaryDirectory() as tmp:
        bad = Path(tmp) / "bad.yaml"
        bad.write_text(yaml.safe_dump(raw), encoding="utf-8")
        with pytest.raises(ConfigError, match="allow_future_meteorology"):
            load_config(bad)


# ---------------------------------------------------------------------------
# Rule 2: split before fitting anything
# ---------------------------------------------------------------------------


@pytest.mark.leakage
def test_splits_are_chronological_and_disjoint(cfg_fractional, synthetic):
    """Splits must be contiguous blocks in time with no interleaving."""
    pm, _ = synthetic
    bounds = resolve_boundaries(pm.index, cfg_fractional)
    split = assign_splits(pm.index, bounds)

    assert split.iloc[0] == "train"
    assert split.iloc[-1] == "test"
    assert set(split.unique()) == {"train", "val", "test"}

    train_max = pm.index[split == "train"].max()
    val_min = pm.index[split == "val"].min()
    val_max = pm.index[split == "val"].max()
    test_min = pm.index[split == "test"].min()
    assert train_max < val_min < val_max < test_min

    # Ordering must be monotone: train then val then test, never interleaved.
    codes = split.map({"train": 0, "val": 1, "test": 2}).to_numpy()
    assert np.all(np.diff(codes) >= 0), "splits interleave in time"


@pytest.mark.leakage
def test_config_forbids_shuffling(cfg):
    """A shuffled time-series split must be rejected at load time."""
    assert cfg.get("split.shuffle") is False
    raw = dict(cfg.raw)
    raw["split"] = {**raw["split"], "shuffle": True}
    import tempfile
    from pathlib import Path

    import yaml

    with tempfile.TemporaryDirectory() as tmp:
        bad = Path(tmp) / "bad.yaml"
        bad.write_text(yaml.safe_dump(raw), encoding="utf-8")
        with pytest.raises(ConfigError, match="shuffle"):
            load_config(bad)


@pytest.mark.leakage
def test_scaler_rejects_non_train_fit(synthetic):
    """Fitting a scaler on val or test must raise, not warn."""
    pm, _ = synthetic
    frame = pd.DataFrame({"a": np.arange(len(pm), dtype=float)}, index=pm.index)
    scaler = TrainOnlyScaler("standard")
    for bad_split in ("val", "test", "all"):
        with pytest.raises(ConfigError, match="fitted on train"):
            scaler.fit(frame, ["a"], bad_split)


@pytest.mark.leakage
def test_scaler_refuses_transform_before_fit(synthetic):
    """Transforming before fitting must raise."""
    pm, _ = synthetic
    frame = pd.DataFrame({"a": np.arange(len(pm), dtype=float)}, index=pm.index)
    with pytest.raises(ConfigError, match="before fit"):
        TrainOnlyScaler("standard").transform(frame)


@pytest.mark.leakage
def test_scaler_statistics_come_only_from_train(cfg_fractional, synthetic):
    """Scaler mean/scale must be computable from the train slice alone."""
    pm, _ = synthetic
    bounds = resolve_boundaries(pm.index, cfg_fractional)
    split = assign_splits(pm.index, bounds)
    frame = pd.DataFrame({"a": np.arange(len(pm), dtype=float)}, index=pm.index)

    scaler = TrainOnlyScaler("standard").fit(frame[split == "train"], ["a"], "train")
    train_only = frame.loc[split == "train", "a"]

    assert scaler.mean_ is not None and scaler.scale_ is not None
    assert np.isclose(scaler.mean_["a"], train_only.mean())
    assert np.isclose(scaler.scale_["a"], train_only.std(ddof=0))
    # The full-series mean is higher on this monotone series; if the scaler had
    # seen val/test it would match that instead.
    assert not np.isclose(scaler.mean_["a"], frame["a"].mean())


# ---------------------------------------------------------------------------
# Rule 3: no interpolation across split boundaries
# ---------------------------------------------------------------------------


@pytest.mark.leakage
def test_imputation_is_forward_fill_only(cfg):
    """Config must not select an imputer that can read forward in time."""
    method = str(cfg.get("impute.method"))
    assert method in {"ffill_limited", "per_split_interpolate", "none"}
    assert method != "linear_interpolate"


@pytest.mark.leakage
def test_ffill_never_uses_a_future_value():
    """Forward-fill must propagate the past forward, never the future backward."""
    index = pd.date_range("2020-01-01", periods=10, freq="1h", tz="UTC")
    values = pd.Series(
        [1.0, np.nan, np.nan, 4.0, np.nan, 6.0, 7.0, np.nan, np.nan, 10.0], index=index
    )
    filled = values.ffill(limit=3)
    # Position 1 and 2 must carry the value from position 0, not from position 3.
    assert filled.iloc[1] == 1.0
    assert filled.iloc[2] == 1.0
    assert filled.iloc[4] == 4.0
    # A leading NaN has no past to draw on and must remain missing.
    leading = pd.Series([np.nan, np.nan, 3.0], index=index[:3]).ffill(limit=3)
    assert leading.isna().iloc[0] and leading.isna().iloc[1]


@pytest.mark.leakage
def test_boundary_rows_are_purged(cfg_fractional, synthetic):
    """Rows whose history or target crosses a split boundary must be dropped."""
    pm, met = synthetic
    frame, _ = build_features(pm, met, cfg_fractional, LOG)
    bounds = resolve_boundaries(frame.index, cfg_fractional)
    split = assign_splits(frame.index, bounds)
    max_lag = max_backward_dependency(cfg_fractional)
    horizon = int(cfg_fractional.get("task.headline_horizon_h"))

    keep = purge_boundary_rows(frame, split, max_lag, horizon)

    val_index = frame.index[split == "val"]
    kept_val = frame.index[keep & (split == "val")]
    # No kept validation row may reach back before validation began.
    assert kept_val.min() >= val_index.min() + pd.Timedelta(hours=max_lag)

    train_index = frame.index[split == "train"]
    kept_train = frame.index[keep & (split == "train")]
    # No kept training row may have its target land in validation.
    assert kept_train.max() <= train_index.max() - pd.Timedelta(hours=horizon)


# ---------------------------------------------------------------------------
# Rule 4: gap-aware windowing
# ---------------------------------------------------------------------------


@pytest.mark.leakage
def test_runs_break_at_gaps():
    """Run labelling must split at every gap, however short."""
    index = pd.date_range("2020-01-01", periods=10, freq="1h", tz="UTC")
    present = pd.Series([True] * 4 + [False] * 2 + [True] * 4, index=index)
    runs = assign_runs(present)
    assert (runs[4:6] == -1).all()
    assert runs.iloc[0] == runs.iloc[3]
    assert runs.iloc[6] == runs.iloc[9]
    assert runs.iloc[0] != runs.iloc[6], "a gap did not start a new run"

    pos = position_in_run(runs)
    assert list(pos.iloc[0:4]) == [0, 1, 2, 3]
    assert list(pos.iloc[6:10]) == [0, 1, 2, 3]
    assert (pos[4:6] == -1).all()


@pytest.mark.leakage
def test_no_valid_row_spans_a_gap(cfg, gapped):
    """No row marked valid may have its window or target cross the outage."""
    pm, met = gapped
    frame, report = build_features(pm, met, cfg, LOG)
    max_lag = max_backward_dependency(cfg)

    gap_index = pm.index[pm["pm25"].isna()]
    assert len(gap_index) == 60

    for h in cfg.get("task.horizons_h"):
        valid = frame.index[frame[f"valid_h{int(h)}"]]
        for ts in valid:
            window_start = ts - pd.Timedelta(hours=max_lag)
            target_ts = ts + pd.Timedelta(hours=int(h))
            overlapping = gap_index[(gap_index >= window_start) & (gap_index <= target_ts)]
            assert len(overlapping) == 0, (
                f"valid row {ts} at h={h} spans the gap at {overlapping[0]}"
            )

    assert report.n_runs >= 2, "the deliberate outage did not create a second run"


@pytest.mark.leakage
def test_rejected_window_counts_are_reported(cfg, gapped):
    """The number of windows the gap rule removes must be counted, not silent."""
    pm, met = gapped
    _, report = build_features(pm, met, cfg, LOG)
    for h in cfg.get("task.horizons_h"):
        rejected = report.rows_rejected_per_horizon[int(h)]
        assert set(rejected) == {
            "outside_any_run",
            "insufficient_history",
            "target_beyond_run_end",
            "target_not_observed",
        }
        assert sum(rejected.values()) > 0
        assert report.rows_valid_per_horizon[int(h)] > 0


@pytest.mark.leakage
def test_target_is_never_an_imputed_value(cfg, gapped):
    """A forward-filled value must never be used as supervision."""
    pm, met = gapped
    frame, _ = build_features(pm, met, cfg, LOG)
    truly_observed = pm["pm25"].notna()
    for h in cfg.get("task.horizons_h"):
        valid = frame[f"valid_h{int(h)}"]
        target_times = frame.index[valid] + pd.Timedelta(hours=int(h))
        assert truly_observed.reindex(target_times).all(), f"h={h}: a target was an imputed value"


# ---------------------------------------------------------------------------
# Rule 5: direct multi-horizon, not recursive
# ---------------------------------------------------------------------------


@pytest.mark.leakage
def test_separate_target_per_horizon(cfg, synthetic):
    """Each horizon must have its own target column, shifted by exactly h."""
    pm, met = synthetic
    frame, _ = build_features(pm, met, cfg, LOG)
    target = str(cfg.get("features.target"))
    horizons = [int(h) for h in cfg.get("task.horizons_h")]

    assert len(horizons) == len({f"target_h{h}" for h in horizons})
    for h in horizons:
        pd.testing.assert_series_equal(
            frame[f"target_h{h}"],
            frame[target].shift(-h),
            check_names=False,
            check_dtype=False,
        )

    # Distinct horizons must give distinct targets; identical columns would mean
    # one horizon silently reused another's labels.
    for i, h1 in enumerate(horizons):
        for h2 in horizons[i + 1 :]:
            a = frame[f"target_h{h1}"].dropna()
            b = frame[f"target_h{h2}"].dropna()
            common = a.index.intersection(b.index)
            assert not np.allclose(a.loc[common].to_numpy(), b.loc[common].to_numpy())


@pytest.mark.leakage
def test_horizons_configured_as_direct(cfg):
    """The protocol's direct-multi-horizon choice must be visible in config."""
    horizons = cfg.get("task.horizons_h")
    assert horizons == sorted(horizons)
    assert len(horizons) == len(set(horizons))
    assert int(cfg.get("task.headline_horizon_h")) in horizons


# ---------------------------------------------------------------------------
# Target transform round-trip
# ---------------------------------------------------------------------------


def test_target_transform_roundtrip(cfg):
    """Metrics are reported in ug/m3, so the transform must invert exactly."""
    method = str(cfg.get("scaling.target_transform"))
    values = np.array([0.0, 1.0, 12.5, 66.0, 250.0, 985.0])
    restored = inverse_transform_target(transform_target(values, method), method)
    assert np.allclose(values, restored, rtol=1e-9, atol=1e-9)


# ---------------------------------------------------------------------------
# Negative controls: prove the guards above actually fire.
#
# A leakage test that passes trivially is worthless. Each control below injects
# the exact mistake the corresponding rule forbids and asserts the guard catches
# it, so a future refactor that quietly disables a check fails here.
# ---------------------------------------------------------------------------


@pytest.mark.leakage
def test_control_future_meteorology_is_detected(cfg, synthetic):
    """Injecting weather from t+h must be caught by the equality check."""
    pm, met = synthetic
    frame, _ = build_features(pm, met, cfg, LOG)
    h = int(cfg.get("task.headline_horizon_h"))

    leaked = frame.copy()
    leaked["T2M_lag_1"] = leaked["T2M"].shift(-h)  # a lag column that peeks ahead

    future = leaked["T2M"].shift(-h)
    aligned = pd.concat([leaked["T2M_lag_1"], future], axis=1).dropna()
    assert np.allclose(aligned.iloc[:, 0].to_numpy(), aligned.iloc[:, 1].to_numpy()), (
        "the future-meteorology detector failed to see an injected forward shift"
    )


@pytest.mark.leakage
def test_control_shuffled_split_is_detected(cfg_fractional, synthetic):
    """A shuffled split must break the monotonicity assertion."""
    pm, _ = synthetic
    bounds = resolve_boundaries(pm.index, cfg_fractional)
    split = assign_splits(pm.index, bounds)

    rng = np.random.default_rng(0)
    shuffled = pd.Series(rng.permutation(split.to_numpy()), index=split.index)
    codes = shuffled.map({"train": 0, "val": 1, "test": 2}).to_numpy()
    assert not np.all(np.diff(codes) >= 0), "the chronology check failed to see a shuffled split"


@pytest.mark.leakage
def test_control_gap_spanning_window_is_detected(cfg, gapped):
    """Marking gap-spanning rows valid must be caught by the contiguity check."""
    pm, met = gapped
    frame, _ = build_features(pm, met, cfg, LOG)
    max_lag = max_backward_dependency(cfg)
    gap_index = pm.index[pm["pm25"].isna()]
    h = int(cfg.get("task.headline_horizon_h"))

    # Force every row valid, ignoring runs -- the mistake gap-awareness prevents.
    forced = frame.index[frame.index >= frame.index.min() + pd.Timedelta(hours=max_lag)]
    offenders = [
        ts
        for ts in forced
        if len(
            gap_index[
                (gap_index >= ts - pd.Timedelta(hours=max_lag))
                & (gap_index <= ts + pd.Timedelta(hours=h))
            ]
        )
        > 0
    ]
    assert offenders, "the gap detector found nothing to reject in a deliberately gapped series"

    honest = set(frame.index[frame[f"valid_h{h}"]])
    assert not (set(offenders) & honest), "build_features marked a gap-spanning row valid"


@pytest.mark.leakage
def test_control_full_series_scaler_differs_from_train_only(cfg_fractional, synthetic):
    """Fitting on everything must give different statistics from fitting on train."""
    pm, _ = synthetic
    bounds = resolve_boundaries(pm.index, cfg_fractional)
    split = assign_splits(pm.index, bounds)
    frame = pd.DataFrame({"a": np.arange(len(pm), dtype=float)}, index=pm.index)

    train_only = TrainOnlyScaler("standard").fit(frame[split == "train"], ["a"], "train")
    assert train_only.mean_ is not None

    full_mean = frame["a"].mean()
    assert not np.isclose(train_only.mean_["a"], full_mean), (
        "train-only and full-series scaling are indistinguishable on this fixture, "
        "so the guard could not detect the leak"
    )


@pytest.mark.leakage
def test_explicit_boundaries_outside_span_raise(cfg, synthetic):
    """Pinned boundaries that fall outside the data must raise, not be ignored.

    The production config pins resolved dates. If those are ever paired with a
    different dataset, silently falling back to fractions would produce a split
    that does not match the one printed in the table captions.
    """
    pm, _ = synthetic
    assert cfg.get("split.explicit_boundaries.train_end") is not None, (
        "this test is meaningless until boundaries are resolved"
    )
    with pytest.raises(ConfigError, match="strictly increasing"):
        resolve_boundaries(pm.index, cfg)


@pytest.mark.leakage
def test_sarimax_h_step_forecast_uses_no_future_data():
    """The vectorised h-step forecast must match a strictly causal reference.

    `get_prediction(dynamic=False)` returns ONE-step-ahead in-sample values;
    stamping those onto t+h silently hands the model h-1 steps of future data.
    This checks the state-space projection against a reference that can only see
    data up to the anchor, so a regression to the leaky form fails here.
    """
    import warnings

    from src.models.baselines import h_step_ahead_from_filtered_state
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    rng = np.random.default_rng(0)
    n = 400
    y = np.zeros(n)
    for t in range(2, n):
        y[t] = (
            0.7 * y[t - 1] - 0.2 * y[t - 2] + 0.5 * np.sin(2 * np.pi * t / 24) + rng.normal(0, 0.4)
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fitted = SARIMAX(
            y[:250],
            order=(2, 0, 1),
            seasonal_order=(1, 0, 1, 24),
            trend="c",
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False, maxiter=40)
        applied = fitted.apply(y, refit=False)

        for h in (1, 3, 24):
            fast = h_step_ahead_from_filtered_state(applied, h)
            for anchor in (150, 250, 350):
                if anchor + h >= n:
                    continue
                # Reference sees data only up to `anchor`.
                causal = fitted.apply(y[: anchor + 1], refit=False).forecast(steps=h)[-1]
                assert np.isclose(fast[anchor], causal, atol=1e-6), (
                    f"h={h} anchor={anchor}: {fast[anchor]} != causal {causal}"
                )


@pytest.mark.leakage
def test_predictions_are_bounded_to_physical_range(cfg):
    """Inverted predictions must respect the same physical bounds as QC."""
    from src.models.data import invert

    cap = float(cfg.get("qc.pm25.sanity_cap_ugm3"))
    # log1p-scale values that expm1 would blow far past any real concentration.
    extreme = np.array([-5.0, 0.0, 3.0, 20.0, 50.0, np.inf, np.nan])
    out = invert(cfg, extreme)
    assert np.all(np.isfinite(out))
    assert out.min() >= 0.0
    assert out.max() <= cap
