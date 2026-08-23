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
    derived_history_columns,
    feature_columns,
    max_backward_dependency,
    position_in_run,
    sequence_channel_columns,
)
from src.features.gap_injection import GapProfile, inject_gaps
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
# Sequence channels: the input the recurrent tier actually receives
#
# Not a leakage rule -- a fairness one. Feeding the sequence tier the engineered
# lag/rolling columns restates what the window already contains and inflates the
# input sixfold with collinear copies, which biases the tier comparison. These
# assert the channel set is what it claims to be.
# ---------------------------------------------------------------------------


def test_sequence_channels_carry_no_derived_history(cfg, synthetic):
    """Contemporaneous mode must expose no lag, rolling, difference or rate column."""
    pm, met = synthetic
    frame, _ = build_features(pm, met, cfg, LOG)
    channels = sequence_channel_columns(frame, cfg)

    assert str(cfg.get("features.sequence_channels.mode")) == "contemporaneous", (
        "this test describes the shipped default; update it deliberately if that changes"
    )
    offenders = [
        c
        for c in channels
        if "_lag_" in c or "_roll" in c or "_diff_" in c or "_roc_" in c or c.startswith("oracle_")
    ]
    assert not offenders, f"derived-history columns leaked into the channel set: {offenders}"


def test_sequence_channels_partition_the_tabular_set(cfg, synthetic):
    """Channels plus derived history must exactly reconstruct the tabular predictors.

    Catches drift in either direction: a feature added to the builder but not to
    ``derived_history_columns`` would silently reach the sequence tier, and a
    renamed column would silently vanish from the tabular one.
    """
    pm, met = synthetic
    frame, _ = build_features(pm, met, cfg, LOG)

    tabular = set(feature_columns(frame, cfg))
    channels = set(sequence_channel_columns(frame, cfg))
    derived = derived_history_columns(cfg, frame)

    assert channels <= tabular, "channels must be a subset of the tabular predictors"
    assert derived <= set(frame.columns), (
        f"derived_history_columns names columns the builder never produced: "
        f"{sorted(derived - set(frame.columns))}"
    )
    assert tabular - channels == derived & tabular, (
        "the channel/derived split does not partition the tabular predictor set"
    )


def test_engineered_mode_restores_the_full_predictor_set(cfg, synthetic):
    """The ablation arm must be a genuine alternative, not a silent no-op."""
    pm, met = synthetic
    frame, _ = build_features(pm, met, cfg, LOG)

    raw = copy.deepcopy(cfg.raw)
    raw["features"]["sequence_channels"]["mode"] = "engineered"
    engineered = Config(raw=raw, path=cfg.path)

    assert sequence_channel_columns(frame, engineered) == feature_columns(frame, engineered)
    assert len(sequence_channel_columns(frame, engineered)) > len(
        sequence_channel_columns(frame, cfg)
    ), "engineered mode must expose strictly more columns than contemporaneous mode"


def test_unknown_sequence_channel_mode_raises(cfg, synthetic):
    """A typo in the mode must fail loudly rather than silently picking a default."""
    pm, met = synthetic
    frame, _ = build_features(pm, met, cfg, LOG)

    raw = copy.deepcopy(cfg.raw)
    raw["features"]["sequence_channels"]["mode"] = "contemporeneous"  # plausible typo
    broken = Config(raw=raw, path=cfg.path)

    with pytest.raises(ValueError, match="contemporaneous"):
        sequence_channel_columns(frame, broken)


# ---------------------------------------------------------------------------
# Gap injection: the controls the ablation's validity rests on
#
# The experiment claims to isolate fragmentation from data volume. That claim is
# only true if the two arms remove identical hour counts, the test period is
# never touched, and a zero-strength injection changes nothing. Each is asserted
# here rather than trusted.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def injectable() -> tuple[pd.DataFrame, GapProfile]:
    """A near-complete hourly record plus a heavy-tailed gap profile."""
    index = pd.date_range("2020-01-01", periods=8760, freq="1h", tz="UTC")
    values = 50.0 + 30.0 * np.sin(np.arange(len(index)) / 24.0)
    values[100:104] = np.nan  # a small pre-existing outage
    frame = pd.DataFrame({"pm25": values}, index=index)
    frame.index.name = "datetime_utc"
    # Mimics a real monitoring record: mostly single hours, occasional long ones.
    profile = GapProfile(lengths=np.array([1] * 60 + [2, 3, 5, 12] * 5 + [72, 200]), source="test")
    return frame, profile


def test_zero_strength_injection_is_a_no_op(injectable):
    """arm='none', and any target at or above current coverage, must change nothing."""
    frame, profile = injectable
    protect = frame.index[6000]

    for arm, coverage in (("none", 1.0), ("fragmented", 1.0), ("contiguous", 1.0)):
        out, report = inject_gaps(
            frame,
            "pm25",
            arm=arm,
            target_coverage=coverage,
            profile=profile,
            protect_from=protect,
            seed=42,
        )
        assert report.hours_removed == 0, f"{arm} removed hours it should not have"
        assert out.equals(frame), f"{arm} altered the record at zero strength"


def test_injection_never_touches_the_protected_period(injectable):
    """The test period must survive intact however hard the record is degraded.

    Every cell in the ablation is scored on these rows. If degradation reached
    them, RMSE would stop being comparable across coverage levels and the whole
    grid would be measuring two things at once.
    """
    frame, profile = injectable
    protect = frame.index[6000]

    for arm in ("fragmented", "contiguous"):
        out, report = inject_gaps(
            frame,
            "pm25",
            arm=arm,
            target_coverage=0.40,
            profile=profile,
            protect_from=protect,
            seed=7,
        )
        assert report.hours_removed > 0, "the degradation did nothing, so this proves nothing"
        before = frame.loc[frame.index >= protect, "pm25"]
        after = out.loc[out.index >= protect, "pm25"]
        assert before.equals(after), f"{arm} degraded the protected period"


def test_both_arms_remove_identical_hour_counts(injectable):
    """Volume must be held constant so the arms differ only in arrangement.

    This is the control that separates fragmentation from having less data. If
    the counts diverge, the arm gap stops being interpretable.
    """
    frame, profile = injectable
    protect = frame.index[6000]

    for coverage in (0.95, 0.90, 0.82, 0.75):
        for seed in (42, 1337):
            _, fragmented = inject_gaps(
                frame,
                "pm25",
                arm="fragmented",
                target_coverage=coverage,
                profile=profile,
                protect_from=protect,
                seed=seed,
            )
            _, contiguous = inject_gaps(
                frame,
                "pm25",
                arm="contiguous",
                target_coverage=coverage,
                profile=profile,
                protect_from=protect,
                seed=seed,
            )
            assert fragmented.hours_removed == contiguous.hours_removed, (
                f"arms removed different hour counts at coverage {coverage}, seed {seed}: "
                f"{fragmented.hours_removed} vs {contiguous.hours_removed}"
            )


def test_fragmented_arm_breaks_contiguity_far_more_than_contiguous(injectable):
    """The manipulated variable must actually differ between arms.

    Equal hour counts alone would be satisfied by two identical arms; the point
    is that one shatters the record and the other does not.
    """
    frame, profile = injectable
    protect = frame.index[6000]
    kwargs = {"target_coverage": 0.82, "profile": profile, "protect_from": protect, "seed": 42}

    _, fragmented = inject_gaps(frame, "pm25", arm="fragmented", **kwargs)
    _, contiguous = inject_gaps(frame, "pm25", arm="contiguous", **kwargs)

    assert fragmented.n_gaps_after > 3 * contiguous.n_gaps_after, (
        f"the arms did not differ in contiguity: {fragmented.n_gaps_after} gaps "
        f"vs {contiguous.n_gaps_after}"
    )


def test_injection_is_reproducible_from_its_seed(injectable):
    """Same seed, same degradation; different seed, different degradation."""
    frame, profile = injectable
    protect = frame.index[6000]
    kwargs = {
        "arm": "fragmented",
        "target_coverage": 0.82,
        "profile": profile,
        "protect_from": protect,
    }

    first, _ = inject_gaps(frame, "pm25", seed=42, **kwargs)
    again, _ = inject_gaps(frame, "pm25", seed=42, **kwargs)
    other, _ = inject_gaps(frame, "pm25", seed=43, **kwargs)

    assert first.equals(again), "the injector is not reproducible from its seed"
    assert not first.equals(other), "different seeds produced identical degradation"


def test_fragmented_arm_requires_a_profile(injectable):
    """Silently falling back to some default distribution would be a hidden choice."""
    frame, _ = injectable
    with pytest.raises(ValueError, match="profile"):
        inject_gaps(frame, "pm25", arm="fragmented", target_coverage=0.8, profile=None)


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


# ---------------------------------------------------------------------------
# Resume must not reuse runs from a different input width
# ---------------------------------------------------------------------------


def _run_record(**over: object) -> dict:
    """Build a minimal completed tier3 run record."""
    base = {
        "model": "gru_h64_l1",
        "tier": "tier3",
        "variant": "w48",
        "horizon_h": 24,
        "seed": 42,
        "completed": True,
        "n_features": 18,
        "metrics": {"rmse": 1.0},
    }
    base.update(over)
    return base


def test_resume_reuses_a_run_at_the_same_input_width():
    """A completed run at the current width is reused rather than retrained."""
    from src.results import find_reusable_run

    payload = {"runs": [_run_record()]}
    found = find_reusable_run(
        payload, model="gru_h64_l1", variant="w48", horizon_h=24, seed=42, n_features=18
    )
    assert found is not None


@pytest.mark.leakage
def test_control_resume_refuses_a_run_from_a_different_input_width():
    """Negative control: a 102-channel record must NOT satisfy an 18-channel sweep.

    This is the defect that mixed two experiments in one results.json. Resume
    consults the results file, not the checkpoints, so clearing checkpoints does
    not protect against it.
    """
    from src.results import find_reusable_run, stale_width_runs

    payload = {"runs": [_run_record(n_features=102, n_params=32321)]}
    assert (
        find_reusable_run(
            payload, model="gru_h64_l1", variant="w48", horizon_h=24, seed=42, n_features=18
        )
        is None
    )
    assert len(stale_width_runs(payload, "tier3", 18)) == 1


@pytest.mark.leakage
def test_control_resume_refuses_a_record_predating_width_tracking():
    """Negative control: a record with no n_features is treated as stale."""
    from src.results import find_reusable_run

    record = _run_record()
    del record["n_features"]
    assert (
        find_reusable_run(
            {"runs": [record]},
            model="gru_h64_l1",
            variant="w48",
            horizon_h=24,
            seed=42,
            n_features=18,
        )
        is None
    )


# ---------------------------------------------------------------------------
# The build shim
# ---------------------------------------------------------------------------


def test_make_shim_does_not_declare_a_powershell_automatic_variable():
    """No make.ps1 parameter may shadow a PowerShell automatic variable.

    ``param([string[]]$Args)`` is accepted by PowerShell and lands in
    ``$PSBoundParameters``, but ``$Args`` read by name always returns the
    *automatic* variable -- empty, since everything bound to a declared
    parameter. ``& $Py $path @Args`` therefore splatted nothing and every step
    ran bare, falling back to its argparse ``--config`` default. That silently
    made ``donors`` re-run the primary city and ``beijing`` write Dhaka results
    into Beijing's slot, and it hid for the whole project because those defaults
    match the ``all`` target.
    """
    import re
    from pathlib import Path

    reserved = {
        "args",
        "input",
        "error",
        "host",
        "home",
        "matches",
        "pid",
        "profile",
        "psitem",
        "pwd",
        "this",
    }
    shim = Path(__file__).resolve().parents[1] / "make.ps1"
    declared = re.findall(
        r"\$(\w+)\s*(?:=|,|\))",
        "\n".join(re.findall(r"param\((.*?)\)", shim.read_text(encoding="utf-8"), re.S)),
    )
    offenders = sorted({d for d in declared if d.lower() in reserved})
    assert not offenders, (
        f"make.ps1 declares parameter(s) {offenders} that shadow PowerShell "
        "automatic variables; they bind but read back empty"
    )


@pytest.mark.leakage
def test_selection_stability_selector_never_sees_a_test_metric():
    """The tier-3 selector must rank on validation loss alone.

    ``19_selection_stability.py`` reports selection *regret*, which needs test
    RMSE, next to the selection rule itself. That adjacency is where a later
    edit would quietly start ranking on test error -- the exact circularity
    ``_best_per_horizon`` is built to avoid. So the selector is handed a frame
    whose test column is poisoned to invert the ranking: if the choice moves,
    test error has entered the selection path.
    """
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts" / "19_selection_stability.py"
    spec = importlib.util.spec_from_file_location("_sel_stability", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    frame = pd.DataFrame(
        {
            "candidate": ["good", "good", "bad", "bad"],
            "seed": [1, 2, 1, 2],
            "val_loss": [0.10, 0.12, 0.30, 0.32],
            "test_rmse": [99.0, 99.0, 1.0, 1.0],  # inverted on purpose
        }
    )
    assert module._select(frame, [1, 2]) == "good"
    assert module._select(frame, [1]) == "good"


@pytest.mark.leakage
def test_control_selector_would_flip_if_it_ranked_on_test():
    """Negative control: the poisoned frame really does invert under test ranking."""
    frame = pd.DataFrame(
        {
            "candidate": ["good", "good", "bad", "bad"],
            "val_loss": [0.10, 0.12, 0.30, 0.32],
            "test_rmse": [99.0, 99.0, 1.0, 1.0],
        }
    )
    assert frame.groupby("candidate")["val_loss"].mean().idxmin() == "good"
    assert frame.groupby("candidate")["test_rmse"].mean().idxmin() == "bad"


# ---------------------------------------------------------------------------
# Lookback cap and the decoupled history floor
# ---------------------------------------------------------------------------


def _capped(cfg: Config, cap: int | None, floor: int | None = None) -> Config:
    """A copy of the config with the lookback cap and history floor set."""
    raw = copy.deepcopy(cfg.raw)
    raw["features"]["lookback_h"] = cap
    raw["features"]["history_floor_h"] = floor
    return Config(raw=raw, path=cfg.path)


@pytest.mark.leakage
def test_lookback_cap_drops_the_columns_it_bounds(cfg, synthetic):
    """A cap must remove the offending columns, not merely rename the bound.

    Keeping ``pm25_lag_168`` while admitting rows that carry only 48 h of
    history would hand the model a NaN, and ``get_split_arrays`` replaces NaN
    with 0.0 *after* scaling -- with the training mean, silently.
    """
    pm, met = synthetic
    variant = _capped(cfg, 48)
    frame, _ = build_features(pm, met, variant, LOG)
    predictors = set(feature_columns(frame, variant))
    for name in ("pm25_lag_168", "pm25_roll168_mean", "pm25_roll168_std"):
        assert name not in predictors
        assert name not in frame.columns
    assert "pm25_lag_48" in predictors


@pytest.mark.leakage
def test_capped_lookbacks_drive_every_reader(cfg, synthetic):
    """The built columns and the generated names must agree at every cap.

    ``derived_history_columns`` exists so the two cannot drift; the cap must be
    applied in one place or that guarantee is lost.
    """
    pm, met = synthetic
    for cap in (None, 48, 24, 12):
        variant = _capped(cfg, cap)
        frame, _ = build_features(pm, met, variant, LOG)
        derived = derived_history_columns(variant, frame)
        built = {c for c in frame.columns if "_lag_" in c or "_roll" in c or "_diff_" in c}
        assert built <= derived, f"cap={cap}: built columns absent from the generated set"
        deepest = max(int(c.rsplit("_", 1)[1]) for c in built if c.startswith("pm25_lag_"))
        assert deepest <= max_backward_dependency(variant)


@pytest.mark.leakage
def test_history_floor_defaults_to_max_backward_dependency(cfg, synthetic):
    """With both keys null the build must be exactly what it was before."""
    from src.features.build_features import history_floor

    assert history_floor(cfg) == max_backward_dependency(cfg)
    pm, met = synthetic
    base, _ = build_features(pm, met, cfg, LOG)
    same, _ = build_features(pm, met, _capped(cfg, None, None), LOG)
    for h in cfg.get("task.horizons_h"):
        assert np.array_equal(base[f"valid_h{h}"].to_numpy(), same[f"valid_h{h}"].to_numpy())


@pytest.mark.leakage
def test_control_history_floor_below_max_lag_raises(cfg):
    """Negative control: the forbidden combination must be refused."""
    from src.features.build_features import history_floor

    with pytest.raises(ConfigError, match="below the deepest configured feature"):
        history_floor(_capped(cfg, None, 48))


@pytest.mark.leakage
def test_control_short_floor_feeds_the_train_mean_to_the_model(cfg, synthetic):
    """Negative control: what the guard above actually prevents.

    Build the forbidden combination by hand, bypassing ``history_floor``, and
    show the deepest lag reaching a row that cannot support it. The value is
    NaN, and every consumer replaces NaN with zero after scaling -- so the model
    receives the training mean dressed as an observation.
    """
    pm, met = synthetic
    frame, _ = build_features(pm, met, cfg, LOG)
    deep = frame["pm25_lag_168"]
    shallow_rows = frame["pos_in_run"].between(48, 167)
    assert shallow_rows.any(), "fixture must contain rows between the two floors"
    assert deep[shallow_rows].isna().all(), "the deepest lag is undefined on those rows"
    assert float(np.nan_to_num(deep[shallow_rows].to_numpy(), nan=0.0).sum()) == 0.0


@pytest.mark.leakage
def test_arm_b_row_set_equals_arm_c_row_set(cfg, gapped):
    """The volume control must reproduce the status quo rows exactly.

    Arm B caps the feature set but holds the floor at the status quo, so it
    differs from arm C in features alone. If the row sets were merely close, the
    A-vs-B and B-vs-C differences would not decompose the A-vs-C difference.
    """
    pm, met = gapped
    deepest = max_backward_dependency(cfg)
    arm_c, _ = build_features(pm, met, cfg, LOG)
    arm_b, _ = build_features(pm, met, _capped(cfg, 48, deepest), LOG)
    arm_a, _ = build_features(pm, met, _capped(cfg, 48), LOG)
    for h in cfg.get("task.horizons_h"):
        assert np.array_equal(arm_c[f"valid_h{h}"].to_numpy(), arm_b[f"valid_h{h}"].to_numpy())
    assert int(arm_a["valid_h24"].sum()) > int(arm_c["valid_h24"].sum())


@pytest.mark.leakage
def test_arm_b_sequence_channels_equal_arm_c(cfg, synthetic):
    """A lookback cap must not change the sequence tier input at all.

    ``sequence_channel_columns`` excludes every derived-history column, so the
    cap can only move the validity floor for tier 3. This is what licenses
    reusing arm C sequence runs for arm B instead of retraining them.
    """
    pm, met = synthetic
    frame_c, _ = build_features(pm, met, cfg, LOG)
    variant = _capped(cfg, 24)
    frame_a, _ = build_features(pm, met, variant, LOG)
    assert sequence_channel_columns(frame_c, cfg) == sequence_channel_columns(frame_a, variant)


# ---------------------------------------------------------------------------
# Evaluation universe and all-hours scoring
# ---------------------------------------------------------------------------


@pytest.mark.leakage
def test_universe_contains_every_arms_served_rows(cfg_fractional):
    """Every arm served set must be a subset of one fixed universe.

    Needs its own record rather than the shared ``gapped`` fixture: the universe
    is floored at 168 h, and a 1,200-hour fixture has a test split shorter than
    that, so the boundary purge would empty it.
    """
    from src.eval.availability import evaluation_universe, served_mask
    from src.features.build_features import scoreable_mask

    index = pd.date_range("2020-01-01", periods=6000, freq="1h", tz="UTC")
    n = len(index)
    values = np.arange(n, dtype=float) + 10.0
    for start in (900, 2400, 3800, 5200):
        values[start : start + 40] = np.nan  # outages the ffill limit cannot bridge
    pm = pd.DataFrame({"pm25": values}, index=index)
    pm.index.name = "datetime_utc"
    met = pd.DataFrame(
        {
            v: np.full(n, 1.0)
            for v in (
                "T2M",
                "T2MDEW",
                "RH2M",
                "WS10M",
                "WD10M",
                "PS",
                "PRECTOTCORR",
                "ALLSKY_SFC_SW_DWN",
            )
        },
        index=index,
    )
    met.index.name = "datetime_utc"

    frame, _ = build_features(pm, met, cfg_fractional, LOG)
    frame["split"] = assign_splits(frame.index, resolve_boundaries(frame.index, cfg_fractional))
    universe = evaluation_universe(frame, "test", 24)
    assert universe.sum() > 0
    for floor in (168, 48, 24, 12):
        served = served_mask(frame, "test", 24, floor)
        assert int((served & ~universe).sum()) == 0, f"floor {floor} serves outside the universe"
    assert int((universe & ~scoreable_mask(frame, 24)).sum()) == 0


@pytest.mark.leakage
def test_control_cascade_ordered_by_test_error_is_rejected():
    """Negative control: rule 6 applied to a fallback chain.

    A chain is a selection. Ordering it by anything measured on the test set --
    which is what sorting by RMSE would be -- must be refused, exactly as
    ranking candidates by test error is.
    """
    from src.eval.metrics import all_hours_skill

    n = 200
    rng = np.random.default_rng(0)
    y = rng.normal(50, 10, n)
    served = {"deep": y + 1.0, "shallow": y + 2.0, "clim": np.full(n, y.mean())}
    need = {"deep": 168, "shallow": 48, "clim": 0}
    ok = all_hours_skill(
        y, served, order=["deep", "shallow", "clim"], history_requirement=need, reference=y + 5.0
    )
    assert ok.availability == 1.0
    with pytest.raises(ValueError, match="non-increasing in history requirement"):
        all_hours_skill(
            y,
            served,
            order=["shallow", "deep", "clim"],
            history_requirement=need,
            reference=y + 5.0,
        )


@pytest.mark.leakage
def test_control_all_hours_reference_must_cover_the_universe():
    """Negative control: a denominator that moves with the arm is not one."""
    from src.eval.metrics import all_hours_skill

    n = 200
    y = np.linspace(10, 90, n)
    served = {"m": y + 1.0, "clim": np.full(n, y.mean())}
    need = {"m": 168, "clim": 0}
    reference = y + 5.0
    reference[7] = np.nan
    with pytest.raises(ValueError, match="denominator must cover the whole universe"):
        all_hours_skill(
            y, served, order=["m", "clim"], history_requirement=need, reference=reference
        )


@pytest.mark.leakage
def test_control_incomplete_cascade_is_rejected():
    """Negative control: silently dropping unanswered hours is the failure mode."""
    from src.eval.metrics import all_hours_skill

    n = 100
    y = np.linspace(10, 90, n)
    partial = y + 1.0
    partial[:20] = np.nan
    with pytest.raises(ValueError, match="unanswered"):
        all_hours_skill(
            y, {"m": partial}, order=["m"], history_requirement={"m": 168}, reference=y + 5.0
        )


# ---------------------------------------------------------------------------
# The frontier must not leak into the headline experiment
# ---------------------------------------------------------------------------


@pytest.mark.leakage
def test_every_runs_consumer_goes_through_main_runs():
    """No script may read the runs list directly; nine sites is nine to forget."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    offenders = []
    for path in sorted((root / "scripts").glob("*.py")) + sorted((root / "src").rglob("*.py")):
        if path.name == "results.py":
            continue
        text = path.read_text(encoding="utf-8")
        if '.get("runs"' in text or 'payload["runs"]' in text:
            offenders.append(path.relative_to(root).as_posix())
    assert not offenders, f"read the runs list directly instead of via main_runs: {offenders}"


@pytest.mark.leakage
def test_main_runs_is_a_no_op_without_side_experiments():
    """Tagging must not disturb records written before tagging existed."""
    from src.results import MAIN_EXPERIMENT, experiment_runs, main_runs

    payload = {"runs": [{"model": "a"}, {"model": "b", "experiment": MAIN_EXPERIMENT}]}
    assert main_runs(payload) == payload["runs"]
    assert experiment_runs(payload, "lookback_frontier") == []


@pytest.mark.leakage
def test_control_lookback_run_cannot_win_the_headline():
    """Negative control: a side-experiment record must not be selectable.

    A lookback arm trains on more rows and can post a better validation loss
    without being the headline model. If it reached ``_best_per_horizon`` it
    would silently replace the reported result.
    """
    from src.results import main_runs

    payload = {
        "runs": [
            {"model": "gru_h64_l1", "variant": "w24", "horizon_h": 24, "seed": 42, "metrics": {}},
            {
                "model": "gru_h64_l1",
                "variant": "w24|R48|A",
                "horizon_h": 24,
                "seed": 42,
                "experiment": "lookback_frontier",
                "metrics": {},
            },
        ]
    }
    selected = main_runs(payload)
    assert len(selected) == 1
    assert selected[0]["variant"] == "w24"


# ---------------------------------------------------------------------------
# Ablation: the control, the timezone, and the two new tests
# ---------------------------------------------------------------------------


@pytest.mark.leakage
def test_climatology_bins_in_the_configured_timezone():
    """Hour bins must follow ``features.calendar_tz``, not a hardcoded city.

    This was hardcoded to Asia/Dhaka, which put every Beijing record bins two
    hours out -- including all 303 cells of the gap-injection grid, where
    climatology represents a reported family.
    """
    from src.models.baselines import ClimatologyModel, predict_climatology

    index = pd.date_range("2020-01-01", periods=240, freq="1h", tz="UTC")
    arrays = type("A", (), {"index": index})()
    table = {(h, m): float(h) for h in range(24) for m in range(1, 13)}
    dhaka = ClimatologyModel(table=table, global_mean=0.0, tz="Asia/Dhaka")
    shanghai = ClimatologyModel(table=table, global_mean=0.0, tz="Asia/Shanghai")
    a = predict_climatology(dhaka, arrays, 24)
    b = predict_climatology(shanghai, arrays, 24)
    assert not np.array_equal(a, b), "two timezones must not give identical bins"
    assert np.array_equal((a + 2.0) % 24.0, b % 24.0)


@pytest.mark.leakage
def test_control_climatology_cannot_predict_in_another_timezone():
    """Negative control: a fit/predict timezone mismatch must be unrepresentable."""
    import inspect

    from src.models.baselines import fit_climatology, predict_climatology

    assert "cfg" not in inspect.signature(predict_climatology).parameters, (
        "predict_climatology must take no config; the timezone rides on the model "
        "so it cannot disagree with the one the table was binned in"
    )
    assert "cfg" in inspect.signature(fit_climatology).parameters


@pytest.mark.leakage
def test_family_contrast_is_paired_within_units():
    """The contrast must lose its signal when the pairing is destroyed."""
    from src.eval.ablation import family_contrasts

    rng = np.random.default_rng(0)
    rows = []
    for cov in (0.95, 0.90, 0.85, 0.82, 0.75):
        for seed in range(10):
            shared = rng.normal(0, 0.05)
            rows.append(
                {
                    "family": "sequence",
                    "target_coverage": cov,
                    "injection_seed": seed,
                    "arm_gap": shared - 0.03,
                }
            )
            rows.append(
                {
                    "family": "trees",
                    "target_coverage": cov,
                    "injection_seed": seed,
                    "arm_gap": shared,
                }
            )
    paired = pd.DataFrame(rows)
    kept = family_contrasts(paired)
    kept_row = kept[kept["family"] == "trees"].iloc[0]
    assert float(kept_row["p_wilcoxon"]) < 0.001

    shuffled = paired.copy()
    mask = shuffled["family"] == "trees"
    shuffled.loc[mask, "arm_gap"] = rng.permutation(shuffled.loc[mask, "arm_gap"].to_numpy())
    broken = family_contrasts(shuffled)
    broken_row = broken[broken["family"] == "trees"].iloc[0]
    kept_width = float(kept_row["ci_high"]) - float(kept_row["ci_low"])
    broken_width = float(broken_row["ci_high"]) - float(broken_row["ci_low"])
    assert broken_width > kept_width, "breaking the pairing must widen the interval"


@pytest.mark.leakage
def test_dose_response_recovers_a_planted_slope():
    """A known slope in the dose must come back inside the interval."""
    from src.eval.ablation import dose_response

    slope = 0.4
    rows = [
        {
            "family": "sequence",
            "target_coverage": cov,
            "injection_seed": seed,
            "arm_gap": slope * (cov - 1.0) + 0.001 * seed,
        }
        for cov in (0.95, 0.90, 0.85, 0.82, 0.75)
        for seed in range(10)
    ]
    out = dose_response(pd.DataFrame(rows))
    row = out[out["family"] == "sequence"].iloc[0]
    assert row["slope_per_unit_dose"] == pytest.approx(slope, rel=1e-6)
    assert row["ci_low"] <= row["gap_change_per_10pp_lost"] <= row["ci_high"]
    # Reported per 10 percentage points LOST, so a positive slope reads negative.
    assert row["gap_change_per_10pp_lost"] == pytest.approx(-0.04, rel=1e-6)


@pytest.mark.leakage
def test_both_configs_declare_the_lookback_keys_as_null():
    """Any key added to one config is inherited by the other with a wrong value.

    That has caused four defects in this repository. Both new keys are
    city-agnostic and null is correct everywhere, so the hazard is structurally
    absent -- and stays absent only if something checks.
    """
    from pathlib import Path

    import yaml

    root = Path(__file__).resolve().parents[1]
    paths = [root / "config.yaml", root / "config_beijing.yaml"]
    paths += sorted((root / "config" / "donors").glob("*.yaml"))
    assert len(paths) >= 3
    for path in paths:
        features = yaml.safe_load(path.read_text(encoding="utf-8"))["features"]
        assert "lookback_h" in features, f"{path.name} is missing features.lookback_h"
        assert "history_floor_h" in features, f"{path.name} is missing features.history_floor_h"
        assert features["lookback_h"] is None, f"{path.name} pins a lookback cap"
        assert features["history_floor_h"] is None, f"{path.name} pins a history floor"


# ---------------------------------------------------------------------------
# Resume identity: a grid must not be resumed under the wrong label
# ---------------------------------------------------------------------------


@pytest.mark.leakage
def test_resume_allows_a_matching_grid():
    """A grid recorded for this donor at this radius may be resumed."""
    from src.eval.ablation import resume_identity_problem

    payload = {"donor": "Beijing Wanliu", "sterilisation_radius_h": 192, "cells": {"a": {}}}
    assert resume_identity_problem(payload, donor="Beijing Wanliu", radius_h=192) is None


@pytest.mark.leakage
def test_resume_accepts_a_grid_predating_the_radius_field():
    """Absence of a recorded radius is not disagreement.

    Every grid written before the mediation experiment ran at the configured
    reach, so refusing them would be a false alarm that forced needless refits.
    """
    from src.eval.ablation import resume_identity_problem

    payload = {"donor": "Beijing Wanliu", "cells": {"a": {}}}
    assert resume_identity_problem(payload, donor="Beijing Wanliu", radius_h=192) is None


@pytest.mark.leakage
def test_control_resume_refuses_a_grid_from_another_donor():
    """Negative control: the defect that reported a complete grid for an untouched station."""
    from src.eval.ablation import resume_identity_problem

    payload = {"donor": "Beijing Wanliu", "sterilisation_radius_h": 192, "cells": {"a": {}}}
    problem = resume_identity_problem(payload, donor="Beijing Dingling", radius_h=192)
    assert problem is not None
    assert "Wanliu" in problem and "Dingling" in problem


@pytest.mark.leakage
def test_control_resume_refuses_a_grid_at_another_radius():
    """Negative control: mixing two backward reaches under one label.

    The mediation experiment's claim is that the reach drives the arm gap, so a
    grid that silently blended two reaches would corrupt the one result it exists
    to produce.
    """
    from src.eval.ablation import resume_identity_problem

    payload = {"donor": "Beijing Wanliu", "sterilisation_radius_h": 192, "cells": {"a": {}}}
    problem = resume_identity_problem(payload, donor="Beijing Wanliu", radius_h=48)
    assert problem is not None
    assert "192" in problem and "48" in problem


@pytest.mark.leakage
def test_empty_grid_is_always_resumable():
    """With no cells recorded there is nothing to mislabel."""
    from src.eval.ablation import resume_identity_problem

    assert resume_identity_problem({}, donor="anything", radius_h=1) is None
    assert (
        resume_identity_problem(
            {"donor": "other", "sterilisation_radius_h": 1, "cells": {}},
            donor="mismatched",
            radius_h=999,
        )
        is None
    )


@pytest.mark.leakage
def test_every_mediation_config_has_a_distinct_output_and_matching_radius():
    """Two grids sharing an output path is the defect this repo hits most often.

    Here it would be worst: resume keys on (arm, coverage, seed) and knows
    nothing about the reach, so a shared path makes every cell read as already
    recorded and the run reports a complete grid for a radius it never ran.
    """
    from pathlib import Path

    import yaml

    root = Path(__file__).resolve().parents[1]
    configs = sorted((root / "config" / "lookback").glob("*.yaml"))
    if not configs:
        pytest.skip("mediation configs not generated")

    seen: dict[str, str] = {}
    for path in configs:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        gap = raw["ablation"]["gap_injection"]
        name = str(gap["output_name"])
        assert name not in seen, f"{path.name} shares {name} with {seen[name]}"
        seen[name] = path.name

        windows = {int(s["window_h"]) for s in gap["sequence_models"]}
        assert len(windows) == 1, f"{path.name} mixes windows {sorted(windows)}"
        lookback = int(raw["features"]["lookback_h"])
        radius = max(lookback, next(iter(windows)) - 1) + int(gap["horizon_h"])
        assert f"_R{radius}." in name, (
            f"{path.name} resolves to radius {radius} h but writes to {name}"
        )
