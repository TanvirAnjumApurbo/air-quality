"""Shared data access for every model tier.

One loader serves baselines, trees and sequence models so that all three are
scored on exactly the same rows. If the tiers selected rows independently, a
comparison between them would silently be a comparison of different test sets.

Sequence batching keeps the **whole scaled feature matrix** resident (79,216 x
102 float32, about 32 MB) and gathers windows by index arithmetic rather than
materialising a windowed tensor. Materialising would cost roughly 2.7 GB at a
168-hour window for no benefit; at 32 MB the matrix fits in GPU memory outright,
so batches are assembled on-device with no host transfer in the training loop.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.eval.split import inverse_transform_target, transform_target
from src.features.build_features import feature_columns
from src.utils import Config


@dataclass
class SplitArrays:
    """Feature matrix, target vector and metadata for one split and horizon.

    Attributes:
        x: Scaled predictors, shape ``(n, n_features)``.
        y: Target in ug/m3, shape ``(n,)``.
        y_transformed: Target on the modelling scale.
        index: UTC timestamps of each row.
        persistence: The value at time t, i.e. the persistence forecast.
        seasonal_naive: The value at t+h-24, i.e. the seasonal-naive forecast.
        month_local: Local calendar month, for seasonal stratification.
        hour_local: Local hour of day.
        feature_names: Column names of ``x``.
    """

    x: np.ndarray
    y: np.ndarray
    y_transformed: np.ndarray
    index: pd.DatetimeIndex
    persistence: np.ndarray
    seasonal_naive: np.ndarray
    month_local: np.ndarray
    hour_local: np.ndarray
    feature_names: list[str]

    def __len__(self) -> int:
        """Number of rows."""
        return int(self.x.shape[0])


def load_features(cfg: Config) -> pd.DataFrame:
    """Load the built feature matrix.

    Args:
        cfg: Loaded configuration.

    Returns:
        The feature frame indexed by UTC timestamp.

    Raises:
        FileNotFoundError: If Phase 2 has not been run.
    """
    path = cfg.path_for("data_processed") / "features.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found -- run scripts/04_build_features.py first")
    return pd.read_parquet(path)


def load_meta(cfg: Config) -> dict[str, Any]:
    """Load the feature-build metadata, including split boundaries.

    Args:
        cfg: Loaded configuration.

    Returns:
        The metadata mapping.
    """
    path = cfg.path_for("data_processed") / "features_meta.json"
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_scaler_params(cfg: Config) -> tuple[pd.Series, pd.Series]:
    """Load the train-fitted standardisation parameters.

    Args:
        cfg: Loaded configuration.

    Returns:
        ``(mean, scale)`` indexed by feature name.
    """
    payload = json.loads(
        (cfg.path_for("data_processed") / "scaler.json").read_text(encoding="utf-8")
    )
    return pd.Series(payload["mean"]), pd.Series(payload["scale"])


def get_split_arrays(
    frame: pd.DataFrame,
    cfg: Config,
    horizon: int,
    split: str,
    *,
    include_oracle: bool = False,
    scaled: bool = True,
) -> SplitArrays:
    """Extract the model-ready arrays for one split and horizon.

    Only rows flagged valid for this horizon are returned, so the gap-aware and
    boundary-purge rules are applied identically for every model.

    Args:
        frame: Built feature frame.
        cfg: Loaded configuration.
        horizon: Forecast horizon in hours.
        split: ``"train"``, ``"val"`` or ``"test"``.
        include_oracle: Include the labelled perfect-forecast columns.
        scaled: Apply the train-fitted standardisation.

    Returns:
        The extracted arrays.
    """
    target_col = str(cfg.get("features.target"))
    mask = (frame["split"] == split) & frame[f"valid_h{horizon}"]
    sub = frame.loc[mask]

    names = feature_columns(frame, cfg, include_oracle=include_oracle)
    if include_oracle:
        # Keep only the oracle columns matching this horizon; other horizons'
        # oracle columns are irrelevant and would inflate the feature count.
        names = [c for c in names if not c.startswith("oracle_") or c.endswith(f"_h{horizon}")]

    x = sub[names].to_numpy(dtype=np.float32)
    if scaled:
        mean, scale = load_scaler_params(cfg)
        common = [c for c in names if c in mean.index]
        idx = [names.index(c) for c in common]
        x[:, idx] = (x[:, idx] - mean[common].to_numpy(dtype=np.float32)) / scale[common].to_numpy(
            dtype=np.float32
        )

    y = sub[f"target_h{horizon}"].to_numpy(dtype=np.float64)
    transform = str(cfg.get("scaling.target_transform"))

    # Persistence uses the observation at t.
    persistence = sub[target_col].to_numpy(dtype=np.float64)

    # Seasonal-naive is y(t+h-24), i.e. the same clock hour one day earlier.
    # That is a lag of (24 - h) from t, which is in the past for every h <= 24
    # and so is a legitimate forecast.
    #
    # Computed by shifting the full series rather than reading a precomputed
    # pm25_lag_k column: the required lags (23, 21, 18, 12, 0) are mostly absent
    # from features.pm25_lags_h, and looking them up there silently yielded NaN.
    #
    # Note h == 24 gives a lag of 0, so seasonal-naive and persistence coincide
    # exactly at the headline horizon. That is a property of the two definitions,
    # not a duplicated row, and it is stated in the results narrative.
    lag_for_seasonal = 24 - horizon
    if lag_for_seasonal >= 0:
        shifted = frame[target_col].shift(lag_for_seasonal)
        seasonal = shifted.loc[mask].to_numpy(dtype=np.float64)
    else:
        seasonal = np.full(len(sub), np.nan)

    return SplitArrays(
        x=np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0),
        y=y,
        y_transformed=transform_target(y, transform),
        index=sub.index,
        persistence=persistence,
        seasonal_naive=seasonal,
        month_local=sub["month_local"].to_numpy(dtype=int),
        hour_local=sub["hour_local"].to_numpy(dtype=int),
        feature_names=names,
    )


def invert(cfg: Config, values: np.ndarray) -> np.ndarray:
    """Map model output back to ug/m3.

    Args:
        cfg: Loaded configuration.
        values: Predictions on the modelling scale.

    Returns:
        Predictions in ug/m3, floored at zero since a negative concentration is
        not physical.
    """
    out = inverse_transform_target(
        np.asarray(values, dtype=float), str(cfg.get("scaling.target_transform"))
    )
    return np.clip(out, 0.0, None)


# ---------------------------------------------------------------------------
# Sequence windowing
# ---------------------------------------------------------------------------


@dataclass
class SequenceIndex:
    """Index-based description of a windowed sequence dataset.

    No windowed tensor is materialised. ``end_positions`` gives the row of the
    full matrix at which each window *ends*; a window is then
    ``matrix[end - window + 1 : end + 1]``.

    Attributes:
        matrix: Full scaled feature matrix for the whole series.
        end_positions: Row index at which each usable window ends.
        y: Target in ug/m3 per window.
        y_transformed: Target on the modelling scale per window.
        index: UTC timestamp of each window end.
        persistence: Persistence forecast per window.
        month_local: Local month per window.
        window: Window length in hours.
        n_features: Number of features per timestep.
    """

    matrix: np.ndarray
    end_positions: np.ndarray
    y: np.ndarray
    y_transformed: np.ndarray
    index: pd.DatetimeIndex
    persistence: np.ndarray
    month_local: np.ndarray
    window: int
    n_features: int

    def __len__(self) -> int:
        """Number of usable windows."""
        return int(self.end_positions.size)

    def nbytes_materialised(self) -> int:
        """Bytes a fully-materialised window tensor would occupy."""
        return int(len(self) * self.window * self.n_features * 4)


def build_sequence_index(
    frame: pd.DataFrame,
    cfg: Config,
    horizon: int,
    window: int,
    split: str,
) -> SequenceIndex:
    """Build the window index for one split, horizon and window length.

    A window is kept only when its entire span lies inside a single gap-free run
    *and* the row is already flagged valid for this horizon. Requiring both means
    a window can never straddle a data gap or a split boundary.

    Args:
        frame: Built feature frame.
        cfg: Loaded configuration.
        horizon: Forecast horizon in hours.
        window: Input window length in hours.
        split: Split name.

    Returns:
        The window index.
    """
    names = feature_columns(frame, cfg, include_oracle=False)
    matrix = frame[names].to_numpy(dtype=np.float32)
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)

    mean, scale = load_scaler_params(cfg)
    common = [c for c in names if c in mean.index]
    idx = [names.index(c) for c in common]
    matrix[:, idx] = (matrix[:, idx] - mean[common].to_numpy(dtype=np.float32)) / scale[
        common
    ].to_numpy(dtype=np.float32)

    valid = (frame["split"] == split) & frame[f"valid_h{horizon}"]
    pos_in_run = frame["pos_in_run"].to_numpy()
    # The window reaches back window-1 steps; that whole span must sit inside the
    # same run, which pos_in_run >= window-1 guarantees.
    deep_enough = pos_in_run >= (window - 1)
    keep = valid.to_numpy() & deep_enough

    positions = np.flatnonzero(keep)
    transform = str(cfg.get("scaling.target_transform"))
    y = frame[f"target_h{horizon}"].to_numpy(dtype=np.float64)[positions]

    return SequenceIndex(
        matrix=matrix,
        end_positions=positions.astype(np.int64),
        y=y,
        y_transformed=transform_target(y, transform),
        index=frame.index[positions],
        persistence=frame[str(cfg.get("features.target"))].to_numpy(dtype=np.float64)[positions],
        month_local=frame["month_local"].to_numpy(dtype=int)[positions],
        window=int(window),
        n_features=len(names),
    )
