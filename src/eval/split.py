"""Chronological splitting and train-only scaling.

Two invariants live here, and both raise rather than warn:

* **Split before fitting anything.** The split is by time, never by shuffling.
  Any request to shuffle is a configuration error, not an option.
* **Scalers see training data only.** :class:`TrainOnlyScaler` refuses to
  ``transform`` before ``fit``, and ``fit`` refuses any frame that is not the
  training slice.

Boundary dates are resolved once from the configured fractions and written back
into ``config.yaml`` so that every table caption can print the exact dates that
produced it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

from src.utils import Config, ConfigError

SplitName = Literal["train", "val", "test"]


@dataclass(frozen=True)
class SplitBoundaries:
    """Resolved chronological split boundaries.

    Attributes:
        train_end: Last timestamp belonging to train, inclusive.
        val_end: Last timestamp belonging to validation, inclusive.
        train_start: First timestamp in the dataset.
        test_end: Last timestamp in the dataset.
    """

    train_start: pd.Timestamp
    train_end: pd.Timestamp
    val_end: pd.Timestamp
    test_end: pd.Timestamp

    def caption(self) -> str:
        """One-line description for table and figure captions."""
        return (
            f"train {self.train_start:%Y-%m-%d} to {self.train_end:%Y-%m-%d}, "
            f"val to {self.val_end:%Y-%m-%d}, "
            f"test to {self.test_end:%Y-%m-%d} (chronological, no shuffling)"
        )

    def to_dict(self) -> dict[str, str]:
        """Serialise boundaries as ISO date strings."""
        return {
            "train_start": self.train_start.isoformat(),
            "train_end": self.train_end.isoformat(),
            "val_end": self.val_end.isoformat(),
            "test_end": self.test_end.isoformat(),
        }


def resolve_boundaries(index: pd.DatetimeIndex, cfg: Config) -> SplitBoundaries:
    """Determine split boundary timestamps.

    Uses explicit boundaries from config when present, otherwise derives them
    from the configured fractions over the *time span*, not over row counts, so
    that gaps in coverage cannot shift a boundary.

    Args:
        index: Sorted UTC index of the full dataset.
        cfg: Loaded configuration.

    Returns:
        The resolved boundaries.

    Raises:
        ConfigError: If shuffling is requested or the fractions are invalid.
    """
    if bool(cfg.get("split.shuffle", False)):
        raise ConfigError("split.shuffle must be false: this is a time series.")

    explicit = cfg.get("split.explicit_boundaries", {}) or {}
    start, end = index.min(), index.max()

    if explicit.get("train_end") and explicit.get("val_end"):
        train_end = pd.Timestamp(explicit["train_end"], tz="UTC")
        val_end = pd.Timestamp(explicit["val_end"], tz="UTC")
    else:
        train_frac = float(cfg.get("split.train_frac"))
        val_frac = float(cfg.get("split.val_frac"))
        test_frac = float(cfg.get("split.test_frac"))
        total = train_frac + val_frac + test_frac
        if not np.isclose(total, 1.0):
            raise ConfigError(f"split fractions sum to {total}, expected 1.0")
        span = end - start
        train_end = start + span * train_frac
        val_end = start + span * (train_frac + val_frac)

    if not start < train_end < val_end < end:
        raise ConfigError(
            f"split boundaries are not strictly increasing within the data span: "
            f"{start} < {train_end} < {val_end} < {end}"
        )
    return SplitBoundaries(train_start=start, train_end=train_end, val_end=val_end, test_end=end)


def assign_splits(index: pd.DatetimeIndex, boundaries: SplitBoundaries) -> pd.Series:
    """Label every timestamp with its split.

    Args:
        index: UTC index.
        boundaries: Resolved boundaries.

    Returns:
        Series of ``"train"``/``"val"``/``"test"`` labels.
    """
    labels = np.where(
        index <= boundaries.train_end,
        "train",
        np.where(index <= boundaries.val_end, "val", "test"),
    )
    return pd.Series(labels, index=index, name="split")


def purge_boundary_rows(
    frame: pd.DataFrame, split: pd.Series, max_lag_h: int, horizon_h: int
) -> pd.Series:
    """Drop rows whose feature window or target crosses a split boundary.

    Without this, the first rows of validation carry lag features drawn from
    training hours, and the last training rows carry targets that fall inside
    validation. Both blur the boundary the split exists to enforce.

    Args:
        frame: Feature frame indexed by UTC timestamp.
        split: Split labels aligned to ``frame``.
        max_lag_h: Longest backward dependency in the feature set.
        horizon_h: Forecast horizon in hours.

    Returns:
        Boolean mask, True for rows safe to keep.
    """
    index = frame.index
    keep = pd.Series(True, index=index)

    for name in ("val", "test"):
        block = index[split == name]
        if len(block) == 0:
            continue
        # A row needs max_lag_h hours of history; anything closer than that to
        # the start of its split would reach back across the boundary.
        keep &= ~((split == name) & (index < block.min() + pd.Timedelta(hours=max_lag_h)))

    for name in ("train", "val"):
        block = index[split == name]
        if len(block) == 0:
            continue
        # A row's target sits horizon_h hours ahead; anything closer than that to
        # the end of its split would reach forward across the boundary.
        keep &= ~((split == name) & (index > block.max() - pd.Timedelta(hours=horizon_h)))

    return keep


class TrainOnlyScaler:
    """Standardiser that structurally cannot be fitted on held-out data.

    ``fit`` records the split it was fitted on and rejects anything other than
    training rows; ``transform`` refuses to run before ``fit``. The guard is not
    decorative -- fitting a scaler on the full series is one of the most common
    silent leaks in time-series work.
    """

    def __init__(self, method: str = "standard") -> None:
        """Initialise the scaler.

        Args:
            method: ``"standard"`` (z-score) or ``"none"``.
        """
        self.method = method
        self.mean_: pd.Series | None = None
        self.scale_: pd.Series | None = None
        self.columns_: list[str] | None = None
        self.fitted_on_: str | None = None

    def fit(self, frame: pd.DataFrame, columns: list[str], split_name: str) -> TrainOnlyScaler:
        """Fit on the training slice only.

        Args:
            frame: Rows to fit on.
            columns: Columns to standardise.
            split_name: Name of the split these rows come from.

        Returns:
            Self.

        Raises:
            ConfigError: If ``split_name`` is not ``"train"``.
        """
        if split_name != "train":
            raise ConfigError(
                f"TrainOnlyScaler.fit called with split {split_name!r}. Scalers, "
                "imputers and encoders are fitted on train and only transform "
                "val/test."
            )
        self.columns_ = list(columns)
        if self.method == "none":
            self.mean_ = pd.Series(0.0, index=self.columns_)
            self.scale_ = pd.Series(1.0, index=self.columns_)
        else:
            sub = frame[self.columns_]
            self.mean_ = sub.mean()
            std = sub.std(ddof=0)
            # A constant column would otherwise divide by zero and produce NaN.
            self.scale_ = std.where(std > 1e-12, 1.0)
        self.fitted_on_ = split_name
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Apply the fitted standardisation.

        Args:
            frame: Rows to transform.

        Returns:
            A copy with the fitted columns standardised.

        Raises:
            ConfigError: If called before :meth:`fit`.
        """
        if self.mean_ is None or self.scale_ is None or self.columns_ is None:
            raise ConfigError("TrainOnlyScaler.transform called before fit")
        out = frame.copy()
        out[self.columns_] = (out[self.columns_] - self.mean_) / self.scale_
        return out

    def to_dict(self) -> dict[str, Any]:
        """Serialise fitted parameters for the results record."""
        if self.mean_ is None or self.scale_ is None:
            return {"method": self.method, "fitted": False}
        return {
            "method": self.method,
            "fitted": True,
            "fitted_on": self.fitted_on_,
            "n_columns": len(self.columns_ or []),
            "mean": {k: float(v) for k, v in self.mean_.items()},
            "scale": {k: float(v) for k, v in self.scale_.items()},
        }


def transform_target(values: np.ndarray, method: str) -> np.ndarray:
    """Apply the configured target transform.

    Args:
        values: Target values in ug/m3.
        method: ``"log1p"`` or ``"none"``.

    Returns:
        Transformed values.
    """
    if method == "log1p":
        return np.log1p(values)
    return values


def inverse_transform_target(values: np.ndarray, method: str) -> np.ndarray:
    """Invert :func:`transform_target`.

    All reported metrics are computed after inversion, so they are always in
    ug/m3 regardless of the scale the model was trained on.

    Args:
        values: Model outputs on the transformed scale.
        method: The transform that was applied.

    Returns:
        Values in ug/m3.
    """
    if method == "log1p":
        return np.expm1(values)
    return values
