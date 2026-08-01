"""Secondary task: next-day AQI-category classification.

Reframes the same problem as a health-advisory decision. Two properties matter
more here than raw accuracy:

* **The averaging basis must match the breakpoints.** US EPA PM2.5 breakpoints
  are defined on a 24-hour mean, so the label is the 24-hour mean ending at
  ``t+h``, not the instantaneous hourly value at ``t+h``. Applying 24-hour
  breakpoints to hourly readings is common in the literature and is simply a
  different quantity from the one the standard defines.
* **Class imbalance is handled explicitly and reported.** Dhaka spends most of
  its hours in two or three categories, so accuracy is a near-useless summary and
  macro-F1 is the headline. Class weights are applied and the resulting per-class
  recall is reported, because a classifier that never predicts "Hazardous" is
  useless as an advisory however good its accuracy looks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.eval.metrics import aqi_categories
from src.utils import Config


@dataclass
class ClassificationData:
    """Feature matrix and AQI labels for one split.

    Attributes:
        x: Scaled predictors.
        y: Integer class labels.
        y_concentration: The underlying 24-hour mean PM2.5, in ug/m3.
        index: UTC timestamps.
        labels: Class names by index.
        feature_names: Predictor names.
    """

    x: np.ndarray
    y: np.ndarray
    y_concentration: np.ndarray
    index: pd.DatetimeIndex
    labels: list[str]
    feature_names: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        """Number of rows."""
        return int(self.x.shape[0])


def build_classification_target(
    frame: pd.DataFrame, cfg: Config, horizon: int
) -> tuple[pd.Series, pd.Series]:
    """Build the AQI label series and the concentration it derives from.

    The 24-hour mean is computed with a right-aligned rolling window ending at
    ``t+h``, so it aggregates observations at and before the target time only.
    The predictor still sees nothing after ``t``.

    Args:
        frame: Built feature frame.
        cfg: Loaded configuration.
        horizon: Forecast horizon in hours.

    Returns:
        ``(labels, concentration)`` aligned to the frame index, where the value
        stamped at ``t`` describes the target period ending at ``t+h``.
    """
    target = str(cfg.get("features.target"))
    basis = str(cfg.get("classification.target_basis", "rolling_24h_mean"))
    edges = [float(e) for e in cfg.get("classification.breakpoints.edges_ugm3")]

    series = frame[target]
    concentration = (
        series.rolling(window=24, min_periods=24).mean() if basis == "rolling_24h_mean" else series
    )
    # Shift back by the horizon so the value at t describes the window at t+h.
    shifted = concentration.shift(-horizon)
    labels = pd.Series(aqi_categories(shifted.to_numpy(), edges), index=frame.index)
    labels = labels.where(shifted.notna())
    return labels, shifted


def get_classification_data(
    frame: pd.DataFrame, cfg: Config, horizon: int, split: str
) -> ClassificationData:
    """Assemble the classification arrays for one split.

    Args:
        frame: Built feature frame.
        cfg: Loaded configuration.
        horizon: Forecast horizon in hours.
        split: Split name.

    Returns:
        The assembled data.
    """
    from src.models.data import get_split_arrays

    arrays = get_split_arrays(frame, cfg, horizon, split)
    labels, concentration = build_classification_target(frame, cfg, horizon)

    aligned_labels = labels.reindex(arrays.index)
    aligned_conc = concentration.reindex(arrays.index)
    keep = aligned_labels.notna().to_numpy()

    return ClassificationData(
        x=arrays.x[keep],
        y=aligned_labels[keep].to_numpy(dtype=int),
        y_concentration=aligned_conc[keep].to_numpy(dtype=float),
        index=arrays.index[keep],
        labels=[str(v) for v in cfg.get("classification.breakpoints.labels")],
        feature_names=arrays.feature_names,
    )


def class_distribution(data: ClassificationData) -> pd.DataFrame:
    """Summarise how many observations fall in each category.

    Args:
        data: The split data.

    Returns:
        One row per class with count and share.
    """
    counts = pd.Series(data.y).value_counts().reindex(range(len(data.labels)), fill_value=0)
    return pd.DataFrame(
        {
            "class_index": list(range(len(data.labels))),
            "label": data.labels,
            "count": counts.to_numpy(),
            "share_pct": (100.0 * counts / max(counts.sum(), 1)).round(2).to_numpy(),
        }
    )


def fit_classifier(
    train: ClassificationData,
    val: ClassificationData,
    cfg: Config,
    seed: int,
    logger: Any,
) -> tuple[Any, dict[str, Any]]:
    """Fit a gradient-boosted classifier with explicit class weighting.

    Args:
        train: Training data.
        val: Validation data, concatenated with train for tuning.
        cfg: Loaded configuration.
        seed: RNG seed.
        logger: Logger.

    Returns:
        ``(fitted_estimator, info)`` where ``info`` records the weighting scheme
        and the classes actually present in training.
    """
    from sklearn.utils.class_weight import compute_sample_weight
    from xgboost import XGBClassifier

    x = np.vstack([train.x, val.x])
    y = np.concatenate([train.y, val.y])

    scheme = str(cfg.get("classification.class_weight", "balanced"))
    weights = compute_sample_weight(class_weight=scheme, y=y) if scheme else None

    present = sorted(set(y.tolist()))
    # XGBoost requires contiguous labels from zero; map to a dense range and
    # remember the mapping so predictions can be mapped back.
    remap = {original: i for i, original in enumerate(present)}
    inverse = {i: original for original, i in remap.items()}
    y_dense = np.array([remap[v] for v in y], dtype=int)

    logger.info(
        "classifier: %d rows, %d of %d classes present in train+val, weighting=%s",
        len(y),
        len(present),
        len(train.labels),
        scheme,
    )

    model = XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.06,
        subsample=0.9,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=len(present),
        random_state=seed,
        tree_method="hist",
        verbosity=0,
    )
    model.fit(x, y_dense, sample_weight=weights)

    return model, {
        "class_weight": scheme,
        "classes_present": present,
        "label_remap": remap,
        "label_inverse": inverse,
        "n_train_rows": len(y),
    }


def predict_classes(model: Any, info: dict[str, Any], x: np.ndarray) -> np.ndarray:
    """Predict original class indices.

    Args:
        model: The fitted estimator.
        info: The mapping information from :func:`fit_classifier`.
        x: Predictors.

    Returns:
        Class indices on the original label scale.
    """
    dense = model.predict(x)
    inverse = info["label_inverse"]
    return np.array([inverse[int(v)] for v in dense], dtype=int)
