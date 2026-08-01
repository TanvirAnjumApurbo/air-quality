"""Forecast evaluation metrics.

All metrics are computed in ug/m3, after any target transform has been inverted,
so a number is never reported on the scale a model happened to be trained on.

Two choices differ from common practice and are deliberate:

* **sMAPE, not MAPE.** Hourly PM2.5 approaches zero often enough that MAPE
  explodes and becomes dominated by a handful of near-zero denominators.
* **Skill score against persistence is a headline column**, not an appendix
  note. It is the number that tells a reader whether a model is worth anything
  at all: a model that cannot beat "tomorrow looks like today" has no claim.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import stats


def _clean(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Drop pairs where either value is missing or non-finite.

    Args:
        y_true: Observed values.
        y_pred: Predicted values.

    Returns:
        The finite subset of both arrays.
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    return y_true[mask], y_pred[mask]


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error, in ug/m3."""
    t, p = _clean(y_true, y_pred)
    return float(np.sqrt(np.mean((t - p) ** 2))) if t.size else float("nan")


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error, in ug/m3."""
    t, p = _clean(y_true, y_pred)
    return float(np.mean(np.abs(t - p))) if t.size else float("nan")


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination against the mean of the observed values."""
    t, p = _clean(y_true, y_pred)
    if t.size < 2:
        return float("nan")
    ss_res = float(np.sum((t - p) ** 2))
    ss_tot = float(np.sum((t - t.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Symmetric mean absolute percentage error, in percent (0-200).

    Uses the ``2|y - yhat| / (|y| + |yhat|)`` form. Pairs where both values are
    zero contribute nothing rather than producing a division by zero.
    """
    t, p = _clean(y_true, y_pred)
    if not t.size:
        return float("nan")
    denominator = np.abs(t) + np.abs(p)
    mask = denominator > 0
    if not mask.any():
        return float("nan")
    return float(100.0 * np.mean(2.0 * np.abs(t[mask] - p[mask]) / denominator[mask]))


def bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean error (prediction minus observation), in ug/m3.

    Reported because a model can have competitive RMSE while systematically
    under-predicting episodes, which matters for a health advisory.
    """
    t, p = _clean(y_true, y_pred)
    return float(np.mean(p - t)) if t.size else float("nan")


def skill_score(rmse_model: float, rmse_reference: float) -> float:
    """Skill relative to a reference forecast.

    ``1 - RMSE_model / RMSE_reference``. Positive means the model beats the
    reference; zero means it merely matches it; negative means it is worse.

    Args:
        rmse_model: RMSE of the model under test.
        rmse_reference: RMSE of the reference, normally persistence.

    Returns:
        The skill score, or NaN if the reference RMSE is zero or undefined.
    """
    if not np.isfinite(rmse_model) or not np.isfinite(rmse_reference) or rmse_reference <= 0:
        return float("nan")
    return float(1.0 - rmse_model / rmse_reference)


def all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute the full metric set for one prediction vector.

    Args:
        y_true: Observed values in ug/m3.
        y_pred: Predicted values in ug/m3.

    Returns:
        Mapping of metric name to value, plus the sample count.
    """
    t, _ = _clean(y_true, y_pred)
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "r2": r2(y_true, y_pred),
        "smape": smape(y_true, y_pred),
        "bias": bias(y_true, y_pred),
        "n": int(t.size),
    }


# ---------------------------------------------------------------------------
# Significance
# ---------------------------------------------------------------------------


@dataclass
class DMResult:
    """Outcome of a Diebold-Mariano test.

    Attributes:
        statistic: The (small-sample corrected) DM statistic.
        p_value: Two-sided p-value.
        n: Number of paired observations.
        horizon: Forecast horizon used to set the HAC truncation lag.
        loss: Loss function name.
        better: Which model has the lower average loss.
        mean_loss_differential: Mean of ``L(e_a) - L(e_b)``.
    """

    statistic: float
    p_value: float
    n: int
    horizon: int
    loss: str
    better: str
    mean_loss_differential: float


def diebold_mariano(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    *,
    horizon: int,
    loss: str = "squared",
    name_a: str = "A",
    name_b: str = "B",
) -> DMResult:
    """Test whether two forecasts differ significantly in accuracy.

    Uses a Newey-West HAC variance with truncation lag ``horizon - 1``, since
    h-step-ahead forecast errors are serially correlated up to lag h-1, plus the
    Harvey-Leybourne-Newbold small-sample correction and a t-reference
    distribution.

    A lower average loss for A gives a negative statistic.

    Args:
        y_true: Observed values.
        pred_a: Predictions from the first model.
        pred_b: Predictions from the second model.
        horizon: Forecast horizon in steps, used for the truncation lag.
        loss: ``"squared"`` or ``"absolute"``.
        name_a: Label for the first model.
        name_b: Label for the second model.

    Returns:
        The test outcome.

    Raises:
        ValueError: If ``loss`` is unknown.
    """
    y = np.asarray(y_true, dtype=float).ravel()
    a = np.asarray(pred_a, dtype=float).ravel()
    b = np.asarray(pred_b, dtype=float).ravel()
    mask = np.isfinite(y) & np.isfinite(a) & np.isfinite(b)
    y, a, b = y[mask], a[mask], b[mask]

    err_a, err_b = y - a, y - b
    if loss == "squared":
        d = err_a**2 - err_b**2
    elif loss == "absolute":
        d = np.abs(err_a) - np.abs(err_b)
    else:
        raise ValueError(f"unknown loss {loss!r}; use 'squared' or 'absolute'")

    n = d.size
    if n < 10 or np.allclose(d, 0.0):
        return DMResult(
            statistic=float("nan"),
            p_value=float("nan"),
            n=int(n),
            horizon=int(horizon),
            loss=loss,
            better="indistinguishable",
            mean_loss_differential=float(np.mean(d)) if n else float("nan"),
        )

    d_bar = float(np.mean(d))
    demeaned = d - d_bar
    gamma0 = float(np.mean(demeaned**2))
    lag_max = max(int(horizon) - 1, 0)
    variance = gamma0
    for lag in range(1, lag_max + 1):
        if lag >= n:
            break
        gamma = float(np.mean(demeaned[lag:] * demeaned[:-lag]))
        # Bartlett weights keep the HAC estimate positive semi-definite.
        weight = 1.0 - lag / (lag_max + 1.0)
        variance += 2.0 * weight * gamma
    if variance <= 0:
        variance = gamma0

    dm = d_bar / np.sqrt(variance / n)

    # Harvey, Leybourne and Newbold (1997) small-sample correction.
    h = int(horizon)
    correction = (n + 1 - 2 * h + h * (h - 1) / n) / n
    correction = max(correction, 1e-8)
    dm_corrected = float(dm * np.sqrt(correction))
    p_value = float(2.0 * (1.0 - stats.t.cdf(abs(dm_corrected), df=n - 1)))

    return DMResult(
        statistic=dm_corrected,
        p_value=p_value,
        n=int(n),
        horizon=h,
        loss=loss,
        better=name_a if d_bar < 0 else name_b,
        mean_loss_differential=d_bar,
    )


@dataclass
class BootstrapCI:
    """Moving-block bootstrap confidence interval.

    Attributes:
        point: Point estimate on the full sample.
        lower: Lower confidence bound.
        upper: Upper confidence bound.
        alpha: Significance level used.
        n_resamples: Number of bootstrap replicates.
        block_size: Block length in observations.
    """

    point: float
    lower: float
    upper: float
    alpha: float
    n_resamples: int
    block_size: int


def block_bootstrap_rmse(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    n_resamples: int = 1000,
    block_size: int = 24,
    alpha: float = 0.05,
    seed: int = 42,
) -> BootstrapCI:
    """Confidence interval for RMSE under serial correlation.

    An i.i.d. bootstrap would understate uncertainty here, because consecutive
    hourly errors are strongly correlated. Resampling contiguous blocks preserves
    that dependence.

    Args:
        y_true: Observed values.
        y_pred: Predicted values.
        n_resamples: Number of bootstrap replicates.
        block_size: Block length in hours.
        alpha: Two-sided significance level.
        seed: RNG seed.

    Returns:
        The interval.
    """
    t, p = _clean(y_true, y_pred)
    n = t.size
    point = rmse(t, p)
    if n < block_size * 2:
        return BootstrapCI(point, float("nan"), float("nan"), alpha, n_resamples, block_size)

    rng = np.random.default_rng(seed)
    squared = (t - p) ** 2
    n_blocks = int(np.ceil(n / block_size))
    starts_max = n - block_size

    estimates = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        starts = rng.integers(0, starts_max + 1, size=n_blocks)
        idx = (starts[:, None] + np.arange(block_size)[None, :]).ravel()[:n]
        estimates[i] = np.sqrt(squared[idx].mean())

    lower, upper = np.quantile(estimates, [alpha / 2.0, 1.0 - alpha / 2.0])
    return BootstrapCI(point, float(lower), float(upper), alpha, n_resamples, block_size)


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


@dataclass
class ClassificationReport:
    """Metrics for the AQI-category task.

    Attributes:
        macro_f1: Unweighted mean F1 across classes.
        weighted_f1: Support-weighted mean F1.
        accuracy: Overall accuracy.
        balanced_accuracy: Mean per-class recall.
        per_class: Precision, recall, F1 and support per class label.
        confusion: Confusion matrix, rows are truth.
        labels: Class labels in matrix order.
    """

    macro_f1: float
    weighted_f1: float
    accuracy: float
    balanced_accuracy: float
    per_class: dict[str, dict[str, float]] = field(default_factory=dict)
    confusion: list[list[int]] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)


def classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]
) -> ClassificationReport:
    """Compute macro-F1, per-class metrics and a confusion matrix.

    Macro-F1 is the headline because the AQI classes are severely imbalanced:
    accuracy alone would be dominated by the majority categories and would hide
    failure on exactly the hazardous classes a health advisory exists to flag.

    Args:
        y_true: True class indices.
        y_pred: Predicted class indices.
        labels: Class names, indexed by class integer.

    Returns:
        The report.
    """
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        confusion_matrix,
        f1_score,
        precision_recall_fscore_support,
    )

    indices = list(range(len(labels)))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=indices, zero_division=0
    )
    per_class = {
        labels[i]: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i in indices
    }
    return ClassificationReport(
        macro_f1=float(f1_score(y_true, y_pred, labels=indices, average="macro", zero_division=0)),
        weighted_f1=float(
            f1_score(y_true, y_pred, labels=indices, average="weighted", zero_division=0)
        ),
        accuracy=float(accuracy_score(y_true, y_pred)),
        balanced_accuracy=float(balanced_accuracy_score(y_true, y_pred)),
        per_class=per_class,
        confusion=confusion_matrix(y_true, y_pred, labels=indices).tolist(),
        labels=list(labels),
    )


def aqi_categories(values: np.ndarray, edges: list[float]) -> np.ndarray:
    """Bucket PM2.5 concentrations into AQI category indices.

    Edges are inclusive upper bounds, so a value exactly on an edge falls in the
    lower category, matching how the published breakpoint table is defined.

    Args:
        values: PM2.5 concentrations in ug/m3, on the averaging basis the
            breakpoints are defined for.
        edges: Ordered inclusive upper bounds; the final class is open-ended.

    Returns:
        Integer category index per value.
    """
    arr = np.asarray(values, dtype=float)
    out = np.full(arr.shape, len(edges), dtype=int)
    for i, edge in enumerate(reversed(edges)):
        out[arr <= edge] = len(edges) - 1 - i
    return out


def summarise(values: list[float]) -> dict[str, Any]:
    """Mean, standard deviation and range across seeds.

    Args:
        values: One value per seed.

    Returns:
        Summary statistics, with NaNs excluded.
    """
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if not arr.size:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "n_seeds": 0,
        }
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "min": float(arr.min()),
        "max": float(arr.max()),
        "n_seeds": int(arr.size),
    }
