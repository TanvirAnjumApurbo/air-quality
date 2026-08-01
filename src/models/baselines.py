"""Tier 1: honest baselines.

These are the paper's integrity, not filler. A forecasting model that cannot beat
persistence has demonstrated nothing, and papers that omit these baselines
routinely report "improvements" that a one-line rule would match.

Four references:

* **Persistence** -- ``yhat(t+h) = y(t)``. The hardest baseline to beat at short
  horizons and the reference for every skill score in this study.
* **Seasonal-naive** -- ``yhat(t+h) = y(t+h-24)``. Exploits the strong diurnal
  cycle. Only defined where ``t+h-24 <= t``, i.e. ``h <= 24``.
* **Climatology** -- the training-set mean by (hour-of-day, month), computed in
  local time. Carries no information about recent conditions at all.
* **SARIMAX** -- a fitted statistical model, for a like-for-like comparison
  against a classical time-series method rather than only against naive rules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.models.data import SplitArrays
from src.utils import Config


@dataclass
class ClimatologyModel:
    """Training-set mean by local hour-of-day and month.

    Attributes:
        table: Mean target keyed by ``(hour, month)``.
        global_mean: Fallback for cells unseen in training.
    """

    table: dict[tuple[int, int], float]
    global_mean: float

    def predict(self, hour_local: np.ndarray, month_local: np.ndarray) -> np.ndarray:
        """Look up the climatological mean for each observation.

        Args:
            hour_local: Local hour of day per row.
            month_local: Local calendar month per row.

        Returns:
            Predicted values in ug/m3.
        """
        return np.array(
            [
                self.table.get((int(h), int(m)), self.global_mean)
                for h, m in zip(hour_local, month_local, strict=True)
            ],
            dtype=float,
        )


def fit_climatology(train: SplitArrays, horizon: int) -> ClimatologyModel:
    """Fit the climatology table on training rows only.

    The cell is keyed by the local hour and month **of the target time**, not of
    the feature time, since that is what is being predicted.

    Args:
        train: Training arrays.
        horizon: Forecast horizon in hours, used to shift the calendar key.

    Returns:
        The fitted model.
    """
    target_time = train.index + pd.Timedelta(hours=horizon)
    # Reuse the same local timezone the calendar features were built in.
    local = target_time.tz_convert(train.index.tz).tz_convert("Asia/Dhaka")
    frame = pd.DataFrame({"hour": local.hour, "month": local.month, "y": train.y}).dropna()
    grouped = frame.groupby(["hour", "month"])["y"].mean()
    return ClimatologyModel(
        table={(int(h), int(m)): float(v) for (h, m), v in grouped.items()},
        global_mean=float(frame["y"].mean()),
    )


def predict_climatology(model: ClimatologyModel, arrays: SplitArrays, horizon: int) -> np.ndarray:
    """Predict with a fitted climatology model.

    Args:
        model: The fitted model.
        arrays: Arrays for the split being predicted.
        horizon: Forecast horizon in hours.

    Returns:
        Predicted values in ug/m3.
    """
    target_time = (arrays.index + pd.Timedelta(hours=horizon)).tz_convert("Asia/Dhaka")
    return model.predict(target_time.hour.to_numpy(), target_time.month.to_numpy())


def predict_persistence(arrays: SplitArrays) -> np.ndarray:
    """Persistence forecast: the value observed at time t.

    Args:
        arrays: Arrays for the split being predicted.

    Returns:
        Predicted values in ug/m3.
    """
    return arrays.persistence.copy()


def predict_seasonal_naive(arrays: SplitArrays) -> np.ndarray:
    """Seasonal-naive forecast: the value 24 hours before the target time.

    Args:
        arrays: Arrays for the split being predicted.

    Returns:
        Predicted values in ug/m3; NaN where the lag is not available.
    """
    return arrays.seasonal_naive.copy()


def fit_predict_sarimax(
    train: SplitArrays,
    test: SplitArrays,
    cfg: Config,
    horizon: int,
    logger: Any,
) -> np.ndarray:
    """Fit SARIMAX on the training tail and forecast the evaluation rows.

    Fitting a seasonal ARIMA on tens of thousands of hourly points is slow enough
    to breach the run-time budget, so the fit uses the most recent
    ``max_train_hours`` of training data. That is stated rather than hidden: it
    makes SARIMAX a *recent-history* statistical baseline.

    Predictions are produced by re-anchoring the fitted model on the observed
    history up to each evaluation point and forecasting h steps ahead, so no
    future information is used.

    Args:
        train: Training arrays.
        test: Arrays for the split being predicted.
        cfg: Loaded configuration.
        horizon: Forecast horizon in hours.
        logger: Logger for progress and failure reporting.

    Returns:
        Predicted values in ug/m3; all-NaN if the fit fails.
    """
    import warnings

    from statsmodels.tsa.statespace.sarimax import SARIMAX

    spec = cfg.get("models.baselines.sarimax")
    max_hours = int(spec.get("max_train_hours", 8760))

    # Reconstruct a regular hourly series from the training rows.
    history = pd.Series(train.persistence, index=train.index).sort_index()
    history = history[~history.index.duplicated(keep="first")]
    full = history.asfreq("1h")
    tail = full.iloc[-max_hours:].ffill(limit=int(cfg.get("impute.max_ffill_hours", 3)))

    # Fit on the longest CONTIGUOUS stretch, not on the non-null rows.
    # Calling .dropna() here would compress the series across gaps, silently
    # changing the sampling interval -- and a seasonal order of (·,·,·,24)
    # assumes a regular hourly spacing, so the seasonal term would then be
    # modelling something that does not exist in the data.
    present = tail.notna()
    blocks = (present != present.shift()).cumsum()
    runs = tail[present].groupby(blocks[present])
    if runs.ngroups == 0:
        logger.warning("SARIMAX: no usable contiguous training stretch; skipping")
        return np.full(len(test), np.nan)
    longest_key = max(runs.groups, key=lambda k: len(runs.groups[k]))
    tail = tail.loc[runs.groups[longest_key]]

    if len(tail) < 500:
        logger.warning(
            "SARIMAX: longest contiguous training stretch is only %d hours; skipping", len(tail)
        )
        return np.full(len(test), np.nan)
    logger.info(
        "SARIMAX h=%d: longest contiguous training stretch %d h (%s to %s)",
        horizon,
        len(tail),
        tail.index.min(),
        tail.index.max(),
    )

    logger.info(
        "SARIMAX h=%d: fitting order=%s seasonal=%s on %d points",
        horizon,
        spec["order"],
        spec["seasonal_order"],
        len(tail),
    )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = SARIMAX(
                tail.to_numpy(dtype=float),
                order=tuple(spec["order"]),
                seasonal_order=tuple(spec["seasonal_order"]),
                trend=spec.get("trend", "c"),
                enforce_stationarity=bool(spec.get("enforce_stationarity", False)),
                enforce_invertibility=bool(spec.get("enforce_invertibility", False)),
            )
            fitted = model.fit(disp=False, maxiter=100)
    except Exception as exc:
        logger.warning("SARIMAX h=%d: fit failed (%s); returning NaN", horizon, exc)
        return np.full(len(test), np.nan)

    logger.info("SARIMAX h=%d: fitted, applying to %d evaluation rows", horizon, len(test))

    # Apply the fitted parameters to the evaluation period's own history, then
    # read the h-step-ahead forecast at each anchor point. `apply` reuses the
    # trained parameters without refitting, which keeps this affordable.
    eval_series = pd.Series(test.persistence, index=test.index).sort_index()
    eval_series = eval_series[~eval_series.index.duplicated(keep="first")].asfreq("1h")
    eval_filled = eval_series.ffill(limit=int(cfg.get("impute.max_ffill_hours", 3)))

    preds = np.full(len(test), np.nan)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            applied = fitted.apply(eval_filled.to_numpy(dtype=float), refit=False)
            # In-sample h-step-ahead predictions across the evaluation window.
            forecasts = applied.get_prediction(
                start=0, end=len(eval_filled) - 1, dynamic=False
            ).predicted_mean
        anchor = pd.Series(forecasts, index=eval_filled.index)
        # The value stamped at t+h is the forecast for that time; align it back
        # to the row anchored at t.
        aligned = anchor.reindex(test.index + pd.Timedelta(hours=horizon))
        preds = aligned.to_numpy(dtype=float)
    except Exception as exc:
        logger.warning("SARIMAX h=%d: apply failed (%s); returning NaN", horizon, exc)

    return np.clip(preds, 0.0, None)
