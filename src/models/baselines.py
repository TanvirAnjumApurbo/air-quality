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
        tz: IANA timezone the bins were built in. Carried on the model so that
            :func:`predict_climatology` cannot bin in a different zone than
            :func:`fit_climatology` did -- a mismatch that silently shifted
            every Beijing record's hour bins by two before it was caught.
    """

    table: dict[tuple[int, int], float]
    global_mean: float
    tz: str

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


def fit_climatology(train: SplitArrays, horizon: int, cfg: Config) -> ClimatologyModel:
    """Fit the climatology table on training rows only.

    The cell is keyed by the local hour and month **of the target time**, not of
    the feature time, since that is what is being predicted.

    The timezone comes from ``features.calendar_tz`` and is stored on the model.
    It used to be hardcoded to Asia/Dhaka, which put every Beijing record's hour
    bins two hours out -- and Beijing is where the gap-injection experiment runs,
    so the error reached the climatology representative of a reported family.

    Args:
        train: Training arrays.
        horizon: Forecast horizon in hours, used to shift the calendar key.
        cfg: Loaded configuration (``features.calendar_tz``).

    Returns:
        The fitted model.
    """
    tz = str(cfg.get("features.calendar_tz"))
    target_time = train.index + pd.Timedelta(hours=horizon)
    # The same local timezone the calendar features were built in.
    local = target_time.tz_convert(tz)
    frame = pd.DataFrame({"hour": local.hour, "month": local.month, "y": train.y}).dropna()
    grouped = frame.groupby(["hour", "month"])["y"].mean()
    return ClimatologyModel(
        table={(int(h), int(m)): float(v) for (h, m), v in grouped.items()},
        global_mean=float(frame["y"].mean()),
        tz=tz,
    )


def predict_climatology(model: ClimatologyModel, arrays: SplitArrays, horizon: int) -> np.ndarray:
    """Predict with a fitted climatology model.

    Takes no configuration: the timezone rides on the model, so predicting in a
    different zone than the table was binned in is unrepresentable rather than
    merely unlikely.

    Args:
        model: The fitted model.
        arrays: Arrays for the split being predicted.
        horizon: Forecast horizon in hours.

    Returns:
        Predicted values in ug/m3.
    """
    target_time = (arrays.index + pd.Timedelta(hours=horizon)).tz_convert(model.tz)
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

    # Reconstruct a regular hourly series from the training rows, keeping gaps
    # as NaN on the regular grid.
    #
    # The grid must stay evenly spaced: a seasonal order of (.,.,.,24) assumes a
    # regular hourly interval, so calling .dropna() would compress the series
    # across gaps and leave the seasonal term modelling a periodicity the data
    # does not have. Restricting instead to the longest contiguous stretch keeps
    # the spacing honest but is also wrong here for a different reason -- the
    # longest run in this record falls in July-October 2019, entirely inside a
    # monsoon, so the model would be fitted on low-season behaviour and applied
    # to a test period spanning both seasons.
    #
    # Neither compromise is necessary. A state-space SARIMAX handles missing
    # observations natively through the Kalman filter, so the full tail is used
    # with its gaps left in place.
    history = pd.Series(train.persistence, index=train.index).sort_index()
    history = history[~history.index.duplicated(keep="first")]
    full = history.asfreq("1h").ffill(limit=int(cfg.get("impute.max_ffill_hours", 3)))
    tail = full.iloc[-max_hours:]

    n_observed = int(tail.notna().sum())
    if n_observed < 500:
        logger.warning("SARIMAX: only %d observed hours in the training tail; skipping", n_observed)
        return np.full(len(test), np.nan)
    logger.info(
        "SARIMAX h=%d: training tail %d h (%s to %s), %d observed (%.1f%%), gaps kept as NaN",
        horizon,
        len(tail),
        tail.index.min().date(),
        tail.index.max().date(),
        n_observed,
        100.0 * n_observed / len(tail),
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
            forecasts = h_step_ahead_from_filtered_state(applied, horizon)
        anchor = pd.Series(forecasts, index=eval_filled.index)
        # anchor[t] is yhat(t+h | data up to t), so it aligns directly to the row
        # stamped at t. Reindexing onto t+h -- the previous behaviour -- is what
        # leaked future observations into the forecast.
        preds = anchor.reindex(test.index).to_numpy(dtype=float)
    except Exception as exc:
        logger.warning("SARIMAX h=%d: apply failed (%s); returning NaN", horizon, exc)

    return np.clip(preds, 0.0, None)


def h_step_ahead_from_filtered_state(results: Any, horizon: int) -> np.ndarray:
    """Compute genuine h-step-ahead forecasts at every anchor time.

    This exists because the obvious approach is wrong. ``get_prediction(...,
    dynamic=False)`` returns *one-step-ahead* in-sample predictions: the value at
    index ``t`` conditions on observations through ``t-1``. Stamping those onto
    ``t+h`` and calling the result an h-step forecast hands the model ``h-1``
    hours of future data. That mistake inflated SARIMAX's skill against
    persistence at h=24 from roughly 0.06 to 0.44 -- a result that looked good
    precisely because it was not a forecast.

    Instead this uses the state-space form directly. The Kalman filter's filtered
    state ``a_{t|t}`` conditions on data up to and including ``t``. Propagating it
    ``h`` steps through the transition matrix and reading it out through the
    design matrix gives

        yhat(t+h | t) = Z (T^h a_{t|t} + sum_j T^j c) + d

    with ``c`` the state intercept and ``d`` the observation intercept. Exact for
    a linear Gaussian state-space model, and it costs one matrix power rather
    than a re-forecast at every timestep.

    Args:
        results: A fitted statsmodels state-space results object.
        horizon: Forecast horizon in steps.

    Returns:
        h-step-ahead forecasts aligned to the anchor time ``t``.
    """
    filtered = results.filter_results
    state = np.asarray(filtered.filtered_state)  # (k_states, n)
    n = state.shape[1]

    k = state.shape[0]

    def _static(matrix: Any, expected_cols: int, default_shape: tuple[int, ...]) -> np.ndarray:
        """Reduce a possibly time-varying system matrix to a single slice.

        statsmodels stores system matrices with a trailing time axis whenever the
        matrix varies, and drops it to length 1 when it does not. Intercepts
        arrive as ``(k, n)`` rather than ``(k, 1, n)``, so the reduction has to
        be driven by the expected column count rather than by dimensionality
        alone.
        """
        if matrix is None:
            return np.zeros(default_shape)
        arr = np.asarray(matrix, dtype=float)
        if arr.ndim == 3:
            return arr[..., -1]
        if arr.ndim == 2 and arr.shape[1] != expected_cols:
            return arr[:, -1:]
        return arr

    transition = _static(filtered.transition, k, (k, k))
    design = _static(filtered.design, k, (1, k))
    state_intercept = _static(getattr(filtered, "state_intercept", None), 1, (k, 1))
    obs_intercept = _static(getattr(filtered, "obs_intercept", None), 1, (1, 1))

    power = np.linalg.matrix_power(transition, int(horizon))

    # Deterministic drift accumulated over the h propagation steps.
    intercept = np.asarray(state_intercept, dtype=float).reshape(-1)
    drift = np.zeros(k, dtype=float)
    if np.any(intercept):
        for _ in range(int(horizon)):
            drift = transition @ drift + intercept

    projected = power @ state + drift[:, None]
    out = (np.asarray(design, dtype=float).reshape(1, k) @ projected).reshape(-1)
    out = out + float(np.asarray(obs_intercept, dtype=float).reshape(-1)[0])

    # The first `horizon` anchors have too little filtered history behind them to
    # be meaningful; leave them missing rather than reporting a warm-up artefact.
    if int(horizon) < n:
        out[: int(horizon)] = np.nan
    return out
