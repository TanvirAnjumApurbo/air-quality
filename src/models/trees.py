"""Tier 2: classical machine learning.

Ridge, random forest, XGBoost and LightGBM on the full tabular feature set.

Hyperparameters are tuned with ``TimeSeriesSplit`` over **train and validation
only**. The test split is never touched during tuning -- the search sees only
folds carved out of the earlier period, so the held-out period stays genuinely
held out. Folds are chronological by construction, which is what makes
``TimeSeriesSplit`` the correct cross-validator here rather than ``KFold``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.models.data import SplitArrays
from src.utils import Config


@dataclass
class TunedModel:
    """A fitted estimator plus the record of how it was selected.

    Attributes:
        name: Model name.
        estimator: The fitted estimator.
        best_params: Selected hyperparameters.
        cv_score: Best cross-validated score (negative RMSE on the modelling scale).
        n_candidates: Number of candidate configurations evaluated.
        fit_seconds: Wall-clock seconds spent tuning and fitting.
        feature_names: Predictor names, in matrix order.
    """

    name: str
    estimator: Any
    best_params: dict[str, Any] = field(default_factory=dict)
    cv_score: float = float("nan")
    n_candidates: int = 0
    fit_seconds: float = 0.0
    feature_names: list[str] = field(default_factory=list)


def _search(
    estimator: Any,
    grid: dict[str, list[Any]],
    cfg: Config,
    x: np.ndarray,
    y: np.ndarray,
    seed: int,
    logger: Any,
    name: str,
) -> tuple[Any, dict[str, Any], float, int]:
    """Run a randomised search under chronological cross-validation.

    Args:
        estimator: Unfitted estimator.
        grid: Candidate hyperparameter values.
        cfg: Loaded configuration.
        x: Predictors from train+val.
        y: Targets from train+val, on the modelling scale.
        seed: RNG seed for the search.
        logger: Logger for progress.
        name: Model name, for logging.

    Returns:
        ``(best_estimator, best_params, best_score, n_candidates)``.
    """
    from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

    tuning = cfg.get("models.trees.tuning")
    n_splits = int(tuning.get("n_splits", 4))
    n_iter = int(tuning.get("n_iter", 20))

    total = 1
    for values in grid.values():
        total *= len(values)
    n_iter = min(n_iter, total)

    logger.info(
        "%s: randomised search, %d of %d configurations, %d chronological folds",
        name,
        n_iter,
        total,
        n_splits,
    )
    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=grid,
        n_iter=n_iter,
        scoring=str(tuning.get("scoring", "neg_root_mean_squared_error")),
        cv=TimeSeriesSplit(n_splits=n_splits),
        random_state=seed,
        n_jobs=1,
        refit=True,
        error_score="raise",
    )
    search.fit(x, y)
    return search.best_estimator_, dict(search.best_params_), float(search.best_score_), n_iter


def fit_ridge(
    train: SplitArrays, val: SplitArrays, cfg: Config, seed: int, logger: Any
) -> TunedModel:
    """Fit ridge regression with the alpha chosen by chronological CV.

    Args:
        train: Training arrays.
        val: Validation arrays.
        cfg: Loaded configuration.
        seed: RNG seed.
        logger: Logger.

    Returns:
        The tuned model.
    """
    import time

    from sklearn.linear_model import Ridge

    x = np.vstack([train.x, val.x])
    y = np.concatenate([train.y_transformed, val.y_transformed])
    alphas = [float(a) for a in cfg.get("models.trees.ridge.alphas")]

    started = time.perf_counter()
    best, params, score, n = _search(
        Ridge(random_state=seed), {"alpha": alphas}, cfg, x, y, seed, logger, "ridge"
    )
    return TunedModel(
        name="ridge",
        estimator=best,
        best_params=params,
        cv_score=score,
        n_candidates=n,
        fit_seconds=time.perf_counter() - started,
        feature_names=train.feature_names,
    )


def fit_random_forest(
    train: SplitArrays, val: SplitArrays, cfg: Config, seed: int, logger: Any
) -> TunedModel:
    """Fit a random forest with hyperparameters chosen by chronological CV."""
    import time

    from sklearn.ensemble import RandomForestRegressor

    spec = cfg.get("models.trees.random_forest")
    x = np.vstack([train.x, val.x])
    y = np.concatenate([train.y_transformed, val.y_transformed])

    started = time.perf_counter()
    best, params, score, n = _search(
        RandomForestRegressor(random_state=seed, n_jobs=int(spec.get("n_jobs", -1))),
        dict(spec["grid"]),
        cfg,
        x,
        y,
        seed,
        logger,
        "random_forest",
    )
    return TunedModel(
        name="random_forest",
        estimator=best,
        best_params=params,
        cv_score=score,
        n_candidates=n,
        fit_seconds=time.perf_counter() - started,
        feature_names=train.feature_names,
    )


def fit_xgboost(
    train: SplitArrays, val: SplitArrays, cfg: Config, seed: int, logger: Any, device: str
) -> TunedModel:
    """Fit XGBoost with hyperparameters chosen by chronological CV."""
    import time

    from xgboost import XGBRegressor

    spec = cfg.get("models.trees.xgboost")
    x = np.vstack([train.x, val.x])
    y = np.concatenate([train.y_transformed, val.y_transformed])

    xgb_device = "cuda" if device == "cuda" and str(spec.get("device", "auto")) != "cpu" else "cpu"
    started = time.perf_counter()
    best, params, score, n = _search(
        XGBRegressor(
            random_state=seed,
            tree_method=str(spec.get("tree_method", "hist")),
            device=xgb_device,
            verbosity=0,
        ),
        dict(spec["grid"]),
        cfg,
        x,
        y,
        seed,
        logger,
        "xgboost",
    )
    return TunedModel(
        name="xgboost",
        estimator=best,
        best_params=params,
        cv_score=score,
        n_candidates=n,
        fit_seconds=time.perf_counter() - started,
        feature_names=train.feature_names,
    )


def fit_lightgbm(
    train: SplitArrays, val: SplitArrays, cfg: Config, seed: int, logger: Any
) -> TunedModel | None:
    """Fit LightGBM, or return None if the package is unusable.

    Returns:
        The tuned model, or None when LightGBM cannot be imported. A missing
        optional dependency is logged and skipped rather than aborting the tier.
    """
    import time

    try:
        from lightgbm import LGBMRegressor
    except Exception as exc:
        logger.warning("lightgbm unavailable (%s); skipping", exc)
        return None

    spec = cfg.get("models.trees.lightgbm")
    x = np.vstack([train.x, val.x])
    y = np.concatenate([train.y_transformed, val.y_transformed])

    started = time.perf_counter()
    best, params, score, n = _search(
        LGBMRegressor(random_state=seed, verbose=-1, n_jobs=-1),
        dict(spec["grid"]),
        cfg,
        x,
        y,
        seed,
        logger,
        "lightgbm",
    )
    return TunedModel(
        name="lightgbm",
        estimator=best,
        best_params=params,
        cv_score=score,
        n_candidates=n,
        fit_seconds=time.perf_counter() - started,
        feature_names=train.feature_names,
    )


def permutation_importance_scores(
    model: TunedModel, arrays: SplitArrays, seed: int, n_repeats: int = 5
) -> dict[str, float]:
    """Permutation importance on the evaluation split.

    Permutation importance is preferred to a tree's built-in split-count
    importance, which is biased towards high-cardinality features.

    Args:
        model: The fitted model.
        arrays: Arrays to measure importance on.
        seed: RNG seed.
        n_repeats: Permutations per feature.

    Returns:
        Mean importance per feature name, sorted descending.
    """
    from sklearn.inspection import permutation_importance

    result = permutation_importance(
        model.estimator,
        arrays.x,
        arrays.y_transformed,
        n_repeats=n_repeats,
        random_state=seed,
        scoring="neg_root_mean_squared_error",
        n_jobs=1,
    )
    scores = {
        name: float(value)
        for name, value in zip(arrays.feature_names, result.importances_mean, strict=True)
    }
    return dict(sorted(scores.items(), key=lambda kv: -kv[1]))
