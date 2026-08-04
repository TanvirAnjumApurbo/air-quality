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
from sklearn.base import BaseEstimator, TransformerMixin

from src.models.data import SplitArrays, load_scaler_params
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


def target_level_columns(cfg: Config, feature_names: list[str]) -> list[int]:
    """Indices of predictors that are PM2.5 *levels* rather than derived contrasts.

    Levels are the target itself, its lags, and the rolling mean/min/max -- all
    non-negative concentrations in ug/m3. Rolling standard deviations,
    differences and rates of change are excluded: they are spreads or signed
    contrasts, so a ``log1p`` is either meaningless or undefined on them.

    Args:
        cfg: Loaded configuration.
        feature_names: Predictor names, in matrix order.

    Returns:
        Positions in ``feature_names`` that hold PM2.5 levels.
    """
    target = str(cfg.get("features.target"))

    def is_level(name: str) -> bool:
        if name == target or name.startswith(f"{target}_lag_"):
            return True
        return name.startswith(f"{target}_roll") and name.rsplit("_", 1)[-1] in {
            "mean",
            "min",
            "max",
        }

    return [i for i, name in enumerate(feature_names) if is_level(name)]


class LogScaleTargetHistory(BaseEstimator, TransformerMixin):
    """Put PM2.5 history on the same ``log1p`` scale as the target.

    The target is modelled as ``log1p(y)``. The PM2.5 predictors arrive on their
    raw ug/m3 scale (standardised, but standardisation is affine and so does not
    change the functional form). A linear model is therefore asked to express
    ``log1p(y(t+h))`` as a linear function of ``y(t)`` -- which it cannot do, and
    the resulting poor fit says nothing about linear methods on this task. Trees
    and recurrent models are unaffected because both are non-linear in the
    inputs; the handicap falls on ridge alone.

    This transformer removes it. Standardisation is inverted to recover ug/m3,
    ``log1p`` is applied to the level columns, and the result is handed on for
    re-standardisation. It sits inside the ridge pipeline rather than in the
    shared feature matrix so that the tier-2 comparison is otherwise untouched:
    adding log columns globally would perturb the tree models' feature
    subsampling for no benefit to them.

    The pipeline placement also means the fix survives cross-validation -- the
    re-standardisation that follows is fitted per fold, not once on everything.
    """

    def __init__(self, cfg: Config | None = None, feature_names: list[str] | None = None) -> None:
        """Record what is needed to invert the shared standardisation.

        Args:
            cfg: Loaded configuration, for the scaler parameters and target name.
            feature_names: Predictor names, in matrix order.
        """
        self.cfg = cfg
        self.feature_names = feature_names

    def fit(self, x: np.ndarray, y: np.ndarray | None = None) -> LogScaleTargetHistory:  # noqa: ARG002
        """Resolve column positions and scaler parameters.

        Args:
            x: Predictor matrix; unused beyond the sklearn contract.
            y: Targets; unused.

        Returns:
            ``self``.
        """
        names = list(self.feature_names or [])
        mean, scale = load_scaler_params(self.cfg)
        common = [c for c in names if c in mean.index]
        self.scaled_idx_ = [names.index(c) for c in common]
        self.mean_ = mean[common].to_numpy(dtype=np.float64)
        self.scale_ = scale[common].to_numpy(dtype=np.float64)
        self.level_idx_ = target_level_columns(self.cfg, names)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Undo standardisation, then log the level columns.

        Args:
            x: Standardised predictor matrix.

        Returns:
            A copy with PM2.5 levels on the ``log1p`` scale.
        """
        out = np.asarray(x, dtype=np.float64).copy()
        if self.scaled_idx_:
            out[:, self.scaled_idx_] = out[:, self.scaled_idx_] * self.scale_ + self.mean_
        if self.level_idx_:
            # Clipped because QC bounds observations at zero but the inverse of a
            # standardisation can land marginally below it on floating point.
            out[:, self.level_idx_] = np.log1p(np.clip(out[:, self.level_idx_], 0.0, None))
        return out


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

    The estimator is a pipeline, not a bare ``Ridge``. Its first step puts the
    PM2.5 history on the ``log1p`` scale the target is modelled on -- see
    :class:`LogScaleTargetHistory` for why a linear model is otherwise being
    asked to do something impossible. The pipeline consumes and returns the same
    standardised matrix every other tier-2 model sees, so nothing downstream
    changes.

    ``models.trees.ridge.log_scale_history`` disables the correction, which
    reproduces the earlier crippled fit for the ablation table.

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
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    x = np.vstack([train.x, val.x])
    y = np.concatenate([train.y_transformed, val.y_transformed])
    alphas = [float(a) for a in cfg.get("models.trees.ridge.alphas")]

    if bool(cfg.get("models.trees.ridge.log_scale_history", True)):
        estimator = Pipeline(
            [
                ("log_history", LogScaleTargetHistory(cfg, list(train.feature_names))),
                # Re-standardise after the log: fitted per CV fold by the
                # pipeline, so this introduces no leakage across folds.
                ("rescale", StandardScaler()),
                ("ridge", Ridge(random_state=seed)),
            ]
        )
        grid = {"ridge__alpha": alphas}
    else:
        logger.warning("ridge: log_scale_history disabled -- fitting the handicapped variant")
        estimator = Ridge(random_state=seed)
        grid = {"alpha": alphas}

    started = time.perf_counter()
    best, params, score, n = _search(estimator, grid, cfg, x, y, seed, logger, "ridge")
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
