"""Stratified evaluation and significance testing.

Overall metrics hide the failure mode that matters most for a health advisory: a
model can look strong on average while being useless on exactly the
high-pollution episodes an advisory exists to warn about. Two stratifications
are therefore mandatory rather than optional:

* **By season** -- monsoon versus dry, using the Department of Environment's
  month boundaries.
* **By pollution level** -- conditional on the observed concentration exceeding
  the Bangladesh national 24-hour standard.

If a model degrades on the high-pollution stratum, that is reported. A
well-documented negative finding is stronger than a hidden one.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.eval.metrics import all_metrics, skill_score
from src.utils import Config


def season_mask(month_local: np.ndarray, cfg: Config) -> np.ndarray:
    """Boolean mask selecting monsoon observations.

    Args:
        month_local: Local calendar month per observation.
        cfg: Loaded configuration (``features.season``).

    Returns:
        True where the month falls in the monsoon season.
    """
    monsoon = sorted(set(cfg.get("features.season.monsoon_months")))
    return np.isin(month_local, monsoon)


def stratified_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    reference: np.ndarray,
    month_local: np.ndarray,
    cfg: Config,
) -> list[dict[str, Any]]:
    """Compute metrics overall and within each stratum.

    The skill score inside a stratum is recomputed against the reference forecast
    *restricted to that stratum*, not against the overall reference RMSE.
    Comparing a stratum's model RMSE to a whole-sample reference would conflate
    difficulty with skill.

    Args:
        y_true: Observed values in ug/m3.
        y_pred: Predicted values in ug/m3.
        reference: Reference (persistence) predictions in ug/m3.
        month_local: Local calendar month per observation.
        cfg: Loaded configuration.

    Returns:
        One record per stratum.
    """
    strat_cfg = cfg.get("evaluation.stratify")
    records: list[dict[str, Any]] = []

    def add(name: str, group: str, mask: np.ndarray) -> None:
        if mask.sum() < 30:
            return
        metrics = all_metrics(y_true[mask], y_pred[mask])
        ref = all_metrics(y_true[mask], reference[mask])["rmse"]
        metrics["skill_vs_persistence"] = skill_score(metrics["rmse"], ref)
        metrics["reference_rmse"] = ref
        records.append({"stratum": name, "group": group, **metrics})

    everything = np.ones(len(y_true), dtype=bool)
    add("overall", "all", everything)

    if bool(strat_cfg.get("by_season", True)):
        monsoon = season_mask(month_local, cfg)
        add("season", "monsoon (wet)", monsoon)
        add("season", "dry", ~monsoon)

    level = strat_cfg.get("by_pollution_level", {})
    if bool(level.get("enabled", True)):
        thresholds = [float(level["threshold_ugm3"])] + [
            float(t) for t in level.get("secondary_thresholds_ugm3", [])
        ]
        for threshold in thresholds:
            above = y_true > threshold
            add("pollution", f"observed > {threshold:g} ug/m3", above)
            add("pollution", f"observed <= {threshold:g} ug/m3", ~above)

    return records


def stratified_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Render stratified records as a table.

    Args:
        records: Output of :func:`stratified_metrics`.

    Returns:
        A tidy DataFrame.
    """
    return pd.DataFrame(records)[
        [
            "stratum",
            "group",
            "n",
            "rmse",
            "mae",
            "r2",
            "smape",
            "bias",
            "reference_rmse",
            "skill_vs_persistence",
        ]
    ]
