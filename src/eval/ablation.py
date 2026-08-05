"""Shared reading and testing for the gap-injection grids.

``17_ablation_analysis.py`` analyses one donor record; ``18_donor_replication.py``
compares several. Both need the same flattening, the same notion of a model
family, and above all the same paired test -- if the two scripts computed the arm
gap even slightly differently, a replication check would be measuring the
difference between the scripts rather than between the donors. So it lives here
once.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from src.eval.metrics import holm_bonferroni

# Grouped so results read as "model class", which is the comparison the
# experiment is about -- not nine individually coloured models.
FAMILIES: dict[str, list[str]] = {
    "sequence": ["gru_h64_l2", "lstm_h64_l1", "gru", "lstm"],
    "linear": ["ridge", "dlinear", "nlinear"],
    "trees": ["random_forest", "xgboost", "lightgbm"],
    "naive": ["persistence", "climatology", "seasonal_naive"],
}

FAMILY_ORDER = ("sequence", "trees", "linear", "naive")


def family_of(model: str) -> str:
    """Map a model name to its class.

    Args:
        model: Model name as recorded in the ablation grid.

    Returns:
        The family label, or ``"other"``.
    """
    for family, members in FAMILIES.items():
        if model in members or any(model.startswith(m) for m in members):
            return family
    return "other"


def fmt_p(p: float) -> str:
    """Format a p-value without rounding a small one to zero.

    ``round(7e-06, 4)`` is ``0.0`` and prints as "0.0000", which is not a
    p-value -- no test returns zero probability. Small values get a threshold.

    Args:
        p: Raw p-value.

    Returns:
        A display string.
    """
    value = float(p)
    if not np.isfinite(value):
        return "—"
    return "<0.0001" if value < 1e-4 else f"{value:.4f}"


def tidy(payload: dict[str, Any]) -> pd.DataFrame:
    """Flatten the nested cell structure into one row per (cell, model).

    Args:
        payload: Contents of an ``ablation_gap_injection*.json``.

    Returns:
        A tidy frame.
    """
    rows = []
    for key, cell in payload.get("cells", {}).items():
        injection = cell["injection"]
        for model, metrics in cell["models"].items():
            rows.append(
                {
                    "cell": key,
                    "arm": cell["arm"],
                    "target_coverage": cell["target_coverage"],
                    "coverage": injection["achieved_coverage"],
                    "injection_seed": cell["injection_seed"],
                    "hours_removed": injection["hours_removed"],
                    "n_gaps": injection["n_gaps_after"],
                    "train_rows": cell["usable_rows"]["train"],
                    "test_rows": cell["test_rows"],
                    "model": model,
                    "family": family_of(model),
                    "rmse": metrics["rmse"],
                    "skill": metrics["skill_vs_persistence"],
                }
            )
    return pd.DataFrame(rows)


def paired_arm_gaps(frame: pd.DataFrame) -> pd.DataFrame:
    """Difference the two arms within each matched (coverage, injection seed).

    The arms are a matched pair by construction: at a given coverage level and
    injection seed both remove the *same number* of observed hours and differ
    only in how those hours are arranged. Differencing within the pair removes
    the level and the draw, leaving the effect of arrangement alone -- which is
    the whole quantity this experiment exists to estimate.

    The representative model per family is held FIXED across arms, chosen once
    on the undegraded reference cell. Taking each arm's own best model instead
    would let the identity of "the sequence model" change between the two halves
    of a difference, so the result would confound the effect of fragmentation
    with a change of model.

    Args:
        frame: Tidy frame from :func:`tidy`, including the undegraded rows.

    Returns:
        One row per (family, target_coverage, injection_seed) with the arm gap.
    """
    reference = frame[frame["arm"] == "none"]
    if reference.empty:
        return pd.DataFrame()
    representative = (
        reference.sort_values("skill", ascending=False)
        .groupby("family", as_index=False)
        .first()[["family", "model"]]
    )
    kept = frame.merge(representative, on=["family", "model"], how="inner")

    wide = kept[kept["arm"] != "none"].pivot_table(
        index=["family", "model", "target_coverage", "injection_seed"],
        columns="arm",
        values="skill",
        aggfunc="first",
    )
    if not {"fragmented", "contiguous"} <= set(wide.columns):
        return pd.DataFrame()
    wide = wide.dropna(subset=["fragmented", "contiguous"]).reset_index()
    wide["arm_gap"] = wide["fragmented"] - wide["contiguous"]
    return wide


def test_arm_gaps(paired: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """Test, per family, whether the paired arm gap differs from zero.

    Wilcoxon signed-rank rather than a paired t-test: the gaps are a handful of
    values per family and their distribution is not known to be symmetric-normal.
    Holm-Bonferroni across the families, because one test per family is a small
    but real multiple-comparison problem and the sequence family's result is the
    one being claimed.

    Args:
        paired: Output of :func:`paired_arm_gaps`.
        alpha: Family-wise error rate.

    Returns:
        One row per family, most negative gap first.
    """
    rows = []
    for family, group in paired.groupby("family"):
        gaps = group["arm_gap"].to_numpy(dtype=float)
        n = len(gaps)
        p = float("nan")
        if n >= 5 and np.any(gaps != 0.0):
            p = float(stats.wilcoxon(gaps, alternative="two-sided").pvalue)
        # Bootstrap rather than a t interval, for the same reason as the test.
        rng = np.random.default_rng(0)
        boot = (
            np.array([rng.choice(gaps, size=n, replace=True).mean() for _ in range(5000)])
            if n >= 2
            else np.array([])
        )
        rows.append(
            {
                "family": family,
                "n_pairs": n,
                "mean_arm_gap": float(gaps.mean()),
                "median_arm_gap": float(np.median(gaps)),
                "ci_low": float(np.percentile(boot, 2.5)) if boot.size else float("nan"),
                "ci_high": float(np.percentile(boot, 97.5)) if boot.size else float("nan"),
                "p_wilcoxon": p,
            }
        )
    out = pd.DataFrame(rows).sort_values("mean_arm_gap").reset_index(drop=True)
    adjusted = holm_bonferroni(out["p_wilcoxon"].tolist(), alpha=alpha)
    out["p_holm"] = [r["p_adjusted"] for r in adjusted]
    out["significant_holm"] = [r["reject"] for r in adjusted]
    return out
