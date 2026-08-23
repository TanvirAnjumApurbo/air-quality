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
    # Named for what it is, not for what it was once captioned as. This family
    # was called "naive" and described as the control that "never reads the
    # training record" -- but its representative is chosen by highest skill on
    # the reference cell, persistence scores identically zero there by
    # construction, and so climatology always wins the slot. Climatology is
    # fitted on the degraded training split. The family is the weakest model
    # that DOES read the record, and it was never the control.
    #
    # There is no useful no-training-data control for SKILL in this design:
    # because the test period is never degraded, any model that reads nothing
    # from training has an arm gap of exactly zero in every cell, so it cannot
    # vary and cannot falsify anything. The invariance is checked by equality
    # instead (see 17_ablation_analysis.py), which is stronger than a test.
    "climatological": ["persistence", "climatology", "seasonal_naive"],
}

FAMILY_ORDER = ("sequence", "trees", "linear", "climatological")


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


def family_contrasts(
    paired: pd.DataFrame,
    *,
    reference: str = "sequence",
    alpha: float = 0.05,
    n_boot: int = 5000,
    by: str | None = None,
) -> pd.DataFrame:
    """Test each family's arm gap directly against a reference family's.

    :func:`test_arm_gaps` asks, four times, whether a family's own gap differs
    from zero; ``18_donor_replication.py`` then asks whether that verdict repeats
    on every donor. Neither is a contrast, and the claim being made -- that the
    sequence tier is the family fragmentation hurts *most* -- is a contrast. A
    family can clear its own null on every donor while its gap is
    indistinguishable from the family it is being compared with, and two families
    can differ sharply while both clear their own nulls.

    So difference the two families *within* each matched unit. The units are
    already matched across families: at one (coverage, injection seed, donor)
    every family was fitted on the same two degraded copies of the same record.

    Because the claim is a conjunction -- worse than *every* other family -- the
    correct global procedure is an intersection-union test, in which each
    contrast is tested at ``alpha`` with no adjustment and rejection requires all
    of them to reject. Holm-adjusted values are reported alongside as the
    conservative alternative; which one licences the claim should be stated.

    Args:
        paired: Output of :func:`paired_arm_gaps`, optionally carrying a
            ``donor`` column so several grids can be pooled.
        reference: Family every other family is contrasted against.
        alpha: Per-contrast level for the IUT, and family-wise rate for Holm.
        n_boot: Bootstrap resamples for the interval.
        by: Optional column to condition on, one set of contrasts per level.
            Pooling across coverage hides that the separation between the
            sequence tier and the tree ensembles lives almost entirely at the
            most severe level, so ``by="target_coverage"`` is worth reporting
            beside the pooled result rather than instead of it.

    Returns:
        One row per non-reference family, most negative contrast first; with
        ``by`` set, one such block per level, carrying the level in a column.

    Raises:
        ValueError: If the reference family is absent from ``paired``.
    """
    if paired.empty:
        return pd.DataFrame()
    if reference not in set(paired["family"]):
        raise ValueError(
            f"reference family {reference!r} absent; have {sorted(set(paired['family']))}"
        )

    if by is not None:
        blocks = []
        for level, group in paired.groupby(by):
            block = family_contrasts(
                group, reference=reference, alpha=alpha, n_boot=n_boot, by=None
            )
            if block.empty:
                continue
            block.insert(0, by, level)
            blocks.append(block)
        return pd.concat(blocks, ignore_index=True) if blocks else pd.DataFrame()

    unit = ["target_coverage", "injection_seed"]
    if "donor" in paired.columns:
        unit.append("donor")

    wide = paired.pivot_table(index=unit, columns="family", values="arm_gap", aggfunc="first")
    others = [f for f in FAMILY_ORDER if f in wide.columns and f != reference]

    rows = []
    for family in others:
        both = wide[[reference, family]].dropna()
        diff = (both[reference] - both[family]).to_numpy(dtype=float)
        n = diff.size
        p = float("nan")
        if n >= 5 and np.any(diff != 0.0):
            p = float(stats.wilcoxon(diff, alternative="two-sided").pvalue)
        rng = np.random.default_rng(0)
        boot = (
            np.array([rng.choice(diff, size=n, replace=True).mean() for _ in range(n_boot)])
            if n >= 2
            else np.array([])
        )
        rows.append(
            {
                "family": family,
                "reference": reference,
                "n_units": n,
                "mean_contrast": float(diff.mean()) if n else float("nan"),
                "median_contrast": float(np.median(diff)) if n else float("nan"),
                "ci_low": float(np.percentile(boot, 2.5)) if boot.size else float("nan"),
                "ci_high": float(np.percentile(boot, 97.5)) if boot.size else float("nan"),
                "p_wilcoxon": p,
            }
        )

    out = pd.DataFrame(rows).sort_values("mean_contrast").reset_index(drop=True)
    if out.empty:
        return out
    out["reject_iut"] = out["p_wilcoxon"] < alpha
    adjusted = holm_bonferroni(out["p_wilcoxon"].tolist(), alpha=alpha)
    out["p_holm"] = [r["p_adjusted"] for r in adjusted]
    out["reject_holm"] = [r["reject"] for r in adjusted]
    return out


def dose_response(
    paired: pd.DataFrame,
    *,
    x: str = "target_coverage",
    unit: tuple[str, ...] = ("injection_seed",),
    scale: float = -0.10,
    alpha: float = 0.05,
    n_boot: int = 5000,
) -> pd.DataFrame:
    """Estimate, test and bound the slope of the arm gap in the dose.

    :func:`test_arm_gaps` pools every coverage level into one signed-rank test.
    That answers "is the gap nonzero on average" and cannot answer "does it grow
    as the record degrades", which is the arm-by-coverage interaction and the
    more informative quantity: a gap flat in coverage is a fixed cost, one that
    steepens is a mechanism. Pooling averages that interaction away, even though
    the per-level trend is the strongest visual evidence in the figure.

    Two stages, so the pairing survives. Within each unit -- one injection seed,
    which contributes the same arrangement draw at every level -- the gap is
    regressed on the dose to give one slope. The slopes are then tested against
    zero by Wilcoxon signed-rank and bounded by a percentile bootstrap over
    units. A paired t-test on the slopes is not used, for the reason given in
    :func:`test_arm_gaps`; a Gaussian random-effects model would reintroduce the
    same assumption one level up.

    Since ``arm_gap`` is already the arm contrast, its slope in coverage *is* the
    interaction, which answers the objection directly rather than around it.

    Args:
        paired: Output of :func:`paired_arm_gaps`.
        x: Dose column to regress on.
        unit: Columns identifying one cluster; one slope is fitted per cluster.
        scale: Reporting scale. The default expresses the slope per 10
            percentage points of coverage **lost**, the direction a reader
            thinks in.
        alpha: Family-wise error rate across families.
        n_boot: Bootstrap resamples for the interval.

    Returns:
        One row per family, steepest first.
    """
    if paired.empty:
        return pd.DataFrame()
    unit_cols = [c for c in unit if c in paired.columns]

    rows = []
    for family, group in paired.groupby("family"):
        clusters = group.groupby(list(unit_cols)) if unit_cols else [((), group)]
        slopes = []
        for _, cluster in clusters:
            xv = cluster[x].to_numpy(dtype=float)
            yv = cluster["arm_gap"].to_numpy(dtype=float)
            if xv.size < 3 or np.ptp(xv) == 0:
                continue
            slopes.append(float(np.polyfit(xv, yv, 1)[0]))
        arr = np.asarray(slopes, dtype=float)
        n = arr.size
        p = float("nan")
        if n >= 5 and np.any(arr != 0.0):
            p = float(stats.wilcoxon(arr, alternative="two-sided").pvalue)
        rng = np.random.default_rng(0)
        boot = (
            np.array([rng.choice(arr, size=n, replace=True).mean() for _ in range(n_boot)])
            if n >= 2
            else np.array([])
        )
        # A negative `scale` reverses the interval, so order the endpoints after
        # scaling rather than before.
        if boot.size:
            lo, hi = sorted(
                (float(np.percentile(boot, 2.5) * scale), float(np.percentile(boot, 97.5) * scale))
            )
        else:
            lo = hi = float("nan")
        rows.append(
            {
                "family": family,
                "n_units": n,
                "slope_per_unit_dose": float(arr.mean()) if n else float("nan"),
                "gap_change_per_10pp_lost": float(arr.mean() * scale) if n else float("nan"),
                "ci_low": lo,
                "ci_high": hi,
                "p_wilcoxon": p,
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.sort_values("gap_change_per_10pp_lost").reset_index(drop=True)
    adjusted = holm_bonferroni(out["p_wilcoxon"].tolist(), alpha=alpha)
    out["p_holm"] = [r["p_adjusted"] for r in adjusted]
    out["significant_holm"] = [r["reject"] for r in adjusted]
    return out
