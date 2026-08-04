r"""Phase 7b: read the gap-injection grid and produce the paper's central figure.

The experiment in ``16_gap_injection.py`` degrades one near-complete record along
two arms that remove identical numbers of hours and differ only in how those
hours are arranged. This script turns that grid into the claim.

What it reports:

* **the key figure** -- skill against coverage, one line per model family, the
  fragmented arm solid and the contiguous control dashed. If the sequence models
  fall away in the fragmented arm and hold in the contiguous one, fragmentation
  is doing the work and data volume is not;
* **the rank-reversal table** -- where each family sits at each coverage level in
  each arm, which is the finding stated numerically;
* **the arm gap** -- fragmented minus contiguous skill at matched coverage, the
  cleanest single-number statement of the effect;
* **the mechanism column** -- usable training rows, showing that fragmentation
  destroys far more supervised data than it removes hours.

Run::

    python scripts/17_ablation_analysis.py --config config_beijing.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import pandas as pd
from src.utils import load_config, setup_logging
from src.viz.figures import save_figure, setup_style
from src.viz.tables import write_table

# Grouped so the figure reads as "model class", which is the comparison the
# experiment is about -- not nine individually coloured models.
FAMILIES = {
    "sequence": ["gru_h64_l2", "lstm_h64_l1", "gru", "lstm"],
    "linear": ["ridge", "dlinear", "nlinear"],
    "trees": ["random_forest", "xgboost", "lightgbm"],
    "naive": ["persistence", "climatology", "seasonal_naive"],
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--config", default="config_beijing.yaml")
    return p.parse_args()


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


def tidy(payload: dict) -> pd.DataFrame:
    """Flatten the nested cell structure into one row per (cell, model).

    Args:
        payload: Contents of ``ablation_gap_injection.json``.

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


def main() -> int:
    """Analyse the gap-injection grid."""
    args = parse_args()
    cfg = load_config(args.config)
    log = setup_logging(cfg, "17_ablation_analysis")

    path = Path(cfg.get("paths.results")) / "ablation_gap_injection.json"
    if not path.exists():
        log.error("%s not found -- run scripts/16_gap_injection.py first", path)
        return 1
    payload = json.loads(path.read_text(encoding="utf-8"))
    frame = tidy(payload)
    if frame.empty:
        log.error("no cells recorded in %s", path)
        return 1

    # The undegraded cell is the reference for both arms, so it is duplicated
    # into each rather than floating outside the comparison.
    reference = frame[frame["arm"] == "none"]
    arms = ["fragmented", "contiguous"]
    expanded = [frame[frame["arm"] != "none"]]
    for arm in arms:
        copy = reference.copy()
        copy["arm"] = arm
        expanded.append(copy)
    frame = pd.concat(expanded, ignore_index=True)

    # Average over injection seeds: a single arrangement is one draw, not a level.
    grouped = (
        frame.groupby(["arm", "target_coverage", "family", "model"], as_index=False)
        .agg(
            skill=("skill", "mean"),
            skill_std=("skill", "std"),
            rmse=("rmse", "mean"),
            train_rows=("train_rows", "mean"),
            n_gaps=("n_gaps", "mean"),
            coverage=("coverage", "mean"),
            n_draws=("skill", "size"),
        )
        .fillna({"skill_std": 0.0})
    )
    # One representative per family: the best model in that family per cell,
    # since the paper's claim is about model classes rather than individual runs.
    best = (
        grouped.sort_values("skill", ascending=False)
        .groupby(["arm", "target_coverage", "family"], as_index=False)
        .first()
    )
    grouped.to_csv(cfg.path_for("tables") / "ablation_by_model.csv", index=False)
    best.to_csv(cfg.path_for("tables") / "ablation_by_family.csv", index=False)

    # ---- the key figure ----------------------------------------------------
    palette = setup_style(cfg)
    fig, (ax_skill, ax_rows) = plt.subplots(1, 2, figsize=(11.5, 4.6))
    order = ["sequence", "trees", "linear", "naive"]
    colours = dict(zip(order, palette, strict=False))

    for family in order:
        for arm, style, marker in (("fragmented", "-", "o"), ("contiguous", "--", "s")):
            sub = best[(best["family"] == family) & (best["arm"] == arm)].sort_values(
                "target_coverage", ascending=False
            )
            if sub.empty:
                continue
            ax_skill.errorbar(
                sub["coverage"] * 100.0,
                sub["skill"],
                yerr=sub["skill_std"],
                linestyle=style,
                marker=marker,
                markersize=4.5,
                capsize=2.5,
                linewidth=1.7,
                color=colours.get(family, "#666666"),
                label=f"{family} ({arm})",
            )

    ax_skill.axhline(0.0, color="#999999", linewidth=0.9, zorder=0)
    ax_skill.set_xlabel("Coverage of the training record (%)")
    ax_skill.set_ylabel("Skill vs persistence")
    ax_skill.invert_xaxis()
    ax_skill.set_title("Solid: fragmented removal.  Dashed: same hours, contiguous.")
    # Below the axes: the crossing point is the finding, and a legend box sitting
    # in the middle of the panel lands squarely on it.
    ax_skill.legend(
        frameon=False,
        fontsize=8,
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        columnspacing=1.0,
        handletextpad=0.4,
    )

    for arm, style, marker in (("fragmented", "-", "o"), ("contiguous", "--", "s")):
        sub = (
            best[best["arm"] == arm]
            .drop_duplicates(subset=["target_coverage"])
            .sort_values("target_coverage", ascending=False)
        )
        ax_rows.plot(
            sub["coverage"] * 100.0,
            sub["train_rows"],
            style,
            marker=marker,
            markersize=4.5,
            linewidth=1.7,
            color="#333333" if arm == "fragmented" else "#999999",
            label=arm,
        )
    ax_rows.set_xlabel("Coverage of the training record (%)")
    ax_rows.set_ylabel("Usable supervised training rows")
    ax_rows.invert_xaxis()
    ax_rows.set_title("The mechanism: identical hours removed, different yield")
    ax_rows.legend(frameon=False, fontsize=9)

    fig.suptitle(
        "Record fragmentation, not data volume, reverses the method ranking",
        fontsize=12,
        y=1.02,
    )
    written = save_figure(cfg, fig, "fig11_gap_injection")
    log.info("wrote %s", ", ".join(str(p.name) for p in written))
    plt.close(fig)

    # ---- rank reversal, stated numerically ---------------------------------
    best = best.copy()
    best["rank"] = best.groupby(["arm", "target_coverage"])["skill"].rank(
        ascending=False, method="min"
    )
    pivot = best.pivot_table(
        index="family", columns=["arm", "target_coverage"], values="rank", aggfunc="first"
    )
    log.info("family ranks by arm and coverage:\n%s", pivot.to_string())

    # ---- the arm gap -------------------------------------------------------
    wide = best.pivot_table(
        index=["family", "target_coverage"], columns="arm", values="skill", aggfunc="first"
    ).reset_index()
    if {"fragmented", "contiguous"} <= set(wide.columns):
        wide["arm_gap"] = wide["fragmented"] - wide["contiguous"]
        display = wide.assign(
            **{
                "Family": wide["family"],
                "Coverage": (wide["target_coverage"] * 100).round(0).astype(int),
                "Fragmented": wide["fragmented"].round(4),
                "Contiguous": wide["contiguous"].round(4),
                "Difference": wide["arm_gap"].round(4),
            }
        )[["Family", "Coverage", "Fragmented", "Contiguous", "Difference"]].sort_values(
            ["Family", "Coverage"], ascending=[True, False]
        )
        write_table(
            cfg,
            display,
            "ablation_arm_gap",
            caption=(
                "Skill vs persistence under the two removal arms at matched coverage. "
                "Both arms remove an identical number of observed hours at each level; "
                "only their arrangement differs, so the Difference column isolates the "
                "effect of fragmentation with data volume held constant. Every cell is "
                "scored on the same, never-degraded test period."
            ),
        )
        print("\n" + "=" * 78)
        print("ARM GAP — fragmented minus contiguous skill, at matched coverage")
        print("=" * 78)
        print(display.to_string(index=False))

        # The undegraded reference is shared by both arms, so its gap is zero by
        # construction; averaging it in would dilute the effect toward zero.
        degraded = wide[wide["target_coverage"] < 1.0]
        summary = degraded.groupby("family")["arm_gap"].mean().sort_values().round(4).to_dict()
        log.info("mean arm gap by family (degraded cells only): %s", summary)
        print("\nMean arm gap by family, degraded cells only")
        print("(negative = fragmentation hurts this family more than volume loss alone):")
        for family, gap in sorted(summary.items(), key=lambda kv: kv[1]):
            print(f"  {family:<10} {gap:+.4f}")

    payload["analysis"] = {
        "by_family": best.to_dict(orient="records"),
        "family_ranks": json.loads(pivot.to_json()),
    }
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {path} and results/figures/.../fig11_gap_injection")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
