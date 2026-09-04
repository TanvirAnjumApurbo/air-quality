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
from src.eval.ablation import (
    dose_response,
    family_contrasts,
    paired_arm_gaps,
    test_arm_gaps,
    tidy,
)
from src.eval.ablation import fmt_p as _fmt_p
from src.utils import load_config, setup_logging
from src.viz.figures import COL_DOUBLE, panel_label, save_figure, setup_style
from src.viz.tables import write_table


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--config", default="config_beijing.yaml")
    return p.parse_args()


def main() -> int:
    """Analyse the gap-injection grid."""
    args = parse_args()
    cfg = load_config(args.config)
    log = setup_logging(cfg, "17_ablation_analysis")

    path = Path(cfg.get("paths.results")) / str(
        cfg.get("ablation.gap_injection.output_name", "ablation_gap_injection.json")
    )
    if not path.exists():
        log.error("%s not found -- run scripts/16_gap_injection.py first", path)
        return 1
    payload = json.loads(path.read_text(encoding="utf-8"))
    frame = tidy(payload)
    if frame.empty:
        log.error("no cells recorded in %s", path)
        return 1

    # A partial grid must not be analysed as if it were whole. 16_gap_injection
    # aborts on a transient fault and resumes on the next invocation, so it is
    # entirely normal to find this file holding, say, 80 of 101 cells -- and
    # nothing downstream would notice. It matters here specifically because the
    # design is paired: a level whose two arms have different seed sets, or
    # fewer seeds than its neighbours, silently reweights the signed-rank test
    # toward whichever levels happen to have finished.
    spec_cfg = cfg.get("ablation.gap_injection") or {}
    expected_seeds = {int(s) for s in spec_cfg.get("injection_seeds", [])}
    expected_levels = {round(float(c), 4) for c in spec_cfg.get("coverage_levels", [])}
    degraded_cells = frame[frame["arm"] != "none"]
    missing: list[str] = []
    for level in sorted(expected_levels, reverse=True):
        for arm in ("fragmented", "contiguous"):
            have = set(
                degraded_cells[
                    (degraded_cells["arm"] == arm)
                    & (degraded_cells["target_coverage"].round(4) == level)
                ]["injection_seed"].astype(int)
            )
            for seed in sorted(expected_seeds - have):
                missing.append(f"{arm}_cov{level:.2f}_s{seed}")
    if missing:
        log.warning(
            "GRID INCOMPLETE: %d of %d degraded cells missing. The paired test below "
            "is computed on the cells that exist and is NOT the designed experiment. "
            "Re-run scripts/16_gap_injection.py to resume, then re-run this script. "
            "Missing: %s",
            len(missing),
            len(expected_levels) * 2 * len(expected_seeds),
            ", ".join(missing[:12]) + (" ..." if len(missing) > 12 else ""),
        )
        print("\n" + "!" * 78)
        print(f"GRID INCOMPLETE — {len(missing)} degraded cells missing; results are provisional")
        print("!" * 78)

    # Paired analysis first, on the frame that still has one row per injection
    # seed. Everything below averages the seeds away, and a difference of two
    # averages discards the pairing that makes this an experiment rather than a
    # comparison of two groups.
    paired = paired_arm_gaps(frame)
    tests = test_arm_gaps(paired) if not paired.empty else pd.DataFrame()

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
    # write_table creates this directory; these two go out through to_csv, which
    # does not. A radius config writes to a directory no earlier stage has made,
    # because the mediation arms run 16 and 17 and nothing else.
    tables_dir = cfg.path_for("tables")
    tables_dir.mkdir(parents=True, exist_ok=True)
    grouped.to_csv(tables_dir / "ablation_by_model.csv", index=False)
    best.to_csv(tables_dir / "ablation_by_family.csv", index=False)

    # ---- the key figure ----------------------------------------------------
    palette = setup_style(cfg)
    fig, (ax_skill, ax_rows) = plt.subplots(1, 2, figsize=(COL_DOUBLE, 3.0))
    order = ["sequence", "trees", "linear", "climatological"]
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
                markersize=3.2,
                capsize=1.8,
                elinewidth=0.7,
                capthick=0.7,
                linewidth=1.25,
                color=colours.get(family, "#666666"),
                label=f"{family} ({arm})",
            )

    ax_skill.axhline(0.0, color="#999999", linewidth=0.8, zorder=0)
    ax_skill.set_xlabel("Coverage of the training record (%)")
    ax_skill.set_ylabel("Skill vs persistence")
    ax_skill.invert_xaxis()
    # Below the axes: the crossing point is the finding, and a legend box sitting
    # in the middle of the panel lands squarely on it. Two rows of four keeps the
    # fragmented/contiguous pair for each family adjacent in the same column.
    ax_skill.legend(
        frameon=False,
        fontsize=6.8,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.24),
        columnspacing=0.9,
        handletextpad=0.4,
        handlelength=1.8,
        labelspacing=0.25,
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
            markersize=3.2,
            linewidth=1.25,
            color="#333333" if arm == "fragmented" else "#999999",
            label=arm,
        )
    ax_rows.set_xlabel("Coverage of the training record (%)")
    ax_rows.set_ylabel("Usable supervised training rows")
    ax_rows.invert_xaxis()
    ax_rows.legend(frameon=False, loc="lower left")

    panel_label(ax_skill, "(a)")
    panel_label(ax_rows, "(b)")
    fig.tight_layout(w_pad=1.8)
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

    if not tests.empty:
        display = tests.assign(
            **{
                "Family": tests["family"],
                "Pairs": tests["n_pairs"],
                "Mean gap": tests["mean_arm_gap"].round(4),
                "95% CI": [
                    f"[{lo:+.4f}, {hi:+.4f}]"
                    for lo, hi in zip(tests["ci_low"], tests["ci_high"], strict=False)
                ],
                "p (Wilcoxon)": tests["p_wilcoxon"].map(_fmt_p),
                "p (Holm)": tests["p_holm"].map(_fmt_p),
            }
        )[["Family", "Pairs", "Mean gap", "95% CI", "p (Wilcoxon)", "p (Holm)"]]
        write_table(
            cfg,
            display,
            "ablation_paired_test",
            caption=(
                "Paired arm gap by model family. Each pair is one (coverage level, "
                "injection seed): both arms remove the same number of observed hours "
                "and differ only in arrangement, so the within-pair difference is the "
                "effect of fragmentation with volume held constant. The representative "
                "model per family is fixed on the undegraded record, never re-chosen "
                "per arm. Wilcoxon signed-rank, Holm-corrected across families. The "
                "climatological family is the weakest model that DOES read the record "
                "-- hour-of-day and month means, insensitive to how the observed hours "
                "are arranged but fitted on them. It is not a no-training-data control: "
                "because the test period is never degraded, any model reading nothing "
                "from training has an arm gap of exactly zero here by construction, so "
                "it cannot vary. That invariance is checked by equality instead."
            ),
        )
        log.info("paired arm-gap tests:\n%s", display.to_string(index=False))
        print("\n" + "=" * 78)
        print("PAIRED ARM-GAP TEST — one pair per (coverage, injection seed)")
        print("=" * 78)
        print(display.to_string(index=False))

    # ---- between-family contrast (the claim is a comparison, so test one) ---
    contrasts = family_contrasts(paired) if not paired.empty else pd.DataFrame()
    contrasts_by_level = (
        family_contrasts(paired, by="target_coverage") if not paired.empty else pd.DataFrame()
    )
    if not contrasts.empty:
        disp = contrasts.assign(
            **{
                "Family": contrasts["family"],
                "Units": contrasts["n_units"],
                "Mean contrast": contrasts["mean_contrast"].round(4),
                "Median": contrasts["median_contrast"].round(4),
                "95% CI": [
                    f"[{lo:+.4f}, {hi:+.4f}]"
                    for lo, hi in zip(contrasts["ci_low"], contrasts["ci_high"], strict=False)
                ],
                "p (Wilcoxon)": contrasts["p_wilcoxon"].map(_fmt_p),
                "p (Holm)": contrasts["p_holm"].map(_fmt_p),
            }
        )[["Family", "Units", "Mean contrast", "Median", "95% CI", "p (Wilcoxon)", "p (Holm)"]]
        write_table(
            cfg,
            disp,
            "ablation_family_contrasts",
            caption=(
                "Arm gap of the sequence family MINUS each other family's, differenced "
                "within the same (coverage level, injection seed). Testing each family "
                "against its own null and then requiring the verdict to repeat is not a "
                "contrast: a family can clear its own null everywhere while remaining "
                "indistinguishable from the family it is compared against. Because the "
                "claim is a conjunction -- worse than every other family -- rejection "
                "requires all rows to reject, each at the unadjusted level "
                "(intersection-union); Holm is reported as the conservative alternative."
            ),
        )
        log.info("family contrasts:%s%s", chr(10), disp.to_string(index=False))

    # ---- dose-response (the pooled test averages the interaction away) -----
    doses = dose_response(paired) if not paired.empty else pd.DataFrame()
    if not doses.empty:
        disp = doses.assign(
            **{
                "Family": doses["family"],
                "Units": doses["n_units"],
                "Gap per 10pp lost": doses["gap_change_per_10pp_lost"].round(4),
                "95% CI": [
                    f"[{lo:+.4f}, {hi:+.4f}]"
                    for lo, hi in zip(doses["ci_low"], doses["ci_high"], strict=False)
                ],
                "p (Wilcoxon)": doses["p_wilcoxon"].map(_fmt_p),
                "p (Holm)": doses["p_holm"].map(_fmt_p),
            }
        )[["Family", "Units", "Gap per 10pp lost", "95% CI", "p (Wilcoxon)", "p (Holm)"]]
        write_table(
            cfg,
            disp,
            "ablation_dose_response",
            caption=(
                "Change in the arm gap per 10 percentage points of coverage lost. One "
                "slope is fitted within each injection seed, which contributes the same "
                "arrangement draw at every level, and the slopes are then tested against "
                "zero by Wilcoxon signed-rank. Since the arm gap is already the arm "
                "contrast, its slope in coverage is the arm-by-coverage interaction -- "
                "the quantity the pooled test averages away. A gap flat in coverage is a "
                "fixed cost; one that steepens is a mechanism."
            ),
        )
        log.info("dose-response:%s%s", chr(10), disp.to_string(index=False))

    # ---- design check: the protected test period makes persistence a constant
    # An equality that must hold is a better control than a test that can only
    # fail to reject. Persistence reads nothing from training and the test period
    # is never degraded, so its RMSE must be identical in every cell.
    pers = frame[frame["model"] == "persistence"]["rmse"].to_numpy(dtype=float)
    pers_spread = float(pers.max() - pers.min()) if pers.size else float("nan")
    pers_invariant = bool(pers.size and pers_spread < 1e-9)
    if not pers_invariant and pers.size:
        log.error(
            "persistence test RMSE varies across cells by %.3e; the test period was "
            "supposed to be protected, so this is a design violation, not noise",
            pers_spread,
        )
    else:
        log.info("persistence RMSE invariant across %d cells (spread %.2e)", pers.size, pers_spread)

    payload["analysis"] = {
        "by_family": best.to_dict(orient="records"),
        "family_contrasts": contrasts.to_dict(orient="records") if not contrasts.empty else [],
        "family_contrasts_by_level": (
            contrasts_by_level.to_dict(orient="records") if not contrasts_by_level.empty else []
        ),
        "dose_response": doses.to_dict(orient="records") if not doses.empty else [],
        "persistence_rmse_invariant": pers_invariant,
        "persistence_rmse_spread": pers_spread,
        "family_ranks": json.loads(pivot.to_json()),
        "paired_gaps": paired.to_dict(orient="records") if not paired.empty else [],
        "paired_tests": tests.to_dict(orient="records") if not tests.empty else [],
        # Carried into the payload so the report can say "provisional" on the
        # reader's behalf. Without it a partial grid renders as a finished
        # section, which is the failure mode this whole guard exists for.
        "grid_complete": not missing,
        "missing_cells": missing,
        "n_expected_degraded_cells": len(expected_levels) * 2 * len(expected_seeds),
    }
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {path} and results/figures/.../fig11_gap_injection")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
