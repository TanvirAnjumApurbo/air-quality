r"""Phase 7c: does the gap-injection result hold on more than one donor record?

``17_ablation_analysis.py`` reports one donor. On a single record a family's
sensitivity to fragmentation is indistinguishable from a property of that
station, so this script reads every ``ablation_gap_injection*.json`` under
``paths.results`` and asks the only question that matters after the first run:

* **does each family's arm gap replicate** -- same sign, and significant under
  Holm, on every donor;
* **does the coverage profile replicate** -- specifically whether the climatological
  control's collapse at severe fragmentation is a general effect or one donor's
  quirk, since that is what decides whether the effect can be called specific to
  models that need contiguous windows.

The paired test is imported from ``src.eval.ablation`` rather than reimplemented,
so a disagreement between donors cannot be an artefact of two scripts computing
the arm gap differently.

What replication does and does not buy here is stated in ``donors.yaml``: the
donors are further stations of one archive, so this is a station-robustness
check and not a multi-city panel. It can show a result is donor-specific, which
is the cheapest way to falsify it; it cannot establish generality.

Run::

    python scripts/18_donor_replication.py --config config_beijing.yaml
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
    FAMILY_ORDER,
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

#: Newline, so log format arguments need no backslash escapes.
NL = chr(10)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--config", default="config_beijing.yaml")
    p.add_argument(
        "--glob",
        default="ablation_gap_injection*.json",
        help="pattern for donor grids under paths.results",
    )
    return p.parse_args()


def _short(donor: str) -> str:
    """Shorten a donor label to its station name.

    Args:
        donor: Full donor label, e.g. ``"Beijing Wanliu (UCI Multi-Site, id 501)"``.

    Returns:
        The station name where one can be recovered, else the input.
    """
    parts = donor.split()
    return parts[1] if len(parts) > 1 and parts[0] == "Beijing" else donor


def main() -> int:
    """Compare the gap-injection result across donor records."""
    args = parse_args()
    cfg = load_config(args.config)
    log = setup_logging(cfg, "18_donor_replication")

    results_dir = Path(cfg.get("paths.results"))
    paths = sorted(results_dir.glob(args.glob))
    if not paths:
        log.error("no donor grids matching %s under %s", args.glob, results_dir)
        return 1

    per_donor: list[pd.DataFrame] = []
    per_level: list[pd.DataFrame] = []
    incomplete: list[str] = []
    paired_by_donor: dict[str, pd.DataFrame] = {}

    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        donor = _short(str(payload.get("donor", path.stem)))
        frame = tidy(payload)
        if frame.empty:
            log.warning("%s holds no cells; skipped", path.name)
            continue
        if payload.get("analysis", {}).get("grid_complete") is False:
            incomplete.append(donor)

        paired = paired_arm_gaps(frame)
        if paired.empty:
            log.warning("%s has no matched pairs; skipped", path.name)
            continue

        paired_by_donor[donor] = paired
        tests = test_arm_gaps(paired)
        tests.insert(0, "donor", donor)
        per_donor.append(tests)

        levels = (
            paired.groupby(["family", "target_coverage"], as_index=False)["arm_gap"]
            .mean()
            .assign(donor=donor)
        )
        per_level.append(levels)
        log.info(
            "%s: %d cells, %d matched pairs per family",
            donor,
            len(payload["cells"]),
            len(paired) // 4,
        )

    if len(per_donor) < 2:
        log.error(
            "only %d donor grid(s) usable; run scripts/16_gap_injection.py for the "
            "replication donors (see donors.yaml) before this script",
            len(per_donor),
        )
        return 1

    tests_all = pd.concat(per_donor, ignore_index=True)
    levels_all = pd.concat(per_level, ignore_index=True)
    donors = list(dict.fromkeys(tests_all["donor"]))

    if incomplete:
        log.warning(
            "donor grid(s) %s are incomplete; their columns are provisional",
            ", ".join(incomplete),
        )

    # ---- does each family replicate? ---------------------------------------
    # Replication is deliberately strict: same sign on every donor AND
    # significant under Holm on every donor. A family that is significant on one
    # donor and null on another has not replicated, and reporting it as though
    # the significant donor settled the matter is the error this script exists
    # to prevent.
    verdict_rows = []
    for family in FAMILY_ORDER:
        sub = tests_all[tests_all["family"] == family]
        if sub.empty:
            continue
        signs = {"-" if g < 0 else "+" for g in sub["mean_arm_gap"]}
        n_sig = int(sub["significant_holm"].sum())
        replicates = len(signs) == 1 and n_sig == len(sub)
        verdict_rows.append(
            {
                "family": family,
                "n_donors": len(sub),
                "mean_gap_across_donors": float(sub["mean_arm_gap"].mean()),
                "min_gap": float(sub["mean_arm_gap"].min()),
                "max_gap": float(sub["mean_arm_gap"].max()),
                "n_donors_significant": n_sig,
                "same_sign": len(signs) == 1,
                "replicates": replicates,
            }
        )
    verdict = pd.DataFrame(verdict_rows)

    # ---- what the replication rule does NOT test ---------------------------
    # Replicating means a family cleared its own null on every donor. It does
    # not mean the family differs from any OTHER family, which is what "the
    # tier fragmentation hurts most" asserts. Pool the matched triples across
    # donors and contrast the families directly; the unit is already matched
    # across families, since at one (coverage, seed, donor) every family was
    # fitted on the same two degraded copies of the same record.
    pooled = pd.concat(
        [g.assign(donor=d) for d, g in paired_by_donor.items() if not g.empty],
        ignore_index=True,
    )
    contrasts = family_contrasts(pooled) if not pooled.empty else pd.DataFrame()
    contrasts_by_level = (
        family_contrasts(pooled, by="target_coverage") if not pooled.empty else pd.DataFrame()
    )
    doses = (
        dose_response(pooled, unit=("injection_seed", "donor"))
        if not pooled.empty
        else pd.DataFrame()
    )
    iut_rejects = bool(
        not contrasts.empty
        and contrasts["reject_iut"].all()
        and (contrasts["mean_contrast"] < 0).all()
    )

    # ---- tables ------------------------------------------------------------
    wide = tests_all.pivot_table(
        index="family", columns="donor", values="mean_arm_gap", aggfunc="first"
    )
    sig = tests_all.pivot_table(
        index="family", columns="donor", values="significant_holm", aggfunc="first"
    )
    order = [f for f in FAMILY_ORDER if f in wide.index]
    display = pd.DataFrame({"Family": order})
    for donor in donors:
        display[donor] = [
            f"{wide.loc[f, donor]:+.4f}{'*' if bool(sig.loc[f, donor]) else ''}" for f in order
        ]
    display["Replicates"] = [
        "yes" if bool(verdict.set_index("family").loc[f, "replicates"]) else "no" for f in order
    ]
    write_table(
        cfg,
        display,
        "donor_replication",
        caption=(
            "Paired arm gap (fragmented minus contiguous skill) by model family and "
            "donor record, each from its own matched (coverage level, injection seed) "
            "pairs. An asterisk marks significance under Holm within that donor. "
            "'Replicates' requires the same sign and Holm significance on every "
            "donor, so a family significant on one record and null on another is "
            "reported as not replicating. All donors are stations of one archive: "
            "this is a station-robustness check, not a multi-city panel."
        ),
    )

    level_table = levels_all.pivot_table(
        index=["family", "donor"], columns="target_coverage", values="arm_gap", aggfunc="first"
    )
    level_table = level_table[sorted(level_table.columns, reverse=True)]
    level_table.columns = [f"{c * 100:.0f}%" for c in level_table.columns]
    idx = [(f, d) for f in FAMILY_ORDER for d in donors if (f, d) in level_table.index]
    level_shown = level_table.loc[idx].round(4).reset_index()
    level_shown.columns = ["Family", "Donor", *list(level_shown.columns[2:])]
    write_table(
        cfg,
        level_shown,
        "donor_replication_by_level",
        caption=(
            "Paired arm gap by family, donor and injected coverage level. The naive "
            "row is the one to read first: it is the control only where it stays "
            "near zero, and a collapse at severe fragmentation that appears on every "
            "donor means fragmentation degrades anything estimated from the record "
            "rather than the windowed model class specifically."
        ),
    )

    # ---- figure ------------------------------------------------------------
    palette = setup_style(cfg)
    families = [f for f in FAMILY_ORDER if f in set(levels_all["family"])]
    fig, axes = plt.subplots(1, len(families), figsize=(COL_DOUBLE, 2.35), sharey=True)
    axes = [axes] if len(families) == 1 else list(axes)
    colours = dict(zip(donors, palette, strict=False))
    tags = "abcdefgh"
    levels = sorted(levels_all["target_coverage"].unique() * 100.0)

    for i, (ax, family) in enumerate(zip(axes, families, strict=False)):
        for donor in donors:
            sub = levels_all[
                (levels_all["family"] == family) & (levels_all["donor"] == donor)
            ].sort_values("target_coverage", ascending=False)
            if sub.empty:
                continue
            ax.plot(
                sub["target_coverage"] * 100.0,
                sub["arm_gap"],
                marker="o",
                markersize=2.8,
                linewidth=1.15,
                color=colours.get(donor, "#666666"),
                label=donor,
            )
        ax.axhline(0.0, color="#999999", linewidth=0.8, zorder=0)
        # The family names are data categories, not chart titles: they identify
        # which model family the panel is about and the caption refers to them.
        panel_label(ax, f"({tags[i]}) {family}")
        ax.set_xlabel("Injected coverage (%)")
        # Tick the coverage levels that were actually injected. The default
        # locator picks round numbers, and at four panels across a page that left
        # two labelled ticks, neither of them a level in the grid.
        ax.set_xticks(levels)
        ax.set_xticklabels([f"{lv:g}" for lv in levels], fontsize=6.6)
        ax.invert_xaxis()
    axes[0].set_ylabel("Fragmented $-$ contiguous skill")
    axes[-1].legend(frameon=True, framealpha=0.92, facecolor="white", edgecolor="#CCCCCC")
    fig.tight_layout(w_pad=0.9)
    written = save_figure(cfg, fig, "fig12_donor_replication")
    log.info("wrote %s", ", ".join(p.name for p in written))
    plt.close(fig)

    # ---- payload -----------------------------------------------------------
    # Its own file, alongside ablation_gap_injection*.json, for the same reason
    # those exist: the experiment spans several donor configs and so has no one
    # city's results.json to live in. Written by code, never by hand.
    if not contrasts.empty:
        disp = contrasts.assign(
            **{
                "Family": contrasts["family"],
                "Triples": contrasts["n_units"],
                "Mean contrast": contrasts["mean_contrast"].round(4),
                "Median": contrasts["median_contrast"].round(4),
                "95% CI": [
                    f"[{lo:+.4f}, {hi:+.4f}]"
                    for lo, hi in zip(contrasts["ci_low"], contrasts["ci_high"], strict=False)
                ],
                "p (Wilcoxon)": contrasts["p_wilcoxon"].map(_fmt_p),
                "p (Holm)": contrasts["p_holm"].map(_fmt_p),
            }
        )[["Family", "Triples", "Mean contrast", "Median", "95% CI", "p (Wilcoxon)", "p (Holm)"]]
        write_table(
            cfg,
            disp,
            "donor_family_contrasts",
            caption=(
                "Sequence-family arm gap MINUS each other family's, differenced within "
                "the same (coverage level, injection seed, donor) triple and pooled "
                "across donors. The replication rule above asks whether a family clears "
                "its OWN null on every donor; it cannot show that one family is affected "
                "more than another, which is the comparison the headline claim makes. "
                "Rejection of that claim requires every row to reject "
                "(intersection-union), so a single non-rejecting row withholds it."
            ),
        )
        log.info("pooled family contrasts:%s%s", NL, disp.to_string(index=False))

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
            "donor_dose_response",
            caption=(
                "Change in the arm gap per 10 percentage points of coverage lost, one "
                "slope per (injection seed, donor). The pooled signed-rank test averages "
                "the arm-by-coverage interaction away; this estimates it. A gap flat in "
                "coverage is a fixed cost, one that steepens is a mechanism."
            ),
        )
        log.info("pooled dose-response:%s%s", NL, disp.to_string(index=False))

    out = results_dir / "donor_replication.json"
    out.write_text(
        json.dumps(
            {
                "donors": donors,
                "incomplete_donors": incomplete,
                "grids": [str(p.name) for p in paths],
                "per_donor_tests": tests_all.to_dict(orient="records"),
                "per_level_gaps": levels_all.to_dict(orient="records"),
                "verdict": verdict.to_dict(orient="records"),
                "family_contrasts_pooled": contrasts.to_dict(orient="records"),
                "family_contrasts_by_level": contrasts_by_level.to_dict(orient="records"),
                "dose_response_pooled": doses.to_dict(orient="records"),
                "iut_sequence_worse_than_every_family": iut_rejects,
                "contrast_note": (
                    "The replication rule tests each family against its own null and "
                    "asks whether that verdict repeats across donors. It is not a "
                    "contrast between families, so it cannot support a claim that one "
                    "family is affected more than another. family_contrasts_pooled "
                    "differences the families within matched triples and tests that "
                    "claim directly; the conjunction is an intersection-union test, so "
                    "iut_sequence_worse_than_every_family is true only if every "
                    "contrast rejects."
                ),
                "replication_rule": (
                    "A family replicates only if its paired arm gap has the same sign "
                    "on every donor and is significant under Holm on every donor."
                ),
                "scope_note": (
                    "All donors are stations of the UCI Beijing Multi-Site archive "
                    "(id 501): one four-year window, one regional weather regime, "
                    "spatially correlated PM2.5. This is a station-robustness check, "
                    "not evidence of generality across records or cities."
                ),
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    # ---- console -----------------------------------------------------------
    print("\n" + "=" * 78)
    print(f"DONOR REPLICATION — {len(donors)} donors: {', '.join(donors)}")
    print("=" * 78)
    for donor in donors:
        sub = tests_all[tests_all["donor"] == donor].sort_values("mean_arm_gap")
        print(f"\n{donor}")
        for _, r in sub.iterrows():
            mark = "*" if r["significant_holm"] else " "
            print(
                f"  {r['family']:<9} {r['mean_arm_gap']:+.4f}{mark} "
                f"[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]  Holm {_fmt_p(r['p_holm'])}"
            )
    print("\n" + "-" * 78)
    print("REPLICATES (same sign and Holm-significant on every donor):")
    for _, r in verdict.iterrows():
        print(
            f"  {r['family']:<9} {'YES' if r['replicates'] else 'no ':<4} "
            f"gap {r['min_gap']:+.4f} to {r['max_gap']:+.4f}, "
            f"significant on {int(r['n_donors_significant'])}/{int(r['n_donors'])}"
        )
    if incomplete:
        print(f"\nPROVISIONAL: incomplete grid(s) for {', '.join(incomplete)}")
    if not contrasts.empty:
        print(f"{NL}between-family contrast (sequence minus family, pooled triples)")
        for r in contrasts.itertuples():
            mark = "*" if r.reject_iut else " "
            print(
                f"  vs {r.family:<15} {r.mean_contrast:+.4f} "
                f"[{r.ci_low:+.4f}, {r.ci_high:+.4f}]  p={_fmt_p(r.p_wilcoxon)}{mark}"
            )
        print(f"  IUT -- sequence worse than EVERY other family: {'YES' if iut_rejects else 'NO'}")
    print()
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
