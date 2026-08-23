r"""Phase 8d: is the sterilisation radius the mediator of the fragmentation penalty?

Sections 8 and 9 make two claims that have so far only been placed side by side.
The gap-injection experiment shows that fragmented removal costs more skill than
contiguous removal at an identical hour count; the amplification law shows that a
gap sterilises the ``lookback + horizon`` hours behind it, so scattering removals
destroys far more supervision than clustering them.

If the second explains the first, shortening the radius must shrink the arm gap.
That is falsifiable, and this script tests it.

The design is a repeated measure. ``inject_gaps`` is deterministic in
``(arm, coverage, seed)`` and runs on the raw record *before* any feature is
built, so the degraded series at one radius is the same series as at another --
the radius changes only what the feature builder can then make of it. Every
(coverage, seed) draw therefore appears at every radius, and the comparison is
within-draw rather than between-grids. The script verifies that identity from the
recorded injection statistics and refuses to compare radii whose draws disagree.

This is not Baron-Kenny and must not be described as such. The putative mediator
is **manipulated directly** by configuration rather than inferred from a
regression, which is strictly stronger and needs no sequential-ignorability
assumption.

Three controls, all of which can falsify the claim:

* the ``linear`` family had a null arm gap on this donor, so its mediation slope
  must also be null;
* the mediator has to be shown to move -- the fragmented-minus-contiguous deficit
  in usable training rows must shrink with the radius, and if it does not there is
  no mediator and nothing here should be reported;
* persistence never reads the training record and the test period is never
  degraded, so its RMSE must be identical across every cell of every radius.

Run::

    python scripts/23_mediation.py --config config_beijing.yaml --slug wanliu
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from src.eval.ablation import FAMILY_ORDER, paired_arm_gaps, tidy
from src.eval.ablation import fmt_p as _fmt_p
from src.eval.metrics import holm_bonferroni
from src.features.build_features import max_backward_dependency
from src.utils import load_config, setup_logging
from src.viz.figures import COL_DOUBLE, panel_label, save_figure, setup_style
from src.viz.tables import write_table

NL = chr(10)

#: Injection fields that must agree across radii for a draw to be the same draw.
DRAW_IDENTITY = ("hours_removed", "n_gaps_after", "longest_gap_after", "achieved_coverage")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--config", default="config_beijing.yaml")
    p.add_argument("--slug", default="wanliu")
    p.add_argument("--alpha", type=float, default=0.05)
    return p.parse_args()


def load_radius_grids(results_dir: Path, slug: str, default_radius: int) -> dict[int, dict]:
    """Read every radius grid for one donor.

    Args:
        results_dir: Directory holding the grids.
        slug: Donor slug used in the filenames.
        default_radius: Radius to attribute to a base grid that predates the
            stamped field.

    Returns:
        Radius in hours to payload.
    """
    grids: dict[int, dict] = {}

    # The base grid is the status-quo radius. It is read here rather than
    # regenerated: a capped config would land in a separate checkpoint tree and
    # retrain 101 cells to reproduce a file already on disk.
    for name in (f"ablation_gap_injection_{slug}.json", "ablation_gap_injection.json"):
        base = results_dir / name
        if base.exists():
            payload = json.loads(base.read_text(encoding="utf-8"))
            if payload.get("cells"):
                grids[int(payload.get("sterilisation_radius_h", default_radius))] = payload
            break

    for path in sorted(results_dir.glob(f"ablation_gap_injection_{slug}_R*.json")):
        m = re.search(r"_R(\d+)\.json$", path.name)
        if not m:
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not payload.get("cells"):
            continue
        grids[int(m.group(1))] = payload
    return grids


def draw_fingerprints(payload: dict) -> dict[tuple, tuple]:
    """Identify each degraded draw by what the injection actually did.

    Args:
        payload: One grid payload.

    Returns:
        ``(arm, coverage, seed)`` to a tuple of injection statistics.
    """
    out: dict[tuple, tuple] = {}
    for cell in payload.get("cells", {}).values():
        inj = cell.get("injection", {})
        key = (cell.get("arm"), cell.get("target_coverage"), cell.get("injection_seed"))
        out[key] = tuple(round(float(inj.get(f, float("nan"))), 6) for f in DRAW_IDENTITY)
    return out


def check_draws_match(grids: dict[int, dict], log: object) -> list[str]:
    """Verify the same injected series appears at every radius.

    The repeated-measure design is what gives this test its power; if the draws
    differ, the comparison is between-grids and the pairing is fiction.

    Args:
        grids: Radius to payload.
        log: Logger.

    Returns:
        Human-readable disagreements; empty when the draws match.
    """
    radii = sorted(grids)
    base_r = radii[0]
    base = draw_fingerprints(grids[base_r])
    problems: list[str] = []
    for r in radii[1:]:
        other = draw_fingerprints(grids[r])
        shared = set(base) & set(other)
        differing = [k for k in sorted(shared, key=str) if base[k] != other[k]]
        if differing:
            problems.append(
                f"R={r} differs from R={base_r} on {len(differing)} draw(s), "
                f"first {differing[0]}: {base[differing[0]]} vs {other[differing[0]]}"
            )
        missing = set(base) - set(other)
        if missing:
            problems.append(f"R={r} is missing {len(missing)} draw(s) present at R={base_r}")
    log.info(  # type: ignore[attr-defined]
        "draw identity across %d radii: %s", len(radii), "OK" if not problems else "FAILED"
    )
    return problems


def main() -> int:
    """Test whether the arm gap is mediated by the sterilisation radius."""
    args = parse_args()
    cfg = load_config(args.config)
    log = setup_logging(cfg, "23_mediation")
    results_dir = cfg.path_for("results")

    default_radius = max(
        max_backward_dependency(cfg),
        max(int(s.get("window_h", 0)) for s in cfg.get("ablation.gap_injection.sequence_models"))
        - 1,
    ) + int(cfg.get("ablation.gap_injection.horizon_h"))
    grids = load_radius_grids(results_dir, args.slug, default_radius)
    if len(grids) < 2:
        log.error(
            "need at least 2 radius grids for %s, found %d. Generate them with "
            "15_make_lookback_configs.py and run 16 against each.",
            args.slug,
            len(grids),
        )
        return 1
    radii = sorted(grids, reverse=True)
    log.info("radii on disk: %s", radii)

    problems = check_draws_match(grids, log)
    if problems:
        for p in problems:
            log.error("%s", p)
        log.error(
            "the radii were not injected identically, so the within-draw pairing "
            "this test depends on does not hold; refusing to report a mediation"
        )
        return 1

    # ---- control: persistence must be invariant everywhere ------------------
    pers = []
    for payload in grids.values():
        frame = tidy(payload)
        pers.extend(frame[frame["model"] == "persistence"]["rmse"].tolist())
    pers_spread = float(max(pers) - min(pers)) if pers else float("nan")
    pers_ok = bool(pers and pers_spread < 1e-9)
    if not pers_ok:
        log.error(
            "persistence RMSE varies by %.3e across radii; the test period was "
            "supposed to be protected, so this is a design violation",
            pers_spread,
        )

    # ---- the mediator must be shown to move ---------------------------------
    mediator_rows = []
    for r in radii:
        frame = tidy(grids[r])
        rows = frame.drop_duplicates(subset=["arm", "target_coverage", "injection_seed"])
        wide = rows.pivot_table(
            index=["target_coverage", "injection_seed"],
            columns="arm",
            values="train_rows",
            aggfunc="first",
        )
        if not {"fragmented", "contiguous"} <= set(wide.columns):
            continue
        deficit = (wide["fragmented"] - wide["contiguous"]).dropna()
        mediator_rows.append(
            {
                "radius_h": r,
                "n_draws": int(deficit.size),
                "mean_row_deficit": float(deficit.mean()),
                "median_row_deficit": float(deficit.median()),
            }
        )
    mediator = pd.DataFrame(mediator_rows).sort_values("radius_h", ascending=False)
    mediator_moves = (
        bool(mediator["mean_row_deficit"].is_monotonic_increasing) if len(mediator) > 1 else False
    )
    log.info(
        "mediator (fragmented minus contiguous usable rows) by radius:%s%s",
        NL,
        mediator.to_string(index=False),
    )

    # ---- the arm gap at each radius, paired within draw ---------------------
    paired = []
    for r in radii:
        g = paired_arm_gaps(tidy(grids[r]))
        if g.empty:
            continue
        g["radius_h"] = r
        paired.append(g)
    if not paired:
        log.error("no matched pairs at any radius")
        return 1
    gaps = pd.concat(paired, ignore_index=True)

    wide = gaps.pivot_table(
        index=["family", "target_coverage", "injection_seed"],
        columns="radius_h",
        values="arm_gap",
        aggfunc="first",
    ).dropna()
    deepest, shallowest = max(radii), min(radii)

    test_rows = []
    for family in FAMILY_ORDER:
        if family not in wide.index.get_level_values("family"):
            continue
        sub = wide.xs(family, level="family")
        delta = (sub[shallowest] - sub[deepest]).to_numpy(dtype=float)
        n = delta.size
        # One-sided and pre-declared: if the radius mediates the penalty, then
        # shortening it makes the (negative) gap LESS negative, so the difference
        # is positive. Testing two-sided here would spend power on a direction the
        # mechanism does not predict.
        p = float("nan")
        if n >= 5 and np.any(delta != 0.0):
            p = float(stats.wilcoxon(delta, alternative="greater").pvalue)
        rng = np.random.default_rng(0)
        boot = np.array([rng.choice(delta, size=n, replace=True).mean() for _ in range(5000)])

        # Slope per halving of the radius, one slope per draw.
        slopes = []
        logr = np.log2(np.array(sorted(sub.columns), dtype=float))
        for _, row in sub.iterrows():
            y = row[sorted(sub.columns)].to_numpy(dtype=float)
            if np.ptp(logr) == 0:
                continue
            slopes.append(float(np.polyfit(logr, y, 1)[0]))
        slope_arr = np.asarray(slopes, dtype=float)

        mean_deep = float(sub[deepest].mean())
        mean_shallow = float(sub[shallowest].mean())
        test_rows.append(
            {
                "family": family,
                "n_draws": n,
                f"gap_R{deepest}": mean_deep,
                f"gap_R{shallowest}": mean_shallow,
                "delta": float(delta.mean()),
                "ci_low": float(np.percentile(boot, 2.5)),
                "ci_high": float(np.percentile(boot, 97.5)),
                "slope_per_halving": float(slope_arr.mean()) if slope_arr.size else float("nan"),
                "proportion_mediated": (
                    float(1.0 - mean_shallow / mean_deep) if mean_deep != 0 else float("nan")
                ),
                "p_wilcoxon_onesided": p,
            }
        )

    tests = pd.DataFrame(test_rows)
    if not tests.empty:
        adjusted = holm_bonferroni(tests["p_wilcoxon_onesided"].tolist(), alpha=args.alpha)
        tests["p_holm"] = [r["p_adjusted"] for r in adjusted]
        tests["significant_holm"] = [r["reject"] for r in adjusted]

    # ---- tables and figure --------------------------------------------------
    write_table(
        cfg,
        mediator,
        "mediation_mediator",
        caption=(
            "The putative mediator, by sterilisation radius: how many more usable "
            "training rows the contiguous arm retains than the fragmented arm at the "
            "same removed-hour count. If this deficit does not shrink as the radius "
            "shortens there is no mediator and the test below has nothing to explain."
        ),
        float_format="%.1f",
    )
    if not tests.empty:
        disp = tests.assign(
            **{
                "Family": tests["family"],
                "Draws": tests["n_draws"],
                f"Gap at R={deepest}": tests[f"gap_R{deepest}"].round(4),
                f"Gap at R={shallowest}": tests[f"gap_R{shallowest}"].round(4),
                "Delta": tests["delta"].round(4),
                "95% CI": [
                    f"[{lo:+.4f}, {hi:+.4f}]"
                    for lo, hi in zip(tests["ci_low"], tests["ci_high"], strict=False)
                ],
                "Mediated": tests["proportion_mediated"].map(lambda v: f"{100 * v:.0f}%"),
                "p (one-sided)": tests["p_wilcoxon_onesided"].map(_fmt_p),
                "p (Holm)": tests["p_holm"].map(_fmt_p),
            }
        )[
            [
                "Family",
                "Draws",
                f"Gap at R={deepest}",
                f"Gap at R={shallowest}",
                "Delta",
                "95% CI",
                "Mediated",
                "p (one-sided)",
                "p (Holm)",
            ]
        ]
        write_table(
            cfg,
            disp,
            "mediation_test",
            caption=(
                "Does shortening the sterilisation radius shrink the fragmentation "
                "penalty? Each draw contributes its arm gap at every radius, and the "
                "same injected series appears at each, so the comparison is within-draw. "
                "One-sided and pre-declared: the mechanism predicts the gap becomes less "
                "negative. The mediator is manipulated by configuration rather than "
                "inferred, so this is a stronger design than a regression-based mediation "
                "and needs no sequential-ignorability assumption."
            ),
        )

    palette = setup_style(cfg)
    families = [f for f in FAMILY_ORDER if f in set(gaps["family"])]
    fig, axes = plt.subplots(1, max(len(families), 1), figsize=(COL_DOUBLE, 2.5), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, family in zip(axes, families, strict=False):
        sub = wide.xs(family, level="family")
        cols = sorted(sub.columns)
        for _, row in sub.iterrows():
            ax.plot(cols, row[cols].to_numpy(), color="#BBBBBB", lw=0.4, alpha=0.5, zorder=1)
        ax.plot(
            cols,
            [float(sub[c].median()) for c in cols],
            color=palette[0],
            lw=1.6,
            marker="o",
            ms=3.5,
            zorder=3,
        )
        ax.axhline(0.0, color="#666666", lw=0.8, ls=":", zorder=2)
        ax.set_xscale("log", base=2)
        ax.set_xticks(cols)
        ax.set_xticklabels([str(c) for c in cols], fontsize=6.4)
        ax.set_xlabel("Sterilisation radius (h)")
        panel_label(ax, f"({chr(97 + list(families).index(family))}) {family}")
    axes[0].set_ylabel("Fragmented $-$ contiguous skill")
    fig.tight_layout(w_pad=0.8)
    written = save_figure(cfg, fig, "fig17_mediation")
    log.info("wrote %s", ", ".join(p.name for p in written))
    plt.close(fig)

    out = results_dir / f"mediation_{args.slug}.json"
    out.write_text(
        json.dumps(
            {
                "donor": args.slug,
                "radii": radii,
                "draws_identical_across_radii": True,
                "persistence_rmse_invariant": pers_ok,
                "persistence_rmse_spread": pers_spread,
                "mediator_deficit_shrinks_with_radius": mediator_moves,
                "mediator": mediator.to_dict(orient="records"),
                "tests": tests.to_dict(orient="records"),
                "design_note": (
                    "The mediator is manipulated directly by configuration, not inferred "
                    "from a regression, so this is not a Baron-Kenny mediation and needs "
                    "no sequential-ignorability assumption. The same injected series "
                    "appears at every radius, verified from the recorded injection "
                    "statistics, so each draw is a repeated measure."
                ),
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    print(NL + "=" * 78)
    print(f"MEDIATION — {args.slug}, radii {radii}")
    print("=" * 78)
    print(NL + "mediator: usable-row deficit (fragmented minus contiguous)")
    print(mediator.to_string(index=False, float_format=lambda v: f"{v:.1f}"))
    print(f"{NL}  deficit shrinks as the radius shortens: {'YES' if mediator_moves else 'NO'}")
    print(f"  persistence invariant across all cells   : {'YES' if pers_ok else 'NO'}")
    if not mediator_moves:
        print(
            f"{NL}  The mediator does not move, so there is nothing for the radius to "
            f"mediate.{NL}  The test below should not be reported."
        )
    if not tests.empty:
        print(NL + "arm gap by radius, paired within draw")
        print(disp.to_string(index=False))
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
