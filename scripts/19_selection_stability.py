r"""Phase 5c: how much does the thin validation split move the tier-3 winner?

``_best_per_horizon`` in ``11_make_report.py`` ranks tier 3 by mean validation
loss across seeds, because ranking candidates by test error and then reporting
that error is circular. That is the right rule, but it inherits whatever noise
the validation split carries -- and this record's validation split is thin: at
h=24 it holds 5,955 valid rows against 9,083 in test, because validity masking
removes a larger share of the validation span than of the test span.

So the question a reviewer will ask is not whether the rule is correct but
whether it is *stable*: with 36 candidates, 5 seeds and a small split, is the
selected architecture a real winner or the luckiest draw? This script answers it
from run records that already exist -- no model is refitted.

Four diagnostics per horizon:

* **agreement** -- would each seed on its own, and each leave-one-seed-out
  subset, choose the same candidate the full selection chose;
* **selection frequency** -- resampling the seeds, how often each candidate wins,
  which turns "the winner" into a distribution over winners;
* **rank agreement** -- Spearman correlation between the validation ranking and
  the test ranking over all candidates, i.e. is validation loss informative here
  at all;
* **selection regret** -- test RMSE of the validation-selected candidate minus
  the best test RMSE available in the pool. This is the honest cost of selecting
  on this validation split, in the units the paper reports.

**Test error is read here as a DIAGNOSTIC and never as a selection criterion.**
``_select`` takes validation loss only and cannot see a test metric; regret and
rank agreement are computed after selection is already fixed. A test-informed
choice would bias the headline downward by the spread of the pool, which is the
mistake this whole ranking discipline exists to prevent. ``tests/test_leakage.py``
asserts the selector ignores test metrics.

Run::

    python scripts/19_selection_stability.py --config config.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from src.results import load_results, main_runs, save_results
from src.utils import check_disk_space, load_config, setup_logging
from src.viz.figures import COL_DOUBLE, panel_label, save_figure, setup_style
from src.viz.tables import write_table


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--config", default="config.yaml")
    p.add_argument(
        "--n-boot",
        type=int,
        default=2000,
        help="seed resamples per horizon for the selection-frequency estimate",
    )
    return p.parse_args()


def _candidate_frame(payload: dict, horizon: int) -> pd.DataFrame:
    """Build the per-seed candidate table for one horizon.

    Args:
        payload: Contents of ``results.json``.
        horizon: Forecast horizon in hours.

    Returns:
        One row per (candidate, seed) with its validation loss and test RMSE.
    """
    rows = []
    for run in main_runs(payload):
        if run.get("tier") != "tier3" or int(run.get("horizon_h", -1)) != horizon:
            continue
        val = run.get("best_val_loss")
        rmse = (run.get("metrics") or {}).get("rmse")
        if val is None or rmse is None:
            continue
        rows.append(
            {
                "candidate": f"{run['model']}_{run['variant']}",
                "seed": int(run["seed"]),
                "val_loss": float(val),
                "test_rmse": float(rmse),
            }
        )
    return pd.DataFrame(rows)


def _select(frame: pd.DataFrame, seeds: list[int]) -> str:
    """Choose a candidate from validation loss over the given seeds.

    This function is the selection rule, and it takes validation loss only. It
    never receives a test metric, so no amount of downstream diagnostic can leak
    test error into the choice.

    Args:
        frame: Per-seed candidate table.
        seeds: Seeds to average over.

    Returns:
        The winning candidate's name.
    """
    sub = frame[frame["seed"].isin(seeds)]
    means = sub.groupby("candidate")["val_loss"].mean()
    return str(means.idxmin())


def _stability_for_horizon(frame: pd.DataFrame, horizon: int, n_boot: int, rng) -> dict:
    """Compute every stability diagnostic for one horizon."""
    seeds = sorted(frame["seed"].unique())
    selected = _select(frame, seeds)

    per_seed = [_select(frame, [s]) for s in seeds]
    loso = [_select(frame, [x for x in seeds if x != s]) for s in seeds]

    # Resample seeds with replacement and re-run the selection rule. The spread
    # of winners is the honest answer to "is this architecture the winner", far
    # more informative than the single argmin the report has to print.
    #
    # This runs on a candidate-by-seed matrix rather than through _select, which
    # filters with `isin`: `isin` would collapse a draw's repeated seeds to one
    # each and silently turn the bootstrap into a random-subset sampler, with
    # every resample weighting a distinct seed equally no matter how many times
    # it was drawn. Column indexing keeps the multiplicities.
    matrix = frame.pivot_table(index="candidate", columns="seed", values="val_loss")
    names = list(matrix.index)
    values = matrix.to_numpy()
    draws = rng.integers(0, values.shape[1], size=(n_boot, values.shape[1]))
    winners = values[:, draws].mean(axis=2).argmin(axis=0)
    counts = np.bincount(winners, minlength=len(names))
    freq = {names[i]: float(counts[i] / n_boot) for i in np.argsort(-counts) if counts[i] > 0}

    agg = frame.groupby("candidate").agg(
        val_mean=("val_loss", "mean"),
        val_sd=("val_loss", "std"),
        test_mean=("test_rmse", "mean"),
    )
    rho, rho_p = stats.spearmanr(agg["val_mean"], agg["test_mean"])

    # Regret is measured AFTER selection is fixed. The oracle row is what the
    # pool's best test RMSE happens to be; it is a yardstick, never a choice.
    oracle = str(agg["test_mean"].idxmin())
    regret = float(agg.loc[selected, "test_mean"] - agg.loc[oracle, "test_mean"])

    ordered = agg.sort_values("val_mean")
    runner_up = ordered.index[1] if len(ordered) > 1 else selected
    margin = float(agg.loc[runner_up, "val_mean"] - agg.loc[selected, "val_mean"])
    noise = float(agg["val_sd"].median())

    # Candidates whose mean validation loss sits inside the winner's own
    # between-seed spread cannot be told apart from it on this split. The count
    # is the plain-language version of the margin.
    band = float(agg.loc[selected, "val_mean"] + agg.loc[selected, "val_sd"])
    indistinguishable = int((agg["val_mean"] <= band).sum())

    return {
        "horizon_h": horizon,
        "n_candidates": len(agg),
        "n_seeds": len(seeds),
        "selected": selected,
        "single_seed_agreement": sum(c == selected for c in per_seed) / len(per_seed),
        "loso_agreement": sum(c == selected for c in loso) / len(loso),
        "bootstrap_selection_frequency": freq.get(selected, 0.0),
        "runner_up": str(runner_up),
        "runner_up_frequency": freq.get(str(runner_up), 0.0),
        "n_distinct_winners": len(freq),
        "val_margin": margin,
        "val_seed_sd_median": noise,
        "margin_over_noise": margin / noise if noise else float("nan"),
        "n_indistinguishable": indistinguishable,
        "spearman_val_test": float(rho),
        "spearman_p": float(rho_p),
        "oracle_on_test": oracle,
        "selected_test_rmse": float(agg.loc[selected, "test_mean"]),
        "oracle_test_rmse": float(agg.loc[oracle, "test_mean"]),
        "regret_rmse": regret,
        "regret_pct": 100.0 * regret / float(agg.loc[oracle, "test_mean"]),
        "selection_frequencies": freq,
    }


def main() -> int:
    """Measure tier-3 selection stability and write it into results.json."""
    args = parse_args()
    cfg = load_config(args.config)
    log = setup_logging(cfg, "19_selection_stability")
    check_disk_space(cfg)

    payload = load_results(cfg)
    horizons = [int(h) for h in cfg.get("task.horizons_h")]
    rng = np.random.default_rng(int(cfg.get("seeds.global", 42)))

    records = []
    for horizon in horizons:
        frame = _candidate_frame(payload, horizon)
        if frame.empty:
            log.warning("h=%d has no tier3 records with a validation loss; skipped", horizon)
            continue
        records.append(_stability_for_horizon(frame, horizon, args.n_boot, rng))
        r = records[-1]
        log.info(
            "h=%-3d selected %-18s  seeds agreeing %.0f%%  bootstrap %.0f%%  "
            "regret %+.2f RMSE (%.1f%%)  Spearman val-test %+.2f",
            horizon,
            r["selected"],
            100 * r["single_seed_agreement"],
            100 * r["bootstrap_selection_frequency"],
            r["regret_rmse"],
            r["regret_pct"],
            r["spearman_val_test"],
        )

    if not records:
        log.error("no tier3 runs carry best_val_loss; run the sequence sweep first")
        return 1

    payload["selection_stability"] = records
    save_results(cfg, payload)

    frame = pd.DataFrame(records)
    shown = pd.DataFrame(
        {
            "Horizon (h)": frame["horizon_h"],
            "Selected": frame["selected"],
            "Seeds agreeing": [f"{v:.0%}" for v in frame["single_seed_agreement"]],
            "LOSO agreeing": [f"{v:.0%}" for v in frame["loso_agreement"]],
            "Bootstrap win rate": [f"{v:.0%}" for v in frame["bootstrap_selection_frequency"]],
            "Distinct winners": frame["n_distinct_winners"],
            "Margin / seed SD": [f"{v:.2f}" for v in frame["margin_over_noise"]],
            "Spearman val-test": [f"{v:+.2f}" for v in frame["spearman_val_test"]],
            "Regret (RMSE)": [f"{v:+.2f}" for v in frame["regret_rmse"]],
        }
    )
    write_table(
        cfg,
        shown,
        "selection_stability",
        caption=(
            "Stability of the tier-3 selection rule, which ranks candidates by mean "
            "validation loss across seeds. 'Seeds agreeing' is the share of single "
            "seeds that would pick the reported winner alone; 'LOSO' drops one seed "
            "at a time; 'bootstrap win rate' resamples the seeds and re-runs the "
            "selection. 'Regret' is the test RMSE of the selected candidate minus "
            "the best test RMSE in the pool -- the cost of selecting on this "
            "validation split, computed after selection and never used to make it."
        ),
    )

    palette = setup_style(cfg)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(COL_DOUBLE, 2.7))
    x = np.arange(len(frame))
    width = 0.27
    for i, (col, label) in enumerate(
        (
            ("single_seed_agreement", "single seed"),
            ("loso_agreement", "leave-one-seed-out"),
            ("bootstrap_selection_frequency", "seed bootstrap"),
        )
    ):
        ax1.bar(
            x + (i - 1) * width,
            frame[col],
            width,
            label=label,
            color=palette[i],
            edgecolor="white",
            linewidth=0.3,
        )
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{h} h" for h in frame["horizon_h"]])
    ax1.set_ylim(0, 1.05)
    ax1.set_ylabel("Agreement with the\nreported winner")
    ax1.set_xlabel("Forecast horizon")
    ax1.legend(
        ncol=3,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        columnspacing=0.9,
        handlelength=1.2,
        fontsize=7.5,
    )

    ax2.bar(
        x, frame["regret_pct"], color=palette[3 % len(palette)], edgecolor="white", linewidth=0.3
    )
    ax2.axhline(0.0, color="#999999", linewidth=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{h} h" for h in frame["horizon_h"]])
    ax2.set_ylabel("Selection regret\n(% of best test RMSE)")
    ax2.set_xlabel("Forecast horizon")

    # Both tags at the same height: (a) has to clear its legend, and a (b) sitting
    # lower reads as a second row of panels rather than the same one.
    panel_label(ax1, "(a)", y=1.20)
    panel_label(ax2, "(b)", y=1.20)
    fig.tight_layout(w_pad=1.8)
    written = save_figure(cfg, fig, "fig13_selection_stability")
    log.info("wrote %s", ", ".join(p.name for p in written))
    plt.close(fig)

    print("\n" + "=" * 78)
    print("TIER-3 SELECTION STABILITY")
    print("=" * 78)
    print(shown.to_string(index=False))
    print(
        "\nRegret is a diagnostic computed after selection. Selection itself never "
        "reads a test metric."
    )
    for r in records:
        top = list(r["selection_frequencies"].items())[:3]
        print(
            f"\nh={r['horizon_h']}: "
            + ", ".join(f"{k} {v:.0%}" for k, v in top)
            + f"   ({r['n_indistinguishable']} of {r['n_candidates']} candidates lie "
            f"within the winner's own between-seed spread)"
        )
    print(f"\nwrote selection_stability into {cfg.get('paths.results_json')}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except BaseException:
        import logging

        logging.getLogger("19_selection_stability").exception("selection-stability run ABORTED")
        raise
