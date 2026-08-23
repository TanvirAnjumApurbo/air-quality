r"""Phase 8: what a gap actually costs, and how often a model can answer.

The gap-injection experiment (``16``/``17``) establishes that fragmented removal
costs more skill than contiguous removal at an identical hour count, and
``fig11`` panel (b) shows why it might: usable training rows collapse between the
arms. Nothing in the pipeline said why *that* happens. This script is the why,
and it needs no training at all -- every number here comes from records and grids
already on disk.

The mechanism is geometric. A row must carry ``floor`` hours of unbroken history
and a target ``horizon`` hours further on inside the same run, so a gap does not
cost the hours it removes: it costs those hours **plus** the ``floor + horizon``
behind it that no longer reach back far enough. The cost is therefore driven by
the **number** of gaps, not their length, which is exactly the arm contrast in
closed form.

Three things are reported:

* **Amplification.** How many usable hours each missing hour destroys. Beijing
  loses 34 per missing hour, Dhaka 2.3 -- the near-complete record is punished
  harder per hour absent, because its few absences are scattered.
* **The law.** ``usable = observed * exp(b0 - alpha * radius * n_gaps /
  observed)``, fitted on one donor's grid and validated out-of-sample on the
  others. ``alpha`` is checked against the fraction of gaps that survive
  forward-filling, which is what it should be if the mechanism is right.
* **Availability.** The share of test hours a model can forecast at all. This is
  the quantity the study has been conditioning on without reporting, and it does
  not rank the cities the way coverage does.

Run::

    python scripts/20_missingness_law.py --config config.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.eval.availability import amplification, availability_record
from src.eval.missingness_law import (
    LawFit,
    exact_usable_rows,
    ffill_survival_fraction,
    fit_survival_law,
    law_diagnostics,
    predict_usable,
    run_length_distribution,
    sterilisation_radius,
)
from src.features.gap_injection import gap_lengths
from src.utils import Config, load_config, setup_logging
from src.viz.figures import COL_DOUBLE, panel_label, save_figure, setup_style
from src.viz.tables import write_table

#: Lookback caps whose availability is reported. 168 is the status quo.
LOOKBACK_LEVELS = (168, 48, 24, 12)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--config", default="config.yaml")
    p.add_argument(
        "--fit-donor",
        default="wanliu",
        help="donor slug whose grid the law is fitted on; the rest are held out",
    )
    p.add_argument(
        "--glob",
        default="ablation_gap_injection*.json",
        help="pattern for donor grids under paths.results",
    )
    return p.parse_args()


def donor_slug(path: Path) -> str:
    """Short donor identifier taken from a grid filename.

    Args:
        path: Path to an ``ablation_gap_injection*.json`` grid.

    Returns:
        The slug, defaulting to ``"wanliu"`` for the unsuffixed primary grid.
    """
    m = re.search(r"ablation_gap_injection_(\w+)\.json$", path.name)
    return m.group(1) if m else "wanliu"


def load_cells(paths: list[Path]) -> pd.DataFrame:
    """Flatten every donor grid to one row per cell.

    Args:
        paths: Grid files to read.

    Returns:
        One row per cell with ``donor``, ``arm``, ``observed``, ``n_gaps`` and
        ``usable`` columns.

    Raises:
        ValueError: If no grid carries any cell.
    """
    rows: list[dict[str, object]] = []
    for path in sorted(paths):
        payload = json.loads(path.read_text(encoding="utf-8"))
        slug = donor_slug(path)
        for key, cell in payload.get("cells", {}).items():
            inj = cell.get("injection", {})
            usable = cell.get("usable_rows", {})
            if "hours_available" not in inj or "train" not in usable:
                continue
            rows.append(
                {
                    "donor": slug,
                    "donor_label": payload.get("donor", slug),
                    "cell": key,
                    "arm": cell.get("arm"),
                    "target_coverage": cell.get("target_coverage"),
                    "injection_seed": cell.get("injection_seed"),
                    "observed": int(inj["hours_available"]) - int(inj.get("hours_removed", 0)),
                    "n_gaps": int(inj.get("n_gaps_after", 0)),
                    "usable": int(usable["train"]),
                }
            )
    if not rows:
        raise ValueError("no usable cells found in any grid")
    return pd.DataFrame(rows)


def record_audit(cfg_path: str, label: str, horizon: int) -> tuple[dict[str, object], list[dict]]:
    """Amplification and availability for one built record.

    Args:
        cfg_path: Path to the record's config.
        label: Short record name for the tables.
        horizon: Forecast horizon in hours.

    Returns:
        ``(amplification_row, availability_rows)``.
    """
    cfg = load_config(cfg_path)
    frame = pd.read_parquet(cfg.path_for("data_processed") / "features.parquet")
    amp = amplification(frame, 168, horizon)
    run_lens = run_length_distribution(frame)

    amp_row = {
        "Record": label,
        "Grid hours": amp["n_grid"],
        "Missing hours": amp["n_missing"],
        "Missing (%)": 100.0 * amp["n_missing"] / amp["n_grid"],
        "Runs": amp["n_runs"],
        "Hours lost to short history": amp["n_lost_to_short_history"],
        "Amplification": amp["amplification"],
        "Identity (geometric)": exact_usable_rows(run_lens, 168, horizon),
        "Actual valid rows": int(frame[f"valid_h{horizon}"].sum()),
    }

    avail_rows: list[dict] = []
    for floor in LOOKBACK_LEVELS:
        rec = availability_record(frame, "test", horizon, floor)
        avail_rows.append(
            {
                "Record": label,
                "Lookback (h)": floor,
                "Radius (h)": rec.radius_h,
                "Universe": rec.n_universe,
                "Served": rec.n_served,
                "Availability": rec.availability,
                "Availability (grid)": rec.availability_grid,
            }
        )
    return amp_row, avail_rows


def make_figure(
    cfg: Config,
    cells: pd.DataFrame,
    fit: LawFit,
    fit_donor: str,
    log: logging.Logger,
) -> None:
    """Draw the law and its out-of-sample validation.

    Args:
        cfg: Loaded configuration.
        cells: Flattened cell frame.
        fit: The fitted law.
        fit_donor: Slug the law was fitted on.
        log: Logger.
    """
    palette = setup_style(cfg)
    fig, axes = plt.subplots(1, 2, figsize=(COL_DOUBLE, 2.9))

    ax = axes[0]
    for i, (arm, sub) in enumerate(cells.groupby("arm")):
        ax.scatter(
            sub["n_gaps"],
            sub["usable"],
            s=11,
            alpha=0.65,
            color=palette[i % len(palette)],
            label=str(arm),
            edgecolors="none",
        )
    grid_gaps = np.linspace(max(cells["n_gaps"].min(), 1), cells["n_gaps"].max(), 200)
    median_obs = float(cells["observed"].median())
    ax.plot(
        grid_gaps,
        predict_usable(fit, np.full_like(grid_gaps, median_obs), grid_gaps),
        color="#1A1A1A",
        lw=1.3,
        ls="--",
        label="law, at median observed hours",
    )
    ax.set_xlabel("Distinct gaps in the degraded record")
    ax.set_ylabel("Usable training rows")
    ax.set_yscale("log")
    ax.legend(frameon=True, framealpha=0.92, facecolor="white", edgecolor="#CCCCCC")
    panel_label(ax, "(a)")

    ax = axes[1]
    held = cells[cells["donor"] != fit_donor]
    fitted = cells[cells["donor"] == fit_donor]
    for i, (name, sub) in enumerate([("fitted on", fitted), ("held out", held)]):
        ax.scatter(
            predict_usable(fit, sub["observed"], sub["n_gaps"]),
            sub["usable"],
            s=13,
            alpha=0.7,
            color=palette[i % len(palette)],
            label=f"{name} ({len(sub)} cells)",
            edgecolors="none",
        )
    lim = [float(cells["usable"].min()) * 0.8, float(cells["usable"].max()) * 1.2]
    ax.plot(lim, lim, color="#666666", lw=0.9, ls=":", zorder=0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("Predicted usable rows")
    ax.set_ylabel("Observed usable rows")
    ax.legend(frameon=True, framealpha=0.92, facecolor="white", edgecolor="#CCCCCC")
    panel_label(ax, "(b)")

    fig.tight_layout(w_pad=1.1)
    written = save_figure(cfg, fig, "fig14_missingness_amplification")
    log.info("wrote %s", ", ".join(p.name for p in written))
    plt.close(fig)


def main() -> int:
    """Fit the amplification law and audit availability."""
    args = parse_args()
    cfg = load_config(args.config)
    log = setup_logging(cfg, "20_missingness_law")

    horizon = int(cfg.get("task.headline_horizon_h"))
    radius = sterilisation_radius(168, horizon)
    results_dir = cfg.path_for("results")
    paths = sorted(results_dir.glob(args.glob))
    if not paths:
        log.error("no donor grids matching %s under %s", args.glob, results_dir)
        return 1

    cells = load_cells(paths)
    log.info("loaded %d cells from %d donor grid(s)", len(cells), cells["donor"].nunique())

    # ---- the law -----------------------------------------------------------
    fit_rows = cells[cells["donor"] == args.fit_donor]
    if fit_rows.empty:
        log.error("no cells for fit donor %r; have %s", args.fit_donor, sorted(set(cells["donor"])))
        return 1
    held = cells[cells["donor"] != args.fit_donor]

    fit = fit_survival_law(
        fit_rows["observed"].to_numpy(),
        fit_rows["n_gaps"].to_numpy(),
        fit_rows["usable"].to_numpy(),
        radius,
        source=f"{args.fit_donor} ({len(fit_rows)} cells)",
    )
    in_sample = law_diagnostics(
        fit_rows["usable"], predict_usable(fit, fit_rows["observed"], fit_rows["n_gaps"])
    )
    out_sample = (
        law_diagnostics(held["usable"], predict_usable(fit, held["observed"], held["n_gaps"]))
        if not held.empty
        else {}
    )

    # alpha should be the share of gaps that outlive the forward-fill. If it is
    # not, the constant is absorbing something other than the stated mechanism.
    target_col = str(cfg.get("features.target"))
    pm = pd.read_parquet(cfg.path_for("data_interim") / str(cfg.get("data.files.target")))
    max_ffill = int(cfg.get("impute.max_ffill_hours", 3))
    predicted_alpha = ffill_survival_fraction(gap_lengths(pm[target_col]), max_ffill)

    # ---- per-record audit --------------------------------------------------
    records = [("Dhaka", "config.yaml"), ("Beijing Wanliu", "config_beijing.yaml")]
    for donor_cfg in sorted(Path("config/donors").glob("*.yaml")):
        records.append((f"Beijing {donor_cfg.stem.title()}", str(donor_cfg)))

    amp_rows, avail_rows = [], []
    for label, path in records:
        try:
            amp_row, avail = record_audit(path, label, horizon)
        except FileNotFoundError:
            log.warning("%s: features not built, skipping", label)
            continue
        amp_rows.append(amp_row)
        avail_rows.extend(avail)
    amp_table = pd.DataFrame(amp_rows)
    avail_table = pd.DataFrame(avail_rows)

    # ---- tables ------------------------------------------------------------
    law_table = pd.DataFrame(
        [
            {
                "Quantity": "alpha (fitted)",
                "Value": fit.alpha,
                "Note": "share of counted gaps that break a run",
            },
            {
                "Quantity": "alpha (predicted)",
                "Value": predicted_alpha,
                "Note": f"share of gaps longer than max_ffill_hours={max_ffill}",
            },
            {"Quantity": "beta0", "Value": fit.beta0, "Note": "log-scale intercept"},
            {
                "Quantity": "R^2 (fitted donor)",
                "Value": in_sample["r2"],
                "Note": f"{in_sample['n']} cells",
            },
            {
                "Quantity": "median APE (fitted donor)",
                "Value": in_sample["median_ape_pct"],
                "Note": "per cent",
            },
        ]
        + (
            [
                {
                    "Quantity": "R^2 (held out)",
                    "Value": out_sample["r2"],
                    "Note": f"{out_sample['n']} cells, donors not fitted on",
                },
                {
                    "Quantity": "median APE (held out)",
                    "Value": out_sample["median_ape_pct"],
                    "Note": "per cent",
                },
            ]
            if out_sample
            else []
        )
    )
    write_table(
        cfg,
        law_table,
        "missingness_law",
        caption=(
            "Amplification law $\\mathrm{usable} = O\\exp(\\beta_0 - \\alpha R\\,k/O)$ for "
            f"observed hours $O$, distinct gaps $k$ and sterilisation radius $R={radius}$ h, "
            f"fitted on {fit.source} and validated on the remaining donor grids. The fitted "
            "$\\alpha$ is compared against the share of gaps outliving the forward-fill, "
            "which is what it should equal if the mechanism is as stated."
        ),
    )
    write_table(
        cfg,
        amp_table,
        "amplification_by_record",
        caption=(
            "How many usable hours each missing hour destroys. A gap costs the hours it "
            f"removes plus the {radius}-hour radius behind it, so amplification is governed "
            "by the number of runs rather than the number of missing hours -- which is why "
            "the near-complete record is punished hardest per hour absent. The geometric "
            "identity is an upper bound on the actual valid rows; the residual is the rule "
            "that a target may never be a forward-filled value."
        ),
        float_format="%.2f",
    )
    write_table(
        cfg,
        avail_table,
        "forecast_availability",
        caption=(
            "Share of test hours a model can forecast at all, against the fixed evaluation "
            "universe and against the raw hourly grid. Every accuracy number in this study "
            "is conditional on this quantity, which has not previously been reported."
        ),
        float_format="%.4f",
    )

    make_figure(cfg, cells, fit, args.fit_donor, log)

    # ---- payload -----------------------------------------------------------
    payload = {
        "fit": fit.to_dict(),
        "alpha_predicted_from_ffill": predicted_alpha,
        "max_ffill_hours": max_ffill,
        "radius_h": radius,
        "horizon_h": horizon,
        "in_sample": in_sample,
        "held_out": out_sample,
        "n_cells": len(cells),
        "donors": sorted(set(cells["donor"])),
        "fit_donor": args.fit_donor,
        "amplification_by_record": amp_table.to_dict(orient="records"),
        "availability": avail_table.to_dict(orient="records"),
        "mechanism_note": (
            "A row needs floor hours of unbroken history and a target horizon hours "
            "further on inside the same run, so each run of length l yields "
            "max(0, l - floor - horizon) rows. The cost of missingness is therefore "
            "driven by the number of gaps, not their total length."
        ),
    }
    out = results_dir / "missingness_law.json"
    out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    log.info("wrote %s", out)

    # ---- console -----------------------------------------------------------
    print("\n" + "=" * 78)
    print(f"MISSINGNESS AMPLIFICATION — radius {radius} h (floor 168 + horizon {horizon})")
    print("=" * 78)
    print(
        f"\nlaw: usable = O * exp({fit.beta0:+.4f} - {fit.alpha:.4f} * {radius} * k / O)"
        f"\n  fitted on {fit.source}"
        f"\n  in-sample  R2 ={in_sample['r2']:7.4f}  median APE {in_sample['median_ape_pct']:5.1f}%"
    )
    if out_sample:
        print(
            f"  HELD OUT   R2 ={out_sample['r2']:7.4f}  median APE "
            f"{out_sample['median_ape_pct']:5.1f}%   ({out_sample['n']} cells)"
        )
    print(
        f"\n  alpha fitted    {fit.alpha:.4f}"
        f"\n  alpha predicted {predicted_alpha:.4f}  (gaps longer than {max_ffill} h)"
        f"\n  agreement       {100 * abs(fit.alpha - predicted_alpha) / fit.alpha:.0f}% apart"
    )
    print("\namplification by record")
    print(
        amp_table[["Record", "Missing hours", "Missing (%)", "Runs", "Amplification"]].to_string(
            index=False, float_format=lambda v: f"{v:.2f}"
        )
    )
    print("\nforecast availability at the headline horizon")
    print(
        avail_table[["Record", "Lookback (h)", "Served", "Availability (grid)"]].to_string(
            index=False, float_format=lambda v: f"{v:.4f}"
        )
    )
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
