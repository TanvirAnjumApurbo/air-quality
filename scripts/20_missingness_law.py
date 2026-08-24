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
from matplotlib.ticker import NullFormatter
from src.eval.availability import amplification, availability_record
from src.eval.missingness_law import (
    LawFit,
    decision_curve,
    exact_usable_rows,
    ffill_survival_fraction,
    ffill_survival_interval,
    fit_survival_law,
    law_diagnostics,
    predict_usable,
    run_length_distribution,
    sterilisation_radius,
)
from src.features.gap_injection import gap_lengths
from src.results import city_suffix
from src.utils import Config, load_config, setup_logging
from src.viz.figures import COL_DOUBLE, panel_label, save_figure, setup_style
from src.viz.tables import write_table

#: Lookback caps whose availability is reported. 168 is the status quo.
LOOKBACK_LEVELS = (168, 48, 24, 12)

#: Radii the decision curves are drawn over. The lower bound is the horizon plus
#: one hour, below which no row can be supervised at all; the upper is a
#: fortnight, past every cap this study reaches.
DECISION_RADII_H = np.arange(25, 337)


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

    Each grid states the radius it was built at, and that is carried onto every
    cell rather than assumed: the mediation grids are the same station at R = 48
    and R = 72, and scoring them against a law evaluated at R = 192 silently
    turned a validated law into a broken one.

    Returns:
        One row per cell with ``donor``, ``radius_h``, ``arm``, ``observed``,
        ``n_gaps`` and ``usable`` columns.

    Raises:
        ValueError: If no grid carries any cell.
    """
    rows: list[dict[str, object]] = []
    for path in sorted(paths):
        payload = json.loads(path.read_text(encoding="utf-8"))
        slug = donor_slug(path)
        grid_radius = int(payload["sterilisation_radius_h"])
        for key, cell in payload.get("cells", {}).items():
            inj = cell.get("injection", {})
            usable = cell.get("usable_rows", {})
            if "hours_available" not in inj or "train" not in usable:
                continue
            rows.append(
                {
                    "donor": slug,
                    "donor_label": payload.get("donor", slug),
                    "radius_h": grid_radius,
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


def record_audit(
    cfg_path: str, label: str, horizon: int
) -> tuple[dict[str, object], list[dict], np.ndarray]:
    """Amplification, availability and run lengths for one built record.

    Args:
        cfg_path: Path to the record's config.
        label: Short record name for the tables.
        horizon: Forecast horizon in hours.

    Returns:
        ``(amplification_row, availability_rows, run_lengths)``.
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
    return amp_row, avail_rows, run_lens


def make_figure(
    cfg: Config,
    cells: pd.DataFrame,
    fit: LawFit,
    fit_donor: str,
    log: logging.Logger,
) -> None:
    """Draw the law and its out-of-sample validation.

    The cells are split three ways rather than two, because "held out" pools two
    different extrapolations: another station at the fitted radius asks whether
    the constants travel between records, and the fitting station at a shorter
    radius asks whether the radius is a factor of the form or a scale the fit
    absorbed. Only the second can falsify the functional form.

    Args:
        cfg: Loaded configuration.
        cells: Flattened cell frame.
        fit: The fitted law.
        fit_donor: Slug the law was fitted on.
        log: Logger.
    """
    palette = setup_style(cfg)
    fig, axes = plt.subplots(1, 2, figsize=(COL_DOUBLE, 2.9))

    # The law is one variable deep: log(usable / observed) against R k / O. Drawn
    # that way every cell at every radius falls on one line, which is the claim,
    # where a plot against the gap count alone would show three parallel clouds
    # and make the radius look like a nuisance rather than the factor it is.
    groups = [
        ("fitted", cells[cells["donor"] == fit_donor]),
        (
            "other stations",
            cells[(cells["donor"] != fit_donor) & (cells["radius_h"] == fit.radius_h)],
        ),
        ("shorter radius", cells[cells["radius_h"] != fit.radius_h]),
    ]

    ax = axes[0]
    for i, (name, sub) in enumerate(groups):
        if sub.empty:
            continue
        ax.scatter(
            sub["radius_h"] * sub["n_gaps"] / sub["observed"],
            sub["usable"] / sub["observed"],
            s=11,
            alpha=0.65,
            color=palette[i % len(palette)],
            label=f"{name} ({len(sub)})",
            edgecolors="none",
        )
    x_law = np.linspace(
        float((cells["radius_h"] * cells["n_gaps"] / cells["observed"]).min()),
        float((cells["radius_h"] * cells["n_gaps"] / cells["observed"]).max()),
        200,
    )
    ax.plot(
        x_law,
        np.exp(fit.beta0 - fit.alpha * x_law),
        color="#1A1A1A",
        lw=1.3,
        ls="--",
        label="law",
    )
    ax.set_xlabel(r"$Rk/O$  (radius $\times$ gaps per observed hour)")
    ax.set_ylabel("Usable share of observed hours")
    ax.set_yscale("log")
    ax.legend(frameon=True, framealpha=0.92, facecolor="white", edgecolor="#CCCCCC", fontsize=6.0)
    panel_label(ax, "(a)")

    ax = axes[1]
    for i, (name, sub) in enumerate(groups):
        if sub.empty:
            continue
        ax.scatter(
            predict_usable(fit, sub["observed"], sub["n_gaps"], sub["radius_h"]),
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
    # Over more than a decade matplotlib labels the log minor ticks too, and at
    # this column width they overprint each other into an unreadable smear.
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("Predicted usable rows")
    ax.set_ylabel("Observed usable rows")
    ax.legend(frameon=True, framealpha=0.92, facecolor="white", edgecolor="#CCCCCC", fontsize=6.0)
    panel_label(ax, "(b)")

    fig.tight_layout(w_pad=1.1)
    written = save_figure(cfg, fig, "fig14_missingness_amplification")
    log.info("wrote %s", ", ".join(p.name for p in written))
    plt.close(fig)


def frontier_outcomes(results_dir: Path, log: logging.Logger) -> pd.DataFrame:
    """What shortening the reach actually bought, on the records that trained it.

    Reads both cities' frontier payloads rather than the running city's alone,
    because the decision this figure supports is made across records. Returns an
    empty frame when the frontier has not run, so the panel is omitted rather
    than invented.

    The radius and the training-row count come from the frontier's own columns.
    Rebuilding the radius from the cap would be wrong for tier 3, where
    ``max(lookback, window - 1)`` binds and two arms sharing a cap can sit at
    different radii; reading it out of the arm label would turn a uniqueness key
    into a data source, which is how the 102-to-18 channel change once became
    invisible.

    Args:
        results_dir: Directory holding the frontier payloads.
        log: Logger.

    Returns:
        One row per (record, arm, model) with the radius, the training rows
        recovered, and the availability and all-hours skill change against the
        status quo.

    Raises:
        ValueError: If an arm's training rows and its availability disagree
            about whether the arm moved, which would mean the arms are
            mislabelled.
    """
    known = (
        ("availability_frontier.json", "lookback_frontier.json", "Dhaka"),
        ("availability_frontier_beijing.json", "lookback_frontier_beijing.json", "Beijing Wanliu"),
    )
    rows = []
    for skill_name, built_name, label in known:
        skill_path, built_path = results_dir / skill_name, results_dir / built_name
        if not (skill_path.exists() and built_path.exists()):
            log.info("%s absent; the decision figure will omit its outcome panel", skill_name)
            continue
        scored = pd.DataFrame(
            json.loads(skill_path.read_text(encoding="utf-8")).get("all_hours", [])
        )
        built = pd.DataFrame(json.loads(built_path.read_text(encoding="utf-8")).get("rows", []))
        if scored.empty or built.empty:
            continue
        built = built.set_index(["model", "arm"])
        single = scored[scored["policy"] == "single"]
        status_quo = single[single["arm"] == "C"].set_index("model")
        for r in single[single["arm"] != "C"].itertuples():
            if r.model not in status_quo.index or (r.model, r.arm) not in built.index:
                continue
            base, here, base_built = (
                status_quo.loc[r.model],
                built.loc[(r.model, r.arm)],
                built.loc[(r.model, "C")],
            )
            gained = float(r.availability) - float(base["availability"])
            # Arm B caps the features but holds the floor, so it trains on
            # exactly the rows the status quo trains on and serves exactly the
            # hours it serves. Reading the control off that equality keeps its
            # definition in the data rather than in its label -- and the two
            # readings must agree, or an arm is not what it says it is.
            same_rows = int(here["train_rows"]) == int(base_built["train_rows"])
            if same_rows != (abs(gained) < 1e-12):
                raise ValueError(
                    f"{label} arm {r.arm} on {r.model}: training rows say "
                    f"{'held' if same_rows else 'moved'} but availability says "
                    f"{'held' if abs(gained) < 1e-12 else 'moved'}"
                )
            rows.append(
                {
                    "Record": label,
                    "arm": r.arm,
                    "model": r.model,
                    "radius_h": int(here["radius_h"]),
                    "base_radius_h": int(base_built["radius_h"]),
                    "rows_gained_pct": 100.0
                    * (float(here["train_rows"]) / float(base_built["train_rows"]) - 1.0),
                    "is_control": same_rows,
                    "d_availability": gained,
                    "d_skill_all_hours": float(r.skill_all_hours) - float(base["skill_all_hours"]),
                }
            )
    return pd.DataFrame(rows)


def row_yield(curves: pd.DataFrame, record: str, radius_h: int, base_radius_h: int) -> float:
    """Usable rows at one radius as a multiple of the rows at another.

    Args:
        curves: Decision curves, one block per record.
        record: Record label.
        radius_h: Radius to price.
        base_radius_h: Radius to price it against, being the status quo's.

    Returns:
        The ratio, or NaN if either radius is off the grid.
    """
    sub = curves[curves["Record"] == record]
    here = sub.loc[sub["radius_h"] == int(radius_h), "usable"]
    base = sub.loc[sub["radius_h"] == int(base_radius_h), "usable"]
    if here.empty or base.empty or not float(base.iloc[0]):
        return float("nan")
    return float(here.iloc[0]) / float(base.iloc[0])


def decision_outcomes(curves: pd.DataFrame, outcomes: pd.DataFrame) -> pd.DataFrame:
    """The price the identity quotes, beside the rows and the skill that followed.

    Four quantities that are usually reported apart. The elasticity is free and
    known before anything is fitted; the rows and the hours served are what the
    build actually produced; the skill change is what the training bought.
    Putting them in one row is what lets the decision rule be stated with its
    limits attached rather than as a promise -- and it is how one sees that the
    two free quantities do not agree about which record had more to gain.

    Predicted and measured rows are not computed on the same footing -- the
    identity runs over the whole record, the build over the training split alone
    -- so they are not expected to agree exactly. This is a calibration report,
    not a guard. Skill is summarised by its median across models as well as its
    best, because a maximum over a pool is a selection and reads better than the
    pool deserves.

    Args:
        curves: Decision curves, one block per record.
        outcomes: Frontier outcomes.

    Returns:
        One row per (record, radius) that moved.

    Raises:
        ValueError: If one radius on one record shows more than one row count,
            which would mean the row count is not a function of the radius.
    """
    moved = outcomes[~outcomes["is_control"]]
    rows = []
    for (record, radius, base), sub in moved.groupby(["Record", "radius_h", "base_radius_h"]):
        measured = sorted(set(sub["rows_gained_pct"].round(9)))
        served = sorted(set(sub["d_availability"].round(12)))
        if len(measured) != 1:
            raise ValueError(f"{record} at R={radius}: {len(measured)} row counts, expected one")
        if len(served) != 1:
            raise ValueError(f"{record} at R={radius}: {len(served)} availabilities, expected one")
        predicted = 100.0 * (row_yield(curves, str(record), int(radius), int(base)) - 1.0)
        at = curves[(curves["Record"] == record) & (curves["radius_h"] == int(base))]["elasticity"]
        rows.append(
            {
                "Record": record,
                "Radius (h)": int(radius),
                "Elasticity at status quo": float(at.iloc[0]) if not at.empty else float("nan"),
                "Predicted rows (%)": predicted,
                "Measured rows (%)": measured[0],
                "Error (pp)": predicted - measured[0],
                "Availability gained (pp)": 100.0 * served[0],
                "Skill gain (median)": float(sub["d_skill_all_hours"].median()),
                "Skill gain (best)": float(sub["d_skill_all_hours"].max()),
                "Models": len(sub),
            }
        )
    return pd.DataFrame(rows)


def make_decision_figure(
    cfg: Config,
    curves: pd.DataFrame,
    outcomes: pd.DataFrame,
    marks: tuple[int, ...],
    log: logging.Logger,
) -> None:
    """Draw the decision rule: what a reach costs, what it prices, what it buys.

    Panels (a) and (b) are identities computed from the run-length distribution
    alone -- free, exact, and available before any model is fitted. Panel (c) is
    the measured outcome on the two records whose arms were trained, and it is
    what stops the first two panels from being read as a promise: the record that
    recovers the most training rows is not the one that gained the most skill.
    The open markers are arm B, which caps the features but holds the floor, so
    it recovers no rows by construction and separates a richness effect from a
    supervision effect at a glance.

    Args:
        cfg: Loaded configuration.
        curves: Decision curves, one block per record.
        outcomes: Frontier outcomes, possibly empty.
        marks: Radii to mark, being the sweep's configured caps plus the horizon.
        log: Logger.
    """
    palette = setup_style(cfg)
    n_panels = 3 if not outcomes.empty else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(COL_DOUBLE, 2.65))
    records = list(dict.fromkeys(curves["Record"]))
    colour = {name: palette[i % len(palette)] for i, name in enumerate(records)}

    ax = axes[0]
    for name in records:
        sub = curves[curves["Record"] == name]
        ax.plot(sub["radius_h"], 100.0 * sub["yield_frac"], lw=1.3, color=colour[name], label=name)
        at_marks = sub[sub["radius_h"].isin(marks)]
        ax.plot(
            at_marks["radius_h"],
            100.0 * at_marks["yield_frac"],
            ls="none",
            marker="o",
            ms=3.0,
            color=colour[name],
        )
    ax.set_xlabel("Sterilisation radius $R$ (h)")
    ax.set_ylabel("Usable rows (% of observed)")
    ax.set_ylim(0, 100)
    ax.legend(frameon=True, framealpha=0.92, facecolor="white", edgecolor="#CCCCCC", fontsize=6.0)
    panel_label(ax, "(a)")

    ax = axes[1]
    for name in records:
        sub = curves[curves["Record"] == name]
        ax.plot(sub["radius_h"], sub["elasticity"], lw=1.3, color=colour[name])
        at_marks = sub[sub["radius_h"].isin(marks)]
        ax.plot(
            at_marks["radius_h"],
            at_marks["elasticity"],
            ls="none",
            marker="o",
            ms=3.0,
            color=colour[name],
        )
    ax.set_xlabel("Sterilisation radius $R$ (h)")
    ax.set_ylabel(r"Elasticity $-\,\mathrm{d}\ln U / \mathrm{d}\ln R$")
    panel_label(ax, "(b)")

    if not outcomes.empty:
        ax = axes[2]
        # Iterate the record order the first two panels use, so one legend
        # reading carries across all three.
        for name in records:
            sub = outcomes[outcomes["Record"] == name]
            if sub.empty:
                continue
            shade = colour.get(str(name), "#666666")
            moved, held = sub[~sub["is_control"]], sub[sub["is_control"]]
            ax.scatter(
                moved["rows_gained_pct"],
                moved["d_skill_all_hours"],
                s=13,
                alpha=0.8,
                color=shade,
                edgecolors="none",
                label=str(name),
            )
            ax.scatter(
                held["rows_gained_pct"],
                held["d_skill_all_hours"],
                s=15,
                facecolors="none",
                edgecolors=shade,
                linewidths=0.7,
            )
        ax.axhline(0.0, color="#666666", lw=0.8, zorder=0)
        ax.set_xlabel("Training rows recovered (%)")
        ax.set_ylabel("All-hours skill gained")
        ax.legend(
            frameon=True,
            framealpha=0.92,
            facecolor="white",
            edgecolor="#CCCCCC",
            fontsize=6.0,
            title="filled: floor lowered\nopen: features capped only",
            title_fontsize=5.6,
        )
        panel_label(ax, "(c)")

    fig.tight_layout(w_pad=1.0)
    written = save_figure(cfg, fig, "fig16_decision_rule")
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
    fit_radii = sorted(set(fit_rows["radius_h"]))
    if len(fit_radii) != 1:
        log.error("fit donor %r spans radii %s; a fit is made at one", args.fit_donor, fit_radii)
        return 1
    held = cells[cells["donor"] != args.fit_donor]

    # The fit is made at the radius its own grid was built with, not at the one
    # the record audit uses. They coincide today; a fit donor chosen from the
    # mediation grids would not, and the fit must follow the data.
    fit_radius = int(fit_radii[0])
    if fit_radius != radius:
        log.info("fit donor grid is at R=%d, record audit at R=%d", fit_radius, radius)
    fit = fit_survival_law(
        fit_rows["observed"].to_numpy(),
        fit_rows["n_gaps"].to_numpy(),
        fit_rows["usable"].to_numpy(),
        fit_radius,
        source=f"{args.fit_donor} ({len(fit_rows)} cells at R={fit_radius})",
    )
    in_sample = law_diagnostics(
        fit_rows["usable"],
        predict_usable(fit, fit_rows["observed"], fit_rows["n_gaps"], fit_rows["radius_h"]),
    )
    out_sample = (
        law_diagnostics(
            held["usable"],
            predict_usable(fit, held["observed"], held["n_gaps"], held["radius_h"]),
        )
        if not held.empty
        else {}
    )
    # Two different extrapolations are pooled in that number and they are not
    # the same claim. Another station at the fitted radius tests whether the
    # constants travel; the fitting station at a shorter radius tests whether
    # the radius is a factor of the form or a scale the fit absorbed.
    out_sample_at_fit_radius = (
        law_diagnostics(held["usable"], predict_usable(fit, held["observed"], held["n_gaps"]))
        if not held.empty
        else {}
    )
    held_groups = []
    for (donor, cell_radius), sub in held.groupby(["donor", "radius_h"]):
        held_groups.append(
            {
                "group": ("other station" if cell_radius == fit_radius else "shorter radius"),
                "donor": donor,
                "radius_h": int(cell_radius),
                **law_diagnostics(
                    sub["usable"],
                    predict_usable(fit, sub["observed"], sub["n_gaps"], sub["radius_h"]),
                ),
            }
        )

    # alpha should be the share of gaps that outlive the forward-fill. If it is
    # not, the constant is absorbing something other than the stated mechanism.
    target_col = str(cfg.get("features.target"))
    pm = pd.read_parquet(cfg.path_for("data_interim") / str(cfg.get("data.files.target")))
    max_ffill = int(cfg.get("impute.max_ffill_hours", 3))
    raw_gaps = gap_lengths(pm[target_col])
    predicted_alpha = ffill_survival_fraction(raw_gaps, max_ffill)
    # A relative percentage is not comparable between a record with 2,074
    # gaps and one with 180. The interval is.
    alpha_lo, alpha_hi, n_raw_gaps = ffill_survival_interval(raw_gaps, max_ffill)

    # ---- per-record audit --------------------------------------------------
    records = [("Dhaka", "config.yaml"), ("Beijing Wanliu", "config_beijing.yaml")]
    for donor_cfg in sorted(Path("config/donors").glob("*.yaml")):
        records.append((f"Beijing {donor_cfg.stem.title()}", str(donor_cfg)))

    amp_rows, avail_rows = [], []
    decision_frames: list[pd.DataFrame] = []
    for label, path in records:
        try:
            amp_row, avail, run_lens = record_audit(path, label, horizon)
        except FileNotFoundError:
            log.warning("%s: features not built, skipping", label)
            continue
        amp_rows.append(amp_row)
        avail_rows.extend(avail)
        curve = decision_curve(run_lens, DECISION_RADII_H)
        curve.insert(0, "Record", label)
        decision_frames.append(curve)
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
                "Note": (
                    f"share of {n_raw_gaps} gaps longer than max_ffill_hours="
                    f"{max_ffill}; 95% CI [{alpha_lo:.4f}, {alpha_hi:.4f}]"
                ),
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
                    "Note": (
                        f"{out_sample['n']} cells: other stations at R={fit_radius}, "
                        f"and the fitted station at shorter radii"
                    ),
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

    # ---- the decision rule --------------------------------------------------
    # The survival law above needs only (observed, gaps) but is fitted on
    # injected grids and does not transfer to a raw record. Anyone holding the
    # record itself has the run lengths, and those give the answer as an identity
    # with no error term at all -- so the rule below is stated on the identity
    # and the law is left doing the job it was validated for.
    decision = pd.concat(decision_frames, ignore_index=True) if decision_frames else pd.DataFrame()
    decision_table = pd.DataFrame()
    recovery = pd.DataFrame()
    marks = tuple(sorted(sterilisation_radius(level, horizon) for level in LOOKBACK_LEVELS))
    if not decision.empty:
        decision_table = (
            decision[decision["radius_h"].isin(marks)]
            .rename(
                columns={
                    "radius_h": "Radius (h)",
                    "observed": "Observed hours",
                    "usable": "Usable rows",
                    "yield_frac": "Yield",
                    "surviving_runs": "Runs surviving",
                    "elasticity": "Elasticity",
                }
            )
            .drop(columns=["marginal_rows_per_h"])
            .reset_index(drop=True)
        )
        write_table(
            cfg,
            decision_table,
            "decision_rule",
            caption=(
                "What a backward reach costs each record, as an identity rather than "
                "a fit. Usable rows are sum(max(0, run length minus radius)), whose "
                "derivative in the radius is minus the number of runs still longer "
                "than it -- so 'Runs surviving' is exactly the rows one further hour "
                "of reach would cost. The elasticity is that marginal cost made "
                "dimensionless, and is the only column comparable across records."
            ),
            float_format="%.4f",
        )
        outcomes = frontier_outcomes(results_dir, log)
        if not outcomes.empty:
            recovery = decision_outcomes(decision, outcomes)
        make_decision_figure(cfg, decision, outcomes, marks, log)

    # ---- payload -----------------------------------------------------------
    payload = {
        "fit": fit.to_dict(),
        "alpha_predicted_from_ffill": predicted_alpha,
        "alpha_predicted_ci": [alpha_lo, alpha_hi],
        "n_raw_gaps": n_raw_gaps,
        "alpha_within_ci": bool(alpha_lo <= fit.alpha <= alpha_hi),
        "max_ffill_hours": max_ffill,
        "radius_h": radius,
        "horizon_h": horizon,
        "in_sample": in_sample,
        "held_out": out_sample,
        "held_out_by_group": held_groups,
        "held_out_at_fit_radius": out_sample_at_fit_radius,
        "held_out_note": (
            "Every held-out cell is scored at the radius its own grid was built with. "
            "The radius enters the law as a factor rather than as a fitted scale, so a "
            "law fitted at one radius has a prediction at any other. "
            "held_out_at_fit_radius is the counterfactual in which that is not believed "
            "and every cell is scored at the fitted radius instead; the difference "
            "between the two is what the radius term is carrying."
        ),
        "fit_radius_h": fit_radius,
        "n_cells": len(cells),
        "donors": sorted(set(cells["donor"])),
        "fit_donor": args.fit_donor,
        "amplification_by_record": amp_table.to_dict(orient="records"),
        "availability": avail_table.to_dict(orient="records"),
        "decision_rule": decision_table.to_dict(orient="records"),
        "decision_outcomes": recovery.to_dict(orient="records"),
        "decision_outcomes_note": (
            "The elasticity is free and known before any fitting; the rows and the "
            "skill are measured. Predicted and measured rows are two footings for one "
            "quantity -- the identity runs over the whole record, the build over the "
            "training split -- so they are not expected to agree exactly. Reported "
            "together so the free calculation can be seen sizing the opportunity, and "
            "so the limit of the rule is visible: on these two records the higher "
            "elasticity is not the larger skill gain."
        ),
        "decision_rule_note": (
            "Identity, not a fit: usable = sum(max(0, run_len - radius)), so one more "
            "hour of reach costs exactly the number of runs still longer than it. The "
            "survival law is the approximation for a caller who has only the observed "
            "hour count and the gap count; applied to these raw records it "
            "under-predicts usable rows by 12-47 percent, because their run lengths "
            "are far from the exponential the survival form assumes."
        ),
        "mechanism_note": (
            "A row needs floor hours of unbroken history and a target horizon hours "
            "further on inside the same run, so each run of length l yields "
            "max(0, l - floor - horizon) rows. The cost of missingness is therefore "
            "driven by the number of gaps, not their total length."
        ),
    }
    # Both configs resolve paths.results to the same directory, so a bare name
    # here is written twice and the city that runs second destroys the first
    # one's. Five artefacts in this repo have already been lost that way.
    out = results_dir / f"missingness_law{city_suffix(cfg)}.json"
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
        for g in sorted(held_groups, key=lambda r: (r["group"], r["radius_h"])):
            print(
                f"    {g['group']:<15} {g['donor']:<12} R={g['radius_h']:<4} "
                f"R2 ={g['r2']:7.4f}  medAPE {g['median_ape_pct']:5.1f}%  ({g['n']} cells)"
            )
    print(
        f"\n  alpha fitted    {fit.alpha:.4f}"
        f"\n  alpha predicted {predicted_alpha:.4f}  [{alpha_lo:.4f}, {alpha_hi:.4f}]"
        f"  the share of {n_raw_gaps} gaps outliving {max_ffill} h"
        f"\n  fitted alpha is {'inside' if alpha_lo <= fit.alpha <= alpha_hi else 'outside'} that interval"
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
    if not decision_table.empty:
        print("\nwhat a backward reach costs, as an identity")
        print(
            decision_table[
                ["Record", "Radius (h)", "Usable rows", "Yield", "Runs surviving", "Elasticity"]
            ].to_string(index=False, float_format=lambda v: f"{v:.3f}")
        )
    if not recovery.empty:
        print("\nwhat the reach was priced at, and what shortening it bought")
        print(recovery.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
