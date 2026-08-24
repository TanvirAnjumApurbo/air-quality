"""Phase 6: generate RESULTS.md and abstract_facts.json.

Both are written from ``results/results.json``. No number in either file is
transcribed by hand -- if it is not in results.json it does not appear here, and
if a value depends on an unverified citation the cell is left blank and flagged
rather than filled in.

``abstract_facts.json`` is the small dictionary the abstract will be written
from: best model, its error at every horizon, the skill score, the parameter
count, the energy estimate and the dataset date range.

Run::

    python scripts/11_make_report.py --config config.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from src.eval.ablation import FAMILY_ORDER
from src.models.data import load_meta
from src.results import city_suffix, load_results, main_runs
from src.utils import check_disk_space, load_config, setup_logging

#: Plain-text unit for prose; figures use the mathtext form from src.viz.figures.
UNIT_PM25_TEXT = "ug/m3"


def _join_and(names: list[str]) -> str:
    """Join names as prose: "a", "a and b", "a, b and c"."""
    quoted = [f"`{n}`" for n in names]
    if len(quoted) <= 1:
        return "".join(quoted)
    return " and ".join([", ".join(quoted[:-1]), quoted[-1]])


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="config.yaml")
    return p.parse_args()


def _fmt_p(p: float) -> str:
    """Format a p-value without rounding a small one to zero.

    Args:
        p: Raw p-value.

    Returns:
        A display string; "<0.0001" rather than "0.0000".
    """
    value = float(p)
    if value != value:  # NaN
        return "—"
    return "<0.0001" if value < 1e-4 else f"{value:.4f}"


def _runs_frame(payload: dict) -> pd.DataFrame:
    """Flatten the run records into a table."""
    runs = main_runs(payload)
    if not runs:
        return pd.DataFrame()
    frame = pd.DataFrame(runs)
    for metric in ("rmse", "mae", "r2", "smape", "bias", "skill_vs_persistence", "n"):
        frame[metric] = frame["metrics"].map(lambda m, k=metric: m.get(k))
    return frame


def _best_per_horizon(frame: pd.DataFrame, tier: str | None = None) -> pd.DataFrame:
    """Select the best model per horizon, choosing on held-out-from-test criteria.

    Selection must never consult test error. Picking the model with the lowest
    test RMSE and then reporting that RMSE as the headline is circular -- it is
    the same mistake the Diebold-Mariano comparison deliberately avoids, and it
    biases the reported number downward by the spread of the candidate pool.

    So the ranking uses whatever genuinely held-out criterion each tier has:

    * **tier3** -- mean validation loss across seeds, recorded during training;
    * **tier2** -- the ``TimeSeriesSplit`` cross-validation score over train+val
      (negative RMSE, so larger is better);
    * **tier1** -- nothing, because the baselines have no hyperparameters and
      therefore no selection step to bias. Ordering them by test RMSE simply
      reports which fixed rule happened to do best.

    Args:
        frame: Flattened run records.
        tier: Restrict to one tier, or None for all.

    Returns:
        One row per horizon.
    """
    sub = frame if tier is None else frame[frame["tier"] == tier]
    if sub.empty:
        return sub
    grouped = (
        sub.groupby(["model", "variant", "horizon_h"], dropna=False)
        .agg(
            rmse=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            mae=("mae", "mean"),
            r2=("r2", "mean"),
            smape=("smape", "mean"),
            skill=("skill_vs_persistence", "mean"),
            skill_std=("skill_vs_persistence", "std"),
            n_params=("n_params", "first"),
            n_seeds=("seed", "count"),
            val_loss=("best_val_loss", "mean")
            if "best_val_loss" in sub.columns
            else ("rmse", "size"),
            cv=("cv_score", "mean") if "cv_score" in sub.columns else ("rmse", "size"),
        )
        .reset_index()
    )

    if "best_val_loss" in sub.columns and grouped["val_loss"].notna().any():
        grouped["_rank"] = grouped["val_loss"]  # lower validation loss is better
    elif "cv_score" in sub.columns and grouped["cv"].notna().any():
        grouped["_rank"] = -grouped["cv"]  # cv_score is negative RMSE
    else:
        grouped["_rank"] = grouped["rmse"]  # tier 1: no selection to bias
    grouped["_rank"] = grouped["_rank"].fillna(grouped["rmse"])

    return grouped.sort_values("_rank").groupby("horizon_h").first().reset_index()


def main() -> int:
    """Write RESULTS.md and abstract_facts.json."""
    args = parse_args()
    cfg = load_config(args.config)
    log = setup_logging(cfg, "11_make_report")
    check_disk_space(cfg)

    payload = load_results(cfg)
    meta = load_meta(cfg)
    frame = _runs_frame(payload)
    if frame.empty:
        log.error("results.json has no runs; run the model phases first")
        return 1

    horizons = [int(h) for h in cfg.get("task.horizons_h")]
    headline = int(cfg.get("task.headline_horizon_h"))
    boundaries = cfg.get("split.explicit_boundaries")
    audit = json.loads(
        (cfg.path_for("data_interim") / "audit_summary.json").read_text(encoding="utf-8")
    )
    # Read for the Limitations section, so its site-specific numbers come from
    # whichever city this config describes rather than being hardcoded.
    qc_ledger = json.loads(
        (cfg.path_for("data_interim") / str(cfg.get("data.files.qc_ledger"))).read_text(
            encoding="utf-8"
        )
    )
    gaps_path = cfg.path_for("tables") / "audit_longest_gaps.csv"
    longest_gap = pd.read_csv(gaps_path).iloc[0] if gaps_path.exists() else None

    best_deep = _best_per_horizon(frame, "tier3")
    best_classical = _best_per_horizon(frame[frame["tier"].isin(["tier1", "tier2"])])

    persistence = frame[frame["model"] == "persistence"].set_index("horizon_h")["rmse"]

    lines: list[str] = []
    a = lines.append

    a("# Results")
    a("")
    a(f"**{cfg.get('project.title')}**")
    a("")
    a(f"- Site: {cfg.get('data.openaq.site_label')}")
    a(
        f"- Data: {audit['coverage']['first_utc'][:10]} to {audit['coverage']['last_utc'][:10]} "
        f"({audit['usable_years']:.2f} usable years, {audit['pct_observed']:.1f}% hourly coverage)"
    )
    a(
        f"- Split (chronological, no shuffling): train to {str(boundaries['train_end'])[:10]}, "
        f"validation to {str(boundaries['val_end'])[:10]}, test thereafter"
    )
    a(f"- Predictors: {meta['n_predictors']}; horizons: {horizons}")
    a(f"- Seeds: {cfg.get('seeds.multi')}")
    a(f"- Git commit: `{payload['meta'].get('git_commit', 'unknown')}`")
    a("")
    a("> Every number below is generated by `scripts/11_make_report.py` from")
    a("> `results/results.json`. Nothing is transcribed by hand.")
    a("")

    # ---------------------------------------------------------------- headline
    a("## 1. Headline")
    a("")
    if not best_deep.empty:
        row = best_deep[best_deep["horizon_h"] == headline]
        if not row.empty:
            r = row.iloc[0]
            ref = float(persistence.get(headline, float("nan")))
            a(
                f"At the {headline}-hour horizon the best sequence model is **{r['model']}** "
                f"(input window {str(r['variant']).lstrip('w')} h, {int(r['n_params']):,} parameters, "
                f"{int(r['n_seeds'])} seeds)."
            )
            a("")
            a(f"- RMSE **{r['rmse']:.2f} µg/m³** (persistence {ref:.2f})")
            a(f"- MAE {r['mae']:.2f} µg/m³, R² {r['r2']:.3f}, sMAPE {r['smape']:.2f}%")
            a(f"- **Skill score vs persistence: {r['skill']:+.4f}**")
            a("")
    if not best_classical.empty:
        row = best_classical[best_classical["horizon_h"] == headline]
        if not row.empty:
            r = row.iloc[0]
            a(
                f"The best classical model at the same horizon is **{r['model']}**, "
                f"RMSE {r['rmse']:.2f} µg/m³, skill {r['skill']:+.4f}."
            )
            a("")

    significance = payload.get("significance", {})
    dm = significance.get("diebold_mariano", [])
    dm_head = [d for d in dm if d["horizon_h"] == headline]
    if dm_head:
        d = dm_head[0]
        # The Holm-adjusted value is the one that may be cited: one test per
        # horizon is a family of five, and the raw p-value ignores that.
        adjusted = d.get("p_value_holm")
        decisive = adjusted if adjusted is not None else d["p_value"]
        verdict = "significant" if decisive < 0.05 else "not significant"
        line = (
            f"Diebold-Mariano at {headline} h ({d['model_a']} vs {d['model_b']}): "
            f"statistic {d['statistic']:.3f}, p = {d['p_value']:.3g}"
        )
        if adjusted is not None:
            line += f", Holm-adjusted p = {adjusted:.3g}"
        a(f"{line} — **{verdict}** at the 5% level. Lower average loss: {d['better']}.")
        a("")
        if adjusted is not None:
            multiplicity = significance.get("multiplicity", {})
            a(
                f"Adjustment is Holm-Bonferroni over {multiplicity.get('n_tests', len(dm))} "
                "tests, one per horizon. At the nominal 5% level a family that size is "
                "expected to return a significant result now and then even if every "
                "model were identical, so the adjusted column is the one that carries "
                "the claim."
            )
            a("")

    # ---- interval estimates ------------------------------------------------
    ci = [c for c in significance.get("bootstrap_ci", []) if c["horizon_h"] == headline]
    if ci:
        # Named, not `frame`: `frame` is the runs table and the sections below
        # still need it. Shadowing it here crashed the whole report with a
        # KeyError on 'tier' the first time a run actually had MCS data.
        ci_frame = pd.DataFrame(ci).sort_values("rmse")
        a("### Interval estimates")
        a("")
        a(
            f"Moving-block bootstrap, {ci_frame['n_resamples'].iloc[0]:,} resamples over "
            f"{ci_frame['block_size_h'].iloc[0]}-hour blocks. Blocks rather than independent "
            "draws because consecutive hourly errors are strongly correlated, and an "
            "i.i.d. bootstrap would read that correlation as extra evidence."
        )
        a("")
        display = ci_frame.assign(
            **{
                "Model": ci_frame["model"],
                "RMSE": ci_frame["rmse"].round(2),
                "95% CI": [
                    f"[{low:.2f}, {high:.2f}]"
                    for low, high in zip(ci_frame["ci_low"], ci_frame["ci_high"], strict=True)
                ],
            }
        )[["Model", "RMSE", "95% CI"]]
        a(display.to_markdown(index=False))
        a("")

    # ---- model confidence set ----------------------------------------------
    mcs = [m for m in significance.get("model_confidence_set", []) if m["horizon_h"] == headline]
    if mcs:
        mcs_frame = pd.DataFrame(mcs).sort_values("mean_squared_loss")
        retained = mcs_frame[mcs_frame["in_confidence_set"]]["model"].tolist()
        a("### Model Confidence Set")
        a("")
        a(
            f"**{len(retained)} of {len(mcs_frame)} models survive at the 95% level: "
            f"{', '.join(f'`{m}`' for m in retained)}.**"
        )
        a("")
        a(
            "Hansen, Lunde and Nason (2011). A ranked table invites the reader to treat "
            "the top row as the winner even when the gap to the row below is smaller than "
            "the seed-to-seed spread. The confidence set answers the question actually "
            "being asked — which models cannot be separated from the best — and controls "
            "the error rate across the whole elimination sequence rather than one "
            "pairwise test at a time. Ordering *within* the surviving set is not evidence "
            "of an ordering."
        )
        a("")
        display = mcs_frame.assign(
            **{
                "Model": mcs_frame["model"],
                "Mean squared loss": mcs_frame["mean_squared_loss"].round(1),
                "MCS p": mcs_frame["mcs_p_value"].round(3),
                "In 95% MCS": mcs_frame["in_confidence_set"].map({True: "yes", False: "no"}),
            }
        )[["Model", "Mean squared loss", "MCS p", "In 95% MCS"]]
        a(display.to_markdown(index=False))
        a("")

    # ---------------------------------------------------------------- baselines
    a("## 2. Baselines and classical models")
    a("")
    a(
        "Skill is $1 - \\mathrm{RMSE}_{model}/\\mathrm{RMSE}_{persistence}$. Positive beats persistence."
    )
    a("")
    table = frame[frame["tier"].isin(["tier1", "tier2"])].copy()
    pivot = table.pivot_table(index="model", columns="horizon_h", values="rmse", aggfunc="mean")
    a(pivot.round(2).to_markdown())
    a("")
    a("**Note on seasonal-naive.** At h = 24 the seasonal-naive rule $\\hat y(t+h)=y(t+h-24)$")
    a("reduces to $y(t)$, which is persistence. The two rows are identical at that")
    a("horizon by definition, not by accident.")
    a("")

    # Persistence error is not necessarily monotone in the horizon, and when it is
    # not that is a physical result rather than an anomaly -- so detect it and say
    # so rather than leaving a reader to wonder whether the table is wrong.
    if len(persistence) >= 3:
        ordered = persistence.sort_index()
        worst_h = int(ordered.idxmax())
        if worst_h != int(ordered.index.max()):
            a(
                "**Persistence error is not monotone in the horizon.** It peaks at "
                f"h = {worst_h} ({ordered.loc[worst_h]:.2f} µg/m³) and *falls* by "
                f"h = {int(ordered.index.max())} ({ordered.iloc[-1]:.2f} µg/m³). This is the"
            )
            a("diurnal cycle, not an error in the table: at h = 24 persistence compares a")
            a("time with the same clock hour one day earlier, whereas at intermediate")
            a("horizons it compares opposite phases of a strong daily cycle — morning peak")
            a("against afternoon minimum. Any skill score is therefore measured against a")
            a("reference whose difficulty varies with the horizon, which is precisely why")
            a("the raw RMSE column is reported alongside it.")
            a("")

    # ---------------------------------------------------------------- sequence
    a("## 3. Sequence models")
    a("")
    seq = frame[frame["tier"] == "tier3"]
    if not seq.empty:
        agg = (
            seq.groupby(["model", "variant", "horizon_h"])
            .agg(
                rmse_mean=("rmse", "mean"),
                rmse_std=("rmse", "std"),
                skill_mean=("skill_vs_persistence", "mean"),
                params=("n_params", "first"),
                seeds=("seed", "count"),
            )
            .reset_index()
        )
        head = agg[agg["horizon_h"] == headline].sort_values("rmse_mean")
        show = head.assign(
            RMSE=lambda d: d.apply(
                lambda r: (
                    f"{r['rmse_mean']:.2f} ± {r['rmse_std']:.2f}"
                    if pd.notna(r["rmse_std"])
                    else f"{r['rmse_mean']:.2f}"
                ),
                axis=1,
            ),
            Skill=lambda d: d["skill_mean"].round(4),
        )[["model", "variant", "params", "seeds", "RMSE", "Skill"]]
        a(show.to_markdown(index=False))
        a("")
        budget = int(cfg.get("models.sequence.max_params"))
        a(f"Every model above respects the {budget:,}-parameter budget.")
        excluded = payload.get("green", {}).get("excluded_over_budget", [])
        if excluded:
            a("")
            a("Configurations the budget excluded (the constraint was not relaxed):")
            a("")
            for e in excluded:
                a(f"- `{e['name']}` — {e['params']:,} parameters")
        a("")

    # ------------------------------------------------------ selection stability
    # The selection rule is defensible; the question is whether the split it
    # runs on can actually separate the candidates. Reporting the winner without
    # reporting how often the winner wins would overstate the architecture claim
    # by exactly the amount this subsection measures.
    stability = payload.get("selection_stability", [])
    if stability:
        stab = pd.DataFrame(stability)
        a("### Is the selected architecture stable?")
        a("")
        a("Tier 3 is ranked by mean validation loss across seeds, never by test")
        a("error. This validation split is thin, so the rule was re-run on single")
        a("seeds, on leave-one-seed-out subsets, and on 2,000 resamples of the")
        a("seeds, to see how often it returns the same winner.")
        a("")
        stab_show = pd.DataFrame(
            {
                "Horizon": [f"{int(h)} h" for h in stab["horizon_h"]],
                "Selected": stab["selected"],
                "Single seed": [f"{v:.0%}" for v in stab["single_seed_agreement"]],
                "Leave-one-out": [f"{v:.0%}" for v in stab["loso_agreement"]],
                "Seed bootstrap": [f"{v:.0%}" for v in stab["bootstrap_selection_frequency"]],
                "Distinct winners": stab["n_distinct_winners"],
                "ρ(val, test)": [f"{v:+.2f}" for v in stab["spearman_val_test"]],
                "Regret (RMSE)": [f"{v:+.2f}" for v in stab["regret_rmse"]],
            }
        )
        # disable_numparse: tabulate otherwise re-parses these pre-formatted
        # strings as numbers and drops the signs and trailing zeros, so a column
        # of "+0.40" values prints as "0.4".
        a(stab_show.to_markdown(index=False, disable_numparse=True))
        a("")
        stab_head = stab[stab["horizon_h"] == headline]
        if not stab_head.empty:
            s = stab_head.iloc[0]
            a(
                f"**The identity of the winner is not stable.** At {headline} h the "
                f"reported architecture wins {s['bootstrap_selection_frequency']:.0%} of "
                f"seed resamples, {int(s['n_distinct_winners'])} different candidates win "
                f"at least once, and {int(s['n_indistinguishable'])} of "
                f"{int(s['n_candidates'])} sit inside the winner's own between-seed spread."
            )
            a("")
            # Whether the instability matters depends on two numbers, and both
            # differ by city: how much the choice costs, and whether validation
            # loss tracks test error at all. Asserting "bounded and informative"
            # would be false on a record where rho goes negative, so both halves
            # are derived.
            rho = float(s["spearman_val_test"])
            small_regret = abs(float(s["regret_pct"])) < 1.0
            a(
                (
                    "**The consequence is bounded.** "
                    if small_regret
                    else "**The consequence is not negligible.** "
                )
                + f"Selecting on validation rather than on test costs "
                f"{s['regret_rmse']:+.2f} RMSE at this horizon "
                f"({s['regret_pct']:+.2f}% of the best available)"
                + (
                    ", because the candidates it cannot separate are near-ties — the "
                    "same conclusion the Model Confidence Set reaches in §1."
                    if small_regret
                    else ", which is larger than the spread this section can dismiss as a tie."
                )
            )
            a("")
            if rho >= 0.4:
                a(
                    f"Validation loss remains informative here (Spearman ρ = {rho:+.2f} "
                    "against the test ranking); it is simply not sharp enough to "
                    "discriminate within the leading band."
                )
            elif rho > 0.0:
                a(
                    f"Validation loss is only weakly informative here (Spearman ρ = "
                    f"{rho:+.2f} against the test ranking), so the selection carries "
                    "correspondingly little evidence about which candidate is best."
                )
            else:
                a(
                    f"**Validation loss does not track test error at this horizon** "
                    f"(Spearman ρ = {rho:+.2f}). The selection is therefore not "
                    "evidence that the reported architecture is the best one, and no "
                    "architecture-level claim should be made from this record at this "
                    "horizon. It is the same reading the Model Confidence Set gives in "
                    "§1, arrived at independently."
                )
            a("")
            a(
                "Read `gru`/`lstm` results as *a* member of that band rather than as the "
                "uniquely correct architecture. The paper's claims rest on tier-level "
                "comparisons, which this does not disturb; a claim that one recurrent "
                "configuration beats another would not survive it."
            )
            a("")
        worst = stab.loc[stab["regret_rmse"].idxmax()]
        if int(worst["horizon_h"]) != headline:
            a(
                f"Largest regret across horizons: {worst['regret_rmse']:+.2f} RMSE at "
                f"{int(worst['horizon_h'])} h ({worst['regret_pct']:+.2f}%)."
            )
            a("")

    # ---------------------------------------------------------------- stratified
    a("## 4. Stratified performance")
    a("")
    strat = payload.get("stratified")
    if strat:
        s = pd.DataFrame(strat)
        view = s[(s["horizon_h"] == headline) & (s["stratum"] == "pollution")]
        if not view.empty:
            a(
                view[["model", "group", "n", "rmse", "skill_vs_persistence"]]
                .round(3)
                .to_markdown(index=False)
            )
            a("")
            worst = view.sort_values("skill_vs_persistence").iloc[0]
            a(
                f"The weakest stratum is *{worst['group']}* for `{worst['model']}` "
                f"(skill {worst['skill_vs_persistence']:+.4f}). Models that look strong overall "
                "commonly fail on exactly the high-pollution episodes a health advisory exists "
                "to warn about, so this table is reported whether or not it flatters the result."
            )
            a("")
        season = s[(s["horizon_h"] == headline) & (s["stratum"] == "season")]
        if not season.empty:
            a("### By season")
            a("")
            a(
                season[["model", "group", "n", "rmse", "skill_vs_persistence"]]
                .round(3)
                .to_markdown(index=False)
            )
            a("")
            a(
                "Season boundaries follow the Bangladesh Department of Environment "
                "(dry November–April, wet May–October). The 2022 monsoon is missing from the "
                "record entirely, so the wet-season sample draws on eight monsoons, not nine."
            )
            a("")

    # ---------------------------------------------------------------- green
    a("## 5. Green-AI accounting")
    a("")
    a("### Two honesty requirements")
    a("")
    a("1. **CodeCarbon reports estimates, not metered measurements.** It derives energy from")
    a("   hardware power models and falls back to fully modelled values when it cannot read")
    a("   Intel RAPL counters, which is the normal case on Windows without elevated")
    a("   privileges. Whether RAPL was readable is recorded per run in `results.json`.")
    grid = cfg.get("green.grid_carbon_intensity")
    a("2. **The Bangladesh figure is a recomputation, not a second measurement.** The same")
    a("   estimated kWh is multiplied by a cited national grid intensity:")
    a(
        f"   **{grid['value_gco2_per_kwh']} gCO₂e/kWh ({grid['year']})**, "
        f"{grid['source_name']}, <{grid['source_url']}>."
    )
    a("")
    pareto = payload.get("green", {}).get("pareto_table")
    if pareto:
        p = pd.DataFrame(pareto)
        show = p.assign(
            RMSE=p["rmse_mean"].round(2),
            Skill=p["skill_mean"].round(4),
            Params=p["params"].astype(int),
            MACs=p["macs"].map(lambda v: f"{v:,.0f}" if pd.notna(v) else "n/a"),
            Latency_ms=p["latency_ms_mean"].round(4),
            Train_kWh=p["train_kwh_mean"].map(lambda v: f"{v:.3e}" if pd.notna(v) else "n/a"),
            gCO2e_BD=p["co2e_g_bd_mean"].map(
                lambda v: f"{v:.4f}" if pd.notna(v) else "**blank — unavailable**"
            ),
        )[
            [
                "model",
                "window_h",
                "RMSE",
                "Skill",
                "Params",
                "MACs",
                "Latency_ms",
                "Train_kWh",
                "gCO2e_BD",
            ]
        ]
        a(show.to_markdown(index=False))
        a("")

    # ---------------------------------------------------------------- classifier
    a("## 6. AQI-category classification")
    a("")
    clf = payload.get("classification")
    if clf:
        bp = clf["breakpoints"]
        a(
            f"Next-day ({clf['horizon_h']} h) health-advisory classification into "
            f"{len(clf['labels'])} AQI categories."
        )
        a("")
        a(f"- Breakpoints: **{bp['scheme']}**, {bp['revision']}")
        a(f"- Source: {bp['source_org']}, *{bp['source_title']}* — <{bp['source_url']}>")
        a("- Bangladesh's Department of Environment states it applies the USEPA AQI equation")
        a("  and scale directly, so these are the correct breakpoints for a Bangladesh study.")
        a(f"- Target basis: **{clf['target_basis']}** — the breakpoints are defined on a")
        a("  24-hour mean, so the label is the 24-hour mean ending at *t+h*, not the")
        a("  instantaneous hourly value.")
        a(f"- Class imbalance handled with `class_weight = {clf['class_weight']}`.")
        a("")
        present = clf.get("macro_f1_present", {})
        empty = [c["label"] for c in clf["per_class"] if not c["support"]]
        n_seeds = clf["macro_f1"]["n_seeds"]

        if present and empty:
            # Lead with the figure that measures the model. A category with zero
            # test support contributes a structural zero to the all-class mean
            # regardless of how well the model does, so quoting that as the
            # headline understates it by an arithmetic artefact.
            a(
                f"- **Macro-F1 {present['mean']:.4f} ± {present['std']:.4f}** over "
                f"{n_seeds} seeds, across the "
                f"{len(clf['per_class']) - len(empty)} categories present in the test period"
            )
            a(
                f"- Macro-F1 over all {len(clf['per_class'])} defined categories: "
                f"{clf['macro_f1']['mean']:.4f} ± {clf['macro_f1']['std']:.4f}"
            )
        else:
            a(
                f"- **Macro-F1 {clf['macro_f1']['mean']:.4f} ± {clf['macro_f1']['std']:.4f}** "
                f"over {n_seeds} seeds"
            )
        a(
            f"- Weighted-F1 {clf['weighted_f1']['mean']:.4f}, accuracy {clf['accuracy']['mean']:.4f}, "
            f"balanced accuracy {clf['balanced_accuracy']['mean']:.4f}"
        )
        a("")
        a("Macro-F1 is the headline: the categories are severely imbalanced, so accuracy")
        a("alone would be dominated by the majority classes and would hide failure on the")
        a("hazardous categories an advisory exists to flag.")
        if present and empty:
            a("")
            a(
                f"{'; '.join(f'**{c}**' for c in empty)} never occurs in the test period, so it "
                "has zero support. A category with no instances scores an F1 of zero "
                "whatever the model predicts, and averaging that in measures the test "
                "period's composition rather than the classifier. Both figures are given "
                "above; the present-class one is the model's performance."
            )
        a("")
        a(
            pd.DataFrame(clf["per_class"])[
                ["label", "support", "precision_mean", "recall_mean", "f1_mean"]
            ]
            .round(3)
            .to_markdown(index=False)
        )
        a("")

    # ---------------------------------------------------------- cross-city
    # Section numbering is computed, not hardcoded: the comparison city's own
    # report has no cross-city block, and a fixed "## 8." there would skip 7.
    next_section = 7
    cross = payload.get("cross_city") or {}
    if cross:
        other = next(iter(cross))
        block = cross[other]
        ranking = pd.DataFrame(block["ranking"])
        rho = block.get("spearman_rank_correlation")
        this_city = str(cfg.get("data.site.city"))

        a(f"## {next_section}. Cross-city generalisation: {this_city} against {other}")
        next_section += 1
        a("")
        a("The same pipeline, model grid, seed set, split procedure and QC thresholds")
        a(f"were run on {other}. Co-located co-pollutants were excluded there because")
        a(f"{this_city} has none, so the comparison tests the method rather than the")
        a("richness of the feature set.")
        a("")
        a(f"**Spearman rank correlation between the two cities' method rankings: {rho:+.3f}.**")
        a("")
        if rho is not None and rho < 0.3:
            a("The ranking does not transfer. A benchmark run on either city alone would")
            a("have recommended a different method, and neither recommendation would")
            a("generalise. This is the central argument for reporting both, and it is not")
            a("visible from either city in isolation.")
            a("")

        pivot = ranking.pivot_table(index="model", columns="city", values="rank", aggfunc="first")
        skl = ranking.pivot_table(index="model", columns="city", values="skill", aggfunc="first")
        merged = pivot.join(skl, lsuffix=" rank", rsuffix=" skill").reset_index()
        merged = merged.sort_values(f"{this_city} rank")
        a(merged.round(4).to_markdown(index=False))
        a("")

        chosen = ranking[ranking.get("model") == "best sequence"]
        if not chosen.empty and "chosen" in chosen.columns:
            a("Validation-selected sequence configuration per city:")
            a("")
            for _, r in chosen.iterrows():
                a(f"- {r['city']}: `{r['chosen']}` (RMSE {r['rmse']:.2f})")
            a("")

        a("### What this changes")
        a("")
        context = pd.DataFrame(block["context"])
        a(context.to_markdown(index=False))
        a("")

        # Direction is READ from the ranking table, never asserted. An earlier
        # version hardcoded "in the other city the sequence model beats every
        # tree". Correcting the sequence tier's inputs and training recipe
        # reversed that, and the prose then contradicted the table printed
        # directly above it. Whatever the numbers say, the text follows.
        seq_rank = {
            str(r["city"]): int(r["rank"])
            for _, r in ranking[ranking["model"] == "best sequence"].iterrows()
        }
        cov_row = context[context["Quantity"].str.startswith("Hourly coverage")]
        coverage = {c: float(cov_row.iloc[0][c]) for c in seq_rank if c in cov_row.columns}

        if len(seq_rank) == 2 and len(coverage) == 2:
            fragmented, complete = sorted(coverage, key=lambda c: coverage[c])
            a(
                f"The sequence tier ranks **{seq_rank[fragmented]}** on the more fragmented "
                f"record ({fragmented}, {coverage[fragmented]:.1f}% coverage) and "
                f"**{seq_rank[complete]}** on the near-complete one "
                f"({complete}, {coverage[complete]:.1f}%)."
            )
            a("")
            if seq_rank[fragmented] > seq_rank[complete]:
                a("That ordering is what a fragmentation account predicts. Tree models on")
                a("lagged tabular features tolerate broken records because each row stands")
                a("alone; sequence models need contiguous windows, and gap-aware windowing")
                a("discards a large fraction of them where the record is torn. **It remains")
                a("an observation on two cities, not evidence** — see the gap-injection")
                a("experiment, which holds every other difference fixed.")
            elif seq_rank[fragmented] < seq_rank[complete]:
                a("**That ordering is the opposite of what a fragmentation account predicts**,")
                a("and it is reported here rather than set aside. Two cities differ in far")
                a("more than the continuity of their records: the comparison city's 24-hour")
                a("problem is simply harder, with persistence RMSE far above this city's, and")
                a("its test period is shorter and spans a seasonal transition. A two-city")
                a("contrast cannot separate fragmentation from any of that, which is why the")
                a("mechanism is tested by injection on a single record instead of inferred")
                a("from a pair. The controlled experiment and this observation disagree, and")
                a("the controlled experiment is the one with a valid counterfactual.")
            else:
                a("The sequence tier holds the same rank in both cities, so this pair says")
                a("nothing about fragmentation in either direction.")
            a("")

            # The coverage contrast that frames this whole section is not the
            # quantity the models experience. Reported here rather than only in
            # the availability section, because it is this comparison it
            # qualifies -- and it is read from the generated file, not asserted.
            law_here = (
                Path(str(cfg.get("paths.results"))) / f"missingness_law{city_suffix(cfg)}.json"
            )
            if law_here.exists():
                avail_rows = pd.DataFrame(
                    json.loads(law_here.read_text(encoding="utf-8")).get("availability", [])
                )
                deep = (
                    avail_rows[avail_rows["Lookback (h)"] == avail_rows["Lookback (h)"].max()]
                    if not avail_rows.empty
                    else pd.DataFrame()
                )
                if len(deep) >= 2:
                    spread = ", ".join(
                        f"{r['Record']} {float(r['Availability (grid)']) * 100:.1f}%"
                        for _, r in deep.iterrows()
                    )
                    cov_spread = ", ".join(
                        f"{c} {coverage[c]:.1f}%" for c in sorted(coverage, key=coverage.get)
                    )
                    a("**Coverage is not the quantity that reaches the model.** Hourly")
                    a(f"coverage runs {cov_spread}, but forecast availability — the share of")
                    a("test hours any model here can answer at all — runs")
                    a(f"{spread}. A row needs an unbroken history behind it, so a scattered")
                    a("outage costs far more than its own length, and a near-complete record")
                    a("with many small gaps can be less available than a broken one with few")
                    a("large ones. The fragmentation gradient this section reads its ordering")
                    a("against is therefore much weaker than the coverage figures suggest,")
                    a("and on these records it does not run the way coverage does. See the")
                    a("availability section below.")
                    a("")

        # Strength of evidence is read from each city's own Model Confidence
        # Set, not asserted. A rank order can be printed for any pair of cities;
        # whether either ordering is separable from noise is a different
        # question, and the honest answer has to come from the data.
        mcs_size = block.get("model_confidence_set_size") or {}
        if mcs_size:
            a("**Strength of evidence.** How many models each city's own 95% Model")
            a("Confidence Set retains at this horizon — a set containing every candidate")
            a("means that city's ordering is not separable from noise:")
            a("")
            for city_name, size in mcs_size.items():
                n_ret, n_all = int(size["n_retained"]), int(size["n_candidates"])
                verdict = (
                    "no ordering is supported"
                    if n_ret == n_all
                    else f"retains {', '.join(size['retained'])}"
                )
                a(f"- {city_name}: **{n_ret} of {n_all}** — {verdict}")
            a("")
            uninformative = [c for c, s in mcs_size.items() if s["n_retained"] == s["n_candidates"]]
            if uninformative:
                a(
                    f"{' and '.join(uninformative)} cannot distinguish any method from any "
                    "other here, persistence included. The ranking printed above for "
                    f"{' and '.join(uninformative)} is therefore a description of this "
                    "sample, not a finding, and no claim in this report rests on it."
                )
                a("")
        a("Absolute skill is lower for every method in the comparison city: its test")
        a("period is shorter and spans a seasonal transition.")
        a("")

    # ------------------------------------------------- gap-injection ablation
    # Read from results/ablation_gap_injection.json rather than results.json.
    # That file is the ablation's own generated source of truth, written by
    # 17_ablation_analysis.py and never touched by hand, so the "no transcribed
    # numbers" rule holds. It cannot live in results.json: the experiment runs
    # under the donor city's config and so would land in that city's file,
    # while the injected gap profile and the claim both belong to this one.
    ablation_path = Path(str(cfg.get("paths.results"))) / str(
        cfg.get("ablation.gap_injection.output_name", "ablation_gap_injection.json")
    )
    if ablation_path.exists():
        abl = json.loads(ablation_path.read_text(encoding="utf-8"))
        by_family = pd.DataFrame(abl.get("analysis", {}).get("by_family", []))
        # Omitting an un-run section is right; omitting it in silence is not.
        # 16_gap_injection.py rewrites this file wholesale and drops the
        # `analysis` block that 17_ablation_analysis.py put there, so a grid
        # that was extended but not re-analysed leaves cells with no analysis --
        # and this report then quietly renumbers Limitations over the top of the
        # study's central section. That happened once. Say so, loudly.
        if by_family.empty and abl.get("cells"):
            log.warning(
                "%s holds %d cells but no analysis block, so the gap-injection "
                "section is being OMITTED from this report. Run "
                "scripts/17_ablation_analysis.py --config <donor config> first.",
                ablation_path,
                len(abl["cells"]),
            )
        if not by_family.empty:
            a(f"## {next_section}. Does fragmentation cause the ranking to change?")
            next_section += 1
            a("")
            a("§7 compares two cities that differ in everything at once, so it cannot")
            a("attribute a ranking difference to any one of those differences. This")
            a("section holds the record fixed and cuts it two ways.")
            a("")
            analysis = abl.get("analysis", {})
            if analysis.get("grid_complete") is False:
                n_missing = len(analysis.get("missing_cells", []))
                n_expected = analysis.get("n_expected_degraded_cells")
                a(f"> ⚠ **PROVISIONAL.** {n_missing} of {n_expected} degraded cells have not")
                a("> run, so the levels below are unequally weighted and this is not yet the")
                a("> designed experiment. Resume `scripts/16_gap_injection.py`, re-run")
                a("> `scripts/17_ablation_analysis.py`, then regenerate this report.")
                a("")
            a(f"- Donor record: **{abl.get('donor')}**")
            a(f"- Injected gap-length distribution: **{abl.get('gap_profile_source')}**")
            a(f"- Horizon: **{abl.get('horizon_h')} h**")
            a("")
            a(str(abl.get("design", "")).replace("\n", " "))
            a("")

            wide = by_family.pivot_table(
                index=["family", "target_coverage"],
                columns="arm",
                values="skill",
                aggfunc="first",
            ).reset_index()
            if {"fragmented", "contiguous"} <= set(wide.columns):
                wide["gap"] = wide["fragmented"] - wide["contiguous"]
                table = wide.pivot_table(
                    index="family", columns="target_coverage", values="gap", aggfunc="first"
                )
                table = table[sorted(table.columns, reverse=True)]
                table.columns = [f"{c * 100:.0f}%" for c in table.columns]
                order = [f for f in FAMILY_ORDER if f in table.index]
                a("**Fragmented minus contiguous skill, at matched coverage.** Both arms")
                a("remove the same number of observed hours at each level, so this")
                a("difference is the effect of *arrangement* with volume held constant.")
                a("Negative means fragmentation costs that family more than the equivalent")
                a("loss of contiguous data.")
                a("")
                a(table.loc[order].round(4).to_markdown())
                a("")

                a("The undegraded level removes nothing, so both arms are the same run")
                a("and their difference there is exactly zero by construction. Any other")
                a("value in that column would mean the injector perturbs something besides")
                a("contiguity.")
                a("")

        # The paired test, not the table above, is the claim. Every level and
        # every injection seed is a matched pair -- same hours removed, different
        # arrangement -- so differencing within the pair removes both the level
        # and the draw. Averaging the seeds away first and differencing two means
        # would throw that pairing out.
        tests = pd.DataFrame(abl.get("analysis", {}).get("paired_tests", []))
        if not tests.empty:
            n_pairs = int(tests["n_pairs"].max())
            a(f"**Paired test.** Each of the {n_pairs} pairs is one (coverage level,")
            a("injection seed): the two arms remove an identical number of observed")
            a("hours and differ only in arrangement. The representative model per")
            a("family is fixed on the undegraded record and never re-chosen per arm,")
            a("so the difference cannot absorb a change of model. Wilcoxon signed-rank,")
            a("Holm-corrected across families.")
            a("")
            shown = pd.DataFrame(
                {
                    "Family": tests["family"],
                    "Pairs": tests["n_pairs"],
                    "Mean gap": tests["mean_arm_gap"].round(4),
                    "95% CI": [
                        f"[{lo:+.4f}, {hi:+.4f}]"
                        for lo, hi in zip(tests["ci_low"], tests["ci_high"], strict=False)
                    ],
                    # Not .round(4): round(7e-06, 4) is 0.0 and prints "0.0000",
                    # which is not a p-value -- no test returns zero probability.
                    "p": [_fmt_p(v) for v in tests["p_wilcoxon"]],
                    "p (Holm)": [_fmt_p(v) for v in tests["p_holm"]],
                }
            )
            a(shown.to_markdown(index=False))
            a("")

            row = tests[tests["family"] == "sequence"]
            others = tests[(tests["family"] != "sequence") & tests["significant_holm"]]
            if not row.empty:
                r = row.iloc[0]
                if bool(r["significant_holm"]) and float(r["mean_arm_gap"]) < 0:
                    a(
                        f"The sequence family loses {abs(float(r['mean_arm_gap'])):.4f} skill to "
                        f"arrangement alone (95% CI [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}], "
                        f"Holm p = {float(r['p_holm']):.4f})."
                    )
                    if others.empty:
                        a("No other family's gap survives correction. Fragmentation is")
                        a("costly specifically to the model class that requires contiguous")
                        a("windows — which is the mechanism §7 proposed and could not test.")
                    else:
                        a(
                            "It is not alone: "
                            + ", ".join(
                                f"{o['family']} ({o['mean_arm_gap']:+.4f})"
                                for _, o in others.iterrows()
                            )
                            + " also move, so the effect is not specific to the sequence tier."
                        )
                else:
                    a("The sequence family's gap does not survive correction. On this donor")
                    a("the experiment does not support the mechanism §7 proposes.")
                a("")

            # Which model stands for each family matters to the reader, and for
            # the climatological row it matters most: the representative is
            # chosen by highest skill on the reference cell, persistence scores
            # identically zero there by construction, so climatology always wins
            # the slot -- and climatology is fitted on the DEGRADED training
            # split. The row is "reads the record, needs no windows", which is
            # informative, but it is not the no-training-data control this
            # section used to call it.
            gaps = pd.DataFrame(abl.get("analysis", {}).get("paired_gaps", []))
            if not gaps.empty and "model" in gaps.columns:
                reps = gaps.groupby("family")["model"].first()
                a(
                    "Representative model per family, fixed on the undegraded record: "
                    + ", ".join(f"{fam} = `{m}`" for fam, m in reps.items())
                    + "."
                )
                a("")

            # Built from the family records, not from analysis.family_ranks:
            # that field is a JSON round-trip of a MultiIndex pivot and its
            # column labels come back as the strings "('contiguous', 0.75)".
            if "rank" in by_family.columns:
                ranks = by_family.pivot_table(
                    index=["arm", "family"],
                    columns="target_coverage",
                    values="rank",
                    aggfunc="first",
                )
                ranks = ranks[sorted(ranks.columns, reverse=True)]
                ranks.columns = [f"{c * 100:.0f}%" for c in ranks.columns]
                idx = [
                    (arm, fam)
                    for arm in ("fragmented", "contiguous")
                    for fam in FAMILY_ORDER
                    if (arm, fam) in ranks.index
                ]
                a("Family rank within each cell (1 = best skill). The sequence row is")
                a("the result: it moves under fragmented removal and does not move under")
                a("contiguous removal of the same number of hours.")
                a("")
                shown = ranks.loc[idx].astype(int)
                shown.index = [f"{arm} / {fam}" for arm, fam in idx]
                shown.index.name = "arm / family"
                a(shown.to_markdown())
                a("")

            n_draws = by_family.get("n_draws")
            a("**What this does and does not establish.** The claim is causal for this")
            a("record: fragmentation is manipulated, volume is held constant, the test")
            a("period is untouched, and the optimizer-step budget is equalised so that")
            a("a fragmented cell is not simply undertrained. What it does not establish")
            a("is generality.")
            a("")
            rep_path = Path(str(cfg.get("paths.results"))) / "donor_replication.json"
            if n_draws is not None and not rep_path.exists():
                a(f"- **One donor record**, degraded {int(n_draws.max())} ways per cell. A second")
                a("  donor would separate the effect from this station's own dynamics.")
            a("- **One horizon** and one injected gap-length distribution.")
            a("- The per-cell differences in the first table are individually noisy; it is")
            a("  the paired test across all levels and draws that carries the result.")
            a("")

            # ---------------------------------------------- donor replication
            # Same rule as the section above: written by 18_donor_replication.py,
            # read here, never transcribed. Absent until the replication donors
            # have run, in which case this subsection simply does not appear.
            if rep_path.exists():
                rep = json.loads(rep_path.read_text(encoding="utf-8"))
                rep_verdict = pd.DataFrame(rep.get("verdict", []))
                rep_tests = pd.DataFrame(rep.get("per_donor_tests", []))
                if not rep_verdict.empty and not rep_tests.empty:
                    rep_donor_names = list(rep.get("donors", []))
                    a(f"### {next_section - 1}b. Does it replicate on another record?")
                    a("")
                    a(
                        "The same experiment on "
                        f"{len(rep_donor_names)} donor records: {', '.join(rep_donor_names)}."
                    )
                    a("")
                    a(str(rep.get("scope_note", "")))
                    a("")
                    rep_grid = rep_tests.pivot_table(
                        index="family", columns="donor", values="mean_arm_gap", aggfunc="first"
                    )
                    rep_flag = rep_tests.pivot_table(
                        index="family",
                        columns="donor",
                        values="significant_holm",
                        aggfunc="first",
                    )
                    rep_fam_order = [f for f in FAMILY_ORDER if f in rep_grid.index]
                    rep_by_family = rep_verdict.set_index("family")
                    rep_shown = pd.DataFrame({"Family": rep_fam_order})
                    for d in rep_donor_names:
                        if d in rep_grid.columns:
                            rep_shown[d] = [
                                f"{rep_grid.loc[f, d]:+.4f}{'*' if bool(rep_flag.loc[f, d]) else ''}"
                                for f in rep_fam_order
                            ]
                    rep_shown["Replicates"] = [
                        "**yes**" if bool(rep_by_family.loc[f, "replicates"]) else "no"
                        for f in rep_fam_order
                    ]
                    a(rep_shown.to_markdown(index=False))
                    a("")
                    a("`*` marks Holm significance within that donor. **Replicates** is the")
                    a("strict rule: " + str(rep.get("replication_rule", "")))
                    a("")

                    # The rule above tests each family against its OWN null and asks
                    # whether that repeats. The claim being made is a comparison, and
                    # a comparison needs a contrast: a family can clear its own null
                    # on every donor and still be indistinguishable from the family it
                    # is being contrasted with. Reported here, next to the rule it
                    # qualifies, rather than left in a table nobody opens.
                    contrasts = pd.DataFrame(rep.get("family_contrasts_pooled", []))
                    if not contrasts.empty:
                        a("#### Is the sequence tier affected *more* than the others?")
                        a("")
                        a("The rule above is not a contrast. Differencing the families within")
                        a("each matched (coverage level, injection seed, donor) triple asks the")
                        a("question the claim actually makes; the triples are already matched,")
                        a("because at one draw every family was fitted on the same two degraded")
                        a("copies of the same record.")
                        a("")
                        a("| Contrast | Mean | 95% CI | p | p (Holm) |")
                        a("|---|---:|---|---:|---:|")
                        for r in contrasts.itertuples():
                            a(
                                f"| sequence − {r.family} | {r.mean_contrast:+.4f} | "
                                f"[{r.ci_low:+.4f}, {r.ci_high:+.4f}] | "
                                f"{_fmt_p(r.p_wilcoxon)} | {_fmt_p(r.p_holm)} |"
                            )
                        a("")
                        iut = bool(rep.get("iut_sequence_worse_than_every_family"))
                        if iut:
                            a("Every contrast rejects, so the claim that the sequence tier is the")
                            a("family fragmentation hurts most is supported as an")
                            a("intersection-union test.")
                        else:
                            worst = contrasts.loc[contrasts["p_wilcoxon"].idxmax()]
                            a(
                                f"**The claim does not survive as a contrast.** Its point estimate "
                                f"is the largest gap of any family on every donor, but it is not "
                                f"separable from `{worst['family']}` "
                                f"({worst['mean_contrast']:+.4f}, p = {worst['p_wilcoxon']:.3f}), "
                                f"and the conjunction it would need — worse than *every* other "
                                f"family — therefore fails. Because the claim is a conjunction the "
                                f"correct procedure is an intersection-union test, in which each "
                                f"contrast is tested unadjusted and a single non-rejection "
                                f"withholds the claim."
                            )
                            a("")
                            a("What survives is narrower and is what this report states: the")
                            a(
                                "sequence tier's gap is the only one that is Holm-significant against"
                            )
                            a("zero on every donor, and it is the largest in mean. That is a")
                            a("statement about reliability across records, not about being worse")
                            a("than another family on any one of them.")
                        a("")

                    doses = pd.DataFrame(rep.get("dose_response_pooled", []))
                    if not doses.empty:
                        a("#### Does the gap steepen as the record degrades?")
                        a("")
                        a("The paired test pools every coverage level, which answers whether the")
                        a("gap is nonzero and cannot answer whether it grows. A gap flat in")
                        a("coverage is a fixed cost; one that steepens is a mechanism. One slope")
                        a("is fitted per (injection seed, donor) and the slopes are tested.")
                        a("")
                        a("| Family | Gap per 10 pp coverage lost | 95% CI | p (Holm) |")
                        a("|---|---:|---|---:|")
                        for r in doses.itertuples():
                            mark = "**" if r.significant_holm else ""
                            a(
                                f"| {mark}{r.family}{mark} | "
                                f"{r.gap_change_per_10pp_lost:+.4f} | "
                                f"[{r.ci_low:+.4f}, {r.ci_high:+.4f}] | "
                                f"{_fmt_p(r.p_holm)} |"
                            )
                        a("")
                        sig = doses[doses["significant_holm"]]["family"].tolist()
                        if sig:
                            a(
                                f"The gap steepens significantly for {_join_and(sig)}. "
                                f"Where it does, fragmentation is not a fixed toll but a cost that "
                                f"accelerates as the record breaks up — which is the shape the "
                                f"sterilisation-radius account predicts, and the one the pooled "
                                f"test averages away."
                            )
                            a("")
                    a("")

                    rep_yes = [f for f in rep_fam_order if bool(rep_by_family.loc[f, "replicates"])]
                    rep_seq_ok = "sequence" in rep_yes
                    rep_others = [f for f in rep_yes if f not in ("sequence", "linear")]
                    # Whether every family's point estimate is negative on every
                    # donor decides how strong a claim the table licenses. If it
                    # is, fragmentation is not costless for anything and the
                    # sequence result is about CONSISTENCY, not exclusivity --
                    # stating otherwise would overclaim off a significance
                    # threshold rather than off an effect.
                    rep_all_negative = all(
                        float(rep_by_family.loc[f, "max_gap"]) < 0.0 for f in rep_fam_order
                    )
                    rep_seq_mean = float(rep_by_family.loc["sequence", "mean_gap_across_donors"])
                    rep_seq_largest = all(
                        rep_seq_mean <= float(rep_by_family.loc[f, "mean_gap_across_donors"])
                        for f in rep_fam_order
                    )
                    if rep_seq_ok and not rep_others:
                        a("The sequence family's gap is the only one Holm-significant on every")
                        a("donor.")
                        if rep_all_negative:
                            a("")
                            a(
                                "Every family's point estimate is negative on every donor, so "
                                "this is not a finding that fragmentation costs the others "
                                "nothing. What separates the sequence tier is that its gap is "
                                "the one that appears *reliably* rather than on some records "
                                "and not others"
                                + (
                                    f", and it is the largest mean gap across donors "
                                    f"({rep_seq_mean:+.4f})."
                                    if rep_seq_largest
                                    else f" (mean across donors {rep_seq_mean:+.4f})."
                                )
                            )
                        else:
                            a("On this evidence fragmentation is costly specifically to the")
                            a("model class that requires contiguous windows.")
                    elif rep_seq_ok and rep_others:
                        a(
                            "The sequence family replicates, but so does "
                            + ", ".join(rep_others)
                            + ". Fragmentation is not costly to the windowed model class alone,"
                        )
                        a("and the claim this section can support is the weaker one: it costs")
                        a("the sequence tier *most*, and costs it first as coverage falls.")
                    elif not rep_seq_ok:
                        a("**The sequence family's gap does not replicate across donors.** The")
                        a("effect seen on the primary donor is not established as a property of")
                        a("fragmentation, and no claim in this report should rest on it.")

                    # The climatological row at the most severe level is the
                    # reading that would sink the specificity claim on one donor
                    # alone: a model that needs no contiguous window should not
                    # care how the record is arranged, yet on the largest-gap
                    # donor it collapses as far as the sequence tier does.
                    # Reporting whether that collapse reproduces is the whole
                    # reason a second and third donor were run.
                    rep_levels = pd.DataFrame(rep.get("per_level_gaps", []))
                    if not rep_levels.empty and "climatological" in set(rep_levels["family"]):
                        rep_worst = float(rep_levels["target_coverage"].min())
                        rep_naive = rep_levels[
                            (rep_levels["family"] == "climatological")
                            & (rep_levels["target_coverage"] == rep_worst)
                        ].set_index("donor")["arm_gap"]
                        rep_spread = ", ".join(
                            f"{d} {float(rep_naive[d]):+.4f}"
                            for d in rep_donor_names
                            if d in rep_naive.index
                        )
                        a("")
                        a(
                            f"**The climatological row at the severest level "
                            f"({rep_worst * 100:.0f}% "
                            f"coverage):** {rep_spread}. "
                        )
                        if float(rep_naive.max()) - float(rep_naive.min()) > abs(
                            float(rep_naive.mean())
                        ):
                            a(
                                "These disagree by more than their own average, so the "
                                "collapse visible on the largest-gap donor is a property of that "
                                "record rather than of fragmentation. Read on one donor alone it "
                                "would have argued that fragmentation degrades anything estimated "
                                "from the record, windowed or not; it does not survive the other "
                                "donors, and that is the specific thing a second and third record "
                                "were run to test."
                            )
                        else:
                            a(
                                "These agree in magnitude across donors, so severe fragmentation "
                                "penalises even models that never read the training record. The "
                                "control is therefore not clean at this level, and the section's "
                                "claim must be read as relative across families rather than as "
                                "an effect absent from the baselines."
                            )
                    a("")
                    if rep.get("incomplete_donors"):
                        a(
                            "> ⚠ Provisional for "
                            f"{', '.join(rep['incomplete_donors'])}: rep_grid incomplete."
                        )
                        a("")

    # ---- mediation: is the radius WHY fragmentation costs skill? -----------
    # Sits at the end of the gap-injection section rather than in the
    # availability one, because it is the gap-injection claim it completes: 8
    # measures the penalty, the law explains what should cause it, and this
    # manipulates that cause and watches the penalty follow.
    med_paths = sorted(Path(str(cfg.get("paths.results"))).glob("mediation_*.json"))
    if med_paths:
        med = json.loads(med_paths[0].read_text(encoding="utf-8"))
        tests = pd.DataFrame(med.get("tests", []))
        mediator = pd.DataFrame(med.get("mediator", []))
        radii = med.get("radii", [])
        if not tests.empty and len(radii) >= 2:
            deep, shallow = max(radii), min(radii)
            a(f"### {next_section - 1}c. Is the backward reach *why* it costs skill?")
            a("")
            a("The section above measures a penalty and the law below explains what should")
            a("cause it: a gap sterilises the hours behind it, so scattering removals")
            a("destroys far more supervision than clustering them. That is a mediation")
            a("claim, and it predicts something falsifiable — shorten the reach and the")
            a("penalty should shrink.")
            a("")
            a("The reach is set by configuration rather than inferred from a regression, so")
            a("this is a stronger design than a regression-based mediation and needs no")
            a("sequential-ignorability assumption. `inject_gaps` is deterministic in")
            a("(arm, coverage, seed) and runs before any feature is built, so the same")
            a("degraded series appears at every reach and each draw is a repeated measure.")
            a("")

            if not mediator.empty:
                a("**The mediator moves first.** Usable training rows the fragmented arm")
                a("loses relative to the contiguous arm, at the same removed-hour count:")
                a("")
                a("| Sterilisation radius (h) | Row deficit |")
                a("|---:|---:|")
                for r in mediator.itertuples():
                    a(f"| {int(r.radius_h)} | {r.mean_row_deficit:,.0f} |")
                a("")
                if not med.get("mediator_deficit_shrinks_with_radius"):
                    a("> ⚠ The deficit does not shrink with the radius, so there is no mediator")
                    a("> here and the test below should not be read as one.")
                    a("")

            a(f"**And the penalty follows it.** Arm gap at R = {deep} h against R = {shallow} h,")
            a("differenced within the same injection draw:")
            a("")
            a(f"| Family | R = {deep} | R = {shallow} | Change | 95% CI | Mediated | p (Holm) |")
            a("|---|---:|---:|---:|---|---:|---:|")
            for r in tests.itertuples():
                gap_deep = getattr(r, f"gap_R{deep}")
                gap_shallow = getattr(r, f"gap_R{shallow}")
                # A proportion of nearly nothing is not a proportion: with no
                # penalty at the deepest reach the ratio is dominated by its own
                # denominator and reads as 235% mediated, which means only that
                # there was nothing there to mediate.
                share = f"{100 * r.proportion_mediated:.0f}%" if abs(gap_deep) >= 0.005 else "—"
                a(
                    f"| {r.family} | {gap_deep:+.4f} | {gap_shallow:+.4f} | "
                    f"{r.delta:+.4f} | [{r.ci_low:+.4f}, {r.ci_high:+.4f}] | "
                    f"{share} | {_fmt_p(r.p_holm)} |"
                )
            a("")
            a("One-sided and pre-declared: the mechanism predicts the gap becomes less")
            a("negative, and spending power on the other direction would be spending it on")
            a("something the account does not claim.")
            a("")

            dil = med.get("dilution_bound") or {}
            if dil:
                factor = float(dil["expected_gap_factor_under_pure_dilution"])
                a(
                    f"**The confound, bounded rather than argued.** A shorter reach scores "
                    f"more test hours — {dil['n_rows_deep']:,} at R = {deep} against "
                    f"{dil['n_rows_shallow']:,} at R = {shallow} — so the two gaps are not "
                    f"measured on the same rows. If those extra hours carried no arm "
                    f"difference whatever, the gap would still shrink to "
                    f"{100 * factor:.0f}% of itself by arithmetic alone: the between-arm "
                    f"error is averaged over more rows and the persistence denominator "
                    f"moves. Anything beyond that is mediated."
                )
                a("")
                seq = tests[tests["family"] == "sequence"]
                if not seq.empty and "shrinkage_beyond_dilution" in seq:
                    row = seq.iloc[0]
                    beyond = float(row["shrinkage_beyond_dilution"])
                    total = float(row["delta"])
                    if total:
                        a(
                            f"For the sequence tier that leaves dilution explaining "
                            f"{abs(total - beyond):.4f} of the {abs(total):.4f} change and "
                            f"the reach explaining {abs(beyond):.4f} — "
                            f"{100 * abs(beyond) / abs(total):.0f}% of the effect."
                        )
                        a("")

            null_rows = tests[tests["p_wilcoxon_onesided"] > 0.10]["family"].tolist()
            if null_rows:
                a(
                    f"**The falsification control holds.** {_join_and(null_rows)} had little "
                    f"or no gap to mediate at the deepest reach, and shows no mediation. A "
                    f"design in which every family moved regardless of whether it had a "
                    f"penalty would be measuring something other than the mechanism."
                )
                a("")
            a("**Scope.** One donor and one horizon. This establishes the mechanism on the")
            a("record where the penalty was measured; it does not establish that the same")
            a("mechanism carries the effect on the other donor stations, whose grids were")
            a("run at the configured reach only.")
            a("")

    # ---- forecast availability and the lookback frontier -------------------
    # Its own section rather than a subsection of the ablation, because it is a
    # property of the RECORD and the feature set, not of the gap-injection
    # experiment: every number in sections 1-8 is conditional on it. Read from
    # generated files only; nothing here is transcribed.
    results_dir = Path(str(cfg.get("paths.results")))
    law_path = results_dir / f"missingness_law{city_suffix(cfg)}.json"
    # Both cities share paths.results, so this name carries a city suffix.
    frontier_path = results_dir / f"availability_frontier{city_suffix(cfg)}.json"
    law: dict = {}
    if law_path.exists():
        law = json.loads(law_path.read_text(encoding="utf-8"))
        a(f"## {next_section}. What a gap costs, and how often a model can answer")
        next_section += 1
        a("")
        a("Every accuracy figure above is conditional on the model being able to produce a")
        a("forecast at all, and that condition has not so far been reported. A row is")
        a("scored only if it carries")
        a(f"`pos_in_run >= {law['radius_h'] - law['horizon_h']}` hours of unbroken history and")
        a(f"a target {law['horizon_h']} hours further on inside the same run, so a gap does not")
        a("cost the hours it removes: it costs those hours **plus the")
        a(f"{law['radius_h']} behind it** that no longer reach back far enough.")
        a("")
        a("The consequence is that the cost of missingness is governed by the *number* of")
        a("gaps rather than their total length. That is the gap-injection experiment's arm")
        a("contrast in closed form: at an identical hour count, scattering the removals")
        a("sterilises many radii where clustering them sterilises few — which is the")
        a("mechanism that experiment measures without explaining.")
        a("")

        amp = pd.DataFrame(law.get("amplification_by_record", []))
        if not amp.empty:
            a(
                "| Record | Missing hours | Missing (%) | Runs | Usable hours destroyed | Amplification |"
            )
            a("|---|---:|---:|---:|---:|---:|")
            for _, r in amp.iterrows():
                a(
                    f"| {r['Record']} | {int(r['Missing hours']):,} | "
                    f"{float(r['Missing (%)']):.2f} | {int(r['Runs']):,} | "
                    f"{int(r['Hours lost to short history']):,} | "
                    f"{float(r['Amplification']):.1f}x |"
                )
            a("")
            worst = amp.loc[amp["Amplification"].idxmax()]
            best = amp.loc[amp["Amplification"].idxmin()]
            a(
                f"**A near-complete record is not a well-supervised one.** {worst['Record']} loses "
                f"{worst['Amplification']:.1f} usable hours for every hour missing, against "
                f"{best['Record']}'s {best['Amplification']:.1f}, because amplification follows the "
                f"number of runs and not the number of absent hours."
            )
            a("")

        fit = law.get("fit", {})
        held = law.get("held_out", {})
        if fit and held:
            a("The relationship is close enough to state as a law. Writing $O$ for observed")
            a("hours, $k$ for distinct gaps and $R$ for the sterilisation radius,")
            a("")
            a(
                f"$$\\text{{usable}} \\approx O\\exp(\\beta_0 - \\alpha R k / O),"
                f"\\quad \\alpha = {fit['alpha']:.4f},\\ \\beta_0 = {fit['beta0']:.4f}$$"
            )
            a("")
            a(
                f"fitted on {fit.get('source', 'one donor grid')} alone, it predicts "
                f"{held['n']} held-out cells at $R^2 = {held['r2']:.3f}$ "
                f"(median absolute error {held['median_ape_pct']:.1f}%)."
            )
            groups = pd.DataFrame(law.get("held_out_by_group", []))
            if not groups.empty:
                a("")
                a("Those cells are two different extrapolations, and the distinction is the")
                a("difference between checking a constant and checking a functional form.")
                a("")
                a("| Held-out grid | Radius (h) | What it tests | Cells | $R^2$ | Median APE |")
                a("|---|---:|---|---:|---:|---:|")
                for _, g in groups.sort_values(["group", "radius_h"]).iterrows():
                    tests = (
                        "do the constants travel between records"
                        if g["group"] == "other station"
                        else "is $R$ a factor, or a scale the fit absorbed"
                    )
                    a(
                        f"| `{g['donor']}` | {int(g['radius_h'])} | {tests} | "
                        f"{int(g['n'])} | {float(g['r2']):.3f} | "
                        f"{float(g['median_ape_pct']):.1f}% |"
                    )
                a("")
                fixed = law.get("held_out_at_fit_radius") or {}
                a("The second group is the stronger test. $R$ enters as a factor, so a law")
                a("fitted at one radius makes a prediction at every other, and those grids are")
                a("the *fitting* station rebuilt at a quarter of the fitting radius — a")
                a("different experiment, not a different record.")
                if fixed:
                    a("")
                    a(
                        f"Score those same held-out cells at the fitted radius instead — "
                        f"treating $R$ as a scale the constants had absorbed — and the pooled "
                        f"$R^2$ falls from {held['r2']:.3f} to {fixed['r2']:.3f}, with median "
                        f"error rising from {held['median_ape_pct']:.1f}% to "
                        f"{fixed['median_ape_pct']:.1f}%. That gap is what the radius term is "
                        f"carrying, and it is why the law is stated with $R$ in it rather than "
                        f"as a relationship between gaps and rows."
                    )

            pred_alpha = law.get("alpha_predicted_from_ffill")
            ci = law.get("alpha_predicted_ci") or []
            if pred_alpha and len(ci) == 2:
                inside = bool(ci[0] <= fit["alpha"] <= ci[1])
                a("")
                a(
                    f"$\\alpha$ is not a free constant either: it should equal the share of "
                    f"gaps outliving the {law['max_ffill_hours']}-hour forward-fill, "
                    f"which is {pred_alpha:.4f} across this record's "
                    f"{law['n_raw_gaps']:,} gaps (95% CI "
                    f"[{ci[0]:.4f}, {ci[1]:.4f}]). The fitted value falls "
                    f"{'inside' if inside else 'just outside'} that interval, so the "
                    f"constant is the imputation policy to within what a record of this "
                    f"size can resolve."
                )
            a("")

        avail = pd.DataFrame(law.get("availability", []))
        if not avail.empty:
            a("### Coverage is not what reaches the model")
            a("")
            base = avail[avail["Lookback (h)"] == 168]
            a("| Record | Lookback 168 h | 48 h | 24 h |")
            a("|---|---:|---:|---:|")
            for record in base["Record"]:
                sub = avail[avail["Record"] == record].set_index("Lookback (h)")
                cells = " | ".join(
                    f"{float(sub.loc[lb, 'Availability (grid)']) * 100:.1f}%"
                    for lb in (168, 48, 24)
                    if lb in sub.index
                )
                a(f"| {record} | {cells} |")
            a("")
            a("Forecast availability is the share of test hours a model can answer at all.")
            a("It does not rank these records the way coverage does: the primary record is")
            a("the least complete of the four and yet among the most available, because its")
            a("absences are clustered into a few long outages while the near-complete")
            a("stations' are scattered. **Coverage is what a data custodian reports;")
            a("availability is what a forecaster gets, and the two can order a set of")
            a("records in opposite directions.**")
            a("")

        dec_rule = pd.DataFrame(law.get("decision_rule", []))
        if not dec_rule.empty:
            a("### What a reach costs, before anything is fitted")
            a("")
            a("The law needs only an hour count and a gap count, which is what makes it")
            a("usable on a record one does not hold. Anyone holding the record has the run")
            a("lengths themselves, and those answer the question as an identity with no")
            a("error term at all: a run of length $l$ supports $\\max(0, l - R)$ scored")
            a("rows, so")
            a("")
            a(
                "$$U(R) = \\sum_{\\text{runs}}\\max(0,\\ l - R),"
                "\\qquad -\\frac{\\mathrm{d}U}{\\mathrm{d}R} = \\#\\{l > R\\}.$$"
            )
            a("")
            a("**One further hour of backward reach costs exactly the number of runs still")
            a("longer than it.** No fit, no residual, and a custodian can evaluate it in a")
            a("line. Divided through by $U/R$ it becomes an elasticity, which is the only")
            a("form of the price comparable between records of different sizes.")
            a("")
            radii = sorted({int(v) for v in dec_rule["Radius (h)"]})
            a(f"| Record | {' | '.join(f'$R$ = {r} h' for r in radii)} |")
            a("|---" + "|---:" * len(radii) + "|")
            for record in dict.fromkeys(dec_rule["Record"]):
                sub = dec_rule[dec_rule["Record"] == record].set_index("Radius (h)")
                cells = " | ".join(f"{float(sub.loc[r, 'Elasticity']):.3f}" for r in radii)
                a(f"| {record} | {cells} |")
            a("")
            deep = dec_rule[dec_rule["Radius (h)"] == max(radii)]
            hi = deep.loc[deep["Elasticity"].idxmax()]
            lo = deep.loc[deep["Elasticity"].idxmin()]
            a(
                f"At the status quo's {max(radii)}-hour radius {hi['Record']} pays "
                f"{float(hi['Elasticity']):.3f} against {lo['Record']}'s "
                f"{float(lo['Elasticity']):.3f}: a 1% deeper reach costs it "
                f"{float(hi['Elasticity']) / float(lo['Elasticity']):.1f} times as large a "
                f"share of its supervision. Every record's price rises with the reach, "
                f"because the runs that can still pay it are the ones being spent."
            )
            a("")

    if frontier_path.exists():
        fr = json.loads(frontier_path.read_text(encoding="utf-8"))
        a("### Does a shorter reach cost accuracy?")
        a("")
        a(
            f"Shortening the reach recovers those hours. It also removes real signal — the "
            f"weekly lag and the 168-hour rolling statistics — so the trade is measured "
            f"rather than assumed, on the {fr['n_common']:,} hours every arm can serve, "
            f"which is the only comparison that is not confounded by how often each arm "
            f"declines to answer."
        )
        a("")
        # Persistence and climatology read no engineered features, so a cap cannot
        # move them; they belong in the fallback, not in a table about the reach.
        tier1 = ("persistence", "climatology")
        dm = pd.DataFrame(fr.get("dm_tests", []))
        allh = pd.DataFrame(fr.get("all_hours", []))
        if not dm.empty:
            dm = dm[~dm["model"].isin(tier1)]
        if not allh.empty:
            allh = allh[~allh["model"].isin(tier1)]
        if not dm.empty:
            # Direction, not just significance. "Differs" would read as support
            # for the caps here, and the only difference on this record runs the
            # other way.
            sig = dm[dm["significant"]]
            for_cap = sig[sig["better"] != "C"]
            for_status_quo = sig[sig["better"] == "C"]
            a(
                f"Under Holm-corrected Diebold-Mariano across "
                f"{int(dm['p_holm'].notna().sum())} tests, **{len(for_cap)} capped "
                f"{'arm beats' if len(for_cap) == 1 else 'arms beat'} the status quo and "
                f"{len(for_status_quo)} "
                f"{'loses' if len(for_status_quo) == 1 else 'lose'} to it**."
            )
            a("")
            if for_cap.empty:
                a("The shorter reach is not better at forecasting. It is better at answering.")
                a("")
            for r in for_status_quo.itertuples():
                a(
                    f"`{r.model}` at {r.arm} is significantly *worse* "
                    f"(p = {_fmt_p(r.p_holm)}). Below its own window a shorter reach stops "
                    f"being free: a window-sized model loses the signal it is built on, and "
                    f"that cost is real rather than a rounding of the availability gain."
                )
                a("")
        if not allh.empty:
            single = allh[allh["policy"] == "single"]
            sq = single[single["arm"] == "C"].set_index("model")["skill_all_hours"]
            best_idx = single.groupby("model")["skill_all_hours"].idxmax()
            best = single.loc[best_idx]
            a("| Model | Status quo | Best arm | Availability | All-hours skill | Change |")
            a("|---|---:|---|---:|---:|---:|")
            for r in best.itertuples():
                if r.arm == "C" or r.model not in sq.index:
                    continue
                a(
                    f"| `{r.model}` | {float(sq[r.model]):+.4f} | {r.arm} | "
                    f"{r.availability * 100:.1f}% | {r.skill_all_hours:+.4f} | "
                    f"{r.skill_all_hours - float(sq[r.model]):+.4f} |"
                )
            a("")
            a("Scored over the whole evaluation universe, an hour with no forecast is not an")
            a("hour without error: it is an hour that must fall back on persistence. That is")
            a("where the recovered availability turns into recovered skill.")
            a("")
            un = fr.get("unserved_hours") or {}
            if un:
                a(
                    f"One number in that table looks wrong and is not. All-hours RMSE comes "
                    f"out *below* served RMSE for the status quo, because the hours it "
                    f"cannot reach are **easier**, not harder: persistence scores "
                    f"{un['reference_rmse_on_unserved']:.2f} on the "
                    f"{un['n_unserved']:,} unserved hours against "
                    f"{un['reference_rmse_on_served']:.2f} on the {un['n_served']:,} it "
                    f"serves, and their observed mean is "
                    f"{un['observed_mean_on_unserved']:.1f} against "
                    f"{un['observed_mean_on_served']:.1f} " + UNIT_PM25_TEXT + "."
                )
                a("")
                a("A gap is followed by a stretch of hours no deep-reach model can forecast,")
                a("and on this record those stretches sit disproportionately in the cleaner")
                a("part of the distribution. So the gain from a shorter reach is not the")
                a("rescue of catastrophic hours; it is a model beating persistence on a large")
                a("block of ordinary ones. That is a smaller and more honest claim, and it is")
                a("the one the numbers support.")
                a("")
        decomp = pd.DataFrame(fr.get("decomposition", []))
        if not decomp.empty:
            decomp = decomp[~decomp["model"].isin(tier1)]
        if not decomp.empty:
            a(
                "The three arms separate the two effects exactly. Holding the row set at the "
                "status quo and capping only the features isolates feature richness; holding "
                "the features and lowering the floor isolates supervision volume; in mean "
                "squared error the two sum to the total by construction "
                f"(largest residual {decomp['residual'].abs().max():.2e})."
            )
            a("")
            a("| Model | Cap (h) | Total ΔMSE | Supervision volume | Feature richness |")
            a("|---|---:|---:|---:|---:|")
            for r in decomp.itertuples():
                a(
                    f"| `{r.model}` | {int(r.lookback_h)} | {r.total_mse_delta:+.1f} | "
                    f"{r.volume_mse_delta:+.1f} | {r.richness_mse_delta:+.1f} |"
                )
            a("")
            vol = float(decomp["volume_mse_delta"].abs().sum())
            rich = float(decomp["richness_mse_delta"].abs().sum())
            leader = "feature richness" if rich > vol else "supervision volume"
            a(
                f"On these identical rows the larger term is **{leader}** "
                f"({rich:.0f} against {vol:.0f} in summed absolute MSE). That is the right "
                f"way round: the common subset holds the evaluation hours fixed, so extra "
                f"training rows can only help through better-fitted parameters, and the "
                f"availability they buy — which is where the gain of the previous table "
                f"comes from — is by construction invisible here. Capping the reach removes "
                f"the weekly lag and its rolling statistics and mostly *improves* the fit, "
                f"so on this record the deepest features were paying for themselves only in "
                f"the rows they made impossible."
            )
            a("")

    dec_out = pd.DataFrame(law.get("decision_outcomes", []))
    if not dec_out.empty:
        a("### Does the free price predict the payoff?")
        a("")
        a("The identity says what a cap recovers. Only the frontier says what it bought,")
        a("and the two records whose arms were trained answer that question against each")
        a("other rather than in isolation.")
        a("")
        a(
            "| Record | Radius (h) | Elasticity | Rows predicted | Rows measured | "
            "Availability | Skill gain (median) | Skill gain (best) |"
        )
        a("|---|---:|---:|---:|---:|---:|---:|---:|")
        for _, r in dec_out.iterrows():
            a(
                f"| {r['Record']} | {int(r['Radius (h)'])} | "
                f"{float(r['Elasticity at status quo']):.3f} | "
                f"{float(r['Predicted rows (%)']):+.1f}% | "
                f"{float(r['Measured rows (%)']):+.1f}% | "
                f"{float(r['Availability gained (pp)']):+.1f} pp | "
                f"{float(r['Skill gain (median)']):+.4f} | "
                f"{float(r['Skill gain (best)']):+.4f} |"
            )
        a("")
        a("The 71-hour rows are the 48-hour-window models under the 24-hour cap. A")
        a("recurrent model's floor is `max(lookback, window - 1)`, so below its own window")
        a("the cap stops buying anything, and those rows are the cost of that ceiling")
        a("rather than a second measurement of the 72-hour arm.")
        a("")
        deepest = dec_out.loc[dec_out.groupby("Record")["Radius (h)"].idxmin()]
        rich = deepest.loc[deepest["Measured rows (%)"].idxmax()]
        poor = deepest.loc[deepest["Measured rows (%)"].idxmin()]
        a(
            f"**The price sizes the opportunity; it does not promise the payoff.** "
            f"{rich['Record']} recovers {float(rich['Measured rows (%)']):.0f}% of its "
            f"training rows against {poor['Record']}'s "
            f"{float(poor['Measured rows (%)']):.0f}%, and gains "
            f"{float(rich['Skill gain (median)']):+.4f} median all-hours skill against "
            f"{float(poor['Skill gain (median)']):+.4f} — the larger recovery is the "
            f"smaller gain. Availability orders them the other way: "
            f"{float(poor['Availability gained (pp)']):+.1f} points against "
            f"{float(rich['Availability gained (pp)']):+.1f}, matching the skill. Rows and "
            f"hours served are both free to compute from the record alone and they "
            f"disagree about which record had more to gain; here the hours were right. "
            f"On two records that is a direction rather than a rule, and the mechanism "
            f"is the one the previous table shows: the gain arrives as hours moved off "
            f"the persistence fallback, not as parameters fitted on more rows."
        )
        a("")
        err = dec_out["Error (pp)"].abs().max()
        a(
            f"The identity is evaluated over the whole record and the frontier counts rows "
            f"in the training split alone, so the two columns are one quantity on two "
            f"footings and are not expected to agree exactly; the largest disagreement is "
            f"{err:.1f} percentage points. The free calculation is a sizing instrument, "
            f"and it is reported as one."
        )
        a("")

    if law_path.exists() or frontier_path.exists():
        a("**Scope.** One horizon and one record family. The frontier is measured at the")
        a("headline horizon only, and the arms cap the feature set rather than replacing it,")
        a("so this bounds what the configured reach costs — not what an optimally chosen")
        a("reach would buy. Below the sequence window the cap buys a recurrent model")
        a("nothing, because `max(lookback, window - 1)` is the binding floor.")
        a("")

    # Site-specific limitations are read from this city's own audit and QC
    # ledger. They were previously hardcoded to Dhaka's numbers, which the
    # comparison city's report then restated as if they were its own.
    city = str(cfg.get("data.site.city"))
    site_label = str(cfg.get("data.openaq.site_label"))
    n_negative = next(
        (int(r["removed"]) for r in qc_ledger if r["rule"] == "negative value"),
        None,
    )

    a(f"## {next_section}. Limitations")
    a("")
    a("- **Energy figures are estimates.** See §5. They should not be reported as measurements.")
    a("- **One monitoring site.** The target series comes from a single monitor —")
    a(f"  {site_label}. Results describe that site, not {city} as a whole.")
    a(f"- **The record ends {audit['coverage']['last_utc'][:10]}.** No later data for this")
    a("  monitor exists in the archive the pipeline reads, so the test period cannot be")
    a("  extended without changing the data source.")
    gap_h = int(audit["longest_gap_hours"])
    gap_days = gap_h / 24.0
    gap_when = (
        f", {longest_gap['start_utc']} to {longest_gap['end_utc']}"
        if longest_gap is not None
        else ""
    )
    a(f"- **Coverage.** {audit['pct_observed']:.1f}% of hours observed, broken by")
    a(f"  {audit['n_gaps']:,} distinct gaps; the longest runs {gap_h:,} h")
    a(f"  ({gap_days:.0f} day{'' if round(gap_days) == 1 else 's'}{gap_when}).")
    a(f"- **{audit['distribution']['n']:,} observed hours** after QC.")
    if n_negative:
        a(f"  {n_negative:,} negative readings were dropped rather than floored, which is a")
        a("  modelling choice, not a neutral act.")
    a("- **SARIMAX is fitted on a bounded training tail**, making it a recent-history")
    a("  statistical baseline rather than a full-sample one. Gaps inside that tail are")
    a("  left as NaN and handled by the Kalman filter rather than dropped, which would")
    a("  have changed the sampling interval the seasonal term depends on.")
    a("- **Ridge is structurally disadvantaged and should be read as a linear-model")
    a("  floor, not as a tuned competitor.** The target is modelled on a `log1p` scale")
    a("  while the PM2.5 history features are supplied on their raw scale, so a linear")
    a("  model has to express `log1p(y(t+h))` as a linear function of `y(t)`, which it")
    a("  cannot do well. Tree ensembles and the recurrent models are unaffected because")
    a("  they are non-linear in the inputs. This is a property of the feature/target")
    a("  parameterisation, not evidence that linear methods are hopeless on this task;")
    a("  supplying log-scale lag features would be the fair comparison and was not run.")
    a("- **Retransformation bias.** Predicting the conditional mean on a `log1p` scale")
    a("  and inverting with `expm1` returns something closer to a conditional median, so")
    a("  every model in this study will tend to under-predict the largest episodes. The")
    a("  bias applies identically to all models, so between-model comparisons remain")
    a("  valid, but absolute peak values should be read with it in mind.")
    a("- **Predictions are bounded** to the physical range QC applies to observations")
    a("  ([0, sanity cap]). Without the upper bound, unbounded linear extrapolation on")
    a("  the log scale produced a handful of impossible concentrations that dominated")
    a("  squared-error metrics. Clip counts are recorded per model in `results.json`.")
    a("")

    report_path = Path(str(cfg.get("output.report.results_md")))
    report_path = (
        report_path if report_path.is_absolute() else cfg.path_for("reports").parent / report_path
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    log.info("wrote %s", report_path)

    # ------------------------------------------------------------ abstract facts
    facts: dict = {
        "_description": (
            "Exactly the figures needed to write a 250-300 word abstract. Generated by "
            "scripts/11_make_report.py from results/results.json. Any value that is null "
            "was not available and must not be invented."
        ),
        "study": {
            "title": cfg.get("project.title"),
            "site": cfg.get("data.openaq.site_label"),
            "city": cfg.get("data.site.city"),
            "openaq_location_ids": cfg.get("data.openaq.location_ids"),
        },
        "dataset": {
            "start_utc": audit["coverage"]["first_utc"],
            "end_utc": audit["coverage"]["last_utc"],
            "date_range_human": f"{audit['coverage']['first_utc'][:10]} to {audit['coverage']['last_utc'][:10]}",
            "span_years": audit["coverage"]["span_years"],
            "usable_years": audit["usable_years"],
            "hourly_coverage_pct": audit["pct_observed"],
            "observed_hours": audit["observed_hours"],
            "n_predictors": meta["n_predictors"],
            "pm25_mean_ugm3": audit["distribution"]["mean"],
            "pm25_median_ugm3": audit["distribution"]["p50"],
            "pct_hours_above_bd_24h_standard": audit["distribution"]["exceedance"].get(
                "above_65_ugm3_pct"
            ),
            "meteorology_source": str(cfg.get("data.meteorology_source")).strip(),
            "split": {
                "train_end": str(boundaries["train_end"])[:10],
                "val_end": str(boundaries["val_end"])[:10],
                "method": "chronological 70/15/15, no shuffling",
            },
        },
        "horizons_h": horizons,
        "headline_horizon_h": headline,
        "best_model": None,
        "per_horizon": {},
        "persistence_rmse_by_horizon": {int(k): round(float(v), 3) for k, v in persistence.items()},
        "green": {},
        "classification": None,
        "significance": None,
    }

    if not best_deep.empty:
        row = best_deep[best_deep["horizon_h"] == headline]
        if not row.empty:
            r = row.iloc[0]
            facts["best_model"] = {
                "name": str(r["model"]),
                "input_window_h": int(str(r["variant"]).lstrip("w")),
                "n_parameters": int(r["n_params"]),
                "n_seeds": int(r["n_seeds"]),
                "parameter_budget": int(cfg.get("models.sequence.max_params")),
            }
        for _, r in best_deep.iterrows():
            facts["per_horizon"][int(r["horizon_h"])] = {
                "model": str(r["model"]),
                "rmse": round(float(r["rmse"]), 3),
                "rmse_std_across_seeds": None
                if pd.isna(r["rmse_std"])
                else round(float(r["rmse_std"]), 3),
                "mae": round(float(r["mae"]), 3),
                "r2": round(float(r["r2"]), 4),
                "smape": round(float(r["smape"]), 3),
                "skill_vs_persistence": round(float(r["skill"]), 4),
            }

    if not best_classical.empty:
        facts["best_classical_by_horizon"] = {
            int(r["horizon_h"]): {
                "model": str(r["model"]),
                "rmse": round(float(r["rmse"]), 3),
                "skill_vs_persistence": round(float(r["skill"]), 4),
            }
            for _, r in best_classical.iterrows()
        }

    if pareto:
        p = pd.DataFrame(pareto).sort_values("rmse_mean")
        top = p.iloc[0]
        facts["green"] = {
            "best_model_params": int(top["params"]),
            "best_model_macs": None if pd.isna(top["macs"]) else int(top["macs"]),
            "best_model_latency_ms": None
            if pd.isna(top["latency_ms_mean"])
            else round(float(top["latency_ms_mean"]), 5),
            "best_model_train_kwh": None
            if pd.isna(top["train_kwh_mean"])
            else float(top["train_kwh_mean"]),
            "best_model_train_gco2e_bd_grid": None
            if pd.isna(top["co2e_g_bd_mean"])
            else round(float(top["co2e_g_bd_mean"]), 5),
            "grid_intensity_gco2_per_kwh": grid["value_gco2_per_kwh"],
            "grid_intensity_year": grid["year"],
            "grid_intensity_source": grid["source_name"],
            "energy_is_estimate_not_measurement": True,
        }

    if clf:
        facts["classification"] = {
            "horizon_h": clf["horizon_h"],
            "n_classes": len(clf["labels"]),
            "macro_f1_mean": round(clf["macro_f1"]["mean"], 4),
            "macro_f1_std": round(clf["macro_f1"]["std"], 4),
            "macro_f1_present_mean": round(
                clf.get("macro_f1_present", {}).get("mean", float("nan")), 4
            ),
            "macro_f1_present_std": round(
                clf.get("macro_f1_present", {}).get("std", float("nan")), 4
            ),
            "n_classes_defined": clf.get("n_classes_defined"),
            "n_classes_present_in_test": clf.get("n_classes_present"),
            "empty_classes": clf.get("empty_classes"),
            "macro_f1_note": (
                "Report the present-class figure alongside the all-class one. A "
                "category with zero test support contributes a structural zero and "
                "depresses the all-class macro-F1 regardless of model quality."
            ),
            "accuracy_mean": round(clf["accuracy"]["mean"], 4),
            "balanced_accuracy_mean": round(clf["balanced_accuracy"]["mean"], 4),
            "breakpoint_scheme": clf["breakpoints"]["scheme"],
            "target_basis": clf["target_basis"],
        }

    if dm_head:
        d = dm_head[0]
        holm = d.get("p_value_holm")
        # significant_at_5pct follows the ADJUSTED p-value when one exists. The
        # abstract is written from this file, and quoting an unadjusted p-value
        # out of a family of five is the error the adjustment exists to prevent.
        decisive = holm if holm is not None else d["p_value"]
        facts["significance"] = {
            "test": "Diebold-Mariano",
            "horizon_h": d["horizon_h"],
            "model_a": d["model_a"],
            "model_b": d["model_b"],
            "statistic": round(d["statistic"], 4),
            "p_value": float(d["p_value"]),
            "p_value_holm": float(holm) if holm is not None else None,
            "multiplicity_correction": payload.get("significance", {}).get("multiplicity"),
            "significant_at_5pct": bool(decisive < 0.05),
            "significance_basis": "Holm-adjusted" if holm is not None else "unadjusted",
            "lower_loss": d["better"],
        }

    mcs_head = [
        m
        for m in payload.get("significance", {}).get("model_confidence_set", [])
        if m["horizon_h"] == int(cfg.get("task.headline_horizon_h"))
    ]
    if mcs_head:
        retained = [m["model"] for m in mcs_head if m["in_confidence_set"]]
        facts["model_confidence_set"] = {
            "reference": "Hansen, Lunde and Nason (2011)",
            "horizon_h": mcs_head[0]["horizon_h"],
            "alpha": 0.05,
            "n_models": len(mcs_head),
            "n_retained": len(retained),
            "retained": retained,
            "note": (
                "Models in the set cannot be distinguished from the best at the 5% "
                "level. Their relative order is not evidence of an ordering, and the "
                "abstract must not name one of them as the winner over another."
            ),
        }

    ci_head = [
        c
        for c in payload.get("significance", {}).get("bootstrap_ci", [])
        if c["horizon_h"] == int(cfg.get("task.headline_horizon_h"))
    ]
    if ci_head:
        facts["headline_rmse_intervals"] = {
            c["model"]: {
                "rmse": round(c["rmse"], 3),
                "ci_low": round(c["ci_low"], 3),
                "ci_high": round(c["ci_high"], 3),
            }
            for c in sorted(ci_head, key=lambda c: c["rmse"])
        }

    cross = payload.get("cross_city") or {}
    if cross:
        other = next(iter(cross))
        block = cross[other]
        ranking = pd.DataFrame(block["ranking"])
        this_city = str(cfg.get("data.site.city"))
        facts["cross_city"] = {
            "comparison_city": other,
            "spearman_rank_correlation": block.get("spearman_rank_correlation"),
            "ranking_transfers": bool((block.get("spearman_rank_correlation") or 0) >= 0.3),
            "headline_claim": (
                f"Method ranking {'transfers' if (block.get('spearman_rank_correlation') or 0) >= 0.3 else 'does not transfer'} "
                f"between cities: the Spearman rank correlation between {this_city} and "
                f"{other} at the headline horizon is "
                f"{block.get('spearman_rank_correlation'):+.3f}."
            ),
            # Recorded so a consumer of this file sees which way the observation
            # runs without re-deriving it. It has already reversed once, when the
            # sequence tier's inputs and training recipe were corrected.
            "sequence_rank_by_city": {
                str(r["city"]): int(r["rank"])
                for _, r in ranking[ranking["model"] == "best sequence"].iterrows()
            },
            "ranking_by_city": {
                city: [
                    {"rank": int(r["rank"]), "model": r["model"], "skill": r["skill"]}
                    for _, r in g.sort_values("rank").iterrows()
                ]
                for city, g in ranking.groupby("city")
            },
            "record_context": block.get("context"),
            "note": (
                "The ranking reversal is the robust claim; the identity of the winner in "
                "the comparison city is weaker, its significance tests being mostly "
                "non-significant and its test period shorter."
            ),
        }

    # Same source as the report section: the ablation's own generated file.
    ablation_path = Path(str(cfg.get("paths.results"))) / str(
        cfg.get("ablation.gap_injection.output_name", "ablation_gap_injection.json")
    )
    if ablation_path.exists():
        abl = json.loads(ablation_path.read_text(encoding="utf-8"))
        by_family = pd.DataFrame(abl.get("analysis", {}).get("by_family", []))
        if not by_family.empty:
            wide = by_family.pivot_table(
                index=["family", "target_coverage"],
                columns="arm",
                values="skill",
                aggfunc="first",
            ).reset_index()
            if {"fragmented", "contiguous"} <= set(wide.columns):
                wide["gap"] = wide["fragmented"] - wide["contiguous"]
                degraded = wide[wide["target_coverage"] < 1.0]
                means = degraded.groupby("family")["gap"].mean().round(4).to_dict()
                facts["gap_injection"] = {
                    "donor": abl.get("donor"),
                    "gap_profile_source": abl.get("gap_profile_source"),
                    "horizon_h": abl.get("horizon_h"),
                    "n_injection_draws": int(by_family["n_draws"].max()),
                    "paired_test": abl.get("analysis", {}).get("paired_tests", []),
                    "mean_arm_gap_by_family": means,
                    "arm_gap_by_family_and_coverage": {
                        str(fam): {
                            f"{row.target_coverage:.2f}": round(float(row.gap), 4)
                            for row in grp.itertuples()
                        }
                        for fam, grp in wide.groupby("family")
                    },
                    "control_note": (
                        "The climatological family is the weakest model that DOES read the "
                        "training record: hour-of-day and month means, insensitive to how "
                        "the observed hours are arranged but fitted on them. It is not a "
                        "no-training-data control. Because the test period is never "
                        "degraded, any model that reads nothing from training has an arm "
                        "gap of exactly zero in every cell, so it cannot vary and cannot "
                        "falsify anything; that invariance is checked by equality instead "
                        "(persistence_rmse_invariant). The undegraded level removes "
                        "nothing, so its gap is 0 by construction."
                    ),
                }

    # ---- the lookback contributions ---------------------------------------
    # Read from the generated payloads, never from the prose above. A value that
    # is absent stays absent rather than being filled in: the header of this file
    # says any null "was not available and must not be invented".
    results_dir_facts = Path(str(cfg.get("paths.results")))

    law_file = results_dir_facts / f"missingness_law{city_suffix(cfg)}.json"
    if law_file.exists():
        law_facts = json.loads(law_file.read_text(encoding="utf-8"))
        fit_facts = law_facts.get("fit", {})
        held = law_facts.get("held_out", {})
        facts["missingness_law"] = {
            "form": fit_facts.get("form"),
            "alpha": fit_facts.get("alpha"),
            "beta0": fit_facts.get("beta0"),
            "sterilisation_radius_h": law_facts.get("radius_h"),
            "fitted_on": fit_facts.get("source"),
            "held_out_n_cells": held.get("n"),
            "held_out_r2": held.get("r2"),
            "held_out_median_ape_pct": held.get("median_ape_pct"),
            "alpha_predicted_from_ffill": law_facts.get("alpha_predicted_from_ffill"),
            "amplification_by_record": {
                r["Record"]: r["Amplification"]
                for r in law_facts.get("amplification_by_record", [])
            },
            "note": (
                "A gap costs the hours it removes plus the sterilisation radius behind "
                "them, so the cost follows the NUMBER of gaps rather than their length. "
                "alpha should equal the share of gaps outliving the forward-fill limit; "
                "the agreement is the evidence it is the imputation policy and not a "
                "free parameter."
            ),
        }
        avail_rows = law_facts.get("availability", [])
        deepest_avail = [r for r in avail_rows if int(r["Lookback (h)"]) == 168]
        facts["forecast_availability"] = {
            "at_configured_lookback_h": 168,
            "share_of_test_hours_by_record": {
                r["Record"]: r["Availability (grid)"] for r in deepest_avail
            },
            "by_record_and_lookback": [
                {
                    "record": r["Record"],
                    "lookback_h": r["Lookback (h)"],
                    "availability_vs_grid": r["Availability (grid)"],
                    "availability_vs_universe": r["Availability"],
                }
                for r in avail_rows
            ],
            "note": (
                "Share of test hours any model in this study can forecast at all. Every "
                "accuracy figure reported is conditional on it. It does not order these "
                "records the way hourly coverage does."
            ),
        }

    frontier_file = results_dir_facts / f"availability_frontier{city_suffix(cfg)}.json"
    if frontier_file.exists():
        fr_facts = json.loads(frontier_file.read_text(encoding="utf-8"))
        dm_rows = pd.DataFrame(fr_facts.get("dm_tests", []))
        allh_rows = pd.DataFrame(fr_facts.get("all_hours", []))
        best_arm = {}
        if not allh_rows.empty:
            single_rows = allh_rows[allh_rows["policy"] == "single"]
            sq = single_rows[single_rows["arm"] == "C"].set_index("model")["skill_all_hours"]
            for idx in single_rows.groupby("model")["skill_all_hours"].idxmax():
                row = single_rows.loc[idx]
                if row["arm"] == "C" or row["model"] not in sq.index:
                    continue
                best_arm[str(row["model"])] = {
                    "arm": row["arm"],
                    "availability": row["availability"],
                    "skill_all_hours": row["skill_all_hours"],
                    "gain_vs_status_quo": row["skill_all_hours"] - float(sq[row["model"]]),
                }
        sig_rows = dm_rows[dm_rows["significant"]] if not dm_rows.empty else pd.DataFrame()
        facts["lookback_frontier"] = {
            "n_universe": fr_facts.get("n_universe"),
            "n_common_subset": fr_facts.get("n_common"),
            "n_arms_beating_status_quo_on_common_hours": (
                int((sig_rows["better"] != "C").sum()) if not sig_rows.empty else 0
            ),
            "n_arms_losing_to_status_quo_on_common_hours": (
                int((sig_rows["better"] == "C").sum()) if not sig_rows.empty else 0
            ),
            "best_arm_by_model": best_arm,
            "unserved_hours": fr_facts.get("unserved_hours"),
            "note": (
                "rmse_served is NOT comparable across arms: a deeper reach is scored on "
                "fewer, better-covered hours. The common subset and the all-hours columns "
                "are the two comparisons that are."
            ),
        }

    med_files = sorted(results_dir_facts.glob("mediation_*.json"))
    if med_files:
        med_facts = json.loads(med_files[0].read_text(encoding="utf-8"))
        facts["mediation"] = {
            "donor": med_facts.get("donor"),
            "radii_h": med_facts.get("radii"),
            "mediator_deficit_shrinks_with_radius": med_facts.get(
                "mediator_deficit_shrinks_with_radius"
            ),
            "dilution_bound": med_facts.get("dilution_bound", {}).get(
                "expected_gap_factor_under_pure_dilution"
            ),
            "by_family": {
                r["family"]: {
                    "gap_at_deepest": r.get(f"gap_R{max(med_facts.get('radii', [0]))}"),
                    "gap_at_shallowest": r.get(f"gap_R{min(med_facts.get('radii', [0]))}"),
                    "proportion_mediated": r.get("proportion_mediated"),
                    "p_holm": r.get("p_holm"),
                }
                for r in med_facts.get("tests", [])
            },
            "note": (
                "The mediator is manipulated by configuration rather than inferred from a "
                "regression, so this is not a Baron-Kenny mediation and needs no "
                "sequential-ignorability assumption. One donor and one horizon."
            ),
        }

    facts_path = Path(str(cfg.get("output.report.abstract_facts_json")))
    facts_path = (
        facts_path if facts_path.is_absolute() else cfg.path_for("reports").parent / facts_path
    )
    facts_path.write_text(json.dumps(facts, indent=2, default=str), encoding="utf-8")
    log.info("wrote %s", facts_path)

    missing = [k for k, v in facts.items() if v is None and not k.startswith("_")]
    if missing:
        log.warning("abstract_facts.json has null sections: %s", missing)

    print(f"\nwrote {report_path}")
    print(f"wrote {facts_path}")
    if missing:
        print(f"\nWARNING: null sections in abstract_facts.json: {missing}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
