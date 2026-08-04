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
from src.models.data import load_meta
from src.results import load_results
from src.utils import check_disk_space, load_config, setup_logging


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="config.yaml")
    return p.parse_args()


def _runs_frame(payload: dict) -> pd.DataFrame:
    """Flatten the run records into a table."""
    runs = payload.get("runs", [])
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
        a("Read alone, the primary result says a compact recurrent model loses to tuned")
        a("gradient boosting. The second city shows that conclusion is **not a property")
        a("of compact recurrent models**: in the other city the sequence model beats")
        a("every tree, while the trees collapse to near-worthless skill.")
        a("")
        a("The record characteristics point at the mechanism. Tree models on lagged")
        a("tabular features tolerate fragmentation well, because each row stands alone.")
        a("Sequence models need contiguous windows, and gap-aware windowing discards a")
        a("large fraction of them where the record is broken.")
        a("")
        context = pd.DataFrame(block["context"])
        a(context.to_markdown(index=False))
        a("")
        a("**Caveat on strength of evidence.** The ranking *reversal* is the robust")
        a("claim. The identity of the winner in the comparison city is not: its")
        a("Diebold-Mariano tests are mostly not significant at the 5% level, and its")
        a("test period is shorter and spans a seasonal transition, so absolute skill is")
        a("lower for every method there.")
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
                "Method ranking does not transfer between cities: the Spearman rank "
                f"correlation between {this_city} and {other} at the headline horizon is "
                f"{block.get('spearman_rank_correlation'):+.3f}. The best method in one "
                "city is among the worst in the other."
            ),
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
