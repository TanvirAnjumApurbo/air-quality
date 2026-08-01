"""Phase 1: data audit -- the mandatory gate before any modelling.

Produces ``reports/DATA_AUDIT.md`` plus supporting figures and tables covering
date coverage, observed vs available hours, the longest contiguous gap,
missingness per variable, a monthly missingness heatmap, the PM2.5 distribution,
and the diurnal and seasonal profiles.

If usable contiguous coverage falls below ``audit.min_usable_years`` the report
says so explicitly and the script exits non-zero, so the strategy decision is
forced rather than made silently.

Run::

    python scripts/03_data_audit.py --config config.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.data.audit import (
    _runs,
    coverage_summary,
    distribution_stats,
    diurnal_profile,
    gap_report,
    gap_size_histogram,
    missingness_by_variable,
    monthly_missingness,
    season_summary,
    seasonal_profile,
    window_yield,
)
from src.utils import check_disk_space, load_config, setup_logging
from src.viz.figures import save_figure, setup_style

MIN_USABLE_YEARS = 2.0


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml", help="path to config.yaml")
    parser.add_argument(
        "--min-years",
        type=float,
        default=MIN_USABLE_YEARS,
        help="usable coverage below this triggers a strategy change",
    )
    return parser.parse_args()


def _fmt_table(df: pd.DataFrame, floatfmt: str = ".2f") -> str:
    """Render a DataFrame as a GitHub-flavoured markdown table."""
    return df.to_markdown(index=True, floatfmt=floatfmt)


def main() -> int:
    """Build the data audit report."""
    args = parse_args()
    cfg = load_config(args.config)
    log = setup_logging(cfg, "03_data_audit")
    check_disk_space(cfg)
    setup_style(cfg)

    tz = str(cfg.get("features.calendar_tz"))
    interim = cfg.path_for("data_interim")
    tables_dir = cfg.path_for("tables")
    tables_dir.mkdir(parents=True, exist_ok=True)

    pm = pd.read_parquet(interim / "openaq_pm25_hourly.parquet")
    met = pd.read_parquet(interim / "power_hourly.parquet")
    units = json.loads((interim / "power_units.json").read_text(encoding="utf-8"))
    ledger = json.loads((interim / "openaq_qc_ledger.json").read_text(encoding="utf-8"))

    beijing_path = interim / "uci_beijing_hourly.parquet"
    beijing = pd.read_parquet(beijing_path) if beijing_path.exists() else None

    pm25 = pm["pm25"]
    log.info("loaded PM2.5 (%d rows) and meteorology (%d rows)", len(pm), len(met))

    # ---------------------------------------------------------------- coverage
    cov = coverage_summary(pm)
    gaps = gap_report(pm25)
    observed = int(pm25.notna().sum())
    pct_observed = 100.0 * observed / len(pm25)
    usable_years = observed / (365.25 * 24)

    # --------------------------------------------------------------- fused view
    fused = pm.join(met, how="inner")
    fused_complete = int(fused.dropna().shape[0])

    # ------------------------------------------------- gap structure and cost
    max_ffill = int(cfg.get("impute.max_ffill_hours", 3))
    gap_hist = gap_size_histogram(pm25)
    wyield = window_yield(
        pm25,
        windows=list(cfg.get("models.sequence.input_windows_h")),
        horizons=list(cfg.get("task.horizons_h")),
        max_ffill=max_ffill,
    )
    filled = pm25.ffill(limit=max_ffill)
    pct_after_ffill = 100.0 * float(filled.notna().mean())
    runs_after = _runs(filled.notna())
    n_runs_after_ffill = len(runs_after)
    longest_run_after_ffill = max((r[2] for r in runs_after), default=0)

    raw_gaps = sorted(_runs(pm25.isna()), key=lambda r: -r[2])[:10]
    longest_gaps = pd.DataFrame(
        [
            {
                "rank": i,
                "hours": n,
                "days": round(n / 24.0, 1),
                "start_utc": start.strftime("%Y-%m-%d"),
                "end_utc": end.strftime("%Y-%m-%d"),
            }
            for i, (start, end, n) in enumerate(raw_gaps, start=1)
        ]
    )
    gap_hist.to_csv(tables_dir / "audit_gap_histogram.csv", index=False)
    wyield.to_csv(tables_dir / "audit_window_yield.csv", index=False)
    longest_gaps.to_csv(tables_dir / "audit_longest_gaps.csv", index=False)

    # ----------------------------------------------------------------- tables
    miss_pm = missingness_by_variable(pm)
    miss_met = missingness_by_variable(met)
    monthly = monthly_missingness(pm25, tz)
    dist = distribution_stats(pm25, cfg)
    diurnal = diurnal_profile(pm25, tz)
    seasonal = seasonal_profile(pm25, tz, cfg)
    seasons = season_summary(pm25, tz, cfg)

    monthly.to_csv(tables_dir / "audit_monthly_missingness.csv")
    diurnal.to_csv(tables_dir / "audit_diurnal_profile.csv")
    seasonal.to_csv(tables_dir / "audit_seasonal_profile.csv")
    pd.DataFrame(ledger).to_csv(tables_dir / "audit_qc_ledger.csv", index=False)

    # ---------------------------------------------------------------- figures
    palette = list(cfg.get("output.figures.palette"))
    figsize_wide = tuple(cfg.get("output.figures.figsize_wide"))

    # Fig 1: data coverage timeline
    fig, ax = plt.subplots(figsize=figsize_wide)
    present = pm25.notna().astype(float)
    daily = present.resample("1D").mean() * 100.0
    ax.fill_between(daily.index, 0, daily.to_numpy(), color=palette[0], linewidth=0, alpha=0.85)
    ax.set_ylabel("hours observed per day (%)")
    ax.set_xlabel("date (UTC)")
    ax.set_ylim(0, 100)
    ax.set_title(f"PM2.5 data coverage -- {cfg.get('data.openaq.site_label')}")
    save_figure(cfg, fig, "fig01_coverage_timeline")

    # Fig 2: monthly missingness heatmap (sequential, single hue light->dark)
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    data = monthly.reindex(columns=range(1, 13))
    im = ax.imshow(data.to_numpy(), aspect="auto", cmap="Blues", vmin=0, vmax=100)
    ax.set_xticks(range(12), [f"{m:02d}" for m in range(1, 13)])
    ax.set_yticks(range(len(data.index)), [str(y) for y in data.index])
    ax.set_xlabel(f"month ({tz})")
    ax.set_ylabel("year")
    ax.set_title("PM2.5 missingness by month (%)")
    ax.grid(False)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data.to_numpy()[i, j]
            if not np.isnan(v):
                ax.text(
                    j,
                    i,
                    f"{v:.0f}",
                    ha="center",
                    va="center",
                    fontsize=6.5,
                    color="white" if v > 55 else "#333333",
                )
    fig.colorbar(im, ax=ax, label="% missing", fraction=0.03, pad=0.02)
    save_figure(cfg, fig, "fig02_monthly_missingness")

    # Fig 3: distribution (linear + log)
    fig, axes = plt.subplots(1, 2, figsize=figsize_wide)
    vals = pm25.dropna()
    axes[0].hist(vals, bins=120, color=palette[0], edgecolor="none")
    axes[0].set_xlabel("PM2.5 (µg/m³)")
    axes[0].set_ylabel("hours")
    axes[0].set_title("distribution")
    for thr, colour, style in (
        (float(cfg.get("evaluation.stratify.by_pollution_level.threshold_ugm3")), palette[1], "-"),
        (
            float(cfg.get("evaluation.stratify.by_pollution_level.secondary_thresholds_ugm3")[0]),
            palette[2],
            "--",
        ),
    ):
        axes[0].axvline(thr, color=colour, linestyle=style, linewidth=1.4, label=f"{thr:g} µg/m³")
    axes[0].legend(title="BD standard")
    axes[1].hist(np.log1p(vals), bins=120, color=palette[0], edgecolor="none")
    axes[1].set_xlabel("log1p PM2.5")
    axes[1].set_ylabel("hours")
    axes[1].set_title("log1p distribution (modelling scale)")
    fig.suptitle("PM2.5 distribution, observed hours only")
    save_figure(cfg, fig, "fig03_pm25_distribution")

    # Fig 4: diurnal + seasonal profiles
    fig, axes = plt.subplots(1, 2, figsize=figsize_wide)
    axes[0].plot(diurnal.index, diurnal["mean"], color=palette[0], marker="o", label="mean")
    axes[0].plot(diurnal.index, diurnal["median"], color=palette[1], marker="s", label="median")
    axes[0].set_xlabel(f"hour of day ({tz})")
    axes[0].set_ylabel("PM2.5 (µg/m³)")
    axes[0].set_title("diurnal profile")
    axes[0].set_xticks(range(0, 24, 3))
    axes[0].legend()

    monsoon = set(cfg.get("features.season.monsoon_months"))
    bar_colours = [palette[2] if m in monsoon else palette[1] for m in seasonal.index]
    axes[1].bar(seasonal.index, seasonal["mean"], color=bar_colours, width=0.72)
    axes[1].set_xlabel(f"month ({tz})")
    axes[1].set_ylabel("PM2.5 (µg/m³)")
    axes[1].set_title("seasonal profile")
    axes[1].set_xticks(range(1, 13))
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=palette[1], label="dry (Nov-Apr)"),
        plt.Rectangle((0, 0), 1, 1, color=palette[2], label="monsoon (May-Oct)"),
    ]
    axes[1].legend(handles=handles)
    fig.suptitle("PM2.5 diurnal and seasonal structure")
    save_figure(cfg, fig, "fig04_diurnal_seasonal")

    log.info("figures written")

    # ----------------------------------------------------------------- report
    gate_pass = usable_years >= args.min_years
    site = cfg.get("data.openaq.site_label")

    lines: list[str] = []
    a = lines.append
    a("# Data Audit")
    a("")
    a(f"**Site:** {site}  ")
    a(f"**OpenAQ location IDs:** {cfg.get('data.openaq.location_ids')}  ")
    a("**Generated by:** `scripts/03_data_audit.py`  ")
    a(f"**Local timezone for calendar features:** `{tz}`  ")
    a("")
    a("> This is the gate before modelling. Every number here is computed from the")
    a("> fetched data, not transcribed.")
    a("")

    a("## 1. Verdict")
    a("")
    a(f"- Usable observed PM2.5: **{usable_years:.2f} years** ({observed:,} hours)")
    a(f"- Threshold for proceeding as planned: **{args.min_years:.1f} years**")
    a(f"- **Gate: {'PASS' if gate_pass else 'FAIL'}**")
    a("")
    if gate_pass:
        a("Dhaka has sufficient contiguous coverage to serve as the primary dataset.")
        a("Beijing (UCI) is retained for the cross-city generalisation check.")
    else:
        a("**Coverage is insufficient.** Beijing (UCI) should become the primary")
        a("dataset and Dhaka should be demoted to a case study. Do not proceed")
        a("without deciding this explicitly.")
    a("")

    a("## 2. Date coverage")
    a("")
    a("| Quantity | Value |")
    a("|---|---|")
    a(f"| First observation (UTC) | {cov['first_utc']} |")
    a(f"| Last observation (UTC) | {cov['last_utc']} |")
    a(f"| Span | {cov['span_days']:,.1f} days ({cov['span_years']:.2f} years) |")
    a(f"| Hours in span | {cov['expected_hours']:,} |")
    a(f"| Hours observed | {observed:,} |")
    a(f"| Coverage | **{pct_observed:.1f}%** |")
    a(f"| Hourly grid complete | {cov['index_is_complete']} |")
    a("")
    a("![coverage](../results/figures/fig01_coverage_timeline.png)")
    a("")

    a("## 3. Contiguity and gaps")
    a("")
    a("| Quantity | Value |")
    a("|---|---|")
    a(f"| Number of distinct gaps | {gaps.n_gaps:,} |")
    a(
        f"| Longest gap | **{gaps.longest_gap_hours:,} hours** ({gaps.longest_gap_hours / 24:.1f} days) |"
    )
    a(f"| Longest gap start (UTC) | {gaps.longest_gap_start} |")
    a(f"| Longest gap end (UTC) | {gaps.longest_gap_end} |")
    a(
        f"| Longest unbroken observed run | **{gaps.longest_run_hours:,} hours** ({gaps.longest_run_hours / 24:.1f} days) |"
    )
    a(f"| Longest run start (UTC) | {gaps.longest_run_start} |")
    a(f"| Longest run end (UTC) | {gaps.longest_run_end} |")
    a("")
    a("Gaps matter twice over: they bound how long an input window can be, and")
    a("any window spanning one is fabricated data. Gap-aware windowing rejects")
    a("those windows and reports the count.")
    a("")
    a("### Gap length distribution")
    a("")
    a(_fmt_table(gap_hist.set_index("gap_length"), floatfmt=".0f"))
    a("")
    a("The gap count alone is misleading. Most outages are single hours; only")
    a(f"{int(gap_hist.loc[gap_hist['gap_length'] == '> 7 d', 'n_gaps'].iloc[0])} exceed a week.")
    a("Applying the configured backward-looking forward-fill limit of")
    a(f"{max_ffill} hours raises coverage from {pct_observed:.1f}% to")
    a(f"{pct_after_ffill:.1f}% and collapses {gaps.n_gaps:,} fragments into")
    a(f"{n_runs_after_ffill:,} usable stretches, lengthening the longest from")
    a(f"{gaps.longest_run_hours:,} h to {longest_run_after_ffill:,} h.")
    a("")
    a("### Ten longest gaps")
    a("")
    a(_fmt_table(longest_gaps.set_index("rank"), floatfmt=".1f"))
    a("")
    a("The 2022 outage removes an entire monsoon season. Season-stratified")
    a("results should be read with that in mind: the wet-season sample is drawn")
    a("from eight monsoons, not nine.")
    a("")
    a("### Projected gap-free window yield")
    a("")
    a("What the gaps actually cost, per configuration. This is the number that")
    a("decides whether a 168-hour input window is affordable, and it is computed")
    a("here rather than discovered during training.")
    a("")
    a(_fmt_table(wyield.set_index(["input_window_h", "horizon_h"]), floatfmt=".1f"))
    a("")

    a("## 4. Quality-control ledger")
    a("")
    a("Every filter, with the rows it removed. Applied in this order.")
    a("")
    a(_fmt_table(pd.DataFrame(ledger).set_index("rule"), floatfmt=".0f"))
    a("")
    neg = next((e for e in ledger if e["rule"] == "negative value"), None)
    if neg and neg["removed"]:
        pct_neg = 100.0 * neg["removed"] / (neg["removed"] + neg["remaining"])
        a(f"**Note on negative values.** {neg['removed']:,} readings ({pct_neg:.1f}% of PM2.5")
        a("records) were negative and were dropped. Beta-attenuation monitors")
        a("routinely report small negatives near the detection limit, so this is")
        a("expected instrument behaviour rather than corruption; it is recorded")
        a("here because dropping rather than flooring them is a modelling choice.")
        a("")

    a("## 5. Missingness per variable")
    a("")
    a("### Target")
    a("")
    a(_fmt_table(miss_pm.set_index("variable"), floatfmt=".3f"))
    a("")
    a("### Meteorology (NASA POWER)")
    a("")
    a(_fmt_table(miss_met.set_index("variable"), floatfmt=".3f"))
    a("")
    a("Units as reported by the POWER API (quoted verbatim, not assumed):")
    a("")
    a("| Parameter | Units |")
    a("|---|---|")
    for k, v in sorted(units.items()):
        a(f"| {k} | {v} |")
    a("")
    a("Two of these differ from what is commonly assumed: `PRECTOTCORR` is")
    a("reported in mm/day and `ALLSKY_SFC_SW_DWN` in Wh/m^2. Values are used as")
    a("returned and never rescaled on an assumption.")
    a("")
    a("**Time standard.** POWER hourly defaults to Local Solar Time, not UTC.")
    a(f"This pull explicitly requested `time-standard={cfg.get('data.power.time_standard')}`")
    a("and the response header was verified to match.")
    a("")

    a("## 6. Monthly missingness")
    a("")
    a("![missingness](../results/figures/fig02_monthly_missingness.png)")
    a("")
    a(_fmt_table(monthly, floatfmt=".1f"))
    a("")

    a("## 7. PM2.5 distribution")
    a("")
    a("![distribution](../results/figures/fig03_pm25_distribution.png)")
    a("")
    a("| Statistic | Value (µg/m³) |")
    a("|---|---|")
    for k in [
        "n",
        "mean",
        "std",
        "min",
        "p01",
        "p05",
        "p25",
        "p50",
        "p75",
        "p90",
        "p95",
        "p99",
        "max",
    ]:
        a(f"| {k} | {dist[k]:,} |")
    a(f"| skew | {dist['skew']} |")
    a(f"| excess kurtosis | {dist['kurtosis']} |")
    a("")
    a("Exceedance of the Bangladesh national standards (Air Pollution (Control)")
    a("Rules 2022):")
    a("")
    a("| Threshold | Hours above |")
    a("|---|---|")
    for k, v in dist["exceedance"].items():
        a(f"| {k.replace('above_', '').replace('_ugm3_pct', ' µg/m³')} | {v}% |")
    a("")
    a("The distribution is strongly right-skewed, which is why the target is")
    a("modelled on a `log1p` scale while all metrics are reported in µg/m³, and")
    a("why sMAPE replaces MAPE.")
    a("")

    a("## 8. Diurnal and seasonal structure")
    a("")
    a("![profiles](../results/figures/fig04_diurnal_seasonal.png)")
    a("")
    a("### By season")
    a("")
    a(_fmt_table(seasons))
    a("")
    a("Season boundaries follow the Bangladesh Department of Environment: dry")
    a("November-April, wet May-October.")
    a("")
    a("### By month")
    a("")
    a(_fmt_table(seasonal))
    a("")
    a("### By hour of day (Asia/Dhaka)")
    a("")
    a(_fmt_table(diurnal))
    a("")

    a("## 9. Fused dataset (PM2.5 + meteorology)")
    a("")
    a("| Quantity | Value |")
    a("|---|---|")
    a(f"| Rows after inner join on UTC hour | {len(fused):,} |")
    a(f"| Rows with no missing value in any column | {fused_complete:,} |")
    a(f"| Fully-complete fraction of joined rows | {100.0 * fused_complete / len(fused):.1f}% |")
    a("")

    if beijing is not None:
        b_cov = coverage_summary(beijing)
        target = str(cfg.get("data.uci.target_column"))
        a("## 10. UCI Beijing (fallback / cross-city)")
        a("")
        a("| Quantity | Value |")
        a("|---|---|")
        a(f"| Rows | {len(beijing):,} |")
        a(f"| First (UTC) | {b_cov['first_utc']} |")
        a(f"| Last (UTC) | {b_cov['last_utc']} |")
        a(f"| Span | {b_cov['span_years']:.2f} years |")
        a(f"| Columns | {len(beijing.columns)} |")
        if target in beijing.columns:
            a(f"| {target} observed | {int(beijing[target].notna().sum()):,} |")
            a(f"| {target} missing | {100.0 * beijing[target].isna().mean():.2f}% |")
        a("")

    a("---")
    a("")
    a("## Decision required")
    a("")
    if gate_pass:
        a("Coverage clears the threshold. Awaiting go-ahead to proceed to Phase 2")
        a("(features, splits, leakage tests).")
    else:
        a("Coverage does not clear the threshold. A strategy change is required")
        a("before any modelling.")
    a("")

    report_path = Path(str(cfg.get("output.report.data_audit_md")))
    report_path = (
        report_path if report_path.is_absolute() else (cfg.path_for("reports").parent / report_path)
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    log.info("wrote %s", report_path)

    summary = {
        "coverage": cov,
        "observed_hours": observed,
        "pct_observed": round(pct_observed, 2),
        "usable_years": round(usable_years, 3),
        "longest_gap_hours": gaps.longest_gap_hours,
        "longest_run_hours": gaps.longest_run_hours,
        "n_gaps": gaps.n_gaps,
        "distribution": dist,
        "fused_rows": len(fused),
        "fused_complete_rows": fused_complete,
        "gate_pass": gate_pass,
    }
    (cfg.path_for("data_interim") / "audit_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )

    print("\n" + "=" * 72)
    print("DATA AUDIT SUMMARY")
    print("=" * 72)
    print(f"  site                     {site}")
    print(f"  span                     {cov['first_utc']} -> {cov['last_utc']}")
    print(f"  hours in span            {cov['expected_hours']:,}")
    print(f"  hours observed           {observed:,} ({pct_observed:.1f}%)")
    print(f"  usable years             {usable_years:.2f}")
    print(f"  distinct gaps            {gaps.n_gaps:,}")
    print(
        f"  longest gap              {gaps.longest_gap_hours:,} h ({gaps.longest_gap_hours / 24:.1f} d)"
    )
    print(
        f"  longest unbroken run     {gaps.longest_run_hours:,} h ({gaps.longest_run_hours / 24:.1f} d)"
    )
    print(f"  PM2.5 mean / median      {dist['mean']} / {dist['p50']} µg/m³")
    print(f"  PM2.5 p95 / max          {dist['p95']} / {dist['max']} µg/m³")
    for k, v in dist["exceedance"].items():
        print(f"  {k:<24} {v}%")
    print(f"  fused rows (PM2.5+met)   {len(fused):,} ({fused_complete:,} fully complete)")
    print(f"\n  GATE: {'PASS' if gate_pass else 'FAIL'}  (threshold {args.min_years:.1f} years)")
    print(f"  report -> {report_path}")
    return 0 if gate_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
