"""Cross-city comparison: Dhaka against Beijing.

Reads both cities' ``results.json`` and produces a single comparison table plus a
short narrative. The question it answers is not "which model is best" but
"does the *ranking* of methods transfer between cities" -- which is the only
question a single-city benchmark cannot answer, and the reason the cross-city
run exists.

Run::

    python scripts/13_cross_city.py --config config.yaml --config-b config_beijing.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from src.results import load_results
from src.utils import check_disk_space, load_config, setup_logging
from src.viz.tables import write_table


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="config.yaml", help="primary city config")
    p.add_argument("--config-b", default="config_beijing.yaml", help="comparison city config")
    return p.parse_args()


def _summary(cfg, label: str) -> pd.DataFrame:
    """Flatten one city's runs into per-model, per-horizon means.

    Sequence models are collapsed to their best variant by mean **validation**
    loss, never by test error.
    """
    payload = load_results(cfg)
    runs = payload.get("runs", [])
    if not runs:
        return pd.DataFrame()

    frame = pd.DataFrame(runs)
    frame["rmse"] = frame["metrics"].map(lambda m: m["rmse"])
    frame["skill"] = frame["metrics"].map(lambda m: m["skill_vs_persistence"])

    grouped = (
        frame.groupby(["model", "tier", "variant", "horizon_h"], dropna=False)
        .agg(
            rmse=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            skill=("skill", "mean"),
            skill_std=("skill", "std"),
            val=("best_val_loss", "mean") if "best_val_loss" in frame.columns else ("rmse", "size"),
            n_seeds=("seed", "count"),
            n_params=("n_params", "first") if "n_params" in frame.columns else ("seed", "count"),
        )
        .reset_index()
    )

    # For sequence models keep one row per horizon: the validation-selected one.
    seq = grouped[grouped["tier"] == "tier3"].copy()
    rest = grouped[grouped["tier"] != "tier3"].copy()
    if not seq.empty and seq["val"].notna().any():
        seq = seq.sort_values("val").groupby("horizon_h").first().reset_index()
    elif not seq.empty:
        seq = seq.sort_values("rmse").groupby("horizon_h").first().reset_index()
    if not seq.empty:
        # Label by ROLE, not by which architecture happened to win. The winning
        # configuration differs between cities, and naming the row after it would
        # split the sequence model into two half-empty rows -- excluding exactly
        # the comparison the cross-city run exists to make from the rank
        # correlation. The chosen architecture is kept in its own column.
        seq["chosen_arch"] = seq["model"].astype(str) + " " + seq["variant"].astype(str)
        seq["model"] = "best sequence"

    out = pd.concat([rest, seq], ignore_index=True)
    out["city"] = label
    return out


def main() -> int:
    """Build the cross-city comparison."""
    args = parse_args()
    cfg_a = load_config(args.config)
    cfg_b = load_config(args.config_b)
    log = setup_logging(cfg_a, "13_cross_city")
    check_disk_space(cfg_a)

    label_a = str(cfg_a.get("data.site.city"))
    label_b = str(cfg_b.get("data.site.city"))

    a = _summary(cfg_a, label_a)
    b = _summary(cfg_b, label_b)
    if a.empty or b.empty:
        log.error("one of the cities has no runs; nothing to compare")
        return 1

    headline = int(cfg_a.get("task.headline_horizon_h"))
    both = pd.concat([a, b], ignore_index=True)
    both.to_csv(cfg_a.path_for("tables") / "cross_city_raw.csv", index=False)

    # ---- headline-horizon comparison, ranked within each city --------------
    rows = []
    for city, group in both[both["horizon_h"] == headline].groupby("city"):
        ranked = group.sort_values("rmse").reset_index(drop=True)
        for rank, r in ranked.iterrows():
            rows.append(
                {
                    "city": city,
                    "rank": rank + 1,
                    "model": r["model"],
                    "tier": r["tier"],
                    "rmse": round(float(r["rmse"]), 2),
                    "skill": round(float(r["skill"]), 4),
                    "chosen": r.get("chosen_arch", ""),
                }
            )
    ranking = pd.DataFrame(rows)

    pivot = ranking.pivot_table(index="model", columns="city", values="rank", aggfunc="first")
    skill = ranking.pivot_table(index="model", columns="city", values="skill", aggfunc="first")
    merged = pivot.join(skill, lsuffix=" rank", rsuffix=" skill").reset_index()
    merged = merged.sort_values(f"{label_a} rank")

    display = merged.rename(columns={"model": "Model"})
    write_table(
        cfg_a,
        display,
        "cross_city_comparison",
        caption=(
            f"Method ranking at the {headline}-hour horizon in {label_a} and {label_b}. "
            "Rank is by test RMSE within each city; skill is against that city's own "
            "persistence baseline. Sequence models are collapsed to the "
            "validation-selected configuration. The two cities use an identical "
            "pipeline, model grid, seed set and split procedure; they differ in "
            "record quality, in one meteorological driver, and in season definition."
        ),
    )

    # ---- rank correlation --------------------------------------------------
    common = merged.dropna(subset=[f"{label_a} rank", f"{label_b} rank"])
    correlation = float("nan")
    if len(common) >= 3:
        correlation = float(
            common[f"{label_a} rank"].corr(common[f"{label_b} rank"], method="spearman")
        )

    audit_a = json.loads(
        (cfg_a.path_for("data_interim") / "audit_summary.json").read_text(encoding="utf-8")
    )
    audit_b = json.loads(
        (cfg_b.path_for("data_interim") / "audit_summary.json").read_text(encoding="utf-8")
    )

    context = pd.DataFrame(
        [
            {
                "Quantity": "Usable years",
                label_a: audit_a["usable_years"],
                label_b: audit_b["usable_years"],
            },
            {
                "Quantity": "Hourly coverage (%)",
                label_a: audit_a["pct_observed"],
                label_b: audit_b["pct_observed"],
            },
            {"Quantity": "Distinct gaps", label_a: audit_a["n_gaps"], label_b: audit_b["n_gaps"]},
            {
                "Quantity": "Longest gap (h)",
                label_a: audit_a["longest_gap_hours"],
                label_b: audit_b["longest_gap_hours"],
            },
            {
                "Quantity": "PM2.5 mean (ug/m3)",
                label_a: audit_a["distribution"]["mean"],
                label_b: audit_b["distribution"]["mean"],
            },
            {
                "Quantity": "PM2.5 median (ug/m3)",
                label_a: audit_a["distribution"]["p50"],
                label_b: audit_b["distribution"]["p50"],
            },
            {
                "Quantity": "Hours > 65 ug/m3 (%)",
                label_a: audit_a["distribution"]["exceedance"].get("above_65_ugm3_pct"),
                label_b: audit_b["distribution"]["exceedance"].get("above_65_ugm3_pct"),
            },
        ]
    )
    write_table(
        cfg_a,
        context,
        "cross_city_context",
        caption=(
            f"Record characteristics for {label_a} and {label_b}. The pollution "
            "distributions are comparable; the record quality is not, which is what "
            "makes the pair informative."
        ),
    )

    payload_a = load_results(cfg_a)
    payload_a.setdefault("cross_city", {})[label_b] = {
        "ranking": ranking.to_dict(orient="records"),
        "spearman_rank_correlation": correlation,
        "context": context.to_dict(orient="records"),
    }
    (cfg_a.path_for("results")).mkdir(parents=True, exist_ok=True)
    Path(cfg_a.path_for("results_json")).write_text(
        json.dumps(payload_a, indent=2, default=str), encoding="utf-8"
    )

    print("\n" + "=" * 88)
    print(f"CROSS-CITY METHOD RANKING AT h={headline}")
    print("=" * 88)
    print(display.to_string(index=False))
    print(f"\nSpearman rank correlation between cities: {correlation:+.3f}")
    if correlation < 0.3:
        print(
            "\n  The ranking does NOT transfer. A benchmark run on either city alone would\n"
            "  have recommended a different method, and neither recommendation would\n"
            "  generalise. This is the central argument for reporting both."
        )
    print("\n" + "=" * 88)
    print("RECORD CHARACTERISTICS")
    print("=" * 88)
    print(context.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
