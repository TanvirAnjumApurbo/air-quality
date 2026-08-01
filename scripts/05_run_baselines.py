"""Phase 3: Tier 1 honest baselines and Tier 2 classical ML.

This tier alone is a publishable results table, and it is the insurance against
running out of time on the later ones. Results are written to
``results/results.json`` and to matched CSV/LaTeX tables.

Tuning uses ``TimeSeriesSplit`` over train+validation only; the test split is
untouched until final scoring.

Run::

    python scripts/05_run_baselines.py --config config.yaml
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from src.eval.metrics import all_metrics, block_bootstrap_rmse, skill_score
from src.models.baselines import (
    fit_climatology,
    fit_predict_sarimax,
    predict_climatology,
    predict_persistence,
    predict_seasonal_naive,
)
from src.models.data import get_split_arrays, invert, load_features
from src.models.trees import (
    fit_lightgbm,
    fit_random_forest,
    fit_ridge,
    fit_xgboost,
    permutation_importance_scores,
)
from src.results import load_results, save_results, upsert_run
from src.utils import check_disk_space, load_config, resolve_device, set_seed, setup_logging
from src.viz.tables import format_mean_std, write_table


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--horizons", type=int, nargs="*", default=None)
    parser.add_argument("--skip-sarimax", action="store_true", help="SARIMAX is the slow one")
    parser.add_argument("--skip-trees", action="store_true")
    parser.add_argument("--device", default=None, choices=["auto", "cuda", "cpu"])
    return parser.parse_args()


def main() -> int:
    """Run Tier 1 and Tier 2 and write the results tables."""
    args = parse_args()
    cfg = load_config(args.config)
    log = setup_logging(cfg, "05_run_baselines")
    check_disk_space(cfg)
    device = resolve_device(cfg, args.device)
    log.info("device: %s", device)

    seed = int(cfg.get("seeds.primary"))
    set_seed(seed, cfg)

    frame = load_features(cfg)
    horizons = args.horizons or [int(h) for h in cfg.get("task.horizons_h")]
    boot = cfg.get("evaluation.bootstrap_ci")

    payload = load_results(cfg)
    rows: list[dict] = []

    for h in horizons:
        log.info("=" * 70)
        log.info("HORIZON h=%d", h)
        log.info("=" * 70)

        train = get_split_arrays(frame, cfg, h, "train")
        val = get_split_arrays(frame, cfg, h, "val")
        test = get_split_arrays(frame, cfg, h, "test")
        log.info("rows: train=%d val=%d test=%d", len(train), len(val), len(test))

        predictions: dict[str, np.ndarray] = {}

        # ---------------- Tier 1 ------------------------------------------
        predictions["persistence"] = predict_persistence(test)

        if h <= 24:
            predictions["seasonal_naive"] = predict_seasonal_naive(test)
        else:
            log.info("seasonal_naive undefined for h>24; skipping")

        clim = fit_climatology(train, h)
        predictions["climatology"] = predict_climatology(clim, test, h)

        if not args.skip_sarimax and cfg.get("models.baselines.sarimax.enabled"):
            started = time.perf_counter()
            predictions["sarimax"] = fit_predict_sarimax(train, test, cfg, h, log)
            log.info("sarimax h=%d took %.1f s", h, time.perf_counter() - started)

        # ---------------- Tier 2 ------------------------------------------
        tuned = []
        if not args.skip_trees:
            if cfg.get("models.trees.ridge.enabled"):
                tuned.append(fit_ridge(train, val, cfg, seed, log))
            if cfg.get("models.trees.random_forest.enabled"):
                tuned.append(fit_random_forest(train, val, cfg, seed, log))
            if cfg.get("models.trees.xgboost.enabled"):
                tuned.append(fit_xgboost(train, val, cfg, seed, log, device))
            if cfg.get("models.trees.lightgbm.enabled"):
                model = fit_lightgbm(train, val, cfg, seed, log)
                if model is not None:
                    tuned.append(model)

        for model in tuned:
            raw = model.estimator.predict(test.x)
            predictions[model.name] = invert(cfg, raw)
            log.info(
                "%s h=%d: cv=%.4f best=%s (%.1f s)",
                model.name,
                h,
                model.cv_score,
                model.best_params,
                model.fit_seconds,
            )

        # ---------------- scoring -----------------------------------------
        persistence_rmse = all_metrics(test.y, predictions["persistence"])["rmse"]

        for name, pred in predictions.items():
            metrics = all_metrics(test.y, pred)
            metrics["skill_vs_persistence"] = skill_score(metrics["rmse"], persistence_rmse)

            ci = block_bootstrap_rmse(
                test.y,
                pred,
                n_resamples=int(boot["n_resamples"]),
                block_size=int(boot["block_size_h"]),
                alpha=float(boot["alpha"]),
                seed=seed,
            )
            metrics["rmse_ci_lower"] = ci.lower
            metrics["rmse_ci_upper"] = ci.upper

            tier = (
                "tier1"
                if name in {"persistence", "seasonal_naive", "climatology", "sarimax"}
                else "tier2"
            )

            record = {
                "model": name,
                "tier": tier,
                "variant": "standard",
                "horizon_h": h,
                "seed": seed,
                "split": "test",
                "metrics": metrics,
            }
            match = next((m for m in tuned if m.name == name), None)
            if match is not None:
                record["best_params"] = match.best_params
                record["cv_score"] = match.cv_score
                record["fit_seconds"] = match.fit_seconds
                record["n_features"] = len(match.feature_names)
            upsert_run(payload, record)

            rows.append({"model": name, "tier": tier, "horizon_h": h, **metrics})
            log.info(
                "  %-16s RMSE %7.2f  MAE %7.2f  R2 %6.3f  sMAPE %6.2f  skill %+.4f",
                name,
                metrics["rmse"],
                metrics["mae"],
                metrics["r2"],
                metrics["smape"],
                metrics["skill_vs_persistence"],
            )

        # ---------------- permutation importance (headline horizon) --------
        if h == int(cfg.get("task.headline_horizon_h")) and tuned:
            tree_models = [m for m in tuned if m.name in {"xgboost", "lightgbm", "random_forest"}]
            if tree_models:
                chosen = tree_models[0]
                log.info("permutation importance for %s at h=%d", chosen.name, h)
                importance = permutation_importance_scores(chosen, test, seed)
                payload.setdefault("importance", {})[f"{chosen.name}_h{h}"] = importance

    save_results(cfg, payload)

    # ---------------- tables ----------------------------------------------
    results = pd.DataFrame(rows)
    results.to_csv(cfg.path_for("tables") / "baselines_raw.csv", index=False)

    order = [
        "persistence",
        "seasonal_naive",
        "climatology",
        "sarimax",
        "ridge",
        "random_forest",
        "xgboost",
        "lightgbm",
    ]
    results["_order"] = results["model"].map({m: i for i, m in enumerate(order)}).fillna(99)

    main_table = (
        results.sort_values(["horizon_h", "_order"])
        .assign(
            RMSE=lambda d: d["rmse"].round(2),
            MAE=lambda d: d["mae"].round(2),
            R2=lambda d: d["r2"].round(3),
            sMAPE=lambda d: d["smape"].round(2),
            Skill=lambda d: d["skill_vs_persistence"].round(4),
        )[["model", "tier", "horizon_h", "RMSE", "MAE", "R2", "sMAPE", "Skill"]]
        .rename(columns={"model": "Model", "tier": "Tier", "horizon_h": "h (hours)"})
    )
    write_table(
        cfg,
        main_table,
        "tier1_tier2_all_horizons",
        caption=(
            "Tier 1 baselines and Tier 2 classical models, all horizons. "
            "Skill is $1 - \\mathrm{RMSE}_{\\mathrm{model}}/\\mathrm{RMSE}_{\\mathrm{persistence}}$; "
            "positive values beat persistence."
        ),
    )

    headline = int(cfg.get("task.headline_horizon_h"))
    head = results[results["horizon_h"] == headline].sort_values("_order")
    head_table = head.assign(
        RMSE=lambda d: [
            format_mean_std(r, np.nan) + f" [{lo:.1f}, {hi:.1f}]"
            for r, lo, hi in zip(d["rmse"], d["rmse_ci_lower"], d["rmse_ci_upper"], strict=True)
        ],
        MAE=lambda d: d["mae"].round(2),
        R2=lambda d: d["r2"].round(3),
        sMAPE=lambda d: d["smape"].round(2),
        Skill=lambda d: d["skill_vs_persistence"].round(4),
    )[["model", "RMSE", "MAE", "R2", "sMAPE", "Skill"]].rename(
        columns={"model": "Model", "RMSE": "RMSE [95\\% CI]"}
    )
    write_table(
        cfg,
        head_table,
        f"tier1_tier2_h{headline}",
        caption=(
            f"Tier 1 and Tier 2 at the headline {headline}-hour horizon. "
            "RMSE intervals are moving-block bootstrap (block 24 h), which "
            "respects the serial correlation an i.i.d. bootstrap would ignore."
        ),
    )

    print("\n" + "=" * 96)
    print("TIER 1 + TIER 2 RESULTS (test period)")
    print("=" * 96)
    print(main_table.to_string(index=False))
    print(f"\nresults -> {cfg.path_for('results_json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
