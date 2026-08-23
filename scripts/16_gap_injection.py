r"""Phase 7: the gap-injection experiment.

Turns the study's central claim from an observation into a controlled test.

Across two cities it looks as though record fragmentation, not model class,
decides which forecasting method wins: tree ensembles lead on Dhaka's broken
record, the sequence model leads on Beijing's near-complete one. But those two
cities differ in coverage, span, climate and instrument simultaneously, so the
comparison cannot separate fragmentation from anything else.

Here fragmentation is manipulated directly. A near-complete donor record is
degraded to a series of coverage levels by two arms that remove **exactly the
same number of hours**: one as many short outages drawn from Dhaka's empirical
gap-length distribution, the other as a few long blocks. Data volume is
therefore held constant and only contiguity differs, so the gap between the arms
is the effect of fragmentation alone.

Two controls make the comparison readable:

* the test period is never degraded, so every cell is scored on identical rows
  and RMSE is comparable across the whole grid;
* model hyperparameters are fixed at the values selected on the undegraded
  record, which is both the honest practitioner setting and the only way to
  avoid re-tuning turning into a second, uncontrolled variable.

Run::

    # see the grid without training anything
    python scripts/16_gap_injection.py --config config_beijing.yaml --dry-run

    # the experiment; resumable, skips cells already recorded
    python scripts/16_gap_injection.py --config config_beijing.yaml --progress plain
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from src.eval.metrics import all_metrics, skill_score
from src.features.gap_injection import empirical_gap_profile, inject_gaps
from src.features.pipeline import build_feature_matrix
from src.models.baselines import fit_climatology, predict_climatology, predict_persistence
from src.models.data import build_sequence_index, get_split_arrays, invert
from src.models.sequence import DeviceWindowSampler as Sampler
from src.models.sequence import (
    ModelSpec,
    _make_loss,
    evaluate_sampler,
    train_one,
)
from src.utils import check_disk_space, load_config, resolve_device, setup_logging


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--config", default="config_beijing.yaml", help="the donor record's config")
    p.add_argument(
        "--profile-config",
        default="config.yaml",
        help="config of the record whose gap-length distribution is injected",
    )
    p.add_argument("--device", default=None, choices=["auto", "cuda", "cpu"])
    p.add_argument("--progress", default="plain", choices=["tqdm", "plain"])
    p.add_argument("--dry-run", action="store_true", help="print the grid and exit")
    p.add_argument("--force", action="store_true", help="recompute cells already recorded")
    p.add_argument(
        "--keep-cells",
        action="store_true",
        help="keep each cell's feature matrix instead of deleting it after scoring",
    )
    p.add_argument("--coverage", type=float, nargs="*", default=None)
    p.add_argument("--arm", default=None, choices=["fragmented", "contiguous", "none"])
    p.add_argument("--seed", type=int, nargs="*", default=None)
    return p.parse_args()


def _tuned_params(payload: dict, model: str, horizon: int) -> dict:
    """Recover the hyperparameters selected on the undegraded record.

    Args:
        payload: The donor city's ``results.json`` contents.
        model: Model name.
        horizon: Forecast horizon.

    Returns:
        The stored ``best_params``, or an empty mapping if the run is absent.
    """
    for record in payload.get("runs", []):
        if record.get("model") == model and record.get("horizon_h") == horizon:
            return dict(record.get("best_params") or {})
    return {}


def _score(
    results: dict[str, dict],
    logger: object,
    name: str,
    truth: np.ndarray,
    pred: np.ndarray,
    persistence_rmse: float,
    extra: dict | None = None,
) -> None:
    """Score one model in a cell and record it.

    Args:
        results: Per-cell result mapping to write into.
        logger: Logger.
        name: Model name.
        truth: Observed test values.
        pred: Predicted test values, in ug/m3.
        persistence_rmse: Persistence RMSE on the same rows, for the skill score.
        extra: Additional fields to record alongside the metrics.
    """
    metrics = all_metrics(truth, pred)
    metrics["skill_vs_persistence"] = skill_score(metrics["rmse"], persistence_rmse)
    results[name] = {**metrics, **(extra or {})}
    logger.info(  # type: ignore[attr-defined]
        "    %-16s RMSE %7.2f  skill %+.4f",
        name,
        metrics["rmse"],
        metrics["skill_vs_persistence"],
    )


def _fit_tabular(name: str, params: dict, cfg, train, val, seed: int):  # noqa: ANN202
    """Fit one tier-2 model at fixed hyperparameters.

    The randomised search is deliberately bypassed. Re-tuning inside every cell
    would let hyperparameter search compensate for the degradation, which is a
    second uncontrolled variable and would blunt exactly the effect being
    measured. Fixing them at the undegraded record's choices is also what a
    practitioner deploying a tuned pipeline onto a worse record actually does.

    Args:
        name: Model name.
        params: Hyperparameters from the undegraded run.
        cfg: Loaded configuration, with paths pointing at this cell.
        train: Training arrays.
        val: Validation arrays.
        seed: RNG seed.

    Returns:
        The fitted estimator, or None if the library is unavailable.
    """
    x = np.vstack([train.x, val.x])
    y = np.concatenate([train.y_transformed, val.y_transformed])

    if name == "ridge":
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from src.models.trees import LogScaleTargetHistory

        alpha = float(params.get("ridge__alpha", params.get("alpha", 1.0)))
        model = Pipeline(
            [
                ("log_history", LogScaleTargetHistory(cfg, list(train.feature_names))),
                ("rescale", StandardScaler()),
                ("ridge", Ridge(alpha=alpha, random_state=seed)),
            ]
        )
    elif name == "random_forest":
        from sklearn.ensemble import RandomForestRegressor

        model = RandomForestRegressor(random_state=seed, n_jobs=-1, **params)
    elif name == "xgboost":
        from xgboost import XGBRegressor

        model = XGBRegressor(random_state=seed, tree_method="hist", verbosity=0, **params)
    elif name == "lightgbm":
        try:
            from lightgbm import LGBMRegressor
        except Exception:
            return None
        model = LGBMRegressor(random_state=seed, verbose=-1, n_jobs=-1, **params)
    else:
        raise ValueError(f"unknown tabular model {name!r}")

    model.fit(x, y)
    return model


def main() -> int:
    """Run the gap-injection grid."""
    args = parse_args()
    cfg = load_config(args.config)
    log = setup_logging(cfg, "16_gap_injection")
    check_disk_space(cfg)
    device = resolve_device(cfg, args.device)

    spec_cfg = cfg.get("ablation.gap_injection")
    horizon = int(spec_cfg.get("horizon_h", cfg.get("task.headline_horizon_h")))
    coverages = args.coverage or [float(c) for c in spec_cfg["coverage_levels"]]
    arms = [args.arm] if args.arm else list(spec_cfg["arms"])
    seeds = args.seed or [int(s) for s in spec_cfg["injection_seeds"]]
    tabular = list(spec_cfg["tabular_models"])
    sequence_specs = list(spec_cfg["sequence_models"])
    model_seeds = [int(s) for s in spec_cfg.get("model_seeds", [42])]
    target_steps = int(spec_cfg.get("target_optimizer_steps", 5100))
    max_epoch_scale = float(spec_cfg.get("max_epoch_scale", 8.0))

    # The undegraded record is the reference cell, and it only needs computing
    # once per donor rather than once per injection seed.
    grid = [("none", 1.0, seeds[0])] + [
        (arm, cov, seed) for cov in coverages for arm in arms for seed in seeds if arm != "none"
    ]

    out_path = Path(cfg.get("paths.results")) / str(
        cfg.get("ablation.gap_injection.output_name", "ablation_gap_injection.json")
    )
    payload = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {}
    cells = payload.get("cells", {})

    # Resume keys on (arm, coverage, injection seed) and says nothing about WHICH
    # record produced the cell. If two configs ever resolve to one output path,
    # every cell reads as "already recorded" and the run reports a complete grid
    # for a station it never touched -- which is exactly what happened when the
    # make shim dropped --config and three stations resumed off Wanliu's file.
    # The donor label is the identity that matters, so disagreement is fatal
    # rather than a warning: a silently mislabelled grid is worse than no grid.
    this_donor = str(cfg.get("data.openaq.site_label") or cfg.get("data.site.city"))
    recorded_donor = str(payload.get("donor", this_donor))
    if cells and recorded_donor != this_donor:
        log.error(
            "%s holds %d cells recorded for %r, but this config is %r. Resuming "
            "would attribute another record's results to this one. Point "
            "ablation.gap_injection.output_name at a distinct file, or delete "
            "the existing one to recompute.",
            out_path,
            len(cells),
            recorded_donor,
            this_donor,
        )
        return 1

    if args.dry_run:
        print(f"\n{len(grid)} cells would run at h={horizon}:")
        for arm, cov, seed in grid:
            print(f"  arm={arm:<11} coverage={cov:.2f}  injection_seed={seed}")
        print(
            f"\nmodels per cell: {tabular + [s['name'] for s in sequence_specs]} + persistence, climatology"
        )
        print(f"sequence seeds per cell: {model_seeds}")
        return 0

    # ---- donor record and the gap pattern being injected --------------------
    interim = cfg.path_for("data_interim")
    pm = pd.read_parquet(interim / str(cfg.get("data.files.target")))
    met = pd.read_parquet(interim / str(cfg.get("data.files.meteorology")))

    profile_cfg = load_config(args.profile_config)
    donor_pm = pd.read_parquet(
        profile_cfg.path_for("data_interim") / str(profile_cfg.get("data.files.target"))
    )
    profile = empirical_gap_profile(
        donor_pm[str(profile_cfg.get("features.target"))],
        str(profile_cfg.get("data.openaq.site_label")),
    )
    log.info("injecting the gap profile of %s: %s", profile.source, profile.summary())

    val_end = pd.Timestamp(str(cfg.get("split.explicit_boundaries.val_end")))
    log.info("test period from %s is protected and never degraded", val_end.date())

    tuned = json.loads(Path(cfg.get("paths.results_json")).read_text(encoding="utf-8"))
    base_processed = cfg.raw["paths"]["data_processed"]
    base_checkpoints = cfg.raw["paths"]["checkpoints"]
    base_tables = cfg.raw["paths"]["tables"]
    cell_root = Path(cfg.get("paths.data_interim")) / "ablation"

    started_all = time.perf_counter()
    for n, (arm, coverage, inj_seed) in enumerate(grid, start=1):
        key = f"{arm}_cov{coverage:.2f}_s{inj_seed}"
        if key in cells and not args.force:
            log.info("[%d/%d] %s: already recorded, skipping", n, len(grid), key)
            continue

        log.info("=" * 78)
        log.info("[%d/%d] cell %s", n, len(grid), key)
        degraded, report = inject_gaps(
            pm,
            str(cfg.get("features.target")),
            arm=arm,
            target_coverage=coverage,
            profile=profile,
            protect_from=val_end,
            n_blocks=int(spec_cfg.get("contiguous_blocks", 8)),
            seed=inj_seed,
        )
        log.info(
            "  removed %d h -> coverage %.4f, %d gaps (longest %d h)",
            report.hours_removed,
            report.achieved_coverage,
            report.n_gaps_after,
            report.longest_gap_after,
        )

        # Redirect every write to this cell so nothing collides with the main run.
        cell_dir = cell_root / key
        cfg.raw["paths"]["data_processed"] = str(cell_dir / "processed")
        cfg.raw["paths"]["checkpoints"] = str(cell_dir / "checkpoints")
        cfg.raw["paths"]["tables"] = str(cell_dir / "tables")

        built = build_feature_matrix(degraded, met, cfg, log, write=True)
        frame = built.frame
        valid = built.meta["valid_rows_per_split_per_horizon"][str(horizon)]
        log.info(
            "  usable rows h=%d: train %d, val %d, test %d",
            horizon,
            valid["train"],
            valid["val"],
            valid["test"],
        )

        train = get_split_arrays(frame, cfg, horizon, "train")
        val = get_split_arrays(frame, cfg, horizon, "val")
        test = get_split_arrays(frame, cfg, horizon, "test")

        results: dict[str, dict] = {}
        persistence = predict_persistence(test)
        persistence_rmse = all_metrics(test.y, persistence)["rmse"]

        _score(results, log, "persistence", test.y, persistence, persistence_rmse)
        _score(
            results,
            log,
            "climatology",
            test.y,
            predict_climatology(fit_climatology(train, horizon, cfg), test, horizon),
            persistence_rmse,
        )

        for name in tabular:
            params = _tuned_params(tuned, name, horizon)
            try:
                model = _fit_tabular(name, params, cfg, train, val, model_seeds[0])
            except Exception as exc:
                log.warning("    %s failed: %s", name, exc)
                continue
            if model is None:
                continue
            _score(
                results,
                log,
                name,
                test.y,
                invert(cfg, model.predict(test.x)),
                persistence_rmse,
                {"params": params},
            )

        loss_fn = _make_loss(cfg)
        batch = int(cfg.get("models.sequence.train.batch_size"))
        train_cfg = cfg.raw["models"]["sequence"]["train"]
        base_schedule = {
            "epochs": int(train_cfg["epochs"]),
            "patience": int(train_cfg["early_stopping"]["patience"]),
            "min_epochs": int(train_cfg["early_stopping"].get("min_epochs", 0)),
        }
        for entry in sequence_specs:
            per_seed = []
            for model_seed in model_seeds:
                spec = ModelSpec(
                    arch=str(entry["arch"]),
                    hidden_size=int(entry.get("hidden_size", 0)),
                    num_layers=int(entry.get("num_layers", 1)),
                    window=int(entry["window_h"]),
                    horizon=horizon,
                    seed=model_seed,
                    kernel_size=25 if entry["arch"] == "dlinear" else None,
                )
                train_idx = build_sequence_index(frame, cfg, horizon, spec.window, "train")
                val_idx = build_sequence_index(frame, cfg, horizon, spec.window, "val")
                test_idx = build_sequence_index(frame, cfg, horizon, spec.window, "test")
                if len(train_idx) < 100 or len(val_idx) < 50:
                    log.warning(
                        "    %s: only %d train / %d val windows survive; skipping",
                        spec.name,
                        len(train_idx),
                        len(val_idx),
                    )
                    break

                # Equalise the OPTIMIZER-STEP budget, not the epoch budget.
                #
                # Fragmentation shrinks the training set, so at a fixed epoch
                # count a fragmented cell takes far fewer gradient updates than a
                # contiguous one -- 16 steps/epoch against 65 at 82% coverage.
                # Left alone, part of the fragmented arm's degradation would be
                # undertraining rather than lost information, and the experiment
                # would not be measuring what it claims to. Epochs, patience and
                # the minimum-epoch floor are all rescaled so every cell sees the
                # same number of updates on the same schedule shape.
                steps_per_epoch = max(1, -(-len(train_idx) // batch))
                scale = target_steps / (steps_per_epoch * base_schedule["epochs"])
                scale = float(np.clip(scale, 1.0, max_epoch_scale))
                train_cfg["epochs"] = round(base_schedule["epochs"] * scale)
                train_cfg["early_stopping"]["patience"] = max(
                    1, round(base_schedule["patience"] * scale)
                )
                train_cfg["early_stopping"]["min_epochs"] = round(
                    base_schedule["min_epochs"] * scale
                )
                log.info(
                    "    %s: %d windows, %d steps/epoch -> %d epochs (~%d optimizer steps)",
                    spec.name,
                    len(train_idx),
                    steps_per_epoch,
                    train_cfg["epochs"],
                    steps_per_epoch * train_cfg["epochs"],
                )
                model, _ = train_one(
                    spec,
                    train_idx,
                    val_idx,
                    cfg,
                    device,
                    log,
                    resume="auto",
                    progress=args.progress,
                    max_minutes=float(cfg.get("runtime.max_minutes_per_run", 10)),
                    run_tag=key,
                )
                _, raw = evaluate_sampler(model, Sampler(test_idx, device), loss_fn, batch)
                preds = invert(cfg, raw)
                metrics = all_metrics(test_idx.y, preds)
                metrics["skill_vs_persistence"] = skill_score(
                    metrics["rmse"], all_metrics(test_idx.y, test_idx.persistence)["rmse"]
                )
                per_seed.append(metrics)

            if per_seed:
                # Averaged over seeds rather than scored once, matching how the
                # main sweep reports the sequence tier.
                results[spec.name] = {
                    metric: float(np.mean([m[metric] for m in per_seed]))
                    for metric in ("rmse", "mae", "r2", "smape", "skill_vs_persistence")
                }
                results[spec.name]["rmse_std"] = float(np.std([m["rmse"] for m in per_seed]))
                results[spec.name]["n_seeds"] = len(per_seed)
                results[spec.name]["epochs"] = int(train_cfg["epochs"])
                log.info(
                    "    %-16s RMSE %7.2f  skill %+.4f  (%d seeds)",
                    spec.name,
                    results[spec.name]["rmse"],
                    results[spec.name]["skill_vs_persistence"],
                    len(per_seed),
                )

        # Restore the shipped schedule; the rescaling above is per cell and must
        # not accumulate across the grid.
        train_cfg["epochs"] = base_schedule["epochs"]
        train_cfg["early_stopping"]["patience"] = base_schedule["patience"]
        train_cfg["early_stopping"]["min_epochs"] = base_schedule["min_epochs"]

        cells[key] = {
            "arm": arm,
            "target_coverage": coverage,
            "injection_seed": inj_seed,
            "horizon_h": horizon,
            "injection": report.to_dict(),
            "usable_rows": valid,
            "test_rows": valid["test"],
            "models": results,
        }
        # Rebuilt from scratch on every cell, which deliberately DROPS the
        # `analysis` block 17_ablation_analysis.py writes here. Do not "fix"
        # this by merging: an analysis computed over a different set of cells
        # is worse than no analysis, because it looks finished. Re-run 17 after
        # this script, always. 11_make_report.py warns when it finds cells with
        # no analysis rather than silently omitting the section.
        payload = {
            "donor": cfg.get("data.openaq.site_label") or cfg.get("data.site.city"),
            "gap_profile_source": profile.source,
            "gap_profile": profile.summary(),
            "horizon_h": horizon,
            "protected_from": str(val_end),
            "design": (
                "Both arms remove an identical number of observed hours at each coverage "
                "level; only their arrangement differs. The test period is never degraded, "
                "so every cell is scored on the same rows. Hyperparameters are fixed at the "
                "undegraded record's selections."
            ),
            "cells": cells,
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

        cfg.raw["paths"]["data_processed"] = base_processed
        cfg.raw["paths"]["checkpoints"] = base_checkpoints
        cfg.raw["paths"]["tables"] = base_tables
        if not args.keep_cells:
            shutil.rmtree(cell_dir / "processed", ignore_errors=True)

    log.info(
        "gap-injection grid complete in %.1f min -> %s",
        (time.perf_counter() - started_all) / 60.0,
        out_path,
    )
    print(f"\nwrote {out_path}")
    print("run scripts/17_ablation_analysis.py to build the figure and tables")
    return 0


if __name__ == "__main__":
    # An uncaught exception here writes its traceback to stderr and nowhere
    # else, so a grid that dies partway leaves results/logs/16_gap_injection.log
    # ending mid-epoch with no reason recorded -- which is what happened at cell
    # 66/101, and the fault turned out to be transient and unreproducible. The
    # logger is name-keyed and setup_logging has already attached the file
    # handler, so re-fetching it here puts the traceback in the log.
    #
    # The grid still aborts rather than skipping the cell and carrying on. Its
    # design is paired -- each (coverage, seed) contributes one fragmented and
    # one contiguous cell to a signed-rank test -- so a silently missing cell
    # would unbalance the pairs. Failing loudly and resuming is correct;
    # continuing past a hole is not.
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except BaseException:
        logging.getLogger("16_gap_injection").exception(
            "gap-injection grid ABORTED; the cell in progress was not recorded. "
            "Re-run with the same command to resume from it."
        )
        raise
