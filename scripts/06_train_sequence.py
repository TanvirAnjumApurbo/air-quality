r"""Phase 4: train the Tier 3 sequence models.

Every stochastic run is repeated across the configured seeds and reported as
mean +/- standard deviation: a single-seed deep-learning number is not a result.

The run is resumable. Each epoch writes ``last.ckpt`` atomically, so an
interrupted sweep continues from the exact epoch it stopped at, and a completed
configuration is skipped on re-invocation unless ``--force`` is given.

Examples::

    # full sweep, resuming anything already started
    python scripts/06_train_sequence.py --config config.yaml --resume auto

    # one configuration
    python scripts/06_train_sequence.py --config config.yaml \\
        --arch gru --hidden 64 --layers 1 --window 48 --horizon 24 --seed 42

    # choose the training recipe on validation loss, before the sweep
    python scripts/06_train_sequence.py --config config.yaml --tune

    # see what would run, without training
    python scripts/06_train_sequence.py --config config.yaml --dry-run
"""

from __future__ import annotations

import argparse
import itertools
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
from src.eval.metrics import all_metrics, skill_score, summarise
from src.green.energy import EnergyTracker
from src.models.data import build_sequence_index, invert, load_features
from src.models.sequence import DeviceWindowSampler as Sampler
from src.models.sequence import (
    ModelSpec,
    _make_loss,
    enumerate_specs,
    evaluate_sampler,
    filter_by_parameter_budget,
    save_history,
    train_one,
)
from src.results import (
    find_reusable_run,
    load_results,
    save_results,
    stale_width_runs,
    upsert_run,
)
from src.utils import check_disk_space, load_config, resolve_device, setup_logging
from src.viz.tables import format_mean_std, write_table


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--resume", default="auto", choices=["auto", "never", "always"])
    p.add_argument("--device", default=None, choices=["auto", "cuda", "cpu"])
    p.add_argument("--progress", default=None, choices=["tqdm", "plain"])
    p.add_argument("--max-minutes", type=float, default=None, help="wall-clock ceiling per run")
    p.add_argument(
        "--cache-gb", type=float, default=None, help="override runtime.ram_cache.budget_gb"
    )
    p.add_argument("--no-cache", action="store_true", help="disable the device-resident cache")
    p.add_argument("--dry-run", action="store_true", help="print the sweep and exit")
    p.add_argument("--force", action="store_true", help="retrain configurations already completed")
    p.add_argument(
        "--tune",
        action="store_true",
        help="search the training recipe on validation loss instead of running the sweep",
    )
    p.add_argument("--arch", default=None, help="restrict to one architecture")
    p.add_argument("--hidden", type=int, default=None)
    p.add_argument("--layers", type=int, default=None)
    p.add_argument("--window", type=int, default=None)
    p.add_argument("--horizon", type=int, nargs="*", default=None)
    p.add_argument("--seed", type=int, nargs="*", default=None)
    return p.parse_args()


def _filter(specs: list[ModelSpec], args: argparse.Namespace) -> list[ModelSpec]:
    """Apply the command-line restrictions to the sweep."""
    out = specs
    if args.arch:
        out = [s for s in out if s.arch == args.arch]
    if args.hidden is not None:
        out = [s for s in out if s.hidden_size == args.hidden]
    if args.layers is not None:
        out = [s for s in out if s.num_layers == args.layers]
    if args.window is not None:
        out = [s for s in out if s.window == args.window]
    if args.horizon:
        out = [s for s in out if s.horizon in set(args.horizon)]
    if args.seed:
        out = [s for s in out if s.seed in set(args.seed)]
    return out


def _run_recipe_search(
    cfg, frame, device: str, progress: str, max_minutes: float, args: argparse.Namespace, log
) -> int:
    """Search the training recipe on validation RMSE and report the winner.

    Test data is never touched. Each candidate recipe is trained on the reduced
    grid in ``models.sequence.tuning``; checkpoints are namespaced per recipe so
    one candidate cannot resume another's weights.

    Ranking is on validation RMSE in ug/m3, **not** on the training loss. The
    grid varies ``huber_delta``, and Huber with a different delta is a different
    function: for every residual above delta it assigns a mechanically smaller
    value. Ranking recipes by that loss therefore rewards the smaller delta for
    reasons unrelated to forecast quality. Measured here: on loss, delta=0.3 beat
    delta=1.0 by 36% (0.067 vs 0.105); on validation RMSE the two differ by 0.1%,
    in the opposite direction. RMSE is invariant to the loss shape, so it is the
    only sound criterion when the loss itself is a search dimension.

    Args:
        cfg: Loaded configuration.
        frame: Built feature frame.
        device: Compute device.
        progress: Progress backend.
        max_minutes: Per-run wall-clock ceiling.
        args: Parsed command line.
        log: Logger.

    Returns:
        Process exit status.
    """
    tuning = cfg.get("models.sequence.tuning")
    if not bool(tuning.get("enabled", False)):
        log.error(
            "models.sequence.tuning.enabled is false in %s -- the recipe is inherited "
            "from the primary city on purpose. Tune there instead.",
            cfg.path,
        )
        return 1

    grid = tuning["grid"]
    keys = sorted(grid)
    combos = [
        dict(zip(keys, values, strict=True))
        for values in itertools.product(*(grid[k] for k in keys))
    ]

    specs = [
        ModelSpec(
            arch=arch,
            hidden_size=int(hidden),
            num_layers=int(layers),
            window=int(window),
            horizon=int(horizon),
            seed=int(seed),
        )
        for arch in tuning["archs"]
        for hidden in tuning["hidden_sizes"]
        for layers in tuning["num_layers"]
        for window in tuning["windows_h"]
        for horizon in tuning["horizons_h"]
        for seed in tuning["seeds"]
    ]

    total = len(combos) * len(specs)
    log.info(
        "recipe search: %d recipes x %d configurations = %d runs", len(combos), len(specs), total
    )
    if args.dry_run:
        print(
            f"\n{total} tuning runs would execute ({len(combos)} recipes x {len(specs)} configs):"
        )
        for combo in combos:
            print("  " + "  ".join(f"{k}={v}" for k, v in sorted(combo.items())))
        return 0

    cache: dict[tuple[int, int, str], object] = {}

    def get_index(horizon: int, window: int, split: str):  # noqa: ANN202
        key = (horizon, window, split)
        if key not in cache:
            cache[key] = build_sequence_index(frame, cfg, horizon, window, split)
        return cache[key]

    train_cfg = cfg.raw["models"]["sequence"]["train"]
    baseline = {k: train_cfg[k] for k in keys}
    rows: list[dict] = []
    done = 0

    batch_size = int(cfg.get("models.sequence.train.batch_size"))

    for combo in combos:
        train_cfg.update(combo)
        tag = "tune_" + "_".join(f"{k}{combo[k]:g}" for k in keys)
        losses: list[float] = []
        rmses: list[float] = []
        loss_fn = _make_loss(cfg)

        for spec in specs:
            done += 1
            log.info("[%d/%d] %s | %s", done, total, tag, spec.run_id)
            model, result = train_one(
                spec,
                get_index(spec.horizon, spec.window, "train"),
                get_index(spec.horizon, spec.window, "val"),
                cfg,
                device,
                log,
                resume=args.resume,
                progress=progress,
                max_minutes=max_minutes,
                run_tag=tag,
            )
            losses.append(result.best_val_loss)

            val_index = get_index(spec.horizon, spec.window, "val")
            _, raw_val = evaluate_sampler(model, Sampler(val_index, device), loss_fn, batch_size)
            val_rmse = all_metrics(val_index.y, invert(cfg, raw_val))["rmse"]
            rmses.append(val_rmse)

            log.info(
                "  %s | %s  val RMSE %.3f, loss %.5f at epoch %d of %d run (%s)",
                tag,
                spec.run_id,
                val_rmse,
                result.best_val_loss,
                result.best_epoch + 1,
                result.total_epochs,
                result.stopped_reason,
            )
            if device == "cuda":
                torch.cuda.empty_cache()

        rows.append(
            {
                **combo,
                "mean_val_rmse": float(np.mean(rmses)),
                "mean_val_loss": float(np.mean(losses)),
                "n_runs": len(rmses),
            }
        )

    train_cfg.update(baseline)  # leave the loaded config as we found it

    table = pd.DataFrame(rows).sort_values("mean_val_rmse").reset_index(drop=True)
    table.to_csv(cfg.path_for("tables") / "sequence_recipe_search.csv", index=False)

    best = table.iloc[0]
    payload = load_results(cfg)
    payload["sequence_recipe_search"] = {
        "selected_on": "validation RMSE (ug/m3), mean over the reduced grid",
        "criterion_note": (
            "RMSE, not training loss: huber_delta is a search dimension, and Huber "
            "loss is not comparable across delta. mean_val_loss is retained per "
            "recipe for reference but is not the ranking key."
        ),
        "n_recipes": len(combos),
        "n_configs_per_recipe": len(specs),
        "grid": {k: list(grid[k]) for k in keys},
        "results": table.to_dict(orient="records"),
        "best": {k: best[k] for k in [*keys, "mean_val_rmse", "mean_val_loss"]},
    }
    save_results(cfg, payload)

    print("\n" + "=" * 78)
    print("SEQUENCE RECIPE SEARCH — ranked by mean validation RMSE (test never read)")
    print("=" * 78)
    print(table.to_string(index=False))
    print("\nPaste into models.sequence.train in both configs, then run the full sweep:\n")
    for key in keys:
        print(f"      {key}: {best[key]:g}")
    return 0


def main() -> int:
    """Run the sequence-model sweep."""
    args = parse_args()
    cfg = load_config(args.config)
    log = setup_logging(cfg, "06_train_sequence")
    check_disk_space(cfg)

    device = resolve_device(cfg, args.device)
    progress = args.progress or str(cfg.get("runtime.progress.backend", "tqdm"))
    max_minutes = args.max_minutes or float(cfg.get("runtime.max_minutes_per_run", 10))
    if args.cache_gb is not None:
        cfg.raw["runtime"]["ram_cache"]["budget_gb"] = args.cache_gb
    if args.no_cache:
        cfg.raw["runtime"]["ram_cache"]["enabled"] = False

    log.info("device=%s progress=%s max_minutes=%.1f", device, progress, max_minutes)

    frame = load_features(cfg)

    if args.tune:
        return _run_recipe_search(cfg, frame, device, progress, max_minutes, args, log)

    horizons = args.horizon or [int(h) for h in cfg.get("task.horizons_h")]
    seeds = args.seed or [int(s) for s in cfg.get("seeds.multi")]

    specs = _filter(enumerate_specs(cfg, horizons, seeds), args)
    n_features = len(build_sequence_index(frame, cfg, horizons[0], 24, "train").matrix[0])
    specs, excluded = filter_by_parameter_budget(specs, cfg, n_features, log)

    payload = load_results(cfg)
    payload.setdefault("green", {})["excluded_over_budget"] = excluded

    stale = stale_width_runs(payload, "tier3", n_features)
    reusable = sum(
        1
        for s in specs
        if find_reusable_run(
            payload,
            model=s.name,
            variant=f"w{s.window}",
            horizon_h=s.horizon,
            seed=s.seed,
            n_features=n_features,
        )
    )

    if args.dry_run:
        print(
            f"\n{len(specs)} runs would execute ({len(seeds)} seeds x "
            f"{len(horizons)} horizons x architectures/windows):"
        )
        print(
            f"  {reusable} already complete at the current width ({n_features} channels) "
            f"and would be reused; {len(specs) - reusable} would train."
        )
        if stale:
            widths = sorted({r.get("n_features") for r in stale})
            print(
                f"  WARNING: {len(stale)} recorded tier3 runs are at a different input "
                f"width ({', '.join('unrecorded' if w is None else str(w) for w in widths)}) "
                f"and will be RETRAINED."
            )
        for s in specs[:40]:
            print(f"  {s.run_id}")
        if len(specs) > 40:
            print(f"  ... and {len(specs) - 40} more")
        print(
            f"\n{len(excluded)} configurations excluded by the {cfg.get('models.sequence.max_params')}-parameter budget:"
        )
        for e in excluded:
            print(f"  {e['name']}: {e['params']:,}")
        return 0

    log.info("sweep: %d runs, %d excluded by parameter budget", len(specs), len(excluded))

    if stale:
        widths = sorted({r.get("n_features") for r in stale})
        log.warning(
            "%d recorded tier3 runs were trained on a different input width (%s, now %d) "
            "and will be RETRAINED, not reused. They are the record of a different "
            "experiment; reporting them alongside runs at the current width would compare "
            "two channel sets and call it a model comparison.",
            len(stale),
            ", ".join("unrecorded" if w is None else str(w) for w in widths),
            n_features,
        )

    # Cache window indices: they depend only on (horizon, window, split), so
    # rebuilding per seed would repeat identical work five times.
    cache: dict[tuple[int, int, str], object] = {}

    def get_index(horizon: int, window: int, split: str):  # noqa: ANN202
        key = (horizon, window, split)
        if key not in cache:
            cache[key] = build_sequence_index(frame, cfg, horizon, window, split)
        return cache[key]

    rows: list[dict] = []
    loss_fn = _make_loss(cfg)
    batch_size = int(cfg.get("models.sequence.train.batch_size"))

    for i, spec in enumerate(specs, start=1):
        existing = find_reusable_run(
            payload,
            model=spec.name,
            variant=f"w{spec.window}",
            horizon_h=spec.horizon,
            seed=spec.seed,
            n_features=n_features,
        )
        if existing and not args.force:
            log.info("[%d/%d] %s: already complete, skipping", i, len(specs), spec.run_id)
            rows.append(
                {
                    "model": spec.name,
                    "window": spec.window,
                    "horizon_h": spec.horizon,
                    "seed": spec.seed,
                    **existing["metrics"],
                    "n_params": existing["n_params"],
                }
            )
            continue

        log.info("[%d/%d] %s", i, len(specs), spec.run_id)
        train_index = get_index(spec.horizon, spec.window, "train")
        val_index = get_index(spec.horizon, spec.window, "val")
        test_index = get_index(spec.horizon, spec.window, "test")

        tracker = EnergyTracker(cfg, f"seq_{spec.run_id}", log)
        with tracker:
            model, result = train_one(
                spec,
                train_index,
                val_index,
                cfg,
                device,
                log,
                resume=args.resume,
                progress=progress,
                max_minutes=max_minutes,
            )
        save_history(result)

        test_sampler = Sampler(test_index, device)
        started = time.perf_counter()
        _, raw_preds = evaluate_sampler(model, test_sampler, loss_fn, batch_size)
        inference_seconds = time.perf_counter() - started

        preds = invert(cfg, raw_preds, log, spec.name)
        metrics = all_metrics(test_index.y, preds)

        persistence_rmse = all_metrics(test_index.y, test_index.persistence)["rmse"]
        metrics["skill_vs_persistence"] = skill_score(metrics["rmse"], persistence_rmse)

        record = {
            "model": spec.name,
            "tier": "tier3",
            "variant": f"w{spec.window}",
            "arch": spec.arch,
            "hidden_size": spec.hidden_size,
            "num_layers": spec.num_layers,
            "window_h": spec.window,
            "horizon_h": spec.horizon,
            "seed": spec.seed,
            "split": "test",
            "completed": True,
            "n_features": n_features,
            "n_params": result.n_params,
            "best_val_loss": result.best_val_loss,
            "best_epoch": result.best_epoch,
            "total_epochs": result.total_epochs,
            "train_seconds": result.train_seconds,
            "inference_seconds": inference_seconds,
            "stopped_reason": result.stopped_reason,
            "energy": tracker.summary(),
            "metrics": metrics,
        }
        upsert_run(payload, record)
        save_results(cfg, payload)

        rows.append(
            {
                "model": spec.name,
                "window": spec.window,
                "horizon_h": spec.horizon,
                "seed": spec.seed,
                "n_params": result.n_params,
                **metrics,
            }
        )
        log.info(
            "  %s  RMSE %7.2f  MAE %7.2f  R2 %6.3f  skill %+.4f  (%d params, %.0f s, %s)",
            spec.run_id,
            metrics["rmse"],
            metrics["mae"],
            metrics["r2"],
            metrics["skill_vs_persistence"],
            result.n_params,
            result.train_seconds,
            result.stopped_reason,
        )
        if device == "cuda":
            torch.cuda.empty_cache()

    save_results(cfg, payload)

    if not rows:
        log.warning("no runs executed")
        return 0

    # ---- aggregate across seeds -------------------------------------------
    raw = pd.DataFrame(rows)
    raw.to_csv(cfg.path_for("tables") / "sequence_runs_raw.csv", index=False)

    agg_rows = []
    for (model, window, horizon), group in raw.groupby(["model", "window", "horizon_h"]):
        entry = {
            "model": model,
            "window_h": window,
            "horizon_h": horizon,
            "n_params": int(group["n_params"].iloc[0]),
            "n_seeds": len(group),
        }
        for metric in ("rmse", "mae", "r2", "smape", "skill_vs_persistence"):
            stats = summarise(group[metric].tolist())
            entry[f"{metric}_mean"] = stats["mean"]
            entry[f"{metric}_std"] = stats["std"]
        agg_rows.append(entry)

    agg = pd.DataFrame(agg_rows).sort_values(["horizon_h", "rmse_mean"])
    payload.setdefault("aggregates", {})["sequence"] = agg.to_dict(orient="records")
    save_results(cfg, payload)

    display = agg.assign(
        RMSE=lambda d: [
            format_mean_std(m, s) for m, s in zip(d["rmse_mean"], d["rmse_std"], strict=True)
        ],
        MAE=lambda d: [
            format_mean_std(m, s) for m, s in zip(d["mae_mean"], d["mae_std"], strict=True)
        ],
        R2=lambda d: [
            format_mean_std(m, s, 3) for m, s in zip(d["r2_mean"], d["r2_std"], strict=True)
        ],
        Skill=lambda d: [
            format_mean_std(m, s, 4)
            for m, s in zip(
                d["skill_vs_persistence_mean"], d["skill_vs_persistence_std"], strict=True
            )
        ],
    )[
        ["model", "window_h", "horizon_h", "n_params", "n_seeds", "RMSE", "MAE", "R2", "Skill"]
    ].rename(
        columns={
            "model": "Model",
            "window_h": "Window (h)",
            "horizon_h": "h (hours)",
            "n_params": "Params",
            "n_seeds": "Seeds",
        }
    )
    write_table(
        cfg,
        display,
        "tier3_sequence",
        caption=(
            "Tier 3 sequence models, mean $\\pm$ standard deviation across "
            f"{len(seeds)} seeds. All models respect the "
            f"{cfg.get('models.sequence.max_params'):,}-parameter budget."
        ),
    )

    print("\n" + "=" * 110)
    print(f"TIER 3 SEQUENCE MODELS ({len(seeds)} seeds, mean ± std)")
    print("=" * 110)
    print(display.to_string(index=False))
    if excluded:
        print(f"\nExcluded by the {cfg.get('models.sequence.max_params'):,}-parameter budget:")
        for e in excluded:
            print(f"  {e['name']:<24} {e['params']:>9,} params")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
