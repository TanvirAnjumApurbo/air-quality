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

    # see what would run, without training
    python scripts/06_train_sequence.py --config config.yaml --dry-run
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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
from src.results import load_results, save_results, upsert_run
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
    horizons = args.horizon or [int(h) for h in cfg.get("task.horizons_h")]
    seeds = args.seed or [int(s) for s in cfg.get("seeds.multi")]

    specs = _filter(enumerate_specs(cfg, horizons, seeds), args)
    n_features = len(build_sequence_index(frame, cfg, horizons[0], 24, "train").matrix[0])
    specs, excluded = filter_by_parameter_budget(specs, cfg, n_features, log)

    payload = load_results(cfg)
    payload.setdefault("green", {})["excluded_over_budget"] = excluded

    if args.dry_run:
        print(
            f"\n{len(specs)} runs would execute ({len(seeds)} seeds x "
            f"{len(horizons)} horizons x architectures/windows):"
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
        existing = next(
            (
                r
                for r in payload.get("runs", [])
                if r.get("model") == spec.name
                and r.get("variant") == f"w{spec.window}"
                and r.get("horizon_h") == spec.horizon
                and r.get("seed") == spec.seed
                and r.get("completed")
            ),
            None,
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

        preds = invert(cfg, raw_preds)
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
