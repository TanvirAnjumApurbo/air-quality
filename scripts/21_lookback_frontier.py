r"""Phase 8b: what shortening the lookback costs, and what it buys.

``20_missingness_law.py`` shows that a gap sterilises the ``lookback + horizon``
hours behind it, so the configured 168-hour reach leaves 23.6% of Dhaka's test
hours and 27.5% of Wanliu's unforecastable by any model in the study. Capping the
reach recovers those hours. It also removes real signal -- ``pm25_lag_168`` is
the weekly cycle. This script measures the trade instead of assuming it.

Three arms, and the third is what makes the first two interpretable:

* **C** -- the status quo: full feature set, floor at 168.
* **B** -- capped features, floor held at 168. Same rows as C, fewer predictors,
  so the difference from C is feature richness alone.
* **A** -- capped features, floor at the cap. More rows than B, identical
  predictors, so the difference from B is supervision volume alone.

Which makes the decomposition exact::

    MSE_A - MSE_C  =  (MSE_A - MSE_B)  +  (MSE_B - MSE_C)
                       supervision volume    feature richness

in mean squared error. RMSE differences are not additive, so the arithmetic is
done in MSE and displayed as RMSE.

Two things are held fixed on purpose. Hyperparameters come from the undegraded
main run and are never re-tuned per arm: letting search compensate for a changed
feature set is a second uncontrolled variable. And every arm is scored over one
**fixed universe** of test hours, with a declared fallback answering the rows an
arm cannot serve -- an arm that simply dropped its unservable hours would be
scored on an easier subset than its rivals, which is the entire problem this
contribution exists to expose.

The sequence tier gets no feature-richness arm, and that is not an omission:
``sequence_channel_columns`` already excludes every derived-history column, so a
cap cannot change tier-3 input by one channel. For tier 3 the only thing a cap
moves is the validity floor, and its runs are a function of the sterilisation
radius ``max(lookback, window - 1) + horizon`` alone. Configurations sharing a
radius share a run.

Run::

    python scripts/21_lookback_frontier.py --config config.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from src.eval.availability import (
    availability_record,
    effective_floor,
    evaluation_universe,
)
from src.eval.metrics import all_metrics
from src.features.build_features import history_floor, max_backward_dependency
from src.features.pipeline import build_feature_matrix
from src.models.baselines import fit_climatology, predict_climatology, predict_persistence
from src.models.data import build_sequence_index, get_split_arrays, invert
from src.models.sequence import ModelSpec, evaluate_sampler, train_one
from src.models.trees import fit_at_fixed_params
from src.results import load_results, main_runs, save_results, upsert_run
from src.utils import Config, check_disk_space, load_config, resolve_device, set_seed, setup_logging

#: Label carried by every run this script writes, so the headline experiment
#: cannot select one of them. See ``src.results.main_runs``.
EXPERIMENT = "lookback_frontier"


@dataclass(frozen=True)
class Arm:
    """One point of the frontier.

    Attributes:
        label: Short name, ``A``/``B``/``C`` with the cap appended.
        lookback_h: Feature cap in hours, or None for the full set.
        floor_h: History floor in hours, or None to follow the cap.
        note: What this arm isolates, for the table.
    """

    label: str
    lookback_h: int | None
    floor_h: int | None
    note: str


def arms_for(lookbacks: list[int], deepest: int) -> list[Arm]:
    """Build the arm list for a set of lookback caps.

    Args:
        lookbacks: Caps to sweep, shortest last.
        deepest: The uncapped backward dependency, used as arm B's floor.

    Returns:
        Arm C once, then arms B and A for each cap.
    """
    arms = [Arm("C", None, None, "status quo: full features, floor 168")]
    for cap in lookbacks:
        arms.append(Arm(f"B{cap}", cap, deepest, f"capped features, floor held at {deepest}"))
        arms.append(Arm(f"A{cap}", cap, None, f"capped features, floor at {cap}"))
    return arms


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--lookbacks", type=int, nargs="*", default=[48, 24])
    p.add_argument("--windows", type=int, nargs="*", default=[48, 24])
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--progress", default="plain", choices=["tqdm", "plain"])
    p.add_argument("--dry-run", action="store_true", help="list the arms and exit")
    p.add_argument("--force", action="store_true", help="refit arms already recorded")
    p.add_argument("--skip-sequence", action="store_true", help="tier 2 only, for a fast pass")
    return p.parse_args()


def variant_config(cfg: Config, arm: Arm, cell_dir: Path) -> Config:
    """A config with the arm's cap and floor set and its writes redirected.

    Args:
        cfg: The base configuration.
        arm: The arm to build.
        cell_dir: Directory this arm may write under.

    Returns:
        A new configuration; the base is not mutated.
    """
    import copy

    raw = copy.deepcopy(cfg.raw)
    raw["features"]["lookback_h"] = arm.lookback_h
    raw["features"]["history_floor_h"] = arm.floor_h
    raw["paths"]["data_processed"] = str(cell_dir / "processed")
    raw["paths"]["checkpoints"] = str(cell_dir / "checkpoints")
    raw["paths"]["tables"] = str(cell_dir / "tables")
    return Config(raw=raw, path=cfg.path)


def tuned_params(payload: dict, model: str, horizon: int) -> dict:
    """Hyperparameters selected on the undegraded main run.

    Never re-tuned per arm: letting a search compensate for a changed feature
    set would be a second uncontrolled variable, and fixing them is also what a
    practitioner moving a tuned pipeline to a shorter reach actually does.

    Args:
        payload: The city's ``results.json`` contents.
        model: Model name.
        horizon: Forecast horizon.

    Returns:
        The stored ``best_params``, or an empty mapping.
    """
    for record in main_runs(payload):
        if record.get("model") == model and record.get("horizon_h") == horizon:
            return dict(record.get("best_params") or {})
    return {}


def align_to_universe(
    universe_index: pd.DatetimeIndex, served_index: pd.DatetimeIndex, values: np.ndarray
) -> np.ndarray:
    """Place a model's predictions into the universe, NaN where it cannot answer.

    Args:
        universe_index: Timestamps of the fixed evaluation universe.
        served_index: Timestamps the model produced predictions for.
        values: Predictions aligned to ``served_index``.

    Returns:
        A universe-length vector.
    """
    out = pd.Series(np.nan, index=universe_index, dtype=float)
    out.loc[served_index.intersection(universe_index)] = pd.Series(
        values, index=served_index
    ).reindex(served_index.intersection(universe_index))
    return out.to_numpy()


def main() -> int:
    """Run the lookback frontier."""
    args = parse_args()
    cfg = load_config(args.config)
    log = setup_logging(cfg, "21_lookback_frontier")
    check_disk_space(cfg)

    horizon = int(cfg.get("task.headline_horizon_h"))
    deepest = max_backward_dependency(cfg)
    arms = arms_for(list(args.lookbacks), deepest)

    print("\n" + "=" * 78)
    print(f"LOOKBACK FRONTIER — {len(arms)} arms at h={horizon}, deepest feature {deepest} h")
    print("=" * 78)
    for arm in arms:
        print(f"  {arm.label:<5} cap={arm.lookback_h!s:<5} floor={arm.floor_h!s:<5} {arm.note}")
    if args.dry_run:
        return 0

    interim = cfg.path_for("data_interim")
    pm = pd.read_parquet(interim / str(cfg.get("data.files.target")))
    met = pd.read_parquet(interim / str(cfg.get("data.files.meteorology")))
    cell_root = interim / "lookback"
    cell_root.mkdir(parents=True, exist_ok=True)

    payload = load_results(cfg)
    device = resolve_device(cfg, args.device)
    tabular_models = list(cfg.get("ablation.gap_injection.tabular_models"))
    seq_specs = list(cfg.get("ablation.gap_injection.sequence_models"))
    model_seeds = list(cfg.get("ablation.gap_injection.model_seeds"))

    universe_index: pd.DatetimeIndex | None = None
    y_universe: np.ndarray | None = None
    reference: np.ndarray | None = None
    rows: list[dict] = []
    predictions: dict[str, np.ndarray] = {}
    radius_cache: dict[int, dict[str, np.ndarray]] = {}

    from src.models.sequence import DeviceWindowSampler as Sampler
    from src.models.sequence import _make_loss

    for arm in arms:
        check_disk_space(cfg)
        cell_dir = cell_root / arm.label
        vcfg = variant_config(cfg, arm, cell_dir)
        floor = history_floor(vcfg)
        cap = max_backward_dependency(vcfg)
        log.info("=== arm %s: cap %s, floor %d ===", arm.label, cap, floor)

        built = build_feature_matrix(pm, met, vcfg, log, write=True)
        frame = built.frame

        # The universe is a property of the record, not of the arm: it uses no
        # history condition and a fixed purge depth. Assert that rather than
        # trust it -- if it moved between arms, nothing below would be comparable.
        this_universe = evaluation_universe(frame, "test", horizon)
        if universe_index is None:
            universe_index = frame.index[this_universe]
            y_universe = frame.loc[this_universe, f"target_h{horizon}"].to_numpy(dtype=float)
            reference = frame.loc[this_universe, str(cfg.get("features.target"))].to_numpy(
                dtype=float
            )
            log.info(
                "universe: %d test hours, persistence RMSE %.4f",
                len(universe_index),
                all_metrics(y_universe, reference)["rmse"],
            )
        elif not frame.index[this_universe].equals(universe_index):
            log.error("arm %s changed the evaluation universe; refusing to continue", arm.label)
            return 1

        train = get_split_arrays(frame, vcfg, horizon, "train")
        val = get_split_arrays(frame, vcfg, horizon, "val")
        test = get_split_arrays(frame, vcfg, horizon, "test")
        avail = availability_record(frame, "test", horizon, floor)
        log.info(
            "  served %d of %d universe hours (%.4f)",
            avail.n_served,
            avail.n_universe,
            avail.availability,
        )

        # ---- tier 1 -------------------------------------------------------
        clim = fit_climatology(train, horizon, vcfg)
        for name, pred in (
            ("persistence", predict_persistence(test)),
            ("climatology", predict_climatology(clim, test, horizon)),
        ):
            key = f"{name}|{arm.label}"
            predictions[key] = align_to_universe(universe_index, test.index, pred)

        # ---- tier 2 -------------------------------------------------------
        for name in tabular_models:
            params = tuned_params(payload, name, horizon)
            model = fit_at_fixed_params(name, params, vcfg, train, val, int(model_seeds[0]))
            if model is None:
                log.warning("  %s unavailable, skipped", name)
                continue
            pred = invert(vcfg, model.predict(test.x))
            key = f"{name}|{arm.label}"
            predictions[key] = align_to_universe(universe_index, test.index, pred)
            m = all_metrics(test.y, pred)
            rows.append(
                {
                    "arm": arm.label,
                    "model": name,
                    "tier": "tier2",
                    "lookback_h": arm.lookback_h,
                    "history_floor_h": floor,
                    "radius_h": floor + horizon,
                    "n_predictors": len(train.feature_names),
                    "train_rows": len(train.y),
                    "availability": avail.availability,
                    "rmse_served": m["rmse"],
                }
            )
            log.info("    %-14s RMSE %7.2f  (served rows)", name, m["rmse"])
            upsert_run(
                payload,
                {
                    "model": name,
                    "tier": "tier2",
                    "variant": f"standard|L{cap}|{arm.label}",
                    "horizon_h": horizon,
                    "seed": int(model_seeds[0]),
                    "split": "test",
                    "experiment": EXPERIMENT,
                    "arm": arm.label,
                    "lookback_h": arm.lookback_h,
                    "history_floor_h": floor,
                    "sterilisation_radius_h": floor + horizon,
                    "availability": avail.availability,
                    "n_features": len(train.feature_names),
                    "metrics": m,
                },
            )

        # ---- tier 3 -------------------------------------------------------
        if args.skip_sequence:
            continue
        loss_fn = _make_loss(vcfg)
        batch = int(cfg.get("models.sequence.train.batch_size", 256))
        for spec_cfg in seq_specs:
            for window in args.windows:
                radius = effective_floor(floor, window) + horizon

                # ModelSpec carries the seed, and `name` is a derived property --
                # so a spec is built per seed inside the loop below, and this one
                # exists only to name the cache entry.
                def _spec(seed: int, entry: dict = spec_cfg, w: int = window) -> ModelSpec:
                    return ModelSpec(
                        arch=str(entry["arch"]),
                        hidden_size=int(entry.get("hidden_size", 0)),
                        num_layers=int(entry.get("num_layers", 1)),
                        window=int(w),
                        horizon=horizon,
                        seed=int(seed),
                        kernel_size=25 if entry["arch"] == "dlinear" else None,
                    )

                spec = _spec(int(model_seeds[0]))
                cache_key = f"{spec.name}|w{window}|R{radius}"
                if cache_key in radius_cache and not args.force:
                    predictions[f"{spec.name}|{arm.label}"] = radius_cache[cache_key]["pred"]
                    log.info("    %-14s reused from radius %d", spec.name, radius)
                    continue

                train_idx = build_sequence_index(frame, vcfg, horizon, window, "train")
                val_idx = build_sequence_index(frame, vcfg, horizon, window, "val")
                test_idx = build_sequence_index(frame, vcfg, horizon, window, "test")
                if len(train_idx) < 100 or len(val_idx) < 50:
                    log.warning("    %s w%d: too few windows, skipped", spec.name, window)
                    continue

                per_seed_pred, per_seed_metrics = [], []
                for seed in model_seeds:
                    set_seed(int(seed), cfg)
                    model, _ = train_one(
                        _spec(int(seed)),
                        train_idx,
                        val_idx,
                        vcfg,
                        device,
                        log,
                        resume="auto",
                        progress=args.progress,
                        max_minutes=float(cfg.get("runtime.max_minutes_per_run", 10)),
                        run_tag=f"{arm.label}_w{window}",
                    )
                    _, raw = evaluate_sampler(model, Sampler(test_idx, device), loss_fn, batch)
                    preds = invert(vcfg, raw)
                    per_seed_pred.append(preds)
                    per_seed_metrics.append(all_metrics(test_idx.y, preds))

                if not per_seed_pred:
                    continue
                mean_pred = np.mean(np.vstack(per_seed_pred), axis=0)
                aligned = align_to_universe(universe_index, test_idx.index, mean_pred)
                radius_cache[cache_key] = {"pred": aligned}
                predictions[f"{spec.name}_w{window}|{arm.label}"] = aligned
                m = {
                    k: float(np.mean([x[k] for x in per_seed_metrics]))
                    for k in ("rmse", "mae", "r2", "smape")
                }
                seq_avail = availability_record(frame, "test", horizon, floor, window)
                rows.append(
                    {
                        "arm": arm.label,
                        "model": f"{spec.name}_w{window}",
                        "tier": "tier3",
                        "lookback_h": arm.lookback_h,
                        "history_floor_h": floor,
                        "radius_h": radius,
                        "n_predictors": test_idx.n_features,
                        "train_rows": len(train_idx),
                        "availability": seq_avail.availability,
                        "rmse_served": m["rmse"],
                    }
                )
                log.info(
                    "    %-14s w%-4d R=%-4d RMSE %7.2f  avail %.4f",
                    spec.name,
                    window,
                    radius,
                    m["rmse"],
                    seq_avail.availability,
                )
                upsert_run(
                    payload,
                    {
                        "model": spec.name,
                        "tier": "tier3",
                        "variant": f"w{window}|R{radius}|{arm.label}",
                        "horizon_h": horizon,
                        "seed": int(model_seeds[0]),
                        "split": "test",
                        "experiment": EXPERIMENT,
                        "arm": arm.label,
                        "lookback_h": arm.lookback_h,
                        "history_floor_h": floor,
                        "sterilisation_radius_h": radius,
                        "availability": seq_avail.availability,
                        "n_features": test_idx.n_features,
                        "n_seeds": len(per_seed_pred),
                        "metrics": m,
                    },
                )

    save_results(cfg, payload)
    frontier = pd.DataFrame(rows)
    out = cfg.path_for("results") / "lookback_frontier.json"
    out.write_text(
        json.dumps(
            {
                "horizon_h": horizon,
                "deepest_feature_h": deepest,
                "n_universe": len(universe_index) if universe_index is not None else 0,
                "reference_rmse_universe": (
                    all_metrics(y_universe, reference)["rmse"] if y_universe is not None else None
                ),
                "arms": [
                    {
                        "label": a.label,
                        "lookback_h": a.lookback_h,
                        "floor_h": a.floor_h,
                        "note": a.note,
                    }
                    for a in arms
                ],
                "rows": frontier.to_dict(orient="records"),
                "predictions_available": sorted(predictions),
                "decomposition_note": (
                    "MSE_A - MSE_C = (MSE_A - MSE_B) + (MSE_B - MSE_C): supervision volume "
                    "plus feature richness, exactly. RMSE differences are not additive."
                ),
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    np.savez_compressed(
        cfg.path_for("results") / "lookback_frontier_predictions.npz",
        universe_index=np.array([str(t) for t in universe_index])
        if universe_index is not None
        else np.array([]),
        y_true=y_universe if y_universe is not None else np.array([]),
        reference=reference if reference is not None else np.array([]),
        **predictions,
    )
    log.info("wrote %s and the prediction bundle", out)

    if not frontier.empty:
        print("\n" + frontier.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
