"""Phase 5: stratified evaluation and significance testing.

Regenerates predictions for every model on the test period, then reports:

* metrics stratified by season and by observed pollution level;
* a Diebold-Mariano test between the best deep model and the best classical
  baseline, so the headline claim carries a significance test rather than a
  bare difference in RMSE.

Model selection for the DM test uses **validation** loss, never test error --
picking the best model by its test score and then testing that score is
circular.

Run::

    python scripts/09_evaluate.py --config config.yaml
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
from src.eval.evaluate import stratified_frame, stratified_metrics
from src.eval.metrics import diebold_mariano
from src.models.baselines import (
    fit_climatology,
    predict_climatology,
    predict_persistence,
    predict_seasonal_naive,
)
from src.models.data import build_sequence_index, get_split_arrays, invert, load_features
from src.models.sequence import (
    DeviceWindowSampler,
    ModelSpec,
    _make_loss,
    build_model,
    evaluate_sampler,
)
from src.results import load_results, save_results
from src.utils import check_disk_space, load_config, resolve_device, setup_logging
from src.viz.tables import write_table


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--device", default=None, choices=["auto", "cuda", "cpu"])
    return p.parse_args()


def load_best_sequence_predictions(
    cfg, frame, horizon: int, device: str, log
) -> tuple[str, np.ndarray, np.ndarray, np.ndarray, dict] | None:
    """Rebuild predictions from the best sequence checkpoint for a horizon.

    "Best" is by mean validation loss across seeds, so the choice never consults
    the test split.

    Returns:
        ``(name, y_true, predictions, persistence, run_record)`` or None.
    """
    payload = load_results(cfg)
    runs = [
        r
        for r in payload.get("runs", [])
        if r.get("tier") == "tier3" and r.get("completed") and r.get("horizon_h") == horizon
    ]
    if not runs:
        return None

    frame_runs = pd.DataFrame(runs)
    grouped = frame_runs.groupby(["model", "variant"])["best_val_loss"].mean().sort_values()
    best_model, best_variant = grouped.index[0]
    candidates = [r for r in runs if r["model"] == best_model and r["variant"] == best_variant]
    chosen = min(candidates, key=lambda r: r["best_val_loss"])

    spec = ModelSpec(
        arch=chosen["arch"],
        hidden_size=chosen["hidden_size"],
        num_layers=chosen["num_layers"],
        window=chosen["window_h"],
        horizon=horizon,
        seed=chosen["seed"],
        attention_dim=32 if chosen["arch"] == "gru_attention" else None,
    )
    ckpt = Path(cfg.path_for("checkpoints")) / spec.run_id / "best.ckpt"
    if not ckpt.exists():
        log.warning("checkpoint missing for %s", spec.run_id)
        return None

    index = build_sequence_index(frame, cfg, horizon, spec.window, "test")
    model = build_model(spec, index.n_features, float(cfg.get("models.sequence.train.dropout")))
    state = torch.load(ckpt, map_location=device, weights_only=False)["model"]
    model.load_state_dict(state)
    model = model.to(device)

    sampler = DeviceWindowSampler(index, device)
    _, raw = evaluate_sampler(
        model, sampler, _make_loss(cfg), int(cfg.get("models.sequence.train.batch_size"))
    )
    preds = invert(cfg, raw)
    log.info(
        "best sequence model at h=%d: %s (val loss %.5f, seed %d)",
        horizon,
        spec.run_id,
        chosen["best_val_loss"],
        spec.seed,
    )
    return spec.name, index.y, preds, index.persistence, chosen


def main() -> int:
    """Run stratified evaluation and significance tests."""
    args = parse_args()
    cfg = load_config(args.config)
    log = setup_logging(cfg, "09_evaluate")
    check_disk_space(cfg)
    device = resolve_device(cfg, args.device)

    frame = load_features(cfg)
    horizons = [int(h) for h in cfg.get("task.horizons_h")]
    payload = load_results(cfg)

    strat_records: list[dict] = []
    dm_records: list[dict] = []

    for h in horizons:
        log.info("=" * 70)
        log.info("HORIZON h=%d", h)
        test = get_split_arrays(frame, cfg, h, "test")
        train = get_split_arrays(frame, cfg, h, "train")

        # Prefer the predictions stage 3 wrote to disk: they cover every classical
        # model including the trees, so the stratified table and the significance
        # test can both use the model that actually won rather than only the
        # naive rules that happen to be cheap to recompute here.
        pred_path = cfg.path_for("data_interim") / "predictions" / f"classical_h{h}.parquet"
        classical: dict[str, np.ndarray] = {}
        if pred_path.exists():
            stored = pd.read_parquet(pred_path).reindex(test.index)
            for col in stored.columns:
                if col not in {"y_true", "month_local"}:
                    classical[col] = stored[col].to_numpy(dtype=float)
            log.info("loaded %d stored classical predictions for h=%d", len(classical), h)
        else:
            log.warning("%s missing; falling back to naive baselines only", pred_path.name)
            classical["persistence"] = predict_persistence(test)
            classical["climatology"] = predict_climatology(fit_climatology(train, h), test, h)
            if h <= 24:
                classical["seasonal_naive"] = predict_seasonal_naive(test)

        # Tree predictions come from results.json rather than being refitted;
        # what is needed here is the stratification of the already-scored runs.
        for name, pred in classical.items():
            for record in stratified_metrics(
                test.y, pred, classical["persistence"], test.month_local, cfg
            ):
                strat_records.append({"model": name, "horizon_h": h, **record})

        best_seq = load_best_sequence_predictions(cfg, frame, h, device, log)
        if best_seq is not None:
            name, y_true, preds, persistence, chosen = best_seq
            for record in stratified_metrics(
                y_true,
                preds,
                persistence,
                build_sequence_index(frame, cfg, h, chosen["window_h"], "test").month_local,
                cfg,
            ):
                strat_records.append({"model": name, "horizon_h": h, **record})

            # ---- Diebold-Mariano vs the best classical baseline -----------
            classical_runs = [
                r
                for r in payload.get("runs", [])
                if r.get("tier") in {"tier1", "tier2"} and r.get("horizon_h") == h
            ]
            if classical_runs:
                best_classical = min(classical_runs, key=lambda r: r["metrics"]["rmse"])
                cname = best_classical["model"]

                # Align the two prediction vectors on their shared timestamps:
                # sequence models drop windows the tabular models keep.
                seq_index = build_sequence_index(frame, cfg, h, chosen["window_h"], "test").index
                common = test.index.intersection(seq_index)
                if len(common) > 100 and cname in classical:
                    a_pos = pd.Index(seq_index).get_indexer(common)
                    b_pos = pd.Index(test.index).get_indexer(common)
                    y_common = y_true[a_pos]
                    dm = diebold_mariano(
                        y_common,
                        preds[a_pos],
                        classical[cname][b_pos],
                        horizon=h,
                        loss=str(cfg.get("evaluation.diebold_mariano.loss", "squared")),
                        name_a=name,
                        name_b=cname,
                    )
                    dm_records.append(
                        {
                            "horizon_h": h,
                            "model_a": name,
                            "model_b": cname,
                            "n_common": len(common),
                            **asdict(dm),
                        }
                    )
                    log.info(
                        "DM h=%d: %s vs %s  stat %.3f  p %.4g  better=%s",
                        h,
                        name,
                        cname,
                        dm.statistic,
                        dm.p_value,
                        dm.better,
                    )

    strat = stratified_frame(strat_records) if strat_records else pd.DataFrame()
    if len(strat):
        strat.insert(0, "horizon_h", [r["horizon_h"] for r in strat_records])
        strat.insert(0, "model", [r["model"] for r in strat_records])
        strat.to_csv(cfg.path_for("tables") / "stratified_metrics_raw.csv", index=False)
        payload["stratified"] = strat.to_dict(orient="records")

        headline = int(cfg.get("task.headline_horizon_h"))
        view = strat[strat["horizon_h"] == headline].copy()
        display = view.assign(
            Model=view["model"],
            Stratum=view["stratum"],
            Group=view["group"],
            N=view["n"],
            RMSE=view["rmse"].round(2),
            MAE=view["mae"].round(2),
            Bias=view["bias"].round(2),
            Skill=view["skill_vs_persistence"].round(4),
        )[["Model", "Stratum", "Group", "N", "RMSE", "MAE", "Bias", "Skill"]]
        write_table(
            cfg,
            display,
            "stratified_metrics",
            caption=(
                f"Performance at {headline} hours, stratified by season and by observed "
                "pollution level. Skill within a stratum is computed against "
                "persistence restricted to that same stratum, so it measures skill "
                "rather than the stratum's intrinsic difficulty. The 65 ug/m3 cut is "
                "the Bangladesh 24-hour standard; 35 ug/m3 is the annual standard, "
                "shown as a more sensitive secondary cut."
            ),
        )
        print("\n" + "=" * 110)
        print(f"STRATIFIED METRICS (h={headline})")
        print("=" * 110)
        print(display.to_string(index=False))

    if dm_records:
        dm = pd.DataFrame(dm_records)
        payload["significance"]["diebold_mariano"] = dm.to_dict(orient="records")
        display = dm.assign(
            **{
                "h (hours)": dm["horizon_h"],
                "Deep model": dm["model_a"],
                "Classical": dm["model_b"],
                "N": dm["n_common"],
                "DM statistic": dm["statistic"].round(3),
                "p-value": dm["p_value"].map(lambda v: f"{v:.3g}"),
                "Lower loss": dm["better"],
            }
        )[["h (hours)", "Deep model", "Classical", "N", "DM statistic", "p-value", "Lower loss"]]
        write_table(
            cfg,
            display,
            "diebold_mariano",
            caption=(
                "Diebold-Mariano tests between the best sequence model and the best "
                "classical model at each horizon, on squared-error loss. Variance uses "
                "a Newey-West HAC estimator with truncation lag $h-1$ and the "
                "Harvey-Leybourne-Newbold small-sample correction. A negative "
                "statistic favours the deep model. Model selection used validation "
                "loss only."
            ),
        )
        print("\n" + "=" * 100)
        print("DIEBOLD-MARIANO TESTS")
        print("=" * 100)
        print(display.to_string(index=False))

    save_results(cfg, payload)
    log.info("wrote evaluation to %s", cfg.path_for("results_json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
