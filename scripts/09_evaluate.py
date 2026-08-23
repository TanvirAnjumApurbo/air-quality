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
from src.eval.metrics import (
    block_bootstrap_rmse,
    diebold_mariano,
    holm_bonferroni,
    model_confidence_set,
)
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
    try:
        model.load_state_dict(state)
    except RuntimeError as exc:
        # Almost always a checkpoint predating a change to the channel set: the
        # weights encode the input width they were trained at. Loading it anyway
        # is not an option, and the raw shape-mismatch traceback does not say
        # what to do about it.
        raise RuntimeError(
            f"{spec.run_id}: checkpoint does not match the current model shape. This "
            f"checkpoint was trained with a different input width than the "
            f"{index.n_features} channels "
            f"features.sequence_channels.mode='{cfg.get('features.sequence_channels.mode')}' "
            f"now produces. Delete {cfg.path_for('checkpoints')} and re-run "
            f"scripts/06_train_sequence.py.\n  original error: {exc}"
        ) from exc
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
    ci_records: list[dict] = []
    mcs_records: list[dict] = []

    boot_cfg = cfg.get("evaluation.bootstrap_ci", {})
    boot_enabled = bool(boot_cfg.get("enabled", True))
    boot_kwargs = {
        "n_resamples": int(boot_cfg.get("n_resamples", 1000)),
        "block_size": int(boot_cfg.get("block_size_h", 24)),
        "alpha": float(boot_cfg.get("alpha", 0.05)),
        "seed": int(cfg.get("seeds.global", 42)),
    }
    mcs_cfg = cfg.get("evaluation.model_confidence_set", {})

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
            classical["climatology"] = predict_climatology(fit_climatology(train, h, cfg), test, h)
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

        # ---- interval estimates for every reported RMSE --------------------
        # evaluation.bootstrap_ci has been configured since the first run and was
        # never actually called, so no reported RMSE carried an interval. A point
        # estimate with no spread invites the reader to treat a 0.5 ug/m3 gap as
        # a result when the seed-to-seed spread alone is twice that.
        if boot_enabled:
            scored = {n: (test.y, p, test.index) for n, p in classical.items()}
            if best_seq is not None:
                scored[best_seq[0]] = (best_seq[1], best_seq[2], None)
            for model_name, (truth, pred, _) in scored.items():
                ci = block_bootstrap_rmse(truth, pred, **boot_kwargs)
                ci_records.append(
                    {
                        "horizon_h": h,
                        "model": model_name,
                        "rmse": ci.point,
                        "ci_low": ci.lower,
                        "ci_high": ci.upper,
                        "alpha": ci.alpha,
                        "n_resamples": ci.n_resamples,
                        "block_size_h": ci.block_size,
                    }
                )

        # ---- model confidence set ------------------------------------------
        # Answers "which models cannot be separated from the best", which is the
        # question a ranked table is read as answering but does not address.
        if bool(mcs_cfg.get("enabled", True)):
            aligned: dict[str, np.ndarray] = {}
            if best_seq is not None:
                seq_idx = build_sequence_index(frame, cfg, h, best_seq[4]["window_h"], "test").index
                common = test.index.intersection(seq_idx)
                a_pos = pd.Index(seq_idx).get_indexer(common)
                b_pos = pd.Index(test.index).get_indexer(common)
                truth = best_seq[1][a_pos]
                aligned[best_seq[0]] = (truth - best_seq[2][a_pos]) ** 2
                for model_name, pred in classical.items():
                    aligned[model_name] = (truth - pred[b_pos]) ** 2
            else:
                for model_name, pred in classical.items():
                    aligned[model_name] = (test.y - pred) ** 2

            if len(aligned) >= 2:
                mcs = model_confidence_set(
                    aligned,
                    alpha=float(mcs_cfg.get("alpha", 0.05)),
                    n_bootstrap=int(mcs_cfg.get("n_bootstrap", 1000)),
                    block_size=int(mcs_cfg.get("block_size_h", 24)),
                    seed=int(cfg.get("seeds.global", 42)),
                )
                for model_name in aligned:
                    mcs_records.append(
                        {
                            "horizon_h": h,
                            "model": model_name,
                            "in_confidence_set": model_name in mcs.included,
                            "mcs_p_value": mcs.p_values.get(model_name, float("nan")),
                            "mean_squared_loss": mcs.mean_loss.get(model_name, float("nan")),
                        }
                    )
                log.info(
                    "MCS h=%d (alpha=%.2f): %d of %d models retained -- %s",
                    h,
                    mcs.alpha,
                    len(mcs.included),
                    len(aligned),
                    ", ".join(mcs.included),
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
        # One test per horizon is still a family of five. Reporting each against
        # a nominal 5% would inflate the chance of calling at least one gap real.
        alpha = float(cfg.get("evaluation.diebold_mariano.alpha", 0.05))
        for record, adjusted in zip(
            dm_records, holm_bonferroni([r["p_value"] for r in dm_records], alpha), strict=True
        ):
            record["p_value_holm"] = adjusted["p_adjusted"]
            record["significant_holm"] = adjusted["reject"]

        dm = pd.DataFrame(dm_records)
        payload["significance"]["diebold_mariano"] = dm.to_dict(orient="records")
        payload["significance"]["multiplicity"] = {
            "method": "Holm-Bonferroni",
            "family": "one DM test per horizon, best sequence vs best classical",
            "n_tests": int(dm["p_value"].notna().sum()),
            "alpha": alpha,
        }
        display = dm.assign(
            **{
                "h (hours)": dm["horizon_h"],
                "Deep model": dm["model_a"],
                "Classical": dm["model_b"],
                "N": dm["n_common"],
                "DM statistic": dm["statistic"].round(3),
                "p-value": dm["p_value"].map(lambda v: f"{v:.3g}"),
                "p (Holm)": dm["p_value_holm"].map(lambda v: f"{v:.3g}"),
                "Lower loss": dm["better"],
            }
        )[
            [
                "h (hours)",
                "Deep model",
                "Classical",
                "N",
                "DM statistic",
                "p-value",
                "p (Holm)",
                "Lower loss",
            ]
        ]
        write_table(
            cfg,
            display,
            "diebold_mariano",
            caption=(
                "Diebold-Mariano tests between the best sequence model and the best "
                "classical model at each horizon, on squared-error loss. Variance uses "
                "a Newey-West HAC estimator with truncation lag $h-1$ and the "
                "Harvey-Leybourne-Newbold small-sample correction. A negative "
                "statistic favours the deep model. The Holm column controls the "
                "family-wise error rate across the five horizons; read it, not the raw "
                "column. Model selection used validation loss only."
            ),
        )
        print("\n" + "=" * 100)
        print("DIEBOLD-MARIANO TESTS (Holm-adjusted across horizons)")
        print("=" * 100)
        print(display.to_string(index=False))

    if ci_records:
        ci = pd.DataFrame(ci_records)
        payload["significance"]["bootstrap_ci"] = ci.to_dict(orient="records")
        ci.to_csv(cfg.path_for("tables") / "bootstrap_ci_raw.csv", index=False)

        headline = int(cfg.get("task.headline_horizon_h"))
        view = ci[ci["horizon_h"] == headline].sort_values("rmse")
        display = view.assign(
            **{
                "Model": view["model"],
                "RMSE": view["rmse"].round(2),
                "95% CI": [
                    f"[{low:.2f}, {high:.2f}]"
                    for low, high in zip(view["ci_low"], view["ci_high"], strict=True)
                ],
            }
        )[["Model", "RMSE", "95% CI"]]
        write_table(
            cfg,
            display,
            "bootstrap_ci",
            caption=(
                f"Test RMSE at {headline} hours with moving-block bootstrap confidence "
                f"intervals ({boot_kwargs['n_resamples']} resamples, "
                f"{boot_kwargs['block_size']}-hour blocks). Blocks rather than "
                "independent draws because consecutive hourly errors are strongly "
                "correlated, which an i.i.d. bootstrap would treat as extra evidence."
            ),
        )
        print("\n" + "=" * 60)
        print(f"BOOTSTRAP CONFIDENCE INTERVALS (h={headline})")
        print("=" * 60)
        print(display.to_string(index=False))

    if mcs_records:
        mcs_frame = pd.DataFrame(mcs_records)
        payload["significance"]["model_confidence_set"] = mcs_frame.to_dict(orient="records")
        mcs_frame.to_csv(cfg.path_for("tables") / "model_confidence_set_raw.csv", index=False)

        headline = int(cfg.get("task.headline_horizon_h"))
        view = mcs_frame[mcs_frame["horizon_h"] == headline].sort_values("mean_squared_loss")
        display = view.assign(
            **{
                "Model": view["model"],
                "Mean squared loss": view["mean_squared_loss"].round(1),
                "MCS p": view["mcs_p_value"].map(lambda v: f"{v:.3f}"),
                "In 95% MCS": view["in_confidence_set"].map({True: "yes", False: "no"}),
            }
        )[["Model", "Mean squared loss", "MCS p", "In 95% MCS"]]
        write_table(
            cfg,
            display,
            "model_confidence_set",
            caption=(
                f"Model Confidence Set at {headline} hours (Hansen, Lunde and Nason, "
                "2011), on squared-error loss with a moving-block bootstrap. Models "
                "marked yes cannot be distinguished from the best at the 5\\% level; "
                "the ranking among them is not evidence of an ordering."
            ),
        )
        print("\n" + "=" * 70)
        print(f"MODEL CONFIDENCE SET (h={headline})")
        print("=" * 70)
        print(display.to_string(index=False))

    save_results(cfg, payload)
    log.info("wrote evaluation to %s", cfg.path_for("results_json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
