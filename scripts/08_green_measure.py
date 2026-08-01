"""Phase 5: green-AI measurement.

Combines three things into the efficiency table:

* **complexity** -- parameters and MACs, measured per architecture;
* **latency** -- single-sample inference time, mean over at least 1000 runs after
  warm-up, with the device synchronised around each iteration;
* **energy** -- the CodeCarbon estimate captured during training, recomputed
  under the Bangladesh grid carbon intensity.

The energy figures are **estimates, not metered measurements**, and the report
says so wherever they appear.

Run::

    python scripts/08_green_measure.py --config config.yaml
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from src.green.complexity import profile_model
from src.models.data import build_sequence_index, load_features
from src.models.sequence import ModelSpec, build_model
from src.results import load_results, save_results
from src.utils import check_disk_space, load_config, resolve_device, setup_logging
from src.viz.tables import write_table


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--device", default=None, choices=["auto", "cuda", "cpu"])
    p.add_argument("--latency-runs", type=int, default=None)
    return p.parse_args()


def main() -> int:
    """Measure complexity and assemble the efficiency table."""
    args = parse_args()
    cfg = load_config(args.config)
    log = setup_logging(cfg, "08_green_measure")
    check_disk_space(cfg)
    device = resolve_device(cfg, args.device)
    if args.latency_runs:
        cfg.raw["green"]["complexity"]["latency"]["n_runs"] = args.latency_runs

    payload = load_results(cfg)
    runs = payload.get("runs", [])
    sequence_runs = [r for r in runs if r.get("tier") == "tier3" and r.get("completed")]
    if not sequence_runs:
        log.error("no completed Tier 3 runs found; run scripts/06_train_sequence.py first")
        return 1

    frame = load_features(cfg)
    headline = int(cfg.get("task.headline_horizon_h"))
    dropout = float(cfg.get("models.sequence.train.dropout"))

    # Profile one instance per (architecture, window): complexity depends on the
    # shape, not the seed or the fitted weights.
    seen: set[tuple[str, int]] = set()
    complexity_rows = []

    for run in sequence_runs:
        key = (run["model"], run["window_h"])
        if key in seen:
            continue
        seen.add(key)

        spec = ModelSpec(
            arch=run["arch"],
            hidden_size=run["hidden_size"],
            num_layers=run["num_layers"],
            window=run["window_h"],
            horizon=run["horizon_h"],
            seed=run["seed"],
            attention_dim=32 if run["arch"] == "gru_attention" else None,
        )
        index = build_sequence_index(frame, cfg, headline, spec.window, "test")
        model = build_model(spec, index.n_features, dropout)
        report = profile_model(model, spec.window, index.n_features, cfg, device, log)

        complexity_rows.append(
            {"model": run["model"], "window_h": run["window_h"], **asdict(report)}
        )
        del model

    complexity = pd.DataFrame(complexity_rows)
    payload.setdefault("green", {})["complexity"] = complexity.to_dict(orient="records")

    # ---- energy, aggregated across seeds ---------------------------------
    energy_rows = []
    for run in sequence_runs:
        energy = run.get("energy") or {}
        energy_rows.append(
            {
                "model": run["model"],
                "window_h": run["window_h"],
                "horizon_h": run["horizon_h"],
                "seed": run["seed"],
                "train_seconds": run.get("train_seconds"),
                "energy_kwh": energy.get("energy_kwh"),
                "co2e_kg_codecarbon": energy.get("co2e_kg_codecarbon"),
                "co2e_g_bd_grid": energy.get("co2e_g_bd_grid"),
                "rapl_available": energy.get("rapl_available"),
                "rmse": run["metrics"]["rmse"],
                "skill": run["metrics"]["skill_vs_persistence"],
                "n_params": run["n_params"],
            }
        )
    energy = pd.DataFrame(energy_rows)
    payload["green"]["energy_runs"] = energy.to_dict(orient="records")

    grid = cfg.get("green.grid_carbon_intensity")
    payload["green"]["grid_carbon_intensity"] = grid
    payload["green"]["energy_caveat"] = (
        "CodeCarbon values are modelled estimates, not metered measurements. Where "
        "Intel RAPL counters are unreadable -- the usual case on Windows without "
        "elevated privileges -- the CPU component is entirely modelled. The "
        "Bangladesh column is the same estimated energy multiplied by a cited "
        "national grid intensity, not an independent measurement."
    )

    # ---- the headline efficiency table -----------------------------------
    at_headline = energy[energy["horizon_h"] == headline]
    if len(at_headline):
        grouped = (
            at_headline.groupby(["model", "window_h"])
            .agg(
                rmse_mean=("rmse", "mean"),
                rmse_std=("rmse", "std"),
                skill_mean=("skill", "mean"),
                params=("n_params", "first"),
                train_kwh_mean=("energy_kwh", "mean"),
                co2e_g_bd_mean=("co2e_g_bd_grid", "mean"),
                train_s_mean=("train_seconds", "mean"),
                n_seeds=("seed", "count"),
            )
            .reset_index()
        )
        merged = grouped.merge(
            complexity[["model", "window_h", "macs", "latency_ms_mean", "peak_gpu_memory_mb"]],
            on=["model", "window_h"],
            how="left",
        ).sort_values("rmse_mean")

        payload["green"]["pareto_table"] = merged.to_dict(orient="records")

        display = merged.assign(
            **{
                "Model": merged["model"],
                "Window (h)": merged["window_h"],
                f"RMSE@{headline}h": merged["rmse_mean"].round(2),
                "Skill": merged["skill_mean"].round(4),
                "Params": merged["params"].astype(int),
                "MACs": merged["macs"].map(lambda v: f"{v:,.0f}" if pd.notna(v) else "n/a"),
                "Latency (ms)": merged["latency_ms_mean"].round(4),
                "Train kWh": merged["train_kwh_mean"].map(
                    lambda v: f"{v:.3e}" if pd.notna(v) else "n/a"
                ),
                "gCO2e (BD)": merged["co2e_g_bd_mean"].map(
                    lambda v: f"{v:.4f}" if pd.notna(v) else "not available"
                ),
            }
        )[
            [
                "Model",
                "Window (h)",
                f"RMSE@{headline}h",
                "Skill",
                "Params",
                "MACs",
                "Latency (ms)",
                "Train kWh",
                "gCO2e (BD)",
            ]
        ]

        intensity = grid.get("value_gco2_per_kwh")
        write_table(
            cfg,
            display,
            "green_efficiency",
            caption=(
                f"Accuracy against cost at the {headline}-hour horizon. "
                "Energy is a CodeCarbon \\emph{estimate}, not a metered measurement; "
                "where RAPL counters are unreadable the CPU term is modelled. "
                f"gCO$_2$e is that estimate recomputed at "
                f"{intensity} gCO$_2$e/kWh for the Bangladesh grid "
                f"({grid.get('year')}, {grid.get('source_name')})."
            ),
        )
        print("\n" + "=" * 120)
        print("GREEN-AI EFFICIENCY")
        print("=" * 120)
        print(display.to_string(index=False))

    complexity_display = complexity.assign(
        Model=complexity["model"],
        **{
            "Window (h)": complexity["window_h"],
            "Params": complexity["n_params"],
            "MACs": complexity["macs"].map(lambda v: f"{v:,.0f}" if pd.notna(v) else "n/a"),
            "Latency mean (ms)": complexity["latency_ms_mean"].round(4),
            "Latency p95 (ms)": complexity["latency_ms_p95"].round(4),
            "Peak GPU (MB)": complexity["peak_gpu_memory_mb"].round(1),
        },
    )[
        [
            "Model",
            "Window (h)",
            "Params",
            "MACs",
            "Latency mean (ms)",
            "Latency p95 (ms)",
            "Peak GPU (MB)",
        ]
    ]
    write_table(
        cfg,
        complexity_display,
        "green_complexity",
        caption=(
            "Model complexity and inference cost. Latency is single-sample, the "
            f"mean of {cfg.get('green.complexity.latency.n_runs')} timed iterations "
            f"after {cfg.get('green.complexity.latency.n_warmup')} warm-up "
            f"iterations, measured on {device} with the device synchronised around "
            "each iteration."
        ),
    )

    excluded = payload.get("green", {}).get("excluded_over_budget", [])
    if excluded:
        excl = pd.DataFrame(excluded).rename(
            columns={"name": "Configuration", "params": "Parameters", "budget": "Budget"}
        )
        write_table(
            cfg,
            excl,
            "green_excluded_configs",
            caption=(
                "Architectures excluded by the parameter budget. The ceiling is the "
                "green-AI constraint of this study and was not relaxed to obtain a "
                "better error; with 102 input features it excludes the wider LSTMs "
                "and the two-layer GRU at hidden size 128."
            ),
        )
        print("\nExcluded by parameter budget:")
        print(excl.to_string(index=False))

    save_results(cfg, payload)
    if np.isnan(float(grid.get("value_gco2_per_kwh") or np.nan)):
        log.warning("grid carbon intensity unavailable; gCO2e column left blank")
    log.info("wrote green measurements to %s", cfg.path_for("results_json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
