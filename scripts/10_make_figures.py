"""Phase 6: generate every paper figure.

All figures are written at the configured DPI in both PNG and PDF, using the
machine-validated colourblind-safe palette in fixed assignment order.

Where more series exist than the palette holds, the chart is faceted by tier
rather than cycling colours: a recycled hue makes two different models look like
the same one.

Run::

    python scripts/10_make_figures.py --config config.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from src.models.data import build_sequence_index, invert, load_features
from src.models.sequence import (
    DeviceWindowSampler,
    ModelSpec,
    _make_loss,
    build_model,
    evaluate_sampler,
)
from src.results import load_results
from src.utils import check_disk_space, load_config, resolve_device, setup_logging
from src.viz.figures import save_figure, setup_style

TIER_LABELS = {
    "tier1": "Tier 1 — baselines",
    "tier2": "Tier 2 — classical ML",
    "tier3": "Tier 3 — sequence",
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--device", default=None, choices=["auto", "cuda", "cpu"])
    return p.parse_args()


def fig_rmse_vs_horizon(cfg, payload, palette, log) -> None:
    """RMSE against forecast horizon, faceted by tier, with seed error bars."""
    runs = pd.DataFrame(payload.get("runs", []))
    if runs.empty:
        log.warning("no runs; skipping RMSE-vs-horizon figure")
        return
    runs["rmse"] = runs["metrics"].map(lambda m: m["rmse"])

    tiers = [t for t in ("tier1", "tier2", "tier3") if t in set(runs["tier"])]
    fig, axes = plt.subplots(1, len(tiers), figsize=(4.2 * len(tiers), 4.0), sharey=True)
    if len(tiers) == 1:
        axes = [axes]

    for ax, tier in zip(axes, tiers, strict=True):
        sub = runs[runs["tier"] == tier]
        models = sorted(sub["model"].unique())
        if tier == "tier3":
            # Keep one representative window per architecture so the panel stays
            # within the palette; the full grid is in the tables.
            best = (
                sub.groupby(["model", "variant"])["rmse"]
                .mean()
                .reset_index()
                .sort_values("rmse")
                .groupby("model")
                .first()
                .reset_index()
            )
            keep = set(zip(best["model"], best["variant"], strict=True))
            sub = sub[[(m, v) in keep for m, v in zip(sub["model"], sub["variant"], strict=True)]]
            models = sorted(sub["model"].unique())

        for i, model in enumerate(models[: len(palette)]):
            group = sub[sub["model"] == model].groupby("horizon_h")["rmse"]
            mean, std = group.mean(), group.std().fillna(0.0)
            ax.errorbar(
                mean.index,
                mean.to_numpy(),
                yerr=std.to_numpy(),
                marker="o",
                capsize=3,
                color=palette[i],
                label=model,
                linewidth=1.6,
            )
        ax.set_title(TIER_LABELS.get(tier, tier))
        ax.set_xlabel("forecast horizon (hours)")
        ax.set_xticks(sorted(runs["horizon_h"].unique()))
        ax.legend(fontsize=7.5)
    axes[0].set_ylabel("RMSE (µg/m³)")
    fig.suptitle("Test-period RMSE against forecast horizon (error bars: ± 1 s.d. across seeds)")
    save_figure(cfg, fig, "fig05_rmse_vs_horizon")
    log.info("wrote fig05_rmse_vs_horizon")


def fig_pred_vs_actual(cfg, frame, payload, palette, device, log) -> None:
    """Predicted against observed over a representative held-out window."""
    headline = int(cfg.get("task.headline_horizon_h"))
    runs = [
        r
        for r in payload.get("runs", [])
        if r.get("tier") == "tier3" and r.get("completed") and r.get("horizon_h") == headline
    ]
    if not runs:
        log.warning("no sequence runs; skipping predicted-vs-actual figure")
        return

    chosen = min(runs, key=lambda r: r["best_val_loss"])
    spec = ModelSpec(
        arch=chosen["arch"],
        hidden_size=chosen["hidden_size"],
        num_layers=chosen["num_layers"],
        window=chosen["window_h"],
        horizon=headline,
        seed=chosen["seed"],
        attention_dim=32 if chosen["arch"] == "gru_attention" else None,
    )
    ckpt = Path(cfg.path_for("checkpoints")) / spec.run_id / "best.ckpt"
    if not ckpt.exists():
        log.warning("no checkpoint for %s", spec.run_id)
        return

    index = build_sequence_index(frame, cfg, headline, spec.window, "test")
    model = build_model(spec, index.n_features, float(cfg.get("models.sequence.train.dropout")))
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=False)["model"])
    model = model.to(device)
    _, raw = evaluate_sampler(
        model,
        DeviceWindowSampler(index, device),
        _make_loss(cfg),
        int(cfg.get("models.sequence.train.batch_size")),
    )
    preds = invert(cfg, raw)

    series = pd.DataFrame(
        {"observed": index.y, "predicted": preds, "persistence": index.persistence},
        index=index.index,
    ).sort_index()

    # Pick the densest 21-day stretch so the panel shows continuous behaviour
    # rather than a window dominated by gaps.
    daily = series["observed"].resample("1D").count()
    if len(daily) >= 21:
        rolling = daily.rolling(21).sum()
        end = rolling.idxmax()
        start = end - pd.Timedelta(days=21)
        window = series.loc[start:end]
    else:
        window = series

    fig, ax = plt.subplots(figsize=tuple(cfg.get("output.figures.figsize_wide")))
    observed_colour = str(cfg.get("output.figures.observed_colour", "#222222"))
    reference_colour = str(cfg.get("output.figures.reference_colour", "#767676"))

    ax.plot(
        window.index, window["observed"], color=observed_colour, linewidth=1.7, label="observed"
    )
    ax.plot(
        window.index,
        window["persistence"],
        color=reference_colour,
        linewidth=1.1,
        linestyle=":",
        label="persistence",
    )
    ax.plot(
        window.index,
        window["predicted"],
        color=palette[0],
        linewidth=1.5,
        label=f"{spec.name} (w={spec.window})",
    )

    threshold = float(cfg.get("evaluation.stratify.by_pollution_level.threshold_ugm3"))
    ax.axhline(threshold, color=palette[1], linewidth=1.1, linestyle="--")
    ax.annotate(
        f"BD 24-h standard {threshold:g} µg/m³",
        xy=(0.01, threshold),
        xycoords=("axes fraction", "data"),
        va="bottom",
        fontsize=8,
        color=palette[1],
    )

    ax.set_ylabel("PM2.5 (µg/m³)")
    ax.set_xlabel("date (UTC)")
    ax.set_title(f"{headline}-hour-ahead forecast over a representative held-out window")
    ax.legend(ncol=3, fontsize=8)
    fig.autofmt_xdate()
    save_figure(cfg, fig, "fig06_pred_vs_actual")
    log.info("wrote fig06_pred_vs_actual")


def fig_feature_importance(cfg, payload, palette, log) -> None:
    """Permutation importance for the tree model at the headline horizon."""
    importance = payload.get("importance", {})
    if not importance:
        log.warning("no importance record; skipping feature-importance figure")
        return
    key = next(iter(importance))
    scores = pd.Series(importance[key]).sort_values(ascending=False).head(20)

    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    ax.barh(range(len(scores)), scores.to_numpy()[::-1], color=palette[0])
    ax.set_yticks(range(len(scores)), [s.replace("_", " ") for s in scores.index[::-1]], fontsize=8)
    ax.set_xlabel("increase in RMSE when permuted (modelling scale)")
    ax.set_title(f"Permutation importance — {key.replace('_', ' ')}")
    ax.grid(axis="y", visible=False)
    save_figure(cfg, fig, "fig07_feature_importance")
    log.info("wrote fig07_feature_importance")


def fig_pareto(cfg, payload, palette, log) -> None:
    """Accuracy against estimated emissions, with the parameter count as size."""
    table = payload.get("green", {}).get("pareto_table")
    if not table:
        log.warning("no green table; skipping Pareto figure")
        return
    df = pd.DataFrame(table)
    if df["co2e_g_bd_mean"].isna().all():
        log.warning("no CO2e estimates available; Pareto figure would be empty")
        return

    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    for i, (model, group) in enumerate(df.groupby("model")):
        ax.scatter(
            group["co2e_g_bd_mean"],
            group["rmse_mean"],
            s=np.clip(group["params"] / 300.0, 25, 400),
            color=palette[i % len(palette)],
            alpha=0.85,
            edgecolor="white",
            linewidth=0.8,
            label=model,
            zorder=3,
        )
        for _, row in group.iterrows():
            ax.annotate(
                f"w{int(row['window_h'])}",
                xy=(row["co2e_g_bd_mean"], row["rmse_mean"]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=7,
                color="#444444",
            )

    ax.set_xlabel("estimated training emissions (gCO₂e, Bangladesh grid)")
    ax.set_ylabel(f"RMSE at {cfg.get('task.headline_horizon_h')} h (µg/m³)")
    ax.set_title("Accuracy against estimated training emissions")
    ax.legend(fontsize=8, title="marker area ∝ parameters", title_fontsize=7.5)
    fig.text(
        0.5,
        -0.04,
        "Energy is a CodeCarbon estimate, not a metered measurement.",
        ha="center",
        fontsize=7.5,
        color="#555555",
    )
    save_figure(cfg, fig, "fig08_pareto_accuracy_emissions")
    log.info("wrote fig08_pareto_accuracy_emissions")


def fig_confusion(cfg, payload, log) -> None:
    """Row-normalised confusion matrix for the AQI classifier."""
    classification = payload.get("classification")
    if not classification:
        log.warning("no classification record; skipping confusion matrix")
        return

    matrix = np.array(classification["confusion_summed"], dtype=float)
    labels = classification["labels"]
    present = matrix.sum(axis=1) > 0
    matrix_shown = matrix[np.ix_(present, present)]
    labels_shown = [lab for lab, keep in zip(labels, present, strict=True) if keep]
    normalised = matrix_shown / np.clip(matrix_shown.sum(axis=1, keepdims=True), 1, None)

    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    im = ax.imshow(normalised, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels_shown)), labels_shown, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels_shown)), labels_shown, fontsize=8)
    ax.set_xlabel("predicted category")
    ax.set_ylabel("observed category")
    ax.grid(False)
    for i in range(len(labels_shown)):
        for j in range(len(labels_shown)):
            ax.text(
                j,
                i,
                f"{normalised[i, j]:.2f}\n({int(matrix_shown[i, j])})",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if normalised[i, j] > 0.55 else "#333333",
            )
    fig.colorbar(im, ax=ax, label="share of observed class", fraction=0.04, pad=0.03)
    ax.set_title(
        f"AQI category confusion, {classification['horizon_h']}-hour advisory\n"
        f"(summed over {len(classification['per_seed'])} seeds)"
    )
    save_figure(cfg, fig, "fig09_confusion_matrix")
    log.info("wrote fig09_confusion_matrix")


def fig_skill_by_stratum(cfg, payload, palette, log) -> None:
    """Skill score by pollution stratum, the honest-reporting figure."""
    strat = payload.get("stratified")
    if not strat:
        log.warning("no stratified metrics; skipping stratum figure")
        return
    df = pd.DataFrame(strat)
    headline = int(cfg.get("task.headline_horizon_h"))
    view = df[(df["horizon_h"] == headline) & (df["stratum"].isin(["season", "pollution"]))]
    if view.empty:
        return

    groups = list(dict.fromkeys(view["group"]))

    # Persistence is the reference, so its skill is identically zero, and at
    # h=24 seasonal-naive reduces to persistence and is zero too. Plotting them
    # spends palette slots and legend entries on invisible bars -- and because
    # the palette is deliberately not cycled, that previously pushed the
    # sequence model, the actual contribution, out of the figure. Drop any series
    # that is flat zero everywhere, then order so the contribution is kept first.
    candidates = []
    for model in dict.fromkeys(view["model"]):
        values = view.loc[view["model"] == model, "skill_vs_persistence"]
        if np.allclose(values.fillna(0.0).to_numpy(), 0.0):
            continue
        candidates.append(model)

    known_baselines = {"persistence", "seasonal_naive", "climatology", "sarimax", "ridge"}
    trees = {"random_forest", "xgboost", "lightgbm"}

    def _priority(name: str) -> int:
        if name not in known_baselines and name not in trees:
            return 0  # sequence models first: they are the contribution
        if name in trees:
            return 1
        return 2

    models = sorted(candidates, key=lambda m: (_priority(m), m))[: len(palette)]
    if not models:
        log.warning("no non-zero skill series to plot; skipping stratum figure")
        return

    fig, ax = plt.subplots(figsize=(9.0, 4.4))
    width = 0.8 / max(len(models), 1)
    positions = np.arange(len(groups))
    for i, model in enumerate(models):
        sub = view[view["model"] == model].set_index("group").reindex(groups)
        ax.bar(
            positions + i * width,
            sub["skill_vs_persistence"].to_numpy(),
            width=width,
            color=palette[i],
            label=model,
        )
    ax.axhline(0.0, color="#767676", linewidth=1.0)
    ax.set_xticks(
        positions + width * (len(models) - 1) / 2, groups, rotation=18, ha="right", fontsize=8
    )
    ax.set_ylabel("skill score vs persistence")
    ax.set_title(f"Skill by stratum at {headline} hours (zero = no better than persistence)")
    ax.legend(fontsize=8, ncol=min(len(models), 4))
    save_figure(cfg, fig, "fig10_skill_by_stratum")
    log.info("wrote fig10_skill_by_stratum")


def main() -> int:
    """Generate every figure that has data available."""
    args = parse_args()
    cfg = load_config(args.config)
    log = setup_logging(cfg, "10_make_figures")
    check_disk_space(cfg)
    palette = setup_style(cfg)
    device = resolve_device(cfg, args.device)

    payload = load_results(cfg)
    frame = load_features(cfg)

    fig_rmse_vs_horizon(cfg, payload, palette, log)
    fig_pred_vs_actual(cfg, frame, payload, palette, device, log)
    fig_feature_importance(cfg, payload, palette, log)
    fig_pareto(cfg, payload, palette, log)
    fig_confusion(cfg, payload, log)
    fig_skill_by_stratum(cfg, payload, palette, log)

    written = sorted(p.name for p in cfg.path_for("figures").glob("*.png"))
    print("\nFigures in results/figures:")
    for name in written:
        print(f"  {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
