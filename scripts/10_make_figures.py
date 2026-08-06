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
from src.viz.figures import (
    COL_DOUBLE,
    COL_SINGLE,
    UNIT_PM25,
    panel_label,
    save_figure,
    setup_style,
)

TIER_LABELS = {
    "tier1": "Tier 1 — baselines",
    "tier2": "Tier 2 — classical ML",
    "tier3": "Tier 3 — sequence",
}
PANEL_TAGS = ("(a)", "(b)", "(c)", "(d)")


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
    fig, axes = plt.subplots(1, len(tiers), figsize=(COL_DOUBLE, 2.75), sharey=True)
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
                markersize=3.0,
                capsize=2.0,
                elinewidth=0.8,
                capthick=0.8,
                color=palette[i],
                label=model.replace("_", " "),
                linewidth=1.2,
            )
        ax.set_xlabel("Forecast horizon (h)")
        ax.set_xticks(sorted(runs["horizon_h"].unique()))
        # The tier name rides on the legend title rather than a panel title:
        # it names the series in the box directly beneath it, and the caption
        # keeps the sentence that used to be the suptitle.
        ax.legend(
            title=TIER_LABELS.get(tier, tier),
            fontsize=6.8,
            title_fontsize=7.2,
            # "best" rather than a fixed corner: the panels share a y-axis set by
            # tier 1, so the empty region is in a different corner in each one.
            loc="best",
            frameon=True,
            framealpha=0.92,
            facecolor="white",
            edgecolor="#CCCCCC",
            labelspacing=0.25,
            handlelength=1.3,
            borderpad=0.3,
        )
    for ax, tag in zip(axes, PANEL_TAGS, strict=False):
        panel_label(ax, tag)
    axes[0].set_ylabel(f"Test RMSE ({UNIT_PM25})")
    fig.tight_layout(w_pad=1.0)
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

    fig, ax = plt.subplots(figsize=(COL_DOUBLE, 2.6))
    observed_colour = str(cfg.get("output.figures.observed_colour", "#222222"))
    reference_colour = str(cfg.get("output.figures.reference_colour", "#767676"))

    ax.plot(
        window.index, window["observed"], color=observed_colour, linewidth=1.15, label="observed"
    )
    ax.plot(
        window.index,
        window["persistence"],
        color=reference_colour,
        linewidth=0.9,
        linestyle=":",
        label="persistence",
    )
    ax.plot(
        window.index,
        window["predicted"],
        color=palette[0],
        linewidth=1.15,
        label=f"{spec.name.replace('_', ' ')} (w = {spec.window} h)",
    )

    threshold = float(cfg.get("evaluation.stratify.by_pollution_level.threshold_ugm3"))
    ax.axhline(
        threshold,
        color=palette[1],
        linewidth=1.0,
        linestyle="--",
        label=f"BD 24-h standard ({threshold:g} {UNIT_PM25})",
    )

    ax.set_ylabel(f"PM2.5 ({UNIT_PM25})")
    ax.set_xlabel("Date (UTC)")
    ax.margins(x=0.005)
    # Legend above the panel: four series over a spiky trace leave no interior
    # region a box can occupy without covering an excursion.
    ax.legend(
        ncol=4,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        columnspacing=1.4,
        handlelength=1.8,
    )
    fig.autofmt_xdate(rotation=0, ha="center")
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

    # Single column: a ranked list of 20 names is tall and narrow by nature, and
    # widening it to the page only stretches the bars away from their labels.
    fig, ax = plt.subplots(figsize=(COL_SINGLE, 3.9))
    ax.barh(range(len(scores)), scores.to_numpy()[::-1], color=palette[0], height=0.74)
    ax.set_yticks(range(len(scores)), [s.replace("_", " ") for s in scores.index[::-1]], fontsize=7)
    ax.set_ylim(-0.7, len(scores) - 0.3)
    ax.set_xlabel("Increase in RMSE when permuted\n(modelling scale)", linespacing=1.4)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
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

    # Twelve model/window combinations exceed the six-colour categorical palette,
    # and a per-point text label is what made the previous version unreadable.
    # Colour therefore carries the architecture family and marker shape carries
    # the input window, which is every dimension the labels held, in the marks.
    family = df["model"].str.extract(r"^([a-z]+)", expand=False)
    family_order = [f for f in ("dlinear", "nlinear", "gru", "lstm") if f in set(family)]
    windows = sorted(df["window_h"].unique())
    markers = ["o", "s", "^", "D", "v"]

    fig, ax = plt.subplots(figsize=(COL_DOUBLE, 3.3))
    for fam_i, fam in enumerate(family_order):
        for win_i, win in enumerate(windows):
            group = df[(family == fam) & (df["window_h"] == win)]
            if group.empty:
                continue
            ax.scatter(
                group["co2e_g_bd_mean"],
                group["rmse_mean"],
                s=np.clip(group["params"] / 260.0, 14, 210),
                color=palette[fam_i],
                marker=markers[win_i % len(markers)],
                alpha=0.80,
                edgecolor="white",
                linewidth=0.6,
                zorder=3,
            )

    family_handles = [
        plt.Line2D(
            [], [], marker="o", linestyle="none", markersize=5, color=palette[i], label=fam.upper()
        )
        for i, fam in enumerate(family_order)
    ]
    window_handles = [
        plt.Line2D(
            [],
            [],
            marker=markers[i % len(markers)],
            linestyle="none",
            markersize=5,
            color="#555555",
            label=f"{int(win)} h",
        )
        for i, win in enumerate(windows)
    ]
    # A size legend rather than a "marker area is proportional to parameters"
    # caption: the reader can measure a marker against it instead of estimating
    # what a proportionality claim means at this scale.
    lo, hi = float(df["params"].min()), float(df["params"].max())
    size_handles = [
        ax.scatter(
            [],
            [],
            s=float(np.clip(p / 260.0, 14, 210)),
            color="#B0B0B0",
            edgecolor="white",
            linewidth=0.6,
            label=f"{p / 1000.0:.1f}k" if p >= 1000 else f"{int(p)}",
        )
        for p in (lo, np.sqrt(lo * hi), hi)
    ]

    # Three legends, attached to the *figure* rather than the axes. Axes.add_artist
    # clips what it is given to the axes patch, so a second legend anchored outside
    # the axes silently disappears; figure-level legends coexist in fig.legends and
    # are picked up by the tight bounding box.
    for handles, title, y in (
        (family_handles, "Architecture", 1.00),
        (window_handles, "Input window", 0.60),
        (size_handles, "Parameters", 0.26),
    ):
        fig.legend(
            handles=handles,
            title=title,
            loc="upper left",
            bbox_to_anchor=(1.02, y),
            bbox_transform=ax.transAxes,
            borderaxespad=0.0,
            labelspacing=0.75 if title == "Parameters" else 0.35,
        )

    ax.set_xlabel("Estimated training emissions (g CO$_2$e, Bangladesh grid)")
    ax.set_ylabel(f"RMSE at {cfg.get('task.headline_horizon_h')} h ({UNIT_PM25})")
    # The caveat travels with the artefact rather than only in the caption: the
    # x-axis is an estimate, and a figure separated from its caption should not
    # read as a measurement.
    ax.text(
        0.0,
        -0.235,
        "Energy is a CodeCarbon estimate, not a metered measurement.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=plt.rcParams["legend.fontsize"] - 0.5,
        color="#666666",
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

    # A confusion matrix is square, so a full-width version would be half a page
    # tall. Single column, with the AQI names wrapped rather than rotated: at
    # this width a 35-degree rotation puts "Unhealthy for Sensitive Groups"
    # further from its column than the neighbouring one.
    wrapped = [lab.replace(" for Sensitive Groups", "\n(sensitive)") for lab in labels_shown]
    wrapped = [lab.replace("Very Unhealthy", "Very\nunhealthy") for lab in wrapped]

    fig, ax = plt.subplots(figsize=(COL_SINGLE, 3.35))
    im = ax.imshow(normalised, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels_shown)), wrapped, fontsize=6.0)
    ax.set_yticks(range(len(labels_shown)), wrapped, fontsize=6.0)
    ax.set_xlabel("Predicted category")
    ax.set_ylabel("Observed category")
    ax.grid(False)
    ax.tick_params(length=0)
    for i in range(len(labels_shown)):
        for j in range(len(labels_shown)):
            ax.text(
                j,
                i,
                f"{normalised[i, j]:.2f}\n({int(matrix_shown[i, j])})",
                ha="center",
                va="center",
                fontsize=5.6,
                linespacing=1.15,
                color="white" if normalised[i, j] > 0.55 else "#2B2B2B",
            )
    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cbar.set_label("Share of observed class", fontsize=plt.rcParams["axes.labelsize"])
    cbar.ax.tick_params(labelsize=plt.rcParams["ytick.labelsize"], length=2)
    cbar.outline.set_linewidth(0.6)
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

    fig, ax = plt.subplots(figsize=(COL_DOUBLE, 2.9))
    width = 0.82 / max(len(models), 1)
    positions = np.arange(len(groups))
    for i, model in enumerate(models):
        sub = view[view["model"] == model].set_index("group").reindex(groups)
        ax.bar(
            positions + i * width,
            sub["skill_vs_persistence"].to_numpy(),
            width=width,
            color=palette[i],
            label=model.replace("_", " "),
            edgecolor="white",
            linewidth=0.3,
        )
    ax.axhline(0.0, color="#767676", linewidth=0.8)
    # The stratum labels are written by the evaluator in ASCII; typeset the unit
    # here so the axis matches every other axis in the paper.
    tick_labels = [
        g.replace("ug/m3", UNIT_PM25).replace("<=", r"$\leq$").replace(">", r"$>$") for g in groups
    ]
    ax.set_xticks(
        positions + width * (len(models) - 1) / 2,
        tick_labels,
        rotation=14,
        ha="right",
        fontsize=7.2,
    )
    ax.set_xlabel("Test-period stratum")
    ax.set_ylabel("Skill score vs persistence")
    ax.legend(
        ncol=min(len(models), 6),
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        columnspacing=1.0,
        handlelength=1.2,
        fontsize=7.5,
    )
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
