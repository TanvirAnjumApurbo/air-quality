"""Phase 5: next-day AQI-category classifier.

Uses the published US EPA PM2.5 breakpoints, which Bangladesh's Department of
Environment states it applies directly. Breakpoints are read from
``config.yaml`` and refuse to load unless marked VERIFIED against their source.

Class imbalance is handled with sample weights and reported rather than glossed:
macro-F1 is the headline, with per-class precision/recall/F1 and a confusion
matrix, because a classifier that never predicts the hazardous categories can
still post a high accuracy.

Run::

    python scripts/07_train_classifier.py --config config.yaml --resume auto
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from src.eval.metrics import classification_metrics, summarise
from src.green.energy import EnergyTracker
from src.models.classifier import (
    class_distribution,
    fit_classifier,
    get_classification_data,
    predict_classes,
)
from src.models.data import load_features
from src.results import load_results, save_results
from src.utils import check_disk_space, load_config, set_seed, setup_logging
from src.viz.tables import format_mean_std, write_table


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--resume", default="auto", choices=["auto", "never", "always"])
    p.add_argument("--seed", type=int, nargs="*", default=None)
    p.add_argument("--horizon", type=int, default=None)
    return p.parse_args()


def main() -> int:
    """Train and evaluate the AQI classifier across seeds."""
    args = parse_args()
    cfg = load_config(args.config)
    log = setup_logging(cfg, "07_train_classifier")
    check_disk_space(cfg)

    # Refuses to proceed on unverified breakpoints -- this is the guard against
    # inventing a health-advisory threshold.
    breakpoints = cfg.require_verified("classification.breakpoints")
    labels = [str(v) for v in breakpoints["labels"]]
    log.info(
        "breakpoints: %s (%s, %s)",
        breakpoints["edges_ugm3"],
        breakpoints["scheme"],
        breakpoints["revision"],
    )
    log.info("target basis: %s", cfg.get("classification.target_basis"))

    horizon = args.horizon or int(cfg.get("classification.horizon_h"))
    seeds = args.seed or [int(s) for s in cfg.get("seeds.multi")]

    frame = load_features(cfg)
    train = get_classification_data(frame, cfg, horizon, "train")
    val = get_classification_data(frame, cfg, horizon, "val")
    test = get_classification_data(frame, cfg, horizon, "test")
    log.info("rows: train=%d val=%d test=%d", len(train), len(val), len(test))

    dist_frames = []
    for name, data in (("train", train), ("val", val), ("test", test)):
        dist = class_distribution(data)
        dist["split"] = name
        dist_frames.append(dist)
    distribution = pd.concat(dist_frames, ignore_index=True)
    log.info(
        "class distribution (test):\n%s",
        distribution[distribution.split == "test"].to_string(index=False),
    )

    payload = load_results(cfg)
    per_seed: list[dict] = []
    reports = []

    for seed in seeds:
        set_seed(seed, cfg)
        tracker = EnergyTracker(cfg, f"clf_h{horizon}_s{seed}", log)
        with tracker:
            model, info = fit_classifier(train, val, cfg, seed, log)
        preds = predict_classes(model, info, test.x)
        report = classification_metrics(test.y, preds, labels)
        reports.append(report)

        per_seed.append(
            {
                "seed": seed,
                "macro_f1": report.macro_f1,
                "weighted_f1": report.weighted_f1,
                "accuracy": report.accuracy,
                "balanced_accuracy": report.balanced_accuracy,
                "energy": tracker.summary(),
            }
        )
        log.info(
            "seed %d: macro-F1 %.4f  weighted-F1 %.4f  accuracy %.4f  balanced-acc %.4f",
            seed,
            report.macro_f1,
            report.weighted_f1,
            report.accuracy,
            report.balanced_accuracy,
        )

    # Aggregate across seeds; the confusion matrix is summed so it reflects all runs.
    confusion = np.sum([np.array(r.confusion) for r in reports], axis=0)
    per_class_rows = []
    for label in labels:
        per_class_rows.append(
            {
                "label": label,
                "precision_mean": float(
                    np.mean([r.per_class[label]["precision"] for r in reports])
                ),
                "precision_std": float(
                    np.std([r.per_class[label]["precision"] for r in reports], ddof=1)
                )
                if len(reports) > 1
                else 0.0,
                "recall_mean": float(np.mean([r.per_class[label]["recall"] for r in reports])),
                "recall_std": float(np.std([r.per_class[label]["recall"] for r in reports], ddof=1))
                if len(reports) > 1
                else 0.0,
                "f1_mean": float(np.mean([r.per_class[label]["f1"] for r in reports])),
                "f1_std": float(np.std([r.per_class[label]["f1"] for r in reports], ddof=1))
                if len(reports) > 1
                else 0.0,
                "support": int(reports[0].per_class[label]["support"]),
            }
        )
    per_class = pd.DataFrame(per_class_rows)

    summary = {
        "horizon_h": horizon,
        "target_basis": cfg.get("classification.target_basis"),
        "breakpoints": breakpoints,
        "class_weight": cfg.get("classification.class_weight"),
        "labels": labels,
        "n_test": len(test),
        "per_seed": per_seed,
        "macro_f1": summarise([r["macro_f1"] for r in per_seed]),
        "weighted_f1": summarise([r["weighted_f1"] for r in per_seed]),
        "accuracy": summarise([r["accuracy"] for r in per_seed]),
        "balanced_accuracy": summarise([r["balanced_accuracy"] for r in per_seed]),
        "per_class": per_class.to_dict(orient="records"),
        "confusion_summed": confusion.tolist(),
        "class_distribution": distribution.to_dict(orient="records"),
    }
    payload["classification"] = summary
    save_results(cfg, payload)

    display = per_class.assign(
        Precision=lambda d: [
            format_mean_std(m, s, 3)
            for m, s in zip(d["precision_mean"], d["precision_std"], strict=True)
        ],
        Recall=lambda d: [
            format_mean_std(m, s, 3) for m, s in zip(d["recall_mean"], d["recall_std"], strict=True)
        ],
        F1=lambda d: [
            format_mean_std(m, s, 3) for m, s in zip(d["f1_mean"], d["f1_std"], strict=True)
        ],
    )[["label", "support", "Precision", "Recall", "F1"]].rename(
        columns={"label": "AQI category", "support": "Test support"}
    )
    write_table(
        cfg,
        display,
        "classification_per_class",
        caption=(
            f"Per-class performance of the {horizon}-hour AQI-category classifier, "
            f"mean $\\pm$ standard deviation over {len(seeds)} seeds. Labels use the "
            "US EPA PM2.5 breakpoints on a 24-hour mean basis, which the Bangladesh "
            "Department of Environment states it applies. Class weights are balanced."
        ),
    )

    overall = pd.DataFrame(
        [
            {
                "Metric": "Macro-F1",
                "Value": format_mean_std(
                    summary["macro_f1"]["mean"], summary["macro_f1"]["std"], 4
                ),
            },
            {
                "Metric": "Weighted-F1",
                "Value": format_mean_std(
                    summary["weighted_f1"]["mean"], summary["weighted_f1"]["std"], 4
                ),
            },
            {
                "Metric": "Accuracy",
                "Value": format_mean_std(
                    summary["accuracy"]["mean"], summary["accuracy"]["std"], 4
                ),
            },
            {
                "Metric": "Balanced accuracy",
                "Value": format_mean_std(
                    summary["balanced_accuracy"]["mean"], summary["balanced_accuracy"]["std"], 4
                ),
            },
        ]
    )
    write_table(
        cfg,
        overall,
        "classification_overall",
        caption=(
            f"Overall AQI-classification performance at {horizon} hours. Macro-F1 is "
            "the headline because the categories are severely imbalanced; accuracy "
            "alone would be dominated by the majority classes."
        ),
    )

    print("\n" + "=" * 80)
    print(f"AQI CLASSIFIER (h={horizon}, {len(seeds)} seeds)")
    print("=" * 80)
    print(overall.to_string(index=False))
    print()
    print(display.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
