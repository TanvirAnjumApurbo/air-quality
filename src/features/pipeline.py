"""Assemble the model matrix: features, splits, boundary purge, scaling.

Extracted from ``scripts/04_build_features.py`` so that anything needing a model
matrix builds it the same way. The gap-injection ablation
(``scripts/16_gap_injection.py``) rebuilds one per degradation cell, and it must
not reimplement any of this: the boundary purge and the per-horizon validity
masks are where the leakage rules are enforced, so a second, drifting copy of
them would be a second, drifting definition of what counts as a usable row.

The function writes exactly what the downstream stages read -- ``features.parquet``,
``scaler.json`` and ``features_meta.json`` -- into ``paths.data_processed``. To
build a variant, point that path elsewhere in the config and call again.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd

from src.eval.split import (
    SplitBoundaries,
    TrainOnlyScaler,
    assign_splits,
    purge_boundary_rows,
    resolve_boundaries,
)
from src.features.build_features import (
    FeatureBuildReport,
    build_features,
    feature_columns,
    max_backward_dependency,
)
from src.utils import Config


@dataclass
class FeatureMatrix:
    """The built matrix and the accounting that produced it.

    Attributes:
        frame: The feature frame, with ``split`` and ``valid_h{h}`` columns.
        boundaries: Resolved chronological split boundaries.
        report: Feature-construction accounting.
        meta: The mapping written to ``features_meta.json``.
        row_accounting: Per-horizon usable/rejected row counts.
        predictors: Predictor column names.
        purge_counts: Rows removed per horizon by the split-boundary purge.
    """

    frame: pd.DataFrame
    boundaries: SplitBoundaries
    report: FeatureBuildReport
    meta: dict[str, Any]
    row_accounting: pd.DataFrame
    predictors: list[str]
    purge_counts: dict[int, int]


def build_feature_matrix(
    pm: pd.DataFrame,
    met: pd.DataFrame,
    cfg: Config,
    logger: Any,
    *,
    write: bool = True,
) -> FeatureMatrix:
    """Build features, split chronologically, purge boundaries and scale.

    The order matters and is not negotiable: features are built first so that
    contiguity runs are known, the split is assigned second, the boundary purge
    third so no row's history or target crosses a split edge, and only then is
    the scaler fitted -- on training rows alone.

    Args:
        pm: Hourly PM2.5 frame, UTC-indexed.
        met: Hourly meteorology frame, UTC-indexed.
        cfg: Loaded configuration. ``paths.data_processed`` and ``paths.tables``
            decide where output lands, so a variant build only needs those
            redirected.
        logger: Logger.
        write: Write ``features.parquet``, ``scaler.json``, ``features_meta.json``
            and the row-accounting CSV. False builds in memory only.

    Returns:
        The built matrix and its accounting.
    """
    processed = cfg.path_for("data_processed")
    frame, report = build_features(pm, met, cfg, logger)
    logger.info(
        "built %d predictor columns (max backward dependency %d h)",
        report.n_features,
        report.max_lag_h,
    )

    boundaries = resolve_boundaries(frame.index, cfg)
    frame["split"] = assign_splits(frame.index, boundaries)
    logger.info("split: %s", boundaries.caption())

    max_lag = max_backward_dependency(cfg)
    horizons = [int(h) for h in cfg.get("task.horizons_h")]

    purge_counts: dict[int, int] = {}
    for h in horizons:
        keep = purge_boundary_rows(frame, frame["split"], max_lag, h)
        before = int(frame[f"valid_h{h}"].sum())
        frame[f"valid_h{h}"] = frame[f"valid_h{h}"] & keep
        after = int(frame[f"valid_h{h}"].sum())
        purge_counts[h] = before - after
        logger.info(
            "h=%3d: boundary purge removed %d rows (%d -> %d)", h, before - after, before, after
        )

    predictors = feature_columns(frame, cfg, include_oracle=False)
    oracle_cols = [c for c in frame.columns if c.startswith("oracle_")]
    train_mask = frame["split"] == "train"

    scaler = TrainOnlyScaler(str(cfg.get("scaling.method"))).fit(
        frame.loc[train_mask], predictors, "train"
    )
    logger.info(
        "scaler fitted on %d training rows over %d predictors",
        int(train_mask.sum()),
        len(predictors),
    )

    meta = {
        "site": cfg.get("data.openaq.site_label"),
        "boundaries": boundaries.to_dict(),
        "caption": boundaries.caption(),
        "max_backward_dependency_h": max_lag,
        "n_predictors": len(predictors),
        "predictors": predictors,
        "n_oracle_columns": len(oracle_cols),
        "horizons": horizons,
        "build_report": asdict(report),
        "boundary_purge_removed": purge_counts,
        "rows_per_split": {
            name: int((frame["split"] == name).sum()) for name in ("train", "val", "test")
        },
        "valid_rows_per_split_per_horizon": {
            str(h): {
                name: int(((frame["split"] == name) & frame[f"valid_h{h}"]).sum())
                for name in ("train", "val", "test")
            }
            for h in horizons
        },
    }

    rows = []
    for h in horizons:
        rejected = report.rows_rejected_per_horizon[h]
        counts = meta["valid_rows_per_split_per_horizon"][str(h)]
        rows.append(
            {
                "horizon_h": h,
                "train": counts["train"],
                "val": counts["val"],
                "test": counts["test"],
                "total_valid": sum(counts.values()),
                "rej_no_run": rejected["outside_any_run"],
                "rej_short_history": rejected["insufficient_history"],
                "rej_target_past_run_end": rejected["target_beyond_run_end"],
                "rej_target_imputed": rejected["target_not_observed"],
                "rej_split_boundary": purge_counts[h],
            }
        )
    row_accounting = pd.DataFrame(rows)

    if write:
        processed.mkdir(parents=True, exist_ok=True)
        (processed / "scaler.json").write_text(
            json.dumps(scaler.to_dict(), indent=2), encoding="utf-8"
        )
        out_path = processed / "features.parquet"
        frame.to_parquet(out_path)
        (processed / "features_meta.json").write_text(
            json.dumps(meta, indent=2, default=str), encoding="utf-8"
        )
        tables_dir = cfg.path_for("tables")
        tables_dir.mkdir(parents=True, exist_ok=True)
        row_accounting.to_csv(tables_dir / "features_row_accounting.csv", index=False)
        logger.info("wrote %s (%d rows, %d cols)", out_path, len(frame), frame.shape[1])

    return FeatureMatrix(
        frame=frame,
        boundaries=boundaries,
        report=report,
        meta=meta,
        row_accounting=row_accounting,
        predictors=predictors,
        purge_counts=purge_counts,
    )
