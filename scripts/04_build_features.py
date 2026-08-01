"""Phase 2: build features, resolve chronological splits, write the model matrix.

Applies the leakage rules structurally (see ``src/features/build_features.py``
and ``src/eval/split.py``), resolves the split boundary dates from the configured
fractions, writes them back into ``config.yaml`` so every table caption can print
them, and reports exactly how many rows each rule discarded.

Run::

    python scripts/04_build_features.py --config config.yaml
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import yaml
from src.eval.split import (
    TrainOnlyScaler,
    assign_splits,
    purge_boundary_rows,
    resolve_boundaries,
)
from src.features.build_features import (
    build_features,
    feature_columns,
    max_backward_dependency,
)
from src.utils import ConfigError, check_disk_space, load_config, setup_logging


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml", help="path to config.yaml")
    parser.add_argument(
        "--no-write-back",
        action="store_true",
        help="do not write resolved split boundaries back into config.yaml",
    )
    return parser.parse_args()


def write_back_boundaries(config_path: Path, boundaries: dict[str, str]) -> None:
    """Persist resolved split boundaries into config.yaml, preserving comments.

    Written back so that the exact dates that produced a result are recorded
    alongside the parameters that produced them, rather than recomputed later
    from fractions that may since have changed.

    Deliberately a targeted line edit rather than a YAML round-trip. Reserialising
    with ``yaml.safe_dump`` silently discards every comment in the file, and this
    config carries its citations, source quotations and the record of two API
    discrepancies in comments -- losing those would destroy the provenance the
    study depends on.

    Args:
        config_path: Path to config.yaml.
        boundaries: Mapping with ``train_end`` and ``val_end`` ISO strings.

    Raises:
        ConfigError: If the expected keys are not found, or the edit does not
            round-trip to the intended values.
    """
    text = config_path.read_text(encoding="utf-8")

    replacements = {
        r"^(\s*train_end:\s*).*$": f'\\g<1>"{boundaries["train_end"]}"',
        r"^(\s*val_end:\s*).*$": f'\\g<1>"{boundaries["val_end"]}"',
    }
    for pattern, replacement in replacements.items():
        text, n = re.subn(pattern, replacement, text, count=1, flags=re.MULTILINE)
        if n != 1:
            raise ConfigError(f"could not locate {pattern!r} in {config_path}")

    text, n = re.subn(
        r"^(\s*)_status: PENDING_VERIFICATION(\s*# filled once real coverage is known)$",
        r"\g<1>_status: VERIFIED   # resolved by scripts/04_build_features.py",
        text,
        count=1,
        flags=re.MULTILINE,
    )
    if n != 1:
        raise ConfigError("could not locate the split _status marker in config.yaml")

    config_path.write_text(text, encoding="utf-8")

    # Verify the edit produced the intended values and left the file parseable.
    check = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    written = check["split"]["explicit_boundaries"]
    if (
        written["train_end"] != boundaries["train_end"]
        or written["val_end"] != boundaries["val_end"]
    ):
        raise ConfigError(f"boundary write-back did not round-trip: {written}")


def main() -> int:
    """Build the model matrix and report the leakage accounting."""
    args = parse_args()
    cfg = load_config(args.config)
    log = setup_logging(cfg, "04_build_features")
    check_disk_space(cfg)

    interim = cfg.path_for("data_interim")
    processed = cfg.path_for("data_processed")
    processed.mkdir(parents=True, exist_ok=True)
    tables_dir = cfg.path_for("tables")

    pm = pd.read_parquet(interim / "openaq_pm25_hourly.parquet")
    met = pd.read_parquet(interim / "power_hourly.parquet")
    log.info("loaded PM2.5 %s and meteorology %s", pm.shape, met.shape)

    frame, report = build_features(pm, met, cfg, log)
    log.info(
        "built %d predictor columns (max backward dependency %d h)",
        report.n_features,
        report.max_lag_h,
    )

    # ---- chronological split ------------------------------------------------
    boundaries = resolve_boundaries(frame.index, cfg)
    frame["split"] = assign_splits(frame.index, boundaries)
    log.info("split: %s", boundaries.caption())

    max_lag = max_backward_dependency(cfg)
    horizons = [int(h) for h in cfg.get("task.horizons_h")]

    # ---- boundary purge, per horizon ---------------------------------------
    purge_counts: dict[int, int] = {}
    for h in horizons:
        keep = purge_boundary_rows(frame, frame["split"], max_lag, h)
        before = int(frame[f"valid_h{h}"].sum())
        frame[f"valid_h{h}"] = frame[f"valid_h{h}"] & keep
        after = int(frame[f"valid_h{h}"].sum())
        purge_counts[h] = before - after
        log.info(
            "h=%3d: boundary purge removed %d rows (%d -> %d)", h, before - after, before, after
        )

    # ---- scaling, fitted on train only -------------------------------------
    predictors = feature_columns(frame, cfg, include_oracle=False)
    oracle_cols = [c for c in frame.columns if c.startswith("oracle_")]
    train_mask = frame["split"] == "train"

    scaler = TrainOnlyScaler(str(cfg.get("scaling.method"))).fit(
        frame.loc[train_mask], predictors, "train"
    )
    log.info(
        "scaler fitted on %d training rows over %d predictors",
        int(train_mask.sum()),
        len(predictors),
    )

    (processed / "scaler.json").write_text(json.dumps(scaler.to_dict(), indent=2), encoding="utf-8")

    # ---- persist ------------------------------------------------------------
    out_path = processed / "features.parquet"
    frame.to_parquet(out_path)
    log.info("wrote %s (%d rows, %d cols)", out_path, len(frame), frame.shape[1])

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
    (processed / "features_meta.json").write_text(
        json.dumps(meta, indent=2, default=str), encoding="utf-8"
    )

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
    summary = pd.DataFrame(rows)
    summary.to_csv(tables_dir / "features_row_accounting.csv", index=False)

    if not args.no_write_back:
        write_back_boundaries(Path(args.config), boundaries.to_dict())
        log.info("wrote resolved boundaries back into %s", args.config)

    print("\n" + "=" * 78)
    print("FEATURE BUILD")
    print("=" * 78)
    print(f"  site                     {meta['site']}")
    print(f"  predictors               {len(predictors)}")
    print(f"  oracle columns (labelled){len(oracle_cols):>4}  (upper bound only, never headline)")
    print(f"  max backward dependency  {max_lag} h")
    print(f"  gap-free runs            {report.n_runs}  (longest {report.longest_run_h} h)")
    print(
        f"  forward-filled hours     {report.n_ffilled:,} (limit {cfg.get('impute.max_ffill_hours')} h)"
    )
    print()
    print("  SPLIT (chronological, no shuffling)")
    print(f"    train  {boundaries.train_start:%Y-%m-%d} -> {boundaries.train_end:%Y-%m-%d}")
    print(f"    val    {boundaries.train_end:%Y-%m-%d} -> {boundaries.val_end:%Y-%m-%d}")
    print(f"    test   {boundaries.val_end:%Y-%m-%d} -> {boundaries.test_end:%Y-%m-%d}")
    print()
    print("  USABLE SUPERVISED ROWS PER HORIZON")
    print(summary.to_string(index=False))
    print(f"\n  wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
