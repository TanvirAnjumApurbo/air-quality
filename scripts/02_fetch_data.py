"""Phase 1: fetch all three data sources.

Downloads, quality-controls and caches:

* **OpenAQ** PM2.5 for the configured location IDs (unsigned S3 archive).
* **NASA POWER** hourly meteorology for the study point (no key required).
* **UCI Beijing Multi-Site** (repo 501), fetched regardless of how the Dhaka pull
  goes -- it is both the fallback dataset and the cross-city generalisation check.

Every filter logs the number of rows it removed. Raw pulls are cached under
``data/raw`` so re-running is cheap and an interrupted fetch resumes.

Run::

    python scripts/02_fetch_data.py --config config.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.fetch_openaq import (
    apply_qc,
    load_openaq_raw,
    to_hourly,
    write_openaq,
)
from src.data.fetch_power import fetch_power, write_power
from src.data.fetch_uci_fallback import fetch_uci_beijing, to_hourly_utc, write_uci
from src.utils import check_disk_space, load_config, setup_logging


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml", help="path to config.yaml")
    parser.add_argument(
        "--skip",
        nargs="*",
        default=[],
        choices=["openaq", "power", "uci"],
        help="sources to skip (already fetched)",
    )
    return parser.parse_args()


def main() -> int:
    """Fetch every configured data source."""
    args = parse_args()
    cfg = load_config(args.config)
    log = setup_logging(cfg, "02_fetch_data")
    free_gb = check_disk_space(cfg)
    log.info("disk check passed: %.1f GB free", free_gb)

    summary: dict[str, object] = {}

    # ---------------------------------------------------------------- OpenAQ
    if "openaq" not in args.skip:
        log.info("=" * 70)
        log.info("OPENAQ PM2.5  locations=%s", cfg.get("data.openaq.location_ids"))
        log.info("=" * 70)
        raw = load_openaq_raw(cfg, log)
        clean, ledger = apply_qc(cfg, raw, log)
        hourly = to_hourly(cfg, clean, log)
        write_openaq(cfg, hourly, log)

        ledger_path = cfg.path_for("data_interim") / "openaq_qc_ledger.json"
        ledger_path.write_text(json.dumps(ledger, indent=2), encoding="utf-8")
        log.info("wrote QC ledger to %s", ledger_path)

        summary["openaq"] = {
            "raw_records": len(raw),
            "after_qc": len(clean),
            "hourly_slots": len(hourly),
            "hourly_observed": int(hourly["pm25"].notna().sum()),
            "first_utc": str(hourly.index.min()),
            "last_utc": str(hourly.index.max()),
        }

    # ----------------------------------------------------------- NASA POWER
    if "power" not in args.skip:
        log.info("=" * 70)
        log.info("NASA POWER meteorology")
        log.info("=" * 70)
        met, units = fetch_power(cfg, log)
        write_power(cfg, met, units, log)
        summary["power"] = {
            "rows": len(met),
            "columns": list(met.columns),
            "units": units,
            "first_utc": str(met.index.min()),
            "last_utc": str(met.index.max()),
            "missing_cells": int(met.isna().sum().sum()),
        }

    # ------------------------------------------------------------ UCI Beijing
    if "uci" not in args.skip:
        log.info("=" * 70)
        log.info("UCI Beijing Multi-Site (fallback + cross-city check)")
        log.info("=" * 70)
        raw_uci = fetch_uci_beijing(cfg, log)
        beijing = to_hourly_utc(cfg, raw_uci, log)
        write_uci(cfg, beijing, log)
        summary["uci"] = {
            "raw_rows": len(raw_uci),
            "station_rows": len(beijing),
            "columns": list(beijing.columns),
            "first_utc": str(beijing.index.min()),
            "last_utc": str(beijing.index.max()),
        }

    out = cfg.path_for("data_interim") / "fetch_summary.json"
    out.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    log.info("=" * 70)
    log.info("wrote fetch summary to %s", out)
    print("\n" + json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
