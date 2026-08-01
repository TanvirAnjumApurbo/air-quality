"""Cross-city check: reshape the Beijing record into the Dhaka pipeline's schema.

Writes the target and meteorology frames under the Beijing configuration's
``data_interim`` directory using the same filenames the Dhaka run uses, so every
downstream stage — audit, features, baselines, sweep, evaluation, report — runs
unchanged against ``--config config_beijing.yaml``.

Run::

    python scripts/12_prepare_beijing.py --config config_beijing.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from src.data.prepare_beijing import prepare, write_prepared
from src.utils import check_disk_space, load_config, setup_logging


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="config_beijing.yaml")
    p.add_argument(
        "--source",
        default="data/interim/uci_beijing_hourly.parquet",
        help="hourly Beijing frame written by scripts/02_fetch_data.py",
    )
    return p.parse_args()


def main() -> int:
    """Prepare the Beijing frames."""
    args = parse_args()
    cfg = load_config(args.config)
    log = setup_logging(cfg, "12_prepare_beijing")
    check_disk_space(cfg)

    source = Path(args.source)
    if not source.exists():
        log.error("%s not found -- run scripts/02_fetch_data.py first", source)
        return 1

    raw = pd.read_parquet(source)
    log.info("loaded %s: %s, %s to %s", source.name, raw.shape, raw.index.min(), raw.index.max())

    pm25, met, ledger = prepare(cfg, raw, log)
    write_prepared(cfg, pm25, met, ledger, log)

    print("\n" + "=" * 72)
    print("BEIJING PREPARED (cross-city generalisation check)")
    print("=" * 72)
    print(f"  site               {cfg.get('data.openaq.site_label')}")
    print(f"  span               {pm25.index.min()} -> {pm25.index.max()}")
    print(f"  hours              {len(pm25):,}")
    print(
        f"  PM2.5 observed     {int(pm25['pm25'].notna().sum()):,} "
        f"({100.0 * pm25['pm25'].notna().mean():.1f}%)"
    )
    print(f"  met drivers        {met.shape[1]} ({', '.join(met.columns)})")
    print(f"  written to         {cfg.path_for('data_interim')}")
    print("\n  Co-pollutants (PM10, SO2, NO2, CO, O3) are deliberately excluded:")
    print("  Dhaka has none, so using them would change the problem rather than")
    print("  test whether the method generalises.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
