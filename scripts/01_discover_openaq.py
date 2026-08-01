"""Phase 1: discover candidate OpenAQ monitors near Dhaka.

Prints every candidate with its ID, name, provider, parameters and first/last
measurement date, and probes the S3 archive for the year partitions that
actually exist. **Downloads no measurement data.** The location ID to use is
chosen by the operator from this table and written into
``data.openaq.location_ids`` in ``config.yaml``.

Run::

    python scripts/01_discover_openaq.py --config config.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.data.fetch_openaq import (  # noqa: E402
    OpenAQError,
    candidates_to_frame,
    discover_locations,
    get_api_key,
    probe_s3_coverage,
)
from src.utils import check_disk_space, load_config, setup_logging  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml", help="path to config.yaml")
    parser.add_argument(
        "--no-s3-probe",
        action="store_true",
        help="skip the S3 partition listing (faster, but coverage is then unverified)",
    )
    return parser.parse_args()


def main() -> int:
    """Run discovery and write the candidate table."""
    args = parse_args()
    cfg = load_config(args.config)
    log = setup_logging(cfg, "01_discover_openaq")
    check_disk_space(cfg)

    site = cfg.get("data.site")
    log.info(
        "discovering monitors: iso=%s, radius=%d m around (%.4f, %.4f)",
        site["country_iso"],
        site["discovery_radius_m"],
        site["latitude"],
        site["longitude"],
    )

    try:
        key = get_api_key(cfg)
    except OpenAQError as exc:
        log.error("%s", exc)
        return 2

    candidates = discover_locations(cfg, key)
    log.info("found %d candidate locations", len(candidates))

    if not args.no_s3_probe:
        log.info("probing S3 archive for actual year partitions (no data downloaded)")
        probe_s3_coverage(cfg, candidates)

    df = candidates_to_frame(candidates)

    tables_dir = cfg.path_for("tables")
    tables_dir.mkdir(parents=True, exist_ok=True)
    out_csv = tables_dir / "openaq_dhaka_candidates.csv"
    df.to_csv(out_csv, index=False)
    log.info("wrote %s (%d rows)", out_csv, len(df))

    pm25 = df[df["has_pm25"]].copy()

    with pd.option_context("display.max_rows", 200, "display.width", 200):
        print("\n" + "=" * 118)
        print("OPENAQ CANDIDATE MONITORS -- PM2.5 CAPABLE")
        print("=" * 118)
        cols = [
            "location_id", "name", "provider", "distance_km", "is_monitor",
            "pm25_units", "api_first_utc", "api_last_utc", "s3_years",
        ]
        print(pm25[cols].to_string(index=False) if len(pm25) else "  (none)")

        others = df[~df["has_pm25"]]
        if len(others):
            print("\n" + "-" * 118)
            print(f"NON-PM2.5 LOCATIONS ({len(others)}) -- listed for completeness, not usable as targets")
            print("-" * 118)
            print(others[["location_id", "name", "provider", "parameters"]].to_string(index=False))

    print("\n" + "=" * 118)
    print("SUMMARY")
    print("=" * 118)
    print(f"  total locations discovered      {len(df)}")
    print(f"  with a PM2.5 sensor             {len(pm25)}")
    print(f"  reference monitors with PM2.5   {int(pm25['is_monitor'].sum()) if len(pm25) else 0}")
    print(f"  with S3 partitions present      {int((df['s3_n_year_partitions'] > 0).sum())}")
    units = sorted({u for u in pm25['pm25_units'].dropna().unique()})
    print(f"  distinct PM2.5 units reported   {units}")
    print(f"\n  candidate table written to      {out_csv}")
    print("\n  STOP: no measurement data has been downloaded.")
    print("  Choose location IDs from the table above and set them in config.yaml:")
    print("      data.openaq.location_ids: [<id>, ...]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
