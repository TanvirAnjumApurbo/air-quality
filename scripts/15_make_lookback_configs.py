r"""Generate one config per sterilisation radius for the mediation grid.

``20_missingness_law.py`` proposes that fragmentation costs skill *because* it
sterilises the backward reach behind every gap. That is a mediation claim, and it
predicts something falsifiable: shorten the radius and the fragmented-versus-
contiguous arm gap should shrink toward zero.

Testing it means running the gap-injection grid again at a shorter radius, which
means a config per radius. Derived by targeted line edit, never by
``yaml.safe_dump`` -- round-tripping through PyYAML would discard every comment,
and those carry the citations and the record of two verified API discrepancies.

The radius a cell actually experiences is::

    R = max(lookback_h, window_h - 1) + horizon_h

so both keys must move together. Changing the lookback alone would leave the
48-hour window binding and produce two arms that are the same experiment --
which is the trap a naive sweep falls into, since lookbacks of 48, 24 and 12 all
collapse onto a 48-hour window.

Every generated config **must** override ``ablation.gap_injection.output_name``.
Two grids sharing an output path is the defect this repository hits most often,
and ``16_gap_injection.py`` keys its resume on ``(arm, coverage, seed)`` alone --
it knows nothing about the radius, so a shared path would make every cell read as
already recorded and report a complete grid for a radius it never ran.

Run::

    python scripts/15_make_lookback_configs.py --base config_beijing.yaml
    python scripts/15_make_lookback_configs.py --check
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml
from src.config_edit import replace_scalar, section_span, verify_overrides

#: (lookback_h, window_h) pairs and the radius each produces at h=24.
#:
#: 192 is the status quo and is generated so the mediation compares like with
#: like -- the same script, the same output-name convention, the same resume
#: semantics -- rather than comparing a generated grid against the hand-run one.
RADII: tuple[tuple[int, int], ...] = ((168, 48), (48, 48), (24, 24))


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--base", default="config_beijing.yaml", help="config to derive from")
    p.add_argument("--slug", default="wanliu", help="donor slug used in the output names")
    p.add_argument("--out-dir", default="config/lookback")
    p.add_argument("--horizon", type=int, default=24)
    p.add_argument("--check", action="store_true", help="verify without writing")
    return p.parse_args()


def set_all_windows(text: str, window_h: int) -> str:
    """Set every sequence spec's ``window_h`` inside the ablation block.

    The specs are inline mappings in a list, so the key appears once per spec and
    a single-match replacement cannot be used. Scoped to the ablation section and
    asserted to hit every spec, because a partial edit would leave the grid
    training a mixture of radii under one label.

    Args:
        text: Whole config text.
        window_h: Replacement window in hours.

    Returns:
        The edited text.

    Raises:
        ValueError: If no ``window_h`` key is found in the ablation block.
    """
    lo, hi = section_span(text, "ablation")
    body = text[lo:hi]
    pattern = re.compile(r"(window_h:\s*)(\d+)")
    if not pattern.search(body):
        raise ValueError("no window_h key inside the ablation block")
    edited = pattern.sub(rf"\g<1>{window_h}", body)
    return text[:lo] + edited + text[hi:]


def build_config(
    base_text: str, *, lookback_h: int, window_h: int, horizon_h: int, slug: str
) -> tuple[str, dict[str, object], int]:
    """Derive one radius config.

    Args:
        base_text: Text of the base config.
        lookback_h: Feature cap in hours.
        window_h: Sequence window in hours.
        horizon_h: Forecast horizon in hours.
        slug: Donor slug, used in the output name.

    Returns:
        ``(text, expected_overrides, radius_h)``.
    """
    radius = max(lookback_h, window_h - 1) + horizon_h
    output_name = f"ablation_gap_injection_{slug}_R{radius}.json"

    text = replace_scalar(base_text, "lookback_h", str(lookback_h), indent=2, section="features")
    text = set_all_windows(text, window_h)
    text = replace_scalar(text, "output_name", output_name, indent=4, section="ablation")

    # The sequence specs are a list of mappings, so verify_overrides sees them as
    # one leaf. Declare the whole edited list rather than leaving the change
    # undeclared -- an undeclared change is exactly what the verifier exists to
    # catch, and silencing it here would defeat the check for every other key.
    specs = yaml.safe_load(text)["ablation"]["gap_injection"]["sequence_models"]
    expected: dict[str, object] = {
        "features.lookback_h": lookback_h,
        "ablation.gap_injection.output_name": output_name,
        "ablation.gap_injection.sequence_models": specs,
    }
    return text, expected, radius


def main() -> int:
    """Generate or verify the radius configs."""
    args = parse_args()
    base_path = Path(args.base)
    if not base_path.exists():
        print(f"base config {base_path} not found", file=sys.stderr)
        return 1
    base_text = base_path.read_text(encoding="utf-8")
    base_parsed = yaml.safe_load(base_text)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    problems_total = 0
    for lookback_h, window_h in RADII:
        text, expected, radius = build_config(
            base_text,
            lookback_h=lookback_h,
            window_h=window_h,
            horizon_h=args.horizon,
            slug=args.slug,
        )
        path = out_dir / f"{args.slug}_R{radius}.yaml"

        # The window lives inside a list of inline mappings, which verify_overrides
        # flattens per element; check it directly instead.
        parsed = yaml.safe_load(text)
        windows = {
            int(s["window_h"]) for s in parsed["ablation"]["gap_injection"]["sequence_models"]
        }
        problems = verify_overrides(base_text, text, expected)
        if windows != {window_h}:
            problems.append(f"sequence windows are {sorted(windows)}, expected all {window_h}")

        # A shared output name silently destroys the grid it collides with.
        base_out = base_parsed["ablation"]["gap_injection"]["output_name"]
        if parsed["ablation"]["gap_injection"]["output_name"] == base_out:
            problems.append("output_name was not overridden; this grid would overwrite the base")

        if problems:
            problems_total += len(problems)
            print(f"{path}: FAILED")
            for problem in problems:
                print(f"  - {problem}")
            continue

        if args.check:
            current = path.read_text(encoding="utf-8") if path.exists() else None
            state = "up to date" if current == text else "STALE"
            print(f"{path}: {state} (radius {radius} h, lookback {lookback_h}, window {window_h})")
            if state == "STALE":
                problems_total += 1
        else:
            path.write_text(text, encoding="utf-8")
            print(
                f"wrote {path}  (radius {radius} h = max({lookback_h}, {window_h - 1}) + "
                f"{args.horizon}, output {expected['ablation.gap_injection.output_name']})"
            )

    if problems_total:
        print(f"\n{problems_total} problem(s)", file=sys.stderr)
        return 1
    print(f"\n{len(RADII)} radius config(s) verified against {base_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
