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

It must override ``paths.figures`` and ``paths.tables`` for the same reason, and
that one was learned the expensive way. The grid file was named per radius from
the start; the two directories that ``17_ablation_analysis.py`` writes alongside
it were not. Running 17 on a radius config therefore overwrote the donor city's
``fig11_gap_injection.pdf`` and every ``ablation_*`` table with the shortened-reach
numbers, and the paper shipped the R=48 experiment under the R=192 caption while
its own table quoted R=192. A shared grid name fails loudly; a shared output
directory fails silently, which is worse.

``paths.results_json`` is deliberately left inherited. ``16_gap_injection.py``
reads the tuned tier-2 hyperparameters out of it, so it is a read dependency on
the donor city rather than a write target, and redirecting it would break the
grid rather than isolate it.

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
from src.config_edit import flatten, replace_scalar, section_span, verify_overrides

#: (lookback_h, window_h) pairs and the radius each produces at h=24.
#:
#: The status quo radius is deliberately absent. The base grid already IS that
#: radius, so generating a config for it would retrain 101 cells to reproduce a
#: file on disk -- and worse, under a capped config it would land in a separate
#: checkpoint tree and lose the resume. 23_mediation.py reads the base grid as
#: the deepest radius instead.
RADII: tuple[tuple[int, int], ...] = ((48, 48), (24, 24))


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
    figures = f"results/figures/lookback/{slug}_R{radius}"
    tables = f"results/tables/lookback/{slug}_R{radius}"

    text = replace_scalar(base_text, "lookback_h", str(lookback_h), indent=2, section="features")
    text = set_all_windows(text, window_h)
    text = replace_scalar(text, "output_name", output_name, indent=4, section="ablation")
    # 17_ablation_analysis.py writes a figure and nine tables next to the grid it
    # analyses. Those go to paths.figures and paths.tables, which are inherited,
    # so without these two lines a radius run lands them in the donor city's
    # directories and destroys the headline versions in place. Scoped to `paths`
    # because both keys also exist at this indent under `output`.
    text = replace_scalar(text, "figures", figures, indent=2, section="paths")
    text = replace_scalar(text, "tables", tables, indent=2, section="paths")

    # The sequence specs are a list of mappings, so verify_overrides sees them as
    # one leaf. Declare the whole edited list rather than leaving the change
    # undeclared -- an undeclared change is exactly what the verifier exists to
    # catch, and silencing it here would defeat the check for every other key.
    specs = yaml.safe_load(text)["ablation"]["gap_injection"]["sequence_models"]
    expected: dict[str, object] = {
        "features.lookback_h": lookback_h,
        "paths.figures": figures,
        "paths.tables": tables,
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

        # Every path a radius run writes to must differ from the base's. The grid
        # is the loud case, since a shared name makes resume report a complete
        # grid for a radius it never ran; the figure and table directories are
        # the quiet one, and the quiet one is what corrupted the paper's fig11.
        base_flat = flatten(base_parsed)
        derived_flat = flatten(parsed)
        for key, artefact in (
            ("ablation.gap_injection.output_name", "grid"),
            ("paths.figures", "figures"),
            ("paths.tables", "tables"),
        ):
            if derived_flat.get(key) == base_flat.get(key):
                problems.append(
                    f"{key} was not overridden; this radius would overwrite the base {artefact}"
                )

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
