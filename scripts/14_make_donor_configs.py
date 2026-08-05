"""Generate one config per gap-injection donor station from ``donors.yaml``.

Each donor config is derived from the base config by **targeted line edit**,
never by ``yaml.safe_dump``. Round-tripping this file through PyYAML discards
every comment in it -- and those comments carry the citations, the source
quotations and the record of two verified API discrepancies. The same
constraint governs ``04_build_features.py::write_back_boundaries``, which is
where this technique comes from.

Deriving from the *comparison city's* config rather than the primary one means
timezone, season months, meteorology variables and split boundaries are already
correct for a Beijing station; the only genuine differences between stations of
one archive are the station name and the output paths.

Every generated file is verified before it is written:

* it parses;
* it differs from the base in exactly the keys this script intended to change,
  so a substitution that silently matched the wrong line is caught rather than
  inherited -- an inherited key with the wrong value has already caused four
  defects in this repository;
* it retains the base's comment lines.

**Generate before running the pipeline, never after.** ``04_build_features.py``
writes the computed split boundaries back into whichever config it was given,
so regenerating a donor afterwards would reset those boundaries to the base
config's. Today that is harmless -- every station of this archive spans the same
four years and resolves to the same boundaries -- but it would not stay harmless
for a donor drawn from a different record. ``--check`` reports a donor as STALE
if its file has drifted from what the registry would produce, which is the
signal that this has happened.

Run::

    python scripts/14_make_donor_configs.py
    python scripts/14_make_donor_configs.py --check    # verify, write nothing
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--registry", default="donors.yaml", help="donor registry")
    p.add_argument(
        "--check",
        action="store_true",
        help="verify the generated configs match the registry without writing",
    )
    return p.parse_args()


def _flatten(node: Any, prefix: str = "") -> dict[str, Any]:
    """Flatten a nested mapping to dotted keys.

    Args:
        node: Mapping to flatten.
        prefix: Key prefix for recursion.

    Returns:
        Dotted key to leaf value.
    """
    out: dict[str, Any] = {}
    for key, value in (node or {}).items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            out.update(_flatten(value, name))
        else:
            out[name] = value
    return out


def _section_span(text: str, section: str) -> tuple[int, int]:
    """Return the character span of one top-level block.

    Args:
        text: Whole config text.
        section: Top-level key, e.g. ``"paths"``.

    Returns:
        ``(start, end)`` offsets covering the block's body.

    Raises:
        ValueError: If the section is absent or appears more than once.
    """
    starts = list(re.finditer(rf"^{re.escape(section)}:[ \t]*$", text, re.MULTILINE))
    if len(starts) != 1:
        raise ValueError(f"expected exactly one top-level {section!r} block, found {len(starts)}")
    start = starts[0].end()
    nxt = re.search(r"^[A-Za-z_][A-Za-z0-9_]*:", text[start:], re.MULTILINE)
    return start, start + nxt.start() if nxt else len(text)


def _replace_scalar(text: str, key: str, value: str, *, indent: int, section: str) -> str:
    """Replace the value of one ``key:`` line, preserving its trailing comment.

    Scoped to a top-level section because indent alone does not identify a key:
    ``tables`` and ``figures`` each appear at indent 2 under both ``paths`` and
    ``output``, and editing the wrong one would send this donor's tables to the
    primary donor's directory -- silently, and in the direction that destroys
    the experiment rather than failing it.

    Args:
        text: Whole config text.
        key: Bare key name, e.g. ``"station"``.
        value: Replacement value, already YAML-quoted if it needs to be.
        indent: Exact leading-space count of the line to edit.
        section: Top-level block the key must live in.

    Returns:
        The edited text.

    Raises:
        ValueError: If the key line is absent or matches more than once.
    """
    lo, hi = _section_span(text, section)
    body = text[lo:hi]
    pattern = re.compile(
        rf"^(?P<lead>{' ' * indent}{re.escape(key)}:)(?P<gap>[ \t]*)"
        rf"(?P<value>[^\n#]*?)(?P<comment>[ \t]*#.*)?$",
        re.MULTILINE,
    )
    matches = list(pattern.finditer(body))
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one {key!r} line at indent {indent} inside {section!r}, "
            f"found {len(matches)}"
        )
    m = matches[0]
    gap = m.group("gap") or " "
    comment = m.group("comment") or ""
    edited = body[: m.start()] + f"{m.group('lead')}{gap}{value}{comment}" + body[m.end() :]
    return text[:lo] + edited + text[hi:]


def _replace_block_scalar(text: str, key: str, value: str, *, indent: int, section: str) -> str:
    """Replace a folded (``>-``) block scalar with a single-line quoted value.

    ``project.title`` is written as a folded block over two lines. Editing it as
    a scalar would leave the continuation lines behind as stray YAML.

    Args:
        text: Whole config text.
        key: Bare key name.
        value: Replacement value, unquoted.
        indent: Leading-space count of the key line.
        section: Top-level block the key must live in.

    Returns:
        The edited text.

    Raises:
        ValueError: If the key line is absent or ambiguous.
    """
    lo, hi = _section_span(text, section)
    body = text[lo:hi]
    lead = " " * indent
    pattern = re.compile(
        rf"^{lead}{re.escape(key)}:[ \t]*>-[ \t]*\n((?:{lead}  .*\n)+)", re.MULTILINE
    )
    matches = list(pattern.finditer(body))
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one folded {key!r} block at indent {indent} inside "
            f"{section!r}, found {len(matches)}"
        )
    m = matches[0]
    escaped = value.replace('"', '\\"')
    edited = body[: m.start()] + f'{lead}{key}: "{escaped}"\n' + body[m.end() :]
    return text[:lo] + edited + text[hi:]


def build_donor_config(base_text: str, slug: str, station: str) -> tuple[str, dict[str, Any]]:
    """Produce one donor config's text and the keys it is expected to change.

    Args:
        base_text: Text of the base config.
        slug: Short donor identifier, used in every generated path.
        station: UCI station name, passed through to ``data.uci.station``.

    Returns:
        ``(text, expected_overrides)`` where the mapping is dotted key to value.
    """
    interim = f"data/interim/donors/{slug}"
    label = f"Beijing {station} (UCI Multi-Site, id 501)"
    title = (
        f"Gap-Injection Donor Replication: the {station} Station of the Beijing Multi-Site Record"
    )

    edits: list[tuple[str, str, int, str, bool]] = [
        # (key, new value as written, indent, top-level section, is_folded_block)
        ("station", f'"{station}"', 4, "data", False),
        ("site_label", f'"{label}"', 4, "data", False),
        ("name", f"donor-{slug}", 2, "project", False),
        ("title", title, 2, "project", True),
        ("data_interim", interim, 2, "paths", False),
        ("data_processed", f"data/processed/donors/{slug}", 2, "paths", False),
        ("tables", f"results/tables/donors/{slug}", 2, "paths", False),
        ("figures", f"results/figures/donors/{slug}", 2, "paths", False),
        ("results_json", f"results/results_donor_{slug}.json", 2, "paths", False),
        ("checkpoints", f"{interim}/checkpoints", 2, "paths", False),
        ("cache", f"{interim}/cache", 2, "paths", False),
        ("results_md", f"reports/donors/{slug}/RESULTS.md", 4, "output", False),
        ("data_audit_md", f"reports/donors/{slug}/DATA_AUDIT.md", 4, "output", False),
        ("abstract_facts_json", f"reports/donors/{slug}/abstract_facts.json", 4, "output", False),
        # Without this every donor overwrites the primary donor's ablation file.
        # Two configs writing to one output path is the defect this repository
        # has hit most often; here it would silently destroy the experiment the
        # donor exists to replicate.
        ("output_name", f"ablation_gap_injection_{slug}.json", 4, "ablation", False),
    ]

    text = base_text
    for key, value, indent, section, folded in edits:
        text = (
            _replace_block_scalar(text, key, value, indent=indent, section=section)
            if folded
            else _replace_scalar(text, key, value, indent=indent, section=section)
        )

    expected = {
        "data.uci.station": station,
        "data.openaq.site_label": label,
        "project.name": f"donor-{slug}",
        "project.title": title,
        "paths.data_interim": interim,
        "paths.data_processed": f"data/processed/donors/{slug}",
        "paths.tables": f"results/tables/donors/{slug}",
        "paths.figures": f"results/figures/donors/{slug}",
        "paths.results_json": f"results/results_donor_{slug}.json",
        "paths.checkpoints": f"{interim}/checkpoints",
        "paths.cache": f"{interim}/cache",
        "output.report.results_md": f"reports/donors/{slug}/RESULTS.md",
        "output.report.data_audit_md": f"reports/donors/{slug}/DATA_AUDIT.md",
        "output.report.abstract_facts_json": f"reports/donors/{slug}/abstract_facts.json",
        "ablation.gap_injection.output_name": f"ablation_gap_injection_{slug}.json",
    }
    return text, expected


def verify(base_text: str, donor_text: str, expected: dict[str, Any]) -> list[str]:
    """Check a generated config changed exactly the intended keys.

    Args:
        base_text: Text of the base config.
        donor_text: Text of the generated config.
        expected: Dotted key to intended value.

    Returns:
        Human-readable problems; empty when the config is sound.
    """
    problems: list[str] = []
    try:
        donor = yaml.safe_load(donor_text)
    except yaml.YAMLError as exc:
        return [f"generated config does not parse: {exc}"]

    base_flat = _flatten(yaml.safe_load(base_text))
    donor_flat = _flatten(donor)

    changed = {k for k in set(base_flat) | set(donor_flat) if base_flat.get(k) != donor_flat.get(k)}
    for key, value in expected.items():
        if donor_flat.get(key) != value:
            problems.append(f"{key}: expected {value!r}, got {donor_flat.get(key)!r}")
    for key in sorted(changed - set(expected)):
        problems.append(
            f"{key}: changed unintentionally ({base_flat.get(key)!r} -> {donor_flat.get(key)!r})"
        )

    # The comments are the reason this script edits lines instead of dumping.
    base_comments = sum(1 for line in base_text.splitlines() if line.lstrip().startswith("#"))
    donor_comments = sum(1 for line in donor_text.splitlines() if line.lstrip().startswith("#"))
    if donor_comments < base_comments:
        problems.append(
            f"lost {base_comments - donor_comments} comment lines "
            f"({base_comments} -> {donor_comments}); the edit is not comment-preserving"
        )
    return problems


def main() -> int:
    """Generate or verify the donor configs."""
    args = parse_args()
    registry_path = Path(args.registry)
    if not registry_path.exists():
        print(f"registry {registry_path} not found", file=sys.stderr)
        return 1

    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    base_path = Path(str(registry["base_config"]))
    out_dir = Path(str(registry["output_dir"]))
    base_text = base_path.read_text(encoding="utf-8")

    failures = 0
    for donor in registry["donors"]:
        slug, station = str(donor["slug"]), str(donor["station"])
        text, expected = build_donor_config(base_text, slug, station)
        problems = verify(base_text, text, expected)

        target = out_dir / f"{slug}.yaml"
        if problems:
            failures += 1
            print(f"\n{target}: {len(problems)} PROBLEM(S)")
            for p in problems:
                print(f"  - {p}")
            continue

        if args.check:
            existing = target.read_text(encoding="utf-8") if target.exists() else None
            state = "up to date" if existing == text else "STALE — re-run without --check"
            if existing != text:
                failures += 1
            print(f"{target}: {state}")
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding="utf-8")
            print(f"wrote {target}  (station {station}, {len(expected)} keys overridden)")

    if failures:
        print(f"\n{failures} donor config(s) failed verification", file=sys.stderr)
        return 1
    print(f"\n{len(registry['donors'])} donor config(s) verified against {base_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
