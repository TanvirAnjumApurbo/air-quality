"""Targeted line edits on a YAML config, preserving comments.

Round-tripping a config through ``yaml.safe_dump`` discards every comment in it,
and in this repository those comments carry the citations, the source quotations
and the record of two verified API discrepancies. So every generated config is
produced by editing the lines that must change and leaving the file otherwise
byte-identical.

The technique originates in ``04_build_features.py::write_back_boundaries`` and
was generalised by ``14_make_donor_configs.py``. It lives here because a second
generator -- the lookback-arm configs -- needs the same primitives, and two
copies of a regex that must match exactly once is two definitions of "exactly
once".

Every edit is scoped to a named top-level section. That is required, not
defensive: ``tables`` and ``figures`` each appear at indent 2 under **both**
``paths`` and ``output``, so an unscoped replacement hits two lines.
"""

from __future__ import annotations

import re
from typing import Any

import yaml


def flatten(node: Any, prefix: str = "") -> dict[str, Any]:
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
            out.update(flatten(value, name))
        else:
            out[name] = value
    return out


def section_span(text: str, section: str) -> tuple[int, int]:
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


def replace_scalar(text: str, key: str, value: str, *, indent: int, section: str) -> str:
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
    lo, hi = section_span(text, section)
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


def replace_block_scalar(text: str, key: str, value: str, *, indent: int, section: str) -> str:
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
    lo, hi = section_span(text, section)
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


def verify_overrides(base_text: str, derived_text: str, expected: dict[str, Any]) -> list[str]:
    """Check a generated config changed exactly the intended keys.

    A key that changed but was not intended to is the failure mode this exists
    to catch: an inherited key with the wrong value has already caused four
    defects in this repository, and every one of them was silent.

    Args:
        base_text: Text of the base config.
        derived_text: Text of the generated config.
        expected: Dotted key to intended value.

    Returns:
        Human-readable problems; empty when the config is sound.
    """
    problems: list[str] = []
    try:
        derived = yaml.safe_load(derived_text)
    except yaml.YAMLError as exc:
        return [f"generated config does not parse: {exc}"]

    base_flat = flatten(yaml.safe_load(base_text))
    derived_flat = flatten(derived)

    changed = {
        k for k in set(base_flat) | set(derived_flat) if base_flat.get(k) != derived_flat.get(k)
    }
    for key, value in expected.items():
        if derived_flat.get(key) != value:
            problems.append(f"{key}: expected {value!r}, got {derived_flat.get(key)!r}")
    for key in sorted(changed - set(expected)):
        problems.append(
            f"{key}: changed unintentionally ({base_flat.get(key)!r} -> {derived_flat.get(key)!r})"
        )

    # The comments are the reason this module edits lines instead of dumping.
    base_comments = sum(1 for line in base_text.splitlines() if line.lstrip().startswith("#"))
    derived_comments = sum(1 for line in derived_text.splitlines() if line.lstrip().startswith("#"))
    if derived_comments < base_comments:
        problems.append(
            f"lost {base_comments - derived_comments} comment lines "
            f"({base_comments} -> {derived_comments}); the edit is not comment-preserving"
        )
    return problems
