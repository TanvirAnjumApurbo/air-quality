"""The single source of truth for every reported number.

``results/results.json`` accumulates one record per model x horizon x seed. Each
pipeline stage merges into it rather than overwriting, so a partial re-run
updates only what it recomputed. Every number that could appear in the paper is
written here by code and read from here by the reporting stage -- nothing is
transcribed by hand.
"""

from __future__ import annotations

import json
import platform
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.utils import Config, git_commit_hash


def _environment() -> dict[str, Any]:
    """Capture the environment a result was produced in."""
    info: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_build"] = torch.version.cuda
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["gpu_capability"] = list(torch.cuda.get_device_capability(0))
    except ImportError:
        info["torch"] = None
    return info


def load_results(cfg: Config) -> dict[str, Any]:
    """Load results.json, creating the skeleton if absent.

    Args:
        cfg: Loaded configuration.

    Returns:
        The results mapping.
    """
    path = cfg.path_for("results_json")
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {
        "meta": {},
        "runs": [],
        "baselines": {},
        "aggregates": {},
        "significance": {},
        "green": {},
        "classification": {},
    }


def save_results(cfg: Config, payload: dict[str, Any]) -> Path:
    """Write results.json with provenance stamped in.

    Args:
        cfg: Loaded configuration.
        payload: The results mapping.

    Returns:
        Path written.
    """
    path = cfg.path_for("results_json")
    path.parent.mkdir(parents=True, exist_ok=True)

    meta = payload.setdefault("meta", {})
    meta["git_commit"] = git_commit_hash()
    meta["written_utc"] = datetime.now(UTC).isoformat()
    meta["environment"] = _environment()
    meta["config_path"] = str(cfg.path)
    meta["site"] = cfg.get("data.openaq.site_label", None)
    meta["horizons"] = cfg.get("task.horizons_h")
    meta["seeds"] = cfg.get("seeds.multi")
    meta["split_boundaries"] = cfg.get("split.explicit_boundaries")

    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path


def upsert_run(payload: dict[str, Any], record: dict[str, Any]) -> None:
    """Insert or replace a run record.

    Identity is ``(model, variant, horizon_h, seed)``, so re-running one
    configuration replaces just that record instead of appending a duplicate.

    Args:
        payload: The results mapping, mutated in place.
        record: The run record to store.
    """
    key = (
        record.get("model"),
        record.get("variant"),
        record.get("horizon_h"),
        record.get("seed"),
    )
    runs = payload.setdefault("runs", [])
    for i, existing in enumerate(runs):
        existing_key = (
            existing.get("model"),
            existing.get("variant"),
            existing.get("horizon_h"),
            existing.get("seed"),
        )
        if existing_key == key:
            runs[i] = record
            return
    runs.append(record)


def find_reusable_run(
    payload: dict[str, Any],
    *,
    model: str,
    variant: str,
    horizon_h: int,
    seed: int,
    n_features: int,
) -> dict[str, Any] | None:
    """Return a completed run that may be reused as-is, or ``None`` to retrain.

    Identity is ``upsert_run``'s key plus the input width. Width is part of the
    test because ``--resume`` consults this file, not the checkpoints: deleting
    the checkpoint directory does not invalidate a record here. When the channel
    set changed from 102 columns to 18, a resumed sweep kept 675 of 900 records
    trained at the old width and retrained only the 225 configurations that were
    new, leaving a ``results.json`` that mixed two experiments and a report that
    compared them as if they were one.

    Records written before the width was tracked have no ``n_features`` and so
    never match, which is the safe direction: the cost is retraining, and the
    cost of the other direction is a silently invalid comparison.

    Args:
        payload: The results mapping.
        model: Model name.
        variant: Variant tag, e.g. ``"w48"``.
        horizon_h: Forecast horizon in hours.
        seed: Random seed.
        n_features: Input channel count the current run would use.

    Returns:
        The matching completed record, or ``None`` if it must be retrained.
    """
    for run in payload.get("runs", []):
        if (
            run.get("model") == model
            and run.get("variant") == variant
            and run.get("horizon_h") == horizon_h
            and run.get("seed") == seed
            and run.get("completed")
            and run.get("n_features") == n_features
        ):
            return run
    return None


def stale_width_runs(payload: dict[str, Any], tier: str, n_features: int) -> list[dict[str, Any]]:
    """Return recorded runs of ``tier`` whose input width is not ``n_features``.

    Args:
        payload: The results mapping.
        tier: Tier to inspect, e.g. ``"tier3"``.
        n_features: The width the current configuration produces.

    Returns:
        Records that will be retrained rather than reused.
    """
    return [
        run
        for run in payload.get("runs", [])
        if run.get("tier") == tier and run.get("n_features") != n_features
    ]
