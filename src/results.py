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


#: Experiment label carried by every run belonging to the headline study.
#:
#: Records written before side experiments existed carry no ``experiment`` key
#: at all, and they are the headline study, so absence means this value.
MAIN_EXPERIMENT = "main"


def main_runs(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Runs belonging to the headline experiment, and only those.

    The lookback frontier writes into the same ``results.json``, keyed by a
    lookback-tagged variant so it cannot collide with an existing record. But
    nothing else stops ``11_make_report.py::_best_per_horizon`` from *selecting*
    a lookback-48 run as the headline model, which would silently change the
    reported result -- the same class of failure as the 675 stale-width records,
    moved to the reporting side.

    Every consumer of ``runs`` goes through here rather than reading the list,
    so a new side experiment cannot leak into the headline by being forgotten at
    one of nine call sites. A test greps for survivors.

    Args:
        payload: The results mapping.

    Returns:
        Run records with no ``experiment`` key, or with it set to
        :data:`MAIN_EXPERIMENT`.
    """
    return [
        run
        for run in payload.get("runs", [])
        if run.get("experiment", MAIN_EXPERIMENT) == MAIN_EXPERIMENT
    ]


def experiment_runs(payload: dict[str, Any], experiment: str) -> list[dict[str, Any]]:
    """Runs belonging to one named side experiment.

    Args:
        payload: The results mapping.
        experiment: The ``experiment`` label to select.

    Returns:
        Matching run records.
    """
    return [run for run in payload.get("runs", []) if run.get("experiment") == experiment]


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


def city_suffix(cfg: Config) -> str:
    """Filename suffix distinguishing this city's outputs from the other's.

    ``paths.results`` is the SAME directory for every city -- only
    ``paths.results_json`` differs -- so any new artefact written to
    ``path_for("results")`` under a bare name is written twice, and the city that
    runs second silently destroys the first one's. That has already happened
    here: the comparison city once overwrote the primary city's RESULTS.md and
    DATA_AUDIT.md, and the gap-injection experiment carries an explicit
    ``output_name`` per donor for exactly this reason.

    Derived from the results-file stem so it cannot drift from the city the run
    actually belongs to: ``results.json`` gives ``""`` and
    ``results_beijing.json`` gives ``"_beijing"``.

    Args:
        cfg: Loaded configuration.

    Returns:
        Empty string for the primary city, ``"_<city>"`` otherwise.
    """
    stem = Path(str(cfg.get("paths.results_json"))).stem
    return stem[len("results") :] if stem.startswith("results") else f"_{stem}"
