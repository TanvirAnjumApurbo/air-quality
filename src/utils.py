"""Shared infrastructure: config loading, seeding, disk guards, logging.

Every script in this pipeline reads its parameters through :func:`load_config`.
Nothing is hardcoded in a script -- horizons, lags, split dates, seeds, model
sizes and learning rates all live in ``config.yaml``.
"""

from __future__ import annotations

import logging
import os
import random
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "config.yaml"

# Sentinel used in config.yaml for values that require a citation and have not
# yet been verified against a source document.
PENDING = "PENDING_VERIFICATION"
VERIFIED = "VERIFIED"


class ConfigError(RuntimeError):
    """Raised when the configuration is missing, malformed, or unverified."""


class DiskSpaceError(RuntimeError):
    """Raised when the project drive has less free space than required."""


@dataclass(frozen=True)
class Config:
    """Immutable view over ``config.yaml`` with dotted-path access.

    Attributes:
        raw: The parsed YAML mapping.
        path: Path the configuration was loaded from.
    """

    raw: dict[str, Any]
    path: Path

    def get(self, dotted: str, default: Any = ...) -> Any:
        """Fetch a value by dotted path, e.g. ``"models.sequence.max_params"``.

        Args:
            dotted: Dot-separated key path.
            default: Returned if the path is absent. If omitted, a missing path
                raises instead, so typos fail loudly rather than silently
                substituting a default.

        Returns:
            The value at ``dotted``.

        Raises:
            ConfigError: If the path is absent and no default was supplied.
        """
        node: Any = self.raw
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                if default is ...:
                    raise ConfigError(f"missing config key: {dotted!r} (in {self.path})")
                return default
            node = node[part]
        return node

    def require_verified(self, dotted: str) -> Any:
        """Fetch a citation-backed value, refusing to return an unverified one.

        Guards the standing rule that a number requiring a citation is never
        fabricated: if ``_status`` under ``dotted`` is not ``VERIFIED``, the
        caller must leave the dependent cell blank and flag it.

        Args:
            dotted: Dot-separated path to a mapping containing a ``_status`` key.

        Returns:
            The mapping at ``dotted``.

        Raises:
            ConfigError: If the block is not marked ``VERIFIED``.
        """
        block = self.get(dotted)
        if not isinstance(block, dict):
            raise ConfigError(f"{dotted!r} is not a mapping; cannot check _status")
        status = block.get("_status")
        if status != VERIFIED:
            raise ConfigError(
                f"{dotted!r} is marked {status!r}, not {VERIFIED!r}. "
                "This value requires a citable source before it can be used in a "
                "reported number. Leave the cell blank and flag it."
            )
        return block

    def path_for(self, key: str) -> Path:
        """Resolve a ``paths.*`` entry to an absolute path.

        Args:
            key: Key under the ``paths`` block, e.g. ``"data_raw"``.

        Returns:
            Absolute path, resolved relative to the repository root.
        """
        value = Path(str(self.get(f"paths.{key}")))
        return value if value.is_absolute() else (REPO_ROOT / value)


def load_config(path: str | Path | None = None) -> Config:
    """Load and lightly validate ``config.yaml``.

    Args:
        path: Config file path. Defaults to ``config.yaml`` at the repo root.

    Returns:
        A :class:`Config` wrapper.

    Raises:
        ConfigError: If the file is missing, unparseable, or violates an
            invariant that must never be relaxed (currently: no shuffling).
    """
    cfg_path = Path(path) if path else DEFAULT_CONFIG
    if not cfg_path.exists():
        raise ConfigError(f"config not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise ConfigError(f"config did not parse to a mapping: {cfg_path}")

    cfg = Config(raw=raw, path=cfg_path)

    # Leakage rule 2 is a config-level invariant, not a runtime option.
    if bool(cfg.get("split.shuffle", False)):
        raise ConfigError(
            "split.shuffle is true. Chronological splitting is non-negotiable for "
            "this study; shuffling a time series leaks the future into training."
        )
    if bool(cfg.get("task.allow_future_meteorology", False)):
        raise ConfigError(
            "task.allow_future_meteorology is true. Observed weather at t+h is a "
            "perfect-forecast oracle. Use task.oracle_met_variant instead, which is "
            "reported as a clearly-labelled upper bound."
        )
    return cfg


def check_disk_space(cfg: Config, target: Path | None = None) -> float:
    """Assert the project drive has at least ``runtime.min_free_disk_gb`` free.

    A previous run on this box filled the system drive completely, so every
    script that writes calls this before doing so.

    Args:
        cfg: Loaded configuration.
        target: Path whose drive is checked. Defaults to the repository root.

    Returns:
        Free space in GB.

    Raises:
        DiskSpaceError: If free space is below the configured minimum.
    """
    required = float(cfg.get("runtime.min_free_disk_gb", 5.0))
    where = target or REPO_ROOT
    free_gb = shutil.disk_usage(where).free / 1024**3
    if free_gb < required:
        raise DiskSpaceError(
            f"only {free_gb:.1f} GB free on {where.drive or where.anchor}; "
            f"{required:.1f} GB required. Free space before continuing."
        )
    return free_gb


def set_seed(seed: int, cfg: Config | None = None) -> None:
    """Seed Python, NumPy and PyTorch, and enable deterministic kernels.

    Args:
        seed: The seed value.
        cfg: Optional config; ``determinism.*`` controls deterministic algorithms
            and the cuBLAS workspace setting required for reproducible CUDA
            matmul reductions.
    """
    deterministic = True
    cudnn_benchmark = False
    workspace = ":4096:8"
    if cfg is not None:
        deterministic = bool(cfg.get("determinism.torch_deterministic", True))
        cudnn_benchmark = bool(cfg.get("determinism.cudnn_benchmark", False))
        workspace = str(cfg.get("determinism.cublas_workspace_config", ":4096:8"))

    if deterministic:
        # Must be set before the first CUDA context is created.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", workspace)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        import numpy as np

        # Legacy global seeding is deliberate: scikit-learn, statsmodels and
        # SHAP all consume the module-level NumPy RNG internally, so a Generator
        # instance would not make those libraries reproducible.
        np.random.seed(seed)  # noqa: NPY002
    except ImportError:  # pragma: no cover - numpy is a hard dependency
        pass

    try:
        import torch
    except ImportError:  # pragma: no cover - torch optional for non-DL scripts
        return

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = cudnn_benchmark
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)


def resolve_device(cfg: Config, override: str | None = None) -> str:
    """Choose the compute device, verifying CUDA with a real kernel launch.

    ``torch.cuda.is_available()`` returns True on Blackwell even when the
    installed wheel has no sm_120 kernels; the failure only surfaces at the first
    dispatch. This runs one to find out now.

    Args:
        cfg: Loaded configuration (``runtime.device``).
        override: CLI override: ``"auto"``, ``"cuda"`` or ``"cpu"``.

    Returns:
        ``"cuda"`` or ``"cpu"``.
    """
    want = (override or str(cfg.get("runtime.device", "auto"))).lower()
    if want == "cpu":
        return "cpu"

    try:
        import torch
    except ImportError:
        return "cpu"

    if not torch.cuda.is_available():
        if want == "cuda":
            logging.getLogger(__name__).warning(
                "device=cuda requested but CUDA is unavailable; falling back to CPU."
            )
        return "cpu"

    try:
        probe = torch.randn(64, 64, device="cuda")
        _ = (probe @ probe).sum().item()
        torch.cuda.synchronize()
        del probe
        torch.cuda.empty_cache()
        return "cuda"
    except RuntimeError as exc:
        logging.getLogger(__name__).warning(
            "CUDA reported available but the kernel launch failed (%s). "
            "This is the Blackwell/sm_120 wheel mismatch -- reinstall torch from "
            "the cu128 index. Falling back to CPU.",
            exc,
        )
        return "cpu"


def git_commit_hash() -> str:
    """Return the current git commit hash, or a marker when unavailable.

    Recorded in ``results/results.json`` so every result is traceable to code.

    Returns:
        Full commit SHA, suffixed ``-dirty`` if the tree has uncommitted changes,
        or ``"unknown"`` if git is unavailable.
    """
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
            timeout=15,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
            timeout=15,
        ).stdout.strip()
        return f"{sha}-dirty" if dirty else sha
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return "unknown"


def setup_logging(cfg: Config, name: str, level: int = logging.INFO) -> logging.Logger:
    """Configure logging to both stdout and ``results/logs/<name>.log``.

    Args:
        cfg: Loaded configuration (``paths.logs``).
        name: Logger name; also the log file stem.
        level: Logging level.

    Returns:
        The configured logger.
    """
    log_dir = cfg.path_for("logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()
    logger.propagate = False

    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s", "%H:%M:%S")

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(fmt)
    logger.addHandler(stream)

    file_handler = logging.FileHandler(log_dir / f"{name}.log", encoding="utf-8")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    return logger
