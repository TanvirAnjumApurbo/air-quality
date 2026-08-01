"""Phase 0 environment check.

Reports the interpreter, PyTorch/CUDA status and free disk space on the project
drive, then runs a real GPU kernel launch.

The kernel launch matters: on Blackwell cards (RTX 50-series, compute capability
sm_120) an older PyTorch wheel will import cleanly and report
``torch.cuda.is_available() == True``, and only fail at the first actual kernel
dispatch with ``no kernel image is available for execution on the device``.
Checking availability alone is therefore not sufficient evidence that training
will run. This script fails loudly rather than deferring that discovery to the
middle of a training run.

Run::

    python scripts/00_check_env.py
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import platform
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
MIN_FREE_DISK_GB_FALLBACK = 5.0


def _rule(title: str = "") -> None:
    """Print a section rule, optionally labelled."""
    if title:
        print(f"\n{'-' * 72}\n{title}\n{'-' * 72}")
    else:
        print("-" * 72)


def _kv(key: str, value: Any) -> None:
    """Print an aligned key/value line."""
    print(f"  {key:<34} {value}")


def load_runtime_config() -> tuple[float, float | None]:
    """Read the runtime constraints this check validates against.

    Falls back to conservative defaults if PyYAML or the config is missing, so
    that the very first script in the pipeline can run before dependencies are
    fully installed.

    Returns:
        ``(min_free_disk_gb, ram_cache_budget_gb)``. The cache budget is None when
        the config could not be read.
    """
    cfg_path = REPO_ROOT / "config.yaml"
    if importlib.util.find_spec("yaml") is None or not cfg_path.exists():
        return MIN_FREE_DISK_GB_FALLBACK, None
    yaml = importlib.import_module("yaml")
    with cfg_path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    runtime = cfg.get("runtime", {})
    min_disk = float(runtime.get("min_free_disk_gb", MIN_FREE_DISK_GB_FALLBACK))
    budget = runtime.get("ram_cache", {}).get("budget_gb")
    return min_disk, (float(budget) if budget is not None else None)


def report_python() -> None:
    """Print interpreter and platform details."""
    _rule("PYTHON / PLATFORM")
    _kv("python version", platform.python_version())
    _kv("python implementation", platform.python_implementation())
    _kv("executable", sys.executable)
    _kv("in virtualenv", sys.prefix != sys.base_prefix)
    _kv("platform", platform.platform())
    _kv("machine", platform.machine())
    _kv("processor", platform.processor() or "n/a")


def report_disk(min_free_gb: float) -> bool:
    """Print free space on the project drive.

    Args:
        min_free_gb: Minimum free space the pipeline requires before writing.

    Returns:
        True if the project drive has at least ``min_free_gb`` free.
    """
    _rule("DISK")
    usage = shutil.disk_usage(REPO_ROOT)
    free_gb = usage.free / 1024**3
    total_gb = usage.total / 1024**3
    used_gb = usage.used / 1024**3
    drive = REPO_ROOT.drive or str(REPO_ROOT.anchor)

    _kv("project root", str(REPO_ROOT))
    _kv("drive", drive)
    _kv("total", f"{total_gb:,.1f} GB")
    _kv("used", f"{used_gb:,.1f} GB")
    _kv("free", f"{free_gb:,.1f} GB")
    _kv("required free", f"{min_free_gb:,.1f} GB")

    ok = free_gb >= min_free_gb
    _kv("status", "OK" if ok else "FAIL - insufficient free space")

    # The OS drive is called out separately: a previous run on this box filled it
    # completely, and pip/matplotlib/CodeCarbon default their caches and temp
    # staging there regardless of where the repo lives. Resolve it from the OS,
    # not from sys.prefix -- sys.prefix points at the venv, which lives on the
    # project drive and would silently report the wrong volume.
    system_root = Path(os.environ.get("SYSTEMROOT", "/")).anchor or "/"
    if Path(system_root).anchor != Path(REPO_ROOT.anchor).anchor:
        sys_usage = shutil.disk_usage(system_root)
        sys_free_gb = sys_usage.free / 1024**3
        _kv(f"OS drive {system_root} free", f"{sys_free_gb:,.1f} GB")
        if sys_free_gb < min_free_gb:
            _kv("OS drive status", "WARNING - pip/temp staging may fail")

    temp_root = Path(tempfile.gettempdir())
    temp_free_gb = shutil.disk_usage(temp_root).free / 1024**3
    _kv("temp dir", str(temp_root))
    _kv("temp dir free", f"{temp_free_gb:,.1f} GB")
    return ok


def report_ram(cache_budget_gb: float | None) -> None:
    """Print host RAM and check it against the configured in-RAM cache budget.

    Args:
        cache_budget_gb: ``runtime.ram_cache.budget_gb`` from config, or None if
            the config could not be read.
    """
    _rule("HOST MEMORY")
    if importlib.util.find_spec("psutil") is None:
        _kv("psutil", "NOT INSTALLED (skipping RAM report)")
        return
    psutil = importlib.import_module("psutil")
    vm = psutil.virtual_memory()
    available_gb = vm.available / 1024**3
    _kv("total RAM", f"{vm.total / 1024**3:,.1f} GB")
    _kv("available RAM", f"{available_gb:,.1f} GB")
    _kv("logical CPUs", psutil.cpu_count(logical=True))
    _kv("physical CPUs", psutil.cpu_count(logical=False))

    if cache_budget_gb is not None:
        _kv("configured RAM cache budget", f"{cache_budget_gb:,.1f} GB")
        if cache_budget_gb > available_gb:
            _kv("RAM cache status", "WARNING - budget exceeds available RAM")
            print("\n  The trainer clamps the cache to a fraction of RAM that is free at")
            print("  run time, so this will not OOM -- but the cache will be smaller than")
            print("  configured. Close other applications for the full benefit.")
        else:
            _kv("RAM cache status", "OK")


def report_torch() -> tuple[bool, bool]:
    """Print PyTorch build and CUDA capability, then launch a real GPU kernel.

    Returns:
        ``(torch_installed, cuda_usable)``. ``cuda_usable`` is True only if a
        kernel actually executed on the device -- not merely if CUDA reported
        itself available.
    """
    _rule("PYTORCH / CUDA")
    if importlib.util.find_spec("torch") is None:
        _kv("torch", "NOT INSTALLED")
        print("\n  Install the CUDA 12.8 build (required for Blackwell / sm_120):")
        print("    pip install torch --index-url https://download.pytorch.org/whl/cu128")
        return False, False

    torch = importlib.import_module("torch")
    _kv("torch version", torch.__version__)
    _kv("torch build CUDA version", torch.version.cuda or "cpu-only build")
    _kv("torch cuDNN version", torch.backends.cudnn.version() or "n/a")
    _kv("compiled arch list", " ".join(torch.cuda.get_arch_list()) or "none")

    available = torch.cuda.is_available()
    _kv("torch.cuda.is_available()", available)

    if not available:
        print("\n  CUDA is NOT available. Continuing on CPU is viable -- every model")
        print("  in this study is small enough to train on CPU.")
        return True, False

    _kv("device count", torch.cuda.device_count())
    _kv("torch.cuda.get_device_name(0)", torch.cuda.get_device_name(0))
    capability = torch.cuda.get_device_capability(0)
    _kv("torch.cuda.get_device_capability(0)", capability)
    _kv("  -> sm_", f"sm_{capability[0]}{capability[1]}")
    props = torch.cuda.get_device_properties(0)
    _kv("device total memory", f"{props.total_memory / 1024**3:,.2f} GB")
    _kv("multiprocessor count", props.multi_processor_count)

    arch_list = torch.cuda.get_arch_list()
    sm_tag = f"sm_{capability[0]}{capability[1]}"
    if arch_list and sm_tag not in arch_list:
        print(f"\n  WARNING: this wheel was not compiled for {sm_tag}.")
        print(f"  Compiled for: {' '.join(arch_list)}")
        print("  A kernel launch is expected to fail below.")

    # --- the check that actually matters -------------------------------------
    _rule("GPU KERNEL SMOKE TEST")
    try:
        a = torch.randn(512, 512, device="cuda")
        b = torch.randn(512, 512, device="cuda")
        c = (a @ b).sum().item()
        torch.cuda.synchronize()
        _kv("matmul on cuda", f"OK (checksum {c:.4f})")

        gru = torch.nn.GRU(input_size=8, hidden_size=32, batch_first=True).cuda()
        out, _ = gru(torch.randn(4, 24, 8, device="cuda"))
        torch.cuda.synchronize()
        _kv("cuDNN GRU forward", f"OK (output {tuple(out.shape)})")

        _kv("peak GPU memory", f"{torch.cuda.max_memory_allocated() / 1024**2:,.1f} MB")
        torch.cuda.empty_cache()
        return True, True
    except RuntimeError as exc:
        _kv("kernel launch", "FAILED")
        print(f"\n  {type(exc).__name__}: {exc}")
        print("\n  This is the classic Blackwell/sm_120 wheel mismatch. Reinstall:")
        print("    pip uninstall -y torch")
        print("    pip install torch --index-url https://download.pytorch.org/whl/cu128")
        return True, False


def report_packages() -> list[str]:
    """Print the version of every pipeline dependency.

    Returns:
        Names of packages that could not be imported.
    """
    _rule("PIPELINE PACKAGES")
    packages = [
        "numpy",
        "pandas",
        "scipy",
        "sklearn",
        "statsmodels",
        "matplotlib",
        "seaborn",
        "yaml",
        "requests",
        "boto3",
        "dotenv",
        "ucimlrepo",
        "xgboost",
        "lightgbm",
        "shap",
        "thop",
        "codecarbon",
        "tqdm",
        "pyarrow",
        "tabulate",
        "pytest",
        "ruff",
    ]
    missing: list[str] = []
    for name in packages:
        if importlib.util.find_spec(name) is None:
            _kv(name, "NOT INSTALLED")
            missing.append(name)
            continue
        try:
            mod = importlib.import_module(name)
            _kv(name, getattr(mod, "__version__", "installed (no __version__)"))
        except Exception as exc:
            _kv(name, f"IMPORT ERROR: {type(exc).__name__}: {exc}")
            missing.append(name)
    return missing


def main() -> int:
    """Run every environment check and return a process exit code."""
    print("=" * 72)
    print("PHASE 0 - ENVIRONMENT CHECK")
    print("Short-Horizon PM2.5 Forecasting for Dhaka")
    print("=" * 72)

    min_free_gb, cache_budget_gb = load_runtime_config()
    report_python()
    disk_ok = report_disk(min_free_gb)
    report_ram(cache_budget_gb)
    torch_ok, cuda_ok = report_torch()
    missing = report_packages()

    _rule("SUMMARY")
    _kv("disk >= required", "PASS" if disk_ok else "FAIL")
    _kv("torch importable", "PASS" if torch_ok else "FAIL")
    _kv("cuda kernels usable", "PASS" if cuda_ok else "NO - CPU fallback")
    _kv("missing packages", ", ".join(missing) if missing else "none")

    if not disk_ok:
        print(
            "\n  BLOCKING: free at least "
            f"{min_free_gb:.1f} GB on the project drive before running the pipeline."
        )
        return 1
    if not cuda_ok:
        print("\n  Proceeding on CPU. This is an accepted fallback for this study;")
        print("  training runs are sized to stay under the configured time budget.")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
