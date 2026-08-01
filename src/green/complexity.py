"""Model complexity: parameters, MACs, latency and peak memory.

Latency is the number most often reported carelessly. Measuring it correctly on a
GPU requires synchronising the device -- CUDA calls are asynchronous, so timing
without a synchronise measures how fast Python can enqueue work, not how fast the
work runs. Warm-up matters too: the first calls pay for allocator setup and
kernel autotuning.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

from src.utils import Config


@dataclass
class ComplexityReport:
    """Measured cost of one model.

    Attributes:
        n_params: Trainable parameters.
        n_params_total: All parameters, including any frozen.
        macs: Multiply-accumulate operations for a single-sample forward pass.
        flops_estimate: Roughly ``2 x macs``, the usual convention.
        latency_ms_mean: Mean single-sample inference latency.
        latency_ms_std: Standard deviation of that latency.
        latency_ms_p50: Median latency.
        latency_ms_p95: 95th percentile latency.
        n_latency_runs: Timed iterations.
        n_warmup: Discarded warm-up iterations.
        peak_gpu_memory_mb: Peak allocated GPU memory during inference.
        device: Device the measurement ran on.
        macs_backend: Which tool produced the MAC count, or why none did.
    """

    n_params: int
    n_params_total: int
    macs: int | None
    flops_estimate: int | None
    latency_ms_mean: float
    latency_ms_std: float
    latency_ms_p50: float
    latency_ms_p95: float
    n_latency_runs: int
    n_warmup: int
    peak_gpu_memory_mb: float | None
    device: str
    macs_backend: str


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Count trainable and total parameters.

    Args:
        model: The network.

    Returns:
        ``(trainable, total)``.
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return int(trainable), int(total)


def count_macs(
    model: nn.Module, sample: torch.Tensor, backend: str = "thop"
) -> tuple[int | None, str]:
    """Count multiply-accumulates for one forward pass.

    Args:
        model: The network.
        sample: A single-sample input tensor.
        backend: ``"thop"`` or ``"ptflops"``.

    Returns:
        ``(macs, backend_used)``; macs is None when no backend could measure it.
    """
    if backend == "thop":
        try:
            from thop import profile

            macs, _ = profile(model, inputs=(sample,), verbose=False)
            return int(macs), "thop"
        except Exception as exc:
            return None, f"thop failed: {type(exc).__name__}: {exc}"

    try:
        from ptflops import get_model_complexity_info

        macs, _ = get_model_complexity_info(
            model, tuple(sample.shape[1:]), as_strings=False, print_per_layer_stat=False
        )
        return int(macs), "ptflops"
    except Exception as exc:
        return None, f"ptflops failed: {type(exc).__name__}: {exc}"


@torch.no_grad()
def measure_latency(
    model: nn.Module,
    sample: torch.Tensor,
    device: str,
    n_warmup: int,
    n_runs: int,
) -> tuple[float, float, float, float]:
    """Time single-sample inference.

    Synchronises the device around every timed iteration, so the result is the
    time the work takes rather than the time to enqueue it.

    Args:
        model: The network, already on ``device``.
        sample: A single-sample input tensor on ``device``.
        device: ``"cuda"`` or ``"cpu"``.
        n_warmup: Iterations to run and discard.
        n_runs: Timed iterations.

    Returns:
        ``(mean_ms, std_ms, p50_ms, p95_ms)``.
    """
    import time

    model.eval()
    for _ in range(n_warmup):
        model(sample)
    if device == "cuda":
        torch.cuda.synchronize()

    timings = np.empty(n_runs, dtype=float)
    for i in range(n_runs):
        if device == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        model(sample)
        if device == "cuda":
            torch.cuda.synchronize()
        timings[i] = (time.perf_counter() - started) * 1000.0

    return (
        float(timings.mean()),
        float(timings.std(ddof=1)) if n_runs > 1 else 0.0,
        float(np.percentile(timings, 50)),
        float(np.percentile(timings, 95)),
    )


def profile_model(
    model: nn.Module,
    window: int,
    n_features: int,
    cfg: Config,
    device: str,
    logger: Any,
) -> ComplexityReport:
    """Measure parameters, MACs, latency and peak memory for one model.

    Args:
        model: The network.
        window: Input window length in hours.
        n_features: Features per timestep.
        cfg: Loaded configuration (``green.complexity``).
        device: Compute device.
        logger: Logger.

    Returns:
        The complexity report.
    """
    spec = cfg.get("green.complexity")
    latency_cfg = spec["latency"]

    model = model.to(device)
    trainable, total = count_parameters(model)

    batch = int(latency_cfg.get("batch_size", 1))
    sample = torch.randn(batch, window, n_features, device=device)

    # MACs are counted on a CPU copy: thop attaches hooks and mutates the module,
    # and doing that to the live GPU model risks perturbing the latency run.
    import copy

    cpu_model = copy.deepcopy(model).to("cpu")
    macs, backend = count_macs(cpu_model, sample.to("cpu"), str(spec.get("macs_backend", "thop")))
    del cpu_model

    if device == "cuda" and bool(spec.get("measure_peak_gpu_memory", True)):
        torch.cuda.reset_peak_memory_stats()

    mean_ms, std_ms, p50, p95 = measure_latency(
        model,
        sample,
        device,
        int(latency_cfg.get("n_warmup", 100)),
        int(latency_cfg.get("n_runs", 1000)),
    )

    peak_mb = None
    if device == "cuda" and bool(spec.get("measure_peak_gpu_memory", True)):
        peak_mb = float(torch.cuda.max_memory_allocated() / 1024**2)

    logger.info(
        "complexity: %d params, %s MACs, %.4f ms/sample (p95 %.4f), peak %s MB",
        trainable,
        f"{macs:,}" if macs else "n/a",
        mean_ms,
        p95,
        f"{peak_mb:.1f}" if peak_mb else "n/a",
    )

    return ComplexityReport(
        n_params=trainable,
        n_params_total=total,
        macs=macs,
        flops_estimate=2 * macs if macs else None,
        latency_ms_mean=mean_ms,
        latency_ms_std=std_ms,
        latency_ms_p50=p50,
        latency_ms_p95=p95,
        n_latency_runs=int(latency_cfg.get("n_runs", 1000)),
        n_warmup=int(latency_cfg.get("n_warmup", 100)),
        peak_gpu_memory_mb=peak_mb,
        device=device,
        macs_backend=backend,
    )
