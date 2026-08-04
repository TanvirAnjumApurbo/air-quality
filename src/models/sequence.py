"""Tier 3: compact sequence forecasters.

The parameter ceiling is the contribution, not an inconvenience. Every
architecture is checked against ``models.sequence.max_params`` before training
and is **skipped and recorded** if it exceeds it, rather than quietly trained
anyway to chase a better RMSE. At the default 18 input channels the ceiling
still bites at the top of the grid -- both two-layer 128-wide recurrent models
are excluded, at 156k and 208k parameters -- while the single-layer variants fit.
It bit harder under the older 102-column input, where ``lstm_h128_l1`` came to
119k and was excluded; at 18 channels the same architecture is 76k and runs.

Two of the architectures here are not recurrent at all. DLinear and NLinear
(Zeng et al., AAAI 2023) are linear maps over the flattened window, costing a few
hundred to a few thousand parameters, and they are included because a compact-
model paper that never checks the simplest compact model has not made its case.

Training is engineered for the operating constraints of the target machine:

* the full scaled channel matrix (~6 MB) lives on the compute device, so batches
  are gathered on-device with no host-to-device copy in the loop;
* every epoch writes ``last.ckpt`` and, on improvement, ``best.ckpt``, both
  atomically, so an interrupted run resumes exactly where it stopped;
* a wall-clock budget aborts a run that would breach the thermal limit rather
  than letting it cook the machine.
"""

from __future__ import annotations

import contextlib
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from src.models.data import SequenceIndex
from src.utils import Config

# ---------------------------------------------------------------------------
# Architectures
# ---------------------------------------------------------------------------


class RecurrentForecaster(nn.Module):
    """A one- or two-layer GRU/LSTM with a linear head.

    The head reads the final hidden state, which for a direct multi-horizon setup
    is the whole forecast: one model per horizon, no recursive rollout.
    """

    def __init__(
        self,
        n_features: int,
        hidden_size: int,
        num_layers: int,
        cell: str,
        dropout: float,
    ) -> None:
        """Build the network.

        Args:
            n_features: Input features per timestep.
            hidden_size: Recurrent hidden width.
            num_layers: Number of stacked recurrent layers.
            cell: ``"gru"`` or ``"lstm"``.
            dropout: Dropout between recurrent layers and before the head.

        Raises:
            ValueError: If ``cell`` is unknown.
        """
        super().__init__()
        self.cell = cell.lower()
        rnn_cls = {"gru": nn.GRU, "lstm": nn.LSTM}.get(self.cell)
        if rnn_cls is None:
            raise ValueError(f"unknown cell {cell!r}; use 'gru' or 'lstm'")

        self.rnn = rnn_cls(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map a batch of windows to a scalar forecast each.

        Args:
            x: Batch of shape ``(batch, window, features)``.

        Returns:
            Predictions of shape ``(batch,)``.
        """
        output, _ = self.rnn(x)
        return self.head(self.dropout(output[:, -1, :])).squeeze(-1)


class AttentionForecaster(nn.Module):
    """A GRU with additive temporal attention over its output sequence.

    Included as the compact attention variant. Attention is over time only --
    there is no cross-feature mechanism -- which keeps the parameter count within
    the same budget as the plain recurrent models.
    """

    def __init__(
        self,
        n_features: int,
        hidden_size: int,
        num_layers: int,
        attention_dim: int,
        dropout: float,
    ) -> None:
        """Build the network.

        Args:
            n_features: Input features per timestep.
            hidden_size: Recurrent hidden width.
            num_layers: Number of stacked GRU layers.
            attention_dim: Width of the attention scoring layer.
            dropout: Dropout before the head.
        """
        super().__init__()
        self.rnn = nn.GRU(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.score = nn.Sequential(
            nn.Linear(hidden_size, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1),
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map a batch of windows to a scalar forecast each.

        Args:
            x: Batch of shape ``(batch, window, features)``.

        Returns:
            Predictions of shape ``(batch,)``.
        """
        output, _ = self.rnn(x)
        weights = torch.softmax(self.score(output).squeeze(-1), dim=1)
        context = torch.bmm(weights.unsqueeze(1), output).squeeze(1)
        return self.head(self.dropout(context)).squeeze(-1)


class SeriesDecomposition(nn.Module):
    """Split a window into a moving-average trend and the seasonal remainder.

    The ends are padded by repeating the first and last observation so the trend
    keeps the input length. Zero-padding would drag the trend towards zero at
    exactly the recent end of the window the forecast depends on most.
    """

    def __init__(self, kernel_size: int) -> None:
        """Build the smoother.

        Args:
            kernel_size: Moving-average width in hours; forced odd so the window
                is symmetric about each point.
        """
        super().__init__()
        self.kernel_size = int(kernel_size) | 1
        self.pad = (self.kernel_size - 1) // 2

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Decompose a batch of windows.

        Args:
            x: Batch of shape ``(batch, window, features)``.

        Returns:
            ``(seasonal, trend)``, both shaped like ``x``.
        """
        front = x[:, :1, :].repeat(1, self.pad, 1)
        back = x[:, -1:, :].repeat(1, self.pad, 1)
        padded = torch.cat([front, x, back], dim=1).transpose(1, 2)
        trend = torch.nn.functional.avg_pool1d(padded, self.kernel_size, stride=1).transpose(1, 2)
        return x - trend, trend


class DLinearForecaster(nn.Module):
    """DLinear (Zeng et al., AAAI 2023) adapted to scalar direct forecasting.

    The original decomposes the input into trend and seasonal components, maps
    each through its own linear layer along the time axis, and sums the results.
    That is reproduced here; the adaptation is in the output.

    The original predicts a horizon-length sequence for every channel. This study
    is direct multi-horizon -- one model per *h* emitting a single scalar -- so
    each component is mapped straight to a scalar by a linear layer over the
    flattened ``(window, channels)`` input. Note that the natural alternative, a
    per-channel temporal filter followed by a channel-mixing head, collapses
    algebraically to exactly this single linear map when the output is scalar, so
    nothing is lost by writing it directly.

    The point of including it is that it is the baseline that showed simple
    linear models matching transformers on long-horizon forecasting. If a linear
    map on the raw window competes with the recurrent tier here, that is a
    finding, and it costs a few thousand parameters to check.
    """

    def __init__(self, n_features: int, window: int, kernel_size: int, dropout: float) -> None:
        """Build the network.

        Args:
            n_features: Input channels per timestep.
            window: Input window length in hours.
            kernel_size: Trend moving-average width; clamped to fit the window.
            dropout: Dropout applied to the flattened components before the head.
        """
        super().__init__()
        self.decomp = SeriesDecomposition(min(int(kernel_size), int(window)))
        self.dropout = nn.Dropout(dropout)
        flat = int(window) * int(n_features)
        self.seasonal = nn.Linear(flat, 1)
        self.trend = nn.Linear(flat, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map a batch of windows to a scalar forecast each.

        Args:
            x: Batch of shape ``(batch, window, features)``.

        Returns:
            Predictions of shape ``(batch,)``.
        """
        seasonal, trend = self.decomp(x)
        seasonal = self.dropout(seasonal.flatten(1))
        trend = self.dropout(trend.flatten(1))
        return (self.seasonal(seasonal) + self.trend(trend)).squeeze(-1)


class NLinearForecaster(nn.Module):
    """NLinear (Zeng et al., AAAI 2023) adapted to a transformed target.

    The original subtracts the last value of the input sequence, applies one
    linear layer, and adds that same value back -- a parameter-free way to
    absorb level shifts between train and test.

    The identity add-back cannot be used here. Input channels are standardised by
    the train-fitted scaler while the target is modelled on ``log1p``, so the last
    observed value and the quantity being predicted live on different scales, and
    adding one to the other would be a unit error rather than a normalisation.
    The level term is therefore reintroduced through a learned affine map on the
    target channel's last value, which preserves the mechanism -- the linear layer
    only has to predict a *deviation* from the current level -- while staying
    dimensionally coherent. The cost is two extra parameters.

    The target channel is index 0 by construction: ``sequence_channel_columns``
    inherits the ordering of ``feature_columns``, which inserts the target first.
    """

    def __init__(self, n_features: int, window: int, dropout: float) -> None:
        """Build the network.

        Args:
            n_features: Input channels per timestep.
            window: Input window length in hours.
            dropout: Dropout applied to the flattened window before the head.
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Linear(int(window) * int(n_features), 1)
        self.level = nn.Linear(1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map a batch of windows to a scalar forecast each.

        Args:
            x: Batch of shape ``(batch, window, features)``.

        Returns:
            Predictions of shape ``(batch,)``.
        """
        last = x[:, -1:, :]
        deviation = self.projection(self.dropout((x - last).flatten(1)))
        return (deviation + self.level(last[:, :, 0])).squeeze(-1)


def count_parameters(model: nn.Module) -> int:
    """Total trainable parameters.

    Args:
        model: The network.

    Returns:
        Number of trainable parameters.
    """
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


# Architectures whose parameter count depends on the input window rather than on
# a hidden width. They are sized by the window, so the budget filter cannot cache
# their counts by name alone and the sweep tables identify them by window only.
WINDOW_SIZED_ARCHS = frozenset({"dlinear", "nlinear"})


def build_model(spec: ModelSpec, n_features: int, dropout: float) -> nn.Module:
    """Instantiate the network described by a specification.

    Args:
        spec: The architecture specification.
        n_features: Input features per timestep.
        dropout: Dropout rate.

    Returns:
        The instantiated model.
    """
    if spec.arch == "gru_attention":
        return AttentionForecaster(
            n_features, spec.hidden_size, spec.num_layers, spec.attention_dim or 32, dropout
        )
    if spec.arch == "dlinear":
        return DLinearForecaster(n_features, spec.window, spec.kernel_size or 25, dropout)
    if spec.arch == "nlinear":
        return NLinearForecaster(n_features, spec.window, dropout)
    return RecurrentForecaster(n_features, spec.hidden_size, spec.num_layers, spec.arch, dropout)


# ---------------------------------------------------------------------------
# Sweep specification
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelSpec:
    """One point in the sequence-model sweep.

    Attributes:
        arch: ``"gru"``, ``"lstm"``, ``"gru_attention"``, ``"dlinear"`` or
            ``"nlinear"``.
        hidden_size: Recurrent hidden width. Unused by the window-sized
            architectures, which have no hidden state.
        num_layers: Stacked recurrent layers. Unused as above.
        window: Input window length in hours.
        horizon: Forecast horizon in hours.
        seed: RNG seed.
        attention_dim: Attention width, for the attention variant only.
        kernel_size: Trend moving-average width, for DLinear only.
    """

    arch: str
    hidden_size: int
    num_layers: int
    window: int
    horizon: int
    seed: int
    attention_dim: int | None = None
    kernel_size: int | None = None

    @property
    def name(self) -> str:
        """Architecture name including size, for tables."""
        if self.arch in WINDOW_SIZED_ARCHS:
            # Sized by the window, which the ``variant`` column already carries;
            # a hidden width here would be a fiction.
            return self.arch
        base = f"{self.arch}_h{self.hidden_size}_l{self.num_layers}"
        return base if self.attention_dim is None else f"{base}_a{self.attention_dim}"

    @property
    def run_id(self) -> str:
        """Unique identifier, used as the checkpoint directory name."""
        return f"{self.name}_w{self.window}_H{self.horizon}_s{self.seed}"


def enumerate_specs(cfg: Config, horizons: list[int], seeds: list[int]) -> list[ModelSpec]:
    """Expand the configured architecture grid into individual runs.

    Args:
        cfg: Loaded configuration.
        horizons: Horizons to cover.
        seeds: Seeds to cover.

    Returns:
        Every specification in the sweep, before the parameter-budget filter.
    """
    sequence = cfg.get("models.sequence")
    specs: list[ModelSpec] = []
    for arch, arch_cfg in sequence["architectures"].items():
        if not arch_cfg.get("enabled", False):
            continue
        # The window-sized architectures have neither a hidden width nor stacked
        # layers, so they declare neither and collapse to a single grid point per
        # window.
        for hidden in arch_cfg.get("hidden_sizes", [0]):
            for layers in arch_cfg.get("num_layers", [1]):
                for window in sequence["input_windows_h"]:
                    for horizon in horizons:
                        for seed in seeds:
                            specs.append(
                                ModelSpec(
                                    arch=arch,
                                    hidden_size=int(hidden),
                                    num_layers=int(layers),
                                    window=int(window),
                                    horizon=int(horizon),
                                    seed=int(seed),
                                    attention_dim=(
                                        int(arch_cfg.get("attention_dim", 32))
                                        if arch == "gru_attention"
                                        else None
                                    ),
                                    kernel_size=(
                                        int(arch_cfg.get("kernel_size", 25))
                                        if arch == "dlinear"
                                        else None
                                    ),
                                )
                            )
    return specs


def filter_by_parameter_budget(
    specs: list[ModelSpec], cfg: Config, n_features: int, logger: Any
) -> tuple[list[ModelSpec], list[dict[str, Any]]]:
    """Split the sweep into runs that fit the parameter ceiling and those that do not.

    Exclusions are returned rather than discarded so the paper can report which
    configurations the green-AI constraint ruled out, which is part of the
    finding rather than a detail to hide.

    Args:
        specs: Candidate specifications.
        cfg: Loaded configuration.
        n_features: Input features per timestep.
        logger: Logger.

    Returns:
        ``(kept, excluded)`` where ``excluded`` records name and parameter count.
    """
    max_params = int(cfg.get("models.sequence.max_params"))
    dropout = float(cfg.get("models.sequence.train.dropout"))

    kept: list[ModelSpec] = []
    excluded: list[dict[str, Any]] = []
    # Keyed by window as well as name: a recurrent model's size is independent of
    # the window, but DLinear and NLinear are sized *by* it, so caching on the
    # name alone would report whichever window happened to be built first.
    seen: dict[tuple[str, int], int] = {}

    for spec in specs:
        key = (spec.name, spec.window)
        if key not in seen:
            model = build_model(spec, n_features, dropout)
            seen[key] = count_parameters(model)
            del model
        n_params = seen[key]
        if n_params <= max_params:
            kept.append(spec)
        else:
            entry = {"name": spec.name, "params": n_params, "budget": max_params}
            if entry not in excluded:
                excluded.append(entry)

    for entry in excluded:
        logger.warning(
            "excluded %s: %d parameters exceeds the %d budget",
            entry["name"],
            entry["params"],
            entry["budget"],
        )
    return kept, excluded


# ---------------------------------------------------------------------------
# Device-resident batching
# ---------------------------------------------------------------------------


class DeviceWindowSampler:
    """Gathers sequence windows on the compute device.

    The full feature matrix is uploaded once. A batch is then a single gather
    over precomputed offsets, so the training loop performs no host-to-device
    transfer and needs no DataLoader workers -- with the data already resident,
    any worker count is strictly slower.
    """

    def __init__(self, index: SequenceIndex, device: str, budget_bytes: int | None = None) -> None:
        """Upload the matrix and prepare the window offsets.

        Args:
            index: The window index for this split.
            device: ``"cuda"`` or ``"cpu"``.
            budget_bytes: Optional cache ceiling; the matrix falls back to CPU if
                it would be exceeded.
        """
        self.index = index
        self.device = device
        matrix_bytes = index.matrix.nbytes
        self.on_device = device == "cuda" and (budget_bytes is None or matrix_bytes <= budget_bytes)

        target = device if self.on_device else "cpu"
        self.matrix = torch.from_numpy(index.matrix).to(target)
        self.ends = torch.from_numpy(index.end_positions).to(target)
        self.y = torch.from_numpy(index.y_transformed.astype(np.float32)).to(target)
        # Offsets are shared by every window, so build them once.
        self.offsets = torch.arange(-index.window + 1, 1, device=target, dtype=torch.long)
        self.matrix_bytes = matrix_bytes

    def __len__(self) -> int:
        """Number of windows."""
        return int(self.ends.numel())

    def batch(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather one batch of windows and targets.

        Args:
            positions: Indices into the window list.

        Returns:
            ``(x, y)`` with ``x`` shaped ``(batch, window, features)``.
        """
        ends = self.ends[positions]
        rows = ends.unsqueeze(1) + self.offsets.unsqueeze(0)
        x = self.matrix[rows]
        y = self.y[positions]
        if not self.on_device and self.device == "cuda":
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
        return x, y


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


@dataclass
class TrainHistory:
    """Per-epoch training record.

    Attributes:
        epochs: Epoch numbers completed.
        train_loss: Training loss per epoch.
        val_loss: Validation loss per epoch.
        lr: Learning rate per epoch.
        epoch_seconds: Wall-clock seconds per epoch.
    """

    epochs: list[int] = field(default_factory=list)
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    lr: list[float] = field(default_factory=list)
    epoch_seconds: list[float] = field(default_factory=list)


@dataclass
class TrainResult:
    """Outcome of one training run.

    Attributes:
        spec: The specification trained.
        n_params: Trainable parameter count.
        best_val_loss: Lowest validation loss reached.
        best_epoch: Epoch at which that occurred.
        epochs_run: Epochs actually executed this invocation.
        total_epochs: Epochs completed including any resumed from.
        train_seconds: Wall-clock training seconds this invocation.
        stopped_reason: Why training ended.
        history: Per-epoch record.
        checkpoint_dir: Where checkpoints were written.
    """

    spec: ModelSpec
    n_params: int
    best_val_loss: float
    best_epoch: int
    epochs_run: int
    total_epochs: int
    train_seconds: float
    stopped_reason: str
    history: TrainHistory
    checkpoint_dir: str


def _atomic_save(payload: dict[str, Any], path: Path, logger: Any = None) -> None:
    """Write a checkpoint atomically, tolerating transient Windows file locks.

    A direct ``torch.save`` to the final path leaves a truncated file if the
    process dies mid-write, which then poisons the next resume -- hence the
    write-then-rename.

    The rename itself needs guarding on Windows. ``os.replace`` raises
    ``PermissionError`` (WinError 5) whenever another process holds a handle to
    the destination, which real-time antivirus scanning does routinely and
    briefly for a freshly written multi-megabyte file. That is a transient
    condition, but unguarded it killed a 675-run sweep at run 191. Retrying with
    a short backoff clears it; if every retry fails the checkpoint is skipped
    with a warning rather than aborting a training run that is otherwise fine.

    Args:
        payload: Checkpoint contents.
        path: Destination path.
        logger: Optional logger for retry and failure reporting.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)

    delay = 0.2
    for attempt in range(6):
        try:
            tmp.replace(path)
            return
        except PermissionError:
            # A stale destination sometimes has to go before the rename lands.
            if attempt >= 2:
                with contextlib.suppress(OSError):
                    path.unlink(missing_ok=True)
            time.sleep(delay)
            delay *= 2

    if logger is not None:
        logger.warning(
            "could not replace %s after 6 attempts (file locked, most likely by "
            "antivirus); leaving the previous checkpoint in place and continuing",
            path.name,
        )
    tmp.unlink(missing_ok=True)


def _make_scheduler(optimiser: torch.optim.Optimizer, cfg: Config, epochs: int) -> tuple[Any, bool]:
    """Build the learning-rate schedule.

    ``reduce_on_plateau`` reacts to the validation loss, which on this data is
    noisy enough that the reaction is often to noise: the original sweep decayed
    at epochs 7 and 12 and was then stopped at 13, so the schedule never
    materially annealed. ``warmup_cosine`` instead follows a fixed path from a
    warmup ramp down to ``min_lr``, decoupling the schedule from validation
    noise and letting every run finish the anneal it started.

    Args:
        optimiser: The optimiser to schedule.
        cfg: Loaded configuration (``models.sequence.train.scheduler``).
        epochs: Total epoch budget, which sets the cosine period.

    Returns:
        ``(scheduler, needs_metric)`` -- ``needs_metric`` is True when
        ``scheduler.step`` expects the validation loss as an argument.

    Raises:
        ValueError: If the scheduler name is unknown.
    """
    spec = cfg.get("models.sequence.train.scheduler")
    name = str(spec.get("name", "warmup_cosine")).lower()

    if name == "reduce_on_plateau":
        return (
            torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimiser,
                mode="min",
                factor=float(spec["factor"]),
                patience=int(spec["patience"]),
                min_lr=float(spec["min_lr"]),
            ),
            True,
        )

    if name != "warmup_cosine":
        raise ValueError(
            f"models.sequence.train.scheduler.name must be 'warmup_cosine' or "
            f"'reduce_on_plateau', got {name!r}"
        )

    base_lr = float(cfg.get("models.sequence.train.lr"))
    warmup = max(1, int(spec.get("warmup_epochs", 3)))
    floor = float(spec.get("min_lr", 1e-5)) / base_lr if base_lr > 0 else 0.0

    def factor(epoch: int) -> float:
        if epoch < warmup:
            return (epoch + 1) / warmup
        progress = (epoch - warmup) / max(1, epochs - warmup)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        return floor + (1.0 - floor) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimiser, factor), False


def _make_loss(cfg: Config) -> nn.Module:
    """Build the configured loss function.

    Returns:
        The loss module. Huber is the default: PM2.5 episodes produce large
        residuals that squared error would let dominate the gradient.
    """
    name = str(cfg.get("models.sequence.train.loss", "huber")).lower()
    if name == "huber":
        return nn.HuberLoss(delta=float(cfg.get("models.sequence.train.huber_delta", 1.0)))
    if name == "mae":
        return nn.L1Loss()
    return nn.MSELoss()


@torch.no_grad()
def evaluate_sampler(
    model: nn.Module, sampler: DeviceWindowSampler, loss_fn: nn.Module, batch_size: int
) -> tuple[float, np.ndarray]:
    """Run the model over a whole split.

    Args:
        model: The network.
        sampler: Window sampler for the split.
        loss_fn: Loss used for reporting.
        batch_size: Evaluation batch size.

    Returns:
        ``(mean_loss, predictions_on_model_scale)``.
    """
    model.eval()
    n = len(sampler)
    preds = torch.empty(n, dtype=torch.float32, device="cpu")
    total = 0.0
    for start in range(0, n, batch_size):
        positions = torch.arange(start, min(start + batch_size, n), device=sampler.ends.device)
        x, y = sampler.batch(positions)
        out = model(x)
        total += float(loss_fn(out, y)) * positions.numel()
        preds[start : start + positions.numel()] = out.detach().float().cpu()
    return total / max(n, 1), preds.numpy()


def train_one(
    spec: ModelSpec,
    train_index: SequenceIndex,
    val_index: SequenceIndex,
    cfg: Config,
    device: str,
    logger: Any,
    *,
    resume: str = "auto",
    progress: str = "tqdm",
    max_minutes: float | None = None,
    run_tag: str = "",
) -> tuple[nn.Module, TrainResult]:
    """Train one sequence model, with checkpointing and resume.

    Args:
        spec: The configuration to train.
        train_index: Training window index.
        val_index: Validation window index.
        cfg: Loaded configuration.
        device: Compute device.
        logger: Logger.
        resume: ``"auto"``, ``"never"`` or ``"always"``.
        progress: ``"tqdm"`` or ``"plain"``.
        max_minutes: Wall-clock ceiling for this run.
        run_tag: Optional checkpoint sub-directory. ``run_id`` encodes the
            architecture but not the training recipe, so a recipe search would
            otherwise resume the previous recipe's checkpoint and silently
            evaluate the wrong thing.

    Returns:
        ``(model_with_best_weights, result)``.
    """
    from src.utils import set_seed

    train_cfg = cfg.get("models.sequence.train")
    set_seed(spec.seed, cfg)

    ckpt_dir = cfg.path_for("checkpoints") / run_tag / spec.run_id
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    last_path = ckpt_dir / "last.ckpt"
    best_path = ckpt_dir / "best.ckpt"

    budget_bytes = None
    ram_cache = cfg.get("runtime.ram_cache", {})
    if ram_cache.get("enabled", True):
        budget_bytes = int(float(ram_cache.get("budget_gb", 8.0)) * 1024**3)

    train_sampler = DeviceWindowSampler(train_index, device, budget_bytes)
    val_sampler = DeviceWindowSampler(val_index, device, budget_bytes)

    model = build_model(spec, train_index.n_features, float(train_cfg["dropout"])).to(device)
    n_params = count_parameters(model)

    optimiser = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    scheduler, scheduler_needs_metric = _make_scheduler(optimiser, cfg, int(train_cfg["epochs"]))
    loss_fn = _make_loss(cfg)

    history = TrainHistory()
    start_epoch = 0
    best_val = math.inf
    best_epoch = -1
    best_state: dict[str, Any] | None = None

    # ---- resume ----------------------------------------------------------
    if resume != "never" and last_path.exists():
        try:
            payload = torch.load(last_path, map_location=device, weights_only=False)
            model.load_state_dict(payload["model"])
            optimiser.load_state_dict(payload["optimiser"])
            scheduler.load_state_dict(payload["scheduler"])
            start_epoch = int(payload["epoch"]) + 1
            best_val = float(payload.get("best_val_loss", math.inf))
            best_epoch = int(payload.get("best_epoch", -1))
            history = TrainHistory(**payload.get("history", {}))
            logger.info(
                "%s: resumed from epoch %d (best val %.5f)", spec.run_id, start_epoch, best_val
            )
        except Exception as exc:
            logger.warning("%s: checkpoint unreadable (%s); starting fresh", spec.run_id, exc)
            start_epoch = 0
    elif resume == "always" and not last_path.exists():
        logger.warning("%s: --resume always but no checkpoint found; starting fresh", spec.run_id)

    if best_path.exists():
        try:
            best_state = torch.load(best_path, map_location=device, weights_only=False)["model"]
        except Exception:
            best_state = None

    epochs = int(train_cfg["epochs"])
    batch_size = int(train_cfg["batch_size"])
    grad_clip = float(train_cfg["grad_clip_norm"])
    early = train_cfg["early_stopping"]
    patience = int(early["patience"])
    min_delta = float(early["min_delta"])
    # Validation loss here is noisy enough that the first few epochs routinely
    # produce a minimum no later epoch beats by min_delta. Without a floor the
    # run then ends at epoch ~13 of 60 with the cosine schedule barely started --
    # which is what the original sweep did, in all 1,353 runs.
    min_epochs = int(early.get("min_epochs", 0))

    amp_cfg = cfg.get("runtime.amp", {})
    use_amp = bool(amp_cfg.get("enabled", True)) and device == "cuda"
    amp_dtype = torch.bfloat16 if str(amp_cfg.get("dtype", "bf16")) == "bf16" else torch.float16

    n_train = len(train_sampler)
    steps_per_epoch = max(1, math.ceil(n_train / batch_size))
    generator = torch.Generator(device="cpu").manual_seed(spec.seed)

    logger.info(
        "%s: %d params | train %d windows, val %d | %d steps/epoch | matrix %.1f MB %s",
        spec.run_id,
        n_params,
        n_train,
        len(val_sampler),
        steps_per_epoch,
        train_sampler.matrix_bytes / 1024**2,
        "on device" if train_sampler.on_device else "on host",
    )

    started = time.perf_counter()
    epochs_run = 0
    stopped_reason = "completed"
    epochs_without_improvement = 0

    bar = None
    if progress == "tqdm" and bool(cfg.get("runtime.progress.enabled", True)):
        try:
            from tqdm.auto import tqdm

            bar = tqdm(
                total=epochs - start_epoch,
                desc=spec.run_id,
                unit="epoch",
                dynamic_ncols=True,
                leave=False,
            )
        except ImportError:
            bar = None

    for epoch in range(start_epoch, epochs):
        epoch_started = time.perf_counter()
        model.train()
        order = torch.randperm(n_train, generator=generator).to(train_sampler.ends.device)
        running = 0.0

        for step in range(steps_per_epoch):
            positions = order[step * batch_size : (step + 1) * batch_size]
            if positions.numel() == 0:
                continue
            x, y = train_sampler.batch(positions)

            optimiser.zero_grad(set_to_none=True)
            if use_amp:
                with torch.autocast("cuda", dtype=amp_dtype):
                    loss = loss_fn(model(x), y)
            else:
                loss = loss_fn(model(x), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimiser.step()
            running += float(loss.detach()) * positions.numel()

        train_loss = running / max(n_train, 1)
        val_loss, _ = evaluate_sampler(model, val_sampler, loss_fn, batch_size)
        # Read the rate the epoch actually ran at, before the step moves it on.
        current_lr = float(optimiser.param_groups[0]["lr"])
        if scheduler_needs_metric:
            scheduler.step(val_loss)
        else:
            scheduler.step()
        elapsed = time.perf_counter() - epoch_started

        history.epochs.append(epoch)
        history.train_loss.append(train_loss)
        history.val_loss.append(val_loss)
        history.lr.append(current_lr)
        history.epoch_seconds.append(elapsed)
        epochs_run += 1

        improved = val_loss < best_val - min_delta
        if improved:
            best_val = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
            _atomic_save(
                {"model": best_state, "epoch": epoch, "val_loss": val_loss, "spec": asdict(spec)},
                best_path,
                logger,
            )
        else:
            epochs_without_improvement += 1

        if bool(cfg.get("runtime.checkpointing.save_every_epoch", True)):
            _atomic_save(
                {
                    "model": model.state_dict(),
                    "optimiser": optimiser.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": epoch,
                    "best_val_loss": best_val,
                    "best_epoch": best_epoch,
                    "history": asdict(history),
                    "spec": asdict(spec),
                },
                last_path,
                logger,
            )

        if bar is not None:
            bar.update(1)
            bar.set_postfix(
                train=f"{train_loss:.5f}", val=f"{val_loss:.5f}", lr=f"{current_lr:.2e}"
            )
        else:
            logger.info(
                "%s epoch %3d/%d  train %.5f  val %.5f  lr %.2e  %.1fs%s",
                spec.run_id,
                epoch + 1,
                epochs,
                train_loss,
                val_loss,
                current_lr,
                elapsed,
                "  *" if improved else "",
            )

        if epochs_without_improvement >= patience and (epoch + 1) >= min_epochs:
            stopped_reason = f"early stopping after {patience} epochs without improvement"
            break

        if max_minutes is not None and (time.perf_counter() - started) / 60.0 > max_minutes:
            stopped_reason = f"wall-clock budget of {max_minutes:.1f} min reached"
            logger.warning("%s: %s", spec.run_id, stopped_reason)
            break

    if bar is not None:
        bar.close()

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, TrainResult(
        spec=spec,
        n_params=n_params,
        best_val_loss=best_val,
        best_epoch=best_epoch,
        epochs_run=epochs_run,
        total_epochs=start_epoch + epochs_run,
        train_seconds=time.perf_counter() - started,
        stopped_reason=stopped_reason,
        history=history,
        checkpoint_dir=str(ckpt_dir),
    )


def save_history(result: TrainResult) -> Path:
    """Write the per-epoch training history alongside the checkpoints.

    Args:
        result: The training outcome.

    Returns:
        Path written.
    """
    path = Path(result.checkpoint_dir) / "history.json"
    path.write_text(
        json.dumps(
            {
                "spec": asdict(result.spec),
                "n_params": result.n_params,
                "best_val_loss": result.best_val_loss,
                "best_epoch": result.best_epoch,
                "total_epochs": result.total_epochs,
                "train_seconds": result.train_seconds,
                "stopped_reason": result.stopped_reason,
                "history": asdict(result.history),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return path
