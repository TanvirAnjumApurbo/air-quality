r"""Figure styling and output helpers.

Every figure is written at the configured DPI in both PNG and PDF so the same
artefact serves screen review and ``\includegraphics``. Styling is driven entirely
by ``output.figures`` in ``config.yaml``.

The categorical palette is applied in a fixed order and never cycled: colour
follows the entity, not its rank, so adding or removing a model must not repaint
the survivors. See the validation record in ``config.yaml`` for why the palette is
the reduced six-colour Okabe-Ito ordering rather than the full eight.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless: figures are files, never windows

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from src.utils import Config


def setup_style(cfg: Config) -> list[str]:
    """Apply the project-wide matplotlib style.

    Args:
        cfg: Loaded configuration (``output.figures``).

    Returns:
        The categorical palette, in assignment order.
    """
    figures = cfg.get("output.figures")
    palette = list(figures["palette"])

    style = str(figures.get("style", "seaborn-v0_8-whitegrid"))
    if style in plt.style.available:
        plt.style.use(style)

    plt.rcParams.update(
        {
            "figure.dpi": 110,  # on-screen; save dpi set at write time
            "savefig.dpi": int(figures.get("dpi", 300)),
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "font.size": float(figures.get("font_size", 11)),
            "axes.titlesize": float(figures.get("font_size", 11)) + 1,
            "axes.labelsize": float(figures.get("font_size", 11)),
            "legend.fontsize": float(figures.get("font_size", 11)) - 1,
            "xtick.labelsize": float(figures.get("font_size", 11)) - 1,
            "ytick.labelsize": float(figures.get("font_size", 11)) - 1,
            "axes.prop_cycle": plt.cycler(color=palette),
            # Recessive chrome: the grid must never compete with the marks.
            "axes.grid": True,
            "grid.color": str(figures.get("grid_colour", "#D9D9D9")),
            "grid.linewidth": 0.6,
            "grid.alpha": 0.9,
            "axes.edgecolor": "#B0B0B0",
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "lines.linewidth": 1.6,
            "lines.markersize": 4.5,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    return palette


def series_colour(cfg: Config, index: int) -> str:
    """Return the categorical colour for a series position.

    Assignment is by fixed position, never cycled past the palette length: a
    seventh series must be folded into a facet or an "other" grouping rather than
    given a generated hue.

    Args:
        cfg: Loaded configuration.
        index: Zero-based series position.

    Returns:
        Hex colour string.

    Raises:
        IndexError: If ``index`` exceeds the palette.
    """
    palette = list(cfg.get("output.figures.palette"))
    if index >= len(palette):
        raise IndexError(
            f"series index {index} exceeds the {len(palette)}-colour categorical palette. "
            "Facet the chart or group the tail rather than generating a new hue."
        )
    return palette[index]


def save_figure(cfg: Config, fig: Figure, name: str, subdir: str | None = None) -> list[Path]:
    """Write a figure in every configured format.

    Args:
        cfg: Loaded configuration (``output.figures.formats``, ``dpi``).
        fig: The figure to write.
        name: File stem, without extension.
        subdir: Optional subdirectory beneath ``results/figures``.

    Returns:
        Paths written.
    """
    figures_dir = cfg.path_for("figures")
    if subdir:
        figures_dir = figures_dir / subdir
    figures_dir.mkdir(parents=True, exist_ok=True)

    dpi = int(cfg.get("output.figures.dpi", 300))
    written: list[Path] = []
    for fmt in cfg.get("output.figures.formats", ["png", "pdf"]):
        path = figures_dir / f"{name}.{fmt}"
        fig.savefig(path, dpi=dpi, format=fmt)
        written.append(path)
    plt.close(fig)
    return written


def label_lines_directly(
    ax: Any, labels: Iterable[str], colours: Iterable[str], x_frac: float = 1.01
) -> None:
    """Place series labels beside the lines instead of relying on a legend alone.

    Identity is then carried by text as well as colour, which is what relieves the
    palette's contrast warning on a light print surface.

    Args:
        ax: Target axes.
        labels: Series labels, in draw order.
        colours: Series colours, in draw order.
        x_frac: Horizontal position in axes fraction for the label column.
    """
    lines = ax.get_lines()
    for line, label, colour in zip(lines, labels, colours, strict=False):
        ydata = line.get_ydata()
        if len(ydata) == 0:
            continue
        ax.annotate(
            label,
            xy=(x_frac, ydata[-1]),
            xycoords=("axes fraction", "data"),
            va="center",
            fontsize=plt.rcParams["legend.fontsize"],
            color=colour,
            annotation_clip=False,
        )
