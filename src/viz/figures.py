r"""Figure styling and output helpers.

Every figure is written at the configured DPI in both PNG and PDF so the same
artefact serves screen review and ``\includegraphics``. Styling is driven by
``output.figures`` in ``config.yaml``.

Figures are sized for a two-column page and carry no titles: the title belongs in
the LaTeX caption, and a title drawn into the artefact is a second, unversioned
copy of it that drifts. Multi-panel figures carry ``(a)``/``(b)`` panel labels so
the caption has something to refer to.

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
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from src.utils import Config

# Column widths in inches for a two-column page. Elsevier is 90 mm single and
# 190 mm double; IEEE is 3.5 in and 7.16 in. The narrower of each pair is used so
# a figure drops into either template at ``width=\linewidth`` with no rescaling.
# Rescaling in LaTeX is what makes legends unreadable: it shrinks the type this
# style set deliberately sized against the printed column.
COL_SINGLE = 3.50
COL_DOUBLE = 7.16

# PM2.5 concentration unit, set in mathtext so it renders identically in the PDF
# and the PNG. The literal "µg/m³" depends on glyphs the serif stack does not
# always carry, and a missing glyph prints as a tofu box rather than failing.
UNIT_PM25 = r"$\mu$g m$^{-3}$"

# The serif stack. STIX Two Text matches the Times-like body type of most
# journal templates; the rest are fallbacks in descending order of similarity.
SERIF_STACK = ["STIX Two Text", "STIXGeneral", "Times New Roman", "DejaVu Serif"]


def setup_style(cfg: Config) -> list[str]:
    """Apply the project-wide matplotlib style.

    Args:
        cfg: Loaded configuration (``output.figures``).

    Returns:
        The categorical palette, in assignment order.
    """
    figures = cfg.get("output.figures")
    palette = list(figures["palette"])
    base = float(figures.get("font_size", 9))

    style = str(figures.get("style", "seaborn-v0_8-whitegrid"))
    if style in plt.style.available:
        plt.style.use(style)

    plt.rcParams.update(
        {
            # Serif body type, matching the journal template rather than the
            # matplotlib default: a sans-serif figure in a serif paper reads as
            # imported from somewhere else.
            "font.family": "serif",
            "font.serif": SERIF_STACK,
            "mathtext.fontset": "stix",
            "font.size": base,
            "axes.titlesize": base,
            "axes.labelsize": base,
            "xtick.labelsize": base - 1.0,
            "ytick.labelsize": base - 1.0,
            "legend.fontsize": base - 1.0,
            "legend.title_fontsize": base - 1.0,
            "figure.dpi": 110,  # on-screen; save dpi set at write time
            "savefig.dpi": int(figures.get("dpi", 300)),
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            # Type 42 embeds the font as outlines the publisher can subset.
            # Type 3 is the matplotlib default and is rejected by several houses.
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.prop_cycle": plt.cycler(color=palette),
            # Recessive chrome: the grid must never compete with the marks.
            "axes.grid": True,
            "grid.color": str(figures.get("grid_colour", "#D9D9D9")),
            "grid.linewidth": 0.5,
            "grid.alpha": 0.9,
            "axes.edgecolor": "#4D4D4D",
            "axes.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "lines.linewidth": 1.3,
            "lines.markersize": 3.6,
            "patch.linewidth": 0.5,
            "legend.frameon": False,
            "legend.handlelength": 1.6,
            "legend.handletextpad": 0.5,
            "legend.columnspacing": 1.1,
            "legend.borderaxespad": 0.4,
            "legend.borderpad": 0.35,
            "legend.labelspacing": 0.35,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            # STIX has no U+2212; drawing the ASCII hyphen avoids a tofu box on
            # every negative tick label.
            "axes.unicode_minus": False,
        }
    )
    return palette


def panel_label(ax: Axes, text: str, *, x: float = 0.5, y: float = 1.015) -> None:
    """Tag a panel with ``(a)``-style identifier the caption can refer to.

    The label sits above the axes rather than inside it, so it never lands on a
    mark, and it is the only text this project draws where a title would go. It
    is centred on the axes for the same reason a title would be: flush left it
    reads as belonging to the y-axis label under it rather than to the panel,
    and in a two-panel figure the pair sits off the figure's own symmetry.

    Args:
        ax: Target axes.
        text: Label text, e.g. ``"(a)"`` or ``"(a) Tier 1 -- baselines"``.
        x: Horizontal position in axes fraction, at the midpoint by default.
        y: Vertical position in axes fraction.
    """
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=plt.rcParams["axes.labelsize"],
        fontweight="bold",
        color="#1A1A1A",
    )


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
