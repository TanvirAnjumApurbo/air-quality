r"""Phase 6: the study architecture, drawn from the record itself.

A schematic made of prose in boxes tells a reader nothing the methods section
does not, so nothing here is written that can be drawn. The series, its gaps,
the split, the supervision a gap destroys and the two degradation arms are all
rendered from ``features.parquet`` and from the same ``inject_gaps`` the
experiment calls, at their real proportions. The only text is a noun per
element.

The middle stage is the one worth the space. A row is scored only if it carries
the feature set's full backward reach of unbroken history, so a gap costs the
hours it removes *plus* the reach behind it -- the excerpt drawn there is 95%
observed and 50% scoreable. That is the paper's mechanism as a picture rather
than as a claim.

Two spans are excerpts rather than whole records, and both are chosen by rule
rather than by eye, because what they carry lives at an hourly scale a nine-year
ribbon cannot resolve: a one-hour gap is a thousandth of a pixel there.

Run::

    python scripts/24_architecture_figure.py --config config.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from src.features.build_features import derived_history_columns
from src.features.gap_injection import empirical_gap_profile, inject_gaps
from src.utils import load_config, setup_logging
from src.viz.figures import COL_DOUBLE, save_figure, series_colour, setup_style

# Geometry. The drawing plane is 100 units wide plus a margin either side, and
# the artefact is written with a STANDARD bounding box so it is 7.16 in by
# construction. A tight box crops to whatever the widest artist happens to be,
# and \includegraphics[width=\linewidth] then rescales the type this style set
# was sized against.
FIG_W_IN = COL_DOUBLE
FIG_H_IN = 2.28
X_UNITS = 100.0
MARGIN_U = 1.0
Y_UNITS = (X_UNITS + 2 * MARGIN_U) * FIG_H_IN / FIG_W_IN

INK = "#1A1A1A"
RULE = "#5A5A5A"
LOST = "#DCDCDC"  # hours no model can be scored on
MUTED = "#6E6E6E"
SPLIT_SHADES = ("#3F3F3F", "#8E8E8E", "#C6C6C6")

FS_STAGE = 7.2
FS_LABEL = 6.2
FS_TICK = 5.6

# Stage columns.
COL_RECORD = (0.0, 25.0)
COL_SUPERVISION = (28.5, 56.0)
COL_MODELS = (59.5, 81.0)
COL_FORECAST = (84.5, 100.0)
GUTTER_U = 6.4  # width reserved for element names, left of every stack

# Rows, top down. Every stage is drawn inside one band and centred on one axis,
# so each arrow between stages -- and the fan of horizons the last one opens --
# sits at the middle of the stage it leaves and of the stage it enters.
STAGE_Y = 30.9
ROW_A = (24.2, 28.4)
ROW_B = (19.8, 22.6)
ROW_C = (16.3, 18.4)
BAND = (ROW_C[0], ROW_A[1])
MID_Y = 0.5 * (BAND[0] + BAND[1])
FLOW_Y = MID_Y
RIBBON_H = 3.0  # observed and scoreable: equal, because the pair is a comparison
BOX_H = 3.2  # one height for every tier box, set by the two-line one
BAND_LBL = 11.4
BAND_A = (6.2, 8.5)
BAND_B = (2.4, 4.7)

# A 420-hour excerpt holds a gap and the whole sterilised stretch behind it. The
# window is chosen by rule: the first in the training split with exactly one gap
# of 8 to 24 hours, early enough that its shadow fits inside the frame.
EXCERPT_H = 420
GAP_MIN_H, GAP_MAX_H = 8, 24

# The arms are drawn over 24 weeks, on the window matching the arm's own removed
# fraction AND its own gaps-per-hour -- both, because a window can hit 25% removed
# with one block or with twenty, and the arrangement is the manipulated variable.
# Matching them makes the excerpt representative rather than picked. A month is
# too short: the fragmented arm averages 42 h per gap, so a 30-day window holds
# about one, and the two arms then look alike for want of span.
ARM_EXCERPT_H = 4032
ARM_COLUMNS = 1200
INJECT_COVERAGE = 0.75


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="config.yaml")
    p.add_argument(
        "--donor-config",
        default="config_beijing.yaml",
        help="record the arms degrade; the primary record supplies the gap profile",
    )
    return p.parse_args()


def axes_at(fig: Figure, x0: float, x1: float, y0: float, y1: float) -> Axes:
    """Open a bare axes over a rectangle of the drawing plane.

    Ticks, grid and spines are cleared individually rather than with
    ``set_axis_off``, which suppresses the frame at draw time whatever the
    spines are later set to -- so a border asked for afterwards never appears.

    Args:
        fig: Target figure.
        x0: Left edge, in drawing units.
        x1: Right edge, in drawing units.
        y0: Bottom edge, in drawing units.
        y1: Top edge, in drawing units.

    Returns:
        An axes with no visible decoration.
    """
    span = X_UNITS + 2 * MARGIN_U
    ax = fig.add_axes([(x0 + MARGIN_U) / span, y0 / Y_UNITS, (x1 - x0) / span, (y1 - y0) / Y_UNITS])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.margins(0)
    return ax


def outline(ax: Axes) -> None:
    """Draw a hairline frame around a panel."""
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.4)
        spine.set_edgecolor(RULE)


def ribbon(ax: Axes, values: np.ndarray, *, high: str, low: str) -> None:
    """Fill an axes with a one-row image, dark where the flag holds.

    The hairline frame is load-bearing: a ribbon that ends in removed hours ends
    in white, and without a border a reader cannot tell a span that was emptied
    from a span that was never drawn.

    Args:
        ax: Target axes.
        values: Row of values in ``[0, 1]``.
        high: Colour at 1.
        low: Colour at 0.
    """
    cmap = mpl.colors.LinearSegmentedColormap.from_list("ribbon", [low, high])
    ax.imshow(values[np.newaxis, :], aspect="auto", cmap=cmap, vmin=0.0, vmax=1.0)
    outline(ax)


def label(
    canvas: Axes, x: float, y: float, text: str, *, colour: str = MUTED, size: float = FS_LABEL
) -> None:
    """Name one drawn element, centred under it; ``x`` is the element's midpoint."""
    canvas.text(x, y, text, ha="center", va="baseline", fontsize=size, color=colour)


def gutter(canvas: Axes, x: float, row: tuple[float, float], text: str) -> None:
    """Name a stacked element from the gutter to its left.

    Every stack in the figure is named this way rather than from above it. The
    rows are 2 to 4 units tall, which is under the descender depth of a 6 pt
    label, so a name set above a strip lands on the strip over it.

    Args:
        canvas: The drawing plane.
        x: Left edge of the named element.
        row: ``(bottom, top)`` of the named element.
        text: The name.
    """
    canvas.text(
        x - 1.1,
        0.5 * (row[0] + row[1]),
        text,
        ha="right",
        va="center",
        fontsize=FS_LABEL,
        color=MUTED,
    )


def arrow(canvas: Axes, start: tuple[float, float], end: tuple[float, float], colour: str) -> None:
    """Draw one connector."""
    canvas.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=5.5,
            linewidth=0.75,
            color=colour,
            shrinkA=0.0,
            shrinkB=0.0,
            zorder=5,
        )
    )


def load_record(cfg, log) -> dict[str, object]:
    """Read the primary record and pick the excerpt the middle stage draws.

    Args:
        cfg: Loaded primary configuration.
        log: Logger.

    Returns:
        Arrays and counts for the first three stages.

    Raises:
        FileNotFoundError: If the feature matrix has not been built.
        RuntimeError: If no excerpt matches the selection rule.
    """
    path = cfg.path_for("data_processed") / "features.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run scripts/04_build_features.py first")

    horizon = int(cfg.get("task.headline_horizon_h"))
    met = list(cfg.get("features.met_vars"))
    frame = pd.read_parquet(
        path, columns=["pm25", "is_observed", "split", f"valid_h{horizon}", *met]
    )

    observed = frame["is_observed"].to_numpy().astype(bool)
    scoreable = frame[f"valid_h{horizon}"].to_numpy().astype(bool)
    split = frame["split"].to_numpy()
    train = split == "train"

    start = None
    for s in range(0, len(frame) - EXCERPT_H, 24):
        if not train[s : s + EXCERPT_H].all():
            continue
        edges = np.diff(np.concatenate([[0], (~observed[s : s + EXCERPT_H]).astype(int), [0]]))
        opens, closes = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
        if len(opens) != 1 or not GAP_MIN_H <= closes[0] - opens[0] <= GAP_MAX_H:
            continue
        if not 0.10 * EXCERPT_H <= opens[0] <= 0.30 * EXCERPT_H:
            continue
        start = s
        break
    if start is None:
        raise RuntimeError("no excerpt with a single short gap; widen GAP_MIN_H/GAP_MAX_H")

    sl = slice(start, start + EXCERPT_H)
    scaled = frame[met].iloc[sl]
    scaled = (scaled - scaled.mean()) / scaled.std(ddof=0)

    log.info(
        "supervision excerpt %s..%s: %.1f%% observed, %.1f%% scoreable at h=%d",
        frame.index[start].date(),
        frame.index[start + EXCERPT_H - 1].date(),
        100 * observed[sl].mean(),
        100 * scoreable[sl].mean(),
        horizon,
    )
    meta = json.loads(
        (cfg.path_for("data_processed") / "features_meta.json").read_text(encoding="utf-8")
    )
    derived = derived_history_columns(cfg)
    return {
        "pm25": frame["pm25"].iloc[sl].to_numpy(),
        "observed": observed[sl],
        "scoreable": scoreable[sl],
        "met": scaled.to_numpy().T,
        "horizon": horizon,
        "radius": int(meta["max_backward_dependency_h"]) + horizon,
        "n_tabular": int(meta["n_predictors"]),
        "n_channels": len([c for c in meta["predictors"] if c not in derived]),
        "split_sizes": [int((split == s).sum()) for s in ("train", "val", "test")],
        "years": (frame.index[0].year, frame.index[-1].year),
        "profile": empirical_gap_profile(frame["pm25"], source="primary record"),
    }


def load_arms(donor_cfg_path: Path, profile, log) -> dict[str, np.ndarray] | None:
    """Degrade the donor record by both arms, through the experiment's own code.

    Args:
        donor_cfg_path: Config naming the record the experiment degrades.
        profile: Gap-length distribution taken from the primary record.
        log: Logger.

    Returns:
        A representative excerpt of each arm's observed mask, or ``None`` if the
        donor's feature matrix has not been built.
    """
    donor = load_config(donor_cfg_path)
    path = donor.path_for("data_processed") / "features.parquet"
    if not path.exists():
        log.warning("no %s; the arms are omitted rather than mocked", path)
        return None

    frame = pd.read_parquet(path, columns=["pm25", "split"])
    test_at = int(np.flatnonzero(frame["split"].to_numpy() == "test")[0])
    protect = frame.index[test_at]
    wanted = 1.0 - INJECT_COVERAGE

    excerpts: dict[str, np.ndarray] = {}
    for arm in ("fragmented", "contiguous"):
        degraded, report = inject_gaps(
            frame[["pm25"]],
            "pm25",
            arm=arm,
            target_coverage=INJECT_COVERAGE,
            profile=profile,
            protect_from=protect,
            n_blocks=int(donor.get("ablation.gap_injection.contiguous_blocks")),
        )
        mask = degraded["pm25"].notna().to_numpy()
        pick = _representative_window(mask, test_at, wanted, report.n_gaps_after)
        excerpts[arm] = pool(mask[pick : pick + ARM_EXCERPT_H], ARM_COLUMNS)
        log.info(
            "%s arm: %d hours removed in %d gaps; excerpt at %s is %.1f%% removed in %d gaps",
            arm,
            report.hours_removed,
            report.n_gaps_after,
            frame.index[pick].date(),
            100 * (1 - mask[pick : pick + ARM_EXCERPT_H].mean()),
            _count_gaps(mask[pick : pick + ARM_EXCERPT_H]),
        )
    return excerpts


def _count_gaps(mask: np.ndarray) -> int:
    """Number of distinct missing runs in a boolean observed mask."""
    return int(np.count_nonzero(np.diff(np.concatenate([[0], (~mask).astype(int), [0]])) == 1))


def _representative_window(mask: np.ndarray, region_h: int, wanted: float, total_gaps: int) -> int:
    """Pick the excerpt that matches the arm on both amount removed and gap count.

    Args:
        mask: Observed flag per hour, after degradation.
        region_h: Length of the injectable region; nothing past it is drawn.
        wanted: Removed fraction the arm was asked for.
        total_gaps: Missing runs the arm left across the whole region.

    Returns:
        Start index of the chosen window.
    """
    per_window = total_gaps * ARM_EXCERPT_H / region_h
    best, best_cost = 0, np.inf
    for start in range(0, region_h - ARM_EXCERPT_H, 24):
        window = mask[start : start + ARM_EXCERPT_H]
        cost = abs((1.0 - window.mean()) - wanted) / wanted + abs(
            _count_gaps(window) - per_window
        ) / max(per_window, 1.0)
        if cost < best_cost:
            best, best_cost = start, cost
    return best


def pool(mask: np.ndarray, columns: int) -> np.ndarray:
    """Average a boolean mask down to a drawable number of columns."""
    trimmed = mask[: len(mask) // columns * columns].astype(float)
    return trimmed.reshape(columns, -1).mean(axis=1)


def draw_record(fig: Figure, canvas: Axes, data: dict[str, object]) -> None:
    """Stage one: what the record is."""
    col0, x1 = COL_RECORD
    x0 = col0 + GUTTER_U  # named in a gutter, as every other stack in the figure is

    gutter(canvas, x0, ROW_A, "PM$_{2.5}$")
    ax = axes_at(fig, x0, x1, *ROW_A)
    series = np.where(data["observed"], data["pm25"], np.nan)
    ax.plot(series, color=INK, linewidth=0.5, solid_capstyle="butt")
    ax.set_xlim(0, len(series))
    ax.set_ylim(0, float(np.nanmax(series)) * 1.10)

    gutter(canvas, x0, ROW_B, "weather")
    ax = axes_at(fig, x0, x1, *ROW_B)
    ax.imshow(data["met"], aspect="auto", cmap="Greys", vmin=-3.2, vmax=3.2)
    outline(ax)

    gutter(canvas, x0, ROW_C, "split")
    ax = axes_at(fig, x0, x1, *ROW_C)
    sizes = np.asarray(data["split_sizes"], dtype=float)
    left = 0.0
    for size, shade, name in zip(sizes, SPLIT_SHADES, ("train", "val", "test"), strict=True):
        ax.barh(0, size, left=left, height=1.0, color=shade, edgecolor="white", linewidth=0.6)
        ax.text(
            left + size / 2,
            0,
            name,
            ha="center",
            va="center",
            fontsize=FS_TICK,
            color="white" if shade == SPLIT_SHADES[0] else INK,
        )
        left += size
    ax.set_xlim(0, sizes.sum())
    ax.set_ylim(-0.5, 0.5)

    first, last = data["years"]
    tick_y = ROW_C[0] - 1.7
    canvas.text(x0, tick_y, str(first), ha="left", va="baseline", fontsize=FS_TICK, color=MUTED)
    canvas.text(x1, tick_y, str(last), ha="right", va="baseline", fontsize=FS_TICK, color=MUTED)


def draw_supervision(fig: Figure, canvas: Axes, data: dict[str, object], accent: str) -> None:
    """Stage two: what a gap costs, at scale."""
    col0, x1 = COL_SUPERVISION
    x0 = col0 + GUTTER_U
    observed = np.asarray(data["observed"])
    scoreable = np.asarray(data["scoreable"])

    # The pair fills the band and is symmetric about the axis, so the stage is
    # as deep as the record beside it and the flow arrows arrive between the two
    # ribbons rather than on the edge of the lower one.
    top = (BAND[1] - RIBBON_H, BAND[1])
    bottom = (BAND[0], BAND[0] + RIBBON_H)
    for name, row, values, low in (
        ("observed", top, observed, "white"),
        ("scoreable", bottom, scoreable, LOST),
    ):
        gutter(canvas, x0, row, name)
        ribbon(axes_at(fig, x0, x1, *row), values.astype(float), high=INK, low=low)

    # The bracket runs from the gap to the far edge of its shadow, which is the
    # quantity the whole mechanism turns on.
    gap = int(np.flatnonzero(~observed)[0])
    span = int(data["radius"]) + int((~observed).sum())
    unit = (x1 - x0) / len(observed)
    left = x0 + gap * unit
    right = x0 + min(len(observed), gap + span) * unit
    stem = bottom[0] - 0.9
    canvas.plot(
        [left, left, right, right],
        [stem + 0.7, stem, stem, stem + 0.7],
        color=accent,
        linewidth=0.8,
    )
    canvas.text(
        0.5 * (left + right),
        stem - 1.8,
        f"$R$ = {data['radius']} h",
        ha="center",
        va="baseline",
        fontsize=FS_LABEL,
        color=accent,
    )
    canvas.plot([left, left], [bottom[0], top[1]], color=accent, linewidth=0.5, zorder=6)


def draw_models(
    canvas: Axes, data: dict[str, object], window: int, budget: int, accent: str
) -> None:
    """Stage three: what reads the record, and in which shape."""
    x0, x1 = COL_MODELS
    grid_x1 = x0 + 6.6
    box_x0, box_x1 = x0 + 10.2, x1 - 1.6

    # The three tiers fill the band at one pitch and straddle the axis, so this
    # stage stands as deep as the two before it and the flow arrow arrives at
    # its middle. Every connector into a box is level, and that is what fixes
    # the tabular row's height: the row is placed from the tiers it feeds rather
    # than the other way round, its own height carrying nothing.
    pitch = 0.5 * (BAND[1] - BAND[0] - BOX_H)
    seq_mid, ml_mid, stat_mid = MID_Y + pitch, MID_Y, MID_Y - pitch
    row_mid = 0.5 * (ml_mid + stat_mid)
    row_y0, row_y1 = row_mid - 0.6, row_mid + 0.6

    # The window a recurrent model consumes, and the row a tabular model sees.
    for i in range(7):
        xi = x0 + i * (grid_x1 - x0) / 6
        canvas.plot([xi, xi], [ROW_A[0], ROW_A[1]], color=RULE, linewidth=0.4)
    for j in range(5):
        yj = ROW_A[0] + j * (ROW_A[1] - ROW_A[0]) / 4
        canvas.plot([x0, grid_x1], [yj, yj], color=RULE, linewidth=0.4)
    grid_mid_x = 0.5 * (x0 + grid_x1)
    label(canvas, grid_mid_x, ROW_A[0] - 1.7, f"{window} $\\times$ {data['n_channels']}")

    canvas.add_patch(
        Rectangle(
            (x0, row_y0),
            grid_x1 - x0,
            row_y1 - row_y0,
            facecolor="white",
            edgecolor=RULE,
            linewidth=0.4,
        )
    )
    for i in range(1, 8):
        xi = x0 + i * (grid_x1 - x0) / 8
        canvas.plot([xi, xi], [row_y0, row_y1], color=RULE, linewidth=0.4)
    label(canvas, grid_mid_x, row_y0 - 1.7, f"1 $\\times$ {data['n_tabular']}")

    tiers = (
        ("Sequence", seq_mid, f"$\\leq${budget}k params"),
        ("Classical ML", ml_mid, None),
        ("Statistical", stat_mid, None),
    )
    text_x = 0.5 * (box_x0 + box_x1)
    for name, mid, note in tiers:
        canvas.add_patch(
            FancyBboxPatch(
                (box_x0, mid - 0.5 * BOX_H),
                box_x1 - box_x0,
                BOX_H,
                boxstyle="round,pad=0,rounding_size=0.55",
                facecolor="white",
                edgecolor=RULE,
                linewidth=0.7,
            )
        )
        # A full line of leading between the two: at 6.2 pt over 5.6 pt a closer
        # pair puts the descenders of the name into the note under it.
        canvas.text(
            text_x,
            mid + (0.72 if note else 0.0),
            name,
            ha="center",
            va="center",
            fontsize=FS_LABEL,
            color=INK,
        )
        if note:
            canvas.text(
                text_x, mid - 0.72, note, ha="center", va="center", fontsize=FS_TICK, color=MUTED
            )
        # Every tier joins one bus, so the forecast reads as the tiers' output
        # and not as the output of whichever box the flow arrow lines up with.
        canvas.plot([box_x1, x1], [mid, mid], color=RULE, linewidth=0.6)
    canvas.plot([x1, x1], [stat_mid, seq_mid], color=RULE, linewidth=0.6)

    # The inputs are drawn as the outputs are. The window enters the sequence
    # tier level, and the tabular row leaves on a stub that meets a riser
    # centred on itself, so the two arrows off it are mirror images. A fan of
    # diagonals could not be symmetric here -- one row cannot be both the
    # midpoint of two boxes and the source of a level arrow to a third.
    split_x = grid_x1 + 1.5
    arrow(canvas, (grid_x1 + 0.4, seq_mid), (box_x0 - 0.3, seq_mid), accent)
    canvas.plot([grid_x1 + 0.4, split_x], [row_mid, row_mid], color=RULE, linewidth=0.75)
    canvas.plot([split_x, split_x], [stat_mid, ml_mid], color=RULE, linewidth=0.75)
    for mid in (ml_mid, stat_mid):
        arrow(canvas, (split_x, mid), (box_x0 - 0.3, mid), RULE)


def draw_forecast(canvas: Axes, horizons: list[int]) -> None:
    """Stage four: one model per horizon, no rollout."""
    x0, x1 = COL_FORECAST
    origin = (x0, FLOW_Y)
    canvas.plot(*origin, marker="o", markersize=1.8, color=INK, zorder=6)
    canvas.text(x0, FLOW_Y - 2.2, "$t$", ha="center", va="baseline", fontsize=FS_LABEL, color=MUTED)

    reach = 0.5 * (BAND[1] - BAND[0]) - 0.4
    tops = np.linspace(MID_Y + reach, MID_Y - reach, len(horizons))
    for h, y in zip(horizons, tops, strict=True):
        arrow(canvas, origin, (x1 - 4.8, y), RULE)
        canvas.plot(x1 - 4.6, y, marker="s", markersize=2.0, color=INK)
        canvas.text(x1 - 3.9, y, f"+{h} h", ha="left", va="center", fontsize=FS_LABEL, color=INK)


def draw_injection(
    fig: Figure, canvas: Axes, arms: dict[str, np.ndarray] | None, accent: str
) -> None:
    """The controlled experiment: the same hours gone, arranged two ways."""
    if arms is None:
        return  # name nothing that is not drawn: no donor matrix, no band at all
    canvas.text(
        0.0,
        BAND_LBL,
        "Gap injection",
        ha="left",
        va="baseline",
        fontsize=FS_STAGE,
        fontweight="bold",
        color=INK,
    )

    x0, x1 = GUTTER_U, 83.0  # aligned with the record strips above
    for name, row in (("fragmented", BAND_A), ("contiguous", BAND_B)):
        gutter(canvas, x0, row, name)
        ribbon(axes_at(fig, x0, x1, *row), arms[name].astype(float), high=INK, low="white")

    canvas.plot(
        [x1 + 0.9, x1 + 1.7, x1 + 1.7, x1 + 0.9],
        [BAND_B[0], BAND_B[0], BAND_A[1], BAND_A[1]],
        color=RULE,
        linewidth=0.7,
    )
    canvas.text(
        x1 + 2.5,
        0.5 * (BAND_B[0] + BAND_A[1]),
        f"{round(100 * (1 - INJECT_COVERAGE))}% of hours\nremoved in both",
        ha="left",
        va="center",
        fontsize=FS_LABEL,
        color=INK,
        linespacing=1.3,
    )
    # The degraded record goes back in at the top of the pipeline.
    arrow(canvas, (11.5, BAND_A[1] + 1.6), (11.5, ROW_C[0] - 0.5), accent)


def check_glyphs(fig: Figure) -> None:
    """Fail rather than print a tofu box for a character the serif stack lacks.

    Args:
        fig: The figure, fully built.

    Raises:
        RuntimeError: If any glyph is missing from the resolved font.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fig.canvas.draw()
    missing = sorted({str(w.message) for w in caught if "missing from font" in str(w.message)})
    if missing:
        raise RuntimeError("architecture figure has unprintable glyphs:\n  " + "\n  ".join(missing))


def main() -> int:
    """Draw the architecture figure."""
    args = parse_args()
    cfg = load_config(args.config)
    log = setup_logging(cfg, "architecture_figure")
    setup_style(cfg)
    accent = series_colour(cfg, 0)

    data = load_record(cfg, log)
    arms = load_arms(Path(args.donor_config), data["profile"], log)

    fig = plt.figure(figsize=(FIG_W_IN, FIG_H_IN))
    canvas = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    canvas.set_xlim(-MARGIN_U, X_UNITS + MARGIN_U)
    canvas.set_ylim(0.0, Y_UNITS)
    canvas.set_axis_off()

    for column, name in (
        (COL_RECORD, "Record"),
        (COL_SUPERVISION, "Gap-aware supervision"),
        (COL_MODELS, "Models"),
        (COL_FORECAST, "Forecast"),
    ):
        canvas.text(
            0.5 * (column[0] + column[1]),
            STAGE_Y,
            name,
            ha="center",
            va="baseline",
            fontsize=FS_STAGE,
            fontweight="bold",
            color=INK,
        )

    draw_record(fig, canvas, data)
    draw_supervision(fig, canvas, data, accent)
    draw_models(
        canvas,
        data,
        int(cfg.get("ablation.gap_injection.sequence_models")[0]["window_h"]),
        int(cfg.get("models.sequence.max_params")) // 1000,
        accent,
    )
    draw_forecast(canvas, [int(h) for h in cfg.get("task.horizons_h")])
    draw_injection(fig, canvas, arms, accent)

    for left, right in (
        (COL_RECORD, COL_SUPERVISION),
        (COL_SUPERVISION, COL_MODELS),
        (COL_MODELS, COL_FORECAST),
    ):
        arrow(canvas, (left[1] + 0.7, FLOW_Y), (right[0] - 0.7, FLOW_Y), RULE)

    check_glyphs(fig)
    with mpl.rc_context({"savefig.bbox": "standard", "savefig.pad_inches": 0.0}):
        written = save_figure(cfg, fig, "fig00_architecture")
    for path in written:
        log.info("wrote %s", path)
    print("\nwrote:")
    for path in written:
        print(f"  {path}")
    print(f"\n{FIG_W_IN:.2f} x {FIG_H_IN:.2f} in, double column, no rescale in LaTeX")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
