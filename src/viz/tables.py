r"""Table output: matched CSV and booktabs LaTeX.

Every table is written twice from the same DataFrame -- ``.csv`` for inspection
and ``.tex`` ready to ``\input{}`` -- so a number in the paper and a number in
the repository cannot drift apart.

Captions always carry the split boundary dates, because a forecasting result is
uninterpretable without knowing which period it was evaluated on.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils import Config

LATEX_ESCAPES = {
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def escape_latex(text: str) -> str:
    """Escape LaTeX special characters in a cell or header.

    Args:
        text: Raw text.

    Returns:
        Text safe to place in a LaTeX table body.
    """
    out = str(text)
    # Backslash first, otherwise it would escape the escapes.
    out = out.replace("\\", r"\textbackslash{}")
    for char, replacement in LATEX_ESCAPES.items():
        out = out.replace(char, replacement)
    return out


def split_caption(cfg: Config) -> str:
    """Human-readable description of the evaluation period.

    Args:
        cfg: Loaded configuration.

    Returns:
        A caption fragment naming the split boundary dates.
    """
    bounds = cfg.get("split.explicit_boundaries", {}) or {}
    train_end = str(bounds.get("train_end") or "?")[:10]
    val_end = str(bounds.get("val_end") or "?")[:10]
    site = cfg.get("data.openaq.site_label", "site")
    return (
        f"{site}. Chronological split, no shuffling: "
        f"train to {train_end}, validation to {val_end}, "
        f"test thereafter. Metrics on the held-out test period only."
    )


def write_table(
    cfg: Config,
    df: pd.DataFrame,
    name: str,
    *,
    caption: str,
    label: str | None = None,
    float_format: str | None = None,
    index: bool = False,
) -> list[Path]:
    """Write a DataFrame as matched CSV and booktabs LaTeX files.

    Args:
        cfg: Loaded configuration.
        df: Table contents.
        name: File stem.
        caption: Table caption; the split description is appended automatically.
        label: LaTeX label; defaults to ``tab:<name>``.
        float_format: Printf-style float format; defaults to config.
        index: Whether to write the index as a column.

    Returns:
        Paths written.
    """
    tables_dir = cfg.path_for("tables")
    tables_dir.mkdir(parents=True, exist_ok=True)
    fmt = float_format or str(cfg.get("output.tables.float_fmt", "%.3f"))

    written: list[Path] = []
    csv_path = tables_dir / f"{name}.csv"
    df.to_csv(csv_path, index=index, float_format=fmt)
    written.append(csv_path)

    full_caption = f"{caption} {split_caption(cfg)}"
    body = df.copy()
    if index:
        body = body.reset_index()
    body.columns = [escape_latex(c) for c in body.columns]
    for col in body.columns:
        if body[col].dtype == object or str(body[col].dtype).startswith("str"):
            body[col] = body[col].map(escape_latex)

    latex = body.to_latex(
        index=False,
        escape=False,
        float_format=lambda v: fmt % v,
        column_format="l" + "r" * (body.shape[1] - 1),
        caption=full_caption,
        label=label or f"tab:{name}",
        position="htbp",
    )
    # to_latex already emits booktabs rules when the option is on; ensure it.
    if r"\toprule" not in latex:
        latex = (
            latex.replace(r"\hline\hline", r"\midrule")
            .replace(r"\hline", r"\toprule", 1)
            .replace(r"\hline", r"\bottomrule")
        )
    tex_path = tables_dir / f"{name}.tex"
    tex_path.write_text(latex, encoding="utf-8")
    written.append(tex_path)
    return written


def format_mean_std(mean: float, std: float, decimals: int = 2) -> str:
    """Render a mean and standard deviation as a single cell.

    Args:
        mean: Mean value.
        std: Standard deviation across seeds.
        decimals: Decimal places.

    Returns:
        ``"mean ± std"``, or just the mean when std is zero or undefined.
    """
    if std is None or not pd.notna(std) or std == 0:
        return f"{mean:.{decimals}f}"
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"
