"""How much supervision a gappy record actually yields.

The gap-injection experiment measures that fragmented removal costs far more
usable training rows than contiguous removal at an identical hour count -- 1,870
rows against 14,727 on one Wanliu cell -- but nothing in the pipeline said why.
This module is the why, and it is geometric rather than statistical.

Every row must carry ``pos_in_run >= floor`` hours of unbroken history and a
target ``horizon`` hours further on inside the same run
(``build_features.build_features``). A run of length ``l`` therefore yields
``max(0, l - floor - horizon)`` rows, and a gap does not cost the hours it
removes: it costs those hours **plus** the ``floor + horizon`` hours behind it
that no longer reach back far enough. Call that span the *sterilisation radius*.

Two consequences follow, and both are load-bearing for the study:

* The cost is driven by the **number** of gaps, not their total length. Removing
  the same hours as many short outages sterilises many radii; removing them as a
  few long blocks sterilises a few. That is the arm contrast, in closed form.
* A near-complete record is not therefore a well-supervised one. Beijing's 117
  missing hours (0.33%) destroy 4,006 usable hours -- a 34-fold amplification --
  against Dhaka's 2.3-fold, because Beijing's few missing hours are scattered
  across 27 runs while Dhaka's are clustered.

Two forms are provided, and they answer different questions.

``exact_usable_rows`` needs the run-length distribution and is an identity, not a
model. ``fit_survival_law``/``predict_usable`` need only the observed-hour count
and the gap count -- the two numbers a data-availability report actually carries
-- and are an approximation whose error is measured rather than assumed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def sterilisation_radius(floor_h: int, horizon_h: int) -> int:
    """Hours behind a gap that the gap renders unusable.

    Args:
        floor_h: Hours of unbroken history a row must carry.
        horizon_h: Forecast horizon in hours.

    Returns:
        The radius in hours.
    """
    return int(floor_h) + int(horizon_h)


def run_length_distribution(frame: pd.DataFrame) -> np.ndarray:
    """Lengths of every maximal gap-free run in a built feature frame.

    Args:
        frame: Built feature frame carrying ``run_id`` and ``run_len``.

    Returns:
        One length in hours per run, ascending by run id.
    """
    in_run = frame["run_id"] >= 0
    return frame.loc[in_run].groupby("run_id")["run_len"].first().to_numpy(dtype=np.int64)


def exact_usable_rows(run_lens: np.ndarray, floor_h: int, horizon_h: int) -> int:
    """Rows a record can supervise, from its run-length distribution alone.

    This is an identity for the geometric conditions -- in a run, enough history
    behind, target still inside the run ahead -- and therefore an **upper bound**
    on ``valid_h{h}``, which additionally requires the target to be a genuine
    observation rather than a forward-filled one. On the records in this study
    that last condition removes a further 1.9-2.5%, and that residual is the
    leakage rule working, not an error in this formula.

    Args:
        run_lens: Length in hours of every gap-free run.
        floor_h: Hours of unbroken history a row must carry.
        horizon_h: Forecast horizon in hours.

    Returns:
        The number of geometrically usable rows.
    """
    radius = sterilisation_radius(floor_h, horizon_h)
    return int(np.maximum(0, np.asarray(run_lens, dtype=np.int64) - radius).sum())


@dataclass
class LawFit:
    """A fitted survival approximation to the usable-row count.

    Attributes:
        alpha: Fraction of counted gaps that actually break a run. Below one
            because ``impute.max_ffill_hours`` bridges the short ones before
            runs are assigned, so a counted gap is not necessarily a run break.
        beta0: Log-scale intercept, absorbing the departure of the run-length
            distribution from the exponential the survival form assumes.
        radius_h: Sterilisation radius the fit was made at.
        n_fit: Number of observations fitted.
        source: What was fitted on, for the record.
    """

    alpha: float
    beta0: float
    radius_h: int
    n_fit: int
    source: str = ""

    def to_dict(self) -> dict[str, object]:
        """Serialisable view of the fit."""
        return {
            "alpha": float(self.alpha),
            "beta0": float(self.beta0),
            "radius_h": int(self.radius_h),
            "n_fit": int(self.n_fit),
            "source": self.source,
            "form": "usable = observed * exp(beta0 - alpha * radius_h * n_gaps / observed)",
        }


def fit_survival_law(
    observed: np.ndarray,
    n_gaps: np.ndarray,
    usable: np.ndarray,
    radius_h: int,
    source: str = "",
) -> LawFit:
    """Fit the survival form of the usable-row count.

    If run lengths were exponential with mean ``observed / n_gaps``, the expected
    surviving rows would be ``observed * exp(-radius * n_gaps / observed)`` with
    no free constant at all. Two things spoil that, and both are absorbed here
    rather than hidden: forward-filling bridges short gaps so not every counted
    gap breaks a run (``alpha``), and real run lengths are not exponential
    (``beta0``).

    Args:
        observed: Observed hours per record or cell.
        n_gaps: Distinct gaps per record or cell, counted before imputation.
        usable: Actual usable row count per record or cell.
        radius_h: Sterilisation radius in hours.
        source: Label describing the fitting set.

    Returns:
        The fit.

    Raises:
        ValueError: If the inputs disagree in length, or fewer than three
            usable observations survive.
    """
    obs = np.asarray(observed, dtype=float)
    gaps = np.asarray(n_gaps, dtype=float)
    use = np.asarray(usable, dtype=float)
    if not (obs.shape == gaps.shape == use.shape):
        raise ValueError(f"shape mismatch: {obs.shape}, {gaps.shape}, {use.shape}")

    keep = (obs > 0) & (use > 0)
    if int(keep.sum()) < 3:
        raise ValueError(f"need at least 3 usable observations to fit, have {int(keep.sum())}")

    x = radius_h * gaps[keep] / obs[keep]
    design = np.vstack([np.ones(x.size), -x]).T
    (beta0, alpha), *_ = np.linalg.lstsq(design, np.log(use[keep] / obs[keep]), rcond=None)
    return LawFit(
        alpha=float(alpha),
        beta0=float(beta0),
        radius_h=int(radius_h),
        n_fit=int(keep.sum()),
        source=source,
    )


def predict_usable(fit: LawFit, observed: np.ndarray, n_gaps: np.ndarray) -> np.ndarray:
    """Predicted usable rows under a fitted law.

    Args:
        fit: A fit from :func:`fit_survival_law`.
        observed: Observed hours per record or cell.
        n_gaps: Distinct gaps per record or cell.

    Returns:
        Predicted usable row counts.
    """
    obs = np.asarray(observed, dtype=float)
    gaps = np.asarray(n_gaps, dtype=float)
    safe = np.where(obs > 0, obs, 1.0)
    ratio = np.where(obs > 0, fit.radius_h * gaps / safe, np.inf)
    return obs * np.exp(fit.beta0 - fit.alpha * ratio)


def law_diagnostics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """Fit quality of a predicted usable-row count.

    Median absolute percentage error is reported beside R-squared because the
    cells span two orders of magnitude in usable rows, and R-squared on that
    spread is flattered by the large cells alone.

    Args:
        actual: Observed usable row counts.
        predicted: Predicted usable row counts.

    Returns:
        ``n``, ``r2``, ``median_ape_pct`` and ``mean_ape_pct``.
    """
    a = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)
    keep = np.isfinite(a) & np.isfinite(p) & (a > 0)
    a, p = a[keep], p[keep]
    ape = np.abs(p - a) / a
    ss_res = float(((a - p) ** 2).sum())
    ss_tot = float(((a - a.mean()) ** 2).sum())
    return {
        "n": int(a.size),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "median_ape_pct": float(100.0 * np.median(ape)),
        "mean_ape_pct": float(100.0 * ape.mean()),
    }


def ffill_survival_fraction(gap_lengths_h: np.ndarray, max_ffill_hours: int) -> float:
    """Fraction of gaps that survive forward-filling and therefore break a run.

    This is the mechanistic prediction of ``LawFit.alpha``. Agreement between the
    two is the evidence that the fitted constant is a property of the imputation
    policy rather than a free parameter absorbing whatever it must.

    Args:
        gap_lengths_h: One length in hours per distinct gap.
        max_ffill_hours: The forward-fill limit from ``impute.max_ffill_hours``.

    Returns:
        The surviving fraction, in ``[0, 1]``.
    """
    lengths = np.asarray(gap_lengths_h, dtype=float)
    if lengths.size == 0:
        return 0.0
    return float(np.mean(lengths > max_ffill_hours))
