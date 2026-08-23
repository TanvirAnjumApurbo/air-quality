"""Forecast availability: how often a model can answer at all.

Every accuracy number in this study is conditional on the model being able to
produce a forecast, and nothing reports how often that is. It is not a detail.
At the configured 168-hour lookback, 23.6% of Dhaka's test hours and 27.5% of
Wanliu's cannot be forecast by any model in the study -- the row does not carry
enough unbroken history to exist -- and those hours are silently absent from
every RMSE in the report rather than counted as failures.

Two things follow.

The first is a correction. Coverage and availability are different quantities and
they do not rank the same way. Dhaka observes 82.3% of hours against Wanliu's
98.9%, a 16.6-point gap in Wanliu's favour; their availabilities are 76.4% and
72.5%, a 3.9-point gap running the *other* way. Dingling and Dongsi, also near
99% covered, fall to 64.4% and 64.3%. Coverage is what a data custodian reports;
availability is what a forecaster gets, and the two can rank a set of records in
opposite orders.

The second is a metric. A model serving 76.4% of hours at RMSE 57 cannot be
ranked against one serving 89.4% at RMSE 60 without saying what happens on the
hours the first refuses. So everything here is scored over a **fixed universe**
that does not move with the treatment, and the unserved hours are answered by a
declared fallback rather than dropped.

The universe is floored at :data:`UNIVERSE_FLOOR_H`, the deepest lookback in the
sweep, so every arm's served set is a subset of it by construction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.eval.split import purge_boundary_rows
from src.features.build_features import scoreable_mask

#: History floor defining the fixed evaluation universe, in hours.
#:
#: The deepest lookback any arm of the frontier uses. Flooring the universe here
#: rather than at each arm's own floor is what makes every arm's served set a
#: subset of one common set, so RMSE over the universe is comparable across arms.
#: It is a module constant rather than a config key because a universe that moved
#: with configuration would not be a universe.
UNIVERSE_FLOOR_H = 168


@dataclass
class AvailabilityRecord:
    """How many hours a configuration can forecast, and out of what.

    Attributes:
        split: Split the record describes.
        horizon_h: Forecast horizon in hours.
        floor_h: History floor this configuration requires.
        window_h: Sequence window in hours, or None for a tabular model.
        radius_h: Sterilisation radius, ``max(floor, window - 1) + horizon``.
        n_grid: Hours of the split on the raw hourly grid.
        n_universe: Hours in the fixed evaluation universe.
        n_served: Hours this configuration can forecast.
        availability: ``n_served / n_universe``.
        availability_grid: ``n_served / n_grid``.
    """

    split: str
    horizon_h: int
    floor_h: int
    window_h: int | None
    radius_h: int
    n_grid: int
    n_universe: int
    n_served: int
    availability: float
    availability_grid: float

    def to_dict(self) -> dict[str, object]:
        """Serialisable view of the record."""
        return {
            "split": self.split,
            "horizon_h": int(self.horizon_h),
            "floor_h": int(self.floor_h),
            "window_h": None if self.window_h is None else int(self.window_h),
            "radius_h": int(self.radius_h),
            "n_grid": int(self.n_grid),
            "n_universe": int(self.n_universe),
            "n_served": int(self.n_served),
            "availability": float(self.availability),
            "availability_grid": float(self.availability_grid),
        }


def effective_floor(floor_h: int, window_h: int | None) -> int:
    """History a configuration actually needs, in hours.

    A sequence model needs its whole window inside one run
    (``models.data.build_sequence_index``), so its binding requirement is the
    larger of the feature floor and ``window - 1``. This is why shortening the
    lookback below the window buys a sequence model nothing: at a 48-hour window,
    floors of 48, 24 and 12 are the same experiment.

    Args:
        floor_h: History floor the feature set requires.
        window_h: Sequence window in hours, or None for a tabular model.

    Returns:
        The binding floor in hours.
    """
    if window_h is None:
        return int(floor_h)
    return max(int(floor_h), int(window_h) - 1)


def sterilisation_radius(floor_h: int, window_h: int | None, horizon_h: int) -> int:
    """Hours behind a gap that a configuration cannot forecast.

    Args:
        floor_h: History floor the feature set requires.
        window_h: Sequence window in hours, or None for a tabular model.
        horizon_h: Forecast horizon in hours.

    Returns:
        The radius in hours.
    """
    return effective_floor(floor_h, window_h) + int(horizon_h)


def evaluation_universe(frame: pd.DataFrame, split: str, horizon_h: int) -> pd.Series:
    """The fixed set of hours every arm is scored over.

    Scoreable at all, in the named split, and clear of the split boundary at the
    deepest floor in the sweep. The boundary purge uses
    :data:`UNIVERSE_FLOOR_H` rather than the arm's own floor precisely so the
    universe does not move between arms.

    Args:
        frame: Built feature frame.
        split: Split name.
        horizon_h: Forecast horizon in hours.

    Returns:
        Boolean mask over the frame's index.
    """
    in_split = frame["split"] == split
    purged = purge_boundary_rows(frame, frame["split"], UNIVERSE_FLOOR_H, horizon_h)
    return scoreable_mask(frame, horizon_h) & in_split & purged


def served_mask(
    frame: pd.DataFrame,
    split: str,
    horizon_h: int,
    floor_h: int,
    window_h: int | None = None,
) -> pd.Series:
    """Hours of the universe a configuration can actually forecast.

    Args:
        frame: Built feature frame.
        split: Split name.
        horizon_h: Forecast horizon in hours.
        floor_h: History floor the feature set requires.
        window_h: Sequence window in hours, or None for a tabular model.

    Returns:
        Boolean mask over the frame's index, a subset of
        :func:`evaluation_universe`.
    """
    deep_enough = frame["pos_in_run"] >= effective_floor(floor_h, window_h)
    return evaluation_universe(frame, split, horizon_h) & deep_enough


def availability_record(
    frame: pd.DataFrame,
    split: str,
    horizon_h: int,
    floor_h: int,
    window_h: int | None = None,
) -> AvailabilityRecord:
    """Count what a configuration can forecast, and out of what.

    Args:
        frame: Built feature frame.
        split: Split name.
        horizon_h: Forecast horizon in hours.
        floor_h: History floor the feature set requires.
        window_h: Sequence window in hours, or None for a tabular model.

    Returns:
        The record.
    """
    universe = evaluation_universe(frame, split, horizon_h)
    served = served_mask(frame, split, horizon_h, floor_h, window_h)
    n_grid = int((frame["split"] == split).sum())
    n_universe = int(universe.sum())
    n_served = int(served.sum())
    return AvailabilityRecord(
        split=split,
        horizon_h=int(horizon_h),
        floor_h=int(floor_h),
        window_h=None if window_h is None else int(window_h),
        radius_h=sterilisation_radius(floor_h, window_h, horizon_h),
        n_grid=n_grid,
        n_universe=n_universe,
        n_served=n_served,
        availability=float(n_served / n_universe) if n_universe else float("nan"),
        availability_grid=float(n_served / n_grid) if n_grid else float("nan"),
    )


def amplification(frame: pd.DataFrame, floor_h: int, horizon_h: int) -> dict[str, float]:
    """How many usable hours each missing hour destroys.

    The headline number of this contribution. A gap costs the hours it removes
    **plus** the sterilisation radius behind it, so a record with few missing
    hours scattered across many runs is punished far harder per missing hour than
    one whose absences are clustered. Beijing loses 34 usable hours per hour
    missing; Dhaka, whose gaps are longer and fewer, loses 2.3.

    Args:
        frame: Built feature frame carrying ``run_id`` and ``pos_in_run``.
        floor_h: History floor in hours.
        horizon_h: Forecast horizon in hours.

    Returns:
        Hour counts and the amplification factor.
    """
    in_run = frame["run_id"] >= 0
    n_missing = int((~in_run).sum())
    n_short_history = int((in_run & (frame["pos_in_run"] < floor_h)).sum())
    n_runs = int(frame.loc[in_run, "run_id"].nunique())
    return {
        "n_grid": len(frame),
        "n_missing": n_missing,
        "n_runs": n_runs,
        "n_lost_to_short_history": n_short_history,
        "closed_form_n_runs_times_radius": int(n_runs * (int(floor_h) + int(horizon_h))),
        "amplification": float(n_short_history / n_missing) if n_missing else float("nan"),
    }


def cascade_predictions(
    served: dict[str, np.ndarray],
    order: list[str],
) -> tuple[np.ndarray, dict[str, float]]:
    """Take, at each row, the first chain member that can answer.

    Args:
        served: Chain member to a prediction vector, NaN where it cannot answer.
        order: Chain members in the order they should be consulted.

    Returns:
        ``(predictions, share)`` where ``share`` is the fraction of rows each
        member answered.

    Raises:
        ValueError: If a member of ``order`` is absent from ``served``, if the
            vectors disagree in length, or if the chain leaves any row
            unanswered.
    """
    missing = [name for name in order if name not in served]
    if missing:
        raise ValueError(f"chain members absent from served: {missing}")
    lengths = {name: len(served[name]) for name in order}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"prediction vectors disagree in length: {lengths}")

    n = next(iter(lengths.values()))
    out = np.full(n, np.nan, dtype=float)
    share: dict[str, float] = {}
    for name in order:
        candidate = np.asarray(served[name], dtype=float)
        take = np.isnan(out) & ~np.isnan(candidate)
        out[take] = candidate[take]
        share[name] = float(take.sum() / n) if n else 0.0

    if np.isnan(out).any():
        raise ValueError(
            f"chain left {int(np.isnan(out).sum())} of {n} rows unanswered; "
            f"the final member must cover the whole universe"
        )
    return out, share
