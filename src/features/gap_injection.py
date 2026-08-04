"""Controlled degradation of a monitoring record's contiguity.

The cross-city result -- trees win on Dhaka's broken record, the sequence model
wins on Beijing's near-complete one -- is an observation over two cities that
differ in coverage, length, climate and instrument at once. It supports a
hypothesis about record fragmentation; it does not test one.

This module makes fragmentation a manipulable variable. A near-complete record is
degraded to a chosen coverage level by removing observed hours, in one of two
ways that remove **exactly the same number of hours**:

``fragmented``
    Removed as many short outages, with lengths drawn from a real record's
    empirical gap-length distribution. Contiguity collapses; window yield falls
    far faster than coverage does.

``contiguous``
    Removed as a few long blocks spread evenly across the span. Coverage falls
    identically, but what remains is still long unbroken runs.

The difference between the arms is the effect of fragmentation with data volume
held fixed. Without the second arm, any degradation in the sequence models could
just as well be explained by having less data to learn from, and the causal claim
would not stand up.

Two further controls matter and are enforced by the caller:

* only the training and validation periods are degraded -- the test period is
  left intact, so every cell is scored on identical rows and RMSE stays
  comparable across levels;
* the contiguous arm uses several blocks rather than one, because a single block
  large enough to drop 25% of a multi-year record would remove entire seasons and
  confound fragmentation with seasonal composition.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class InjectionReport:
    """What a degradation actually did to the record.

    Attributes:
        arm: ``"fragmented"``, ``"contiguous"`` or ``"none"``.
        target_coverage: Requested coverage within the injectable region.
        achieved_coverage: Coverage after removal, within that region.
        coverage_before: Coverage before removal, within that region.
        hours_removed: Observed hours set to missing.
        hours_available: Observed hours in the region before removal.
        slots_in_region: Total hourly slots in the region.
        n_gaps_before: Distinct missing runs before removal.
        n_gaps_after: Distinct missing runs after removal.
        longest_gap_after: Longest missing run after removal, hours.
        seed: RNG seed used.
        region: Human-readable description of the injectable region.
    """

    arm: str
    target_coverage: float
    achieved_coverage: float
    coverage_before: float
    hours_removed: int
    hours_available: int
    slots_in_region: int
    n_gaps_before: int
    n_gaps_after: int
    longest_gap_after: int
    seed: int
    region: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return the report as a plain mapping for results.json."""
        return {
            "arm": self.arm,
            "target_coverage": self.target_coverage,
            "achieved_coverage": self.achieved_coverage,
            "coverage_before": self.coverage_before,
            "hours_removed": self.hours_removed,
            "hours_available": self.hours_available,
            "slots_in_region": self.slots_in_region,
            "n_gaps_before": self.n_gaps_before,
            "n_gaps_after": self.n_gaps_after,
            "longest_gap_after": self.longest_gap_after,
            "seed": self.seed,
            "region": self.region,
        }


@dataclass
class GapProfile:
    """An empirical distribution of outage lengths taken from a real record.

    Attributes:
        lengths: Observed gap lengths in hours, one entry per distinct gap.
        source: Where the profile came from, for the record.
    """

    lengths: np.ndarray
    source: str = ""
    _summary: dict[str, float] = field(default_factory=dict)

    def summary(self) -> dict[str, float]:
        """Descriptive statistics of the profile."""
        if not self._summary and self.lengths.size:
            self._summary = {
                "n_gaps": int(self.lengths.size),
                "total_hours": int(self.lengths.sum()),
                "median_h": float(np.median(self.lengths)),
                "mean_h": float(self.lengths.mean()),
                "max_h": int(self.lengths.max()),
                "pct_single_hour": float(100.0 * np.mean(self.lengths == 1)),
            }
        return self._summary


def gap_lengths(series: pd.Series) -> np.ndarray:
    """Lengths of every maximal run of missing values.

    Args:
        series: A value series on a regular hourly index; NaN marks missing.

    Returns:
        One length in hours per distinct gap, in chronological order.
    """
    missing = series.isna().to_numpy()
    if not missing.any():
        return np.empty(0, dtype=int)
    # Run boundaries are where the missing flag changes.
    change = np.flatnonzero(np.diff(missing.astype(np.int8)) != 0) + 1
    starts = np.concatenate([[0], change])
    ends = np.concatenate([change, [missing.size]])
    return np.array(
        [end - start for start, end in zip(starts, ends, strict=True) if missing[start]], dtype=int
    )


def empirical_gap_profile(series: pd.Series, source: str = "") -> GapProfile:
    """Extract a record's gap-length distribution for reuse as an injection pattern.

    Sampling from a real record's outages rather than a parametric distribution
    matters: monitoring failures are strongly bimodal -- a large majority are
    single missed hours, but a handful run for weeks -- and that shape is what
    determines how badly windowing suffers.

    Args:
        series: The donor record's value series, NaN where missing.
        source: Label describing where the profile came from.

    Returns:
        The empirical profile.
    """
    return GapProfile(lengths=gap_lengths(series), source=source)


def _place_fragmented(
    eligible: np.ndarray, n_remove: int, profile: GapProfile, rng: np.random.Generator
) -> np.ndarray:
    """Choose positions to remove as many short outages.

    Gap lengths are drawn with replacement from the empirical profile and placed
    at random eligible starts. Overlap with an already-removed stretch is allowed
    and simply merges the two, which is what happens in a real record; the loop
    tracks the count actually removed rather than the count attempted, so the
    total lands exactly on ``n_remove``.

    Args:
        eligible: Positions that may be removed (currently observed).
        n_remove: Exact number of positions to remove.
        profile: Empirical gap-length distribution to draw from.
        rng: Seeded generator.

    Returns:
        Positions to remove.
    """
    eligible_set = set(eligible.tolist())
    chosen: set[int] = set()
    lengths = profile.lengths if profile.lengths.size else np.array([1])

    # Bounded rather than while-True: a pathological profile (every draw landing
    # on already-removed hours) would otherwise spin forever.
    max_attempts = max(1000, 20 * n_remove)
    attempts = 0
    while len(chosen) < n_remove and attempts < max_attempts:
        attempts += 1
        length = int(rng.choice(lengths))
        start = int(rng.choice(eligible))
        for pos in range(start, start + length):
            if len(chosen) >= n_remove:
                break
            if pos in eligible_set:
                chosen.add(pos)

    if len(chosen) < n_remove:
        # Top up deterministically from whatever remains, so the two arms always
        # remove identical counts even on an awkward profile.
        remaining = np.array(sorted(eligible_set - chosen))
        extra = rng.choice(remaining, size=n_remove - len(chosen), replace=False)
        chosen.update(int(p) for p in np.atleast_1d(extra))

    return np.array(sorted(chosen), dtype=int)


def _place_contiguous(
    eligible: np.ndarray, n_remove: int, n_blocks: int, rng: np.random.Generator
) -> np.ndarray:
    """Choose positions to remove as a few long blocks spread across the span.

    Blocks are spaced evenly and jittered within their slot, so the removal is
    balanced across seasons while what survives between blocks stays contiguous.
    A single block would remove entire seasons and confound fragmentation with
    seasonal composition, which is the whole point of having this arm.

    Args:
        eligible: Positions that may be removed (currently observed).
        n_remove: Exact number of positions to remove.
        n_blocks: Number of blocks to spread the removal over.
        rng: Seeded generator.

    Returns:
        Positions to remove.
    """
    n_blocks = max(1, min(n_blocks, n_remove))
    # Operate on ranks within `eligible` so blocks are contiguous in observed
    # time rather than in wall-clock time, which keeps block sizes equal even
    # where the record is already patchy.
    n_eligible = eligible.size
    per_block = n_remove // n_blocks
    remainder = n_remove % n_blocks
    slot = n_eligible / n_blocks

    chosen: list[int] = []
    for b in range(n_blocks):
        size = per_block + (1 if b < remainder else 0)
        low = int(b * slot)
        high = max(low, int((b + 1) * slot) - size)
        start = int(rng.integers(low, high + 1)) if high > low else low
        start = min(start, n_eligible - size)
        chosen.extend(eligible[start : start + size].tolist())

    chosen_set = set(chosen)
    if len(chosen_set) < n_remove:
        remaining = np.array(sorted(set(eligible.tolist()) - chosen_set))
        extra = rng.choice(remaining, size=n_remove - len(chosen_set), replace=False)
        chosen_set.update(int(p) for p in np.atleast_1d(extra))

    return np.array(sorted(chosen_set), dtype=int)


def inject_gaps(
    pm: pd.DataFrame,
    target_column: str,
    *,
    arm: str,
    target_coverage: float,
    profile: GapProfile | None = None,
    protect_from: pd.Timestamp | None = None,
    n_blocks: int = 8,
    seed: int = 42,
) -> tuple[pd.DataFrame, InjectionReport]:
    """Degrade a record to a target coverage, by one of the two arms.

    Args:
        pm: Hourly target frame, UTC-indexed, on a regular grid.
        target_column: Column holding the concentration.
        arm: ``"fragmented"``, ``"contiguous"`` or ``"none"``.
        target_coverage: Desired observed fraction within the injectable region,
            in ``(0, 1]``. A value at or above current coverage removes nothing.
        profile: Empirical gap-length distribution; required for ``fragmented``.
        protect_from: Timestamps at or after this are never touched. Used to keep
            the test period intact so every cell is scored on the same rows.
        n_blocks: Blocks to spread a ``contiguous`` removal over.
        seed: RNG seed.

    Returns:
        ``(degraded_frame, report)``. The frame is a copy; removed hours are NaN.

    Raises:
        ValueError: If the arm is unknown, or ``fragmented`` is asked for without
            a profile.
    """
    if arm not in {"fragmented", "contiguous", "none"}:
        raise ValueError(f"unknown arm {arm!r}; use 'fragmented', 'contiguous' or 'none'")
    if arm == "fragmented" and (profile is None or not profile.lengths.size):
        raise ValueError("the fragmented arm needs a non-empty gap profile")

    out = pm.copy()
    values = out[target_column]

    region = (
        np.ones(len(out), dtype=bool)
        if protect_from is None
        else np.asarray(out.index < protect_from, dtype=bool)
    )
    region_label = (
        "entire record" if protect_from is None else f"before {protect_from:%Y-%m-%d} (test intact)"
    )

    observed = values.notna().to_numpy() & region
    slots = int(region.sum())
    available = int(observed.sum())
    coverage_before = available / slots if slots else 0.0
    gaps_before = int(gap_lengths(values).size)

    target_observed = round(target_coverage * slots)
    n_remove = max(0, available - target_observed)

    if arm == "none" or n_remove == 0:
        return out, InjectionReport(
            arm="none" if arm == "none" else arm,
            target_coverage=target_coverage,
            achieved_coverage=coverage_before,
            coverage_before=coverage_before,
            hours_removed=0,
            hours_available=available,
            slots_in_region=slots,
            n_gaps_before=gaps_before,
            n_gaps_after=gaps_before,
            longest_gap_after=int(gap_lengths(values).max()) if gaps_before else 0,
            seed=seed,
            region=region_label,
        )

    rng = np.random.default_rng(seed)
    eligible = np.flatnonzero(observed)
    if arm == "fragmented":
        positions = _place_fragmented(eligible, n_remove, profile, rng)
    else:
        positions = _place_contiguous(eligible, n_remove, n_blocks, rng)

    out.iloc[positions, out.columns.get_loc(target_column)] = np.nan
    after = out[target_column]
    after_lengths = gap_lengths(after)

    achieved = int((after.notna().to_numpy() & region).sum())
    return out, InjectionReport(
        arm=arm,
        target_coverage=target_coverage,
        achieved_coverage=achieved / slots if slots else 0.0,
        coverage_before=coverage_before,
        hours_removed=int(positions.size),
        hours_available=available,
        slots_in_region=slots,
        n_gaps_before=gaps_before,
        n_gaps_after=int(after_lengths.size),
        longest_gap_after=int(after_lengths.max()) if after_lengths.size else 0,
        seed=seed,
        region=region_label,
    )
