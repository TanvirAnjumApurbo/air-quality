"""Training energy and CO2e accounting via CodeCarbon.

**These are estimates, not metered measurements.** CodeCarbon derives energy from
hardware power models and, where it cannot read Intel RAPL counters, falls back
to modelled values entirely. On Windows RAPL is normally unreadable without
elevated privileges, so the CPU component here is very likely modelled rather
than measured. Whether RAPL was available is recorded per run and surfaced in the
results, because a reader cannot otherwise tell which is which.

The Bangladesh figure is a recomputation, not a second measurement: the same
estimated kWh is multiplied by a citable national grid intensity from
``green.grid_carbon_intensity``. If that value is not marked VERIFIED, the
recomputed column is left blank rather than filled with a guess.
"""

from __future__ import annotations

import contextlib
import time
from typing import Any

from src.utils import Config, ConfigError


class EnergyTracker:
    """Context manager wrapping a CodeCarbon ``EmissionsTracker``.

    Degrades to wall-clock timing only if CodeCarbon is unavailable or fails to
    start, so a tracker problem can never abort a training run.
    """

    def __init__(self, cfg: Config, run_name: str, logger: Any) -> None:
        """Prepare a tracker for one run.

        Args:
            cfg: Loaded configuration.
            run_name: Identifier recorded with the measurement.
            logger: Logger.
        """
        self.cfg = cfg
        self.run_name = run_name
        self.logger = logger
        self._tracker: Any = None
        self._started = 0.0
        self.duration_s = 0.0
        self.energy_kwh: float | None = None
        self.emissions_kg: float | None = None
        self.error: str | None = None
        self.rapl_available: bool | None = None

    def __enter__(self) -> EnergyTracker:
        """Start tracking."""
        self._started = time.perf_counter()
        spec = self.cfg.get("green.energy", {})
        try:
            from codecarbon import EmissionsTracker

            self._tracker = EmissionsTracker(
                project_name=self.cfg.get("project.name", "pipeline"),
                experiment_id=self.run_name,
                measure_power_secs=int(spec.get("measure_power_secs", 15)),
                output_dir=str(self.cfg.path_for("logs")),
                output_file=str(spec.get("output_file", "emissions.csv")),
                log_level=str(spec.get("log_level", "warning")),
                country_iso_code=str(spec.get("country_iso_code", "BGD")),
                save_to_file=True,
                allow_multiple_runs=True,
            )
            self._tracker.start()
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            self.logger.warning("CodeCarbon unavailable (%s); timing only", self.error)
            self._tracker = None
        return self

    def __exit__(self, *exc_info: object) -> None:
        """Stop tracking and record the estimate."""
        self.duration_s = time.perf_counter() - self._started
        if self._tracker is None:
            return
        try:
            emissions = self._tracker.stop()
            self.emissions_kg = float(emissions) if emissions is not None else None
            data = getattr(self._tracker, "final_emissions_data", None)
            if data is not None:
                self.energy_kwh = float(getattr(data, "energy_consumed", 0.0)) or None
                # CodeCarbon reports zero CPU energy when it cannot read RAPL.
                cpu_energy = float(getattr(data, "cpu_energy", 0.0) or 0.0)
                self.rapl_available = cpu_energy > 0.0
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            self.logger.warning("CodeCarbon stop failed (%s)", self.error)

    def summary(self) -> dict[str, Any]:
        """Serialise the measurement, including its provenance caveats.

        Returns:
            Mapping recorded alongside the run in results.json.
        """
        out: dict[str, Any] = {
            "duration_s": round(self.duration_s, 3),
            "energy_kwh": self.energy_kwh,
            "co2e_kg_codecarbon": self.emissions_kg,
            "tracker": "codecarbon" if self._tracker is not None else "unavailable",
            "rapl_available": self.rapl_available,
            "is_estimate": True,
            "note": (
                "CodeCarbon reports modelled estimates, not metered measurements. "
                "When Intel RAPL counters are unreadable (the usual case on Windows) "
                "the CPU component is fully modelled."
            ),
        }
        if self.error:
            out["error"] = self.error

        recomputed = recompute_for_grid(self.cfg, self.energy_kwh)
        out.update(recomputed)
        return out


def recompute_for_grid(cfg: Config, energy_kwh: float | None) -> dict[str, Any]:
    """Recompute CO2e under the Bangladesh grid carbon intensity.

    Refuses to produce a number when the intensity is not backed by a verified
    citation: the cell is left null and flagged instead.

    Args:
        cfg: Loaded configuration.
        energy_kwh: Estimated energy for the run.

    Returns:
        Mapping with the recomputed grams of CO2e and its provenance, or a
        flagged null when the intensity is unverified.
    """
    try:
        block = cfg.require_verified("green.grid_carbon_intensity")
    except ConfigError as exc:
        return {
            "co2e_g_bd_grid": None,
            "grid_intensity_gco2_per_kwh": None,
            "grid_intensity_status": f"UNVERIFIED -- {exc}",
        }

    intensity = block.get("value_gco2_per_kwh")
    if intensity is None or energy_kwh is None:
        return {
            "co2e_g_bd_grid": None,
            "grid_intensity_gco2_per_kwh": intensity,
            "grid_intensity_source": block.get("source_name"),
            "grid_intensity_year": block.get("year"),
        }

    return {
        "co2e_g_bd_grid": float(energy_kwh) * float(intensity),
        "grid_intensity_gco2_per_kwh": float(intensity),
        "grid_intensity_year": block.get("year"),
        "grid_intensity_source": block.get("source_name"),
        "grid_intensity_url": block.get("source_url"),
    }


@contextlib.contextmanager
def optional_tracker(cfg: Config, run_name: str, logger: Any, enabled: bool = True):  # noqa: ANN201
    """Track energy only when enabled, with the same interface either way.

    Args:
        cfg: Loaded configuration.
        run_name: Identifier for the run.
        logger: Logger.
        enabled: Whether to actually track.

    Yields:
        An :class:`EnergyTracker`, started if enabled.
    """
    if not enabled:
        tracker = EnergyTracker(cfg, run_name, logger)
        tracker._started = time.perf_counter()
        yield tracker
        tracker.duration_s = time.perf_counter() - tracker._started
        return
    with EnergyTracker(cfg, run_name, logger) as tracker:
        yield tracker
