from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np

from core.collisions import (
    SecondaryElectronEventBuffer,
    equal_mass_elastic_collision_3d,
    perform_mcc_electron_1d3v,
)
from core.fields import solve_poisson_cylindrical


EPSILON_0 = 8.8541878128e-12
PASS = "PASS"
FAIL = "FAIL"
CONVERGENCE_AXES = ("time_step", "grid", "particle_count")
OML_MAX_RADIUS_RATIO = 0.2


def _json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_json_value(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    return value


@dataclass(frozen=True)
class BenchmarkResult:
    """Store one benchmark result."""

    name: str
    status: str
    metrics: dict[str, Any]
    thresholds: dict[str, Any]
    message: str

    def __post_init__(self) -> None:
        if self.status not in (PASS, FAIL):
            raise ValueError("The benchmark status must be PASS or FAIL.")

    @property
    def passed(self) -> bool:
        return self.status == PASS

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "metrics": _json_value(self.metrics),
            "thresholds": _json_value(self.thresholds),
            "message": self.message,
        }


@dataclass(frozen=True)
class ConvergenceSeries:
    """Store one ordered convergence series."""

    axis: str
    levels: tuple[float, ...]
    values: tuple[float, ...]
    standard_uncertainties: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        levels = tuple(float(value) for value in self.levels)
        values = tuple(float(value) for value in self.values)
        uncertainties = tuple(float(value) for value in self.standard_uncertainties)
        if not uncertainties:
            uncertainties = (0.0,) * len(values)
        object.__setattr__(self, "levels", levels)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "standard_uncertainties", uncertainties)
        self._validate()

    def _validate(self) -> None:
        if self.axis not in CONVERGENCE_AXES:
            raise ValueError("The convergence axis is not correct.")
        if len(self.levels) < 3:
            raise ValueError("Supply at least three convergence levels.")
        if len(self.values) != len(self.levels):
            raise ValueError("The convergence values do not match the levels.")
        if len(self.standard_uncertainties) != len(self.levels):
            raise ValueError("The uncertainties do not match the levels.")
        if not all(math.isfinite(value) and value > 0.0 for value in self.levels):
            raise ValueError("Each convergence level must be positive and finite.")
        if not all(math.isfinite(value) for value in self.values):
            raise ValueError("Each convergence value must be finite.")
        if not all(
            math.isfinite(value) and value >= 0.0
            for value in self.standard_uncertainties
        ):
            raise ValueError("Each uncertainty must be nonnegative and finite.")
        pairs = zip(self.levels, self.levels[1:], strict=False)
        if self.axis == "time_step":
            ordered = all(first > second for first, second in pairs)
        else:
            ordered = all(first < second for first, second in pairs)
        if not ordered:
            raise ValueError("Put the convergence levels in coarse-to-fine order.")


@dataclass(frozen=True)
class ConvergenceMatrix:
    """Store convergence data for three numerical controls."""

    time_step: ConvergenceSeries
    grid: ConvergenceSeries
    particle_count: ConvergenceSeries
    observable_name: str = "observable"

    def __post_init__(self) -> None:
        series = (self.time_step, self.grid, self.particle_count)
        if tuple(item.axis for item in series) != CONVERGENCE_AXES:
            raise ValueError("The convergence series do not match their fields.")
        if not self.observable_name.strip():
            raise ValueError("Supply an observable name.")


def cylindrical_vacuum_solution(
    n_cells: int,
    *,
    probe_radius_m: float = 5.0e-4,
    wall_radius_m: float = 5.0e-3,
    probe_voltage_v: float = -100.0,
    wall_voltage_v: float = 0.0,
    epsilon_0: float = EPSILON_0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate the numerical and analytical vacuum potentials."""

    if isinstance(n_cells, bool) or not isinstance(n_cells, (int, np.integer)):
        raise ValueError("The cell count must be an integer.")
    if n_cells < 4:
        raise ValueError("The cell count must be at least four.")
    values = (
        probe_radius_m,
        wall_radius_m,
        probe_voltage_v,
        wall_voltage_v,
        epsilon_0,
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Each capacitor input must be finite.")
    if probe_radius_m <= 0.0 or wall_radius_m <= probe_radius_m:
        raise ValueError("The capacitor radii are not correct.")
    if epsilon_0 <= 0.0:
        raise ValueError("The permittivity must be positive.")

    radial_step_m = (wall_radius_m - probe_radius_m) / n_cells
    radius_m = np.linspace(probe_radius_m, wall_radius_m, n_cells + 1)
    charge_density = np.zeros(n_cells + 1, dtype=np.float64)
    numerical_potential_v = np.empty(n_cells + 1, dtype=np.float64)
    solve_poisson_cylindrical(
        charge_density,
        numerical_potential_v,
        probe_radius_m,
        radial_step_m,
        epsilon_0,
        probe_voltage_v,
        wall_voltage_v,
    )
    voltage_fraction = np.log(radius_m / wall_radius_m) / math.log(
        probe_radius_m / wall_radius_m
    )
    analytical_potential_v = wall_voltage_v + (
        probe_voltage_v - wall_voltage_v
    ) * voltage_fraction
    return radius_m, numerical_potential_v, analytical_potential_v


def cylindrical_vacuum_capacitor_benchmark(
    cell_counts: Sequence[int] = (32, 64, 128),
    *,
    maximum_relative_error: float = 1.0e-4,
    minimum_order: float = 1.8,
    probe_radius_m: float = 5.0e-4,
    wall_radius_m: float = 5.0e-3,
    probe_voltage_v: float = -100.0,
    wall_voltage_v: float = 0.0,
) -> BenchmarkResult:
    """Calculate the capacitor error and the grid convergence order."""

    supplied_counts = tuple(cell_counts)
    if any(
        isinstance(value, bool) or not isinstance(value, (int, np.integer))
        for value in supplied_counts
    ):
        raise ValueError("Each cell count must be an integer.")
    counts = tuple(int(value) for value in supplied_counts)
    if len(counts) < 3 or any(value < 4 for value in counts):
        raise ValueError("Supply at least three applicable cell counts.")
    if any(first >= second for first, second in zip(counts, counts[1:])):
        raise ValueError("Put the cell counts in increasing order.")
    if maximum_relative_error <= 0.0 or not math.isfinite(
        maximum_relative_error
    ):
        raise ValueError("The error limit must be positive and finite.")
    if minimum_order <= 0.0 or not math.isfinite(minimum_order):
        raise ValueError("The order limit must be positive and finite.")

    voltage_scale = max(abs(probe_voltage_v - wall_voltage_v), 1.0)
    relative_linf_errors: list[float] = []
    relative_l2_errors: list[float] = []
    for count in counts:
        _, numerical, analytical = cylindrical_vacuum_solution(
            count,
            probe_radius_m=probe_radius_m,
            wall_radius_m=wall_radius_m,
            probe_voltage_v=probe_voltage_v,
            wall_voltage_v=wall_voltage_v,
        )
        difference = numerical - analytical
        relative_linf_errors.append(
            float(np.max(np.abs(difference)) / voltage_scale)
        )
        relative_l2_errors.append(
            float(np.sqrt(np.mean(difference * difference)) / voltage_scale)
        )

    orders: list[float] = []
    for index in range(1, len(counts)):
        coarse_error = relative_linf_errors[index - 1]
        fine_error = relative_linf_errors[index]
        refinement = counts[index] / counts[index - 1]
        if coarse_error > 0.0 and fine_error > 0.0:
            orders.append(math.log(coarse_error / fine_error) / math.log(refinement))
        elif coarse_error == fine_error:
            orders.append(0.0)
        else:
            orders.append(math.inf)

    finest_error = relative_linf_errors[-1]
    measured_minimum_order = min(orders)
    passed = (
        finest_error <= maximum_relative_error
        and measured_minimum_order >= minimum_order
    )
    message = (
        "The vacuum solution agrees with the analytical solution."
        if passed
        else "The vacuum solution does not satisfy the error limits."
    )
    return BenchmarkResult(
        name="cylindrical_vacuum_capacitor",
        status=PASS if passed else FAIL,
        metrics={
            "cell_counts": counts,
            "relative_linf_errors": relative_linf_errors,
            "relative_l2_errors": relative_l2_errors,
            "observed_orders": orders,
            "finest_relative_linf_error": finest_error,
            "minimum_observed_order": measured_minimum_order,
        },
        thresholds={
            "maximum_finest_relative_linf_error": maximum_relative_error,
            "minimum_grid_convergence_order": minimum_order,
        },
        message=message,
    )


def _empty_secondary_buffer() -> SecondaryElectronEventBuffer:
    return SecondaryElectronEventBuffer(
        parent_index=np.empty(0, dtype=np.int64),
        event_time_s=np.empty(0, dtype=np.float64),
        vx=np.empty(0, dtype=np.float64),
        vy=np.empty(0, dtype=np.float64),
        vz=np.empty(0, dtype=np.float64),
        energy_ev=np.empty(0, dtype=np.float64),
    )


def constant_cross_section_collision_benchmark(
    *,
    particle_count: int = 2_000,
    replicate_count: int = 6,
    neutral_density_m3: float = 0.75,
    cross_section_m2: float = 1.0,
    speed_m_s: float = 1.0,
    interval_s: float = 1.0,
    seed: int = 20260729,
    maximum_z_score: float = 5.0,
    max_events_per_particle: int = 64,
) -> BenchmarkResult:
    """Compare constant-cross-section event counts with a Poisson rate."""

    integer_values = (
        ("particle_count", particle_count),
        ("replicate_count", replicate_count),
        ("max_events_per_particle", max_events_per_particle),
    )
    for name, value in integer_values:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must be an integer.")
        if value <= 0:
            raise ValueError(f"{name} must be positive.")
    rate_values = (
        neutral_density_m3,
        cross_section_m2,
        speed_m_s,
        interval_s,
        maximum_z_score,
    )
    if not all(math.isfinite(value) and value > 0.0 for value in rate_values):
        raise ValueError("Each collision-box input must be positive and finite.")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError("The seed must be an integer.")

    mean_events_per_particle = (
        neutral_density_m3 * cross_section_m2 * speed_m_s * interval_s
    )
    counts: list[int] = []
    event_limit_stops = 0
    overflow_lookups = 0
    constant_sigma = np.array(
        [cross_section_m2, cross_section_m2],
        dtype=np.float64,
    )
    zero_sigma = np.zeros(2, dtype=np.float64)
    collision_energy = speed_m_s * speed_m_s
    for replicate in range(replicate_count):
        velocity_x = np.full(particle_count, speed_m_s, dtype=np.float64)
        velocity_y = np.zeros(particle_count, dtype=np.float64)
        velocity_z = np.zeros(particle_count, dtype=np.float64)
        alive = np.ones(particle_count, dtype=np.bool_)
        collision_result = perform_mcc_electron_1d3v(
            velocity_x,
            velocity_y,
            velocity_z,
            alive,
            n_g=neutral_density_m3,
            sigma_el=constant_sigma,
            sigma_exc=zero_sigma,
            sigma_ion=zero_sigma,
            dt=interval_s,
            m_e=2.0,
            excitation_threshold_ev=collision_energy + 1.0,
            ionization_threshold_ev=collision_energy + 1.0,
            e_min=0.0,
            inv_de=0.0,
            e_charge=1.0,
            secondary_buffer=_empty_secondary_buffer(),
            seed=int(seed),
            step_index=replicate,
            stream_id=73,
            max_events_per_particle=max_events_per_particle,
        )
        counts.append(int(collision_result.elastic_events))
        event_limit_stops += int(collision_result.event_limit_stops)
        overflow_lookups += int(collision_result.energy_table_overflow_lookups)

    observed_total = sum(counts)
    expected_per_replicate = particle_count * mean_events_per_particle
    expected_total = replicate_count * expected_per_replicate
    poisson_standard_deviation = math.sqrt(expected_total)
    z_score = abs(observed_total - expected_total) / poisson_standard_deviation
    observed_mean = observed_total / (particle_count * replicate_count)
    passed = (
        z_score <= maximum_z_score
        and event_limit_stops == 0
        and overflow_lookups == 0
    )
    message = (
        "The collision count agrees with the Poisson rate."
        if passed
        else "The collision count does not satisfy the Poisson-rate limits."
    )
    return BenchmarkResult(
        name="constant_cross_section_collision_box",
        status=PASS if passed else FAIL,
        metrics={
            "replicate_event_counts": counts,
            "observed_total_events": observed_total,
            "expected_total_events": expected_total,
            "observed_mean_events_per_particle": observed_mean,
            "expected_mean_events_per_particle": mean_events_per_particle,
            "absolute_z_score": z_score,
            "event_limit_stops": event_limit_stops,
            "energy_table_overflow_lookups": overflow_lookups,
        },
        thresholds={
            "maximum_absolute_z_score": maximum_z_score,
            "maximum_event_limit_stops": 0,
            "maximum_energy_table_overflow_lookups": 0,
        },
        message=message,
    )


def equal_mass_elastic_conservation_benchmark(
    ion_velocity_m_s: Sequence[float] = (3.0, -1.0, 2.0),
    neutral_velocity_m_s: Sequence[float] = (-0.5, 0.75, 1.25),
    *,
    direction_cosine: float = 0.2,
    azimuth_rad: float = 1.3,
    particle_mass_kg: float = 1.0,
    maximum_momentum_error: float = 1.0e-12,
    maximum_relative_energy_error: float = 1.0e-12,
) -> BenchmarkResult:
    """Calculate momentum and energy errors for one elastic collision."""

    ion_before = np.asarray(ion_velocity_m_s, dtype=np.float64)
    neutral_before = np.asarray(neutral_velocity_m_s, dtype=np.float64)
    if ion_before.shape != (3,) or neutral_before.shape != (3,):
        raise ValueError("Each velocity must have three components.")
    scalar_values = (
        direction_cosine,
        azimuth_rad,
        particle_mass_kg,
        maximum_momentum_error,
        maximum_relative_energy_error,
    )
    if not all(math.isfinite(value) for value in scalar_values):
        raise ValueError("Each elastic-collision input must be finite.")
    if particle_mass_kg <= 0.0:
        raise ValueError("The particle mass must be positive.")
    if maximum_momentum_error <= 0.0 or maximum_relative_energy_error <= 0.0:
        raise ValueError("Each conservation limit must be positive.")

    ion_after, neutral_after = equal_mass_elastic_collision_3d(
        ion_before,
        neutral_before,
        direction_cosine,
        azimuth_rad,
    )
    momentum_before = particle_mass_kg * (ion_before + neutral_before)
    momentum_after = particle_mass_kg * (ion_after + neutral_after)
    momentum_error = float(np.linalg.norm(momentum_after - momentum_before))
    energy_before = 0.5 * particle_mass_kg * (
        float(np.dot(ion_before, ion_before))
        + float(np.dot(neutral_before, neutral_before))
    )
    energy_after = 0.5 * particle_mass_kg * (
        float(np.dot(ion_after, ion_after))
        + float(np.dot(neutral_after, neutral_after))
    )
    relative_energy_error = abs(energy_after - energy_before) / max(
        abs(energy_before),
        np.finfo(float).tiny,
    )
    passed = (
        momentum_error <= maximum_momentum_error
        and relative_energy_error <= maximum_relative_energy_error
    )
    message = (
        "The elastic collision keeps momentum and energy constant."
        if passed
        else "The elastic collision does not satisfy the conservation limits."
    )
    return BenchmarkResult(
        name="equal_mass_elastic_conservation",
        status=PASS if passed else FAIL,
        metrics={
            "ion_velocity_after_m_s": ion_after,
            "neutral_velocity_after_m_s": neutral_after,
            "absolute_momentum_error_kg_m_s": momentum_error,
            "relative_energy_error": relative_energy_error,
            "energy_before_j": energy_before,
            "energy_after_j": energy_after,
        },
        thresholds={
            "maximum_absolute_momentum_error_kg_m_s": maximum_momentum_error,
            "maximum_relative_energy_error": maximum_relative_energy_error,
        },
        message=message,
    )


def oml_applicability_benchmark(
    probe_radius_m: float,
    debye_length_m: float,
    *,
    maximum_radius_ratio: float = OML_MAX_RADIUS_RATIO,
) -> BenchmarkResult:
    """Apply the probe-radius limit for an OML comparison."""

    values = (probe_radius_m, debye_length_m, maximum_radius_ratio)
    if not all(math.isfinite(value) and value > 0.0 for value in values):
        raise ValueError("Each OML input must be positive and finite.")
    if maximum_radius_ratio > OML_MAX_RADIUS_RATIO:
        raise ValueError("The OML radius-ratio limit cannot be more than 0.2.")
    ratio = probe_radius_m / debye_length_m
    passed = ratio <= maximum_radius_ratio
    message = (
        "The probe radius is inside the OML limit."
        if passed
        else "The probe radius is outside the OML limit."
    )
    return BenchmarkResult(
        name="oml_applicability",
        status=PASS if passed else FAIL,
        metrics={
            "probe_radius_to_debye_length": ratio,
            "comparison_permitted": passed,
        },
        thresholds={
            "maximum_probe_radius_to_debye_length": maximum_radius_ratio,
        },
        message=message,
    )


def oml_ion_current_comparison(
    probe_radius_m: float,
    debye_length_m: float,
    bias_voltage_v: Sequence[float],
    ion_current_a: Sequence[float],
    *,
    maximum_radius_ratio: float = OML_MAX_RADIUS_RATIO,
    minimum_r_squared: float = 0.98,
) -> BenchmarkResult:
    """Compare ion-current data with the OML linear relation."""

    gate = oml_applicability_benchmark(
        probe_radius_m,
        debye_length_m,
        maximum_radius_ratio=maximum_radius_ratio,
    )
    thresholds = {
        "maximum_probe_radius_to_debye_length": maximum_radius_ratio,
        "minimum_r_squared": minimum_r_squared,
        "positive_fit_slope_required": True,
    }
    if not gate.passed:
        return BenchmarkResult(
            name="oml_ion_current_scaling",
            status=FAIL,
            metrics={
                "probe_radius_to_debye_length": gate.metrics[
                    "probe_radius_to_debye_length"
                ],
                "comparison_performed": False,
                "r_squared": None,
                "fit_slope": None,
            },
            thresholds=thresholds,
            message="The OML gate stopped the current comparison.",
        )
    if not math.isfinite(minimum_r_squared) or not 0.0 <= minimum_r_squared <= 1.0:
        raise ValueError("The R-squared limit must be from zero through one.")

    voltage = np.asarray(bias_voltage_v, dtype=np.float64)
    current = np.asarray(ion_current_a, dtype=np.float64)
    data_are_correct = (
        voltage.ndim == 1
        and current.ndim == 1
        and voltage.shape == current.shape
        and voltage.size >= 3
        and np.all(np.isfinite(voltage))
        and np.all(np.isfinite(current))
    )
    if not data_are_correct:
        return BenchmarkResult(
            name="oml_ion_current_scaling",
            status=FAIL,
            metrics={
                "probe_radius_to_debye_length": gate.metrics[
                    "probe_radius_to_debye_length"
                ],
                "comparison_performed": False,
                "r_squared": None,
                "fit_slope": None,
            },
            thresholds=thresholds,
            message="The OML data are not correct.",
        )

    voltage_magnitude = np.abs(voltage)
    squared_current = current * current
    if np.ptp(voltage_magnitude) <= 0.0 or np.ptp(squared_current) <= 0.0:
        return BenchmarkResult(
            name="oml_ion_current_scaling",
            status=FAIL,
            metrics={
                "probe_radius_to_debye_length": gate.metrics[
                    "probe_radius_to_debye_length"
                ],
                "comparison_performed": False,
                "r_squared": None,
                "fit_slope": None,
            },
            thresholds=thresholds,
            message="The OML data do not have a sufficient range.",
        )

    slope, intercept = np.polyfit(voltage_magnitude, squared_current, 1)
    fitted = slope * voltage_magnitude + intercept
    residual_sum = float(np.sum((squared_current - fitted) ** 2))
    total_sum = float(
        np.sum((squared_current - np.mean(squared_current)) ** 2)
    )
    r_squared = 1.0 - residual_sum / total_sum
    passed = r_squared >= minimum_r_squared and slope > 0.0
    message = (
        "The current data satisfy the OML comparison limits."
        if passed
        else "The current data do not satisfy the OML comparison limits."
    )
    return BenchmarkResult(
        name="oml_ion_current_scaling",
        status=PASS if passed else FAIL,
        metrics={
            "probe_radius_to_debye_length": gate.metrics[
                "probe_radius_to_debye_length"
            ],
            "comparison_performed": True,
            "r_squared": float(r_squared),
            "fit_slope": float(slope),
            "fit_intercept": float(intercept),
            "point_count": int(voltage.size),
        },
        thresholds=thresholds,
        message=message,
    )


def evaluate_convergence_matrix(
    matrix: ConvergenceMatrix,
    *,
    relative_tolerance: float = 0.02,
    absolute_tolerance: float = 0.0,
    uncertainty_multiplier: float = 2.0,
) -> BenchmarkResult:
    """Apply one convergence criterion to each numerical control."""

    limits = (relative_tolerance, absolute_tolerance, uncertainty_multiplier)
    if not all(math.isfinite(value) and value >= 0.0 for value in limits):
        raise ValueError("Each convergence limit must be nonnegative and finite.")
    if relative_tolerance == 0.0 and absolute_tolerance == 0.0:
        raise ValueError("Supply a nonzero convergence tolerance.")

    axis_metrics: dict[str, dict[str, Any]] = {}
    axis_passes: list[bool] = []
    for series in (matrix.time_step, matrix.grid, matrix.particle_count):
        coarse_value = series.values[-2]
        fine_value = series.values[-1]
        difference = abs(fine_value - coarse_value)
        scale = max(abs(coarse_value), abs(fine_value))
        combined_uncertainty = math.hypot(
            series.standard_uncertainties[-2],
            series.standard_uncertainties[-1],
        )
        allowed_difference = (
            absolute_tolerance
            + relative_tolerance * scale
            + uncertainty_multiplier * combined_uncertainty
        )
        relative_change = difference / max(scale, np.finfo(float).tiny)
        axis_passed = difference <= allowed_difference
        axis_passes.append(axis_passed)
        axis_metrics[series.axis] = {
            "coarse_level": series.levels[-2],
            "fine_level": series.levels[-1],
            "coarse_value": coarse_value,
            "fine_value": fine_value,
            "absolute_change": difference,
            "relative_change": relative_change,
            "combined_standard_uncertainty": combined_uncertainty,
            "allowed_absolute_change": allowed_difference,
            "status": PASS if axis_passed else FAIL,
        }

    passed = all(axis_passes)
    message = (
        "All numerical controls satisfy the convergence limits."
        if passed
        else "One or more numerical controls do not satisfy the convergence limits."
    )
    return BenchmarkResult(
        name="numerical_convergence_matrix",
        status=PASS if passed else FAIL,
        metrics={
            "observable_name": matrix.observable_name,
            "axes": axis_metrics,
        },
        thresholds={
            "maximum_relative_change": relative_tolerance,
            "absolute_change_allowance": absolute_tolerance,
            "uncertainty_multiplier": uncertainty_multiplier,
            "minimum_levels_per_axis": 3,
        },
        message=message,
    )


def run_fast_benchmarks(
    *,
    vacuum_cell_counts: Sequence[int] = (32, 64, 128),
    collision_particle_count: int = 2_000,
    collision_replicate_count: int = 6,
    collision_mean_events: float = 0.75,
    maximum_collision_z_score: float = 5.0,
) -> tuple[BenchmarkResult, ...]:
    """Do the fast benchmarks that do not use a long PIC calculation."""

    return (
        cylindrical_vacuum_capacitor_benchmark(vacuum_cell_counts),
        constant_cross_section_collision_benchmark(
            particle_count=collision_particle_count,
            replicate_count=collision_replicate_count,
            neutral_density_m3=collision_mean_events,
            cross_section_m2=1.0,
            speed_m_s=1.0,
            interval_s=1.0,
            maximum_z_score=maximum_collision_z_score,
        ),
        equal_mass_elastic_conservation_benchmark(),
    )


def make_benchmark_report(
    results: Sequence[BenchmarkResult],
) -> dict[str, Any]:
    """Make one JSON-compatible report."""

    items = tuple(results)
    passed = all(result.passed for result in items)
    return {
        "suite_status": PASS if passed else FAIL,
        "benchmark_count": len(items),
        "passed_count": sum(result.passed for result in items),
        "failed_count": sum(not result.passed for result in items),
        "results": [result.to_dict() for result in items],
    }


def write_benchmark_report(
    report: dict[str, Any],
    output_dir: str | Path,
) -> Path:
    """Write a report only in the supplied output directory."""

    directory = Path(output_dir).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "benchmark_results.json"
    with target.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(_json_value(report), stream, indent=2, sort_keys=True)
        stream.write("\n")
    return target


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Do fast checks for the probe simulation.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Write benchmark_results.json in this directory.",
    )
    parser.add_argument(
        "--vacuum-cells",
        type=int,
        nargs="+",
        default=[32, 64, 128],
        help="Supply the ordered cell counts for the vacuum check.",
    )
    parser.add_argument(
        "--collision-particles",
        type=int,
        default=2_000,
        help="Set the particle count for each collision-box sample.",
    )
    parser.add_argument(
        "--collision-replicates",
        type=int,
        default=6,
        help="Set the number of collision-box samples.",
    )
    parser.add_argument(
        "--collision-mean-events",
        type=float,
        default=0.75,
        help="Set the expected event count for each particle.",
    )
    parser.add_argument(
        "--collision-max-z-score",
        type=float,
        default=5.0,
        help="Set the maximum absolute Poisson z-score.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_argument_parser()
    arguments = parser.parse_args(argv)
    try:
        results = run_fast_benchmarks(
            vacuum_cell_counts=arguments.vacuum_cells,
            collision_particle_count=arguments.collision_particles,
            collision_replicate_count=arguments.collision_replicates,
            collision_mean_events=arguments.collision_mean_events,
            maximum_collision_z_score=arguments.collision_max_z_score,
        )
        report = make_benchmark_report(results)
        if arguments.output_dir is not None:
            write_benchmark_report(report, arguments.output_dir)
    except (OSError, ValueError) as error:
        print(
            json.dumps(
                {
                    "suite_status": FAIL,
                    "error": str(error),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2

    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["suite_status"] == PASS else 1


__all__ = [
    "FAIL",
    "OML_MAX_RADIUS_RATIO",
    "PASS",
    "BenchmarkResult",
    "ConvergenceMatrix",
    "ConvergenceSeries",
    "constant_cross_section_collision_benchmark",
    "cylindrical_vacuum_capacitor_benchmark",
    "cylindrical_vacuum_solution",
    "equal_mass_elastic_conservation_benchmark",
    "evaluate_convergence_matrix",
    "main",
    "make_benchmark_report",
    "oml_applicability_benchmark",
    "oml_ion_current_comparison",
    "run_fast_benchmarks",
    "write_benchmark_report",
]


if __name__ == "__main__":
    raise SystemExit(main())
