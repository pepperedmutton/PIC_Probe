from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from statistics import NormalDist
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence

import numpy as np

from core.config import Config
from core.data_output import DATASET_SCHEMA_VERSION, make_run_id, write_dataset
from core.provenance import PHYSICS_MODEL_VERSION, canonical_json
from core.rng import derive_seed
from core.simulation import PICSimulation


PREVIEW_QUALITY = "PREVIEW"
FAIL_QUALITY = "FAIL"
DEFAULT_ION_TRANSIT_MULTIPLES = 3.0
_DIRECTION_TAGS = {"up": 0, "down": 1}


class SimulationLike(Protocol):
    config: Config
    dt: float
    probe_length: float
    qe: float
    qi: float
    root_seed: int
    v_bias: float

    def step(self) -> tuple[int, int]:
        ...

    def _update_fields(self) -> None:
        ...

    def run_manifest(self) -> dict[str, Any]:
        ...


SimulationFactory = Callable[..., SimulationLike]


@dataclass(frozen=True)
class StudySettings:
    """Keep the controls for one research-preview study."""

    voltages: tuple[float, ...]
    repeats: int = 3
    root_seed: int = 20260729
    n_particles: int = 4000
    probe_length_m: float = 1.0
    sigma_cex_m2: float = 5.0e-19
    ramp_steps: int = 200
    minimum_burn_steps: int = 1000
    convergence_block_steps: int = 500
    maximum_convergence_blocks: int = 8
    required_stable_pairs: int = 2
    convergence_relative_tolerance: float = 0.05
    convergence_absolute_tolerance_a: float = 1.0e-9
    sampling_steps: int | None = None
    ion_transit_multiples: float = DEFAULT_ION_TRANSIT_MULTIPLES
    confidence_level: float = 0.95

    def __post_init__(self) -> None:
        if len(self.voltages) < 2:
            raise ValueError("Set voltages to at least two values.")
        if not all(math.isfinite(float(value)) for value in self.voltages):
            raise ValueError("Set voltages to finite values.")
        if any(
            self.voltages[index] >= self.voltages[index + 1]
            for index in range(len(self.voltages) - 1)
        ):
            raise ValueError("Set voltages in increasing order.")
        for name, value, minimum in (
            ("repeats", self.repeats, 1),
            ("n_particles", self.n_particles, 1),
            ("ramp_steps", self.ramp_steps, 0),
            ("minimum_burn_steps", self.minimum_burn_steps, 0),
            ("convergence_block_steps", self.convergence_block_steps, 2),
            ("maximum_convergence_blocks", self.maximum_convergence_blocks, 2),
            ("required_stable_pairs", self.required_stable_pairs, 1),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"Set {name} to an integer not less than {minimum}.")
        if self.required_stable_pairs >= self.maximum_convergence_blocks:
            raise ValueError(
                "Set required_stable_pairs below maximum_convergence_blocks."
            )
        if self.sampling_steps is not None:
            if (
                isinstance(self.sampling_steps, bool)
                or not isinstance(self.sampling_steps, int)
                or self.sampling_steps < 2
            ):
                raise ValueError("Set sampling_steps to an integer not less than 2.")
        for name, value in (
            ("probe_length_m", self.probe_length_m),
            ("ion_transit_multiples", self.ion_transit_multiples),
            ("confidence_level", self.confidence_level),
        ):
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"Set {name} to a value greater than zero.")
        if self.confidence_level >= 1.0:
            raise ValueError("Set confidence_level to a value less than 1.")
        for name, value in (
            ("sigma_cex_m2", self.sigma_cex_m2),
            (
                "convergence_relative_tolerance",
                self.convergence_relative_tolerance,
            ),
            (
                "convergence_absolute_tolerance_a",
                self.convergence_absolute_tolerance_a,
            ),
        ):
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"Set {name} to zero or a positive value.")
        if isinstance(self.root_seed, bool) or not isinstance(self.root_seed, int):
            raise TypeError("Set root_seed to an integer.")
        if self.root_seed < 0 or self.root_seed > (1 << 64) - 1:
            raise ValueError("Set root_seed from 0 through 2^64 - 1.")


@dataclass(frozen=True)
class CurrentSummary:
    """Keep the statistics for one current series."""

    mean_a: float
    standard_error_a: float
    confidence_low_a: float
    confidence_high_a: float
    batch_mean_count: int
    sample_count: int


@dataclass(frozen=True)
class ConvergenceReport:
    """Keep the result of one convergence check."""

    converged: bool
    blocks_used: int
    stable_pairs: int
    last_difference_a: float
    last_threshold_a: float
    block_means_a: tuple[float, ...]


def calculate_ion_transit_time(config: Config) -> float:
    """Calculate one ion transit time across the radial domain."""
    if config.ION_INJECTION_BOHM:
        characteristic_speed = math.sqrt(config.e * config.Te / config.m_i)
    else:
        characteristic_speed = math.sqrt(config.e * config.Ti / config.m_i)
    return (config.R_MAX - config.R_MIN) / characteristic_speed


def calculate_minimum_sampling_steps(
    config: Config,
    ion_transit_multiples: float = DEFAULT_ION_TRANSIT_MULTIPLES,
) -> int:
    """Calculate the minimum number of sampling steps."""
    if not math.isfinite(ion_transit_multiples) or ion_transit_multiples <= 0.0:
        raise ValueError("Set ion_transit_multiples to a value greater than zero.")
    duration = ion_transit_multiples * calculate_ion_transit_time(config)
    return max(2, int(math.ceil(duration / config.DT)))


def resolve_sampling_steps(config: Config, settings: StudySettings) -> tuple[int, int]:
    """Give the sampling steps and the required minimum."""
    minimum_steps = calculate_minimum_sampling_steps(
        config,
        settings.ion_transit_multiples,
    )
    if settings.sampling_steps is None:
        return minimum_steps, minimum_steps
    if settings.sampling_steps < minimum_steps:
        raise ValueError(
            f"Set sampling_steps to at least {minimum_steps}. "
            f"This value covers {settings.ion_transit_multiples:g} ion transit times."
        )
    return settings.sampling_steps, minimum_steps


def make_seed_plan(root_seed: int, repeats: int) -> list[dict[str, int | str]]:
    """Calculate an independent seed for each directional replicate."""
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
        raise ValueError("Set repeats to an integer not less than 1.")
    plan: list[dict[str, int | str]] = []
    used: set[int] = set()
    for pair_replicate in range(repeats):
        for direction, direction_tag in _DIRECTION_TAGS.items():
            seed = derive_seed(root_seed, pair_replicate, direction_tag)
            if seed in used:
                raise RuntimeError("The seed plan has a duplicate seed.")
            used.add(seed)
            plan.append(
                {
                    "direction": direction,
                    "direction_tag": direction_tag,
                    "pair_replicate": pair_replicate,
                    "seed": seed,
                }
            )
    return plan


def estimate_batch_mean_count(values: Sequence[float]) -> int:
    """Give the number of nonoverlapping sample batches."""
    samples = np.asarray(values, dtype=float)
    if samples.ndim != 1 or samples.size < 2:
        raise ValueError("Give at least two current samples.")
    if not np.all(np.isfinite(samples)):
        raise ValueError("Give only finite current samples.")
    batch_count = min(samples.size, max(2, int(math.sqrt(samples.size))))
    return int(batch_count)


def summarize_current(
    values: Sequence[float],
    confidence_level: float = 0.95,
) -> CurrentSummary:
    """Calculate the mean, confidence interval, and effective sample count."""
    samples = np.asarray(values, dtype=float)
    if samples.ndim != 1 or samples.size < 2:
        raise ValueError("Give at least two current samples.")
    if not np.all(np.isfinite(samples)):
        raise ValueError("Give only finite current samples.")
    if not math.isfinite(confidence_level) or not 0.0 < confidence_level < 1.0:
        raise ValueError("Set confidence_level between 0 and 1.")

    mean = float(np.mean(samples))
    batch_count = estimate_batch_mean_count(samples)
    batch_means = np.asarray(
        [float(np.mean(batch)) for batch in np.array_split(samples, batch_count)]
    )
    standard_error = float(
        np.std(batch_means, ddof=1) / math.sqrt(batch_count)
    )
    probability = 0.5 + confidence_level / 2.0
    multiplier = NormalDist().inv_cdf(probability)
    margin = multiplier * standard_error
    return CurrentSummary(
        mean_a=mean,
        standard_error_a=standard_error,
        confidence_low_a=mean - margin,
        confidence_high_a=mean + margin,
        batch_mean_count=batch_count,
        sample_count=int(samples.size),
    )


def sample_current_series(
    sim: SimulationLike,
    sampling_steps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Record the current for each sampling step."""
    if isinstance(sampling_steps, bool) or not isinstance(sampling_steps, int):
        raise TypeError("Set sampling_steps to an integer.")
    if sampling_steps < 2:
        raise ValueError("Set sampling_steps to an integer not less than 2.")
    electron = np.empty(sampling_steps, dtype=float)
    ion = np.empty(sampling_steps, dtype=float)
    for index in range(sampling_steps):
        electron_hits, ion_hits = sim.step()
        electron[index] = (electron_hits * -sim.qe) / sim.dt
        ion[index] = (ion_hits * sim.qi) / sim.dt
    if sim.probe_length != 1.0:
        electron *= sim.probe_length
        ion *= sim.probe_length
    return electron, ion, electron - ion


def sample_current(
    sim: SimulationLike,
    sampling_steps: int,
) -> tuple[float, float, float]:
    """Give the mean electron, ion, and total currents."""
    electron, ion, total = sample_current_series(sim, sampling_steps)
    return float(np.mean(electron)), float(np.mean(ion)), float(np.mean(total))


def ramp_bias(
    sim: SimulationLike,
    v_from: float,
    v_to: float,
    ramp_steps: int,
) -> None:
    """Change the probe bias in equal steps."""
    if isinstance(ramp_steps, bool) or not isinstance(ramp_steps, int):
        raise TypeError("Set ramp_steps to an integer.")
    if ramp_steps < 0:
        raise ValueError("Set ramp_steps to zero or a positive integer.")
    if ramp_steps == 0:
        sim.v_bias = float(v_to)
        sim._update_fields()
        return
    for step_index in range(ramp_steps):
        fraction = (step_index + 1) / ramp_steps
        sim.v_bias = float(v_from + fraction * (v_to - v_from))
        sim._update_fields()
        sim.step()


def stabilize_bias(
    sim: SimulationLike,
    bias: float,
    minimum_burn_steps: int,
    block_steps: int,
    maximum_blocks: int,
    relative_tolerance: float,
    absolute_tolerance_a: float,
    required_stable_pairs: int,
) -> ConvergenceReport:
    """Check consecutive current blocks for convergence."""
    sim.v_bias = float(bias)
    sim._update_fields()
    for _ in range(max(0, int(minimum_burn_steps))):
        sim.step()

    block_means: list[float] = []
    stable_pairs = 0
    last_difference = math.inf
    last_threshold = max(0.0, float(absolute_tolerance_a))
    for block_number in range(1, max(2, int(maximum_blocks)) + 1):
        _, _, total_current = sample_current_series(sim, max(2, int(block_steps)))
        block_mean = float(np.mean(total_current))
        block_means.append(block_mean)
        if len(block_means) < 2:
            continue
        previous = block_means[-2]
        last_difference = abs(block_mean - previous)
        last_threshold = max(
            float(absolute_tolerance_a),
            float(relative_tolerance) * max(abs(previous), abs(block_mean)),
        )
        if last_difference <= last_threshold:
            stable_pairs += 1
        else:
            stable_pairs = 0
        if stable_pairs >= required_stable_pairs:
            return ConvergenceReport(
                converged=True,
                blocks_used=block_number,
                stable_pairs=stable_pairs,
                last_difference_a=last_difference,
                last_threshold_a=last_threshold,
                block_means_a=tuple(block_means),
            )
    return ConvergenceReport(
        converged=False,
        blocks_used=len(block_means),
        stable_pairs=stable_pairs,
        last_difference_a=last_difference,
        last_threshold_a=last_threshold,
        block_means_a=tuple(block_means),
    )


def assign_quality(gates: Mapping[str, bool]) -> str:
    """Give FAIL when one or more gates are not satisfactory."""
    return PREVIEW_QUALITY if gates and all(bool(value) for value in gates.values()) else FAIL_QUALITY


def _combine_estimates(
    means: Sequence[float],
    standard_errors: Sequence[float],
    confidence_level: float,
) -> dict[str, float | int]:
    values = np.asarray(means, dtype=float)
    errors = np.asarray(standard_errors, dtype=float)
    if values.size < 1 or values.size != errors.size:
        raise ValueError("Give equal nonempty estimate and error arrays.")
    mean = float(np.mean(values))
    between_error = (
        float(np.std(values, ddof=1) / math.sqrt(values.size))
        if values.size > 1
        else 0.0
    )
    within_error = float(math.sqrt(float(np.sum(errors * errors))) / values.size)
    combined_error = math.sqrt(between_error * between_error + within_error * within_error)
    multiplier = NormalDist().inv_cdf(0.5 + confidence_level / 2.0)
    margin = multiplier * combined_error
    return {
        "mean_a": mean,
        "standard_error_a": combined_error,
        "confidence_low_a": mean - margin,
        "confidence_high_a": mean + margin,
        "replicate_count": int(values.size),
    }


def make_curve_summary(
    curve_records: Iterable[Mapping[str, Any]],
    confidence_level: float,
) -> list[dict[str, Any]]:
    """Calculate replicate statistics for each voltage and direction."""
    grouped: dict[tuple[str, float], list[Mapping[str, Any]]] = {}
    for record in curve_records:
        key = (str(record["scan_direction"]), float(record["bias_v"]))
        grouped.setdefault(key, []).append(record)
    summary: list[dict[str, Any]] = []
    for (direction, bias), records in sorted(
        grouped.items(),
        key=lambda item: (_DIRECTION_TAGS[item[0][0]], item[0][1]),
    ):
        combined = _combine_estimates(
            [float(record["total_current_a"]) for record in records],
            [float(record["current_sem_a"]) for record in records],
            confidence_level,
        )
        summary.append(
            {
                "scan_direction": direction,
                "bias_v": bias,
                **combined,
            }
        )
    return summary


def make_hysteresis_report(
    curve_summary: Sequence[Mapping[str, Any]],
    confidence_level: float,
) -> dict[str, Any]:
    """Calculate the current difference between the two scan directions."""
    by_key = {
        (str(record["scan_direction"]), float(record["bias_v"])): record
        for record in curve_summary
    }
    biases = sorted(
        {
            bias
            for direction, bias in by_key
            if ("up", bias) in by_key and ("down", bias) in by_key
        }
    )
    multiplier = NormalDist().inv_cdf(0.5 + confidence_level / 2.0)
    points: list[dict[str, Any]] = []
    for bias in biases:
        up = by_key[("up", bias)]
        down = by_key[("down", bias)]
        difference = float(up["mean_a"]) - float(down["mean_a"])
        standard_error = math.hypot(
            float(up["standard_error_a"]),
            float(down["standard_error_a"]),
        )
        margin = multiplier * standard_error
        low = difference - margin
        high = difference + margin
        points.append(
            {
                "bias_v": bias,
                "up_minus_down_a": difference,
                "standard_error_a": standard_error,
                "confidence_low_a": low,
                "confidence_high_a": high,
                "confidence_excludes_zero": bool(low > 0.0 or high < 0.0),
            }
        )

    loop_area = 0.0
    for left, right in zip(points, points[1:]):
        voltage_width = float(right["bias_v"]) - float(left["bias_v"])
        left_difference = abs(float(left["up_minus_down_a"]))
        right_difference = abs(float(right["up_minus_down_a"]))
        loop_area += 0.5 * (left_difference + right_difference) * voltage_width
    return {
        "definition": "up current minus down current at equal bias",
        "loop_area_abs_a_v": loop_area,
        "maximum_absolute_difference_a": max(
            (abs(float(point["up_minus_down_a"])) for point in points),
            default=0.0,
        ),
        "points": points,
        "significant_point_count": sum(
            bool(point["confidence_excludes_zero"]) for point in points
        ),
    }


def _direction_voltages(
    voltages: Sequence[float],
    direction: str,
) -> tuple[float, ...]:
    if direction == "up":
        return tuple(float(value) for value in voltages)
    if direction == "down":
        return tuple(float(value) for value in reversed(voltages))
    raise ValueError("Set direction to 'up' or 'down'.")


def _make_run_record(
    *,
    sim: SimulationLike,
    settings: StudySettings,
    direction: str,
    pair_replicate: int,
    record_replicate: int,
    seed: int,
    run_id: str,
    manifest: Mapping[str, Any],
    gates: Mapping[str, bool],
    sampling_steps: int,
    minimum_sampling_steps: int,
) -> dict[str, Any]:
    config = sim.config
    versions = dict(manifest.get("versions", {}))
    return {
        "run_id": run_id,
        "config_sha256": config.fingerprint(),
        "config_json": config.canonical_json(),
        "root_seed": seed,
        "replicate": record_replicate,
        "ne_m3": config.N0,
        "te_ev": config.Te,
        "ti_ev": config.Ti,
        "vp_v": config.V_WALL,
        "pressure_pa": config.pressure_pa,
        "gas": config.CROSS_SECTION_TARGET,
        "probe_radius_m": config.R_MIN,
        "probe_length_m": settings.probe_length_m,
        "n_cells": config.N_CELLS,
        "dt_s": config.DT,
        "n_particles": settings.n_particles,
        "cross_section_hashes_json": canonical_json(
            manifest.get("source_files", {})
        ),
        "stability_pass": bool(gates["stability"]),
        "convergence_pass": bool(gates["convergence"]),
        "quality_status": assign_quality(gates),
        "software_version": str(versions.get("software_version", "0+uninstalled")),
        "physics_model_version": str(
            versions.get("physics_model_version", PHYSICS_MODEL_VERSION)
        ),
        "data_schema_version": DATASET_SCHEMA_VERSION,
        "scan_direction": direction,
        "pair_replicate": pair_replicate,
        "study_root_seed": settings.root_seed,
        "sampling_duration_pass": bool(gates["sampling_duration"]),
        "sampling_steps": sampling_steps,
        "minimum_sampling_steps": minimum_sampling_steps,
    }


def execute_direction(
    config: Config,
    settings: StudySettings,
    seed_record: Mapping[str, Any],
    sampling_steps: int,
    minimum_sampling_steps: int,
    *,
    simulation_factory: SimulationFactory = PICSimulation,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    """Execute one independent directional replicate."""
    direction = str(seed_record["direction"])
    pair_replicate = int(seed_record["pair_replicate"])
    seed = int(seed_record["seed"])
    record_replicate = pair_replicate * 2 + _DIRECTION_TAGS[direction]
    voltages = _direction_voltages(settings.voltages, direction)
    sim = simulation_factory(
        config,
        n_particles=settings.n_particles,
        v_bias=voltages[0],
        probe_length=settings.probe_length_m,
        sigma_cex=settings.sigma_cex_m2,
        seed=seed,
    )
    run_id = make_run_id(config.fingerprint(), seed, record_replicate)
    curve_records: list[dict[str, Any]] = []
    convergence_reports: list[dict[str, Any]] = []
    previous_bias = voltages[0]

    for index, bias in enumerate(voltages):
        if index > 0:
            ramp_bias(sim, previous_bias, bias, settings.ramp_steps)
        report = stabilize_bias(
            sim,
            bias,
            settings.minimum_burn_steps,
            settings.convergence_block_steps,
            settings.maximum_convergence_blocks,
            settings.convergence_relative_tolerance,
            settings.convergence_absolute_tolerance_a,
            settings.required_stable_pairs,
        )
        electron, ion, total = sample_current_series(sim, sampling_steps)
        electron_summary = summarize_current(electron, settings.confidence_level)
        ion_summary = summarize_current(ion, settings.confidence_level)
        total_summary = summarize_current(total, settings.confidence_level)
        curve_records.append(
            {
                "run_id": run_id,
                "bias_v": bias,
                "replicate": record_replicate,
                "electron_current_a": electron_summary.mean_a,
                "ion_current_a": ion_summary.mean_a,
                "total_current_a": total_summary.mean_a,
                "conventional_current_a": -total_summary.mean_a,
                "current_sem_a": total_summary.standard_error_a,
                "sample_count": total_summary.sample_count,
                "converged": report.converged,
                "scan_direction": direction,
                "pair_replicate": pair_replicate,
                "batch_mean_count": total_summary.batch_mean_count,
                "sample_independence_method": "nonoverlapping_batch_means",
                "confidence_level": settings.confidence_level,
                "confidence_low_a": total_summary.confidence_low_a,
                "confidence_high_a": total_summary.confidence_high_a,
                "electron_sem_a": electron_summary.standard_error_a,
                "ion_sem_a": ion_summary.standard_error_a,
                "sampling_duration_s": sampling_steps * config.DT,
                "ion_transit_time_s": calculate_ion_transit_time(config),
                "ion_transit_multiples": (
                    sampling_steps * config.DT / calculate_ion_transit_time(config)
                ),
                "convergence_blocks": report.blocks_used,
                "convergence_last_difference_a": report.last_difference_a,
                "convergence_threshold_a": report.last_threshold_a,
            }
        )
        convergence_reports.append(
            {
                "bias_v": bias,
                **asdict(report),
            }
        )
        previous_bias = bias

    sim_manifest = sim.run_manifest()
    gates = {
        "stability": not config.stability_warnings(),
        "sampling_duration": sampling_steps >= minimum_sampling_steps,
        "convergence": all(
            bool(record["converged"]) for record in curve_records
        ),
        "simulation_state": (
            str(sim_manifest.get("simulation", {}).get("status", "READY"))
            != "FAILED"
        ),
        "particle_ledger": (
            sim_manifest.get("simulation", {})
            .get("numerical_diagnostics", {})
            .get("electron_particle_ledger_residual", 0)
            == 0
            and sim_manifest.get("simulation", {})
            .get("numerical_diagnostics", {})
            .get("ion_particle_ledger_residual", 0)
            == 0
        ),
        "cross_section_range": (
            sim_manifest.get("simulation", {})
            .get("numerical_diagnostics", {})
            .get("energy_table_overflow_lookups", 0)
            == 0
        ),
        "runtime_limits": not bool(
            sim_manifest.get("simulation", {})
            .get("numerical_diagnostics", {})
            .get("runtime_warnings", [])
        ),
    }
    run_record = _make_run_record(
        sim=sim,
        settings=settings,
        direction=direction,
        pair_replicate=pair_replicate,
        record_replicate=record_replicate,
        seed=seed,
        run_id=run_id,
        manifest=sim_manifest,
        gates=gates,
        sampling_steps=sampling_steps,
        minimum_sampling_steps=minimum_sampling_steps,
    )
    trace = {
        "run_id": run_id,
        "direction": direction,
        "pair_replicate": pair_replicate,
        "seed": seed,
        "gates": gates,
        "quality_status": run_record["quality_status"],
        "convergence": convergence_reports,
        "simulation_manifest": sim_manifest,
    }
    return run_record, curve_records, trace


def execute_study(
    config: Config,
    settings: StudySettings,
    output_dir: str | Path,
    *,
    simulation_factory: SimulationFactory = PICSimulation,
) -> dict[str, Path]:
    """Execute the study and write its traceable dataset."""
    if config.is_production:
        raise ValueError("Use a research configuration for this preview entry point.")
    sampling_steps, minimum_sampling_steps = resolve_sampling_steps(config, settings)
    seed_plan = make_seed_plan(settings.root_seed, settings.repeats)
    run_records: list[dict[str, Any]] = []
    curve_records: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    for seed_record in seed_plan:
        run_record, directional_curves, trace = execute_direction(
            config,
            settings,
            seed_record,
            sampling_steps,
            minimum_sampling_steps,
            simulation_factory=simulation_factory,
        )
        run_records.append(run_record)
        curve_records.extend(directional_curves)
        traces.append(trace)

    curve_summary = make_curve_summary(curve_records, settings.confidence_level)
    hysteresis = make_hysteresis_report(curve_summary, settings.confidence_level)
    overall_gates = {
        "all_runs_satisfactory": all(
            record["quality_status"] == PREVIEW_QUALITY for record in run_records
        ),
        "all_directions_present": {
            str(record["scan_direction"]) for record in run_records
        }
        == set(_DIRECTION_TAGS),
        "independent_seeds": len({record["root_seed"] for record in run_records})
        == len(run_records),
    }
    manifest = {
        "artifact_type": "research_preview_iv_study",
        "created_utc": datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
        "quality_status": assign_quality(overall_gates),
        "quality_statement": (
            "This dataset is a research preview. "
            "External physical comparison is necessary before diagnostic use."
        ),
        "study_root_seed": settings.root_seed,
        "seed_plan": seed_plan,
        "settings": asdict(settings),
        "sampling": {
            "sample_independence_method": "nonoverlapping_batch_means",
            "independence_statement": (
                "Batch means give a conservative error estimate. "
                "They do not prove sample independence."
            ),
            "sampling_steps": sampling_steps,
            "minimum_sampling_steps": minimum_sampling_steps,
            "ion_transit_time_s": calculate_ion_transit_time(config),
            "requested_ion_transit_multiples": settings.ion_transit_multiples,
            "actual_ion_transit_multiples": (
                sampling_steps * config.DT / calculate_ion_transit_time(config)
            ),
        },
        "gates": overall_gates,
        "curve_summary": curve_summary,
        "hysteresis": hysteresis,
        "runs": traces,
    }
    return write_dataset(
        output_dir,
        run_records,
        curve_records,
        manifest,
        overwrite=False,
    )


def make_default_preview_config() -> Config:
    """Give the default argon research-preview configuration."""
    return Config.research(
        N0=1.0e16,
        Te=3.0,
        Ti=0.03,
        P_Torr=0.3 / 133.322368,
        CROSS_SECTION_TARGET="Ar",
        R_MIN=0.2e-3,
        R_MAX=2.0e-3,
        N_CELLS=400,
        DT=2.0e-12,
        LXCAT_ELECTRON_FILE=None,
        LXCAT_ION_FILE=None,
        ENABLE_IONIZATION_SECONDARIES=True,
        SMOOTH_DENSITY=True,
        N_SMOOTHING_PASSES=3,
    )


def build_parser() -> argparse.ArgumentParser:
    """Give the command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Make a research-preview I-V dataset. "
            "The program keeps this file name for compatibility."
        )
    )
    parser.add_argument(
        "--output-dir",
        "--output",
        dest="output_dir",
        type=Path,
        default=None,
        help="Set the output directory. Existing dataset files stop the operation.",
    )
    parser.add_argument(
        "--root-seed",
        type=int,
        default=20260729,
        help="Set the traceable study seed.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Set the number of independent replicates in each direction.",
    )
    parser.add_argument(
        "--particles",
        type=int,
        default=4000,
        help="Set the nominal particle count.",
    )
    parser.add_argument(
        "--v-start",
        type=float,
        default=-30.0,
        help="Set the minimum probe bias in volts.",
    )
    parser.add_argument(
        "--v-end",
        type=float,
        default=30.0,
        help="Set the maximum probe bias in volts.",
    )
    parser.add_argument(
        "--points",
        type=int,
        default=9,
        help="Set the number of voltage points.",
    )
    parser.add_argument(
        "--sampling-steps",
        type=int,
        default=None,
        help=(
            "Set the sampling steps. "
            "The program calculates the minimum when this option is absent."
        ),
    )
    parser.add_argument(
        "--ion-transit-multiples",
        type=float,
        default=DEFAULT_ION_TRANSIT_MULTIPLES,
        help="Set the minimum sampling time in ion transit times.",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Show the sampling and seed plan without simulation.",
    )
    return parser


def _settings_from_arguments(arguments: argparse.Namespace) -> StudySettings:
    if arguments.points < 2:
        raise ValueError("Set points to an integer not less than 2.")
    voltages = tuple(
        float(value)
        for value in np.linspace(arguments.v_start, arguments.v_end, arguments.points)
    )
    return StudySettings(
        voltages=voltages,
        repeats=arguments.repeats,
        root_seed=arguments.root_seed,
        n_particles=arguments.particles,
        sampling_steps=arguments.sampling_steps,
        ion_transit_multiples=arguments.ion_transit_multiples,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Operate the research-preview batch entry point."""
    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        config = make_default_preview_config()
        settings = _settings_from_arguments(arguments)
        sampling_steps, minimum_steps = resolve_sampling_steps(config, settings)
        seed_plan = make_seed_plan(settings.root_seed, settings.repeats)
        output_dir = arguments.output_dir
        print("This operation makes a research-preview dataset.")
        print(
            json.dumps(
                {
                    "output_directory": (
                        str(output_dir) if output_dir is not None else None
                    ),
                    "sampling_steps": sampling_steps,
                    "minimum_sampling_steps": minimum_steps,
                    "ion_transit_multiples": (
                        sampling_steps
                        * config.DT
                        / calculate_ion_transit_time(config)
                    ),
                    "seed_plan": seed_plan,
                },
                indent=2,
                sort_keys=True,
            )
        )
        if arguments.plan_only:
            return 0
        if output_dir is None:
            raise ValueError("Set --output-dir, or use --plan-only.")
        paths = execute_study(config, settings, output_dir)
    except (FileExistsError, RuntimeError, TypeError, ValueError) as error:
        parser.error(str(error))
    print("The program wrote these dataset files:")
    for name, path in sorted(paths.items()):
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
