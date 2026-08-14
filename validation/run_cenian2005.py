from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import subprocess
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np

from core.config import Config
from core.cross_sections import (
    build_electron_tables_from_lxcat,
    build_ion_tables_from_lxcat,
)
from core.lxcat_parser import parse_cross_section_file
from core.provenance import (
    json_sha256,
    sha256_bytes,
    sha256_file,
    version_record,
)
from core.simulation import PICSimulation


QUALITY_STATUS = "PREVIEW"
EXPERIMENT_DOI = "10.1063/1.1938275"
EXPERIMENT_FIGURE = "Figure 2"
PROBE_RADIUS_M = 313.0e-6
PROBE_LENGTH_M = 0.047
MINIMUM_DOMAIN_RADIUS_M = 0.012
EXPECTED_VOLTAGES_V = tuple(float(value) for value in range(-60, -9, 5))
DEFAULT_EXPERIMENTAL_CSV = (
    Path(__file__).resolve().parent
    / "experimental"
    / "cenian2005_fig2_case_rp_lambda_0p26.csv"
)
DEFAULT_CROSS_SECTION_MANIFEST = (
    Path(__file__).resolve().parent
    / "cross_sections"
    / "lxcat_manifest.json"
)
DEFAULT_EXPERIMENTAL_MANIFEST = (
    Path(__file__).resolve().parent
    / "experimental"
    / "dataset_manifest.json"
)
PRIMARY_ELECTRON_ROLE = "electron_argon_primary"
PRIMARY_ION_ROLE = "argon_ion_in_argon"


@dataclass(frozen=True)
class ExperimentalPoint:
    """Keep one digitized experimental point."""

    voltage_v: float
    current_a: float
    digitization_uncertainty_a: float
    source_doi: str
    source_figure: str


@dataclass(frozen=True)
class PilotSettings:
    """Keep the numerical controls for one pilot study."""

    cells: int = 64
    particles: int = 1024
    dt_s: float = 3.0e-11
    domain_radius_m: float = MINIMUM_DOMAIN_RADIUS_M
    seeds: tuple[int, ...] = (20260814,)
    sample_steps: int = 31_708
    warmup_steps: int = 31_708

    def __post_init__(self) -> None:
        for name, value, minimum in (
            ("cells", self.cells, 2),
            ("particles", self.particles, 1),
            ("sample_steps", self.sample_steps, 2),
            ("warmup_steps", self.warmup_steps, 0),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"Set {name} to an integer.")
            if value < minimum:
                raise ValueError(
                    f"Set {name} to an integer not less than {minimum}."
                )
        if not math.isfinite(self.dt_s) or self.dt_s <= 0.0:
            raise ValueError("Set dt_s to a value greater than zero.")
        if (
            not math.isfinite(self.domain_radius_m)
            or self.domain_radius_m < MINIMUM_DOMAIN_RADIUS_M
        ):
            raise ValueError(
                "Set domain_radius_m to at least 0.012 m. "
                "The paper reports an approximately 0.008 m sheath."
            )
        if not self.seeds:
            raise ValueError("Give at least one seed.")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("Give each seed one time.")
        for seed in self.seeds:
            if isinstance(seed, bool) or not isinstance(seed, int):
                raise TypeError("Set each seed to an integer.")
            if seed < 0 or seed >= 1 << 64:
                raise ValueError("Set each seed from 0 through 2^64 - 1.")


class SimulationResultLike(Protocol):
    avg_current: float
    current_sem: float
    sample_count: int
    batch_mean_count: int


class SimulationLike(Protocol):
    def run(self, n_steps: int, n_warmup: int) -> SimulationResultLike:
        ...

    def run_manifest(self) -> dict[str, Any]:
        ...


SimulationFactory = Callable[..., SimulationLike]


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


def _finite_float(name: str, value: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"The experimental {name} is not numeric.") from error
    if not math.isfinite(number):
        raise ValueError(f"The experimental {name} is not finite.")
    return number


def source_tree_record() -> dict[str, Any]:
    """Give the Git source state without making Git a runtime requirement."""
    source_root = Path(__file__).resolve().parents[1]
    try:
        commit_result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=source_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5.0,
        )
        status_result = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=source_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5.0,
        )
    except (OSError, subprocess.SubprocessError):
        return {
            "commit": None,
            "dirty": None,
            "status": "git_unavailable",
            "status_sha256": None,
        }
    status_text = status_result.stdout
    return {
        "commit": commit_result.stdout.strip(),
        "dirty": bool(status_text.strip()),
        "status": "available",
        "status_sha256": sha256_bytes(status_text.encode("utf-8")),
    }


def require_experimental_dataset_file(
    path: str | Path,
    manifest_path: str | Path = DEFAULT_EXPERIMENTAL_MANIFEST,
) -> tuple[Path, Path, Path]:
    """Require the fixed experimental CSV and its provenance record."""
    csv_path = Path(path).expanduser().resolve()
    provenance_path = csv_path.with_suffix(".provenance.json")
    manifest_file = Path(manifest_path).expanduser().resolve()
    if not manifest_file.is_file():
        raise FileNotFoundError(
            f"The experimental manifest was not found: {manifest_file}."
        )
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    files = manifest.get("files", {})
    for role, file_path in (
        ("csv", csv_path),
        ("provenance", provenance_path),
    ):
        if not file_path.is_file():
            raise FileNotFoundError(
                f"The experimental {role} file was not found: {file_path}."
            )
        record = files.get(role)
        if not isinstance(record, Mapping):
            raise ValueError(
                f"The experimental manifest has no {role} record."
            )
        if sha256_file(file_path).casefold() != str(
            record.get("sha256", "")
        ).casefold():
            raise ValueError(
                f"The experimental {role} hash does not match "
                "dataset_manifest.json."
            )
    return csv_path, provenance_path, manifest_file


def load_experimental_points(
    path: str | Path = DEFAULT_EXPERIMENTAL_CSV,
) -> tuple[ExperimentalPoint, ...]:
    """Read and examine the selected Cenian Figure 2 data."""
    source_path = Path(path).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(
            f"The experimental CSV was not found: {source_path}."
        )
    required_columns = {
        "voltage_V",
        "current_A",
        "digitization_uncertainty_A",
        "source_doi",
        "source_figure",
    }
    with source_path.open("r", encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None or not required_columns.issubset(
            reader.fieldnames
        ):
            raise ValueError(
                "The experimental CSV does not have all necessary columns."
            )
        points: list[ExperimentalPoint] = []
        for row in reader:
            point = ExperimentalPoint(
                voltage_v=_finite_float("voltage", row["voltage_V"]),
                current_a=_finite_float("current", row["current_A"]),
                digitization_uncertainty_a=_finite_float(
                    "digitization uncertainty",
                    row["digitization_uncertainty_A"],
                ),
                source_doi=str(row["source_doi"]).strip(),
                source_figure=str(row["source_figure"]).strip(),
            )
            if point.digitization_uncertainty_a <= 0.0:
                raise ValueError(
                    "The digitization uncertainty must be greater than zero."
                )
            if point.source_doi != EXPERIMENT_DOI:
                raise ValueError("The experimental DOI is not the selected DOI.")
            if point.source_figure != EXPERIMENT_FIGURE:
                raise ValueError(
                    "The experimental figure is not the selected figure."
                )
            points.append(point)
    voltages = tuple(point.voltage_v for point in points)
    if voltages != EXPECTED_VOLTAGES_V:
        raise ValueError(
            "The experimental voltages must be -60 V through -10 V "
            "in 5 V increments."
        )
    if any(point.current_a >= 0.0 for point in points):
        raise ValueError(
            "The selected negative-bias experimental current must be negative."
        )
    return tuple(points)


def select_experimental_points(
    points: Sequence[ExperimentalPoint],
    voltages: Sequence[float] | None,
) -> tuple[ExperimentalPoint, ...]:
    """Select a traceable voltage subset while keeping source order."""
    if voltages is None:
        return tuple(points)
    requested = tuple(float(value) for value in voltages)
    if not requested:
        raise ValueError("Give at least one voltage.")
    if len(set(requested)) != len(requested):
        raise ValueError("Give each selected voltage one time.")
    available = {point.voltage_v: point for point in points}
    missing = [value for value in requested if value not in available]
    if missing:
        raise ValueError(
            "Selected voltages are not in the fixed experiment grid: "
            + ", ".join(f"{value:g}" for value in missing)
            + "."
        )
    requested_set = set(requested)
    return tuple(
        point for point in points if point.voltage_v in requested_set
    )


def require_lxcat_database_file(
    path: str | Path,
    database_name: str,
) -> Path:
    """Make sure that a local LXCat file identifies the selected database."""
    file_path = Path(path).expanduser().resolve()
    if not file_path.exists():
        raise FileNotFoundError(f"The LXCat file was not found: {file_path}.")
    if not file_path.is_file():
        raise IsADirectoryError(
            f"The LXCat path does not identify a file: {file_path}."
        )
    if file_path.stat().st_size == 0:
        raise ValueError(f"The LXCat file is empty: {file_path}.")
    text = file_path.read_text(encoding="utf-8")
    database_pattern = re.compile(
        rf"(?:DATABASE\s*:\s*[^\r\n]*\b{re.escape(database_name)}\b|"
        rf"PERMLINK\s*:\s*[^\r\n]*/{re.escape(database_name)}\b)",
        flags=re.IGNORECASE,
    )
    if database_pattern.search(text) is None:
        raise ValueError(
            f"The LXCat file does not identify the {database_name} database: "
            f"{file_path}."
        )
    return file_path


def require_lxcat_dataset_file(
    path: str | Path,
    role: str,
    manifest_path: str | Path = DEFAULT_CROSS_SECTION_MANIFEST,
) -> Path:
    """Require the exact LXCat retrieval recorded for this validation."""
    manifest_file = Path(manifest_path).expanduser().resolve()
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    datasets = {
        str(item["role"]): item for item in manifest.get("datasets", [])
    }
    if role not in datasets:
        raise ValueError(f"The LXCat manifest has no {role} dataset.")
    selected = datasets[role]
    file_path = require_lxcat_database_file(path, str(selected["database"]))
    actual_hash = sha256_file(file_path)
    expected_hash = str(selected["sha256"]).casefold()
    if actual_hash.casefold() != expected_hash:
        raise ValueError(
            f"The {role} file hash does not match lxcat_manifest.json."
        )
    processes = parse_cross_section_file(file_path, strict=True)
    process_counts = Counter(
        process.process_type for process in processes
    )
    if dict(process_counts) != selected["process_counts"]:
        raise ValueError(
            f"The {role} process counts do not match lxcat_manifest.json."
        )
    if role == PRIMARY_ELECTRON_ROLE:
        build_electron_tables_from_lxcat(
            file_path,
            target="Ar",
            e_max=100.0,
            n_bins=2001,
            strict=True,
        )
    elif role == PRIMARY_ION_ROLE:
        build_ion_tables_from_lxcat(
            file_path,
            target="Ar",
            e_max=100.0,
            n_bins=2001,
            strict=True,
            confirm_symmetric_backscatter_as_cex=True,
            ion_species="Ar+",
        )
    return file_path


def accelerated_electron_cfl(config: Config) -> float:
    """Calculate a conservative electron CFL value at -60 V."""
    maximum_energy_ev = config.Te + abs(min(EXPECTED_VOLTAGES_V))
    maximum_speed = math.sqrt(
        2.0 * config.e * maximum_energy_ev / config.m_e
    )
    return maximum_speed * config.DT / config.dr


def make_cenian_config(
    settings: PilotSettings,
    electron_lxcat_file: str | Path,
    ion_lxcat_file: str | Path,
) -> Config:
    """Make the research configuration for the selected experiment."""
    electron_file = require_lxcat_dataset_file(
        electron_lxcat_file,
        PRIMARY_ELECTRON_ROLE,
    )
    ion_file = require_lxcat_dataset_file(
        ion_lxcat_file,
        PRIMARY_ION_ROLE,
    )
    config = Config.research(
        N0=7.15e13,
        Te=1.9,
        Ti=0.025,
        P_Torr=0.0013,
        T_GAS_K=290.1,
        R_MIN=PROBE_RADIUS_M,
        R_MAX=settings.domain_radius_m,
        V_WALL=0.0,
        N_CELLS=settings.cells,
        DT=settings.dt_s,
        CROSS_SECTION_TARGET="Ar",
        NEUTRAL_SPECIES="Ar",
        ION_SPECIES="Ar+",
        LXCAT_ELECTRON_FILE=str(electron_file),
        LXCAT_ION_FILE=str(ion_file),
        CROSS_SECTION_STRICT=True,
        CONFIRM_SYMMETRIC_BACKSCATTER_AS_CEX=True,
        EN_CS_E_MAX=100.0,
        EN_CS_N=2001,
        ION_CS_E_MAX=100.0,
        ION_CS_N=2001,
        ENABLE_IONIZATION_SECONDARIES=True,
        ENABLE_ION_NEUTRAL_ELASTIC=True,
        ENABLE_COULOMB_COLLISIONS=False,
        ION_INJECTION_BOHM=False,
        SMOOTH_DENSITY=False,
    )
    config.require_stable()
    cfl = accelerated_electron_cfl(config)
    if cfl > config.MAX_CFL:
        raise ValueError(
            "Decrease dt_s or increase cells. "
            f"The accelerated-electron CFL value is {cfl:.6g}."
        )
    return config


def calculate_ion_transit_time(config: Config) -> float:
    """Calculate one ion transit time with the selected boundary speed."""
    energy_ev = config.Te if config.ION_INJECTION_BOHM else config.Ti
    ion_speed = math.sqrt(config.e * energy_ev / config.m_i)
    return (config.R_MAX - config.R_MIN) / ion_speed


def make_validation_plan(
    config: Config,
    settings: PilotSettings,
    experimental_points: Sequence[ExperimentalPoint],
    experimental_csv: str | Path,
) -> dict[str, Any]:
    """Give the traceable plan without simulation."""
    csv_path, provenance_path, experimental_manifest_path = (
        require_experimental_dataset_file(experimental_csv)
    )
    transit_time = calculate_ion_transit_time(config)
    source_tree = source_tree_record()
    published_config = config.canonical_dict()
    for field in ("LXCAT_ELECTRON_FILE", "LXCAT_ION_FILE"):
        if published_config.get(field):
            published_config[field] = Path(str(published_config[field])).name
    plan: dict[str, Any] = {
        "artifact_type": "cenian2005_external_validation_plan",
        "created_utc": _utc_now(),
        "quality_status": QUALITY_STATUS,
        "quality_statement": (
            "This operation is a research preview. "
            "It does not give production release evidence."
        ),
        "absolute_current_no_scaling": True,
        "signed_current_no_scaling": True,
        "current_definition": (
            "Use PICSimulation.avg_current: signed I_electron_magnitude - "
            "I_ion_magnitude in amperes for the 0.047 m probe. "
            "Do not use avg_conventional_current."
        ),
        "experiment": {
            "doi": EXPERIMENT_DOI,
            "figure": EXPERIMENT_FIGURE,
            "point_count": len(experimental_points),
            "probe_length_m": PROBE_LENGTH_M,
            "probe_radius_m": PROBE_RADIUS_M,
            "voltage_reference": "plasma_potential",
            "boundary_mapping": (
                "Maxwellian thermal plasma-bulk influx; "
                "ION_INJECTION_BOHM=False."
            ),
            "probe_surface_mapping": "Fully absorbing probe surface.",
            "geometry_mapping": (
                "Infinite cylindrical model multiplied by the 0.047 m "
                "measured length; probe-end current is not modeled."
            ),
            "voltages_V": [
                point.voltage_v for point in experimental_points
            ],
        },
        "input_files": {
            "experimental_csv": {
                "name": csv_path.name,
                "sha256": sha256_file(csv_path),
            },
            "experimental_provenance": {
                "name": provenance_path.name,
                "sha256": sha256_file(provenance_path),
            },
            "experimental_manifest": {
                "name": experimental_manifest_path.name,
                "sha256": sha256_file(experimental_manifest_path),
            },
            "lxcat_manifest": {
                "name": DEFAULT_CROSS_SECTION_MANIFEST.name,
                "sha256": sha256_file(DEFAULT_CROSS_SECTION_MANIFEST),
            },
            "lxcat_phelps_electron": {
                "name": Path(config.LXCAT_ELECTRON_FILE or "").name,
                "sha256": sha256_file(config.LXCAT_ELECTRON_FILE or ""),
            },
            "lxcat_phelps_ion": {
                "name": Path(config.LXCAT_ION_FILE or "").name,
                "sha256": sha256_file(config.LXCAT_ION_FILE or ""),
            },
        },
        "source_tree": source_tree,
        "config": published_config,
        "config_paths_redacted": True,
        "config_sha256": config.fingerprint(),
        "settings": asdict(settings),
        "numerical_checks": {
            "config_stability": config.stability_metrics_dict(),
            "accelerated_electron_cfl": accelerated_electron_cfl(config),
            "ion_transit_time_s": transit_time,
            "sample_duration_s": settings.sample_steps * config.DT,
            "sample_ion_transit_multiples": (
                settings.sample_steps * config.DT / transit_time
            ),
        },
        "known_limits": [
            "The experimental points came from one figure digitization that has not been independently reviewed.",
            "The experimental density and temperature came from the same probe curve.",
            "The pilot sample time is not a production acceptance time.",
            "The model uses an infinite cylinder multiplied by the measured probe length.",
            "The source reports significant finite-length effects above 50 V magnitude; this affects the -55 V and -60 V points.",
            "The 0.012 m pilot domain is smaller than the source model domains, which reached about 0.067 m at -50 V.",
            "The 2026 LXCat Phelps retrieval has not been shown to be numerically identical to the 1997 table used by the source.",
            "The experiment does not give point-by-point system uncertainty.",
        ],
    }
    plan["plan_sha256"] = json_sha256(plan)
    return plan


def _simulation_manifest_summary(simulation: SimulationLike) -> dict[str, Any]:
    manifest = simulation.run_manifest()
    simulation_data = manifest.get("simulation", {})
    if not isinstance(simulation_data, Mapping):
        simulation_data = {}
    diagnostics = simulation_data.get("numerical_diagnostics", {})
    if not isinstance(diagnostics, Mapping):
        diagnostics = {}
    return {
        "manifest_sha256": manifest.get("manifest_sha256"),
        "state_sha256": simulation_data.get("state_sha256"),
        "simulation_status": simulation_data.get("status", "UNKNOWN"),
        "numerical_status": diagnostics.get("status", "UNKNOWN"),
        "physics_limitations": simulation_data.get("physics_limitations", []),
        "stability_warnings": diagnostics.get("stability_warnings", []),
        "runtime_warnings": diagnostics.get("runtime_warnings", []),
        "energy_table_overflow_lookups": diagnostics.get(
            "energy_table_overflow_lookups"
        ),
        "electron_particle_ledger_residual": diagnostics.get(
            "electron_particle_ledger_residual"
        ),
        "ion_particle_ledger_residual": diagnostics.get(
            "ion_particle_ledger_residual"
        ),
    }


def _simulation_records(
    config: Config,
    settings: PilotSettings,
    experimental_points: Sequence[ExperimentalPoint],
    simulation_factory: SimulationFactory,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    trace: list[dict[str, Any]] = []
    total_steps = settings.warmup_steps + settings.sample_steps
    for point in experimental_points:
        for seed in settings.seeds:
            simulation = simulation_factory(
                config,
                n_particles=settings.particles,
                v_bias=point.voltage_v,
                sigma_cex=0.0,
                probe_length=PROBE_LENGTH_M,
                seed=seed,
            )
            result = simulation.run(
                n_steps=total_steps,
                n_warmup=settings.warmup_steps,
            )
            current = float(result.avg_current)
            current_sem = float(result.current_sem)
            if not math.isfinite(current):
                raise RuntimeError("The simulation current is not finite.")
            if not math.isfinite(current_sem) or current_sem < 0.0:
                raise RuntimeError(
                    "The simulation current standard error is not valid."
                )
            summary = _simulation_manifest_summary(simulation)
            if summary["simulation_status"] != "READY":
                raise RuntimeError(
                    "The simulation manifest must have READY status."
                )
            if summary["numerical_status"] != "PASS":
                raise RuntimeError(
                    "The numerical diagnostics must have PASS status."
                )
            if summary["energy_table_overflow_lookups"] != 0:
                raise RuntimeError("The energy-table overflow count is not zero.")
            if summary["electron_particle_ledger_residual"] != 0:
                raise RuntimeError("The electron particle ledger is not balanced.")
            if summary["ion_particle_ledger_residual"] != 0:
                raise RuntimeError("The ion particle ledger is not balanced.")
            if summary["stability_warnings"]:
                raise RuntimeError("The simulation has stability warnings.")
            if summary["runtime_warnings"]:
                raise RuntimeError("The simulation has runtime warnings.")
            records.append(
                {
                    "voltage_V": point.voltage_v,
                    "seed": seed,
                    "source_aligned_current_A": current,
                    "current_sem_A": current_sem,
                    "sample_steps": settings.sample_steps,
                    "warmup_steps": settings.warmup_steps,
                    "sample_count": int(result.sample_count),
                    "batch_mean_count": int(result.batch_mean_count),
                    "numerical_status": summary["numerical_status"],
                    "simulation_manifest_sha256": summary[
                        "manifest_sha256"
                    ],
                }
            )
            trace.append(
                {
                    "voltage_V": point.voltage_v,
                    "seed": seed,
                    **summary,
                }
            )
    return records, trace


def make_comparison_records(
    experimental_points: Sequence[ExperimentalPoint],
    simulation_points: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Compare absolute ampere values without a fitted scale factor."""
    by_voltage: dict[float, list[Mapping[str, Any]]] = {}
    for record in simulation_points:
        voltage = float(record["voltage_V"])
        by_voltage.setdefault(voltage, []).append(record)
    comparison: list[dict[str, Any]] = []
    for point in experimental_points:
        records = by_voltage.get(point.voltage_v, [])
        if not records:
            raise ValueError(
                f"No simulation current is available at {point.voltage_v:g} V."
            )
        currents = np.asarray(
            [
                float(record["source_aligned_current_A"])
                for record in records
            ],
            dtype=float,
        )
        internal_sem = np.asarray(
            [float(record["current_sem_A"]) for record in records],
            dtype=float,
        )
        count = int(currents.size)
        mean_current = float(np.mean(currents))
        between_seed_sem = (
            float(np.std(currents, ddof=1) / math.sqrt(count))
            if count > 1
            else 0.0
        )
        combined_internal_sem = float(
            math.sqrt(float(np.sum(internal_sem * internal_sem))) / count
        )
        simulation_sem = (
            between_seed_sem if count > 1 else combined_internal_sem
        )
        total_uncertainty = math.hypot(
            simulation_sem,
            point.digitization_uncertainty_a,
        )
        residual = mean_current - point.current_a
        z_score = residual / total_uncertainty
        comparison.append(
            {
                "voltage_V": point.voltage_v,
                "experimental_source_aligned_current_A": point.current_a,
                "digitization_uncertainty_A": (
                    point.digitization_uncertainty_a
                ),
                "simulation_source_aligned_current_A": mean_current,
                "simulation_standard_error_A": simulation_sem,
                "between_seed_standard_error_A": between_seed_sem,
                "combined_internal_standard_error_A": combined_internal_sem,
                "simulation_standard_error_method": (
                    "between_seed_standard_error"
                    if count > 1
                    else "single_run_internal_standard_error"
                ),
                "combined_uncertainty_A": total_uncertainty,
                "residual_A": residual,
                "absolute_residual_A": abs(residual),
                "relative_error": residual / abs(point.current_a),
                "z_score": z_score,
                "within_combined_2sigma": abs(z_score) <= 2.0,
                "seed_count": count,
                "source_doi": point.source_doi,
                "source_figure": point.source_figure,
            }
        )
    return comparison


def calculate_metrics(
    comparison: Sequence[Mapping[str, Any]],
    simulation_row_count: int,
    seed_count: int,
) -> dict[str, Any]:
    """Calculate research-preview comparison metrics."""
    if not comparison:
        raise ValueError("Give at least one comparison point.")
    residuals = np.asarray(
        [float(record["residual_A"]) for record in comparison],
        dtype=float,
    )
    experimental = np.asarray(
        [
            float(record["experimental_source_aligned_current_A"])
            for record in comparison
        ],
        dtype=float,
    )
    z_scores = np.asarray(
        [float(record["z_score"]) for record in comparison],
        dtype=float,
    )
    within = np.asarray(
        [bool(record["within_combined_2sigma"]) for record in comparison],
        dtype=bool,
    )
    rmse = float(math.sqrt(float(np.mean(residuals * residuals))))
    reference_rms = float(
        math.sqrt(float(np.mean(experimental * experimental)))
    )
    metrics: dict[str, Any] = {
        "quality_status": QUALITY_STATUS,
        "release_decision": "NOT_EVALUATED_RESEARCH_PREVIEW",
        "absolute_current_no_scaling": True,
        "signed_current_no_scaling": True,
        "experimental_point_count": len(comparison),
        "simulation_row_count": simulation_row_count,
        "seed_count": seed_count,
        "mean_bias_A": float(np.mean(residuals)),
        "mean_absolute_error_A": float(np.mean(np.abs(residuals))),
        "root_mean_square_error_A": rmse,
        "maximum_absolute_error_A": float(np.max(np.abs(residuals))),
        "normalized_root_mean_square_error": rmse / reference_rms,
        "mean_absolute_relative_error": float(
            np.mean(np.abs(residuals) / np.abs(experimental))
        ),
        "root_mean_square_z_score": float(
            math.sqrt(float(np.mean(z_scores * z_scores)))
        ),
        "points_within_combined_2sigma": int(np.count_nonzero(within)),
        "fraction_within_combined_2sigma": float(np.mean(within)),
        "uncertainty_statement": (
            "The combined value includes digitization uncertainty and "
            "simulation standard error. It does not include full experimental "
            "system uncertainty."
        ),
    }
    return metrics


def _prepare_output_directory(path: str | Path) -> Path:
    output_dir = Path(path).expanduser().resolve()
    if output_dir.exists():
        if not output_dir.is_dir():
            raise NotADirectoryError(
                f"The output path is not a directory: {output_dir}."
            )
        if any(output_dir.iterdir()):
            raise FileExistsError(
                f"The output directory is not empty: {output_dir}."
            )
    else:
        output_dir.mkdir(parents=True)
    return output_dir


def _write_csv(
    path: Path,
    records: Sequence[Mapping[str, Any]],
) -> None:
    if not records:
        raise ValueError("Give at least one CSV record.")
    fieldnames = list(records[0].keys())
    with path.open("x", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    text = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    path.write_text(text + "\n", encoding="utf-8")


def execute_validation(
    config: Config,
    settings: PilotSettings,
    experimental_points: Sequence[ExperimentalPoint],
    experimental_csv: str | Path,
    output_dir: str | Path,
    *,
    validation_plan: Mapping[str, Any] | None = None,
    simulation_factory: SimulationFactory = PICSimulation,
) -> dict[str, Path]:
    """Operate the pilot and write its comparison files."""
    if config.is_production:
        raise ValueError("Use a research configuration for this entry point.")
    target_dir = _prepare_output_directory(output_dir)
    simulation_records, run_trace = _simulation_records(
        config,
        settings,
        experimental_points,
        simulation_factory,
    )
    comparison_records = make_comparison_records(
        experimental_points,
        simulation_records,
    )
    metrics = calculate_metrics(
        comparison_records,
        simulation_row_count=len(simulation_records),
        seed_count=len(settings.seeds),
    )

    simulation_path = target_dir / "simulation_points.csv"
    comparison_path = target_dir / "comparison.csv"
    metrics_path = target_dir / "metrics.json"
    manifest_path = target_dir / "manifest.json"
    _write_csv(simulation_path, simulation_records)
    _write_csv(comparison_path, comparison_records)
    _write_json(metrics_path, metrics)

    plan = (
        dict(validation_plan)
        if validation_plan is not None
        else make_validation_plan(
            config,
            settings,
            experimental_points,
            experimental_csv,
        )
    )
    source_tree = plan.get("source_tree", source_tree_record())
    source_commit = (
        source_tree.get("commit")
        if isinstance(source_tree, Mapping)
        else None
    )
    manifest: dict[str, Any] = {
        "artifact_type": "cenian2005_external_validation_result",
        "created_utc": _utc_now(),
        "quality_status": QUALITY_STATUS,
        "quality_statement": (
            "These results are a research preview. "
            "They are not a production validation certificate."
        ),
        "absolute_current_no_scaling": True,
        "signed_current_no_scaling": True,
        "plan": plan,
        "metrics": metrics,
        "run_trace": run_trace,
        "versions": version_record(source_commit=source_commit),
        "source_tree": source_tree,
        "outputs": {
            "simulation_points.csv": sha256_file(simulation_path),
            "comparison.csv": sha256_file(comparison_path),
            "metrics.json": sha256_file(metrics_path),
        },
    }
    manifest["manifest_sha256"] = json_sha256(manifest)
    _write_json(manifest_path, manifest)
    return {
        "simulation_points": simulation_path,
        "comparison": comparison_path,
        "metrics": metrics_path,
        "manifest": manifest_path,
    }


def build_parser() -> argparse.ArgumentParser:
    """Give the command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Compare a research-preview PIC-MCC result with Cenian 2005 "
            "Figure 2."
        )
    )
    parser.add_argument(
        "--electron-lxcat",
        type=Path,
        required=True,
        help="Set the pinned local LXCat Phelps electron cross-section file.",
    )
    parser.add_argument(
        "--ion-lxcat",
        type=Path,
        required=True,
        help="Set a local LXCat Phelps ion cross-section file.",
    )
    parser.add_argument(
        "--experimental-csv",
        type=Path,
        default=DEFAULT_EXPERIMENTAL_CSV,
        help="Set the digitized Cenian Figure 2 CSV file.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--cells", type=int, default=64)
    parser.add_argument("--particles", type=int, default=1024)
    parser.add_argument("--dt-s", type=float, default=3.0e-11)
    parser.add_argument(
        "--domain-radius-m",
        type=float,
        default=MINIMUM_DOMAIN_RADIUS_M,
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[20260814],
    )
    parser.add_argument(
        "--voltages",
        type=float,
        nargs="+",
        default=None,
        help="Select a subset of the fixed experiment voltages.",
    )
    parser.add_argument("--sample-steps", type=int, default=31_708)
    parser.add_argument("--warmup-steps", type=int, default=31_708)
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Show the checked plan without simulation.",
    )
    return parser


def _settings_from_arguments(arguments: argparse.Namespace) -> PilotSettings:
    return PilotSettings(
        cells=arguments.cells,
        particles=arguments.particles,
        dt_s=arguments.dt_s,
        domain_radius_m=arguments.domain_radius_m,
        seeds=tuple(arguments.seeds),
        sample_steps=arguments.sample_steps,
        warmup_steps=arguments.warmup_steps,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Operate the Cenian research-preview validation entry point."""
    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        settings = _settings_from_arguments(arguments)
        experimental_points = select_experimental_points(
            load_experimental_points(arguments.experimental_csv),
            arguments.voltages,
        )
        config = make_cenian_config(
            settings,
            arguments.electron_lxcat,
            arguments.ion_lxcat,
        )
        plan = make_validation_plan(
            config,
            settings,
            experimental_points,
            arguments.experimental_csv,
        )
        print("This operation is a research preview.")
        print(json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True))
        if arguments.plan_only:
            return 0
        if arguments.output_dir is None:
            raise ValueError("Set --output-dir, or use --plan-only.")
        paths = execute_validation(
            config,
            settings,
            experimental_points,
            arguments.experimental_csv,
            arguments.output_dir,
            validation_plan=plan,
        )
    except (
        FileExistsError,
        FileNotFoundError,
        IsADirectoryError,
        NotADirectoryError,
        OSError,
        RuntimeError,
        TypeError,
        UnicodeError,
        ValueError,
    ) as error:
        parser.error(str(error))
    print("The program wrote these research-preview files:")
    for name, path in sorted(paths.items()):
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
