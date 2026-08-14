from __future__ import annotations

import math
from hashlib import sha256
from numbers import Integral, Real
import warnings
from typing import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from core.collisions import (
    ElectronCollisionResult,
    IonCollisionResult,
    SecondaryElectronEventBuffer,
    perform_mcc_electron_channels_1d3v_variable_time,
    perform_mcc_ion_1d3v_variable_time,
)
from core.config import Config
from core.cross_sections import (
    build_constant_electron_tables,
    build_constant_ion_tables,
    build_electron_tables_from_lxcat,
    build_ion_tables_from_lxcat,
    load_cross_sections_from_custom_file,
    load_lxcat_text,
)
from core.fields import compute_electric_field, solve_poisson_cylindrical, smooth_density_cylindrical
from core.particles import (
    compute_shell_volumes,
    cylindrical_drift,
    radial_kick,
    weight_charge_cic,
)
from core.provenance import build_run_manifest, canonical_json, json_sha256


@dataclass
class SimulationResult:
    avg_current: float
    avg_conventional_current: float
    current_sem: float
    sample_count: int
    batch_mean_count: int
    r_grid: np.ndarray
    phi: np.ndarray
    ne: np.ndarray
    ni: np.ndarray
    ne_raw: np.ndarray
    ni_raw: np.ndarray
    ion_r: np.ndarray
    ion_vr: np.ndarray
    ion_vz: np.ndarray


@dataclass(frozen=True)
class BoundaryInjectionResult:
    residual: float
    crossing_events: int
    active_particles: int
    probe_hits: int
    active_slots: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.int64),
        compare=False,
        repr=False,
    )
    time_inside_s: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float64),
        compare=False,
        repr=False,
    )


def batch_mean_statistics(
    values: np.ndarray,
) -> tuple[float, float, int]:
    """Calculate a mean and a conservative standard error."""
    samples = np.asarray(values, dtype=float)
    if samples.ndim != 1 or samples.size == 0:
        raise ValueError("Set values to a nonempty one-dimensional array.")
    if not np.all(np.isfinite(samples)):
        raise ValueError("Set values to finite numbers.")

    count = int(samples.size)
    mean = float(np.mean(samples))
    if count == 1:
        return mean, math.nan, 0

    batch_size = max(1, int(math.sqrt(count)))
    batch_count = count // batch_size
    batches = np.array_split(samples, batch_count)
    batch_means = np.asarray(
        [float(np.mean(batch)) for batch in batches],
        dtype=float,
    )
    sem = float(np.std(batch_means, ddof=1) / math.sqrt(batch_count))
    return mean, sem, batch_count


def _json_safe_value(value: object) -> object:
    if isinstance(value, dict):
        return {
            str(key): _json_safe_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_safe_value(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    return value


class PICSimulation:
    def __init__(
        self,
        config: Config,
        n_particles: int = 5000,
        v_bias: float = 0.0,
        reflect_wall: bool = False,
        sigma_cex: float = 1.0e-18,
        seed: int | None = None,
        probe_length: float = 1.0,
        headroom_factor: float = 0.2,
        max_collision_events_per_particle: int = 64,
        max_secondary_buffer_bytes: int = 256 * 1024 * 1024,
    ) -> None:
        if isinstance(n_particles, bool) or not isinstance(n_particles, Integral):
            raise TypeError("Set n_particles to an integer.")
        if int(n_particles) < 1:
            raise ValueError("Set n_particles to an integer greater than zero.")
        if isinstance(probe_length, bool) or not isinstance(probe_length, Real):
            raise TypeError("Set probe_length to a real number.")
        if not math.isfinite(float(probe_length)) or float(probe_length) <= 0.0:
            raise ValueError("Set probe_length to a finite value greater than zero.")
        if isinstance(headroom_factor, bool) or not isinstance(headroom_factor, Real):
            raise TypeError("Set headroom_factor to a real number.")
        if not math.isfinite(float(headroom_factor)) or float(headroom_factor) < 0.0:
            raise ValueError("Set headroom_factor to zero or a positive value.")
        if (
            isinstance(max_collision_events_per_particle, bool)
            or not isinstance(max_collision_events_per_particle, Integral)
        ):
            raise TypeError("Set max_collision_events_per_particle to an integer.")
        if int(max_collision_events_per_particle) < 1:
            raise ValueError(
                "Set max_collision_events_per_particle to an integer greater than zero."
            )
        if (
            isinstance(max_secondary_buffer_bytes, bool)
            or not isinstance(max_secondary_buffer_bytes, Integral)
        ):
            raise TypeError("Set max_secondary_buffer_bytes to an integer.")
        if int(max_secondary_buffer_bytes) < 48:
            raise ValueError(
                "Set max_secondary_buffer_bytes to an integer not less than 48."
            )
        if not isinstance(reflect_wall, bool):
            raise TypeError("Set reflect_wall to a Boolean value.")
        if isinstance(v_bias, bool) or not isinstance(v_bias, Real):
            raise TypeError("Set v_bias to a real number.")
        if not math.isfinite(float(v_bias)):
            raise ValueError("Set v_bias to a finite value.")
        if isinstance(sigma_cex, bool) or not isinstance(sigma_cex, Real):
            raise TypeError("Set sigma_cex to a real number.")
        if not math.isfinite(float(sigma_cex)) or float(sigma_cex) < 0.0:
            raise ValueError("Set sigma_cex to zero or a positive value.")
        if seed is not None:
            if isinstance(seed, bool) or not isinstance(seed, Integral):
                raise TypeError("Set seed to an integer or None.")
            if int(seed) < 0 or int(seed) >= 1 << 64:
                raise ValueError("Set seed to an integer from 0 through 2^64 - 1.")
        elif config.is_production:
            raise ValueError("Set seed for a production run.")
        if config.is_production and not config.PHYSICS_RELEASE_READY:
            raise RuntimeError(
                "Production simulation is locked until the physics release gates pass."
            )
        if config.is_production and (
            not config.LXCAT_ELECTRON_FILE or not config.LXCAT_ION_FILE
        ):
            raise ValueError(
                "Set verified electron and ion cross-section files for a production run."
            )
        if (
            config.CROSS_SECTION_TARGET.casefold()
            != config.NEUTRAL_SPECIES.casefold()
        ):
            raise ValueError(
                "Set CROSS_SECTION_TARGET to the configured neutral species."
            )
        mass_difference = abs(config.m_i - config.m_neutral)
        mass_scale = max(config.m_i, config.m_neutral)
        if mass_difference > 1.0e-6 * mass_scale:
            raise ValueError(
                "This ion collision model requires equal ion and neutral masses."
            )

        self.config = config
        self.n_nominal = int(n_particles)
        self.n_particles = max(
            self.n_nominal,
            int(math.ceil(self.n_nominal * (1.0 + float(headroom_factor)))),
        )
        self.v_bias = float(v_bias)
        self.initial_v_bias = float(v_bias)
        self.probe_length = float(probe_length)
        self.reflect_wall = reflect_wall
        self.sigma_cex = float(sigma_cex)
        self.headroom_factor = float(headroom_factor)
        if seed is None:
            seed = int(np.random.SeedSequence().entropy) & ((1 << 64) - 1)
        self.root_seed = int(seed)
        self.rng = np.random.default_rng(self.root_seed)
        self.step_index = 0
        self.execution_history: list[dict[str, object]] = []
        self.max_collision_events_per_particle = int(
            max_collision_events_per_particle
        )
        self.max_secondary_buffer_bytes = int(max_secondary_buffer_bytes)
        self.failure_reason: str | None = None

        for message in config.stability_warnings():
            warnings.warn(message, RuntimeWarning)

        self.r_min = config.R_MIN
        self.r_max = config.R_MAX
        self.dt = config.DT
        self.dr = config.dr

        self.n_nodes = config.N_CELLS + 1
        self.r_grid = self.r_min + self.dr * np.arange(self.n_nodes)

        self.phi = np.zeros(self.n_nodes)
        self.E = np.zeros(self.n_nodes)
        self.rho = np.zeros(self.n_nodes)
        self.rho_e = np.zeros(self.n_nodes)
        self.rho_i = np.zeros(self.n_nodes)
        self.rho_e_raw = np.zeros(self.n_nodes)
        self.rho_i_raw = np.zeros(self.n_nodes)
        self.ne = np.zeros(self.n_nodes)
        self.ni = np.zeros(self.n_nodes)
        self.vol = np.zeros(self.n_nodes)
        compute_shell_volumes(self.r_min, self.dr, self.vol)

        self.r_e = np.zeros(self.n_particles)
        self.vr_e = np.zeros(self.n_particles)
        self.vt_e = np.zeros(self.n_particles)
        self.vz_e = np.zeros(self.n_particles)
        self.r_i = np.zeros(self.n_particles)
        self.vr_i = np.zeros(self.n_particles)
        self.vt_i = np.zeros(self.n_particles)
        self.vz_i = np.zeros(self.n_particles)

        self.vth_e = math.sqrt(config.e * config.Te / config.m_e)
        self.vth_i = math.sqrt(config.e * config.Ti / config.m_i)
        self.u_bohm = math.sqrt(config.e * max(config.Te, 0.1) / config.m_i)

        self.q_weight = self._compute_macro_weight()
        self.qe = -config.e * self.q_weight
        self.qi = config.ION_CHARGE_STATE * config.e * self.q_weight
        self.qe_arr = np.full(self.n_particles, self.qe)
        self.qi_arr = np.full(self.n_particles, self.qi)

        self.n_g, self.vth_gas = self._compute_neutral_properties()
        self.inject_target_e = self._compute_injection_target_flux(
            self.vth_e,
            0.0,
        )
        self.ion_inject_drift = 0.0
        if config.ION_INJECTION_BOHM:
            self.ion_inject_drift = self.u_bohm
            self.inject_target_i = self._compute_injection_target_flux(
                self.vth_i,
                self.u_bohm,
            )
        else:
            self.inject_target_i = self._compute_injection_target_flux(
                self.vth_i,
                0.0,
            )
        self._flux_tables: dict[tuple[float, float], tuple[np.ndarray, np.ndarray]] = {}
        self.inject_residual_e = 0.0
        self.inject_residual_i = 0.0
        self.sigma_en_elastic = config.SIGMA_EN_ELASTIC
        self.sigma_en_exc = config.SIGMA_EN_EXC
        self.sigma_en_ion = config.SIGMA_EN_ION
        self.e_exc_j = config.E_EXC_EV * config.e
        self.e_ion_j = config.E_ION_EV * config.e
        self.enable_secondaries = config.ENABLE_IONIZATION_SECONDARIES
        self.enable_ion_elastic = config.ENABLE_ION_NEUTRAL_ELASTIC
        self.enable_coulomb = config.ENABLE_COULOMB_COLLISIONS

        self._load_cross_sections()

        self.collision_counters = {
            "electron_elastic": 0,
            "electron_excitation": 0,
            "electron_ionization": 0,
            "ion_charge_exchange": 0,
            "ion_elastic": 0,
            "ion_collision_candidates": 0,
            "ion_null_collision_rejections": 0,
            "secondary_electrons": 0,
            "secondary_ions": 0,
            "electron_coulomb": 0,
            "ion_coulomb": 0,
            "electron_energy_table_overflow_lookups": 0,
            "ion_energy_table_overflow_lookups": 0,
            "boundary_entry_particles_processed": 0,
            "ionization_descendants_processed": 0,
        }
        self.excitation_channel_counters = [
            0 for _ in self.en_excitation_thresholds_ev
        ]
        self.ionization_channel_counters = [
            0 for _ in self.en_ionization_thresholds_ev
        ]
        self.runtime_metrics = {
            "maximum_electron_radial_cell_crossing": 0.0,
            "maximum_ion_radial_cell_crossing": 0.0,
            "maximum_electron_density_m3": 0.0,
            "maximum_local_cell_to_debye_ratio": 0.0,
            "maximum_local_dt_omega_pe": 0.0,
        }
        self.runtime_warnings: list[str] = []
        self._runtime_warning_keys: set[str] = set()

        self.nu_ei = self._compute_coulomb_frequency_ei() if self.enable_coulomb else 0.0
        self.nu_ii = self._compute_coulomb_frequency_ii() if self.enable_coulomb else 0.0

        self._initialize_particles()
        self.particle_counters = {
            "initial_electrons": self.initial_electron_count,
            "initial_ions": self.initial_ion_count,
            "injected_electrons": 0,
            "injected_ions": 0,
            "probe_absorbed_electrons": 0,
            "probe_absorbed_ions": 0,
            "wall_absorbed_electrons": 0,
            "wall_absorbed_ions": 0,
        }
        self._update_fields()

    def _load_cross_sections(self) -> None:
        """Load energy-dependent cross sections (LXCat) or fall back to constants."""
        target = self.config.CROSS_SECTION_TARGET
        strict = self.config.is_production or self.config.CROSS_SECTION_STRICT
        self.cross_section_strict = strict
        self.ion_backscatter_mapping_requested = (
            self.config.CONFIRM_SYMMETRIC_BACKSCATTER_AS_CEX
        )
        self.ion_backscatter_mapping_applied = False

        def resolve_path(path_str: str) -> str:
            path = Path(path_str)
            if path.is_absolute():
                return str(path.resolve())
            root = Path(__file__).resolve().parents[1]
            return str((root / path).resolve())

        electron_path = (
            resolve_path(self.config.LXCAT_ELECTRON_FILE)
            if self.config.LXCAT_ELECTRON_FILE
            else None
        )
        ion_path = (
            resolve_path(self.config.LXCAT_ION_FILE)
            if self.config.LXCAT_ION_FILE
            else None
        )
        if strict and (electron_path is None or ion_path is None):
            raise ValueError(
                "Strict cross-section loading requires both electron and ion files."
            )
        self.cross_section_source_files = tuple(
            sorted(
                {
                    path
                    for path in (electron_path, ion_path)
                    if path is not None
                }
            )
        )
        self.cross_section_inventory: list[dict[str, object]] = []
        for source_path in self.cross_section_source_files:
            source_processes = load_lxcat_text(
                source_path,
                default_target=target,
                strict=strict,
            )
            counts: dict[str, int] = {}
            species: set[str] = set()
            for process in source_processes:
                counts[process.process_type] = (
                    counts.get(process.process_type, 0) + 1
                )
                species.add(
                    f"{process.incident_particle.strip()} / "
                    f"{process.target_particle.strip()}"
                )
            self.cross_section_inventory.append(
                {
                    "source_path": source_path,
                    "source_sha256": (
                        source_processes[0].source_sha256
                        if source_processes
                        else None
                    ),
                    "process_counts": dict(sorted(counts.items())),
                    "species_pairs": sorted(species),
                }
            )
            if source_path == ion_path and any(
                process.process_type == "BACKSCATTER"
                for process in source_processes
            ):
                self.ion_backscatter_mapping_applied = bool(
                    self.ion_backscatter_mapping_requested
                )
        if electron_path is None and ion_path is None:
            self.cross_section_model = "constant_test_tables"
        elif electron_path is not None and ion_path is not None:
            self.cross_section_model = "file_tables"
        else:
            self.cross_section_model = "mixed_file_and_constant_test_tables"

        if electron_path is not None and electron_path == ion_path:
            electron_tables, ion_tables = load_cross_sections_from_custom_file(
                electron_path,
                e_max=self.config.EN_CS_E_MAX,
                n_bins=self.config.EN_CS_N,
                target=target,
                strict=strict,
                ion_e_max=self.config.ION_CS_E_MAX,
                ion_n_bins=self.config.ION_CS_N,
                confirm_symmetric_backscatter_as_cex=(
                    self.ion_backscatter_mapping_requested
                ),
                ion_species=self.config.ION_SPECIES,
            )
        else:
            if electron_path is not None:
                electron_tables = build_electron_tables_from_lxcat(
                    electron_path,
                    target=target,
                    e_max=self.config.EN_CS_E_MAX,
                    n_bins=self.config.EN_CS_N,
                    strict=strict,
                )
            else:
                electron_tables = build_constant_electron_tables(
                    self.config.SIGMA_EN_ELASTIC,
                    self.config.SIGMA_EN_EXC,
                    self.config.SIGMA_EN_ION,
                    e_max=self.config.EN_CS_E_MAX,
                    n_bins=self.config.EN_CS_N,
                    excitation_threshold_ev=self.config.E_EXC_EV,
                    ionization_threshold_ev=self.config.E_ION_EV,
                )

            if ion_path is not None:
                ion_tables = build_ion_tables_from_lxcat(
                    ion_path,
                    target=target,
                    e_max=self.config.ION_CS_E_MAX,
                    n_bins=self.config.ION_CS_N,
                    fallback_cex=self.sigma_cex,
                    strict=strict,
                    confirm_symmetric_backscatter_as_cex=(
                        self.ion_backscatter_mapping_requested
                    ),
                    ion_species=self.config.ION_SPECIES,
                )
            else:
                ion_tables = build_constant_ion_tables(
                    sigma_cex=self.sigma_cex,
                    sigma_elastic=self.config.SIGMA_IN_ELASTIC,
                    e_max=self.config.ION_CS_E_MAX,
                    n_bins=self.config.ION_CS_N,
                )

        if not self.enable_ion_elastic:
            ion_tables = ion_tables.__class__(
                e_min=ion_tables.e_min,
                inv_de=ion_tables.inv_de,
                sigma_cex=ion_tables.sigma_cex,
                sigma_elastic=np.zeros_like(ion_tables.sigma_elastic),
            )

        self.en_e_min = electron_tables.e_min
        self.en_inv_de = electron_tables.inv_de
        self.en_sigma_elastic = electron_tables.sigma_elastic
        self.en_sigma_exc = electron_tables.sigma_excitation
        self.en_sigma_ion = electron_tables.sigma_ionization
        self.en_excitation_thresholds_ev = (
            electron_tables.excitation_thresholds_ev
        )
        self.en_excitation_channel_tables = (
            electron_tables.excitation_channel_tables
        )
        self.en_ionization_thresholds_ev = (
            electron_tables.ionization_thresholds_ev
        )
        self.en_ionization_channel_tables = (
            electron_tables.ionization_channel_tables
        )

        self.ion_e_min = ion_tables.e_min
        self.ion_inv_de = ion_tables.inv_de
        self.ion_sigma_cex = ion_tables.sigma_cex
        self.ion_sigma_elastic = ion_tables.sigma_elastic

    def _compute_coulomb_frequency_ei(self) -> float:
        """NRL formulary-style electron-ion collision frequency (s^-1)."""
        n_cm3 = self.config.N0 * 1.0e-6
        te = max(self.config.Te, 1.0e-3)
        return 2.91e-6 * n_cm3 * self.config.COULOMB_LOG / (te ** 1.5)

    def _compute_coulomb_frequency_ii(self) -> float:
        """NRL formulary-style ion-ion collision frequency (s^-1)."""
        n_cm3 = self.config.N0 * 1.0e-6
        ti = max(self.config.Ti, 1.0e-4)
        mu = self.config.m_i / self.config.m_p
        return 4.80e-8 * n_cm3 * self.config.COULOMB_LOG / (ti ** 1.5 * math.sqrt(mu))

    def _compute_macro_weight(self) -> float:
        area = math.pi * (self.r_max * self.r_max - self.r_min * self.r_min)
        return self.config.N0 * area / float(self.n_nominal)

    def _compute_neutral_properties(self) -> tuple[float, float]:
        n_g = self.config.pressure_pa / (
            self.config.k_B * self.config.T_GAS_K
        )
        v_th = math.sqrt(
            self.config.k_B * self.config.T_GAS_K / self.config.m_neutral
        )
        return n_g, v_th

    def _compute_injection_target_flux(self, vth: float, drift: float) -> float:
        """Calculate the inward flux for a drifting Maxwellian."""
        if vth <= 0.0:
            velocity_flux = max(drift, 0.0)
        else:
            ratio = drift / vth
            gaussian = math.exp(-0.5 * ratio * ratio) / math.sqrt(2.0 * math.pi)
            cumulative = 0.5 * (1.0 + math.erf(ratio / math.sqrt(2.0)))
            velocity_flux = vth * gaussian + drift * cumulative
        flux = self.config.N0 * velocity_flux
        boundary_area = 2.0 * math.pi * self.r_max
        n_phys = flux * boundary_area * self.dt
        return n_phys / self.q_weight

    def _sample_inward_flux_speed(
        self,
        vth: float,
        drift: float,
        size: int,
    ) -> np.ndarray:
        """Sample the inward speed from a flux-weighted Maxwellian."""
        key = (float(vth), float(drift))
        table = self._flux_tables.get(key)
        if table is None:
            speed_max = max(drift + 8.0 * vth, 8.0 * vth, 1.0e-12)
            speed = np.linspace(0.0, speed_max, 4097)
            density = speed * np.exp(-0.5 * ((speed - drift) / vth) ** 2)
            increments = 0.5 * (density[:-1] + density[1:]) * np.diff(speed)
            cdf = np.empty_like(speed)
            cdf[0] = 0.0
            cdf[1:] = np.cumsum(increments)
            if cdf[-1] <= 0.0:
                raise ValueError("The injection speed distribution is empty.")
            cdf /= cdf[-1]
            table = (speed, cdf)
            self._flux_tables[key] = table
        return np.interp(self.rng.random(size), table[1], table[0])

    def _estimate_sheath_width(self) -> float:
        lambda_d = self.config.debye_length()
        domain = self.r_max - self.r_min
        te = max(self.config.Te, 0.1)
        scale = abs(self.v_bias - self.config.V_WALL) / te
        if scale < 1.0:
            scale = 1.0
        width = 5.0 * lambda_d * math.sqrt(scale)
        width = max(width, 5.0 * self.dr)
        width = min(width, 0.3 * domain)
        return width

    def _initial_potential_profile(self) -> np.ndarray:
        width = self._estimate_sheath_width()
        phi = np.full(self.n_nodes, self.config.V_WALL)
        if width <= 0.0:
            phi[0] = self.v_bias
            return phi

        cutoff = self.r_min + width
        mask = self.r_grid <= cutoff
        xi = (self.r_grid[mask] - self.r_min) / width
        phi[mask] = self.v_bias + (self.config.V_WALL - self.v_bias) * np.power(xi, 4.0 / 3.0)
        return phi

    def _profile_particle_count(self, n_profile: np.ndarray) -> int:
        integrand = n_profile * self.r_grid
        profile_area_density = 2.0 * math.pi * float(
            np.sum(
                0.5
                * (integrand[:-1] + integrand[1:])
                * np.diff(self.r_grid)
            )
        )
        domain_area_density = self.config.N0 * math.pi * (
            self.r_max * self.r_max - self.r_min * self.r_min
        )
        fraction = profile_area_density / domain_area_density
        count = int(round(self.n_nominal * fraction))
        return min(max(count, 0), self.n_particles)

    def _sample_positions_from_density(
        self,
        n_profile: np.ndarray,
        n_samples: int,
    ) -> np.ndarray:
        if n_samples == 0:
            return np.empty(0, dtype=float)
        if n_profile.shape != self.r_grid.shape:
            raise ValueError("The density profile and radial grid have different shapes.")
        if not np.all(np.isfinite(n_profile)) or np.any(n_profile < 0.0):
            raise ValueError("The density profile has an invalid value.")

        integrand = n_profile * self.r_grid
        increments = (
            0.5
            * (integrand[:-1] + integrand[1:])
            * np.diff(self.r_grid)
        )
        cdf = np.empty_like(self.r_grid)
        cdf[0] = 0.0
        cdf[1:] = np.cumsum(increments)
        if cdf[-1] <= 0.0:
            raise ValueError("The density profile has zero integral.")
        samples = self.rng.random(n_samples) * cdf[-1]
        cells = np.searchsorted(cdf, samples, side="right") - 1
        cells = np.clip(cells, 0, self.r_grid.size - 2)
        local_integral = samples - cdf[cells]
        cell_width = self.r_grid[cells + 1] - self.r_grid[cells]
        left_value = integrand[cells]
        slope = (
            integrand[cells + 1] - integrand[cells]
        ) / cell_width
        discriminant = np.maximum(
            0.0,
            left_value * left_value + 2.0 * slope * local_integral,
        )
        denominator = left_value + np.sqrt(discriminant)
        local_position = np.divide(
            2.0 * local_integral,
            denominator,
            out=np.zeros_like(local_integral),
            where=denominator > 0.0,
        )
        nearly_constant = np.abs(slope) <= np.finfo(float).eps * np.maximum(
            1.0,
            np.abs(left_value) / cell_width,
        )
        local_position[nearly_constant] = np.divide(
            local_integral[nearly_constant],
            left_value[nearly_constant],
            out=np.zeros(np.count_nonzero(nearly_constant)),
            where=left_value[nearly_constant] > 0.0,
        )
        positions = self.r_grid[cells] + np.clip(
            local_position,
            0.0,
            cell_width,
        )
        return np.clip(
            positions,
            np.nextafter(self.r_min, self.r_max),
            np.nextafter(self.r_max, self.r_min),
        )

    def _initialize_particles(self) -> None:
        phi_init = self._initial_potential_profile()
        phi_rel = phi_init - self.config.V_WALL
        te = max(self.config.Te, 0.1)
        n0 = self.config.N0

        n_e = n0 * np.exp(phi_rel / te)
        n_e = np.clip(n_e, n0 * 1.0e-4, n0)

        u_b = math.sqrt(self.config.e * te / self.config.m_i)
        phi_drop = self.config.V_WALL - phi_init
        phi_drop = np.maximum(phi_drop, 0.0)
        v_i = np.sqrt(u_b * u_b + 2.0 * self.config.e * phi_drop / self.config.m_i)
        n_i = n0 * u_b / v_i
        n_i = np.clip(n_i, n0 * 1.0e-3, n0)

        self.initial_electron_count = self._profile_particle_count(n_e)
        self.initial_ion_count = self._profile_particle_count(n_i)
        self.r_e[: self.initial_electron_count] = self._sample_positions_from_density(
            n_e,
            self.initial_electron_count,
        )
        self.r_i[: self.initial_ion_count] = self._sample_positions_from_density(
            n_i,
            self.initial_ion_count,
        )


        self.vr_e[:] = self.rng.normal(0.0, self.vth_e, self.n_particles)
        self.vt_e[:] = self.rng.normal(0.0, self.vth_e, self.n_particles)
        self.vz_e[:] = self.rng.normal(0.0, self.vth_e, self.n_particles)
        self.vr_i[:] = self.rng.normal(0.0, self.vth_i, self.n_particles)
        self.vt_i[:] = self.rng.normal(0.0, self.vth_i, self.n_particles)
        self.vz_i[:] = self.rng.normal(0.0, self.vth_i, self.n_particles)

    def _update_fields(self) -> None:
        weight_charge_cic(
            self.r_e,
            self.qe_arr,
            self.r_min,
            self.dr,
            self.rho_e_raw,
            self.vol,
        )
        weight_charge_cic(
            self.r_i,
            self.qi_arr,
            self.r_min,
            self.dr,
            self.rho_i_raw,
            self.vol,
        )
        self.rho_e[:] = self.rho_e_raw
        self.rho_i[:] = self.rho_i_raw
        
        if self.config.SMOOTH_DENSITY:
            smooth_density_cylindrical(
                self.rho_e,
                self.config.N_SMOOTHING_PASSES,
                self.vol,
            )
            smooth_density_cylindrical(
                self.rho_i,
                self.config.N_SMOOTHING_PASSES,
                self.vol,
            )

        self.rho[:] = self.rho_e + self.rho_i
        solve_poisson_cylindrical(
            self.rho,
            self.phi,
            self.r_min,
            self.dr,
            self.config.epsilon_0,
            self.v_bias,
            self.config.V_WALL,
        )
        compute_electric_field(self.phi, self.dr, self.E)
        self._update_runtime_density_metrics()

    def inject_particles(
        self,
        r: np.ndarray,
        vr: np.ndarray,
        vt: np.ndarray,
        vz: np.ndarray,
        vth: float,
        target_per_step: float,
        residual: float,
        drift: float = 0.0,
    ) -> float:
        dead_idx = np.flatnonzero((r <= self.r_min) | (r >= self.r_max))
        n_dead = int(dead_idx.size)
        target_total = target_per_step + residual
        if n_dead == 0:
            return target_total

        n_inject = int(target_total)
        residual = target_total - n_inject
        if n_inject <= 0:
            return residual

        if n_inject > n_dead:
            n_inject = n_dead
            residual = target_total - n_inject

        if n_inject < n_dead:
            pick = self.rng.permutation(n_dead)[:n_inject]
            idx = dead_idx[pick]
        else:
            idx = dead_idx

        r[idx] = self.r_max - 0.5 * self.dr * self.rng.random(n_inject)
        vr_in = self._sample_inward_flux_speed(vth, drift, n_inject)
        vr[idx] = -vr_in
        vt[idx] = self.rng.normal(0.0, vth, n_inject)
        vz[idx] = self.rng.normal(0.0, vth, n_inject)
        return residual

    def _inject_boundary_arrivals(
        self,
        r: np.ndarray,
        vr: np.ndarray,
        vt: np.ndarray,
        vz: np.ndarray,
        vth: float,
        target_per_step: float,
        residual: float,
        *,
        drift: float,
        charge: float,
        mass: float,
    ) -> BoundaryInjectionResult:
        target_total = target_per_step + residual
        crossing_events = int(target_total)
        residual = target_total - crossing_events
        if crossing_events == 0:
            return BoundaryInjectionResult(residual, 0, 0, 0)

        dead = np.flatnonzero((r <= self.r_min) | (r >= self.r_max))
        if dead.size < crossing_events:
            self._stop_failed_run("The boundary source has no particle slots.")
        slots = dead[:crossing_events]

        inward_speed = self._sample_inward_flux_speed(
            vth,
            drift,
            crossing_events,
        )
        transverse_speed = self.rng.normal(0.0, vth, crossing_events)
        axial_speed = self.rng.normal(0.0, vth, crossing_events)
        time_inside = self.dt * self.rng.random(crossing_events)
        boundary_acceleration = (
            (charge / mass) * self.E[-1]
            + transverse_speed * transverse_speed / self.r_max
        )
        position = (
            self.r_max
            - inward_speed * time_inside
            + 0.5 * boundary_acceleration * time_inside * time_inside
        )
        radial_speed = -inward_speed + boundary_acceleration * time_inside
        position = np.minimum(
            position,
            np.nextafter(self.r_max, self.r_min),
        )
        probe_mask = position <= self.r_min
        active_mask = ~probe_mask
        active_slots = slots[active_mask]
        active_position = position[active_mask]

        r[slots] = self.r_max + (self.r_max - self.r_min)
        vr[slots] = 0.0
        vt[slots] = 0.0
        vz[slots] = 0.0
        r[active_slots] = active_position
        vr[active_slots] = radial_speed[active_mask]
        vt[active_slots] = (
            transverse_speed[active_mask]
            * self.r_max
            / active_position
        )
        vz[active_slots] = axial_speed[active_mask]
        return BoundaryInjectionResult(
            residual=residual,
            crossing_events=crossing_events,
            active_particles=int(np.count_nonzero(active_mask)),
            probe_hits=int(np.count_nonzero(probe_mask)),
            active_slots=active_slots.copy(),
            time_inside_s=time_inside[active_mask].copy(),
        )

    def _ensure_injection_capacity(self) -> None:
        requested_e = int(self.inject_target_e + self.inject_residual_e)
        requested_i = int(self.inject_target_i + self.inject_residual_i)
        dead_e = int(
            np.count_nonzero(
                (self.r_e <= self.r_min) | (self.r_e >= self.r_max)
            )
        )
        dead_i = int(
            np.count_nonzero(
                (self.r_i <= self.r_min) | (self.r_i >= self.r_max)
            )
        )
        shortage = max(requested_e - dead_e, requested_i - dead_i, 0)
        if shortage:
            self._ensure_particle_capacity(self.n_particles + shortage)

    def _stop_failed_run(self, reason: str) -> None:
        self.failure_reason = reason
        raise RuntimeError(reason)

    @staticmethod
    def _grow_array(
        values: np.ndarray,
        new_size: int,
        fill_value: float,
    ) -> np.ndarray:
        result = np.empty(new_size, dtype=values.dtype)
        old_size = values.size
        result[:old_size] = values
        result[old_size:] = fill_value
        return result

    def _ensure_particle_capacity(self, required_capacity: int) -> None:
        if required_capacity <= self.n_particles:
            return

        new_capacity = max(
            int(required_capacity),
            self.n_particles + 1,
            int(math.ceil(1.5 * self.n_particles)),
        )
        self.r_e = self._grow_array(self.r_e, new_capacity, 0.0)
        self.vr_e = self._grow_array(self.vr_e, new_capacity, 0.0)
        self.vt_e = self._grow_array(self.vt_e, new_capacity, 0.0)
        self.vz_e = self._grow_array(self.vz_e, new_capacity, 0.0)
        self.r_i = self._grow_array(self.r_i, new_capacity, 0.0)
        self.vr_i = self._grow_array(self.vr_i, new_capacity, 0.0)
        self.vt_i = self._grow_array(self.vt_i, new_capacity, 0.0)
        self.vz_i = self._grow_array(self.vz_i, new_capacity, 0.0)
        self.qe_arr = self._grow_array(self.qe_arr, new_capacity, self.qe)
        self.qi_arr = self._grow_array(self.qi_arr, new_capacity, self.qi)
        self.n_particles = new_capacity

    def _make_secondary_buffer(
        self,
        capacity: int,
    ) -> SecondaryElectronEventBuffer:
        capacity = max(int(capacity), 1)
        required_bytes = capacity * 48
        if required_bytes > self.max_secondary_buffer_bytes:
            self._stop_failed_run(
                "The bounded ionization event buffer is larger than the memory limit."
            )
        return SecondaryElectronEventBuffer(
            parent_index=np.empty(capacity, dtype=np.int64),
            event_time_s=np.empty(capacity, dtype=float),
            vx=np.empty(capacity, dtype=float),
            vy=np.empty(capacity, dtype=float),
            vz=np.empty(capacity, dtype=float),
            energy_ev=np.empty(capacity, dtype=float),
        )

    def _spawn_ionization_event_products(
        self,
        events: SecondaryElectronEventBuffer,
    ) -> tuple[np.ndarray, np.ndarray]:
        n_new = int(events.count)
        if n_new == 0:
            empty = np.empty(0, dtype=np.int64)
            return empty, empty.copy()
        if not self.enable_secondaries:
            self._stop_failed_run(
                "Ionization occurred while ionization products were disabled."
            )

        parent_index = events.parent_index[:n_new]
        if np.any(parent_index < 0) or np.any(parent_index >= self.n_particles):
            self._stop_failed_run("An ionization event has an invalid parent index.")
        parent_r = self.r_e[parent_index].copy()
        if np.any((parent_r <= self.r_min) | (parent_r >= self.r_max)):
            self._stop_failed_run("An ionization event has an inactive parent.")

        dead_e = np.flatnonzero(
            (self.r_e <= self.r_min) | (self.r_e >= self.r_max)
        )
        dead_i = np.flatnonzero(
            (self.r_i <= self.r_min) | (self.r_i >= self.r_max)
        )
        shortage = max(n_new - dead_e.size, n_new - dead_i.size, 0)
        if shortage:
            self._ensure_particle_capacity(self.n_particles + shortage)
            dead_e = np.flatnonzero(
                (self.r_e <= self.r_min) | (self.r_e >= self.r_max)
            )
            dead_i = np.flatnonzero(
                (self.r_i <= self.r_min) | (self.r_i >= self.r_max)
            )

        electron_slots = dead_e[:n_new]
        ion_slots = dead_i[:n_new]
        if electron_slots.size != n_new or ion_slots.size != n_new:
            self._stop_failed_run("The ionization products have no particle slots.")

        self.r_e[electron_slots] = parent_r
        self.r_i[ion_slots] = parent_r
        self.vr_e[electron_slots] = events.vx[:n_new]
        self.vt_e[electron_slots] = events.vy[:n_new]
        self.vz_e[electron_slots] = events.vz[:n_new]
        self.vr_i[ion_slots] = self.rng.normal(0.0, self.vth_gas, n_new)
        self.vt_i[ion_slots] = self.rng.normal(0.0, self.vth_gas, n_new)
        self.vz_i[ion_slots] = self.rng.normal(0.0, self.vth_gas, n_new)
        self.collision_counters["secondary_electrons"] += n_new
        self.collision_counters["secondary_ions"] += n_new
        return electron_slots.copy(), ion_slots.copy()

    def _spawn_ionization_events(
        self,
        events: SecondaryElectronEventBuffer,
    ) -> int:
        """Create ionization products and give their count."""
        electron_slots, _ = self._spawn_ionization_event_products(events)
        return int(electron_slots.size)

    @staticmethod
    def _table_energy_max(e_min: float, inv_de: float, size: int) -> float:
        if inv_de <= 0.0 or size < 2:
            return math.inf
        return e_min + (size - 1) / inv_de

    def _check_collision_energy_range(
        self,
        alive_e: np.ndarray,
        alive_i: np.ndarray,
    ) -> None:
        if not self.cross_section_source_files:
            return

        electron_max = self._table_energy_max(
            self.en_e_min,
            self.en_inv_de,
            self.en_sigma_elastic.size,
        )
        ion_max = self._table_energy_max(
            self.ion_e_min,
            self.ion_inv_de,
            self.ion_sigma_cex.size,
        )
        electron_energy = (
            0.5
            * self.config.m_e
            * (
                self.vr_e[alive_e] ** 2
                + self.vt_e[alive_e] ** 2
                + self.vz_e[alive_e] ** 2
            )
            / self.config.e
        )
        ion_speed = np.sqrt(
            self.vr_i[alive_i] ** 2
            + self.vt_i[alive_i] ** 2
            + self.vz_i[alive_i] ** 2
        )
        ion_energy = (
            0.5
            * self.config.m_i
            * (ion_speed + 6.0 * self.vth_gas) ** 2
            / self.config.e
        )
        electron_out = bool(
            electron_energy.size and np.max(electron_energy) > electron_max
        )
        ion_out = bool(ion_energy.size and np.max(ion_energy) > ion_max)
        if not electron_out and not ion_out:
            return

        detail = []
        if electron_out:
            detail.append(
                f"electron energy exceeds {electron_max:.6g} eV"
            )
        if ion_out:
            detail.append(f"ion relative energy can exceed {ion_max:.6g} eV")
        message = "The cross-section table range is too small: " + "; ".join(detail)
        if self.config.is_production:
            self._stop_failed_run(message)
        warning_key = "cross_section_energy_range"
        if warning_key not in getattr(self, "_reported_warnings", set()):
            warnings.warn(message, RuntimeWarning)
            if not hasattr(self, "_reported_warnings"):
                self._reported_warnings: set[str] = set()
            self._reported_warnings.add(warning_key)

    def _apply_coulomb_scatter_3d(
        self,
        r: np.ndarray,
        vr: np.ndarray,
        vt: np.ndarray,
        vz: np.ndarray,
        nu: float,
        duration: float,
    ) -> int:
        if nu <= 0.0 or duration <= 0.0:
            return 0
        active = np.flatnonzero((r > self.r_min) & (r < self.r_max))
        if active.size == 0:
            return 0
        probability = 1.0 - math.exp(-nu * duration)
        selected = active[self.rng.random(active.size) < probability]
        if selected.size == 0:
            return 0
        speed = np.sqrt(
            vr[selected] ** 2 + vt[selected] ** 2 + vz[selected] ** 2
        )
        direction_z = 2.0 * self.rng.random(selected.size) - 1.0
        azimuth = 2.0 * math.pi * self.rng.random(selected.size)
        transverse = np.sqrt(np.maximum(0.0, 1.0 - direction_z**2))
        vr[selected] = speed * transverse * np.cos(azimuth)
        vt[selected] = speed * transverse * np.sin(azimuth)
        vz[selected] = speed * direction_z
        return int(selected.size)

    def _apply_coulomb_scatter_variable_3d(
        self,
        r: np.ndarray,
        vr: np.ndarray,
        vt: np.ndarray,
        vz: np.ndarray,
        nu: float,
        particle_duration_s: np.ndarray,
    ) -> int:
        if nu <= 0.0:
            return 0
        active = np.flatnonzero(
            (r > self.r_min)
            & (r < self.r_max)
            & (particle_duration_s > 0.0)
        )
        if active.size == 0:
            return 0
        probability = 1.0 - np.exp(-nu * particle_duration_s[active])
        selected = active[self.rng.random(active.size) < probability]
        if selected.size == 0:
            return 0
        speed = np.sqrt(
            vr[selected] ** 2 + vt[selected] ** 2 + vz[selected] ** 2
        )
        direction_z = 2.0 * self.rng.random(selected.size) - 1.0
        azimuth = 2.0 * math.pi * self.rng.random(selected.size)
        transverse = np.sqrt(np.maximum(0.0, 1.0 - direction_z**2))
        vr[selected] = speed * transverse * np.cos(azimuth)
        vt[selected] = speed * transverse * np.sin(azimuth)
        vz[selected] = speed * direction_z
        return int(selected.size)

    def _record_ion_collision_result(self, result: IonCollisionResult) -> None:
        if result.event_limit_stops:
            self._stop_failed_run(
                "The ion collision event limit stopped one or more particles."
            )
        if result.candidate_limit_stops:
            self._stop_failed_run(
                "The ion null-collision candidate limit stopped one or more particles."
            )
        self.collision_counters["ion_collision_candidates"] += result.candidate_events
        self.collision_counters[
            "ion_null_collision_rejections"
        ] += result.null_collision_rejections
        self.collision_counters[
            "ion_energy_table_overflow_lookups"
        ] += result.energy_table_overflow_lookups
        if result.energy_table_overflow_lookups:
            message = "An ion collision used energy above the cross-section table."
            if self.config.is_production:
                self._stop_failed_run(message)
            if "ion_collision_energy_range" not in getattr(
                self,
                "_reported_warnings",
                set(),
            ):
                warnings.warn(message, RuntimeWarning)
                if not hasattr(self, "_reported_warnings"):
                    self._reported_warnings = set()
                self._reported_warnings.add("ion_collision_energy_range")
        self.collision_counters[
            "ion_charge_exchange"
        ] += result.charge_exchange_events
        self.collision_counters["ion_elastic"] += result.elastic_events

    def _record_electron_collision_result(
        self,
        result: ElectronCollisionResult,
    ) -> None:
        if result.secondary_events_dropped:
            self._stop_failed_run(
                "The ionization event buffer dropped one or more products."
            )
        if result.event_limit_stops:
            self._stop_failed_run(
                "The electron collision event limit stopped one or more particles."
            )
        self.collision_counters[
            "electron_energy_table_overflow_lookups"
        ] += result.energy_table_overflow_lookups
        if result.energy_table_overflow_lookups:
            message = "An electron collision used energy above the cross-section table."
            if self.config.is_production:
                self._stop_failed_run(message)
            if "electron_collision_energy_range" not in getattr(
                self,
                "_reported_warnings",
                set(),
            ):
                warnings.warn(message, RuntimeWarning)
                if not hasattr(self, "_reported_warnings"):
                    self._reported_warnings = set()
                self._reported_warnings.add("electron_collision_energy_range")
        if result.ionization_events != result.secondary_events_written:
            self._stop_failed_run(
                "The ionization event count does not equal the product count."
            )
        if int(np.sum(result.excitation_channel_events)) != result.excitation_events:
            self._stop_failed_run(
                "The excitation channel count does not equal the event count."
            )
        if int(np.sum(result.ionization_channel_events)) != result.ionization_events:
            self._stop_failed_run(
                "The ionization channel count does not equal the event count."
            )
        self.collision_counters["electron_elastic"] += result.elastic_events
        self.collision_counters[
            "electron_excitation"
        ] += result.excitation_events
        self.collision_counters[
            "electron_ionization"
        ] += result.ionization_events
        for index, count in enumerate(result.excitation_channel_events):
            self.excitation_channel_counters[index] += int(count)
        for index, count in enumerate(result.ionization_channel_events):
            self.ionization_channel_counters[index] += int(count)

    def _apply_collision_intervals(
        self,
        electron_duration_s: np.ndarray,
        ion_duration_s: np.ndarray,
        *,
        step_index: int,
        stream_id: int,
    ) -> SecondaryElectronEventBuffer:
        if (
            electron_duration_s.shape != (self.n_particles,)
            or ion_duration_s.shape != (self.n_particles,)
        ):
            raise ValueError("The collision-duration arrays have invalid shapes.")
        if (
            np.any(~np.isfinite(electron_duration_s))
            or np.any(electron_duration_s < 0.0)
            or np.any(~np.isfinite(ion_duration_s))
            or np.any(ion_duration_s < 0.0)
        ):
            raise ValueError("A collision-duration array has an invalid value.")

        alive_i = (
            (self.r_i > self.r_min)
            & (self.r_i < self.r_max)
            & (ion_duration_s > 0.0)
        )
        if np.any(alive_i):
            ion_result = perform_mcc_ion_1d3v_variable_time(
                self.vr_i,
                self.vt_i,
                self.vz_i,
                alive_i,
                self.n_g,
                self.ion_sigma_cex,
                self.ion_sigma_elastic,
                ion_duration_s,
                self.vth_gas,
                self.ion_e_min,
                self.ion_inv_de,
                self.config.e,
                self.config.m_i,
                seed=self.root_seed,
                step_index=step_index,
                stream_id=stream_id,
                max_events_per_particle=self.max_collision_events_per_particle,
            )
            self._record_ion_collision_result(ion_result)

        alive_e = (
            (self.r_e > self.r_min)
            & (self.r_e < self.r_max)
            & (electron_duration_s > 0.0)
        )
        buffer_capacity = (
            int(np.count_nonzero(alive_e))
            * self.max_collision_events_per_particle
        )
        secondary_buffer = self._make_secondary_buffer(buffer_capacity)
        if np.any(alive_e):
            electron_result = perform_mcc_electron_channels_1d3v_variable_time(
                self.vr_e,
                self.vt_e,
                self.vz_e,
                alive_e,
                self.n_g,
                self.en_sigma_elastic,
                self.en_excitation_channel_tables,
                self.en_excitation_thresholds_ev,
                self.en_ionization_channel_tables,
                self.en_ionization_thresholds_ev,
                electron_duration_s,
                self.config.m_e,
                self.en_e_min,
                self.en_inv_de,
                self.config.e,
                secondary_buffer,
                seed=self.root_seed,
                step_index=step_index,
                stream_id=stream_id,
                max_events_per_particle=self.max_collision_events_per_particle,
            )
            self._record_electron_collision_result(electron_result)

        if self.enable_coulomb:
            self.collision_counters[
                "electron_coulomb"
            ] += self._apply_coulomb_scatter_variable_3d(
                self.r_e,
                self.vr_e,
                self.vt_e,
                self.vz_e,
                self.nu_ei,
                electron_duration_s,
            )
            self.collision_counters[
                "ion_coulomb"
            ] += self._apply_coulomb_scatter_variable_3d(
                self.r_i,
                self.vr_i,
                self.vt_i,
                self.vz_i,
                self.nu_ii,
                ion_duration_s,
            )
        return secondary_buffer

    def _process_ionization_descendants(
        self,
        events: SecondaryElectronEventBuffer,
        parent_duration_s: np.ndarray,
        *,
        step_index: int,
        stream_id: int,
    ) -> None:
        current_events = events
        current_parent_duration = parent_duration_s
        while current_events.count:
            count = int(current_events.count)
            parent_index = current_events.parent_index[:count]
            if (
                np.any(parent_index < 0)
                or np.any(parent_index >= current_parent_duration.size)
            ):
                self._stop_failed_run(
                    "An ionization event has an invalid duration parent."
                )
            available = current_parent_duration[parent_index]
            event_time = current_events.event_time_s[:count]
            tolerance = np.maximum(np.finfo(float).tiny, available) * 1.0e-12
            if np.any(event_time < -tolerance) or np.any(
                event_time > available + tolerance
            ):
                self._stop_failed_run(
                    "An ionization event time is outside its collision interval."
                )
            remaining = np.maximum(0.0, available - event_time)
            electron_slots, ion_slots = self._spawn_ionization_event_products(
                current_events
            )

            electron_duration = np.zeros(self.n_particles, dtype=np.float64)
            ion_duration = np.zeros(self.n_particles, dtype=np.float64)
            electron_duration[electron_slots] = remaining
            ion_duration[ion_slots] = remaining
            self.collision_counters[
                "ionization_descendants_processed"
            ] += 2 * int(np.count_nonzero(remaining > 0.0))
            current_events = self._apply_collision_intervals(
                electron_duration,
                ion_duration,
                step_index=step_index,
                stream_id=stream_id,
            )
            current_parent_duration = electron_duration

    def _apply_collision_half_step(self, phase: int) -> None:
        duration = 0.5 * self.dt
        alive_e = (self.r_e > self.r_min) & (self.r_e < self.r_max)
        alive_i = (self.r_i > self.r_min) & (self.r_i < self.r_max)
        self._check_collision_energy_range(alive_e, alive_i)

        electron_duration = np.zeros(self.n_particles, dtype=np.float64)
        ion_duration = np.zeros(self.n_particles, dtype=np.float64)
        electron_duration[alive_e] = duration
        ion_duration[alive_i] = duration
        collision_step = 2 * self.step_index + phase
        events = self._apply_collision_intervals(
            electron_duration,
            ion_duration,
            step_index=collision_step,
            stream_id=0,
        )
        self._process_ionization_descendants(
            events,
            electron_duration,
            step_index=collision_step,
            stream_id=0,
        )

    def _apply_boundary_entry_collisions(
        self,
        electron_injection: BoundaryInjectionResult,
        ion_injection: BoundaryInjectionResult,
    ) -> None:
        electron_duration = np.zeros(self.n_particles, dtype=np.float64)
        ion_duration = np.zeros(self.n_particles, dtype=np.float64)
        electron_duration[
            electron_injection.active_slots
        ] = electron_injection.time_inside_s
        ion_duration[
            ion_injection.active_slots
        ] = ion_injection.time_inside_s
        processed = int(np.count_nonzero(electron_duration > 0.0))
        processed += int(np.count_nonzero(ion_duration > 0.0))
        self.collision_counters["boundary_entry_particles_processed"] += processed
        collision_step = 2 * self.step_index + 1
        events = self._apply_collision_intervals(
            electron_duration,
            ion_duration,
            step_index=collision_step,
            stream_id=1,
        )
        self._process_ionization_descendants(
            events,
            electron_duration,
            step_index=collision_step,
            stream_id=1,
        )

    def _record_runtime_warning(self, key: str, message: str) -> None:
        if key in self._runtime_warning_keys:
            return
        self._runtime_warning_keys.add(key)
        self.runtime_warnings.append(message)
        warnings.warn(message, RuntimeWarning)

    def _runtime_limit(self, research_limit: float, production_limit: float) -> float:
        return production_limit if self.config.is_production else research_limit

    def _check_runtime_particle_motion(self) -> None:
        alive_e = (self.r_e > self.r_min) & (self.r_e < self.r_max)
        alive_i = (self.r_i > self.r_min) & (self.r_i < self.r_max)
        for values, alive, species, metric_name in (
            (
                self.vr_e,
                alive_e,
                "electron",
                "maximum_electron_radial_cell_crossing",
            ),
            (
                self.vr_i,
                alive_i,
                "ion",
                "maximum_ion_radial_cell_crossing",
            ),
        ):
            active_values = values[alive]
            if active_values.size and not np.all(np.isfinite(active_values)):
                self._stop_failed_run(
                    f"A live {species} has a nonfinite radial velocity."
                )
            crossing = (
                float(np.max(np.abs(active_values))) * self.dt / self.dr
                if active_values.size
                else 0.0
            )
            self.runtime_metrics[metric_name] = max(
                self.runtime_metrics[metric_name],
                crossing,
            )
            crossing_limit = self._runtime_limit(
                self.config.MAX_CFL,
                self.config.PRODUCTION_MAX_CFL,
            )
            if crossing > crossing_limit:
                message = (
                    f"The runtime {species} radial cell crossing is more than "
                    f"{crossing_limit:.3g}."
                )
                if self.config.is_production:
                    self._stop_failed_run(message)
                self._record_runtime_warning(
                    f"{species}_radial_cell_crossing",
                    message,
                )

    def _update_runtime_density_metrics(self) -> None:
        electron_density = np.maximum(
            0.0,
            -self.rho_e / self.config.e,
        )
        maximum_density = (
            float(np.max(electron_density))
            if electron_density.size
            else 0.0
        )
        self.runtime_metrics["maximum_electron_density_m3"] = max(
            self.runtime_metrics["maximum_electron_density_m3"],
            maximum_density,
        )
        if maximum_density <= 0.0:
            return

        local_debye = math.sqrt(
            self.config.epsilon_0
            * self.config.Te
            * self.config.e
            / (maximum_density * self.config.e * self.config.e)
        )
        local_cell_ratio = self.dr / local_debye
        local_dt_omega = self.dt * math.sqrt(
            maximum_density
            * self.config.e
            * self.config.e
            / (self.config.epsilon_0 * self.config.m_e)
        )
        self.runtime_metrics["maximum_local_cell_to_debye_ratio"] = max(
            self.runtime_metrics["maximum_local_cell_to_debye_ratio"],
            local_cell_ratio,
        )
        self.runtime_metrics["maximum_local_dt_omega_pe"] = max(
            self.runtime_metrics["maximum_local_dt_omega_pe"],
            local_dt_omega,
        )

        cell_limit = self._runtime_limit(
            self.config.MAX_CELL_TO_DEBYE_RATIO,
            self.config.PRODUCTION_MAX_CELL_TO_DEBYE_RATIO,
        )
        time_limit = self._runtime_limit(
            self.config.MAX_DT_OMEGA_PE,
            self.config.PRODUCTION_MAX_DT_OMEGA_PE,
        )
        for key, value, limit, label in (
            (
                "local_debye_resolution",
                local_cell_ratio,
                cell_limit,
                "runtime dr / local lambda_D",
            ),
            (
                "local_plasma_time_resolution",
                local_dt_omega,
                time_limit,
                "runtime dt * local omega_pe",
            ),
        ):
            if value > limit:
                message = f"The {label} value is more than {limit:.3g}."
                if self.config.is_production:
                    self._stop_failed_run(message)
                self._record_runtime_warning(key, message)

    def _active_particle_counts(self) -> tuple[int, int]:
        active_e = int(
            np.count_nonzero(
                (self.r_e > self.r_min) & (self.r_e < self.r_max)
            )
        )
        active_i = int(
            np.count_nonzero(
                (self.r_i > self.r_min) & (self.r_i < self.r_max)
            )
        )
        return active_e, active_i

    def numerical_diagnostics(self) -> dict[str, object]:
        active_e, active_i = self._active_particle_counts()
        expected_e = (
            self.particle_counters["initial_electrons"]
            + self.particle_counters["injected_electrons"]
            + self.collision_counters["secondary_electrons"]
            - self.particle_counters["probe_absorbed_electrons"]
            - self.particle_counters["wall_absorbed_electrons"]
        )
        expected_i = (
            self.particle_counters["initial_ions"]
            + self.particle_counters["injected_ions"]
            + self.collision_counters["secondary_ions"]
            - self.particle_counters["probe_absorbed_ions"]
            - self.particle_counters["wall_absorbed_ions"]
        )
        electron_residual = active_e - expected_e
        ion_residual = active_i - expected_i
        overflow_lookups = (
            self.collision_counters[
                "electron_energy_table_overflow_lookups"
            ]
            + self.collision_counters["ion_energy_table_overflow_lookups"]
        )
        warnings_present = (
            bool(self.config.stability_warnings())
            or bool(self.runtime_warnings)
            or bool(overflow_lookups)
        )
        if (
            self.failure_reason is not None
            or electron_residual != 0
            or ion_residual != 0
        ):
            status = "FAIL"
        elif warnings_present:
            status = "WARN"
        else:
            status = "PASS"
        return {
            "status": status,
            "active_electrons": active_e,
            "active_ions": active_i,
            "expected_electrons": expected_e,
            "expected_ions": expected_i,
            "electron_particle_ledger_residual": electron_residual,
            "ion_particle_ledger_residual": ion_residual,
            "energy_table_overflow_lookups": overflow_lookups,
            "stability_warnings": self.config.stability_warnings(),
            "runtime_warnings": list(self.runtime_warnings),
            "runtime_metrics": dict(sorted(self.runtime_metrics.items())),
            "failure_reason": self.failure_reason,
        }

    def _require_particle_ledger(self) -> None:
        diagnostics = self.numerical_diagnostics()
        if (
            diagnostics["electron_particle_ledger_residual"] != 0
            or diagnostics["ion_particle_ledger_residual"] != 0
        ):
            self._stop_failed_run("The macro-particle ledger is not balanced.")

    def _record_failure(
        self,
        error: Exception,
        *,
        mark_without_history: bool = False,
    ) -> None:
        error_type = type(error).__name__
        error_message = str(error)
        active_records = [
            record
            for record in self.execution_history
            if record.get("status") == "STARTED"
        ]
        if not active_records and not mark_without_history:
            return
        if self.failure_reason is None:
            self.failure_reason = f"{error_type}: {error_message}"
        for record in reversed(active_records):
            record["end_step_index"] = self.step_index
            record["status"] = "FAILED"
            record["error_type"] = error_type
            record["error_message"] = error_message

    def step(self) -> tuple[int, int]:
        """Advance one complete time step."""
        try:
            return self._step_impl()
        except Exception as error:
            self._record_failure(error, mark_without_history=True)
            raise

    def _step_impl(self) -> tuple[int, int]:
        if self.failure_reason is not None:
            raise RuntimeError(
                "This simulation stopped after an earlier failure: "
                f"{self.failure_reason}"
            )
        self._apply_collision_half_step(phase=0)

        radial_kick(
            self.r_e,
            self.vr_e,
            self.vt_e,
            self.E,
            -self.config.e,
            self.config.m_e,
            self.dt,
            self.r_min,
            self.r_max,
            self.dr,
        )
        radial_kick(
            self.r_i,
            self.vr_i,
            self.vt_i,
            self.E,
            self.config.ION_CHARGE_STATE * self.config.e,
            self.config.m_i,
            self.dt,
            self.r_min,
            self.r_max,
            self.dr,
        )
        self._check_runtime_particle_motion()
        e_hits, e_wall_hits = cylindrical_drift(
            self.r_e,
            self.vr_e,
            self.vt_e,
            self.vz_e,
            self.dt,
            self.r_min,
            self.r_max,
            self.reflect_wall,
        )
        i_hits, i_wall_hits = cylindrical_drift(
            self.r_i,
            self.vr_i,
            self.vt_i,
            self.vz_i,
            self.dt,
            self.r_min,
            self.r_max,
            self.reflect_wall,
        )
        self.particle_counters["probe_absorbed_electrons"] += e_hits
        self.particle_counters["probe_absorbed_ions"] += i_hits
        if not self.reflect_wall:
            self.particle_counters["wall_absorbed_electrons"] += e_wall_hits
            self.particle_counters["wall_absorbed_ions"] += i_wall_hits

        self._update_fields()
        radial_kick(
            self.r_e,
            self.vr_e,
            self.vt_e,
            self.E,
            -self.config.e,
            self.config.m_e,
            self.dt,
            self.r_min,
            self.r_max,
            self.dr,
        )
        radial_kick(
            self.r_i,
            self.vr_i,
            self.vt_i,
            self.E,
            self.config.ION_CHARGE_STATE * self.config.e,
            self.config.m_i,
            self.dt,
            self.r_min,
            self.r_max,
            self.dr,
        )
        self._apply_collision_half_step(phase=1)

        self._ensure_injection_capacity()
        electron_injection = self._inject_boundary_arrivals(
            self.r_e,
            self.vr_e,
            self.vt_e,
            self.vz_e,
            self.vth_e,
            self.inject_target_e,
            self.inject_residual_e,
            drift=0.0,
            charge=-self.config.e,
            mass=self.config.m_e,
        )
        ion_injection = self._inject_boundary_arrivals(
            self.r_i,
            self.vr_i,
            self.vt_i,
            self.vz_i,
            self.vth_i,
            self.inject_target_i,
            self.inject_residual_i,
            drift=self.ion_inject_drift,
            charge=self.config.ION_CHARGE_STATE * self.config.e,
            mass=self.config.m_i,
        )
        self.inject_residual_e = electron_injection.residual
        self.inject_residual_i = ion_injection.residual
        self.particle_counters[
            "injected_electrons"
        ] += electron_injection.crossing_events
        self.particle_counters[
            "injected_ions"
        ] += ion_injection.crossing_events
        self.particle_counters[
            "probe_absorbed_electrons"
        ] += electron_injection.probe_hits
        self.particle_counters[
            "probe_absorbed_ions"
        ] += ion_injection.probe_hits
        e_hits += electron_injection.probe_hits
        i_hits += ion_injection.probe_hits
        self._apply_boundary_entry_collisions(
            electron_injection,
            ion_injection,
        )
        self._update_fields()
        self._require_particle_ledger()
        self.step_index += 1
        return e_hits, i_hits

    def run(self, n_steps: int = 2000, n_warmup: int = 1000) -> SimulationResult:
        """Run one fixed-bias simulation."""
        try:
            return self._run_impl(n_steps=n_steps, n_warmup=n_warmup)
        except Exception as error:
            self._record_failure(error)
            raise

    def _run_impl(
        self,
        n_steps: int = 2000,
        n_warmup: int = 1000,
    ) -> SimulationResult:
        if isinstance(n_steps, bool) or not isinstance(n_steps, Integral):
            raise TypeError("Set n_steps to an integer.")
        if isinstance(n_warmup, bool) or not isinstance(n_warmup, Integral):
            raise TypeError("Set n_warmup to an integer.")
        if int(n_steps) < 1:
            raise ValueError("Set n_steps to an integer greater than zero.")
        if int(n_warmup) < 0 or int(n_warmup) >= int(n_steps):
            raise ValueError("Set n_warmup from zero through n_steps minus one.")

        history_record: dict[str, object] = {
            "operation": "run",
            "start_step_index": self.step_index,
            "n_steps": int(n_steps),
            "n_warmup": int(n_warmup),
            "status": "STARTED",
        }
        self.execution_history.append(history_record)
        current_values = np.empty(int(n_steps) - int(n_warmup))
        sample_index = 0
        for step_idx in range(int(n_steps)):
            e_hits, i_hits = self.step()
            if step_idx >= n_warmup:
                # Electron current is reported as positive magnitude.
                current_values[sample_index] = (
                    (e_hits * -self.qe) - (i_hits * self.qi)
                ) / self.dt
                sample_index += 1

        avg_current, current_sem, batch_count = batch_mean_statistics(
            current_values
        )
        avg_current *= self.probe_length
        current_sem *= self.probe_length

        self.ne[:] = -self.rho_e / self.config.e
        self.ni[:] = self.rho_i / self.config.e
        history_record["end_step_index"] = self.step_index
        history_record["status"] = "COMPLETE"

        return SimulationResult(
            avg_current=avg_current,
            avg_conventional_current=-avg_current,
            current_sem=current_sem,
            sample_count=int(current_values.size),
            batch_mean_count=batch_count,
            r_grid=self.r_grid.copy(),
            phi=self.phi.copy(),
            ne=self.ne.copy(),
            ni=self.ni.copy(),
            ne_raw=(-self.rho_e_raw / self.config.e).copy(),
            ni_raw=(self.rho_i_raw / self.config.e).copy(),
            ion_r=self.r_i.copy(),
            ion_vr=self.vr_i.copy(),
            ion_vz=self.vz_i.copy(),
        )

    def scan_voltage_range(
        self,
        v_start: float,
        v_end: float,
        n_steps: int,
        n_burn_in: int,
        n_sampling: int,
        n_initial_burn_in: int | None = None,
        ramp_steps: int = 0,
        progress_cb: Callable[[int, int, float], None] | None = None,
    ) -> dict[str, np.ndarray]:
        """Sweep the probe bias."""
        try:
            return self._scan_voltage_range_impl(
                v_start=v_start,
                v_end=v_end,
                n_steps=n_steps,
                n_burn_in=n_burn_in,
                n_sampling=n_sampling,
                n_initial_burn_in=n_initial_burn_in,
                ramp_steps=ramp_steps,
                progress_cb=progress_cb,
            )
        except Exception as error:
            self._record_failure(error)
            raise

    def _scan_voltage_range_impl(
        self,
        v_start: float,
        v_end: float,
        n_steps: int,
        n_burn_in: int,
        n_sampling: int,
        n_initial_burn_in: int | None = None,
        ramp_steps: int = 0,
        progress_cb: Callable[[int, int, float], None] | None = None,
    ) -> dict[str, np.ndarray]:
        """Sweep probe bias with warm start and return averaged I-V data."""
        for name, value, minimum in (
            ("n_steps", n_steps, 1),
            ("n_burn_in", n_burn_in, 0),
            ("n_sampling", n_sampling, 1),
            ("ramp_steps", ramp_steps, 0),
        ):
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f"Set {name} to an integer.")
            if int(value) < minimum:
                raise ValueError(
                    f"Set {name} to an integer not less than {minimum}."
                )
        for name, value in (("v_start", v_start), ("v_end", v_end)):
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"Set {name} to a real number.")
            if not math.isfinite(float(value)):
                raise ValueError(f"Set {name} to a finite value.")
        if progress_cb is not None and not callable(progress_cb):
            raise TypeError("Set progress_cb to a callable value or None.")

        if n_initial_burn_in is None:
            n_initial_burn_in = n_burn_in
        elif (
            isinstance(n_initial_burn_in, bool)
            or not isinstance(n_initial_burn_in, Integral)
        ):
            raise TypeError("Set n_initial_burn_in to an integer or None.")
        if int(n_initial_burn_in) < 0:
            raise ValueError(
                "Set n_initial_burn_in to an integer not less than zero."
            )

        n_voltage_points = int(n_steps)
        sampling_steps = int(n_sampling)
        voltages = np.linspace(float(v_start), float(v_end), n_voltage_points)
        history_record: dict[str, object] = {
            "operation": "voltage_scan",
            "start_step_index": self.step_index,
            "v_start": float(v_start),
            "v_end": float(v_end),
            "n_voltage_points": n_voltage_points,
            "n_burn_in": int(n_burn_in),
            "n_initial_burn_in": int(n_initial_burn_in),
            "n_sampling": sampling_steps,
            "ramp_steps": int(ramp_steps),
            "status": "STARTED",
        }
        self.execution_history.append(history_record)
        i_total = np.zeros(n_voltage_points)
        i_e = np.zeros(n_voltage_points)
        i_i = np.zeros(n_voltage_points)
        total_sem = np.zeros(n_voltage_points)
        electron_sem = np.zeros(n_voltage_points)
        ion_sem = np.zeros(n_voltage_points)
        batch_mean_counts = np.zeros(n_voltage_points, dtype=np.int64)

        for idx, v in enumerate(voltages):
            if idx > 0 and ramp_steps > 0:
                prev_v = float(voltages[idx - 1])
                for step_idx in range(int(ramp_steps)):
                    frac = (step_idx + 1) / ramp_steps
                    self.v_bias = prev_v + frac * (float(v) - prev_v)
                    self._update_fields()
                    self.step()

            self.v_bias = float(v)
            self._update_fields()

            # First step often needs longer to settle from initial conditions
            steps_to_burn = (
                int(n_initial_burn_in) if idx == 0 else int(n_burn_in)
            )
            for _ in range(steps_to_burn):
                self.step()

            electron_samples = np.empty(sampling_steps)
            ion_samples = np.empty(sampling_steps)
            for sample_index in range(sampling_steps):
                e_hits, i_hits = self.step()
                electron_samples[sample_index] = (
                    e_hits * -self.qe
                ) / self.dt
                ion_samples[sample_index] = (i_hits * self.qi) / self.dt

            total_samples = electron_samples - ion_samples
            i_e[idx], electron_sem[idx], electron_batches = (
                batch_mean_statistics(electron_samples)
            )
            i_i[idx], ion_sem[idx], ion_batches = batch_mean_statistics(
                ion_samples
            )
            i_total[idx], total_sem[idx], total_batches = (
                batch_mean_statistics(total_samples)
            )
            batch_mean_counts[idx] = min(
                electron_batches,
                ion_batches,
                total_batches,
            )

            if progress_cb is not None:
                progress_cb(idx + 1, n_voltage_points, float(v))

        if self.probe_length != 1.0:
            i_e *= self.probe_length
            i_i *= self.probe_length
            i_total *= self.probe_length
            electron_sem *= self.probe_length
            ion_sem *= self.probe_length
            total_sem *= self.probe_length

        history_record["end_step_index"] = self.step_index
        history_record["status"] = "COMPLETE"
        return {
            "voltages": voltages,
            "I_total": i_total,
            "I_conventional": -i_total,
            "I_electron": i_e,
            "I_ion": i_i,
            "I_total_sem": total_sem,
            "I_electron_sem": electron_sem,
            "I_ion_sem": ion_sem,
            "sample_count": np.full(
                n_voltage_points,
                sampling_steps,
                dtype=np.int64,
            ),
            "batch_mean_count": batch_mean_counts,
        }

    def state_sha256(self) -> str:
        """Calculate a hash of the complete mutable simulation state."""
        rng_state = _json_safe_value(self.rng.bit_generator.state)
        metadata = {
            "root_seed": self.root_seed,
            "step_index": self.step_index,
            "voltage_bias_v": self.v_bias,
            "injection_residual_e": self.inject_residual_e,
            "injection_residual_i": self.inject_residual_i,
            "collision_counters": self.collision_counters,
            "excitation_channel_counters": self.excitation_channel_counters,
            "ionization_channel_counters": self.ionization_channel_counters,
            "particle_counters": self.particle_counters,
            "runtime_metrics": self.runtime_metrics,
            "runtime_warnings": self.runtime_warnings,
            "execution_history": self.execution_history,
            "failure_reason": self.failure_reason,
            "numpy_rng_state": rng_state,
        }
        digest = sha256(canonical_json(metadata).encode("utf-8"))
        arrays = {
            "E": self.E,
            "phi": self.phi,
            "rho": self.rho,
            "rho_e": self.rho_e,
            "rho_e_raw": self.rho_e_raw,
            "rho_i": self.rho_i,
            "rho_i_raw": self.rho_i_raw,
            "ne": self.ne,
            "ni": self.ni,
            "r_e": self.r_e,
            "vr_e": self.vr_e,
            "vt_e": self.vt_e,
            "vz_e": self.vz_e,
            "r_i": self.r_i,
            "vr_i": self.vr_i,
            "vt_i": self.vt_i,
            "vz_i": self.vz_i,
        }
        for name, values in sorted(arrays.items()):
            contiguous = np.ascontiguousarray(values)
            digest.update(name.encode("ascii"))
            digest.update(contiguous.dtype.str.encode("ascii"))
            digest.update(canonical_json(contiguous.shape).encode("ascii"))
            digest.update(contiguous.view(np.uint8))
        return digest.hexdigest()

    def run_manifest(
        self,
        *,
        source_commit: str | None = None,
        created_utc: str | None = None,
    ) -> dict[str, object]:
        """Give the traceability data for this simulation."""
        root = Path(__file__).resolve().parents[1]
        manifest = build_run_manifest(
            self.config,
            root_seed=self.root_seed,
            source_files=self.cross_section_source_files,
            source_root=root,
            source_commit=source_commit,
            created_utc=created_utc,
        )
        manifest.pop("manifest_sha256", None)
        manifest["simulation"] = {
            "n_nominal_particles": self.n_nominal,
            "initial_electron_particles": self.initial_electron_count,
            "initial_ion_particles": self.initial_ion_count,
            "particle_capacity": self.n_particles,
            "probe_length_m": self.probe_length,
            "initial_voltage_bias_v": self.initial_v_bias,
            "step_index": self.step_index,
            "voltage_bias_v": self.v_bias,
            "reflect_wall": self.reflect_wall,
            "headroom_factor": self.headroom_factor,
            "fallback_charge_exchange_cross_section_m2": self.sigma_cex,
            "status": "FAILED" if self.failure_reason is not None else "READY",
            "failure_reason": self.failure_reason,
            "max_collision_events_per_particle": (
                self.max_collision_events_per_particle
            ),
            "max_secondary_buffer_bytes": self.max_secondary_buffer_bytes,
            "cross_section_model": self.cross_section_model,
            "cross_section_strict": self.cross_section_strict,
            "cross_section_inventory": self.cross_section_inventory,
            "ion_backscatter_mapping_requested": (
                self.ion_backscatter_mapping_requested
            ),
            "ion_backscatter_mapping": (
                "explicit symmetric ion-parent-neutral charge exchange"
                if self.ion_backscatter_mapping_applied
                else "not applied"
            ),
            "physics_limitations": [
                "boundary_entry_motion_uses_precollision_velocity",
                "external_validation_not_complete",
                "preview_coulomb_operator_not_release_model",
            ]
            + (
                ["constant_test_cross_sections"]
                if self.cross_section_model != "file_tables"
                else []
            ),
            "collision_counters": dict(sorted(self.collision_counters.items())),
            "electron_excitation_channels": {
                "thresholds_ev": self.en_excitation_thresholds_ev.tolist(),
                "event_counts": list(self.excitation_channel_counters),
            },
            "electron_ionization_channels": {
                "thresholds_ev": self.en_ionization_thresholds_ev.tolist(),
                "event_counts": list(self.ionization_channel_counters),
            },
            "particle_counters": dict(sorted(self.particle_counters.items())),
            "numerical_diagnostics": self.numerical_diagnostics(),
            "execution_history": _json_safe_value(self.execution_history),
            "numpy_rng_bit_generator": type(self.rng.bit_generator).__name__,
            "numpy_rng_state": _json_safe_value(self.rng.bit_generator.state),
            "state_sha256": self.state_sha256(),
            "current_sign_convention": {
                "I_electron": "positive electron-current magnitude",
                "I_ion": "positive ion current",
                "I_total": "I_electron - I_ion (legacy)",
                "I_conventional": "I_ion - I_electron",
            },
        }
        manifest["manifest_sha256"] = json_sha256(manifest)
        return manifest
