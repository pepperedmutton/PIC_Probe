from __future__ import annotations

import math
import warnings
from typing import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from core.collisions import perform_coulomb_scatter, perform_mcc_electron, perform_mcc_ion
from core.config import Config
from core.cross_sections import (
    build_constant_electron_tables,
    build_constant_ion_tables,
    build_electron_tables_from_lxcat,
    build_ion_tables_from_lxcat,
    load_cross_sections_from_custom_file,
)
from core.fields import compute_electric_field, solve_poisson_cylindrical, smooth_density_cylindrical
from core.particles import compute_shell_volumes, push_particles, weight_charge_cic


@dataclass
class SimulationResult:
    avg_current: float
    r_grid: np.ndarray
    phi: np.ndarray
    ne: np.ndarray
    ni: np.ndarray
    ion_r: np.ndarray
    ion_vr: np.ndarray


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
    ) -> None:
        self.config = config
        self.n_nominal = n_particles
        self.n_particles = int(n_particles * (1.0 + headroom_factor))
        self.v_bias = v_bias
        self.probe_length = probe_length
        self.reflect_wall = reflect_wall
        self.sigma_cex = sigma_cex

        for message in config.stability_warnings():
            warnings.warn(message, RuntimeWarning)

        if seed is not None:
            np.random.seed(seed)

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
        self.ne = np.zeros(self.n_nodes)
        self.ni = np.zeros(self.n_nodes)
        self.vol = np.zeros(self.n_nodes)
        compute_shell_volumes(self.r_min, self.dr, self.vol)

        self.r_e = np.zeros(self.n_particles)
        self.vr_e = np.zeros(self.n_particles)
        self.vt_e = np.zeros(self.n_particles)
        self.r_i = np.zeros(self.n_particles)
        self.vr_i = np.zeros(self.n_particles)
        self.vt_i = np.zeros(self.n_particles)

        self.vth_e = math.sqrt(config.e * config.Te / config.m_e)
        self.vth_i = math.sqrt(config.e * config.Ti / config.m_i)
        self.u_bohm = math.sqrt(config.e * max(config.Te, 0.1) / config.m_i)

        self.q_weight = self._compute_macro_weight()
        self.qe = -config.e * self.q_weight
        self.qi = config.e * self.q_weight
        self.qe_arr = np.full(self.n_particles, self.qe)
        self.qi_arr = np.full(self.n_particles, self.qi)

        self.n_g, self.vth_gas = self._compute_neutral_properties()
        self.inject_target_e = self._compute_injection_target_maxwellian(self.vth_e)
        self.ion_inject_drift = 0.0
        if config.ION_INJECTION_BOHM:
            self.ion_inject_drift = self.u_bohm
            self.inject_target_i = self._compute_injection_target_drift(self.u_bohm)
        else:
            self.inject_target_i = self._compute_injection_target_maxwellian(self.vth_i)
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

        self.ionized = np.zeros(self.n_particles, dtype=np.int8)
        self.sec_energy_ev = np.zeros(self.n_particles)

        self.nu_ei = self._compute_coulomb_frequency_ei() if self.enable_coulomb else 0.0
        self.nu_ii = self._compute_coulomb_frequency_ii() if self.enable_coulomb else 0.0

        self._initialize_particles()
        self._update_fields()

    def _load_cross_sections(self) -> None:
        """Load energy-dependent cross sections (LXCat) or fall back to constants."""
        target = self.config.CROSS_SECTION_TARGET

        def resolve_path(path_str: str) -> str:
            path = Path(path_str)
            if path.is_absolute():
                return str(path)
            root = Path(__file__).resolve().parents[1]
            return str(root / path)

        if self.config.LXCAT_ELECTRON_FILE == "CS.txt" and self.config.LXCAT_ION_FILE == "CS.txt":
            # High-priority custom loading for CS.txt
            electron_tables, ion_tables = load_cross_sections_from_custom_file(
                resolve_path("CS.txt"),
                e_max=self.config.EN_CS_E_MAX,
                n_bins=self.config.EN_CS_N,
            )
        else:
            if self.config.LXCAT_ELECTRON_FILE:
                electron_tables = build_electron_tables_from_lxcat(
                    resolve_path(self.config.LXCAT_ELECTRON_FILE),
                    target=target,
                    e_max=self.config.EN_CS_E_MAX,
                    n_bins=self.config.EN_CS_N,
                )
            else:
                electron_tables = build_constant_electron_tables(
                    self.config.SIGMA_EN_ELASTIC,
                    self.config.SIGMA_EN_EXC,
                    self.config.SIGMA_EN_ION,
                    e_max=self.config.EN_CS_E_MAX,
                    n_bins=self.config.EN_CS_N,
                )

            if self.config.LXCAT_ION_FILE:
                ion_tables = build_ion_tables_from_lxcat(
                    resolve_path(self.config.LXCAT_ION_FILE),
                    target=target,
                    e_max=self.config.ION_CS_E_MAX,
                    n_bins=self.config.ION_CS_N,
                    fallback_cex=self.sigma_cex,
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
        p_pa = self.config.P_Torr * 133.322368
        t_gas_ev = 0.026
        t_gas_k = t_gas_ev * self.config.e / self.config.k_B
        n_g = p_pa / (self.config.k_B * t_gas_k)
        v_th = math.sqrt(self.config.k_B * t_gas_k / self.config.m_i)
        return n_g, v_th

    def _compute_injection_target_maxwellian(self, vth: float) -> float:
        flux = self.config.N0 * vth / math.sqrt(2.0 * math.pi)
        boundary_area = 2.0 * math.pi * self.r_max
        n_phys = flux * boundary_area * self.dt
        return n_phys / self.q_weight

    def _compute_injection_target_drift(self, drift: float) -> float:
        flux = self.config.N0 * max(drift, 0.0)
        boundary_area = 2.0 * math.pi * self.r_max
        n_phys = flux * boundary_area * self.dt
        return n_phys / self.q_weight

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

    def _sample_positions_from_density(self, n_profile: np.ndarray, n_samples: int) -> np.ndarray:
        weights = n_profile * self.r_grid
        cdf = np.cumsum(weights)
        total = cdf[-1] if cdf[-1] > 0.0 else 1.0
        cdf /= total
        u = np.random.random(n_samples)
        return np.interp(u, cdf, self.r_grid)

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

        self.r_e[:self.n_nominal] = self._sample_positions_from_density(n_e, self.n_nominal)
        self.r_i[:self.n_nominal] = self._sample_positions_from_density(n_i, self.n_nominal)


        self.vr_e[:] = np.random.normal(0.0, self.vth_e, self.n_particles)
        self.vt_e[:] = np.random.normal(0.0, self.vth_e, self.n_particles)
        self.vr_i[:] = np.random.normal(0.0, self.vth_i, self.n_particles)
        self.vt_i[:] = np.random.normal(0.0, self.vth_i, self.n_particles)

    def _update_fields(self) -> None:
        weight_charge_cic(self.r_e, self.qe_arr, self.r_min, self.dr, self.rho_e, self.vol)
        weight_charge_cic(self.r_i, self.qi_arr, self.r_min, self.dr, self.rho_i, self.vol)
        self.rho[:] = self.rho_e + self.rho_i
        
        if self.config.SMOOTH_DENSITY:
            smooth_density_cylindrical(self.rho, self.config.N_SMOOTHING_PASSES)
            
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

    def inject_particles(
        self,
        r: np.ndarray,
        vr: np.ndarray,
        vt: np.ndarray,
        vth: float,
        target_per_step: float,
        residual: float,
        drift: float = 0.0,
    ) -> float:
        dead_idx = np.flatnonzero((r <= self.r_min) | (r >= self.r_max))
        n_dead = int(dead_idx.size)
        if n_dead == 0:
            return residual

        target_total = target_per_step + residual
        n_inject = int(target_total)
        residual = target_total - n_inject
        if n_inject <= 0:
            return residual

        if n_inject > n_dead:
            n_inject = n_dead
            residual = 0.0

        if n_inject < n_dead:
            pick = np.random.permutation(n_dead)[:n_inject]
            idx = dead_idx[pick]
        else:
            idx = dead_idx

        r[idx] = self.r_max - 0.5 * self.dr * np.random.random(n_inject)
        u = np.random.random(n_inject)
        u = np.clip(u, 1.0e-12, 1.0 - 1.0e-12)
        vr_in = drift + vth * np.sqrt(-2.0 * np.log(u))
        vr[idx] = -vr_in
        vt[idx] = np.random.normal(0.0, vth, n_inject)
        return residual

    def step(self) -> tuple[int, int]:
        self.inject_residual_e = self.inject_particles(
            self.r_e,
            self.vr_e,
            self.vt_e,
            self.vth_e,
            self.inject_target_e,
            self.inject_residual_e,
            drift=0.0,
        )
        self.inject_residual_i = self.inject_particles(
            self.r_i,
            self.vr_i,
            self.vt_i,
            self.vth_i,
            self.inject_target_i,
            self.inject_residual_i,
            drift=self.ion_inject_drift,
        )

        e_hits, _ = push_particles(
            self.r_e,
            self.vr_e,
            self.vt_e,
            self.E,
            self.qe / self.q_weight,
            self.config.m_e,
            self.dt,
            self.r_min,
            self.r_max,
            self.dr,
            self.reflect_wall,
        )

        i_hits, _ = push_particles(
            self.r_i,
            self.vr_i,
            self.vt_i,
            self.E,
            self.qi / self.q_weight,
            self.config.m_i,
            self.dt,
            self.r_min,
            self.r_max,
            self.dr,
            self.reflect_wall,
        )

        perform_mcc_electron(
            self.r_e,
            self.vr_e,
            self.vt_e,
            self.r_min,
            self.r_max,
            self.n_g,
            self.en_sigma_elastic,
            self.en_sigma_exc,
            self.en_sigma_ion,
            self.dt,
            self.config.m_e,
            self.e_exc_j,
            self.e_ion_j,
            self.en_e_min,
            self.en_inv_de,
            self.config.e,
            self.ionized,
            self.sec_energy_ev,
        )

        if self.enable_secondaries:
            self._spawn_ionization_secondaries()

        perform_mcc_ion(
            self.r_i,
            self.vr_i,
            self.vt_i,
            self.r_min,
            self.r_max,
            self.n_g,
            self.ion_sigma_cex,
            self.ion_sigma_elastic,
            self.dt,
            self.vth_gas,
            self.ion_e_min,
            self.ion_inv_de,
            self.config.e,
            self.config.m_i,
        )

        if self.enable_coulomb:
            perform_coulomb_scatter(
                self.r_e,
                self.vr_e,
                self.vt_e,
                self.r_min,
                self.r_max,
                self.nu_ei,
                self.dt,
            )
            perform_coulomb_scatter(
                self.r_i,
                self.vr_i,
                self.vt_i,
                self.r_min,
                self.r_max,
                self.nu_ii,
                self.dt,
            )

        self._update_fields()
        return e_hits, i_hits

    def _spawn_ionization_secondaries(self) -> None:
        ionized_idx = np.flatnonzero(self.ionized)
        if ionized_idx.size == 0:
            return

        dead_e = np.flatnonzero((self.r_e <= self.r_min) | (self.r_e >= self.r_max))
        dead_i = np.flatnonzero((self.r_i <= self.r_min) | (self.r_i >= self.r_max))
        n_new = min(ionized_idx.size, dead_e.size, dead_i.size)
        if n_new <= 0:
            return

        if ionized_idx.size > n_new:
            ionized_idx = ionized_idx[np.random.permutation(ionized_idx.size)[:n_new]]
        if dead_e.size > n_new:
            dead_e = dead_e[np.random.permutation(dead_e.size)[:n_new]]
        if dead_i.size > n_new:
            dead_i = dead_i[np.random.permutation(dead_i.size)[:n_new]]

        r_new = self.r_e[ionized_idx]
        self.r_e[dead_e] = r_new
        self.r_i[dead_i] = r_new

        sec_energy = self.sec_energy_ev[ionized_idx]
        speed = np.sqrt(2.0 * self.config.e * sec_energy / self.config.m_e)
        angles = 2.0 * math.pi * np.random.random(n_new)
        self.vr_e[dead_e] = speed * np.cos(angles)
        self.vt_e[dead_e] = speed * np.sin(angles)

        self.vr_i[dead_i] = np.random.normal(0.0, self.vth_gas, n_new)
        self.vt_i[dead_i] = np.random.normal(0.0, self.vth_gas, n_new)

    def run(self, n_steps: int = 2000, n_warmup: int = 1000) -> SimulationResult:
        current_sum = 0.0
        current_samples = 0

        for step_idx in range(n_steps):
            e_hits, i_hits = self.step()
            if step_idx >= n_warmup:
                # Electron current is reported as positive magnitude.
                current_sum += ((e_hits * -self.qe) - (i_hits * self.qi)) / self.dt
                current_samples += 1

        denom = float(current_samples) if current_samples > 0 else 1.0
        avg_current = current_sum / denom
        avg_current *= self.probe_length

        self.ne[:] = -self.rho_e / self.config.e
        self.ni[:] = self.rho_i / self.config.e

        return SimulationResult(
            avg_current=avg_current,
            r_grid=self.r_grid.copy(),
            phi=self.phi.copy(),
            ne=self.ne.copy(),
            ni=self.ni.copy(),
            ion_r=self.r_i.copy(),
            ion_vr=self.vr_i.copy(),
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
        """Sweep probe bias with warm start and return averaged I-V data."""
        if n_steps < 1:
            raise ValueError("n_steps must be >= 1")
        if ramp_steps < 0:
            raise ValueError("ramp_steps must be >= 0")

        if n_initial_burn_in is None:
            n_initial_burn_in = n_burn_in

        voltages = np.linspace(v_start, v_end, n_steps)
        i_total = np.zeros(n_steps)
        i_e = np.zeros(n_steps)
        i_i = np.zeros(n_steps)

        for idx, v in enumerate(voltages):
            if idx > 0 and ramp_steps > 0:
                prev_v = float(voltages[idx - 1])
                for step_idx in range(ramp_steps):
                    frac = (step_idx + 1) / ramp_steps
                    self.v_bias = prev_v + frac * (float(v) - prev_v)
                    self._update_fields()
                    self.step()

            self.v_bias = float(v)
            self._update_fields()

            # First step often needs longer to settle from initial conditions
            steps_to_burn = n_initial_burn_in if idx == 0 else n_burn_in
            for _ in range(steps_to_burn):
                self.step()

            acc_e = 0.0
            acc_i = 0.0
            sample_count = n_sampling if n_sampling > 0 else 1
            for _ in range(n_sampling):
                e_hits, i_hits = self.step()
                acc_e += (e_hits * -self.qe) / self.dt
                acc_i += (i_hits * self.qi) / self.dt

            # Electron current is reported as positive magnitude.
            i_e[idx] = acc_e / sample_count
            i_i[idx] = acc_i / sample_count
            i_total[idx] = i_e[idx] - i_i[idx]

            if progress_cb is not None:
                progress_cb(idx + 1, n_steps, v)

        if self.probe_length != 1.0:
            i_e *= self.probe_length
            i_i *= self.probe_length
            i_total *= self.probe_length

        return {
            "voltages": voltages,
            "I_total": i_total,
            "I_electron": i_e,
            "I_ion": i_i,
        }
