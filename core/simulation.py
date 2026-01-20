from __future__ import annotations

import math
import warnings
from typing import Callable
from dataclasses import dataclass

import numpy as np

from core.collisions import perform_mcc_electron, perform_mcc_ion
from core.config import Config
from core.fields import compute_electric_field, solve_poisson_cylindrical
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
    ) -> None:
        self.config = config
        self.n_particles = n_particles
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

        self.r_e = np.zeros(n_particles)
        self.vr_e = np.zeros(n_particles)
        self.vt_e = np.zeros(n_particles)
        self.r_i = np.zeros(n_particles)
        self.vr_i = np.zeros(n_particles)
        self.vt_i = np.zeros(n_particles)

        self.vth_e = math.sqrt(config.e * config.Te / config.m_e)
        self.vth_i = math.sqrt(config.e * config.Ti / config.m_i)
        self.u_bohm = math.sqrt(config.e * max(config.Te, 0.1) / config.m_i)

        self.q_weight = self._compute_macro_weight()
        self.qe = -config.e * self.q_weight
        self.qi = config.e * self.q_weight
        self.qe_arr = np.full(n_particles, self.qe)
        self.qi_arr = np.full(n_particles, self.qi)

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

        self._initialize_particles()
        self._update_fields()

    def _compute_macro_weight(self) -> float:
        area = math.pi * (self.r_max * self.r_max - self.r_min * self.r_min)
        return self.config.N0 * area / float(self.n_particles)

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

    def _sample_positions_from_density(self, n_profile: np.ndarray) -> np.ndarray:
        weights = n_profile * self.r_grid
        cdf = np.cumsum(weights)
        total = cdf[-1] if cdf[-1] > 0.0 else 1.0
        cdf /= total
        u = np.random.random(self.n_particles)
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

        self.r_e[:] = self._sample_positions_from_density(n_e)
        self.r_i[:] = self._sample_positions_from_density(n_i)

        self.vr_e[:] = np.random.normal(0.0, self.vth_e, self.n_particles)
        self.vt_e[:] = np.random.normal(0.0, self.vth_e, self.n_particles)
        self.vr_i[:] = np.random.normal(0.0, self.vth_i, self.n_particles)
        self.vt_i[:] = np.random.normal(0.0, self.vth_i, self.n_particles)

    def _update_fields(self) -> None:
        weight_charge_cic(self.r_e, self.qe_arr, self.r_min, self.dr, self.rho_e, self.vol)
        weight_charge_cic(self.r_i, self.qi_arr, self.r_min, self.dr, self.rho_i, self.vol)
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

        if self.sigma_en_elastic > 0.0 or self.sigma_en_exc > 0.0 or self.sigma_en_ion > 0.0:
            perform_mcc_electron(
                self.r_e,
                self.vr_e,
                self.vt_e,
                self.r_min,
                self.r_max,
                self.n_g,
                self.sigma_en_elastic,
                self.sigma_en_exc,
                self.sigma_en_ion,
                self.dt,
                self.config.m_e,
                self.e_exc_j,
                self.e_ion_j,
            )

        perform_mcc_ion(
            self.r_i,
            self.vr_i,
            self.vt_i,
            self.r_min,
            self.r_max,
            self.n_g,
            self.sigma_cex,
            self.dt,
            self.vth_gas,
        )

        self._update_fields()
        return e_hits, i_hits

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
        progress_cb: Callable[[int, int, float], None] | None = None,
    ) -> dict[str, np.ndarray]:
        """Sweep probe bias with warm start and return averaged I-V data."""
        if n_steps < 1:
            raise ValueError("n_steps must be >= 1")

        voltages = np.linspace(v_start, v_end, n_steps)
        i_total = np.zeros(n_steps)
        i_e = np.zeros(n_steps)
        i_i = np.zeros(n_steps)

        for idx, v in enumerate(voltages):
            self.v_bias = float(v)
            self._update_fields()

            for _ in range(n_burn_in):
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
