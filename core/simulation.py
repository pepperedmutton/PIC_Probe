from __future__ import annotations

import math
import warnings
from typing import Callable
from dataclasses import dataclass

import numpy as np

from core.collisions import perform_mcc_electron, perform_mcc_ion
from core.config import Config
from core.fields import compute_electric_field, solve_poisson_cylindrical, smooth_density_cylindrical
from core.gpu_backend import build_cuda_backend_or_none
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
        self.wall_bc_mode = str(config.WALL_BC_MODE).strip().lower()
        if self.wall_bc_mode not in {"fixed", "floating_insulator", "bulk_plasma"}:
            raise ValueError(
                "WALL_BC_MODE must be 'fixed', 'floating_insulator', or 'bulk_plasma'"
            )
        self.outer_injection_mode = str(config.OUTER_INJECTION_MODE).strip().lower()
        if self.outer_injection_mode not in {"flux", "fill_dead"}:
            raise ValueError("OUTER_INJECTION_MODE must be 'flux' or 'fill_dead'")
        if self.wall_bc_mode == "floating_insulator" and self.reflect_wall:
            warnings.warn(
                "reflect_wall=True is incompatible with floating insulator wall; forcing absorption.",
                RuntimeWarning,
            )
            self.reflect_wall = False

        self.adaptive_stability = config.ADAPTIVE_STABILITY
        self.stability_check_every = max(1, int(config.STABILITY_CHECK_EVERY))
        self.grid_refine_factor = max(2, int(config.GRID_REFINE_FACTOR))
        self.max_n_cells = max(4, int(config.MAX_N_CELLS))
        self.max_refinements_per_check = max(1, int(config.MAX_REFINEMENTS_PER_CHECK))
        self.dt_safety_factor = float(config.DT_SAFETY_FACTOR)
        self.cfl_safety_factor = float(config.CFL_SAFETY_FACTOR)
        self.cfl_percentile = float(config.CFL_PERCENTILE)
        self.omega_pe_limit = float(config.OMEGA_PE_LIMIT)
        self.min_dt = float(config.MIN_DT)
        self.use_cuda = bool(config.USE_CUDA)
        self.cuda_threads_per_block = max(32, int(config.CUDA_THREADS_PER_BLOCK))
        self.cuda_backend = None
        self._step_counter = 0

        if not self.adaptive_stability:
            for message in config.stability_warnings():
                warnings.warn(message, RuntimeWarning)

        if seed is None:
            seed = int(np.random.randint(0, 2**31 - 1))
        self.seed = int(seed)
        np.random.seed(self.seed)

        self.r_min = config.R_MIN
        self.r_max = config.R_MAX
        self.dt = config.DT
        self.dr = config.dr
        self.bulk_phi_ref = float(config.BULK_PHI_REF)
        self.bulk_dphi_dr = float(config.BULK_DPHI_DR)
        self.bulk_self_consistent_outer = bool(config.BULK_SELF_CONSISTENT_OUTER)
        self.wall_phi = float(config.V_WALL)
        if self.wall_bc_mode == "floating_insulator":
            self.wall_phi = float(config.WALL_PHI_INIT)
        elif self.wall_bc_mode == "bulk_plasma":
            self.wall_phi = self.bulk_phi_ref
        self.wall_phi_min = float(config.WALL_PHI_MIN)
        self.wall_phi_max = float(config.WALL_PHI_MAX)
        self.wall_charge = 0.0  # C/m
        self.wall_cap_per_length = float(config.WALL_CAPACITANCE_PER_LENGTH)
        self.enable_outer_injection = bool(config.OUTER_BOUNDARY_INJECTION)
        if self.wall_bc_mode == "floating_insulator" and config.AUTO_DISABLE_INJECTION_FOR_INSULATOR:
            self.enable_outer_injection = False

        self.n_nodes = config.N_CELLS + 1
        self.r_grid = self.r_min + self.dr * np.arange(self.n_nodes)
        self.wall_cap_per_length = self._compute_wall_capacitance_per_length()

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
        if self.use_cuda:
            self.cuda_backend = build_cuda_backend_or_none(
                n_particles=self.n_particles,
                n_nodes=self.n_nodes,
                qe_arr=self.qe_arr,
                qi_arr=self.qi_arr,
                vol=self.vol,
                seed=self.seed,
                threads_per_block=self.cuda_threads_per_block,
            )
            if self.cuda_backend is None:
                self.use_cuda = False

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
        self.enable_ionization_secondaries = bool(config.ENABLE_IONIZATION_SECONDARIES)
        self.ionization_secondaries_in_bulk_only = bool(config.IONIZATION_SECONDARIES_IN_BULK_ONLY)
        self.max_ionization_pairs_per_step = max(0, int(config.MAX_IONIZATION_PAIRS_PER_STEP))
        self.vth_secondary_e = math.sqrt(
            config.e * max(float(config.SECONDARY_E_EV), 1.0e-3) / config.m_e
        )
        self.vth_secondary_i = math.sqrt(
            config.e * max(float(config.SECONDARY_I_EV), 1.0e-3) / config.m_i
        )
        self.enable_bulk_feedback_source = bool(config.ENABLE_BULK_FEEDBACK_SOURCE)
        self.bulk_source_r_frac_min = float(config.BULK_SOURCE_R_FRAC_MIN)
        self.bulk_source_gain = max(0.0, float(config.BULK_SOURCE_GAIN))
        self.bulk_source_max_pairs_per_step = max(0, int(config.BULK_SOURCE_MAX_PAIRS_PER_STEP))
        self.bulk_source_rel_deadband = max(0.0, float(config.BULK_SOURCE_REL_DEADBAND))

        self._initialize_particles()
        if self.use_cuda and self.cuda_backend is not None:
            self.cuda_backend.upload_particles_from_host(
                self.r_e,
                self.vr_e,
                self.vt_e,
                self.r_i,
                self.vr_i,
                self.vt_i,
            )
        self._apply_adaptive_stability(force=True, refresh_fields=False)
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

    def _compute_wall_capacitance_per_length(self) -> float:
        if self.config.WALL_CAPACITANCE_PER_LENGTH > 0.0:
            return float(self.config.WALL_CAPACITANCE_PER_LENGTH)
        # Simple local sheath-capacitance estimate: C' ≈ epsilon_0 * A'/dr, A' = 2*pi*R_MAX.
        dr_eff = max(self.dr, 1.0e-9)
        return self.config.epsilon_0 * (2.0 * math.pi * self.r_max) / dr_eff

    def _outer_reference_potential(self) -> float:
        if self.wall_bc_mode == "bulk_plasma":
            return self.bulk_phi_ref
        return self.wall_phi

    def _estimate_sheath_width(self) -> float:
        lambda_d = self.config.debye_length()
        te = max(self.config.Te, 0.1)
        phi_outer_ref = self._outer_reference_potential()
        scale = abs(self.v_bias - phi_outer_ref) / te
        if scale < 1.0:
            scale = 1.0
        width = 5.0 * lambda_d * math.sqrt(scale)
        width = max(width, 5.0 * self.dr)
        return width

    def _initial_potential_profile(self) -> np.ndarray:
        width = self._estimate_sheath_width()
        phi_outer_ref = self._outer_reference_potential()
        phi = np.full(self.n_nodes, phi_outer_ref)
        if width <= 0.0:
            phi[0] = self.v_bias
            return phi

        cutoff = self.r_min + width
        mask = self.r_grid <= cutoff
        xi = (self.r_grid[mask] - self.r_min) / width
        phi[mask] = self.v_bias + (phi_outer_ref - self.v_bias) * np.power(xi, 4.0 / 3.0)
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
        phi_outer_ref = self._outer_reference_potential()
        phi_rel = phi_init - phi_outer_ref
        te = max(self.config.Te, 0.1)
        n0 = self.config.N0

        n_e = n0 * np.exp(phi_rel / te)
        n_e = np.clip(n_e, n0 * 1.0e-4, n0)

        u_b = math.sqrt(self.config.e * te / self.config.m_i)
        phi_drop = phi_outer_ref - phi_init
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
        use_bulk_bc = self.wall_bc_mode == "bulk_plasma"
        use_sc_outer = use_bulk_bc and self.bulk_self_consistent_outer
        v_outer = self.wall_phi if not use_bulk_bc else self.bulk_phi_ref
        if self.use_cuda and self.cuda_backend is not None:
            self.cuda_backend.deposit_charge_device(self.r_min, self.dr)
            self.cuda_backend.solve_fields_device(
                n_smoothing_passes=self.config.N_SMOOTHING_PASSES if self.config.SMOOTH_DENSITY else 0,
                r_min=self.r_min,
                dr=self.dr,
                epsilon_0=self.config.epsilon_0,
                v_bias=self.v_bias,
                v_outer=v_outer,
                outer_neumann=(use_bulk_bc and not use_sc_outer),
                outer_dphi_dr=self.bulk_dphi_dr,
                outer_self_consistent=use_sc_outer,
            )
        else:
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
                v_outer,
                outer_neumann=(use_bulk_bc and not use_sc_outer),
                outer_dphi_dr=self.bulk_dphi_dr,
                outer_self_consistent=use_sc_outer,
            )
            compute_electric_field(self.phi, self.dr, self.E)

    def _update_floating_wall_potential(self, e_wall_hits: int, i_wall_hits: int) -> None:
        if self.wall_bc_mode != "floating_insulator":
            return
        if e_wall_hits == 0 and i_wall_hits == 0:
            return
        cap = max(self.wall_cap_per_length, 1.0e-18)
        dq = (e_wall_hits * self.qe) + (i_wall_hits * self.qi)
        self.wall_charge += dq
        phi_new = self.config.WALL_PHI_INIT + self.wall_charge / cap
        if phi_new < self.wall_phi_min:
            phi_new = self.wall_phi_min
            self.wall_charge = (phi_new - self.config.WALL_PHI_INIT) * cap
        elif phi_new > self.wall_phi_max:
            phi_new = self.wall_phi_max
            self.wall_charge = (phi_new - self.config.WALL_PHI_INIT) * cap
        self.wall_phi = phi_new

    def _refresh_injection_targets(self) -> None:
        self.inject_target_e = self._compute_injection_target_maxwellian(self.vth_e)
        if self.config.ION_INJECTION_BOHM:
            self.inject_target_i = self._compute_injection_target_drift(self.ion_inject_drift)
        else:
            self.inject_target_i = self._compute_injection_target_maxwellian(self.vth_i)

    def _max_active_speed(self, r: np.ndarray, vr: np.ndarray, vt: np.ndarray) -> float:
        active = (r > self.r_min) & (r < self.r_max)
        if not np.any(active):
            return 0.0
        speed2 = vr[active] * vr[active] + vt[active] * vt[active]
        if speed2.size == 0:
            return 0.0
        speeds = np.sqrt(speed2)
        if speeds.size < 64:
            return float(np.max(speeds))
        p = min(100.0, max(50.0, self.cfl_percentile))
        return float(np.percentile(speeds, p))

    def _stability_metrics(self) -> dict[str, float]:
        lambda_d = max(self.config.debye_length(), 1.0e-18)
        omega_pe = max(self.config.plasma_frequency(), 1.0e-18)
        if self.use_cuda and self.cuda_backend is not None:
            observed_speed = max(
                self.cuda_backend.max_active_speed("e", self.r_min, self.r_max),
                self.cuda_backend.max_active_speed("i", self.r_min, self.r_max),
            )
            # Avoid pathological single-particle spikes forcing dt to MIN_DT on GPU path.
            observed_speed = min(observed_speed, 10.0 * self.vth_e)
            max_speed = max(
                self.vth_e,
                self.vth_i,
                observed_speed,
                1.0e-12,
            )
        else:
            max_speed = max(
                self.vth_e,
                self.vth_i,
                self._max_active_speed(self.r_e, self.vr_e, self.vt_e),
                self._max_active_speed(self.r_i, self.vr_i, self.vt_i),
                1.0e-12,
            )
        dr_safe = max(self.dr, 1.0e-18)
        dt_safe = max(self.dt, 0.0)

        return {
            "lambda_d": lambda_d,
            "omega_pe": omega_pe,
            "max_speed": max_speed,
            "dr_over_lambda": dr_safe / lambda_d,
            "dt_omega": dt_safe * omega_pe,
            "cfl": max_speed * dt_safe / dr_safe,
        }

    def _target_dt(self, omega_pe: float, max_speed: float) -> float:
        dt_plasma = self.omega_pe_limit / max(omega_pe, 1.0e-18)
        dt_cfl = self.cfl_safety_factor * self.dr / max(max_speed, 1.0e-12)
        dt_limit = min(dt_plasma, dt_cfl)
        dt_target = self.dt_safety_factor * dt_limit
        return max(dt_target, self.min_dt)

    def _required_n_cells_for_debye(self, lambda_d: float) -> int:
        domain = self.r_max - self.r_min
        if lambda_d <= 0.0:
            return self.n_nodes - 1
        return max(4, int(math.ceil(domain / lambda_d)))

    def _recast_fields_to_new_grid(self, new_n_cells: int) -> None:
        if self.use_cuda and self.cuda_backend is not None:
            self.cuda_backend.copy_fields_to_host(
                self.rho_e,
                self.rho_i,
                self.rho,
                self.phi,
                self.E,
            )
        old_r = self.r_grid
        old_phi = self.phi
        old_E = self.E
        old_rho = self.rho
        old_rho_e = self.rho_e
        old_rho_i = self.rho_i
        old_ne = self.ne
        old_ni = self.ni

        self.n_nodes = new_n_cells + 1
        self.dr = (self.r_max - self.r_min) / new_n_cells
        self.r_grid = self.r_min + self.dr * np.arange(self.n_nodes)

        if old_r.size > 1:
            self.phi = np.interp(self.r_grid, old_r, old_phi)
            self.E = np.interp(self.r_grid, old_r, old_E)
            self.rho = np.interp(self.r_grid, old_r, old_rho)
            self.rho_e = np.interp(self.r_grid, old_r, old_rho_e)
            self.rho_i = np.interp(self.r_grid, old_r, old_rho_i)
            self.ne = np.interp(self.r_grid, old_r, old_ne)
            self.ni = np.interp(self.r_grid, old_r, old_ni)
        else:
            self.phi = np.zeros(self.n_nodes)
            self.E = np.zeros(self.n_nodes)
            self.rho = np.zeros(self.n_nodes)
            self.rho_e = np.zeros(self.n_nodes)
            self.rho_i = np.zeros(self.n_nodes)
            self.ne = np.zeros(self.n_nodes)
            self.ni = np.zeros(self.n_nodes)

        self.vol = np.zeros(self.n_nodes)
        compute_shell_volumes(self.r_min, self.dr, self.vol)
        self.wall_cap_per_length = self._compute_wall_capacitance_per_length()
        if self.use_cuda and self.cuda_backend is not None:
            self.cuda_backend.reconfigure_grid(self.n_nodes, self.vol)

    def _apply_adaptive_stability(self, force: bool = False, refresh_fields: bool = True) -> None:
        if not self.adaptive_stability:
            return
        if not force and (self._step_counter % self.stability_check_every != 0):
            return

        grid_changed = False
        for _ in range(self.max_refinements_per_check):
            metrics = self._stability_metrics()
            violate_debye = metrics["dr_over_lambda"] > 1.0
            violate_dt_omega = metrics["dt_omega"] > self.omega_pe_limit
            violate_cfl = metrics["cfl"] > 1.0
            violated = violate_debye or violate_dt_omega or violate_cfl
            if not violated:
                break

            changed = False
            current_cells = self.n_nodes - 1
            if violate_debye and current_cells < self.max_n_cells:
                target_cells = max(
                    current_cells * self.grid_refine_factor,
                    self._required_n_cells_for_debye(metrics["lambda_d"]),
                )
                target_cells = min(target_cells, self.max_n_cells)
                if target_cells > current_cells:
                    self._recast_fields_to_new_grid(target_cells)
                    grid_changed = True
                    changed = True

            metrics = self._stability_metrics()
            dt_target = self._target_dt(metrics["omega_pe"], metrics["max_speed"])
            if self.dt > dt_target:
                self.dt = dt_target
                self._refresh_injection_targets()
                changed = True

            if not changed:
                warnings.warn(
                    "Adaptive stability reached refinement/time-step limits before satisfying all criteria.",
                    RuntimeWarning,
                )
                break

        if grid_changed and refresh_fields:
            self._update_fields()

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

    def _inject_particles_gpu(
        self,
        species: str,
        vth: float,
        target_per_step: float,
        residual: float,
        drift: float = 0.0,
    ) -> float:
        if self.cuda_backend is None:
            return residual
        target_total = target_per_step + residual
        n_inject = int(target_total)
        residual = target_total - n_inject
        if n_inject <= 0:
            return residual

        n_dead = self.cuda_backend.inject_species(
            species=species,
            vth=vth,
            drift=drift,
            n_inject=n_inject,
            r_min=self.r_min,
            r_max=self.r_max,
            dr=self.dr,
        )
        if n_inject > n_dead:
            residual = 0.0
        return residual

    def _spawn_pairs_cpu(
        self,
        n_pairs: int,
        r_lo: float,
        r_hi: float,
        vth_e: float,
        vth_i: float,
    ) -> int:
        n_req = max(0, int(n_pairs))
        if n_req <= 0:
            return 0

        r_lo = max(self.r_min + 1.0e-12, float(r_lo))
        r_hi = min(self.r_max - 1.0e-12, float(r_hi))
        if r_hi <= r_lo:
            return 0

        dead_e = np.flatnonzero((self.r_e <= self.r_min) | (self.r_e >= self.r_max))
        dead_i = np.flatnonzero((self.r_i <= self.r_min) | (self.r_i >= self.r_max))
        n_spawn = min(n_req, int(dead_e.size), int(dead_i.size))
        if n_spawn <= 0:
            return 0

        if n_spawn < dead_e.size:
            idx_e = dead_e[np.random.permutation(dead_e.size)[:n_spawn]]
        else:
            idx_e = dead_e
        if n_spawn < dead_i.size:
            idx_i = dead_i[np.random.permutation(dead_i.size)[:n_spawn]]
        else:
            idx_i = dead_i

        u = np.random.random(n_spawn)
        r_new = np.sqrt(r_lo * r_lo + (r_hi * r_hi - r_lo * r_lo) * u)
        r_new = np.clip(r_new, r_lo, r_hi)

        self.r_e[idx_e] = r_new
        self.r_i[idx_i] = r_new
        self.vr_e[idx_e] = np.random.normal(0.0, vth_e, n_spawn)
        self.vt_e[idx_e] = np.random.normal(0.0, vth_e, n_spawn)
        self.vr_i[idx_i] = np.random.normal(0.0, vth_i, n_spawn)
        self.vt_i[idx_i] = np.random.normal(0.0, vth_i, n_spawn)
        return n_spawn

    def _spawn_pairs_gpu(
        self,
        n_pairs: int,
        r_lo: float,
        r_hi: float,
        vth_e: float,
        vth_i: float,
    ) -> int:
        if self.cuda_backend is None:
            return 0
        n_req = max(0, int(n_pairs))
        if n_req <= 0:
            return 0
        r_lo = max(self.r_min + 1.0e-12, float(r_lo))
        r_hi = min(self.r_max - 1.0e-12, float(r_hi))
        if r_hi <= r_lo:
            return 0
        dead_e = self.cuda_backend.count_dead_species("e", self.r_min, self.r_max)
        dead_i = self.cuda_backend.count_dead_species("i", self.r_min, self.r_max)
        n_spawn = min(n_req, dead_e, dead_i)
        if n_spawn <= 0:
            return 0
        n_spawn_e = self.cuda_backend.fill_dead_species_in_region(
            species="e",
            vth=vth_e,
            n_fill=n_spawn,
            domain_min=self.r_min,
            domain_max=self.r_max,
            src_min=r_lo,
            src_max=r_hi,
        )
        n_spawn_i = self.cuda_backend.fill_dead_species_in_region(
            species="i",
            vth=vth_i,
            n_fill=n_spawn,
            domain_min=self.r_min,
            domain_max=self.r_max,
            src_min=r_lo,
            src_max=r_hi,
        )
        return min(n_spawn_e, n_spawn_i)

    def _spawn_pairs(
        self,
        n_pairs: int,
        r_lo: float,
        r_hi: float,
        vth_e: float,
        vth_i: float,
    ) -> int:
        if self.use_cuda and self.cuda_backend is not None:
            return self._spawn_pairs_gpu(n_pairs, r_lo, r_hi, vth_e, vth_i)
        return self._spawn_pairs_cpu(n_pairs, r_lo, r_hi, vth_e, vth_i)

    def _spawn_ionization_pairs(self, n_pairs: int) -> int:
        if not self.enable_ionization_secondaries:
            return 0
        capped = max(0, min(int(n_pairs), self.max_ionization_pairs_per_step))
        if capped <= 0:
            return 0
        if self.ionization_secondaries_in_bulk_only:
            r_lo, r_hi = self._bulk_source_bounds()
        else:
            r_lo, r_hi = self.r_min, self.r_max
        return self._spawn_pairs(
            n_pairs=capped,
            r_lo=r_lo,
            r_hi=r_hi,
            vth_e=self.vth_secondary_e,
            vth_i=self.vth_secondary_i,
        )

    def _bulk_source_bounds(self) -> tuple[float, float]:
        frac = min(0.98, max(0.0, self.bulk_source_r_frac_min))
        domain = self.r_max - self.r_min
        r_lo = self.r_min + frac * domain
        r_hi = self.r_max - 1.0e-12
        if r_hi <= r_lo:
            r_lo = self.r_max - max(self.dr, 1.0e-9)
            r_hi = self.r_max - 1.0e-12
        return r_lo, r_hi

    def _bulk_mean_density_counts(self, r_lo: float, r_hi: float) -> tuple[float, float]:
        area = math.pi * max(r_hi * r_hi - r_lo * r_lo, 1.0e-24)
        if self.use_cuda and self.cuda_backend is not None:
            cnt_e = self.cuda_backend.count_alive_species_in_region(
                "e", self.r_min, self.r_max, r_lo, r_hi
            )
            cnt_i = self.cuda_backend.count_alive_species_in_region(
                "i", self.r_min, self.r_max, r_lo, r_hi
            )
        else:
            cnt_e = int(np.sum((self.r_e > r_lo) & (self.r_e < r_hi)))
            cnt_i = int(np.sum((self.r_i > r_lo) & (self.r_i < r_hi)))
        n_e = (cnt_e * self.q_weight) / area
        n_i = (cnt_i * self.q_weight) / area
        return n_e, n_i

    def _apply_bulk_feedback_source(self) -> int:
        if not self.enable_bulk_feedback_source:
            return 0

        r_lo, r_hi = self._bulk_source_bounds()
        area = math.pi * max(r_hi * r_hi - r_lo * r_lo, 1.0e-24)
        n_e, n_i = self._bulk_mean_density_counts(r_lo, r_hi)
        n_bulk = 0.5 * (n_e + n_i)
        n_target = max(self.config.N0, 1.0)
        rel_deficit = (n_target - n_bulk) / n_target
        if rel_deficit <= self.bulk_source_rel_deadband:
            return 0

        target_macro = n_target * area / max(self.q_weight, 1.0e-24)
        current_macro = 0.5 * (n_e + n_i) * area / max(self.q_weight, 1.0e-24)
        deficit_macro = max(0.0, target_macro - current_macro)
        n_pairs = int(math.ceil(self.bulk_source_gain * deficit_macro))
        n_pairs = min(n_pairs, self.bulk_source_max_pairs_per_step)
        if n_pairs <= 0:
            return 0

        return self._spawn_pairs(
            n_pairs=n_pairs,
            r_lo=r_lo,
            r_hi=r_hi,
            vth_e=self.vth_e,
            vth_i=self.vth_i,
        )

    def _sync_device_state_to_host(self, include_particles: bool = False) -> None:
        if not (self.use_cuda and self.cuda_backend is not None):
            return
        self.cuda_backend.copy_fields_to_host(
            self.rho_e,
            self.rho_i,
            self.rho,
            self.phi,
            self.E,
        )
        if include_particles:
            self.cuda_backend.copy_particles_to_host(
                self.r_e,
                self.vr_e,
                self.vt_e,
                self.r_i,
            self.vr_i,
            self.vt_i,
        )

    def sync_state_from_device(self, include_particles: bool = False) -> None:
        """Public helper: sync GPU-resident state back to host arrays."""
        self._sync_device_state_to_host(include_particles=include_particles)

    def _step_gpu(self) -> tuple[int, int]:
        if self.cuda_backend is None:
            raise RuntimeError("CUDA step requested but backend is unavailable.")

        e_hits, e_wall_hits = self.cuda_backend.push_species(
            "e",
            (self.qe / self.q_weight) / self.config.m_e,
            self.dt,
            self.r_min,
            self.r_max,
            self.dr,
            self.reflect_wall,
        )
        i_hits, i_wall_hits = self.cuda_backend.push_species(
            "i",
            (self.qi / self.q_weight) / self.config.m_i,
            self.dt,
            self.r_min,
            self.r_max,
            self.dr,
            self.reflect_wall,
        )

        ionization_events = 0
        if self.sigma_en_elastic > 0.0 or self.sigma_en_exc > 0.0 or self.sigma_en_ion > 0.0:
            ionization_events = self.cuda_backend.collide_electrons(
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
        self.cuda_backend.collide_ions(
            self.r_min,
            self.r_max,
            self.n_g,
            self.sigma_cex,
            self.dt,
            self.vth_gas,
        )
        self._spawn_ionization_pairs(ionization_events)
        self._apply_bulk_feedback_source()

        self._update_floating_wall_potential(e_wall_hits, i_wall_hits)
        use_bulk_bc = self.wall_bc_mode == "bulk_plasma"
        use_sc_outer = use_bulk_bc and self.bulk_self_consistent_outer
        v_outer = self.wall_phi if not use_bulk_bc else self.bulk_phi_ref
        self.cuda_backend.deposit_charge_device(self.r_min, self.dr)
        self.cuda_backend.solve_fields_device(
            n_smoothing_passes=self.config.N_SMOOTHING_PASSES if self.config.SMOOTH_DENSITY else 0,
            r_min=self.r_min,
            dr=self.dr,
            epsilon_0=self.config.epsilon_0,
            v_bias=self.v_bias,
            v_outer=v_outer,
            outer_neumann=(use_bulk_bc and not use_sc_outer),
            outer_dphi_dr=self.bulk_dphi_dr,
            outer_self_consistent=use_sc_outer,
        )

        self._step_counter += 1
        return e_hits, i_hits

    def step(self) -> tuple[int, int]:
        self._apply_adaptive_stability()

        if self.enable_outer_injection:
            inject_target_e = self.inject_target_e
            inject_target_i = self.inject_target_i
            if self.outer_injection_mode == "fill_dead":
                # Keep the macro-particle inventory from drifting downward in long scans.
                inject_target_e = float(self.n_particles)
                inject_target_i = float(self.n_particles)
            if self.use_cuda and self.cuda_backend is not None:
                self.inject_residual_e = self._inject_particles_gpu(
                    species="e",
                    vth=self.vth_e,
                    target_per_step=inject_target_e,
                    residual=self.inject_residual_e,
                    drift=0.0,
                )
                self.inject_residual_i = self._inject_particles_gpu(
                    species="i",
                    vth=self.vth_i,
                    target_per_step=inject_target_i,
                    residual=self.inject_residual_i,
                    drift=self.ion_inject_drift,
                )
            else:
                self.inject_residual_e = self.inject_particles(
                    self.r_e,
                    self.vr_e,
                    self.vt_e,
                    self.vth_e,
                    inject_target_e,
                    self.inject_residual_e,
                    drift=0.0,
                )
                self.inject_residual_i = self.inject_particles(
                    self.r_i,
                    self.vr_i,
                    self.vt_i,
                    self.vth_i,
                    inject_target_i,
                    self.inject_residual_i,
                    drift=self.ion_inject_drift,
                )

        if self.use_cuda and self.cuda_backend is not None:
            return self._step_gpu()

        e_hits, e_wall_hits = push_particles(
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

        i_hits, i_wall_hits = push_particles(
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

        ionization_events = 0
        if self.sigma_en_elastic > 0.0 or self.sigma_en_exc > 0.0 or self.sigma_en_ion > 0.0:
            _, _, ionization_events = perform_mcc_electron(
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
        self._spawn_ionization_pairs(ionization_events)
        self._apply_bulk_feedback_source()

        self._update_floating_wall_potential(e_wall_hits, i_wall_hits)
        self._update_fields()
        self._step_counter += 1
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
        self._sync_device_state_to_host(include_particles=True)

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
