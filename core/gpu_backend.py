from __future__ import annotations

import math
import warnings

import numpy as np

try:
    from numba import cuda
    from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32
except Exception:  # pragma: no cover
    cuda = None
    create_xoroshiro128p_states = None
    xoroshiro128p_uniform_float32 = None


CUDA_AVAILABLE = bool(cuda is not None and cuda.is_available())


if cuda is not None:
    @cuda.jit(device=True)
    def _normal_sample(states, idx: int, mean: float, std: float) -> float:
        u1 = xoroshiro128p_uniform_float32(states, idx)
        if u1 < 1.0e-7:
            u1 = 1.0e-7
        u2 = xoroshiro128p_uniform_float32(states, idx)
        z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
        return mean + std * z


    @cuda.jit
    def _fill_zero_kernel(arr: np.ndarray) -> None:
        i = cuda.grid(1)
        if i < arr.shape[0]:
            arr[i] = 0.0


    @cuda.jit
    def _copy_array_kernel(src: np.ndarray, dst: np.ndarray) -> None:
        i = cuda.grid(1)
        if i < src.shape[0]:
            dst[i] = src[i]


    @cuda.jit
    def _sum_arrays_kernel(a: np.ndarray, b: np.ndarray, out: np.ndarray) -> None:
        i = cuda.grid(1)
        if i < out.shape[0]:
            out[i] = a[i] + b[i]


    @cuda.jit
    def _smooth_density_pass_kernel(src: np.ndarray, dst: np.ndarray) -> None:
        i = cuda.grid(1)
        n = src.shape[0]
        if i >= n:
            return
        if n < 3:
            dst[i] = src[i]
            return
        if i == 0:
            dst[0] = 0.666 * src[0] + 0.334 * src[1]
            return
        if i == n - 1:
            dst[n - 1] = 0.666 * src[n - 1] + 0.334 * src[n - 2]
            return
        dst[i] = 0.25 * src[i - 1] + 0.5 * src[i] + 0.25 * src[i + 1]


    @cuda.jit
    def _compute_electric_field_kernel(phi: np.ndarray, dr: float, e_out: np.ndarray) -> None:
        i = cuda.grid(1)
        n = phi.shape[0]
        if i >= n:
            return
        if n == 1:
            e_out[0] = 0.0
            return
        if i == 0:
            e_out[0] = -(phi[1] - phi[0]) / dr
            return
        if i == n - 1:
            e_out[n - 1] = -(phi[n - 1] - phi[n - 2]) / dr
            return
        e_out[i] = -(phi[i + 1] - phi[i - 1]) / (2.0 * dr)


    @cuda.jit
    def _solve_poisson_cylindrical_kernel(
        rho: np.ndarray,
        phi: np.ndarray,
        r_min: float,
        dr: float,
        epsilon_0: float,
        v_bias: float,
        v_wall: float,
        outer_neumann: bool,
        outer_dphi_dr: float,
        outer_self_consistent: bool,
        a: np.ndarray,
        b: np.ndarray,
        c: np.ndarray,
        d: np.ndarray,
    ) -> None:
        if cuda.grid(1) != 0:
            return

        n = rho.shape[0]
        if n == 0:
            return

        phi[0] = v_bias
        if n == 1:
            return
        if outer_self_consistent:
            phi[n - 1] = phi[0]
        elif outer_neumann:
            phi[n - 1] = phi[n - 2] + outer_dphi_dr * dr
        else:
            phi[n - 1] = v_wall
        if n == 2:
            return

        n_int = n - 2
        dr2 = dr * dr
        for j in range(1, n - 1):
            r_j = r_min + dr * j
            r_p = r_j + 0.5 * dr
            r_m = r_j - 0.5 * dr
            idx = j - 1
            a[idx] = r_m / (r_j * dr2)
            b[idx] = -(r_p + r_m) / (r_j * dr2)
            c[idx] = r_p / (r_j * dr2)
            d[idx] = -rho[j] / epsilon_0

        d[0] -= a[0] * phi[0]
        if outer_self_consistent:
            last = n_int - 1
            a[last] = a[last] - c[last]
            b[last] = b[last] + 2.0 * c[last]
            c[last] = 0.0
        elif outer_neumann:
            b[n_int - 1] += c[n_int - 1]
            d[n_int - 1] -= c[n_int - 1] * outer_dphi_dr * dr
        else:
            d[n_int - 1] -= c[n_int - 1] * phi[n - 1]

        for i in range(1, n_int):
            w = a[i] / b[i - 1]
            b[i] = b[i] - w * c[i - 1]
            d[i] = d[i] - w * d[i - 1]

        phi[n - 2] = d[n_int - 1] / b[n_int - 1]
        for i in range(n_int - 2, -1, -1):
            phi[i + 1] = (d[i] - c[i] * phi[i + 2]) / b[i]

        if outer_self_consistent:
            if n >= 3:
                phi[n - 1] = 2.0 * phi[n - 2] - phi[n - 3]
            else:
                phi[n - 1] = phi[n - 2]
        elif outer_neumann:
            phi[n - 1] = phi[n - 2] + outer_dphi_dr * dr


    @cuda.jit
    def _weight_charge_cic_kernel(
        r: np.ndarray,
        q: np.ndarray,
        r_min: float,
        dr: float,
        rho: np.ndarray,
    ) -> None:
        i = cuda.grid(1)
        if i >= r.shape[0]:
            return

        n_nodes = rho.shape[0]
        r_max = r_min + dr * (n_nodes - 1)
        ri = r[i]
        if ri <= r_min or ri >= r_max:
            return

        xi = (ri - r_min) / dr
        j = int(xi)
        if j < 0 or j >= n_nodes - 1:
            return

        w = xi - j
        qi = q[i]
        cuda.atomic.add(rho, j, qi * (1.0 - w))
        cuda.atomic.add(rho, j + 1, qi * w)


    @cuda.jit
    def _normalize_rho_kernel(rho: np.ndarray, vol: np.ndarray) -> None:
        j = cuda.grid(1)
        if j >= rho.shape[0]:
            return
        if vol[j] > 0.0:
            rho[j] = rho[j] / vol[j]
        else:
            rho[j] = 0.0


    @cuda.jit
    def _push_particles_kernel(
        r: np.ndarray,
        vr: np.ndarray,
        vt: np.ndarray,
        e_grid: np.ndarray,
        q_over_m: float,
        dt: float,
        r_min: float,
        r_max: float,
        dr: float,
        reflect_wall: bool,
        counts: np.ndarray,
    ) -> None:
        i = cuda.grid(1)
        if i >= r.shape[0]:
            return

        ri = r[i]
        if ri <= r_min or ri >= r_max:
            return

        n_nodes = e_grid.shape[0]
        xi = (ri - r_min) / dr
        j = int(xi)
        if j < 0:
            j = 0
        elif j >= n_nodes - 1:
            j = n_nodes - 2
        w = xi - j
        e_val = (1.0 - w) * e_grid[j] + w * e_grid[j + 1]

        vt_old = vt[i]
        a_r = q_over_m * e_val + (vt_old * vt_old) / ri
        vr_old = vr[i]
        r_new = ri + vr_old * dt + 0.5 * a_r * dt * dt

        r_span = r_max - r_min
        r_dead = r_max + r_span

        if r_new <= r_min:
            cuda.atomic.add(counts, 0, 1)
            r[i] = r_dead
            vr[i] = 0.0
            vt[i] = 0.0
            return

        if r_new >= r_max:
            cuda.atomic.add(counts, 1, 1)
            if reflect_wall:
                r_reflect = r_max - (r_new - r_max)
                if r_reflect < r_min:
                    r_reflect = r_min
                r_new = r_reflect
                if r_new > 0.0:
                    vt_new = vt_old * (ri / r_new)
                else:
                    vt_new = 0.0

                xi2 = (r_new - r_min) / dr
                j2 = int(xi2)
                if j2 < 0:
                    j2 = 0
                elif j2 >= n_nodes - 1:
                    j2 = n_nodes - 2
                w2 = xi2 - j2
                e2 = (1.0 - w2) * e_grid[j2] + w2 * e_grid[j2 + 1]
                a_r_new = q_over_m * e2 + (vt_new * vt_new) / r_new
                vr_new = -(vr_old + 0.5 * (a_r + a_r_new) * dt)

                r[i] = r_new
                vr[i] = vr_new
                vt[i] = vt_new
            else:
                r[i] = r_dead
                vr[i] = 0.0
                vt[i] = 0.0
            return

        vt_new = vt_old * (ri / r_new)
        xi2 = (r_new - r_min) / dr
        j2 = int(xi2)
        if j2 < 0:
            j2 = 0
        elif j2 >= n_nodes - 1:
            j2 = n_nodes - 2
        w2 = xi2 - j2
        e2 = (1.0 - w2) * e_grid[j2] + w2 * e_grid[j2 + 1]
        a_r_new = q_over_m * e2 + (vt_new * vt_new) / r_new
        vr_new = vr_old + 0.5 * (a_r + a_r_new) * dt

        r[i] = r_new
        vr[i] = vr_new
        vt[i] = vt_new


    @cuda.jit
    def _inject_particles_kernel(
        r: np.ndarray,
        vr: np.ndarray,
        vt: np.ndarray,
        r_min: float,
        r_max: float,
        dr: float,
        vth: float,
        drift: float,
        n_inject: int,
        rng_states: np.ndarray,
        dead_counter: np.ndarray,
    ) -> None:
        i = cuda.grid(1)
        if i >= r.shape[0]:
            return

        ri = r[i]
        if not (ri <= r_min or ri >= r_max):
            return

        slot = cuda.atomic.add(dead_counter, 0, 1)
        if slot >= n_inject:
            return

        # Place injected particles near outer boundary.
        ur = xoroshiro128p_uniform_float32(rng_states, i)
        r[i] = r_max - 0.5 * dr * ur

        u = xoroshiro128p_uniform_float32(rng_states, i)
        if u < 1.0e-7:
            u = 1.0e-7
        if u > 1.0 - 1.0e-7:
            u = 1.0 - 1.0e-7
        vr_in = drift + vth * math.sqrt(-2.0 * math.log(u))
        vr[i] = -vr_in
        vt[i] = _normal_sample(rng_states, i, 0.0, vth)


    @cuda.jit
    def _count_dead_kernel(
        r: np.ndarray,
        r_min: float,
        r_max: float,
        counter: np.ndarray,
    ) -> None:
        i = cuda.grid(1)
        if i >= r.shape[0]:
            return
        ri = r[i]
        if ri <= r_min or ri >= r_max:
            cuda.atomic.add(counter, 0, 1)


    @cuda.jit
    def _fill_dead_bulk_kernel(
        r: np.ndarray,
        vr: np.ndarray,
        vt: np.ndarray,
        domain_min: float,
        domain_max: float,
        src_min: float,
        src_max: float,
        vth: float,
        n_fill: int,
        rng_states: np.ndarray,
        dead_counter: np.ndarray,
    ) -> None:
        i = cuda.grid(1)
        if i >= r.shape[0]:
            return

        ri = r[i]
        if not (ri <= domain_min or ri >= domain_max):
            return

        slot = cuda.atomic.add(dead_counter, 0, 1)
        if slot >= n_fill:
            return

        # Uniform-in-area radial sampling for cylindrical geometry.
        u_area = xoroshiro128p_uniform_float32(rng_states, i)
        r_sq = src_min * src_min + (src_max * src_max - src_min * src_min) * u_area
        r_new = math.sqrt(r_sq)
        if r_new <= src_min:
            r_new = src_min + 1.0e-12
        if r_new >= src_max:
            r_new = src_max - 1.0e-12
        r[i] = r_new

        vr[i] = _normal_sample(rng_states, i, 0.0, vth)
        vt[i] = _normal_sample(rng_states, i, 0.0, vth)


    @cuda.jit
    def _count_alive_in_region_kernel(
        r: np.ndarray,
        domain_min: float,
        domain_max: float,
        r_lo: float,
        r_hi: float,
        counter: np.ndarray,
    ) -> None:
        i = cuda.grid(1)
        if i >= r.shape[0]:
            return
        ri = r[i]
        if ri <= domain_min or ri >= domain_max:
            return
        if ri >= r_lo and ri < r_hi:
            cuda.atomic.add(counter, 0, 1)


    @cuda.jit
    def _mcc_ion_kernel(
        r: np.ndarray,
        vr: np.ndarray,
        vt: np.ndarray,
        r_min: float,
        r_max: float,
        n_g: float,
        sigma: float,
        dt: float,
        v_th: float,
        rng_states: np.ndarray,
    ) -> None:
        i = cuda.grid(1)
        if i >= r.shape[0]:
            return
        ri = r[i]
        if ri <= r_min or ri >= r_max:
            return

        vi = math.sqrt(vr[i] * vr[i] + vt[i] * vt[i])
        p = 1.0 - math.exp(-n_g * sigma * vi * dt)
        u = xoroshiro128p_uniform_float32(rng_states, i)
        if u < p:
            vr[i] = _normal_sample(rng_states, i, 0.0, v_th)
            vt[i] = _normal_sample(rng_states, i, 0.0, v_th)


    @cuda.jit
    def _mcc_electron_kernel(
        r: np.ndarray,
        vr: np.ndarray,
        vt: np.ndarray,
        r_min: float,
        r_max: float,
        n_g: float,
        sigma_el: float,
        sigma_exc: float,
        sigma_ion: float,
        dt: float,
        m_e: float,
        e_exc: float,
        e_ion: float,
        rng_states: np.ndarray,
        ionization_counter: np.ndarray,
    ) -> None:
        i = cuda.grid(1)
        if i >= r.shape[0]:
            return
        ri = r[i]
        if ri <= r_min or ri >= r_max:
            return

        v2 = vr[i] * vr[i] + vt[i] * vt[i]
        if v2 <= 0.0:
            return
        v = math.sqrt(v2)
        energy = 0.5 * m_e * v2

        sigma_exc_eff = sigma_exc if energy > e_exc else 0.0
        sigma_ion_eff = sigma_ion if energy > e_ion else 0.0
        sigma_total = sigma_el + sigma_exc_eff + sigma_ion_eff
        if sigma_total <= 0.0:
            return

        p = 1.0 - math.exp(-n_g * sigma_total * v * dt)
        if xoroshiro128p_uniform_float32(rng_states, i) >= p:
            return

        pick = xoroshiro128p_uniform_float32(rng_states, i) * sigma_total
        angle = 2.0 * math.pi * xoroshiro128p_uniform_float32(rng_states, i)
        if pick < sigma_el:
            vr[i] = v * math.cos(angle)
            vt[i] = v * math.sin(angle)
        elif pick < sigma_el + sigma_exc_eff:
            energy_new = energy - e_exc
            if energy_new < 0.0:
                energy_new = 0.0
            v_new = math.sqrt(2.0 * energy_new / m_e) if energy_new > 0.0 else 0.0
            vr[i] = v_new * math.cos(angle)
            vt[i] = v_new * math.sin(angle)
        else:
            energy_new = energy - e_ion
            if energy_new < 0.0:
                energy_new = 0.0
            v_new = math.sqrt(2.0 * energy_new / m_e) if energy_new > 0.0 else 0.0
            vr[i] = v_new * math.cos(angle)
            vt[i] = v_new * math.sin(angle)
            cuda.atomic.add(ionization_counter, 0, 1)


    @cuda.jit
    def _max_speed_kernel(
        r: np.ndarray,
        vr: np.ndarray,
        vt: np.ndarray,
        r_min: float,
        r_max: float,
        max_speed_out: np.ndarray,
    ) -> None:
        i = cuda.grid(1)
        if i >= r.shape[0]:
            return
        ri = r[i]
        if ri <= r_min or ri >= r_max:
            return
        v = math.sqrt(vr[i] * vr[i] + vt[i] * vt[i])
        cuda.atomic.max(max_speed_out, 0, np.float32(v))


class CUDABackend:
    def __init__(
        self,
        n_particles: int,
        n_nodes: int,
        qe_arr: np.ndarray,
        qi_arr: np.ndarray,
        vol: np.ndarray,
        seed: int,
        threads_per_block: int = 256,
    ) -> None:
        if not CUDA_AVAILABLE:
            raise RuntimeError("CUDA backend requested but CUDA is not available.")
        if create_xoroshiro128p_states is None:
            raise RuntimeError("CUDA random module is unavailable.")

        self.n_particles = int(n_particles)
        self.threads_per_block = max(32, int(threads_per_block))
        self._particle_blocks = self._blocks_for(self.n_particles)

        self.d_r_e = cuda.device_array(self.n_particles, dtype=np.float64)
        self.d_vr_e = cuda.device_array(self.n_particles, dtype=np.float64)
        self.d_vt_e = cuda.device_array(self.n_particles, dtype=np.float64)
        self.d_r_i = cuda.device_array(self.n_particles, dtype=np.float64)
        self.d_vr_i = cuda.device_array(self.n_particles, dtype=np.float64)
        self.d_vt_i = cuda.device_array(self.n_particles, dtype=np.float64)

        self.d_qe = cuda.to_device(np.asarray(qe_arr, dtype=np.float64))
        self.d_qi = cuda.to_device(np.asarray(qi_arr, dtype=np.float64))

        self._counts_host = np.zeros(2, dtype=np.int32)
        self.d_counts_e = cuda.device_array(2, dtype=np.int32)
        self.d_counts_i = cuda.device_array(2, dtype=np.int32)

        self._counter_host = np.zeros(1, dtype=np.int32)
        self.d_dead_counter = cuda.device_array(1, dtype=np.int32)
        self.d_ionization_counter = cuda.device_array(1, dtype=np.int32)

        self._max_speed_host = np.zeros(1, dtype=np.float32)
        self.d_max_speed = cuda.device_array(1, dtype=np.float32)

        self.rng_states_e = create_xoroshiro128p_states(self.n_particles, seed=seed + 11)
        self.rng_states_i = create_xoroshiro128p_states(self.n_particles, seed=seed + 97)

        self.n_nodes = 0
        self._grid_blocks = 1
        self.d_e = None
        self.d_phi = None
        self.d_vol = None
        self.d_rho_e = None
        self.d_rho_i = None
        self.d_rho = None
        self.d_rho_tmp = None
        self.d_a = None
        self.d_b = None
        self.d_c = None
        self.d_d = None
        self.reconfigure_grid(n_nodes, vol)

    def _blocks_for(self, n: int) -> int:
        return max(1, (int(n) + self.threads_per_block - 1) // self.threads_per_block)

    def reconfigure_grid(self, n_nodes: int, vol: np.ndarray) -> None:
        n_nodes = int(n_nodes)
        if n_nodes <= 0:
            raise ValueError("n_nodes must be positive")
        if vol.shape[0] != n_nodes:
            raise ValueError("vol shape mismatch in CUDA grid reconfigure")

        if self.n_nodes != n_nodes:
            self.n_nodes = n_nodes
            self._grid_blocks = self._blocks_for(n_nodes)
            self.d_e = cuda.device_array(n_nodes, dtype=np.float64)
            self.d_phi = cuda.device_array(n_nodes, dtype=np.float64)
            self.d_vol = cuda.device_array(n_nodes, dtype=np.float64)
            self.d_rho_e = cuda.device_array(n_nodes, dtype=np.float64)
            self.d_rho_i = cuda.device_array(n_nodes, dtype=np.float64)
            self.d_rho = cuda.device_array(n_nodes, dtype=np.float64)
            self.d_rho_tmp = cuda.device_array(n_nodes, dtype=np.float64)
            self.d_a = cuda.device_array(n_nodes, dtype=np.float64)
            self.d_b = cuda.device_array(n_nodes, dtype=np.float64)
            self.d_c = cuda.device_array(n_nodes, dtype=np.float64)
            self.d_d = cuda.device_array(n_nodes, dtype=np.float64)

        self.d_vol.copy_to_device(np.asarray(vol, dtype=np.float64))

    def upload_particles_from_host(
        self,
        r_e: np.ndarray,
        vr_e: np.ndarray,
        vt_e: np.ndarray,
        r_i: np.ndarray,
        vr_i: np.ndarray,
        vt_i: np.ndarray,
    ) -> None:
        self.d_r_e.copy_to_device(np.asarray(r_e, dtype=np.float64))
        self.d_vr_e.copy_to_device(np.asarray(vr_e, dtype=np.float64))
        self.d_vt_e.copy_to_device(np.asarray(vt_e, dtype=np.float64))
        self.d_r_i.copy_to_device(np.asarray(r_i, dtype=np.float64))
        self.d_vr_i.copy_to_device(np.asarray(vr_i, dtype=np.float64))
        self.d_vt_i.copy_to_device(np.asarray(vt_i, dtype=np.float64))

    def copy_particles_to_host(
        self,
        r_e: np.ndarray,
        vr_e: np.ndarray,
        vt_e: np.ndarray,
        r_i: np.ndarray,
        vr_i: np.ndarray,
        vt_i: np.ndarray,
    ) -> None:
        self.d_r_e.copy_to_host(r_e)
        self.d_vr_e.copy_to_host(vr_e)
        self.d_vt_e.copy_to_host(vt_e)
        self.d_r_i.copy_to_host(r_i)
        self.d_vr_i.copy_to_host(vr_i)
        self.d_vt_i.copy_to_host(vt_i)

    def copy_fields_to_host(
        self,
        rho_e_host: np.ndarray,
        rho_i_host: np.ndarray,
        rho_host: np.ndarray,
        phi_host: np.ndarray,
        e_host: np.ndarray,
    ) -> None:
        self.d_rho_e.copy_to_host(rho_e_host)
        self.d_rho_i.copy_to_host(rho_i_host)
        self.d_rho.copy_to_host(rho_host)
        self.d_phi.copy_to_host(phi_host)
        self.d_e.copy_to_host(e_host)

    def push_species(
        self,
        species: str,
        q_over_m: float,
        dt: float,
        r_min: float,
        r_max: float,
        dr: float,
        reflect_wall: bool,
    ) -> tuple[int, int]:
        self._counts_host[:] = 0
        if species == "e":
            self.d_counts_e.copy_to_device(self._counts_host)
            _push_particles_kernel[self._particle_blocks, self.threads_per_block](
                self.d_r_e,
                self.d_vr_e,
                self.d_vt_e,
                self.d_e,
                q_over_m,
                dt,
                r_min,
                r_max,
                dr,
                reflect_wall,
                self.d_counts_e,
            )
            self.d_counts_e.copy_to_host(self._counts_host)
        elif species == "i":
            self.d_counts_i.copy_to_device(self._counts_host)
            _push_particles_kernel[self._particle_blocks, self.threads_per_block](
                self.d_r_i,
                self.d_vr_i,
                self.d_vt_i,
                self.d_e,
                q_over_m,
                dt,
                r_min,
                r_max,
                dr,
                reflect_wall,
                self.d_counts_i,
            )
            self.d_counts_i.copy_to_host(self._counts_host)
        else:
            raise ValueError("species must be 'e' or 'i'")
        return int(self._counts_host[0]), int(self._counts_host[1])

    def inject_species(
        self,
        species: str,
        vth: float,
        drift: float,
        n_inject: int,
        r_min: float,
        r_max: float,
        dr: float,
    ) -> int:
        if n_inject <= 0:
            return 0
        self._counter_host[0] = 0
        self.d_dead_counter.copy_to_device(self._counter_host)

        if species == "e":
            _inject_particles_kernel[self._particle_blocks, self.threads_per_block](
                self.d_r_e,
                self.d_vr_e,
                self.d_vt_e,
                r_min,
                r_max,
                dr,
                vth,
                drift,
                int(n_inject),
                self.rng_states_e,
                self.d_dead_counter,
            )
        elif species == "i":
            _inject_particles_kernel[self._particle_blocks, self.threads_per_block](
                self.d_r_i,
                self.d_vr_i,
                self.d_vt_i,
                r_min,
                r_max,
                dr,
                vth,
                drift,
                int(n_inject),
                self.rng_states_i,
                self.d_dead_counter,
            )
        else:
            raise ValueError("species must be 'e' or 'i'")

        self.d_dead_counter.copy_to_host(self._counter_host)
        return int(self._counter_host[0])

    def count_dead_species(self, species: str, r_min: float, r_max: float) -> int:
        self._counter_host[0] = 0
        self.d_dead_counter.copy_to_device(self._counter_host)
        if species == "e":
            _count_dead_kernel[self._particle_blocks, self.threads_per_block](
                self.d_r_e,
                r_min,
                r_max,
                self.d_dead_counter,
            )
        elif species == "i":
            _count_dead_kernel[self._particle_blocks, self.threads_per_block](
                self.d_r_i,
                r_min,
                r_max,
                self.d_dead_counter,
            )
        else:
            raise ValueError("species must be 'e' or 'i'")
        self.d_dead_counter.copy_to_host(self._counter_host)
        return int(self._counter_host[0])

    def fill_dead_species_in_region(
        self,
        species: str,
        vth: float,
        n_fill: int,
        domain_min: float,
        domain_max: float,
        src_min: float,
        src_max: float,
    ) -> int:
        if n_fill <= 0:
            return 0
        self._counter_host[0] = 0
        self.d_dead_counter.copy_to_device(self._counter_host)

        if species == "e":
            _fill_dead_bulk_kernel[self._particle_blocks, self.threads_per_block](
                self.d_r_e,
                self.d_vr_e,
                self.d_vt_e,
                domain_min,
                domain_max,
                src_min,
                src_max,
                vth,
                int(n_fill),
                self.rng_states_e,
                self.d_dead_counter,
            )
        elif species == "i":
            _fill_dead_bulk_kernel[self._particle_blocks, self.threads_per_block](
                self.d_r_i,
                self.d_vr_i,
                self.d_vt_i,
                domain_min,
                domain_max,
                src_min,
                src_max,
                vth,
                int(n_fill),
                self.rng_states_i,
                self.d_dead_counter,
            )
        else:
            raise ValueError("species must be 'e' or 'i'")

        self.d_dead_counter.copy_to_host(self._counter_host)
        return min(int(n_fill), int(self._counter_host[0]))

    def fill_dead_bulk_species(
        self,
        species: str,
        vth: float,
        n_fill: int,
        r_min: float,
        r_max: float,
    ) -> int:
        return self.fill_dead_species_in_region(
            species=species,
            vth=vth,
            n_fill=n_fill,
            domain_min=r_min,
            domain_max=r_max,
            src_min=r_min,
            src_max=r_max,
        )

    def count_alive_species_in_region(
        self,
        species: str,
        domain_min: float,
        domain_max: float,
        r_lo: float,
        r_hi: float,
    ) -> int:
        if r_hi <= r_lo:
            return 0
        self._counter_host[0] = 0
        self.d_dead_counter.copy_to_device(self._counter_host)
        if species == "e":
            _count_alive_in_region_kernel[self._particle_blocks, self.threads_per_block](
                self.d_r_e,
                domain_min,
                domain_max,
                r_lo,
                r_hi,
                self.d_dead_counter,
            )
        elif species == "i":
            _count_alive_in_region_kernel[self._particle_blocks, self.threads_per_block](
                self.d_r_i,
                domain_min,
                domain_max,
                r_lo,
                r_hi,
                self.d_dead_counter,
            )
        else:
            raise ValueError("species must be 'e' or 'i'")
        self.d_dead_counter.copy_to_host(self._counter_host)
        return int(self._counter_host[0])

    def collide_ions(
        self,
        r_min: float,
        r_max: float,
        n_g: float,
        sigma: float,
        dt: float,
        v_th: float,
    ) -> None:
        if sigma <= 0.0:
            return
        _mcc_ion_kernel[self._particle_blocks, self.threads_per_block](
            self.d_r_i,
            self.d_vr_i,
            self.d_vt_i,
            r_min,
            r_max,
            n_g,
            sigma,
            dt,
            v_th,
            self.rng_states_i,
        )

    def collide_electrons(
        self,
        r_min: float,
        r_max: float,
        n_g: float,
        sigma_el: float,
        sigma_exc: float,
        sigma_ion: float,
        dt: float,
        m_e: float,
        e_exc: float,
        e_ion: float,
    ) -> int:
        if sigma_el <= 0.0 and sigma_exc <= 0.0 and sigma_ion <= 0.0:
            return 0
        self._counter_host[0] = 0
        self.d_ionization_counter.copy_to_device(self._counter_host)
        _mcc_electron_kernel[self._particle_blocks, self.threads_per_block](
            self.d_r_e,
            self.d_vr_e,
            self.d_vt_e,
            r_min,
            r_max,
            n_g,
            sigma_el,
            sigma_exc,
            sigma_ion,
            dt,
            m_e,
            e_exc,
            e_ion,
            self.rng_states_e,
            self.d_ionization_counter,
        )
        self.d_ionization_counter.copy_to_host(self._counter_host)
        return int(self._counter_host[0])

    def deposit_charge_device(self, r_min: float, dr: float) -> None:
        _fill_zero_kernel[self._grid_blocks, self.threads_per_block](self.d_rho_e)
        _fill_zero_kernel[self._grid_blocks, self.threads_per_block](self.d_rho_i)

        _weight_charge_cic_kernel[self._particle_blocks, self.threads_per_block](
            self.d_r_e, self.d_qe, r_min, dr, self.d_rho_e
        )
        _weight_charge_cic_kernel[self._particle_blocks, self.threads_per_block](
            self.d_r_i, self.d_qi, r_min, dr, self.d_rho_i
        )
        _normalize_rho_kernel[self._grid_blocks, self.threads_per_block](self.d_rho_e, self.d_vol)
        _normalize_rho_kernel[self._grid_blocks, self.threads_per_block](self.d_rho_i, self.d_vol)

    def solve_fields_device(
        self,
        n_smoothing_passes: int,
        r_min: float,
        dr: float,
        epsilon_0: float,
        v_bias: float,
        v_outer: float,
        outer_neumann: bool,
        outer_dphi_dr: float,
        outer_self_consistent: bool,
    ) -> None:
        _sum_arrays_kernel[self._grid_blocks, self.threads_per_block](
            self.d_rho_e,
            self.d_rho_i,
            self.d_rho,
        )

        if n_smoothing_passes > 0 and self.n_nodes >= 3:
            src = self.d_rho
            dst = self.d_rho_tmp
            for _ in range(int(n_smoothing_passes)):
                _smooth_density_pass_kernel[self._grid_blocks, self.threads_per_block](src, dst)
                src, dst = dst, src
            if src is not self.d_rho:
                _copy_array_kernel[self._grid_blocks, self.threads_per_block](src, self.d_rho)

        _solve_poisson_cylindrical_kernel[1, 1](
            self.d_rho,
            self.d_phi,
            r_min,
            dr,
            epsilon_0,
            v_bias,
            v_outer,
            outer_neumann,
            outer_dphi_dr,
            outer_self_consistent,
            self.d_a,
            self.d_b,
            self.d_c,
            self.d_d,
        )
        _compute_electric_field_kernel[self._grid_blocks, self.threads_per_block](
            self.d_phi,
            dr,
            self.d_e,
        )

    def max_active_speed(self, species: str, r_min: float, r_max: float) -> float:
        self._max_speed_host[0] = 0.0
        self.d_max_speed.copy_to_device(self._max_speed_host)

        if species == "e":
            _max_speed_kernel[self._particle_blocks, self.threads_per_block](
                self.d_r_e, self.d_vr_e, self.d_vt_e, r_min, r_max, self.d_max_speed
            )
        elif species == "i":
            _max_speed_kernel[self._particle_blocks, self.threads_per_block](
                self.d_r_i, self.d_vr_i, self.d_vt_i, r_min, r_max, self.d_max_speed
            )
        else:
            raise ValueError("species must be 'e' or 'i'")

        self.d_max_speed.copy_to_host(self._max_speed_host)
        return float(self._max_speed_host[0])


def build_cuda_backend_or_none(
    n_particles: int,
    n_nodes: int,
    qe_arr: np.ndarray,
    qi_arr: np.ndarray,
    vol: np.ndarray,
    seed: int,
    threads_per_block: int,
) -> CUDABackend | None:
    if not CUDA_AVAILABLE:
        return None
    try:
        return CUDABackend(
            n_particles=n_particles,
            n_nodes=n_nodes,
            qe_arr=qe_arr,
            qi_arr=qi_arr,
            vol=vol,
            seed=seed,
            threads_per_block=threads_per_block,
        )
    except Exception as exc:  # pragma: no cover
        warnings.warn(f"Failed to initialize CUDA backend; falling back to CPU. {exc}", RuntimeWarning)
        return None
