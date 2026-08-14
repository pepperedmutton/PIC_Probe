from __future__ import annotations

import math

import numpy as np
from numba import jit, prange


@jit(nopython=True, parallel=True)
def compute_shell_volumes(r_min: float, dr: float, vol: np.ndarray) -> None:
    """Calculate cylindrical control volumes for unit axial length."""
    n_nodes = vol.shape[0]
    if n_nodes == 0:
        return
    r_max = r_min + dr * (n_nodes - 1)
    for j in prange(n_nodes):
        r_j = r_min + dr * j
        r_left = r_min if j == 0 else r_j - 0.5 * dr
        r_right = r_max if j == n_nodes - 1 else r_j + 0.5 * dr
        vol[j] = math.pi * (r_right * r_right - r_left * r_left)


@jit(nopython=True)
def deposit_charge_cic(
    r: np.ndarray,
    q: np.ndarray,
    r_min: float,
    dr: float,
    node_charge: np.ndarray,
) -> None:
    """Deposit particle charge on the internal grid nodes."""
    n_nodes = node_charge.shape[0]
    for j in range(n_nodes):
        node_charge[j] = 0.0

    if n_nodes < 3:
        return

    r_max = r_min + dr * (n_nodes - 1)
    for i in range(r.shape[0]):
        ri = r[i]
        if ri <= r_min or ri >= r_max:
            continue

        xi = (ri - r_min) / dr
        j = int(xi)
        if j < 0 or j >= n_nodes - 1:
            continue

        w_right = xi - j
        w_left = 1.0 - w_right
        qi = q[i]

        node_charge[j] += qi * w_left
        node_charge[j + 1] += qi * w_right


@jit(nopython=True, parallel=True)
def weight_charge_cic(
    r: np.ndarray,
    q: np.ndarray,
    r_min: float,
    dr: float,
    rho: np.ndarray,
    vol: np.ndarray,
) -> None:
    """Deposit particle charge and calculate the charge density."""
    n_nodes = rho.shape[0]
    deposit_charge_cic(r, q, r_min, dr, rho)

    for j in prange(n_nodes):
        if vol[j] > 0.0:
            rho[j] /= vol[j]
        else:
            rho[j] = 0.0


@jit(nopython=True, inline="always")
def gather_field_linear(
    position: float,
    E_grid: np.ndarray,
    r_min: float,
    dr: float,
) -> float:
    """Calculate the field at one particle position."""
    n_nodes = E_grid.shape[0]
    xi = (position - r_min) / dr
    j = int(xi)
    if j < 0:
        j = 0
    elif j >= n_nodes - 1:
        j = n_nodes - 2
    weight = xi - j
    return (1.0 - weight) * E_grid[j] + weight * E_grid[j + 1]


@jit(nopython=True, parallel=True)
def radial_kick(
    r: np.ndarray,
    vr: np.ndarray,
    vt: np.ndarray,
    E_grid: np.ndarray,
    q: float,
    m: float,
    dt: float,
    r_min: float,
    r_max: float,
    dr: float,
) -> None:
    """Apply one radial half-kick to the active particles."""
    half_dt = 0.5 * dt
    for i in prange(r.shape[0]):
        ri = r[i]
        if ri <= r_min or ri >= r_max:
            continue
        field = gather_field_linear(ri, E_grid, r_min, dr)
        acceleration = (q / m) * field + vt[i] * vt[i] / ri
        vr[i] += acceleration * half_dt


@jit(nopython=True, parallel=True)
def cylindrical_drift(
    r: np.ndarray,
    vr: np.ndarray,
    vt: np.ndarray,
    vz: np.ndarray,
    dt: float,
    r_min: float,
    r_max: float,
    reflect_wall: bool,
) -> tuple[int, int]:
    """Move particles and keep their axial angular momentum."""
    n_probe_hits = 0
    n_wall_hits = 0
    if dt <= 0.0:
        return n_probe_hits, n_wall_hits

    r_span = r_max - r_min
    r_dead = r_max + r_span

    for i in prange(r.shape[0]):
        ri = r[i]
        if ri <= r_min or ri >= r_max:
            continue

        radial_velocity = vr[i]
        angular_momentum = ri * vt[i]
        trial_radius = ri + radial_velocity * dt

        if trial_radius <= r_min:
            n_probe_hits += 1
            r[i] = r_dead
            vr[i] = 0.0
            vt[i] = 0.0
            vz[i] = 0.0
            continue

        if trial_radius >= r_max:
            n_wall_hits += 1
            if not reflect_wall:
                r[i] = r_dead
                vr[i] = 0.0
                vt[i] = 0.0
                vz[i] = 0.0
                continue

            time_to_wall = (r_max - ri) / radial_velocity
            remaining_time = dt - time_to_wall
            reflected_velocity = -radial_velocity
            reflected_radius = r_max + reflected_velocity * remaining_time

            if reflected_radius <= r_min:
                n_probe_hits += 1
                r[i] = r_dead
                vr[i] = 0.0
                vt[i] = 0.0
                vz[i] = 0.0
                continue

            if reflected_radius >= r_max:
                reflected_radius = np.nextafter(r_max, r_min)
            r[i] = reflected_radius
            vr[i] = reflected_velocity
            vt[i] = angular_momentum / reflected_radius
            continue

        r[i] = trial_radius
        vt[i] = angular_momentum / trial_radius

    return n_probe_hits, n_wall_hits


@jit(nopython=True, parallel=True)
def initialize_leapfrog_velocity(
    r: np.ndarray,
    vr: np.ndarray,
    vt: np.ndarray,
    E_grid: np.ndarray,
    q: float,
    m: float,
    dt: float,
    r_min: float,
    r_max: float,
    dr: float,
) -> None:
    """Move radial velocity back by one half time step."""
    for i in prange(r.shape[0]):
        ri = r[i]
        if ri <= r_min or ri >= r_max:
            continue
        field = gather_field_linear(ri, E_grid, r_min, dr)
        acceleration = (q / m) * field + vt[i] * vt[i] / ri
        vr[i] -= 0.5 * acceleration * dt


@jit(nopython=True, parallel=True)
def push_particles_leapfrog(
    r: np.ndarray,
    vr: np.ndarray,
    vt: np.ndarray,
    vz: np.ndarray,
    E_grid: np.ndarray,
    q: float,
    m: float,
    dt: float,
    r_min: float,
    r_max: float,
    dr: float,
    reflect_wall: bool,
) -> tuple[int, int]:
    """Advance 1D3V particles with a staggered radial velocity."""
    n_probe_hits = 0
    n_wall_hits = 0
    r_span = r_max - r_min
    r_dead = r_max + r_span

    for i in prange(r.shape[0]):
        ri = r[i]
        if ri <= r_min or ri >= r_max:
            continue

        field = gather_field_linear(ri, E_grid, r_min, dr)
        acceleration = (q / m) * field + vt[i] * vt[i] / ri
        vr_new = vr[i] + acceleration * dt
        r_new = ri + vr_new * dt

        if r_new <= r_min:
            n_probe_hits += 1
            r[i] = r_dead
            vr[i] = 0.0
            vt[i] = 0.0
            vz[i] = 0.0
            continue

        if r_new >= r_max:
            n_wall_hits += 1
            if reflect_wall:
                r_new = r_max - (r_new - r_max)
                if r_new <= r_min:
                    r_new = r_min + np.finfo(np.float64).eps
                vr_new = -abs(vr_new)
                vt[i] = vt[i] * ri / r_new
                r[i] = r_new
                vr[i] = vr_new
            else:
                r[i] = r_dead
                vr[i] = 0.0
                vt[i] = 0.0
                vz[i] = 0.0
            continue

        vt[i] = vt[i] * ri / r_new
        r[i] = r_new
        vr[i] = vr_new

    return n_probe_hits, n_wall_hits


@jit(nopython=True, parallel=True)
def push_particles(
    r: np.ndarray,
    vr: np.ndarray,
    vt: np.ndarray,
    E_grid: np.ndarray,
    q: float,
    m: float,
    dt: float,
    r_min: float,
    r_max: float,
    dr: float,
    reflect_wall: bool,
) -> tuple:
    """Advance particles one step (velocity Verlet) with boundary handling.

    Returns:
        (n_probe_hits, n_wall_hits)
    """
    n_nodes = E_grid.shape[0]
    n_probe_hits = 0
    n_wall_hits = 0
    r_span = r_max - r_min
    r_dead = r_max + r_span

    for i in prange(r.shape[0]):
        ri = r[i]
        if ri <= r_min or ri >= r_max:
            continue

        # Linear gather of E field at particle position
        xi = (ri - r_min) / dr
        j = int(xi)
        if j < 0:
            j = 0
        elif j >= n_nodes - 1:
            j = n_nodes - 2
        w = xi - j
        E = (1.0 - w) * E_grid[j] + w * E_grid[j + 1]

        vt_old = vt[i]
        a_r = (q / m) * E + (vt_old * vt_old) / ri
        vr_old = vr[i]
        r_new = ri + vr_old * dt + 0.5 * a_r * dt * dt

        if r_new <= r_min:
            n_probe_hits += 1
            r[i] = r_dead
            vr[i] = 0.0
            vt[i] = 0.0
            continue

        if r_new >= r_max:
            n_wall_hits += 1
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
                E2 = (1.0 - w2) * E_grid[j2] + w2 * E_grid[j2 + 1]
                a_r_new = (q / m) * E2 + (vt_new * vt_new) / r_new
                vr_new = -(vr_old + 0.5 * (a_r + a_r_new) * dt)

                r[i] = r_new
                vr[i] = vr_new
                vt[i] = vt_new
            else:
                r[i] = r_dead
                vr[i] = 0.0
                vt[i] = 0.0
            continue

        vt_new = vt_old * (ri / r_new)

        xi2 = (r_new - r_min) / dr
        j2 = int(xi2)
        if j2 < 0:
            j2 = 0
        elif j2 >= n_nodes - 1:
            j2 = n_nodes - 2
        w2 = xi2 - j2
        E2 = (1.0 - w2) * E_grid[j2] + w2 * E_grid[j2 + 1]
        a_r_new = (q / m) * E2 + (vt_new * vt_new) / r_new
        vr_new = vr_old + 0.5 * (a_r + a_r_new) * dt

        r[i] = r_new
        vr[i] = vr_new
        vt[i] = vt_new

    return n_probe_hits, n_wall_hits
