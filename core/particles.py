from __future__ import annotations

import math

import numpy as np
from numba import jit


@jit(nopython=True)
def compute_shell_volumes(r_min: float, dr: float, vol: np.ndarray) -> None:
    """Fill volume array for cylindrical shells (per unit length)."""
    n_nodes = vol.shape[0]
    for j in range(n_nodes):
        r_j = r_min + dr * j
        vol[j] = 2.0 * math.pi * r_j * dr


@jit(nopython=True)
def weight_charge_cic(
    r: np.ndarray,
    q: np.ndarray,
    r_min: float,
    dr: float,
    rho: np.ndarray,
    vol: np.ndarray,
) -> None:
    """Scatter particle charge to grid with cylindrical volume correction."""
    n_nodes = rho.shape[0]
    for j in range(n_nodes):
        rho[j] = 0.0

    r_max = r_min + dr * (n_nodes - 1)
    for i in range(r.shape[0]):
        ri = r[i]
        if ri <= r_min or ri >= r_max:
            continue
        xi = (ri - r_min) / dr
        j = int(xi)
        if j < 0 or j >= n_nodes - 1:
            continue
        w = xi - j
        qi = q[i]
        rho[j] += qi * (1.0 - w)
        rho[j + 1] += qi * w

    for j in range(n_nodes):
        if vol[j] > 0.0:
            rho[j] /= vol[j]
        else:
            rho[j] = 0.0


@jit(nopython=True)
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

    for i in range(r.shape[0]):
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
