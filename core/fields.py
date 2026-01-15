from __future__ import annotations

import math

import numpy as np
from numba import jit


@jit(nopython=True)
def solve_poisson_cylindrical(
    rho: np.ndarray,
    phi: np.ndarray,
    r_min: float,
    dr: float,
    epsilon_0: float,
    v_bias: float,
    v_wall: float,
) -> None:
    """Solve cylindrical Poisson equation with Dirichlet BCs using TDMA."""
    n = rho.shape[0]
    if n == 0:
        return

    phi[0] = v_bias
    if n == 1:
        return
    phi[n - 1] = v_wall
    if n == 2:
        return

    n_int = n - 2
    a = np.empty(n_int)
    b = np.empty(n_int)
    c = np.empty(n_int)
    d = np.empty(n_int)

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
    d[n_int - 1] -= c[n_int - 1] * phi[n - 1]

    for i in range(1, n_int):
        w = a[i] / b[i - 1]
        b[i] = b[i] - w * c[i - 1]
        d[i] = d[i] - w * d[i - 1]

    phi[n - 2] = d[n_int - 1] / b[n_int - 1]
    for i in range(n_int - 2, -1, -1):
        phi[i + 1] = (d[i] - c[i] * phi[i + 2]) / b[i]


@jit(nopython=True)
def compute_electric_field(phi: np.ndarray, dr: float, E: np.ndarray) -> None:
    """Compute radial electric field E = -dphi/dr on the same grid as phi."""
    n = phi.shape[0]
    if n == 0:
        return
    if n == 1:
        E[0] = 0.0
        return

    E[0] = -(phi[1] - phi[0]) / dr
    for j in range(1, n - 1):
        E[j] = -(phi[j + 1] - phi[j - 1]) / (2.0 * dr)
    E[n - 1] = -(phi[n - 1] - phi[n - 2]) / dr
