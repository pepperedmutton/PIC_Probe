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
    outer_neumann: bool = False,
    outer_dphi_dr: float = 0.0,
    outer_self_consistent: bool = False,
) -> None:
    """Solve cylindrical Poisson equation using TDMA.

    Boundary conditions:
    - Inner boundary (`r_min`): Dirichlet `phi = v_bias`
    - Outer boundary (`r_max`):
      - Self-consistent extrapolation `phi_{N} = 2*phi_{N-1} - phi_{N-2}`
        when `outer_self_consistent=True`
      - Dirichlet `phi = v_wall` when `outer_neumann=False`
      - Neumann `dphi/dr = outer_dphi_dr` when `outer_neumann=True`
    """
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
    if outer_self_consistent:
        last = n_int - 1
        a[last] = a[last] - c[last]
        b[last] = b[last] + 2.0 * c[last]
        c[last] = 0.0
    elif outer_neumann:
        # phi_{n-1} = phi_{n-2} + outer_dphi_dr * dr
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


@jit(nopython=True)
def smooth_density_cylindrical(rho: np.ndarray, n_passes: int) -> None:
    """Apply binomial smoothing (1-2-1) to density array.

    Args:
        rho: Charge density array (modified in-place)
        n_passes: Number of smoothing passes
    """
    n = rho.shape[0]
    if n < 3 or n_passes <= 0:
        return

    rho_new = np.empty_like(rho)
    
    for _ in range(n_passes):
        # Interior points: 0.25 * rho[j-1] + 0.5 * rho[j] + 0.25 * rho[j+1]
        for j in range(1, n - 1):
            rho_new[j] = 0.25 * rho[j - 1] + 0.5 * rho[j] + 0.25 * rho[j + 1]
            
        # Boundaries: Asymmetric 2-point smooth
        rho_new[0] = 0.666 * rho[0] + 0.334 * rho[1]
        rho_new[n - 1] = 0.666 * rho[n - 1] + 0.334 * rho[n - 2]
        
        # Copy back
        for j in range(n):
            rho[j] = rho_new[j]
