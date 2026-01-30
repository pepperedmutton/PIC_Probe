from __future__ import annotations

import math

import numpy as np
from numba import jit, prange


@jit(nopython=True, parallel=True)
def perform_mcc_ion(
    r: np.ndarray,
    vr: np.ndarray,
    vt: np.ndarray,
    r_min: float,
    r_max: float,
    n_g: float,
    sigma: float,
    dt: float,
    v_th: float,
) -> int:
    """Ion-neutral charge exchange (CEX) collisions."""
    n_collisions = 0
    for i in prange(r.shape[0]):
        ri = r[i]
        if ri <= r_min or ri >= r_max:
            continue
        vi = math.sqrt(vr[i] * vr[i] + vt[i] * vt[i])
        p = 1.0 - math.exp(-n_g * sigma * vi * dt)
        if np.random.random() < p:
            vr[i] = np.random.normal(0.0, v_th)
            vt[i] = np.random.normal(0.0, v_th)
            n_collisions += 1
    return n_collisions


@jit(nopython=True, parallel=True)
def perform_mcc_electron(
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
) -> tuple[int, int, int]:
    """Electron-neutral elastic/excitation/ionization collisions."""
    n_el = 0
    n_exc = 0
    n_ion = 0
    if sigma_el <= 0.0 and sigma_exc <= 0.0 and sigma_ion <= 0.0:
        return n_el, n_exc, n_ion

    for i in prange(r.shape[0]):
        ri = r[i]
        if ri <= r_min or ri >= r_max:
            continue

        v2 = vr[i] * vr[i] + vt[i] * vt[i]
        if v2 <= 0.0:
            continue
        v = math.sqrt(v2)
        energy = 0.5 * m_e * v2

        sigma_exc_eff = sigma_exc if energy > e_exc else 0.0
        sigma_ion_eff = sigma_ion if energy > e_ion else 0.0
        sigma_total = sigma_el + sigma_exc_eff + sigma_ion_eff
        if sigma_total <= 0.0:
            continue

        p = 1.0 - math.exp(-n_g * sigma_total * v * dt)
        if np.random.random() >= p:
            continue

        pick = np.random.random() * sigma_total
        angle = 2.0 * math.pi * np.random.random()
        if pick < sigma_el:
            vr[i] = v * math.cos(angle)
            vt[i] = v * math.sin(angle)
            n_el += 1
        elif pick < sigma_el + sigma_exc_eff:
            energy_new = energy - e_exc
            if energy_new < 0.0:
                energy_new = 0.0
            v_new = math.sqrt(2.0 * energy_new / m_e) if energy_new > 0.0 else 0.0
            vr[i] = v_new * math.cos(angle)
            vt[i] = v_new * math.sin(angle)
            n_exc += 1
        else:
            energy_new = energy - e_ion
            if energy_new < 0.0:
                energy_new = 0.0
            v_new = math.sqrt(2.0 * energy_new / m_e) if energy_new > 0.0 else 0.0
            vr[i] = v_new * math.cos(angle)
            vt[i] = v_new * math.sin(angle)
            n_ion += 1

    return n_el, n_exc, n_ion
