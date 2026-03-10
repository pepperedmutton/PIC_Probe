from __future__ import annotations

import math

import numpy as np
from numba import jit, prange


@jit(nopython=True)
def sigma_from_uniform_table(
    energy_ev: float,
    sigma_table: np.ndarray,
    e_min: float,
    inv_de: float,
) -> float:
    n = sigma_table.shape[0]
    if n == 0:
        return 0.0
    if energy_ev <= e_min:
        return sigma_table[0]
    idx = int((energy_ev - e_min) * inv_de)
    if idx >= n - 1:
        return sigma_table[n - 1]
    frac = (energy_ev - e_min) * inv_de - idx
    return sigma_table[idx] + frac * (sigma_table[idx + 1] - sigma_table[idx])


@jit(nopython=True, parallel=True)
def perform_mcc_ion(
    r: np.ndarray,
    vr: np.ndarray,
    vt: np.ndarray,
    r_min: float,
    r_max: float,
    n_g: float,
    sigma_cex: np.ndarray,
    sigma_el: np.ndarray,
    dt: float,
    v_th: float,
    e_min: float,
    inv_de: float,
    e_charge: float,
    m_i: float,
) -> int:
    """Ion-neutral charge exchange (CEX) + elastic collisions."""
    n_collisions = 0
    for i in prange(r.shape[0]):
        ri = r[i]
        if ri <= r_min or ri >= r_max:
            continue
        vi = math.sqrt(vr[i] * vr[i] + vt[i] * vt[i])
        if vi <= 0.0:
            continue
        energy_ev = 0.5 * m_i * vi * vi / e_charge
        sigma_cex_val = sigma_from_uniform_table(energy_ev, sigma_cex, e_min, inv_de)
        sigma_el_val = sigma_from_uniform_table(energy_ev, sigma_el, e_min, inv_de)
        sigma_total = sigma_cex_val + sigma_el_val
        if sigma_total <= 0.0:
            continue
        p = 1.0 - math.exp(-n_g * sigma_total * vi * dt)
        if np.random.random() < p:
            pick = np.random.random() * sigma_total
            if pick < sigma_cex_val:
                vr[i] = np.random.normal(0.0, v_th)
                vt[i] = np.random.normal(0.0, v_th)
            else:
                angle = 2.0 * math.pi * np.random.random()
                vr[i] = vi * math.cos(angle)
                vt[i] = vi * math.sin(angle)
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
    sigma_el: np.ndarray,
    sigma_exc: np.ndarray,
    sigma_ion: np.ndarray,
    dt: float,
    m_e: float,
    e_exc: float,
    e_ion: float,
    e_min: float,
    inv_de: float,
    e_charge: float,
    ionized: np.ndarray,
    sec_energy_ev: np.ndarray,
) -> tuple[int, int, int]:
    """Electron-neutral elastic/excitation/ionization collisions."""
    n_el = 0
    n_exc = 0
    n_ion = 0

    for i in prange(r.shape[0]):
        ionized[i] = 0
        sec_energy_ev[i] = 0.0
        ri = r[i]
        if ri <= r_min or ri >= r_max:
            continue

        v2 = vr[i] * vr[i] + vt[i] * vt[i]
        if v2 <= 0.0:
            continue
        v = math.sqrt(v2)
        energy = 0.5 * m_e * v2
        energy_ev = energy / e_charge

        sigma_el_val = sigma_from_uniform_table(energy_ev, sigma_el, e_min, inv_de)
        sigma_exc_val = sigma_from_uniform_table(energy_ev, sigma_exc, e_min, inv_de) if energy > e_exc else 0.0
        sigma_ion_val = sigma_from_uniform_table(energy_ev, sigma_ion, e_min, inv_de) if energy > e_ion else 0.0
        sigma_total = sigma_el_val + sigma_exc_val + sigma_ion_val
        if sigma_total <= 0.0:
            continue

        p = 1.0 - math.exp(-n_g * sigma_total * v * dt)
        if np.random.random() >= p:
            continue

        pick = np.random.random() * sigma_total
        angle = 2.0 * math.pi * np.random.random()
        if pick < sigma_el_val:
            vr[i] = v * math.cos(angle)
            vt[i] = v * math.sin(angle)
            n_el += 1
        elif pick < sigma_el_val + sigma_exc_val:
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
            # Split remaining energy between primary and secondary electron.
            energy_share = 0.5 * energy_new
            v_new = math.sqrt(2.0 * energy_share / m_e) if energy_share > 0.0 else 0.0
            vr[i] = v_new * math.cos(angle)
            vt[i] = v_new * math.sin(angle)
            ionized[i] = 1
            sec_energy_ev[i] = energy_share / e_charge if energy_share > 0.0 else 0.0
            n_ion += 1

    return n_el, n_exc, n_ion


@jit(nopython=True, parallel=True)
def perform_coulomb_scatter(
    r: np.ndarray,
    vr: np.ndarray,
    vt: np.ndarray,
    r_min: float,
    r_max: float,
    nu: float,
    dt: float,
) -> int:
    """Simple pitch-angle scattering to approximate Coulomb collisions."""
    if nu <= 0.0:
        return 0
    n_events = 0
    p = 1.0 - math.exp(-nu * dt)
    for i in prange(r.shape[0]):
        ri = r[i]
        if ri <= r_min or ri >= r_max:
            continue
        if np.random.random() < p:
            v2 = vr[i] * vr[i] + vt[i] * vt[i]
            if v2 <= 0.0:
                continue
            v = math.sqrt(v2)
            angle = 2.0 * math.pi * np.random.random()
            vr[i] = v * math.cos(angle)
            vt[i] = v * math.sin(angle)
            n_events += 1
    return n_events
