from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from core.cs_txt_adapter import parse_cs_txt

import numpy as np


@dataclass(frozen=True)
class CrossSectionProcess:
    kind: str
    target: str
    label: str
    energy_ev: np.ndarray
    sigma_m2: np.ndarray


@dataclass(frozen=True)
class ElectronXSections:
    e_min: float
    inv_de: float
    sigma_elastic: np.ndarray
    sigma_excitation: np.ndarray
    sigma_ionization: np.ndarray


@dataclass(frozen=True)
class IonXSections:
    e_min: float
    inv_de: float
    sigma_cex: np.ndarray
    sigma_elastic: np.ndarray


_KNOWN_KINDS = (
    "ELASTIC",
    "EFFECTIVE",
    "EXCITATION",
    "IONIZATION",
    "ATTACHMENT",
    "CHARGE EXCHANGE",
    "CHARGE-EXCHANGE",
    "BACKSCATTER",
)


def _normalize_kind(line: str) -> str | None:
    text = " ".join(line.strip().split())
    if not text:
        return None
    upper = text.upper()
    if upper in _KNOWN_KINDS:
        return upper
    if upper.startswith("CHARGE"):
        return "CHARGE EXCHANGE"
    return None


def _is_separator(line: str) -> bool:
    stripped = line.strip()
    return stripped.startswith("-") and len(stripped) >= 5


def _parse_target_line(line: str) -> tuple[str, str]:
    text = line.strip()
    if "<->" in text:
        left, right = text.split("<->", 1)
        return left.strip(), right.strip()
    if "->" in text:
        left, right = text.split("->", 1)
        return left.strip(), right.strip()
    return text, ""


def _parse_table(lines: list[str], start: int) -> tuple[np.ndarray, np.ndarray, int]:
    energy: list[float] = []
    sigma: list[float] = []
    i = start
    while i < len(lines) and not _is_separator(lines[i]):
        parts = lines[i].strip().split()
        if len(parts) >= 2:
            try:
                energy.append(float(parts[0]))
                sigma.append(float(parts[1]))
            except ValueError:
                pass
        i += 1
    return np.asarray(energy, dtype=float), np.asarray(sigma, dtype=float), i


def _scale_from_headers(headers: list[str]) -> float:
    sigma_scale = 1.0
    for header in headers:
        upper = header.upper()
        if "CM2" in upper or "CM^2" in upper:
            sigma_scale = 1.0e-4
            break
    return sigma_scale


def _extract_target_from_species(species: str) -> str:
    text = species.strip()
    if "/" in text:
        return text.split("/", 1)[1].strip()
    return text


def _infer_kind_from_process(process: str) -> str | None:
    upper = process.upper()
    if "IONIZATION" in upper:
        return "IONIZATION"
    if "EXCITATION" in upper:
        return "EXCITATION"
    if "ATTACH" in upper:
        return "ATTACHMENT"
    if "CHARGE" in upper:
        return "CHARGE EXCHANGE"
    if "ELASTIC" in upper or "BACKSCAT" in upper or "BACKSCATTER" in upper or "MOMENTUM" in upper:
        return "ELASTIC"
    return None


def _process_exists(
    processes: list[CrossSectionProcess],
    kind: str,
    target: str,
    energy_ev: np.ndarray,
) -> bool:
    for proc in processes:
        if proc.kind != kind:
            continue
        if proc.target.strip().lower() != target.strip().lower():
            continue
        if proc.energy_ev.size != energy_ev.size:
            continue
        if proc.energy_ev.size > 0 and proc.energy_ev[0] == energy_ev[0] and proc.energy_ev[-1] == energy_ev[-1]:
            return True
    return False


def load_lxcat_text(path: Path | str, default_target: str | None = None) -> list[CrossSectionProcess]:
    """Parse LXCat-style text file into cross-section process blocks."""
    lines = Path(path).read_text(encoding="utf-8", errors="ignore").splitlines()
    processes: list[CrossSectionProcess] = []

    # First pass: keyword-based blocks (ELASTIC / IONIZATION / etc).
    i = 0
    while i < len(lines):
        kind = _normalize_kind(lines[i])
        if kind is None:
            i += 1
            continue

        if i + 1 >= len(lines):
            break
        target, label = _parse_target_line(lines[i + 1])
        i += 2

        header_lines: list[str] = []
        while i < len(lines) and not _is_separator(lines[i]):
            header_lines.append(lines[i])
            i += 1
        if i >= len(lines):
            break
        i += 1  # skip separator

        sigma_scale = _scale_from_headers(header_lines)
        energy, sigma, i = _parse_table(lines, i)
        if sigma_scale != 1.0:
            sigma = sigma * sigma_scale
        i += 1  # skip end separator

        if energy.size:
            e_arr = energy
            s_arr = sigma
            order = np.argsort(e_arr)
            processes.append(
                CrossSectionProcess(
                    kind=kind,
                    target=target,
                    label=label,
                    energy_ev=e_arr[order],
                    sigma_m2=s_arr[order],
                )
            )

    # Second pass: SPECIES/PROCESS blocks.
    pending_species: str | None = None
    pending_process: str | None = None
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("SPECIES:"):
            pending_species = line.split(":", 1)[1].strip()
        elif line.startswith("PROCESS:"):
            pending_process = line.split(":", 1)[1].strip()
        elif line.startswith("COLUMNS:"):
            # Find start of numeric table after separator.
            i += 1
            while i < len(lines) and not _is_separator(lines[i]):
                i += 1
            if i >= len(lines):
                break
            i += 1

            energy, sigma, i = _parse_table(lines, i)
            if energy.size and pending_species and pending_process:
                kind = _infer_kind_from_process(pending_process)
                if kind:
                    target = _extract_target_from_species(pending_species)
                    if not _process_exists(processes, kind, target, energy):
                        order = np.argsort(energy)
                        processes.append(
                            CrossSectionProcess(
                                kind=kind,
                                target=target,
                                label=pending_process,
                                energy_ev=energy[order],
                                sigma_m2=sigma[order],
                            )
                        )
        i += 1

    # Fallback: leading numeric table (implicit elastic).
    if default_target:
        has_elastic = any(proc.kind == "ELASTIC" and proc.target.lower() == default_target.lower() for proc in processes)
        if not has_elastic:
            i = 0
            while i < len(lines) and lines[i].strip() == "":
                i += 1
            if i < len(lines):
                parts = lines[i].strip().split()
                if len(parts) >= 2:
                    try:
                        float(parts[0])
                        float(parts[1])
                        energy, sigma, i = _parse_table(lines, i)
                        if energy.size:
                            order = np.argsort(energy)
                            processes.append(
                                CrossSectionProcess(
                                    kind="ELASTIC",
                                    target=default_target,
                                    label=f"{default_target} (implicit elastic)",
                                    energy_ev=energy[order],
                                    sigma_m2=sigma[order],
                                )
                            )
                    except ValueError:
                        pass
    return processes


def _select_processes(
    processes: Iterable[CrossSectionProcess],
    kinds: set[str],
    target: str,
) -> list[CrossSectionProcess]:
    target_lower = target.strip().lower()
    return [
        proc
        for proc in processes
        if proc.kind in kinds and proc.target.strip().lower() == target_lower
    ]


def _sum_processes(processes: list[CrossSectionProcess]) -> tuple[np.ndarray, np.ndarray]:
    energies = np.unique(np.concatenate([p.energy_ev for p in processes]))
    total = np.zeros_like(energies)
    for proc in processes:
        total += np.interp(energies, proc.energy_ev, proc.sigma_m2, left=0.0, right=proc.sigma_m2[-1])
    return energies, total


def _build_uniform_table(
    energy_ev: np.ndarray,
    sigma_m2: np.ndarray,
    e_max: float,
    n_bins: int,
) -> tuple[float, float, np.ndarray]:
    e_min = 0.0
    grid = np.linspace(e_min, e_max, n_bins)
    sigma = np.interp(grid, energy_ev, sigma_m2, left=sigma_m2[0], right=sigma_m2[-1])
    inv_de = (n_bins - 1) / (e_max - e_min) if e_max > e_min else 1.0
    return e_min, inv_de, sigma


def build_electron_tables_from_lxcat(
    path: Path | str,
    target: str,
    e_max: float,
    n_bins: int,
) -> ElectronXSections:
    processes = load_lxcat_text(path, default_target=target)

    elastic = _select_processes(processes, {"ELASTIC"}, target)
    if not elastic:
        elastic = _select_processes(processes, {"EFFECTIVE"}, target)
    if not elastic:
        raise ValueError("No ELASTIC/EFFECTIVE cross section found for target.")
    elastic_energy, elastic_sigma = _sum_processes(elastic[:1])

    excitation = _select_processes(processes, {"EXCITATION"}, target)
    if excitation:
        exc_energy, exc_sigma = _sum_processes(excitation)
    else:
        exc_energy = elastic_energy
        exc_sigma = np.zeros_like(elastic_energy)

    ionization = _select_processes(processes, {"IONIZATION"}, target)
    if ionization:
        ion_energy, ion_sigma = _sum_processes(ionization)
    else:
        ion_energy = elastic_energy
        ion_sigma = np.zeros_like(elastic_energy)

    e_min, inv_de, sigma_el = _build_uniform_table(elastic_energy, elastic_sigma, e_max, n_bins)
    _, _, sigma_exc = _build_uniform_table(exc_energy, exc_sigma, e_max, n_bins)
    _, _, sigma_ion = _build_uniform_table(ion_energy, ion_sigma, e_max, n_bins)
    return ElectronXSections(e_min=e_min, inv_de=inv_de, sigma_elastic=sigma_el, sigma_excitation=sigma_exc, sigma_ionization=sigma_ion)


def build_ion_tables_from_lxcat(
    path: Path | str,
    target: str,
    e_max: float,
    n_bins: int,
    fallback_cex: float | None = None,
) -> IonXSections:
    processes = load_lxcat_text(path, default_target=target)

    cex = _select_processes(processes, {"CHARGE EXCHANGE", "CHARGE-EXCHANGE"}, target)
    if not cex:
        cex = [p for p in processes if "CHARGE" in p.kind and p.target.strip().lower() == target.lower()]
    if cex:
        cex_energy, cex_sigma = _sum_processes(cex)
    elif fallback_cex and fallback_cex > 0.0:
        cex_energy = np.array([0.0, e_max], dtype=float)
        cex_sigma = np.array([fallback_cex, fallback_cex], dtype=float)
    else:
        cex_energy = np.array([0.0, e_max], dtype=float)
        cex_sigma = np.zeros(2, dtype=float)

    elastic = _select_processes(processes, {"ELASTIC", "EFFECTIVE"}, target)
    if elastic:
        el_energy, el_sigma = _sum_processes(elastic[:1])
    else:
        el_energy = np.array([0.0, e_max], dtype=float)
        el_sigma = np.zeros(2, dtype=float)

    e_min, inv_de, sigma_cex = _build_uniform_table(cex_energy, cex_sigma, e_max, n_bins)
    _, _, sigma_el = _build_uniform_table(el_energy, el_sigma, e_max, n_bins)
    return IonXSections(e_min=e_min, inv_de=inv_de, sigma_cex=sigma_cex, sigma_elastic=sigma_el)


def build_constant_electron_tables(
    sigma_el: float,
    sigma_exc: float,
    sigma_ion: float,
    e_max: float,
    n_bins: int,
) -> ElectronXSections:
    e_min = 0.0
    inv_de = (n_bins - 1) / (e_max - e_min) if e_max > e_min else 1.0
    sigma_elastic = np.full(n_bins, sigma_el, dtype=float)
    sigma_excitation = np.full(n_bins, sigma_exc, dtype=float)
    sigma_ionization = np.full(n_bins, sigma_ion, dtype=float)
    return ElectronXSections(e_min=e_min, inv_de=inv_de, sigma_elastic=sigma_elastic, sigma_excitation=sigma_excitation, sigma_ionization=sigma_ionization)


def build_constant_ion_tables(
    sigma_cex: float,
    sigma_elastic: float,
    e_max: float,
    n_bins: int,
) -> IonXSections:
    e_min = 0.0
    inv_de = (n_bins - 1) / (e_max - e_min) if e_max > e_min else 1.0
    sigma_cex_arr = np.full(n_bins, sigma_cex, dtype=float)
    sigma_el_arr = np.full(n_bins, sigma_elastic, dtype=float)
    return IonXSections(e_min=e_min, inv_de=inv_de, sigma_cex=sigma_cex_arr, sigma_elastic=sigma_el_arr)

def load_cross_sections_from_custom_file(
    path: Path | str,
    e_max: float,
    n_bins: int,
) -> tuple[ElectronXSections, IonXSections]:
    """
    Load cross sections from a custom CS.txt file.
    
    Returns:
        (ElectronXSections, IonXSections)
    """
    sections = parse_cs_txt(path)
    
    # --- Build Electron Tables ---
    # Elastic
    if "electron_elastic" in sections:
        data = sections["electron_elastic"]
        e_el, s_el = data.energy, data.sigma
    else:
        # Fallback to zero if missing (unlikely if parser works)
        e_el, s_el = np.array([0.0, e_max]), np.zeros(2)

    # Excitation (Assume zero if not present, CS.txt doesn't seem to have it?)
    # If CS.txt has excitation, the parser needs to find it. 
    # Current parser maps "IONIZATION" and "Backscat", everything else is elastic?
    # Actually current parser only finds 3 blocks.
    e_exc, s_exc = np.array([0.0, e_max]), np.zeros(2)

    # Ionization
    if "electron_ionization" in sections:
        data = sections["electron_ionization"]
        e_ion, s_ion = data.energy, data.sigma
    else:
        e_ion, s_ion = np.array([0.0, e_max]), np.zeros(2)
        
    e_min, inv_de, sigma_el_arr = _build_uniform_table(e_el, s_el, e_max, n_bins)
    _, _, sigma_exc_arr = _build_uniform_table(e_exc, s_exc, e_max, n_bins)
    _, _, sigma_ion_arr = _build_uniform_table(e_ion, s_ion, e_max, n_bins)
    
    electron_xs = ElectronXSections(
        e_min=e_min, 
        inv_de=inv_de, 
        sigma_elastic=sigma_el_arr, 
        sigma_excitation=sigma_exc_arr, 
        sigma_ionization=sigma_ion_arr
    )

    # --- Build Ion Tables ---
    # CEX
    if "ion_cex" in sections:
        data = sections["ion_cex"]
        e_cex, s_cex = data.energy, data.sigma
    else:
        e_cex, s_cex = np.array([0.0, e_max]), np.zeros(2)
        
    # Elastic (Ion) - Not explicitly in CS.txt, assume zero or included in CEX?
    # CS.txt has "Ar+ + Ar -> , Backscat", which is predominantly CEX/Backscatter.
    e_iel, s_iel = np.array([0.0, e_max]), np.zeros(2)
    
    e_min_i, inv_de_i, sigma_cex_arr = _build_uniform_table(e_cex, s_cex, e_max, n_bins)
    _, _, sigma_iel_arr = _build_uniform_table(e_iel, s_iel, e_max, n_bins)
    
    ion_xs = IonXSections(
        e_min=e_min_i,
        inv_de=inv_de_i,
        sigma_cex=sigma_cex_arr,
        sigma_elastic=sigma_iel_arr
    )
    
    return electron_xs, ion_xs
