from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np

from core.cs_txt_adapter import (
    CrossSectionProcess,
    parse_cross_section_file,
    parse_cs_txt,
)


@dataclass(frozen=True)
class ElectronXSections:
    """Electron cross sections on one uniform energy grid.

    Total tables have shape (n_bins,). Channel tables have shape
    (n_channels, n_bins). Threshold arrays have shape (n_channels,).
    All arrays have type float64 and are read-only.
    """

    e_min: float
    inv_de: float
    sigma_elastic: np.ndarray
    sigma_excitation: np.ndarray
    sigma_ionization: np.ndarray
    excitation_thresholds_ev: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float64)
    )
    excitation_channel_tables: np.ndarray = field(
        default_factory=lambda: np.empty((0, 0), dtype=np.float64)
    )
    ionization_thresholds_ev: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float64)
    )
    ionization_channel_tables: np.ndarray = field(
        default_factory=lambda: np.empty((0, 0), dtype=np.float64)
    )

    def __post_init__(self) -> None:
        sigma_elastic = _readonly_vector(
            self.sigma_elastic,
            "The electron elastic table",
        )
        sigma_excitation = _readonly_vector(
            self.sigma_excitation,
            "The electron excitation table",
        )
        sigma_ionization = _readonly_vector(
            self.sigma_ionization,
            "The electron ionization table",
        )
        if not (
            sigma_elastic.shape
            == sigma_excitation.shape
            == sigma_ionization.shape
        ):
            raise ValueError("The electron cross-section tables have different shapes.")

        excitation_thresholds, excitation_channels = _normalize_channels(
            self.excitation_thresholds_ev,
            self.excitation_channel_tables,
            sigma_excitation,
            "excitation",
        )
        ionization_thresholds, ionization_channels = _normalize_channels(
            self.ionization_thresholds_ev,
            self.ionization_channel_tables,
            sigma_ionization,
            "ionization",
        )
        object.__setattr__(self, "sigma_elastic", sigma_elastic)
        object.__setattr__(self, "sigma_excitation", sigma_excitation)
        object.__setattr__(self, "sigma_ionization", sigma_ionization)
        object.__setattr__(
            self,
            "excitation_thresholds_ev",
            excitation_thresholds,
        )
        object.__setattr__(
            self,
            "excitation_channel_tables",
            excitation_channels,
        )
        object.__setattr__(
            self,
            "ionization_thresholds_ev",
            ionization_thresholds,
        )
        object.__setattr__(
            self,
            "ionization_channel_tables",
            ionization_channels,
        )


@dataclass(frozen=True)
class IonXSections:
    e_min: float
    inv_de: float
    sigma_cex: np.ndarray
    sigma_elastic: np.ndarray


def _readonly_vector(values: np.ndarray, name: str) -> np.ndarray:
    vector = np.array(values, dtype=np.float64, copy=True, order="C")
    if vector.ndim != 1:
        raise ValueError(f"{name} must have one dimension.")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} has a nonfinite value.")
    if np.any(vector < 0.0):
        raise ValueError(f"{name} has a negative value.")
    vector.flags.writeable = False
    return vector


def _normalize_channels(
    thresholds_ev: np.ndarray,
    channel_tables: np.ndarray,
    total_table: np.ndarray,
    process_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    thresholds = np.array(
        thresholds_ev,
        dtype=np.float64,
        copy=True,
        order="C",
    )
    channels = np.array(
        channel_tables,
        dtype=np.float64,
        copy=True,
        order="C",
    )
    if thresholds.ndim != 1:
        raise ValueError(f"The {process_name} thresholds must have one dimension.")
    if channels.ndim != 2:
        raise ValueError(f"The {process_name} channel tables must have two dimensions.")

    n_bins = total_table.size
    if thresholds.size == 0 and channels.size == 0:
        if np.any(total_table > 0.0):
            thresholds = np.zeros(1, dtype=np.float64)
            channels = np.array(total_table[np.newaxis, :], copy=True)
        else:
            channels = np.empty((0, n_bins), dtype=np.float64)
    if channels.shape != (thresholds.size, n_bins):
        raise ValueError(
            f"The {process_name} channel tables must have shape "
            f"({thresholds.size}, {n_bins})."
        )
    if not np.all(np.isfinite(thresholds)):
        raise ValueError(f"The {process_name} thresholds have a nonfinite value.")
    if np.any(thresholds < 0.0):
        raise ValueError(f"The {process_name} thresholds have a negative value.")
    if not np.all(np.isfinite(channels)):
        raise ValueError(f"The {process_name} channel tables have a nonfinite value.")
    if np.any(channels < 0.0):
        raise ValueError(f"The {process_name} channel tables have a negative value.")
    if not np.allclose(
        np.sum(channels, axis=0),
        total_table,
        rtol=1.0e-13,
        atol=0.0,
    ):
        raise ValueError(
            f"The {process_name} channel tables do not sum to the total table."
        )
    thresholds.flags.writeable = False
    channels.flags.writeable = False
    return thresholds, channels


def _is_electron(text: str) -> bool:
    return text.strip().casefold() in {"e", "e-", "electron"}


def load_lxcat_text(
    path: Path | str,
    default_target: str | None = None,
    *,
    strict: bool = False,
) -> list[CrossSectionProcess]:
    del default_target
    return parse_cross_section_file(path, strict=strict)


def _select_processes(
    processes: Iterable[CrossSectionProcess],
    kinds: set[str],
    target: str,
    *,
    electron: bool,
) -> list[CrossSectionProcess]:
    target_key = target.strip().casefold()
    selected: list[CrossSectionProcess] = []
    for process in processes:
        if process.process_type not in kinds:
            continue
        if process.target_particle.strip().casefold() != target_key:
            continue
        if _is_electron(process.incident_particle) != electron:
            continue
        selected.append(process)
    return selected


def _sum_processes(
    processes: list[CrossSectionProcess],
) -> tuple[np.ndarray, np.ndarray]:
    energies = np.unique(
        np.concatenate([process.energy_ev for process in processes])
    )
    total = np.zeros_like(energies)
    for process in processes:
        total += np.interp(
            energies,
            process.energy_ev,
            process.sigma_m2,
            left=0.0,
            right=0.0,
        )
    return energies, total


def _validate_uniform_request(e_max: float, n_bins: int) -> None:
    if e_max <= 0.0:
        raise ValueError("The maximum energy must be positive.")
    if n_bins < 2:
        raise ValueError("The cross-section table needs two or more bins.")


def _build_uniform_table(
    energy_ev: np.ndarray,
    sigma_m2: np.ndarray,
    e_max: float,
    n_bins: int,
) -> tuple[float, float, np.ndarray]:
    _validate_uniform_request(e_max, n_bins)
    if energy_ev.size == 0 or sigma_m2.size == 0:
        raise ValueError("The cross-section table has no values.")
    if energy_ev.shape != sigma_m2.shape:
        raise ValueError("The energy and cross-section arrays have different shapes.")
    if not np.all(np.isfinite(energy_ev)) or not np.all(np.isfinite(sigma_m2)):
        raise ValueError("The cross-section table has a nonfinite value.")
    if np.any(energy_ev < 0.0) or np.any(sigma_m2 < 0.0):
        raise ValueError("The cross-section table has a negative value.")
    if energy_ev.size > 1 and np.any(np.diff(energy_ev) <= 0.0):
        raise ValueError("Energy values do not increase.")

    e_min = 0.0
    grid = np.linspace(e_min, e_max, n_bins)
    sigma = np.interp(
        grid,
        energy_ev,
        sigma_m2,
        left=0.0,
        right=0.0,
    )
    inv_de = (n_bins - 1) / (e_max - e_min)
    return e_min, inv_de, sigma


def _validate_confirmed_mapping(
    processes: Iterable[CrossSectionProcess],
    *,
    strict: bool,
) -> None:
    if not strict:
        return
    for process in processes:
        if not process.mapping_confirmed:
            raise ValueError(
                f"The mapping for {process.process_type} is not confirmed."
            )


def _process_threshold(
    process: CrossSectionProcess,
    *,
    strict: bool,
) -> float:
    named_values = (
        ("threshold", process.threshold_ev),
        ("energy loss", process.energy_loss_ev),
    )
    finite_values: list[float] = []
    for name, raw_value in named_values:
        if raw_value is None:
            continue
        value = float(raw_value)
        if not np.isfinite(value):
            if strict:
                raise ValueError(
                    f"The {process.process_type} {name} is not finite."
                )
            continue
        if value < 0.0:
            raise ValueError(
                f"The {process.process_type} {name} is negative."
            )
        finite_values.append(value)

    if not finite_values:
        if strict:
            raise ValueError(
                f"The {process.process_type} process has no finite threshold."
            )
        return 0.0
    if (
        strict
        and len(finite_values) == 2
        and not np.isclose(
            finite_values[0],
            finite_values[1],
            rtol=1.0e-12,
            atol=1.0e-12,
        )
    ):
        raise ValueError(
            f"The {process.process_type} threshold and energy loss do not agree."
        )
    return finite_values[0]


def _electron_product_count(token: str) -> int:
    value = token.strip().casefold()
    if value in {"e", "e-", "electron"}:
        return 1
    match = re.fullmatch(r"(\d+)\s*(?:e|e-|electrons?)", value)
    if match is None:
        return 0
    return int(match.group(1))


def _has_unsupported_ionization_products(process: CrossSectionProcess) -> bool:
    if "->" not in process.label:
        return False
    products_text = process.label.split("->", 1)[1].split(",", 1)[0]
    product_tokens = re.split(r"\s+\+\s+", products_text.strip())
    electron_count = sum(
        _electron_product_count(token) for token in product_tokens
    )
    positive_ion_count = 0
    for token in product_tokens:
        value = token.strip().replace(" ", "")
        if _electron_product_count(token):
            continue
        if "++" in value or re.search(r"\^(?:\{)?[2-9]\d*\+", value):
            return True
        if value.endswith("+"):
            positive_ion_count += 1
    return electron_count > 2 or positive_ion_count > 1


def _build_channel_tables(
    processes: list[CrossSectionProcess],
    e_max: float,
    n_bins: int,
    *,
    strict: bool,
    reject_multiple_products: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    thresholds: list[float] = []
    channel_tables: list[np.ndarray] = []
    grid = np.linspace(0.0, e_max, n_bins)
    _validate_confirmed_mapping(processes, strict=strict)
    for process in processes:
        if (
            strict
            and reject_multiple_products
            and _has_unsupported_ionization_products(process)
        ):
            raise ValueError(
                "The ionization process has multiple products that the "
                "collision model cannot represent."
            )
        threshold = _process_threshold(process, strict=strict)
        _, _, table = _build_uniform_table(
            process.energy_ev,
            process.sigma_m2,
            e_max,
            n_bins,
        )
        table[grid < threshold] = 0.0
        thresholds.append(threshold)
        channel_tables.append(table)

    if channel_tables:
        channels = np.asarray(channel_tables, dtype=np.float64)
        total = np.sum(channels, axis=0, dtype=np.float64)
    else:
        channels = np.empty((0, n_bins), dtype=np.float64)
        total = np.zeros(n_bins, dtype=np.float64)
    return np.asarray(thresholds, dtype=np.float64), channels, total


def _build_electron_tables(
    processes: list[CrossSectionProcess],
    target: str,
    e_max: float,
    n_bins: int,
    *,
    strict: bool,
) -> ElectronXSections:
    elastic = _select_processes(
        processes,
        {"ELASTIC"},
        target,
        electron=True,
    )
    if not elastic:
        elastic = _select_processes(
            processes,
            {"EFFECTIVE"},
            target,
            electron=True,
        )
    if not elastic:
        raise ValueError("No electron elastic cross section is available for the target.")
    _validate_confirmed_mapping(elastic, strict=strict)
    elastic_energy, elastic_sigma = _sum_processes(elastic[:1])

    excitation = _select_processes(
        processes,
        {"EXCITATION"},
        target,
        electron=True,
    )
    if strict and not excitation:
        raise ValueError("No electron excitation process is available for the target.")
    (
        excitation_thresholds_ev,
        excitation_channel_tables,
        sigma_excitation,
    ) = _build_channel_tables(
        excitation,
        e_max,
        n_bins,
        strict=strict,
        reject_multiple_products=False,
    )

    ionization = _select_processes(
        processes,
        {"IONIZATION"},
        target,
        electron=True,
    )
    (
        ionization_thresholds_ev,
        ionization_channel_tables,
        sigma_ionization,
    ) = _build_channel_tables(
        ionization,
        e_max,
        n_bins,
        strict=strict,
        reject_multiple_products=True,
    )

    e_min, inv_de, sigma_elastic = _build_uniform_table(
        elastic_energy,
        elastic_sigma,
        e_max,
        n_bins,
    )
    return ElectronXSections(
        e_min=e_min,
        inv_de=inv_de,
        sigma_elastic=sigma_elastic,
        sigma_excitation=sigma_excitation,
        sigma_ionization=sigma_ionization,
        excitation_thresholds_ev=excitation_thresholds_ev,
        excitation_channel_tables=excitation_channel_tables,
        ionization_thresholds_ev=ionization_thresholds_ev,
        ionization_channel_tables=ionization_channel_tables,
    )


def build_electron_tables_from_lxcat(
    path: Path | str,
    target: str,
    e_max: float,
    n_bins: int,
    *,
    strict: bool = False,
) -> ElectronXSections:
    processes = load_lxcat_text(path, default_target=target, strict=strict)
    return _build_electron_tables(
        processes,
        target,
        e_max,
        n_bins,
        strict=strict,
    )


def _build_ion_tables(
    processes: list[CrossSectionProcess],
    target: str,
    e_max: float,
    n_bins: int,
    *,
    fallback_cex: float | None,
    strict: bool,
) -> IonXSections:
    if fallback_cex is not None and fallback_cex < 0.0:
        raise ValueError("The fallback cross section is negative.")

    charge_exchange = _select_processes(
        processes,
        {"CHARGE EXCHANGE"},
        target,
        electron=False,
    )
    backscatter = _select_processes(
        processes,
        {"BACKSCATTER"},
        target,
        electron=False,
    )
    if charge_exchange:
        cex_energy, cex_sigma = _sum_processes(charge_exchange)
    elif backscatter and strict:
        raise ValueError(
            "BACKSCATTER is not a confirmed CHARGE EXCHANGE process."
        )
    elif strict:
        raise ValueError(
            "No confirmed ion charge-exchange process is available for the target."
        )
    elif fallback_cex is not None and fallback_cex > 0.0:
        cex_energy = np.array([0.0, e_max], dtype=float)
        cex_sigma = np.array([fallback_cex, fallback_cex], dtype=float)
    elif backscatter:
        cex_energy, cex_sigma = _sum_processes(backscatter)
    else:
        cex_energy = np.array([0.0, e_max], dtype=float)
        cex_sigma = np.zeros(2, dtype=float)

    elastic = _select_processes(
        processes,
        {"ELASTIC", "EFFECTIVE"},
        target,
        electron=False,
    )
    if elastic:
        elastic_energy, elastic_sigma = _sum_processes(elastic[:1])
    else:
        elastic_energy = np.array([0.0, e_max], dtype=float)
        elastic_sigma = np.zeros(2, dtype=float)

    e_min, inv_de, sigma_cex = _build_uniform_table(
        cex_energy,
        cex_sigma,
        e_max,
        n_bins,
    )
    _, _, sigma_elastic = _build_uniform_table(
        elastic_energy,
        elastic_sigma,
        e_max,
        n_bins,
    )
    return IonXSections(
        e_min=e_min,
        inv_de=inv_de,
        sigma_cex=sigma_cex,
        sigma_elastic=sigma_elastic,
    )


def build_ion_tables_from_lxcat(
    path: Path | str,
    target: str,
    e_max: float,
    n_bins: int,
    fallback_cex: float | None = None,
    *,
    strict: bool = False,
) -> IonXSections:
    processes = load_lxcat_text(path, default_target=target, strict=strict)
    return _build_ion_tables(
        processes,
        target,
        e_max,
        n_bins,
        fallback_cex=fallback_cex,
        strict=strict,
    )


def build_constant_electron_tables(
    sigma_el: float,
    sigma_exc: float,
    sigma_ion: float,
    e_max: float,
    n_bins: int,
    *,
    excitation_threshold_ev: float = 0.0,
    ionization_threshold_ev: float = 0.0,
) -> ElectronXSections:
    if min(sigma_el, sigma_exc, sigma_ion) < 0.0:
        raise ValueError("A constant electron cross section is negative.")
    if not np.isfinite(excitation_threshold_ev) or excitation_threshold_ev < 0.0:
        raise ValueError("The excitation threshold is invalid.")
    if not np.isfinite(ionization_threshold_ev) or ionization_threshold_ev < 0.0:
        raise ValueError("The ionization threshold is invalid.")
    _validate_uniform_request(e_max, n_bins)
    e_min = 0.0
    inv_de = (n_bins - 1) / (e_max - e_min)
    grid = np.linspace(e_min, e_max, n_bins)
    sigma_elastic = np.full(n_bins, sigma_el, dtype=float)
    sigma_excitation = np.full(n_bins, sigma_exc, dtype=float)
    sigma_ionization = np.full(n_bins, sigma_ion, dtype=float)
    sigma_excitation[grid < excitation_threshold_ev] = 0.0
    sigma_ionization[grid < ionization_threshold_ev] = 0.0
    return ElectronXSections(
        e_min=e_min,
        inv_de=inv_de,
        sigma_elastic=sigma_elastic,
        sigma_excitation=sigma_excitation,
        sigma_ionization=sigma_ionization,
        excitation_thresholds_ev=np.array(
            [excitation_threshold_ev],
            dtype=np.float64,
        ),
        excitation_channel_tables=sigma_excitation[np.newaxis, :],
        ionization_thresholds_ev=np.array(
            [ionization_threshold_ev],
            dtype=np.float64,
        ),
        ionization_channel_tables=sigma_ionization[np.newaxis, :],
    )


def build_constant_ion_tables(
    sigma_cex: float,
    sigma_elastic: float,
    e_max: float,
    n_bins: int,
) -> IonXSections:
    if min(sigma_cex, sigma_elastic) < 0.0:
        raise ValueError("A constant ion cross section is negative.")
    _validate_uniform_request(e_max, n_bins)
    e_min = 0.0
    inv_de = (n_bins - 1) / (e_max - e_min)
    sigma_cex_array = np.full(n_bins, sigma_cex, dtype=float)
    sigma_elastic_array = np.full(n_bins, sigma_elastic, dtype=float)
    return IonXSections(
        e_min=e_min,
        inv_de=inv_de,
        sigma_cex=sigma_cex_array,
        sigma_elastic=sigma_elastic_array,
    )


def load_cross_sections_from_custom_file(
    path: Path | str,
    e_max: float,
    n_bins: int,
    *,
    target: str = "Ar",
    strict: bool = False,
    ion_e_max: float | None = None,
    ion_n_bins: int | None = None,
) -> tuple[ElectronXSections, IonXSections]:
    processes = load_lxcat_text(path, default_target=target, strict=strict)
    electron_tables = _build_electron_tables(
        processes,
        target,
        e_max,
        n_bins,
        strict=strict,
    )
    ion_tables = _build_ion_tables(
        processes,
        target,
        e_max if ion_e_max is None else ion_e_max,
        n_bins if ion_n_bins is None else ion_n_bins,
        fallback_cex=None,
        strict=strict,
    )
    return electron_tables, ion_tables


__all__ = [
    "CrossSectionProcess",
    "ElectronXSections",
    "IonXSections",
    "build_constant_electron_tables",
    "build_constant_ion_tables",
    "build_electron_tables_from_lxcat",
    "build_ion_tables_from_lxcat",
    "load_cross_sections_from_custom_file",
    "load_lxcat_text",
    "parse_cs_txt",
]
