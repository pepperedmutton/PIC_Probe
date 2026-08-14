from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


_PROCESS_TYPES = {
    "ELASTIC",
    "EFFECTIVE",
    "EXCITATION",
    "IONIZATION",
    "ATTACHMENT",
    "CHARGE EXCHANGE",
    "BACKSCATTER",
}

_SCATTERING_MODELS = {
    "ELASTIC": "momentum-transfer",
    "EFFECTIVE": "effective-momentum-transfer",
    "EXCITATION": "inelastic-unspecified",
    "IONIZATION": "ionization-unspecified",
    "ATTACHMENT": "attachment-unspecified",
    "CHARGE EXCHANGE": "charge-exchange",
    "BACKSCATTER": "backscatter",
}

_NUMBER_PATTERN = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?"


@dataclass(frozen=True)
class CrossSectionProcess:
    kind: str
    target: str
    label: str
    energy_ev: np.ndarray
    sigma_m2: np.ndarray
    incident_particle: str = ""
    target_particle: str = ""
    process_type: str = ""
    threshold_ev: float | None = None
    energy_loss_ev: float | None = None
    energy_unit: str = "eV"
    cross_section_unit: str = "m2"
    scattering_model: str = "unspecified"
    source_path: str = ""
    source_sha256: str = ""
    source_energy_unit: str = "eV"
    source_cross_section_unit: str = "m2"
    mapping_confirmed: bool = True

    def __post_init__(self) -> None:
        if not self.process_type:
            object.__setattr__(self, "process_type", self.kind)
        if not self.target_particle:
            object.__setattr__(self, "target_particle", self.target)

    @property
    def energy(self) -> np.ndarray:
        return self.energy_ev

    @property
    def sigma(self) -> np.ndarray:
        return self.sigma_m2

    @property
    def mapping_status(self) -> str:
        return "confirmed" if self.mapping_confirmed else "unconfirmed"


@dataclass(frozen=True)
class CrossSectionData:
    energy: np.ndarray
    sigma: np.ndarray
    process: CrossSectionProcess | None = None

    @classmethod
    def from_process(cls, process: CrossSectionProcess) -> CrossSectionData:
        return cls(energy=process.energy_ev, sigma=process.sigma_m2, process=process)

    @property
    def incident_particle(self) -> str:
        return "" if self.process is None else self.process.incident_particle

    @property
    def target_particle(self) -> str:
        return "" if self.process is None else self.process.target_particle

    @property
    def process_type(self) -> str:
        return "" if self.process is None else self.process.process_type

    @property
    def threshold_ev(self) -> float | None:
        return None if self.process is None else self.process.threshold_ev

    @property
    def energy_loss_ev(self) -> float | None:
        return None if self.process is None else self.process.energy_loss_ev

    @property
    def energy_unit(self) -> str:
        return "" if self.process is None else self.process.energy_unit

    @property
    def cross_section_unit(self) -> str:
        return "" if self.process is None else self.process.cross_section_unit

    @property
    def scattering_model(self) -> str:
        return "" if self.process is None else self.process.scattering_model

    @property
    def source_path(self) -> str:
        return "" if self.process is None else self.process.source_path

    @property
    def source_sha256(self) -> str:
        return "" if self.process is None else self.process.source_sha256

    @property
    def mapping_confirmed(self) -> bool:
        return self.process is not None and self.process.mapping_confirmed


def _is_separator(line: str) -> bool:
    text = line.strip()
    return len(text) >= 5 and set(text) == {"-"}


def _next_nonempty(lines: list[str], start: int) -> int | None:
    for index in range(start, len(lines)):
        if lines[index].strip():
            return index
    return None


def _next_separator(lines: list[str], start: int) -> int | None:
    for index in range(start, len(lines)):
        if _is_separator(lines[index]):
            return index
    return None


def _normalize_process_type(text: str) -> str | None:
    value = " ".join(text.strip().upper().replace("_", " ").split())
    value = value.replace("CHARGE-EXCHANGE", "CHARGE EXCHANGE")
    if value in _PROCESS_TYPES:
        return value
    return None


def _infer_process_type(text: str) -> str | None:
    value = " ".join(text.upper().split())
    if "BACKSCAT" in value:
        return "BACKSCATTER"
    if "CHARGE EXCHANGE" in value or "CHARGE-EXCHANGE" in value:
        return "CHARGE EXCHANGE"
    if re.search(r"\bCEX\b", value):
        return "CHARGE EXCHANGE"
    if "IONIZATION" in value:
        return "IONIZATION"
    if "EXCITATION" in value:
        return "EXCITATION"
    if "ATTACH" in value:
        return "ATTACHMENT"
    if "EFFECTIVE" in value:
        return "EFFECTIVE"
    if "ELASTIC" in value or "MOMENTUM TRANSFER" in value:
        return "ELASTIC"
    return None


def _metadata_value(lines: list[str], prefix: str) -> str | None:
    prefix_upper = prefix.upper()
    for line in lines:
        text = line.strip()
        if text.upper().startswith(prefix_upper):
            return text.split(":", 1)[1].strip()
    return None


def _parse_species(text: str) -> tuple[str, str]:
    if "/" not in text:
        raise ValueError("The SPECIES field does not identify two particles.")
    incident, target = text.split("/", 1)
    incident = incident.strip()
    target = target.strip()
    if not incident or not target:
        raise ValueError("The SPECIES field does not identify two particles.")
    return incident, target


def _parse_target(text: str) -> tuple[str, str]:
    value = text.strip()
    if "<->" in value:
        target, state = value.split("<->", 1)
        return target.strip(), state.strip()
    if "->" in value:
        target, state = value.split("->", 1)
        return target.strip(), state.strip()
    return value, ""


def _compact_unit(text: str) -> str:
    return (
        text.strip()
        .lower()
        .replace("²", "2")
        .replace("^", "")
        .replace(" ", "")
    )


def _parse_units(lines: list[str]) -> tuple[str, str, float]:
    columns = _metadata_value(lines, "COLUMNS:")
    if columns is None:
        raise ValueError("Column units are not available.")
    match = re.search(
        r"energy\s*\(([^)]+)\).*?cross[- ]*section\s*\(([^)]+)\)",
        columns,
        flags=re.IGNORECASE,
    )
    if match is None:
        raise ValueError("Column units are not known.")

    source_energy_unit = match.group(1).strip()
    source_cross_section_unit = match.group(2).strip()
    energy_unit = _compact_unit(source_energy_unit)
    cross_section_unit = _compact_unit(source_cross_section_unit)

    if energy_unit != "ev":
        raise ValueError(f"Energy unit is not known: {source_energy_unit}.")
    if cross_section_unit == "m2":
        scale = 1.0
    elif cross_section_unit == "cm2":
        scale = 1.0e-4
    else:
        raise ValueError(
            f"Cross-section unit is not known: {source_cross_section_unit}."
        )
    return source_energy_unit, source_cross_section_unit, scale


def _parse_number(text: str) -> float:
    return float(text.replace("D", "E").replace("d", "e"))


def _parse_table(
    lines: list[str],
    start: int,
    sigma_scale: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    rows: list[tuple[float, float]] = []
    index = start
    while index < len(lines) and not _is_separator(lines[index]):
        text = lines[index].strip()
        if text:
            fields = text.split()
            if len(fields) < 2:
                raise ValueError("A cross-section row has less than two values.")
            try:
                energy = _parse_number(fields[0])
                sigma = _parse_number(fields[1]) * sigma_scale
            except ValueError as exc:
                raise ValueError("A cross-section row has an invalid value.") from exc
            rows.append((energy, sigma))
        index += 1

    if index >= len(lines):
        raise ValueError("The cross-section table has no end separator.")
    if not rows:
        raise ValueError("The cross-section table has no values.")

    values = np.asarray(rows, dtype=np.float64)
    energy_ev = values[:, 0]
    sigma_m2 = values[:, 1]
    if not np.all(np.isfinite(values)):
        raise ValueError("The cross-section table has a nonfinite value.")
    if np.any(energy_ev < 0.0) or np.any(sigma_m2 < 0.0):
        raise ValueError("The cross-section table has a negative value.")
    if energy_ev.size > 1 and np.any(np.diff(energy_ev) <= 0.0):
        raise ValueError("Energy values do not increase.")
    return energy_ev, sigma_m2, index


def _extract_energy_loss(
    process_type: str,
    header_lines: list[str],
    target_index: int | None,
) -> float | None:
    if process_type not in {"EXCITATION", "IONIZATION"}:
        return None

    parameter = _metadata_value(header_lines, "PARAM.:")
    if parameter is not None:
        match = re.search(
            rf"(?:^|\bE\s*=)\s*({_NUMBER_PATTERN})\s*eV\b",
            parameter,
            flags=re.IGNORECASE,
        )
        if match is not None:
            value = _parse_number(match.group(1))
            if value < 0.0:
                raise ValueError("The energy loss is negative.")
            return value

    start = 0 if target_index is None else target_index + 1
    for line in header_lines[start:]:
        text = line.strip()
        match = re.fullmatch(rf"({_NUMBER_PATTERN})(?:\s+{_NUMBER_PATTERN})?", text)
        if match is not None:
            value = _parse_number(match.group(1))
            if value < 0.0:
                raise ValueError("The energy loss is negative.")
            return value
    raise ValueError(f"The {process_type} process has no energy loss.")


def _make_process(
    header_lines: list[str],
    keyword: str | None,
    target_index: int | None,
    energy_ev: np.ndarray,
    sigma_m2: np.ndarray,
    source_path: str,
    source_sha256: str,
    source_energy_unit: str,
    source_cross_section_unit: str,
) -> CrossSectionProcess:
    process_label = _metadata_value(header_lines, "PROCESS:")
    if keyword is None:
        if process_label is None:
            raise ValueError("The PROCESS field is not available.")
        process_type = _infer_process_type(process_label)
        if process_type is None:
            raise ValueError(f"Process type is not known: {process_label}.")
    else:
        process_type = keyword
        inferred_type = (
            None if process_label is None else _infer_process_type(process_label)
        )
        if inferred_type is not None and inferred_type != process_type:
            compatible = {inferred_type, process_type} <= {"ELASTIC", "EFFECTIVE"}
            if not compatible:
                raise ValueError("The process keyword and PROCESS field do not agree.")

    species = _metadata_value(header_lines, "SPECIES:")
    if species is not None:
        incident_particle, target_particle = _parse_species(species)
    elif keyword is not None and target_index is not None:
        incident_particle = "e"
        target_particle, _ = _parse_target(header_lines[target_index])
    else:
        raise ValueError("The incident particle or target particle is not available.")

    state = ""
    if target_index is not None:
        target_from_line, state = _parse_target(header_lines[target_index])
        if target_from_line.casefold() != target_particle.casefold():
            raise ValueError("The target fields do not agree.")

    energy_loss_ev = _extract_energy_loss(
        process_type,
        header_lines,
        target_index,
    )
    label = process_label or state or f"{incident_particle} / {target_particle}"
    mapping_confirmed = process_type != "BACKSCATTER"
    return CrossSectionProcess(
        kind=process_type,
        target=target_particle,
        label=label,
        energy_ev=energy_ev,
        sigma_m2=sigma_m2,
        incident_particle=incident_particle,
        target_particle=target_particle,
        process_type=process_type,
        threshold_ev=energy_loss_ev,
        energy_loss_ev=energy_loss_ev,
        energy_unit="eV",
        cross_section_unit="m2",
        scattering_model=_SCATTERING_MODELS[process_type],
        source_path=source_path,
        source_sha256=source_sha256,
        source_energy_unit=source_energy_unit,
        source_cross_section_unit=source_cross_section_unit,
        mapping_confirmed=mapping_confirmed,
    )


def _looks_like_unknown_keyword(lines: list[str], index: int) -> bool:
    text = lines[index].strip()
    if not re.fullmatch(r"[A-Z][A-Z -]{2,39}", text):
        return False
    for candidate in lines[index + 1 : index + 13]:
        if candidate.strip().upper().startswith("COLUMNS:") or _is_separator(candidate):
            return True
    return False


def _is_electron(text: str) -> bool:
    return text.strip().casefold() in {"e", "e-", "electron"}


def parse_cross_section_file(
    file_path: str | Path,
    *,
    strict: bool = False,
) -> list[CrossSectionProcess]:
    path = Path(file_path).expanduser().resolve(strict=True)
    source_bytes = path.read_bytes()
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    lines = source_bytes.decode("utf-8-sig").splitlines()
    source_path = str(path)
    processes: list[CrossSectionProcess] = []

    index = 0
    while index < len(lines):
        text = lines[index].strip()
        keyword = _normalize_process_type(text)

        if keyword is not None:
            target_line_index = _next_nonempty(lines, index + 1)
            if target_line_index is None:
                raise ValueError("The target particle is not available.")
            separator_index = _next_separator(lines, target_line_index + 1)
            if separator_index is None:
                raise ValueError("The cross-section table has no start separator.")
            header_lines = lines[target_line_index:separator_index]
            source_energy_unit, source_cross_section_unit, sigma_scale = _parse_units(
                header_lines
            )
            energy_ev, sigma_m2, end_index = _parse_table(
                lines,
                separator_index + 1,
                sigma_scale,
            )
            processes.append(
                _make_process(
                    header_lines=header_lines,
                    keyword=keyword,
                    target_index=0,
                    energy_ev=energy_ev,
                    sigma_m2=sigma_m2,
                    source_path=source_path,
                    source_sha256=source_sha256,
                    source_energy_unit=source_energy_unit,
                    source_cross_section_unit=source_cross_section_unit,
                )
            )
            index = end_index + 1
            continue

        if text.upper().startswith("SPECIES:"):
            separator_index = _next_separator(lines, index + 1)
            if separator_index is None:
                raise ValueError("The cross-section table has no start separator.")
            header_lines = lines[index:separator_index]
            source_energy_unit, source_cross_section_unit, sigma_scale = _parse_units(
                header_lines
            )
            energy_ev, sigma_m2, end_index = _parse_table(
                lines,
                separator_index + 1,
                sigma_scale,
            )
            processes.append(
                _make_process(
                    header_lines=header_lines,
                    keyword=None,
                    target_index=None,
                    energy_ev=energy_ev,
                    sigma_m2=sigma_m2,
                    source_path=source_path,
                    source_sha256=source_sha256,
                    source_energy_unit=source_energy_unit,
                    source_cross_section_unit=source_cross_section_unit,
                )
            )
            index = end_index + 1
            continue

        if _looks_like_unknown_keyword(lines, index):
            raise ValueError(f"Process type is not known: {text}.")
        index += 1

    if strict:
        electron_processes = [
            process
            for process in processes
            if _is_electron(process.incident_particle)
        ]
        has_excitation = any(
            process.process_type == "EXCITATION"
            for process in electron_processes
        )
        if electron_processes and not has_excitation:
            raise ValueError("No electron excitation process is available.")
    return processes


def _legacy_key(
    process: CrossSectionProcess,
    *,
    strict: bool,
) -> str | None:
    electron = _is_electron(process.incident_particle)
    if electron and process.process_type in {"ELASTIC", "EFFECTIVE"}:
        return "electron_elastic"
    if electron and process.process_type == "EXCITATION":
        return "electron_excitation"
    if electron and process.process_type == "IONIZATION":
        return "electron_ionization"
    if electron and process.process_type == "ATTACHMENT":
        return "electron_attachment"
    if not electron and process.process_type == "CHARGE EXCHANGE":
        return "ion_cex"
    if not electron and process.process_type == "BACKSCATTER":
        return "ion_backscatter" if strict else "ion_cex"
    if not electron and process.process_type in {"ELASTIC", "EFFECTIVE"}:
        return "ion_elastic"
    return None


def _add_legacy_section(
    sections: dict[str, CrossSectionData],
    key: str,
    data: CrossSectionData,
) -> None:
    if key not in sections:
        sections[key] = data
        return
    number = 2
    while f"{key}_{number}" in sections:
        number += 1
    sections[f"{key}_{number}"] = data


def parse_cs_txt(
    file_path: str | Path,
    *,
    strict: bool = False,
) -> dict[str, CrossSectionData]:
    processes = parse_cross_section_file(file_path, strict=strict)
    sections: dict[str, CrossSectionData] = {}
    for process in processes:
        key = _legacy_key(process, strict=strict)
        if key is None:
            continue
        data = CrossSectionData.from_process(process)
        _add_legacy_section(sections, key, data)
    return sections
