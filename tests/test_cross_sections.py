from pathlib import Path

import numpy as np
import pytest

from core.cross_sections import (
    build_electron_tables_from_lxcat,
    load_cross_sections_from_custom_file,
)
from core.lxcat_parser import parse_cross_section_file


def make_process_block(
    process_type: str,
    incident: str,
    target: str,
    table_max_ev: float,
    *,
    energy_loss_ev: float | None = None,
    sigma_start_m2: float = 1.0e-20,
    sigma_end_m2: float = 2.0e-20,
    process_label_type: str | None = None,
) -> str:
    label_type = process_type if process_label_type is None else process_label_type
    lines = [process_type, target]
    if energy_loss_ev is not None:
        lines.append(f"{energy_loss_ev:g}")
    elif incident == "e" and process_type in {"ELASTIC", "EFFECTIVE"}:
        lines.append("1.36e-5")
    lines.extend(
        [
            f"SPECIES: {incident} / {target}",
            f"PROCESS: {incident} + {target} -> products, {label_type}",
            "COLUMNS: Energy (eV) | Cross section (m2)",
            "-----",
            f"0 {sigma_start_m2:g}",
            f"{table_max_ev:g} {sigma_end_m2:g}",
            "-----",
        ]
    )
    return "\n".join(lines)


@pytest.fixture
def legacy_cross_section_file(tmp_path: Path) -> Path:
    """Create an incomplete legacy-style input without redistributed data."""
    path = tmp_path / "legacy_cross_sections.txt"
    path.write_text(
        "\n\n".join(
            [
                make_process_block("ELASTIC", "e", "Ar", 200.0),
                make_process_block(
                    "IONIZATION",
                    "e",
                    "Ar",
                    200.0,
                    energy_loss_ev=15.76,
                ),
                make_process_block("BACKSCATTER", "Ar+", "Ar", 200.0),
            ]
        ),
        encoding="utf-8",
    )
    return path


def write_complete_cross_sections(
    tmp_path: Path,
    *,
    include_ionization: bool = True,
    electron_elastic_max_ev: float = 20.0,
    ion_incident: str = "Ar+",
) -> Path:
    blocks = [
        make_process_block("ELASTIC", "e", "Ar", electron_elastic_max_ev),
        make_process_block(
            "EXCITATION",
            "e",
            "Ar",
            20.0,
            energy_loss_ev=11.5,
        ),
    ]
    if include_ionization:
        blocks.append(
            make_process_block(
                "IONIZATION",
                "e",
                "Ar",
                20.0,
                energy_loss_ev=15.76,
            )
        )
    blocks.extend(
        [
            make_process_block("BACKSCATTER", ion_incident, "Ar", 20.0),
            make_process_block("ISOTROPIC", ion_incident, "Ar", 20.0),
        ]
    )
    path = tmp_path / "complete_cross_sections.txt"
    path.write_text("\n\n".join(blocks), encoding="utf-8")
    return path


def test_legacy_cross_section_file_is_available_in_permissive_mode(
    legacy_cross_section_file: Path,
) -> None:
    processes = parse_cross_section_file(
        legacy_cross_section_file,
        strict=False,
    )
    process_types = {process.process_type for process in processes}
    assert process_types == {"ELASTIC", "IONIZATION", "BACKSCATTER"}
    backscatter = next(
        process for process in processes if process.process_type == "BACKSCATTER"
    )
    assert not backscatter.mapping_confirmed


def test_permissive_tables_preserve_available_channels(
    legacy_cross_section_file: Path,
) -> None:
    electron, ion = load_cross_sections_from_custom_file(
        legacy_cross_section_file,
        200.0,
        2001,
        target="Ar",
        strict=False,
        ion_e_max=200.0,
        ion_n_bins=2001,
    )
    assert electron.excitation_channel_tables.shape == (0, 2001)
    assert electron.ionization_channel_tables.shape == (1, 2001)
    assert np.allclose(
        electron.sigma_excitation,
        np.sum(electron.excitation_channel_tables, axis=0),
    )
    assert np.allclose(
        electron.sigma_ionization,
        np.sum(electron.ionization_channel_tables, axis=0),
    )
    assert np.max(ion.sigma_cex) == 0.0
    assert np.max(ion.sigma_elastic) == 0.0


def test_strict_parser_rejects_incomplete_legacy_cross_sections(
    legacy_cross_section_file: Path,
) -> None:
    with pytest.raises(ValueError, match="No electron excitation process"):
        parse_cross_section_file(legacy_cross_section_file, strict=True)


def test_strict_table_loader_rejects_incomplete_legacy_cross_sections(
    legacy_cross_section_file: Path,
) -> None:
    with pytest.raises(ValueError, match="No electron excitation process"):
        load_cross_sections_from_custom_file(
            legacy_cross_section_file,
            200.0,
            2001,
            target="Ar",
            strict=True,
            ion_e_max=200.0,
            ion_n_bins=2001,
        )


@pytest.mark.parametrize(
    ("keyword_type", "label_type"),
    [("ELASTIC", "EFFECTIVE"), ("EFFECTIVE", "ELASTIC")],
)
def test_strict_parser_rejects_keyword_process_type_mismatch(
    tmp_path: Path,
    keyword_type: str,
    label_type: str,
) -> None:
    path = tmp_path / "mismatched_process_type.txt"
    path.write_text(
        make_process_block(
            keyword_type,
            "e",
            "Ar",
            20.0,
            process_label_type=label_type,
        ),
        encoding="utf-8",
    )

    permissive = parse_cross_section_file(path, strict=False)
    assert permissive[0].process_type == keyword_type
    with pytest.raises(ValueError, match="keyword and PROCESS field do not agree"):
        parse_cross_section_file(path, strict=True)


@pytest.mark.parametrize("extra_type", ["ELASTIC", "EFFECTIVE"])
def test_strict_electron_loader_rejects_multiple_elastic_candidates(
    tmp_path: Path,
    extra_type: str,
) -> None:
    path = write_complete_cross_sections(tmp_path)
    path.write_text(
        path.read_text(encoding="utf-8")
        + "\n\n"
        + make_process_block(extra_type, "e", "Ar", 20.0),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="strict electron loader has 2 elastic"):
        load_cross_sections_from_custom_file(
            path,
            20.0,
            21,
            strict=True,
            ion_e_max=20.0,
            ion_n_bins=21,
            confirm_symmetric_backscatter_as_cex=True,
        )


@pytest.mark.parametrize("extra_type", ["ELASTIC", "EFFECTIVE", "ISOTROPIC"])
def test_strict_ion_loader_rejects_multiple_elastic_candidates(
    tmp_path: Path,
    extra_type: str,
) -> None:
    path = write_complete_cross_sections(tmp_path)
    path.write_text(
        path.read_text(encoding="utf-8")
        + "\n\n"
        + make_process_block(extra_type, "Ar+", "Ar", 20.0),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="strict ion loader has 2 elastic"):
        load_cross_sections_from_custom_file(
            path,
            20.0,
            21,
            strict=True,
            ion_e_max=20.0,
            ion_n_bins=21,
            confirm_symmetric_backscatter_as_cex=True,
        )


def test_phelps_isotropic_process_is_parsed(tmp_path) -> None:
    path = tmp_path / "phelps_isotropic.txt"
    path.write_text(
        "\n".join(
            [
                "SPECIES: Ar+ / Ar",
                "PROCESS: Ar+ + Ar -> Ar+ + Ar, ISOTROPIC",
                "COLUMNS: Energy (eV) | Cross section (m2)",
                "-----",
                "0 1e-20",
                "20 2e-20",
                "-----",
            ]
        ),
        encoding="utf-8",
    )
    processes = parse_cross_section_file(path)
    assert len(processes) == 1
    assert processes[0].process_type == "ISOTROPIC"
    assert processes[0].scattering_model == "isotropic"


def test_strict_loader_rejects_unconfirmed_backscatter(tmp_path) -> None:
    path = write_complete_cross_sections(tmp_path)
    with pytest.raises(ValueError, match="BACKSCATTER is not a confirmed"):
        load_cross_sections_from_custom_file(
            path,
            20.0,
            21,
            strict=True,
            ion_e_max=20.0,
            ion_n_bins=21,
        )


def test_strict_loader_accepts_confirmed_symmetric_backscatter(tmp_path) -> None:
    path = write_complete_cross_sections(tmp_path)
    electron, ion = load_cross_sections_from_custom_file(
        path,
        20.0,
        21,
        strict=True,
        ion_e_max=20.0,
        ion_n_bins=21,
        confirm_symmetric_backscatter_as_cex=True,
    )
    assert np.max(electron.sigma_ionization) > 0.0
    assert np.max(ion.sigma_cex) > 0.0
    assert np.max(ion.sigma_elastic) > 0.0


def test_strict_loader_requires_electron_ionization(tmp_path) -> None:
    path = write_complete_cross_sections(tmp_path, include_ionization=False)
    with pytest.raises(ValueError, match="No electron ionization process"):
        load_cross_sections_from_custom_file(
            path,
            20.0,
            21,
            strict=True,
            ion_e_max=20.0,
            ion_n_bins=21,
            confirm_symmetric_backscatter_as_cex=True,
        )


def test_strict_loader_requires_energy_coverage(tmp_path) -> None:
    path = write_complete_cross_sections(
        tmp_path,
        electron_elastic_max_ev=10.0,
    )
    with pytest.raises(ValueError, match="below the requested 20 eV"):
        load_cross_sections_from_custom_file(
            path,
            20.0,
            21,
            strict=True,
            ion_e_max=20.0,
            ion_n_bins=21,
            confirm_symmetric_backscatter_as_cex=True,
        )


def test_effective_table_has_inelastic_channels_removed(tmp_path) -> None:
    path = tmp_path / "effective.txt"
    path.write_text(
        "\n\n".join(
            [
                make_process_block(
                    "EFFECTIVE",
                    "e",
                    "Ar",
                    20.0,
                    sigma_start_m2=6.0e-20,
                    sigma_end_m2=6.0e-20,
                ),
                make_process_block(
                    "EXCITATION",
                    "e",
                    "Ar",
                    20.0,
                    energy_loss_ev=0.0,
                    sigma_start_m2=1.0e-20,
                    sigma_end_m2=1.0e-20,
                ),
                make_process_block(
                    "IONIZATION",
                    "e",
                    "Ar",
                    20.0,
                    energy_loss_ev=0.0,
                    sigma_start_m2=2.0e-20,
                    sigma_end_m2=2.0e-20,
                ),
            ]
        ),
        encoding="utf-8",
    )
    tables = build_electron_tables_from_lxcat(
        path,
        "Ar",
        20.0,
        21,
        strict=True,
    )
    assert np.allclose(tables.sigma_elastic, 3.0e-20)
    assert np.allclose(
        tables.sigma_elastic
        + tables.sigma_excitation
        + tables.sigma_ionization,
        6.0e-20,
    )


def test_effective_table_rejects_negative_derived_elastic(tmp_path) -> None:
    path = tmp_path / "invalid_effective.txt"
    path.write_text(
        "\n\n".join(
            [
                make_process_block(
                    "EFFECTIVE",
                    "e",
                    "Ar",
                    20.0,
                    sigma_start_m2=1.0e-20,
                    sigma_end_m2=1.0e-20,
                ),
                make_process_block(
                    "EXCITATION",
                    "e",
                    "Ar",
                    20.0,
                    energy_loss_ev=0.0,
                    sigma_start_m2=1.0e-20,
                    sigma_end_m2=1.0e-20,
                ),
                make_process_block(
                    "IONIZATION",
                    "e",
                    "Ar",
                    20.0,
                    energy_loss_ev=0.0,
                    sigma_start_m2=1.0e-20,
                    sigma_end_m2=1.0e-20,
                ),
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="smaller than the summed inelastic"):
        build_electron_tables_from_lxcat(
            path,
            "Ar",
            20.0,
            21,
            strict=True,
        )


@pytest.mark.parametrize("incident", ["Ar", "Ar-", "Ar2+", "Xe+"])
def test_strict_ion_loader_requires_configured_incident_species(
    tmp_path,
    incident,
) -> None:
    path = write_complete_cross_sections(tmp_path, ion_incident=incident)
    with pytest.raises(ValueError, match="unexpected incident species"):
        load_cross_sections_from_custom_file(
            path,
            20.0,
            21,
            strict=True,
            ion_e_max=20.0,
            ion_n_bins=21,
            confirm_symmetric_backscatter_as_cex=True,
            ion_species="Ar+",
        )
