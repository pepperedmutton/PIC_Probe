from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from validation.run_cenian2005 import (
    DEFAULT_EXPERIMENTAL_CSV,
    PilotSettings,
    accelerated_electron_cfl,
    execute_validation,
    load_experimental_points,
    main,
    make_cenian_config,
    require_experimental_dataset_file,
    require_lxcat_database_file,
    require_lxcat_dataset_file,
    select_experimental_points,
)


def _lxcat_files(tmp_path: Path) -> tuple[Path, Path]:
    electron = tmp_path / "phelps_electron.txt"
    ion = tmp_path / "phelps.txt"
    electron.write_text("DATABASE: Phelps\n", encoding="utf-8")
    ion.write_text("DATABASE: Phelps\n", encoding="utf-8")
    return electron, ion


def _allow_test_lxcat(monkeypatch) -> None:
    monkeypatch.setattr(
        "validation.run_cenian2005.require_lxcat_dataset_file",
        lambda path, role: Path(path).resolve(),
    )


class FakeSimulation:
    calls: list[dict[str, object]] = []

    def __init__(
        self,
        config,
        *,
        n_particles: int,
        v_bias: float,
        sigma_cex: float,
        probe_length: float,
        seed: int,
    ) -> None:
        self.config = config
        self.v_bias = float(v_bias)
        self.seed = int(seed)
        self.probe_length = float(probe_length)
        self.calls.append(
            {
                "n_particles": n_particles,
                "v_bias": self.v_bias,
                "sigma_cex": sigma_cex,
                "probe_length": self.probe_length,
                "seed": self.seed,
            }
        )

    def run(self, n_steps: int, n_warmup: int):
        sample_count = n_steps - n_warmup
        return SimpleNamespace(
            avg_current=self.v_bias * 1.0e-6,
            current_sem=1.0e-7,
            sample_count=sample_count,
            batch_mean_count=2,
        )

    def run_manifest(self) -> dict[str, object]:
        return {
            "manifest_sha256": f"manifest-{self.seed}-{self.v_bias:g}",
            "simulation": {
                "status": "READY",
                "state_sha256": f"state-{self.seed}-{self.v_bias:g}",
                "physics_limitations": ["research_preview"],
                "numerical_diagnostics": {
                    "status": "PASS",
                    "stability_warnings": [],
                    "runtime_warnings": [],
                    "energy_table_overflow_lookups": 0,
                    "electron_particle_ledger_residual": 0,
                    "ion_particle_ledger_residual": 0,
                },
            },
        }


def test_experimental_csv_has_the_selected_case() -> None:
    points = load_experimental_points()
    assert len(points) == 11
    assert points[0].voltage_v == -60.0
    assert points[-1].voltage_v == -10.0
    assert points[0].current_a == pytest.approx(-2.055e-5)
    assert all(point.current_a < 0.0 for point in points)


def test_experimental_dataset_hashes_are_fixed(tmp_path: Path) -> None:
    csv_path = tmp_path / DEFAULT_EXPERIMENTAL_CSV.name
    provenance_path = csv_path.with_suffix(".provenance.json")
    csv_path.write_bytes(DEFAULT_EXPERIMENTAL_CSV.read_bytes())
    source_provenance = DEFAULT_EXPERIMENTAL_CSV.with_suffix(
        ".provenance.json"
    )
    provenance_path.write_bytes(source_provenance.read_bytes())
    checked_csv, checked_provenance, _ = require_experimental_dataset_file(
        csv_path
    )
    assert checked_csv == csv_path.resolve()
    assert checked_provenance == provenance_path.resolve()
    csv_path.write_text("changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="csv hash does not match"):
        require_experimental_dataset_file(csv_path)


def test_experimental_voltage_subset_keeps_source_order() -> None:
    points = load_experimental_points()
    selected = select_experimental_points(points, (-10.0, -50.0, -30.0))
    assert [point.voltage_v for point in selected] == [-50.0, -30.0, -10.0]
    with pytest.raises(ValueError, match="not in the fixed experiment grid"):
        select_experimental_points(points, (-12.0,))


def test_cenian_config_uses_strict_local_cross_sections(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _allow_test_lxcat(monkeypatch)
    electron, ion = _lxcat_files(tmp_path)
    settings = PilotSettings()
    config = make_cenian_config(settings, electron, ion)

    assert config.N0 == 7.15e13
    assert config.Te == 1.9
    assert config.Ti == 0.025
    assert config.P_Torr == 0.0013
    assert config.R_MIN == 313.0e-6
    assert config.R_MAX == 0.012
    assert config.V_WALL == 0.0
    assert config.CROSS_SECTION_STRICT is True
    assert config.CONFIRM_SYMMETRIC_BACKSCATTER_AS_CEX is True
    assert config.ION_INJECTION_BOHM is False
    assert Path(config.LXCAT_ELECTRON_FILE or "") == electron.resolve()
    assert Path(config.LXCAT_ION_FILE or "") == ion.resolve()
    assert config.stability_metrics().is_stable
    assert accelerated_electron_cfl(config) < 1.0


def test_lxcat_database_marker_is_necessary(tmp_path: Path) -> None:
    path = tmp_path / "not_biagi.txt"
    path.write_text("DATABASE: Phelps\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Biagi"):
        require_lxcat_database_file(path, "Biagi")


def test_pinned_lxcat_dataset_rejects_unrecorded_file(tmp_path: Path) -> None:
    electron, _ = _lxcat_files(tmp_path)
    with pytest.raises(ValueError, match="hash does not match"):
        require_lxcat_dataset_file(electron, "electron_argon_primary")


def test_plan_only_does_not_operate_the_simulation(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    _allow_test_lxcat(monkeypatch)
    electron, ion = _lxcat_files(tmp_path)
    FakeSimulation.calls.clear()
    result = main(
        [
            "--electron-lxcat",
            str(electron),
            "--ion-lxcat",
            str(ion),
            "--plan-only",
        ]
    )
    output = capsys.readouterr().out
    assert result == 0
    assert "research preview" in output
    assert '"absolute_current_no_scaling": true' in output
    assert '"ION_INJECTION_BOHM": false' in output
    assert '"config_paths_redacted": true' in output
    assert str(tmp_path) not in output
    assert FakeSimulation.calls == []


def test_validation_writes_direct_ampere_comparison(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _allow_test_lxcat(monkeypatch)
    electron, ion = _lxcat_files(tmp_path)
    settings = PilotSettings(
        particles=16,
        seeds=(101, 202),
        sample_steps=4,
        warmup_steps=2,
    )
    config = make_cenian_config(settings, electron, ion)
    points = load_experimental_points(DEFAULT_EXPERIMENTAL_CSV)
    output_dir = tmp_path / "output"
    FakeSimulation.calls.clear()

    paths = execute_validation(
        config,
        settings,
        points,
        DEFAULT_EXPERIMENTAL_CSV,
        output_dir,
        simulation_factory=FakeSimulation,
    )

    assert set(paths) == {
        "simulation_points",
        "comparison",
        "metrics",
        "manifest",
    }
    assert all(path.is_file() for path in paths.values())
    assert len(FakeSimulation.calls) == 22
    assert all(
        call["probe_length"] == 0.047 for call in FakeSimulation.calls
    )
    assert all(call["sigma_cex"] == 0.0 for call in FakeSimulation.calls)

    with paths["simulation_points"].open(
        "r", encoding="utf-8", newline=""
    ) as source:
        simulation_rows = list(csv.DictReader(source))
    assert float(
        simulation_rows[0]["source_aligned_current_A"]
    ) == pytest.approx(-60.0e-6)

    with paths["comparison"].open(
        "r", encoding="utf-8", newline=""
    ) as source:
        comparison_rows = list(csv.DictReader(source))
    assert float(
        comparison_rows[0]["simulation_source_aligned_current_A"]
    ) == pytest.approx(-60.0e-6)

    metrics = json.loads(paths["metrics"].read_text(encoding="utf-8"))
    assert metrics["absolute_current_no_scaling"] is True
    assert metrics["signed_current_no_scaling"] is True
    assert metrics["quality_status"] == "PREVIEW"
    assert metrics["simulation_row_count"] == 22
    assert math.isfinite(metrics["root_mean_square_error_A"])

    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    assert manifest["quality_status"] == "PREVIEW"
    assert manifest["absolute_current_no_scaling"] is True
    assert manifest["signed_current_no_scaling"] is True
    assert manifest["versions"]["physics_model_version"] == "3"
    assert "source_tree" in manifest
    assert manifest["plan"]["experiment"]["probe_length_m"] == 0.047
    assert (
        manifest["plan"]["experiment"]["boundary_mapping"]
        == "Maxwellian thermal plasma-bulk influx; ION_INJECTION_BOHM=False."
    )


def test_output_directory_must_be_empty(tmp_path: Path, monkeypatch) -> None:
    _allow_test_lxcat(monkeypatch)
    electron, ion = _lxcat_files(tmp_path)
    settings = PilotSettings(sample_steps=2, warmup_steps=0)
    config = make_cenian_config(settings, electron, ion)
    output_dir = tmp_path / "not_empty"
    output_dir.mkdir()
    (output_dir / "keep.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError, match="not empty"):
        execute_validation(
            config,
            settings,
            load_experimental_points(),
            DEFAULT_EXPERIMENTAL_CSV,
            output_dir,
            simulation_factory=FakeSimulation,
        )
