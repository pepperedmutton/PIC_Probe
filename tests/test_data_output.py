from hashlib import sha256
import json

import pandas as pd
import pytest

from core.config import Config
from core.data_output import make_run_id, read_dataset, write_dataset
from core.provenance import canonical_json


def make_records() -> tuple[pd.DataFrame, pd.DataFrame]:
    config = Config.research()
    config_json = config.canonical_json()
    config_sha256 = sha256(canonical_json(json.loads(config_json)).encode()).hexdigest()
    run_id = make_run_id(config_sha256, 11, 0)
    runs = pd.DataFrame(
        [
            {
                "run_id": run_id,
                "config_sha256": config_sha256,
                "config_json": config_json,
                "root_seed": 11,
                "replicate": 0,
                "ne_m3": config.N0,
                "te_ev": config.Te,
                "ti_ev": config.Ti,
                "vp_v": 0.0,
                "pressure_pa": config.pressure_pa,
                "gas": "Ar",
                "probe_radius_m": config.R_MIN,
                "probe_length_m": 1.0,
                "n_cells": config.N_CELLS,
                "dt_s": config.DT,
                "n_particles": 100,
                "cross_section_hashes_json": json.dumps({}),
                "stability_pass": False,
                "convergence_pass": False,
                "quality_status": "PREVIEW",
                "software_version": "test",
                "physics_model_version": "2",
                "data_schema_version": 1,
            }
        ]
    )
    curves = pd.DataFrame(
        [
            {
                "run_id": run_id,
                "bias_v": -1.0,
                "replicate": 0,
                "electron_current_a": 1.0,
                "ion_current_a": 0.25,
                "total_current_a": 0.75,
                "current_sem_a": 0.01,
                "sample_count": 10,
                "converged": False,
            }
        ]
    )
    return runs, curves


def test_dataset_round_trip(tmp_path) -> None:
    runs, curves = make_records()
    write_dataset(tmp_path, runs, curves, {"quality_status": "PREVIEW"})
    loaded_runs, loaded_curves, manifest = read_dataset(tmp_path)
    pd.testing.assert_frame_equal(loaded_runs, runs)
    pd.testing.assert_frame_equal(loaded_curves, curves)
    assert manifest["dataset_schema_version"] == 1


def test_dataset_detects_manifest_tampering(tmp_path) -> None:
    runs, curves = make_records()
    write_dataset(tmp_path, runs, curves, {"quality_status": "PREVIEW"})
    path = tmp_path / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["quality_status"] = "PASS"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        read_dataset(tmp_path)
