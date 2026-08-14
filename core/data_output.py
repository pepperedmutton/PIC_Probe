from __future__ import annotations

from hashlib import sha256
import json
from numbers import Integral
from pathlib import Path
from string import hexdigits
from typing import Any, Iterable, Mapping
from uuid import uuid4

import numpy as np
import pandas as pd

from core.provenance import canonical_json, sha256_file


DATASET_SCHEMA_VERSION = 1

RUN_REQUIRED_COLUMNS = (
    "run_id",
    "config_sha256",
    "config_json",
    "root_seed",
    "replicate",
    "ne_m3",
    "te_ev",
    "ti_ev",
    "vp_v",
    "pressure_pa",
    "gas",
    "probe_radius_m",
    "probe_length_m",
    "n_cells",
    "dt_s",
    "n_particles",
    "cross_section_hashes_json",
    "stability_pass",
    "convergence_pass",
    "quality_status",
    "software_version",
    "physics_model_version",
    "data_schema_version",
)

CURVE_REQUIRED_COLUMNS = (
    "run_id",
    "bias_v",
    "replicate",
    "electron_current_a",
    "ion_current_a",
    "total_current_a",
    "current_sem_a",
    "sample_count",
    "converged",
)

QUALITY_VALUES = frozenset({"PREVIEW", "PASS", "FAIL"})


def make_run_id(config_sha256: str, root_seed: int, replicate: int) -> str:
    """Calculate a stable identifier for one simulation run."""
    if not isinstance(config_sha256, str) or len(config_sha256) != 64:
        raise ValueError("Set config_sha256 to a 64-character value.")
    if isinstance(root_seed, bool) or not isinstance(root_seed, Integral):
        raise TypeError("Set root_seed to an integer.")
    if isinstance(replicate, bool) or not isinstance(replicate, Integral):
        raise TypeError("Set replicate to an integer.")
    if int(root_seed) < 0 or int(replicate) < 0:
        raise ValueError("Set root_seed and replicate to positive values or zero.")
    source = f"{config_sha256}:{int(root_seed)}:{int(replicate)}"
    return sha256(source.encode("ascii")).hexdigest()[:32]


def _frame(records: pd.DataFrame | Iterable[Mapping[str, Any]]) -> pd.DataFrame:
    if isinstance(records, pd.DataFrame):
        return records.copy()
    return pd.DataFrame.from_records(records)


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...], name: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"The {name} data does not have these columns: {missing}.")
    if frame.loc[:, list(columns)].isnull().any().any():
        raise ValueError(f"The {name} data has an empty required value.")


def _require_finite(frame: pd.DataFrame, columns: tuple[str, ...], name: str) -> None:
    for column in columns:
        values = frame[column].to_numpy(dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"The {name} column {column} has a nonfinite value.")


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in hexdigits for character in value)
    )


def _require_integer_column(
    frame: pd.DataFrame,
    column: str,
    name: str,
    *,
    minimum: int,
) -> None:
    for value in frame[column]:
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, Integral)
            or int(value) < minimum
        ):
            raise ValueError(
                f"The {name} column {column} has an invalid integer."
            )


def _require_boolean_column(
    frame: pd.DataFrame,
    column: str,
    name: str,
) -> None:
    if not all(isinstance(value, (bool, np.bool_)) for value in frame[column]):
        raise ValueError(f"The {name} column {column} has a non-Boolean value.")


def validate_run_records(frame: pd.DataFrame) -> None:
    """Make sure that the run records agree with the data contract."""
    _require_columns(frame, RUN_REQUIRED_COLUMNS, "run")
    if frame["run_id"].duplicated().any():
        raise ValueError("The run data has a duplicate run_id.")
    if not all(
        isinstance(value, str)
        and len(value) == 32
        and all(character in hexdigits for character in value)
        for value in frame["run_id"]
    ):
        raise ValueError("The run data has an invalid run_id.")
    if not all(_is_sha256(value) for value in frame["config_sha256"]):
        raise ValueError("The run data has an invalid config_sha256.")
    if not set(frame["quality_status"]).issubset(QUALITY_VALUES):
        raise ValueError("The run data has an unknown quality_status.")
    _require_finite(
        frame,
        (
            "root_seed",
            "replicate",
            "ne_m3",
            "te_ev",
            "ti_ev",
            "vp_v",
            "pressure_pa",
            "probe_radius_m",
            "probe_length_m",
            "n_cells",
            "dt_s",
            "n_particles",
            "data_schema_version",
        ),
        "run",
    )
    if (frame["probe_length_m"] <= 0.0).any():
        raise ValueError("The run data has a probe length that is not positive.")
    for column in ("ne_m3", "te_ev", "ti_ev", "probe_radius_m", "dt_s"):
        if (frame[column] <= 0.0).any():
            raise ValueError(f"The run data column {column} must be positive.")
    if (frame["pressure_pa"] < 0.0).any():
        raise ValueError("The run data pressure must not be negative.")
    for column, minimum in (
        ("root_seed", 0),
        ("replicate", 0),
        ("n_cells", 2),
        ("n_particles", 1),
        ("data_schema_version", DATASET_SCHEMA_VERSION),
    ):
        _require_integer_column(
            frame,
            column,
            "run",
            minimum=minimum,
        )
    if (frame["root_seed"].map(int) > (1 << 64) - 1).any():
        raise ValueError("The run data root seed is outside the uint64 range.")
    for column in ("stability_pass", "convergence_pass"):
        _require_boolean_column(frame, column, "run")
    pass_rows = frame["quality_status"] == "PASS"
    if (
        pass_rows.any()
        and not (
            frame.loc[pass_rows, "stability_pass"]
            & frame.loc[pass_rows, "convergence_pass"]
        ).all()
    ):
        raise ValueError("A PASS run does not satisfy all required gates.")
    if (frame["data_schema_version"] != DATASET_SCHEMA_VERSION).any():
        raise ValueError("The run data has an unsupported schema version.")
    for config_json, config_sha256 in zip(
        frame["config_json"],
        frame["config_sha256"],
        strict=True,
    ):
        try:
            normalized = canonical_json(json.loads(config_json))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError("The run data has invalid config_json.") from exc
        if sha256(normalized.encode("utf-8")).hexdigest() != config_sha256:
            raise ValueError("The run configuration hash is not correct.")
    for value in frame["cross_section_hashes_json"]:
        try:
            hashes = json.loads(value)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(
                "The run data has invalid cross-section hashes."
            ) from exc
        if not isinstance(hashes, dict) or not all(
            isinstance(label, str) and _is_sha256(digest)
            for label, digest in hashes.items()
        ):
            raise ValueError("The run data has invalid cross-section hashes.")


def validate_curve_records(
    frame: pd.DataFrame,
    *,
    run_ids: set[str] | None = None,
) -> None:
    """Make sure that the curve records agree with the data contract."""
    _require_columns(frame, CURVE_REQUIRED_COLUMNS, "curve")
    if run_ids is not None and not set(frame["run_id"]).issubset(run_ids):
        raise ValueError("The curve data refers to an unknown run_id.")
    if frame.duplicated(["run_id", "bias_v", "replicate"]).any():
        raise ValueError("The curve data has a duplicate voltage record.")
    _require_finite(
        frame,
        (
            "bias_v",
            "replicate",
            "electron_current_a",
            "ion_current_a",
            "total_current_a",
            "current_sem_a",
            "sample_count",
        ),
        "curve",
    )
    expected_total = frame["electron_current_a"] - frame["ion_current_a"]
    if not np.allclose(
        frame["total_current_a"],
        expected_total,
        rtol=1.0e-12,
        atol=1.0e-15,
    ):
        raise ValueError("The total current does not equal electron current minus ion current.")
    if (frame["sample_count"] < 1).any():
        raise ValueError("The curve data has a sample count less than one.")
    if (frame["current_sem_a"] < 0.0).any():
        raise ValueError("The curve data has a negative standard error.")
    if "conventional_current_a" in frame.columns:
        _require_finite(
            frame,
            ("conventional_current_a",),
            "curve",
        )
        if not np.allclose(
            frame["conventional_current_a"],
            -frame["total_current_a"],
            rtol=1.0e-12,
            atol=1.0e-15,
        ):
            raise ValueError(
                "The conventional current does not equal ion current minus electron current."
            )
    _require_integer_column(frame, "replicate", "curve", minimum=0)
    _require_integer_column(frame, "sample_count", "curve", minimum=1)
    _require_boolean_column(frame, "converged", "curve")


def _temporary_path(target: Path) -> Path:
    return target.with_name(f".{target.name}.{uuid4().hex}.tmp")


def write_dataset(
    output_dir: str | Path,
    runs: pd.DataFrame | Iterable[Mapping[str, Any]],
    curves: pd.DataFrame | Iterable[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    *,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Write one traceable dataset to an explicit directory."""
    directory = Path(output_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    targets = {
        "runs": directory / "runs.parquet",
        "curves": directory / "curves.parquet",
        "manifest": directory / "manifest.json",
    }
    if not overwrite:
        existing = [str(path) for path in targets.values() if path.exists()]
        if existing:
            raise FileExistsError(f"The dataset files already exist: {existing}.")

    run_frame = _frame(runs)
    curve_frame = _frame(curves)
    validate_run_records(run_frame)
    validate_curve_records(curve_frame, run_ids=set(run_frame["run_id"]))
    if set(run_frame["run_id"]) != set(curve_frame["run_id"]):
        raise ValueError("Each run must have one or more curve records.")
    replicate_by_run = dict(
        zip(run_frame["run_id"], run_frame["replicate"], strict=True)
    )
    if any(
        int(replicate) != int(replicate_by_run[run_id])
        for run_id, replicate in zip(
            curve_frame["run_id"],
            curve_frame["replicate"],
            strict=True,
        )
    ):
        raise ValueError("A curve replicate does not match its run record.")

    run_temp = _temporary_path(targets["runs"])
    curve_temp = _temporary_path(targets["curves"])
    manifest_temp = _temporary_path(targets["manifest"])
    temporary_paths = (run_temp, curve_temp, manifest_temp)
    try:
        run_frame.to_parquet(run_temp, index=False)
        curve_frame.to_parquet(curve_temp, index=False)

        manifest_data = dict(manifest)
        manifest_data["dataset_schema_version"] = DATASET_SCHEMA_VERSION
        manifest_data["files"] = {
            "curves.parquet": sha256_file(curve_temp),
            "runs.parquet": sha256_file(run_temp),
        }
        manifest_data["dataset_sha256"] = sha256(
            canonical_json(manifest_data).encode("utf-8")
        ).hexdigest()
        manifest_temp.write_text(
            json.dumps(
                manifest_data,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

        run_temp.replace(targets["runs"])
        curve_temp.replace(targets["curves"])
        manifest_temp.replace(targets["manifest"])
    finally:
        for temporary_path in temporary_paths:
            temporary_path.unlink(missing_ok=True)
    return targets


def read_dataset(output_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Read and check one dataset."""
    directory = Path(output_dir).resolve()
    runs = pd.read_parquet(directory / "runs.parquet")
    curves = pd.read_parquet(directory / "curves.parquet")
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    validate_run_records(runs)
    validate_curve_records(curves, run_ids=set(runs["run_id"]))
    if set(runs["run_id"]) != set(curves["run_id"]):
        raise ValueError("Each run must have one or more curve records.")
    recorded_dataset_hash = manifest.get("dataset_sha256")
    hash_source = dict(manifest)
    hash_source.pop("dataset_sha256", None)
    expected_dataset_hash = sha256(
        canonical_json(hash_source).encode("utf-8")
    ).hexdigest()
    if recorded_dataset_hash != expected_dataset_hash:
        raise ValueError("The dataset checksum is not correct.")
    for name in ("runs.parquet", "curves.parquet"):
        if manifest["files"][name] != sha256_file(directory / name):
            raise ValueError(f"The checksum is not correct for {name}.")
    return runs, curves, manifest
