from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
from importlib import metadata
import json
from numbers import Integral
from pathlib import Path
import platform
from typing import Any, Iterable

from core.config import Config
from core.rng import RNG_ALGORITHM_VERSION


MANIFEST_SCHEMA_VERSION = 2
DATA_SCHEMA_VERSION = "1"
PHYSICS_MODEL_VERSION = "3"
DEFAULT_DISTRIBUTION_NAME = "pic-probe-simulation"
UNINSTALLED_VERSION = "0+uninstalled"
_U64_MAX = (1 << 64) - 1


def sha256_bytes(data: bytes) -> str:
    """Calculate the SHA-256 value of byte data."""
    return sha256(data).hexdigest()


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Calculate the SHA-256 value of a file."""
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, Integral):
        raise TypeError("Set chunk_size to an integer.")
    if chunk_size < 1:
        raise ValueError("Set chunk_size to an integer greater than zero.")

    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"The file was not found: {file_path}.")
    if not file_path.is_file():
        raise IsADirectoryError(f"The path does not identify a file: {file_path}.")

    digest = sha256()
    with file_path.open("rb") as source:
        while True:
            block = source.read(int(chunk_size))
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


file_sha256 = sha256_file


def hash_files(
    paths: Iterable[str | Path],
    *,
    root: str | Path | None = None,
) -> dict[str, str]:
    """Calculate SHA-256 values for a set of files."""
    root_path = Path(root).resolve() if root is not None else None
    records: dict[str, str] = {}
    for path in paths:
        raw_path = Path(path)
        if root_path is not None and not raw_path.is_absolute():
            file_path = (root_path / raw_path).resolve()
        else:
            file_path = raw_path.resolve()
        if root_path is None:
            label = file_path.as_posix()
        else:
            try:
                label = file_path.relative_to(root_path).as_posix()
            except ValueError:
                label = file_path.as_posix()
        if label in records:
            raise ValueError(f"The file label occurs more than one time: {label}.")
        records[label] = sha256_file(file_path)
    return dict(sorted(records.items()))


def get_distribution_version(
    distribution_name: str = DEFAULT_DISTRIBUTION_NAME,
) -> str:
    """Give the installed software version."""
    if not isinstance(distribution_name, str) or not distribution_name.strip():
        raise ValueError("Set distribution_name to a nonempty value.")
    try:
        return metadata.version(distribution_name)
    except metadata.PackageNotFoundError:
        return UNINSTALLED_VERSION


def version_record(
    *,
    software_version: str | None = None,
    distribution_name: str = DEFAULT_DISTRIBUTION_NAME,
    physics_model_version: str = PHYSICS_MODEL_VERSION,
    data_schema_version: str = DATA_SCHEMA_VERSION,
    rng_algorithm_version: str = RNG_ALGORITHM_VERSION,
    source_commit: str | None = None,
) -> dict[str, Any]:
    """Give software and runtime version data."""
    version = software_version or get_distribution_version(distribution_name)
    return {
        "data_schema_version": str(data_schema_version),
        "dependencies": {
            name: get_distribution_version(name)
            for name in ("numba", "numpy")
        },
        "distribution_name": distribution_name,
        "physics_model_version": str(physics_model_version),
        "platform": platform.platform(),
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "rng_algorithm_version": str(rng_algorithm_version),
        "software_version": version,
        "source_commit": source_commit,
    }


def canonical_json(data: Any) -> str:
    """Give normalized JSON data."""
    return json.dumps(
        data,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def json_sha256(data: Any) -> str:
    """Calculate the SHA-256 value of JSON data."""
    return sha256_bytes(canonical_json(data).encode("utf-8"))


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _root_seed(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError("Set root_seed to an integer.")
    number = int(value)
    if number < 0 or number > _U64_MAX:
        raise ValueError("Set root_seed to an integer from 0 through 2^64 - 1.")
    return number


def build_run_manifest(
    config: Config,
    *,
    root_seed: int,
    source_files: Iterable[str | Path] = (),
    source_root: str | Path | None = None,
    software_version: str | None = None,
    distribution_name: str = DEFAULT_DISTRIBUTION_NAME,
    physics_model_version: str = PHYSICS_MODEL_VERSION,
    data_schema_version: str = DATA_SCHEMA_VERSION,
    rng_algorithm_version: str = RNG_ALGORITHM_VERSION,
    source_commit: str | None = None,
    created_utc: str | None = None,
) -> dict[str, Any]:
    """Give traceability data for one run."""
    if not isinstance(config, Config):
        raise TypeError("Set config to a Config object.")

    manifest: dict[str, Any] = {
        "config": config.canonical_dict(),
        "config_sha256": config.fingerprint(),
        "created_utc": created_utc or _utc_now(),
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "root_seed": _root_seed(root_seed),
        "source_files": hash_files(source_files, root=source_root),
        "stability": config.stability_metrics().as_dict(),
        "stability_warnings": config.stability_warnings(),
        "versions": version_record(
            software_version=software_version,
            distribution_name=distribution_name,
            physics_model_version=physics_model_version,
            data_schema_version=data_schema_version,
            rng_algorithm_version=rng_algorithm_version,
            source_commit=source_commit,
        ),
    }
    manifest["manifest_sha256"] = json_sha256(manifest)
    return manifest
