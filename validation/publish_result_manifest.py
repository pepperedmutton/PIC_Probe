from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from core.provenance import json_sha256, sha256_file


def _redact_path_record(record: Any) -> None:
    if not isinstance(record, dict):
        return
    raw_path = record.pop("path", None)
    if raw_path is not None:
        record["name"] = Path(str(raw_path)).name


def publish_manifest(path: str | Path) -> Path:
    """Redact machine-local paths while preserving runtime hashes."""
    manifest_path = Path(path).expanduser().resolve()
    raw_manifest_sha256 = sha256_file(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    plan = manifest.get("plan")
    if not isinstance(plan, dict):
        raise ValueError("The result manifest has no validation plan.")

    config = plan.get("config")
    if isinstance(config, dict):
        for field in ("LXCAT_ELECTRON_FILE", "LXCAT_ION_FILE"):
            value = config.get(field)
            if value:
                config[field] = Path(str(value)).name
    input_files = plan.get("input_files")
    if isinstance(input_files, Mapping):
        for record in input_files.values():
            _redact_path_record(record)

    plan["config_paths_redacted"] = True
    plan.pop("plan_sha256", None)
    plan["plan_sha256"] = json_sha256(plan)
    manifest["publication"] = {
        "machine_local_paths_redacted": True,
        "raw_manifest_sha256_before_redaction": raw_manifest_sha256,
        "runtime_config_sha256_preserved": True,
        "statement": (
            "Only machine-local path strings were replaced with file names. "
            "Runtime configuration, state, input, and output hashes remain."
        ),
    }
    manifest.pop("manifest_sha256", None)
    manifest["manifest_sha256"] = json_sha256(manifest)
    text = json.dumps(
        manifest,
        allow_nan=False,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    manifest_path.write_text(text + "\n", encoding="utf-8")
    return manifest_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Redact machine-local paths from a result manifest."
    )
    parser.add_argument("manifests", type=Path, nargs="+")
    arguments = parser.parse_args(argv)
    for manifest in arguments.manifests:
        print(publish_manifest(manifest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
