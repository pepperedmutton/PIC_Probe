from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Sequence

from core.cross_sections import (
    build_electron_tables_from_lxcat,
    build_ion_tables_from_lxcat,
)
from core.lxcat_parser import parse_cross_section_file


HERE = Path(__file__).resolve().parent
DEFAULT_MANIFEST = HERE / "lxcat_manifest.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_files(
    data_dir: Path,
    manifest_path: Path = DEFAULT_MANIFEST,
    *,
    electron_e_max_ev: float = 100.0,
    ion_e_max_ev: float = 100.0,
    table_points: int = 2001,
) -> dict[str, object]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    datasets = {item["role"]: item for item in manifest["datasets"]}
    electron_path = (
        data_dir / datasets["electron_argon_primary"]["local_filename"]
    )
    sensitivity_path = (
        data_dir / datasets["electron_argon_sensitivity"]["local_filename"]
    )
    ion_path = data_dir / datasets["argon_ion_in_argon"]["local_filename"]

    hashes = {
        "electron_argon_primary": _sha256(electron_path),
        "electron_argon_sensitivity": _sha256(sensitivity_path),
        "argon_ion_in_argon": _sha256(ion_path),
    }
    for role, actual_hash in hashes.items():
        expected_hash = str(datasets[role]["sha256"]).casefold()
        if actual_hash.casefold() != expected_hash:
            raise ValueError(
                f"The {role} file hash does not match lxcat_manifest.json."
            )

    electron_processes = parse_cross_section_file(electron_path, strict=True)
    sensitivity_processes = parse_cross_section_file(
        sensitivity_path,
        strict=True,
    )
    ion_processes = parse_cross_section_file(ion_path, strict=True)
    electron_counts = Counter(
        process.process_type for process in electron_processes
    )
    sensitivity_counts = Counter(
        process.process_type for process in sensitivity_processes
    )
    ion_counts = Counter(process.process_type for process in ion_processes)

    if (
        dict(electron_counts)
        != datasets["electron_argon_primary"]["process_counts"]
    ):
        raise ValueError("The electron process counts do not match the manifest.")
    if (
        dict(sensitivity_counts)
        != datasets["electron_argon_sensitivity"]["process_counts"]
    ):
        raise ValueError(
            "The sensitivity process counts do not match the manifest."
        )
    if dict(ion_counts) != datasets["argon_ion_in_argon"]["process_counts"]:
        raise ValueError("The ion process counts do not match the manifest.")

    electron_tables = build_electron_tables_from_lxcat(
        electron_path,
        target="Ar",
        e_max=electron_e_max_ev,
        n_bins=table_points,
        strict=True,
    )
    sensitivity_tables = build_electron_tables_from_lxcat(
        sensitivity_path,
        target="Ar",
        e_max=electron_e_max_ev,
        n_bins=table_points,
        strict=True,
    )
    ion_tables = build_ion_tables_from_lxcat(
        ion_path,
        target="Ar",
        e_max=ion_e_max_ev,
        n_bins=table_points,
        strict=True,
        confirm_symmetric_backscatter_as_cex=True,
    )
    return {
        "status": "PASS",
        "manifest": str(manifest_path.resolve()),
        "hashes": hashes,
        "electron_process_counts": dict(sorted(electron_counts.items())),
        "electron_sensitivity_process_counts": dict(
            sorted(sensitivity_counts.items())
        ),
        "ion_process_counts": dict(sorted(ion_counts.items())),
        "electron_excitation_channels": int(
            electron_tables.excitation_channel_tables.shape[0]
        ),
        "electron_ionization_channels": int(
            electron_tables.ionization_channel_tables.shape[0]
        ),
        "electron_sensitivity_excitation_channels": int(
            sensitivity_tables.excitation_channel_tables.shape[0]
        ),
        "electron_e_max_ev": electron_e_max_ev,
        "ion_e_max_ev": ion_e_max_ev,
        "table_points": table_points,
        "ion_backscatter_mapping": "explicit symmetric Ar+/Ar CEX",
        "maximum_cross_sections_m2": {
            "electron_elastic": float(electron_tables.sigma_elastic.max()),
            "electron_excitation_total": float(
                electron_tables.sigma_excitation.max()
            ),
            "electron_ionization_total": float(
                electron_tables.sigma_ionization.max()
            ),
            "ion_charge_exchange": float(ion_tables.sigma_cex.max()),
            "ion_isotropic_elastic": float(ion_tables.sigma_elastic.max()),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate local LXCat inputs without redistributing them."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(".validation_private/lxcat"),
    )
    parser.add_argument("--electron-e-max", type=float, default=100.0)
    parser.add_argument("--ion-e-max", type=float, default=100.0)
    parser.add_argument("--table-points", type=int, default=2001)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = validate_files(
        args.data_dir,
        electron_e_max_ev=args.electron_e_max,
        ion_e_max_ev=args.ion_e_max,
        table_points=args.table_points,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
