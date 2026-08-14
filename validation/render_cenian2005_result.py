from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


EXPECTED_VOLTAGES = np.arange(-60.0, -9.0, 5.0)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as source:
        return list(csv.DictReader(source))


def _number(record: Mapping[str, Any], name: str) -> float:
    value = float(record[name])
    if not math.isfinite(value):
        raise ValueError(f"The {name} value is not finite.")
    return value


def verify_result_directory(result_dir: str | Path) -> dict[str, Any]:
    """Verify the stored pilot data and recalculate its main metrics."""
    directory = Path(result_dir).expanduser().resolve()
    comparison = _read_csv(directory / "comparison.csv")
    simulation = _read_csv(directory / "simulation_points.csv")
    metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))

    voltages = np.asarray(
        [_number(record, "voltage_V") for record in comparison],
        dtype=float,
    )
    selected_voltage_set = set(float(value) for value in voltages)
    expected_selected = np.asarray(
        [
            value
            for value in EXPECTED_VOLTAGES
            if float(value) in selected_voltage_set
        ],
        dtype=float,
    )
    if not np.array_equal(voltages, expected_selected):
        raise ValueError("The comparison voltage grid is not a source-order subset.")
    seed_count = int(metrics["seed_count"])
    expected_simulation_rows = len(comparison) * seed_count
    if len(comparison) < 1 or len(simulation) != expected_simulation_rows:
        raise ValueError("The comparison and simulation row counts are inconsistent.")

    for voltage in voltages:
        selected = [
            record
            for record in simulation
            if _number(record, "voltage_V") == voltage
        ]
        if (
            len(selected) != seed_count
            or len({record["seed"] for record in selected}) != seed_count
        ):
            raise ValueError(
                f"The {voltage:g} V point does not have {seed_count} seeds."
            )
        mean_current = float(
            np.mean(
                [
                    _number(record, "source_aligned_current_A")
                    for record in selected
                ]
            )
        )
        stored = next(
            record
            for record in comparison
            if _number(record, "voltage_V") == voltage
        )
        if not math.isclose(
            mean_current,
            _number(stored, "simulation_source_aligned_current_A"),
            rel_tol=1.0e-12,
            abs_tol=1.0e-18,
        ):
            raise ValueError(f"The {voltage:g} V stored mean is inconsistent.")

    trace = manifest.get("run_trace", [])
    if len(trace) != expected_simulation_rows:
        raise ValueError("The manifest run-trace count is inconsistent.")
    for item in trace:
        if item.get("simulation_status") != "READY":
            raise ValueError("A simulation trace is not READY.")
        if item.get("numerical_status") != "PASS":
            raise ValueError("A numerical trace is not PASS.")
        if item.get("energy_table_overflow_lookups") != 0:
            raise ValueError("A trace has an energy-table overflow.")
        if item.get("electron_particle_ledger_residual") != 0:
            raise ValueError("A trace has an electron ledger residual.")
        if item.get("ion_particle_ledger_residual") != 0:
            raise ValueError("A trace has an ion ledger residual.")
        if item.get("stability_warnings") or item.get("runtime_warnings"):
            raise ValueError("A trace has a numerical warning.")

    experiment = np.asarray(
        [
            _number(record, "experimental_source_aligned_current_A")
            for record in comparison
        ],
        dtype=float,
    )
    model = np.asarray(
        [
            _number(record, "simulation_source_aligned_current_A")
            for record in comparison
        ],
        dtype=float,
    )
    residual = model - experiment
    recalculated = {
        "mean_bias_A": float(np.mean(residual)),
        "mean_absolute_error_A": float(np.mean(np.abs(residual))),
        "root_mean_square_error_A": float(np.sqrt(np.mean(residual**2))),
        "normalized_root_mean_square_error": float(
            np.sqrt(np.mean(residual**2)) / np.sqrt(np.mean(experiment**2))
        ),
        "mean_absolute_relative_error": float(
            np.mean(np.abs(residual) / np.abs(experiment))
        ),
    }
    for name, value in recalculated.items():
        if not math.isclose(
            value,
            float(metrics[name]),
            rel_tol=1.0e-12,
            abs_tol=1.0e-18,
        ):
            raise ValueError(f"The stored {name} value is inconsistent.")
    return {
        "directory": directory,
        "comparison": comparison,
        "simulation": simulation,
        "metrics": metrics,
        "recalculated": recalculated,
    }


def render_comparison(result_dir: str | Path) -> Path:
    """Render the experiment, three seeds, mean, and residual."""
    verified = verify_result_directory(result_dir)
    directory = verified["directory"]
    comparison = verified["comparison"]
    simulation = verified["simulation"]
    voltage = np.asarray(
        [_number(record, "voltage_V") for record in comparison]
    )
    experiment = 1.0e6 * np.asarray(
        [
            _number(record, "experimental_source_aligned_current_A")
            for record in comparison
        ]
    )
    experiment_uncertainty = 1.0e6 * np.asarray(
        [_number(record, "digitization_uncertainty_A") for record in comparison]
    )
    model = 1.0e6 * np.asarray(
        [
            _number(record, "simulation_source_aligned_current_A")
            for record in comparison
        ]
    )
    model_sem = 1.0e6 * np.asarray(
        [_number(record, "simulation_standard_error_A") for record in comparison]
    )

    figure, (current_axis, residual_axis) = plt.subplots(
        2,
        1,
        figsize=(8.2, 7.6),
        sharex=True,
        gridspec_kw={"height_ratios": [2.15, 1.0]},
    )
    seeds = sorted({int(record["seed"]) for record in simulation})
    for index, seed in enumerate(seeds):
        selected = sorted(
            (
                record
                for record in simulation
                if int(record["seed"]) == seed
            ),
            key=lambda record: _number(record, "voltage_V"),
        )
        current_axis.scatter(
            [_number(record, "voltage_V") for record in selected],
            [
                1.0e6 * _number(record, "source_aligned_current_A")
                for record in selected
            ],
            s=22,
            alpha=0.45,
            color="#3b82f6",
            marker=("x", "+", "1")[index],
            label=f"Simulation seed {seed}",
            zorder=2,
        )
    current_axis.errorbar(
        voltage,
        experiment,
        yerr=experiment_uncertainty,
        color="#111827",
        marker="o",
        linewidth=1.8,
        capsize=3,
        label="Cenian 2005 digitized experiment",
        zorder=4,
    )
    current_axis.errorbar(
        voltage,
        model,
        yerr=model_sem,
        color="#1d4ed8",
        marker="s",
        linewidth=2.0,
        capsize=3,
        label="PIC-MCC mean (3 seeds)",
        zorder=3,
    )
    current_axis.set_ylabel("Signed probe current (µA)")
    current_axis.set_title(
        "Cenian 2005 Figure 2 vs PIC-MCC pilot\n"
        "Phelps e-Ar and Ar+-Ar; no fitted scale factor"
    )
    current_axis.grid(True, alpha=0.25)
    current_axis.legend(loc="lower right", fontsize=8)

    residual = model - experiment
    residual_axis.axhline(0.0, color="#111827", linewidth=1.0)
    residual_axis.errorbar(
        voltage,
        residual,
        yerr=np.hypot(experiment_uncertainty, model_sem),
        color="#b91c1c",
        marker="o",
        linewidth=1.8,
        capsize=3,
    )
    residual_axis.set_xlabel("Probe bias relative to plasma potential (V)")
    residual_axis.set_ylabel("Sim. − exp. (µA)")
    residual_axis.grid(True, alpha=0.25)
    residual_axis.text(
        0.01,
        0.06,
        "PREVIEW: short transient, 12 mm domain, 16 particles/cell",
        transform=residual_axis.transAxes,
        fontsize=8,
        color="#7f1d1d",
    )
    figure.tight_layout()
    output = directory / "comparison_plot.png"
    figure.savefig(output, dpi=200, metadata={"Software": "PIC_Probe"})
    plt.close(figure)
    return output


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify and plot a Cenian 2005 pilot result."
    )
    parser.add_argument("result_dir", type=Path)
    arguments = parser.parse_args(argv)
    output = render_comparison(arguments.result_dir)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
