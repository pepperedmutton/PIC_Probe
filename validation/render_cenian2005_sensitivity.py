from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def render_sensitivity(result_dir: str | Path) -> Path:
    directory = Path(result_dir).expanduser().resolve()
    with (directory / "sensitivity.csv").open(
        "r", encoding="utf-8", newline=""
    ) as source:
        rows = list(csv.DictReader(source))
    if len(rows) != 3:
        raise ValueError("Expected three sensitivity rows.")

    voltage = np.asarray([float(row["voltage_V"]) for row in rows])
    experiment = 1.0e6 * np.asarray(
        [float(row["experimental_current_A"]) for row in rows]
    )
    baseline = 1.0e6 * np.asarray(
        [float(row["baseline_current_A"]) for row in rows]
    )
    domain = 1.0e6 * np.asarray(
        [float(row["domain20_current_A"]) for row in rows]
    )
    long_warmup = 1.0e6 * np.asarray(
        [float(row["long_warmup_current_A"]) for row in rows]
    )

    figure, axis = plt.subplots(figsize=(7.8, 5.3))
    axis.plot(voltage, experiment, "o-", color="#111827", label="Experiment")
    axis.plot(voltage, baseline, "s-", color="#2563eb", label="12 mm, 0.02 τi warmup")
    axis.plot(voltage, domain, "^-", color="#059669", label="20 mm, 0.02 τi warmup")
    axis.plot(voltage, long_warmup, "D-", color="#dc2626", label="12 mm, 0.10 τi warmup")
    axis.set_title("Cenian 2005 three-point sensitivity screen")
    axis.set_xlabel("Probe bias relative to plasma potential (V)")
    axis.set_ylabel("Signed probe current (µA)")
    axis.grid(True, alpha=0.25)
    axis.legend(fontsize=8)
    axis.text(
        0.01,
        0.02,
        "PREVIEW: one seed; not a convergence study",
        transform=axis.transAxes,
        fontsize=8,
        color="#7f1d1d",
    )
    figure.tight_layout()
    output = directory / "sensitivity_plot.png"
    figure.savefig(output, dpi=200, metadata={"Software": "PIC_Probe"})
    plt.close(figure)
    return output


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Render the three-point Cenian sensitivity screen."
    )
    parser.add_argument("result_dir", type=Path)
    arguments = parser.parse_args(argv)
    print(render_sensitivity(arguments.result_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
