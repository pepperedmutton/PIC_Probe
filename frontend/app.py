from __future__ import annotations

from collections.abc import Mapping
import csv
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
from hashlib import sha256
import io
import json
import math
from numbers import Integral, Real
from pathlib import Path
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config import Config
from core.provenance import canonical_json, json_sha256
from core.simulation import PICSimulation


PA_PER_TORR = 133.322
PREVIEW_MODE = "preview"
VALIDATED_MODE = "validated"
PRODUCTION_LOCK_MESSAGE = (
    "You cannot select Validated Run. "
    "External physics validation is not available."
)
MODE_BY_LABEL = {
    "Quick Preview": PREVIEW_MODE,
    "Validated Run": VALIDATED_MODE,
}
IV_COLUMNS = (
    ("voltages", "voltage_V"),
    ("I_total", "I_total_A"),
    ("I_conventional", "I_conventional_A"),
    ("I_electron", "I_electron_A"),
    ("I_ion", "I_ion_A"),
)

PRESETS: dict[str, dict[str, Any]] = {
    "Argon": {
        "gas_name": "Argon (Ar+)",
        "runnable": True,
        "pressure_value": 50.0,
        "pressure_unit": "Torr",
        "probe_length_m": 0.01,
        "root_seed": 20260309,
        "v_start": -50.0,
        "v_end": 50.0,
        "config": {
            "N0": 1.0e16,
            "Te": 3.0,
            "Ti": 0.03,
            "P_Torr": 50.0,
            "m_i": 6.6335209e-26,
            "CROSS_SECTION_TARGET": "Ar",
            "R_MIN": 5.0e-4,
            "R_MAX": 5.0e-2,
            "N_CELLS": 400,
            "DT": 2.0e-11,
            "LXCAT_ELECTRON_FILE": None,
            "LXCAT_ION_FILE": None,
        },
    },
    "Hydrogen": {
        "gas_name": "Hydrogen (H+)",
        "runnable": False,
        "pressure_value": 0.3,
        "pressure_unit": "Pa",
        "probe_length_m": 1.0,
        "root_seed": 20260309,
        "v_start": -30.0,
        "v_end": 100.0,
        "config": {
            "N0": 1.0e16,
            "Te": 1.0,
            "Ti": 1.0,
            "P_Torr": 0.3 / PA_PER_TORR,
            "m_i": 1.67262192369e-27,
            "m_neutral": 3.34524384738e-27,
            "CROSS_SECTION_TARGET": "H2",
            "NEUTRAL_SPECIES": "H2",
            "ION_SPECIES": "H+",
            "R_MIN": 0.2e-3,
            "R_MAX": 2.0e-3,
            "N_CELLS": 400,
            "DT": 5.0e-13,
            "LXCAT_ELECTRON_FILE": None,
            "LXCAT_ION_FILE": None,
            "SIGMA_EN_ELASTIC": 6.0e-20,
            "SIGMA_EN_EXC": 1.5e-20,
            "SIGMA_EN_ION": 2.0e-21,
            "SIGMA_IN_ELASTIC": 2.0e-19,
            "E_EXC_EV": 10.2,
            "E_ION_EV": 15.4,
            "ENABLE_IONIZATION_SECONDARIES": True,
            "SMOOTH_DENSITY": True,
            "N_SMOOTHING_PASSES": 3,
        },
    },
}

RUN_DEFAULTS = {
    PREVIEW_MODE: {
        "voltage_steps": 9,
        "n_particles": 1000,
        "n_burn_in": 200,
        "n_sampling": 200,
    },
    VALIDATED_MODE: {
        "voltage_steps": 21,
        "n_particles": 12000,
        "n_burn_in": 12000,
        "n_sampling": 12000,
    },
}


def preset_values(name: str) -> dict[str, Any]:
    """Give an independent copy of one preset."""
    try:
        return deepcopy(PRESETS[name])
    except KeyError as exc:
        raise ValueError(f"Select a known gas preset. The preset is {name!r}.") from exc


def _real_value(name: str, value: Real) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"Set {name} to a real number.")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"Set {name} to a finite value.")
    return number


def _integer_value(
    name: str,
    value: Integral,
    *,
    minimum: int,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"Set {name} to an integer.")
    number = int(value)
    if number < minimum:
        raise ValueError(f"Set {name} to an integer not less than {minimum}.")
    if maximum is not None and number > maximum:
        raise ValueError(f"Set {name} to an integer not more than {maximum}.")
    return number


def pressure_to_torr(value: Real, unit: str) -> float:
    """Convert the pressure to Torr for Config."""
    pressure = _real_value("pressure", value)
    if pressure < 0.0:
        raise ValueError("Set pressure to zero or a positive value.")
    if unit == "Torr":
        return pressure
    if unit == "Pa":
        return pressure / PA_PER_TORR
    raise ValueError("Set the pressure unit to Pa or Torr.")


def validate_probe_length(value: Real) -> float:
    """Make sure that the probe length is positive."""
    length = _real_value("probe_length", value)
    if length <= 0.0:
        raise ValueError("Set probe_length to a value greater than zero.")
    return length


def validate_root_seed(value: Integral) -> int:
    """Make sure that the root seed is in the permitted range."""
    return _integer_value(
        "root_seed",
        value,
        minimum=0,
        maximum=(1 << 64) - 1,
    )


def make_config_values(
    preset_name: str,
    *,
    pressure_value: Real,
    pressure_unit: str,
    density: Real,
    electron_temperature_ev: Real,
    ion_temperature_ev: Real,
    probe_radius_m: Real,
    radial_limit_m: Real,
    radial_cells: Integral,
    time_step_s: Real,
    wall_potential_v: Real,
) -> dict[str, Any]:
    """Give Config values from the selected inputs."""
    preset = preset_values(preset_name)
    values = dict(preset["config"])
    values.update(
        {
            "P_Torr": pressure_to_torr(pressure_value, pressure_unit),
            "N0": _real_value("density", density),
            "Te": _real_value("electron_temperature_ev", electron_temperature_ev),
            "Ti": _real_value("ion_temperature_ev", ion_temperature_ev),
            "R_MIN": _real_value("probe_radius_m", probe_radius_m),
            "R_MAX": _real_value("radial_limit_m", radial_limit_m),
            "N_CELLS": _integer_value("radial_cells", radial_cells, minimum=2),
            "DT": _real_value("time_step_s", time_step_s),
            "V_WALL": _real_value("wall_potential_v", wall_potential_v),
        }
    )
    values.pop("RUN_MODE", None)
    return values


def make_research_config(values: Mapping[str, Any]) -> Config:
    """Make a research Config for input checks."""
    config_values = dict(values)
    config_values.pop("RUN_MODE", None)
    return Config.research(**config_values)


def prepare_run_config(config: Config, execution_mode: str) -> Config:
    """Give the Config for the selected execution mode."""
    if not isinstance(config, Config):
        raise TypeError("Set config to a Config object.")
    if execution_mode == PREVIEW_MODE:
        return replace(config, RUN_MODE="research")
    if execution_mode == VALIDATED_MODE:
        raise ValueError(PRODUCTION_LOCK_MESSAGE)
    raise ValueError("Select Quick Preview.")


def make_input_snapshot(
    config: Config,
    *,
    preset_name: str,
    gas_name: str,
    execution_mode: str,
    pressure_value: Real,
    pressure_unit: str,
    probe_length_m: Real,
    root_seed: Integral,
    v_start: Real,
    v_end: Real,
    voltage_steps: Integral,
    n_particles: Integral,
    n_burn_in: Integral,
    n_sampling: Integral,
) -> dict[str, Any]:
    """Give normalized data for all simulation inputs."""
    if not isinstance(config, Config):
        raise TypeError("Set config to a Config object.")
    if execution_mode == VALIDATED_MODE:
        raise ValueError(PRODUCTION_LOCK_MESSAGE)
    if execution_mode != PREVIEW_MODE:
        raise ValueError("Select Quick Preview.")
    if not isinstance(preset_name, str) or not preset_name:
        raise ValueError("Select a gas preset.")
    if not isinstance(gas_name, str) or not gas_name:
        raise ValueError("Set a gas name.")

    pressure_torr = pressure_to_torr(pressure_value, pressure_unit)
    if not math.isclose(pressure_torr, config.P_Torr, rel_tol=1.0e-12, abs_tol=0.0):
        raise ValueError("The pressure input does not match the Config pressure.")

    return {
        "configuration": {
            "data": config.canonical_dict(),
            "sha256": config.fingerprint(),
        },
        "execution": {
            "mode": execution_mode,
            "n_burn_in": _integer_value("n_burn_in", n_burn_in, minimum=0),
            "n_particles": _integer_value("n_particles", n_particles, minimum=1),
            "n_sampling": _integer_value("n_sampling", n_sampling, minimum=1),
            "root_seed": validate_root_seed(root_seed),
        },
        "gas": {
            "name": gas_name,
            "preset": preset_name,
            "pressure_config_torr": pressure_torr,
            "pressure_input_unit": pressure_unit,
            "pressure_input_value": _real_value("pressure", pressure_value),
        },
        "probe": {
            "length_m": validate_probe_length(probe_length_m),
            "radius_m": float(config.R_MIN),
        },
        "sweep": {
            "end_voltage_V": _real_value("v_end", v_end),
            "start_voltage_V": _real_value("v_start", v_start),
            "voltage_steps": _integer_value(
                "voltage_steps",
                voltage_steps,
                minimum=1,
            ),
        },
    }


def input_fingerprint(snapshot: Mapping[str, Any]) -> str:
    """Calculate the SHA-256 value of an input snapshot."""
    if not isinstance(snapshot, Mapping):
        raise TypeError("Set snapshot to a mapping.")
    return json_sha256(dict(snapshot))


def select_result_for_input(
    result: Mapping[str, Any] | None,
    current_input_fingerprint: str | None,
) -> tuple[Mapping[str, Any] | None, bool]:
    """Select a result only when its input fingerprint is current."""
    if result is None:
        return None, False
    if not isinstance(result, Mapping):
        return None, True
    if not current_input_fingerprint:
        return None, True
    if result.get("input_fingerprint") != current_input_fingerprint:
        return None, True
    return result, False


def normalize_iv_data(iv_data: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Give validated copies of the I-V data arrays."""
    if not isinstance(iv_data, Mapping):
        raise TypeError("Set iv_data to a mapping.")

    arrays: dict[str, np.ndarray] = {}
    for key, _ in IV_COLUMNS:
        if key not in iv_data:
            raise ValueError(f"The I-V data does not contain {key}.")
        try:
            array = np.asarray(iv_data[key], dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"The I-V data for {key} is not numeric.") from exc
        if array.ndim != 1:
            raise ValueError(f"Set the I-V data for {key} to one dimension.")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"Set finite I-V data for {key}.")
        arrays[key] = array.copy()

    sizes = {array.size for array in arrays.values()}
    if len(sizes) != 1 or not sizes or next(iter(sizes)) < 1:
        raise ValueError("Set all I-V arrays to the same nonzero size.")
    if not np.allclose(
        arrays["I_total"],
        arrays["I_electron"] - arrays["I_ion"],
        rtol=1.0e-12,
        atol=1.0e-15,
    ):
        raise ValueError("I_total must equal I_electron minus I_ion.")
    if not np.allclose(
        arrays["I_conventional"],
        -arrays["I_total"],
        rtol=1.0e-12,
        atol=1.0e-15,
    ):
        raise ValueError("I_conventional must equal I_ion minus I_electron.")
    return arrays


def iv_data_to_csv(iv_data: Mapping[str, Any]) -> str:
    """Give the I-V data as CSV text."""
    arrays = normalize_iv_data(iv_data)
    target = io.StringIO(newline="")
    writer = csv.writer(target, lineterminator="\n")
    writer.writerow([column for _, column in IV_COLUMNS])
    for row in zip(*(arrays[key] for key, _ in IV_COLUMNS), strict=True):
        writer.writerow([format(float(value), ".17g") for value in row])
    return target.getvalue()


def manifest_to_json(manifest: Mapping[str, Any]) -> str:
    """Give the manifest as formatted JSON text."""
    return (
        json.dumps(
            dict(manifest),
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def make_result_manifest(
    config: Config,
    *,
    simulation_manifest: Mapping[str, Any],
    input_snapshot: Mapping[str, Any],
    input_sha256: str,
    root_seed: Integral,
    iv_data: Mapping[str, Any],
) -> dict[str, Any]:
    """Give the result manifest."""
    if not isinstance(config, Config):
        raise TypeError("Set config to a Config object.")
    if not isinstance(simulation_manifest, Mapping):
        raise TypeError("Set simulation_manifest to a mapping.")
    if input_fingerprint(input_snapshot) != input_sha256:
        raise ValueError("The input fingerprint does not match the input snapshot.")

    arrays = normalize_iv_data(iv_data)
    csv_text = iv_data_to_csv(arrays)
    manifest = json.loads(canonical_json(dict(simulation_manifest)))
    simulation_sha256 = manifest.pop("manifest_sha256", None)
    if simulation_sha256 != json_sha256(manifest):
        raise ValueError("The simulation manifest hash is not correct.")
    if manifest.get("config_sha256") != config.fingerprint():
        raise ValueError("The simulation manifest does not match the Config.")
    if manifest.get("root_seed") != validate_root_seed(root_seed):
        raise ValueError("The simulation manifest does not match the root seed.")
    if not isinstance(manifest.get("source_files"), dict):
        raise ValueError("The simulation manifest does not contain source file hashes.")

    manifest["frontend_input"] = {
        "fingerprint_sha256": input_sha256,
        "snapshot": json.loads(canonical_json(dict(input_snapshot))),
    }
    manifest["result_data"] = {
        "columns": [column for _, column in IV_COLUMNS],
        "csv_sha256": sha256(csv_text.encode("utf-8")).hexdigest(),
        "point_count": int(arrays["voltages"].size),
    }
    manifest["manifest_sha256"] = json_sha256(manifest)
    return manifest


def make_result_record(
    iv_data: Mapping[str, Any],
    config: Config,
    *,
    simulation_manifest: Mapping[str, Any],
    input_snapshot: Mapping[str, Any],
    input_sha256: str,
    root_seed: Integral,
) -> dict[str, Any]:
    """Give a result record with its configuration data."""
    arrays = normalize_iv_data(iv_data)
    manifest = make_result_manifest(
        config,
        simulation_manifest=simulation_manifest,
        input_snapshot=input_snapshot,
        input_sha256=input_sha256,
        root_seed=root_seed,
        iv_data=arrays,
    )
    return {
        "config_fingerprint": config.fingerprint(),
        "config_snapshot": config.canonical_dict(),
        "csv_text": iv_data_to_csv(arrays),
        "input_fingerprint": input_sha256,
        "input_snapshot": json.loads(canonical_json(dict(input_snapshot))),
        "iv_data": arrays,
        "manifest": manifest,
        "manifest_json": manifest_to_json(manifest),
    }


def utc_timestamp() -> str:
    """Give the current UTC time."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def apply_style() -> None:
    """Set the page style."""
    st.set_page_config(page_title="PICSIMU", layout="wide")
    st.markdown(
        """
        <style>
        :root {
            --bg-1: #f6efe4;
            --bg-2: #dbe7f2;
            --ink: #1d2327;
            --muted: #5c6b73;
            --accent: #b24a2f;
            --accent-2: #2b6f73;
            --card: rgba(255, 255, 255, 0.78);
            --border: rgba(29, 35, 39, 0.12);
            --shadow: 0 20px 60px rgba(29, 35, 39, 0.12);
        }
        .stApp {
            background: radial-gradient(circle at 12% 8%, rgba(178, 74, 47, 0.12), transparent 40%),
                        radial-gradient(circle at 86% 6%, rgba(43, 111, 115, 0.10), transparent 38%),
                        linear-gradient(135deg, var(--bg-1), var(--bg-2));
            color: var(--ink);
        }
        .hero {
            padding: 1.2rem 1.4rem;
            border: 1px solid var(--border);
            background: rgba(255, 255, 255, 0.7);
            border-radius: 18px;
            box-shadow: var(--shadow);
        }
        .kicker {
            text-transform: uppercase;
            font-size: 0.78rem;
            letter-spacing: 0.2rem;
            color: var(--muted);
            margin-bottom: 0.4rem;
        }
        .card {
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 1rem 1.2rem;
            box-shadow: var(--shadow);
        }
        .pill {
            display: inline-block;
            padding: 0.2rem 0.6rem;
            border-radius: 999px;
            background: rgba(178, 74, 47, 0.14);
            color: var(--accent);
            font-weight: 600;
            font-size: 0.8rem;
        }
        .stButton > button {
            background: linear-gradient(135deg, var(--accent), #d27d4d);
            color: white;
            border: none;
            border-radius: 999px;
            font-weight: 600;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def plot_iv_curve(
    v: np.ndarray,
    i_total: np.ndarray,
    i_e: np.ndarray,
    i_i: np.ndarray,
) -> plt.Figure:
    """Make the I-V figure."""
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.plot(
        v,
        i_total,
        color="#1d2327",
        linewidth=2.2,
        label="I_e - I_i (legacy)",
    )
    ax.plot(v, i_e, color="#b24a2f", linewidth=2.0, label="I_e")
    ax.plot(v, i_i, color="#2b6f73", linewidth=2.0, label="I_i")
    ax.set_xlabel("Probe bias (V)")
    ax.set_ylabel("Electron-minus-ion current (A)")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def plot_electron_semilog(v: np.ndarray, i_e: np.ndarray) -> plt.Figure:
    """Make the logarithmic electron-current figure."""
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.semilogy(v, np.maximum(i_e, 1.0e-30), color="#b24a2f", linewidth=2.0)
    ax.set_xlabel("Probe bias (V)")
    ax.set_ylabel("|I_e| (A)")
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    return fig


def main() -> None:
    """Show the Streamlit application."""
    apply_style()

    st.markdown(
        """
        <div class="hero">
            <div class="kicker">One-dimensional cylindrical PIC-MCC</div>
            <h1>PICSIMU Probe Simulation</h1>
            <p>
                Use Quick Preview for research calculations.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.warning(
        f"{PRODUCTION_LOCK_MESSAGE} "
        "Use Quick Preview for research only."
    )

    st.write("")
    left, right = st.columns([1.05, 1.4], gap="large")

    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Inputs")

        preset_name = st.selectbox("Gas preset", tuple(PRESETS))
        preset = preset_values(preset_name)
        preset_key = preset_name.lower()
        gas_name = st.text_input(
            "Gas",
            value=str(preset["gas_name"]),
            disabled=True,
            key=f"{preset_key}_gas",
        )
        mode_label = st.radio(
            "Execution mode",
            tuple(MODE_BY_LABEL),
            horizontal=True,
            disabled=True,
        )
        execution_mode = MODE_BY_LABEL[mode_label]
        st.caption("Quick Preview uses fewer particles and time steps.")
        st.warning(
            "Quick Preview uses constant test cross sections. "
            "It does not use verified LXCat data. "
            "Do not use its results for quantitative plasma diagnosis."
        )
        if not bool(preset.get("runnable", False)):
            st.warning(
                "This species is not available. "
                "The current ion collision model requires equal ion and neutral masses."
            )

        pressure_unit = str(preset["pressure_unit"])
        pressure_value = st.number_input(
            f"Neutral pressure ({pressure_unit})",
            min_value=0.0,
            value=float(preset["pressure_value"]),
            step=0.1,
            format="%.6f",
            key=f"{preset_key}_pressure",
        )
        density = st.number_input(
            "Plasma density (m^-3)",
            min_value=1.0,
            value=float(preset["config"]["N0"]),
            step=1.0e15,
            format="%.6e",
            key=f"{preset_key}_density",
        )
        electron_temperature_ev = st.number_input(
            "Electron temperature (eV)",
            min_value=1.0e-6,
            value=float(preset["config"]["Te"]),
            format="%.6f",
            key=f"{preset_key}_te",
        )
        ion_temperature_ev = st.number_input(
            "Ion temperature (eV)",
            min_value=1.0e-6,
            value=float(preset["config"]["Ti"]),
            format="%.6f",
            key=f"{preset_key}_ti",
        )
        probe_radius_m = st.number_input(
            "Probe radius (m)",
            min_value=1.0e-9,
            value=float(preset["config"]["R_MIN"]),
            step=1.0e-5,
            format="%.6e",
            key=f"{preset_key}_probe_radius",
        )
        probe_length_m = st.number_input(
            "Probe length (m)",
            min_value=1.0e-9,
            value=float(preset["probe_length_m"]),
            step=1.0e-3,
            format="%.6e",
            key=f"{preset_key}_probe_length",
        )
        root_seed = st.number_input(
            "Root seed",
            min_value=0,
            max_value=2_147_483_647,
            value=int(preset["root_seed"]),
            step=1,
            key=f"{preset_key}_root_seed",
        )

        v_start = st.number_input(
            "Start bias (V)",
            value=float(preset["v_start"]),
            step=1.0,
            key=f"{preset_key}_v_start",
        )
        v_end = st.number_input(
            "End bias (V)",
            value=float(preset["v_end"]),
            step=1.0,
            key=f"{preset_key}_v_end",
        )

        run_defaults = RUN_DEFAULTS[execution_mode]
        run_key = f"{preset_key}_{execution_mode}"
        voltage_steps = st.number_input(
            "Voltage points",
            min_value=1,
            max_value=101,
            value=int(run_defaults["voltage_steps"]),
            step=1,
            key=f"{run_key}_voltage_steps",
        )

        with st.expander("Advanced settings", expanded=False):
            radial_limit_m = st.number_input(
                "Radial limit (m)",
                min_value=1.0e-9,
                value=float(preset["config"]["R_MAX"]),
                step=1.0e-4,
                format="%.6e",
                key=f"{preset_key}_radial_limit",
            )
            radial_cells = st.number_input(
                "Radial cells",
                min_value=2,
                max_value=10000,
                value=int(preset["config"]["N_CELLS"]),
                step=1,
                key=f"{preset_key}_radial_cells",
            )
            time_step_s = st.number_input(
                "Time step (s)",
                min_value=1.0e-16,
                value=float(preset["config"]["DT"]),
                step=1.0e-13,
                format="%.6e",
                key=f"{preset_key}_time_step",
            )
            n_particles = st.number_input(
                "Macro particles for each species",
                min_value=1,
                max_value=50000,
                value=int(run_defaults["n_particles"]),
                step=100,
                key=f"{run_key}_n_particles",
            )
            n_burn_in = st.number_input(
                "Burn-in steps",
                min_value=0,
                max_value=50000,
                value=int(run_defaults["n_burn_in"]),
                step=100,
                key=f"{run_key}_n_burn_in",
            )
            n_sampling = st.number_input(
                "Sampling steps",
                min_value=1,
                max_value=50000,
                value=int(run_defaults["n_sampling"]),
                step=100,
                key=f"{run_key}_n_sampling",
            )
            wall_potential_v = st.number_input(
                "Wall potential (V)",
                value=float(preset["config"].get("V_WALL", 0.0)),
                step=1.0,
                key=f"{preset_key}_wall_potential",
            )
            semilog_e = st.checkbox(
                "Show electron current on a logarithmic scale",
                value=False,
            )
            st.caption("The electron current is a positive magnitude.")
            st.caption("A probe length of 1 m gives current in A/m.")

        config: Config | None = None
        snapshot: dict[str, Any] | None = None
        current_input_sha256: str | None = None
        input_error: str | None = None
        try:
            config_values = make_config_values(
                preset_name,
                pressure_value=pressure_value,
                pressure_unit=pressure_unit,
                density=density,
                electron_temperature_ev=electron_temperature_ev,
                ion_temperature_ev=ion_temperature_ev,
                probe_radius_m=probe_radius_m,
                radial_limit_m=radial_limit_m,
                radial_cells=radial_cells,
                time_step_s=time_step_s,
                wall_potential_v=wall_potential_v,
            )
            config = make_research_config(config_values)
            snapshot = make_input_snapshot(
                config,
                preset_name=preset_name,
                gas_name=gas_name,
                execution_mode=execution_mode,
                pressure_value=pressure_value,
                pressure_unit=pressure_unit,
                probe_length_m=probe_length_m,
                root_seed=root_seed,
                v_start=v_start,
                v_end=v_end,
                voltage_steps=voltage_steps,
                n_particles=n_particles,
                n_burn_in=n_burn_in,
                n_sampling=n_sampling,
            )
            current_input_sha256 = input_fingerprint(snapshot)
        except (TypeError, ValueError, FileNotFoundError) as exc:
            input_error = str(exc)
            st.error(f"Configuration error: {input_error}")

        stability_warnings = config.stability_warnings() if config is not None else []
        production_blocked = (
            execution_mode != PREVIEW_MODE
            or not bool(preset.get("runnable", False))
        )
        run_clicked = st.button(
            "Start Quick Preview",
            disabled=input_error is not None or production_blocked,
            use_container_width=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    stored_result = st.session_state.get("run_result")
    result, stale_result = select_result_for_input(
        stored_result,
        current_input_sha256,
    )
    if stale_result:
        st.session_state.pop("run_result", None)

    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Stability")
        if config is None:
            st.error("Set valid configuration values.")
        else:
            metrics = config.stability_metrics()
            metric_columns = st.columns(4)
            metric_columns[0].metric(
                "Cell / Debye",
                f"{metrics.cell_to_debye_ratio:.3g}",
            )
            metric_columns[1].metric(
                "dt * omega_pe",
                f"{metrics.dt_omega_pe:.3g}",
            )
            metric_columns[2].metric(
                "Electron CFL",
                f"{metrics.electron_cfl:.3g}",
            )
            metric_columns[3].metric(
                "Ion CFL",
                f"{metrics.ion_cfl:.3g}",
            )
            if metrics.is_stable:
                st.success("All stability checks are complete.")
            else:
                st.warning("One or more stability checks failed.")
                for message in stability_warnings:
                    st.markdown(f"- {message}")
            st.caption(f"Configuration SHA-256: `{config.fingerprint()}`")
            if current_input_sha256 is not None:
                st.caption(f"Input SHA-256: `{current_input_sha256}`")
        st.markdown("</div>", unsafe_allow_html=True)

        if stale_result:
            st.info("Input values changed. The previous result was removed.")

        st.write("")
        status = st.empty()
        if result is None:
            status.info("No current result is available.")

        if (
            run_clicked
            and config is not None
            and snapshot is not None
            and current_input_sha256 is not None
        ):
            progress = st.progress(0.0)

            def progress_cb(done: int, total: int, voltage: float) -> None:
                progress.progress(done / total)
                status.info(f"Voltage: {voltage:.1f} V. Point {done} of {total}.")

            try:
                run_config = prepare_run_config(config, execution_mode)
                valid_probe_length = validate_probe_length(probe_length_m)
                with st.spinner("The simulation is active."):
                    simulation = PICSimulation(
                        run_config,
                        n_particles=int(n_particles),
                        v_bias=float(v_start),
                        probe_length=valid_probe_length,
                        seed=int(root_seed),
                    )
                    iv_data = simulation.scan_voltage_range(
                        v_start=float(v_start),
                        v_end=float(v_end),
                        n_steps=int(voltage_steps),
                        n_burn_in=int(n_burn_in),
                        n_sampling=int(n_sampling),
                        progress_cb=progress_cb,
                    )
                created_utc = utc_timestamp()
                simulation_manifest = simulation.run_manifest(
                    created_utc=created_utc,
                )
                result = make_result_record(
                    iv_data,
                    run_config,
                    simulation_manifest=simulation_manifest,
                    input_snapshot=snapshot,
                    input_sha256=current_input_sha256,
                    root_seed=int(root_seed),
                )
                st.session_state["run_result"] = result
                status.success("The result is complete.")
            except Exception as exc:
                result = None
                st.session_state.pop("run_result", None)
                status.error(f"The simulation did not complete. {exc}")
            finally:
                progress.empty()

        if result is not None:
            iv_data = result["iv_data"]
            st.markdown(
                f"<div class='pill'>Points: {len(iv_data['voltages'])}</div>",
                unsafe_allow_html=True,
            )
            file_tag = str(result["input_fingerprint"])[:12]
            download_columns = st.columns(2)
            download_columns[0].download_button(
                "Download result CSV",
                data=result["csv_text"],
                file_name=f"iv_result_{file_tag}.csv",
                mime="text/csv",
                use_container_width=True,
            )
            download_columns[1].download_button(
                "Download manifest JSON",
                data=result["manifest_json"],
                file_name=f"iv_manifest_{file_tag}.json",
                mime="application/json",
                use_container_width=True,
            )

    if result is not None:
        iv_data = result["iv_data"]
        st.write("")
        fig_iv = plot_iv_curve(
            iv_data["voltages"],
            iv_data["I_total"],
            iv_data["I_electron"],
            iv_data["I_ion"],
        )
        st.pyplot(fig_iv, clear_figure=True)

        if semilog_e:
            fig_log = plot_electron_semilog(
                iv_data["voltages"],
                iv_data["I_electron"],
            )
            st.pyplot(fig_log, clear_figure=True)


if __name__ == "__main__":
    main()
