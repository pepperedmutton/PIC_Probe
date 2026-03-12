from __future__ import annotations

from datetime import datetime
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from core.config import Config
from core.simulation import PICSimulation

try:
    from numba.core.errors import NumbaPerformanceWarning

    warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)
except Exception:
    pass


EXP_VOLTAGES = np.array(
    [-30.0, -20.0, -10.0, 0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0],
    dtype=float,
)
EXP_CURRENT_MA_APPROX = np.array(
    [-0.7, -0.2, 1.8, 4.0, 6.0, 7.5, 8.8, 10.0, 11.2, 12.1, 12.9, 13.5],
    dtype=float,
)


def sample_current(sim: PICSimulation, n_samples: int) -> tuple[float, float, float]:
    n = max(int(n_samples), 1)
    acc_e = 0.0
    acc_i = 0.0
    for _ in range(n):
        e_hits, i_hits = sim.step()
        acc_e += (e_hits * -sim.qe) / sim.dt
        acc_i += (i_hits * sim.qi) / sim.dt

    i_e = acc_e / n
    i_i = acc_i / n
    i_total = i_e - i_i

    if sim.probe_length != 1.0:
        i_e *= sim.probe_length
        i_i *= sim.probe_length
        i_total *= sim.probe_length

    return i_e, i_i, i_total


def ramp_bias(sim: PICSimulation, v_from: float, v_to: float, ramp_steps: int) -> None:
    if ramp_steps <= 0:
        sim.v_bias = float(v_to)
        sim._update_fields()
        return
    for idx in range(ramp_steps):
        frac = (idx + 1) / ramp_steps
        sim.v_bias = float(v_from + frac * (v_to - v_from))
        sim._update_fields()
        sim.step()


def run_bulk_physical_scan() -> None:
    pressure_pa = 0.3
    pressure_torr = pressure_pa / 133.322

    m_h_plus = 1.67262192369e-27
    probe_diameter_m = 0.15e-3
    probe_radius_m = 0.5 * probe_diameter_m
    probe_length_m = 10.0e-3
    r_max_m = 2.0e-3
    n_cells = 220

    # Boundary setting requested by user:
    # fixed grounded wall (0 V reference).
    cfg = Config(
        N0=1.1e16,
        Te=1.35,
        Ti=0.2,
        P_Torr=pressure_torr,
        m_i=m_h_plus,
        R_MIN=probe_radius_m,
        R_MAX=r_max_m,
        N_CELLS=n_cells,
        DT=3.0e-12,
        WALL_BC_MODE="fixed",
        V_WALL=0.0,
        OUTER_BOUNDARY_INJECTION=True,
        OUTER_INJECTION_MODE="flux",
        ION_INJECTION_BOHM=False,
        SIGMA_EN_ELASTIC=8.0e-20,
        SIGMA_EN_EXC=1.2e-20,
        SIGMA_EN_ION=2.5e-20,
        ENABLE_IONIZATION_SECONDARIES=True,
        SECONDARY_E_EV=2.0,
        SECONDARY_I_EV=0.05,
        MAX_IONIZATION_PAIRS_PER_STEP=70000,
        ENABLE_BULK_FEEDBACK_SOURCE=True,
        BULK_SOURCE_R_FRAC_MIN=0.77,
        BULK_SOURCE_GAIN=0.8,
        BULK_SOURCE_MAX_PAIRS_PER_STEP=70000,
        BULK_SOURCE_REL_DEADBAND=0.01,
        E_EXC_EV=10.2,
        E_ION_EV=15.4,
        SMOOTH_DENSITY=True,
        N_SMOOTHING_PASSES=1,
        ADAPTIVE_STABILITY=True,
        USE_CUDA=True,
        CUDA_THREADS_PER_BLOCK=256,
    )

    warnings = cfg.stability_warnings()
    if warnings:
        print("Initial stability warnings:")
        for msg in warnings:
            print(f"  - {msg}")

    # Fitting mode: per-bias relaxation tuned for experiment-shape matching.
    burn_in_steps = 2500
    sampling_steps = 2500
    ramp_steps = 80
    n_particles = 26000
    sigma_cex = 2.7e-19
    base_seed = 20260312

    print(
        "Running bulk scan: "
        f"P={pressure_pa:.3f} Pa, N0={cfg.N0:.2e} m^-3, Te={cfg.Te:.2f} eV, Ti={cfg.Ti:.2f} eV, "
        f"d_probe={probe_diameter_m * 1e3:.3f} mm, Rmax={cfg.R_MAX * 1e3:.2f} mm, cells={cfg.N_CELLS}"
    )
    print(
        "Boundary settings: "
        f"mode={cfg.WALL_BC_MODE}, V_wall={cfg.V_WALL:.2f} V, "
        f"outer_injection={cfg.OUTER_BOUNDARY_INJECTION}, "
        f"injection_mode={cfg.OUTER_INJECTION_MODE}, ion_bohm={cfg.ION_INJECTION_BOHM}"
    )
    print(
        "Source settings: "
        f"ionization_secondaries={cfg.ENABLE_IONIZATION_SECONDARIES}, "
        f"bulk_feedback={cfg.ENABLE_BULK_FEEDBACK_SOURCE}, "
        f"bulk_r_min_frac={cfg.BULK_SOURCE_R_FRAC_MIN:.2f}, "
        f"bulk_gain={cfg.BULK_SOURCE_GAIN:.2f}"
    )
    print(
        "Per-bias evolution: "
        f"burn_in={burn_in_steps} steps ({burn_in_steps * cfg.DT:.3e} s), "
        f"sample={sampling_steps} steps ({sampling_steps * cfg.DT:.3e} s)"
    )
    print(f"Voltage ramp steps between points: {ramp_steps}")

    sim = PICSimulation(
        cfg,
        n_particles=n_particles,
        v_bias=float(EXP_VOLTAGES[0]),
        probe_length=probe_length_m,
        sigma_cex=sigma_cex,
        seed=base_seed,
    )
    print(f"Compute backend: {'CUDA' if sim.use_cuda else 'CPU'}")

    i_total_a = np.zeros_like(EXP_VOLTAGES)
    i_e_a = np.zeros_like(EXP_VOLTAGES)
    i_i_a = np.zeros_like(EXP_VOLTAGES)
    profile_rows: list[dict[str, float]] = []

    # Warm-start scan: each bias inherits the final state from previous bias.
    for idx, v in enumerate(EXP_VOLTAGES):
        if idx > 0:
            ramp_bias(sim, float(EXP_VOLTAGES[idx - 1]), float(v), ramp_steps)

        sim.v_bias = float(v)
        sim._update_fields()

        for _ in range(burn_in_steps):
            sim.step()

        sim.sync_state_from_device(include_particles=False)
        ne_profile = -sim.rho_e / sim.config.e
        for ridx, r_val in enumerate(sim.r_grid):
            profile_rows.append(
                {
                    "scan_idx": float(idx),
                    "V_bias_V": float(v),
                    "r_m": float(r_val),
                    "phi_V": float(sim.phi[ridx]),
                    "ne_m3": float(ne_profile[ridx]),
                }
            )

        i_e, i_i, i_total = sample_current(sim, sampling_steps)
        i_total_a[idx] = i_total
        i_e_a[idx] = i_e
        i_i_a[idx] = i_i

        print(
            f"{idx + 1:02d}/{EXP_VOLTAGES.size}: V={v:>6.1f} V, "
            f"I_total={i_total * 1e3:>8.3f} mA"
        )

    i_total_ma = i_total_a * 1e3
    rmse_ma = float(np.sqrt(np.mean((i_total_ma - EXP_CURRENT_MA_APPROX) ** 2)))

    out_dir = Path("results") / "test_runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    csv_path = out_dir / f"iv_bulk_physical_tuned_{ts}.csv"
    png_path = out_dir / f"iv_bulk_physical_tuned_{ts}.png"
    profile_csv_path = out_dir / f"profiles_stepwise_bulk_physical_tuned_{ts}.csv"

    df = pd.DataFrame(
        {
            "V_bias_V": EXP_VOLTAGES,
            "I_total_A": i_total_a,
            "I_total_mA": i_total_ma,
            "I_electron_A": i_e_a,
            "I_ion_A": i_i_a,
            "I_exp_mA_approx": EXP_CURRENT_MA_APPROX,
        }
    )
    df.to_csv(csv_path, index=False)
    profile_df = pd.DataFrame(profile_rows)
    profile_df.to_csv(profile_csv_path, index=False)

    plt.figure(figsize=(8.5, 5.2))
    plt.plot(EXP_VOLTAGES, i_total_ma, "o-", linewidth=2.0, label="Simulation (bulk)")
    plt.plot(EXP_VOLTAGES, EXP_CURRENT_MA_APPROX, "s--", linewidth=1.8, label="Kakati 2017 (approx.)")
    plt.grid(True, alpha=0.3)
    plt.xlabel("Probe bias (V)")
    plt.ylabel("Current (mA)")
    plt.title(
        "Hydrogen I-V, 0.3 Pa, 0.15 mm probe\n"
        f"bulk BC tuned run (RMSE = {rmse_ma:.3f} mA)"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(png_path, dpi=170)

    print("\nFinished.")
    print(
        f"Adaptive state (last bias): dt={sim.dt:.3e} s, "
        f"dr={sim.dr:.3e} m, cells={sim.n_nodes - 1}"
    )
    print(f"RMSE vs approximate experiment: {rmse_ma:.3f} mA")
    print(f"CSV: {csv_path}")
    print(f"Plot: {png_path}")
    print(f"Stepwise profiles CSV (phi, ne): {profile_csv_path}")


if __name__ == "__main__":
    run_bulk_physical_scan()
