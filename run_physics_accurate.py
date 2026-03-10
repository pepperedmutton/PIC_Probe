
from pathlib import Path
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from core.config import Config
from core.simulation import PICSimulation


def sample_current(
    sim: PICSimulation,
    sampling_steps: int,
) -> tuple[float, float, float]:
    """Sample currents at current bias and return averaged (Ie, Ii, Itotal)."""
    n_samples = max(int(sampling_steps), 1)
    acc_e = 0.0
    acc_i = 0.0
    for _ in range(n_samples):
        e_hits, i_hits = sim.step()
        acc_e += (e_hits * -sim.qe) / sim.dt
        acc_i += (i_hits * sim.qi) / sim.dt

    i_e = acc_e / n_samples
    i_i = acc_i / n_samples
    i_total = i_e - i_i

    if sim.probe_length != 1.0:
        i_e *= sim.probe_length
        i_i *= sim.probe_length
        i_total *= sim.probe_length

    return i_e, i_i, i_total


def ramp_bias(sim: PICSimulation, v_from: float, v_to: float, ramp_steps: int) -> None:
    """Smoothly ramp probe bias between two points."""
    if ramp_steps <= 0:
        sim.v_bias = float(v_to)
        sim._update_fields()
        return
    for step_idx in range(ramp_steps):
        frac = (step_idx + 1) / ramp_steps
        sim.v_bias = float(v_from + frac * (v_to - v_from))
        sim._update_fields()
        sim.step()


def stabilize_bias(
    sim: PICSimulation,
    bias: float,
    min_burn_in: int,
    block_steps: int,
    max_blocks: int,
    rel_tol: float,
    abs_tol: float,
) -> int:
    """Run until consecutive current blocks converge; returns blocks consumed."""
    sim.v_bias = float(bias)
    sim._update_fields()

    for _ in range(max(int(min_burn_in), 0)):
        sim.step()

    prev_block_i: float | None = None
    for block_idx in range(1, max(int(max_blocks), 1) + 1):
        _, _, block_i = sample_current(sim, block_steps)
        if prev_block_i is not None:
            threshold = max(abs_tol, rel_tol * max(abs(prev_block_i), abs(block_i), 1.0))
            if abs(block_i - prev_block_i) <= threshold:
                return block_idx
        prev_block_i = block_i
    return max(int(max_blocks), 1)


def run_hydrogen_test() -> None:
    # Requested plasma conditions:
    # Hydrogen plasma, P = 0.3 Pa, n = 1e16 m^-3, T = 1 eV.
    pressure_pa = 0.3
    pressure_torr = pressure_pa / 133.322
    density = 1.0e16
    temperature_ev = 1.0
    m_h_plus = 1.67262192369e-27
    probe_diameter_m = 0.4e-3
    probe_radius_m = 0.5 * probe_diameter_m

    print(
        f"Running HYDROGEN test: n={density:.2e} m^-3, "
        f"P={pressure_pa} Pa ({pressure_torr:.4e} Torr), T={temperature_ev} eV, "
        f"probe_d={probe_diameter_m * 1e3:.3f} mm"
    )

    # Hydrogen cross sections:
    # Use hydrogen-target constants (fallback) because no local H2 LXCat file is present.
    cfg = Config(
        N0=density,
        Te=temperature_ev,
        Ti=temperature_ev,
        P_Torr=pressure_torr,
        m_i=m_h_plus,
        CROSS_SECTION_TARGET="H2",
        R_MIN=probe_radius_m,
        R_MAX=2.0e-3,
        N_CELLS=400,
        DT=5.0e-13,
        LXCAT_ELECTRON_FILE=None,
        LXCAT_ION_FILE=None,
        SIGMA_EN_ELASTIC=6.0e-20,
        SIGMA_EN_EXC=1.5e-20,
        SIGMA_EN_ION=2.0e-21,
        SIGMA_IN_ELASTIC=2.0e-19,
        E_EXC_EV=10.2,
        E_ION_EV=15.4,
        ENABLE_IONIZATION_SECONDARIES=False,
        SMOOTH_DENSITY=True,
        N_SMOOTHING_PASSES=3,
    )

    warnings = cfg.stability_warnings()
    if warnings:
        print("Stability warnings:")
        for w in warnings:
            print(f"  - {w}")

    sim = PICSimulation(
        cfg,
        n_particles=12000,
        v_bias=-30.0,
        probe_length=1.0,
        sigma_cex=5.0e-19,
        seed=20260309,
    )

    # Stronger preconditioning before scan.
    precondition_steps = 80000
    print(f"Preconditioning at -30 V for {precondition_steps} steps...")
    for _ in range(precondition_steps):
        sim.step()

    # Requested sweep controls.
    v_start = -30.0
    v_end = 100.0
    n_points = 14
    n_repeats = 5
    ramp_steps = 2000

    # Increased burn-in and stricter per-step stabilization.
    stabilize_min_burn = 20000
    stabilize_block_steps = 4000
    stabilize_max_blocks = 6
    stabilize_rel_tol = 0.03
    stabilize_abs_tol = 0.2
    n_sampling = 6000

    voltages = np.linspace(v_start, v_end, n_points)
    i_total = np.zeros(n_points)
    i_electron = np.zeros(n_points)
    i_ion = np.zeros(n_points)
    i_total_std = np.zeros(n_points)
    i_electron_std = np.zeros(n_points)
    i_ion_std = np.zeros(n_points)

    print("Starting hydrogen I-V scan (stabilized + 5 repeats/point)...")
    for idx, v in enumerate(voltages):
        print(f"\nStep {idx + 1}/{n_points}: target V={v:.1f} V")

        if idx > 0:
            ramp_bias(sim, float(voltages[idx - 1]), float(v), ramp_steps)

        rep_i_t: list[float] = []
        rep_i_e: list[float] = []
        rep_i_i: list[float] = []

        for rep in range(n_repeats):
            extra_first = 20000 if (idx == 0 and rep == 0) else 0
            blocks = stabilize_bias(
                sim=sim,
                bias=float(v),
                min_burn_in=stabilize_min_burn + extra_first,
                block_steps=stabilize_block_steps,
                max_blocks=stabilize_max_blocks,
                rel_tol=stabilize_rel_tol,
                abs_tol=stabilize_abs_tol,
            )
            i_e, i_i, i_t = sample_current(sim, n_sampling)
            rep_i_e.append(i_e)
            rep_i_i.append(i_i)
            rep_i_t.append(i_t)
            print(
                f"  repeat {rep + 1}/{n_repeats}: "
                f"stable_blocks={blocks}, I_total={i_t:.6e} A/m"
            )

        i_total[idx] = float(np.mean(rep_i_t))
        i_electron[idx] = float(np.mean(rep_i_e))
        i_ion[idx] = float(np.mean(rep_i_i))
        if n_repeats > 1:
            i_total_std[idx] = float(np.std(rep_i_t, ddof=1))
            i_electron_std[idx] = float(np.std(rep_i_e, ddof=1))
            i_ion_std[idx] = float(np.std(rep_i_i, ddof=1))
        print(
            f"  step mean: I_total={i_total[idx]:.6e} A/m, "
            f"std={i_total_std[idx]:.6e}"
        )

    results = {
        "voltages": voltages,
        "I_total": i_total,
        "I_electron": i_electron,
        "I_ion": i_ion,
        "I_total_std": i_total_std,
        "I_electron_std": i_electron_std,
        "I_ion_std": i_ion_std,
    }

    out_dir = Path("results") / "test_runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = out_dir / f"iv_curve_{timestamp}.csv"
    png_path = out_dir / f"iv_curve_{timestamp}.png"

    df = pd.DataFrame(results)
    df.to_csv(csv_path, index=False)

    plt.figure(figsize=(8, 5))
    plt.errorbar(
        results["voltages"],
        results["I_total"],
        yerr=results["I_total_std"],
        fmt="o-",
        capsize=3,
        label="Total current (mean±std)",
    )
    plt.plot(results["voltages"], results["I_electron"], "x--", label="Electron current")
    plt.plot(results["voltages"], -results["I_ion"], "s--", label="Ion current (neg)")
    plt.grid(True, alpha=0.3)
    plt.xlabel("Probe voltage (V)")
    plt.ylabel("Current (A/m)")
    plt.title("Synthetic I-V (H+, 0.3 Pa, 1e16 m^-3, 1 eV, d=0.4 mm, Vmax=100 V)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(png_path, dpi=160)

    print(f"Saved synthetic data to: {csv_path}")
    print(f"Saved I-V plot to: {png_path}")


if __name__ == "__main__":
    run_hydrogen_test()
