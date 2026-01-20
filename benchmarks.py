from __future__ import annotations

from pathlib import Path
import sys
import time

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config import Config
from core.fields import solve_poisson_cylindrical
from core.simulation import PICSimulation


def ensure_results_dir() -> Path:
    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    return out_dir


def neutral_density_from_torr(pressure_torr: float, t_gas_k: float) -> float:
    p_pa = pressure_torr * 133.322368
    return p_pa / (1.380649e-23 * t_gas_k)


def test_vacuum_cylindrical_capacitor(out_dir: Path) -> float:
    print("\n[Test 1/4] Running vacuum cylindrical capacitor test...")
    r_min = 0.5e-3
    r_max = 5.0e-3
    v_bias = -100.0
    v_wall = 0.0
    n_cells = 200

    dr = (r_max - r_min) / n_cells
    n_nodes = n_cells + 1
    r_grid = r_min + dr * np.arange(n_nodes)
    rho = np.zeros(n_nodes)
    phi = np.zeros(n_nodes)

    solve_poisson_cylindrical(rho, phi, r_min, dr, 8.8541878128e-12, v_bias, v_wall)

    denom = np.log(r_min / r_max)
    phi_theory = v_bias * np.log(r_grid / r_max) / denom

    max_err = np.max(np.abs(phi - phi_theory))
    span = np.max(np.abs(phi_theory)) if np.max(np.abs(phi_theory)) > 0 else 1.0
    rel_err = max_err / span

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.plot(r_grid * 1e3, phi_theory, color="#1d2327", linewidth=2.0, label="Analytical Theory")
    ax.scatter(r_grid * 1e3, phi, s=14, color="#b24a2f", label="PIC Solver")
    ax.set_xlabel("r (mm)")
    ax.set_ylabel("phi (V)")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "benchmark_test1_vacuum_capacitor.png", dpi=160)
    data = np.column_stack((r_grid, phi, phi_theory))
    np.savetxt(
        out_dir / "benchmark_test1_vacuum_capacitor.csv",
        data,
        delimiter=",",
        header="r_m,phi_pic,phi_theory",
        comments="",
    )
    return rel_err


def test_electron_temperature(out_dir: Path) -> tuple[float, float]:
    print("\n[Test 2/4] Running electron temperature measurement...")
    cfg = Config(
        N_CELLS=100,
        DT=1.0e-11,
        R_MIN=5.0e-4,
        R_MAX=5.0e-3,
        N0=1.0e15,
        Te=2.0,
        Ti=0.026,
        P_Torr=0.0,
        V_WALL=0.0,
        SIGMA_EN_ELASTIC=0.0,
        SIGMA_EN_EXC=0.0,
        SIGMA_EN_ION=0.0,
    )

    sim = PICSimulation(
        cfg,
        n_particles=8000,
        v_bias=-10.0,
        probe_length=1.0,
        sigma_cex=0.0,
        seed=10,
    )

    def progress_callback(done, total, v):
        pass  # Progress handled by tqdm

    print("  Running voltage scan...")
    iv = sim.scan_voltage_range(
        v_start=-10.0,
        v_end=-2.0,
        n_steps=9,
        n_burn_in=6000,
        n_sampling=6000,
        progress_cb=progress_callback,
    )

    voltages = iv["voltages"]
    i_e = iv["I_electron"]
    mask = i_e > 0.0
    if mask.sum() < 2:
        return 0.0, 0.0

    x = voltages[mask]
    y = np.log(i_e[mask])
    coeff = np.polyfit(x, y, 1)
    slope = coeff[0]
    inferred_te = 1.0 / slope if slope != 0.0 else 0.0

    fit_line = np.polyval(coeff, x)
    fit_full = np.full_like(voltages, np.nan, dtype=float)
    fit_full[mask] = fit_line

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.scatter(voltages, np.log(np.maximum(i_e, 1.0e-30)), s=18, color="#2b6f73", label="PIC Data")
    ax.plot(x, fit_line, color="#1d2327", linewidth=2.0, label="Fit")
    ax.set_xlabel("V_bias (V)")
    ax.set_ylabel("ln(I_e)")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    ax.set_title(f"Inferred Te = {inferred_te:.2f} eV (slope={slope:.3f})")
    fig.tight_layout()
    fig.savefig(out_dir / "benchmark_test2_electron_temperature.png", dpi=160)
    print(f"  ✓ Test 2 complete: Te = {inferred_te:.2f} eV (slope = {slope:.3f})")
    data = np.column_stack((voltages, i_e, np.log(np.maximum(i_e, 1.0e-30)), fit_full))
    np.savetxt(
        out_dir / "benchmark_test2_electron_temperature.csv",
        data,
        delimiter=",",
        header="V_bias,I_electron,ln_Ie,fit_ln_Ie",
        comments="",
    )

    return slope, inferred_te


def test_oml_regime(out_dir: Path) -> float:
    print("\n[Test 3/4] Running OML regime linearity test...")
    print("  This test may take several minutes...")
    cfg = Config(
        N_CELLS=120,
        DT=1.0e-11,
        R_MIN=5.0e-4,       # 修复：500 μm >> λ_D (67 μm)
        R_MAX=5.0e-3,
        N0=5.0e15,          # 方案2: 从10^14增大到5x10^15
        Te=2.0,
        Ti=0.026,
        P_Torr=0.0,
        V_WALL=0.0,
        SIGMA_EN_ELASTIC=0.0,
        SIGMA_EN_EXC=0.0,
        SIGMA_EN_ION=0.0,
    )

    sim = PICSimulation(
        cfg,
        n_particles=20000,
        v_bias=-50.0,
        probe_length=1.0,
        sigma_cex=0.0,
        seed=11,
    )

    print(f"  Parameters: {sim.n_particles} particles, {cfg.N0:.1e} m^-3 density")
    print(f"  Scanning {9} voltage points from -50V to -10V...")
    
    def progress_callback(done, total, v):
        pct = (done / total) * 100
        bar_len = 30
        filled = int(bar_len * done / total)
        bar = '█' * filled + '░' * (bar_len - filled)
        print(f"\r  Progress: [{bar}] {pct:.0f}% ({done}/{total}) @ {v:.1f}V", end='', flush=True)
    
    iv = sim.scan_voltage_range(
        v_start=-50.0,
        v_end=-10.0,
        n_steps=9,
        n_burn_in=200000,
        n_sampling=80000,
        progress_cb=progress_callback,
    )
    print()  # New line after progress bar

    voltages = iv["voltages"]
    i_i = iv["I_ion"]
    x = np.abs(voltages)
    y = i_i * i_i
    coeff = np.polyfit(x, y, 1)
    y_fit = np.polyval(coeff, x)

    ss_res = np.sum((y - y_fit) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.scatter(x, y, s=18, color="#b24a2f", label="PIC Data")
    ax.plot(x, y_fit, color="#1d2327", linewidth=2.0, label="Linear Fit")
    ax.set_xlabel("|V_bias| (V)")
    ax.set_ylabel("I_i^2 (A^2)")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    ax.set_title(f"OML check: R^2 = {r2:.3f}")
    fig.tight_layout()
    fig.savefig(out_dir / "benchmark_test3_oml_ion.png", dpi=160)
    data = np.column_stack((voltages, x, i_i, y, y_fit))
    np.savetxt(
        out_dir / "benchmark_test3_oml_ion.csv",
        data,
        delimiter=",",
        header="V_bias,abs_V,I_ion,I_ion_sq,fit_Iion_sq",
        comments="",
    )
    print(f"  ✓ Test 3 complete: OML R² = {r2:.3f}")
    return r2


def test_collisional_damping(out_dir: Path) -> float:
    print("\n[Test 4/4] Running collisional damping test...")
    print("  This test may take several minutes...")
    pressure_list = np.array([0.0, 0.1, 0.5, 1.0, 3.0, 5.0, 10.0])

    cfg = Config(
        N_CELLS=160,
        DT=1.0e-11,
        R_MIN=1.5e-4,
        R_MAX=1.0e-2,
        N0=1.0e16,
        Te=2.0,
        Ti=0.026,
        P_Torr=0.0,
        V_WALL=0.0,
    )

    sim = PICSimulation(
        cfg,
        n_particles=30000,
        v_bias=-40.0,
        probe_length=1.0,
        sigma_cex=8.0e-18,
        seed=22,
    )

    t_gas_k = 300.0
    sim.vth_gas = np.sqrt(cfg.k_B * t_gas_k / cfg.m_i)

    burn_in = 5000
    sampling = 5000
    i_ion = np.zeros(pressure_list.shape[0])

    for idx, p in enumerate(pressure_list):
        sim.n_g = neutral_density_from_torr(float(p), t_gas_k)

        for _ in range(burn_in):
            sim.step()

        acc_i = 0.0
        for _ in range(sampling):
            _, i_hits = sim.step()
            acc_i += (i_hits * sim.qi) / sim.dt

        i_ion[idx] = acc_i / sampling
        print(f"  P = {p:>4.1f} Torr -> I_ion = {i_ion[idx]:.3e} A/m")

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.plot(pressure_list, i_ion, marker="o", color="#2b6f73", linewidth=2.0)
    ax.set_xlabel("Neutral Pressure (Torr)")
    ax.set_ylabel("Ion Current (A/m)")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "benchmark_test4_collisional_damping.png", dpi=160)

    data = np.column_stack((pressure_list, i_ion))
    np.savetxt(
        out_dir / "benchmark_test4_collisional_damping.csv",
        data,
        delimiter=",",
        header="pressure_torr,I_ion",
        comments="",
    )

    i0 = i_ion[0] if i_ion[0] != 0.0 else 1.0
    suppression_ratio = i_ion[-1] / i0
    print(f"  ✓ Test 4 complete: suppression ratio = {suppression_ratio:.3f}")
    return suppression_ratio


def main() -> None:
    print("="*60)
    print("PICSIMU Benchmark Suite")
    print("="*60)
    start_time = time.time()
    
    out_dir = ensure_results_dir()
    print(f"Output directory: {out_dir}\n")
    
    err = test_vacuum_cylindrical_capacitor(out_dir)
    slope, te_inferred = test_electron_temperature(out_dir)
    r2 = test_oml_regime(out_dir)
    suppression_ratio = test_collisional_damping(out_dir)

    elapsed = time.time() - start_time
    print("\n" + "="*60)
    print("BENCHMARK RESULTS")
    print("="*60)
    print(f"Test 1 - Vacuum capacitor max relative error: {err * 100:.4f}%")
    if slope != 0.0:
        print(f"Test 2 - ln(I_e) slope: {slope:.3f}, inferred Te: {te_inferred:.2f} eV")
    else:
        print("Test 2 - ln(I_e) fit failed (insufficient positive current samples).")
    print(f"Test 3 - OML linearity R^2: {r2:.3f}")
    print(f"Test 4 - Current suppression ratio (10 Torr / 0 Torr): {suppression_ratio:.3f}")
    print("="*60)
    print(f"Total runtime: {elapsed:.1f} seconds")
    print(f"Saved plots to: {out_dir}")
    print("="*60)
    print(f"Total runtime: {elapsed:.1f} seconds")
    print(f"Saved plots to: {out_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
