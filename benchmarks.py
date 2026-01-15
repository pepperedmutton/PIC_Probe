from __future__ import annotations

from pathlib import Path
import sys

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


def test_vacuum_cylindrical_capacitor(out_dir: Path) -> float:
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

    iv = sim.scan_voltage_range(
        v_start=-10.0,
        v_end=-2.0,
        n_steps=9,
        n_burn_in=6000,
        n_sampling=6000,
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
    cfg = Config(
        N_CELLS=120,
        DT=1.0e-11,
        R_MIN=2.0e-5,      # 进一步减小探针半径以满足 r_p << lambda_D
        R_MAX=2.0e-3,
        N0=5.0e15,         # 方案2: 从10^14增大到5x10^15
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
        n_particles=50000,  # 方案3: 从8000增加到50000
        v_bias=-50.0,
        probe_length=1.0,
        sigma_cex=0.0,
        seed=11,
    )

    iv = sim.scan_voltage_range(
        v_start=-50.0,
        v_end=-10.0,
        n_steps=9,
        n_burn_in=20000,    # 方案3: 从8000增加到20000
        n_sampling=30000,   # 方案3: 从8000增加到30000
    )

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
    return r2


def main() -> None:
    out_dir = ensure_results_dir()
    err = test_vacuum_cylindrical_capacitor(out_dir)
    slope, te_inferred = test_electron_temperature(out_dir)
    r2 = test_oml_regime(out_dir)

    print(f"Test 1 - Vacuum capacitor max relative error: {err * 100:.4f}%")
    if slope != 0.0:
        print(f"Test 2 - ln(I_e) slope: {slope:.3f}, inferred Te: {te_inferred:.2f} eV")
    else:
        print("Test 2 - ln(I_e) fit failed (insufficient positive current samples).")
    print(f"Test 3 - OML linearity R^2: {r2:.3f}")
    print(f"Saved plots to {out_dir}")


if __name__ == "__main__":
    main()
