
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from core.config import Config
from core.simulation import PICSimulation

def run_robust_sim():
    # 1. Configure the simulation
    # 0.1 Pa ~= 7.5e-4 Torr
    pressure_pa = 0.1
    pressure_torr = pressure_pa / 133.322
    
    # 1e15 m^-3 density
    density = 1.0e15
    
    print(f"Running ROBUST sim (Optimized) with Density={density:.1e}, Pressure={pressure_pa} Pa")

    # Domain Optimization:
    # Reducing R_MAX from 2.5cm to 1.0cm.
    # The sheath is ~2mm, so 1cm is plenty of buffer (5x sheath max).
    # This reduces volume by ~6x, allowing us to use fewer particles for same density.
    R_MIN = 5.0e-4
    R_MAX = 1.0e-2 
    
    cfg = Config(
        N0=density,
        P_Torr=pressure_torr,
        Te=3.0,
        Ti=0.03,
        R_MIN=R_MIN,
        R_MAX=R_MAX,
        N_CELLS=200,      # High resolution (dr ~ 0.05 mm)
        DT=5.0e-11,       # 50ps (Stable)
        SMOOTH_DENSITY=True,
        N_SMOOTHING_PASSES=5
    )

    print("Config created. Warnings:", cfg.stability_warnings())
    
    # 2. Initialize Simulation
    # 40k particles in 1cm domain is equivalent to ~240k particles in 2.5cm domain.
    # This gives us better statistics than the original 100k run, with much less work.
    sim = PICSimulation(cfg, n_particles=40000, seed=123) 
    
    # 3. Create I-V Scan
    # Scan -40V to +100V.
    print("Starting I-V scan (Robust Mode)...")
    
    # Burn-in Strategy:
    # Ion plasma period (Ar+, 1e15) is roughly 1/f_pi. 
    # f_pi = 1.05e6 Hz => T_pi ~ 1 us.
    # 500,000 steps * 50 ps = 25 us.
    # This gives ions plenty of time to reach equilibrium sheaths.
    
    results = sim.scan_voltage_range(
        v_start=-40.0,
        v_end=100.0,
        n_steps=15,          # 10V increments
        n_burn_in=50000,     # 2.5us inter-step burn-in
        n_initial_burn_in=500000, # 25 us initial burn-in to KILL THE SPIKE.
        n_sampling=50000,    # Good statistics (2.5 us averaging window).
        ramp_steps=5000,     # Smooth transitions
        progress_cb=lambda idx, total, v: print(f"Step {idx}/{total}: V={v:.2f} V")
    )
    
    # 4. Save and Plot
    df = pd.DataFrame(results)
    df.to_csv("results/iv_curve_robust.csv", index=False)
    print("Data saved to results/iv_curve_robust.csv")
    
    plt.figure()
    plt.plot(results["voltages"], results["I_total"], 'o-', label="Total Current")
    plt.plot(results["voltages"], results["I_electron"], 'x--', label="Electron Current")
    plt.plot(results["voltages"], -results["I_ion"], 's--', label="Ion Current (Neg)")
    plt.grid(True)
    plt.xlabel("Voltage (V)")
    plt.ylabel("Current (A/m)")
    plt.title(f"I-V Curve (Robust Run)")
    plt.legend()
    plt.savefig("results/iv_curve_robust.png")
    print("Plot saved to results/iv_curve_robust.png")

if __name__ == "__main__":
    run_robust_sim()
