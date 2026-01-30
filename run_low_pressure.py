
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from core.config import Config
from core.simulation import PICSimulation

def run_low_pressure_sim():
    # 1. Configure the simulation
    # 0.1 Pa ~= 7.5e-4 Torr
    pressure_pa = 0.1
    pressure_torr = pressure_pa / 133.322
    
    # 1e15 m^-3 density
    density = 1.0e15
    
    # Te = 3.0 eV.
    print(f"Running sim with Density={density:.1e}, Pressure={pressure_pa} Pa")

    # Domain:
    # R_MIN = 0.5 mm. 
    # R_MAX = 2.5 cm (25 mm) covers the sheath (~4 mm) + presheath.
    R_MIN = 5.0e-4
    R_MAX = 2.5e-2 
    
    # Grid:
    # N_CELLS = 120 => dr ~ 0.204 mm.
    
    # Timestep:
    # omega_pe ~ 1.78e9 rad/s. Stability limit ~ 110 ps.
    # We use 50 ps.
    
    cfg = Config(
        N0=density,
        P_Torr=pressure_torr,
        Te=3.0,
        Ti=0.03,
        R_MIN=R_MIN,
        R_MAX=R_MAX,
        N_CELLS=120,
        DT=5.0e-11,
        SMOOTH_DENSITY=True,
        N_SMOOTHING_PASSES=5
    )

    print("Config created. Warnings:", cfg.stability_warnings())
    
    # 2. Initialize Simulation
    # Reduced particle count for faster runs
    sim = PICSimulation(cfg, n_particles=100000, seed=42) 
    
    # 3. Create I-V Scan
    # Scan -40V to +10V.
    print("Starting I-V scan (High Precision)...")
    
    results = sim.scan_voltage_range(
        v_start=-40.0,
        v_end=100.0,
        n_steps=15,
        n_burn_in=100000, 
        n_initial_burn_in=150000, 
        n_sampling=30000, 
        ramp_steps=5000,
        progress_cb=lambda idx, total, v: print(f"Step {idx}/{total}: V={v:.2f} V")
    )
    
    # 4. Save and Plot
    df = pd.DataFrame(results)
    df.to_csv("results/iv_curve_low_pressure.csv", index=False)
    print("Data saved to results/iv_curve_low_pressure.csv")
    
    plt.figure()
    plt.plot(results["voltages"], results["I_total"], 'o-', label="Total Current")
    plt.plot(results["voltages"], results["I_electron"], 'x--', label="Electron Current")
    plt.plot(results["voltages"], -results["I_ion"], 's--', label="Ion Current (Neg)")
    plt.grid(True)
    plt.xlabel("Voltage (V)")
    plt.ylabel("Current (A/m)")
    plt.title(f"I-V Curve (P={pressure_pa} Pa, n={density:.0e} m^-3)")
    plt.legend()
    plt.savefig("results/iv_curve_low_pressure.png")
    print("Plot saved to results/iv_curve_low_pressure.png")

if __name__ == "__main__":
    run_low_pressure_sim()
