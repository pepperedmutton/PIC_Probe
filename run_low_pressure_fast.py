
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from core.config import Config
from core.simulation import PICSimulation

def run_low_pressure_sim_fast():
    # 1. Configure the simulation
    pressure_pa = 0.1
    pressure_torr = pressure_pa / 133.322
    density = 1.0e15
    
    print(f"Running FAST sim with Density={density:.1e}, Pressure={pressure_pa} Pa")

    R_MIN = 5.0e-4
    R_MAX = 2.5e-2 
    
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
    # Reduced particle count for speed (100k -> 10k)
    sim = PICSimulation(cfg, n_particles=10000, seed=42) 
    
    # 3. Create I-V Scan
    # Fewer points, less burning
    print("Starting I-V scan (Fast Mode)...")
    
    results = sim.scan_voltage_range(
        v_start=-40.0,
        v_end=100.0,
        n_steps=15,          # 10V increments
        n_burn_in=5000,     # 100k -> 5k
        n_initial_burn_in=10000, # 150k -> 10k
        n_sampling=2000,    # 30k -> 2k
        ramp_steps=1000,    # 5k -> 1k
        progress_cb=lambda idx, total, v: print(f"Step {idx}/{total}: V={v:.2f} V")
    )
    
    # 4. Save and Plot
    df = pd.DataFrame(results)
    df.to_csv("results/iv_curve_low_pressure_fast.csv", index=False)
    print("Data saved to results/iv_curve_low_pressure_fast.csv")
    
    plt.figure()
    plt.plot(results["voltages"], results["I_total"], 'o-', label="Total Current")
    plt.plot(results["voltages"], results["I_electron"], 'x--', label="Electron Current")
    plt.plot(results["voltages"], -results["I_ion"], 's--', label="Ion Current (Neg)")
    plt.grid(True)
    plt.xlabel("Voltage (V)")
    plt.ylabel("Current (A/m)")
    plt.title(f"I-V Curve (Fast Run)")
    plt.legend()
    plt.savefig("results/iv_curve_low_pressure_fast.png")
    print("Plot saved to results/iv_curve_low_pressure_fast.png")

if __name__ == "__main__":
    run_low_pressure_sim_fast()
