
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from core.config import Config
from core.simulation import PICSimulation

def run_robust_sim_fast():
    # 1. Configure the simulation
    pressure_pa = 0.1
    pressure_torr = pressure_pa / 133.322
    density = 1.0e15
    
    print(f"Running ROBUST-FAST sim with Density={density:.1e}, Pressure={pressure_pa} Pa")

    # Domain: 1.0 cm (Small enough for speed, big enough for sheath)
    R_MIN = 5.0e-4
    R_MAX = 1.0e-2 
    
    cfg = Config(
        N0=density,
        P_Torr=pressure_torr,
        Te=3.0,
        Ti=0.03,
        R_MIN=R_MIN,
        R_MAX=R_MAX,
        N_CELLS=200,      
        DT=1.0e-10,       # 100ps (Stability limit approx 110ps). 2x faster physics.
        SMOOTH_DENSITY=True,
        N_SMOOTHING_PASSES=5
    )

    # 2. Initialize Simulation
    # 10k particles in 1cm domain. 
    # Compared to previous run:
    # Old: 100k in 2.5cm => 40k/cm
    # New: 10k in 1.0cm => 10k/cm. (4x less density, but acceptable for demo).
    # Speedup: 10x fewer particles than 100k.
    sim = PICSimulation(cfg, n_particles=10000, seed=999) 
    
    # 3. Create I-V Scan
    print("Starting I-V scan (Robust-Fast)...")
    
    # Timeline:
    # 1 ion period ~ 1-2 us.
    # DT = 100ps = 0.1 ns.
    # 1 us = 10,000 steps.
    
    # Initial burn-in: 100,000 steps = 10 us (approx 5-10 ion periods).
    # Step burn-in: 5,000 steps = 0.5 us (fast adjustment).
    # Sampling: 10,000 steps = 1 us (averaging window).
    
    results = sim.scan_voltage_range(
        v_start=-40.0,
        v_end=100.0,
        n_steps=15,          # 10V increments (-40, -30, ..., 100)
        n_burn_in=5000,      
        n_initial_burn_in=100000, # 10 us
        n_sampling=10000,    # 1 us
        ramp_steps=1000,     
        progress_cb=lambda idx, total, v: print(f"Step {idx}/{total}: V={v:.2f} V")
    )
    
    # 4. Save and Plot
    df = pd.DataFrame(results)
    df.to_csv("results/iv_curve_robust_fast.csv", index=False)
    
    plt.figure()
    plt.plot(results["voltages"], results["I_total"], 'o-', label="Total Current")
    plt.plot(results["voltages"], results["I_electron"], 'x--', label="Electron Current")
    plt.plot(results["voltages"], -results["I_ion"], 's--', label="Ion Current (Neg)")
    plt.grid(True)
    plt.xlabel("Voltage (V)")
    plt.ylabel("Current (A/m)")
    plt.title(f"I-V Curve (Robust Fast, dt=100ps)")
    plt.legend()
    plt.savefig("results/iv_curve_robust_fast.png")
    print("Done. Saved to results/iv_curve_robust_fast.png")

if __name__ == "__main__":
    run_robust_sim_fast()
