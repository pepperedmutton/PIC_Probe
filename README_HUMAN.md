# PICSIMU - Human Overview

PICSIMU is a 1D cylindrical Particle-in-Cell simulation with Monte Carlo
collisions (PIC-MCC) that produces Langmuir probe I-V curves for
high-pressure, collisional plasmas. It exists to generate synthetic data and
support plasma parameter inference where collisionless OML models break down.

## Warning

Do not modify code manually. Code changes must be executed by an agent using at
least a DeepSeekR1 core.

## What it does

- Simulates the radial sheath around a cylindrical probe in 1D geometry
- Tracks electrons and ions with angular momentum conservation
- Solves the cylindrical Poisson equation with ion-neutral CEX and simplified electron-neutral collisions
- Outputs I-V curves plus potential and density profiles

## Why it exists

- Extends Langmuir probe diagnostics into high-pressure regimes (1-200 Torr)
- Generates synthetic data for ML-based parameter inference
- Targets industrial and atmospheric-pressure plasma applications

## Repository layout

- `core/`: Numba-accelerated physics engine
- `frontend/`: Streamlit UI for configuration and plots
- `benchmarks.py`: benchmark-suite CLI entry
- `run_physics_accurate.py`: primary CLI test/data-generation entry
- `results/`: benchmark outputs and example curves
- `README.md`: full technical / AI-facing documentation

Legacy root-level experiment/debug scripts have been removed to keep the repo
focused on canonical entry points.

## Quick start

```powershell
streamlit run frontend/app.py
```

Minimal CLI run (for a quick smoke test):

```powershell
@'
from core.config import Config
from core.simulation import PICSimulation

cfg = Config()
sim = PICSimulation(cfg, n_particles=2000, v_bias=-10.0, seed=1)
res = sim.run(n_steps=200, n_warmup=100)
print(res.avg_current)
'@ | python -
```

## Typical inputs

- Pressure (Torr), density (m^-3), electron temperature (eV)
- Probe bias voltage (V)
- Numerical settings: grid cells, time step, particle count

## Outputs

- I-V curve data (total, electron, ion currents)
- Potential and density profiles vs radius
- Benchmark plots in `results/`

## Validation status (high level)

- Poisson solver: passed vacuum cylindrical capacitor test
- Electron temperature check: passed (Te ≈ 2.02 eV)
- OML ion dynamics: passed (R² = 0.993 for I_i^2 vs |V|)
- Collisional damping: passed (ion current suppression with pressure)

## Benchmarks (what they check)

- Vacuum cylindrical capacitor: validates the Poisson solver and the 1/r geometry term
- Electron temperature check: verifies velocity sampling and the Boltzmann relation in the retarding region
- OML ion dynamics: checks angular momentum conservation and I_i^2 ∝ |V| behavior
- Collisional damping: checks ion current suppression vs pressure (CEX)

## Model assumptions (brief)

- 1D radial cylindrical geometry, no axial/azimuthal spatial variation
- Electrostatic approximation only (no magnetic fields)
- Species are electrons and singly charged ions (argon)
- Collisions include ion-neutral CEX + elastic scattering and electron-neutral processes
- Energy-dependent cross sections supported via LXCat tables (optional)
- Electron-impact ionization can spawn secondary e-/ion macro-particles (optional)
- Probe and wall are absorbing; outer boundary injects particles to maintain density

## Key limitations

- 1D radial model only (no axial/azimuthal spatial variation)
- Collision models remain simplified (e.g., no energy-dependent data unless LXCat tables supplied)

Note: the default config will attempt to read `CS.txt` in the repo root as a local LXCat export if present.
- Electrostatic approximation (no magnetic fields)

For full technical details, see `README.md`.

## 2026-03-11 Kakati 2017 Comparison Update (0.15 mm Probe)

Reference:
- Kakati et al., *Scientific Reports* 7, 490 (2017), PMCID: PMC5593904
- Target trace: Figure 1 clean H-plasma I-V curve (image-read points)

### Current simulation setup used for comparison

- Probe geometry: diameter `0.15 mm`, length `10 mm`
- Domain: `R_MIN = 7.5e-5 m`, `R_MAX = 2.0e-3 m`
- Pressure: `0.08 Pa` (`8e-4 mbar`)
- Sweep: `-30 V` to `+80 V`, 12 points
- Tuned model knobs:
  - `N0 = 9.6e15 m^-3`
  - `Te = 0.6432 eV`
  - `Ti = 0.2167 eV`
  - `V_WALL = -9.0938 V`
  - `sigma_cex = 1.7e-19 m^2`
  - `ION_INJECTION_BOHM = False`
- Numerical controls:
  - `n_particles = 6000` (per species)
  - `n_initial_burn_in = 2800`
  - `n_burn_in = 1500`
  - `n_sampling = 2200`
  - `ramp_steps = 180`
  - `seed = 20260314`
  - `ADAPTIVE_STABILITY = True`
  - End-of-run state: `dr = 8.75e-6 m`, `dt = 1.890431e-12 s`

### Artifacts

- Data: `results/test_runs/iv_kakati_tuned_0p15mm_20260311_200007.csv`
- Plot: `results/test_runs/iv_kakati_tuned_0p15mm_20260311_200007.png`

### Inline comparison plot

![Kakati 2017 comparison (0.15 mm probe)](results/test_runs/iv_kakati_tuned_0p15mm_20260311_200007.png)

### Result summary

- RMSE over 12 points: `0.904 mA`
- `0 V`: simulation `4.296 mA`, experiment `~4.0 mA`
- `+80 V`: simulation `12.453 mA`, experiment `~13.5 mA`

Note:
- Experimental values are approximate image-read points from Figure 1 (no raw
  table in the paper).
- `Te` and `N0` were treated as fitting knobs for this curve-matching pass.
