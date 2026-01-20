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
- `results/`: benchmark outputs and example curves
- `README.md`: full technical / AI-facing documentation

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
- Collisions include ion-neutral CEX and simplified electron-neutral losses (no secondaries)
- Probe and wall are absorbing; outer boundary injects particles to maintain density

## Key limitations

- 1D radial model only (no axial/azimuthal spatial variation)
- Simplified collision models and no secondary particle creation
- Electrostatic approximation (no magnetic fields)

For full technical details, see `README.md`.
