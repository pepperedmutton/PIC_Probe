# Agent Guide: PICSIMU

This document is for automation agents and other bots interacting with this
repository. It describes the architecture, coding rules, and the expected
workflow when extending the simulation.

## Purpose

Build a 1D radial (cylindrical) Particle-in-Cell (PIC) simulation with
Monte Carlo Collisions (MCC) capable of reproducing Langmuir probe I–V curves
in high-pressure regimes. The project prioritizes physical correctness and
performance.

## Required tech stack

- Python 3.10+
- NumPy
- Numba (all heavy loops must be `nopython=True`)
- Streamlit (frontend)
- Matplotlib (plots)

## Architectural constraints

1. Geometry is 1D radial cylindrical, domain `[R_MIN, R_MAX]`.
2. Particles track `(r, v_r, v_theta)`; `v_theta` is required.
3. The radial equation of motion includes a centrifugal term:
   `dv_r/dt = (q/m) E_r + v_theta^2 / r`.
4. Angular momentum conservation must be enforced:
   `v_theta_new = v_theta_old * (r_old / r_new)`.
5. Charge density weighting must account for cylindrical shell volume:
   `V_j ≈ 2 * pi * r_j * dr` (per unit length).
6. The Poisson solver must use the cylindrical Laplacian and TDMA.
7. Monte Carlo collisions are ion-neutral charge exchange (CEX).
8. All heavy loops in the physics core must be Numba-jitted.

## Module responsibilities

`core/config.py`
- Holds constants and simulation parameters.
- Provides helper methods for Debye length and plasma frequency.

`core/particles.py`
- Particle push (mover) with radial + angular physics.
- Boundary conditions (probe absorption, wall handling).
- Charge weighting (CIC) with cylindrical volume correction.

`core/fields.py`
- Discretize and solve cylindrical Poisson equation.
- TDMA tridiagonal solve.
- Apply Dirichlet boundary conditions.

`core/collisions.py`
- Ion-neutral CEX MCC.
- Sample neutral thermal velocities.

`core/simulation.py`
- Main PIC loop and orchestration.
- Particle injection at `R_MAX`.
- Current accumulation at probe.

`frontend/app.py`
- Streamlit UI and plotting.

## Data layout expectations

Use flat NumPy arrays for particle data:

- `r`, `vr`, `vt` as 1D arrays (float64).
- Arrays are passed into JIT functions without Python objects.

Grid arrays are 1D float64:
- `r_grid`, `phi`, `E`, `rho`.

Avoid lists or Python objects inside JIT regions.

## Physics validation checks

When adding features, ensure:
- `dt` resolves `omega_pe` and electron motion.
- `dr` resolves `lambda_D`.
- Probe boundary removes particles and increments current.
- Charge density weighting correctly normalizes by cylindrical shell volume.
- Electric field derived from potential uses consistent radial discretization.

## Behavior expectations

When implementing or modifying:
- Keep all tight loops in `@numba.jit(nopython=True)` functions.
- Avoid allocations inside per-step loops.
- Use deterministic random streams if needed for testing.

## UI behavior

The frontend should expose:
- Pressure (Torr)
- Density (m^-3)
- Electron temperature (eV)
- Probe bias (V)

The frontend should display:
- Potential vs radius
- Electron/ion density vs radius
- Ion phase space (v_r vs r)

## Extensibility notes

Future additions may include:
- Electron collisions or elastic scattering.
- Multi-species ions.
- Diagnostics (energy, sheath width, etc).

Keep these in mind when naming and structuring interfaces.
