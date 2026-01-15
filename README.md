# PICSIMU: 1D Cylindrical PIC-MCC (Langmuir Probe)

This project builds a 1D radial (cylindrical) Particle-in-Cell simulation with
Monte Carlo Collisions (MCC) aimed at high-pressure Langmuir probe I–V curves.
The physics core is optimized with `numba.jit(nopython=True)` for particle
movers, charge weighting, field solves, and collisions. A Streamlit frontend
provides interactive control and visualization.

All distances are in meters and all velocities in m/s unless explicitly stated.
Temperatures are specified in eV in the configuration.

## Architecture overview

The project is split into two top-level packages:

- `core/`: High-performance physics engine. All heavy loops are JIT-compiled.
- `frontend/`: Streamlit UI that configures, runs, and visualizes the simulation.

Core modules are designed so the frontend can treat them as a pure API:
configure parameters, run a simulation, and pull arrays for plotting.

## Project structure

```
PICSIMU/
  core/
    config.py         # Constants + parameter container + physics helpers
    particles.py      # Particle mover + charge weighting (cylindrical)
    fields.py         # Poisson solver (cylindrical Laplacian) via TDMA
    collisions.py     # Ion-neutral CEX MCC
    simulation.py     # Main PIC loop: inject->push->collide->weight->solve
  frontend/
    app.py            # Streamlit UI + plotting
  README.md
  agent.md
```

## Physics model summary

Geometry
- 1D radial domain: `r ∈ [R_MIN, R_MAX]` with a probe at `R_MIN` and chamber wall
  at `R_MAX`.
- The grid is cell-centered or node-centered depending on solver implementation,
  but volume factors always use cylindrical shell volumes.

Particles
- Particles track `(r, v_r, v_theta)`; `v_theta` ensures angular momentum
  conservation.
- Radial equation of motion (ions and electrons share form):
  `dv_r/dt = (q/m) E_r + v_theta^2 / r`.
- Angular momentum conservation:
  `v_theta_new = v_theta_old * (r_old / r_new)`.

Weighting
- Cloud-in-Cell (linear) weighting to grid.
- Cylindrical volume correction:
  `V_j ≈ 2 * pi * r_j * dr` (per unit length).
  Charge density on node `j` is the weighted charge divided by `V_j`.

Field solve
- Poisson equation:
  `(1/r) d/dr (r dphi/dr) = -rho / epsilon_0`.
- Finite difference yields a tridiagonal system solved by TDMA in `O(N)`.
- Dirichlet boundaries:
  `phi(R_MIN) = V_bias`, `phi(R_MAX) = 0`.

Collisions
- Ion-neutral charge exchange (CEX) in high-pressure regime.
- Probability per step:
  `P = 1 - exp(-n_g * sigma * v * dt)`.
- If collision occurs, replace ion velocity with Maxwellian neutral sample
  at `T_gas ≈ 0.026 eV`.
- Electron-neutral collisions include elastic, excitation, and ionization with
  constant cross sections and energy thresholds; excitation/ionization apply
  energy loss only (no secondary particle creation).

Injection and currents
- Particles that hit the probe are absorbed and contribute to probe current.
- Particles reaching the wall are absorbed or reflected (policy defined in
  `particles.py`).
- Injection at `R_MAX` uses a Maxwellian half-flux estimate to set the per-step
  injection count, capped by available dead particles. The injected normal
  velocity follows the flux distribution (Rayleigh), not a half-Gaussian.
- Current sign convention:
  `I_electron = (N_e_hit * |q_e|) / dt`, `I_ion = (N_i_hit * q_i) / dt`,
  `I_total = I_electron - I_ion` (electron current reported as positive).

Initialization
- Particles are initialized with a Child-Langmuir-shaped potential profile to
  seed a sheath-like density gradient (electron Boltzmann response, ion
  continuity-based depletion) and reduce burn-in time.

## Data model (core)

All per-species data are stored in flat NumPy arrays to maximize Numba speed:

- `r`: radial position array (size Np)
- `vr`: radial velocity array
- `vt`: tangential velocity array

Fields and grid arrays are 1D:

- `r_grid`: node locations
- `phi`: electrostatic potential
- `E`: radial electric field
- `rho`: charge density

All arrays passed into `numba.jit(nopython=True)` functions are explicitly
typed via NumPy dtypes at construction time.

## Performance rules

- All heavy loops (push, weight, solve, collisions) are implemented in Numba.
- Avoid Python allocations inside jitted functions.
- Use fixed-size temporary arrays where possible.
- Keep branching minimal inside tight loops.

## Simulation flow (per time step)

1. Inject new particles at `R_MAX` (maintain density).
2. Push particles under `E_r` and centrifugal term.
3. Apply MCC (ion-neutral CEX).
4. Scatter charge to grid using cylindrical CIC.
5. Solve Poisson for `phi`, compute `E_r`.
6. Accumulate probe current from absorbed particles.

The simulation runs until a steady-state current is reached; the reported
current is typically an average over the late-time window.

## Frontend behavior

The UI exposes sliders for:
- Pressure (Torr)
- Density (m^-3)
- Electron temperature (eV)
- Probe bias voltage (V)
