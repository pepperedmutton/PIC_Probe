from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from numba import jit, prange

from core.rng import counter_normal, counter_u64, counter_uniform, derive_seed


_ENERGY_BOUNDARY_RTOL = 64.0 * np.finfo(np.float64).eps


@jit(nopython=True, inline="always")
def _energy_exceeds_table(energy_ev: float, table_max_energy: float) -> bool:
    tolerance = _ENERGY_BOUNDARY_RTOL * max(1.0, abs(table_max_energy))
    return energy_ev > table_max_energy + tolerance


@jit(nopython=True)
def sigma_from_uniform_table(
    energy_ev: float,
    sigma_table: np.ndarray,
    e_min: float,
    inv_de: float,
) -> float:
    n = sigma_table.shape[0]
    if n == 0:
        return 0.0
    if energy_ev <= e_min:
        return sigma_table[0]
    idx = int((energy_ev - e_min) * inv_de)
    if idx >= n - 1:
        return sigma_table[n - 1]
    frac = (energy_ev - e_min) * inv_de - idx
    return sigma_table[idx] + frac * (sigma_table[idx + 1] - sigma_table[idx])


@jit(nopython=True, parallel=True)
def perform_mcc_ion(
    r: np.ndarray,
    vr: np.ndarray,
    vt: np.ndarray,
    r_min: float,
    r_max: float,
    n_g: float,
    sigma_cex: np.ndarray,
    sigma_el: np.ndarray,
    dt: float,
    v_th: float,
    e_min: float,
    inv_de: float,
    e_charge: float,
    m_i: float,
) -> int:
    """Calculate ion-neutral CEX and elastic collisions."""
    n_collisions = 0
    for i in prange(r.shape[0]):
        ri = r[i]
        if ri <= r_min or ri >= r_max:
            continue
        vi = math.sqrt(vr[i] * vr[i] + vt[i] * vt[i])
        if vi <= 0.0:
            continue
        energy_ev = 0.5 * m_i * vi * vi / e_charge
        sigma_cex_val = sigma_from_uniform_table(energy_ev, sigma_cex, e_min, inv_de)
        sigma_el_val = sigma_from_uniform_table(energy_ev, sigma_el, e_min, inv_de)
        sigma_total = sigma_cex_val + sigma_el_val
        if sigma_total <= 0.0:
            continue
        p = 1.0 - math.exp(-n_g * sigma_total * vi * dt)
        if np.random.random() < p:
            pick = np.random.random() * sigma_total
            if pick < sigma_cex_val:
                vr[i] = np.random.normal(0.0, v_th)
                vt[i] = np.random.normal(0.0, v_th)
            else:
                angle = 2.0 * math.pi * np.random.random()
                vr[i] = vi * math.cos(angle)
                vt[i] = vi * math.sin(angle)
            n_collisions += 1
    return n_collisions


@jit(nopython=True, parallel=True)
def perform_mcc_electron(
    r: np.ndarray,
    vr: np.ndarray,
    vt: np.ndarray,
    r_min: float,
    r_max: float,
    n_g: float,
    sigma_el: np.ndarray,
    sigma_exc: np.ndarray,
    sigma_ion: np.ndarray,
    dt: float,
    m_e: float,
    e_exc: float,
    e_ion: float,
    e_min: float,
    inv_de: float,
    e_charge: float,
    ionized: np.ndarray,
    sec_energy_ev: np.ndarray,
) -> tuple[int, int, int]:
    """Calculate electron-neutral elastic, excitation, and ionization collisions."""
    n_el = 0
    n_exc = 0
    n_ion = 0

    for i in prange(r.shape[0]):
        ionized[i] = 0
        sec_energy_ev[i] = 0.0
        ri = r[i]
        if ri <= r_min or ri >= r_max:
            continue

        v2 = vr[i] * vr[i] + vt[i] * vt[i]
        if v2 <= 0.0:
            continue
        v = math.sqrt(v2)
        energy = 0.5 * m_e * v2
        energy_ev = energy / e_charge

        sigma_el_val = sigma_from_uniform_table(
            energy_ev,
            sigma_el,
            e_min,
            inv_de,
        )
        sigma_exc_val = (
            sigma_from_uniform_table(energy_ev, sigma_exc, e_min, inv_de)
            if energy > e_exc
            else 0.0
        )
        sigma_ion_val = (
            sigma_from_uniform_table(energy_ev, sigma_ion, e_min, inv_de)
            if energy > e_ion
            else 0.0
        )
        sigma_total = sigma_el_val + sigma_exc_val + sigma_ion_val
        if sigma_total <= 0.0:
            continue

        p = 1.0 - math.exp(-n_g * sigma_total * v * dt)
        if np.random.random() >= p:
            continue

        pick = np.random.random() * sigma_total
        angle = 2.0 * math.pi * np.random.random()
        if pick < sigma_el_val:
            vr[i] = v * math.cos(angle)
            vt[i] = v * math.sin(angle)
            n_el += 1
        elif pick < sigma_el_val + sigma_exc_val:
            energy_new = energy - e_exc
            if energy_new < 0.0:
                energy_new = 0.0
            v_new = math.sqrt(2.0 * energy_new / m_e) if energy_new > 0.0 else 0.0
            vr[i] = v_new * math.cos(angle)
            vt[i] = v_new * math.sin(angle)
            n_exc += 1
        else:
            energy_new = energy - e_ion
            if energy_new < 0.0:
                energy_new = 0.0
            # Split remaining energy between primary and secondary electron.
            energy_share = 0.5 * energy_new
            v_new = math.sqrt(2.0 * energy_share / m_e) if energy_share > 0.0 else 0.0
            vr[i] = v_new * math.cos(angle)
            vt[i] = v_new * math.sin(angle)
            ionized[i] = 1
            sec_energy_ev[i] = energy_share / e_charge if energy_share > 0.0 else 0.0
            n_ion += 1

    return n_el, n_exc, n_ion


@jit(nopython=True, parallel=True)
def perform_coulomb_scatter(
    r: np.ndarray,
    vr: np.ndarray,
    vt: np.ndarray,
    r_min: float,
    r_max: float,
    nu: float,
    dt: float,
) -> int:
    """Calculate pitch-angle scattering for Coulomb collisions."""
    if nu <= 0.0:
        return 0
    n_events = 0
    p = 1.0 - math.exp(-nu * dt)
    for i in prange(r.shape[0]):
        ri = r[i]
        if ri <= r_min or ri >= r_max:
            continue
        if np.random.random() < p:
            v2 = vr[i] * vr[i] + vt[i] * vt[i]
            if v2 <= 0.0:
                continue
            v = math.sqrt(v2)
            angle = 2.0 * math.pi * np.random.random()
            vr[i] = v * math.cos(angle)
            vt[i] = v * math.sin(angle)
            n_events += 1
    return n_events


_ELECTRON_SEED_COMPONENT = 101
_ION_SEED_COMPONENT = 102
_PARTICLE_STREAM = 1
_WAIT_STREAM = 2
_BRANCH_STREAM = 3
_ENERGY_SHARE_STREAM = 4
_DIRECTION_MU_STREAM = 5
_DIRECTION_AZIMUTH_STREAM = 6
_NEUTRAL_X_STREAM = 7
_NEUTRAL_Y_STREAM = 8
_NEUTRAL_Z_STREAM = 9
_ION_PROCESS_STREAM = 10
_MIN_UNIFORM = 1.0 / float(1 << 53)
_MAJORANT_MARGIN = 1.0 + 1.0e-12


@dataclass
class SecondaryElectronEventBuffer:
    parent_index: np.ndarray
    event_time_s: np.ndarray
    vx: np.ndarray
    vy: np.ndarray
    vz: np.ndarray
    energy_ev: np.ndarray
    count: int = 0

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        arrays = (
            self.parent_index,
            self.event_time_s,
            self.vx,
            self.vy,
            self.vz,
            self.energy_ev,
        )
        if any(array.ndim != 1 for array in arrays):
            raise ValueError("Event buffer arrays are not one-dimensional.")
        if any(array.shape != arrays[0].shape for array in arrays[1:]):
            raise ValueError("Event buffer arrays have different shapes.")
        if self.parent_index.dtype != np.dtype(np.int64):
            raise ValueError("The parent-index array does not have type int64.")
        if any(
            array.dtype != np.dtype(np.float64)
            for array in arrays[1:]
        ):
            raise ValueError("Event data arrays do not have type float64.")
        if isinstance(self.count, bool) or not isinstance(
            self.count,
            (int, np.integer),
        ):
            raise ValueError("The event buffer count is not an integer.")
        if self.count < 0 or self.count > self.capacity:
            raise ValueError("The event buffer count is outside its capacity.")

    @property
    def capacity(self) -> int:
        return int(self.parent_index.size)

    def record(
        self,
        parent_index: int,
        event_time_s: float,
        vx: float,
        vy: float,
        vz: float,
        energy_ev: float,
    ) -> bool:
        if self.count >= self.capacity:
            return False
        index = self.count
        self.parent_index[index] = parent_index
        self.event_time_s[index] = event_time_s
        self.vx[index] = vx
        self.vy[index] = vy
        self.vz[index] = vz
        self.energy_ev[index] = energy_ev
        self.count += 1
        return True


@dataclass(frozen=True)
class ElectronCollisionResult:
    elastic_events: int
    excitation_events: int
    ionization_events: int
    secondary_events_written: int
    buffer_full_stops: int
    event_limit_stops: int
    energy_table_overflow_lookups: int
    dead_particles_skipped: int
    excitation_channel_events: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.int64),
        compare=False,
    )
    ionization_channel_events: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.int64),
        compare=False,
    )

    @property
    def total_events(self) -> int:
        return self.elastic_events + self.excitation_events + self.ionization_events

    @property
    def buffer_full_count(self) -> int:
        return self.buffer_full_stops

    @property
    def secondary_events_dropped(self) -> int:
        return self.buffer_full_stops

    @property
    def event_limit_count(self) -> int:
        return self.event_limit_stops

    @property
    def energy_table_overflow_count(self) -> int:
        return self.energy_table_overflow_lookups


@dataclass(frozen=True)
class IonCollisionResult:
    charge_exchange_events: int
    elastic_events: int
    event_limit_stops: int
    energy_table_overflow_lookups: int
    dead_particles_skipped: int
    candidate_events: int = 0
    null_collision_rejections: int = 0
    candidate_limit_stops: int = 0

    @property
    def total_events(self) -> int:
        return self.charge_exchange_events + self.elastic_events

    @property
    def event_limit_count(self) -> int:
        return self.event_limit_stops

    @property
    def energy_table_overflow_count(self) -> int:
        return self.energy_table_overflow_lookups

    @property
    def candidate_limit_count(self) -> int:
        return self.candidate_limit_stops


@jit(nopython=True, inline="always")
def _draw_uniform(
    seed: int,
    counter: int,
    stream: int,
) -> tuple[float, int]:
    value = counter_uniform(seed, counter, stream)
    return value, counter + 1


@jit(nopython=True, inline="always")
def _draw_normal(
    seed: int,
    counter: int,
    stream: int,
) -> tuple[float, int]:
    value = counter_normal(seed, counter, stream)
    return value, counter + 1


@jit(nopython=True, inline="always")
def _sample_wait(
    rate: float,
    seed: int,
    counter: int,
) -> tuple[float, int]:
    uniform, counter = _draw_uniform(seed, counter, _WAIT_STREAM)
    if uniform < _MIN_UNIFORM:
        uniform = _MIN_UNIFORM
    return -math.log(uniform) / rate, counter


@jit(nopython=True, inline="always")
def _isotropic_velocity(
    speed: float,
    seed: int,
    counter: int,
) -> tuple[float, float, float, int]:
    mu_value, counter = _draw_uniform(seed, counter, _DIRECTION_MU_STREAM)
    azimuth_value, counter = _draw_uniform(
        seed,
        counter,
        _DIRECTION_AZIMUTH_STREAM,
    )
    mu = 2.0 * mu_value - 1.0
    azimuth = 2.0 * math.pi * azimuth_value
    transverse = math.sqrt(max(0.0, 1.0 - mu * mu))
    return (
        speed * transverse * math.cos(azimuth),
        speed * transverse * math.sin(azimuth),
        speed * mu,
        counter,
    )


@jit(nopython=True, inline="always")
def _thermal_velocity(
    thermal_speed: float,
    seed: int,
    counter: int,
) -> tuple[float, float, float, int]:
    normal_x, counter = _draw_normal(seed, counter, _NEUTRAL_X_STREAM)
    normal_y, counter = _draw_normal(seed, counter, _NEUTRAL_Y_STREAM)
    normal_z, counter = _draw_normal(seed, counter, _NEUTRAL_Z_STREAM)
    return (
        thermal_speed * normal_x,
        thermal_speed * normal_y,
        thermal_speed * normal_z,
        counter,
    )


@jit(nopython=True, inline="always")
def _equal_mass_elastic_components(
    ion_x: float,
    ion_y: float,
    ion_z: float,
    neutral_x: float,
    neutral_y: float,
    neutral_z: float,
    mu: float,
    azimuth: float,
) -> tuple[float, float, float, float, float, float]:
    center_x = 0.5 * (ion_x + neutral_x)
    center_y = 0.5 * (ion_y + neutral_y)
    center_z = 0.5 * (ion_z + neutral_z)
    relative_x = ion_x - neutral_x
    relative_y = ion_y - neutral_y
    relative_z = ion_z - neutral_z
    relative_speed = math.sqrt(
        relative_x * relative_x
        + relative_y * relative_y
        + relative_z * relative_z
    )
    transverse = math.sqrt(max(0.0, 1.0 - mu * mu))
    scattered_x = relative_speed * transverse * math.cos(azimuth)
    scattered_y = relative_speed * transverse * math.sin(azimuth)
    scattered_z = relative_speed * mu
    ion_out_x = center_x + 0.5 * scattered_x
    ion_out_y = center_y + 0.5 * scattered_y
    ion_out_z = center_z + 0.5 * scattered_z
    neutral_out_x = center_x - 0.5 * scattered_x
    neutral_out_y = center_y - 0.5 * scattered_y
    neutral_out_z = center_z - 0.5 * scattered_z
    return (
        ion_out_x,
        ion_out_y,
        ion_out_z,
        neutral_out_x,
        neutral_out_y,
        neutral_out_z,
    )


def equal_mass_elastic_collision_3d(
    ion_velocity: np.ndarray,
    neutral_velocity: np.ndarray,
    mu: float,
    azimuth: float,
) -> tuple[np.ndarray, np.ndarray]:
    ion = np.asarray(ion_velocity, dtype=float)
    neutral = np.asarray(neutral_velocity, dtype=float)
    if ion.shape != (3,) or neutral.shape != (3,):
        raise ValueError("Each velocity input must have three components.")
    if not np.all(np.isfinite(ion)) or not np.all(np.isfinite(neutral)):
        raise ValueError("A velocity input has a nonfinite value.")
    if not math.isfinite(mu) or mu < -1.0 or mu > 1.0:
        raise ValueError("The direction cosine is outside the range -1 through 1.")
    if not math.isfinite(azimuth):
        raise ValueError("The azimuth has a nonfinite value.")
    values = _equal_mass_elastic_components(
        float(ion[0]),
        float(ion[1]),
        float(ion[2]),
        float(neutral[0]),
        float(neutral[1]),
        float(neutral[2]),
        mu,
        azimuth,
    )
    return np.asarray(values[:3]), np.asarray(values[3:])


def _validate_particle_arrays(
    vx: np.ndarray,
    vy: np.ndarray,
    vz: np.ndarray,
    alive: np.ndarray,
) -> None:
    arrays = (vx, vy, vz, alive)
    if any(array.ndim != 1 for array in arrays):
        raise ValueError("Particle arrays are not one-dimensional.")
    if any(array.shape != vx.shape for array in arrays[1:]):
        raise ValueError("Particle arrays have different shapes.")
    if any(array.dtype != np.dtype(np.float64) for array in (vx, vy, vz)):
        raise ValueError("Velocity arrays do not have type float64.")
    if alive.dtype != np.dtype(np.bool_):
        raise ValueError("The alive array does not have a Boolean type.")


def _validate_sigma_tables(*tables: np.ndarray) -> None:
    for table in tables:
        if table.ndim != 1 or table.size == 0:
            raise ValueError("A cross-section table has an invalid shape.")
        if table.dtype != np.dtype(np.float64):
            raise ValueError("A cross-section table does not have type float64.")
        if not np.all(np.isfinite(table)):
            raise ValueError("A cross-section table has a nonfinite value.")
        if np.any(table < 0.0):
            raise ValueError("A cross-section table has a negative value.")


def _validate_channel_tables(
    channel_tables: np.ndarray,
    thresholds_ev: np.ndarray,
    table_size: int,
    process_name: str,
) -> None:
    if (
        channel_tables.ndim != 2
        or channel_tables.shape[1] != table_size
    ):
        raise ValueError(
            f"The {process_name} channel tables have an invalid shape."
        )
    if channel_tables.dtype != np.dtype(np.float64):
        raise ValueError(
            f"The {process_name} channel tables do not have type float64."
        )
    if not np.all(np.isfinite(channel_tables)):
        raise ValueError(
            f"The {process_name} channel tables have a nonfinite value."
        )
    if np.any(channel_tables < 0.0):
        raise ValueError(
            f"The {process_name} channel tables have a negative value."
        )
    if thresholds_ev.ndim != 1 or thresholds_ev.size != channel_tables.shape[0]:
        raise ValueError(
            f"The {process_name} threshold array has an invalid shape."
        )
    if thresholds_ev.dtype != np.dtype(np.float64):
        raise ValueError(
            f"The {process_name} threshold array does not have type float64."
        )
    if not np.all(np.isfinite(thresholds_ev)):
        raise ValueError(
            f"The {process_name} threshold array has a nonfinite value."
        )
    if np.any(thresholds_ev < 0.0):
        raise ValueError(
            f"The {process_name} threshold array has a negative value."
        )


def _validate_reference_values(
    n_g: float,
    dt: float,
    mass: float,
    e_charge: float,
    max_events_per_particle: int,
) -> None:
    if not math.isfinite(n_g) or n_g < 0.0:
        raise ValueError("The neutral density is invalid.")
    if not math.isfinite(dt) or dt < 0.0:
        raise ValueError("The time step is invalid.")
    if not math.isfinite(mass) or mass <= 0.0:
        raise ValueError("The particle mass must be positive.")
    if not math.isfinite(e_charge) or e_charge <= 0.0:
        raise ValueError("The elementary charge must be positive.")
    if (
        isinstance(max_events_per_particle, bool)
        or not isinstance(max_events_per_particle, int)
        or max_events_per_particle < 0
    ):
        raise ValueError("The event limit must be a nonnegative integer.")


def _validate_particle_durations(
    particle_duration_s: np.ndarray,
    particle_shape: tuple[int, ...],
) -> np.ndarray:
    durations = np.asarray(particle_duration_s, dtype=np.float64)
    if durations.ndim != 1 or durations.shape != particle_shape:
        raise ValueError("The particle-duration array has an invalid shape.")
    if not np.all(np.isfinite(durations)) or np.any(durations < 0.0):
        raise ValueError("The particle-duration array has an invalid value.")
    return durations


@jit(nopython=True)
def _perform_mcc_electron_1d3v_core(
    vx: np.ndarray,
    vy: np.ndarray,
    vz: np.ndarray,
    alive: np.ndarray,
    n_g: float,
    sigma_el: np.ndarray,
    sigma_exc: np.ndarray,
    sigma_ion: np.ndarray,
    dt: float,
    m_e: float,
    excitation_threshold_ev: float,
    ionization_threshold_ev: float,
    e_min: float,
    inv_de: float,
    e_charge: float,
    buffer_parent_index: np.ndarray,
    buffer_event_time_s: np.ndarray,
    buffer_vx: np.ndarray,
    buffer_vy: np.ndarray,
    buffer_vz: np.ndarray,
    buffer_energy_ev: np.ndarray,
    buffer_count: int,
    base_seed: int,
    max_events_per_particle: int,
) -> tuple[int, int, int, int, int, int, int, int, int]:
    elastic_events = 0
    excitation_events = 0
    ionization_events = 0
    secondary_events_written = 0
    buffer_full_stops = 0
    event_limit_stops = 0
    energy_table_overflow_lookups = 0
    dead_particles_skipped = 0
    table_size = min(sigma_el.size, sigma_exc.size, sigma_ion.size)
    if inv_de > 0.0:
        table_max_energy = e_min + (table_size - 1) / inv_de
    else:
        table_max_energy = math.inf

    for particle_index in range(vx.size):
        if not alive[particle_index]:
            dead_particles_skipped += 1
            continue

        particle_seed = counter_u64(
            base_seed,
            particle_index,
            _PARTICLE_STREAM,
        )
        draw_counter = 0
        elapsed = 0.0
        particle_events = 0
        while elapsed < dt:
            velocity_squared = (
                vx[particle_index] * vx[particle_index]
                + vy[particle_index] * vy[particle_index]
                + vz[particle_index] * vz[particle_index]
            )
            if velocity_squared <= 0.0:
                break
            speed = math.sqrt(velocity_squared)
            energy_ev = 0.5 * m_e * velocity_squared / e_charge
            if _energy_exceeds_table(energy_ev, table_max_energy):
                energy_table_overflow_lookups += 1
                break
            sigma_elastic = sigma_from_uniform_table(
                energy_ev,
                sigma_el,
                e_min,
                inv_de,
            )
            sigma_excitation = 0.0
            if energy_ev >= excitation_threshold_ev:
                sigma_excitation = sigma_from_uniform_table(
                    energy_ev,
                    sigma_exc,
                    e_min,
                    inv_de,
                )
            sigma_ionization = 0.0
            if energy_ev >= ionization_threshold_ev:
                sigma_ionization = sigma_from_uniform_table(
                    energy_ev,
                    sigma_ion,
                    e_min,
                    inv_de,
                )
            sigma_total = sigma_elastic + sigma_excitation + sigma_ionization
            rate = n_g * sigma_total * speed
            if rate <= 0.0:
                break

            wait, draw_counter = _sample_wait(
                rate,
                particle_seed,
                draw_counter,
            )
            if wait >= dt - elapsed:
                break
            if particle_events >= max_events_per_particle:
                event_limit_stops += 1
                break
            elapsed += wait

            branch_value, draw_counter = _draw_uniform(
                particle_seed,
                draw_counter,
                _BRANCH_STREAM,
            )
            pick = branch_value * sigma_total
            if pick < sigma_elastic:
                (
                    vx[particle_index],
                    vy[particle_index],
                    vz[particle_index],
                    draw_counter,
                ) = _isotropic_velocity(
                    speed,
                    particle_seed,
                    draw_counter,
                )
                elastic_events += 1
            elif pick < sigma_elastic + sigma_excitation:
                new_energy_ev = energy_ev - excitation_threshold_ev
                new_speed = math.sqrt(2.0 * new_energy_ev * e_charge / m_e)
                (
                    vx[particle_index],
                    vy[particle_index],
                    vz[particle_index],
                    draw_counter,
                ) = _isotropic_velocity(
                    new_speed,
                    particle_seed,
                    draw_counter,
                )
                excitation_events += 1
            else:
                if buffer_count >= buffer_parent_index.size:
                    buffer_full_stops += 1
                    break
                available_energy_ev = energy_ev - ionization_threshold_ev
                secondary_fraction, draw_counter = _draw_uniform(
                    particle_seed,
                    draw_counter,
                    _ENERGY_SHARE_STREAM,
                )
                secondary_energy_ev = available_energy_ev * secondary_fraction
                primary_energy_ev = available_energy_ev - secondary_energy_ev
                primary_speed = math.sqrt(
                    2.0 * primary_energy_ev * e_charge / m_e
                )
                secondary_speed = math.sqrt(
                    2.0 * secondary_energy_ev * e_charge / m_e
                )
                (
                    primary_x,
                    primary_y,
                    primary_z,
                    draw_counter,
                ) = _isotropic_velocity(
                    primary_speed,
                    particle_seed,
                    draw_counter,
                )
                (
                    secondary_x,
                    secondary_y,
                    secondary_z,
                    draw_counter,
                ) = _isotropic_velocity(
                    secondary_speed,
                    particle_seed,
                    draw_counter,
                )
                buffer_parent_index[buffer_count] = particle_index
                buffer_event_time_s[buffer_count] = elapsed
                buffer_vx[buffer_count] = secondary_x
                buffer_vy[buffer_count] = secondary_y
                buffer_vz[buffer_count] = secondary_z
                buffer_energy_ev[buffer_count] = secondary_energy_ev
                buffer_count += 1
                vx[particle_index] = primary_x
                vy[particle_index] = primary_y
                vz[particle_index] = primary_z
                secondary_events_written += 1
                ionization_events += 1
            particle_events += 1

    return (
        elastic_events,
        excitation_events,
        ionization_events,
        secondary_events_written,
        buffer_full_stops,
        event_limit_stops,
        energy_table_overflow_lookups,
        dead_particles_skipped,
        buffer_count,
    )


@jit(nopython=True)
def _perform_mcc_electron_channels_1d3v_core(
    vx: np.ndarray,
    vy: np.ndarray,
    vz: np.ndarray,
    alive: np.ndarray,
    n_g: float,
    sigma_el: np.ndarray,
    excitation_channel_tables: np.ndarray,
    excitation_thresholds_ev: np.ndarray,
    ionization_channel_tables: np.ndarray,
    ionization_thresholds_ev: np.ndarray,
    particle_duration_s: np.ndarray,
    m_e: float,
    e_min: float,
    inv_de: float,
    e_charge: float,
    buffer_parent_index: np.ndarray,
    buffer_event_time_s: np.ndarray,
    buffer_vx: np.ndarray,
    buffer_vy: np.ndarray,
    buffer_vz: np.ndarray,
    buffer_energy_ev: np.ndarray,
    buffer_count: int,
    excitation_channel_events: np.ndarray,
    ionization_channel_events: np.ndarray,
    base_seed: int,
    max_events_per_particle: int,
) -> tuple[int, int, int, int, int, int, int, int, int]:
    elastic_events = 0
    excitation_events = 0
    ionization_events = 0
    secondary_events_written = 0
    buffer_full_stops = 0
    event_limit_stops = 0
    energy_table_overflow_lookups = 0
    dead_particles_skipped = 0
    if inv_de > 0.0:
        table_max_energy = e_min + (sigma_el.size - 1) / inv_de
    else:
        table_max_energy = math.inf

    for particle_index in range(vx.size):
        if not alive[particle_index]:
            dead_particles_skipped += 1
            continue

        duration = particle_duration_s[particle_index]
        if duration <= 0.0:
            continue

        particle_seed = counter_u64(
            base_seed,
            particle_index,
            _PARTICLE_STREAM,
        )
        draw_counter = 0
        elapsed = 0.0
        particle_events = 0
        while elapsed < duration:
            velocity_squared = (
                vx[particle_index] * vx[particle_index]
                + vy[particle_index] * vy[particle_index]
                + vz[particle_index] * vz[particle_index]
            )
            if velocity_squared <= 0.0:
                break
            speed = math.sqrt(velocity_squared)
            energy_ev = 0.5 * m_e * velocity_squared / e_charge
            if _energy_exceeds_table(energy_ev, table_max_energy):
                energy_table_overflow_lookups += 1
                break

            sigma_elastic = sigma_from_uniform_table(
                energy_ev,
                sigma_el,
                e_min,
                inv_de,
            )
            sigma_total = sigma_elastic
            for channel_index in range(excitation_thresholds_ev.size):
                if energy_ev >= excitation_thresholds_ev[channel_index]:
                    sigma_total += sigma_from_uniform_table(
                        energy_ev,
                        excitation_channel_tables[channel_index],
                        e_min,
                        inv_de,
                    )
            for channel_index in range(ionization_thresholds_ev.size):
                if energy_ev >= ionization_thresholds_ev[channel_index]:
                    sigma_total += sigma_from_uniform_table(
                        energy_ev,
                        ionization_channel_tables[channel_index],
                        e_min,
                        inv_de,
                    )

            rate = n_g * sigma_total * speed
            if rate <= 0.0:
                break
            wait, draw_counter = _sample_wait(
                rate,
                particle_seed,
                draw_counter,
            )
            if wait >= duration - elapsed:
                break
            if particle_events >= max_events_per_particle:
                event_limit_stops += 1
                break
            elapsed += wait

            branch_value, draw_counter = _draw_uniform(
                particle_seed,
                draw_counter,
                _BRANCH_STREAM,
            )
            pick = branch_value * sigma_total
            cumulative_sigma = sigma_elastic
            if pick < cumulative_sigma:
                (
                    vx[particle_index],
                    vy[particle_index],
                    vz[particle_index],
                    draw_counter,
                ) = _isotropic_velocity(
                    speed,
                    particle_seed,
                    draw_counter,
                )
                elastic_events += 1
                particle_events += 1
                continue

            selected_excitation_channel = -1
            for channel_index in range(excitation_thresholds_ev.size):
                if energy_ev < excitation_thresholds_ev[channel_index]:
                    continue
                cumulative_sigma += sigma_from_uniform_table(
                    energy_ev,
                    excitation_channel_tables[channel_index],
                    e_min,
                    inv_de,
                )
                if pick < cumulative_sigma:
                    selected_excitation_channel = channel_index
                    break
            if selected_excitation_channel >= 0:
                new_energy_ev = (
                    energy_ev
                    - excitation_thresholds_ev[selected_excitation_channel]
                )
                new_speed = math.sqrt(
                    2.0 * new_energy_ev * e_charge / m_e
                )
                (
                    vx[particle_index],
                    vy[particle_index],
                    vz[particle_index],
                    draw_counter,
                ) = _isotropic_velocity(
                    new_speed,
                    particle_seed,
                    draw_counter,
                )
                excitation_events += 1
                excitation_channel_events[selected_excitation_channel] += 1
                particle_events += 1
                continue

            selected_ionization_channel = -1
            for channel_index in range(ionization_thresholds_ev.size):
                if energy_ev < ionization_thresholds_ev[channel_index]:
                    continue
                cumulative_sigma += sigma_from_uniform_table(
                    energy_ev,
                    ionization_channel_tables[channel_index],
                    e_min,
                    inv_de,
                )
                if pick < cumulative_sigma:
                    selected_ionization_channel = channel_index
                    break
            if selected_ionization_channel < 0:
                break
            if buffer_count >= buffer_parent_index.size:
                buffer_full_stops += 1
                break

            ionization_threshold_ev = ionization_thresholds_ev[
                selected_ionization_channel
            ]
            available_energy_ev = energy_ev - ionization_threshold_ev
            secondary_fraction, draw_counter = _draw_uniform(
                particle_seed,
                draw_counter,
                _ENERGY_SHARE_STREAM,
            )
            secondary_energy_ev = available_energy_ev * secondary_fraction
            primary_energy_ev = available_energy_ev - secondary_energy_ev
            primary_speed = math.sqrt(
                2.0 * primary_energy_ev * e_charge / m_e
            )
            secondary_speed = math.sqrt(
                2.0 * secondary_energy_ev * e_charge / m_e
            )
            (
                primary_x,
                primary_y,
                primary_z,
                draw_counter,
            ) = _isotropic_velocity(
                primary_speed,
                particle_seed,
                draw_counter,
            )
            (
                secondary_x,
                secondary_y,
                secondary_z,
                draw_counter,
            ) = _isotropic_velocity(
                secondary_speed,
                particle_seed,
                draw_counter,
            )
            buffer_parent_index[buffer_count] = particle_index
            buffer_event_time_s[buffer_count] = elapsed
            buffer_vx[buffer_count] = secondary_x
            buffer_vy[buffer_count] = secondary_y
            buffer_vz[buffer_count] = secondary_z
            buffer_energy_ev[buffer_count] = secondary_energy_ev
            buffer_count += 1
            vx[particle_index] = primary_x
            vy[particle_index] = primary_y
            vz[particle_index] = primary_z
            secondary_events_written += 1
            ionization_events += 1
            ionization_channel_events[selected_ionization_channel] += 1
            particle_events += 1

    return (
        elastic_events,
        excitation_events,
        ionization_events,
        secondary_events_written,
        buffer_full_stops,
        event_limit_stops,
        energy_table_overflow_lookups,
        dead_particles_skipped,
        buffer_count,
    )


@jit(nopython=True, inline="always")
def _sigma_speed_product(
    energy_ev: float,
    sigma_total: float,
    e_charge: float,
    mass: float,
) -> float:
    if energy_ev <= 0.0 or sigma_total <= 0.0:
        return 0.0
    speed = math.sqrt(2.0 * energy_ev * e_charge / mass)
    return sigma_total * speed


@jit(nopython=True)
def _ion_collision_majorant(
    sigma_cex: np.ndarray,
    sigma_el: np.ndarray,
    e_min: float,
    inv_de: float,
    e_charge: float,
    m_i: float,
) -> tuple[float, float]:
    table_size = min(sigma_cex.size, sigma_el.size)
    table_max_energy = e_min + (table_size - 1) / inv_de
    maximum = 0.0

    for index in range(table_size):
        energy = e_min + index / inv_de
        sigma_total = sigma_cex[index] + sigma_el[index]
        value = _sigma_speed_product(
            energy,
            sigma_total,
            e_charge,
            m_i,
        )
        if value > maximum:
            maximum = value

    for index in range(table_size - 1):
        energy_left = e_min + index / inv_de
        energy_right = e_min + (index + 1) / inv_de
        sigma_left = sigma_cex[index] + sigma_el[index]
        sigma_right = sigma_cex[index + 1] + sigma_el[index + 1]
        slope = (sigma_right - sigma_left) / (energy_right - energy_left)
        if slope >= 0.0:
            continue
        intercept = sigma_left - slope * energy_left
        critical_energy = -intercept / (3.0 * slope)
        if critical_energy <= energy_left or critical_energy >= energy_right:
            continue
        critical_sigma = intercept + slope * critical_energy
        value = _sigma_speed_product(
            critical_energy,
            critical_sigma,
            e_charge,
            m_i,
        )
        if value > maximum:
            maximum = value

    return maximum * _MAJORANT_MARGIN, table_max_energy


@jit(nopython=True)
def _perform_mcc_ion_1d3v_core(
    vx: np.ndarray,
    vy: np.ndarray,
    vz: np.ndarray,
    alive: np.ndarray,
    n_g: float,
    sigma_cex: np.ndarray,
    sigma_el: np.ndarray,
    particle_duration_s: np.ndarray,
    neutral_thermal_speed: float,
    e_min: float,
    inv_de: float,
    e_charge: float,
    m_i: float,
    base_seed: int,
    max_events_per_particle: int,
    max_candidates_per_particle: int,
) -> tuple[int, int, int, int, int, int, int, int]:
    charge_exchange_events = 0
    elastic_events = 0
    event_limit_stops = 0
    energy_table_overflow_lookups = 0
    dead_particles_skipped = 0
    candidate_events = 0
    null_collision_rejections = 0
    candidate_limit_stops = 0
    majorant, table_max_energy = _ion_collision_majorant(
        sigma_cex,
        sigma_el,
        e_min,
        inv_de,
        e_charge,
        m_i,
    )
    candidate_rate = n_g * majorant
    if candidate_rate <= 0.0:
        for particle_index in range(vx.size):
            if not alive[particle_index]:
                dead_particles_skipped += 1
        return (
            charge_exchange_events,
            elastic_events,
            event_limit_stops,
            energy_table_overflow_lookups,
            dead_particles_skipped,
            candidate_events,
            null_collision_rejections,
            candidate_limit_stops,
        )

    for particle_index in range(vx.size):
        if not alive[particle_index]:
            dead_particles_skipped += 1
            continue

        duration = particle_duration_s[particle_index]
        if duration <= 0.0:
            continue

        particle_seed = counter_u64(
            base_seed,
            particle_index,
            _PARTICLE_STREAM,
        )
        draw_counter = 0
        elapsed = 0.0
        particle_events = 0
        particle_candidates = 0
        while elapsed < duration:
            wait, draw_counter = _sample_wait(
                candidate_rate,
                particle_seed,
                draw_counter,
            )
            if wait >= duration - elapsed:
                break
            if particle_candidates >= max_candidates_per_particle:
                candidate_limit_stops += 1
                break
            elapsed += wait
            particle_candidates += 1
            candidate_events += 1

            (
                neutral_x,
                neutral_y,
                neutral_z,
                draw_counter,
            ) = _thermal_velocity(
                neutral_thermal_speed,
                particle_seed,
                draw_counter,
            )
            relative_x = vx[particle_index] - neutral_x
            relative_y = vy[particle_index] - neutral_y
            relative_z = vz[particle_index] - neutral_z
            relative_speed_squared = (
                relative_x * relative_x
                + relative_y * relative_y
                + relative_z * relative_z
            )
            if relative_speed_squared <= 0.0:
                null_collision_rejections += 1
                continue
            relative_speed = math.sqrt(relative_speed_squared)
            energy_ev = 0.5 * m_i * relative_speed_squared / e_charge
            if _energy_exceeds_table(energy_ev, table_max_energy):
                energy_table_overflow_lookups += 1
                break
            sigma_cex_value = sigma_from_uniform_table(
                energy_ev,
                sigma_cex,
                e_min,
                inv_de,
            )
            sigma_elastic = sigma_from_uniform_table(
                energy_ev,
                sigma_el,
                e_min,
                inv_de,
            )
            sigma_total = sigma_cex_value + sigma_elastic
            collision_weight = sigma_total * relative_speed
            acceptance_value, draw_counter = _draw_uniform(
                particle_seed,
                draw_counter,
                _BRANCH_STREAM,
            )
            if acceptance_value * majorant >= collision_weight:
                null_collision_rejections += 1
                continue
            if particle_events >= max_events_per_particle:
                event_limit_stops += 1
                break

            branch_value, draw_counter = _draw_uniform(
                particle_seed,
                draw_counter,
                _ION_PROCESS_STREAM,
            )
            pick = branch_value * sigma_total
            if pick < sigma_cex_value:
                vx[particle_index] = neutral_x
                vy[particle_index] = neutral_y
                vz[particle_index] = neutral_z
                charge_exchange_events += 1
            else:
                mu_value, draw_counter = _draw_uniform(
                    particle_seed,
                    draw_counter,
                    _DIRECTION_MU_STREAM,
                )
                azimuth_value, draw_counter = _draw_uniform(
                    particle_seed,
                    draw_counter,
                    _DIRECTION_AZIMUTH_STREAM,
                )
                mu = 2.0 * mu_value - 1.0
                azimuth = 2.0 * math.pi * azimuth_value
                values = _equal_mass_elastic_components(
                    vx[particle_index],
                    vy[particle_index],
                    vz[particle_index],
                    neutral_x,
                    neutral_y,
                    neutral_z,
                    mu,
                    azimuth,
                )
                vx[particle_index] = values[0]
                vy[particle_index] = values[1]
                vz[particle_index] = values[2]
                elastic_events += 1
            particle_events += 1

    return (
        charge_exchange_events,
        elastic_events,
        event_limit_stops,
        energy_table_overflow_lookups,
        dead_particles_skipped,
        candidate_events,
        null_collision_rejections,
        candidate_limit_stops,
    )


def perform_mcc_electron_1d3v_reference(
    vx: np.ndarray,
    vy: np.ndarray,
    vz: np.ndarray,
    alive: np.ndarray,
    n_g: float,
    sigma_el: np.ndarray,
    sigma_exc: np.ndarray,
    sigma_ion: np.ndarray,
    dt: float,
    m_e: float,
    excitation_threshold_ev: float,
    ionization_threshold_ev: float,
    e_min: float,
    inv_de: float,
    e_charge: float,
    secondary_buffer: SecondaryElectronEventBuffer,
    *,
    seed: int,
    step_index: int = 0,
    stream_id: int = 0,
    max_events_per_particle: int = 64,
) -> ElectronCollisionResult:
    _validate_particle_arrays(vx, vy, vz, alive)
    _validate_sigma_tables(sigma_el, sigma_exc, sigma_ion)
    _validate_reference_values(
        n_g,
        dt,
        m_e,
        e_charge,
        max_events_per_particle,
    )
    if excitation_threshold_ev < 0.0 or not math.isfinite(
        excitation_threshold_ev
    ):
        raise ValueError("The excitation threshold is invalid.")
    if ionization_threshold_ev < 0.0 or not math.isfinite(
        ionization_threshold_ev
    ):
        raise ValueError("The ionization threshold is invalid.")
    if not math.isfinite(e_min) or not math.isfinite(inv_de) or inv_de < 0.0:
        raise ValueError("The cross-section energy grid is invalid.")
    if not isinstance(secondary_buffer, SecondaryElectronEventBuffer):
        raise TypeError("Use a SecondaryElectronEventBuffer instance.")
    secondary_buffer.validate()
    live = np.asarray(alive, dtype=bool)
    if (
        np.any(~np.isfinite(vx[live]))
        or np.any(~np.isfinite(vy[live]))
        or np.any(~np.isfinite(vz[live]))
    ):
        raise ValueError("A live particle has a nonfinite velocity.")

    base_seed = derive_seed(
        seed,
        step_index,
        stream_id,
        _ELECTRON_SEED_COMPONENT,
    )
    (
        elastic_events,
        excitation_events,
        ionization_events,
        secondary_events_written,
        buffer_full_stops,
        event_limit_stops,
        energy_table_overflow_lookups,
        dead_particles_skipped,
        buffer_count,
    ) = _perform_mcc_electron_1d3v_core(
        vx,
        vy,
        vz,
        alive,
        n_g,
        sigma_el,
        sigma_exc,
        sigma_ion,
        dt,
        m_e,
        excitation_threshold_ev,
        ionization_threshold_ev,
        e_min,
        inv_de,
        e_charge,
        secondary_buffer.parent_index,
        secondary_buffer.event_time_s,
        secondary_buffer.vx,
        secondary_buffer.vy,
        secondary_buffer.vz,
        secondary_buffer.energy_ev,
        secondary_buffer.count,
        np.uint64(base_seed),
        max_events_per_particle,
    )
    secondary_buffer.count = buffer_count
    return ElectronCollisionResult(
        elastic_events=elastic_events,
        excitation_events=excitation_events,
        ionization_events=ionization_events,
        secondary_events_written=secondary_events_written,
        buffer_full_stops=buffer_full_stops,
        event_limit_stops=event_limit_stops,
        energy_table_overflow_lookups=energy_table_overflow_lookups,
        dead_particles_skipped=dead_particles_skipped,
    )


def _perform_mcc_electron_channels_1d3v_durations(
    vx: np.ndarray,
    vy: np.ndarray,
    vz: np.ndarray,
    alive: np.ndarray,
    n_g: float,
    sigma_el: np.ndarray,
    excitation_channel_tables: np.ndarray,
    excitation_thresholds_ev: np.ndarray,
    ionization_channel_tables: np.ndarray,
    ionization_thresholds_ev: np.ndarray,
    particle_duration_s: np.ndarray,
    m_e: float,
    e_min: float,
    inv_de: float,
    e_charge: float,
    secondary_buffer: SecondaryElectronEventBuffer,
    *,
    seed: int,
    step_index: int = 0,
    stream_id: int = 0,
    max_events_per_particle: int = 64,
) -> ElectronCollisionResult:
    _validate_particle_arrays(vx, vy, vz, alive)
    durations = _validate_particle_durations(particle_duration_s, vx.shape)
    _validate_sigma_tables(sigma_el)
    _validate_channel_tables(
        excitation_channel_tables,
        excitation_thresholds_ev,
        sigma_el.size,
        "excitation",
    )
    _validate_channel_tables(
        ionization_channel_tables,
        ionization_thresholds_ev,
        sigma_el.size,
        "ionization",
    )
    _validate_reference_values(
        n_g,
        float(np.max(durations, initial=0.0)),
        m_e,
        e_charge,
        max_events_per_particle,
    )
    if not math.isfinite(e_min) or not math.isfinite(inv_de) or inv_de < 0.0:
        raise ValueError("The cross-section energy grid is invalid.")
    if not isinstance(secondary_buffer, SecondaryElectronEventBuffer):
        raise TypeError("Use a SecondaryElectronEventBuffer instance.")
    secondary_buffer.validate()
    live = np.asarray(alive, dtype=bool)
    if (
        np.any(~np.isfinite(vx[live]))
        or np.any(~np.isfinite(vy[live]))
        or np.any(~np.isfinite(vz[live]))
    ):
        raise ValueError("A live particle has a nonfinite velocity.")

    excitation_channel_events = np.zeros(
        excitation_thresholds_ev.size,
        dtype=np.int64,
    )
    ionization_channel_events = np.zeros(
        ionization_thresholds_ev.size,
        dtype=np.int64,
    )
    base_seed = derive_seed(
        seed,
        step_index,
        stream_id,
        _ELECTRON_SEED_COMPONENT,
    )
    (
        elastic_events,
        excitation_events,
        ionization_events,
        secondary_events_written,
        buffer_full_stops,
        event_limit_stops,
        energy_table_overflow_lookups,
        dead_particles_skipped,
        buffer_count,
    ) = _perform_mcc_electron_channels_1d3v_core(
        vx,
        vy,
        vz,
        alive,
        n_g,
        sigma_el,
        excitation_channel_tables,
        excitation_thresholds_ev,
        ionization_channel_tables,
        ionization_thresholds_ev,
        durations,
        m_e,
        e_min,
        inv_de,
        e_charge,
        secondary_buffer.parent_index,
        secondary_buffer.event_time_s,
        secondary_buffer.vx,
        secondary_buffer.vy,
        secondary_buffer.vz,
        secondary_buffer.energy_ev,
        secondary_buffer.count,
        excitation_channel_events,
        ionization_channel_events,
        np.uint64(base_seed),
        max_events_per_particle,
    )
    secondary_buffer.count = buffer_count
    excitation_channel_events.setflags(write=False)
    ionization_channel_events.setflags(write=False)
    return ElectronCollisionResult(
        elastic_events=elastic_events,
        excitation_events=excitation_events,
        ionization_events=ionization_events,
        secondary_events_written=secondary_events_written,
        buffer_full_stops=buffer_full_stops,
        event_limit_stops=event_limit_stops,
        energy_table_overflow_lookups=energy_table_overflow_lookups,
        dead_particles_skipped=dead_particles_skipped,
        excitation_channel_events=excitation_channel_events,
        ionization_channel_events=ionization_channel_events,
    )


def perform_mcc_electron_channels_1d3v(
    vx: np.ndarray,
    vy: np.ndarray,
    vz: np.ndarray,
    alive: np.ndarray,
    n_g: float,
    sigma_el: np.ndarray,
    excitation_channel_tables: np.ndarray,
    excitation_thresholds_ev: np.ndarray,
    ionization_channel_tables: np.ndarray,
    ionization_thresholds_ev: np.ndarray,
    dt: float,
    m_e: float,
    e_min: float,
    inv_de: float,
    e_charge: float,
    secondary_buffer: SecondaryElectronEventBuffer,
    *,
    seed: int,
    step_index: int = 0,
    stream_id: int = 0,
    max_events_per_particle: int = 64,
) -> ElectronCollisionResult:
    durations = np.full(vx.shape, dt, dtype=np.float64)
    return _perform_mcc_electron_channels_1d3v_durations(
        vx,
        vy,
        vz,
        alive,
        n_g,
        sigma_el,
        excitation_channel_tables,
        excitation_thresholds_ev,
        ionization_channel_tables,
        ionization_thresholds_ev,
        durations,
        m_e,
        e_min,
        inv_de,
        e_charge,
        secondary_buffer,
        seed=seed,
        step_index=step_index,
        stream_id=stream_id,
        max_events_per_particle=max_events_per_particle,
    )


def perform_mcc_electron_channels_1d3v_variable_time(
    vx: np.ndarray,
    vy: np.ndarray,
    vz: np.ndarray,
    alive: np.ndarray,
    n_g: float,
    sigma_el: np.ndarray,
    excitation_channel_tables: np.ndarray,
    excitation_thresholds_ev: np.ndarray,
    ionization_channel_tables: np.ndarray,
    ionization_thresholds_ev: np.ndarray,
    particle_duration_s: np.ndarray,
    m_e: float,
    e_min: float,
    inv_de: float,
    e_charge: float,
    secondary_buffer: SecondaryElectronEventBuffer,
    *,
    seed: int,
    step_index: int = 0,
    stream_id: int = 0,
    max_events_per_particle: int = 64,
) -> ElectronCollisionResult:
    """Calculate collisions for particle-specific time intervals."""
    return _perform_mcc_electron_channels_1d3v_durations(
        vx,
        vy,
        vz,
        alive,
        n_g,
        sigma_el,
        excitation_channel_tables,
        excitation_thresholds_ev,
        ionization_channel_tables,
        ionization_thresholds_ev,
        particle_duration_s,
        m_e,
        e_min,
        inv_de,
        e_charge,
        secondary_buffer,
        seed=seed,
        step_index=step_index,
        stream_id=stream_id,
        max_events_per_particle=max_events_per_particle,
    )


def _perform_mcc_ion_1d3v_durations(
    vx: np.ndarray,
    vy: np.ndarray,
    vz: np.ndarray,
    alive: np.ndarray,
    n_g: float,
    sigma_cex: np.ndarray,
    sigma_el: np.ndarray,
    particle_duration_s: np.ndarray,
    neutral_thermal_speed: float,
    e_min: float,
    inv_de: float,
    e_charge: float,
    m_i: float,
    *,
    seed: int,
    step_index: int = 0,
    stream_id: int = 0,
    max_events_per_particle: int = 64,
    max_candidates_per_particle: int | None = None,
) -> IonCollisionResult:
    _validate_particle_arrays(vx, vy, vz, alive)
    durations = _validate_particle_durations(particle_duration_s, vx.shape)
    _validate_sigma_tables(sigma_cex, sigma_el)
    _validate_reference_values(
        n_g,
        float(np.max(durations, initial=0.0)),
        m_i,
        e_charge,
        max_events_per_particle,
    )
    if not math.isfinite(neutral_thermal_speed) or neutral_thermal_speed < 0.0:
        raise ValueError("The neutral thermal speed is invalid.")
    if not math.isfinite(e_min) or e_min < 0.0:
        raise ValueError("The cross-section energy grid is invalid.")
    if not math.isfinite(inv_de) or inv_de <= 0.0:
        raise ValueError("The cross-section energy grid is invalid.")
    if sigma_cex.shape != sigma_el.shape or sigma_cex.size < 2:
        raise ValueError("Ion cross-section tables have different grids.")
    if max_candidates_per_particle is None:
        max_candidates_per_particle = max(
            1024,
            32 * max_events_per_particle,
        )
    if (
        isinstance(max_candidates_per_particle, bool)
        or not isinstance(max_candidates_per_particle, int)
        or max_candidates_per_particle < 0
    ):
        raise ValueError("The candidate limit must be a nonnegative integer.")
    live = np.asarray(alive, dtype=bool)
    if (
        np.any(~np.isfinite(vx[live]))
        or np.any(~np.isfinite(vy[live]))
        or np.any(~np.isfinite(vz[live]))
    ):
        raise ValueError("A live particle has a nonfinite velocity.")

    base_seed = derive_seed(
        seed,
        step_index,
        stream_id,
        _ION_SEED_COMPONENT,
    )
    (
        charge_exchange_events,
        elastic_events,
        event_limit_stops,
        energy_table_overflow_lookups,
        dead_particles_skipped,
        candidate_events,
        null_collision_rejections,
        candidate_limit_stops,
    ) = _perform_mcc_ion_1d3v_core(
        vx,
        vy,
        vz,
        alive,
        n_g,
        sigma_cex,
        sigma_el,
        durations,
        neutral_thermal_speed,
        e_min,
        inv_de,
        e_charge,
        m_i,
        np.uint64(base_seed),
        max_events_per_particle,
        max_candidates_per_particle,
    )
    return IonCollisionResult(
        charge_exchange_events=charge_exchange_events,
        elastic_events=elastic_events,
        event_limit_stops=event_limit_stops,
        energy_table_overflow_lookups=energy_table_overflow_lookups,
        dead_particles_skipped=dead_particles_skipped,
        candidate_events=candidate_events,
        null_collision_rejections=null_collision_rejections,
        candidate_limit_stops=candidate_limit_stops,
    )


def perform_mcc_ion_1d3v_reference(
    vx: np.ndarray,
    vy: np.ndarray,
    vz: np.ndarray,
    alive: np.ndarray,
    n_g: float,
    sigma_cex: np.ndarray,
    sigma_el: np.ndarray,
    dt: float,
    neutral_thermal_speed: float,
    e_min: float,
    inv_de: float,
    e_charge: float,
    m_i: float,
    *,
    seed: int,
    step_index: int = 0,
    stream_id: int = 0,
    max_events_per_particle: int = 64,
    max_candidates_per_particle: int | None = None,
) -> IonCollisionResult:
    durations = np.full(vx.shape, dt, dtype=np.float64)
    return _perform_mcc_ion_1d3v_durations(
        vx,
        vy,
        vz,
        alive,
        n_g,
        sigma_cex,
        sigma_el,
        durations,
        neutral_thermal_speed,
        e_min,
        inv_de,
        e_charge,
        m_i,
        seed=seed,
        step_index=step_index,
        stream_id=stream_id,
        max_events_per_particle=max_events_per_particle,
        max_candidates_per_particle=max_candidates_per_particle,
    )


def perform_mcc_ion_1d3v_variable_time(
    vx: np.ndarray,
    vy: np.ndarray,
    vz: np.ndarray,
    alive: np.ndarray,
    n_g: float,
    sigma_cex: np.ndarray,
    sigma_el: np.ndarray,
    particle_duration_s: np.ndarray,
    neutral_thermal_speed: float,
    e_min: float,
    inv_de: float,
    e_charge: float,
    m_i: float,
    *,
    seed: int,
    step_index: int = 0,
    stream_id: int = 0,
    max_events_per_particle: int = 64,
    max_candidates_per_particle: int | None = None,
) -> IonCollisionResult:
    """Calculate ion collisions for particle-specific time intervals."""
    return _perform_mcc_ion_1d3v_durations(
        vx,
        vy,
        vz,
        alive,
        n_g,
        sigma_cex,
        sigma_el,
        particle_duration_s,
        neutral_thermal_speed,
        e_min,
        inv_de,
        e_charge,
        m_i,
        seed=seed,
        step_index=step_index,
        stream_id=stream_id,
        max_events_per_particle=max_events_per_particle,
        max_candidates_per_particle=max_candidates_per_particle,
    )


perform_mcc_electron_1d3v = perform_mcc_electron_1d3v_reference
perform_mcc_ion_1d3v = perform_mcc_ion_1d3v_reference


__all__ = [
    "ElectronCollisionResult",
    "IonCollisionResult",
    "SecondaryElectronEventBuffer",
    "equal_mass_elastic_collision_3d",
    "perform_coulomb_scatter",
    "perform_mcc_electron",
    "perform_mcc_electron_1d3v",
    "perform_mcc_electron_channels_1d3v",
    "perform_mcc_electron_channels_1d3v_variable_time",
    "perform_mcc_electron_1d3v_reference",
    "perform_mcc_ion",
    "perform_mcc_ion_1d3v",
    "perform_mcc_ion_1d3v_reference",
    "perform_mcc_ion_1d3v_variable_time",
    "sigma_from_uniform_table",
]
