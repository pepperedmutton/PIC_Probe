import math

import numpy as np

from core.collisions import (
    SecondaryElectronEventBuffer,
    perform_mcc_electron_channels_1d3v_variable_time,
    perform_mcc_ion_1d3v_variable_time,
)
from core.config import Config
from core.simulation import BoundaryInjectionResult, PICSimulation


def make_buffer(capacity: int) -> SecondaryElectronEventBuffer:
    return SecondaryElectronEventBuffer(
        parent_index=np.empty(capacity, dtype=np.int64),
        event_time_s=np.empty(capacity, dtype=np.float64),
        vx=np.empty(capacity, dtype=np.float64),
        vy=np.empty(capacity, dtype=np.float64),
        vz=np.empty(capacity, dtype=np.float64),
        energy_ev=np.empty(capacity, dtype=np.float64),
    )


def make_simulation() -> PICSimulation:
    config = Config.research(
        N0=1.0e12,
        Te=1.0,
        Ti=0.03,
        R_MIN=1.0e-3,
        R_MAX=2.0e-3,
        N_CELLS=8,
        DT=1.0e-12,
        P_Torr=1.0e-6,
    )
    return PICSimulation(config, n_particles=8, seed=17)


def test_electron_variable_time_skips_zero_duration_particle() -> None:
    mass = 9.1093837015e-31
    charge = 1.602176634e-19
    speed = math.sqrt(2.0 * charge / mass)
    vx = np.full(2, speed)
    vy = np.zeros(2)
    vz = np.zeros(2)

    result = perform_mcc_electron_channels_1d3v_variable_time(
        vx,
        vy,
        vz,
        np.ones(2, dtype=bool),
        1.0,
        np.full(2, 10.0 / speed),
        np.empty((0, 2)),
        np.empty(0),
        np.empty((0, 2)),
        np.empty(0),
        np.array([1.0, 0.0]),
        mass,
        0.0,
        1.0,
        charge,
        make_buffer(1),
        seed=31,
    )

    assert result.elastic_events > 0
    np.testing.assert_array_equal(
        np.array([vx[1], vy[1], vz[1]]),
        np.array([speed, 0.0, 0.0]),
    )


def test_ion_variable_time_skips_zero_duration_particle() -> None:
    mass = 6.6335209e-26
    charge = 1.602176634e-19
    speed = math.sqrt(2.0 * charge / mass)
    vx = np.full(2, speed)
    vy = np.zeros(2)
    vz = np.zeros(2)

    result = perform_mcc_ion_1d3v_variable_time(
        vx,
        vy,
        vz,
        np.ones(2, dtype=bool),
        1.0,
        np.full(2, 10.0 / speed),
        np.zeros(2),
        np.array([1.0, 0.0]),
        0.0,
        0.0,
        1.0,
        charge,
        mass,
        seed=37,
    )

    assert result.charge_exchange_events == 1
    np.testing.assert_array_equal(
        np.array([vx[1], vy[1], vz[1]]),
        np.array([speed, 0.0, 0.0]),
    )


def test_ionization_products_use_parent_remaining_time(monkeypatch) -> None:
    simulation = make_simulation()
    interval = 4.0e-10
    event_time = 1.0e-10
    events = make_buffer(1)
    assert events.record(
        parent_index=0,
        event_time_s=event_time,
        vx=2.0,
        vy=3.0,
        vz=4.0,
        energy_ev=1.0,
    )
    parent_duration = np.zeros(simulation.n_particles)
    parent_duration[0] = interval
    captured: dict[str, object] = {}

    def capture_intervals(
        electron_duration_s: np.ndarray,
        ion_duration_s: np.ndarray,
        *,
        step_index: int,
        stream_id: int,
    ) -> SecondaryElectronEventBuffer:
        captured["electron"] = electron_duration_s.copy()
        captured["ion"] = ion_duration_s.copy()
        captured["step_index"] = step_index
        captured["stream_id"] = stream_id
        return make_buffer(1)

    monkeypatch.setattr(
        simulation,
        "_apply_collision_intervals",
        capture_intervals,
    )
    simulation._process_ionization_descendants(
        events,
        parent_duration,
        step_index=12,
        stream_id=3,
    )

    electron_duration = captured["electron"]
    ion_duration = captured["ion"]
    assert isinstance(electron_duration, np.ndarray)
    assert isinstance(ion_duration, np.ndarray)
    np.testing.assert_array_equal(
        electron_duration[electron_duration > 0.0],
        np.array([interval - event_time]),
    )
    np.testing.assert_array_equal(
        ion_duration[ion_duration > 0.0],
        np.array([interval - event_time]),
    )
    assert captured["step_index"] == 12
    assert captured["stream_id"] == 3
    assert simulation.collision_counters[
        "ionization_descendants_processed"
    ] == 2


def test_boundary_arrivals_use_each_residence_time(monkeypatch) -> None:
    simulation = make_simulation()
    electron_slot = int(
        np.flatnonzero(
            (simulation.r_e <= simulation.r_min)
            | (simulation.r_e >= simulation.r_max)
        )[0]
    )
    ion_slot = int(
        np.flatnonzero(
            (simulation.r_i <= simulation.r_min)
            | (simulation.r_i >= simulation.r_max)
        )[0]
    )
    midpoint = 0.5 * (simulation.r_min + simulation.r_max)
    simulation.r_e[electron_slot] = midpoint
    simulation.r_i[ion_slot] = midpoint
    electron_time = 0.2 * simulation.dt
    ion_time = 0.8 * simulation.dt
    electron_injection = BoundaryInjectionResult(
        residual=0.0,
        crossing_events=1,
        active_particles=1,
        probe_hits=0,
        active_slots=np.array([electron_slot], dtype=np.int64),
        time_inside_s=np.array([electron_time]),
    )
    ion_injection = BoundaryInjectionResult(
        residual=0.0,
        crossing_events=1,
        active_particles=1,
        probe_hits=0,
        active_slots=np.array([ion_slot], dtype=np.int64),
        time_inside_s=np.array([ion_time]),
    )
    captured: dict[str, object] = {}

    def capture_intervals(
        electron_duration_s: np.ndarray,
        ion_duration_s: np.ndarray,
        *,
        step_index: int,
        stream_id: int,
    ) -> SecondaryElectronEventBuffer:
        captured["electron"] = electron_duration_s.copy()
        captured["ion"] = ion_duration_s.copy()
        captured["step_index"] = step_index
        captured["stream_id"] = stream_id
        return make_buffer(1)

    monkeypatch.setattr(
        simulation,
        "_apply_collision_intervals",
        capture_intervals,
    )
    simulation._apply_boundary_entry_collisions(
        electron_injection,
        ion_injection,
    )

    electron_duration = captured["electron"]
    ion_duration = captured["ion"]
    assert isinstance(electron_duration, np.ndarray)
    assert isinstance(ion_duration, np.ndarray)
    assert electron_duration[electron_slot] == electron_time
    assert ion_duration[ion_slot] == ion_time
    assert captured["step_index"] == 1
    assert captured["stream_id"] == 1
    assert simulation.collision_counters[
        "boundary_entry_particles_processed"
    ] == 2


def test_step_processes_boundary_arrivals_and_updates_limitations() -> None:
    simulation = make_simulation()
    simulation.inject_target_e = 1.0
    simulation.inject_target_i = 1.0

    simulation.step()

    assert simulation.collision_counters[
        "boundary_entry_particles_processed"
    ] == 2
    limitations = simulation.run_manifest()["simulation"]["physics_limitations"]
    assert "boundary_entry_motion_uses_precollision_velocity" in limitations
    assert (
        "boundary_injection_skips_remaining_entry_step_collisions"
        not in limitations
    )
    assert (
        "ionization_products_skip_remaining_collision_half_step"
        not in limitations
    )
