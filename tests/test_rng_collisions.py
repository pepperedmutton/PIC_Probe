import math

import numpy as np

from benchmarks import (
    constant_cross_section_collision_benchmark,
    equal_mass_elastic_conservation_benchmark,
)
from core.collisions import (
    SecondaryElectronEventBuffer,
    perform_mcc_electron_channels_1d3v,
    sigma_from_uniform_table,
)
from core.rng import counter_uniform, derive_seed


def make_buffer(capacity: int) -> SecondaryElectronEventBuffer:
    return SecondaryElectronEventBuffer(
        parent_index=np.empty(capacity, dtype=np.int64),
        event_time_s=np.empty(capacity, dtype=np.float64),
        vx=np.empty(capacity, dtype=np.float64),
        vy=np.empty(capacity, dtype=np.float64),
        vz=np.empty(capacity, dtype=np.float64),
        energy_ev=np.empty(capacity, dtype=np.float64),
    )


def test_counter_rng_is_deterministic() -> None:
    values = [counter_uniform(123, index, 7) for index in range(8)]
    assert values == [counter_uniform(123, index, 7) for index in range(8)]
    assert len(set(values)) == len(values)
    assert derive_seed(123, 4, 5) == derive_seed(123, 4, 5)


def test_uniform_table_interpolation() -> None:
    table = np.array([0.0, 2.0, 4.0], dtype=float)
    assert sigma_from_uniform_table(0.5, table, 0.0, 1.0) == 1.0
    assert sigma_from_uniform_table(9.0, table, 0.0, 1.0) == 4.0


def test_fast_collision_benchmarks_pass() -> None:
    collision = constant_cross_section_collision_benchmark(
        particle_count=400,
        replicate_count=4,
    )
    conservation = equal_mass_elastic_conservation_benchmark()
    assert collision.passed
    assert conservation.passed


def test_right_table_boundary_does_not_create_false_overflow() -> None:
    count = 400
    mass = 9.1093837015e-31
    charge = 1.602176634e-19
    speed = math.sqrt(2.0 * charge / mass)
    vx = np.full(count, speed)
    vy = np.zeros(count)
    vz = np.zeros(count)
    alive = np.ones(count, dtype=bool)
    result = perform_mcc_electron_channels_1d3v(
        vx,
        vy,
        vz,
        alive,
        1.0,
        np.array([10.0 / speed, 10.0 / speed]),
        np.empty((0, 2)),
        np.empty(0),
        np.empty((0, 2)),
        np.empty(0),
        1.0,
        mass,
        0.0,
        1.0,
        charge,
        make_buffer(1),
        seed=17,
    )
    assert result.energy_table_overflow_lookups == 0
    assert result.total_events > count


def test_channel_collisions_repeat_with_the_same_seed() -> None:
    count = 64
    mass = 9.1093837015e-31
    charge = 1.602176634e-19
    speed = math.sqrt(2.0 * 30.0 * charge / mass)

    def run_once() -> tuple[np.ndarray, object, SecondaryElectronEventBuffer]:
        vx = np.full(count, speed)
        vy = np.zeros(count)
        vz = np.zeros(count)
        result = perform_mcc_electron_channels_1d3v(
            vx,
            vy,
            vz,
            np.ones(count, dtype=bool),
            1.0e20,
            np.zeros(101),
            np.empty((0, 101)),
            np.empty(0),
            np.full((1, 101), 1.0e-20),
            np.array([15.7596119]),
            2.0e-7,
            mass,
            0.0,
            1.0,
            charge,
            buffer := make_buffer(count * 4),
            seed=901,
        )
        return np.column_stack((vx, vy, vz)), result, buffer

    velocity_a, result_a, buffer_a = run_once()
    velocity_b, result_b, buffer_b = run_once()
    np.testing.assert_array_equal(velocity_a, velocity_b)
    assert result_a == result_b
    assert result_a.ionization_events > 0
    np.testing.assert_array_equal(
        buffer_a.event_time_s[: buffer_a.count],
        buffer_b.event_time_s[: buffer_b.count],
    )
