import numpy as np

from benchmarks import cylindrical_vacuum_capacitor_benchmark
from core.fields import compute_electric_field, smooth_density_cylindrical
from core.particles import compute_shell_volumes, deposit_charge_cic


def test_vacuum_poisson_converges() -> None:
    result = cylindrical_vacuum_capacitor_benchmark()
    assert result.passed
    assert result.metrics["minimum_observed_order"] >= 1.8


def test_electric_field_of_linear_potential_is_constant() -> None:
    phi = np.array([4.0, 3.0, 2.0, 1.0], dtype=float)
    field = np.empty_like(phi)
    compute_electric_field(phi, 0.5, field)
    np.testing.assert_allclose(field, 2.0)


def test_cic_deposition_conserves_charge() -> None:
    positions = np.array([1.25, 1.75, 2.50], dtype=float)
    charges = np.array([-2.0, 1.0, 4.0], dtype=float)
    nodes = np.empty(5, dtype=float)
    deposit_charge_cic(positions, charges, 1.0, 0.5, nodes)
    assert np.sum(nodes) == np.sum(charges)


def test_density_smoothing_conserves_weighted_charge() -> None:
    volumes = np.empty(9, dtype=float)
    compute_shell_volumes(1.0e-3, 1.0e-4, volumes)
    density = np.array([0.0, 1.0, 5.0, 2.0, 9.0, 3.0, 1.0, 0.5, 0.0])
    initial = float(np.sum(density * volumes))
    smooth_density_cylindrical(density, 4, volumes)
    assert np.isclose(np.sum(density * volumes), initial, rtol=1.0e-14)
