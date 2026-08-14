import pytest

from core.config import Config
from core.simulation import PICSimulation


def test_config_fingerprint_is_stable() -> None:
    first = Config.research(N0=2.0e15, Te=2.5)
    second = Config.research(N0=2.0e15, Te=2.5)
    assert first.fingerprint() == second.fingerprint()
    assert len(first.fingerprint()) == 64


def test_config_rejects_invalid_geometry() -> None:
    with pytest.raises(ValueError, match="R_MAX"):
        Config.research(R_MIN=1.0e-3, R_MAX=1.0e-3)


def test_production_simulation_remains_locked() -> None:
    config = Config.production(
        N0=1.0e12,
        Te=1.0,
        Ti=0.03,
        R_MIN=1.0e-4,
        R_MAX=2.0e-2,
        N_CELLS=20,
        DT=1.0e-12,
        LXCAT_ELECTRON_FILE="not-read-electron.txt",
        LXCAT_ION_FILE="not-read-ion.txt",
    )
    with pytest.raises(RuntimeError, match="locked"):
        PICSimulation(config, n_particles=16, seed=1)


def test_research_stability_metrics_are_finite() -> None:
    metrics = Config.research().stability_metrics_dict()
    assert metrics["debye_length_m"] > 0.0
    assert metrics["plasma_frequency_rad_s"] > 0.0


def test_strict_cross_sections_require_both_files() -> None:
    config = Config.research(
        N0=1.0e12,
        Te=1.0,
        Ti=0.03,
        R_MIN=1.0e-4,
        R_MAX=2.0e-2,
        N_CELLS=20,
        DT=1.0e-12,
        CROSS_SECTION_STRICT=True,
    )
    with pytest.raises(ValueError, match="requires both electron and ion files"):
        PICSimulation(config, n_particles=16, seed=1)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("CROSS_SECTION_TARGET", "Xe"),
        ("NEUTRAL_SPECIES", "Xe"),
        ("ION_SPECIES", "Ar2+"),
    ],
)
def test_config_rejects_unsupported_species(field, value) -> None:
    with pytest.raises(ValueError, match="supports only neutral Ar"):
        Config.research(**{field: value})
