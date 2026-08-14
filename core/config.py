from __future__ import annotations

from dataclasses import dataclass, fields
from enum import Enum
from hashlib import sha256
import json
import math
from numbers import Integral, Real
from typing import Any, ClassVar


class RunMode(str, Enum):
    """Give the permitted run modes."""

    RESEARCH = "research"
    PRODUCTION = "production"


@dataclass(frozen=True)
class StabilityMetrics:
    """Keep the calculated stability values."""

    cell_size_m: float
    debye_length_m: float
    cell_to_debye_ratio: float
    plasma_frequency_rad_s: float
    dt_omega_pe: float
    electron_thermal_speed_m_s: float
    electron_cfl: float
    ion_thermal_speed_m_s: float
    ion_cfl: float
    debye_resolved: bool
    plasma_time_resolved: bool
    electron_cfl_resolved: bool
    ion_cfl_resolved: bool

    @property
    def is_stable(self) -> bool:
        return (
            self.debye_resolved
            and self.plasma_time_resolved
            and self.electron_cfl_resolved
            and self.ion_cfl_resolved
        )

    def as_dict(self) -> dict[str, float | bool]:
        """Give the stability values as a dictionary."""
        data: dict[str, float | bool] = {
            field.name: getattr(self, field.name) for field in fields(self)
        }
        data["is_stable"] = self.is_stable
        return dict(sorted(data.items()))


@dataclass(frozen=True)
class Config:
    """Keep the physical and numerical configuration."""

    CONFIG_SCHEMA_VERSION: ClassVar[int] = 3
    MAX_CELL_TO_DEBYE_RATIO: ClassVar[float] = 1.0
    MAX_DT_OMEGA_PE: ClassVar[float] = 0.2
    MAX_CFL: ClassVar[float] = 1.0
    PRODUCTION_MAX_CELL_TO_DEBYE_RATIO: ClassVar[float] = 0.5
    PRODUCTION_MAX_DT_OMEGA_PE: ClassVar[float] = 0.1
    PRODUCTION_MAX_CFL: ClassVar[float] = 0.5
    PHYSICS_RELEASE_READY: ClassVar[bool] = False

    # Physical constants (SI)
    e: float = 1.602176634e-19
    m_e: float = 9.1093837015e-31
    m_i: float = 6.6335209e-26  # Argon ion mass (Ar+)
    m_neutral: float = 6.6335209e-26
    m_p: float = 1.67262192369e-27
    epsilon_0: float = 8.8541878128e-12
    k_B: float = 1.380649e-23

    # Simulation parameters
    N_CELLS: int = 200
    DT: float = 1.0e-10
    R_MIN: float = 5.0e-4
    R_MAX: float = 5.0e-2
    V_WALL: float = 0.0

    # Plasma parameters
    N0: float = 1.0e16
    Te: float = 3.0  # eV
    Ti: float = 0.03  # eV
    P_Torr: float = 50.0
    T_GAS_K: float = 300.0
    SIGMA_EN_ELASTIC: float = 4.0e-20
    SIGMA_EN_EXC: float = 2.0e-20
    SIGMA_EN_ION: float = 1.0e-20
    SIGMA_IN_ELASTIC: float = 0.0
    E_EXC_EV: float = 11.6
    E_ION_EV: float = 15.8
    ION_INJECTION_BOHM: bool = True

    # Energy-dependent cross-section tables (LXCat text files)
    CROSS_SECTION_TARGET: str = "Ar"
    NEUTRAL_SPECIES: str = "Ar"
    ION_SPECIES: str = "Ar+"
    ION_CHARGE_STATE: int = 1
    LXCAT_ELECTRON_FILE: str | None = None
    LXCAT_ION_FILE: str | None = None
    CROSS_SECTION_STRICT: bool = False
    CONFIRM_SYMMETRIC_BACKSCATTER_AS_CEX: bool = False
    EN_CS_E_MAX: float = 200.0
    EN_CS_N: int = 2001
    ION_CS_E_MAX: float = 200.0
    ION_CS_N: int = 2001

    # Collision feature toggles
    ENABLE_IONIZATION_SECONDARIES: bool = True
    ENABLE_ION_NEUTRAL_ELASTIC: bool = True
    ENABLE_COULOMB_COLLISIONS: bool = False
    COULOMB_LOG: float = 10.0

    # Numerical stability features
    SMOOTH_DENSITY: bool = False
    N_SMOOTHING_PASSES: int = 1
    RUN_MODE: str = RunMode.RESEARCH.value

    def __post_init__(self) -> None:
        self._normalize_mode()
        self._validate_values()
        if self.is_production:
            self.require_stable()

    @classmethod
    def research(cls, **values: Any) -> Config:
        """Give a research configuration."""
        values["RUN_MODE"] = RunMode.RESEARCH.value
        return cls(**values)

    @classmethod
    def production(cls, **values: Any) -> Config:
        """Give a production configuration."""
        values["RUN_MODE"] = RunMode.PRODUCTION.value
        return cls(**values)

    @property
    def mode(self) -> RunMode:
        return RunMode(self.RUN_MODE)

    @property
    def is_production(self) -> bool:
        return self.mode is RunMode.PRODUCTION

    def _normalize_mode(self) -> None:
        value = self.RUN_MODE.value if isinstance(self.RUN_MODE, RunMode) else self.RUN_MODE
        if not isinstance(value, str):
            raise TypeError("Set RUN_MODE to 'research' or 'production'.")
        try:
            mode = RunMode(value.strip().lower())
        except ValueError as exc:
            raise ValueError("Set RUN_MODE to 'research' or 'production'.") from exc
        object.__setattr__(self, "RUN_MODE", mode.value)

    @staticmethod
    def _require_real(name: str, value: Real) -> float:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"Set {name} to a real number.")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"Set {name} to a finite value.")
        return number

    @staticmethod
    def _require_integer(name: str, value: Integral, minimum: int) -> int:
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(f"Set {name} to an integer.")
        number = int(value)
        if number < minimum:
            raise ValueError(f"Set {name} to an integer not less than {minimum}.")
        return number

    @classmethod
    def _require_positive(cls, name: str, value: Real) -> None:
        if cls._require_real(name, value) <= 0.0:
            raise ValueError(f"Set {name} to a value greater than zero.")

    @classmethod
    def _require_nonnegative(cls, name: str, value: Real) -> None:
        if cls._require_real(name, value) < 0.0:
            raise ValueError(f"Set {name} to zero or a positive value.")

    def _validate_values(self) -> None:
        for name in (
            "e",
            "m_e",
            "m_i",
            "m_neutral",
            "m_p",
            "epsilon_0",
            "k_B",
        ):
            self._require_positive(name, getattr(self, name))

        self._require_integer("N_CELLS", self.N_CELLS, 2)
        self._require_positive("DT", self.DT)
        self._require_positive("R_MIN", self.R_MIN)
        self._require_real("R_MAX", self.R_MAX)
        if self.R_MAX <= self.R_MIN:
            raise ValueError("Set R_MAX to a value greater than R_MIN.")
        self._require_real("V_WALL", self.V_WALL)

        self._require_positive("N0", self.N0)
        self._require_positive("Te", self.Te)
        self._require_positive("Ti", self.Ti)
        self._require_nonnegative("P_Torr", self.P_Torr)
        self._require_positive("T_GAS_K", self.T_GAS_K)

        for name in (
            "SIGMA_EN_ELASTIC",
            "SIGMA_EN_EXC",
            "SIGMA_EN_ION",
            "SIGMA_IN_ELASTIC",
            "E_EXC_EV",
            "E_ION_EV",
            "COULOMB_LOG",
        ):
            self._require_nonnegative(name, getattr(self, name))

        if self.ENABLE_COULOMB_COLLISIONS and self.COULOMB_LOG <= 0.0:
            raise ValueError(
                "Set COULOMB_LOG to a value greater than zero when Coulomb collisions are active."
            )

        self._require_positive("EN_CS_E_MAX", self.EN_CS_E_MAX)
        self._require_positive("ION_CS_E_MAX", self.ION_CS_E_MAX)
        self._require_integer("EN_CS_N", self.EN_CS_N, 2)
        self._require_integer("ION_CS_N", self.ION_CS_N, 2)
        self._require_integer("ION_CHARGE_STATE", self.ION_CHARGE_STATE, 1)
        self._require_integer("N_SMOOTHING_PASSES", self.N_SMOOTHING_PASSES, 0)
        if self.ION_CHARGE_STATE != 1:
            raise ValueError("Set ION_CHARGE_STATE to 1 for this physics model.")

        if self.SMOOTH_DENSITY and self.N_SMOOTHING_PASSES < 1:
            raise ValueError(
                "Set N_SMOOTHING_PASSES to at least 1 when density smoothing is active."
            )

        for name in (
            "ION_INJECTION_BOHM",
            "ENABLE_IONIZATION_SECONDARIES",
            "ENABLE_ION_NEUTRAL_ELASTIC",
            "ENABLE_COULOMB_COLLISIONS",
            "SMOOTH_DENSITY",
            "CROSS_SECTION_STRICT",
            "CONFIRM_SYMMETRIC_BACKSCATTER_AS_CEX",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"Set {name} to a Boolean value.")

        if not self.ENABLE_IONIZATION_SECONDARIES:
            raise ValueError(
                "Keep ENABLE_IONIZATION_SECONDARIES active to conserve ionization products."
            )
        if self.is_production and self.ENABLE_COULOMB_COLLISIONS:
            raise ValueError(
                "Do not use the preview Coulomb collision model in a production run."
            )

        if not isinstance(self.CROSS_SECTION_TARGET, str) or not self.CROSS_SECTION_TARGET.strip():
            raise ValueError("Set CROSS_SECTION_TARGET to a nonempty value.")
        object.__setattr__(self, "CROSS_SECTION_TARGET", self.CROSS_SECTION_TARGET.strip())
        for name in ("NEUTRAL_SPECIES", "ION_SPECIES"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Set {name} to a nonempty value.")
            object.__setattr__(self, name, value.strip())

        supported_species = (
            self.CROSS_SECTION_TARGET,
            self.NEUTRAL_SPECIES,
            self.ION_SPECIES,
            self.ION_CHARGE_STATE,
        )
        if supported_species != ("Ar", "Ar", "Ar+", 1):
            raise ValueError(
                "This physics model supports only neutral Ar and singly "
                "charged Ar+. Set CROSS_SECTION_TARGET='Ar', "
                "NEUTRAL_SPECIES='Ar', ION_SPECIES='Ar+', and "
                "ION_CHARGE_STATE=1."
            )

        for name in ("LXCAT_ELECTRON_FILE", "LXCAT_ION_FILE"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"Set {name} to a nonempty file name or None.")
            if isinstance(value, str):
                object.__setattr__(self, name, value.strip())

    @property
    def dr(self) -> float:
        """Give the radial cell size in meters."""
        return (self.R_MAX - self.R_MIN) / self.N_CELLS

    @property
    def pressure_pa(self) -> float:
        """Give the neutral pressure in pascals."""
        return self.P_Torr * 133.322368

    def debye_length(self) -> float:
        """Calculate the electron Debye length in meters."""
        te_joule = self.Te * self.e
        return math.sqrt(self.epsilon_0 * te_joule / (self.N0 * self.e * self.e))

    def plasma_frequency(self) -> float:
        """Calculate the electron plasma frequency in radians per second."""
        return math.sqrt(self.N0 * self.e * self.e / (self.epsilon_0 * self.m_e))

    def thermal_speed_e(self) -> float:
        """Calculate the electron thermal speed in meters per second."""
        return math.sqrt(self.e * self.Te / self.m_e)

    def thermal_speed_i(self) -> float:
        """Calculate the ion thermal speed in meters per second."""
        return math.sqrt(self.e * self.Ti / self.m_i)

    def stability_metrics(self) -> StabilityMetrics:
        """Calculate the numerical stability values."""
        lambda_d = self.debye_length()
        omega_pe = self.plasma_frequency()
        vth_e = self.thermal_speed_e()
        vth_i = self.thermal_speed_i()
        cell_to_debye = self.dr / lambda_d
        dt_omega_pe = self.DT * omega_pe
        electron_cfl = vth_e * self.DT / self.dr
        ion_cfl = vth_i * self.DT / self.dr
        cell_limit = (
            self.PRODUCTION_MAX_CELL_TO_DEBYE_RATIO
            if self.is_production
            else self.MAX_CELL_TO_DEBYE_RATIO
        )
        time_limit = (
            self.PRODUCTION_MAX_DT_OMEGA_PE
            if self.is_production
            else self.MAX_DT_OMEGA_PE
        )
        cfl_limit = (
            self.PRODUCTION_MAX_CFL
            if self.is_production
            else self.MAX_CFL
        )
        return StabilityMetrics(
            cell_size_m=self.dr,
            debye_length_m=lambda_d,
            cell_to_debye_ratio=cell_to_debye,
            plasma_frequency_rad_s=omega_pe,
            dt_omega_pe=dt_omega_pe,
            electron_thermal_speed_m_s=vth_e,
            electron_cfl=electron_cfl,
            ion_thermal_speed_m_s=vth_i,
            ion_cfl=ion_cfl,
            debye_resolved=cell_to_debye <= cell_limit,
            plasma_time_resolved=dt_omega_pe <= time_limit,
            electron_cfl_resolved=electron_cfl <= cfl_limit,
            ion_cfl_resolved=ion_cfl <= cfl_limit,
        )

    def stability_metrics_dict(self) -> dict[str, float | bool]:
        """Give the numerical stability values as a dictionary."""
        return self.stability_metrics().as_dict()

    def stability_warnings(self) -> list[str]:
        """Give the numerical stability warnings."""
        warnings: list[str] = []
        metrics = self.stability_metrics()
        cell_limit = (
            self.PRODUCTION_MAX_CELL_TO_DEBYE_RATIO
            if self.is_production
            else self.MAX_CELL_TO_DEBYE_RATIO
        )
        time_limit = (
            self.PRODUCTION_MAX_DT_OMEGA_PE
            if self.is_production
            else self.MAX_DT_OMEGA_PE
        )
        cfl_limit = (
            self.PRODUCTION_MAX_CFL
            if self.is_production
            else self.MAX_CFL
        )
        if not metrics.debye_resolved:
            warnings.append(
                f"The value of dr / lambda_D is more than {cell_limit:.3g}."
            )

        if not metrics.plasma_time_resolved:
            warnings.append(
                f"The value of dt * omega_pe is more than {time_limit:.3g}. "
                "Explicit PIC can be unstable or noisy."
            )

        if not metrics.electron_cfl_resolved:
            warnings.append(
                f"The electron CFL value is more than {cfl_limit:.3g}."
            )

        if not metrics.ion_cfl_resolved:
            warnings.append(
                f"The ion CFL value is more than {cfl_limit:.3g}."
            )

        return warnings

    def require_stable(self) -> None:
        """Make sure that the configuration is stable."""
        warnings = self.stability_warnings()
        if warnings:
            detail = " | ".join(warnings)
            raise ValueError(
                f"Change the numerical parameters to remove all stability warnings. {detail}"
            )

    @staticmethod
    def _canonical_value(value: Any) -> Any:
        if isinstance(value, Enum):
            return value.value
        if value is None or isinstance(value, (str, bool)):
            return value
        if isinstance(value, Integral):
            return int(value)
        if isinstance(value, Real):
            number = float(value)
            return 0.0 if number == 0.0 else number
        raise TypeError(f"The configuration value type is not valid: {type(value).__name__}.")

    def canonical_dict(self) -> dict[str, Any]:
        """Give the normalized configuration data."""
        data: dict[str, Any] = {"config_schema_version": self.CONFIG_SCHEMA_VERSION}
        for field in fields(self):
            value = self._canonical_value(getattr(self, field.name))
            if field.name in {"LXCAT_ELECTRON_FILE", "LXCAT_ION_FILE"} and value is not None:
                value = value.replace("\\", "/")
            data[field.name] = value
        return dict(sorted(data.items()))

    def normalized_dict(self) -> dict[str, Any]:
        """Give the normalized configuration data."""
        return self.canonical_dict()

    def canonical_json(self) -> str:
        """Give the normalized configuration as JSON."""
        return json.dumps(
            self.canonical_dict(),
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )

    def fingerprint(self) -> str:
        """Calculate the SHA-256 configuration value."""
        return sha256(self.canonical_json().encode("utf-8")).hexdigest()

    @property
    def sha256_fingerprint(self) -> str:
        return self.fingerprint()
