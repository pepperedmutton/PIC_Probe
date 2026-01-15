from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Config:
    # Physical constants (SI)
    e: float = 1.602176634e-19
    m_e: float = 9.1093837015e-31
    m_i: float = 6.6335209e-26  # Argon ion mass (Ar+)
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
    SIGMA_EN_ELASTIC: float = 4.0e-20
    SIGMA_EN_EXC: float = 2.0e-20
    SIGMA_EN_ION: float = 1.0e-20
    E_EXC_EV: float = 11.6
    E_ION_EV: float = 15.8

    @property
    def dr(self) -> float:
        return (self.R_MAX - self.R_MIN) / self.N_CELLS

    def debye_length(self) -> float:
        """Electron Debye length (m). Assumes Te is in eV."""
        te_joule = self.Te * self.e
        return math.sqrt(self.epsilon_0 * te_joule / (self.N0 * self.e * self.e))

    def plasma_frequency(self) -> float:
        """Electron plasma frequency (rad/s)."""
        return math.sqrt(self.N0 * self.e * self.e / (self.epsilon_0 * self.m_e))

    def thermal_speed_e(self) -> float:
        """Electron thermal speed (m/s)."""
        return math.sqrt(self.e * self.Te / self.m_e)

    def thermal_speed_i(self) -> float:
        """Ion thermal speed (m/s)."""
        return math.sqrt(self.e * self.Ti / self.m_i)

    def stability_warnings(self) -> list[str]:
        """Return human-readable stability warnings based on dt and dr."""
        warnings: list[str] = []
        lambda_d = self.debye_length()
        if self.dr > lambda_d:
            warnings.append(
                f"dr ({self.dr:.3e} m) exceeds Debye length lambda_D ({lambda_d:.3e} m)."
            )

        omega_pe = self.plasma_frequency()
        if self.DT * omega_pe > 0.2:
            warnings.append(
                "dt * omega_pe exceeds 0.2; explicit PIC may be unstable or noisy."
            )

        vth_e = self.thermal_speed_e()
        if vth_e * self.DT > self.dr:
            warnings.append(
                "Electron CFL violated: vth_e * dt exceeds dr."
            )

        vth_i = self.thermal_speed_i()
        if vth_i * self.DT > self.dr:
            warnings.append(
                "Ion CFL violated: vth_i * dt exceeds dr."
            )

        return warnings
