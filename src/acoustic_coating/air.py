"""Ambient-air properties and lossless plane-wave quantities (SI units)."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class AirProperties:
    """Room-temperature air properties, all in SI units.

    Attributes are density [kg/m^3], sound speed [m/s], dynamic viscosity
    [Pa s], pressure [Pa], heat-capacity ratio [-], thermal conductivity
    [W/(m K)], and constant-pressure specific heat [J/(kg K)].
    """

    rho0: float = 1.2041
    c0: float = 343.2
    eta: float = 1.825e-5
    P0: float = 101_325.0
    gamma: float = 1.4
    kappa: float = 0.0257
    Cp: float = 1005.0

    @property
    def Z0(self) -> float:
        """Characteristic impedance of air [Pa s/m]."""
        return self.rho0 * self.c0

    def omega(self, frequency: ArrayLike) -> NDArray[np.float64]:
        """Angular frequency [rad/s]."""
        frequency = np.asarray(frequency, dtype=float)
        if np.any(frequency <= 0):
            raise ValueError("Frequencies must be strictly positive.")
        return 2.0 * np.pi * frequency


DEFAULT_AIR = AirProperties()

