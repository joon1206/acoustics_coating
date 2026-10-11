"""Simplified 1D effective viscoelastic longitudinal-wave layer."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .air import AirProperties, DEFAULT_AIR
from .jca import _passive_pair


@dataclass(frozen=True)
class ViscoelasticLayer:
    """Homogeneous effective longitudinal layer (not a 3D solid model).

    modulus is M' [Pa], density is kg/m^3, loss_factor is dimensionless, and
    thickness is m.  M*=M'(1+j eta_L) is passive for exp(+j omega t).
    """

    modulus: float
    density: float
    loss_factor: float
    thickness: float

    def __post_init__(self) -> None:
        if self.modulus <= 0 or self.density <= 0 or self.thickness <= 0:
            raise ValueError("modulus, density, and thickness must be positive.")
        if self.loss_factor < 0:
            raise ValueError("loss_factor cannot be negative.")

    def complex_modulus(self, frequency: ArrayLike) -> NDArray[np.complex128]:
        frequencies = np.asarray(frequency, dtype=float)
        return np.full(frequencies.shape, self.modulus * (1.0 + 1j * self.loss_factor))

    def wave_properties(
        self, frequency: ArrayLike, air: AirProperties = DEFAULT_AIR
    ) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
        omega = air.omega(frequency)
        modulus = self.complex_modulus(frequency)
        rho = np.full(omega.shape, self.density, dtype=complex)
        return _passive_pair(rho, modulus, omega)

    def characteristic_impedance(self, frequency: ArrayLike) -> NDArray[np.complex128]:
        return self.wave_properties(frequency)[0]

    def wavenumber(self, frequency: ArrayLike) -> NDArray[np.complex128]:
        return self.wave_properties(frequency)[1]

