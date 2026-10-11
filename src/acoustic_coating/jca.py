"""Johnson--Champoux--Allard rigid-frame equivalent-fluid model."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .air import AirProperties, DEFAULT_AIR


def _passive_pair(
    rho: NDArray[np.complex128],
    bulk: NDArray[np.complex128],
    omega: NDArray[np.float64],
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    """Choose coupled square-root signs for Re(Z)>=0 and Im(k)<=0.

    With exp(+j omega t) and forward waves exp(-j k x), attenuation requires
    Im(k) <= 0.  Z and k must be flipped together to retain k = omega rho/Z.
    """
    impedance = np.sqrt(rho * bulk).astype(complex)
    wavenumber = (omega * rho / impedance).astype(complex)
    flip = (np.imag(wavenumber) > 0) | (
        np.isclose(np.imag(wavenumber), 0.0) & (np.real(impedance) < 0)
    )
    impedance = np.where(flip, -impedance, impedance)
    wavenumber = np.where(flip, -wavenumber, wavenumber)
    return impedance, wavenumber


@dataclass(frozen=True)
class JCAMaterial:
    """Effective porous material parameters in SI units."""

    phi: float
    sigma: float
    alpha_inf: float
    Lambda: float
    Lambda_prime: float
    thickness: float

    def __post_init__(self) -> None:
        if not 0.0 < self.phi <= 1.0:
            raise ValueError("phi must lie in (0, 1].")
        for name in ("sigma", "alpha_inf", "Lambda", "Lambda_prime", "thickness"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive.")
        if self.alpha_inf < 1.0:
            raise ValueError("alpha_inf must be at least one.")

    def effective_density(
        self, frequency: ArrayLike, air: AirProperties = DEFAULT_AIR
    ) -> NDArray[np.complex128]:
        """Dynamic effective density [kg/m^3] for exp(+j omega t)."""
        omega = air.omega(frequency)
        viscous_root = np.sqrt(
            1.0
            + 1j
            * 4.0
            * self.alpha_inf**2
            * air.eta
            * air.rho0
            * omega
            / (self.sigma**2 * self.Lambda**2 * self.phi**2)
        )
        correction = self.sigma * self.phi / (
            1j * omega * air.rho0 * self.alpha_inf
        ) * viscous_root
        return np.asarray(
            self.alpha_inf * air.rho0 / self.phi * (1.0 + correction),
            dtype=complex,
        )

    def effective_bulk_modulus(
        self, frequency: ArrayLike, air: AirProperties = DEFAULT_AIR
    ) -> NDArray[np.complex128]:
        """Champoux--Allard dynamic bulk modulus [Pa]."""
        omega = air.omega(frequency)
        thermal_root = np.sqrt(
            1.0
            + 1j
            * self.Lambda_prime**2
            * air.Cp
            * air.rho0
            * omega
            / (16.0 * air.kappa)
        )
        thermal_bracket = 1.0 - 1j * 8.0 * air.kappa / (
            self.Lambda_prime**2 * air.Cp * air.rho0 * omega
        ) * thermal_root
        denominator = air.gamma - (air.gamma - 1.0) / thermal_bracket
        return np.asarray((air.gamma * air.P0 / self.phi) / denominator, dtype=complex)

    def wave_properties(
        self, frequency: ArrayLike, air: AirProperties = DEFAULT_AIR
    ) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
        """Return passive characteristic impedance [Pa s/m] and k [rad/m]."""
        omega = air.omega(frequency)
        rho = self.effective_density(frequency, air)
        bulk = self.effective_bulk_modulus(frequency, air)
        return _passive_pair(rho, bulk, omega)

    def characteristic_impedance(
        self, frequency: ArrayLike, air: AirProperties = DEFAULT_AIR
    ) -> NDArray[np.complex128]:
        return self.wave_properties(frequency, air)[0]

    def wavenumber(
        self, frequency: ArrayLike, air: AirProperties = DEFAULT_AIR
    ) -> NDArray[np.complex128]:
        return self.wave_properties(frequency, air)[1]

