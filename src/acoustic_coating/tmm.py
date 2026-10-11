"""Vectorized normal-incidence transfer-matrix calculations."""

from collections.abc import Iterable
from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .air import AirProperties, DEFAULT_AIR


class Layer(Protocol):
    thickness: float

    def wave_properties(self, frequency: ArrayLike, air: AirProperties = DEFAULT_AIR): ...


def layer_matrix(
    impedance: ArrayLike, wavenumber: ArrayLike, thickness: float
) -> NDArray[np.complex128]:
    """Matrix mapping [p,u] at the right face to the left face.

    For exp(+j omega t), a forward wave is exp(-j k x), giving positive-j
    off-diagonal terms.  Input arrays must share a frequency shape.
    """
    z = np.asarray(impedance, dtype=complex)
    k = np.asarray(wavenumber, dtype=complex)
    kd = k * thickness
    matrix = np.empty(k.shape + (2, 2), dtype=complex)
    matrix[..., 0, 0] = np.cos(kd)
    matrix[..., 0, 1] = 1j * z * np.sin(kd)
    matrix[..., 1, 0] = 1j * np.sin(kd) / z
    matrix[..., 1, 1] = np.cos(kd)
    return matrix


def air_layer_matrix(
    frequency: ArrayLike, thickness: float, air: AirProperties = DEFAULT_AIR
) -> NDArray[np.complex128]:
    omega = air.omega(frequency)
    z = np.full(omega.shape, air.Z0, dtype=complex)
    k = omega / air.c0
    return layer_matrix(z, k, thickness)


def material_layer_matrix(
    layer: Layer, frequency: ArrayLike, air: AirProperties = DEFAULT_AIR
) -> NDArray[np.complex128]:
    z, k = layer.wave_properties(frequency, air)
    return layer_matrix(z, k, layer.thickness)


def stack_matrix(
    layers: Iterable[Layer], frequency: ArrayLike, air: AirProperties = DEFAULT_AIR
) -> NDArray[np.complex128]:
    frequencies = np.asarray(frequency, dtype=float)
    total = np.zeros(frequencies.shape + (2, 2), dtype=complex)
    total[..., 0, 0] = total[..., 1, 1] = 1.0
    for layer in layers:
        total = np.matmul(total, material_layer_matrix(layer, frequencies, air))
    return total


def rigid_backed_response(
    layer_or_matrix: Layer | NDArray[np.complex128],
    frequency: ArrayLike | None = None,
    air: AirProperties = DEFAULT_AIR,
) -> dict[str, NDArray]:
    """Surface impedance, reflection, and clipped absorption at a rigid wall."""
    if frequency is None:
        matrix = np.asarray(layer_or_matrix, dtype=complex)
    else:
        matrix = material_layer_matrix(layer_or_matrix, frequency, air)  # type: ignore[arg-type]
    surface_impedance = matrix[..., 0, 0] / matrix[..., 1, 0]
    reflection = (surface_impedance - air.Z0) / (surface_impedance + air.Z0)
    raw_absorption = 1.0 - np.abs(reflection) ** 2
    absorption = np.clip(np.real(raw_absorption), 0.0, 1.0)
    return {
        "surface_impedance": surface_impedance,
        "reflection": reflection,
        "absorption": absorption,
    }


def air_backed_response(
    matrix: NDArray[np.complex128], air: AirProperties = DEFAULT_AIR
) -> dict[str, NDArray]:
    """Reflection, pressure transmission, power transmission, absorption, TL.

    The incident and transmitted half-spaces are the same air, so their real
    impedances cancel in the power ratio and tau=|t|^2.
    """
    m = np.asarray(matrix, dtype=complex)
    a, b, c, d = m[..., 0, 0], m[..., 0, 1], m[..., 1, 0], m[..., 1, 1]
    denominator = a + b / air.Z0 + c * air.Z0 + d
    transmission = 2.0 / denominator
    reflection = (a + b / air.Z0 - c * air.Z0 - d) / denominator
    tau = np.maximum(np.abs(transmission) ** 2, np.finfo(float).tiny)
    reflected_power = np.abs(reflection) ** 2
    absorption = np.clip(1.0 - reflected_power - tau, 0.0, 1.0)
    return {
        "reflection": reflection,
        "pressure_transmission": transmission,
        "power_transmission": tau,
        "absorption": absorption,
        "transmission_loss_db": -10.0 * np.log10(tau),
        "power_balance": reflected_power + tau + absorption,
    }
