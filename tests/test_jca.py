import numpy as np
import pytest

from acoustic_coating.jca import JCAMaterial


@pytest.fixture
def material() -> JCAMaterial:
    return JCAMaterial(0.95, 20_000, 1.0, 80e-6, 200e-6, 0.015)


def test_jca_is_vectorized_finite_and_passive(material: JCAMaterial) -> None:
    frequency = np.geomspace(20, 20_000, 501)
    rho = material.effective_density(frequency)
    bulk = material.effective_bulk_modulus(frequency)
    z, k = material.wave_properties(frequency)
    assert rho.shape == bulk.shape == z.shape == k.shape == frequency.shape
    assert np.all(np.isfinite(rho)) and np.all(np.isfinite(bulk))
    assert np.all(np.real(z) > 0)
    assert np.all(np.imag(k) <= 1e-12)
    assert np.all(-np.imag(rho) >= 0)


def test_jca_qualitative_limits(material: JCAMaterial) -> None:
    frequency = np.array([100.0, 1000.0, 10_000.0])
    rho = material.effective_density(frequency)
    bulk = material.effective_bulk_modulus(frequency)
    assert np.all(rho.real > 1.0)
    assert np.all(bulk.real > 50_000.0)
    assert abs(rho.imag[-1]) < abs(rho.imag[0])


def test_parameter_validation() -> None:
    with pytest.raises(ValueError):
        JCAMaterial(1.1, 20_000, 1.0, 80e-6, 200e-6, 0.015)

