import numpy as np

from acoustic_coating.air import DEFAULT_AIR
from acoustic_coating.jca import JCAMaterial
from acoustic_coating.tmm import air_backed_response, air_layer_matrix, rigid_backed_response, stack_matrix
from acoustic_coating.viscoelastic import ViscoelasticLayer


def test_lossless_air_layer_conserves_power() -> None:
    frequency = np.linspace(100, 5000, 250)
    response = air_backed_response(air_layer_matrix(frequency, 0.123))
    assert np.allclose(response["power_transmission"], 1.0, atol=1e-12)
    assert np.allclose(response["reflection"], 0.0, atol=1e-12)
    assert np.allclose(response["transmission_loss_db"], 0.0, atol=1e-11)


def test_jca_rigid_absorption_is_bounded() -> None:
    frequency = np.linspace(100, 5000, 500)
    material = JCAMaterial(0.95, 20_000, 1.0, 80e-6, 200e-6, 0.015)
    response = rigid_backed_response(material, frequency)
    assert np.all((response["absorption"] >= 0) & (response["absorption"] <= 1))
    assert np.max(response["absorption"]) > 0.2


def test_passive_multilayer_has_no_power_gain() -> None:
    frequency = np.linspace(100, 5000, 500)
    porous = JCAMaterial(0.95, 20_000, 1.0, 80e-6, 200e-6, 0.015)
    polymer = ViscoelasticLayer(2e8, 1100, 0.15, 0.003)
    response = air_backed_response(stack_matrix([porous, polymer], frequency))
    assert np.all(response["power_transmission"] <= 1 + 1e-10)
    assert np.all(np.abs(response["reflection"]) <= 1 + 1e-10)
    assert np.allclose(response["power_balance"], 1.0, atol=1e-10)

