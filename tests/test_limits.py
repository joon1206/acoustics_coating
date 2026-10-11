import numpy as np
import pandas as pd
from scipy.integrate import trapezoid

from acoustic_coating.design_space import evaluate_batched, evaluate_serial, generate_designs
from acoustic_coating.jca import JCAMaterial
from acoustic_coating.tmm import rigid_backed_response
from acoustic_coating.validation import run_validation


def test_extremely_thin_rigid_backed_layer_approaches_reflection() -> None:
    frequency = np.linspace(100, 5000, 300)
    thin = JCAMaterial(0.95, 20_000, 1.0, 80e-6, 200e-6, 1e-9)
    alpha = rigid_backed_response(thin, frequency)["absorption"]
    assert np.max(alpha) < 1e-4


def test_frequency_resolution_converges() -> None:
    material = JCAMaterial(0.95, 20_000, 1.0, 80e-6, 200e-6, 0.015)
    means = []
    for n in (500, 1000, 2000):
        f = np.linspace(100, 5000, n)
        a = rigid_backed_response(material, f)["absorption"]
        band = (f >= 500) & (f <= 4000)
        means.append(trapezoid(a[band], f[band]) / (f[band][-1] - f[band][0]))
    assert abs(means[-1] - means[-2]) < abs(means[-2] - means[-3])
    assert abs(means[-1] - means[-2]) < 1e-3


def test_batched_matches_serial() -> None:
    bounds = {
        "phi": [0.7, 0.98], "sigma_pa_s_m2": [5000, 80000], "alpha_inf": [1, 2],
        "lambda_m": [30e-6, 200e-6], "lambda_prime_m": [60e-6, 500e-6],
        "coating_thickness_m": [0.005, 0.03], "coating_bulk_density_kg_m3": [250, 1000],
        "polymer_modulus_pa": [5e7, 2e9], "polymer_density_kg_m3": [900, 1600],
        "polymer_loss_factor": [0.02, 0.5], "polymer_thickness_m": [0.0005, 0.005],
    }
    designs = generate_designs(1000, bounds, seed=7).iloc[:5]
    frequency = np.linspace(100, 5000, 180)
    serial = evaluate_serial(designs, frequency)
    batched = evaluate_batched(designs, frequency, batch_size=3)
    columns = ["mean_absorption_500_4000", "mean_tl_db_500_4000", "broadband_score"]
    assert np.allclose(serial[columns], batched[columns], rtol=1e-11, atol=1e-11)


def test_loss_factor_smooths_in_band_resonance() -> None:
    validation = run_validation()
    assert validation["damping_reduces_spectral_sharpness"]
