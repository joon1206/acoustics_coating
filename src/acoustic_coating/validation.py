"""Numerical and physical sanity checks without experimental-data claims."""

from __future__ import annotations

import numpy as np

from .jca import JCAMaterial
from .metrics import band_mean
from .tmm import air_backed_response, rigid_backed_response, stack_matrix
from .viscoelastic import ViscoelasticLayer


def representative_material(thickness: float = 0.015) -> JCAMaterial:
    return JCAMaterial(0.95, 20_000.0, 1.0, 80e-6, 200e-6, thickness)


def run_validation() -> dict[str, float | bool]:
    """Run limiting, convergence, damping, and passivity demonstrations."""
    fine_frequency = np.linspace(100.0, 5000.0, 2400)
    material = representative_material()
    baseline = rigid_backed_response(material, fine_frequency)
    alpha = baseline["absorption"]

    thin = representative_material(1e-7)
    thin_alpha = rigid_backed_response(thin, fine_frequency)["absorption"]

    convergence = []
    for points in (250, 500, 1000, 2000):
        f = np.linspace(100.0, 5000.0, points)
        a = rigid_backed_response(material, f)["absorption"]
        convergence.append(band_mean(f, a))

    spectral_variations = []
    for loss in (0.02, 0.30):
        # A deliberately soft/thick effective layer places longitudinal features
        # inside the analysis band, making resonance smoothing observable.
        polymer = ViscoelasticLayer(1e7, 1100.0, loss, 0.012)
        response = air_backed_response(stack_matrix([material, polymer], fine_frequency))
        tl = response["transmission_loss_db"]
        # Total variation quantifies sharp spectral structure; damping reduces it.
        spectral_variations.append(float(np.sum(np.abs(np.diff(tl)))))

    multilayer = air_backed_response(
        stack_matrix([material, ViscoelasticLayer(2e8, 1100.0, 0.15, 0.003)], fine_frequency)
    )
    balance = multilayer["power_balance"]
    return {
        "baseline_alpha_min": float(alpha.min()),
        "baseline_alpha_max": float(alpha.max()),
        "baseline_bounded": bool(np.all((alpha >= 0) & (alpha <= 1))),
        "thin_layer_max_absorption": float(thin_alpha.max()),
        "frequency_convergence_error": float(abs(convergence[-1] - convergence[-2])),
        "low_loss_total_variation_db": spectral_variations[0],
        "high_loss_total_variation_db": spectral_variations[1],
        "damping_reduces_spectral_sharpness": bool(spectral_variations[1] < spectral_variations[0]),
        "max_power_balance_error": float(np.max(np.abs(balance - 1.0))),
        "passive_transmission": bool(np.all(multilayer["power_transmission"] <= 1.0 + 1e-9)),
    }

