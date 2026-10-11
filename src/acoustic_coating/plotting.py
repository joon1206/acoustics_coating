"""Publication-oriented figures for spectra and design-space results."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .jca import JCAMaterial
from .tmm import air_backed_response, rigid_backed_response, stack_matrix
from .viscoelastic import ViscoelasticLayer


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fig.get_layout_engine() is None:
        fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_baseline_diagnostics(
    frequency: np.ndarray,
    porous: JCAMaterial,
    polymer: ViscoelasticLayer,
    output_dir: Path,
) -> None:
    rho = porous.effective_density(frequency)
    bulk = porous.effective_bulk_modulus(frequency)
    z, k = porous.wave_properties(frequency)
    rigid = rigid_backed_response(porous, frequency)
    multi = air_backed_response(stack_matrix([porous, polymer], frequency))

    fig, ax = plt.subplots()
    ax.plot(frequency, rigid["absorption"], label="Porous, rigid-backed")
    ax.plot(frequency, multi["absorption"], label="Porous + polymer, air-backed")
    ax.set(xlabel="Frequency [Hz]", ylabel="Absorption coefficient [-]", ylim=(0, 1), title="Normal-incidence absorption")
    ax.grid(alpha=0.25); ax.legend()
    _save(fig, output_dir / "absorption_spectra.png")

    fig, ax = plt.subplots()
    ax.plot(frequency, multi["transmission_loss_db"])
    ax.set(xlabel="Frequency [Hz]", ylabel="Transmission loss [dB]", title="Porous + viscoelastic multilayer")
    ax.grid(alpha=0.25)
    _save(fig, output_dir / "multilayer_transmission_loss.png")

    pairs = ((rho, "Effective density", "kg/m³", "effective_density.png"), (bulk, "Effective bulk modulus", "Pa", "effective_bulk_modulus.png"))
    for values, title, units, filename in pairs:
        fig, ax = plt.subplots()
        ax.plot(frequency, values.real, label="Real")
        ax.plot(frequency, values.imag, label="Imaginary")
        ax.set(xlabel="Frequency [Hz]", ylabel=f"{title} [{units}]", title=f"JCA {title.lower()}")
        ax.grid(alpha=0.25); ax.legend()
        _save(fig, output_dir / filename)

    fig, axes = plt.subplots(2, 1, sharex=True)
    axes[0].plot(frequency, np.abs(z)); axes[0].set(ylabel="|Zc| [Pa s/m]", title="JCA characteristic impedance")
    axes[1].plot(frequency, np.angle(z, deg=True)); axes[1].set(xlabel="Frequency [Hz]", ylabel="Phase [deg]")
    for ax in axes: ax.grid(alpha=0.25)
    _save(fig, output_dir / "characteristic_impedance.png")

    fig, axes = plt.subplots(2, 1, sharex=True)
    axes[0].plot(frequency, k.real); axes[0].set(ylabel="Re(k) [rad/m]", title="JCA complex wavenumber")
    axes[1].plot(frequency, -k.imag); axes[1].set(xlabel="Frequency [Hz]", ylabel="Attenuation -Im(k) [Np/m]")
    for ax in axes: ax.grid(alpha=0.25)
    _save(fig, output_dir / "complex_wavenumber.png")


def plot_parameter_comparisons(frequency: np.ndarray, base: JCAMaterial, output_dir: Path) -> None:
    studies = {
        "thickness_comparison.png": ("Thickness", [(0.008, "8 mm"), (0.015, "15 mm"), (0.025, "25 mm")]),
        "resistivity_comparison.png": ("Flow resistivity", [(10_000, "10 kPa s/m²"), (20_000, "20 kPa s/m²"), (50_000, "50 kPa s/m²")]),
        "porosity_comparison.png": ("Porosity", [(0.75, "0.75"), (0.90, "0.90"), (0.98, "0.98")]),
    }
    from dataclasses import replace
    for filename, (parameter, values) in studies.items():
        fig, ax = plt.subplots()
        for value, label in values:
            field = {"Thickness": "thickness", "Flow resistivity": "sigma", "Porosity": "phi"}[parameter]
            material = replace(base, **{field: value})
            ax.plot(frequency, rigid_backed_response(material, frequency)["absorption"], label=label)
        ax.set(xlabel="Frequency [Hz]", ylabel="Absorption coefficient [-]", ylim=(0, 1), title=f"Effect of {parameter.lower()}")
        ax.grid(alpha=0.25); ax.legend()
        _save(fig, output_dir / filename)

    fig, ax = plt.subplots()
    for loss in (0.02, 0.10, 0.30, 0.50):
        polymer = ViscoelasticLayer(1e7, 1100, loss, 0.012)
        response = air_backed_response(stack_matrix([base, polymer], frequency))
        ax.plot(frequency, response["transmission_loss_db"], label=f"ηL={loss:.2f}")
    ax.set(xlabel="Frequency [Hz]", ylabel="Transmission loss [dB]", title="Effect of viscoelastic loss factor")
    ax.grid(alpha=0.25); ax.legend()
    _save(fig, output_dir / "loss_factor_comparison.png")


def plot_design_space(results: pd.DataFrame, output_dir: Path) -> None:
    relationships = (
        ("phi", "Porosity [-]"),
        ("sigma_pa_s_m2", "Flow resistivity [Pa s/m²]"),
        ("coating_thickness_m", "Coating thickness [m]"),
        ("polymer_loss_factor", "Polymer loss factor [-]"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), layout="constrained")
    for ax, (column, label) in zip(axes.flat, relationships):
        scatter = ax.scatter(results[column], results["broadband_score"], c=results["areal_mass_kg_m2"], s=8, alpha=0.45)
        ax.set(xlabel=label, ylabel="Broadband score [-]")
        ax.grid(alpha=0.2)
    fig.colorbar(scatter, ax=axes, label="Areal mass [kg/m²]")
    _save(fig, output_dir / "parameter_sensitivity.png")

    fig, ax = plt.subplots()
    ax.scatter(results["areal_mass_kg_m2"], results["broadband_score"], s=8, alpha=0.3, label="Designs")
    if "pareto_optimal" in results:
        pareto = results[results["pareto_optimal"]].sort_values("areal_mass_kg_m2")
        ax.plot(pareto["areal_mass_kg_m2"], pareto["broadband_score"], marker="o", label="Pareto front")
    ax.set(xlabel="Areal mass [kg/m²]", ylabel="Broadband score [-]", title="Performance--mass trade-off")
    ax.grid(alpha=0.25); ax.legend()
    _save(fig, output_dir / "pareto_performance_mass.png")

    parameters = [column for column, _ in relationships]
    correlations = results[parameters + ["mean_absorption_500_4000", "peak_frequency_hz", "mean_tl_db_500_4000", "mass_normalized_score"]].corr(method="spearman").loc[parameters]
    fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
    image = ax.imshow(correlations.to_numpy(), vmin=-1, vmax=1, cmap="coolwarm", aspect="auto")
    ax.set_xticks(range(len(correlations.columns)), correlations.columns, rotation=35, ha="right")
    ax.set_yticks(range(len(parameters)), parameters)
    fig.colorbar(image, ax=ax, label="Spearman ρ")
    ax.set_title("Rank-based sensitivity summary")
    _save(fig, output_dir / "spearman_sensitivity.png")
