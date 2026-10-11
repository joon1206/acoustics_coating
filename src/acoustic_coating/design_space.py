"""Synthetic Latin-hypercube design generation and high-throughput evaluation."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
from scipy.integrate import trapezoid
from scipy.stats import qmc

from .air import AirProperties, DEFAULT_AIR
from .jca import JCAMaterial
from .metrics import spectrum_metrics
from .tmm import air_backed_response, rigid_backed_response, stack_matrix
from .viscoelastic import ViscoelasticLayer


PARAMETER_COLUMNS = (
    "phi",
    "sigma_pa_s_m2",
    "alpha_inf",
    "lambda_m",
    "lambda_prime_m",
    "coating_thickness_m",
    "coating_bulk_density_kg_m3",
    "polymer_modulus_pa",
    "polymer_density_kg_m3",
    "polymer_loss_factor",
    "polymer_thickness_m",
)


def generate_designs(
    n_cases: int, bounds: Mapping[str, list[float]], seed: int = 20261006
) -> pd.DataFrame:
    """Generate reproducible hypothetical cases by Latin hypercube sampling."""
    if not 1_000 <= n_cases <= 100_000:
        raise ValueError("n_cases must be between 1,000 and 100,000.")
    missing = set(PARAMETER_COLUMNS) - set(bounds)
    if missing:
        raise KeyError(f"Missing design bounds: {sorted(missing)}")
    lower = np.array([bounds[name][0] for name in PARAMETER_COLUMNS], dtype=float)
    upper = np.array([bounds[name][1] for name in PARAMETER_COLUMNS], dtype=float)
    sample = qmc.LatinHypercube(d=len(PARAMETER_COLUMNS), seed=seed).random(n_cases)
    values = qmc.scale(sample, lower, upper)
    frame = pd.DataFrame(values, columns=PARAMETER_COLUMNS)
    frame.insert(0, "case_id", np.arange(n_cases, dtype=int))
    return frame


def _materials_from_row(row: Mapping[str, float]) -> tuple[JCAMaterial, ViscoelasticLayer]:
    porous = JCAMaterial(
        phi=float(row["phi"]),
        sigma=float(row["sigma_pa_s_m2"]),
        alpha_inf=float(row["alpha_inf"]),
        Lambda=float(row["lambda_m"]),
        Lambda_prime=float(row["lambda_prime_m"]),
        thickness=float(row["coating_thickness_m"]),
    )
    polymer = ViscoelasticLayer(
        modulus=float(row["polymer_modulus_pa"]),
        density=float(row["polymer_density_kg_m3"]),
        loss_factor=float(row["polymer_loss_factor"]),
        thickness=float(row["polymer_thickness_m"]),
    )
    return porous, polymer


def evaluate_design(
    row: Mapping[str, float], frequency: np.ndarray, air: AirProperties = DEFAULT_AIR
) -> dict[str, float]:
    """Reference evaluation of one coating+polymer design."""
    porous, polymer = _materials_from_row(row)
    rigid = rigid_backed_response(porous, frequency, air)
    transmitted = air_backed_response(stack_matrix([porous, polymer], frequency, air), air)
    result = spectrum_metrics(
        frequency,
        rigid["absorption"],
        transmitted["transmission_loss_db"],
    )
    areal_mass = (
        float(row["coating_bulk_density_kg_m3"]) * porous.thickness
        + polymer.density * polymer.thickness
    )
    result["areal_mass_kg_m2"] = areal_mass
    result["mass_normalized_score"] = result["broadband_score"] / areal_mass
    return result


def evaluate_serial(designs: pd.DataFrame, frequency: np.ndarray) -> pd.DataFrame:
    """Clear case-by-case NumPy reference implementation."""
    records = [evaluate_design(row, frequency) for row in designs.to_dict("records")]
    return pd.concat([designs.reset_index(drop=True), pd.DataFrame(records)], axis=1)


def evaluate_batched(
    designs: pd.DataFrame, frequency: np.ndarray, batch_size: int = 512
) -> pd.DataFrame:
    """True case-by-frequency NumPy batching with bounded memory."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    pieces: list[pd.DataFrame] = []
    for start in range(0, len(designs), batch_size):
        block = designs.iloc[start : start + batch_size].reset_index(drop=True)
        pieces.append(_evaluate_vectorized_block(block, frequency))
    return pd.concat(pieces, ignore_index=True)


def _evaluate_vectorized_block(designs: pd.DataFrame, frequency: np.ndarray) -> pd.DataFrame:
    """Evaluate one block; leading dimension is design, trailing is frequency."""
    f = np.asarray(frequency, dtype=float)
    omega = 2.0 * np.pi * f[None, :]
    def column(name: str) -> np.ndarray:
        return designs[name].to_numpy(dtype=float)[:, None]

    phi, sigma, alpha = column("phi"), column("sigma_pa_s_m2"), column("alpha_inf")
    lam, lam_p, thickness = column("lambda_m"), column("lambda_prime_m"), column("coating_thickness_m")
    air = DEFAULT_AIR
    viscous_root = np.sqrt(1 + 1j * 4 * alpha**2 * air.eta * air.rho0 * omega / (sigma**2 * lam**2 * phi**2))
    rho = alpha * air.rho0 / phi * (1 + sigma * phi / (1j * omega * air.rho0 * alpha) * viscous_root)
    thermal_root = np.sqrt(1 + 1j * lam_p**2 * air.Cp * air.rho0 * omega / (16 * air.kappa))
    thermal_bracket = 1 - 1j * 8 * air.kappa / (lam_p**2 * air.Cp * air.rho0 * omega) * thermal_root
    bulk = (air.gamma * air.P0 / phi) / (air.gamma - (air.gamma - 1) / thermal_bracket)
    zp = np.sqrt(rho * bulk)
    kp = omega * rho / zp
    flip = np.imag(kp) > 0
    zp, kp = np.where(flip, -zp, zp), np.where(flip, -kp, kp)

    cp, sp = np.cos(kp * thickness), np.sin(kp * thickness)
    ap, bp, cp21, dp = cp, 1j * zp * sp, 1j * sp / zp, cp
    zin = ap / cp21
    reflection_rigid = (zin - air.Z0) / (zin + air.Z0)
    absorption = np.clip(1 - np.abs(reflection_rigid) ** 2, 0, 1)

    modulus = column("polymer_modulus_pa") * (1 + 1j * column("polymer_loss_factor"))
    density = column("polymer_density_kg_m3")
    zsolid = np.sqrt(density * modulus)
    ksolid = omega * density / zsolid
    solid_flip = np.imag(ksolid) > 0
    zsolid, ksolid = np.where(solid_flip, -zsolid, zsolid), np.where(solid_flip, -ksolid, ksolid)
    solid_thickness = column("polymer_thickness_m")
    cs, ss = np.cos(ksolid * solid_thickness), np.sin(ksolid * solid_thickness)
    ass, bs, css, ds = cs, 1j * zsolid * ss, 1j * ss / zsolid, cs

    # Matrix product porous @ solid, expanded to avoid a large (...,2,2) array.
    a = ap * ass + bp * css
    b = ap * bs + bp * ds
    c = cp21 * ass + dp * css
    d = cp21 * bs + dp * ds
    denominator = a + b / air.Z0 + c * air.Z0 + d
    transmission = 2 / denominator
    tau = np.maximum(np.abs(transmission) ** 2, np.finfo(float).tiny)
    tl = -10 * np.log10(tau)

    mask = (f >= 500) & (f <= 4000)
    band_f = f[mask]
    mean_alpha = trapezoid(absorption[:, mask], band_f, axis=1) / (band_f[-1] - band_f[0])
    mean_tl = trapezoid(tl[:, mask], band_f, axis=1) / (band_f[-1] - band_f[0])
    peak_index = np.argmax(absorption[:, mask], axis=1)
    max_alpha = np.take_along_axis(absorption[:, mask], peak_index[:, None], axis=1)[:, 0]
    min_tl = np.min(tl[:, mask], axis=1)
    areal_mass = (
        designs["coating_bulk_density_kg_m3"].to_numpy() * designs["coating_thickness_m"].to_numpy()
        + designs["polymer_density_kg_m3"].to_numpy() * designs["polymer_thickness_m"].to_numpy()
    )
    score = mean_alpha + mean_tl / 50
    metrics = pd.DataFrame({
        "mean_absorption_500_4000": mean_alpha,
        "max_absorption_500_4000": max_alpha,
        "peak_frequency_hz": band_f[peak_index],
        "mean_tl_db_500_4000": mean_tl,
        "min_tl_db_500_4000": min_tl,
        "broadband_score": score,
        "areal_mass_kg_m2": areal_mass,
        "mass_normalized_score": score / areal_mass,
    })
    return pd.concat([designs, metrics], axis=1)


def pareto_mask(performance: np.ndarray, mass: np.ndarray) -> np.ndarray:
    """Return non-dominated designs: maximize performance, minimize mass."""
    performance = np.asarray(performance, dtype=float)
    mass = np.asarray(mass, dtype=float)
    order = np.argsort(mass, kind="stable")
    keep = np.zeros(len(performance), dtype=bool)
    best = -np.inf
    for index in order:
        if performance[index] > best:
            keep[index] = True
            best = performance[index]
    return keep

