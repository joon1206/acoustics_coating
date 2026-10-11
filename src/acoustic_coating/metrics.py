"""Engineering response metrics for spectra."""

import numpy as np
from numpy.typing import ArrayLike
from scipy.integrate import trapezoid


def band_mean(frequency: ArrayLike, values: ArrayLike, low: float = 500, high: float = 4000) -> float:
    f = np.asarray(frequency, dtype=float)
    y = np.asarray(values, dtype=float)
    mask = (f >= low) & (f <= high)
    if np.count_nonzero(mask) < 2:
        raise ValueError("At least two frequency samples are required in the band.")
    return float(trapezoid(y[mask], f[mask]) / (f[mask][-1] - f[mask][0]))


def spectrum_metrics(
    frequency: ArrayLike,
    absorption: ArrayLike,
    transmission_loss_db: ArrayLike,
    low: float = 500,
    high: float = 4000,
) -> dict[str, float]:
    f = np.asarray(frequency, dtype=float)
    alpha = np.asarray(absorption, dtype=float)
    tl = np.asarray(transmission_loss_db, dtype=float)
    mask = (f >= low) & (f <= high)
    peak_idx = int(np.argmax(alpha[mask]))
    band_f = f[mask]
    mean_alpha = band_mean(f, alpha, low, high)
    mean_tl = band_mean(f, tl, low, high)
    return {
        "mean_absorption_500_4000": mean_alpha,
        "max_absorption_500_4000": float(alpha[mask][peak_idx]),
        "peak_frequency_hz": float(band_f[peak_idx]),
        "mean_tl_db_500_4000": mean_tl,
        "min_tl_db_500_4000": float(np.min(tl[mask])),
        "broadband_score": float(mean_alpha + mean_tl / 50.0),
    }
