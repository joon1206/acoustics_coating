#!/usr/bin/env python
"""Rank designs, plot sensitivities/Pareto front, and report simple statistics."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import zscore

from acoustic_coating.config import PROJECT_ROOT
from acoustic_coating.design_space import PARAMETER_COLUMNS, pareto_mask
from acoustic_coating.plotting import plot_design_space


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=PROJECT_ROOT / "results" / "data" / "design_space.csv")
    parser.add_argument("--figures", type=Path, default=PROJECT_ROOT / "results" / "figures")
    args = parser.parse_args()
    results = pd.read_parquet(args.input) if args.input.suffix.lower() == ".parquet" else pd.read_csv(args.input)
    results["pareto_optimal"] = pareto_mask(results["broadband_score"].to_numpy(), results["areal_mass_kg_m2"].to_numpy())
    plot_design_space(results, args.figures)

    outputs = ["mean_absorption_500_4000", "peak_frequency_hz", "mean_tl_db_500_4000", "mass_normalized_score"]
    spearman = results[list(PARAMETER_COLUMNS) + outputs].corr(method="spearman").loc[list(PARAMETER_COLUMNS), outputs]
    spearman.to_csv(args.input.with_name("spearman_correlations.csv"))

    # Standardized multivariate linear relationships are descriptive, not causal.
    x = zscore(np.log10(results[list(PARAMETER_COLUMNS)].clip(lower=1e-12)), axis=0)
    x = np.column_stack([np.ones(len(x)), x])
    coefficients = {}
    for output in outputs:
        y = zscore(results[output].to_numpy())
        coefficients[output] = np.linalg.lstsq(x, y, rcond=None)[0][1:]
    standardized = pd.DataFrame(coefficients, index=PARAMETER_COLUMNS)
    standardized.to_csv(args.input.with_name("standardized_coefficients.csv"))
    results[results["pareto_optimal"]].sort_values("areal_mass_kg_m2").to_csv(args.input.with_name("pareto_designs.csv"), index=False)
    print("Largest absolute Spearman correlations by response:")
    for output in outputs:
        print(f"\n{output}\n{spearman[output].abs().sort_values(ascending=False).head(4).to_string()}")


if __name__ == "__main__":
    main()

