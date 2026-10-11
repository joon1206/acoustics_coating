#!/usr/bin/env python
"""Generate and evaluate a synthetic Latin-hypercube design space."""

import argparse
from pathlib import Path

import numpy as np

from acoustic_coating.config import DEFAULT_CONFIG, PROJECT_ROOT, load_config
from acoustic_coating.design_space import evaluate_batched, evaluate_serial, generate_designs, pareto_mask
from acoustic_coating.parallel import evaluate_parallel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=int, default=10_000, choices=range(1_000, 100_001))
    parser.add_argument("--mode", choices=("serial", "batched", "parallel"), default="batched")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "results" / "data" / "design_space.csv")
    args = parser.parse_args()
    config = load_config(args.config)
    freq_cfg = config["frequency"]
    frequency = np.linspace(freq_cfg["minimum_hz"], freq_cfg["maximum_hz"], freq_cfg["points"])
    designs = generate_designs(args.cases, config["design_bounds"], config["seed"])
    if args.mode == "serial":
        results = evaluate_serial(designs, frequency)
    elif args.mode == "parallel":
        results = evaluate_parallel(designs, frequency, args.workers, args.batch_size)
    else:
        results = evaluate_batched(designs, frequency, args.batch_size)
    results["pareto_optimal"] = pareto_mask(results["broadband_score"].to_numpy(), results["areal_mass_kg_m2"].to_numpy())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.suffix.lower() == ".parquet":
        results.to_parquet(args.output, index=False)
    else:
        results.to_csv(args.output, index=False)
    ranked = results.nlargest(25, "broadband_score")
    ranked.to_csv(args.output.with_name("top_designs.csv"), index=False)
    print(f"Saved {len(results):,} evaluated designs to {args.output}")
    print(ranked[["case_id", "broadband_score", "mean_absorption_500_4000", "mean_tl_db_500_4000", "areal_mass_kg_m2"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()

