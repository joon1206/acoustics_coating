#!/usr/bin/env python
"""Compare serial, batched NumPy, and CPU-process design evaluation."""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from acoustic_coating.config import DEFAULT_CONFIG, PROJECT_ROOT, load_config
from acoustic_coating.design_space import generate_designs
from acoustic_coating.parallel import benchmark_modes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="+", default=[100, 1000, 10_000])
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--frequency-points", type=int, default=320)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "results" / "data" / "benchmark.csv")
    args = parser.parse_args()
    if any(size < 1 or size > 100_000 for size in args.sizes):
        parser.error("Each size must be between 1 and 100,000.")
    config = load_config(args.config)
    frequency = np.linspace(100, 5000, args.frequency_points)
    all_designs = generate_designs(max(1000, max(args.sizes)), config["design_bounds"], config["seed"])
    tables = [benchmark_modes(all_designs.iloc[:size].copy(), frequency, args.workers, args.batch_size) for size in args.sizes]
    benchmark = pd.concat(tables, ignore_index=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    benchmark.to_csv(args.output, index=False)
    print(benchmark.to_string(index=False))


if __name__ == "__main__":
    main()

