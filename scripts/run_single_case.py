#!/usr/bin/env python
"""Generate baseline spectra, comparison figures, and validation summary."""

import argparse
import json
from pathlib import Path

import numpy as np

from acoustic_coating.config import PROJECT_ROOT
from acoustic_coating.plotting import plot_baseline_diagnostics, plot_parameter_comparisons
from acoustic_coating.validation import representative_material, run_validation
from acoustic_coating.viscoelastic import ViscoelasticLayer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--points", type=int, default=1600)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "results" / "figures")
    args = parser.parse_args()
    frequency = np.linspace(100.0, 5000.0, args.points)
    porous = representative_material(0.015)
    polymer = ViscoelasticLayer(2.0e8, 1100.0, 0.15, 0.003)
    plot_baseline_diagnostics(frequency, porous, polymer, args.output)
    plot_parameter_comparisons(frequency, porous, args.output)
    validation = run_validation()
    validation_path = PROJECT_ROOT / "results" / "data" / "validation.json"
    validation_path.parent.mkdir(parents=True, exist_ok=True)
    validation_path.write_text(json.dumps(validation, indent=2), encoding="utf-8")
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()

