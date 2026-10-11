# Research notes

## Equations and assumptions

The implementation uses the JCA dynamic density and Champoux–Allard bulk modulus shown in the README, with $Z_c=\sqrt{\widetilde\rho\widetilde K}$ and $k_c=\omega\widetilde\rho/Z_c$. For $e^{j\omega t}$ and forward waves $e^{-jkx}$, the coupled square-root signs enforce $\operatorname{Im}(k)\le0$. Layer matrices use positive-$j$ off-diagonal terms. The polymer uses $M^*=M'(1+j\eta_L)$.

The model assumes normal-incidence plane waves, a rigid porous frame, homogeneous layers, perfect interfaces, and a constant polymer loss factor. Every design-space range is hypothetical; none is a measured formulation or commercial material.

## Validation performed

- JCA finiteness, vectorization, passive branch, and qualitative trends.
- Bounded absorption and lossless-air conservation.
- Extremely thin rigid-backed coating limit.
- Frequency-grid convergence.
- Passive multilayer power balance.
- Serial/batched numerical equivalence.
- Loss-factor comparison of resonance structure.

Run `python scripts/run_single_case.py` and `pytest` to reproduce the numerical values. The generated `results/data/validation.json` is the machine-readable record.

## Numerical findings

For the 15 mm representative JCA coating over 100–5000 Hz, rigid-backed absorption spans 0.007204–0.859461. An extremely thin 0.1 µm layer has maximum absorption $2.10\times10^{-6}$. The final frequency-refinement step changes the 500–4000 Hz average by $2.53\times10^{-5}$. A passive porous/polymer stack closes power balance to $1.11\times10^{-16}$ and has no transmission gain.

For a deliberately soft/thick polymer example that places longitudinal structure in-band, increasing $\eta_L$ from 0.02 to 0.30 reduces TL total variation from 75.36 to 34.80 dB. This supports the expected smoothing/broadening interpretation without claiming monotonic improvement at every frequency.

The fixed-seed 10,000-case sweep produces 38 mass/performance Pareto designs. The highest combined score is 1.824819, with mean absorption 0.883395, mean TL 47.071 dB, and areal mass 18.214 kg/m². The lightest Pareto point is 2.195 kg/m². These are synthetic optima within independent effective-property bounds, not manufacturable recipes.

## HPC scaling findings

On the verification machine, 10,000 cases require 5.759 s serial, 1.627 s batched, and 3.440 s with four worker processes. Relative to serial, the speedups are 3.54× and 1.67× respectively; four-process efficiency is 0.418. At 100 cases the process mode takes 2.61 s versus 0.063 s serial, and at 1,000 it takes 2.66 s versus 0.585 s serial: startup and serialization overhead dominate. NumPy batching is the fastest of these modes at every checked size. Timings are machine-specific and should be regenerated.

## Physical interpretations

Within the sampled box, Spearman correlation identifies coating thickness as the leading mean-absorption control ($\rho_s=0.920$) and leading inverse peak-frequency control ($\rho_s=-0.783$). Polymer thickness most strongly controls mean TL ($\rho_s=0.854$). Coating bulk density has the strongest negative association with mass-normalized score ($\rho_s=-0.856$). Thermal length and flow resistivity have weaker, range-dependent effects on mean absorption. Polymer loss factor has little monotonic correlation across the broad independent sample, although the controlled resonance demonstration shows its local smoothing role.

Interpret effective parameters rather than mineral names: thickness and flow resistivity shape dissipation and quarter-wave behavior; porosity, tortuosity, and characteristic lengths change dynamic inertia and thermal/viscous relaxation; polymer modulus, thickness, density, and loss shift and damp longitudinal resonances. Correlations describe this sampled range and do not establish causal chemistry/property relations.

## Limitations and next steps

The study lacks experimental calibration, parameter covariance, microstructural prediction, frame elasticity, shear waves, and oblique/random incidence. Immediate next steps are impedance-tube measurement and inverse fitting of effective parameters, followed by uncertainty quantification and frequency-dependent polymer rheology. Biot poroelasticity and finite-element structural acoustics are appropriate later extensions for frame motion, finite geometry, bonding, and coupled structural modes.

