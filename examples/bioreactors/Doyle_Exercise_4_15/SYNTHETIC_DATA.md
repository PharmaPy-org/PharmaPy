# Synthetic parameter-estimation data: Doyle Exercise 4.15

Run `PYTHONPATH=../../.. python generate_synthetic.py` from this example folder
using the PharmaPy environment. Outputs are in `outputs/synthetic_data/` and
are ignored by Git. The generator performs no fitting. The companion
`calibration.ipynb` and `calibrate.py` select these assets for a shared-estimator
workflow; see the example README.

The independent textbook concentration-balance implementation in
`run_benchmarks.py` generates truth with DOP853; Radau independently checks it.
These are same-model synthetic qualification data, not experimental measurements.
Source parameters remain mu_max=0.20/h, Ks=1 g/L, Y=0.5 g/g and feed=10 g/L
from Exercise 4.15, printed p.67, Seborg et al., fourth edition (2017).
Initial biomass/substrate are 4.5/1 g/L in every run. Dilution changes at t=0.

The default generator retains mechanistic truth (theta=0). To generate the
explicit synthetic hybrid, use `--theta 0.1823215567939546` (ln(1.2)); `--output`
selects a separate destination. Both growth and substrate consumption use the
same correction `exp(theta*S/(S+1 g/L))`. The feature scale is read from the
existing hybrid configuration and theta must stay in its declared domain.
The manifest records theta, the feature scale, the effective design and source
checksums; observation checksums use filenames relative to the dataset folder
so the assets can be relocated.

## Explicit experimental-design choices

`inputs/synthetic_design.json` declares all sampling/noise settings. The new
0.06 and 0.14/h fitting conditions and 0.12/h held-out condition are our synthetic
design, not textbook experiments. The positive steady-state substrate spans
approximately 0.43 to 2.33 g/L, bracketing Ks=1 g/L. This improves excitation
over +/-10% steps but does not by itself establish joint identifiability.
Fifteen sample times concentrate observations on transients while retaining
late samples through 100 h. Known inputs and initial states are treated as exact;
the t=0 observation is still noisy and must not replace the known initial state.

For each nonzero noise level there are 20 independently seeded realizations.
Realizations are alternative noisy versions of an experiment, not 20 actual
biological replicates to pool indiscriminately. Seed 4152026 is an arbitrary,
fixed reproducibility choice. The same standardized draws are paired across
1% and 5% for controlled noise-level comparisons. Errors are independent across
species, times, experiments and realizations, but not across those paired levels.

Noise standard deviation is sqrt(0.001^2 + (fraction * truth)^2) g/L. Thus 1%
and 5% denote the relative component, not a fixed error magnitude or bound.
The 0.001 g/L absolute component is a declared synthetic measurement-resolution
assumption, not an experimentally sourced assay specification. Gaussian errors
are neither clipped nor rescaled to force their realized RMS to equal a target.
The zero-noise case has exactly zero error and sigma; do NOT divide by that
sigma when fitting. Use explicitly declared numerical residual scales instead.

## Files and use

- `truth.csv`: noise-free sampled concentrations, kept separate from observations.
- `noise_0pct.csv`: one exact dataset, 90 observations.
- `noise_1pct.csv`, `noise_5pct.csv`: 20 realizations each, 1800 observations each.
- `manifest.json`: truth parameters, experiment settings, solver agreement,
  row/negative-value checks, versions and SHA-256 checksums.

Each observation identifies experiment, fit/validation split, realization,
actual time in hours, variable, value in g/L and generating standard deviation
in g/L. Biomass and substrate only; the inactive product species is omitted.
Split by entire experiment, not random rows. Fit using only `split=fit`; assess
the withheld experiment afterward. Known generating uncertainties are supplied
as oracle weights for this initial qualification, not estimated assay errors.

The worked workflow verifies noise-free parameter recovery, noisy fitting and
held-out predictions, with separate stress tests for initialization and numerical
refinement. A full distributional study over the declared realizations remains
distinct from this bounded numerical qualification.
Generating these datasets alone establishes neither parameter identifiability
nor successful parameter estimation. No production code or equations changed.
