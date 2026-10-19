# E. coli source documentation and evidence

This package documents the scientific source and the available reproduction
evidence. It does not supply raw experimental observations or establish
independent experimental validation.

## Scientific source

Radhakrishnan Mahadevan, Jeremy S. Edwards, and Francis J. Doyle III (2002).
*Dynamic Flux Balance Analysis of Diauxic Growth in Escherichia coli*.
Biophysical Journal, 83, 1331–1340.
[Original paper](https://doi.org/10.1016/S0006-3495(02)73903-9).

The example targets the qualitative diauxic behavior associated with the paper's
Figure 4 relaxed-flux-rate static-optimization formulation. It is not a
reproduction of every formulation in the paper: the rate-constrained Figure 5
case and full-horizon dynamic optimization are distinct comparisons.

## Model inputs and numerical benchmarks

- [mechanism.json](../inputs/mechanism.json): pathway coefficients, bounds,
  initial state, and transport definitions used by this example.
- [thermo.json](../inputs/thermo.json): native phase-property inputs.
- [workflow.ipynb](../workflow.ipynb): native object construction, execution, and reports.

The notebook loads the three forward inputs and exports local trajectories.
Published panels, historical trajectories and generated comparison figures are
excluded from this branch. Consult the original paper via its DOI.

No raw experimental trajectory dataset or measured uncertainty accompanies this
batch example. Quantitative validation requires independent observations,
provenance, uncertainty and declared comparison criteria.
