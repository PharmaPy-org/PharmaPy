# Frozen E. coli fed-batch reconciliation assessment

This configuration transfers the existing F10/F11 experimental reproduction into
`rate-reconciled-culture`. The source coefficients, initial phase amounts, feeds,
samples, base additions and solver settings are unchanged. The parent README
records the original source processing corrections and biological limitations.
No coefficients are fitted here. The original direct-kinetic example remains intact.

Run `workflow.ipynb` with Run All. It uses the same six forward-only steps and
native `build_bioreactor -> assembly.solve -> SemiBatchReactor.solve_unit` path.
Inputs are under `inputs/` and `inputs/F11/`; the unchanged parent
`../inputs/thermo.json` is shared. Outputs are CSV trajectories and PNG/SVG figures.
There are no notebook-local mechanisms or solvers.

## Input field guide: rate-reconciliation template

Use `inputs/mechanism.json` for F10 or `inputs/F11/mechanism.json` for F11.
Both share `../inputs/thermo.json`; do not duplicate or silently substitute its
properties. These are literature-calibrated reproduction cases with experimental
observations, not independently validated predictions for arbitrary organisms.
See the [parent source and assumptions](../README.md#source-and-corrections).

| Section | Category | What to configure |
| --- | --- | --- |
| `schema_version`, `mechanism` | Required for this model | Keep version 3 and `rate-reconciled-culture` for this formulation. |
| `$comment` | Descriptive | Explain the case without changing scientific inputs. |
| `operation` | Required for this model | Working-volume basis, carrier, initial mass/volume, runtime, conditions, and F10/F11 recipe selection. |
| `state` | Required for this model | Initial species amounts and dry biomass. Here phase biomass and `biomass_kg` describe the same population and must agree. |
| `parameters` | Required for this model | Referenced source kinetic coefficients and yields; empty groups remain empty. |
| `model.kinetics` | Required for this model | Rate relationships, target outputs, exchange mappings and dry-mass basis. Preserve explicit unit conversions. |
| `model.reconciliation`, `network` | Required for this model | Yield balances, reaction bounds, targeted internal rates, reconciliation policy and reference scales. |
| `rate_provider` | Required for this model | Select the rule graph and declare its output units; its identifier/version also document provenance. |
| `recipes` | Required for this model | Feed flows, base additions and sampling times/volumes. These are operating inputs, not adjustable fitting corrections. |
| `model.pathways`, `transport`, `flux_time_unit`, blank cell-count fields | Optional | Retain the inactive empty/null slots. Do not replace intentional zero rates or the zero product mapping with blanks. |
| `numerical` | Numerical | Solver, requested time grid, tolerances, maximum step and Jacobian settings. |
| Shared `thermo.json` | Required for this model / optional properties | Preserve molecular-weight bookkeeping and liquid densities; unused thermal properties stay blank. |

Reconciliation weights/scales, constraints and secondary selection policies can
affect accepted fluxes. They remain model choices, not harmless solver controls.
The `network.objective` field does not make this formulation growth-maximizing.
The configured network represents reduced yield balances, not full elemental
conservation. Reusing the template requires appropriate biology and operating
records; do not borrow these values for an unrelated experiment.

### Mathematical transfer

Let `q_s`, `q_ox`, `q_a`, `q_b`, and `q_ac` be the source specific glucose uptake,
oxidative glucose use, two yeast-fraction uptakes, and acetate uptake in g/(gDW h).
The source predicts

```
q_overflow = q_s - q_ox
q_acetate_net = Y_AS*q_overflow - q_ac
mu = Y_XSem*(q_ox - qm) + Y_XYEF_A*q_a + Y_XYEF_B*q_b + Y_XA*q_ac
```

The configured network imposes these same yield relationships as linear balances.
The glucose and acetate pools use mmol; yeast-fraction and biomass pools use
formal mmol with the existing 1 g/mol bookkeeping basis. Internal mass-based
reaction extents are converted by 1000 mmol/formal mol. The growth column uses
1/h and consumes 1000 formal mmol biomass per unit specific growth. The exchange
mappings undo these coordinate changes before native kg/s balances are applied.

The source's kinetic rates target glucose/acetate exchanges, yeast-fraction
uptakes, oxidative glucose use, acetate use, maintenance and net growth. Remaining
rates are determined by the balances. A unit-scaled quadratic objective reconciles
these targets; its reference scales are declared in JSON. Compatible targets
have objective zero regardless of positive weights. The network's `objective`
entry is not a growth-maximization instruction for this closure.

Finite bounds follow from the source maximum uptake rates and yields. The lower
net-growth bound is `-qm*Y_XSem`, preserving source biomass loss under starvation.
No density cap, new survival law, ATP requirement or artificial cell mass is added.
The required product mapping has zero flux because this source has no separate
product model.

These are **reduced yield balances, not an independently established intracellular
or elemental network**. The maintenance term retains the source's mathematical
meaning as a biomass-growth offset; it is not an identified ATP maintenance flux.

## General implementation changes

The existing `state.biomass_kg` selects a dry-mass population in reconciliation.
Its rate basis converts to kgDW directly; cell-count configurations retain their
existing behavior. A biomass population cannot silently use cell-count rules or
cell-count initial states. Native sampling removes the same fraction of biomass
and liquid inventories. In this example the phase biomass and population state
represent the same quantity, whose agreement is checked explicitly.

The closure also checks whether supplied targets and balances uniquely specify a
feasible zero-residual optimum. If so, it returns that optimum directly. Otherwise
it retains the existing constrained optimization and tie-break. This avoids
optimization noise in an already-consistent model without changing its objective.
No native reactor class, schema template, or source-specific architectural branch
was added.

## Acceptance criteria fixed before the campaign

- Full concentrations and volume, including both event sides: agree with the
  freshly executed direct model at `atol=2e-5, rtol=2e-5`.
- Independent concentration-based Radau balances: same tolerance away from jumps.
- Population biomass agrees with phase biomass; sampling preserves concentrations
  and removes proportional inventories; exported inventories remain nonnegative
  to roundoff.
- Accepted-state target deviations below `2e-6` after scaling by `max(1, |target|)`;
  network balance residual below `2e-7` in declared coordinates.
- Experimental RMSE changes by less than `2e-4 g/L` from the frozen direct model.
  This tests preservation of prior agreement, not a new industry accuracy standard.
- Equivalent physical units and existing cell-based reconciliation regressions pass.

Generate the assessment and comparison figures with:

```sh
BIOREACTOR_RECONCILIATION_OUTPUT=examples/bioreactors/ecoli_ye_fed_batch/reconciliation/outputs \
MPLBACKEND=Agg python -m pytest tests/Bioreactor/test_bioreactor_ecoli_reconciliation.py -q
```

`outputs/assessment.json` records measured results. Successful preservation supports
experimental reproduction through reconciliation. It does not show reconciliation
can repair incorrect kinetics, establish complete elemental conservation, validate
oxygen control, or resolve NISTCHO's biological-model transfer failure.

## Reference-branch outcome (historical)

The full reference branch reported both runs passing the frozen-model preservation criteria.
These historical numbers are not verification of subsequent backend changes. Maximum concentration
changes from the direct implementation are 1.72e-6 g/L (F10) and 1.08e-6 g/L (F11).
All checked accepted-state targets are retained exactly, with maximum network
balance residual 2.40e-13. Experimental RMSEs remain:

| g/L | F10 | F11 |
| --- | --- | --- |
| Biomass | 1.79174 | 1.65165 |
| Glucose | 1.87115 | 1.63870 |
| Acetate | 0.10722 | 0.10170 |

The two-run forward workflow took approximately 13.7 s including export. Tightened
integration reduces the F10 independent error below 8e-7 g/L; maximum-step
refinement also remains within the original acceptance gate. Rerun the command above for current assessment and comparison files. These
generated files are excluded from Git. See the validation ladder for the
qualification scope; the historical test count is not a current branch sign-off.

## Kinetics selection

Both F10 and F11 now select `monod` for `q_s`, `q_a`, `q_b` and the nested uptake term in `q_ac`. The existing `additive-inhibition`, `capacity-limited-rate` and `yield-sum` compositions remain explicit. Concentrations/affinities use their declared mass bases and uptake maxima their source-specific normalization. Edit the relevant `inputs/mechanism.json` or `inputs/F11/mechanism.json`; Step 2 displays their parameter/rule selections. Retain each run's feed history, source coefficients and reconciliation constraints to reproduce the published case.

The notebook exposes `kinetic_forms = relationship_catalog()` with supported names, aliases, required inputs, equations and units. For rule graphs, change `relationship` and supply every required input; for pathway optimization, keep the distinct uptake-constraint interface. No parameter, yield or physical-basis conversion is inferred.

Named Monod/Haldane forms are equivalent to the former rational expressions on the physical nonnegative-substrate domain. They inherit the existing zero-uptake extension for negative trial substrate; negative inventories are not validated biological states. Signed product inhibition and custom hybrid equations remain explicit. The numerical and independent-reference checks retain their existing acceptance budgets.

### Neutral kinetic correction

The kinetic input files expose `parameters.corrections.theta = 0` (dimensionless).
The correction multiplies the selected substrate-dependent rate by
`exp(theta * substrate / (substrate + reference_concentration))`; zero preserves
the original rate. Each declaration explicitly binds the concentration and its
units. The ordinary kinetic examples reuse that rate's half-saturation
concentration as the reference scale, a configurable assumption rather than a
calibrated value. Doyle's `hybrid.json` retains its separate documented scale.
A shared theta changes all declared corrections in that file; derived yields
and balance expressions continue to use the corrected rates. Change individual
bindings to use independent coefficients if required. Selecting another kinetic
relationship also requires reviewing its input slots and correction bindings.

Pathway examples now connect empirical uptake capacities through
`model.pathways.rule_graph` and the existing `rule-graph` provider, with
`parameters.corrections.theta = 0`. Edit the capacity expression's `relationship`
and inputs to select a kinetic form; constant capacities use a parameter node.
The correction scale is the Monod affinity, or an explicit configurable
`parameters.corrections.reference_concentration = 1` [mmol/L] for constant
capacities. These are normalization assumptions, not fitted values. Monod's
`greater-select.threshold` preserves the native depletion cutoff; update it
alongside `model.pathways.depletion_tolerance`. Negative trial concentrations
use the existing zero extension. The oxygen `transfer-cap`, inventory limits,
and metabolic balances remain independent constraints, even for nonzero theta.
