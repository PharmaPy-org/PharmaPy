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

## Mathematical transfer

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

## Recorded outcome

Both runs pass the frozen-model preservation criteria. Maximum concentration
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
refinement also remains within the original acceptance gate. See
`outputs/comparison.png`, `outputs/assessment.json`, and the validation ladder
for detailed evidence and the refinement qualification. The 251 focused checks
pass; a full repository test-suite pass is not claimed.
