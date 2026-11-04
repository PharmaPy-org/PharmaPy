# Generic fed batch

The same model with a scheduled feed pulse, dilution and feed-corrected material balances.

Run all six steps of [workflow.ipynb](workflow.ipynb). Edit the values in
`inputs/mechanism.json` and `inputs/thermo.json` to change
operating conditions, kinetics and properties. The shared `build_bioreactor`
workflow creates native PharmaPy reactors and exports CSV/PNG/SVG trajectories
under `outputs/`. No runner or verification helper is required.

The synthetic inputs demonstrate execution and material accounting, not
experimental validation. Analytical checks are in
`tests/Bioreactor/test_bioreactor_generic_example.py`.

The [reconciliation benchmark](../../../tests/Bioreactor/fixtures/reconciliation/README.md) is maintained with the test fixtures
with known-answer metabolic networks and progressively harder numerical tests.

## Kinetics selection

The shipped empirical uptake capacity is defined in `model.pathways.rule_graph` and connected by a `type: "provider"` constraint. Select its `relationship` and fill the required inputs; constant capacities use a parameter expression. Set `parameters.corrections.theta` to change the optional correction. Native `constant`, `monod`, and `transfer-cap` constraints remain supported; switching back to them also requires removing unused provider bindings. The oxygen transfer cap remains a native constraint.

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
