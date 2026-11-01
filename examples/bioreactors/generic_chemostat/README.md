# Generic chemostat

Synthetic, isothermal, sterile-feed Monod culture using PharmaPy's native
`ContinuousReactor`, liquid streams and volume controller. These inputs are a
mathematical qualification case, not experimental data or fitted biology.

The 1 L culture receives 0.1 L/h containing 10 mmol/L nutrient. The native
well-mixed outlet maintains the target volume. Declared uptake is
`q = 4 S/(1 + S)` mol/(kgDW h), with S in mmol/L; yield is 0.1 kgDW/mol.
Thus maximum growth is 0.4/h and dilution is 0.1/h. No maintenance, death,
cell retention, gas transfer or oxygen limitation is represented.

At the positive equilibrium, `S* = Ks D/(mu_max-D) = 1/3 mmol/L` and
`X* = Y(Sin-S*) = 0.9667 gDW/L`. The washout threshold is
`Dcrit = mu(Sin) = 0.3636/h`. With growth disabled, biomass follows
`X(t) = X(0) exp(-D t)`. These are independent test oracles.

Run the six notebook steps. The default `inputs/` selects pathway optimization.
To run the same Monod model as configured kinetics, select `INPUT_SET = "kinetics"` in Step 1; both paths update together.
That representation includes biomass as a phase species; its formula weight is
bookkeeping. Both representations use the same yield and operating conditions.
Exports are local generated files and are excluded from version control.

`operation.mode: "continuous"` is the only new selection. Existing flow events
set, change or stop inlet streams. `working-volume` assumes additive solution
volumes with explicit feed densities. The native controller targets the initial
volume; after a sample or bolus it relaxes back toward that target when the available inlet flow permits. This is a
well-mixed chemostat, not perfusion or selective cell retention. Exposure integrals
remain accumulated history rather than being withdrawn with cells.

## Kinetics selection

In Step 1 choose `INPUT_SET = "pathway"` or `"kinetics"`; input and thermo paths change together. The output table handles either biomass representation and all event segments. In direct kinetics, select `relationship: "monod"` inside `biomass_rate` and `nutrient_rate`. The two rates share `parameters.affinities.nutrient` [mmol/L]; growth is [1/h] and uptake [mol/(kgDW h)]. Preserve their stated yield relationship when editing either maximum. The pathway route uses `relationship: "monod"` in `model.pathways.rule_graph`, connected through a `type: "provider"` uptake constraint.

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
