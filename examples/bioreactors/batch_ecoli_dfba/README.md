# Native E. coli batch dFBA

[Scientific sources, packaged evidence, and validation limitations](documentation/README.md).

Reproduces the qualitative diauxic-growth behavior reported by Mahadevan et al.
(2002) using a native `LiquidPhase`, arbitrary-pathway `PathwayMetabolism`, and
native `BatchReactor`. All pathway identities, coefficients, constraints, state,
and transport are declared in `inputs/mechanism.json`; source attribution is
given in the linked scientific documentation.

Run Steps 1–6 of [workflow.ipynb](workflow.ipynb) for forward simulation.

## Input field guide: reduced-pathway template

Use this runnable example as a starting configuration, not as a universal
E. coli parameter set. See the [source notes](documentation/README.md) for
literature provenance, reconstruction assumptions, and validation limitations.

Read `inputs/mechanism.json` from top to bottom in setup order:

| Section | Category | What to configure |
| --- | --- | --- |
| `schema_version`, `mechanism` | Required for this model | Keep version 3 and `pathway-lp` for this formulation. |
| `$comment`, `model.pathways.problem_name` | Descriptive | Explain and identify the case; these do not define biological rates. |
| `operation` | Required for this model | Mode, volume/mass basis, carrier, duration, and selected recipe. |
| `state` | Required for this model | Initial species amounts with their unit, and biomass in kg. |
| `parameters` | Required for this model | Supply the coefficients referenced by the uptake and transfer rules; retain empty parameter groups. |
| `model.pathways` | Required for this model | Define species/pathway order, exchange matrix, growth yields, objective and uptake constraints. Preserve matching matrix/vector dimensions. |
| `transport`, `flux_time_unit` | Required for this model | This oxygen-transfer example uses both; specify transfer coefficients in the declared units. |
| `recipes` | Required for this model | The selected `batch` recipe is empty because no scheduled additions are made. |
| Inactive `model` sections, `network`, `rate_provider`, blank state/condition fields | Optional | Leave their existing empty/null slots intact; they are inactive, not zero-valued biological parameters. |
| `numerical`, `model.pathways.depletion_tolerance` | Numerical | Solver, sampling grid, tolerances, maximum step, Jacobian settings, and depletion threshold. |
| `thermo.json` | Required for this model / optional properties | Keep species/carrier molecular weights and liquid densities. Unused properties remain blank for this isothermal calculation. |

The objective and pathway bounds are scientific modeling choices, not merely
solver settings. Changing them can change predictions. `numerical.step` specifies
the requested grid for this SciPy example, not a fixed internal integration step.
Empty thermodynamic slots must be populated before enabling calculations that
need them. Do not reorder species/pathway arrays independently of their matrices.

### Run the notebook

Open [workflow.ipynb](workflow.ipynb) and run all six steps with Run All or
Shift+Enter: select folders, load/edit inputs, simulate, extract trajectories,
plot, and export. Only `mechanism.json` and `thermo.json` are loaded.
`WRITE_ARTIFACTS = True` replaces `outputs/trajectories.csv`, `trajectories.png`,
and `trajectories.svg`. Change `EXPORT_DIR` to preserve earlier files.
Concentrations use the recorded volume, and feed boundaries retain pre/post rows.

Numerical checks belong to the test suite. Published panels and historical
comparison figures are excluded. Consult the linked original paper for qualitative
comparison; generated trajectories do not establish experimental validation.

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
