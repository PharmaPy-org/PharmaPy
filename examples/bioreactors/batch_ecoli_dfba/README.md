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
