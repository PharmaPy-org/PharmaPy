# Standard forward-simulation inputs

Copy `case.json`, `mechanism.json`, and `thermo.json` into your input directory.
These are **blank templates**, not runnable examples. Follow `guide.json` to fill
them, or use a populated example as a starting point. The guide is documentation,
not a fourth runtime input.

The case and mechanism use schema version 2; thermo retains the native PharmaPy
species-property format. The shared structure applies to all three mechanism families.
Species identifiers, coefficients, equations, matrices, and record-list lengths
are user inputs. The format does not infer missing biology or material data.

- `case.json`: operation, initial volume/mass, conditions, numerics, selected recipe.
- `mechanism.json`: mechanism selection, parameters, mathematical model, initial
  quantities, network/provider, transport, and scheduled material events.
- `thermo.json`: native species properties needed by the chosen calculation.

Parameters use seven shared groups: rates, affinities, yields, maintenance,
responses, coefficients, and corrections. Populate relevant groups with numeric
maps using your identifiers. Equations explicitly reference their dotted paths;
category names do not select or modify equations.

Fill `model.pathways` for pathway optimization, or `model.kinetics` and
`model.reconciliation` for rate reconciliation. For direct configured kinetics,
fill `model.kinetics.rule_graph`, `species_mass_rates`, and `rate_unit`; leave
network, transport and flux_time_unit null. Leave rate_provider null for default direct kinetics, or declare a provider to substitute outputs. Pathway models also accept providers for explicitly selected uptake constraints; see ../USER_GUIDE.md. Leave inactive sections blank or
`{}` and inactive scalar values null. Initial species amounts share
`state.species.unit` and `state.species.amounts`; initial volume is specified only
in `case.operation.initial_volume`. Null means absent, not a physical zero.

Feed and stock records can declare `composition` as `supplied`, `nutrient-free`,
or `unknown`. Supplied means the concentration map is complete; unlisted solutes
are zero. Unknown may include partial known concentrations. Legacy omitted
composition is reported as unspecified. Only declared material is added.
Set `case.recipe.require_complete_inputs` to true to reject incomplete feeds
before solving; its compatibility default is false.

`stock_to_target` calculates stock volume and dilution under the declared constant
density. The older `feed` event retains its carrier-volume convention, and
`target_concentration` retains its solute-only convention. See the guide for units
and complete event records. `operation.temperature_k` sets the isothermal liquid
temperature; null keeps the native 298.15 K default.

For rate reconciliation, optional `model.kinetics.flux_basis` replaces opaque
source-conversion factors with exchange, growth and product declarations. Each
specifies amount unit, population normalization and time unit. See `guide.json`
for supported bases and required physical properties. Empty means legacy,
unverified conversions. Declaration does not infer the scientific meaning of
a reaction or rescale its stoichiometric column.

Pathways accept the same declarations under `model.pathways.flux_basis`, with
`exchange` and `growth` only. A conventional dry-biomass basis is:

```json
"flux_basis": {
  "exchange": {"amount_unit": "mmol", "normalization": "gDW", "time_unit": "h"},
  "growth": {"amount_unit": "1", "normalization": "none", "time_unit": "h"}
}
```

Both times must match `flux_time_unit`. Pathway matrix coefficients and uptake
maxima must be supplied on the declared amount/population basis; saturation and
half-saturation concentrations remain mol/m3. Mass-transfer parameters use the
declared time unit. The converter maps accepted exchange rates to mol/kgDW/time,
then native kg/s. It also converts the physical transfer capacity into the LP's
declared uptake units. Specific growth remains inverse time.

For a cell-based culture, a consistent choice is exchange in pmol/(cell day),
growth in 1/day, and product in pg/(cell day). Declare those three records under
`model.kinetics.flux_basis`, with matching provider output labels and correctly
based network columns/kinetic targets. Other supported prefixes and normalizations
are interchangeable when the numerical model is expressed consistently.

Converting between cell count and dry biomass requires an explicitly supplied
`biomass_g_per_million_cells` in the relevant rate record. No such property is
needed for gDW-to-kgDW or cell-to-million-cell conversions. Carbon-based product
or biomass conversion requires the existing explicit composition/content fields;
no standard cell mass or carbon content is assumed. Empty declarations preserve
existing behavior and its reported qualifications; they do not certify the basis.

`build_bioreactor` automatically uses `BioreactorUnitConverter` for biological
source factors and JSON time/volume/initial molecular amounts. No fourth input
file is needed. Initial amounts accept mol through fmol; volumes accept m3, L,
mL and uL; times accept s, min, h and day. Fields with units in their names
(such as `temperature_k` or `concentrations_mmol_l`) retain those fixed units.
Use standalone conversions before assigning differently measured values:

```python
from pathlib import Path
from PharmaPy.Bioreactors import BioreactorUnitConverter as Units

inputs = Path("path/to/inputs")
units = Units.from_inputs(inputs / "case.json", inputs / "mechanism.json")
temperature_k = Units.convert(37., "degC", "K")
concentration = Units.convert(1., "g/L", "mmol/L", molecular_weight_g_mol=180.)
# Arrays are supported; molecular weights can also come from a native phase.mw.
```

Standalone conversions support mass kg through pg, molecular amounts mol through
fmol, the corresponding `mol-C` prefixes, cell/`10^6 cell`, amount/volume
concentrations, and K/degC/degF. M/mM/uM/nM are concentration aliases. Molecular
mass conversion requires molecular weight; carbon-mole mass conversion requires
`mass_per_cmol_g`; molecular/carbon conversion requires `carbon_atoms` or both
mass properties. Cell/mass conversion requires `biomass_g_per_million_cells`.
Units are case-sensitive. Unsupported units or absent required properties fail
explicitly. Compound biological rates use the structured `flux_basis` declarations,
not the standalone converter. Neither API infers reaction stoichiometry or rescales
equations and bounds automatically.

Both mechanism families constrain accepted fluxes against mapped resource
availability: end inventories for fixed-step propagation, net production at
exhausted pools for adaptive evaluation. Scheduled feeds use the same event
workflow. Growth/product coupling still follows the declared biological model;
unmapped external resources are not invented or verified.

For pathways, transport-capacity parameters and an actual `transport` record must
agree on maximum supply after unit conversion. Conflicting values fail at
construction. The existing positive-concentration transfer-cap approximation is
retained. Both mechanisms convert accepted molecular sources to native kg/s through
the same unit converter. Pathway coefficients must obey the declared amount/population
and growth normalization; the converter cannot derive the biological normalization.

`assembly.diagnostics(histories)` provides common conservation and optimization
sections. A pathway closed-uptake screen is not a full elemental balance. The
optimization section reports only the last evaluated closure. Pathway alternate
optima are selected lexicographically by sorted pathway ID, so column order has
no effect; this numerical priority is not a biological objective.

For metabolic networks, optional `representation` (`reduced` or `chemical`) and
`exchange_roles` declare scope and boundary assumptions. These fields do not alter
flux bounds. Diagnostics distinguish structural consistency from elemental closure
and numerical feasibility; a reduced network can fail the structural test while
still satisfying its declared steady-state equations.

```python
import json
from pathlib import Path
from PharmaPy.Bioreactors import build_bioreactor

inputs = Path("path/to/inputs")
case = json.loads((inputs / "case.json").read_text())
mechanism = json.loads((inputs / "mechanism.json").read_text())
assembly = build_bioreactor(case, mechanism, inputs / "thermo.json")
print(assembly.input_audit())
histories = assembly.solve(verbose=False)
print(assembly.diagnostics(histories))
```

Schema 1 and unversioned inputs remain accepted. Existing equations, solver
settings, and material increments are preserved unless the user changes them.

Named `flow` events configure piecewise-constant continuous inlets; zero flow
stops the named inlet. `sample` events withdraw well-mixed liquid and its extensive
biological inventories. These use the same builder for the three mechanism
families. See `guide.json` for explicit quantity-unit records and volume assumptions.

## Optional population coupling and kinetic validity

For rate reconciliation, `kinetic_outputs.growth_reaction` already selects growth
from an accepted network flux. Species exchanges already permit either direction
when the supplied stoichiometry, bounds and kinetic rules allow it. A species is
not intrinsically an uptake-only or production-only species. Configure both
directions when justified; the architecture does not invent a production term.

An optional `model.kinetics.kinetic_outputs.death_increment` evaluates a user-supplied
survival/stress expression **after** reconciliation. For example, with a declared
uptake-positive reaction named `resource_uptake`:

```json
"death_increment": {
  "unit": "1/day",
  "expression": {
    "op": "multiply",
    "args": [
      {"parameter": "responses.deficit_sensitivity"},
      {"op": "maximum", "args": [0,
        {"op": "subtract", "args": [
          {"parameter": "maintenance.required_uptake"},
          {"accepted_flux": "resource_uptake"}
        ]}
      ]}
    ]
  }
}
```

This illustrates configuration syntax, not a universal death law. The user must
supply and justify the uptake requirement, sensitivity, resource roles and sign
conventions. Equivalent resources can be combined using supplied coefficients.
No requirement, threshold, capacity or cell composition is inferred automatically.

`accepted_flux` references a declared exchange output and returns a canonical
value through `BioreactorUnitConverter`: molecular exchanges are in
mmol/(million cells day), product exchange in g/(million cells day), and the
selected growth reaction in 1/day. Declared `flux_basis` is required. Parameter
units must match the resulting expression; the example sensitivity has units
million cells/mmol. Ordinary parameter/constant/state/condition expressions and
references to assigned provider outputs remain available. Accepted flux references
are rejected in pre-reconciliation kinetics to avoid an implicit feedback loop.

The increment must be finite and nonnegative and is added to the provider's basal
death rate. Set basal death to zero if this expression supplies all mortality.
The closure first uses basal death for finite-step exposure; after the response,
inventory constraints are rechecked with actual death. If reduced biological
production no longer offsets an independent loss, the step fails explicitly:
reduce the step or use adaptive integration. No unresolved fixed-point iteration
or silent clipping is introduced. The expression changes neither flux targets nor
the reconciliation objective, and is not a DAE formulation.

Rule-graph providers can also declare optional validity bounds on named graph
values, including intermediate rules that expose concentrations or cell density:

```json
"validity_domain": {"configured_density_rule": [0, 20]},
"extrapolation": "error"
```

Bounds use that rule's units. The example bound is illustrative, not a recommended
cell-density limit. `error` rejects out-of-domain evaluations; `allow` preserves
the calculated rates but marks provider validity false. Bounds are not clamps or
biological constraints. They apply to solver stage evaluations as well as saved
states; they do not locate domain-boundary events. Omitting the domain preserves
existing behavior. Hybrid providers retain validity information from their children.

`assembly.mechanism.last_population_rates` separates actual growth, basal death,
the extra death response and net growth. `assembly.diagnostics(histories)` includes
these rates and provider validity, explicitly scoped to the last evaluation, not
a trajectory-wide biological certificate. Use recorded-state replay for full traces.
