# Configuring a native bioreactor

## 1. Choose the biological model

| Input mechanism | Choose when | Starting example |
| --- | --- | --- |
| `configured-kinetics` | You have explicit volumetric biological source equations or fitted source-rate models. | `ecoli_ye_fed_batch` |
| `pathway-lp` | A reduced pathway map and objective determine feasible activity under uptake capacities. | `generic_batch` |
| `rate-reconciled-culture` | Predicted/observed rate targets need reconciliation with a metabolic network. | `tests/Bioreactor/fixtures/reconciliation/analytic_balances` |

These are model choices, not organism restrictions. Supply a defensible model,
units and parameters for your process; the simulator does not infer missing biology.

## 2. Configure and run

Copy an example's three inputs. `case.json` declares operation, initial volume,
temperature, conditions and numerical settings. `mechanism.json` declares initial
species/populations, equations or network, parameters and feed/sample recipe.
`thermo.json` supplies the native phase properties needed by that run. Blank
properties are not zero; populate them before using additional thermal/phase physics.

```python
import json
from pathlib import Path
from PharmaPy.Bioreactors import build_bioreactor

folder = Path('examples/bioreactors/generic_batch/inputs')
case = json.loads((folder / 'case.json').read_text())
model = json.loads((folder / 'mechanism.json').read_text())
assembly = build_bioreactor(case, model, folder / 'thermo.json')
histories = assembly.solve(verbose=False)
```

The example notebooks show CSV/PNG/SVG exports from native DynamicResult histories.
Use the shared [input templates](templates/README.md) for complete schemas. The
following fragments show the active choices, not complete runnable input documents.

Direct kinetics select `mechanism: "configured-kinetics"` with
`model.kinetics.rule_graph`, `species_mass_rates` (species → output rule), and
`rate_unit`, e.g. `g/(L day)`. Unmapped species have no biological source.
dFBA selects `mechanism: "pathway-lp"` and supplies `model.pathways` with state/pathway
IDs, exchange/growth/objective coefficients and uptake constraints. Reconciliation
selects `mechanism: "rate-reconciled-culture"` with `network`, kinetic output maps,
explicit flux bases and a reconciliation policy. Keep inactive sections empty.

### Optional pathway selection among primary optima

In `model.pathways`, add this optional configuration, replacing the identifiers
and scales with those of your declared pathways:

```json
"secondary_optimization": {
  "policy": "scaled-minimum-norm",
  "reference_scales": {"pathway_a": 1.0, "pathway_b": 2.0}
}
```

The primary objective and constraints remain unchanged. On that optimal face,
the solver minimizes `sum((v_j / reference_scales[j])**2)`. Every pathway needs
one finite positive scale, in the same units and normalization as its flux.
Scales define a modeling preference, not fitted trajectory corrections or an
inferred enzyme cost. Rename or reorder scales with their associated pathways;
convert them with the flux coordinates when changing units.

The result reports `closure_policy_id = highs:pathway-lp:scaled-minimum-norm`
and the negative squared norm as `secondary_objective` (maximization convention).
Failed feasibility or optimality checks raise an error without reverting to the
alphabetical selection. Omitting the field, or setting it to `null`, preserves
the existing alphabetical tie-break and all current example configurations.
This option does not add automatic flux-variability/ambiguity screening or claim
that a unique numerical allocation is a biologically validated prediction.

## 3. Replace kinetics without replacing the reactor

The built-in `rule-graph` provider evaluates declared kinetic expressions and
returns `RateProviderResult`: named finite rates, matching units, provider identity,
version, validity status and diagnostics. Declare `output_units` for every output.
Optional `validity_domain` bounds refer to named rules; explicitly choose
`extrapolation: "error"` or `"allow"` when bounds are supplied.

For a custom model, call `register_rate_provider(name, factory)` before construction.
The factory receives the declaration, model definition and expected output names.
Its instance implements `evaluate(snapshot, conditions)`. Concentrations in the
snapshot use mmol/L; population and other state fields retain their declared bases.
Direct kinetics outputs match `species_mass_rates` and `rate_unit`. Reconciliation
outputs match the configured flux bases. Pathway uptake providers return
nonnegative capacities in mol/(kgDW time), with time matching `flux_time_unit`.
Replace coupled outputs consistently to preserve the declared balances.

Whole mechanisms are interchangeable for one case through the same
`build_bioreactor` call. `tests/Bioreactor/test_bioreactor_mechanism_interchange.py`
solves `generic_batch` and `generic_fed_batch` with their `pathway-lp` mechanism
and with `configured-kinetics` rules that encode the LP optimum; both routes match
the closed-form biomass and nutrient solution. Configured kinetics have no
population state, so biomass is declared as a phase species in an extended
thermo file. Those runs end before nutrient depletion, so they do not compare
the LP's depletion switching.

Affine and hybrid provider implementations belong to WP3.2 and are not included.
The registration interface is retained for later substitution without changing
reactor balances.

## 4. Predict measured fields

A measurement declaration (for example
`examples/bioreactors/ecoli_ye_fed_batch/inputs/observations.json`) names each
measured field, the phase species it measures, `quantity: "concentration"`, the
reported unit (such as `g/L` or `mmol/L`), the source-file `time_key` and
`value_key`, and `event_side`. At a scheduled event time the solution holds the
state before and after the event; `pre` selects the first, `post` the second.

```python
observations = read_observations(declaration, reference["measurements"])
histories = assembly.solve(report_times=sampling_times(observations))
prediction = predict_measurements(assembly, histories, declaration, observations)
```

Each field keeps its own sampling grid and source order. Times are in seconds and
values use the declared unit; concentrations are species inventory divided by the
recorded liquid volume. `report_times` adds the sampling times to the native
grid, so predictions come from accepted solver states rather than interpolation.
Adaptive SciPy steps are unchanged by reporting. Fixed-step grids define the
numerical method, so their sampling times must already lie on the grid.
Missing observations are omitted from the source, never encoded as NaN.

`prediction["time_s"][field]` and `prediction["values"][field]` match the per-field
`x_data`/`y_data` entries of the keyed `ParameterEstimation` interface on `master`
(#238). This branch's base predates that interface, and no experiment or
measurement object hierarchy is defined here (#346).

## 5. Report algebraic kinetic quantities

Configured kinetics accept optional `model.kinetics.reported_rates` entries:

```json
"reported_rates": {
  "oxygen_uptake": {
    "rule": "oxygen_uptake",
    "unit": "g/(L h)",
    "basis": "g oxygen per litre of liquid per hour"
  }
}
```

The named rule must exist and any replacement rate provider must return it with
its declared unit. This registers a native postprocessed field (for example
`oxygen_uptake_liquid0`); it does not add a material source or solver state.
Values use the declared reporting unit without implicit conversion. Equations
must therefore match their units and basis, just as for other rule expressions.
Omitting `reported_rates` preserves existing behavior. The direct F10/F11
examples use this to calculate published oxygen demand; they do not model a
dissolved-oxygen inventory or oxygen-control dynamics.

## 6. Verify before scientific use

Run `python -m pytest -q -p no:cacheprovider -o pythonpath=. tests/Bioreactor` in
the documented Python 3.11 environment. Tests select headless plotting themselves.
For a new process, assess inventory/constraint residuals, time-step/tolerance
refinement, rate-provider domain coverage and independent measurements. Numerical
verification, approximation accuracy and biological validation are different claims.

