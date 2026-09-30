# Configuring a native bioreactor

## 1. Choose the biological model

| Input mechanism | Choose when | Starting example |
| --- | --- | --- |
| `configured-kinetics` | You have explicit volumetric biological source equations or fitted source-rate models. | `mammalian_surrogate_fed_batch` |
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

## 3. Replace kinetics without replacing the reactor

An affine provider declaration has this form (replace the names and coefficients):

```json
{
  "type": "affine-surrogate",
  "identifier": "fitted-capacity",
  "inputs": {"s": {"source": "concentration", "name": "nutrient", "unit": "mmol/L"}},
  "outputs": {"capacity": {"unit": "mol/(kgDW h)", "intercept": 0.0, "coefficients": {"s": 0.2}}},
  "validity_domain": {"s": [0.0, 10.0]},
  "extrapolation": "error"
}
```

Put it in the existing top-level `rate_provider` field. For a pathway capacity,
replace the relevant uptake rule with:

```json
{"identifier": "nutrient_uptake", "type": "provider", "species": "nutrient", "output": "capacity"}
```

Existing constant/Monod/transfer-cap rules can coexist. Provider capacities use
mol/(kgDW time), with time matching `flux_time_unit`; negative capacities raise.
For direct kinetics, provider output names match `species_mass_rates` and units
match `rate_unit`. A null provider preserves the rule graph. Reconciliation units
must match its declared output/flux bases. Concentration inputs use mmol/L.
State inputs refer to available mechanism snapshot fields; condition inputs use
explicitly supplied conditions and their declared physical basis.

Use `type: "hybrid"`, `providers`, `default_provider` and `output_sources` to assign
different outputs to different providers. Outputs are selected independently;
substituting one output does not rewrite dependent equations in another provider.
Replace coupled source outputs together when necessary to preserve consistency.

For a custom model, register a constructor with `register_rate_provider(name,
factory)` before construction. It must implement `evaluate(snapshot, conditions)`
and return RateProviderResult with finite rates, correct units and domain diagnostics.
The [synthetic mammalian notebook](../../../examples/bioreactors/mammalian_surrogate_fed_batch/README.md)
fits both affine and non-affine surrogates from synthetic rates, registers a small
polynomial provider, compares trajectories and saves reusable mechanism JSONs.
It does not require changes to any native reactor or solver.

## 4. Verify before scientific use

Run `python -m pytest -q -p no:cacheprovider -o pythonpath=. tests/Bioreactor` in
the documented Python 3.11 environment. Tests select headless plotting themselves.
For a new process, assess inventory/constraint residuals, time-step/tolerance
refinement, surrogate domain coverage and independent measurements. Numerical
verification, approximation accuracy and biological validation are different claims.
