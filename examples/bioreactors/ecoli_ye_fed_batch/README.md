# E. coli yeast-extract fed-batch forward reproduction

Both direct and reconciled F10/F11 mechanisms now select reusable
[biological relationships](../../../PharmaPy/Bioreactors/TECHNICAL_SUMMARY.md#biological-relationships) inside
the existing rule graph. Expansion preserves the original source equations,
parameters and target mappings exactly; explicit rule expressions remain supported.

This example runs F10 and F11 through native PharmaPy `SemiBatchReactor.solve_unit`, using configured direct kinetics. It reproduces the major biomass, glucose, and acetate trends without parameter fitting. It is not a rate-reconciliation or oxygen-control validation.

The separate [frozen-model reconciliation assessment](reconciliation/README.md)
runs the same source equations and operating inputs through the shared
rate-reconciliation mechanism, with its own forward notebook and outputs.
It tests preservation of this experimental agreement without refitting.

Open `workflow.ipynb` from the repository or example folder and use Run All. The six commented steps load inputs, build native reactors, simulate, tabulate, plot, and export. `inputs/` contains the F10 configuration; `inputs/F11/` contains the second case and mechanism, sharing `inputs/thermo.json`. Change `INPUT_DIRS` to select configurations. The notebook reads no reference measurements and performs no comparison analysis.

## Configuration and native execution

All species, equations, coefficients, initial amounts, feed compositions, and event schedules come from JSON. The notebook calls shared `build_bioreactor` and `assembly.solve()`; it contains no local mechanisms or event loop. `ConfiguredRates` evaluates the existing rule graph and returns native mass-inventory rates. `WorkingVolume` integrates declared continuous inlet flows and applies scheduled volume changes. Native `LiquidStream` objects supply feed mass. Additions and well-mixed withdrawals update inventories between ordinary native solves.

Select `mechanism: "configured-kinetics"` using the same top-level input layout as the metabolic models. Fill `model.kinetics.rule_graph`, `species_mass_rates`, and `rate_unit`; inactive sections and `flux_time_unit` remain empty/null. The generic shared capabilities support batch and fed-batch, with named piecewise-constant inlets, additions, and sampling. A flow event replaces its named inlet; zero flow stops it. No organism-specific equations are embedded in the architecture.

`BioreactorUnitConverter` handles input quantities, kinetic source conversion, and output reporting. Initial amounts may use supported mass or molecular units; volume, runtime, output step and maximum solver step use their declared `unit`/`value`. Molecular conversions use the configured species molecular weights. The existing example values and equations are unchanged.

Recipe events also accept explicit quantities, for example:

```json
{
  "event_type": "flow",
  "time": {"value": 60, "unit": "min"},
  "volume_flow": {"value": 1, "unit": "mL/min"},
  "concentrations": {"unit": "g/L", "values": {"glucose": 600}},
  "density": {"value": 1336, "unit": "kg/m3"}
}
```

Use optional `inlet` identifiers for multiple concurrent feeds. `recipe.time_unit` declares the default for numeric event times (hours here); explicit time quantities override it. For additions and samples use `volume: {"value": 10, "unit": "mL"}`. Additions in this example declare `volume_basis: "solution"` to preserve the complete-solution volume basis. In these examples numeric event times are hours, and the explicit legacy suffixes `volume_l_h`, `volume_l`, `concentrations_g_l`, and `density_kg_l` retain their meanings. Do not supply both forms of a quantity. Absent feed density uses the configured carrier density (1 kg/L here); declare a density for a different stock.

`model.kinetics.rate_unit` declares the rule outputs, such as `g/(L h)` or `mol/(m3 min)`. The converter maps them to kg/(m3 s), then the mechanism multiplies by native volume to obtain kg/s. Rule-graph concentration inputs remain mmol/L, as expected by the existing rule evaluator. Changing a rate-unit label requires expressing the equations and coefficients in that basis; conversion cannot infer the dimensions inside an arbitrary expression. Native inventories remain kg, volume m3, and solver time seconds; exported tables retain h, L, g/L and g. Biomass and lumped yeast fractions use formal 1 g/mol bookkeeping units, not molecular identities. Glucose and acetate use molecular weights of 180.156 and 60.052 g/mol. Thermophysical values are bookkeeping placeholders for this isothermal, prescribed-volume test; it does not validate thermodynamics.

## Source and corrections

Source: Schröder-Kleeberg et al. (2025), [Modelling of Escherichia coli Batch and Fed-Batch Processes in Semi-Defined Yeast Extract Media](https://doi.org/10.3390/bioengineering12101081). Parameters come from [supplement S3](https://mdpi-res.com/d_attachment/bioengineering/bioengineering-12-01081/article_deploy/bioengineering-12-01081-s001.zip), sheet `3L`, column `GSF_YE_Paper_FedBatch`. Initial states, processed measurements, and operating histories come from the authors' [saved F10/F11 simulations](https://git.tu-berlin.de/bvt-htbd/public/schroeder-kleeberg_2024_ye_model). The reference files retain those processed measurements, including their 0.37 biomass conversion, rather than mixing them with the differently processed supplementary figure spreadsheet.

- Acetate production has the positive sign implied by the source's definition of net production; the printed balance has an inconsistent sign.
- Scheduled feed adds 0.1725 L containing 103.5 g glucose per run. The saved source also applies most recorded feed volumes as nutrient-free dilution. That second volume contribution is excluded here.
- Base additions and samples remain separate. Final volumes are 1.1061 L (F10) and 1.1008 L (F11).
- Stock density 1.336 kg/L and base density 1 kg/L are declared assumptions affecting carrier mass, not prescribed feed volume or glucose dose. Base is modeled as a nutrient-free addition, without ionic chemistry. Evaporation is omitted.
- The configured BDF Jacobian perturbations avoid a failure near depletion. Absolute tolerance is 1e-15 in native solver units because the initial biomass inventory is only 7.4e-8 kg; relative tolerance is 1e-10. These are numerical settings, not fitted biological coefficients.

## Results and limits

`outputs/F10` and `outputs/F11` contain forward CSV spreadsheets and PNG/SVG figures. The separately generated `outputs/comparison.*` and `validation.json` compare against measurements and an independent concentration-based Radau calculation. The forward notebook does not regenerate these validation artifacts.

| Experimental RMSE (g/L) | F10 | F11 |
| --- | --- | --- |
| Biomass | 1.792 | 1.652 |
| Glucose | 1.871 | 1.639 |
| Acetate | 0.107 | 0.102 |

Native trajectories are checked against independently implemented balances away from event discontinuities. Tests also check event conservation, multiple inlets and stopping, sampling of biological inventories, final volume, equivalent units, and renamed species with a different recipe.

Agreement is reasonable at the measured points, not exact throughout the process. Final biomass is about 44 g/L versus about 40 g/L in the saved source. Late acetate differs materially, and its final collapse is not experimentally established by these measurements. Oxygen transfer/control, induction, temperature shifts, exact yeast composition, and complete elemental conservation are not validated. Published coefficients were previously calibrated by the authors; this is a reproduction benchmark, not a prospective prediction of an unseen process.

Run the independent numerical checks with:

```sh
MPLBACKEND=Agg python -m pytest tests/Bioreactor/test_bioreactor_ecoli_ye.py -q
```

## Feed interpretation: pending domain review

| Interpretation | Treatment | Consequence |
| --- | --- | --- |
| Current inputs | Each declared glucose feed contributes its glucose and solution volume once; base and samples remain separate. | F10/F11 final volumes 1.1061/1.1008 L; F10 biomass RMSE about 1.79 g/L. |
| Authors’ saved simulation | Most recorded feed volumes also enter as nutrient-free dilution. | Approximately 10% larger final volume; source F10 biomass RMSE about 1.46 g/L. |

The chosen interpretation follows the declared feed dose (103.5 g glucose in
0.1725 L solution) and avoids counting that solution volume twice. However, the
saved source may encode an additional real liquid contribution not distinguished
in the available declarations. Better curve agreement alone cannot decide this.
The comparison is documented in `outputs/validation.json`; interpretation remains
**pending domain review**. Neither inputs nor trajectories were retuned in response
to this review. A source-matching scenario is not silently substituted.
