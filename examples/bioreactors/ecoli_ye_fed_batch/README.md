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

Open `workflow.ipynb` from the repository or example folder and use Run All. The six commented steps load inputs, build native reactors, simulate, tabulate, plot, and export. `inputs/` contains the F10 configuration; `inputs/F11/` contains the second case and mechanism, sharing `inputs/thermo.json`. Change `INPUT_DIRS` to select configurations. The notebook reads no reference measurements and performs no comparison analysis. `inputs/observations.json` declares the measured biomass, glucose and acetate fields of `reference/F10.json` and `F11.json`; `predict_measurements` returns the matching predictions at each field's own sampling times (see the [user guide](../../../doc/online_docs/bioreactors/USER_GUIDE.md#4-predict-measured-fields)).

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
- Scheduled feed adds 0.1725 L containing 103.5 g glucose per run. The saved source volume history is consistent with also applying most logged glucose-feed volumes as nutrient-free dilution. That second volume contribution is excluded here.
- Base additions and samples remain separate. Final volumes are 1.1061 L (F10) and 1.1008 L (F11).
- Stock density 1.336 kg/L and base density 1 kg/L are declared assumptions affecting carrier mass, not prescribed feed volume or glucose dose. Base is modeled as a nutrient-free addition, without ionic chemistry. Evaporation is omitted.
- The configured BDF Jacobian perturbations avoid a failure near depletion. Absolute tolerance is 1e-15 in native solver units because the initial biomass inventory is only 7.4e-8 kg; relative tolerance is 1e-10. These are numerical settings, not fitted biological coefficients.

## Results and limits

The notebook generates forward CSV and PNG/SVG exports under `outputs/F10` and
`outputs/F11`. Outputs are excluded from Git. Tests generate validation evidence
in temporary directories and compare against the source observations and an
independent concentration-based Radau calculation. The figures below summarize
the reference branch assessment; rerun the tests to qualify changed code.

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

## Feed-volume source audit

The authors' repository at revision `c8d18b53105f695af0d99ee0126544a623b6ff65`
identifies `Feed` as `Cumulated_feed_volume_glucose`. Its original
`ExpData/3L_cultivation/Generic_Excel_Sheet_GSF_F10F11.xlsx` labels the two
channels **Glu (Feed [V])** and **Glu (Profile)**. Base and samples occupy
separate columns; the saved acid and medium additions are empty. The extracted
[operating records](reference/operating_records.json) preserve these original
observations and their source identity.

This evidence identifies two descriptions of the same glucose feed, rather than
a documented second liquid addition. The current inputs already count the
nominal glucose profile's solution volume once. No feed or carbon-kinetics
parameters were changed by this audit.

| Quantity over the modeled 26.5 h | F10 | F11 |
| --- | ---: | ---: |
| Nominal profile integral, glucose [g] | 103.5 | 103.5 |
| Nominal solution volume at 600 g/L [L] | 0.1725 | 0.1725 |
| Sum of logged glucose-feed additions [L] | 0.125374 | 0.125524 |
| Separate base additions [L] | 0.0696 | 0.0643 |
| Samples removed [L] | 0.136 | 0.136 |
| Current final liquid volume [L] | 1.1061 | 1.1008 |
| Historical saved simulation final volume [L] | 1.217926 | 1.216519 |

The historical final-volume excess is consistent with including most logged
feed additions a second time. The actual source simulator is an external
`ecoli_sim_and_pe` submodule whose code was unavailable for this audit; this is
an inference from input identities and saved volume histories, not a source-code
trace. The recorded induction addition occurs at the endpoint of this
pre-induction model and is not added to improve agreement.

**The nominal profile and logged dose still disagree.** The 103.5 g value is the
configured profile integral, not proof of the experimentally delivered dose.
An experiment owner must resolve that discrepancy before claiming a validated
actual-feed reconstruction. Adding both volumes or selecting the interpretation
with lower biomass RMSE would not resolve it. The retained scenario reproduces
the nominal source-model feed schedule without duplicate dilution.

### Stock-basis hypothesis (pending experiment-owner confirmation)

One explanation for the dose discrepancy is that the stock was 60 % glucose by
mass at the declared 1.336 kg/L, i.e. 0.60 x 1336 = 801.6 g/L, rather than the
published 600 g/L. The `Profile` channel is a glucose mass rate [g/h]
(4.8 g/h / 600 g/L = 0.008 L/h, the configured first flow), so each logged
half-hour `Feed` volume can be compared with the volume the profile implies
under either basis:

| F10 interval [h] | Logged [mL] | At 600 g/L [mL] | At 801.6 g/L [mL] |
| --- | ---: | ---: | ---: |
| 18.675-19.175 | 9.81 | 12.38 | 9.26 |
| 20.175-23.675 | 13.10 | 17.40 | 13.03 |
| 24.175-24.675 | 13.62 | 18.00 | 13.47 |
| 25.175-25.675 | 13.32 | 18.00 | 13.47 |
| All 14 logged intervals (16.175-26.175) | 125.37 | 170.88 | 127.90 |

F11 behaves the same (125.52 mL logged; 170.89 and 127.91 mL predicted). Steady
intervals agree within 1-2 % at 801.6 g/L, while 600 g/L is about 33 % high
throughout. The first, ramp and last intervals deviate more, consistent with
actual pump transitions (pause near 20.08 h, final reduction near 25.82 h)
differing from the nominal schedule.

This is not yet a confirmed correction:

- The 1.336 kg/L density is a declared assumption without a cited source. For
  reference, tabulated pure glucose solutions are about 1.22 kg/L at 50 % w/w
  (20 °C), and 600 g/L corresponds to roughly 49 % w/w; 1.336 kg/L fits neither
  basis well unless it was measured.
- If the logger recorded feed mass and converted it with 1.336 kg/L, the logged
  volumes would match 801.6 g/L whatever the true stock concentration.
- An unlabelled workbook flow of 36 alongside a 21.6 g/h profile does not
  discriminate: 36 g/h solution at 60 % w/w and 36 mL/h at 0.600 g/mL both give
  21.6 g/h glucose.

Sensitivity (not an adopted input): keeping the glucose mass-rate profile and
1.336 kg/L, and scaling each flow by 600/801.6, lowers the glucose-feed volume
to 129.12 mL and the final volume to 1.0627 L (F10) and 1.0574 L (F11).
Predictions at the measurement times rise by at most 3.6 %; RMSE changes from
1.79 to 2.06 g/L (biomass), 1.87 to 1.94 (glucose) and 0.107 to 0.116 (acetate)
for F10, and similarly for F11. The published coefficients were calibrated under
the authors' feed accounting, so fit quality cannot decide the stock basis.

## Oxygen uptake and its limits

The direct F10/F11 mechanisms now report the paper's Eq. 19 oxygen demand using
native algebraic outputs: `specific_oxygen_uptake_liquid0` [g O2/(gDW h)] and
`oxygen_uptake_liquid0` [g O2/(L h)]. The latter is specific demand multiplied
by biomass concentration. This implements Eq. 19 as printed; it does not
establish an independently closed elemental/respiratory balance. The notebook exports both alongside the existing
carbon trajectories. Reconciliation inputs are unchanged.

The oxygen yields come from S3, sheet `3L`, column `GSF_YE_Paper_FedBatch`:
1.067, 1.067 and 1.163 g O2/g substrate for glucose, acetate and yeast extract.
Carbon contents come from S3 `Constants`, `LabScale`: biomass 0.041,
glucose/acetate 1/30 and yeast extract 0.0364 C-mol/g. Equations, coefficients,
units and output bases are explicit mechanism inputs. Reported rates add no
integrated state and do not change the carbon balances.

This is a demand calculation, **not dissolved-oxygen or oxygen-control
validation**. The source model's Eq. 20 also requires oxygen transfer and its
Eq. 21 a sensor response. We have not qualified those transfer/controller
assumptions for this reproduction. A fixed transfer coefficient would not by
itself establish the actual variable-agitation response. The original
[processed oxygen observations](reference/oxygen_observations.json) fall to
2% and 5.1% air saturation in F10/F11 despite the nominal 30% control target;
therefore we do not assume oxygen was always nonlimiting. Sensor readings
alone do not establish the magnitude of growth limitation. A dynamic oxygen
inventory and limitation feedback remain unqualified until their required
transfer/control evidence is established.

## Forward provenance

[inputs/provenance.json](inputs/provenance.json) records source revisions,
transformation history, sharing conditions, parameter units/bases and reported
source bounds. It explicitly distinguishes the authors' fitted coefficients
from this unfitted forward reproduction. Source estimation settings are retained
as metadata, not silently converted into statistical weights.

The [forward-record guide](../../../doc/online_docs/bioreactors/USER_GUIDE.md#5-save-and-reload-forward-provenance)
shows how to snapshot inputs, observations, predictions and native fields and
reload them for another solve. Generated records belong in temporary or ignored
output directories. This adopts the forward provenance requirements of #351;
it does not implement the pending shared Experiment/Measurement API (#346).
