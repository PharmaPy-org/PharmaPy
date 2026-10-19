# E. coli yeast-extract fed-batch: source and assumptions

The user-facing demonstrator is [F10/F11 rate reconciliation](reconciliation/README.md).
Run its `reconciliation/workflow.ipynb` to load inputs, build native reactors,
solve, extract trajectories, plot and export. F10 and F11 configurations are in
`reconciliation/inputs/` and `reconciliation/inputs/F11/`, sharing
`inputs/thermo.json`. Source observations, provenance and operating records remain
here for both runs.

The source's direct-kinetics model is retained only as a regression reference in
`tests/Bioreactor/fixtures/ecoli_ye_direct`, not as another demonstration.
Tests compare reconciliation with that reference and independently implemented
concentration balances. Neither route is refitted to improve agreement.

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
- Stock density 1.336 kg/L is the authors' feed-logging mass-to-volume factor (see the feed record basis below), not a documented stock density; with base density 1 kg/L it affects carrier mass, not prescribed feed volume or glucose dose. Base is modeled as a nutrient-free addition, without ionic chemistry. Evaporation is omitted.
- The configured BDF Jacobian perturbations avoid a failure near depletion. Absolute tolerance is 1e-15 in native solver units because the initial biomass inventory is only 7.4e-8 kg; relative tolerance is 1e-10. These are numerical settings, not fitted biological coefficients.

## Results and limits

The reconciliation notebook generates forward CSV and PNG/SVG exports under `outputs/F10` and
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
| Sum of logged glucose-feed additions [L] (balance mass / 1336 g/L) | 0.125374 | 0.125524 |
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

**The nominal profile and logged dose disagree.** The 103.5 g value is the
configured profile integral, not proof of the experimentally delivered dose, so
this example does not claim a validated actual-feed reconstruction. Adding both
volumes or selecting the interpretation with lower biomass RMSE would not
resolve it. The retained scenario reproduces the nominal source-model feed
schedule without duplicate dilution.

### Feed record basis and glucose-dose uncertainty

The logged glucose-feed channel is gravimetric. Every `Feed` value multiplied by
1336 is an exact multiple of 0.1 g in both runs, while base and sample volumes are
exact multiples of 0.1 mL. The logged "volumes" are therefore balance masses
divided by 1336 g/L, and the 1.336 kg/L used for the configured stock is that
processing factor, not a documented stock density. The logs fix the delivered
solution mass, 167.5 g (F10) and 167.7 g (F11). They do not fix the delivered
glucose (mass fraction x mass) or volume (mass / true density).

The `Profile` channel is read here as the glucose mass-rate setpoint [g/h]
(4.8 g/h / 600 g/L = 0.008 L/h, the configured first flow). Over the logged
window it integrates to 102.53 g, about 0.61 of the logged solution mass. On the
21.6 g/h plateau (23.675-25.675 h) the setpoint glucose is 0.59-0.61 of the
logged solution mass in every half hour, consistent with a dosing controller
that assumed a 60 % w/w stock. Logged mass in the ramp intervals is 1-10 % above
that relationship, and the first and last
intervals deviate strongly, consistent with transitions shifted from the nominal
schedule. With otherwise nominal rates, the last interval implies the final
reduction at about 25.80-25.81 h instead of 26 h. Half-hour logging cannot
resolve the other transition times.

The nominal scenario retained above (103.5 g glucose in 172.5 mL) is not
consistent with the logged mass under either plausible composition: it would
require about 230 g of solution at 1.336 kg/L. Scenarios consistent with the
logged mass bound the effect. Densities are assumptions: 1.217 kg/L at
49.3 % w/w is interpolated from tabulated pure glucose at 20 °C (about
1.219 kg/L at 50 % w/w), and 1.27 kg/L at 60 % w/w is extrapolated.

| F10 scenario (not adopted inputs) | Glucose [g] | Feed [mL] | Final volume [L] | Largest prediction change |
| --- | ---: | ---: | ---: | --- |
| Retained nominal, 600 g/L | 103.5 | 172.5 | 1.1061 | reference |
| Stock 60 % w/w, 1.27 kg/L | 101.5 | 133.1 | 1.0667 | glucose 7.6 %, acetate 18 % |
| Stock truly 600 g/L (49.3 % w/w), controller assumed 60 % | 83.4 | 138.9 | 1.0725 | biomass 11.5 %; glucose and acetate up to about 90 % at low concentrations |

F11 is similar (largest changes 7.7 % and 18 % for the 60 % w/w stock; 11.7 %
biomass and up to 93 % glucose for a true 600 g/L stock). The largest effect is
therefore the glucose dose (about 83-102 g), not the solution volume. The
coefficients were calibrated by the authors under their own feed accounting, so
fit quality cannot decide between these scenarios.

**Decision (WP3.1).** Source records that could settle the stock preparation,
the dosing controller's assumed concentration, or the meaning of the 1336 g/L
conversion are not available. The nominal 600 g/L scenario is therefore
retained as the published-model reproduction, and the 83-102 g delivered-dose
range is a documented, unresolved input uncertainty. Agreement with the F10/F11
measurements is reported as reproduction under that assumption, not as
validation of the delivered feed. Any recalibration (WP3.3) must carry this
uncertainty rather than fixing a single dose.

## Oxygen uptake and its limits

The direct F10/F11 mechanisms now report the paper's Eq. 19 oxygen demand using
native algebraic outputs: `specific_oxygen_uptake_liquid0` [g O2/(gDW h)] and
`oxygen_uptake_liquid0` [g O2/(L h)]. The latter is specific demand multiplied
by biomass concentration. This implements Eq. 19 as printed; it does not
establish an independently closed elemental/respiratory balance. The direct-model regression fixture checks these outputs alongside carbon
trajectories. The reconciliation demonstrator does not report oxygen demand.

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
alone do not establish the magnitude of growth limitation.

**Decision (WP3.1).** Oxygen is represented by the reported Eq. 19 demand plus
this applicability statement; no dissolved-oxygen state is added. The transfer
coefficient, agitation and aeration control records needed for Eq. 20-21 are not
available, and adding them would require unsourced parameters. Predictions
during the recorded low-DOT periods may therefore overstate growth if oxygen
limited it; that limitation is neither modeled nor excluded.

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
