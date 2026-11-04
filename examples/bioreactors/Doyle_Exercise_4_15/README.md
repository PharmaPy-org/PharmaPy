# Doyle — Exercise 4.15

Open `workbook.ipynb` with the PharmaPy Python environment and run all cells.
The notebook shows the six explicit forward-simulation steps: choose folders,
load/edit inputs, construct and solve, extract trajectories, plot, and export.
It does not call a runner or apply benchmark scenario overrides. The nominal
input is an equilibrium; Step 2 shows how to request a dilution-step transient.

For the separate automated numerical comparisons, from this folder run:

```sh
PYTHONPATH=../../.. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python run_benchmarks.py
```

Only this example's scenarios are declared in `benchmark.json`. Inputs and the
unexecuted notebook are committable; generated `outputs/` remain local and ignored.
The runner is self-contained, requiring no Alterable files.

## Optional synthetic hybrid and estimation connection

`inputs/hybrid.json` is an explicitly selected extension of the same model;
`inputs/mechanism.json` and the forward workbook retain the original physical model; named kinetic selection changes its declaration.
Both run through the existing rule-graph provider and native reactor balances.
For substrate S [g/L], the hybrid uses
`mu = mu_Monod * exp(theta * S/(S + S_ref))` [1/h].
The fixed feature scale S_ref=1 g/L equals the textbook Ks; it is a numerical
design choice, not another fitted biological affinity. Only theta [-] is fitted
in the initial qualification. Theta=0 recovers the mechanistic model exactly.
The declared range |theta|<=ln(2) restricts the positive multiplier to between
one half and two for nonnegative substrate. This range is an explicit synthetic
modeling assumption, not a literature calibration range.

Biomass production and substrate consumption use the same corrected growth,
preserving `r_S + r_X/Y = 0`. In nominal sterile continuous culture,
`S+X/Y` stays at 10 g substrate-equivalent/L. The declared S range [0,10] g/L
and X range [0,5] gDW/L follow this nominal invariant with Y=0.5 gDW/g substrate.
These limits are specific to this benchmark. Formal weights of 1000 g/mol make
the numerical mmol/L values equal to g/L here. The hybrid graph nevertheless
explicitly converts each species using its declared g/mmol factor before
applying mass-basis equations and domain bounds; the callback does not infer
that equivalence for other models. No complete elemental or oxygen closure is claimed.

The public `simulate_measurements` function connects the existing input builder
and observation mapper to shared `ParameterEstimation`. It accepts explicit
parameter order, returns explicit field order/units, and reconstructs the model
for every trial so feeds and numerical histories do not leak across evaluations.
See the [user guide](../../../doc/online_docs/bioreactors/USER_GUIDE.md#4-predict-measured-fields).

Run the local synthetic qualification from the repository root:

```sh
python -m pytest tests/Bioreactor/test_bioreactor_estimation_handoff.py -q
python -m pytest tests/Bioreactor/test_bioreactor_estimation_stress.py -q
```

The tests independently integrate concentration balances, check the yield
invariant, neutral equivalence, zero inventories, domain failures, repeated calls,
legacy inputs, field/parameter ordering and repeated sampling. The shared
estimator fits theta from noiseless synthetic biomass/substrate observations
over the initial 8 h transient, including unequal sampling grids. Both fitted
fields use g/L, with unit numerical residual scales of 1 g/L; no experimental
assay uncertainties or statistical confidence are inferred from this noiseless fit.
The synthetic
truth theta=ln(1.2) is a declared small positive correction; the held-out dilution
0.12/h is assessed after the nominal 0.1/h fit. This is verification of the
connection and model, not experimental training.

The separate stress suite uses the existing `synthetic_design.json` noise
assumptions: independent Gaussian errors with standard deviation
`sqrt((0.001 g/L)^2 + (fraction * concentration)^2)`, at 1% and 5% relative
noise. It checks two seeded realizations, three interior starting coefficients,
and unequal sampling grids at D=0.06/h and 0.14/h. Fits are compared with an
independently integrated and minimized weighted objective, then predictions
at the held-out D=0.12/h are compared with that reference fit. The 5e-4
coefficient comparison budget covers forward-difference and integration error;
it is not a confidence interval. Two realizations test reproducibility and
handoffs, not statistical coverage or the full 20-realization design.

Additional checks cover exhausted optimizer budgets, domain-error recovery,
finite-difference and integration refinement, reduced information in short
observation windows, singular information from initial-only measurements,
substrate depletion, washout, and repeated reconciled-model evaluations.
No new optimizer or solver is introduced. This qualification does not replace
experimental validation or statistical coverage studies.
Explicit finite-difference steps are checked at a zero coefficient; analytic
parameter sensitivities are not supplied. The broader WP3.2 scientific and
integration acceptance remains separate from this synthetic qualification.

## Complete calibration and validation workflow

Open `calibration.ipynb` from the repository root or this example folder and run
its six steps. It is a companion to the forward `workbook.ipynb`.
From the repository root the equivalent commands are:

```sh
python -m examples.bioreactors.Doyle_Exercise_4_15.calibrate
python -m examples.bioreactors.Doyle_Exercise_4_15.calibrate --replay --refit
```

Use `--output /path/to/new-directory` to preserve existing results. A nonempty
destination is rejected. Generation and fitting are separate functions in the
existing example; the generator itself still performs no fitting.
Optional `--settings` and `--design` select other explicit JSON inputs for a
new run. Replay uses the saved inputs and rejects replacement settings.

`inputs/calibration.json` declares the nonneutral truth, selected noise level
and realization, parameter seed, finite-difference step, optimizer settings,
field/parameter order and replay budgets. `inputs/synthetic_design.json` retains
the 15 reporting times through 100 h, the two entire training experiments at
D=0.06/h and 0.14/h, and the untouched D=0.12/h validation experiment. The
generator independently cross-checks DOP853 and Radau. Only one alternative
realization is fitted; the 20 generated realizations are never pooled as
additional biological experiments. Initial states and mechanistic parameters
remain exact synthetic assumptions.

The workflow passes known positive synthetic standard deviations through
`Measurement.uncertainty`, builds shared `Experiment` objects and calls existing
`ParameterEstimation` with `simulate_measurements`. The zero-noise control uses
an explicitly declared numerical residual scale rather than dividing by zero.
The saved inverse-information matrix is not a statistical uncertainty estimate
for that control. LM does not enforce recorded domains; invalid native trials
raise. `tol_fun=0` deliberately disables residual-norm termination for this
noisy-data qualification; gradient, step and evaluation-budget criteria remain active.
An exhausted budget is saved as nonconvergence and the CLI returns a nonzero
exit status. Source observations are not modified or clipped.

Generated evidence includes:

- Original observation CSVs, truth and generator manifest under `data/`.
- Native input JSON for each experiment, copied thermo input, saved numerical
  configuration, and separate `*-fitted.json` files under `inputs/`.
- `predictions.csv` with experiment/field identity, held-out membership,
  observations, fitted and neutral predictions, residuals and numerical scales.
- `fit-residual-layout.csv` from the shared estimator, showing exactly which
  observations participated in fitting.
- `calibration-record.json`, saved by the native shared record API.
- `fit-report.json` with convergence status, accepted theta, absolute-form
  linearized covariance, training/validation errors, code revision and dirty
  state, versions and content checksums.

Replay verifies assets and model-source checksums, rebuilds from the saved native
inputs, and compares predictions. `--refit` uses the native record's estimation
and optimizer inputs, then checks the rerun through `CalibrationRecord.check_rerun`.
Only the verified thermo asset path is rebound when the directory moves. The directory can be relocated; no original dataset
path or serialized executable is needed. Use the same model source when replaying.
Checksums detect changed content, not malicious modification or authenticity.

This uses the native `CalibrationRecord` API on the reconciled
`MPVRefactor-formerge` baseline `fa47a388`. `calibration-record.json` stores the
training experiments, measurements, parameter order/units, source provenance,
solver settings and fitted covariance through that shared API. Held-out data
remain in the source CSV and verification outputs, outside the fitted record.
The legacy forward-record adapter continues to report `calibration: not_performed`.
No empirical calibration, complete elemental closure, joint identifiability or
WP3.3 experiment-design acceptance is claimed.

## Input field guide: direct-kinetics template

### Named selections and the native estimator

The [user guide](../../../doc/online_docs/bioreactors/USER_GUIDE.md#select-a-named-relationship)
shows a drop-in named Monod selection in the existing hybrid input. The
[catalog](../../../PharmaPy/Bioreactors/TECHNICAL_SUMMARY.md#equation-and-physical-basis-contract)
also documents Haldane–Andrews, first-order death, and the existing Pirt and
Luedeking–Piret compositions. Preserve the raw named declaration when changing
trial parameters: the public callback replaces parameters **before** normalization,
then constructs a fresh native reactor for each evaluation. An already expanded
graph is also supported, but will no longer retain its original named selection.

The fitting route is unchanged:
`ParameterEstimation -> simulate_measurements -> build_bioreactor ->`
`normalize_forward_inputs -> expand_relationships -> configured rates ->`
`native reactor/ScipyBackend -> predict_measurements`.
Pass the selected raw `model` as `Experiment.kwargs['case']`, bind scalar paths
with `parameter_names`, and use the observation declaration's field insertion
order for `output_names`. Measurement labels do not convert concentrations;
the observation declaration does. Each trial retains fixed inputs and starts
from the declared initial state and recipe.

`tests/Bioreactor/test_bioreactor_named_kinetics.py` supplies executable, small
recovery demonstrations using native `Experiment`, `Measurement` and
`ParameterEstimation.optimize_fn(method='least_squares', bounds=...)`:

- Monod and substrate-inhibited growth fit only `rates.mu_max`; affinity,
  inhibition, yield and neutral hybrid correction stay fixed. Independent
  concentration balances supply the synthetic truth, including the inhibition
  term when selected. This does not establish joint identifiability.
- A batch death-only model fits `rates.death` to `X0 exp(-kd t)` and reconstructs
  an ordinary `build_bioreactor` forward run with the accepted coefficient.
- Both input layouts and unequal formal molecular weights are checked against
  explicit equations; invalid affinity, output dimensions and provider domains
  remain errors. Existing handoff/stress/workflow tests qualify ordering, event
  sides, noisy multistarts, missing observations, derivatives and saved replay.

Run the focused demonstrations from the repository root:

```sh
python -m pytest tests/Bioreactor/test_bioreactor_named_kinetics.py
```

These recovery tests use declared positive numerical residual scales for
noise-free data, not empirical assay errors. Bounds are passed to the actual
bounded optimizer; the original LM workflow above remains unbounded and rejects
invalid provider trials. The death fixture explicitly removes biomass without
a dead pool and therefore does not claim material closure. Named forms supply
no analytic sensitivities; numerical differences use the same native forward
equations. Source hashes from the existing calibration workflow identify the
actual implementation in addition to its Git revision and dirty state.

The additional `test_bioreactor_named_kinetics_stress.py` checks inhibition ratios
and concentration rescalings, long-horizon productive/washout attraction basins,
coupled product/maintenance/death accounting, zero inventories and hazards,
parameter sensitivities, and bounded recovery from multiple seeds. These fixtures
are synthetic numerical stress cases, not extensions of biological validity.
Stricter absolute prediction or derivative budgets can require tighter integration
tolerances; local solver tolerances are not guarantees of global prediction error.

For bounded `least_squares` starting exactly at zero, default relative-cost
termination can report success before a meaningful step away from the bound.
The synthetic death stress tests use `optim_options={'ftol': None, 'xtol': 1e-12,
'gtol': 1e-10, 'max_nfev': 128}` and an explicit finite-difference step [1/h].
These are qualified settings for those fixtures, not universal optimizer defaults.
Inspect residuals, optimality, accepted parameters and termination reason;
`success=True` alone does not establish parameter recovery or identifiability.

This example supplies a textbook numerical reference, not measured cultivation
data. Use its structure for direct kinetics; do not treat its coefficients or
lumped species properties as universal biological defaults.

| Section | Category | What to configure |
| --- | --- | --- |
| `schema_version`, `mechanism` | Required for this model | Keep version 3 and `configured-kinetics` for this formulation. |
| `$comment` | Descriptive | Explain the case; it does not affect the equations. |
| `operation` | Required for this model | Continuous mode, working volume, carrier, duration and selected recipe. The 1 L volume and density are benchmark conventions. |
| `state.species` | Required for this model | Initial amounts and their unit (`g` here). Product is intentionally zero and inactive, not a missing input. |
| `parameters` | Required for this model | Monod maximum growth rate, affinity and biomass yield; keep unused groups empty. |
| `model.kinetics` | Required for this model | Explicit rate graph, species mass-rate mapping, constants and rate units. |
| `recipes` | Required for this model | Sterile feed flow and composition; continuous operation supplies withdrawal through the native reactor. |
| Other model sections, `network`, `rate_provider`, `transport`, blank state fields | Optional | Leave the inactive slots empty/null; they are not needed for this direct-kinetics formulation. |
| `numerical` | Numerical | Solver, requested output grid, tolerances and maximum step. |
| `thermo.json` | Required for this model / optional properties | Preserve formal molecular weights and density conventions described below. Unused thermal properties remain blank. |
| `benchmark.json` | Numerical verification / scenario inputs | The verification runner uses declared scenario changes and acceptance limits. This is not a third runtime input required by `build_bioreactor`. |

For a different process, change equations and parameters together with their
units. Preserve the distinction between initial amounts and concentrations.
The separate verification runner applies declared benchmark scenarios; editing only
the nominal mechanism does not replace those overrides.

### Source and methodology

Source: Seborg, Edgar, Mellichamp and Doyle, *Process Dynamics and Control*,
fourth edition (Wiley, 2017). Exercise 4.15: printed p. 67 (PDF p. 79).
No copyrighted page images are redistributed here.

## Purpose and execution

These are numerical reproductions of a textbook model, not experimental data,
parameter estimation, or complete elemental-balance validation. They run through
unchanged `build_bioreactor -> NativeBioreactorAssembly.solve ->
ContinuousReactor.solve_unit -> ScipyBackend` using `configured-kinetics`.
They do not qualify the pathway LP or rate-reconciliation optimizer.

From an environment containing the existing PharmaPy dependencies:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH="/absolute/path/to/Bioreactor WP3.1" \
python run_benchmarks.py
```

The mechanism JSON file specifies every biological equation and coefficient.
`inputs/mechanism.json` includes physical `operation` and solver `numerical` settings.
`benchmark.json` specifies operating scenarios and acceptance limits. The runner
changes only the declared initial amounts and inlet flow for each scenario and
saves the complete resolved case/mechanism inputs beside the generated outputs.
Reference equations are independently written concentration balances; they do
not call the production rate evaluator, vessel balances, or rule interpreter.

## Inputs, units, and assumptions

- This example uses a perfectly mixed, isothermal continuous culture with
  sterile substrate feed, fixed working volume, and no cell retention.
- Working volume 1 L and density 1 kg/L are explicit benchmark conventions,
  not textbook measurements. Volume cancels from the concentration equations
  because feed flow is set to dilution rate times volume.
- Biomass, substrate and product are unspecified lumped masses. Formal molecular
  weights of 1000 g/mol make each numerical concentration in mmol/L equal to
  its value in g/L (1 g per formal mmol). This is a bookkeeping conversion,
  not a chemical identity. Water has MW 18.015 g/mol. No thermal properties or
  organism-specific constants are inferred.
- Kinetic rates returned by the graph are g/(L h). Native integration uses
  kg, m3 and seconds. No empirical correction, clipping of the textbook growth
  equation, or fitted coefficient is introduced.
- The substrate-plus-biomass invariant is the one implied by the printed
  model. Matching it does not establish complete elemental conservation.

## Exercise 4.15

The Monod model uses mu_max=0.20/h, Ks=1 g/L, Y=0.5 gDW/g substrate,
feed substrate=10 g/L. Initial X=4.5 gDW/L and S=1 g/L are the stated
equilibrium for D=0.1/h. Product is an inactive zero-valued bookkeeping species.
The +/-10% dilution steps and 100 h horizon are declared extensions for
transient verification; the exercise itself asks for a linearized model.

Balances are X'=(mu-D)X, S'=D(Sin-S)-mu X/Y, with mu=mu_max S/(Ks+S).
The exact positive equilibrium is S*=Ks D/(mu_max-D), X*=Y(Sin-S*), where
positive biomass is feasible. Its transfer function can independently be
derived; full nonlinear histories are compared with separate integrations.

## Predeclared acceptance

`benchmark.json` was fixed before the first simulation. Maximum full-history
absolute concentration discrepancy must be <=1e-5 g/L (well below the book's
two-decimal initial-state precision). Independent DOP853 and Radau reference
solutions must agree within 1e-7 g/L. Tightening native relative/absolute
tolerances tenfold and halving maximum step must change predictions by
<=1e-5 g/L. The analytical equivalent-substrate balance must hold to
1e-6 g/L, constant volume to 1e-9 L, and nonnegativity to 1e-9 g/L.

The reference identity is:

`S + X/Y = Sin + (S0 + X0/Y - Sin) exp(-D t)`.

Reference integrations use rtol=1e-11, atol=1e-12 g/L, max step=0.1 h.
Native baseline uses rtol=1e-9, atol=1e-12 in native units, max step=0.25 h.
All histories are compared every 0.25 h; these are reporting points, not an
assumption that the adaptive integrator uses fixed steps.

## Output and scope

`outputs/report.json` records source revision, runtime environment, input and runner
checksums, measured errors and all acceptance decisions. Each scenario has
resolved inputs, `trajectories.csv`, `comparison.png` and `metrics.json`.
Generated outputs are ignored by Git. Keep the source inputs, runner and this
documentation if later committing this standalone benchmark package.

This benchmark is a focused numerical qualification, not a clean-installation
sign-off or independent experimental validation.

## Numerical fit acceptance

The calibration report distinguishes solver termination from local numerical
acceptance. `fit_acceptance` checks the feasible Gauss–Newton displacement for
the single correction parameter at two secant resolutions, using the declared
replay parameter budget and the provider's validity bounds. `converged` requires
both a successful stopping reason and this assessment; replay with refitting
applies the same check. A heavily damped, unchanged seed can therefore report
`Small step` while remaining unaccepted. Training observations alone enter this
assessment. It does not establish a global optimum, biological adequacy, or
confidence-interval coverage. Exported covariance remains local inverse
information and must be interpreted alongside acceptance and noise assumptions.


## Kinetics selection

Select `monod` in the `mu` rule of `inputs/mechanism.json`, or the `mu_mechanistic` rule of `inputs/hybrid.json`. Inputs bind `rates.mu_max` [1/h], `affinities.Ks` [g/L], and substrate. The hybrid retains its explicit mass conversions and exponential correction. The calibration workbook still fits only `corrections.theta`; its independent synthetic-data generator assumes Monod growth. A different truth/model requires a separately qualified calibration design.

The notebook exposes `kinetic_forms = relationship_catalog()` with supported names, aliases, required inputs, equations and units. For rule graphs, change `relationship` and supply every required input; for pathway optimization, keep the distinct uptake-constraint interface. No parameter, yield or physical-basis conversion is inferred.

Named Monod/Haldane forms are equivalent to the former rational expressions on the physical nonnegative-substrate domain. They inherit the existing zero-uptake extension for negative trial substrate; negative inventories are not validated biological states. Signed product inhibition and custom hybrid equations remain explicit. The numerical and independent-reference checks retain their existing acceptance budgets.

### Configurable correction

`inputs/hybrid.json` now declares the existing multiplier with the generic
`correction` convenience on the `mu` expression. Its base remains the separately
named `mu_mechanistic` rate. Select that base relationship independently, and set
`parameters.corrections.theta` [-] to choose correction strength; zero retains
the base exactly. The reference concentration remains the declared
`feature_scale_g_l` constant [g/L]. Growth and substrate consumption continue to
share the corrected rate, and all existing validity limits remain in place.

The [generic correction contract](../../../doc/online_docs/bioreactors/USER_GUIDE.md#optional-configurable-kinetic-correction)
also permits attaching the same `correction` object directly to any named
relationship. `input: "substrate_g_l"` binds the existing converted graph rule;
it does not ask the simulator to infer a species or its units. This syntax selects
an explicit hypothesis; fitting theta or selecting correction forms from data is
a separate workflow. Existing explicit operator expressions remain valid.

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
