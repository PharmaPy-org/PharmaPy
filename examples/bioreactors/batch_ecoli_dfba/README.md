# Native E. coli batch dFBA

## Calibrated experiment design

Run the experiment-design cells in [workflow.ipynb](workflow.ipynb)
from the repository root to regenerate `outputs/experiment_design.json`.
The study settings and scientific rationale are in `inputs/design.json`.

This is a full offline qualification, not a quick screen: one baseline fit,
ten follow-up joint fits, and sensitivity/solver refinements can take tens of
minutes. The notebook executes the same complete study without rewriting outputs.

This separate synthetic study calibrates uptake capacity from a low-glucose
batch, then selects an initial loading for another batch using the fitted
parameters and absolute baseline information. Biomass measurements at 0.5, 1, and 2 h
are used for fitting; predictions at 4.5 and 5 h at the flagship loading are
withheld. This does not replace the glucose/biomass estimation demonstration below.

An initial runtime-only pilot used a 1–4 h fitting window. The final early-growth
window avoids repeatedly resolving low-loading depletion inside every fit;
confirmation outcomes were not used to select it. Seeds and benefit thresholds
were retained, and later predictions still test extrapolation beyond fitting.

Both examples call the same `PharmaPy.Bioreactors.experiments` coordinator and
native construction/estimation/design APIs. For independent experiments it adds
`F_total = F_baseline + J.T @ inv(R) @ J`, in the declared parameter order.
The prior is computed from weighted calibration sensitivities, not by inverting
residual-scaled fitted covariance. Reported absolute local SDs come from
`sqrt(diag(inv(F_total)))`. No regularization is used to disguise rank deficiency.

The selected experiment is compared with an equal-cost repeat of the baseline
using the same assays and timing. Five predeclared, paired synthetic-noise seeds
generate new observations; each arm is jointly fitted with the original data as
separate reactor experiments, never as a concatenated artificial timeline.
The report includes observations, fits, expected SD reduction, individual and
aggregate withheld errors, paired-benefit standard error, and numerical checks.
Baseline noise is held fixed across repetitions; these are small conditional
synthetic comparisons, not independent experimental validation or statistical
certification. A better design need not win on every noisy realization.

The declared WP3 `capability` profile requires successful locally identifiable
fits, replay, equal cost, finite confirmation diagnostics, at least 1% predicted
SD reduction versus the matched repeat, and stable selection/expected benefit
when sensitivity steps are halved/doubled and integration tolerances are tightened
tenfold. Numerical failures still block acceptance. `capability_status` and
`empirical_benefit_status` are reported separately; top-level `status` applies
to the declared profile.

The general `predictive-benefit` profile additionally requires lower mean
withheld squared error and remains the default if the profile is omitted.
`OBSERVED` means benefit in the finite synthetic comparison, not statistical
certification; unfavorable or tied results remain `NOT_DEMONSTRATED`. Nonfinite
results are `INVALID` and fail both profiles. Thresholds, seeds, and illustrative
measurement noise remain explicit inputs, not industry-wide accuracy standards.

This study uses a declared constant density of 1000 kg/m3 for observation
conversion and each sample's recorded mass. The framework does not infer assay
units or supply case-specific kinetic values.

The qualified run selected 0.0108 mol initial glucose and reduced expected
parameter SD by 10.36% against the matched repeat. Mean withheld standardized
squared error was 486.89 versus 555.93 for the control (baseline-only: 1153.00).
These correspond to absolute pooled RMS biomass errors of approximately
0.002207 and 0.002358 gDW/L, respectively. They remain much larger than the
illustrative 0.0001 gDW/L assay SD: this design study demonstrates a relative
improvement, not satisfaction of the separate estimation example's 3-SD gate.
The paired mean benefit was 69.04 with standard error 65.45 over five seeds;
that small conditional comparison does not establish statistical significance.

[Scientific sources, packaged evidence, and validation limitations](documentation/README.md).
The source manifest's evidence paths resolve locally within this example.

Reproduces the qualitative diauxic-growth behavior reported by Mahadevan et al.
(2002) using a native `LiquidPhase`, arbitrary-pathway `PathwayMetabolism`, and
native `BatchReactor`. All pathway identities, coefficients, constraints, state,
and transport are declared in `inputs/mechanism.json`; source attribution is
kept separately in `inputs/source_manifest.json`.

Run Steps 1–6 of [workflow.ipynb](workflow.ipynb) for forward simulation.

Run Steps 7–10 of the notebook for the dynamic parameter-estimation demonstration.

The estimation workflow integrates a fresh native batch reactor at every trial
`glucose_uptake_max`. It fits glucose and biomass at 1, 2, 3, and 4 hours and
evaluates predictions at withheld times 4.5, 5, 5.5, 6, and 7 hours without refitting.
All times, measurement definitions, parameter bounds, noise covariance, and
acceptance thresholds are declared in `inputs/estimation.json`. Hours are
converted to solver seconds; concentration values are never used as times.

Measurements are explicitly synthetic: the same reactor model generates the
noise-free trajectory with the declared true parameter, then a seeded Gaussian
noise model generates observations. The assumed standard deviations (0.03 mmol/L
glucose and 0.0001 gDW/L biomass) are illustrative assay errors, not a claim about
published experimental uncertainty. These sampling times cover growth before
substrate exhaustion and reserve later times, including glucose depletion, for
prediction assessment. The pathway LP uses feasibility tolerances tighter than
its unchanged constraint audit. `case.json` selects controlled numerical Jacobian
differences for the optimization-coupled BDF integration: perturbation j equals
`jacobian_relative_step * max(abs(y[j]), jacobian_state_scale[j])`. The scale can
be a positive scalar or a vector in native solver-state order and units; this
example uses a 1e-6 kg scale for its mass states. These are numerical settings,
not physical inventory floors. Other SciPy processes keep their default Jacobian
behavior unless these options are supplied.

Acceptance requires optimizer convergence, full local sensitivity rank, parameter
recovery within 5%, training and withheld standardized RMSE at most 3 for each
measurement, and predictions changing by no more than 0.1 measurement standard
deviation when integration tolerances are tightened tenfold. The report records
every observation, its split, noise-free value, fitted prediction, and metric.
The known truth and held-out values are used only for generation and assessment;
the optimizer receives training observations only. Reported parameter covariance
is a local residual-scaled approximation, not a guarantee of global identifiability.
This demonstrates dynamic inference and extrapolation on synthetic data, not
independent experimental validation. The command exits unsuccessfully if any
declared acceptance check fails.

Executable workflow notebook: open
`examples/bioreactors/batch_ecoli_dfba/workflow.ipynb` in VS Code or Jupyter and
run all cells. With Jupyter installed, execute it unattended from the
repository root with:

`jupyter nbconvert --execute --to notebook --output /tmp/ecoli_workflow.ipynb examples/bioreactors/batch_ecoli_dfba/workflow.ipynb`

The repository test suite also executes the notebook without making Jupyter a
runtime dependency:

`PYTHONPATH=. python3 -m pytest -q tests/Bioreactor/test_bioreactor_workflow_notebooks.py -k ecoli`

The notebook performs configuration loading, native construction, simulation,
trajectory extraction, plotting, and inference directly through PharmaPy APIs;
it does not import the example runners. Each numbered step has a comment block
explaining what to edit and how to execute the cell (Shift+Enter). Restart and
run in order after changing configuration. The full synthetic design study
still includes calibration, candidate ranking, matched-control refits, and
numerical refinement checks.

Copy the notebook and compatible `inputs/` files for a new case, update
`example_dir`, and edit the Step 2 dictionaries. Initial species identifiers,
units, parameter paths, observation definitions, and schedules must agree.
The concentration conversions assume the declared constant density. For the
E. coli standalone fit, the visible adapter is a batch pathway template;
scheduled culture observations use the shared event-aware design adapter.

Default execution writes no artifacts. Original-example regression runs only
when all input documents and thermodynamic properties match the distributed
case; changed cases report `NOT_CHECKED` for that comparison, without skipping
physical or study acceptance checks. Set `WRITE_ARTIFACTS=True` to export to
`EXPORT_DIR`, which defaults to a separate `custom_results/` folder. Synthetic
truth-based assessment must be replaced when using real observations.

Outputs include the full trajectory, plot, validation result, balance audit, and
summary. Citation details are in `inputs/source_manifest.json`.

## Presentation and paper-layout figures

From the repository root in the configured environment, run:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. MPLBACKEND=Agg python examples/bioreactors/reproduce_figures.py
```

This reruns both canonical forward simulations, checks their model regression,
and writes PNG/SVG plots, plotted data, and generation metadata under
[`outputs/figures/`](outputs/figures/). Existing simulation artifacts and regression
baselines are not overwritten. E. coli pathway activities are re-evaluated with
the native LP at reported states; they are not the BDF internal stage history.
CHO product is plotted in g/L, and viable cells in million cells/mL, using each
time point's liquid volume.

The [side-by-side comparison](outputs/figures/paper_comparison.png) includes the
[published panel from the supplied presentation](documentation/published_figures/README.md).
CHO shows the current pH 7.0 reconstruction; the reference panel includes several
pH conditions. Figure layouts support qualitative inspection, not a claim of
independent experimental validation. The corrected CHO curves intentionally
retain their differences from the older implementation and published plots.
