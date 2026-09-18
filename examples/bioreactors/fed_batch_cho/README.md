# Native CHO fed-batch culture

**Physical accounting and convergence: PASS.**
Inventory-aware metabolic feasibility passes the declared three-mesh checks
without native limiter clipping. Historical endpoints no longer agree; active
regression checks use explicitly generated current-model baselines, not those
untraced values or experimental measurements. The packaged
`outputs/consistency.json` records quantitative accounting and three-mesh
convergence evidence. See
[the consistency review](documentation/README.md#coupling-review-and-current-acceptance-status).
The historical terminal regression fails after the correction; older design
and notebook results are not qualification of the corrected model.

[Scientific sources, packaged evidence, and validation limitations](documentation/README.md).
The source manifest's evidence paths resolve locally within this example.

Reconstructs the public Reddy CHO fed-batch case using a native `LiquidPhase`,
input-declared kinetic rule graph, rate-reconciled metabolic mechanism, discrete
feed/target events, and native `SemiBatchReactor`. Scientific identities and fitted
values occur only under `inputs/`.

The declared `rate_provider` selects the kinetic source implementation. The
same reactor lifecycle accepts registered mechanistic, affine-surrogate, or
hybrid providers without changing its balance equations.

Run Steps 1–6 of [workflow.ipynb](workflow.ipynb) for forward simulation and
Steps 7–9 for the calibrated pH experiment-design study.

`inputs/estimation.json` retains the separate product-trajectory fit configuration:
local productivity coefficient, pH 6.8 operating condition, product observations,
covariance, seed, training days 2, 3.5, 5.5, and withheld days 6.5, 8.5.
`outputs/parameter_estimation.json` is its packaged reference result. The notebook
performs calibration within the `design.json` study; a separate estimation
command-line script is not distributed. The shared observation adapter and
parameter-estimation API remain available for independently configured fits.

The separate fit's mesh criterion is 5% peak-relative half-step agreement, also
used by the consistency study. Its assay-SD refinement diagnostic is reported
separately from that mesh gate.

The full study includes eleven fits and sensitivity/solver refinements and can
take tens of minutes. The notebook reruns that full qualification without
rewriting the canonical artifacts.

`inputs/design.json` declares the decision, parameter bounds, samples, noise,
costs, seeds, and acceptance thresholds. A baseline culture at pH 6.8 calibrates
the local productivity coefficient using product concentration at days 2, 3.5,
and 5.5. These samples cross actual feed events. Candidate pH values 6.8, 7.0,
and 7.2 are evaluated at the fitted parameter, not at synthetic truth. Product
predictions at days 6.5 and 8.5 at pH 7.0 are withheld from all fitting.

This uses the same general study coordinator as the batch example. Each
observation is taken from its own recorded segment and volume, with an explicit
pre/post event convention. Fixed-step samples must lie on the declared mesh;
the solver does not silently change its step to match a measurement.
Volume conversion assumes the input-declared constant density 1000 kg/m3.

The study adds independent baseline and proposed Fisher information, and ranks
the total information using the existing D criterion. Baseline information comes
from weighted sensitivities at the calibrated parameter, not the inverse of
residual-scaled fit covariance. The report distinguishes incremental information,
total information, absolute local parameter SD, and fitted covariance.

Five predeclared paired seeds generate follow-up observations for the selected
pH and an equal-cost additional baseline culture. Both arms are jointly fitted
with the same initial calibration data. The report includes every new observation,
joint estimates, withheld prediction errors, paired-benefit standard error, and
expected SD reduction. Baseline noise is held fixed; this is a small conditional
synthetic demonstration, not statistically certified experimental performance.

The input explicitly selects the WP3 `capability` acceptance profile. It requires
all fits to converge with local identifiability, selected-design replay, equal
cost, finite confirmation diagnostics, and at least 1% expected SD reduction.
Selection and expected benefit must persist with half/double sensitivity
perturbations and a halved propagation step. Numerical failures still block
acceptance. The assumed product assay SD of 0.001 g/L is illustrative, not
measured experimental uncertainty.

The separate `empirical_benefit_status` reports whether the selected arm actually
has lower aggregate withheld error. An unfavorable result is retained as
`NOT_DEMONSTRATED`; it is not turned into favorable data by capability acceptance.
The general `predictive-benefit` profile additionally requires this observed
comparison to pass and remains the default for inputs without a profile.
`OBSERVED` means only the declared finite synthetic sample, not statistically
established superiority or independent validation. Nonfinite empirical metrics
are `INVALID` and fail both profiles.

The narrow coefficient interval expresses local calibration around the source
parameterization; the study does not identify an arbitrary CHO kinetic model.
Product observations now refer to the reconciled product state. The separate
`outputs/consistency.json` report records checks of inventory availability, metabolic constraints,
material accounting, and time-step convergence; it does not assert full elemental
closure for untracked pools.

The original stricter-policy run failed its realized-prediction-benefit gate.
That result is not retroactively relabeled PASS. The new capability-profile study reproduces
all of its numerical/evidence fields exactly and reports empirical benefit as
`NOT_DEMONSTRATED`. Selected/control mean standardized withheld MSE is
0.143257713/0.137690468 despite 17.26% expected local parameter-SD reduction.
Three of five paired seeds favor the selected arm; this small conditional study
does not establish predictive superiority, statistical significance, or independent
experimental validity. No numeric threshold or seed was changed. The selected
pH is an output, not a coded acceptance requirement.

The previous full verification had 52 passed and 3 failures from that gate.
Current notebook acceptance is against the explicitly selected capability
profile, while retaining trajectory/metric comparisons and empirical-status
reporting. See the parent folder's `installation_report.json` for the latest
environment and full-suite outcome. A capability PASS does not qualify empirical
predictive superiority.

The new uninterrupted pinned-environment verification passed all 67 tests,
including both notebook replays, on 2026-09-14. The CHO notebook reports
`capability_status: PASS` and `empirical_benefit_status: NOT_DEMONSTRATED`.

Executable workflow notebook: open
`examples/bioreactors/fed_batch_cho/workflow.ipynb` in VS Code or Jupyter and
run all cells. With Jupyter installed, execute it unattended from the
repository root with:

`jupyter nbconvert --execute --to notebook --output /tmp/cho_workflow.ipynb examples/bioreactors/fed_batch_cho/workflow.ipynb`

The repository test suite also executes the notebook without making Jupyter a
runtime dependency:

`PYTHONPATH=. python3 -m pytest -q tests/Bioreactor/test_bioreactor_workflow_notebooks.py -k cho`

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
summary. Citation and evidence limitations are in `inputs/source_manifest.json`.

### Numerical regression tolerances

The notebook compares generated trajectories and design-score improvement with
saved outputs using `rtol=1e-4` (0.01%) and `atol=1e-8` in each output's reported
units. The time grid must match exactly. Terminal relative-error fractions use
`rtol=0` and `atol=1e-7` (0.00001 percentage points), avoiding relative comparison
of an already-small error. Failures identify the output and numerical mismatch;
nonfinite values are rejected.

These are project-specific reproducibility tolerances, not a universal industry
standard. They allow margin over the observed optimizer/environment drift
(about 0.0024% in aspartate and 0.0012% in ammonia), while keeping the relative
trajectory tolerance 200 times smaller than the separate 2% terminal benchmark.
Simulation/design acceptance and the selected design must still pass.
The regression gate uses current saved outputs; tolerances are unchanged.

This follows [NumPy's guidance on selecting comparison tolerances for the use
case](https://numpy.org/doc/stable/reference/generated/numpy.allclose.html).

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
