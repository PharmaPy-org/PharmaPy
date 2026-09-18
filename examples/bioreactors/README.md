# Native bioreactor demonstrators

These examples use `build_bioreactor` to construct standard PharmaPy
`LiquidPhase`, biological `Mechanism`, and native `BatchReactor` or
`SemiBatchReactor` objects. JSON is only a reproducible serialization of the
scientific and operating inputs; it does not invoke a separate simulator.

The shared construction path is:

`input mappings -> LiquidPhase + Mechanism -> native reactor -> solve_unit`

- `batch_ecoli_dfba`: arbitrary-pathway batch dFBA and published qualitative benchmark.
- `fed_batch_cho`: input-declared kinetic rules, rate-reconciled MFA, and fed-batch events.

Biological-rate providers share a validated native mechanism contract. The
registered `rule-graph`, `affine-surrogate`, and `hybrid` providers can be
selected through the mechanism input; provider identity, units, validity-domain
status, and composition are retained in runtime diagnostics.

## Editable workflow notebooks

Each example's `workflow.ipynb` now runs the example tasks directly through
PharmaPy APIs instead of importing `run_*.py`. Numbered markdown sections and
Python comment blocks describe what to edit, required units, and how to execute
each cell with Shift+Enter. Configuration is loaded into editable dictionaries;
construction, propagation, result extraction, plots, and inference are visible.
The shared `run_design_study` library coordinator retains the full calibrated
synthetic design/refit/refinement study. It is not a laboratory data-acquisition
workflow. Redundant command-line runner scripts are not distributed.

Default execution leaves packaged inputs and outputs unchanged. A custom case
skips reference regression when its configuration differs; acceptance checks
for physical results and the declared study remain active. Optional export goes
to a separate folder and includes the effective input dictionaries. See each
notebook's introductory instructions before adapting species or model families.

## Reproduce and check the demonstrator

From a source checkout, using Python 3.11:

```sh
python3.11 examples/bioreactors/verify_installation.py
```

This creates a new isolated environment outside the checkout, installs the exact
versions in `requirements-py311.txt`, runs `pip check`, and executes `tests/Bioreactor`
including both full notebook workflows. Packaged simulation and inference outputs
are preserved as references. It exits nonzero on failed acceptance and writes
`installation_report.json`. Allow time for repeated dynamic fits. The environment
is retained; `--environment /path/to/new/environment` chooses its location.

The qualification scope is Python 3.11 with the examples' SciPy and fixed-step
backends. Assimulo/IDA and other operating systems are not qualified by this
command. This is source-checkout execution, not installation through the legacy
root `requirements.txt`, which also requests optional Assimulo. No inherited
`PYTHONPATH`, external dependency overlay, or existing PharmaPy environment is used.

After setup, open either `workflow.ipynb` using that environment and execute its
cells in order. For a forward simulation alone, run Steps 1–6. Library callers
can use `build_bioreactor(case, mechanism, thermo_path).solve()` directly.

### Explicit experiment-design acceptance

Both flagship `inputs/design.json` files select `acceptance.profile: capability`
for the WP3 methodology demonstration. Reports distinguish `capability_status`
from `empirical_benefit_status`; top-level `status` applies to the declared profile.
The capability profile requires replay, equal cost, stable model-based uncertainty
benefit, and finite confirmation results. Failed fitting or numerical solutions
still raise errors. Independent simulation/accounting checks remain mandatory in
the complete verification command.

The `predictive-benefit` profile additionally requires lower mean withheld MSE
in the declared noisy confirmations. It remains the default when no profile is
provided. Neither profile permits skipping the numerical/capability gates.
`OBSERVED` means benefit in that finite synthetic sample, not statistical proof;
`NOT_DEMONSTRATED` retains an unfavorable or tied comparison, and `INVALID`
identifies nonfinite confirmation diagnostics. Capability PASS must never be
presented as empirical predictive validation.

### Packaged evidence

`installation_report.json` records the historical environment that generated the
packaged reports; its runner names describe that earlier execution. Running the
current verification command replaces only the installation report with the new
test outcome. Scientific input values, reference outputs, seeds, noise, and
acceptance thresholds are not automatically refreshed.

The selected CHO design predicts a 17.26% local parameter-SD reduction, but its
five-seed mean standardized withheld MSE is 0.14326 versus 0.13769 for the matched
control. This small conditional comparison does not demonstrate predictive
superiority. Local information improvement and realized noisy prediction benefit
are different claims. CHO remains `NOT_DEMONSTRATED` for empirical benefit and
would still fail the `predictive-benefit` profile.

## Evidence and baselines

Each example packages its scientific references under `documentation/` and records
evidence checksums in `inputs/source_manifest.json`. Current endpoint checks use
`inputs/regression_baseline.json`: generated model outputs with input/source hashes
and generation metadata, **not experimental measurements**. Previously untraced
constants are retained there as superseded history. The explicit maintenance
command `PYTHONPATH=. python examples/bioreactors/refresh_regression.py` changes baselines;
normal verification never refreshes them. Review scientific changes before using it.

Each flagship folder also includes one executable `workflow.ipynb` that calls
the PharmaPy library directly without writing artifacts. The notebooks demonstrate the
complete simulation plus E. coli parameter estimation/design or CHO experiment design;
they do not contain an alternative model or execution path.
