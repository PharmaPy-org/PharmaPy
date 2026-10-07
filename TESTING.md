# Testing

See the canonical [installation guide](INSTALLATION.md) for environment setup,
package-name migration guidance, platform support, and verification details.

Run the core pytest slice without the optional Assimulo solver stack:

```bash
python -m pip uninstall -y PharmaPy pharmapy-sim  # when reusing an environment
python -m pip install -e ".[test]"
python -m pytest --collect-only
python -m pytest tests/ -m "not assimulo"
```

The CI core lane requires both Assimulo and cyipopt to be absent before it
collects tests, so their missing-dependency fallbacks are also exercised under
genuine absence. The backend-absence tests block the optional import inside an
isolated child process, so they also run in a richer local environment that
has either backend installed; use the explicit optional-backend environments
for installed-backend coverage.

Run both locked pixi test lanes:

```bash
pixi install --all --locked
pixi run test
pixi run -e assimulo test-assimulo
```

For the unlocked manual conda-forge fallback, run:

```bash
conda env create -f environment.yml
conda activate pharmapy-assimulo
python -m pip uninstall -y PharmaPy pharmapy-sim  # when reusing an environment
python -m pip install -e . --no-deps
python -m pytest tests/ -v -m assimulo
```

`python -m pip install -e ".[assimulo]"` installs only pip-available build
helpers for source workflows; it does not install Assimulo itself. Use the
conda environment above for the supported Assimulo test path.

GitHub Actions gates the locked pixi install on both Linux and Windows. The
Assimulo test job is intentionally informational and independent of that
two-platform matrix, so failures in the external solver stack do not block core
tests and its Linux signal is still reported if another platform fails.

## Real cyipopt parameter-estimation tests

The optional optimizer regressions exercise real IPOPT solves for correlated
GLS, staggered observations with three sensitivity modes, and bootstrap fit
preservation. A second module checks the public `SimulationExec` handoff with
a real CVode reactor, including an active parameter bound. Expected values come
from independent normal equations and the analytic first-order reaction.

Use a separate conda-forge environment; the core and locked Assimulo environments
deliberately do not install cyipopt. This manual environment is not locked:

```bash
conda create -n pharmapy-ipopt -c conda-forge python=3.11 numpy=1.26.4 \
    cyipopt=1.7.0 assimulo=3.4.3 scipy matplotlib pandas pytest pip
conda activate pharmapy-ipopt
python -m pip install -e . --no-deps
python -c "import cyipopt, assimulo; print(cyipopt.__version__, cyipopt.IPOPT_VERSION, assimulo.__version__)"
python -m pytest tests/test_paramestim_cyipopt.py tests/test_simexec_cyipopt.py -v
```

NumPy 1.26.4 satisfies the Assimulo 3.4.3 binary's NumPy constraint. The first
module needs only cyipopt; the second requires both cyipopt and Assimulo. Each
module skips at collection when its backend is absent, preserving the core
missing-dependency lane. Existing CI does not install cyipopt, so run this
explicit command when changing the IPOPT estimation path. Headless machines can
set `MPLBACKEND=Agg` and `QT_QPA_PLATFORM=offscreen` for these plotting imports.

MCR's scalar-gradient and post-solve result-assembly defects remain tracked in
[#240](https://github.com/PharmaPy-org/PharmaPy/issues/240#issuecomment-6040658764).
These tests cover ordinary `ParameterEstimation`; they do not encode known MCR
failures as expected success or certify MCR/IPOPT fitting.

## Parameter-estimation workshop regression

Run the solved parameter-estimation notebook in fresh kernels from both supported
working directories (repository root and notebook directory):

```bash
pixi run --locked -e assimulo python -m pytest tests/test_workshop_param_estimation.py -v
```

The notebook module skips during collection if Assimulo or any imported notebook
tool (`nbformat`, `nbclient`, or `jupyter_client`) is unavailable. This keeps core
tests runnable in older manual solver environments without the workshop tools.
The locked Assimulo environment supplies the tools for notebook execution;
`tests/test_workshop_collection.py` checks each missing-tool path in isolation.

The test executes every cell, including sensitivity analysis, parameter fits,
confidence intervals, and plots. It takes about 20 seconds for both launches and
also runs in the existing informational Assimulo CI lane on every push and PR.
It checks the initial four-species concentration trajectory against an independent
matrix-exponential solution and conservation of total reactive concentration.
The absolute concentration tolerance is 1e-5 mol/L (ten default CVode absolute
tolerances to allow accumulated integration error). Final Arrhenius parameters
are compared with the unchanged 2023 table in
`tests/fixtures/param_estimation_2023.json`: relative tolerance 1e-6 (CVode default)
plus an absolute 0.5e-6 in each column to account for the table's six-decimal rounding. Columns are ln(A / (1/s))
[dimensionless] and Ea/R [K]; reaction and column order are also checked.
This is a compatibility regression, not a statistical accuracy claim.

To refresh published outputs after the regression passes, use the installed
Assimulo environment's `python3` kernel:

```bash
pixi run --locked -e assimulo jupyter execute doc/online_docs/examples/param_estimation_solved.ipynb --inplace
pixi run --locked docs
```

Inspect the notebook diff, keep cell sources and output refreshes in separate
commits, and retain the historical fixture unchanged. Before committing outputs,
remove transient cell timing metadata and normalize local traceback paths to
repository-relative paths while retaining the complete warning text. Do not suppress warnings
or overwrite numerical references just to pass a test. Zero-inventory and
missing-Utility warnings currently remain visible because this historical
example fits concentrations without specifying a physical batch inventory or
utility. Their model-assumption review and broader workshop
certification work remain under
[#172](https://github.com/PharmaPy-org/PharmaPy/issues/172).

## PFR/batch process-optimization workshop regression

```bash
pixi run --locked -e assimulo python -m pytest tests/test_workshop_pfr_batch.py -v
```

Two quick cases run the nominal flowsheet and initial costing from the repository
root and notebook directory. A third case runs every cell, including both
original optimization budgets (10 and 400 function evaluations), in a fresh
kernel. The complete run takes about five minutes locally and runs in the
existing informational Assimulo lane on every PR and master push; no additional
CI job is required. Notebook tools remain optional at collection time, and the
missing-tool checks above cover both workshop modules.

The tests check species/cost ordering, feed volume, the cooling endpoint, and
crystallizer-to-filter solid-mass transfer. Initial raw-material costs retain the
unchanged historical values in `tests/fixtures/pfr_batch_2023.json`, with a 1e-10
relative tolerance for these algebraic calculations. Reactant costs are also
derived independently from feed moles, molecular weights, and prices.

The complete run checks finite bounded decision variables, the evaluation budget,
truthful optimizer-status reporting, and mean-size plotting after particles form.
It does **not** certify the historical final optimized design or feasibility:
current execution exhausts the original budget with a positive production
shortfall, while the original stored result also reported `success=False`.
Reproduction of historical numerical endpoints remains under
[#172](https://github.com/PharmaPy-org/PharmaPy/issues/172); do not replace reference
values or increase optimization budgets merely to make a regression green.

Refresh published outputs using the same output/metadata hygiene described above:

```bash
pixi run --locked -e assimulo jupyter execute doc/online_docs/examples/PFR_Batch_solved.ipynb --inplace
pixi run --locked docs
```
