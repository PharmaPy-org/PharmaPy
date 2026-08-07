# Dependency Policy

For runnable pip, pixi, and conda installation instructions, see
[`INSTALLATION.md`](INSTALLATION.md).

The default install keeps only the core runtime dependencies. PharmaPy uses
Assimulo to connect dynamic unit-operation models to the SUNDIALS CVode and IDA
integrators for ordinary differential equations (ODEs) and differential-
algebraic equations (DAEs), respectively. It remains optional because
solver-independent workflows do not invoke those integrations.

PharmaPy supports Assimulo 3.4.3, which conda-forge distributes as a compiled
package but PyPI does not provide for PharmaPy's supported Python versions. A
pip-only installation therefore requires a compatible native build toolchain
for Assimulo's Cython, C, Fortran, and SUNDIALS components. For solver-backed
runs, use the pixi environment defined in `pyproject.toml` (`pixi run -e
assimulo ...`), the conda-forge environment in `environment.yml`, or a local
source build of Assimulo.

Unit-operation modules and their solver-free calculations remain importable
without Assimulo. Attempting to construct an Assimulo-backed problem or solver
without that optional dependency raises an installation error at the solver
boundary.

pixi is the recommended developer/CI environment manager: its `[tool.pixi]`
configuration in `pyproject.toml` provides a `default` (core, Assimulo-free),
an `assimulo` (full backend), and a `docs` (documentation toolchain)
environment backed by a committed `pixi.lock`,
which is this repository's locked, reproducible way to obtain the conda-forge
solver stack. `[project]` remains the source of truth for pip users under the
`pharmapy-sim` distribution name; the Python import remains `PharmaPy`. The pixi
dependency bounds mirror the core project metadata and must stay synchronized.

Current bounds:

`pyproject.toml` is the source of truth for install metadata. The
`requirements.txt` and `requirements-assimulo.txt` files are convenience
mirrors for pip-based workflows and should stay synchronized with
`[project.dependencies]` and the pip-installable entries in the `assimulo`
optional extra.

- `numpy>=1.22`: the lower bound gives a modern baseline for supported Python
  versions while allowing the NumPy 2 line to be exercised by CI and Dependabot.
- `scipy>=1.9`: the previous SciPy upper cap was tied to
  `scipy.integrate.simps`, which was removed in SciPy 1.14. PharmaPy now uses
  `scipy.integrate.simpson`, so current SciPy releases can be tested directly.
- `matplotlib>=3.5` and `pandas>=1.5`: lower bounds avoid unbounded old
  releases while leaving current releases open for Dependabot and CI.
- `cython>=0.29,<3`: kept in the `assimulo` optional extra as a pip-installable
  build helper for source-install workflows. The extra does not install
  Assimulo itself.
- `assimulo==3.4.3`: the supported Assimulo version for the conda-forge
  environment and source-build workflows. It is documented here instead of
  pinned in pip metadata because that release is not available from PyPI.
  Dependabot intentionally ignores Assimulo updates and Cython major-version
  updates until a broader solver stack is verified.

## Documentation dependencies

The Sphinx toolchain lives in the `docs` optional extra
(`[project.optional-dependencies].docs`) and is mirrored by the pixi `docs`
feature and environment. `pixi run docs` reproduces the same warning-as-error
build that the Sphinx docs CI job and Read the Docs run, from the committed
lock. The pip path (`pip install ".[docs]"`) is what CI and Read the Docs
themselves use, so both surfaces must stay synchronized like the core ones.
The Linux leg of the locked pixi CI job runs `pixi run docs` itself, so a
divergence between the locked task and the pip-based build fails a check
rather than waiting for a contributor to hit it.

Upper caps exclude the next major of a rendering-relevant package until the
built site has been checked against it; lower bounds keep a modern baseline
rather than encoding a specific known-bad release. The exact versions that make
a build reproducible come from `pixi.lock`, not from these ranges.

- `sphinx>=7.2,<9`: the site builder. The cap excludes the unreleased Sphinx 9
  major. Note that the effective floor is set by the extensions rather than by
  this bound: `sphinxcontrib-bibtex` 2.7 requires `Sphinx>=7.4`, and
  `nbsphinx` 0.9.8 excludes the broken Sphinx 8.2.0 and 8.2.1 point releases.
- `sphinx-rtd-theme>=2,<4`: the published theme. Theme majors change rendered
  output and navigation, so the cap keeps an unvalidated major from silently
  changing the site.
- `nbsphinx>=0.9`: renders the example notebooks. Requires the `pandoc` binary,
  which pip metadata cannot express; CI and Read the Docs install it from apt
  and the pixi `docs` feature resolves it from conda-forge.
- `sphinxcontrib-bibtex>=2.6`: renders `references.bib`. 2.6 is the first line
  supporting the Sphinx 7 series that this extra's lower bound admits.
- `ipykernel>=6`: gives contributors a kernel in the locked docs environment
  when regenerating committed notebook outputs. The documentation build itself
  consumes those outputs and does not execute notebooks.

Documentation builds require Python 3.10 or newer, above the package's own
`requires-python = ">=3.9"`, because `doc/online_docs/conf.py` resolves the
installed distribution with `importlib.metadata.packages_distributions`, added
in 3.10. CI, Read the Docs, and the pixi `docs` environment all pin 3.11.
