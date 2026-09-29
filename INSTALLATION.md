# Installing PharmaPy

PharmaPy has a pip-installable core and an optional Assimulo solver backend.
Choose the installation path that matches the work you need to do:

- Use **pip** for the core library and solver-free development.
- Use **pixi** for a reproducible development environment or any work that
  requires Assimulo.

pixi is a development and solver-environment tool. It is not a PharmaPy runtime
dependency and does not replace the package metadata used by pip.

The installation commands intentionally produce different environments:

| Command | Assimulo included? | Intended use |
| --- | --- | --- |
| `python -m pip install .` | No | Core library for users |
| `pixi install --locked` | No | Reproducible core development |
| `pixi install -e assimulo --locked` | Yes | Solver-backed development and use |
| `pixi install --all --locked` | Yes, in the separate `assimulo` environment | Maintainer verification of both environments |

## Package and import names

The planned PyPI distribution name is **`pharmapy-sim`**. The Python import name
remains **`PharmaPy`**:

```python
import PharmaPy
```

There is not yet a PharmaPy process-simulation release on PyPI. Until one is
published, install this project from a source checkout as described below. Do
not run `pip install pharmapy`: that name belongs to an unrelated project.
The `pharmapy-sim` name remains unreserved until the first release is uploaded;
maintainers must recheck its availability immediately before publishing and
must not advertise the index-install command until that upload succeeds.
Release ownership is tracked in
[issue #146](https://github.com/PharmaPy-org/PharmaPy/issues/146).

## Upgrading from an earlier PharmaPy install

Earlier source checkouts installed a distribution named `PharmaPy`. pip treats
that name and the new `pharmapy-sim` distribution name as different projects,
so both installation records can coexist even though they own the same
`PharmaPy` import files.

If an environment has ever installed PharmaPy from an earlier checkout,
uninstall both possible distribution records before installing this tree:

```bash
python -m pip uninstall -y PharmaPy pharmapy-sim
python -m pip install .
```

Run the commands with the same Python interpreter used for PharmaPy. This
cleanup is required only when reusing an existing environment; a new virtual
environment does not contain the legacy distribution.

## Core installation with pip

The core installation supports Python 3.9 or newer and does not install
Assimulo.

Clone the repository and enter it:

```bash
git clone https://github.com/PharmaPy-org/PharmaPy.git
cd PharmaPy
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Linux or macOS:

```bash
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install PharmaPy:

```bash
python -m pip install --upgrade pip
python -m pip install .
```

Verify the installation:

```bash
python -c "import PharmaPy; print(PharmaPy.__file__)"
```

For editable development and the core test dependencies, use:

```bash
python -m pip uninstall -y PharmaPy pharmapy-sim  # when reusing an environment
python -m pip install -e ".[test]"
python -m pytest tests/ -m "not assimulo"
```

After the first `pharmapy-sim` release is published, the core installation
command will be:

```bash
python -m pip install pharmapy-sim
```

The `assimulo` optional extra contains only the Cython helper used by source
build workflows; it cannot install the current supported Assimulo release from
PyPI.

## Why pixi is used

pip remains the primary installation mechanism for the PharmaPy package. pixi
is used for the development and solver environment because Assimulo wraps
compiled SUNDIALS, Cython, C, and Fortran components. The
[Assimulo release on PyPI](https://pypi.org/project/Assimulo/) does not provide
a compatible wheel for PharmaPy's supported Python versions, whereas
[conda-forge provides compiled Assimulo packages](https://anaconda.org/conda-forge/assimulo).

pixi can resolve those conda-forge binaries, install the local PharmaPy package
editably, and record the complete result in `pixi.lock`. Keeping pixi
configuration under `[tool.pixi]` means that it does not become a dependency of
the PharmaPy wheel and does not change how the package is built or published to
PyPI.

Plain `environment.yml` remains a simpler fallback, but it does not pin the
complete transitive environment. The lockfile is the reason to retain pixi; if
the project stops testing and maintaining that lockfile, pixi would add
maintenance cost without providing its intended reproducibility benefit.

## Reproducible installation with pixi

The committed `pixi.lock` supports `linux-64` and `win-64`. Install pixi using
its
[official installation instructions](https://pixi.prefix.dev/latest/installation/).
The native Windows installer used to verify this guide can be run from
PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -c "irm -useb https://pixi.sh/install.ps1 | iex"
```

Open a new terminal after installation so that `pixi` is available on `PATH`.
Then clone and enter the repository:

```bash
git clone https://github.com/PharmaPy-org/PharmaPy.git
cd PharmaPy
```

The available environments are:

- `default`: core dependencies without Assimulo
- `assimulo`: the full solver stack with Assimulo 3.4.3 and SUNDIALS

For work that requires Assimulo, install its locked environment explicitly:

```bash
pixi install -e assimulo --locked
```

This command installs both PharmaPy and Assimulo in the named environment.
Verify that both imports work:

```bash
pixi run -e assimulo python -c "import PharmaPy, assimulo"
```

For core-only development, install the default environment:

```bash
pixi install --locked
```

Maintainers can install both environments together:

```bash
pixi install --all --locked
```

Run the matching test lanes:

```bash
pixi run test
pixi run -e assimulo test-assimulo
```

Run Python or a script in either environment:

```bash
pixi run python -c "import PharmaPy"
pixi run -e assimulo python -c "import PharmaPy, assimulo"
pixi run -e assimulo python path/to/script.py
```

You can also start an activated shell with `pixi shell` or
`pixi shell -e assimulo`.

macOS is not currently present in the committed lockfile. Maintainers extending
platform support must add the platform under `[tool.pixi.workspace].platforms`,
regenerate `pixi.lock`, and verify both test lanes on that platform.

## What is tested

The following installation paths were run successfully on `linux-64` when this
guide was introduced:

- A fresh Python 3.11 virtual environment ran `python -m pip install .`,
  `python -m pip check`, and `import PharmaPy`. Wheel and source-distribution
  builds also passed.
- `pixi install --all --locked` installed both locked environments.
- The core pixi test lane passed.
- The Assimulo pixi test lane passed and imported both `PharmaPy` and Assimulo
  3.4.3.

The native `win-64` lock is generated with pixi 0.74.0 and Python 3.11. The
gating CI job verifies that both environments install and import on Windows.
During the initial native Windows validation:

- `pixi install --all --locked` installed the core and Assimulo environments.
- The core lane passed 56 tests, with 4 skipped and 6 deselected.
- The Assimulo lane imported Assimulo 3.4.3 and passed all 14 selected
  integration tests.
- The lock contains no non-local PyPI artifacts; SciPy remains owned by its
  conda-forge package in both environments.

Continuous integration exercises both installation promises:

- The core job installs PharmaPy editably with pip and runs the solver-free
  tests.
- The gating locked-environment job runs on Linux and Windows, installs both
  pixi environments from the committed lockfile, reruns the documented
  `pixi install --all --locked` command, and verifies the core and Assimulo
  imports.
- A separate advisory job runs the solver-backed tests with
  `continue-on-error: true` and no dependency on the two-platform lock matrix,
  so it reports Linux solver health even if a locked-install job fails.

Maintainers changing dependencies should also run the same checks locally:

```bash
pixi lock --check
pixi install --all --locked
pixi run test
pixi run -e assimulo test-assimulo
```

Keeping the pip-based core CI lane alongside a gating locked-pixi job proves
both installation promises: PharmaPy remains pip-installable, and the compiled
solver environment remains reproducible. The advisory solver-test job reports
integration health separately.

## Manual conda fallback for Assimulo

`environment.yml` remains available as an unlocked conda-forge fallback:

```bash
conda env create -f environment.yml
conda activate pharmapy-assimulo
python -m pip uninstall -y PharmaPy pharmapy-sim  # when reusing an environment
python -m pip install -e . --no-deps
python -m pytest tests/ -m assimulo
```

Prefer pixi when possible because its committed lockfile records the complete
resolved environment.
