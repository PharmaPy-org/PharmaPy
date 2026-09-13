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
collects tests. This keeps their missing-dependency fallbacks observable. When
the same command is run in a richer local environment, backend-absence tests
may skip while the remaining core tests continue to run; use the explicit
optional-backend environments for installed-backend coverage.

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
