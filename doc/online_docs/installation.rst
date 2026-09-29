============
Installation
============

The canonical `PharmaPy installation guide`_ contains the current commands for:

* a core installation from source with pip
* editable development and core tests
* the reproducible pixi environments
* the optional Assimulo solver backend
* the manual conda fallback and platform limitations

The planned PyPI distribution name is ``pharmapy-sim``, while the Python import
name remains ``PharmaPy``. No PharmaPy process-simulation release is on PyPI
yet, so install from the source repository for now. Do not run
``pip install pharmapy``; that distribution is an unrelated project.

For the core library from an existing source checkout:

.. code-block:: console

   python -m venv .venv
   python -m pip install .
   python -c "import PharmaPy; print(PharmaPy.__file__)"

For the locked ``linux-64`` and ``win-64`` development and Assimulo
environments:

.. code-block:: console

   pixi install --all --locked
   pixi run test
   pixi run -e assimulo test-assimulo

pixi is a development and solver-environment tool, not a runtime dependency of
the pip distribution. See the canonical guide for environment activation,
Windows commands, installation of pixi itself, and complete verification
instructions.

.. _PharmaPy installation guide: https://github.com/PharmaPy-org/PharmaPy/blob/master/INSTALLATION.md
