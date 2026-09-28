# PharmaPy

PharmaPy is a pythonic library for the analysis of pharmaceutical manufacturing systems.\

It allows to simulate the dynamics of standalone, drug substance unit operations in a variety of operating modes (batch, continuous, semibatch). Also, PharmaPy facilitates setting up and simulating pharmaceutical **flowsheets**, i.e., interconnected unit operations in a variety of operation modes, ranging from end-to-end batch, end-to-end continuous, and hybrid operation (combination of batch and/or continuous and semicontinuous unit operations).

## Getting started
Read our [link to documentation page] for more information on how to install and use PharmaPy.

### Installation
PharmaPy is published on PyPI as `pharmapy-org`, and you import it as `PharmaPy`:

```
pip install pharmapy-org
```

```python
import PharmaPy
```

The only integrator this installs is scipy's. Unit operations built on `MultiPhaseVessel` (the `*_Refactored` modules) default to `ScipyBackend`. Two other solver backends are optional:

- **Assimulo (SUNDIALS CVode/IDA)** is used by `AssimuloBackend` and `AssimuloDAEBackend`. It is also needed by the legacy unit operations that have not been refactored yet: `Reactors`, `Crystallizers`, `Evaporators`, `Distillation`, `SolidLiquidSep`, `Containers`, `Drying_Model`, `DynamicExtraction` and `ThreePhaseSettler`. Install it from conda-forge, because the PyPI `assimulo` package is outdated and has no wheels:

  ```
  conda install -c conda-forge assimulo
  ```

- **Julia (DifferentialEquations.jl)** is used by `DiffeqpyBackend` and needs Python 3.10 or newer:

  ```
  pip install "pharmapy-org[julia]"
  python -c "import diffeqpy; diffeqpy.install()"
  ```

To get a development install with every backend, download the source code and follow `install_instructions.txt`. It sets up a conda environment that includes Assimulo and installs PharmaPy in editable mode.

<!-- BEGIN Status badges -->
[![Downloads]]
<!-- END Status badges -->
