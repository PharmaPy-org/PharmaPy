# PharmaPy

<img align="left" src="./doc/online_docs/images/PharmaPy_logo.jpeg" alt="PharmaPy_logo" height="250">

<!-- BEGIN Status badges -->
[![CI](https://github.com/PharmaPy-org/PharmaPy/actions/workflows/ci.yml/badge.svg)](https://github.com/PharmaPy-org/PharmaPy/actions/workflows/ci.yml)
[![Historical CryPTSys release downloads](https://img.shields.io/github/downloads/CryPTSys/PharmaPy/total)](https://github.com/CryPTSys/PharmaPy/releases)
[![DOI](https://img.shields.io/badge/DOI-10.1016%2Fj.compchemeng.2021.107408-blue)](https://www.sciencedirect.com/science/article/abs/pii/S0098135421001861)
<!-- END Status badges -->

PharmaPy is a pythonic library for the analysis of pharmaceutical manufacturing systems.

It allows to simulate the dynamics of standalone, drug substance unit operations in a variety of operating modes (batch, continuous, semibatch). Also, PharmaPy facilitates setting up and simulating pharmaceutical **flowsheets**, i.e., interconnected unit operations running in one or more operation modes, offering flexibility to simulate end-to-end batch, end-to-end continuous, and hybrid operation schemes (combination of batch and/or continuous and semicontinuous unit operations).

<br clear="left"/>

## Installation

Install the core library from a source checkout with:

```bash
python -m pip install .
```

Use the locked pixi environment when you need the optional Assimulo solver
backend. The planned PyPI distribution name is `pharmapy-sim`; the import name
will remain `PharmaPy`.

See the [complete installation guide](https://github.com/PharmaPy-org/PharmaPy/blob/master/INSTALLATION.md)
for virtual-environment setup, editable development, pixi, Assimulo, platform
support, and verification commands.

See the [contributing guide](https://github.com/PharmaPy-org/PharmaPy/blob/master/CONTRIBUTING.md)
for the development workflow and integration contract.

Read our [documentation](https://pharmapy.readthedocs.io/en/latest/) or chat with the [PharmaPy Simulation Assistant](https://chatgpt.com/g/g-679bb3b5c5188191b26680b147a4f4a2-pharmapy-simulation-assistant) for more information on how to install and how to use PharmaPy.
