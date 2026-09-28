============
Installation
============

There are two ways to install PharmaPy. The first is for users who want to run the software without editing it. The second is for developers or advanced users who want to create their own models and add them to PharmaPy.

Standard Installation
=====================

We recommend installing PharmaPy and its dependencies in a Python or conda virtual environment. PharmaPy is published on PyPI as ``pharmapy-org``, and you import it as ``PharmaPy``:

.. code-block:: bash

   pip install pharmapy-org

This installs PharmaPy with scipy as its only integrator. Unit operations built on ``MultiPhaseVessel`` (the ``*_Refactored`` modules) use ``ScipyBackend`` by default. Two other solver backends are optional:

* **Assimulo** (SUNDIALS CVode/IDA) is used by ``AssimuloBackend`` and ``AssimuloDAEBackend``, and by the legacy unit operations that have not been refactored yet. Install it from conda-forge, because the PyPI package is outdated and has no wheels: ``conda install -c conda-forge assimulo``.
* **Julia** (DifferentialEquations.jl) is used by ``DiffeqpyBackend``: run ``pip install "pharmapy-org[julia]"`` and then ``python -c "import diffeqpy; diffeqpy.install()"``.

To edit and run code, we recommend also installing an IDE or JupyterLab (``pip install jupyterlab``).

Developer Installation
======================

For development, we recommend using conda environments to manage PharmaPy and its dependencies. `miniconda`_, a lightweight version of conda, is a good option for new users.

Download the source code from our `Github repository`_ and navigate (:code:`cd`) to the directory that contains :code:`pyproject.toml`. Then follow the instructions in :code:`install_instructions.txt`. They set up a fresh conda environment that includes Assimulo and install PharmaPy in editable mode:

.. code-block:: bash

   conda install --file requirements.txt -c conda-forge
   python -m pip install -e .

.. _Github repository: https://github.com/CryPTSys/PharmaPy/tree/develop
.. _miniconda: https://github.com/CryPTSys/PharmaPy/

Once the software is installed, install and/or use your preferred IDE or text editor to construct PharmaPy flowsheets. For instance, on an active conda environment, install the `Spyder IDE`_ by doing :code:`conda -c conda-forge install spyder`, which provides a nice development environment very well suited for scientific computing. 

.. _Spyder IDE: https://github.com/spyder-ide/spyder

Tutorials in the format of Jupyter notebooks are available for users and developers getting started with PharmaPy. Also, on this site, documentation for all unit operations is available.
