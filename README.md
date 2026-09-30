# PharmaPy

PharmaPy is a pythonic library for the analysis of pharmaceutical manufacturing systems.\

It allows to simulate the dynamics of standalone, drug substance unit operations in a variety of operating modes (batch, continuous, semibatch). Also, PharmaPy facilitates setting up and simulating pharmaceutical **flowsheets**, i.e., interconnected unit operations in a variety of operation modes, ranging from end-to-end batch, end-to-end continuous, and hybrid operation (combination of batch and/or continuous and semicontinuous unit operations).

## Getting started
Read our [link to documentation page] for more information on how to install and use PharmaPy.

To install PharmaPy, download and unzip the code from this page, and then follow the instructions on the `install_instructions.txt` file.

## Bioreactor examples

The [bioreactor guide](examples/bioreactors/README.md) describes native batch and
fed-batch simulations, JSON inputs, executable notebooks, and verification.
Start with the [generic batch](examples/bioreactors/generic_batch/workflow.ipynb)
or [generic fed-batch](examples/bioreactors/generic_fed_batch/workflow.ipynb)
notebook for a small forward-only example with analytical and mass-balance checks.
The E. coli examples add metabolic modeling; the synthetic mammalian example demonstrates cells, product and surrogate substitution. Start with the [configuration guide](doc/online_docs/bioreactors/USER_GUIDE.md). These demonstrations do not establish independent experimental accuracy.

For the isolated bioreactor verification environment, use Python 3.11 and install the pinned requirements linked in that guide;
its optional-solver scope differs
from the legacy installation instructions above.

<!-- BEGIN Status badges -->
[![Downloads]]
<!-- END Status badges -->
