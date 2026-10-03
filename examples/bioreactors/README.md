# Native bioreactor forward examples

Each core example supplies `inputs/case.json`, `inputs/mechanism.json`,
`inputs/thermo.json`, a forward `workflow.ipynb`. Running a notebook generates local CSV/PNG/SVG outputs; generated files are not distributed.
The shared path is `build_bioreactor -> native BatchReactor/SemiBatchReactor/ContinuousReactor -> solve_unit`.

See the concise [configuration and provider guide](../../doc/online_docs/bioreactors/USER_GUIDE.md).

| Example | Purpose |
| --- | --- |
| [Generic chemostat](generic_chemostat/README.md) | Synthetic Monod equilibrium, dilution and washout, with interchangeable kinetic routes. |
| [Generic batch](generic_batch/README.md) | Small configurable batch with analytical checks. |
| [Generic fed-batch](generic_fed_batch/README.md) | Feed events, dilution and material accounting. Synthetic reconciliation benchmarks are maintained separately in the test fixtures. |
| [Batch E. coli](batch_ecoli_dfba/README.md) | Reduced-pathway optimization and qualitative diauxic growth reproduction. |
| [Fed-batch E. coli](ecoli_ye_fed_batch/README.md) | Experimental reproduction with continuous feed and sampling. Its [reconciliation assessment](ecoli_ye_fed_batch/reconciliation/README.md) checks the frozen model through metabolic reconciliation. |

Run the notebook cells in order: select folders, load/edit inputs, construct and
solve, extract trajectories, plot, export. Change the input values using the
[shared templates](../../doc/online_docs/bioreactors/templates/README.md).
Core forward notebooks need no runner or verification scripts. They do not fit biological parameters or
select experiments. Their exports overwrite the configured output paths.

`case.operation.volume_policy` selects native mass/density volume or the declared
working-volume basis. Both execute through native reactors. Working-volume
supports declared continuous inlets, additions and well-mixed samples; continuous
well-mixed outlets are supported in continuous mode. Perfusion and unmanaged connections remain unsupported. Configure rate
units and biological conversion properties explicitly; the converter cannot infer
missing cell mass, feed composition or biological coefficients.

## Installation and execution

From the repository root, install the package and test extra in your environment:

```sh
python -m pip install -e '.[test]'
python -m pytest tests/ -m "not assimulo"
```

Runtime dependencies are declared in `pyproject.toml`. The provisioned qualification
environment uses Python 3.11; this is not a clean-installation qualification.

For interactive execution, open an example's `workflow.ipynb` in a notebook
editor and select this environment as its Python kernel. If you do not already
have an editor/kernel integration, install the optional notebook tools:

```sh
python -m pip install jupyterlab ipykernel
python -m jupyter lab examples/bioreactors
```

Run the notebook cells in order. Notebook tools are optional, unpinned interface
dependencies, not required by the simulator or test suite. The supplied
SciPy and fixed-step demonstrations do not require Assimulo; using the optional
IDA backend requires a compatible Assimulo installation separately.

Retired CHO, hybridoma and duplicate demonstration folders, old campaign outputs,
and installation/regression-refresh helpers have been removed. Minimal historical
inputs and captured numerical failure states remain under `tests/Bioreactor/fixtures`
only where active tests need them. Synthetic benchmark inputs and mathematical references are maintained in
[reconciliation test fixtures](../../tests/Bioreactor/fixtures/reconciliation/README.md). The [validation ladder](../../PharmaPy/Bioreactors/TECHNICAL_SUMMARY.md#qualification-and-limits)
records the qualification scope; numerical correctness does not establish general
biological predictivity. Published figures and historical generated evidence are excluded; source citations remain.

## Empty optional input slots

The core examples are isothermal liquid-phase simulations. Their
`thermo.json` files retain molecular weights (`mw`) and liquid densities
(`rho_liq`), which native construction and balances use. Unused property arrays
(including `cp_liq` and `p_vap`) are `[]`; unused scalar/text properties are `null`.
Every property key is retained. Empty arrays satisfy phase/stream construction
without supplying invented heat-capacity or vapor-pressure coefficients.

These blanks mean **not supplied**, not zero. Repopulate the corresponding
properties before requesting energy balances, vapor-liquid calculations, or other
property calculations that need them. No native thermodynamic functionality was
removed. The supplied forward trajectories are unchanged by this input cleanup.
Populated case/model values, intentional zero rates, required mappings and
construction/validation fields remain; unused optional model sections already
use empty objects or `null`.
