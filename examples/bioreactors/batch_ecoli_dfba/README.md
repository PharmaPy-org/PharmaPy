# Native E. coli batch dFBA

[Scientific sources, packaged evidence, and validation limitations](documentation/README.md).

Reproduces the qualitative diauxic-growth behavior reported by Mahadevan et al.
(2002) using a native `LiquidPhase`, arbitrary-pathway `PathwayMetabolism`, and
native `BatchReactor`. All pathway identities, coefficients, constraints, state,
and transport are declared in `inputs/mechanism.json`; source attribution is
given in the linked scientific documentation.

Run Steps 1–6 of [workflow.ipynb](workflow.ipynb) for forward simulation.

## Forward-only notebook

Open [workflow.ipynb](workflow.ipynb) and run all six steps with Run All or
Shift+Enter: select folders, load/edit inputs, simulate, extract trajectories,
plot, and export. Only `case.json`, `mechanism.json`, and `thermo.json` are loaded.
`WRITE_ARTIFACTS = True` replaces `outputs/trajectories.csv`, `trajectories.png`,
and `trajectories.svg`. Change `EXPORT_DIR` to preserve earlier files.
Concentrations use the recorded volume, and feed boundaries retain pre/post rows.

Numerical checks belong to the test suite. Published panels and historical
comparison figures are excluded. Consult the linked original paper for qualitative
comparison; generated trajectories do not establish experimental validation.
