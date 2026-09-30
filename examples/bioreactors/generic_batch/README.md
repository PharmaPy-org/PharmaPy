# Generic batch

Batch growth and nutrient consumption with an analytical solution.

Run all six steps of [workflow.ipynb](workflow.ipynb). Edit the values in
`inputs/case.json`, `inputs/mechanism.json` and `inputs/thermo.json` to change
operating conditions, kinetics and properties. The shared `build_bioreactor`
workflow creates native PharmaPy reactors and exports CSV/PNG/SVG trajectories
under `outputs/`. No runner or verification helper is required.

The synthetic inputs demonstrate execution and material accounting, not
experimental validation. Analytical checks are in
`tests/Bioreactor/test_bioreactor_generic_example.py`.
