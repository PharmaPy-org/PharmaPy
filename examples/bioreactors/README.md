# Native bioreactor examples

These input-configured examples use native PharmaPy phases, streams, vessels,
mechanisms, and integration. Run each unexecuted workflow notebook from the
repository root; generated outputs are not versioned.

Each example uses `inputs/mechanism.json` for the model, physical `operation`,
and solver `numerical` settings, plus native `inputs/thermo.json` properties.
Construct it with `build_bioreactor(config, thermo_path=thermo_path)`.
The separate case/mechanism three-argument interface remains supported for
existing clients. The combined layout changes packaging, not model equations.

- [Batch](generic_batch/README.md): synthetic reduced-pathway batch example.
- [Fed-batch](generic_fed_batch/README.md): synthetic feed and pathway example.
- [Chemostat](generic_chemostat/README.md): dilution, steady state, and an alternative configured-kinetics route.

These are analytical/synthetic capability demonstrations, not independent
experimental validation. WP3.2 and WP3.3 are outside this implementation.

## Construction and numerical checks

The existing process JSON can select native SciPy methods or optional Assimulo
CVode. Explicit `numerical.inventory_acceptance` values set unit-bearing output
budgets separately from solver tolerances; a permissive output budget never
authorizes consumption from an empty pool.

Optional `operation.density` distinguishes a constant mixture density from
native composition-dependent mixture properties. `operation.conservation`
checks a declared stoichiometric matrix against named invariant coefficients.
The [construction contracts](../../PharmaPy/Bioreactors/construction_checks.py)
document their units, required fields and failure conditions.

`assembly.input_audit()` records construction assumptions. Terminal recipe
operations are retained as a one-point post-event history; reset before replay.
`assembly.balance_report()` requires independently supplied source ledgers and
error budgets. [Convergence trials](../../PharmaPy/Bioreactors/verification.py)
retain every success and failure on an explicitly supplied common reporting
grid. These checks qualify numerical behavior and declared physical balances;
experimental predictive accuracy still requires independent measurements.
