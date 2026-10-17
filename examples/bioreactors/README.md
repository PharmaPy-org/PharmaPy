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
