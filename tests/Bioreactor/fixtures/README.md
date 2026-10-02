# Active regression fixtures

`fed_batch_cho/inputs` and `khare_cho_fed_batch/inputs` retain only the three
input files needed by construction, kinetics, unit-conversion, volume and
physical-feasibility tests. Their original demonstration folders were retired;
these fixtures do not assert successful biological reproduction.

`reconciliation/closure_states.json` contains selected historical numerical
failure states, copied without changing their values from the historical coupled-system
reports and trajectories. The regression test independently resolves each
problem, including original constraints and objective checks. Keeping these
small snapshots avoids retaining hundreds of megabytes of campaign output.
