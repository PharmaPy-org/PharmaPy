# Reconciliation verification fixtures

These manufactured inputs verify numerical behavior, not biological prediction.
Their scientific configuration values are unchanged from the original benchmarks.

| Group | Verification purpose |
| --- | --- |
| [Analytic balances](analytic_balances/README.md) | Known fluxes, feed/sample accounting, depletion |
| [Population dynamics](population_dynamics/README.md) | Growth, death and population–flux coupling |
| [Network structure](network_structure/README.md) | Branching, competing nutrients and nonuniqueness |
| [Optimization constraints](optimization_constraints/README.md) | Weighting, bounds and objective policies |
| [Operating transitions](operating_transitions/README.md) | Supply, events, depletion and refinement |
| [Coupled systems](coupled_systems/README.md) | Larger networks, scaling and solver regression |

From the repository root:

```sh
python -m pytest -q -p no:cacheprovider -o pythonpath=. tests/Bioreactor
```

`test_bioreactor_known_answer.py` and `test_bioreactor_qualification.py`
check the inputs against independent mathematics, including references in
`reconciliation_reference.py`. `closure_states.json` retains historical failure
states used by `test_bioreactor_reconciliation_regressions.py`.

The tests do not require saved plots, reports or notebooks. For optional
operating-transition report exports, set `BIOREACTOR_VERIFICATION_OUTPUT` to
an output directory outside the source tree.
