# Level 6: coupled synthetic network regression inputs

Four standard input triples (`coupled04`, `coupled10`, `coupled20`, and
`coupled20_scaled`) exercise progressively larger coupled reconciliation systems.
The forward notebook uses the shared native builder. Edit its input/output paths
to choose a configuration; executing it regenerates trajectory CSV/PNG/SVG outputs.

The historical full campaign passed its numerical gates. Its large intermediate
output trees, report generators and artifact-only acceptance tests were removed.
This directory is now retained for repeatable equation-level regression checks,
not as a packaged copy of the complete historical qualification campaign.

Run `tests/Bioreactor/test_bioreactor_qualification.py` and
`tests/Bioreactor/test_bioreactor_reconciliation_regressions.py` with pytest.
`tests/Bioreactor/reconciliation_reference.py` supplies independent reference mathematics;
`tests/Bioreactor/fixtures/reconciliation/closure_states.json` retains only the
specific numerical states replayed by the tests. These exercise flux precision,
original-problem feasibility, recovery and finite-exposure optimality.

See the validation ladder for the distinction between numerical verification
and biological prediction. These synthetic configurations establish neither
CHO coefficient validity nor independent experimental agreement.
