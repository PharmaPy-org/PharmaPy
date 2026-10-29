"""Capturing and rerunning an IPOPT fit in a calibration record (#351).

Scope: ``CalibrationRecord.from_estimation`` for ``method='IPOPT'``, which
requires explicit ``bounds`` because ``optimize_fn`` does not store them,
records the options that cyipopt leaves in the caller's dict (the
caller's plus those cyipopt writes, bytes entries decoded, ``print_level``
injected by ``optimize_fn`` omitted; issue #476) and no
``default_options`` (Ipopt's native defaults are not recorded; #483),
and records bounds that ``optimizer_inputs``
passes back to IPOPT on a rerun, including a seed outside its bounds; a
simulated (``simulate=True``) estimator is rejected. Fixture: the user
guide's synthetic two-experiment decay
(see ``test_records_calibration.py``). Requires the optional cyipopt
backend (manual environment in TESTING.md); skipped at collection without
it, like ``test_paramestim_cyipopt.py``.
"""

import numpy as np
import pytest

from PharmaPy.ParamEstim import ParameterEstimation
import test_records_calibration as case

pytestmark = pytest.mark.integration
# ParamEstim imports cyipopt optionally; only the solve below needs it.
pytest.importorskip('cyipopt')

# Synthetic admissible range of k containing the documented optimum
# (about 0.501 1/s) in its interior.
RATE_BOUNDS = (0.01, 5.0)  # [1/s]
# A range that excludes the seed 0.3 1/s but contains the optimum: IPOPT
# starts from the infeasible seed and moves it into the bounds.
ABOVE_SEED_BOUNDS = (0.4, 5.0)  # [1/s]
# IPOPT's convergence tolerance tol is a scaled NLP error [-]; 1e-10 is far
# below the 1e-8 default so the one-parameter optimum is resolved well
# beyond the replay check. max_iter [-] is a generous ceiling: this smooth
# one-parameter fit converges in about ten iterations.
IPOPT_OPTIONS = {'tol': 1e-10, 'max_iter': 100}  # [-], [-]
# The replay tolerance case.REPLAY_RTOL (four ulps) also applies to IPOPT:
# the rerun passes bit-identical data, seeds, bounds and the effective
# options to the same deterministic solver, so iterates are reproduced.


def test_ipopt_capture_requires_bounds_and_reruns():
    experiments = case.make_experiments()
    estimator = ParameterEstimation(case.decay, [case.SEED_RATE],
                                    experiments, name_params=['k'])
    estimator.optimize_fn(optim_options=dict(IPOPT_OPTIONS), method='IPOPT',
                          bounds=[RATE_BOUNDS], verbose=False)
    with pytest.raises(ValueError, match="bounds is required for IPOPT"):
        case.capture(estimator, experiments)
    record = case.capture(estimator, experiments, bounds=[RATE_BOUNDS])

    assert record.solver.method == 'IPOPT'
    # cyipopt rewrote the options dict in place with bytes entries (#476);
    # the record holds the decoded caller options plus the entries cyipopt
    # wrote, without print_level.
    assert any(isinstance(key, bytes) for key in estimator.optim_options)
    options = record.solver.options
    assert all(type(key) is str for key in options)
    assert not any(isinstance(value, bytes) for value in options.values())
    assert 'print_level' not in options
    for key, value in IPOPT_OPTIONS.items():
        assert options[key] == value
    assert 'num_iter' not in record.solver.outcome  # IPOPT info lacks it
    # Ipopt's native defaults are not recorded (#483): no default_options,
    # and a rerun passes exactly the recorded options.
    assert record.solver.default_options is None
    assert record.optimizer_inputs()['optim_options'] == options
    assert record.optimizer_inputs()['bounds'] == [RATE_BOUNDS]
    rerun = record.rerun(case.decay, rtol=case.REPLAY_RTOL,
                         atol=case.REPLAY_ATOL)
    np.testing.assert_allclose(rerun.params_convg, estimator.params_convg,
                               rtol=case.REPLAY_RTOL, atol=case.REPLAY_ATOL)


def test_ipopt_infeasible_seed_is_captured_and_reruns():
    experiments = case.make_experiments()
    estimator = ParameterEstimation(case.decay, [case.SEED_RATE],
                                    experiments, name_params=['k'])
    estimator.optimize_fn(optim_options=dict(IPOPT_OPTIONS), method='IPOPT',
                          bounds=[ABOVE_SEED_BOUNDS], verbose=False)
    lower, upper = ABOVE_SEED_BOUNDS  # [1/s]
    assert lower <= estimator.params_convg[0] <= upper
    record = case.capture(estimator, experiments, bounds=[ABOVE_SEED_BOUNDS])
    parameter = record.parameters[0]
    assert parameter.seed < parameter.lower  # stored as an infeasible start
    loaded = case.CalibrationRecord.from_json(record.to_json())
    rerun = loaded.rerun(case.decay, rtol=case.REPLAY_RTOL,
                         atol=case.REPLAY_ATOL)
    np.testing.assert_allclose(rerun.params_convg, estimator.params_convg,
                               rtol=case.REPLAY_RTOL, atol=case.REPLAY_ATOL)


def test_simulated_ipopt_estimators_are_not_captured():
    experiments = case.make_experiments()
    estimator = ParameterEstimation(case.decay, [case.SEED_RATE],
                                    experiments, name_params=['k'])
    estimator.optimize_fn(optim_options=dict(IPOPT_OPTIONS), method='IPOPT',
                          bounds=[RATE_BOUNDS], simulate=True, verbose=False)
    with pytest.raises(ValueError, match="simulate=True"):
        case.capture(estimator, experiments, bounds=[RATE_BOUNDS])
