"""Bounded CVode/IPOPT reactor fitting and accepted bootstrap state (#397).

Requires both real optional backends. The first-order reaction's analytic
solution supplies the GLS optimum, residuals and sensitivity information.
MCR optimization remains a separate, unresolved contract under #240.
"""

import copy

import numpy as np
import pytest
from scipy.optimize import minimize_scalar

pytestmark = [pytest.mark.assimulo, pytest.mark.integration]
pytest.importorskip('cyipopt')
pytest.importorskip('assimulo')

from PharmaPy.StatsModule import StatisticsClass
import test_simexec_gls_weighting as reactor_case
import test_simexec_bootstrap_state as state_checks


# Synthetic rate interval containing the five-second reaction time constant.
# The upper-bound case deliberately excludes the unconstrained fitted rate,
# which is about 0.207 /s, to verify that the public handoff enforces bounds.
RATE_BOUNDS = (0.01, 1.0)  # [1/s]
ACTIVE_UPPER_BOUND = 0.18  # [1/s], below the known unconstrained optimum
IPOPT_OPTIONS = {'tol': 1e-7, 'max_iter': 100}  # [-], [-], small-fit budget
REFERENCE_XTOL = 1e-12  # [1/s], tighter than the CVode/finite-difference error
# PharmaPy uses a relative forward-difference step of 1e-3. Near k=0.2 /s,
# this 1e-4 /s fit budget covers derivative truncation and integration noise,
# while remaining below 0.1% of the rate.
RATE_ATOL = 1e-4  # [1/s]
# Weighted squared errors are O(0.1). CVode's 1e-9 relative tolerance and
# 1e-11 mol/L absolute tolerance imply much less than 1e-6 objective error.
OBJECTIVE_RTOL = 1e-5  # [-]
# The analytic sensitivities differ from forward differences by at most
# (step * maximum time)/2, about 7e-4 here; their squared sum has twice that
# relative error. 2e-3 covers it and the smaller integration error.
INFORMATION_RTOL = 2e-3  # [-]
# IPOPT relaxes bounds by 1e-8 by default. This budget covers that convention
# and float64 roundoff without accepting an unconstrained fitted rate.
BOUND_ATOL = 2e-8  # [1/s]


def _analytic_objective(rate: float, observations: np.ndarray) -> float:
    """Evaluate the analytic reaction's dimensionless GLS loss.

    Parameters
    ----------
    rate : float
        First-order rate constant [1/s].
    observations : numpy.ndarray
        Concentrations [mol/L], shape ``(5, 2)``, columns A and B.

    Returns
    -------
    float
        Half the precision-weighted squared concentration error [-].
    """
    residual = reactor_case._analytic_profiles(rate) - observations  # [mol/L]
    return 0.5 * np.einsum('ki,ij,kj->', residual, reactor_case.PRECISION,
                           residual)


@pytest.mark.parametrize('upper_bound', [RATE_BOUNDS[1], ACTIVE_UPPER_BOUND])
def test_simexec_ipopt_fit_and_bootstrap_preserve_accepted_state(upper_bound):
    """Check bounds, then statistics and bootstrap for the interior fit.

    Parameters
    ----------
    upper_bound : float
        Upper rate bound [1/s]; either encloses or excludes the free optimum.
    """
    simulation = reactor_case._batch_simulation()
    observations = (reactor_case._analytic_profiles(reactor_case.TRUE_RATE)
                    + reactor_case.MEASUREMENT_ERRORS_MOL_L)  # [mol/L]
    simulation.SetParamEstimation(
        reactor_case.TIME_S, observations, measured_ind=[0, 1],
        optimize_flags=[True, False], weight_matrix=reactor_case.COVARIANCE,
        wrapper_kwargs={'sundials_opts': {'rtol': reactor_case.SOLVER_RTOL,
                                          'atol': reactor_case.SOLVER_ATOL}})
    bounds = (RATE_BOUNDS[0], upper_bound)  # [1/s]
    reference = minimize_scalar(
        _analytic_objective, args=(observations,), bounds=bounds,
        method='bounded', options={'xatol': REFERENCE_XTOL})
    assert reference.success

    # [1/s], [1/s**2], and dimensionless residual / [s] Jacobian outputs.
    accepted, covariance, info = simulation.EstimateParams(
        method='IPOPT', bounds=[bounds], verbose=False,
        optim_options=dict(IPOPT_OPTIONS))

    assert bounds[0] - BOUND_ATOL <= accepted[0] <= bounds[1] + BOUND_ATOL
    assert accepted[0] == pytest.approx(reference.x, rel=0.0, abs=RATE_ATOL)
    assert 0.5 * info['fun'] @ info['fun'] == pytest.approx(
        _analytic_objective(accepted[0], observations), rel=OBJECTIVE_RTOL)
    if upper_bound == ACTIVE_UPPER_BOUND:
        # Bound-aware covariance and retention of bounds in bootstrap refits
        # are not established by this fixture. This case verifies the solve's
        # bound handoff only; the interior case below verifies fit statistics.
        return

    reactant = reactor_case._analytic_profiles(accepted[0])[:, 0]  # [mol/L]
    derivative = reactor_case.TIME_S * reactant  # [mol*s/L]
    sensitivities = np.column_stack((-derivative, derivative))  # [mol*s/L]
    information = np.einsum('ki,ij,kj->', sensitivities,
                             reactor_case.PRECISION, sensitivities)  # [s**2]
    np.testing.assert_allclose(info['jac'] @ info['jac'].T, [[information]],
                               rtol=INFORMATION_RTOL, atol=0.0)
    dof = observations.size - accepted.size  # [-], observations minus parameters
    expected_covariance = (2 * _analytic_objective(accepted[0], observations)
                           / dof / information)  # [1/s**2]
    np.testing.assert_allclose(covariance, [[expected_covariance]],
                               rtol=INFORMATION_RTOL, atol=0.0)

    estimator = simulation.ParamInst
    statistics = StatisticsClass(estimator)
    before = {name: copy.deepcopy(getattr(estimator, name))
              for name in state_checks.FITTED_ATTRIBUTES}
    previous_random_state = np.random.get_state()
    try:
        np.random.seed(state_checks.BOOTSTRAP_SEED)
        fitted = statistics.bootstrap_params(state_checks.NUM_SAMPLES)  # [1/s]
    finally:
        np.random.set_state(previous_random_state)

    assert fitted.shape == (state_checks.NUM_SAMPLES, 1)
    assert np.all(np.isfinite(fitted))
    assert np.all(np.abs(fitted[:, 0] - accepted[0]) > RATE_ATOL)
    for name, value in before.items():
        state_checks._assert_same(value, getattr(estimator, name), name)
    assert simulation.R01.Kinetics.concat_params()[0] == pytest.approx(
        accepted[0], rel=0.0, abs=REFERENCE_XTOL)
