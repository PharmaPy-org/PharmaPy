"""Accepted-point reporting for #252 using the real estimator and native LM.

Synthetic squared gains force a rejected terminal trial without replacing the
solver. Weighted, unequal experiments exercise ordering and fixed parameters;
StatisticsClass exercises the consumer of the reported trajectories.
"""

import numpy as np
import pytest

from PharmaPy.ParamEstim import ParameterEstimation
from PharmaPy.StatsModule import StatisticsClass


pytestmark = pytest.mark.unit
# Roundoff allowance for the small, exact-algebra fixtures below.
ROUNDING_ATOL = 64 * np.finfo(float).eps  # [-]
# Allow cancellation in the explicit forward-difference quotient below.
DERIVATIVE_RTOL = 32 * np.sqrt(np.finfo(float).eps)  # [-]
FINITE_DIFFERENCE_STEP = 1.0e-4  # [-], resolves curvature at the .1 gain seed


def squared_gain(parameters, inputs):
    """Evaluate a nonlinear verification response.

    Parameters
    ----------
    parameters : ndarray
        One gain [-], shape (1,).
    inputs : ndarray
        Controlled inputs [-], shape (n_observations,).

    Returns
    -------
    ndarray
        Response [-], shape (n_observations,).
    """
    return parameters[0] ** 2 * inputs


def squared_gain_jacobian(parameters, inputs):
    """Differentiate the squared-gain response analytically.

    Parameters
    ----------
    parameters : ndarray
        One gain [-], shape (1,).
    inputs : ndarray
        Controlled inputs [-], shape (n_observations,).

    Returns
    -------
    ndarray
        Response derivative [-], shape (n_observations, 1).
    """
    return (2 * parameters[0] * inputs)[:, np.newaxis]


@pytest.mark.parametrize("jacobian", [None, squared_gain_jacobian])
@pytest.mark.parametrize("store_iter", [False, True])
def test_rejected_terminal_trial_reports_accepted_point(jacobian, store_iter):
    inputs = np.array([1.0, 2.0, 3.0])  # [-], identifiable gain experiment
    observations = np.array([1.0, 2.0, 3.0])  # [-], independent truth at gain 1
    seed = np.array([0.1])  # [-], small derivative causes first-trial overshoot
    estimator = ParameterEstimation(squared_gain, seed, inputs, observations,
                                    jac_fun=jacobian,
                                    dx_finitediff=FINITE_DIFFERENCE_STEP)
    # One trial tests rejection reporting, not convergence to the truth.
    accepted, _, info = estimator.optimize_fn(
        optim_options={"max_fun_eval": 1}, verbose=False, store_iter=store_iter)
    assert info["num_fun_eval"] == 1 and info["num_iter"] == 0
    np.testing.assert_allclose(accepted, seed, rtol=0.0, atol=ROUNDING_ATOL)
    expected_model = np.array([0.01, 0.02, 0.03])  # [-], seed squared * inputs
    expected_residual = np.array([-0.99, -1.98, -2.97])  # [-], prediction - data
    expected_jacobian = np.array([[0.2, 0.4, 0.6]])  # [-], 2 * seed * inputs
    if jacobian is None:
        # For a quadratic the forward quotient is exactly (2 * seed + dx) * x.
        expected_jacobian += FINITE_DIFFERENCE_STEP * inputs  # [-]
    # Native LM already returns these accepted values on the broken base.
    np.testing.assert_allclose(info["x"], seed, rtol=0.0, atol=ROUNDING_ATOL)
    np.testing.assert_allclose(info["fun"], expected_residual,
                               rtol=0.0, atol=ROUNDING_ATOL)
    np.testing.assert_allclose(info["jac"], expected_jacobian,
                               rtol=DERIVATIVE_RTOL, atol=ROUNDING_ATOL)
    np.testing.assert_allclose(estimator.params_residuals, seed,
                               rtol=0.0, atol=ROUNDING_ATOL)
    for reported in (estimator.y_model[0], estimator.y_runs[0]):
        np.testing.assert_allclose(reported.ravel(), expected_model,
                                   rtol=0.0, atol=ROUNDING_ATOL)
    for residual in (estimator.resid_runs[0], estimator.residuals,
                     estimator.weighted_residuals):
        np.testing.assert_allclose(residual.ravel(), expected_residual,
                                   rtol=0.0, atol=ROUNDING_ATOL)
    assert len(estimator.params_iter) == len(estimator.objfun_iter)
    statistics = StatisticsClass(estimator)
    np.testing.assert_allclose(statistics.y_nominal[0].ravel(), expected_model,
                               rtol=0.0, atol=ROUNDING_ATOL)
    np.testing.assert_allclose(statistics.residuals[0].ravel(), expected_residual,
                               rtol=0.0, atol=ROUNDING_ATOL)


def three_state_gain(parameters, inputs):
    """Return distinct states while keeping the first parameter fixed.

    Parameters
    ----------
    parameters : ndarray
        Fixed multiplier and fitted gain [-], shape (2,).
    inputs : ndarray
        Controlled inputs [-], shape (n_observations,).

    Returns
    -------
    ndarray
        Response [-], shape (n_observations, 3). State 2 is twice state 0;
        the middle state is an unmeasured constant reference of unity.
    """
    response = parameters[0] * parameters[1] ** 2 * inputs  # [-]
    return np.column_stack((response, np.ones_like(inputs), 2 * response))


def test_rejected_trial_preserves_weighting_state_order_and_fixed_parameters():
    inputs = [np.array([1.0, 2.0, 3.0]),
              np.array([0.5, 1.5, 2.5, 3.5, 4.5])]  # [-], unequal schedules
    observations = [np.array([[4.0, 2.0], [8.0, 4.0], [12.0, 6.0]]),
                    np.array([[2.0, 1.0], [6.0, 3.0], [10.0, 5.0],
                              [14.0, 7.0], [18.0, 9.0]])]  # [-], states [2, 0]
    seed = np.array([2.0, 0.1])  # [-], fixed multiplier 2; overshooting gain .1
    weights = np.diag([4.0, 9.0])  # [-], standard deviations 2 and 3
    estimator = ParameterEstimation(
        three_state_gain, seed, inputs, observations, measured_ind=[2, 0],
        optimize_flags=[False, True], weight_matrix=weights)
    accepted, _, info = estimator.optimize_fn(
        optim_options={"max_fun_eval": 1}, verbose=False)
    assert info["num_iter"] == 0 and info["num_fun_eval"] == 1
    np.testing.assert_allclose(accepted, [0.1], rtol=0.0, atol=ROUNDING_ATOL)
    # At the retained seed the response is exactly 1/100 of the observations.
    for reported, raw, observation in zip(estimator.y_model,
                                           estimator.resid_runs, observations):
        np.testing.assert_allclose(reported, observation / 100,
                                   rtol=0.0, atol=ROUNDING_ATOL)
        np.testing.assert_allclose(raw, -0.99 * observation,
                                   rtol=0.0, atol=ROUNDING_ATOL)
    # Independent state-major ordering, experiment by experiment; divide each
    # residual by its known standard deviation once (not by its variance).
    expected_weighted = np.array([
        -1.98, -3.96, -5.94, -0.66, -1.32, -1.98,
        -0.99, -2.97, -4.95, -6.93, -8.91,
        -0.33, -0.99, -1.65, -2.31, -2.97])  # [-]
    np.testing.assert_allclose(info["fun"], expected_weighted,
                               rtol=0.0, atol=ROUNDING_ATOL)
    np.testing.assert_allclose(estimator.weighted_residuals, expected_weighted,
                               rtol=0.0, atol=ROUNDING_ATOL)


@pytest.mark.parametrize("store_iter", [False, True])
def test_repeated_fit_replaces_reported_trajectories(store_iter):
    inputs = np.array([1.0, 2.0, 3.0])  # [-]
    observations = np.array([1.0, 2.0, 3.0])  # [-], truth at gain 1
    seed = np.array([0.1])  # [-], nonzero residuals avoid zero-variance covariance
    estimator = ParameterEstimation(squared_gain, seed, inputs, observations,
                                    jac_fun=squared_gain_jacobian)
    # No trials isolates assembly and reuse from the rejected-trial repair.
    estimator.optimize_fn(optim_options={"max_fun_eval": 0}, verbose=False,
                          store_iter=store_iter)
    previous_statistics = StatisticsClass(estimator)
    estimator.param_seed = np.array([0.2])  # [-], distinct second accepted seed
    estimator.optimize_fn(optim_options={"max_fun_eval": 0}, verbose=False,
                          store_iter=store_iter)
    assert len(estimator.y_model) == estimator.num_datasets == 1
    expected_model = np.array([0.04, 0.08, 0.12])  # [-], second seed squared
    np.testing.assert_allclose(estimator.y_model[0].ravel(), expected_model,
                               rtol=0.0, atol=ROUNDING_ATOL)
    # Rebuilding the list must not mutate a previous statistics snapshot.
    previous_model = np.array([0.01, 0.02, 0.03])  # [-], first seed squared
    np.testing.assert_allclose(previous_statistics.y_nominal[0].ravel(),
                               previous_model, rtol=0.0, atol=ROUNDING_ATOL)


def test_converged_fit_reports_the_same_accepted_solution():
    inputs = np.array([1.0, 2.0, 3.0])  # [-]
    # Asymmetric perturbations keep the fitted residual variance nonzero.
    observations = np.array([1.1, 1.9, 3.05])  # [-]
    seed = np.array([0.8])  # [-], near the truth to exercise accepted steps
    estimator = ParameterEstimation(squared_gain, seed, inputs, observations,
                                    jac_fun=squared_gain_jacobian)
    accepted, _, info = estimator.optimize_fn(verbose=False)
    assert info["num_iter"] > 0
    # The normal equation for slope = gain**2 gives (1.1+3.8+9.15)/14.
    expected_gain = np.sqrt(281 / 280)  # [-], positive branch selected by seed
    np.testing.assert_allclose(accepted, [expected_gain],
                               rtol=DERIVATIVE_RTOL, atol=ROUNDING_ATOL)
    reevaluated = squared_gain(accepted, inputs)  # [-]
    np.testing.assert_allclose(estimator.y_model[0].ravel(), reevaluated,
                               rtol=0.0, atol=ROUNDING_ATOL)
    np.testing.assert_allclose(info["fun"], reevaluated - observations,
                               rtol=0.0, atol=ROUNDING_ATOL)
