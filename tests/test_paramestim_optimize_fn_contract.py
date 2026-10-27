"""Result and state contract of ``ParameterEstimation.optimize_fn``.

Scope, by issue:

- The Levenberg-Marquardt ``tol_fun`` stop reached through ``optimize_fn`` and,
  directly, after a rejected trial with an ``arctan`` residual (#474).

Main fixture: a noise-free proportional model ``y = k * t``, time [s], rate
constant ``k`` [mol/L/s] and concentration [mol/L], whose exact least-squares
solution is the generating rate. Fits are cheap LM or SciPy solves in the core
environment.
"""

import inspect

import numpy as np
import pytest

from PharmaPy.LevMarq import levenberg_marquardt
from PharmaPy.ParamEstim import ParameterEstimation


pytestmark = pytest.mark.unit

TIME_S = np.linspace(1.0, 4.0, 4)  # [s]
TRUE_RATE = 0.5  # [mol/L/s], generating rate of the exact data
SEED_RATE = 0.3  # [mol/L/s], deliberately away from the solution
# Default residual threshold of levenberg_marquardt, read from its signature.
LM_TOL_FUN = inspect.signature(
    levenberg_marquardt).parameters['tol_fun'].default  # [mol/L]
# Exact data reach machine-precision residuals before the gradient or step
# could vanish; switching both criteria off leaves tol_fun as the only
# convergence stop (max_fun_eval only bounds the budget).
NO_GRADIENT_OR_STEP_STOP = {'eps_1': 0.0, 'eps_2': 0.0}  # [mol*s/L], [-]
TOL_FUN_REASON = 'Value of the objective function lower than tolerance'


def proportional(params, time_s):
    """Return concentrations of a constant-rate production.

    Parameters
    ----------
    params : array_like
        ``[k]``, rate constant [mol/L/s].
    time_s : numpy.ndarray
        Sample times [s], shape ``(n_times,)``.

    Returns
    -------
    numpy.ndarray
        Concentration ``k * t`` [mol/L], shape ``(n_times,)``.
    """
    return params[0] * time_s


def test_lm_tol_fun_stop_reports_accepted_point():
    data = proportional([TRUE_RATE], TIME_S)  # [mol/L]
    estimator = ParameterEstimation(proportional, [SEED_RATE], TIME_S, data)

    fitted, _, info = estimator.optimize_fn(
        optim_options=dict(NO_GRADIENT_OR_STEP_STOP), verbose=False)

    assert info['stop_criterion'] == TOL_FUN_REASON
    # norm(r) = |k - k_true| * norm(t) for this linear model, so the stop
    # bounds the parameter error by tol_fun / norm(t).
    rate_bound = LM_TOL_FUN / np.linalg.norm(TIME_S)  # [mol/L/s]
    assert abs(fitted[0] - TRUE_RATE) <= rate_bound
    np.testing.assert_array_equal(info['x'], fitted)
    # info['fun'] are the residuals of the returned rate: model minus data.
    # Both sides form the same difference; allow one rounding of the
    # largest concentration.
    residual_atol = np.finfo(float).eps * np.max(np.abs(data))  # [mol/L]
    np.testing.assert_allclose(info['fun'],
                               proportional(fitted, TIME_S) - data,
                               rtol=0.0, atol=residual_atol)
    assert np.linalg.norm(info['fun']) < LM_TOL_FUN
    np.testing.assert_array_equal(estimator.params_convg, fitted)


# Scalar residual r(x) = arctan(x) [-] of a dimensionless x [-]. From
# x0 = 2 its Gauss-Newton step overshoots: Newton's iteration on arctan
# diverges for |x0| above about 1.39, so the first LM trial is rejected.
ARCTAN_SEED = 2.0  # [-]
# levenberg_marquardt's default initial damping, mu = 1e-2 * max(diag(A)).
NIELSEN_DAMPING_FACTOR = 1e-2  # [-]


def arctan_jacobian(x):
    """Return the Jacobian of ``numpy.arctan`` in LM orientation.

    Parameters
    ----------
    x : numpy.ndarray
        Decision variable, shape ``(1,)`` [-].

    Returns
    -------
    numpy.ndarray
        ``1 / (1 + x**2)``, shape ``(1, 1)`` [-].
    """
    return (1 / (1 + np.asarray(x)**2)).reshape(1, -1)


def test_lm_tol_fun_stop_after_rejected_trial_keeps_initial_point():
    # First trial, derived by hand: A = J**2, b = J * r and
    # step = -b / (A + mu) with J = 1 / (1 + x0**2) and r = arctan(x0).
    jac = 1 / (1 + ARCTAN_SEED**2)  # [-]
    residual = np.arctan(ARCTAN_SEED)  # [-]
    damping = NIELSEN_DAMPING_FACTOR * jac**2  # [-]
    trial = ARCTAN_SEED - jac * residual / (jac**2 + damping)  # [-]
    trial_residual = abs(np.arctan(trial))  # [-]
    assert trial_residual > residual  # the trial is rejected
    # A threshold met by the initial point but not by the rejected trial.
    tol_fun = 0.5 * (residual + trial_residual)  # [-]

    x, _, output = levenberg_marquardt(
        [ARCTAN_SEED], np.arctan, arctan_jacobian, eps_1=0.0, eps_2=0.0,
        tol_fun=tol_fun, full_output=True)

    assert output['stop_criterion'] == TOL_FUN_REASON
    assert output['num_iter'] == 0
    assert output['num_fun_eval'] == 1
    np.testing.assert_array_equal(x, [ARCTAN_SEED])
    np.testing.assert_array_equal(output['fun'], [residual])
