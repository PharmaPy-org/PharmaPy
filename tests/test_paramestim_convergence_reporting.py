"""Uniform convergence reporting of LM and SciPy fits (#352, criterion 5).

Scope: ``ParameterEstimation.optimize_fn`` and ``estimate_parameters``
report ``info['success']`` and ``info['message']`` for PharmaPy's
Levenberg-Marquardt solver and ``scipy.optimize.least_squares`` as IPOPT
fits do (#475), warn with an actionable ``RuntimeWarning`` when the fit
stops without converging (naming the evaluation limit only when it
stopped the fit), and record the effective stopping criteria in
``info['tolerances']``. The returned parameters are those of the solver,
unchanged. The IPOPT entries of ``info['tolerances']`` are checked here
through their solver-independent selection from cyipopt's rewritten
options; real IPOPT fits run in ``test_paramestim_cyipopt``.

Fixture: the synthetic two-experiment first-order decay A -> B of
``test_records_calibration`` (c_A [mol/L] and T [K] with declared
uncertainties, seed k = 0.3 1/s, optimum near 0.501 1/s). Expected
defaults are the values documented by ``levenberg_marquardt`` and
``scipy.optimize.least_squares``, written out by hand. Core lane; each fit
takes milliseconds.
"""

import inspect
import warnings

import numpy as np
import pytest
from scipy.optimize import least_squares

from PharmaPy.LevMarq import levenberg_marquardt
from PharmaPy import ParamEstim
from PharmaPy.ParamEstim import ParameterEstimation
from test_records_calibration import SEED_RATE, decay, make_experiments

pytestmark = pytest.mark.unit

# Documented keyword defaults of levenberg_marquardt (its docstring in
# PharmaPy/LevMarq.py): max_fun_eval [-], eps_1 [residual units**2 /
# parameter units], eps_2 [-], tol_fun [residual units].
LM_DOCUMENTED_DEFAULTS = {'max_fun_eval': 100, 'eps_1': 1e-8, 'eps_2': 1e-8,
                          'tol_fun': 1e-12}
# Documented defaults of scipy.optimize.least_squares: ftol, xtol and gtol
# 1e-8 [-] and max_nfev None, SciPy's automatic evaluation limit.
SCIPY_DOCUMENTED_DEFAULTS = {'ftol': 1e-8, 'xtol': 1e-8, 'gtol': 1e-8,
                             'max_nfev': None}
# A single LM trial: the converged fit of this fixture needs nine, so LM
# stops at the limit after one accepted step.
ONE_EVALUATION = 1  # [-]
# An evaluation-only LM call: residuals and Jacobian at the seed, no trial.
EVALUATION_ONLY = 0  # [-]
# A non-default SciPy tolerance, distinct from the 1e-8 default, that must
# win over the default in info['tolerances'].
TIGHT_FTOL = 1e-10  # [-]
# A non-default LM gradient tolerance [s]: the gradient J @ r of the
# standardized residuals [-] with respect to k [1/s] has units [s].
LOOSE_EPS_1 = 1e-6  # [s]
NOT_CONVERGED = 'not a converged estimate'
# LM gradient tolerance [s] that ends the fit on 'Small gradient': J @ J.T
# of this fixture is 1.8e4 s**2 near the optimum, so a gradient norm of
# 0.1 s still asks for a Gauss-Newton step of about 0.1 / 1.8e4 = 5.6e-6
# 1/s, far above the small-step threshold eps_2 * |k| = 1e-8 * 0.5 1/s;
# the gradient test is therefore met before the step test, and 0.1 s is
# far below the 7.2e3 s gradient norm at the seed.
GRADIENT_EPS_1 = 1e-1  # [s]
# LM residual tolerance [-] above the 34 [-] residual norm of the seed.
# levenberg_marquardt tests the residual norm only after a trial, on the
# accepted point, whose norm cannot exceed the seed's, so the fit stops on
# 'Value of the objective function lower than tolerance' after one trial.
RESIDUAL_TOL_FUN = 1e3  # [-]
# A SciPy budget that leaves the seed but stops short of the optimum: the
# converged fit needs six residual evaluations, and after three the rate is
# about 0.499 1/s, between the seed and CONVERGED_RATE_FLOOR.
PARTIAL_MAX_NFEV = 3  # [-]
# A gtol [-] just above machine epsilon, with ftol and xtol disabled
# (None): least_squares requires at least one tolerance above machine
# epsilon. For this fixture the finite-difference gradient never meets it,
# so only SciPy's automatic evaluation limit stops the fit.
NEAR_EPSILON_GTOL = 2 * np.finfo(float).eps  # [-]
# SciPy's automatic max_nfev for 'trf' is 100 * n for n parameters
# (scipy.optimize.least_squares documentation); this fixture has n = 1.
SCIPY_AUTOMATIC_NFEV_PER_PARAMETER = 100  # [-]
# OptimizeResult.status values of scipy.optimize.least_squares.
SCIPY_BUDGET_STATUS = 0  # [-], max_nfev exceeded
SCIPY_CALLBACK_STATUS = -2  # [-], callback raised StopIteration
# The converged rate lies near 0.501 1/s (test_records_calibration); one LM
# step from the 0.3 1/s seed stops well short of it.
CONVERGED_RATE_FLOOR = 0.5  # [1/s]


def estimator():
    """Return an unfitted estimator of the decay rate.

    Returns
    -------
    ParameterEstimation
        Estimator of k [1/s] seeded at ``SEED_RATE``.
    """
    return ParameterEstimation(decay, [SEED_RATE], make_experiments(),
                               name_params=['k'])


def fit(call, **kwargs):
    """Run a fit and record every warning it emits.

    Parameters
    ----------
    call : str
        ``'optimize_fn'`` or ``'estimate_parameters'``.
    **kwargs
        Keyword arguments of the call; ``verbose=False`` is added.

    Returns
    -------
    tuple
        ``(params, info, caught)``: the returned parameters [1/s], shape
        ``(1,)``, the returned ``info`` and the list of
        ``warnings.WarningMessage``.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        params, _, info = getattr(estimator(), call)(verbose=False, **kwargs)
    return params, info, caught


def runtime_warnings(caught):
    """Return the ``RuntimeWarning`` records.

    Parameters
    ----------
    caught : list of warnings.WarningMessage
        Recorded warnings.

    Returns
    -------
    list of warnings.WarningMessage
        Those of category ``RuntimeWarning``.
    """
    return [item for item in caught
            if issubclass(item.category, RuntimeWarning)]


@pytest.mark.parametrize('method, expected_message', [
    ('LM', 'Small step'),
    ('scipy', '`ftol` termination condition is satisfied.'),
    ('least_squares', '`ftol` termination condition is satisfied.')])
def test_converged_fit_reports_success_without_warning(method,
                                                       expected_message):
    params, info, caught = fit('optimize_fn', method=method)

    assert runtime_warnings(caught) == []
    assert info['success'] is True
    assert info['message'] == expected_message
    assert params[0] > CONVERGED_RATE_FLOOR


@pytest.mark.parametrize('options, expected_message, expected_trials', [
    ({'eps_1': GRADIENT_EPS_1}, 'Small gradient', None),
    ({'tol_fun': RESIDUAL_TOL_FUN},
     'Value of the objective function lower than tolerance', 1)])
def test_every_lm_convergence_test_counts_as_success(options,
                                                     expected_message,
                                                     expected_trials):
    _, info, caught = fit('optimize_fn', method='LM', optim_options=options)

    assert runtime_warnings(caught) == []
    assert info['stop_criterion'] == expected_message
    assert info['success'] is True
    assert info['message'] == expected_message
    if expected_trials is not None:
        assert info['num_fun_eval'] == expected_trials


def test_lm_at_evaluation_limit_warns_and_keeps_last_iterate():
    params, info, caught = fit('optimize_fn', method='LM',
                               optim_options={'max_fun_eval': ONE_EVALUATION})

    # The solver's own last iterate under the same budget, from a direct
    # call on an independent estimator.
    reference = estimator()
    expected = levenberg_marquardt(
        reference.param_seed, reference.get_objective,
        reference.get_gradient, max_fun_eval=ONE_EVALUATION,
        args=(True,))  # [1/s]

    np.testing.assert_array_equal(params, expected)
    assert SEED_RATE < params[0] < CONVERGED_RATE_FLOOR
    assert info['success'] is False
    assert info['stop_criterion'] == 'Maximum iterations exceeded'
    assert info['message'] == 'Maximum iterations exceeded'
    unconverged = runtime_warnings(caught)
    assert len(unconverged) == 1
    message = str(unconverged[0].message)
    assert message.startswith(
        "Levenberg-Marquardt (method='LM') stopped without converging "
        "(Maximum iterations exceeded) after 1 trial evaluations and 1 "
        "accepted steps, bounded by max_fun_eval=1;")
    assert message.endswith(
        f"{NOT_CONVERGED}. Raise max_fun_eval or review the seed, scaling "
        "and tolerances (info['tolerances']).")
    assert unconverged[0].filename == __file__


@pytest.mark.parametrize('method', ['scipy', 'least_squares'])
def test_scipy_at_evaluation_limit_warns_and_keeps_last_iterate(method):
    params, info, caught = fit('optimize_fn', method=method,
                               optim_options={'max_nfev': PARTIAL_MAX_NFEV})

    # The solver's own last iterate under the same budget, from a direct
    # call on an independent estimator.
    reference = estimator()
    expected = least_squares(
        reference.get_objective, reference.param_seed,
        jac=lambda params_eval, *args: reference.get_gradient(
            params_eval, *args).T,
        args=(True,), max_nfev=PARTIAL_MAX_NFEV).x  # [1/s]

    np.testing.assert_array_equal(params, expected)
    assert SEED_RATE < params[0] < CONVERGED_RATE_FLOOR
    assert info['success'] is False
    assert info['status'] == SCIPY_BUDGET_STATUS
    assert info['message'] == ('The maximum number of function evaluations '
                               'is exceeded.')
    unconverged = runtime_warnings(caught)
    assert len(unconverged) == 1
    message = str(unconverged[0].message)
    assert message.startswith(
        f"scipy.optimize.least_squares (method={method!r}) stopped without "
        "converging (status 0: The maximum number of function evaluations "
        f"is exceeded.) after {info['nfev']} residual and {info['njev']} "
        "Jacobian evaluations, bounded by max_nfev=3;")
    assert message.endswith(
        f"{NOT_CONVERGED}. Raise max_nfev or review the seed, scaling and "
        "tolerances (info['tolerances']).")
    assert unconverged[0].filename == __file__


def test_scipy_automatic_evaluation_limit_is_named_as_such():
    params, info, caught = fit(
        'optimize_fn', method='scipy',
        optim_options={'ftol': None, 'xtol': None,
                       'gtol': NEAR_EPSILON_GTOL})

    assert info['success'] is False
    assert info['status'] == SCIPY_BUDGET_STATUS
    assert info['nfev'] == SCIPY_AUTOMATIC_NFEV_PER_PARAMETER * params.size
    assert info['tolerances']['max_nfev'] is None
    unconverged = runtime_warnings(caught)
    assert len(unconverged) == 1
    message = str(unconverged[0].message)
    assert ("Jacobian evaluations, bounded by max_nfev=None, SciPy's "
            "automatic limit; the returned parameters") in message
    assert unconverged[0].filename == __file__


@pytest.mark.skipif(
    'callback' not in inspect.signature(least_squares).parameters,
    reason='scipy.optimize.least_squares has no callback before SciPy 1.16')
def test_scipy_stop_without_budget_cause_names_no_budget():
    def stop(intermediate_result):
        """Stop SciPy at its first iteration.

        Parameters
        ----------
        intermediate_result : scipy.optimize.OptimizeResult
            SciPy's current iterate (unused).

        Raises
        ------
        StopIteration
            Always, which SciPy reports as status -2.
        """
        raise StopIteration

    _, info, caught = fit('optimize_fn', method='scipy',
                          optim_options={'callback': stop})

    assert info['success'] is False
    assert info['status'] == SCIPY_CALLBACK_STATUS
    unconverged = runtime_warnings(caught)
    assert len(unconverged) == 1
    message = str(unconverged[0].message)
    assert message == (
        "scipy.optimize.least_squares (method='scipy') stopped without "
        f"converging (status -2: {info['message']}) after {info['nfev']} "
        f"residual and {info['njev']} Jacobian evaluations; the returned "
        "parameters, covariance and statistics describe the last accepted "
        f"iterate, {NOT_CONVERGED}. Review the solver status and message.")
    assert 'max_nfev' not in message
    assert unconverged[0].filename == __file__


def test_lm_evaluation_only_call_reports_without_warning():
    params, info, caught = fit('optimize_fn', method='LM',
                               optim_options={'max_fun_eval': EVALUATION_ONLY})

    assert caught == []
    np.testing.assert_array_equal(params, [SEED_RATE])
    assert 'stop_criterion' not in info
    assert info['success'] is False
    assert info['message'] == ('No iterations were requested '
                               '(max_fun_eval=0); the parameters are the '
                               'seed, evaluated only')


def test_tolerances_hold_documented_defaults():
    _, lm_info, _ = fit('optimize_fn', method='LM')
    _, scipy_info, _ = fit('optimize_fn', method='scipy')

    assert lm_info['tolerances'] == LM_DOCUMENTED_DEFAULTS
    assert list(lm_info['tolerances']) == ['max_fun_eval', 'eps_1', 'eps_2',
                                           'tol_fun']
    assert scipy_info['tolerances'] == SCIPY_DOCUMENTED_DEFAULTS
    assert list(scipy_info['tolerances']) == ['ftol', 'xtol', 'gtol',
                                              'max_nfev']


def test_given_tolerances_override_defaults():
    _, lm_info, _ = fit('optimize_fn', method='LM',
                        optim_options={'eps_1': LOOSE_EPS_1})
    _, scipy_info, _ = fit('optimize_fn', method='scipy',
                           optim_options={'ftol': TIGHT_FTOL})

    assert lm_info['tolerances'] == {**LM_DOCUMENTED_DEFAULTS,
                                     'eps_1': LOOSE_EPS_1}
    assert scipy_info['tolerances'] == {**SCIPY_DOCUMENTED_DEFAULTS,
                                        'ftol': TIGHT_FTOL}


def test_estimate_parameters_keyword_options_reach_tolerances_and_warning():
    params, info, caught = fit('estimate_parameters', method='LM',
                               max_fun_eval=ONE_EVALUATION)

    assert info['tolerances'] == {**LM_DOCUMENTED_DEFAULTS,
                                  'max_fun_eval': ONE_EVALUATION}
    assert info['success'] is False
    unconverged = runtime_warnings(caught)
    assert len(unconverged) == 1
    assert 'bounded by max_fun_eval=1;' in str(unconverged[0].message)
    # Through estimate_parameters the warning still points at this file.
    assert unconverged[0].filename == __file__

    _, merged_info, _ = fit('estimate_parameters', method='scipy',
                            optim_options={'xtol': TIGHT_FTOL},
                            ftol=TIGHT_FTOL)
    assert merged_info['tolerances'] == {**SCIPY_DOCUMENTED_DEFAULTS,
                                         'ftol': TIGHT_FTOL,
                                         'xtol': TIGHT_FTOL}


def test_ipopt_tolerances_are_the_termination_options_present():
    # Options as cyipopt 1.7 leaves them: bytes keys, its tol and
    # mu_strategy defaults and the injected print_level added.
    rewritten = {b'max_iter': 1, b'acceptable_tol': 1e-6,
                 b'print_level': 0, b'tol': 1e-8,
                 b'mu_strategy': b'adaptive',
                 b'hessian_approximation': b'limited-memory'}

    tolerances = ParamEstim._stopping_tolerances('IPOPT', rewritten)

    assert tolerances == {'tol': 1e-8, 'max_iter': 1, 'acceptable_tol': 1e-6}
    assert list(tolerances) == ['tol', 'max_iter', 'acceptable_tol']

