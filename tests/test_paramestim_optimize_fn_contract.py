"""Result and state contract of ``ParameterEstimation.optimize_fn``.

Scope, by issue:

- The Levenberg-Marquardt ``tol_fun`` stop reached through ``optimize_fn`` and,
  directly, after a rejected trial with an ``arctan`` residual (#474).
- The estimator state after a call that raises (#479): an unknown method, a
  failed solve, a singular covariance (a second parameter that stops affecting
  the model), a failed first fit, IPOPT without cyipopt (run in a child process
  that blocks cyipopt imports, the helper of
  ``test_paramestim_calibration_fit_predict``, so it runs whether or not
  cyipopt is installed), the residuals that ``StatisticsClass.get_bootsamples``
  resamples after a failed refit (data with fixed asymmetric errors), and a
  failed ``MultipleCurveResolution`` refit (the fixture of
  ``test_paramestim_mcr_observation_masks``), as well as the iteration history
  of refits with and without ``store_iter``.

Main fixture: a noise-free proportional model ``y = k * t``, time [s], rate
constant ``k`` [mol/L/s] and concentration [mol/L], whose exact least-squares
solution is the generating rate. Fits are cheap LM or SciPy solves in the core
environment.
"""

import inspect
import textwrap

import numpy as np
import pytest

from PharmaPy.LevMarq import levenberg_marquardt
from PharmaPy.ParamEstim import ParameterEstimation
from PharmaPy.StatsModule import StatisticsClass
from test_paramestim_calibration_fit_predict import _run_with_cyipopt_blocked
import test_paramestim_mcr_observation_masks as mcr_case


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


class ModelFailure(Exception):
    """Sentinel raised by the guarded model of the failed-solve test."""


def fitted_estimator(model=proportional):
    """Return an estimator after a successful LM fit of the exact data.

    Parameters
    ----------
    model : callable, optional
        Model callback with the signature of ``proportional``.

    Returns
    -------
    ParameterEstimation
        Estimator whose ``params_convg`` is the LM estimate [mol/L/s].
    """
    data = proportional([TRUE_RATE], TIME_S)  # [mol/L]
    estimator = ParameterEstimation(model, [SEED_RATE], TIME_S, data)
    estimator.optimize_fn(verbose=False)
    return estimator


# Fit results compared by identity: a failed call must not rebind them.
FIT_OBJECTS = ('optim_options', 'info_opt', 'y_model', 'covar_params',
               'correl_params', 'paramest_df')
# Fit arrays compared by value: params_convg [mol/L/s] and the histories,
# params_iter [mol/L/s] and objfun_iter [(mol/L)**2].
FIT_ARRAYS = ('params_convg', 'params_iter', 'objfun_iter')


def fit_state(estimator):
    """Return the attributes that describe the stored fit.

    Parameters
    ----------
    estimator : ParameterEstimation
        Fitted estimator.

    Returns
    -------
    dict
        ``opt_method``, ``optimize_flag``, the objects in ``FIT_OBJECTS``
        and copies of the arrays in ``FIT_ARRAYS``, in their own units.
    """
    state = {'opt_method': estimator.opt_method,
             'optimize_flag': estimator.optimize_flag}
    state.update({name: getattr(estimator, name) for name in FIT_OBJECTS})
    state.update({name: np.array(getattr(estimator, name), copy=True)
                  for name in FIT_ARRAYS})
    return state


def assert_same_fit(before, estimator):
    """Assert that ``estimator`` still describes the fit in ``before``.

    Parameters
    ----------
    before : dict
        Result of ``fit_state`` before the failed call.
    estimator : ParameterEstimation
        Estimator after the failed call.
    """
    after = fit_state(estimator)
    assert after['opt_method'] == before['opt_method'] == 'LM'
    assert after['optimize_flag'] is before['optimize_flag'] is True
    for name in FIT_OBJECTS:
        assert after[name] is before[name], name
    for name in FIT_ARRAYS:
        np.testing.assert_array_equal(after[name], before[name],
                                      err_msg=name)


def test_unknown_method_keeps_previous_fit():
    estimator = fitted_estimator()
    before = fit_state(estimator)

    with pytest.raises(ValueError, match=(
            "Unknown optimization method 'lm'; expected 'LM', 'IPOPT', "
            "'scipy' or 'least_squares'")):
        estimator.optimize_fn(method='lm', simulate=True, verbose=False)

    assert_same_fit(before, estimator)


def test_failed_solve_keeps_previous_fit():
    guard = {'fail': False}

    def guarded(params, time_s):
        """Return ``proportional`` unless ``guard['fail']`` is set.

        Parameters
        ----------
        params : array_like
            ``[k]``, rate constant [mol/L/s].
        time_s : numpy.ndarray
            Sample times [s].

        Returns
        -------
        numpy.ndarray
            Concentration [mol/L].

        Raises
        ------
        ModelFailure
            When ``guard['fail']`` is True.
        """
        if guard['fail']:
            raise ModelFailure('model evaluation failed')
        return proportional(params, time_s)

    estimator = fitted_estimator(guarded)
    before = fit_state(estimator)
    guard['fail'] = True

    with pytest.raises(ModelFailure, match='model evaluation failed'):
        estimator.optimize_fn(method='scipy', simulate=True, verbose=False)

    assert_same_fit(before, estimator)


def test_singular_covariance_keeps_previous_fit():
    """A fit whose covariance cannot be formed leaves the previous fit.

    In ``dead`` mode the second parameter has no effect on the model, so
    the solve succeeds but ``J @ J.T`` is singular and ``get_covariance``
    raises after the new results were assembled.
    """
    guard = {'dead': False}

    def curvature(params, time_s):
        """Return ``k * t + a * t**2``, without the ``a`` term when dead.

        Parameters
        ----------
        params : array_like
            ``[k, a]``: rate [mol/L/s] and its change [mol/L/s**2].
        time_s : numpy.ndarray
            Sample times [s].

        Returns
        -------
        numpy.ndarray
            Concentration [mol/L].
        """
        quadratic = 0.0 if guard['dead'] else params[1] * time_s**2
        return params[0] * time_s + quadratic

    data = proportional([TRUE_RATE], TIME_S)  # [mol/L]
    estimator = ParameterEstimation(curvature, [SEED_RATE, SEED_RATE],
                                    TIME_S, data)
    estimator.optimize_fn(verbose=False)
    before = fit_state(estimator)
    guard['dead'] = True

    with pytest.raises(np.linalg.LinAlgError, match='Singular matrix'):
        estimator.optimize_fn(method='scipy', simulate=True, verbose=False)

    assert_same_fit(before, estimator)


# Fixed asymmetric measurement errors of the bootstrap case [mol/L], so
# the fit has nonzero residuals to resample.
DATA_ERRORS = np.array([0.01, -0.02, 0.0, 0.01])  # [mol/L]
BOOTSTRAP_SAMPLES = 5  # [-]
BOOTSTRAP_SEED = 20261027  # [-], arbitrary fixed NumPy seed
# An LM budget of the wrong type: LM evaluates the seed, then fails when it
# compares its evaluation count with it.
INVALID_BUDGET = {'max_fun_eval': 'invalid'}


def seeded_bootsamples(statistics):
    """Draw residual-bootstrap datasets with NumPy's generator seeded.

    Parameters
    ----------
    statistics : StatisticsClass
        Statistics of the fitted estimator.

    Returns
    -------
    list of list of numpy.ndarray
        ``get_bootsamples(BOOTSTRAP_SAMPLES)`` [mol/L]; the previous
        generator state is restored afterwards.
    """
    previous_state = np.random.get_state()
    try:
        np.random.seed(BOOTSTRAP_SEED)
        return statistics.get_bootsamples(BOOTSTRAP_SAMPLES)
    finally:
        np.random.set_state(previous_state)


def test_failed_refit_keeps_bootstrap_residuals():
    data = proportional([TRUE_RATE], TIME_S) + DATA_ERRORS  # [mol/L]
    estimator = ParameterEstimation(proportional, [SEED_RATE], TIME_S, data)
    estimator.optimize_fn(verbose=False)
    statistics = StatisticsClass(estimator)
    expected = seeded_bootsamples(statistics)  # [mol/L]
    before = fit_state(estimator)

    with pytest.raises(TypeError, match="'<' not supported"):
        estimator.optimize_fn(optim_options=dict(INVALID_BUDGET),
                              verbose=False)

    assert_same_fit(before, estimator)
    for samples in (seeded_bootsamples(statistics),
                    seeded_bootsamples(StatisticsClass(estimator))):
        for actual_runs, expected_runs in zip(samples, expected):
            for actual, wanted in zip(actual_runs, expected_runs):
                np.testing.assert_array_equal(actual, wanted)


@pytest.mark.parametrize('stage', ['solve', 'covariance'])
def test_failed_first_fit_leaves_an_unfitted_estimator(stage):
    """A first fit that raises leaves no fit results behind.

    ``'solve'`` fails inside LM after the seed evaluation; ``'covariance'``
    finishes a SciPy solve whose second parameter has no effect, so
    ``get_covariance`` raises after the results were assembled.
    """
    def unused_second(params, time_s):
        """Return ``k * t``; the second parameter [mol/L/s**2] is unused.

        Parameters
        ----------
        params : array_like
            ``[k, a]``: rate [mol/L/s] and an unused coefficient.
        time_s : numpy.ndarray
            Sample times [s].

        Returns
        -------
        numpy.ndarray
            Concentration [mol/L].
        """
        return params[0] * time_s

    data = proportional([TRUE_RATE], TIME_S) + DATA_ERRORS  # [mol/L]
    estimator = ParameterEstimation(unused_second, [SEED_RATE, SEED_RATE],
                                    TIME_S, data)
    histories = (estimator.params_iter, estimator.objfun_iter)
    buffers = {name: getattr(estimator, name)
               for name in ('y_runs', 'resid_runs', 'weighted_residuals',
                            'params_residuals', 'sens', 'y_model')}

    if stage == 'solve':
        with pytest.raises(TypeError, match="'<' not supported"):
            estimator.optimize_fn(optim_options=dict(INVALID_BUDGET),
                                  verbose=False)
    else:
        with pytest.raises(np.linalg.LinAlgError, match='Singular matrix'):
            estimator.optimize_fn(method='scipy', verbose=False)

    for name in ('opt_method', 'optim_options', 'params_convg', 'info_opt',
                 'covar_params', 'correl_params', 'paramest_df',
                 'residuals'):
        assert not hasattr(estimator, name), name
    assert estimator.params_iter is histories[0] and histories[0] == []
    assert estimator.objfun_iter is histories[1] and histories[1] == []
    for name, value in buffers.items():
        assert getattr(estimator, name) is value, name


def test_refit_records_its_own_iteration_history():
    first_data = proportional([TRUE_RATE], TIME_S) + DATA_ERRORS  # [mol/L]
    estimator = ParameterEstimation(proportional, [SEED_RATE], TIME_S,
                                    first_data)
    first, _, _ = estimator.optimize_fn(verbose=False)  # [mol/L/s]
    # New data from twice the rate: a refit, e.g. after correcting them.
    estimator.y_data = [proportional([2 * TRUE_RATE], TIME_S)[:, None]
                        + DATA_ERRORS[:, None]]  # [mol/L]

    second, _, info = estimator.optimize_fn(verbose=False)  # [mol/L/s]

    assert info['stop_criterion'] in ('Small step', 'Small gradient')
    rates = estimator.paramest_df['theta_1'].to_numpy()  # [mol/L/s]
    # LM stopped before a further trial, so the last unique iterate is the
    # accepted rate; the first fit's estimate is not in this history.
    assert rates[-1] == second[0]
    assert first[0] not in rates
    np.testing.assert_array_equal(estimator.params_iter[:, 0], rates)
    assert len(estimator.objfun_iter) == len(rates)


def test_unstored_refit_replaces_the_stored_history():
    """A ``store_iter=False`` refit keeps only its own evaluations."""
    first_data = proportional([TRUE_RATE], TIME_S) + DATA_ERRORS  # [mol/L]
    estimator = ParameterEstimation(proportional, [SEED_RATE], TIME_S,
                                    first_data)
    first, _, _ = estimator.optimize_fn(verbose=False)  # [mol/L/s]
    assert hasattr(estimator, 'paramest_df')
    estimator.y_data = [proportional([2 * TRUE_RATE], TIME_S)[:, None]
                        + DATA_ERRORS[:, None]]  # [mol/L]

    second, _, _ = estimator.optimize_fn(store_iter=False,
                                         verbose=False)  # [mol/L/s]

    assert not hasattr(estimator, 'paramest_df')
    rates = estimator.params_iter[:, 0]  # [mol/L/s], every evaluation
    assert rates[0] == SEED_RATE
    assert second[0] in rates
    assert first[0] not in rates
    assert len(estimator.objfun_iter) == len(rates)


@pytest.mark.parametrize('failure', ['model', 'budget'])
def test_failed_mcr_refit_restores_absorptivities(failure):
    """A refit failing after one complete evaluation restores the MCR fit.

    The first evaluation of the refit, at the seed, resolves new
    absorptivities and caches them for ``get_gradient``. With ``'model'``
    the model then fails at its next call; with ``'budget'`` LM fails on
    ``INVALID_BUDGET`` after evaluating the seed, leaving that evaluation
    cached unless the restore clears it.
    """
    guard = {'fail': False, 'calls': 0}

    def model(params, time_s, **kwargs):
        """Return the MCR fixture states until the refit's second evaluation.

        Parameters
        ----------
        params : array_like
            ``[k]``, rate constant [1/s].
        time_s : numpy.ndarray
            Sample times [s].
        **kwargs
            Passed to ``model_states_only``.

        Returns
        -------
        numpy.ndarray
            States of ``mcr_case.model_states_only``.

        Raises
        ------
        ModelFailure
            When ``guard['fail']`` is True, after one call per experiment.
        """
        if guard['fail']:
            guard['calls'] += 1
            if guard['calls'] > len(mcr_case.SCHEDULES):
                raise ModelFailure('model evaluation failed')
        return mcr_case.model_states_only(params, time_s, **kwargs)

    estimator = mcr_case._estimator(model, 'correlated')
    estimator.optimize_fn(verbose=False)
    before = fit_state(estimator)
    absorptivity = estimator.epsilon_mcr
    saved = np.array(absorptivity, copy=True)  # [L/mol]

    if failure == 'model':
        guard['fail'] = True
        with pytest.raises(ModelFailure, match='model evaluation failed'):
            estimator.optimize_fn(verbose=False)
        assert guard['calls'] == len(mcr_case.SCHEDULES) + 1
    else:
        with pytest.raises(TypeError, match="'<' not supported"):
            estimator.optimize_fn(optim_options=dict(INVALID_BUDGET),
                                  verbose=False)

    assert_same_fit(before, estimator)
    assert estimator.epsilon_mcr is absorptivity
    np.testing.assert_array_equal(estimator.epsilon_mcr, saved)
    assert estimator._params_mcr_eval is None
    assert estimator._mcr_evaluation is None


def test_ipopt_without_cyipopt_keeps_previous_lm_fit(tmp_path):
    script = textwrap.dedent('''
        import numpy as np

        from PharmaPy import ParamEstim

        if ParamEstim.have_cyipopt is not False:
            raise AssertionError("blocked cyipopt was reported as available")

        time_s = np.linspace(1.0, 4.0, 4)  # [s]

        def proportional(params, time_s):
            """Return concentration [mol/L] from rate [mol/L/s], time [s]."""
            return params[0] * time_s

        seed_rate = [0.3]  # [mol/L/s], away from the solution
        true_rate = [0.5]  # [mol/L/s], generating rate of the exact data
        estimator = ParamEstim.ParameterEstimation(
            proportional, seed_rate, time_s, proportional(true_rate, time_s))
        estimator.optimize_fn(verbose=False)
        fitted = np.array(estimator.params_convg, copy=True)  # [mol/L/s]
        info, options = estimator.info_opt, estimator.optim_options

        try:
            estimator.optimize_fn(method="IPOPT", simulate=True,
                                  verbose=False)
        except ImportError as exc:
            if "cyipopt is an optional import" not in str(exc):
                raise AssertionError(str(exc)) from exc
        else:
            raise AssertionError("IPOPT selection unexpectedly succeeded")

        state = (estimator.opt_method, estimator.optimize_flag)
        if state != ("LM", True):
            raise AssertionError(f"failed IPOPT call relabelled: {state}")
        if estimator.info_opt is not info:
            raise AssertionError("info_opt replaced")
        if estimator.optim_options is not options:
            raise AssertionError("optim_options replaced")
        if not np.array_equal(estimator.params_convg, fitted):
            raise AssertionError("params_convg changed")
        ''')

    result = _run_with_cyipopt_blocked(script, tmp_path)

    assert result.returncode == 0, result.stderr
