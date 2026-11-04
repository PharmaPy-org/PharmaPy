"""Custom-optimizer replay through the public residual-bootstrap APIs.

The concentration model c = rate * time [mol/L] has an independently known
least-squares solution. A genuine SciPy least_squares adapter exercises both
custom-callable entry forms, with asymmetric observation errors. Core lane;
no optional solver or model substitute is used. Seeded sampling reuses the
state-preserving test helper from the bootstrap-state regressions.
"""

from functools import partial

import numpy as np
import pytest
from scipy.optimize import least_squares

from PharmaPy.ParamEstim import ParameterEstimation
from PharmaPy.StatsModule import ParallelProblem, StatisticsClass
from test_statistics_bootstrap_state import _assert_same, _seeded, _state

pytestmark = pytest.mark.unit

TIME_S = np.array([1.0, 2.0, 3.0, 4.0])  # [s], four distinct sampling times
TRUE_RATE = 2.0  # [mol/L/s], slope of the synthetic concentration response
SEED_RATE = np.array([1.0])  # [mol/L/s], initial guess below the true rate
# [mol/L], asymmetric errors keep residual variance nonzero for statistics.
OBSERVATION_ERRORS = np.array([0.1, -0.1, 0.1, -0.1])
SAMPLE_SHIFT = 0.2  # [mol/L], changed dataset with a distinct fitted slope
NUM_SAMPLES = 5  # [-], several draws exercise repeated isolated refits
# These smooth linear fits use an analytic Jacobian. SciPy's 1e-12 stopping
# tolerances and a 1e-10 relative comparison permit roundoff while resolving
# sample-to-sample slope differences, which are of order 1e-2 mol/L/s.
SOLVER_TOLERANCE = 1e-12  # [-], SciPy ftol/xtol/gtol stopping thresholds
REFIT_RTOL = 1e-10  # [-], relative slope comparison tolerance


def _linear_concentration(params, time_s):
    """Return the zero-intercept concentration model.

    Parameters
    ----------
    params : numpy.ndarray
        Rate, shape (1,) [mol/L/s].
    time_s : numpy.ndarray
        Sampling times, shape (n_times,) [s].

    Returns
    -------
    numpy.ndarray
        Concentrations, shape (n_times,) [mol/L].
    """
    return params[0] * time_s


def _linear_sensitivity(params, time_s):
    """Return the exact concentration derivative with respect to rate.

    Parameters
    ----------
    params : numpy.ndarray
        Rate, shape (1,) [mol/L/s]; unused in this linear derivative.
    time_s : numpy.ndarray
        Sampling times, shape (n_times,) [s].

    Returns
    -------
    numpy.ndarray
        Sensitivities, shape (n_times, 1) [s].
    """
    return time_s[:, np.newaxis]


def _scipy_optimizer(seed, objective_fun, gradient_fun, **options):
    """Fit residuals using a real SciPy optimizer and PharmaPy callbacks.

    Parameters
    ----------
    seed : numpy.ndarray
        Rate seed, shape (1,) [mol/L/s].
    objective_fun : callable
        PharmaPy objective; out_array=True returns residuals [mol/L].
    gradient_fun : callable
        PharmaPy gradient; out_array=True returns the residual Jacobian
        with shape (1, n_times) [s].
    **options : dict
        Native least_squares options, including stopping tolerances [-].

    Returns
    -------
    scipy.optimize.OptimizeResult
        Fit and native solver diagnostics; x holds the rate [mol/L/s].
    """
    def residual_jacobian(params):
        """Return the SciPy-oriented residual Jacobian.

        Parameters
        ----------
        params : numpy.ndarray
            Trial rate, shape (1,) [mol/L/s].

        Returns
        -------
        numpy.ndarray
            Residual derivative, shape (n_times, 1) [s].
        """
        return gradient_fun(params, out_array=True).T

    return least_squares(partial(objective_fun, out_array=True), seed,
                         jac=residual_jacobian, **options)


def _fitted_estimator(entry_form):
    """Fit the synthetic concentrations using one custom entry form.

    Parameters
    ----------
    entry_form : {'callable', 'options'}
        Invocation spelling of the same real custom optimizer.

    Returns
    -------
    estimator : ParameterEstimation
        Accepted fit with stored private solver options and callable.
    observed : numpy.ndarray
        Original observations, shape (4,) [mol/L].
    """
    observed = TRUE_RATE * TIME_S + OBSERVATION_ERRORS  # [mol/L]
    estimator = ParameterEstimation(
        _linear_concentration, SEED_RATE, TIME_S, observed,
        jac_fun=_linear_sensitivity, name_params=['rate'])
    options = dict(xtol=SOLVER_TOLERANCE, ftol=SOLVER_TOLERANCE,
                   gtol=SOLVER_TOLERANCE)
    method = _scipy_optimizer
    if entry_form == 'options':
        method = 'custom'
        options['optimizer'] = _scipy_optimizer
    estimator.optimize_fn(method=method, optim_options=options,
                          verbose=False, store_iter=False)
    return estimator, observed


@pytest.mark.parametrize('entry_form', ['callable', 'options'])
def test_parallel_refit_preserves_custom_optimizer_and_original_fit(entry_form):
    estimator, observed = _fitted_estimator(entry_form)
    before = _state(estimator)
    sample = observed + SAMPLE_SHIFT  # [mol/L]
    # Least squares of c = rate * t has rate = sum(t*c)/sum(t**2).
    expected = TIME_S @ sample / (TIME_S @ TIME_S)  # [mol/L/s]

    actual = ParallelProblem(estimator).run_fit([sample])  # [mol/L/s]

    np.testing.assert_allclose(actual, [expected], rtol=REFIT_RTOL)
    assert not np.isclose(expected, estimator.params_convg[0],
                          rtol=REFIT_RTOL, atol=0)
    _assert_same(before, _state(estimator))
    assert estimator._custom_optimizer is _scipy_optimizer
    assert ('optimizer' in estimator.optim_options) == (entry_form == 'options')


@pytest.mark.parametrize('entry_form', ['callable', 'options'])
def test_bootstrap_preserves_custom_optimizer_and_original_fit(entry_form):
    estimator, _ = _fitted_estimator(entry_form)
    statistics = StatisticsClass(estimator)
    before = _state(estimator)
    samples = _seeded(statistics.get_bootsamples, NUM_SAMPLES)
    # The identical seeded draws are fitted below. Each expected slope follows
    # from the analytic scalar least-squares normal equation, not a refit.
    expected = np.array([
        TIME_S @ sample[:, 0] / (TIME_S @ TIME_S)
        for sample in samples[0]])  # [mol/L/s]

    actual = _seeded(statistics.bootstrap_params, NUM_SAMPLES)  # [mol/L/s]

    assert actual.shape == (NUM_SAMPLES, 1)
    np.testing.assert_allclose(actual[:, 0], expected, rtol=REFIT_RTOL)
    assert np.ptp(expected) > REFIT_RTOL * abs(estimator.params_convg[0])
    _assert_same(before, _state(estimator))
    assert estimator._custom_optimizer is _scipy_optimizer
    assert ('optimizer' in estimator.optim_options) == (entry_form == 'options')
