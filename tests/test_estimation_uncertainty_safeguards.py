"""Real constant-response fits qualify failed bootstrap draws and covariance.

The mean of [1, 2, 3, 4, 5] g/L is exactly 3 g/L. One evaluation fits
that seed but cannot fit bootstrap datasets with different means.
"""

import warnings
from functools import partial

import numpy as np
import pytest
from scipy.optimize import Bounds, least_squares

from PharmaPy.ParamEstim import Experiment, Measurement, ParameterEstimation
from PharmaPy.StatsModule import StatisticsClass, _interval_bounds
from test_statistics_bootstrap_state import _seeded

pytestmark = pytest.mark.unit


# Real fixture construction and independent qualification helpers.
def mean_model(parameters, times):
    """Evaluate a constant concentration.

    Parameters
    ----------
    parameters : array_like
        Mean concentration [g/L], shape (1,).
    times : array_like
        Sampling coordinates [-].

    Returns
    -------
    numpy.ndarray
        Concentrations [g/L], shape (n_times, 1).
    """
    return np.full((len(times), 1), parameters[0])


def scipy_optimizer(seed, objective_fun, gradient_fun, **options):
    """Use real SciPy through the native custom-optimizer interface.

    Parameters
    ----------
    seed : array_like
        Initial parameters [model parameter units], shape (n_parameters,).
    objective_fun : callable
        Native objective; ``out_array=True`` returns weighted residuals [-].
    gradient_fun : callable
        Native gradient callback [1 / parameter unit]. SciPy's forward differences
        are used here, so this callback is not required by the adapter.
    **options : dict
        Native solver options, including evaluation budget [-] and parameter bounds.

    Returns
    -------
    scipy.optimize.OptimizeResult
        Real solver status and fitted values [model parameter units].
    """
    return least_squares(partial(objective_fun, out_array=True), seed, **options)


def mean_estimator(values, budget=64, bounds=(0., 6.), *, custom=False):
    """Fit a mean with known unit measurement standard deviation.

    Parameters
    ----------
    values : array_like
        Observations [g/L].
    budget : int, optional
        Evaluation budget [-]; 64 permits convergence for this linear case.
    bounds : tuple, optional
        Lower and upper mean concentration [g/L].
    custom : bool, optional
        Exercise the native custom-optimizer entry point with real SciPy.
        Otherwise use the native built-in least-squares entry point.

    Returns
    -------
    ParameterEstimation
        Real fitted estimator seeded at 3 g/L.
    """
    experiment = Experiment({'mean': Measurement('mean', np.arange(len(values)),
        values, uncertainty=1., units='g/L', x_units=None)})
    estimator = ParameterEstimation(mean_model, [3.], {'mean': experiment},
        output_names=('mean',), name_params=['mean'], dx_finitediff=1e-6)
    method = scipy_optimizer if custom else 'least_squares'
    estimator.optimize_fn(method=method, bounds=bounds,
        optim_options={'max_nfev': budget}, verbose=False)
    return estimator


def affine_model(parameters, times):
    """Evaluate concentration with a constant production rate.

    Parameters
    ----------
    parameters : array_like
        Initial concentration [g/L] and production rate [g/(L s)].
    times : array_like
        Sampling times [s].

    Returns
    -------
    numpy.ndarray
        Concentrations [g/L], shape (n_times, 1).
    """
    return (parameters[0] + parameters[1]*times)[:, None]


# Public-path regressions, independent balances, and failure qualification.
@pytest.mark.parametrize('custom', [False, True])
def test_bootstrap_marks_unsuccessful_refits_without_changing_fit(custom):
    """Retain failure status through both native optimizer entry points.

    Parameters
    ----------
    custom : bool
        Use the real custom SciPy adapter instead of the built-in entry point.
    """
    values = np.arange(1., 6.)  # [g/L], symmetric observations about 3
    estimator = mean_estimator(values, budget=1, custom=custom)
    statistics = StatisticsClass(estimator)
    samples = _seeded(statistics.get_bootsamples, 32)[0]  # [g/L]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        draws = _seeded(statistics.bootstrap_params, 32)  # [g/L]
    failed = np.isnan(draws[:, 0])
    assert failed.any()
    diagnostics = [item for item in caught
                   if 'Bootstrap optimization failed' in str(item.message)]
    assert len(diagnostics) == failed.sum()
    assert all('sample' in str(item.message) for item in diagnostics)
    assert len(statistics.bootstrap_failures) == failed.sum()
    expected = np.mean(samples, axis=1)[:, 0]  # [g/L], exact sample optima
    np.testing.assert_array_equal(failed, np.abs(expected-3.) > 1e-8)
    assert [entry['sample_index'] for entry in statistics.bootstrap_failures] == np.flatnonzero(failed).tolist()
    np.testing.assert_allclose(estimator.params_convg, [3.])
    np.testing.assert_array_equal(estimator.y_data[0][:, 0], values)


@pytest.mark.parametrize('custom', [False, True])
def test_bootstrap_adequate_budget_recovers_sample_means(custom):
    """Recover independently calculated sample means through both entry points.

    Parameters
    ----------
    custom : bool
        Use the real custom SciPy adapter instead of the built-in entry point.
    """
    estimator = mean_estimator(np.arange(1., 6.), custom=custom)
    statistics = StatisticsClass(estimator)
    samples = _seeded(statistics.get_bootsamples, 32)[0]  # [g/L]
    draws = _seeded(statistics.bootstrap_params, 32)  # [g/L]
    np.testing.assert_allclose(draws, np.mean(samples, axis=1), atol=1e-7, rtol=0.)
    assert statistics.bootstrap_failures == []


def test_single_observation_absolute_information_is_defined():
    with pytest.warns(RuntimeWarning, match='degrees of freedom'):
        estimator = mean_estimator([3.])
    assert np.isnan(estimator.covar_params).all()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        covariance = estimator.get_covariance(include_mse=False)
    assert not caught
    np.testing.assert_allclose(covariance, [[1.]], rtol=1e-8)


@pytest.mark.parametrize('bounds', [(0., 4.), Bounds(0., 4.)])
@pytest.mark.parametrize('custom', [False, True])
def test_intervals_warn_when_crossing_parameter_bounds(bounds, custom):
    """Preserve untruncated intervals for both native optimization routes.

    Parameters
    ----------
    bounds : tuple or scipy.optimize.Bounds
        Mean concentration limits [g/L].
    custom : bool
        Use the real custom SciPy adapter instead of the built-in entry point.
    """
    estimator = mean_estimator([4., 5., 6., 7., 8.], bounds=bounds, custom=custom)
    with pytest.warns(RuntimeWarning, match='bound'):
        intervals = estimator.get_confidence_intervals(verbose=False)
    assert intervals['mean'][1] > 4.  # [g/L], diagnostic preserved, never clipped


def test_custom_ambiguous_bounds_warn_without_losing_intervals():
    """A two-parameter custom layout cannot be inferred from the fitted values."""
    times = np.arange(5.)  # [s]; five observations for two independent parameters
    offsets = np.array([.1, -.2, .2, -.2, .1])  # [g/L]; orthogonal to constant and time columns
    initial = 3.  # [g/L]; synthetic intercept
    rate = 1.  # [g/(L s)]; synthetic constant production rate
    values = initial + rate * times + offsets  # [g/L]; orthogonal noise preserves both exact coefficients
    bounds = ([0., 0.], [2 * initial, 2 * rate])  # [g/L; g/(L s)] per side; positive ranges twice the true coefficients
    experiment = Experiment({'concentration': Measurement('concentration', times,
        values, uncertainty=1., units='g/L', x_units='s')})
    intervals = []
    for method in ('least_squares', scipy_optimizer):
        estimator = ParameterEstimation(affine_model, [initial, rate], {'run': experiment},
            output_names=('concentration',), name_params=['initial', 'rate'])
        estimator.optimize_fn(method=method, bounds=bounds, verbose=False)
        np.testing.assert_allclose(estimator.params_convg, [initial, rate], rtol=1e-7, atol=0.)  # [g/L; g/(L s)]; independently known linear-fit coefficients
        if callable(method):
            with pytest.warns(RuntimeWarning, match='Custom optimizer bounds cannot be interpreted'):
                result = estimator.get_confidence_intervals(verbose=False)
        else:
            result = estimator.get_confidence_intervals(verbose=False)
        intervals.append(np.array([result['initial'], result['rate']]))  # [g/L; g/(L s)]
        assert intervals[-1].shape == (2, 2)
        assert np.isfinite(intervals[-1]).all()
    np.testing.assert_allclose(intervals[1], intervals[0], rtol=1e-7, atol=0.)  # [-]; smooth linear model, finite-difference roundoff


def test_interval_bounds_preserve_ipopt_parameter_pairs():
    """Qualify the solver-independent IPOPT layout without substituting a solver."""
    lower, upper = _interval_bounds([(0., 4.), (None, 7.)], ['first', 'second'], 'IPOPT')  # [g/L]; second parameter is unbounded below
    np.testing.assert_array_equal(lower, [0., -np.inf])
    np.testing.assert_array_equal(upper, [4., 7.])


def test_information_assessment_uses_exact_mean_derivative():
    estimator = mean_estimator([1., 2., 3., 4., 5.])
    # Unit standard deviations make d(weighted residual)/d(mean) exactly one.
    exact = np.ones((1, 5))  # [L/g]
    report = estimator.assess_information([1.], 1e-5, exact)
    assert report['full_rank']
    assert report['derivative_refinement_checked']
    np.testing.assert_allclose(report['scaled_singular_values'], [np.sqrt(5.)], rtol=1e-8)
    assert report['relative_derivative_discrepancy'] < 1e-8


@pytest.mark.parametrize('scales,tolerance', [([0.], 1e-5), ([1.], 0.), ([1.], np.nan)])
def test_information_assessment_rejects_invalid_resolution(scales, tolerance):
    estimator = mean_estimator([1., 2., 3., 4., 5.])
    with pytest.raises(ValueError, match='parameter_scales|relative_tolerance'):
        estimator.assess_information(scales, tolerance)


@pytest.mark.parametrize('duration_s,expected_rank', [(4., 2), (4e-6, 1)])
def test_information_resolves_only_measured_parameter_directions(duration_s, expected_rank):
    times = np.linspace(0., duration_s, 5)  # [s], resolved vs extremely short observation window
    values = 3. + times  # [g/L], c0=3 g/L and rate=1 g/(L s)
    experiment = Experiment({'concentration': Measurement('concentration', times,
        values, uncertainty=1., units='g/L', x_units='s')})
    estimator = ParameterEstimation(affine_model, [3., 1.], {'run': experiment},
        output_names=('concentration',), dx_finitediff=1e-6)
    estimator.optimize_fn(method='least_squares', verbose=False)
    exact = np.vstack((np.ones(5), times))  # [L/g; L s/g], known SD=1 g/L
    # A 1e-5 resolved fraction separates the deliberately six-decade windows.
    report = estimator.assess_information([1., 1.], 1e-5, exact)
    assert report['resolved_rank'] == expected_rank
    assert report['full_rank'] == (expected_rank == 2)


def test_intervals_warn_after_unsuccessful_optimization():
    estimator = mean_estimator([4., 5., 6., 7., 8.], budget=1)
    assert not estimator.info_opt['success']
    with pytest.warns(RuntimeWarning, match='unsuccessful fit'):
        estimator.get_confidence_intervals(verbose=False)
