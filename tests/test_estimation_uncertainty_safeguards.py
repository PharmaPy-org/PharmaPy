"""Real constant-response fits qualify failed bootstrap draws and covariance.

The mean of [1, 2, 3, 4, 5] g/L is exactly 3 g/L. One evaluation fits
that seed but cannot fit bootstrap datasets with different means.
"""

import warnings

import numpy as np
import pytest
from scipy.optimize import Bounds

from PharmaPy.ParamEstim import Experiment, Measurement, ParameterEstimation
from PharmaPy.StatsModule import StatisticsClass
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


def mean_estimator(values, budget=64, bounds=(0., 6.)):
    """Fit a mean with known unit measurement standard deviation.

    Parameters
    ----------
    values : array_like
        Observations [g/L].
    budget : int, optional
        Evaluation budget [-]; 64 permits convergence for this linear case.
    bounds : tuple, optional
        Lower and upper mean concentration [g/L].

    Returns
    -------
    ParameterEstimation
        Real fitted estimator seeded at 3 g/L.
    """
    experiment = Experiment({'mean': Measurement('mean', np.arange(len(values)),
        values, uncertainty=1., units='g/L', x_units=None)})
    estimator = ParameterEstimation(mean_model, [3.], {'mean': experiment},
        output_names=('mean',), name_params=['mean'], dx_finitediff=1e-6)
    estimator.optimize_fn(method='least_squares', bounds=bounds,
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
def test_bootstrap_marks_unsuccessful_refits_without_changing_fit():
    values = np.arange(1., 6.)  # [g/L], symmetric observations about 3
    estimator = mean_estimator(values, budget=1)
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


def test_bootstrap_adequate_budget_recovers_sample_means():
    estimator = mean_estimator(np.arange(1., 6.))
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
def test_intervals_warn_when_crossing_parameter_bounds(bounds):
    estimator = mean_estimator([4., 5., 6., 7., 8.], bounds=bounds)
    with pytest.warns(RuntimeWarning, match='bound'):
        intervals = estimator.get_confidence_intervals(verbose=False)
    assert intervals['mean'][1] > 4.  # [g/L], diagnostic preserved, never clipped


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
