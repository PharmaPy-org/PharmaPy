"""Full parameter coordinates, early name validation and record replay.

The core fixture is an exact quadratic concentration history at four unequal
times. Fits free one coordinate or two coordinates around a fixed entry,
exposing both fixed-slot placement and optimized-name ordering. Expectations
are tabulated independently of the model callback.
"""

import numpy as np
import pytest

from PharmaPy.ParamEstim import Experiment, Measurement, ParameterEstimation
from PharmaPy.Records import CalibrationRecord, DataSource, ModelRevision

pytestmark = pytest.mark.unit

TIMES = np.array([0.0, 1.0, 2.0, 4.0])  # [s], unequal intervals expose ordering
TRUTH = np.array([1.0, 2.0, 3.0])  # [g/L], [g/L/s], [g/L/s**2]
OBSERVATIONS = np.array([1.0, 6.0, 17.0, 57.0])  # [g/L], 1 + 2*t + 3*t**2
NAMES = ['charge', 'rate', 'curvature']
UNITS = ['g/L', 'g/L/s', 'g/L/s**2']
# LM's objective tolerance is 1e-12; a 1e-6 absolute coefficient budget
# bounds its square-root residual scale for this small, O(1) design matrix.
FIT_ATOL = 1e-6  # [coefficient units], numerical contract rather than data accuracy


def quadratic(parameters: np.ndarray, time_s: np.ndarray) -> np.ndarray:
    """Evaluate the synthetic quadratic concentration history.

    Parameters
    ----------
    parameters : ndarray
        Charge [g/L], rate [g/L/s], curvature [g/L/s**2], shape (3,).
    time_s : ndarray
        Sample times [s], shape (n_times,).

    Returns
    -------
    ndarray
        Concentration [g/L], shape (n_times,).
    """
    return np.polynomial.polynomial.polyval(time_s, parameters)


@pytest.mark.parametrize('free_indices', [(0,), (1,), (2,), (0, 2)],
                         ids=['charge-only', 'rate-only', 'curvature-only',
                              'charge-and-curvature'])
@pytest.mark.parametrize('full_names', [False, True])
def test_both_name_forms_preserve_coordinates_through_fit_and_record(
        free_indices, full_names):
    """Keep named free coordinates in physical order across fitting and replay.

    Parameters
    ----------
    free_indices : tuple of int
        Free charge, rate or curvature coordinates in full-vector order;
        their units are [g/L], [g/L/s] and [g/L/s**2], respectively.
    full_names : bool
        Supply all coordinate names or only the free-coordinate names.
    """
    flags = [index in free_indices for index in range(len(TRUTH))]
    optimized_names = [NAMES[index] for index in free_indices]
    names = NAMES.copy() if full_names else optimized_names.copy()
    expected_names = (NAMES.copy() if full_names
                      else ['theta_1', 'theta_2', 'theta_3'])
    for index in free_indices:
        expected_names[index] = NAMES[index]
    seed = TRUTH.copy()  # [coefficient units], perturb only free entries
    seed[flags] /= 2  # [coefficient units], start halfway to the known solution
    experiments = {'run': Experiment({
        'concentration': Measurement(0, TIMES, OBSERVATIONS, units='g/L',
                                     x_units='s')})}
    estimator = ParameterEstimation(quadratic, seed, experiments,
                                    optimize_flags=flags, name_params=names)
    assert estimator.name_params == optimized_names
    assert estimator.name_params_total == expected_names
    estimator.optimize_fn(verbose=False)
    np.testing.assert_allclose(estimator.params_convg, TRUTH[flags],
                               rtol=0, atol=FIT_ATOL)
    record = CalibrationRecord.from_estimation(
        estimator, experiments, case='quadratic-identity', parameter_units=UNITS,
        model=ModelRevision('exact quadratic'),
        sources=(DataSource('analytic', 'synthetic', 'test-generated'),))
    loaded = CalibrationRecord.from_json(record.to_json())
    assert [item.name for item in loaded.parameters] == expected_names
    assert [item.units for item in loaded.parameters] == UNITS
    assert [item.optimized for item in loaded.parameters] == flags
    np.testing.assert_array_equal([item.seed for item in loaded.parameters], seed)
    for index, item in enumerate(loaded.parameters):
        if index in free_indices:
            assert item.fitted == pytest.approx(TRUTH[index], abs=FIT_ATOL, rel=0)
        else:
            assert item.fitted is None
    replay = loaded.rerun(quadratic, rtol=0, atol=FIT_ATOL)
    assert replay.name_params_total == expected_names
    np.testing.assert_allclose(replay.reconstruct_params(replay.params_convg),
                               TRUTH, rtol=0, atol=FIT_ATOL)
    assert names == (NAMES if full_names else optimized_names)


@pytest.mark.parametrize('names', [[], ['a', 'b'], ['a', 'b', 'c', 'd']])
def test_invalid_name_lengths_raise_at_construction(names):
    with pytest.raises(ValueError, match=(
            r'name_params must contain one name per parameter \(3\) '
            r'or per optimized parameter \(1\); got')):
        ParameterEstimation(quadratic, TRUTH, TIMES, OBSERVATIONS,
                            optimize_flags=[False, True, False],
                            name_params=names)


def test_optimized_name_is_preserved_when_it_matches_a_fixed_default():
    estimator = ParameterEstimation(
        quadratic, TRUTH, TIMES, OBSERVATIONS,
        optimize_flags=[False, True, False], name_params=['theta_1'])
    assert estimator.name_params == ['theta_1']
    assert estimator.name_params_total == ['theta_1_fixed', 'theta_1', 'theta_3']


def test_twelve_default_names_keep_braced_multi_digit_plot_labels():
    # Construction does not evaluate the callback, so the fixture need not
    # supply a separate twelve-parameter model for this naming-only contract.
    estimator = ParameterEstimation(quadratic, np.ones(12), TIMES,
                                    OBSERVATIONS)
    assert estimator.name_params_total == [f'theta_{index}'
                                          for index in range(1, 13)]
    assert estimator.name_params_plot == [rf'$\theta_{{{index}}}$'
                                         for index in range(1, 13)]
