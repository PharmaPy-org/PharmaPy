"""Statistics, covariance and bootstrap refits of Experiment input (#346).

Scope: ``StatisticsClass`` residual bootstrap (``get_bootsamples``,
``bootstrap_params``) and ``ParameterEstimation.get_covariance`` for
estimators built from ``Experiment``/``Measurement`` objects with unequal
grids, replicate samples, a missing (NaN) observation, a single-point
measurement and a measurement absent from one experiment, with and without
declared standard deviations in two physical units ([mol/L] and [K]).

Fixture: zero-order product formation in two batch experiments, product
concentration ``c = c0 + k t`` [mol/L] and adiabatic temperature
``T = T0 + rise * k * t`` [K], linear in the rate ``k`` [mol/L/s]. Fitted
rates, covariances and bootstrap refits therefore have closed-form weighted
least-squares (WLS) values, computed here from the hand-written fixture
rather than from estimator internals. Bootstrap draws use NumPy's global
generator at a fixed seed, restored afterwards. Everything runs in well
under a second with PharmaPy's Levenberg-Marquardt (LM) solver.
"""

import copy

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import curve_fit

from PharmaPy.ParamEstim import Experiment, Measurement, ParameterEstimation
from PharmaPy.StatsModule import StatisticsClass
from test_statistics_bootstrap_state import _assert_same, _state

pytestmark = pytest.mark.unit

TRUE_RATE = 0.3  # [mol/L/s], rate that generates the synthetic data
SEED_RATE = np.array([0.1])  # [mol/L/s], LM seed below every estimate
# Experiment -> callback args (initial product c0 [mol/L],) and kwargs
# (initial temperature [K], adiabatic rise per produced mol/L [K*L/mol]).
CALLBACK = {
    'warm': ((0.2,), {'temp0_k': 300.0, 'rise_k_l_mol': 5.0}),
    'cool': ((0.1,), {'temp0_k': 290.0, 'rise_k_l_mol': 8.0}),
}
# Experiment -> measurement name -> (field, times [s], offsets added to the
# noise-free data [mol/L or K], standard deviations [mol/L or K], units).
# 'warm' has replicate samples at 1 s, a missing c_P observation at 2 s and
# a single temperature sample; 'cool' measures no temperature. Standard
# deviations differ within each c_P measurement (heteroscedastic errors);
# the one at the missing sample is ignored.
MEASUREMENTS = {
    'warm': {
        'c_P': (0, [0.0, 1.0, 1.0, 2.0, 4.0],
                [0.010, -0.020, 0.015, np.nan, -0.030],
                [0.01, 0.02, 0.02, 0.03, 0.05], 'mol/L'),
        'T': (1, [2.0], [0.4], [0.5], 'K'),
    },
    'cool': {
        'c_P': (0, [0.5, 3.0, 5.0], [0.020, -0.010, 0.040],
                [0.02, 0.04, 0.08], 'mol/L'),
    },
}
# Hand-written layout of the compiled data: experiment -> model grid [s]
# (multiset union of the measurement grids) and observation mask, columns
# (c_P, T).
GRIDS = {'warm': [0.0, 1.0, 1.0, 2.0, 4.0], 'cool': [0.5, 3.0, 5.0]}  # [s]
MASKS = {'warm': [[1, 0], [1, 0], [1, 0], [0, 1], [1, 0]],
         'cool': [[1, 0], [1, 0], [1, 0]]}
NUM_OBSERVED = 8  # [-], warm 4 c_P + 1 T, cool 3 c_P
NUM_SAMPLES = 40  # [-], bootstrap datasets
BOOTSTRAP_SEED = 346  # [-], arbitrary fixed seed; any value works
# LM stops within 1e-8 relative of the optimum of these linear problems;
# 1e-6 leaves two decades of margin while staying far below the 1e-2
# relative spread of the bootstrap estimates.
FIT_RTOL = 1e-6  # [-]
# Generated data are differences of O(1e2) values; a few ulps bound the
# roundoff of recovering a drawn residual from them.
DRAW_ATOL = 1e-9  # [-], on standardized residuals of O(1)
# Model values up to 310 K carry roundoff of a few 1e-14 K.
VALUE_ATOL = 1e-10  # [mol/L or K]
# Closed-form and production sums of eight squared residuals differ only by
# accumulated double roundoff (a few 1e-16 relative); 1e-12 is far above it
# and far below the factor 1/sigma**2 (4 to 1e4) separating weighted and raw
# sums here.
ROUNDOFF_RTOL = 1e-12  # [-]
# Minimum |reduced chi-square - 1| for the two covariance forms to differ
# by at least 10 %, five decades above FIT_RTOL.
CHI_SQUARE_SEPARATION = 0.1  # [-]


def model(params, time_s, initial_mol_l, temp0_k=300.0, rise_k_l_mol=5.0):
    """Return product concentration and adiabatic temperature.

    Parameters
    ----------
    params : numpy.ndarray
        Zero-order rate ``k``, shape ``(1,)`` [mol/L/s].
    time_s : numpy.ndarray
        Model grid, shape ``(n_times,)`` [s].
    initial_mol_l : float
        Initial product concentration [mol/L].
    temp0_k : float, optional
        Initial temperature [K].
    rise_k_l_mol : float, optional
        Adiabatic temperature rise per produced concentration [K*L/mol].

    Returns
    -------
    numpy.ndarray
        Shape ``(n_times, 2)``: concentration [mol/L], temperature [K].
    """
    produced = params[0] * time_s  # [mol/L]
    return np.column_stack((initial_mol_l + produced,
                            temp0_k + rise_k_l_mol * produced))


def sensitivity(params, time_s, initial_mol_l, temp0_k=300.0,
                rise_k_l_mol=5.0):
    """Return the state-major rate sensitivities of ``model``.

    Parameters
    ----------
    params, time_s, initial_mol_l, temp0_k, rise_k_l_mol
        As for ``model``.

    Returns
    -------
    numpy.ndarray
        Shape ``(2 * n_times, 1)``: dc/dk [s], then dT/dk [K*L*s/mol].
    """
    return np.concatenate((time_s, rise_k_l_mol * time_s))[:, np.newaxis]


def entries(uncertainty=True):
    """List the observed entries in closed form.

    Parameters
    ----------
    uncertainty : bool, optional
        If False, every standard deviation is 1 in the measurement unit
        (identity weighting).

    Returns
    -------
    list of dict
        One dict per observed entry, in experiment, measurement and sample
        order, with keys ``experiment``, ``column`` (0 for c_P, 1 for T),
        ``row`` (model-grid row), ``offset`` (model value at k = 0
        [mol/L or K]), ``slope`` (d model / dk [s or K*L*s/mol]), ``y``
        (observation [mol/L or K]) and ``sigma`` ([mol/L or K]).
    """
    rows = []
    for experiment, measurements in MEASUREMENTS.items():
        (initial,), kwargs = CALLBACK[experiment]
        grid = GRIDS[experiment]
        for name, (field, times, offsets, sigmas, _) in measurements.items():
            for count, (time, noise, sigma) in enumerate(
                    zip(times, offsets, sigmas)):
                if np.isnan(noise):
                    continue
                if field == 0:
                    offset = initial  # [mol/L]
                    slope = time  # [s]
                else:
                    offset = kwargs['temp0_k']  # [K]
                    slope = kwargs['rise_k_l_mol'] * time  # [K*L*s/mol]
                # Replicates map to successive grid rows with equal time.
                row = grid.index(time) + times[:count].count(time)
                rows.append({'experiment': experiment, 'column': field,
                             'row': row, 'offset': offset, 'slope': slope,
                             'y': offset + slope * TRUE_RATE + noise,
                             'sigma': sigma if uncertainty else 1.0})
    return rows


def wls_rate(rows, observations=None):
    """Return the closed-form weighted least-squares rate.

    Parameters
    ----------
    rows : list of dict
        Output of ``entries``.
    observations : dict, optional
        Experiment -> generated data, shape ``(n_times, 2)``
        [mol/L, K]; replaces each entry's ``y``. The default uses the
        fixture observations.

    Returns
    -------
    float
        ``sum(g (y - y0) / s**2) / sum(g**2 / s**2)`` [mol/L/s].
    """
    numerator = 0.0  # [1/(mol/L/s)], sum of g (y - y0) / s**2
    denominator = 0.0  # [1/(mol/L/s)**2], sum of g**2 / s**2
    for entry in rows:
        # [mol/L or K], observed or generated value of this entry
        y = (entry['y'] if observations is None else
             observations[entry['experiment']][entry['row'],
                                               entry['column']])
        weight = 1.0 / entry['sigma'] ** 2  # [1/(mol/L)**2 or 1/K**2]
        numerator += weight * entry['slope'] * (y - entry['offset'])
        denominator += weight * entry['slope'] ** 2
    return numerator / denominator


def fitted_value(experiment, column, time, rate):
    """Return the closed-form model value of one entry.

    Parameters
    ----------
    experiment : str
        Key of ``CALLBACK``.
    column : int
        0 for concentration, 1 for temperature.
    time : float
        Time [s].
    rate : float
        Rate ``k`` [mol/L/s].

    Returns
    -------
    float
        Concentration [mol/L] or temperature [K].
    """
    (initial,), kwargs = CALLBACK[experiment]
    produced = rate * time  # [mol/L]
    if column == 0:
        return initial + produced
    return kwargs['temp0_k'] + kwargs['rise_k_l_mol'] * produced


def build_experiments(uncertainty=True):
    """Build the fixture as caller arrays and Experiment objects.

    Parameters
    ----------
    uncertainty : bool, optional
        Declare the fixture standard deviations.

    Returns
    -------
    experiments : dict
        Experiment name -> ``Experiment``.
    caller_arrays : list of numpy.ndarray
        The arrays handed to ``Measurement``, for mutation checks.
    """
    experiments, caller_arrays = {}, []
    for experiment, measurements in MEASUREMENTS.items():
        args, kwargs = CALLBACK[experiment]
        objects = {}
        for name, (field, times, offsets, sigmas, units) in (
                measurements.items()):
            time = np.array(times)  # [s]
            values = np.array([
                model([TRUE_RATE], np.array([t]), *args, **kwargs)[0, field]
                + noise for t, noise in zip(times, offsets)])  # [units]
            sigma = np.array(sigmas)  # [units]
            caller_arrays += [time, values, sigma]
            objects[name] = Measurement(
                field, time, values, units=units,
                uncertainty=sigma if uncertainty else None)
        experiments[experiment] = Experiment(objects, args=args,
                                             kwargs=dict(kwargs))
    return experiments, caller_arrays


def fitted_statistics(uncertainty=True):
    """Fit the fixture and wrap it for statistics.

    Parameters
    ----------
    uncertainty : bool, optional
        Declare the fixture standard deviations.

    Returns
    -------
    statistics : StatisticsClass
        Statistics of the LM fit.
    experiments : dict
        The ``Experiment`` objects given to the estimator.
    caller_arrays : list of numpy.ndarray
        The arrays handed to ``Measurement``.
    """
    experiments, caller_arrays = build_experiments(uncertainty)
    estimator = ParameterEstimation(model, SEED_RATE, experiments,
                                    jac_fun=sensitivity,
                                    name_params=['rate_mol_l_s'])
    estimator.optimize_fn(verbose=False)
    return StatisticsClass(estimator), experiments, caller_arrays


def seeded(function, *args):
    """Call ``function`` with NumPy's global generator at a fixed seed.

    Parameters
    ----------
    function : callable
        Function drawing from ``numpy.random``.
    *args
        Positional arguments of ``function``.

    Returns
    -------
    object
        Return value of ``function``; the previous generator state is
        restored afterwards.
    """
    previous_state = np.random.get_state()
    try:
        np.random.seed(BOOTSTRAP_SEED)
        return function(*args)
    finally:
        np.random.set_state(previous_state)


def experiment_snapshot(experiments):
    """Copy every public field of the Experiment and Measurement objects.

    Parameters
    ----------
    experiments : dict
        Experiment name -> ``Experiment``.

    Returns
    -------
    dict
        Nested plain copies: names, callback args and kwargs, and per
        measurement the field, x [s], values, uncertainty and labels.
    """
    return {name: {
        'names': experiment.measurement_names,
        'args': copy.deepcopy(experiment.args),
        'kwargs': copy.deepcopy(dict(experiment.kwargs)),
        'measurements': {
            meas: {'field': m.field, 'x': m.x.copy(),
                   'values': m.values.copy(),
                   'uncertainty': (None if m.uncertainty is None
                                   else m.uncertainty.copy()),
                   'labels': (m.units, m.basis, m.x_units)}
            for meas, m in experiment.measurements.items()}}
        for name, experiment in experiments.items()}


def test_fixture_compiles_to_the_hand_written_layout():
    statistics, _, _ = fitted_statistics()
    estimator = statistics.inst
    assert estimator.experim_names == list(GRIDS)
    for index, name in enumerate(GRIDS):
        np.testing.assert_array_equal(estimator.x_model[index], GRIDS[name])
        np.testing.assert_array_equal(estimator.x_masks[index],
                                      np.array(MASKS[name], dtype=bool))
    assert estimator.num_data_total == NUM_OBSERVED
    assert len(entries()) == NUM_OBSERVED
    assert statistics.dof == NUM_OBSERVED - 1
    np.testing.assert_allclose(statistics.params, [wls_rate(entries())],
                               rtol=FIT_RTOL)


@pytest.mark.parametrize('uncertainty', [True, False],
                         ids=['declared_sigma', 'identity'])
def test_covariance_matches_closed_form_linear_model(uncertainty):
    """Both ``include_mse`` forms equal the linear-model closed forms.

    For residuals ``(y0 + g k - y) / s``, ``J J^T = sum(g**2 / s**2)``.
    ``include_mse=False`` gives ``1 / sum(g**2 / s**2)``, the covariance
    for absolute standard deviations; the default multiplies it by the
    reduced chi-square ``sum(r**2) / (n - p)`` over observed entries only.
    ``scipy.optimize.curve_fit`` provides the same two forms through
    ``absolute_sigma``.
    """
    statistics, _, _ = fitted_statistics(uncertainty)
    estimator = statistics.inst
    rows = entries(uncertainty)
    rate = wls_rate(rows)  # [mol/L/s]
    # [1/(mol/L/s)**2] with declared sigma. With identity weights it sums
    # [s**2] (concentration) and [(K*L*s/mol)**2] (temperature) terms, the
    # dimensional inhomogeneity that declared uncertainties remove.
    information = sum(entry['slope'] ** 2 / entry['sigma'] ** 2
                      for entry in rows)
    chi_square = sum(
        ((entry['offset'] + entry['slope'] * rate - entry['y'])
         / entry['sigma']) ** 2 for entry in rows)  # [-] with sigma
    reduced = chi_square / (NUM_OBSERVED - 1)  # [-] with sigma

    # [(mol/L/s)**2] with declared sigma; identity weights leave the
    # measurement units in it (see ``information``).
    absolute = estimator.get_covariance(include_mse=False)
    np.testing.assert_allclose(absolute, [[1.0 / information]],
                               rtol=FIT_RTOL)
    # [(mol/L/s)**2]; with identity weights a unit-dependent number, since
    # the residuals mix [mol/L] and [K].
    scaled = estimator.get_covariance()
    np.testing.assert_allclose(scaled, [[reduced / information]],
                               rtol=FIT_RTOL)
    assert estimator.covar_params is scaled
    # Non-vacuity: the two forms differ by far more than FIT_RTOL.
    assert abs(reduced - 1.0) > CHI_SQUARE_SEPARATION

    def linear(index, rate_mol_l_s):
        """Evaluate the fixture model at observed entries for ``curve_fit``.

        Parameters
        ----------
        index : numpy.ndarray
            Positions in ``rows``, shape ``(n,)`` [-].
        rate_mol_l_s : float
            Zero-order rate ``k`` [mol/L/s].

        Returns
        -------
        numpy.ndarray
            Shape ``(n,)``: model value of each entry, in its measurement's
            unit (concentration [mol/L] or temperature [K]).
        """
        return np.array([rows[int(i)]['offset']
                         + rows[int(i)]['slope'] * rate_mol_l_s
                         for i in index])

    for absolute_sigma, expected in ((True, absolute), (False, scaled)):
        _, reference = curve_fit(
            linear, np.arange(len(rows)), [entry['y'] for entry in rows],
            p0=SEED_RATE, sigma=[entry['sigma'] for entry in rows],
            absolute_sigma=absolute_sigma)
        np.testing.assert_allclose(expected, reference, rtol=FIT_RTOL)


@pytest.mark.parametrize('fix_initial', [False, True])
def test_bootstrap_resamples_standardized_residuals(fix_initial):
    """Declared standard deviations make the resampled errors exchangeable.

    Every observed generated entry ``i`` must equal ``fit_i - s_i * z``
    with ``z`` one of the standardized residuals ``r / s`` of the same
    experiment and measurement; unobserved entries are NaN. Raw-residual
    resampling, used without declared uncertainties, would place errors of
    precise samples at imprecise ones, which the last assertion rules out.
    """
    statistics, _, _ = fitted_statistics()
    rate = statistics.params[0]  # [mol/L/s]
    # Per experiment, datasets of shape (n_times, 2) [mol/L, K].
    samples = seeded(statistics.get_bootsamples, NUM_SAMPLES, fix_initial)

    raw_mismatch = False
    for index, experiment in enumerate(GRIDS):
        mask = np.array(MASKS[experiment], dtype=bool)
        assert len(samples[index]) == NUM_SAMPLES
        for column in range(mask.shape[1]):
            rows = [entry for entry in entries()
                    if entry['experiment'] == experiment
                    and entry['column'] == column
                    and entry['row'] >= fix_initial]
            fitted = {entry['row']: entry['offset'] + entry['slope'] * rate
                      for entry in rows}  # [mol/L or K]
            raw = np.array([fitted[entry['row']] - entry['y']
                            for entry in rows])  # [mol/L or K]
            pool = raw / np.array([entry['sigma'] for entry in rows])  # [-]
            for sample in samples[index]:
                assert sample.shape == mask.shape
                if fix_initial:
                    # Row 0 keeps its fitted value (NaN if unobserved)
                    # [mol/L or K].
                    expected_first = (fitted_value(experiment, column,
                                                   GRIDS[experiment][0], rate)
                                      if mask[0, column] else np.nan)
                    np.testing.assert_allclose(
                        sample[0, column], expected_first,
                        rtol=0, atol=VALUE_ATOL)
                unobserved = ~mask[fix_initial:, column]
                assert np.all(np.isnan(sample[fix_initial:][unobserved,
                                                            column]))
                for entry in rows:
                    # [mol/L or K], error placed at this entry
                    drawn = fitted[entry['row']] - sample[entry['row'],
                                                          column]
                    # [-], to the nearest standardized residual of the pool
                    z_distance = np.abs(drawn / entry['sigma'] - pool).min()
                    assert z_distance <= DRAW_ATOL
                    if np.abs(drawn - raw).min() > DRAW_ATOL * entry['sigma']:
                        raw_mismatch = True
    assert raw_mismatch


def test_bootstrap_without_uncertainty_keeps_raw_residual_draws():
    """Without declared uncertainties the draws are those of earlier releases.

    The expected datasets re-run the pre-#346 expression,
    ``y - np.random.choice(r[observed], size=(n, n_observed))`` for each
    experiment and measurement column, from the same seed.
    """
    statistics, _, _ = fitted_statistics(uncertainty=False)

    def earlier_bootsamples():
        """Return datasets built by the pre-#346 expression.

        Returns
        -------
        list of list of numpy.ndarray
            One list per experiment of ``NUM_SAMPLES`` arrays, shape
            ``(n_times, 2)`` [mol/L, K], NaN where unobserved.
        """
        samples = []
        for name, residual, fitted in zip(GRIDS, statistics.residuals,
                                          statistics.y_nominal):
            mask = np.array(MASKS[name], dtype=bool)
            columns = []
            for res, y, observed in zip(residual.T, fitted.T, mask.T):
                # [mol/L or K], NaN where unobserved
                generated = np.full((NUM_SAMPLES, len(observed)), np.nan)
                pool = res[observed]  # [mol/L or K]
                if pool.size:
                    generated[:, observed] = y[observed] - np.random.choice(
                        pool, size=(NUM_SAMPLES, pool.size), replace=True)
                columns.append(generated)
            samples.append([np.column_stack(row) for row in zip(*columns)])
        return samples

    expected = seeded(earlier_bootsamples)  # [mol/L, K] columns
    samples = seeded(statistics.get_bootsamples, NUM_SAMPLES)  # [mol/L, K]
    for experiment_expected, experiment_samples in zip(expected, samples):
        assert len(experiment_samples) == NUM_SAMPLES
        for expected_sample, sample in zip(experiment_expected,
                                           experiment_samples):
            np.testing.assert_array_equal(sample, expected_sample)


@pytest.mark.parametrize('uncertainty', [True, False],
                         ids=['declared_sigma', 'identity'])
def test_bootstrap_refits_leave_data_objects_and_fit_unchanged(uncertainty):
    """Repeated bootstrap refits of Experiment input are isolated.

    Each refit must reach the closed-form WLS rate of its generated
    dataset; two seeded runs must agree exactly. Neither the caller's
    arrays, the ``Measurement``/``Experiment`` objects nor the fitted
    estimator (observations, masks, grids, standard deviations, accepted
    parameters, residuals, predictions, solver information, covariance)
    may change.
    """
    statistics, experiments, caller_arrays = fitted_statistics(uncertainty)
    estimator = statistics.inst
    arrays_before = [array.copy() for array in caller_arrays]
    objects_before = experiment_snapshot(experiments)
    estimator_before = _state(estimator)
    accepted = estimator.params_convg.copy()  # [mol/L/s]
    # [-] with declared sigma; with identity weights a sum of [(mol/L)**2]
    # and [K**2] terms.
    objective = estimator.get_objective(accepted)

    samples = seeded(statistics.get_bootsamples, NUM_SAMPLES)
    rows = entries(uncertainty)
    estimates = np.array([
        wls_rate(rows, {name: samples[index][draw]
                        for index, name in enumerate(GRIDS)})
        for draw in range(NUM_SAMPLES)])  # [mol/L/s]
    # Non-vacuity: the generated datasets differ from the observations.
    assert np.all(estimates != accepted[0])

    first = seeded(statistics.bootstrap_params, NUM_SAMPLES)  # [mol/L/s]
    second = seeded(statistics.bootstrap_params, NUM_SAMPLES)  # [mol/L/s]

    np.testing.assert_allclose(first[:, 0], estimates, rtol=FIT_RTOL)
    np.testing.assert_array_equal(second, first)
    for array, before in zip(caller_arrays, arrays_before):
        np.testing.assert_array_equal(array, before)
    _assert_same(objects_before, experiment_snapshot(experiments),
                 'experiments')
    _assert_same(estimator_before, _state(estimator), 'estimator')
    assert estimator.get_objective(accepted) == objective


@pytest.mark.parametrize('weighting', ['declared_sigma', 'identity',
                                       'weight_matrix'])
def test_objective_history_records_weighted_sum_with_uncertainties(
        weighting):
    """``paramest_df['obj_fun']`` is dimensionless with declared sigma.

    With declared uncertainties each recorded value is the weighted sum of
    squares ``sum(((y0 + g k - y) / s)**2)`` [-] at that iterate, twice the
    objective; the raw residuals mix [mol/L] and [K]. Without them the
    legacy history is kept: the unweighted raw sum of squares, also with a
    labelled diagonal ``weight_matrix`` of the declared variances. Expected
    values come from the closed-form entries at each recorded rate.
    """
    experiments, _ = build_experiments(weighting == 'declared_sigma')
    options = {}
    if weighting == 'weight_matrix':
        # [(mol/L)**2], [K**2]: c_P variance of its first warm sample and
        # the T variance, labelled by measurement name.
        variances = {'c_P': MEASUREMENTS['warm']['c_P'][3][0] ** 2,
                     'T': MEASUREMENTS['warm']['T'][3][0] ** 2}
        options['weight_matrix'] = pd.DataFrame(
            np.diag(list(variances.values())), index=list(variances),
            columns=list(variances))
    estimator = ParameterEstimation(model, SEED_RATE, experiments,
                                    jac_fun=sensitivity, **options)
    estimator.optimize_fn(verbose=False)

    rows = entries(weighting == 'declared_sigma')
    history = estimator.paramest_df
    assert len(history) > 1
    for objective_value, rate in zip(history['obj_fun'],
                                     history[estimator.name_params[0]]):
        expected = sum(  # [-] with sigma; mixed squared units otherwise
            ((entry['offset'] + entry['slope'] * rate - entry['y'])
             / entry['sigma']) ** 2 for entry in rows)
        assert objective_value == pytest.approx(expected, rel=ROUNDOFF_RTOL)
        if weighting == 'declared_sigma':
            assert objective_value == pytest.approx(
                2 * estimator.get_objective([rate], set_self=False),
                rel=ROUNDOFF_RTOL)
