"""Bounded synthetic stress campaign through the shared estimator and real solves.

Reuses the independent concentration balances from the handoff qualification.
Two declared noise levels, two reproducible realizations and three initial
parameters check a small, deliberately asymmetric multi-experiment problem.
Independent bounded minimization supplies the noisy-data optimum; recovering
its generating parameter exactly would be an incorrect noisy-data oracle.
"""

from copy import deepcopy
import json

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar

from PharmaPy.Bioreactors import simulate_measurements
from PharmaPy.ParamEstim import Experiment, Measurement, ParameterEstimation
from .test_bioreactor_estimation_handoff import (
    EXAMPLE, HOUR_S, SAMPLE_H, TRUTH_THETA, FD_STEP, CONCENTRATION_ATOL,
    REFERENCE_RTOL, REFERENCE_ATOL, callback_inputs, configuration,
    independent_prediction, independent_rates,
)


pytestmark = pytest.mark.integration
DESIGN = json.loads((EXAMPLE / 'inputs/synthetic_design.json').read_text())
THETA_LIMIT = np.log(2.)  # [-], declared hybrid validity bound
# Five forward-difference steps cover O(h) optimization bias and solver noise;
# the independent optimum uses a tighter scalar minimization tolerance.
FIT_ATOL = 5. * FD_STEP  # [-], qualification budget, not a statistical interval
REFERENCE_XTOL = 1e-8  # [-], 1/50000 of the parameter comparison budget
# Only two deterministic realizations qualify mechanics, not Monte Carlo
# coverage or statistical calibration. The full design retains 20 realizations.
REALIZATIONS = (0, 1)
STARTS = (-THETA_LIMIT / 2., 0., THETA_LIMIT / 2.)  # [-], interior domain points


# Real fixture construction and independent qualification helpers.
def noisy_experiments(noise_fraction, realization):
    """Create independent noisy low/high dilution experiments.

    Parameters
    ----------
    noise_fraction : float
        Relative synthetic standard deviation [-], from the example design.
    realization : int
        Deterministic realization index [-].

    Returns
    -------
    tuple
        Experiment mapping and reference tuples (D [1/h], times [s],
        data [g/L], standard deviations [g/L]); X omits alternating samples.
    """
    rng = np.random.default_rng(np.random.SeedSequence([DESIGN['seed'], realization]))
    experiments, records = {}, []
    for design in DESIGN['experiments']:
        if design['split'] != 'fit':
            continue
        dilution = design['dilution_h_inv']  # [1/h]
        times = SAMPLE_H * HOUR_S  # [s]
        truth = independent_prediction(TRUTH_THETA, times, dilution)  # [g/L]
        std = np.sqrt(DESIGN['absolute_sd_g_l'] ** 2
                      + (noise_fraction * truth) ** 2)  # [g/L], independent error terms
        data = truth + rng.normal(size=truth.shape) * std  # [g/L], no clipping
        model = configuration(True)
        model['recipes']['continuous'][0]['volume_l_h'] = dilution  # [L/h], 1 L volume
        kwargs = callback_inputs(model)
        kwargs['declaration']['fields']['S']['unit'] = 'g/L'
        experiments[design['name']] = Experiment({
            'substrate': Measurement('S', times, data[:, 0],
                                     uncertainty=std[:, 0], units='g/L'),
            'biomass': Measurement('X', times[1::2], data[1::2, 1],
                                   uncertainty=std[1::2, 1], units='g/L')}, kwargs=kwargs)
        records.append((dilution, times, data, std))
    return experiments, records


def reference_objective(theta, records):
    """Evaluate independent weighted least squares with the observed mask.

    Parameters
    ----------
    theta : float
        Log correction [-].
    records : list of tuple
        Reference records described by ``noisy_experiments``.

    Returns
    -------
    float
        Half the sum of squared standardized residuals [-].
    """
    objective = 0.  # [-]
    for dilution, times, data, std in records:
        residual = (independent_prediction(theta, times, dilution) - data) / std  # [-]
        objective += (np.sum(residual[:, 0] ** 2)
                      + np.sum(residual[1::2, 1] ** 2)) / 2.  # [-]
    return objective


def estimator_for(experiments, start=0.):
    """Configure the existing shared estimator with an explicit difference step.

    Parameters
    ----------
    experiments : dict
        Real Experiment objects, concentrations [g/L], uncertainties [g/L].
    start : float, optional
        Initial synthetic correction [-].

    Returns
    -------
    ParameterEstimation
        Native estimator with dimensionless standardized residuals.
    """
    return ParameterEstimation(simulate_measurements, np.array([start]), experiments,
        output_names=('S', 'X'), name_params=['corrections.theta'], dx_finitediff=FD_STEP)


# Public-path regressions, independent balances, and failure qualification.
@pytest.mark.parametrize('noise_fraction', [DESIGN['noise_fractions']['noise_1pct'],
                                           DESIGN['noise_fractions']['noise_5pct']])
@pytest.mark.parametrize('realization', REALIZATIONS)
def test_noisy_multistart_fit_matches_independent_optimum_and_heldout(noise_fraction, realization):
    experiments, records = noisy_experiments(noise_fraction, realization)
    original_kwargs = {name: deepcopy(dict(experiment.kwargs))
                       for name, experiment in experiments.items()}
    original_data = {name: {key: measurement.values.copy()
                           for key, measurement in experiment.measurements.items()}
                     for name, experiment in experiments.items()}
    reference = minimize_scalar(reference_objective, args=(records,),
        bounds=(-THETA_LIMIT, THETA_LIMIT), method='bounded',
        options={'xatol': REFERENCE_XTOL})
    assert reference.success
    assert abs(reference.x) < THETA_LIMIT
    for start in STARTS:
        estimator = estimator_for(experiments, start)
        fitted, covariance, info = estimator.optimize_fn(verbose=False)
        np.testing.assert_allclose(fitted, [reference.x], rtol=0., atol=FIT_ATOL)
        assert info['stop_criterion'] != 'Maximum iterations exceeded'
        assert np.isfinite(covariance).all() and covariance[0, 0] > 0.
        # The objective oracle uses separate equations, integrator and packing.
        np.testing.assert_allclose(estimator.get_objective(fitted),
                                   reference_objective(float(fitted[0]), records),
                                   rtol=1e-6, atol=1e-6)  # [-], native error / assay scale
        heldout = configuration(True)
        dilution = next(item['dilution_h_inv'] for item in DESIGN['experiments']
                        if item['split'] == 'validation')  # [1/h]
        heldout['recipes']['continuous'][0]['volume_l_h'] = dilution  # [L/h]
        times = SAMPLE_H * HOUR_S  # [s]
        prediction = simulate_measurements(fitted, times, **callback_inputs(heldout))  # [g/L]
        expected = independent_prediction(reference.x, times, dilution)  # [g/L]
        # Independent local sensitivity propagates the qualified parameter error.
        sensitivity = (independent_prediction(reference.x + FIT_ATOL, times, dilution)
                       - independent_prediction(reference.x - FIT_ATOL, times, dilution)) / 2.  # [g/L]
        assert np.all(np.abs(prediction - expected)
                      <= np.abs(sensitivity) + CONCENTRATION_ATOL)
    for name, experiment in experiments.items():
        assert experiment.kwargs == original_kwargs[name]
        for key, measurement in experiment.measurements.items():
            np.testing.assert_array_equal(measurement.values, original_data[name][key])


def test_budget_exhaustion_and_domain_error_leave_reusable_estimator():
    experiments, records = noisy_experiments(.01, 0)  # [-], declared 1% noise
    estimator = estimator_for(experiments)
    with pytest.raises(ValueError, match='outside validity domain'):
        estimator.get_objective(np.array([2. * THETA_LIMIT]))
    expected = reference_objective(0., records)  # [-]
    np.testing.assert_allclose(estimator.get_objective(np.array([0.])), expected,
                               rtol=1e-6, atol=1e-6)
    # One allowed evaluation deliberately exhausts the budget. Zero thresholds
    # disable alternative stopping tests; they are not numerical tolerances.
    fitted, _, info = estimator.optimize_fn(verbose=False,
        optim_options={'max_fun_eval': 1, 'eps_1': 0., 'eps_2': 0., 'tol_fun': 0.})
    assert info['stop_criterion'] == 'Maximum iterations exceeded'
    assert info['num_fun_eval'] == 1
    np.testing.assert_array_equal(info['x'], fitted)
    np.testing.assert_allclose(info['fun'], estimator.get_objective(fitted, out_array=True),
                               rtol=0., atol=0.)
    assert np.linalg.norm(info['fun']) > 0.
    recovered, _, recovered_info = estimator.optimize_fn(verbose=False)
    reference = minimize_scalar(reference_objective, args=(records,),
        bounds=(-THETA_LIMIT, THETA_LIMIT), method='bounded',
        options={'xatol': REFERENCE_XTOL})
    assert reference.success
    np.testing.assert_allclose(recovered, [reference.x], rtol=0., atol=FIT_ATOL)
    assert recovered_info['stop_criterion'] != 'Maximum iterations exceeded'
    assert len(estimator.y_model) == len(experiments)


@pytest.mark.parametrize('theta', [0., TRUTH_THETA])
def test_step_refinement_and_tighter_integration_preserve_sensitivity(theta):
    times = SAMPLE_H * HOUR_S  # [s]
    kwargs = callback_inputs(configuration(True))
    step = FD_STEP / 2.  # [-], central independent reference
    reference = (independent_prediction(theta + step, times)
                 - independent_prediction(theta - step, times)) / (2. * step)  # [g/L per theta]
    baseline = simulate_measurements([theta], times, **kwargs)  # [g/L]
    for difference_step in (10. * FD_STEP, FD_STEP, FD_STEP / 10.):
        derivative = (simulate_measurements([theta + difference_step], times, **kwargs)
                      - baseline) / difference_step  # [g/L per theta]
        # Forward O(h) truncation plus 1e-7 concentration budget divided by h;
        # native tighter solves must also agree in absolute concentrations.
        np.testing.assert_allclose(derivative, reference,
                                   rtol=10. * difference_step,
                                   atol=CONCENTRATION_ATOL / difference_step)
    refined = deepcopy(kwargs)
    refined['case']['numerical']['relative_tolerance'] /= 10.  # [-], decade refinement
    refined['case']['numerical']['absolute_tolerance'] /= 10.  # [native inventory units]
    refined['case']['numerical']['maximum_step']['value'] /= 2.  # [h], step-halving
    np.testing.assert_allclose(simulate_measurements([theta], times, **refined),
                               baseline, rtol=0., atol=CONCENTRATION_ATOL)


def test_initial_only_data_report_singular_information_instead_of_precision():
    kwargs = callback_inputs(configuration(True))
    experiment = Experiment({'S': Measurement('S', [0., 0.], [1., 1.],
                                               uncertainty=[.001, .001], units='g/L')},
                            kwargs=kwargs)
    estimator = estimator_for({'unidentifiable': experiment})
    estimator.get_objective(np.array([0.]))
    jacobian = estimator.get_gradient(np.array([0.]), out_array=True)  # [- per theta]
    np.testing.assert_array_equal(jacobian, np.zeros_like(jacobian))
    # Initial conditions have no theta dependence. The existing API reports
    # singular information; it must not invent a finite confidence interval.
    with pytest.raises(np.linalg.LinAlgError, match='Singular matrix'):
        estimator.optimize_fn(verbose=False)
    np.testing.assert_array_equal(experiment.measurements['S'].values, [1., 1.])


@pytest.mark.parametrize('regime', ['depletion', 'washout'])
def test_limiting_regimes_match_independent_balances(regime):
    model = configuration(True)
    initial = np.array([4.5, 1.])  # [g/L], benchmark initial biomass, substrate
    dilution = .1  # [1/h], nominal continuous case
    if regime == 'depletion':
        model['operation']['mode'] = 'batch'
        model['recipes']['continuous'] = []
        initial[1] = .01  # [g/L], 1% of Ks probes positive substrate limitation
        model['state']['species']['amounts']['substrate'] = initial[1]  # [g], 1 L
        dilution = 0.  # [1/h], no batch feed
    else:
        # D=2*mu_max*maximum correction exceeds every admissible growth rate.
        dilution = 2. * .2 * np.exp(THETA_LIMIT)  # [1/h]
        model['recipes']['continuous'][0]['volume_l_h'] = dilution  # [L/h]
    times = SAMPLE_H * HOUR_S  # [s]
    reference = solve_ivp(independent_rates, (0., SAMPLE_H[-1]), initial,
        args=(TRUTH_THETA, dilution), t_eval=SAMPLE_H, method='DOP853',
        rtol=REFERENCE_RTOL, atol=REFERENCE_ATOL)
    assert reference.success
    actual = simulate_measurements([TRUTH_THETA], times, **callback_inputs(model))  # [g/L]
    np.testing.assert_allclose(actual, reference.y[::-1].T, rtol=0., atol=CONCENTRATION_ATOL)
    assert np.all(actual >= -CONCENTRATION_ATOL)
    if regime == 'washout':
        upper_biomass = initial[0] * np.exp(-dilution / 2. * SAMPLE_H)  # [g/L]
        assert np.all(actual[:, 1] <= upper_biomass + CONCENTRATION_ATOL)
        np.testing.assert_allclose(actual[:, 0] + actual[:, 1] / .5,
                                   10., rtol=0., atol=CONCENTRATION_ATOL)
    else:
        np.testing.assert_allclose(actual[:, 0] + actual[:, 1] / .5,
                                   initial[1] + initial[0] / .5,
                                   rtol=0., atol=CONCENTRATION_ATOL)
        assert actual[-1, 0] < initial[1]


def test_short_observation_window_has_less_information_than_transient_window():
    information = []  # [1/theta**2], independent-error Fisher information
    std = .001  # [g/L], absolute noise floor in the committed synthetic design
    for scale in (1. / HOUR_S, 1.):  # [-], compress hours to seconds or retain hours
        times = SAMPLE_H * HOUR_S * scale  # [s]
        values = independent_prediction(TRUTH_THETA, times)  # [g/L]
        kwargs = callback_inputs(configuration(True))
        kwargs['declaration']['fields']['S']['unit'] = 'g/L'
        experiments = {'window': Experiment({
            'S': Measurement('S', times, values[:, 0], uncertainty=std, units='g/L'),
            'X': Measurement('X', times, values[:, 1], uncertainty=std, units='g/L')}, kwargs=kwargs)}
        estimator = estimator_for(experiments)
        estimator.get_objective(np.array([TRUTH_THETA]))
        jacobian = estimator.get_gradient(np.array([TRUTH_THETA]), out_array=True)  # [1/theta]
        reference = (independent_prediction(TRUTH_THETA + FD_STEP / 2., times)
                    - independent_prediction(TRUTH_THETA - FD_STEP / 2., times)) / (FD_STEP * std)  # [1/theta]
        np.testing.assert_allclose(jacobian, reference.T.reshape(1, -1),
                                   rtol=1e-3, atol=1e-3)  # [-], O(h) plus solver noise / sigma
        information.append(float(np.sum(jacobian ** 2)))
    assert 0. < information[0] < information[1]


def test_reconciled_callback_rebuilds_each_trial_without_target_history_leaks():
    from .test_bioreactor_known_answer import load

    case, definition, thermo = load('consistent')
    original_case, original_definition = deepcopy(case), deepcopy(definition)
    day_s = 86400.  # [s/day], exact conversion
    times_day = np.array([0., .1, .2])  # [day], before first feed change
    fields = {'nutrient': {'species': 'nutrient', 'quantity': 'concentration',
                           'unit': 'mmol/L', 'event_side': 'post',
                           'time_key': 't', 'value_key': 'y'}}
    kwargs = dict(case=case, definition=definition, thermo_path=thermo,
        parameter_names=('rates.uptake_target', 'rates.product_target'),
        declaration={'schema_version': 1, 'time_unit': 's', 'fields': fields})
    # Equal-weight projection onto uptake=conversion=secretion gives the mean
    # of the two targets, within [0,10] mmol/(million cell day).
    for targets in ([2., 4.], [4., 2.], [2., 2.], [2., 4.]):
        rate = sum(targets) / 2.  # [mmol/(million cell day)]
        expected = (10. + (.1 * 2. - rate) * times_day) / (1. + .1 * times_day)  # [mmol/L]
        actual = simulate_measurements(targets, times_day * day_s, **kwargs)  # [mmol/L]
        np.testing.assert_allclose(actual[:, 0], expected, rtol=0., atol=1e-8)  # [mmol/L], linear inventory benchmark
    assert case == original_case
    assert definition == original_definition
