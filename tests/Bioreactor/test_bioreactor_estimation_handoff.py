"""Native observation/estimation handoff with real SciPy reactor solves.

The Doyle Monod model supplies a synthetic continuous-culture case. Independent
concentration balances check trajectories and a substrate-equivalent invariant;
no experimental calibration or complete elemental closure is claimed.
"""

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from PharmaPy.Bioreactors import simulate_measurements
from PharmaPy.Bioreactors.input_format import normalize_forward_inputs, split_forward_inputs
from .test_bioreactor_thermal import (thermal_case, CP, INITIAL_TEMPERATURE,
                                      RATE_CONSTANT, REACTION_ENTHALPY)
from PharmaPy.ParamEstim import Experiment, Measurement, ParameterEstimation


pytestmark = pytest.mark.integration
EXAMPLE = Path(__file__).parent / 'fixtures/wp32_kinetics'
HOUR_S = 3600.  # [s/h], exact conversion
HORIZON_H = 8.  # [h], initial transient qualification rather than a steady fit
SAMPLE_H = np.array([0., .5, 1., 2., 4., 8.])  # [h], early and late transient samples
TRUTH_THETA = np.log(1.2)  # [-], synthetic positive correction, multiplier <= 1.2
FD_STEP = 1e-4  # [-], forward parameter step; refined explicitly below
REFERENCE_RTOL = 1e-11  # [-], reference integration tighter than native 1e-9
REFERENCE_ATOL = 1e-12  # [g/L], independent concentration-balance integration
CONCENTRATION_ATOL = 1e-7  # [g/L], 100 times below textbook benchmark's 1e-5 budget


# Real fixture construction and independent qualification helpers.
def configuration(hybrid=False):
    """Read an independent input copy for a short native solve.

    Parameters
    ----------
    hybrid : bool, optional
        Select the separately configured synthetic hybrid model.

    Returns
    -------
    dict
        Combined input, with runtime [h] shortened for transient tests.
    """
    name = 'hybrid.json' if hybrid else 'mechanism.json'
    model = json.loads((EXAMPLE / 'inputs' / name).read_text())
    model['operation']['runtime']['value'] = HORIZON_H
    return model


def declaration():
    """Declare asymmetric output order and units for observable concentrations.

    Returns
    -------
    dict
        Substrate [mmol/L] then biomass [g/L], time [s]. Formal species weights
        make the substrate numerical value equal to [g/L] in this benchmark.
    """
    return {'schema_version': 1, 'time_unit': 's', 'fields': {
        'S': {'species': 'substrate', 'quantity': 'concentration', 'unit': 'mmol/L',
              'event_side': 'post', 'time_key': 't', 'value_key': 'y'},
        'X': {'species': 'biomass', 'quantity': 'concentration', 'unit': 'g/L',
              'event_side': 'post', 'time_key': 't', 'value_key': 'y'}}}


def callback_inputs(model, names=('corrections.theta',)):
    """Bind existing construction/observation inputs to the shared callback.

    Parameters
    ----------
    model : dict
        Combined model with parameters in their input-declared units.
    names : tuple of str, optional
        Selected scalar parameter paths, in callback-vector order.

    Returns
    -------
    dict
        Callback keyword arguments; no observation or fitting state is added.
    """
    return dict(case=model, parameter_names=names, declaration=declaration(),
                thermo_path=EXAMPLE / 'inputs/thermo.json')


def independent_rates(time_h, state, theta, dilution):
    """Evaluate the synthetic hybrid concentration balances independently.

    Parameters
    ----------
    time_h : float
        Time [h], unused for constant feed.
    state : ndarray
        Biomass and substrate [g/L].
    theta : float
        Correction [-].
    dilution : float
        Dilution [1/h].

    Returns
    -------
    ndarray
        Biomass/substrate derivatives [g/(L h)].

    Notes
    -----
    Exercise 4.15 supplies mu_max=0.2/h, Ks=1 g/L, Y=0.5 g/g and
    feed=10 g/L. The feature scale is Ks; theta is a synthetic assumption.
    """
    biomass, substrate = state  # [g/L]
    growth = .2 * substrate / (1. + substrate)  # [1/h], textbook Monod model
    growth *= np.exp(theta * substrate / (1. + substrate))  # [1/h], synthetic correction
    return np.array([(growth - dilution) * biomass,
                     dilution * (10. - substrate) - growth * biomass / .5])


def independent_prediction(theta, times_s, dilution=.1):
    """Integrate independently written Doyle concentration balances.

    Parameters
    ----------
    theta : float
        Synthetic log correction [-]; zero gives the textbook model.
    times_s : ndarray
        Non-decreasing, unique requested times [s].
    dilution : float, optional
        Sterile-feed dilution [1/h], textbook nominal value 0.1/h.

    Returns
    -------
    ndarray
        Substrate then biomass [g/L], shape (n_times, 2).
    """
    solution = solve_ivp(independent_rates, (0., HORIZON_H), [4.5, 1.],
                         args=(theta, dilution), t_eval=times_s / HOUR_S,
                         method='DOP853', rtol=REFERENCE_RTOL, atol=REFERENCE_ATOL)
    assert solution.success, solution.message
    return solution.y[::-1].T


def coefficient_configuration() -> dict:
    """Place the correction in an existing coefficient-list input slot.

    Returns
    -------
    dict
        Fresh native case with a dimensionless correction [-] in list entry 0.
    """
    model = configuration(True)
    model['parameters']['coefficients']['response'] = [0.]  # [-], neutral correction
    model['parameters']['corrections'] = {}
    next(rule for rule in model['model']['kinetics']['rule_graph']
         if rule['identifier'] == 'theta')['expression'] = {'parameter': 'coefficients.response.0'}
    next(rule for rule in model['model']['kinetics']['rule_graph']
         if rule['identifier'] == 'mu')['expression']['correction']['parameters'][
             'theta'] = {'parameter': 'coefficients.response.0'}
    return model


# Public-path regressions, independent balances, and failure qualification.
def test_callback_preserves_requested_rows_order_units_and_input_isolation():
    model = configuration(hybrid=True)
    original = deepcopy(model)
    times_s = np.array([8., 0.5, 0.5, 2.]) * HOUR_S  # [s], unsorted with replicate
    first = simulate_measurements([TRUTH_THETA], times_s, **callback_inputs(model))  # [mmol/L, g/L]
    other = simulate_measurements([0.], times_s, **callback_inputs(model))  # [mmol/L, g/L]
    repeated = simulate_measurements([TRUTH_THETA], times_s, **callback_inputs(model))  # [mmol/L, g/L]
    unique, inverse = np.unique(times_s, return_inverse=True)  # [s], bookkeeping row indices
    expected = independent_prediction(TRUTH_THETA, unique)[inverse]  # [g/L]
    np.testing.assert_allclose(first, expected, rtol=0., atol=CONCENTRATION_ATOL)
    np.testing.assert_array_equal(first, repeated)
    assert np.max(np.abs(first - other)) > CONCENTRATION_ATOL
    assert model == original
    single = simulate_measurements([TRUTH_THETA], times_s[:1], **callback_inputs(model))  # [mmol/L, g/L]
    np.testing.assert_array_equal(single, first[:1])
    one_field_kwargs = callback_inputs(model)
    one_field_kwargs['declaration']['fields'].pop('X')
    one_field = simulate_measurements([TRUTH_THETA], times_s[:1], **one_field_kwargs)  # [mmol/L]
    np.testing.assert_array_equal(one_field, first[:1, :1])


def test_neutral_hybrid_matches_mechanistic_model_and_declared_yield_balance():
    times_s = SAMPLE_H * HOUR_S  # [s]
    hybrid = simulate_measurements([0.], times_s, **callback_inputs(configuration(True)))  # [mmol/L, g/L]
    baseline = simulate_measurements([.2], times_s,
        **callback_inputs(configuration(), ('rates.mu_max',)))  # [mmol/L, g/L]
    np.testing.assert_allclose(hybrid, baseline, rtol=0., atol=CONCENTRATION_ATOL)
    corrected = simulate_measurements([TRUTH_THETA], times_s,
                                      **callback_inputs(configuration(True)))  # [mmol/L, g/L]
    # Sterile nominal feed: Z0=1+4.5/0.5=10, so Z stays at feed concentration.
    np.testing.assert_allclose(corrected[:, 0] + corrected[:, 1] / .5,
                               10., rtol=0., atol=CONCENTRATION_ATOL)


@pytest.mark.parametrize('theta', [-np.log(2.), np.log(2.)])
def test_declared_correction_range_matches_independent_balances(theta):
    times_s = SAMPLE_H * HOUR_S  # [s]
    observed = simulate_measurements([theta], times_s,
        **callback_inputs(configuration(True)))  # [mmol/L, g/L]
    np.testing.assert_allclose(observed, independent_prediction(theta, times_s),
                               rtol=0., atol=CONCENTRATION_ATOL)
    assert np.all(observed >= 0.)


def test_explicit_graph_conversions_preserve_mass_basis_with_other_formal_weights(tmp_path):
    model = configuration(True)
    times_s = SAMPLE_H * HOUR_S  # [s]
    expected = independent_prediction(TRUTH_THETA, times_s)  # [g/L, g/L]
    thermo = json.loads((EXAMPLE / 'inputs/thermo.json').read_text())
    # Unequal formal weights expose silent concentration-basis equivalence.
    thermo['substrate']['mw'] /= 2.  # [g/mol], halve the formal nutrient weight
    thermo['biomass']['mw'] *= 2.  # [g/mol], double the formal biomass weight
    for species in ('substrate', 'biomass'):
        model['model']['kinetics']['constants'][species + '_g_per_mmol'] = (
            thermo[species]['mw'] / 1000.)  # [g/mmol], exact mol-to-mmol conversion
    thermo_path = tmp_path / 'thermo.json'
    thermo_path.write_text(json.dumps(thermo))
    kwargs = callback_inputs(model)
    kwargs['thermo_path'] = thermo_path
    observed = simulate_measurements([TRUTH_THETA], times_s, **kwargs)  # [mmol/L, g/L]
    expected[:, 0] *= 2.  # [mmol/L], twice as many formal substrate moles per gram
    np.testing.assert_allclose(observed, expected, rtol=0., atol=CONCENTRATION_ATOL)


def test_legacy_inputs_and_reordered_parameter_and_field_maps_keep_values():
    model = configuration(True)
    times_s = SAMPLE_H * HOUR_S  # [s]
    expected = simulate_measurements([TRUTH_THETA, .2], times_s,
        **callback_inputs(model, ('corrections.theta', 'rates.mu_max')))  # [mmol/L, g/L]
    case, definition = split_forward_inputs(model)
    kwargs = callback_inputs(case, ('rates.mu_max', 'corrections.theta'))
    kwargs['definition'] = definition
    kwargs['declaration']['fields'] = dict(reversed(list(kwargs['declaration']['fields'].items())))
    observed = simulate_measurements([.2, TRUTH_THETA], times_s, **kwargs)  # [g/L, mmol/L]
    np.testing.assert_array_equal(observed, expected[:, ::-1])


def test_coefficient_list_and_flattened_legacy_names_use_existing_input_conventions():
    """Rebind both the correction and its domain-reporting rule to a list slot."""
    model = configuration(True)
    times_s = SAMPLE_H * HOUR_S  # [s]
    expected = simulate_measurements([TRUTH_THETA], times_s, **callback_inputs(model))  # [mmol/L, g/L]
    model = coefficient_configuration()
    observed = simulate_measurements([TRUTH_THETA], times_s,
        **callback_inputs(model, ('coefficients.response.0',)))  # [mmol/L, g/L]
    np.testing.assert_array_equal(observed, expected)
    case, definition = normalize_forward_inputs(model)
    kwargs = callback_inputs(case, ('coefficients.response.0',))
    kwargs['definition'] = definition
    np.testing.assert_array_equal(simulate_measurements([TRUTH_THETA], times_s, **kwargs), expected)


def test_equivalent_list_paths_cannot_bind_one_scalar_twice():
    """Reject distinct text names that would silently overwrite one parameter."""
    model = coefficient_configuration()
    original = deepcopy(model)
    with pytest.raises(ValueError, match='distinct scalar'):
        simulate_measurements([0., TRUTH_THETA], SAMPLE_H * HOUR_S,
            **callback_inputs(model, ('coefficients.response.0', 'coefficients.response.00')))
    assert model == original


def test_single_equivalent_list_path_preserves_existing_binding():
    """Keep an individually valid leading-zero list index compatible."""
    model = coefficient_configuration()
    times_s = SAMPLE_H * HOUR_S  # [s]
    expected = simulate_measurements([TRUTH_THETA], times_s,
        **callback_inputs(model, ('coefficients.response.0',)))  # [mmol/L, g/L]
    actual = simulate_measurements([TRUTH_THETA], times_s,
        **callback_inputs(model, ('coefficients.response.00',)))  # [mmol/L, g/L]
    np.testing.assert_array_equal(actual, expected)


def test_pathway_callback_keeps_feed_event_sides_and_physical_units():
    folder = Path(__file__).parents[2] / 'examples/bioreactors/generic_fed_batch/inputs'
    model = json.loads((folder / 'mechanism.json').read_text())
    fields = {side: {'species': 'nutrient', 'quantity': 'concentration',
                    'unit': 'mmol/L' if side == 'pre' else 'g/L',
                    'event_side': side, 'time_key': 't', 'value_key': 'y'}
              for side in ('post', 'pre')}
    kwargs = dict(case=model, parameter_names=('rates.uptake.nutrient',),
                  declaration={'schema_version': 1, 'time_unit': 'h', 'fields': fields},
                  thermo_path=folder / 'thermo.json')
    prediction = simulate_measurements([0.], [3. * HOUR_S], **kwargs)  # [g/L, mmol/L]
    # With zero uptake, the 3 h feed adds 5 mmol nutrient and 0.1 kg carrier.
    # Nutrient MW=100 g/mol adds 0.0005 kg; rho=1 kg/L fixes final liquid volume.
    expected_post_g_l = 1.5 / 1.1005  # [g/L], 15 mmol * 0.1 g/mmol / liquid volume
    np.testing.assert_allclose(prediction, [[expected_post_g_l, 10.]], rtol=0., atol=CONCENTRATION_ATOL)


def test_shared_estimator_fits_real_native_trajectory_and_zero_seed_derivative():
    times_s = SAMPLE_H * HOUR_S  # [s]
    expected = independent_prediction(TRUTH_THETA, times_s)  # [g/L]
    fit_kwargs = callback_inputs(configuration(True))
    # Common mass-concentration units give a coherent noiseless objective.
    # Unit scales of 1 g/L are numerical conventions, not measured assay errors.
    fit_kwargs['declaration']['fields']['S']['unit'] = 'g/L'
    experiments = {'synthetic': Experiment({
        'substrate': Measurement('S', times_s, expected[:, 0], units='g/L'),
        'biomass': Measurement('X', times_s[1::2], expected[1::2, 1], units='g/L')},
        kwargs=fit_kwargs)}
    estimator = ParameterEstimation(simulate_measurements, np.array([0.]), experiments,
        output_names=('S', 'X'), name_params=['corrections.theta'], dx_finitediff=FD_STEP)
    estimator.get_objective(np.array([0.]))
    jacobian = estimator.get_gradient(np.array([0.]), out_array=True)  # [(g/L) / theta]
    assert np.isfinite(jacobian).all() and np.linalg.norm(jacobian) > 0.
    derivative = (independent_prediction(FD_STEP / 2., times_s)
                  - independent_prediction(-FD_STEP / 2., times_s)) / FD_STEP  # [g/L per theta]
    # Shared residual layout is field-major; unobserved X rows carry zero information.
    derivative[::2, 1] = 0.
    expected_jacobian = derivative.T.reshape(1, -1)  # [(g/L) / theta]
    # Forward-step truncation is O(1e-4); the 1e-4 absolute allowance also
    # covers native concentration roundoff amplified by 1/FD_STEP.
    np.testing.assert_allclose(jacobian, expected_jacobian, rtol=1e-3, atol=1e-4)
    estimator.dx_fd = FD_STEP / 2.  # [-], public configured finite-difference step
    refined = estimator.get_gradient(np.array([0.]), out_array=True)  # [(g/L) / theta]
    np.testing.assert_allclose(refined, expected_jacobian, rtol=1e-3, atol=1e-4)
    fitted, _, info = estimator.optimize_fn(verbose=False)  # theta [-], covariance [-], mixed solver metadata
    # 1e-4 FD and native 1e-9 integration qualify recovery to 1e-5 in theta.
    np.testing.assert_allclose(fitted, [TRUTH_THETA], rtol=0., atol=1e-5)
    assert np.max(np.abs(info['fun'])) < 1e-5  # [field concentration unit]
    np.testing.assert_array_equal(experiments['synthetic'].measurements['substrate'].values,
                                  expected[:, 0])
    heldout = configuration(True)
    heldout['recipes']['continuous'][0]['volume_l_h'] = .12  # [L/h], existing synthetic holdout D=0.12/h at 1 L
    heldout_kwargs = callback_inputs(heldout)
    heldout_kwargs['declaration']['fields']['S']['unit'] = 'g/L'
    prediction = simulate_measurements(fitted, times_s, **heldout_kwargs)  # [g/L, g/L]
    np.testing.assert_allclose(prediction, independent_prediction(TRUTH_THETA, times_s, .12),
                               rtol=0., atol=1e-5)  # [g/L], parameter-recovery propagation budget


@pytest.mark.parametrize('values,names,message', [
    ([0., 0.], ('corrections.theta',), 'parameter vector'),
    ([float('nan')], ('corrections.theta',), 'finite'),
    ([0.], ('corrections.missing',), 'unknown parameter'),
    ([0., 0.], ('corrections.theta', 'corrections.theta'), 'unique'),
])
def test_invalid_parameter_bindings_fail_before_simulation(values, names, message):
    with pytest.raises(ValueError, match=message):
        simulate_measurements(values, [0.], **callback_inputs(configuration(True), names))


@pytest.mark.parametrize('times', [[], [[0.]], [float('nan')], [-1.], [9. * HOUR_S]])
def test_invalid_requested_times_are_actionable(times):
    with pytest.raises(ValueError, match='time|times'):
        simulate_measurements([0.], times, **callback_inputs(configuration(True)))


def test_declared_hybrid_domain_failure_does_not_poison_next_evaluation():
    kwargs = callback_inputs(configuration(True))
    with pytest.raises(ValueError, match='outside validity domain'):
        simulate_measurements([2.], [0.], **kwargs)
    prediction = simulate_measurements([0.], [0.], **kwargs)  # [mmol/L, g/L]
    np.testing.assert_allclose(prediction, [[1., 4.5]], rtol=0., atol=CONCENTRATION_ATOL)


@pytest.mark.parametrize('empty_species', ['biomass', 'substrate'])
def test_hybrid_zero_inventory_has_no_spontaneous_growth(empty_species):
    model = configuration(True)
    model['operation']['mode'] = 'batch'
    model['recipes']['continuous'] = []
    model['state']['species']['amounts'][empty_species] = 0.  # [g], empty biological pool
    prediction = simulate_measurements([TRUTH_THETA], SAMPLE_H * HOUR_S,
                                       **callback_inputs(model))  # [mmol/L, g/L]
    initial = np.array([model['state']['species']['amounts']['substrate'],
                        model['state']['species']['amounts']['biomass']])  # [g/L], 1 L working volume
    np.testing.assert_allclose(prediction, np.broadcast_to(initial, prediction.shape),
                               rtol=0., atol=CONCENTRATION_ATOL)


def test_temperature_measurement_callback_retains_field_time_order_and_trials(thermal_case):
    document, path = thermal_case
    document["operation"]["thermal"]["heat_generation_w"]["args"][0] = {"parameter": "coefficients.enthalpy"}
    document["parameters"]["coefficients"]["enthalpy"] = REACTION_ENTHALPY  # [J/kg substrate]
    original = deepcopy(document)
    declaration = {"schema_version": 1, "time_unit": "s", "fields": {
        "temperature": {"species": None, "quantity": "temperature", "unit": "K", "event_side": "post", "time_key": "t", "value_key": "v"},
        "substrate": {"species": "substrate", "quantity": "concentration", "unit": "g/L", "event_side": "post", "time_key": "t", "value_key": "v"}}}
    times = np.array([100.0, 0.0, 50.0, 50.0])  # [s]; unsorted and repeated, with unequal row/column dimensions
    for enthalpy in (REACTION_ENTHALPY, 0.0, REACTION_ENTHALPY):  # [J/kg]; changed trial then repeat
        actual = simulate_measurements([enthalpy], times, case=document,
            parameter_names=["coefficients.enthalpy"], declaration=declaration, thermo_path=path)
        substrate = 10.0 * np.exp(-RATE_CONSTANT * times)  # [g/L]; 10 g initially in 1 L
        temperature = INITIAL_TEMPERATURE - enthalpy * (0.01 - substrate / 1000.0) / CP  # [K]
        assert actual.shape == (4, 2)
        # Separate absolute budgets prevent a 300 K offset from masking
        # temperature error or mixing concentration and temperature scales.
        np.testing.assert_allclose(actual[:, 0], temperature, rtol=0.0, atol=1e-6)  # [K]
        np.testing.assert_allclose(actual[:, 1], substrate, rtol=0.0, atol=1e-6)  # [g/L]
    assert document == original
