"""Experiment identity regressions using real core estimation paths.

https://github.com/PharmaPy-org/PharmaPy/issues/238
"""

import numpy as np
import pytest

from PharmaPy.ParamEstim import ParameterEstimation, MultipleCurveResolution

pytestmark = pytest.mark.unit
ROUNDING_ATOL = 64 * np.finfo(float).eps  # [-], exact-algebra roundoff budget


def accumulating_concentration(parameters, time_s, initial_mol_l, gain=1.0):
    """Evaluate a constant-rate accumulation verification model.

    Parameters
    ----------
    parameters : ndarray
        One concentration-production rate [mol/L/s], shape (1,).
    time_s : ndarray
        Measurement times [s], shape (n_times,).
    initial_mol_l : float
        Initial concentration [mol/L].
    gain : float, optional
        Known experiment-specific production multiplier [-].

    Returns
    -------
    ndarray
        Concentration [mol/L], shape (n_times,).
    """
    return initial_mol_l + gain * parameters[0] * time_s


@pytest.mark.parametrize("reordered_input", ["y_data", "args_fun", "kwargs_fun"])
def test_issue238_experiment_identity_is_invariant_to_mapping_order(reordered_input):
    """Reorder one keyed input while preserving both physical experiments."""
    rate = np.array([2.0])  # [mol/L/s], exact synthetic rate for both experiments
    times = {"dilute": np.array([0.0, 1.0, 3.0]),
             "concentrated": np.array([0.0, 2.0, 3.0])}  # [s], distinct schedules
    # Independent arithmetic: C0 + gain * rate * time at the schedules above.
    observations = {"dilute": np.array([1.0, 3.0, 7.0]),
                    "concentrated": np.array([10.0, 18.0, 22.0])}  # [mol/L]
    initial_values = {"dilute": (1.0,), "concentrated": (10.0,)}  # [mol/L]
    gains = {"dilute": {"gain": 1.0},
             "concentrated": {"gain": 2.0}}  # [-], distinguish callback kwargs
    inputs = {"y_data": observations, "args_fun": initial_values,
              "kwargs_fun": gains}
    reference = ParameterEstimation(accumulating_concentration, rate, times,
                                    **inputs)
    expected_residual = np.zeros(6)  # [mol/L] under default unit numerical weights
    np.testing.assert_allclose(reference.get_objective(rate, out_array=True),
                               expected_residual, rtol=0.0, atol=ROUNDING_ATOL)
    inputs[reordered_input] = dict(reversed(list(inputs[reordered_input].items())))
    reordered = ParameterEstimation(accumulating_concentration, rate, times,
                                    **inputs)
    residual = reordered.get_objective(rate, out_array=True)  # unit-weighted residual
    np.testing.assert_allclose(residual, expected_residual, rtol=0.0,
                               atol=ROUNDING_ATOL)



def concentration_jacobian(parameters, time_s, initial_mol_l, gain=1.0):
    """Return the exact constant-rate model sensitivity.

    Parameters
    ----------
    parameters : ndarray
        Rate [mol/L/s], unused because the model is linear in rate.
    time_s : ndarray
        Measurement times [s].
    initial_mol_l : float
        Initial concentration [mol/L], independent of the fitted rate.
    gain : float, optional
        Known rate multiplier [-].

    Returns
    -------
    ndarray
        Sensitivity [s], shape (1, n_times).
    """
    return gain * time_s[np.newaxis, :]


@pytest.fixture
def experiments():
    """Construct asymmetric concentration experiments with a known LS fit.

    Returns
    -------
    dict
        Times [s], observations and initial values [mol/L], gains [-].
        Measurements at time zero have offsets 0.1 and 0.2 mol/L so the
        optimum rate remains exactly 2 mol/L/s and variance is nonzero.
    """
    return {
        'x_data': {'dilute': np.array([0., 1., 3.]),
                   'concentrated': np.array([0., 2., 3.])},  # [s]
        'y_data': {'dilute': np.array([1.1, 3., 7.]),
                   'concentrated': np.array([10.2, 18., 22.])},  # [mol/L]
        'args_fun': {'dilute': (1.,), 'concentrated': (10.,)},  # [mol/L]
        'kwargs_fun': {'dilute': {'gain': 1.},
                       'concentrated': {'gain': 2.}},  # [-]
    }


@pytest.mark.parametrize('reordered', ['x_data', 'y_data', 'args_fun',
                                      'kwargs_fun', 'all'])
def test_objective_and_real_lm_fit_are_invariant(experiments, reordered):
    for label in experiments if reordered == 'all' else [reordered]:
        experiments[label] = dict(reversed(list(experiments[label].items())))
    estimator = ParameterEstimation(
        accumulating_concentration, [1.5], jac_fun=concentration_jacobian,
        **experiments)  # [mol/L/s], non-optimal positive seed
    # Sum of squared rate offsets: .5**2*(1+9+16+36); noise adds .01+.04.
    expected_objective = 7.775  # [-], identity numerical weighting
    assert estimator.get_objective([1.5]) == pytest.approx(
        expected_objective, rel=ROUNDING_ATOL)
    fitted, _, info = estimator.optimize_fn(verbose=False)
    # [-], 1e-6 budget exceeds the native LM default termination tolerances.
    np.testing.assert_allclose(fitted, [2.], rtol=1e-6, atol=0.)  # [mol/L/s]
    assert estimator.get_objective(fitted) == pytest.approx(.025, rel=1e-6)
    assert estimator.experim_names == list(experiments['x_data'])
    assert np.all(np.isfinite(info['fun']))


@pytest.mark.parametrize('label', ['y_data', 'args_fun', 'kwargs_fun'])
@pytest.mark.parametrize('defect', ['missing', 'unexpected', 'renamed'])
def test_experiment_key_mismatch_is_actionable(experiments, label, defect):
    if defect in ('unexpected', 'renamed'):
        experiments[label]['unknown'] = experiments[label]['concentrated']
    if defect in ('missing', 'renamed'):
        del experiments[label]['concentrated']
    with pytest.raises(ValueError, match=label + ' experiment keys must match x_data') as error:
        ParameterEstimation(accumulating_concentration, [2.], **experiments)
    assert ('concentrated' in str(error.value)) == (defect != 'unexpected')
    assert ('unknown' in str(error.value)) == (defect != 'missing')


@pytest.mark.parametrize('label', ['y_data', 'args_fun', 'kwargs_fun'])
@pytest.mark.parametrize('count', [1, 3])
def test_positional_count_mismatch_cannot_silently_drop_experiments(
        experiments, label, count):
    experiments = {key: list(value.values()) for key, value in experiments.items()}
    experiments[label] = (experiments[label] * count)[:count]
    with pytest.raises(ValueError, match='experiments|per experiment'):
        ParameterEstimation(accumulating_concentration, [2.], **experiments)


@pytest.mark.parametrize('named', [False, True])
@pytest.mark.parametrize('representation', ['list', 'tuple'])
def test_positional_multiple_experiments_keep_order(experiments, named, representation):
    container = list if representation == 'list' else tuple
    inputs = {key: container(value.values()) for key, value in experiments.items()}
    if named:
        inputs['x_data'] = experiments['x_data']
    estimator = ParameterEstimation(accumulating_concentration, [2.], **inputs)
    expected = np.array([-.1, 0., 0., -.2, 0., 0.])  # [mol/L], unit weights
    np.testing.assert_allclose(estimator.get_objective([2.], out_array=True),
                               expected, rtol=0., atol=ROUNDING_ATOL)


@pytest.mark.parametrize('named, representation', [
    (False, 'direct'), (True, 'direct'), (False, 'list'), (True, 'list'),
    (True, 'keyed')])
def test_single_experiment_callback_conventions(named, representation):
    times = np.array([0., 1., 3.])  # [s]
    observations = np.array([1., 5., 13.])  # [mol/L], C0=1, gain=2, rate=2
    args = (1.,)  # [mol/L]
    kwargs = {'gain': 2.}  # [-]
    if representation == 'list':
        args, kwargs = [args], [kwargs]
    elif representation == 'keyed':
        args, kwargs = {'batch': args}, {'batch': kwargs}
    estimator = ParameterEstimation(
        accumulating_concentration, [2.],
        {'batch': times} if named else times,
        {'batch': observations} if named else observations,
        args_fun=args, kwargs_fun=kwargs)
    np.testing.assert_allclose(estimator.get_objective([2.], out_array=True),
                               np.zeros(3), rtol=0., atol=ROUNDING_ATOL)


def test_direct_single_positional_list_remains_callback_arguments():
    times = np.array([0., 1., 3.])  # [s]
    observations = np.array([1., 3., 7.])  # [mol/L], C0=1, rate=2
    estimator = ParameterEstimation(accumulating_concentration, [2.], times,
                                    observations, args_fun=[1.])  # [mol/L]
    assert estimator.get_objective([2.]) == pytest.approx(0., abs=ROUNDING_ATOL)


def test_nested_spectral_observations_keep_fields_masks_and_caller_data():
    # Different spectral/non-spectral grids exercise the inherited MCR adapter.
    times = {'first': {'spectra': np.array([0., 2.]),
                       'non_spectra': np.array([1.])},
             'second': {'spectra': np.array([0., 3.]),
                        'non_spectra': np.array([2.])}}  # [s]
    observations = {
        'second': {'spectra': np.array([[20., 21.], [22., 23.]]),
                   'non_spectra': np.array([24.])},
        'first': {'spectra': np.array([[10., 11.], [12., 13.]]),
                  'non_spectra': np.array([14.])},
    }  # spectra [-]; non-spectral concentration [mol/L], synthetic adapter data
    estimator = MultipleCurveResolution(
        accumulating_concentration, [2.], times, observations,
        measured_ind={'spectra': [0], 'non_spectra': [0]},
        name_states=['concentration'])
    assert estimator.experim_names == ['first', 'second']
    for index, name in enumerate(times):
        np.testing.assert_array_equal(estimator.x_model[index],
                                      [0., index + 1., index + 2.])
        assert list(estimator.y_data[index]) == ['spectra', 'non_spectra']
        np.testing.assert_array_equal(estimator.x_masks[index]['spectra'],
                                      [True, False, True])
        np.testing.assert_array_equal(estimator.y_data[index]['spectra'][[0, 2]],
                                      observations[name]['spectra'])
        np.testing.assert_array_equal(estimator.y_data[index]['non_spectra'][[1]],
                                      observations[name]['non_spectra'][:, None])
        assert np.isnan(estimator.y_data[index]['spectra'][1]).all()
    assert list(observations) == ['second', 'first']
    np.testing.assert_array_equal(observations['first']['spectra'],
                                  [[10., 11.], [12., 13.]])


def test_empty_experiments_are_rejected():
    with pytest.raises(ValueError, match='nonzero number of experiments'):
        ParameterEstimation(accumulating_concentration, [2.], {}, {})


def configured_concentration(parameters, time_s, batch):
    """Evaluate a rate with a nested callback configuration.

    Parameters
    ----------
    parameters : ndarray
        Rate [mol/L/s].
    time_s : ndarray
        Measurement times [s].
    batch : dict
        Configuration containing a dimensionless ``gain`` [-].

    Returns
    -------
    ndarray
        Concentrations [mol/L] from zero initial concentration.
    """
    return parameters[0] * time_s * batch['gain']


@pytest.mark.parametrize('name', ['experiment', 'batch'])
def test_single_nested_callback_keywords_and_empty_positional_list(name):
    times = {name: np.array([0., 1., 3.])}  # [s]
    observations = {name: np.array([0., 4., 12.])}  # [mol/L]
    kwargs = {'batch': {'gain': 2.}}  # [-]
    if name == 'batch':
        kwargs = [kwargs]  # Explicitly escape the reserved experiment-key form.
    estimator = ParameterEstimation(configured_concentration, [2.], times,
                                    observations, args_fun=[], kwargs_fun=kwargs)
    assert estimator.get_objective([2.]) == pytest.approx(0., abs=ROUNDING_ATOL)


def test_keyword_entries_must_be_dictionaries(experiments):
    experiments['kwargs_fun']['dilute'] = ()
    with pytest.raises(TypeError, match='Each kwargs_fun entry must be a dictionary'):
        ParameterEstimation(accumulating_concentration, [2.], **experiments)


@pytest.mark.parametrize('named', [False, True])
@pytest.mark.parametrize('invalid', [1., np.array(1.)], ids=['scalar', 'zero-d-array'])
def test_positional_entries_reject_noniterables_at_construction(
        experiments, named, invalid):
    # Both malformed values represent the dilute initial concentration [mol/L].
    experiments['args_fun']['dilute'] = invalid
    if not named:
        experiments = {key: list(value.values())
                       for key, value in experiments.items()}
    with pytest.raises(TypeError, match='args_fun.*iterable') as error:
        ParameterEstimation(accumulating_concentration, [2.], **experiments)
    identity = "['dilute']" if named else '[0]'
    assert 'offending experiments: ' + identity in str(error.value)


@pytest.mark.parametrize('representation', ['direct', 'unnamed-mapping'])
def test_single_positional_scalar_is_rejected_at_construction(representation):
    times = np.array([0., 1., 3.])  # [s], constant-rate verification schedule
    observations = np.array([1., 3., 7.])  # [mol/L], C0=1, rate=2
    args = 1. if representation == 'direct' else {'initial': 1.}  # [mol/L]
    with pytest.raises(TypeError, match=r'args_fun.*offending experiments: \[0\]'):
        ParameterEstimation(accumulating_concentration, [2.], times,
                            observations, args_fun=args)


@pytest.mark.parametrize('container', [list, np.array], ids=['list', 'array'])
def test_iterable_positional_entries_preserve_callback_values(experiments, container):
    experiments['args_fun'] = {
        name: container(values) for name, values in experiments['args_fun'].items()
    }  # [mol/L], retain valid iterable containers rather than requiring tuples
    estimator = ParameterEstimation(accumulating_concentration, [2.], **experiments)
    expected = np.array([-.1, 0., 0., -.2, 0., 0.])  # [mol/L], unit weights
    np.testing.assert_allclose(estimator.get_objective([2.], out_array=True),
                               expected, rtol=0., atol=ROUNDING_ATOL)


def tuple_initial_concentration(parameters, time_s, initial):
    """Evaluate accumulation with a tuple-valued callback argument.

    Parameters
    ----------
    parameters : ndarray
        Concentration-production rate [mol/L/s], shape (1,).
    time_s : ndarray
        Measurement times [s], shape (n_times,).
    initial : tuple
        One initial concentration [mol/L], shape (1,).

    Returns
    -------
    ndarray
        Concentrations [mol/L], shape (n_times,).
    """
    initial_mol_l, = initial  # [mol/L], scalar input must not replace this tuple
    return accumulating_concentration(parameters, time_s, initial_mol_l)


@pytest.mark.parametrize('wrapped', [False, True])
def test_single_tuple_valued_callback_argument_has_an_explicit_escape(wrapped):
    times = np.array([0., 1., 3.])  # [s], constant-rate verification schedule
    observations = np.array([1., 3., 7.])  # [mol/L], C0=1, rate=2
    args = ((1.,),)  # [mol/L], one callback argument that is itself a tuple
    if wrapped:
        args = [args]  # One experiment's argument container.
    estimator = ParameterEstimation(tuple_initial_concentration, [2.], times,
                                    observations, args_fun=args)
    assert isinstance(estimator.args_fun[0][0], tuple)
    np.testing.assert_allclose(estimator.get_objective([2.], out_array=True),
                               np.zeros(3), rtol=0., atol=ROUNDING_ATOL)
