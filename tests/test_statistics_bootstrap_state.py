"""Regressions for estimator state across residual-bootstrap refits.

Issue #262: ``StatisticsClass.bootstrap_params`` refits generated datasets
and must leave the fitted ``ParameterEstimation`` (observations, accepted
parameters, residuals, predictions, solver information, covariance and
solver options) unchanged, whether every sample succeeds, a sample's
singular-system failure is recorded as a NaN row, or a sample's exception
propagates.

The fixture is a one-parameter concentration model c = k t [mol/L] over two
experiments with different sampling times [s] and asymmetric errors, fitted
by PharmaPy's Levenberg-Marquardt (LM) solver. Draws use NumPy's global
generator at a fixed seed, restored afterwards, so the generated datasets and
their closed-form least-squares estimates are known in advance. Runs in
well under a second.
"""

import copy
import warnings

import numpy as np
import pytest

from PharmaPy.ParamEstim import ParameterEstimation
from PharmaPy.StatsModule import StatisticsClass


pytestmark = pytest.mark.unit

TIMES_S = {'short': np.array([0.5, 1.0, 2.0]),
           'long': np.array([1.0, 2.5, 4.0, 6.0])}  # [s]
# Fixed asymmetric measurement errors at TIMES_S [mol/L].
ERRORS_MOL_L = {'short': np.array([0.05, -0.12, 0.30]),
                'long': np.array([-0.20, 0.45, -0.10, 0.60])}
TRUE_RATE = 1.0  # [mol/L/s]
SEED_RATE = np.array([2.0])  # [mol/L/s], above every fitted rate below
NUM_SAMPLES = 12  # [-], bootstrap datasets per run
BOOTSTRAP_SEED = 262  # [-], arbitrary fixed seed; any value works
# For a one-parameter linear least-squares problem every LM trial lies
# between the current point and the optimum (the damped step is a shortened
# Newton step), so fits from SEED_RATE approach their estimate from above
# and never visit rates below it.
# Refits stop on LM's default 1e-8 relative step / 1e-8 gradient criteria;
# 1e-6 relative to the closed-form rate covers that and roundoff while
# staying far below the 1e-2 spread of the sample estimates.
REFIT_RTOL = 1e-6  # [-]


def _observations():
    """Return the observed concentrations of both experiments.

    Returns
    -------
    dict
        Experiment name -> concentrations at ``TIMES_S`` [mol/L].
    """
    return {name: TRUE_RATE * times + ERRORS_MOL_L[name]
            for name, times in TIMES_S.items()}


def _linear_jacobian(params, time_s):
    """Return d(c)/dk = t, shape ``(n_times, 1)`` [s].

    Parameters
    ----------
    params : numpy.ndarray
        Rate, shape ``(1,)`` [mol/L/s]; unused for this linear model.
    time_s : numpy.ndarray
        Times, shape ``(n_times,)`` [s].

    Returns
    -------
    numpy.ndarray
        Sensitivities, shape ``(n_times, 1)`` [s].
    """
    return time_s[:, np.newaxis]


def _guarded_model(lower_rate, error_type, last_call=None):
    """Return c = k t with a lower validity bound on the rate.

    Parameters
    ----------
    lower_rate : float
        Smallest admissible rate [mol/L/s]; ``-inf`` disables the guard.
    error_type : type
        Exception raised below ``lower_rate``.
    last_call : dict, optional
        If given, ``last_call['rate']`` records the rate of every call
        [mol/L/s], like the kinetics a unit-operation callback retains.

    Returns
    -------
    callable
        Model ``f(params, time_s)`` returning concentrations [mol/L].
    """
    def linear_model(params, time_s):
        """Return c = k t, rejecting rates below the bound.

        Parameters
        ----------
        params : numpy.ndarray
            Rate k, shape ``(1,)`` [mol/L/s].
        time_s : numpy.ndarray
            Times, shape ``(n_times,)`` [s].

        Returns
        -------
        numpy.ndarray
            Concentrations, shape ``(n_times,)`` [mol/L].

        Raises
        ------
        error_type
            If ``params[0] < lower_rate``.
        """
        if last_call is not None:
            last_call['rate'] = params[0]  # [mol/L/s]
        if params[0] < lower_rate:
            raise error_type(
                f"rate {params[0]:.6f} mol/L/s is below the validated "
                f"range starting at {lower_rate:.6f} mol/L/s")
        return params[0] * time_s

    return linear_model


def _fitted_statistics(model, store_iter=True):
    """Fit the fixture and wrap it for bootstrap statistics.

    Parameters
    ----------
    model : callable
        Model ``f(params, time_s)`` [mol/L].
    store_iter : bool, optional
        Passed to ``optimize_fn``; False keeps ``objfun_iter`` a list.

    Returns
    -------
    StatisticsClass
        Statistics of an LM fit run with ``verbose=True``, so the stored
        solver options record a verbose fit.
    """
    estimator = ParameterEstimation(model, SEED_RATE, TIMES_S,
                                    _observations(),
                                    jac_fun=_linear_jacobian,
                                    name_params=['rate_mol_l_s'])
    estimator.optimize_fn(verbose=True, store_iter=store_iter)
    return StatisticsClass(estimator)


def _seeded(function, *args):
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
        The return value of ``function``; the previous generator state is
        restored afterwards.
    """
    previous_state = np.random.get_state()
    try:
        np.random.seed(BOOTSTRAP_SEED)
        return function(*args)
    finally:
        np.random.set_state(previous_state)


def _sample_estimates(statistics):
    """Return closed-form estimates of the seeded bootstrap datasets.

    Parameters
    ----------
    statistics : StatisticsClass
        Statistics of the fitted fixture.

    Returns
    -------
    numpy.ndarray
        Least-squares rates ``sum(t y) / sum(t**2)``, shape
        ``(NUM_SAMPLES,)`` [mol/L/s], for the datasets ``bootstrap_params``
        draws first under ``BOOTSTRAP_SEED``.
    """
    samples = _seeded(statistics.get_bootsamples, NUM_SAMPLES)
    times = np.concatenate(list(TIMES_S.values()))  # [s]
    return np.array([
        np.concatenate([experiment[index][:, 0] for experiment in samples])
        @ times / (times @ times) for index in range(NUM_SAMPLES)])


def _state(obj):
    """Return a deep copy of an object's non-callable attributes.

    Parameters
    ----------
    obj : object
        Estimator or statistics object.

    Returns
    -------
    dict
        Attribute name -> deep copy; callables and the referenced estimator
        map to themselves and are compared by identity.
    """
    return {name: (value if callable(value)
                   or isinstance(value, ParameterEstimation)
                   else copy.deepcopy(value))
            for name, value in vars(obj).items()}


def _assert_same(expected, actual, path='state'):
    """Assert recursive, bit-identical equality of attribute values.

    Parameters
    ----------
    expected, actual : object
        Values to compare; arrays compare exactly with NaN equal to NaN.
    path : str, optional
        Location reported on failure.
    """
    if isinstance(expected, dict):
        assert isinstance(actual, dict), path
        assert list(expected) == list(actual), path
        for key in expected:
            _assert_same(expected[key], actual[key], f'{path}[{key!r}]')
    elif isinstance(expected, (list, tuple)):
        assert type(actual) is type(expected), path
        assert len(actual) == len(expected), path
        for index, (item, other) in enumerate(zip(expected, actual)):
            _assert_same(item, other, f'{path}[{index}]')
    elif isinstance(expected, np.ndarray):
        assert isinstance(actual, np.ndarray), path
        np.testing.assert_array_equal(actual, expected, err_msg=path)
    elif callable(expected) or isinstance(expected, ParameterEstimation):
        assert actual is expected, path
    elif hasattr(expected, 'equals'):  # pandas objects
        assert expected.equals(actual), path
    else:
        assert actual == expected, path


def test_successful_bootstrap_keeps_estimator_state():
    last_call = {}
    statistics = _fitted_statistics(
        _guarded_model(-np.inf, ValueError, last_call))
    estimator = statistics.inst
    accepted = estimator.params_convg  # [mol/L/s]
    objective = estimator.get_objective(accepted)  # [(mol/L)**2]
    estimator_before = _state(estimator)
    statistics_before = _state(statistics)
    estimates = _sample_estimates(statistics)  # [mol/L/s]
    # The generated datasets differ from the observations.
    assert np.all(estimates != accepted[0])

    boot_params = _seeded(statistics.bootstrap_params, NUM_SAMPLES)

    # Each refit reaches its dataset's closed-form estimate.
    np.testing.assert_allclose(boot_params[:, 0], estimates,
                               rtol=REFIT_RTOL)
    assert boot_params.shape == (NUM_SAMPLES, 1)
    _assert_same(estimator_before, _state(estimator), 'estimator')
    assert estimator.optim_options['verbose'] is True
    # The stateful model was returned to the accepted rate by bootstrap.
    assert last_call['rate'] == accepted[0]
    statistics_after = _state(statistics)
    assert statistics_after.pop('boot_params') is not None
    assert statistics_after.pop('num_samples') == NUM_SAMPLES
    assert statistics.boot_params is boot_params
    statistics_before.pop('boot_params')
    _assert_same(statistics_before, statistics_after, 'statistics')
    assert estimator.get_objective(accepted) == objective


@pytest.mark.parametrize('error_type', [ValueError, np.linalg.LinAlgError])
def test_failed_sample_keeps_estimator_state(error_type):
    """A sample that leaves the model's validity range keeps the fit intact.

    The bound sits midway between the accepted rate and the largest sample
    estimate below it, so the original fit never reaches it and exactly the
    samples estimated below it fail. ``ValueError`` propagates;
    ``LinAlgError`` is recorded as a NaN row with a warning.
    """
    reference = _fitted_statistics(_guarded_model(-np.inf, error_type))
    estimates = _sample_estimates(reference)  # [mol/L/s]
    accepted = reference.inst.params_convg[0]  # [mol/L/s]
    below = estimates[estimates < accepted]
    assert below.size > 0 and below.size < NUM_SAMPLES
    lower_rate = 0.5 * (accepted + below.max())  # [mol/L/s]
    failing = estimates < lower_rate

    last_call = {}
    statistics = _fitted_statistics(
        _guarded_model(lower_rate, error_type, last_call))
    estimator = statistics.inst
    assert estimator.params_convg[0] == accepted
    objective = estimator.get_objective(estimator.params_convg)  # [(mol/L)**2]
    estimator_before = _state(estimator)

    if error_type is ValueError:
        with pytest.raises(ValueError, match='below the validated range'):
            _seeded(statistics.bootstrap_params, NUM_SAMPLES)
        assert statistics.boot_params is None
    else:
        with pytest.warns(RuntimeWarning, match='below the validated range'):
            boot_params = _seeded(statistics.bootstrap_params, NUM_SAMPLES)
        np.testing.assert_array_equal(np.isnan(boot_params[:, 0]), failing)
        np.testing.assert_allclose(boot_params[~failing, 0],
                                   estimates[~failing], rtol=REFIT_RTOL)

    _assert_same(estimator_before, _state(estimator), 'estimator')
    # Restored before the error propagated or the rows were returned.
    assert last_call['rate'] == accepted
    assert estimator.get_objective(estimator.params_convg) == objective


def _switchable_model(failure):
    """Return c = k t that fails on request, counting its calls.

    Parameters
    ----------
    failure : dict
        ``failure['when']`` is None (never fail), ``'always'``, or
        ``'accepted'`` (fail only at ``failure['rate']`` [mol/L/s]).
        ``failure['calls']`` counts evaluations.

    Returns
    -------
    callable
        Model ``f(params, time_s)`` returning concentrations [mol/L] and
        raising ``ValueError`` naming the call number when it fails.
    """
    def linear_model(params, time_s):
        """Return c = k t or raise the requested failure.

        Parameters
        ----------
        params : numpy.ndarray
            Rate k, shape ``(1,)`` [mol/L/s].
        time_s : numpy.ndarray
            Times, shape ``(n_times,)`` [s].

        Returns
        -------
        numpy.ndarray
            Concentrations, shape ``(n_times,)`` [mol/L].

        Raises
        ------
        ValueError
            On every call when ``failure['when'] == 'always'``, or at the
            rate ``failure['rate']`` when it is ``'accepted'``; the message
            names the call number.
        """
        failure['calls'] += 1
        fails = (failure['when'] == 'always'
                 or (failure['when'] == 'accepted'
                     and params[0] == failure['rate']))
        if fails:
            raise ValueError(f"model failure at call {failure['calls']}")
        return params[0] * time_s

    return linear_model


def test_restore_failure_after_successful_samples_propagates():
    """A fit that can no longer be evaluated is reported, not hidden."""
    failure = {'when': None, 'calls': 0}
    statistics = _fitted_statistics(_switchable_model(failure))
    estimator_before = _state(statistics.inst)
    failure.update(when='accepted', rate=statistics.inst.params_convg[0])

    with pytest.raises(ValueError, match='model failure at call') as raised:
        _seeded(statistics.bootstrap_params, NUM_SAMPLES)

    # The restoring evaluation, the last model call, raised: no sample did.
    assert str(raised.value) == f"model failure at call {failure['calls']}"
    assert statistics.boot_params is None
    _assert_same(estimator_before, _state(statistics.inst), 'estimator')


def test_restore_failure_does_not_mask_sample_error():
    """The sample's error propagates; the restore failure is a warning."""
    failure = {'when': None, 'calls': 0}
    statistics = _fitted_statistics(_switchable_model(failure))
    estimator_before = _state(statistics.inst)
    failure['when'] = 'always'
    first_failing_call = failure['calls'] + 1  # [-]

    with pytest.warns(RuntimeWarning, match='Could not restore') as caught:
        with pytest.raises(ValueError) as raised:
            _seeded(statistics.bootstrap_params, NUM_SAMPLES)

    assert str(raised.value) == f'model failure at call {first_failing_call}'
    assert len(caught) == 1
    assert f'call {first_failing_call + 1}' in str(caught[0].message)
    _assert_same(estimator_before, _state(statistics.inst), 'estimator')


@pytest.mark.parametrize('sample_error', [False, True],
                         ids=['success', 'sample_error'])
def test_list_histories_of_unstored_fit_are_not_extended(sample_error):
    """Refits and the restore do not append to the fit's history lists.

    ``optimize_fn(store_iter=False)`` leaves ``objfun_iter`` a list, which
    the objective callback appends to after every successful evaluation.
    With ``sample_error`` the rate bound of
    ``test_failed_sample_keeps_estimator_state`` makes a refit fail after
    successful evaluations, and the error propagates.
    """
    lower_rate = -np.inf  # [mol/L/s]
    if sample_error:
        reference = _fitted_statistics(_guarded_model(-np.inf, ValueError))
        estimates = _sample_estimates(reference)  # [mol/L/s]
        accepted = reference.inst.params_convg[0]  # [mol/L/s]
        lower_rate = 0.5 * (accepted
                            + estimates[estimates < accepted].max())
    statistics = _fitted_statistics(_guarded_model(lower_rate, ValueError),
                                    store_iter=False)
    estimator = statistics.inst
    histories = {name: getattr(estimator, name)
                 for name in ('params_iter', 'objfun_iter')}
    assert isinstance(histories['objfun_iter'], list)
    contents = copy.deepcopy(histories)
    estimator_before = _state(estimator)

    if sample_error:
        with pytest.raises(ValueError, match='below the validated range'):
            _seeded(statistics.bootstrap_params, NUM_SAMPLES)
    else:
        _seeded(statistics.bootstrap_params, NUM_SAMPLES)

    for name, history in histories.items():
        assert getattr(estimator, name) is history
        _assert_same(contents[name], history, name)
    _assert_same(estimator_before, _state(estimator), 'estimator')


def test_sample_error_propagates_when_warnings_are_errors():
    """A warnings-as-errors filter cannot replace the sample's exception."""
    failure = {'when': None, 'calls': 0}
    statistics = _fitted_statistics(_switchable_model(failure))
    failure['when'] = 'always'
    first_failing_call = failure['calls'] + 1  # [-]

    with warnings.catch_warnings():
        warnings.simplefilter('error')
        with pytest.raises(ValueError) as raised:
            _seeded(statistics.bootstrap_params, NUM_SAMPLES)

    assert str(raised.value) == f'model failure at call {first_failing_call}'
