"""Regressions for ``StatsModule.ParallelProblem.run_fit`` (#399).

``run_fit(sample)`` must fit the given dataset, in the per-experiment
layout of ``StatisticsClass.get_bootsamples``, on an isolated copy of the
fitted ``ParameterEstimation``, leave the estimator unchanged, and return
a stateful model callable to the accepted parameters.

Fixture: first-order decay ``c = c0 * exp(-k t)`` with ``c0 = 1 mol/L`` and
rate ``k`` [1/s], observed with fixed asymmetric errors [mol/L] in one
experiment (9 samples) or two experiments with different grids (4 and 6
samples). Expected refits are computed independently with
``scipy.optimize.least_squares`` on the plain residuals. The
``MultipleCurveResolution`` estimator of
``test_paramestim_mcr_observation_masks`` checks the explicit rejection of
nested observations (#505). Runs in about a second.
"""

import numpy as np
import pytest
from scipy.optimize import least_squares

from PharmaPy.ParamEstim import ParameterEstimation
from PharmaPy.StatsModule import ParallelProblem, StatisticsClass
from test_statistics_bootstrap_state import _assert_same, _state
import test_paramestim_mcr_observation_masks as mcr_fixture

pytestmark = pytest.mark.unit

INITIAL_CONC = 1.0  # [mol/L], c0 of the decay model
TRUE_RATE = 0.5  # [1/s]
SEED_RATE = np.array([0.4])  # [1/s]
TIME_S = np.linspace(0.5, 4.5, 9)  # [s]
# Fixed asymmetric measurement errors at TIME_S [mol/L] (#399 reproduction).
NOISE_MOL_L = np.array([0.012, -0.008, 0.015, -0.011, 0.006, -0.014, 0.009,
                        -0.005, 0.010])
SHIFT_MOL_L = 0.2  # [mol/L], offset that turns the data into a new sample
TWO_EXPERIMENT_TIMES_S = {'short': np.array([0.3, 0.9, 1.6, 2.4]),
                          'long': np.array([0.5, 1.5, 3.0, 4.5, 6.0, 8.0])}
TWO_EXPERIMENT_NOISE = {'short': np.array([0.02, -0.01, 0.015, -0.02]),
                        'long': np.array([-0.01, 0.03, -0.02, 0.01, 0.005,
                                          -0.004])}  # [mol/L]
# The PharmaPy LM refit stops on 1e-8 relative step / gradient criteria
# (LevMarq defaults); the SciPy reference is solved to 1e-14. A 1e-6
# relative bound covers both while staying far below the ~0.2 1/s
# difference between the original fit and the shifted-sample fit.
REFIT_RTOL = 1e-6  # [-]
REFERENCE_TOL = 1e-14  # [-], least_squares xtol/ftol/gtol
BOOTSTRAP_SEED = 399  # [-], arbitrary fixed seed


def _decay(params, time_s):
    """Return the first-order decay profile.

    Parameters
    ----------
    params : numpy.ndarray
        Rate constant, shape ``(1,)`` [1/s].
    time_s : numpy.ndarray
        Times, shape ``(n_times,)`` [s].

    Returns
    -------
    numpy.ndarray
        Concentrations ``c0 * exp(-k t)``, shape ``(n_times,)`` [mol/L].
    """
    return INITIAL_CONC * np.exp(-params[0] * time_s)


def _decay_jacobian(params, time_s):
    """Return the analytic sensitivity of the decay to its rate.

    An exact Jacobian makes the LM fixed point the exact least-squares
    optimum; the default forward differences (relative step 1e-3) would
    shift it by ~1e-5 relative.

    Parameters
    ----------
    params : numpy.ndarray
        Rate constant, shape ``(1,)`` [1/s].
    time_s : numpy.ndarray
        Times, shape ``(n_times,)`` [s].

    Returns
    -------
    numpy.ndarray
        ``dc/dk = -t c0 exp(-k t)``, shape ``(n_times, 1)`` [mol*s/L].
    """
    return (-time_s * _decay(params, time_s))[:, np.newaxis]


def _reference_fit(times, observations):
    """Fit the decay rate with SciPy, independently of PharmaPy.

    Parameters
    ----------
    times : list of numpy.ndarray
        Times per experiment [s].
    observations : list of numpy.ndarray
        Concentrations per experiment, same shapes [mol/L].

    Returns
    -------
    float
        Least-squares rate constant [1/s].
    """
    def residuals(rate):
        return np.concatenate([_decay(rate, t) - y
                               for t, y in zip(times, observations)])

    result = least_squares(residuals, SEED_RATE, xtol=REFERENCE_TOL,
                           ftol=REFERENCE_TOL, gtol=REFERENCE_TOL)
    return result.x[0]


def _fitted_single(model=_decay):
    """Fit the single-experiment fixture.

    Parameters
    ----------
    model : callable, optional
        Model ``f(params, time_s)`` [mol/L].

    Returns
    -------
    estimator : ParameterEstimation
        LM fit of the observations.
    observed : numpy.ndarray
        Observed concentrations, shape ``(9,)`` [mol/L].
    """
    observed = _decay([TRUE_RATE], TIME_S) + NOISE_MOL_L  # [mol/L]
    estimator = ParameterEstimation(model, SEED_RATE, TIME_S, observed,
                                    jac_fun=_decay_jacobian)
    estimator.optimize_fn(verbose=False, store_iter=False)
    return estimator, observed


def test_run_fit_fits_the_given_sample_without_touching_the_estimator():
    estimator, observed = _fitted_single()
    accepted = estimator.params_convg.copy()  # [1/s]
    before = _state(estimator)
    shifted = observed + SHIFT_MOL_L  # [mol/L]

    refit = ParallelProblem(estimator).run_fit([shifted])

    expected = _reference_fit([TIME_S], [shifted])  # [1/s]
    assert refit.shape == (1,)
    assert refit[0] == pytest.approx(expected, rel=REFIT_RTOL)
    # Guard against a vacuous pass: the sample's fit (~0.27 1/s) differs
    # from the original estimate (~0.50 1/s) by far more than the tolerance.
    assert abs(expected - accepted[0]) > 0.1  # [1/s]
    assert not hasattr(estimator, 'y_fit')
    _assert_same(before, _state(estimator))


def test_run_fit_matches_a_bootstrap_params_sample():
    estimator, _ = _fitted_single()
    statistics = StatisticsClass(estimator)
    state = np.random.get_state()
    try:
        np.random.seed(BOOTSTRAP_SEED)
        boot_rate = statistics.bootstrap_params(1)[0]  # [1/s]
        np.random.seed(BOOTSTRAP_SEED)
        boots = statistics.get_bootsamples(1)
    finally:
        np.random.set_state(state)

    sample = [experiment[0] for experiment in boots]  # (9, 1) [mol/L]
    refit = ParallelProblem(estimator).run_fit(sample)

    np.testing.assert_allclose(refit, boot_rate, rtol=REFIT_RTOL)
    expected = _reference_fit([TIME_S], [sample[0][:, 0]])  # [1/s]
    assert refit[0] == pytest.approx(expected, rel=REFIT_RTOL)


def test_run_fit_with_asymmetric_experiments():
    names = list(TWO_EXPERIMENT_TIMES_S)
    observed = {name: _decay([TRUE_RATE], TWO_EXPERIMENT_TIMES_S[name])
                + TWO_EXPERIMENT_NOISE[name] for name in names}  # [mol/L]
    estimator = ParameterEstimation(_decay, SEED_RATE,
                                    TWO_EXPERIMENT_TIMES_S, observed,
                                    jac_fun=_decay_jacobian)
    estimator.optimize_fn(verbose=False)
    before = _state(estimator)

    # Different shifts per experiment, in x_data order [mol/L].
    sample = [observed['short'] + 0.15, observed['long'] - 0.05]
    refit = ParallelProblem(estimator).run_fit(sample)

    expected = _reference_fit(
        [TWO_EXPERIMENT_TIMES_S[name] for name in names], sample)  # [1/s]
    assert refit[0] == pytest.approx(expected, rel=REFIT_RTOL)
    # Swapping the experiments' data must be rejected (lengths 4 and 6).
    with pytest.raises(ValueError, match=r"sample entry 0 \('short'\) has "
                                         r"shape \(6,\); expected \(4, 1\)"):
        ParallelProblem(estimator).run_fit(sample[::-1])
    _assert_same(before, _state(estimator))


def test_run_fit_restores_a_stateful_model():
    last_call = {}

    def recording_decay(params, time_s):
        """Return the decay profile and record the rate used.

        Parameters
        ----------
        params : numpy.ndarray
            Rate constant, shape ``(1,)`` [1/s].
        time_s : numpy.ndarray
            Times, shape ``(n_times,)`` [s].

        Returns
        -------
        numpy.ndarray
            Concentrations, shape ``(n_times,)`` [mol/L].
        """
        last_call['rate'] = params[0]  # [1/s]
        return _decay(params, time_s)

    estimator, observed = _fitted_single(recording_decay)

    refit = ParallelProblem(estimator).run_fit([observed + SHIFT_MOL_L])

    assert refit[0] != pytest.approx(estimator.params_convg[0])
    assert last_call['rate'] == estimator.params_convg[0]


def test_run_fit_validates_the_sample():
    estimator, observed = _fitted_single()
    problem = ParallelProblem(estimator)

    with pytest.raises(ValueError, match="one array per experiment"):
        problem.run_fit(observed)
    with pytest.raises(ValueError, match="got a list of 2 entries"):
        problem.run_fit([observed, observed])
    with pytest.raises(ValueError, match=r"shape \(8,\); expected \(9, 1\)"):
        problem.run_fit([observed[:-1]])

    unfitted = ParameterEstimation(_decay, SEED_RATE, TIME_S, observed)
    with pytest.raises(RuntimeError, match="run optimize_fn first"):
        ParallelProblem(unfitted).run_fit([observed])


def _guarded_decay(lower_rate, last_call, broken_after_failure=False):
    """Return a recording decay model that rejects low rates.

    Parameters
    ----------
    lower_rate : float
        Smallest admissible rate [1/s].
    last_call : dict
        ``last_call['rate']`` records the rate of every call [1/s].
    broken_after_failure : bool, optional
        If True, every call after the first rejection raises
        ``RuntimeError``, so restoring the accepted parameters fails too.

    Returns
    -------
    callable
        Model ``f(params, time_s)`` returning concentrations [mol/L].
    """
    def guarded_decay(params, time_s):
        """Return the decay profile, rejecting rates below ``lower_rate``.

        Parameters
        ----------
        params : numpy.ndarray
            Rate constant, shape ``(1,)`` [1/s].
        time_s : numpy.ndarray
            Times, shape ``(n_times,)`` [s].

        Returns
        -------
        numpy.ndarray
            Concentrations, shape ``(n_times,)`` [mol/L].

        Raises
        ------
        RuntimeError
            After a rejection, when ``broken_after_failure`` is True.
        ValueError
            If ``params[0] < lower_rate``.
        """
        if last_call.get('failed') and broken_after_failure:
            raise RuntimeError("model left unusable by the failed fit")
        last_call['rate'] = params[0]  # [1/s]
        if params[0] < lower_rate:
            last_call['failed'] = True
            raise ValueError(f"rate {params[0]:.4f} 1/s below validity")
        return _decay(params, time_s)

    return guarded_decay


# Admissible-rate bound [1/s] between the shifted-sample optimum
# (~0.27 1/s) and the original fit (~0.50 1/s), so the sample fit fails.
GUARD_RATE = 0.35  # [1/s]


def test_run_fit_failure_propagates_and_restores_the_model():
    last_call = {}
    estimator, observed = _fitted_single(
        _guarded_decay(GUARD_RATE, last_call))
    before = _state(estimator)

    with pytest.raises(ValueError, match="below validity"):
        ParallelProblem(estimator).run_fit([observed + SHIFT_MOL_L])

    assert last_call['rate'] == estimator.params_convg[0]
    _assert_same(before, _state(estimator))


def test_run_fit_failure_is_not_masked_by_a_restore_failure():
    last_call = {}
    estimator, observed = _fitted_single(
        _guarded_decay(GUARD_RATE, last_call, broken_after_failure=True))

    with pytest.warns(RuntimeWarning,
                      match="after the sample fit failed: model left"):
        with pytest.raises(ValueError, match="below validity"):
            ParallelProblem(estimator).run_fit([observed + SHIFT_MOL_L])


@pytest.mark.parametrize("bad_value", [np.nan, np.inf])
def test_run_fit_rejects_non_finite_observed_values(bad_value):
    estimator, observed = _fitted_single()
    sample = observed + SHIFT_MOL_L  # [mol/L]
    sample[2] = bad_value

    with pytest.raises(ValueError, match=r"sample entry 0 \('exp_1'\) is NaN "
                                         r"or infinite at observed \(row, "
                                         r"column\) positions \[\(2, 0\)\]"):
        ParallelProblem(estimator).run_fit([sample])


def test_run_fit_accepts_nan_at_unobserved_positions():
    # Observation 4 is missing, so the estimator masks it out.
    missing = 4  # [-], row index
    observed = _decay([TRUE_RATE], TIME_S) + NOISE_MOL_L  # [mol/L]
    observed[missing] = np.nan
    estimator = ParameterEstimation(_decay, SEED_RATE, TIME_S, observed,
                                    jac_fun=_decay_jacobian)
    estimator.optimize_fn(verbose=False, store_iter=False)
    assert not estimator.x_masks[0][missing, 0]
    shifted = observed + SHIFT_MOL_L  # [mol/L], NaN stays at the mask

    refit = ParallelProblem(estimator).run_fit([shifted])

    keep = np.isfinite(shifted)
    expected = _reference_fit([TIME_S[keep]], [shifted[keep]])  # [1/s]
    assert refit[0] == pytest.approx(expected, rel=REFIT_RTOL)


def test_run_fit_rejects_multiple_curve_resolution():
    estimator = mcr_fixture._estimator(mcr_fixture.model_with_sensitivities,
                                       None)
    estimator.optimize_fn(verbose=False)
    sample = [np.zeros((len(times), 3))
              for times, _ in mcr_fixture.SCHEDULES.values()]

    with pytest.raises(ValueError, match=r"MultipleCurveResolution.*#505"):
        ParallelProblem(estimator).run_fit(sample)
