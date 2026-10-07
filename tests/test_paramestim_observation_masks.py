"""Regressions for staggered observation grids in parameter estimation.

Issue #236: when measured states are sampled at different times, the
estimator fits on the union model grid and masks unobserved entries. The
residuals, their sensitivities (analytical and finite-difference), the
correlated-error weighting, the parameter covariance, and the residual
bootstrap must all use the observed entries only.

The fixtures are two experiments of a two-state concentration model [mol/L]
over time [s], linear in three parameters, with different sampling times
per state and per experiment. Expected values come from closed-form
generalized least squares (GLS) assembled here from the per-state sampling
schedules and ``numpy.linalg.inv`` of each observed covariance block, not
from the estimator's masks or weight roots. Everything runs in milliseconds
with PharmaPy's Levenberg-Marquardt (LM) solver.
"""

import numpy as np
import pytest

from PharmaPy.ParamEstim import MultipleCurveResolution, ParameterEstimation
from PharmaPy.StatsModule import StatisticsClass


pytestmark = pytest.mark.unit

# Per-experiment sampling times of states (c_A, c_B) [s]. Each experiment has
# fully observed rows, rows with only c_A, and rows with only c_B, and the
# experiments use different grids and observation counts.
TIMES_S = {
    'early': (np.array([0.5, 1.0, 2.0, 3.5]), np.array([1.0, 3.5, 5.0])),
    'late': (np.array([0.8, 2.5]), np.array([0.8, 1.6, 2.5, 4.0])),
}
# Fixed asymmetric measurement errors at the times above [mol/L].
ERRORS_MOL_L = {
    'early': (np.array([0.30, -0.45, 0.20, 0.55]),
              np.array([-0.10, 0.25, -0.30])),
    'late': (np.array([-0.35, 0.15]),
             np.array([0.40, -0.20, 0.05, -0.25])),
}
NUM_OBSERVED = 13  # [-], 7 + 6 observations above
# Rate k [mol/L/s], shared offset c0 [mol/L], quadratic coefficient
# a [mol/L/s**2].
TRUE_PARAMS = np.array([0.8, 0.5, 0.12])  # [mol/L/s], [mol/L], [mol/L/s**2]
SEED_PARAMS = np.array([1.0, 0.2, 0.1])  # [mol/L/s], [mol/L], [mol/L/s**2]
NUM_PARAMS = TRUE_PARAMS.size  # [-]

# Variance scale 1/16 [(mol/L)**2] is a power of two, so the correlated
# covariance below is exactly representable; correlation -2/sqrt(5) [-].
VARIANCE_SCALE = 0.0625  # [(mol/L)**2]
WEIGHTS = {
    'identity': None,
    'diagonal': np.diag([4.0, 1.0]) * VARIANCE_SCALE,  # [(mol/L)**2]
    'correlated': np.array(
        [[5.0, -2.0], [-2.0, 1.0]]) * VARIANCE_SCALE,  # [(mol/L)**2]
}

# LM gradient-norm stop with the step-size stop disabled (eps_2 = 0). For a
# linear model the parameter error is then below LM_GRADIENT_TOL /
# lambda_min(X^T P X); 1e-6 sits above the gradient roundoff floor of these
# O(1e2)-O(1e4) normal matrices while keeping that error below 1e-7.
LM_GRADIENT_TOL = 1e-6  # [weighted objective unit / parameter unit]
# Roundoff budget for affine-model identities and closed-form comparisons:
# sums of at most 13 products of O(1e2) terms, near 1e-13 relative.
ROUNDOFF_RTOL = 1e-10  # [-]
# Non-vacuity margin: a pre-#236 quantity must differ from the observed-data
# expectation by at least 10 %, eight orders above ROUNDOFF_RTOL, so a guard
# cannot pass through roundoff. The fixtures differ by 20 % to 5x.
SEPARATION_MARGIN = 0.1  # [-]
# Bootstrap draws per test. Before #236, 200 samples draw at least one
# phantom zero residual with probability above 1 - 1e-90.
BOOTSTRAP_SAMPLES = 200  # [-]
# A drawn error is recovered as fitted - generated; subtracting and
# re-adding O(1) mol/L responses loses at most a few ulps of the residual.
DRAW_MATCH_ATOL = 1e-12  # [mol/L]
# Smallest |residual| treated as distinguishable from a phantom zero: nine
# orders above DRAW_MATCH_ATOL and two below the 0.1-0.5 mol/L errors.
NONZERO_RESIDUAL_FLOOR = 1e-3  # [mol/L]
# Fixed seed for the draw-compatibility test; any value works. The global
# NumPy state is restored afterwards (see _seeded).
BOOTSTRAP_SEED = 236  # [-]


def two_state_model(params, time_s):
    """Return concentrations of the two measured states.

    Parameters
    ----------
    params : numpy.ndarray
        Shape ``(3,)``: rate k [mol/L/s], offset c0 [mol/L], quadratic
        coefficient a [mol/L/s**2].
    time_s : numpy.ndarray
        Model times, shape ``(n_times,)`` [s].

    Returns
    -------
    numpy.ndarray
        Shape ``(n_times, 2)``: ``c_A = k t + c0`` and ``c_B = c0 + a t**2``
        [mol/L].
    """
    rate, offset, quadratic = params
    return np.column_stack(
        (rate * time_s + offset, offset + quadratic * time_s**2))


def two_state_jacobian(params, time_s):
    """Return analytic state-major sensitivities on the full model grid.

    Parameters
    ----------
    params : numpy.ndarray
        Shape ``(3,)``; unused because the model is linear.
    time_s : numpy.ndarray
        Model times, shape ``(n_times,)`` [s].

    Returns
    -------
    numpy.ndarray
        Shape ``(2 * n_times, 3)``: rows of c_A then c_B, columns d/dk [s],
        d/dc0 [-], d/da [s**2]. Unobserved entries are deliberately
        included, as a model callback cannot know the observation masks.
    """
    zeros = np.zeros_like(time_s)
    ones = np.ones_like(time_s)
    return np.vstack((np.column_stack((time_s, ones, zeros)),
                      np.column_stack((zeros, ones, time_s**2))))


def model_with_sensitivities(params, time_s):
    """Return states together with their analytic sensitivities.

    Parameters
    ----------
    params : numpy.ndarray
        Shape ``(3,)``, as in ``two_state_model``.
    time_s : numpy.ndarray
        Model times, shape ``(n_times,)`` [s].

    Returns
    -------
    tuple of numpy.ndarray
        ``two_state_model`` output [mol/L] and ``two_state_jacobian``
        output, the layout ``ParameterEstimation`` accepts from models that
        integrate their own sensitivities.
    """
    return (two_state_model(params, time_s),
            two_state_jacobian(params, time_s))


def _regressors(time):
    """Return the regressor block of one sample time.

    Parameters
    ----------
    time : float
        Sample time [s].

    Returns
    -------
    numpy.ndarray
        Shape ``(2, 3)``: rows c_A, c_B; columns [s], [-], [s**2].
    """
    return np.array([[time, 1.0, 0.0], [0.0, 1.0, time**2]])


def _datasets(errors=ERRORS_MOL_L):
    """Return per-state observations of each experiment.

    Parameters
    ----------
    errors : dict, optional
        Per-experiment tuples of measurement errors [mol/L].

    Returns
    -------
    x_data, y_data : dict
        Experiment name -> list of per-state times [s] / observations
        [mol/L], in (c_A, c_B) order.
    """
    x_data, y_data = {}, {}
    for name, state_times in TIMES_S.items():
        x_data[name] = list(state_times)
        y_data[name] = [
            two_state_model(TRUE_PARAMS, times)[:, state] + errors[name][state]
            for state, times in enumerate(state_times)]
    return x_data, y_data


def _observed_rows(y_data):
    """List observed sample rows from the per-state schedules.

    Parameters
    ----------
    y_data : dict
        Per-state observations [mol/L] keyed like ``TIMES_S``.

    Returns
    -------
    list of tuple
        ``(observed_states, regressors, observations)`` per sample time of
        each experiment, in experiment and time order: state indices,
        ``(n_o, 3)`` regressor rows, and ``(n_o,)`` values [mol/L].
    """
    rows = []
    for name, state_times in TIMES_S.items():
        for time in np.union1d(*state_times):
            observed = [state for state, times in enumerate(state_times)
                        if time in times]
            values = np.array([
                y_data[name][state][np.flatnonzero(
                    state_times[state] == time)[0]]
                for state in observed])
            rows.append((observed, _regressors(time)[observed], values))
    return rows


def _covariance_matrix(weight_key):
    """Return the measurement covariance used by the closed forms.

    Parameters
    ----------
    weight_key : str
        Key of ``WEIGHTS``.

    Returns
    -------
    numpy.ndarray
        Shape ``(2, 2)`` [(mol/L)**2]; the identity for default weights.
    """
    weights = WEIGHTS[weight_key]
    return np.eye(2) if weights is None else weights


def _closed_form(weight_key, params, y_data):
    """Assemble the observed-data GLS quantities.

    Parameters
    ----------
    weight_key : str
        Key of ``WEIGHTS``.
    params : numpy.ndarray
        Parameters at which the objective and gradient are evaluated.
    y_data : dict
        Per-state observations [mol/L].

    Returns
    -------
    dict
        ``objective`` 1/2 sum r_o^T inv(S_oo) r_o, ``gradient``
        sum X_o^T inv(S_oo) r_o, ``normal`` sum X_o^T inv(S_oo) X_o,
        ``estimate`` normal^-1 sum X_o^T inv(S_oo) y_o, ``covariance``
        s2 normal^-1 with s2 the weighted SSR at ``estimate`` over
        (NUM_OBSERVED - NUM_PARAMS), and ``phantom_normal``,
        sum X^T inv(S) X over every union-grid row with both states, which
        is what ``jac @ jac.T`` contained before #236.
    """
    # Units below are for covariance weights; identity weights are [-] and
    # leave the objective in [(mol/L)**2].
    covariance = _covariance_matrix(weight_key)  # [(mol/L)**2]
    precision_full = np.linalg.inv(covariance)  # [(L/mol)**2]
    # Entry (i, j): [(L/mol)**2] times regressor units of params i and j.
    normal = np.zeros((NUM_PARAMS, NUM_PARAMS))
    rhs = np.zeros(NUM_PARAMS)  # [L/mol] times each regressor unit
    gradient = np.zeros(NUM_PARAMS)  # [1/parameter unit]
    objective = 0.0  # [-]
    phantom_normal = np.zeros((NUM_PARAMS, NUM_PARAMS))  # as ``normal``
    rows = _observed_rows(y_data)
    for observed, regressors, values in rows:
        precision = np.linalg.inv(
            covariance[np.ix_(observed, observed)])  # [(L/mol)**2]
        residual = regressors @ params - values  # [mol/L]
        normal += regressors.T @ precision @ regressors
        rhs += regressors.T @ precision @ values
        gradient += regressors.T @ precision @ residual
        objective += 0.5 * residual @ precision @ residual
    for name, state_times in TIMES_S.items():
        for time in np.union1d(*state_times):
            full = _regressors(time)
            phantom_normal += full.T @ precision_full @ full
    estimate = np.linalg.solve(normal, rhs)  # parameter units
    weighted_ssr = sum(  # [-]
        (regressors @ estimate - values)
        @ np.linalg.inv(covariance[np.ix_(observed, observed)])
        @ (regressors @ estimate - values)
        for observed, regressors, values in rows)
    # Entry (i, j): product of the units of parameters i and j.
    covariance_params = (weighted_ssr / (NUM_OBSERVED - NUM_PARAMS)
                         * np.linalg.inv(normal))
    return {'objective': objective, 'gradient': gradient, 'normal': normal,
            'estimate': estimate, 'covariance': covariance_params,
            'phantom_normal': phantom_normal}


def _estimator(weight_key, sensitivity, y_data=None):
    """Build the staggered-grid estimator.

    Parameters
    ----------
    weight_key : str
        Key of ``WEIGHTS``.
    sensitivity : {'jac_fun', 'finite_difference', 'model_output'}
        Source of the model sensitivities: ``two_state_jacobian`` as
        ``jac_fun``, PharmaPy finite differences, or the model's own
        ``(states, sensitivities)`` return value.
    y_data : dict, optional
        Per-state observations; defaults to ``_datasets()``.

    Returns
    -------
    ParameterEstimation
        Estimator seeded at ``SEED_PARAMS``.
    """
    x_data, default_y = _datasets()
    model = (model_with_sensitivities if sensitivity == 'model_output'
             else two_state_model)
    return ParameterEstimation(
        model, SEED_PARAMS, x_data,
        default_y if y_data is None else y_data,
        jac_fun=two_state_jacobian if sensitivity == 'jac_fun' else None,
        weight_matrix=WEIGHTS[weight_key],
        name_params=['rate_mol_l_s', 'offset_mol_l', 'quadratic_mol_l_s2'],
        name_states=['c_A', 'c_B'])


def _lm_options():
    """Return LM options whose only convergence stop is the gradient norm.

    Returns
    -------
    dict
        ``eps_1 = LM_GRADIENT_TOL`` and ``eps_2 = 0``.
    """
    return {'eps_1': LM_GRADIENT_TOL, 'eps_2': 0.0}


def _parameter_atol(normal):
    """Return the LM parameter-error bound for a linear model.

    Parameters
    ----------
    normal : numpy.ndarray
        Observed-data normal matrix ``X^T P X``, shape ``(3, 3)``.

    Returns
    -------
    float
        ``2 * LM_GRADIENT_TOL / lambda_min(normal)`` in parameter units:
        ``theta - theta* = normal^-1 g`` with ``||g|| < LM_GRADIENT_TOL``;
        the factor 2 covers roundoff in the gradient and closed form.
    """
    return 2 * LM_GRADIENT_TOL / np.linalg.eigvalsh(normal)[0]


def test_fixture_masks_are_staggered():
    estimator = _estimator('identity', 'jac_fun')
    assert estimator.num_data_total == NUM_OBSERVED
    masks = estimator.x_masks
    np.testing.assert_array_equal(
        masks[0], [[True, False], [True, True], [True, False],
                   [True, True], [False, True]])
    np.testing.assert_array_equal(
        masks[1], [[True, True], [False, True], [True, True],
                   [False, True]])


@pytest.mark.parametrize('weight_key', list(WEIGHTS))
@pytest.mark.parametrize('sensitivity',
                         ['jac_fun', 'finite_difference', 'model_output'])
def test_jacobian_is_derivative_of_weighted_residuals(weight_key,
                                                      sensitivity):
    """The LM Jacobian equals the derivative of the returned residuals.

    The residuals are affine in the parameters, so ``r(theta + e_p) -
    r(theta)`` is the exact column ``p`` of their derivative. Unobserved
    entries must have zero residual and zero sensitivity.
    """
    estimator = _estimator(weight_key, sensitivity)
    base = estimator.get_objective(SEED_PARAMS, out_array=True)
    jacobian = estimator.get_gradient(SEED_PARAMS, out_array=True)
    assert jacobian.shape == (NUM_PARAMS, base.size)

    for param_index in range(NUM_PARAMS):
        shifted = SEED_PARAMS + np.eye(NUM_PARAMS)[param_index]  # unit step
        difference = (estimator.get_objective(shifted, out_array=True)
                      - base)
        # Finite-difference sensitivities of this affine model carry
        # O(eps / dx) = O(1e-10) relative roundoff (dx about 1e-6 theta).
        np.testing.assert_allclose(
            jacobian[param_index], difference, rtol=1e-6,
            atol=1e-6 * np.abs(difference).max())

    # State-major unobserved entries: early c_A at t=5, c_B at t=0.5, 2.0;
    # late c_A at t=1.6, 4.0.
    unobserved = np.concatenate(
        [mask.T.ravel() for mask in estimator.x_masks]) == 0
    assert unobserved.sum() == base.size - NUM_OBSERVED
    np.testing.assert_array_equal(base[unobserved], 0.0)
    np.testing.assert_array_equal(jacobian[:, unobserved], 0.0)


@pytest.mark.parametrize('weight_key', list(WEIGHTS))
def test_callbacks_match_observed_data_closed_form(weight_key):
    """Objective, IPOPT gradient and LM information use observed data."""
    estimator = _estimator(weight_key, 'jac_fun')
    _, y_data = _datasets()
    expected = _closed_form(weight_key, SEED_PARAMS, y_data)

    objective = estimator.get_objective(SEED_PARAMS)  # [-]
    gradient = estimator.get_gradient(SEED_PARAMS)  # [1/parameter unit]
    # [weighted residual unit / parameter unit]
    jacobian = estimator.get_gradient(SEED_PARAMS, out_array=True)

    assert objective == pytest.approx(expected['objective'],
                                      rel=ROUNDOFF_RTOL)
    np.testing.assert_allclose(gradient, expected['gradient'],
                               rtol=ROUNDOFF_RTOL,
                               atol=ROUNDOFF_RTOL
                               * np.abs(expected['gradient']).max())
    np.testing.assert_allclose(jacobian @ jacobian.T, expected['normal'],
                               rtol=ROUNDOFF_RTOL, atol=0.0)
    # Non-vacuity: unobserved regressors change the information materially.
    assert not np.allclose(expected['phantom_normal'], expected['normal'],
                           rtol=SEPARATION_MARGIN)


def test_partially_observed_rows_use_marginal_precision():
    """Correlated errors weight each row by inv(Sigma_oo), not P_oo.

    Zero-filling the unobserved state and whitening with the full root
    gives the conditional precision ``inv(Sigma)[o, o]``, five times the
    marginal ``1 / Sigma[o, o]`` for these correlated states.
    """
    covariance = WEIGHTS['correlated']  # [(mol/L)**2]
    precision_full = np.linalg.inv(covariance)  # [(L/mol)**2]
    # Conditional over marginal precision of c_A: 5 * 1 / det = 5 [-].
    assert precision_full[0, 0] * covariance[0, 0] == pytest.approx(5.0)

    estimator = _estimator('correlated', 'jac_fun')
    _, y_data = _datasets()
    expected = _closed_form('correlated', SEED_PARAMS, y_data)  # [-] etc.
    conditional = 0.0  # [-], objective of the pre-#236 weighting
    for observed, regressors, values in _observed_rows(y_data):
        residual = regressors @ SEED_PARAMS - values  # [mol/L]
        conditional += 0.5 * residual @ precision_full[
            np.ix_(observed, observed)] @ residual
    assert (abs(conditional - expected['objective'])
            > SEPARATION_MARGIN * expected['objective'])

    assert estimator.get_objective(SEED_PARAMS) == pytest.approx(
        expected['objective'], rel=ROUNDOFF_RTOL)


def test_diagonal_weights_keep_zero_filled_root_weighting():
    """Uncorrelated weights reduce exactly to the established weighting."""
    estimator = _estimator('diagonal', 'jac_fun')
    weighted = estimator.get_objective(SEED_PARAMS, out_array=True)
    zero_filled = np.concatenate([
        np.dot(raw, estimator.sigma_inv).T.ravel()
        for raw in estimator.resid_runs])
    np.testing.assert_array_equal(weighted, zero_filled)


@pytest.mark.parametrize('weight_key', list(WEIGHTS))
def test_lm_covariance_counts_observed_information_only(weight_key):
    estimator = _estimator(weight_key, 'finite_difference')
    _, y_data = _datasets()
    expected = _closed_form(weight_key, SEED_PARAMS, y_data)

    estimate, covariance, info = estimator.optimize_fn(
        optim_options=_lm_options(), verbose=False)

    assert info['stop_criterion'] == 'Small gradient'
    np.testing.assert_allclose(estimate, expected['estimate'], rtol=0.0,
                               atol=_parameter_atol(expected['normal']))
    # The LM offset changes the weighted SSR only at second order
    # (relative < 1e-12), and finite-difference sensitivities of this
    # affine model carry O(1e-9) relative roundoff.
    covariance_rtol = 1e-7  # [-]
    np.testing.assert_allclose(covariance, expected['covariance'],
                               rtol=covariance_rtol, atol=0.0)

    statistics = StatisticsClass(estimator)
    np.testing.assert_allclose(statistics.a_matrix, expected['normal'],
                               rtol=covariance_rtol, atol=0.0)
    assert statistics.dof == NUM_OBSERVED - NUM_PARAMS


def test_bootstrap_samples_draw_only_observed_residuals():
    """Resampled errors come from each state's observed residuals.

    No random seed is fixed: the assertions hold for every draw. Before
    #236 each pool also held the zero residuals of unobserved entries (see
    ``BOOTSTRAP_SAMPLES``).
    """
    estimator = _estimator('diagonal', 'jac_fun')
    estimator.optimize_fn(optim_options=_lm_options(), verbose=False)
    statistics = StatisticsClass(estimator)

    samples = statistics.get_bootsamples(BOOTSTRAP_SAMPLES)

    assert len(samples) == estimator.num_datasets
    for experiment, (mask, fitted, residual) in enumerate(zip(
            estimator.x_masks, estimator.y_model, estimator.resid_runs)):
        assert len(samples[experiment]) == BOOTSTRAP_SAMPLES
        for state in range(mask.shape[1]):
            observed = mask[:, state]
            pool = residual[observed, state]  # [mol/L]
            assert np.all(np.abs(pool) > NONZERO_RESIDUAL_FLOOR)
            for sample in samples[experiment]:
                assert sample.shape == mask.shape
                assert np.all(np.isnan(sample[~observed, state]))
                _assert_drawn_from(
                    fitted[observed, state] - sample[observed, state], pool)


def _assert_drawn_from(drawn, pool):
    """Assert that every drawn residual is a member of ``pool``.

    Parameters
    ----------
    drawn : numpy.ndarray
        Recovered resampled residuals, shape ``(n,)`` [mol/L].
    pool : numpy.ndarray
        Admissible residuals, shape ``(m,)`` [mol/L].
    """
    distance = np.abs(drawn[:, np.newaxis] - pool).min(axis=1)  # [mol/L]
    assert distance.max() <= DRAW_MATCH_ATOL


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
        The return value of ``function``.

    Notes
    -----
    ``get_bootsamples`` draws from the global generator, so the seed is set
    there and the previous generator state is restored afterwards.
    """
    previous_state = np.random.get_state()
    try:
        np.random.seed(BOOTSTRAP_SEED)
        return function(*args)
    finally:
        np.random.set_state(previous_state)


@pytest.mark.parametrize('fix_initial', [False, True])
def test_unmasked_bootstrap_draws_are_unchanged(fix_initial):
    """Without masks the samples equal the pre-#236 expression bit for bit.

    The expected samples re-run the earlier implementation,
    ``y[fi:] - np.random.choice(res[fi:], size=(n, len(res[fi:])))`` per
    experiment and state, with the first row reinserted for
    ``fix_initial``, from the same seed.
    """
    times = {'early': np.array([0.5, 1.0, 2.0, 3.5, 5.0]),
             'late': np.array([0.8, 1.6, 2.5, 4.0])}  # [s]
    # Fixed asymmetric errors, columns (c_A, c_B) [mol/L].
    errors = {'early': np.array([[0.30, -0.10], [-0.45, 0.25],
                                 [0.20, -0.30], [0.55, 0.05],
                                 [-0.35, 0.40]]),
              'late': np.array([[-0.35, 0.40], [0.15, -0.20],
                                [0.10, 0.05], [-0.05, -0.25]])}
    y_data = {name: two_state_model(TRUE_PARAMS, times[name]) + errors[name]
              for name in times}  # [mol/L]
    estimator = ParameterEstimation(two_state_model, SEED_PARAMS, times,
                                    y_data, jac_fun=two_state_jacobian)
    estimator.optimize_fn(optim_options=_lm_options(), verbose=False)
    statistics = StatisticsClass(estimator)
    assert all(mask is None for mask in estimator.x_masks)

    def earlier_bootsamples():
        """Return samples built by the pre-#236 expression.

        Returns
        -------
        list of list of numpy.ndarray
            One list per experiment, in ``x_data`` order, of
            ``BOOTSTRAP_SAMPLES`` arrays of shape ``(n_times, n_states)``
            [mol/L], columns (c_A, c_B).
        """
        samples = []
        for residual, fitted in zip(statistics.residuals,
                                    statistics.y_nominal):
            states = []
            for res, y in zip(residual.T, fitted.T):
                resid = res[fix_initial:]
                boots = np.random.choice(
                    resid, size=(BOOTSTRAP_SAMPLES, len(resid)), replace=True)
                generated = y[fix_initial:] - boots
                if fix_initial:
                    generated = np.insert(generated, 0, y[0], axis=1)
                states.append(generated)
            samples.append([np.column_stack(columns)
                            for columns in zip(*states)])
        return samples

    expected = _seeded(earlier_bootsamples)
    samples = _seeded(statistics.get_bootsamples, BOOTSTRAP_SAMPLES,
                      fix_initial)

    for experiment_expected, experiment_samples in zip(expected, samples):
        assert len(experiment_samples) == BOOTSTRAP_SAMPLES
        for expected_sample, sample in zip(experiment_expected,
                                           experiment_samples):
            np.testing.assert_array_equal(sample, expected_sample)


def test_masked_bootstrap_keeps_first_row_with_fix_initial():
    """``fix_initial`` keeps row 0 fitted and resamples later observations.

    Row 0 of ``early`` observes only c_A and of ``late`` both states. Its
    fitted values (NaN where unobserved) are kept, and rows 1 onward draw
    only from the observed residuals of rows 1 onward.
    """
    estimator = _estimator('diagonal', 'jac_fun')
    estimator.optimize_fn(optim_options=_lm_options(), verbose=False)
    statistics = StatisticsClass(estimator)

    samples = statistics.get_bootsamples(BOOTSTRAP_SAMPLES, True)

    for mask, fitted, residual, experiment_samples in zip(
            estimator.x_masks, estimator.y_model, estimator.resid_runs,
            samples):
        for state in range(mask.shape[1]):
            later = mask[1:, state]  # observed rows 1 onward
            pool = residual[1:][later, state]  # [mol/L]
            if mask[0, state]:
                # Row 0's own residual is excluded from the pool.
                assert (np.abs(pool - residual[0, state]).min()
                        > NONZERO_RESIDUAL_FLOOR)
            for sample in experiment_samples:
                np.testing.assert_array_equal(sample[0, state],
                                              fitted[0, state])
                assert np.all(np.isnan(sample[1:][~later, state]))
                _assert_drawn_from(
                    fitted[1:][later, state] - sample[1:][later, state],
                    pool)


def mcr_model(params, time_s, reord_sens=False):
    """Return species and tracer concentrations for an MCR fit.

    Parameters
    ----------
    params : numpy.ndarray
        Shape ``(1,)``: first-order A -> B rate constant [1/s].
    time_s : numpy.ndarray
        Times, shape ``(n_times,)`` [s].
    reord_sens : bool, optional
        Keyword passed by ``MultipleCurveResolution``; unused.

    Returns
    -------
    numpy.ndarray
        Shape ``(n_times, 3)``: A, B (unit initial concentration) and a
        tracer ``0.3 + 0.1 t`` [mol/L].
    """
    reactant = np.exp(-params[0] * time_s)  # [mol/L]
    return np.column_stack((reactant, 1.0 - reactant, 0.3 + 0.1 * time_s))


def test_mcr_bootstrap_masks_non_spectral_columns():
    """Multivariate curve resolution masks map onto its residual columns.

    Spectra are recorded at every time; the tracer only at three of them,
    so its residual column is zero at the other rows and must be excluded
    from its pool, while spectral channels resample their own residuals.
    """
    time_s = np.array([0.5, 1.0, 2.0, 3.5, 5.0])  # [s]
    tracer_rows = [0, 2, 4]
    rate = np.array([0.3])  # [1/s]
    absorptivity = np.array([[1.2, 0.4], [0.3, 0.9]])  # [L/mol], rows A, B
    concentrations = mcr_model(np.array([0.4]), time_s)  # [mol/L]
    spectra = concentrations[:, :2] @ absorptivity + np.array([
        [0.020, -0.010], [-0.030, 0.025], [0.010, -0.020],
        [0.035, 0.005], [-0.020, 0.030]])  # [-], absorbance
    tracer = (concentrations[tracer_rows, 2]
              + np.array([0.010, -0.020, 0.015]))  # [mol/L]
    estimator = MultipleCurveResolution(
        mcr_model, rate,
        {'run': {'spectra': time_s, 'non_spectra': time_s[tracer_rows]}},
        {'run': {'spectra': spectra, 'non_spectra': tracer}},
        measured_ind={'spectra': [0, 1], 'non_spectra': [2]},
        name_states=['A', 'B', 'tracer'])
    # Reporting at the seed is enough to populate the bootstrap inputs.
    estimator.optimize_fn(optim_options={'max_fun_eval': 0}, verbose=False)
    statistics = StatisticsClass(estimator)

    samples = statistics.get_bootsamples(BOOTSTRAP_SAMPLES)

    residual = estimator.resid_runs[0]  # [-] spectra, [mol/L] tracer
    fitted = estimator.y_model[0]
    tracer_observed = np.isin(np.arange(time_s.size), tracer_rows)
    observed = np.column_stack((np.ones((time_s.size, 2), dtype=bool),
                                tracer_observed))
    assert np.all(np.abs(residual[observed]) > NONZERO_RESIDUAL_FLOOR)
    assert len(samples[0]) == BOOTSTRAP_SAMPLES
    for sample in samples[0]:
        assert sample.shape == residual.shape
        assert np.all(np.isnan(sample[~tracer_observed, 2]))
        for column in range(residual.shape[1]):
            rows = observed[:, column]
            _assert_drawn_from(fitted[rows, column] - sample[rows, column],
                               residual[rows, column])


def test_caller_covariance_mutation_does_not_change_weighting():
    """The estimator keeps its own copy of ``weight_matrix``.

    Marginal precision roots are built on first use; they must come from
    the covariance given at construction even if the caller's array is
    later modified.
    """
    caller_covariance = WEIGHTS['correlated'].copy()  # [(mol/L)**2]
    estimator = ParameterEstimation(
        two_state_model, SEED_PARAMS, *_datasets(),
        jac_fun=two_state_jacobian, weight_matrix=caller_covariance)
    caller_covariance[:] = np.diag([4.0, 1.0]) * VARIANCE_SCALE

    objective = estimator.get_objective(SEED_PARAMS)  # [-]

    reference = _estimator('correlated', 'jac_fun')
    assert objective == reference.get_objective(SEED_PARAMS)
    _, y_data = _datasets()
    assert objective == pytest.approx(
        _closed_form('correlated', SEED_PARAMS, y_data)['objective'],
        rel=ROUNDOFF_RTOL)


def test_bootstrap_fits_reproduce_single_observation_per_state():
    """``bootstrap_params`` refits observed-only resampled datasets.

    With one observation per state and experiment, every observed-only draw
    returns that state's own residual, so each bootstrap dataset equals the
    original data and each refit returns the closed-form estimate. Drawing
    the zero residual of an unobserved entry (probability 1/2 per state and
    experiment before #236) would move the estimate.
    """
    times = {'early': (np.array([1.0]), np.array([3.0])),
             'late': (np.array([2.0]), np.array([0.5]))}  # [s]
    errors = {'early': (np.array([0.3]), np.array([-0.2])),
              'late': (np.array([-0.1]), np.array([0.25]))}  # [mol/L]
    x_data = {name: list(state_times) for name, state_times in times.items()}
    y_data = {name: [two_state_model(TRUE_PARAMS, state_times[state])[:, state]
                     + errors[name][state] for state in range(2)]
              for name, state_times in times.items()}  # [mol/L]
    estimator = ParameterEstimation(
        two_state_model, SEED_PARAMS, x_data, y_data,
        jac_fun=two_state_jacobian)

    # Closed-form least squares on the four observations (identity weights).
    regressors = np.array([
        _regressors(1.0)[0], _regressors(3.0)[1],
        _regressors(2.0)[0], _regressors(0.5)[1]])
    values = np.array([y_data['early'][0][0], y_data['early'][1][0],
                       y_data['late'][0][0], y_data['late'][1][0]])
    normal = regressors.T @ regressors
    expected = np.linalg.solve(normal, regressors.T @ values)
    assert np.all(np.abs(regressors @ expected - values)
                  > NONZERO_RESIDUAL_FLOOR)

    estimate, _, _ = estimator.optimize_fn(optim_options=_lm_options(),
                                           verbose=False)
    statistics = StatisticsClass(estimator)
    # Eight refits: before #236 all would reproduce the data with
    # probability 16**-8, about 2e-10.
    num_fits = 8  # [-]
    boot_params = statistics.bootstrap_params(num_samples=num_fits)

    atol = _parameter_atol(normal)  # parameter units
    np.testing.assert_allclose(estimate, expected, rtol=0.0, atol=atol)
    np.testing.assert_allclose(
        boot_params, np.tile(expected, (num_fits, 1)), rtol=0.0, atol=atol)
