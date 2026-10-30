"""Regressions for StatisticsClass.bootstrap_confid and plot_boots (#400).

``bootstrap_confid`` draws parameter vectors from ``N(params_convg,
covar_params)`` and evaluates the model at each on an isolated estimator;
``plot_boots`` plots the residual-bootstrap datasets of one experiment
against its model grid. Neither may modify the fitted estimator.

Fixtures:

* two-parameter first-order decay ``c = c0 * exp(-k t)`` (``c0`` [mol/L],
  ``k`` [1/s]) observed in two experiments with different grids (4 and 6
  samples) and fixed asymmetric errors [mol/L];
* a three-state decay ``A -> B`` (``c_A``, ``c_B``, ``c_A + c_B``)
  [mol/L] for the subplot layout;
* the ``Experiment``-object estimator of ``test_statistics_experiment_objects``
  (#346): unequal and replicate grids, a missing observation, and a
  measurement absent from one experiment.

Draws use NumPy's global generator at fixed seeds, restored afterwards.
Expected values are closed-form model evaluations and statistical bounds.
The 4000-draw statistical test dominates the run time (about a second).
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

from PharmaPy.ParamEstim import ParameterEstimation
from PharmaPy.StatsModule import StatisticsClass
from test_statistics_bootstrap_state import _assert_same, _state
import test_statistics_experiment_objects as experiment_fixture

pytestmark = pytest.mark.unit

TRUE_PARAMS = np.array([1.0, 0.5])  # [mol/L], [1/s]
SEED_PARAMS = np.array([0.8, 0.3])  # [mol/L], [1/s]
TIMES_S = {'short': np.array([0.2, 0.8, 1.5, 2.5]),
           'long': np.array([0.5, 1.5, 3.0, 4.5, 6.0, 8.0])}  # [s]
NOISE_MOL_L = {'short': np.array([0.021, -0.013, 0.017, -0.024]),
               'long': np.array([-0.011, 0.026, -0.019, 0.008, 0.012,
                                 -0.006])}  # [mol/L]
SEED = 400  # [-], arbitrary fixed generator seed
NUM_SMALL = 7  # [-], draws for value checks
NUM_LARGE = 4000  # [-], draws for the statistical check
# Statistical bounds at NUM_LARGE draws, five standard errors each
# (two-sided probability of a false failure below 1e-6 per bound):
# sample mean of parameter i: SE = sqrt(var_i / N);
# sample variance: relative SE = sqrt(2 / N) ~ 0.022;
# sample correlation: SE ~ (1 - rho**2) / sqrt(N) <= 0.016.
NUM_STANDARD_ERRORS = 5.0  # [-]
# Closed-form model values against PharmaPy evaluations of the same
# exponentials: double round-off only.
VALUE_RTOL = 1e-12  # [-]
# Drawn parameters against the same standard normals mapped by hand: a few
# floating operations, amplified at most by the covariance condition number
# (about 7 for the decay fixture) in the Mahalanobis check.
DRAW_RTOL = 1e-10  # [-]


def _decay(params, time_s):
    """Return the two-parameter first-order decay.

    Parameters
    ----------
    params : numpy.ndarray
        ``[c0, k]``, shape ``(2,)`` [mol/L], [1/s].
    time_s : numpy.ndarray
        Times, shape ``(n_times,)`` [s].

    Returns
    -------
    numpy.ndarray
        Concentrations ``c0 * exp(-k t)``, shape ``(n_times,)`` [mol/L].
    """
    return params[0] * np.exp(-params[1] * time_s)


def _three_states(params, time_s):
    """Return ``c_A``, ``c_B`` and their sum for ``A -> B``.

    Parameters
    ----------
    params : numpy.ndarray
        ``[c0, k]``, shape ``(2,)`` [mol/L], [1/s].
    time_s : numpy.ndarray
        Times, shape ``(n_times,)`` [s].

    Returns
    -------
    numpy.ndarray
        Shape ``(n_times, 3)``: ``c_A``, ``c_B``, ``c_A + c_B`` [mol/L].
    """
    conc_a = _decay(params, time_s)  # [mol/L]
    conc_b = params[0] - conc_a  # [mol/L]
    return np.column_stack((conc_a, conc_b, conc_a + conc_b))


# Experiment layouts of the decay fixture: one experiment given as plain
# arrays ('long' only), or both experiments as dictionaries.
LAYOUTS = ('single', 'two')


def _layout_grids(layout):
    """Return the model grids of a decay-fixture layout.

    Parameters
    ----------
    layout : str
        ``'single'`` or ``'two'``.

    Returns
    -------
    list of numpy.ndarray
        Times per experiment in ``x_data`` order [s].
    """
    if layout == 'single':
        return [TIMES_S['long']]
    return list(TIMES_S.values())


def _fitted_decay(model=_decay, layout='two'):
    """Fit the decay fixture.

    Parameters
    ----------
    model : callable, optional
        Model ``f(params, time_s)`` [mol/L].
    layout : str, optional
        ``'two'``: both experiments as dictionaries; ``'single'``: the
        ``'long'`` experiment as plain ``x_data``/``y_data`` arrays.

    Returns
    -------
    StatisticsClass
        Statistics of the LM fit.
    """
    observed = {name: _decay(TRUE_PARAMS, time) + NOISE_MOL_L[name]
                for name, time in TIMES_S.items()}  # [mol/L]
    if layout == 'single':
        x_data, y_data = TIMES_S['long'], observed['long']
    else:
        x_data, y_data = TIMES_S, observed
    estimator = ParameterEstimation(model, SEED_PARAMS, x_data, y_data,
                                    name_params=['c0', 'k'])
    estimator.optimize_fn(verbose=False)
    return StatisticsClass(estimator)


def _fitted_three_states():
    """Fit the three-state ``A -> B`` fixture on both experiments.

    Returns
    -------
    StatisticsClass
        Statistics of the LM fit; three measured columns [mol/L].
    """
    # Column-wise noise scale [-] so the three columns carry distinct
    # synthetic errors from the same fixed sequences.
    column_noise_scale = np.array([1.0, -0.8, 0.5])  # [-]
    observed = {name: _three_states(TRUE_PARAMS, time)
                + np.outer(NOISE_MOL_L[name], column_noise_scale)
                for name, time in TIMES_S.items()}  # [mol/L]
    estimator = ParameterEstimation(_three_states, SEED_PARAMS, TIMES_S,
                                    observed, name_params=['c0', 'k'])
    estimator.optimize_fn(verbose=False)
    return StatisticsClass(estimator)


def _seeded(function, *args, **kwargs):
    """Call ``function`` with NumPy's global generator seeded at ``SEED``.

    Parameters
    ----------
    function : callable
        Function drawing from ``numpy.random``.
    *args, **kwargs
        Arguments of ``function``.

    Returns
    -------
    object
        Return value of ``function``; the previous generator state is
        restored afterwards.
    """
    previous_state = np.random.get_state()
    try:
        np.random.seed(SEED)
        return function(*args, **kwargs)
    finally:
        np.random.set_state(previous_state)


@pytest.fixture(autouse=True)
def _close_figures():
    """Close the figures each test creates.

    Yields
    ------
    None
    """
    yield
    plt.close('all')


@pytest.mark.parametrize("layout", LAYOUTS)
def test_bootstrap_confid_draws_and_predictions(layout):
    statistics = _fitted_decay(layout=layout)
    grids = _layout_grids(layout)
    estimator = statistics.inst
    params = estimator.params_convg  # [mol/L], [1/s]
    covar = estimator.covar_params  # [mol**2/L**2, mol/L/s; ..., 1/s**2]
    estimator_before = _state(estimator)
    statistics_before = _state(statistics)

    samples, predictions, fig, axis = _seeded(
        statistics.bootstrap_confid, NUM_SMALL, plot=False)

    assert fig is None and axis is None
    assert samples.shape == (NUM_SMALL, 2)
    z_draws = _seeded(np.random.standard_normal, (NUM_SMALL, 2))  # [-]
    deviations = samples - params  # [mol/L], [1/s]
    # Any square-root factor F of covar maps z to F z, which keeps the
    # Mahalanobis norm: d^T covar^-1 d = z^T z.
    mahalanobis = np.einsum('si,ij,sj->s', deviations, np.linalg.inv(covar),
                            deviations)  # [-]
    np.testing.assert_allclose(mahalanobis, np.sum(z_draws ** 2, axis=1),
                               rtol=DRAW_RTOL)
    # Lower-triangular factor: the first parameter uses only z_1.
    np.testing.assert_allclose(deviations[:, 0],
                               np.sqrt(covar[0, 0]) * z_draws[:, 0],
                               rtol=DRAW_RTOL)

    assert len(predictions) == NUM_SMALL
    for sample, prediction in zip(samples, predictions):
        assert len(prediction) == len(grids)
        for time, values in zip(grids, prediction):
            assert values.shape == (len(time), 1)
            np.testing.assert_allclose(values[:, 0], _decay(sample, time),
                                       rtol=VALUE_RTOL)

    _assert_same(estimator_before, _state(estimator))
    _assert_same(statistics_before, _state(statistics))


def test_bootstrap_confid_diagonal_covariance_is_exact():
    statistics = _fitted_decay()
    params = statistics.inst.params_convg  # [mol/L], [1/s]
    std = np.array([0.03, 0.01])  # [mol/L], [1/s]

    samples, _, _, _ = _seeded(statistics.bootstrap_confid, NUM_SMALL,
                               plot=False, covar_params=np.diag(std ** 2))

    z_draws = _seeded(np.random.standard_normal, (NUM_SMALL, 2))  # [-]
    # theta = params + std * z: standard deviations, not variances.
    np.testing.assert_allclose(samples, params + std * z_draws,
                               rtol=DRAW_RTOL)


def test_bootstrap_confid_sample_moments():
    statistics = _fitted_decay()
    params = statistics.inst.params_convg  # [mol/L], [1/s]
    covar = statistics.inst.covar_params

    samples, _, _, _ = _seeded(statistics.bootstrap_confid, NUM_LARGE,
                               plot=False)

    std = np.sqrt(np.diag(covar))  # [mol/L], [1/s]
    mean_bound = NUM_STANDARD_ERRORS * std / np.sqrt(NUM_LARGE)
    assert np.all(np.abs(samples.mean(axis=0) - params) < mean_bound)

    sample_covar = np.cov(samples.T)
    var_bound = NUM_STANDARD_ERRORS * np.sqrt(2 / NUM_LARGE)  # [-]
    np.testing.assert_allclose(np.diag(sample_covar), np.diag(covar),
                               rtol=var_bound)
    rho = covar[0, 1] / (std[0] * std[1])  # [-]
    sample_rho = sample_covar[0, 1] / np.sqrt(
        sample_covar[0, 0] * sample_covar[1, 1])  # [-]
    rho_bound = NUM_STANDARD_ERRORS * (1 - rho ** 2) / np.sqrt(NUM_LARGE)
    # The fixture's parameters are correlated (rho ~ 0.73), so the
    # off-diagonal term is exercised rather than vacuous; 0.5 is a guard,
    # far from both 0 and the fixture value.
    assert abs(rho) > 0.5
    assert abs(sample_rho - rho) < rho_bound


@pytest.mark.parametrize("layout", LAYOUTS)
def test_bootstrap_confid_plot_shows_nominal_and_samples(layout):
    statistics = _fitted_decay(layout=layout)
    grids = _layout_grids(layout)
    params = statistics.inst.params_convg  # [mol/L], [1/s]

    samples, predictions, fig, axes = _seeded(
        statistics.bootstrap_confid, NUM_SMALL, plot=True)

    assert isinstance(fig, matplotlib.figure.Figure)
    assert axes.shape == (len(grids),)
    for ind, (axis, time) in enumerate(zip(axes, grids)):
        nominal, *sample_lines = axis.get_lines()
        assert len(sample_lines) == NUM_SMALL
        np.testing.assert_allclose(nominal.get_xdata(), time)
        np.testing.assert_allclose(nominal.get_ydata(),
                                   _decay(params, time), rtol=VALUE_RTOL)
        for line, prediction in zip(sample_lines, predictions):
            assert line.get_color() == nominal.get_color()
            np.testing.assert_allclose(line.get_xdata(), time)
            np.testing.assert_array_equal(line.get_ydata(),
                                          prediction[ind][:, 0])


def test_bootstrap_confid_experiment_objects_predict_on_model_grids():
    statistics, _, _ = experiment_fixture.fitted_statistics(True)

    samples, predictions, _, _ = _seeded(statistics.bootstrap_confid, 3,
                                         plot=False)

    for sample, prediction in zip(samples, predictions):
        for name, values in zip(experiment_fixture.GRIDS, prediction):
            grid = np.array(experiment_fixture.GRIDS[name])  # [s]
            args, kwargs = experiment_fixture.CALLBACK[name]
            expected = experiment_fixture.model(sample, grid, *args,
                                                **kwargs)
            # Both columns, also T of 'cool', which is never observed.
            assert values.shape == (len(grid), 2)
            np.testing.assert_allclose(values, expected, rtol=VALUE_RTOL)


def test_bootstrap_confid_restores_a_stateful_model():
    last_call = {}

    def recording_decay(params, time_s):
        """Return the decay and record the rate used.

        Parameters
        ----------
        params : numpy.ndarray
            ``[c0, k]``, shape ``(2,)`` [mol/L], [1/s].
        time_s : numpy.ndarray
            Times, shape ``(n_times,)`` [s].

        Returns
        -------
        numpy.ndarray
            Concentrations, shape ``(n_times,)`` [mol/L].
        """
        last_call['rate'] = params[1]  # [1/s]
        return _decay(params, time_s)

    statistics = _fitted_decay(recording_decay)

    samples, _, _, _ = _seeded(statistics.bootstrap_confid, NUM_SMALL,
                               plot=False)

    assert samples[-1, 1] != statistics.inst.params_convg[1]
    assert last_call['rate'] == statistics.inst.params_convg[1]


def test_bootstrap_confid_failure_propagates_and_restores_the_model():
    last_call = {}
    # Admissible-rate bound [1/s] just above the fitted rate (~0.51 1/s),
    # so draws with a rate standard deviation of 0.2 1/s soon exceed it.
    upper_rate = 0.6  # [1/s]

    def guarded_decay(params, time_s):
        """Return the decay, rejecting rates above ``upper_rate``.

        Parameters
        ----------
        params : numpy.ndarray
            ``[c0, k]``, shape ``(2,)`` [mol/L], [1/s].
        time_s : numpy.ndarray
            Times, shape ``(n_times,)`` [s].

        Returns
        -------
        numpy.ndarray
            Concentrations, shape ``(n_times,)`` [mol/L].

        Raises
        ------
        ValueError
            If ``params[1] > upper_rate``.
        """
        last_call['rate'] = params[1]  # [1/s]
        if params[1] > upper_rate:
            raise ValueError(f"rate {params[1]:.3f} 1/s above validity")
        return _decay(params, time_s)

    statistics = _fitted_decay(guarded_decay)
    before = _state(statistics.inst)
    wide_covar = np.diag([0.01, 0.2]) ** 2  # [mol**2/L**2], [1/s**2]

    with pytest.raises(ValueError, match="above validity"):
        _seeded(statistics.bootstrap_confid, 50, plot=False,
                covar_params=wide_covar)

    assert last_call['rate'] == statistics.inst.params_convg[1]
    _assert_same(before, _state(statistics.inst))


@pytest.mark.parametrize("unit_scale", [np.ones(2), np.array([1e-4, 1e4])])
def test_bootstrap_confid_rejects_asymmetric_covariance_in_any_units(
        unit_scale):
    """Symmetry is judged on the correlation scale, not in raw units.

    The matrix has variances 1e8 and 1e-8 and off-diagonal entries 0.9 and
    0, i.e. correlations 0.9 and 0; ``unit_scale`` rescales the parameters
    (``D C D``) to unit variances without changing that asymmetry.
    """
    statistics = _fitted_decay()
    covar = np.array([[1e8, 0.9], [0.0, 1e-8]])
    scaled = np.diag(unit_scale) @ covar @ np.diag(unit_scale)

    with pytest.raises(ValueError, match="must be symmetric; largest "
                                         "asymmetry on the correlation "
                                         "scale is 0.9"):
        statistics.bootstrap_confid(2, plot=False, covar_params=scaled)


# Four nearly collinear exponential modes, rates 0.5 * (1 + i * 0.02) [1/s].
# On the locked linux-64 environment the fitted covariance has cond ~ 7e13;
# its LU inverse is symmetric only to ~2e-8 on the correlation scale (above
# the sqrt(eps) ~ 1.5e-8 tolerance applied to user matrices), while its
# symmetric part stays positive definite with a smallest-to-largest
# eigenvalue ratio ~ 1e-14, about 15 times the n * eps ~ 9e-16 near which a
# Cholesky factorization fails. More collinear modes would push the
# asymmetry further but bring the Cholesky factorization to that limit.
COLLINEAR_RATE_STEP = 0.02  # [-]
NUM_COLLINEAR = 4  # [-]


def _collinear_modes(params, time_s):
    """Return a sum of nearly collinear first-order decays.

    Parameters
    ----------
    params : numpy.ndarray
        Mode amplitudes, shape ``(4,)`` [mol/L].
    time_s : numpy.ndarray
        Times, shape ``(n_times,)`` [s].

    Returns
    -------
    numpy.ndarray
        Concentrations, shape ``(n_times,)`` [mol/L].
    """
    rates = (TRUE_PARAMS[1]
             * (1 + COLLINEAR_RATE_STEP * np.arange(NUM_COLLINEAR)))  # [1/s]
    return np.exp(-np.outer(time_s, rates)) @ params


def test_bootstrap_confid_accepts_ill_conditioned_estimator_covariance():
    times = np.linspace(0.5, 4.5, 9)  # [s]
    noise = np.concatenate((NOISE_MOL_L['short'], NOISE_MOL_L['long'][:5]))
    amplitudes = 0.5 ** np.arange(NUM_COLLINEAR)  # [mol/L]
    estimator = ParameterEstimation(
        _collinear_modes, 0.9 * amplitudes, times,
        _collinear_modes(amplitudes, times) + noise)
    estimator.optimize_fn(verbose=False)
    covar = estimator.covar_params  # [mol**2/L**2]
    std = np.sqrt(np.diag(covar))  # [mol/L]
    # Guard: the estimator covariance is not exactly symmetric, so the
    # default path must symmetrize rather than demand exact symmetry.
    assert np.max(np.abs(covar - covar.T) / np.outer(std, std)) > 0
    statistics = StatisticsClass(estimator)
    num_draws = 8  # [-], more draws than parameters

    samples, _, _, _ = _seeded(statistics.bootstrap_confid, num_draws,
                               plot=False)

    # Recover the factor from the draws: deviations = z @ L.T.
    z_draws = _seeded(np.random.standard_normal,
                      (num_draws, NUM_COLLINEAR))  # [-]
    factor_t, *_ = np.linalg.lstsq(z_draws,
                                   samples - estimator.params_convg,
                                   rcond=None)
    factor = factor_t.T  # [mol/L]
    np.testing.assert_allclose(np.triu(factor, 1), 0.0,
                               atol=DRAW_RTOL * np.abs(factor).max())
    np.testing.assert_allclose(factor @ factor.T, (covar + covar.T) / 2,
                               rtol=0.0,
                               atol=DRAW_RTOL * np.abs(covar).max())


def test_bootstrap_confid_validates_inputs():
    statistics = _fitted_decay()

    with pytest.raises(ValueError, match="num_samples must be a positive"):
        statistics.bootstrap_confid(0, plot=False)
    with pytest.raises(ValueError, match=r"shape \(3, 3\); expected \(2, 2\)"):
        statistics.bootstrap_confid(2, plot=False, covar_params=np.eye(3))
    with pytest.raises(ValueError, match="must be symmetric"):
        statistics.bootstrap_confid(
            2, plot=False, covar_params=np.array([[1.0, 0.5], [0.0, 1.0]]))
    with pytest.raises(ValueError, match=r"diagonal \(variances\) is "
                                         r"\[1.0, 0.0\]"):
        statistics.bootstrap_confid(
            2, plot=False, covar_params=np.diag([1.0, 0.0]))
    with pytest.raises(ValueError, match="must be positive definite"):
        statistics.bootstrap_confid(
            2, plot=False, covar_params=np.array([[1.0, 2.0], [2.0, 1.0]]))


def test_plot_boots_experiment_objects():
    statistics, _, _ = experiment_fixture.fitted_statistics(True)
    estimator_before = _state(statistics.inst)
    num_samples = 5  # [-]

    for ind, name in enumerate(experiment_fixture.GRIDS):
        fig, axes, samples = _seeded(statistics.plot_boots, num_samples,
                                     ind)

        expected = _seeded(statistics.get_bootsamples, num_samples)[ind]
        assert len(samples) == num_samples
        for sample, reference in zip(samples, expected):
            np.testing.assert_array_equal(sample, reference)

        grid = np.array(experiment_fixture.GRIDS[name])  # [s]
        mask = np.array(experiment_fixture.MASKS[name], dtype=bool)
        assert axes.shape == (2,)  # columns c_P [mol/L] and T [K]
        assert list(fig.axes) == list(axes)
        for col, axis in enumerate(axes):
            lines = axis.get_lines()
            assert len(lines) == num_samples
            for line, sample in zip(lines, samples):
                np.testing.assert_array_equal(line.get_xdata(),
                                              grid[mask[:, col]])
                np.testing.assert_array_equal(line.get_ydata(),
                                              sample[mask[:, col], col])
                assert np.all(np.isfinite(line.get_ydata()))

    # 'cool' never measures T: its subplot has empty lines.
    assert all(len(line.get_xdata()) == 0 for line in axes[1].get_lines())
    _assert_same(estimator_before, _state(statistics.inst))


def test_plot_boots_removes_empty_trailing_axes():
    statistics = _fitted_three_states()

    fig, axes, samples = _seeded(statistics.plot_boots, 4, 1)

    assert samples[0].shape == (len(TIMES_S['long']), 3)
    # Three columns on a 2 x 2 grid: the fourth cell is removed.
    assert axes.shape == (3,)
    assert list(fig.axes) == list(axes)
    for col, axis in enumerate(axes):
        for line, sample in zip(axis.get_lines(), samples):
            np.testing.assert_array_equal(line.get_xdata(),
                                          TIMES_S['long'])
            np.testing.assert_array_equal(line.get_ydata(), sample[:, col])


def test_plot_boots_single_experiment():
    statistics = _fitted_decay(layout='single')
    num_samples = 5  # [-]

    fig, axes, samples = _seeded(statistics.plot_boots, num_samples, 0)

    expected = _seeded(statistics.get_bootsamples, num_samples)
    assert len(expected) == 1
    assert axes.shape == (1,)
    assert list(fig.axes) == list(axes)
    lines = axes[0].get_lines()
    assert len(lines) == num_samples
    for line, sample, reference in zip(lines, samples, expected[0]):
        np.testing.assert_array_equal(sample, reference)
        np.testing.assert_array_equal(line.get_xdata(), TIMES_S['long'])
        np.testing.assert_array_equal(line.get_ydata(), reference[:, 0])


def test_plot_boots_validates_inputs():
    statistics = _fitted_decay()

    with pytest.raises(ValueError, match=r"from 0 to 1; got 2"):
        statistics.plot_boots(3, 2)
    with pytest.raises(ValueError, match=r"from 0 to 1; got -1"):
        statistics.plot_boots(3, -1)
    with pytest.raises(ValueError, match="num_samples must be a positive"):
        statistics.plot_boots(0, 0)
