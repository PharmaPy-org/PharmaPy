"""Undefined covariance of rank-deficient or unit-scaled fits (#532).

Scope: ``ParameterEstimation.get_covariance`` and ``optimize_fn`` when the
information matrix ``J @ J.T`` of the weighted residual Jacobian is
singular or numerically singular: the solved fit is kept, the covariance
and correlations are all NaN, a ``RuntimeWarning`` names the parameters and
directions the data do not determine, and the downstream consumers
(``get_confidence_intervals``, ``plot_correlation``, ``StatisticsClass``,
``CalibrationRecord.from_estimation``) fail with an actionable
``ValueError`` instead of crashing or returning meaningless numbers. The
other two diagnostics, a non-finite Jacobian and an inversion that fails
although ``J @ J.T`` passes the conditioning test (sensitivities near the
underflow limit, and seeded near-threshold Jacobians), are covered too.
Also the scale invariance of the test (a well-conditioned fit with
parameters in very different units keeps its covariance, bit for bit as
before #532) and the final inverse of ``levenberg_marquardt``.

Fixture: the synthetic two-experiment first-order decay of
``test_records_calibration.py`` (hot/cold runs, c_A [mol/L] and T [K],
declared uncertainties), whose one-parameter optimum ``k*`` is derived here
from the closed form by root finding. The two-parameter models below are
deliberately rank deficient: a second parameter without effect, a product
``k = a * b``, a sum ``k + 1e-9 * b`` and a near-collinear pair. They are
verification fixtures, not physical kinetics. Core lane; every fit takes
milliseconds.
"""

import re
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import brentq

import test_records_calibration as case
from PharmaPy.LevMarq import levenberg_marquardt
from PharmaPy.ParamEstim import ParameterEstimation
from PharmaPy.StatsModule import StatisticsClass

pytestmark = pytest.mark.unit

EPS = np.finfo(float).eps  # [-], float64 unit roundoff
METHODS = ('LM', 'scipy')
# Seed of the second, unidentifiable parameter: nonzero, so the product
# a * b starts with a nonzero derivative in a.
SECOND_SEED = 1.0  # [-]
# Seed of b in the near-collinear pair [1/s]. From b = 1, SciPy drifts along
# the undetermined direction k - b to |b| ~ 1e7, where the NEAR_DELTA term
# is no longer negligible against k + b; from zero it stays near the start.
NEAR_SECOND_SEED = 0.0  # [1/s]
# Start of every undefined-covariance warning of get_covariance.
UNDEFINED = re.escape('The linearized parameter covariance is undefined')
# Diagnostic of the identifiability kind of that warning.
UNIDENTIFIED = r'not \(locally\) identifiable from the observed data'
# ValueError of the consumers of an undefined covariance.
CONSUMER_ERROR = r'covariance, which is undefined \(all NaN\) for this fit'
# A pair of parameters whose Jacobi-scaled sensitivity rows coincide gives
# A_s = [[1, 1], [1, 1]], whose null vector is (1, -1) / sqrt(2): written to
# the three decimals of the diagnostic, 0.707 and -0.707.
PAIR_DIRECTION = '0.707*{} - 0.707*{}'
# Fraction of the second parameter in the sum k + 1e-9 * b of the issue:
# exactly unidentifiable. The unscaled cond(J J^T), near 1e25, flags it
# too, but its null vector is almost entirely b; the Jacobi-scaled test
# reports the direction 0.707*k - 0.707*b, in which k and b, measured in
# their own standard errors, are equally undetermined.
SUM_WEIGHT = 1e-9  # [-]
# Extra time dependence [1/s] of the near-collinear pair: the sensitivity
# rows -t*c and -(1 + DELTA*t)*t*c differ by a relative DELTA*t <= 4e-10
# over the 0-4 s grid, so the smallest eigenvalue of A_s is of order
# (DELTA*t)**2 ~ 1e-20, far below eps * lambda_max ~ 4e-16.
NEAR_DELTA = 1e-10  # [1/s]
# Rate unit of the well-conditioned case: the rate is fitted in units of
# 1e-10 1/s, so its Jacobian row is 1e10 times smaller than in [1/s] and
# cond(J J^T) exceeds 1/eps while the Jacobi-scaled condition number stays
# near 2.6 (the rise and the rate correlate by about 0.44).
RATE_UNIT_SCALE = 1e10  # [-], (1e-10 1/s) per (1/s)
# The exact optimum is bracketed by rates that over- and under-predict the
# hot-run decay of the fixture data (0.1 and 2 1/s).
RATE_BRACKET = (0.1, 2.0)  # [1/s]
# Relative tolerance of an analytic-Jacobian fit to k*: 100 times the
# 1e-8 relative stopping thresholds of LM (eps_2) and SciPy (xtol, ftol).
ANALYTIC_RTOL = 1e-6  # [-]
# Relative tolerance of a finite-difference fit to k*. The default forward
# difference (step 1e-3 * |p|) of exp(-k t) has relative error at most
# 1e-3 * k * t_max / 2 = 1e-3 here; to first order it moves the stationary
# point by that fraction of ||r|| / (k ||J||) = 0.021 at k* (computed from
# the closed form), i.e. about 2e-5. 1e-4 adds headroom for the stopping
# thresholds.
FD_RTOL = 1e-4  # [-]
# The well-conditioned covariance is compared with an explicit 2x2
# inverse; both are accurate to a few ulps times cond(A_s) ~ 2.6, so 1e-12
# leaves about three orders of magnitude of headroom.
COVARIANCE_RTOL = 1e-12  # [-]
# Exactly proportional sensitivities that the rounding of J @ J.T hides:
# the columns exp(-0.5 t) and 9.5 * exp(-0.5 t) on 14 samples over 0-4 s
# (found by scanning simple grids and factors). Rounded, the smallest
# eigenvalue of the Jacobi-scaled J J^T is 6.1e-16, above eps * 2, so
# cond(A_s) evaluated from that product is 3.6e15 < 1/eps and inv(J J^T)
# returns finite variances near 2e14; the singular values of D J resolve
# the exact rank 1. (Values on x86-64 with OpenBLAS; the exact rank, and so
# the expected outcome, does not depend on the platform.)
PROPORTIONAL_TIME = np.linspace(0.0, 4.0, 14)  # [s]
PROPORTIONAL_DECAY = 0.5  # [1/s]
PROPORTIONAL_FACTOR = 9.5  # [-]
PROPORTIONAL_AMPLITUDE = 1.0  # [mol/L], generating k + 9.5 b
PROPORTIONAL_SEED = [0.3, 0.1]  # [mol/L], arbitrary start of [k, b]
BOOTSTRAP_SAMPLES = 3  # [-]
BOOTSTRAP_SEED = 20261029  # [-], arbitrary fixed NumPy seed


def exact_rate():
    """Return the exact one-parameter optimum of the fixture.

    Returns
    -------
    float
        Rate constant k* [1/s] at which the gradient of the closed-form
        weighted sum of squares vanishes.
    """
    def gradient(rate):
        """Return the closed-form gradient of the weighted sum of squares.

        Parameters
        ----------
        rate : float
            Rate constant k [1/s].

        Returns
        -------
        float
            ``sum(r * dr/dk)`` [s] of the standardized residuals.
        """
        residuals = case.standardized_residuals(rate)  # [-]
        derivatives = []  # [s]
        for experiment, column, time, _, sigma in case.OBSERVED_ENTRIES:
            initial, rise, _ = case.CALLBACK[experiment]
            # [mol/L/(1/s)]
            d_conc = -time * initial * np.exp(-rate * time)
            derivative = d_conc if column == 0 else -rise * d_conc
            derivatives.append(derivative / sigma)
        return float(residuals @ np.array(derivatives))

    return brentq(gradient, *RATE_BRACKET, xtol=EPS, rtol=4 * EPS)


K_STAR = exact_rate()  # [1/s]


def no_effect(params, time_s, initial_mol_l, rise_k_l_mol=0.0,
              temp0_k=300.0):
    """Return the decay outputs; the second parameter has no effect.

    Parameters
    ----------
    params : numpy.ndarray
        ``[k, b]``: rate constant [1/s] and an unused parameter [-].
    time_s : numpy.ndarray
        Model grid [s].
    initial_mol_l : float
        Initial concentration of A [mol/L].
    rise_k_l_mol : float, optional
        Adiabatic temperature rise per converted concentration [K*L/mol].
    temp0_k : float, optional
        Initial temperature [K].

    Returns
    -------
    numpy.ndarray
        Shape (n_times, 2): c_A [mol/L] and T [K].
    """
    return case.decay(params[:1], time_s, initial_mol_l, rise_k_l_mol,
                      temp0_k)


def product(params, time_s, initial_mol_l, rise_k_l_mol=0.0, temp0_k=300.0):
    """Return the decay outputs for the rate ``k = a * b``.

    Parameters
    ----------
    params : numpy.ndarray
        ``[a, b]``: rate factor [1/s] and dimensionless multiplier [-].
    time_s : numpy.ndarray
        Model grid [s].
    initial_mol_l : float
        Initial concentration of A [mol/L].
    rise_k_l_mol : float, optional
        Adiabatic temperature rise per converted concentration [K*L/mol].
    temp0_k : float, optional
        Initial temperature [K].

    Returns
    -------
    numpy.ndarray
        Shape (n_times, 2): c_A [mol/L] and T [K].
    """
    return case.decay(np.array([params[0] * params[1]]), time_s,
                      initial_mol_l, rise_k_l_mol, temp0_k)


def _exponent_terms(params, time_s, weight, delta):
    """Return the decay exponent's derivatives for a two-parameter rate.

    Parameters
    ----------
    params : numpy.ndarray
        ``[k, b]`` [1/s].
    time_s : numpy.ndarray
        Model grid [s].
    weight : float
        Weight of ``b`` in the rate [-].
    delta : float
        Extra time dependence of the ``b`` term [1/s].

    Returns
    -------
    exponent : numpy.ndarray
        ``k t + weight * b * t * (1 + delta * t)`` [-].
    d_exponent : numpy.ndarray
        Shape (n_times, 2): its derivatives with respect to k and b [s].
    """
    b_term = weight * time_s * (1 + delta * time_s)  # [s]
    exponent = params[0] * time_s + params[1] * b_term  # [-]
    return exponent, np.column_stack((time_s, b_term))


def make_pair(weight, delta):
    """Build a two-parameter decay and its analytic Jacobian.

    The rate exponent is ``k t + weight * b * t * (1 + delta * t)``.

    Parameters
    ----------
    weight : float
        Weight of ``b`` in the rate [-].
    delta : float
        Extra time dependence of the ``b`` term [1/s].

    Returns
    -------
    model : callable
        ``model(params, time_s, initial_mol_l, rise_k_l_mol=0.0,
        temp0_k=300.0)`` returning c_A [mol/L] and T [K], shape
        (n_times, 2).
    jacobian : callable
        Same signature; returns the state-major sensitivities, shape
        ``(2 * n_times, 2)``: rows of d c_A / d[k, b] [mol/L*s], then
        d T / d[k, b] [K*s].
    """
    def model(params, time_s, initial_mol_l, rise_k_l_mol=0.0,
              temp0_k=300.0):
        """Return c_A [mol/L] and T [K], shape (n_times, 2).

        Parameters
        ----------
        params : numpy.ndarray
            ``[k, b]`` [1/s].
        time_s : numpy.ndarray
            Model grid [s].
        initial_mol_l : float
            Initial concentration of A [mol/L].
        rise_k_l_mol : float, optional
            Adiabatic temperature rise [K*L/mol].
        temp0_k : float, optional
            Initial temperature [K].

        Returns
        -------
        numpy.ndarray
            Shape (n_times, 2).
        """
        exponent, _ = _exponent_terms(params, time_s, weight, delta)
        remaining = initial_mol_l * np.exp(-exponent)  # [mol/L]
        return np.column_stack((
            remaining, temp0_k + rise_k_l_mol * (initial_mol_l - remaining)))

    def jacobian(params, time_s, initial_mol_l, rise_k_l_mol=0.0,
                 temp0_k=300.0):
        """Return the state-major sensitivities, shape (2 * n_times, 2).

        Parameters
        ----------
        params : numpy.ndarray
            ``[k, b]`` [1/s].
        time_s : numpy.ndarray
            Model grid [s].
        initial_mol_l : float
            Initial concentration of A [mol/L].
        rise_k_l_mol : float, optional
            Adiabatic temperature rise [K*L/mol].
        temp0_k : float, optional
            Initial temperature [K]; it does not enter the derivatives.

        Returns
        -------
        numpy.ndarray
            d c_A / d[k, b] [mol/L*s] stacked over d T / d[k, b] [K*s].
        """
        exponent, d_exponent = _exponent_terms(params, time_s, weight, delta)
        remaining = initial_mol_l * np.exp(-exponent)  # [mol/L]
        d_conc = -remaining[:, np.newaxis] * d_exponent  # [mol/L*s]
        return np.vstack((d_conc, -rise_k_l_mol * d_conc))

    return model, jacobian


SCALED_SUM = make_pair(SUM_WEIGHT, 0.0)
NEAR_PAIR = make_pair(1.0, NEAR_DELTA)

# name -> model, analytic Jacobian (None: finite differences), parameter
# names, seed of the second parameter, effective rate [1/s] of the fitted
# [p0, p1], its relative tolerance [-] and the expected diagnostic
# fragment. The two finite-difference cases reproduce issue #532. The
# sum and near pairs use exact sensitivities: forward-difference noise in
# the weak b row (relative error up to about 1e-4 for the sum, whose b
# step changes the rate by only 1e-12 1/s) would separate the scaled rows
# far more than eps and make them look identifiable.
CASES = {
    'no_effect': dict(
        model=no_effect, jacobian=None, names=['k', 'b'], seed=SECOND_SEED,
        effective=lambda p: p[0], rtol=FD_RTOL,
        fragment=re.escape('(zero information): b.')),
    'product': dict(
        model=product, jacobian=None, names=['a', 'b'], seed=SECOND_SEED,
        effective=lambda p: p[0] * p[1], rtol=FD_RTOL,
        fragment=re.escape(PAIR_DIRECTION.format('a', 'b'))),
    'scaled_sum': dict(
        model=SCALED_SUM[0], jacobian=SCALED_SUM[1], names=['k', 'b'],
        seed=SECOND_SEED, effective=lambda p: p[0] + SUM_WEIGHT * p[1],
        rtol=ANALYTIC_RTOL,
        fragment=re.escape(PAIR_DIRECTION.format('k', 'b'))),
    # k + b to first order; the DELTA term shifts it by about 1e-10.
    'near_collinear': dict(
        model=NEAR_PAIR[0], jacobian=NEAR_PAIR[1], names=['k', 'b'],
        seed=NEAR_SECOND_SEED, effective=lambda p: p[0] + p[1],
        rtol=ANALYTIC_RTOL,
        fragment=re.escape(PAIR_DIRECTION.format('k', 'b'))),
}


def undefined_warnings(record):
    """Return the undefined-covariance warnings of a recorded block.

    Parameters
    ----------
    record : list of warnings.WarningMessage
        Warnings recorded by ``pytest.warns`` or ``catch_warnings``.

    Returns
    -------
    list of warnings.WarningMessage
        The ``RuntimeWarning`` entries stating that the covariance is
        undefined.
    """
    return [item for item in record
            if issubclass(item.category, RuntimeWarning)
            and re.search(UNDEFINED, str(item.message))]


def fit(name, method):
    """Fit one rank-deficient case.

    Parameters
    ----------
    name : str
        Key of ``CASES``.
    method : str
        ``'LM'`` or ``'scipy'``.

    Returns
    -------
    tuple
        ``(estimator, opt_par, covariance, info, warnings)``: the
        estimator, the returned parameters, covariance and solver
        information, and the recorded undefined-covariance warnings.
    """
    spec = CASES[name]
    estimator = ParameterEstimation(spec['model'],
                                    [case.SEED_RATE, spec['seed']],
                                    case.make_experiments(),
                                    name_params=spec['names'],
                                    jac_fun=spec['jacobian'])
    with pytest.warns(RuntimeWarning, match=UNDEFINED) as record:
        opt_par, covariance, info = estimator.optimize_fn(verbose=False,
                                                          method=method)
    return estimator, opt_par, covariance, info, \
        undefined_warnings(record)


@pytest.mark.parametrize('method', METHODS)
@pytest.mark.parametrize('name', list(CASES))
def test_rank_deficient_fit_keeps_solution_with_nan_covariance(name, method):
    spec = CASES[name]
    estimator, opt_par, covariance, info, caught = fit(name, method)

    # The solved fit is stored, not discarded or replaced by the seed.
    np.testing.assert_array_equal(estimator.params_convg, opt_par)
    assert estimator.info_opt is info
    assert estimator.opt_method == method
    assert spec['effective'](opt_par) == pytest.approx(K_STAR,
                                                       rel=spec['rtol'])

    assert covariance.shape == (2, 2)
    assert np.isnan(covariance).all()
    assert np.isnan(estimator.covar_params).all()
    assert np.isnan(estimator.correl_params).all()
    assert estimator.correl_params.shape == (2, 2)

    assert len(caught) == 1
    message = str(caught[0].message)
    assert re.search(UNIDENTIFIED, message), message
    assert re.search(spec['fragment'], message), message
    # The warning points at this call, not at PharmaPy internals.
    assert caught[0].filename == __file__

    with pytest.raises(ValueError, match=CONSUMER_ERROR):
        estimator.get_confidence_intervals(verbose=False)


@pytest.mark.parametrize('method', METHODS)
def test_proportional_rows_hidden_by_normal_matrix_rounding_are_flagged(
        method):
    def model(params, time_s):
        """Return ``(k + 9.5 b) * exp(-0.5 t)`` [mol/L].

        Parameters
        ----------
        params : numpy.ndarray
            ``[k, b]`` [mol/L].
        time_s : numpy.ndarray
            Sample times [s].

        Returns
        -------
        numpy.ndarray
            Concentration [mol/L], shape (n_times,).
        """
        amplitude = params[0] + PROPORTIONAL_FACTOR * params[1]  # [mol/L]
        return amplitude * np.exp(-PROPORTIONAL_DECAY * time_s)

    def jacobian(params, time_s):
        """Return the exact sensitivities, shape (n_times, 2) [-].

        Parameters
        ----------
        params : numpy.ndarray
            ``[k, b]`` [mol/L].
        time_s : numpy.ndarray
            Sample times [s].

        Returns
        -------
        numpy.ndarray
            Columns ``exp(-0.5 t)`` and ``9.5 * exp(-0.5 t)``.
        """
        profile = np.exp(-PROPORTIONAL_DECAY * time_s)  # [-]
        return np.column_stack((profile, PROPORTIONAL_FACTOR * profile))

    data = model([PROPORTIONAL_AMPLITUDE, 0.0], PROPORTIONAL_TIME)  # [mol/L]
    estimator = ParameterEstimation(model, PROPORTIONAL_SEED,
                                    PROPORTIONAL_TIME, data,
                                    name_params=['k', 'b'], jac_fun=jacobian)
    with pytest.warns(RuntimeWarning, match=UNDEFINED) as record:
        opt_par, covariance, _ = estimator.optimize_fn(verbose=False,
                                                       method=method)

    amplitude = opt_par[0] + PROPORTIONAL_FACTOR * opt_par[1]  # [mol/L]
    assert amplitude == pytest.approx(PROPORTIONAL_AMPLITUDE,
                                      rel=ANALYTIC_RTOL)
    assert np.isnan(covariance).all()
    caught = undefined_warnings(record)
    assert len(caught) == 1
    assert PAIR_DIRECTION.format('k', 'b') in str(caught[0].message)


def test_unit_scaled_well_conditioned_fit_keeps_covariance():
    """A rate in 1e-10 1/s units is not mistaken for unidentifiable.

    The rise [K*L/mol] and the rate are identifiable from the hot-run
    temperature and the concentrations; only the rate's unit makes
    ``J @ J.T`` badly scaled. The covariance at the returned parameters
    must equal ``s2 * inv(J J^T)`` written out from the closed form and
    inverted explicitly. SciPy's Jacobian-based ``x_scale`` lets the solver
    converge despite the scaling; LM's step test mixes the parameter units
    and stops at the seed, and the covariance path is method-independent.
    """
    def model(params, time_s, initial_mol_l, temp0_k=300.0):
        """Return c_A [mol/L] and T [K] for ``[rise, k_scaled]``.

        Parameters
        ----------
        params : numpy.ndarray
            Rise [K*L/mol], then the rate [1e-10 1/s].
        time_s : numpy.ndarray
            Model grid [s].
        initial_mol_l : float
            Initial concentration of A [mol/L].
        temp0_k : float, optional
            Initial temperature [K].

        Returns
        -------
        numpy.ndarray
            Shape (n_times, 2).
        """
        rate = np.array([params[1] / RATE_UNIT_SCALE])  # [1/s]
        return case.decay(rate, time_s, initial_mol_l, params[0], temp0_k)

    def jacobian(params, time_s, initial_mol_l, temp0_k=300.0):
        """Return the state-major sensitivities, shape (2 * n_times, 2).

        Parameters
        ----------
        params : numpy.ndarray
            Rise [K*L/mol], then the rate [1e-10 1/s].
        time_s : numpy.ndarray
            Model grid [s].
        initial_mol_l : float
            Initial concentration of A [mol/L].
        temp0_k : float, optional
            Initial temperature [K]; it does not enter the derivatives.

        Returns
        -------
        numpy.ndarray
            d c_A / d[rise, k_scaled] over d T / d[rise, k_scaled].
        """
        rise, rate = params[0], params[1] / RATE_UNIT_SCALE  # [K*L/mol, 1/s]
        remaining = initial_mol_l * np.exp(-rate * time_s)  # [mol/L]
        # [mol/L / (1e-10 1/s)]
        d_conc_rate = -time_s * remaining / RATE_UNIT_SCALE
        d_conc = np.column_stack((np.zeros_like(time_s), d_conc_rate))
        d_temp = np.column_stack((initial_mol_l - remaining,
                                  -rise * d_conc_rate))
        return np.vstack((d_conc, d_temp))

    estimator = ParameterEstimation(
        model, [case.RISE_PARAMETER, case.SEED_RATE * RATE_UNIT_SCALE],
        case.experiments_without_rise(), name_params=['rise', 'k'],
        jac_fun=jacobian)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter('always')
        opt_par, covariance, info = estimator.optimize_fn(
            verbose=False, method='scipy', optim_options={'x_scale': 'jac'})
    assert undefined_warnings(record) == []

    # Guard: an unscaled condition-number test would flag this fit.
    jac = info['jac']
    assert np.linalg.cond(jac @ jac.T) * EPS >= 1

    rise, rate = opt_par[0], opt_par[1] / RATE_UNIT_SCALE  # [K*L/mol, 1/s]
    residuals, rows = [], []  # [-], [-/K*L/mol, -/(1e-10 1/s)]
    for experiment, column, time, value, sigma in case.OBSERVED_ENTRIES:
        initial, _, temp0 = case.CALLBACK[experiment]
        remaining = initial * np.exp(-rate * time)  # [mol/L]
        d_rate = -time * remaining / RATE_UNIT_SCALE  # [mol/L/(1e-10 1/s)]
        if column == 0:
            residuals.append((remaining - value) / sigma)
            rows.append((0.0, d_rate / sigma))
        else:
            temperature = temp0 + rise * (initial - remaining)  # [K]
            residuals.append((temperature - value) / sigma)
            rows.append(((initial - remaining) / sigma,
                         -rise * d_rate / sigma))
    residuals = np.array(residuals)
    rows = np.array(rows)
    info_rise = rows[:, 0] @ rows[:, 0]
    info_cross = rows[:, 0] @ rows[:, 1]
    info_rate = rows[:, 1] @ rows[:, 1]
    determinant = info_rise * info_rate - info_cross ** 2
    dof = len(residuals) - 2  # [-]
    residual_variance = residuals @ residuals / dof  # [-]
    expected = residual_variance / determinant * np.array(
        [[info_rate, -info_cross], [-info_cross, info_rise]])

    np.testing.assert_allclose(covariance, expected, rtol=COVARIANCE_RTOL,
                               atol=0.0)
    # Compatibility contract: a defined covariance is still the expression
    # get_covariance evaluated before #532, mse * inv(J J^T), bit for bit.
    fun = info['fun']  # [-]
    dof_fit = estimator.num_data_total - estimator.num_params  # [-]
    mse = 1 / dof_fit * np.dot(fun.T, fun)  # [-]
    np.testing.assert_array_equal(
        covariance, mse * np.linalg.inv(np.dot(jac, jac.T)))
    assert np.isfinite(estimator.correl_params).all()
    # inv(J J^T) of the statistics object, without the residual variance.
    np.testing.assert_allclose(StatisticsClass(estimator).a_inv,
                               expected / residual_variance,
                               rtol=COVARIANCE_RTOL, atol=0.0)
    intervals = estimator.get_confidence_intervals(verbose=False)
    assert np.isfinite(intervals['k']).all()


def test_levenberg_marquardt_singular_final_matrix_returns_nan_covariance():
    """A variable without effect makes ``A`` singular at the solution.

    Residuals ``[x0 - 1, x0 - 2]`` do not depend on ``x1``: the damped
    steps are solvable, the least-squares optimum is ``x0 = 1.5`` and
    ``x1`` keeps its start value, but ``A = [[2, 0], [0, 0]]``.
    """
    targets = np.array([1.0, 2.0])  # [-]

    def residual(x):
        """Return ``x0 - targets`` [-], independent of ``x1``.

        Parameters
        ----------
        x : numpy.ndarray
            Decision variables, shape (2,) [-].

        Returns
        -------
        numpy.ndarray
            Shape (2,) [-].
        """
        return x[0] - targets

    def derivative(x):
        """Return the Jacobian rows ``[1, 1]`` and ``[0, 0]`` [-].

        Parameters
        ----------
        x : numpy.ndarray
            Decision variables, shape (2,) [-].

        Returns
        -------
        numpy.ndarray
            Shape (2, 2), one row per decision variable.
        """
        return np.array([[1.0, 1.0], [0.0, 0.0]])

    start = np.array([0.0, 3.0])  # [-]
    x, covar_x, output = levenberg_marquardt(start, residual, derivative,
                                             full_output=True)

    assert x[0] == pytest.approx(targets.mean(), rel=ANALYTIC_RTOL)
    assert x[1] == start[1]
    assert covar_x.shape == (2, 2)
    assert np.isnan(covar_x).all()
    assert output['num_iter'] > 0


@pytest.fixture
def unidentified():
    """Return an LM fit whose second parameter has no effect.

    Returns
    -------
    ParameterEstimation
        Fitted ``no_effect`` estimator with an all-NaN covariance.
    """
    estimator, _, _, _, _ = fit('no_effect', 'LM')
    return estimator


def test_statistics_consumers_reject_undefined_covariance(unidentified):
    statistics = StatisticsClass(unidentified)
    assert np.isnan(statistics.a_inv).all()

    with pytest.raises(ValueError, match=CONSUMER_ERROR):
        statistics.get_intervals(verbose=False)
    with pytest.raises(ValueError, match=CONSUMER_ERROR):
        statistics.confidence_regions(pd.Series(unidentified.params_convg))
    with pytest.raises(ValueError, match=CONSUMER_ERROR):
        statistics.bootstrap_confid(BOOTSTRAP_SAMPLES, plot=False)


def test_plot_correlation_rejects_undefined_correlations(unidentified):
    """Earlier releases drew an empty heatmap for all-NaN correlations."""
    with pytest.raises(ValueError, match=CONSUMER_ERROR):
        unidentified.plot_correlation()


def test_bootstrap_refits_keep_estimates_of_unidentifiable_samples(
        unidentified):
    """Each refit warns and keeps its estimates; earlier they were NaN rows.

    The bootstrap only needs the refitted parameters, which exist although
    every refit's covariance is undefined.
    """
    statistics = StatisticsClass(unidentified)
    previous_state = np.random.get_state()
    try:
        np.random.seed(BOOTSTRAP_SEED)
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter('always')
            boot_params = statistics.bootstrap_params(BOOTSTRAP_SAMPLES)
    finally:
        np.random.set_state(previous_state)

    caught = undefined_warnings(record)
    assert len(caught) == BOOTSTRAP_SAMPLES
    assert all(item.filename == __file__ for item in caught)
    assert not any('Bootstrap optimization failed' in str(item.message)
                   for item in record)
    assert boot_params.shape == (BOOTSTRAP_SAMPLES, 2)
    assert np.isfinite(boot_params).all()
    # The parameter without information never moves from its seed.
    np.testing.assert_array_equal(boot_params[:, 1], SECOND_SEED)


def test_calibration_record_rejects_undefined_covariance(unidentified):
    with pytest.warns(RuntimeWarning, match=UNIDENTIFIED):
        with pytest.raises(ValueError, match=CONSUMER_ERROR):
            case.capture(unidentified, case.make_experiments(),
                         parameter_units={'k': '1/s', 'b': '-'})
    # The estimator keeps its stored (NaN) covariance.
    assert np.isnan(unidentified.covar_params).all()


# Power-law fixture of the non-finite Jacobian: y = k * t**n on a grid that
# starts at t = 0, where dy/dn = k * t**n * ln(t) is 0 * (-inf), undefined.
POWER_TIME = np.linspace(0.0, 4.0, 9)  # [s]
POWER_SEED = [0.3, 1.2]  # k [mol/L/s**n], n [-]
# Evaluation-only LM run: no trial step, so the model is evaluated at the
# seed and the covariance path sees the NaN sensitivity directly.
EVALUATION_ONLY = {'max_fun_eval': 0}  # [-]


def test_non_finite_jacobian_is_reported_as_such():
    """A NaN sensitivity is named, not reported as non-identifiability."""
    def model(params, time_s):
        """Return ``k * t**n`` [mol/L].

        Parameters
        ----------
        params : numpy.ndarray
            ``[k, n]``: coefficient [mol/L/s**n] and exponent [-].
        time_s : numpy.ndarray
            Sample times [s].

        Returns
        -------
        numpy.ndarray
            Shape (n_times,) [mol/L].
        """
        return params[0] * time_s ** params[1]

    def jacobian(params, time_s):
        """Return d y / d[k, n], NaN for n at t = 0.

        Parameters
        ----------
        params : numpy.ndarray
            ``[k, n]``: coefficient [mol/L/s**n] and exponent [-].
        time_s : numpy.ndarray
            Sample times [s].

        Returns
        -------
        numpy.ndarray
            Shape (n_times, 2): [s**n] and [mol/L].
        """
        # [-], ln(t), undefined (NaN) at t = 0
        log_time = np.log(time_s, out=np.full_like(time_s, np.nan),
                          where=time_s > 0)
        power = time_s ** params[1]  # [s**n]
        return np.column_stack((power, params[0] * power * log_time))

    data = model(POWER_SEED, POWER_TIME)  # [mol/L]
    estimator = ParameterEstimation(model, POWER_SEED, POWER_TIME, data,
                                    name_params=['k', 'n'],
                                    jac_fun=jacobian)
    with pytest.warns(RuntimeWarning, match=UNDEFINED) as record:
        opt_par, covariance, _ = estimator.optimize_fn(
            optim_options=dict(EVALUATION_ONLY), verbose=False)

    np.testing.assert_array_equal(opt_par, POWER_SEED)
    assert np.isnan(covariance).all()
    caught = undefined_warnings(record)
    assert len(caught) == 1
    message = str(caught[0].message)
    assert 'non-finite entries for parameters n.' in message
    assert not re.search(UNIDENTIFIED, message)

    with pytest.raises(ValueError, match=CONSUMER_ERROR) as error:
        estimator.get_confidence_intervals(verbose=False)
    assert not re.search(UNIDENTIFIED, str(error.value))


# Sensitivity scales [mol/L per parameter unit] of a well-conditioned pair,
# rows scale * t and scale * 1 (Jacobi-scaled condition number about 12):
# with 1e-160 the entries of J @ J.T, of order 1e-319, are subnormal and its
# inverse overflows; with 1e-170 they underflow to zero and J @ J.T is
# exactly singular in floating point, although no row of J is zero.
UNDERFLOW_CASES = [(1e-160, 'LM'), (1e-160, 'scipy'), (1e-170, 'scipy')]
UNDERFLOW_TIME = np.linspace(0.0, 4.0, 9)  # [s]
UNDERFLOW_SEED = [0.3, 1.2]  # k [1/s], b [-], both in scaled units


@pytest.mark.parametrize('scale, method', UNDERFLOW_CASES)
def test_information_matrix_at_underflow_uses_the_inversion_fallback(
        scale, method):
    """The inversion fallback names its parameters and least direction.

    The pair is identifiable, so the conditioning test passes, but the
    computed ``J @ J.T`` cannot be inverted to valid variances. (LM is not
    used with 1e-170: its initial damping, 1e-2 * max(diag(A)), is then
    zero and its first damped solve is singular, as its docstring states.)
    """
    def model(params, time_s):
        """Return ``scale * (k t + b)`` [mol/L].

        Parameters
        ----------
        params : numpy.ndarray
            ``[k, b]`` [1/s, -].
        time_s : numpy.ndarray
            Sample times [s].

        Returns
        -------
        numpy.ndarray
            Shape (n_times,) [mol/L].
        """
        return scale * (params[0] * time_s + params[1])

    def jacobian(params, time_s):
        """Return ``scale * [t, 1]``, shape (n_times, 2).

        Parameters
        ----------
        params : numpy.ndarray
            ``[k, b]`` [1/s, -].
        time_s : numpy.ndarray
            Sample times [s].

        Returns
        -------
        numpy.ndarray
            [mol/L*s] and [mol/L].
        """
        return scale * np.column_stack((time_s, np.ones_like(time_s)))

    data = model(UNDERFLOW_SEED, UNDERFLOW_TIME)  # [mol/L]
    estimator = ParameterEstimation(model, UNDERFLOW_SEED, UNDERFLOW_TIME,
                                    data, name_params=['k', 'b'],
                                    jac_fun=jacobian)
    with pytest.warns(RuntimeWarning, match=UNDEFINED) as record:
        _, covariance, _ = estimator.optimize_fn(verbose=False,
                                                 method=method)

    assert np.isnan(covariance).all()
    caught = undefined_warnings(record)
    assert len(caught) == 1
    message = str(caught[0].message)
    assert 'passed the conditioning test' in message
    assert 'for parameters k, b.' in message
    assert 'Least-determined parameter direction' in message
    # Neither an all-zero row nor an unidentifiable pair.
    assert 'zero information' not in message
    assert not re.search(UNIDENTIFIED, message)


# Seeded near-threshold Jacobians, rows u and u + NEAR_THRESHOLD_STEP * w of
# PCG64 standard normals (bit-identical on every platform): their
# Jacobi-scaled condition number is about 3.2e15, below 1/eps = 4.5e15, so
# they pass the conditioning test. Forming J @ J.T adds rounding errors of
# order sqrt(100) * u relative to its entries, which on x86-64 with
# OpenBLAS make seed 294 invert to negative variances and seed 297 exactly
# singular (found by scanning seeds; review of #532). Other BLAS builds may
# round differently, so the test asserts the contract for either outcome.
NEAR_THRESHOLD_SEEDS = (294, 297)  # [-]
NEAR_THRESHOLD_STEP = 3e-8  # [-]
NEAR_THRESHOLD_ENTRIES = 100  # [-]
NEAR_THRESHOLD_PARAMS = [0.3, 1.0]  # [-]


@pytest.mark.parametrize('rng_seed', NEAR_THRESHOLD_SEEDS)
def test_near_threshold_fit_never_returns_invalid_variances(rng_seed):
    """A covariance is either valid, as before #532, or NaN with names."""
    generator = np.random.default_rng(rng_seed)
    base = generator.standard_normal(NEAR_THRESHOLD_ENTRIES)  # [-]
    offset = generator.standard_normal(NEAR_THRESHOLD_ENTRIES)  # [-]
    sensitivities = np.column_stack(
        (base, base + NEAR_THRESHOLD_STEP * offset))  # [-], (100, 2)

    def model(params, index):
        """Return the linear response ``S @ params`` [-].

        Parameters
        ----------
        params : numpy.ndarray
            ``[foo, bar]`` [-].
        index : numpy.ndarray
            Sample indices [-].

        Returns
        -------
        numpy.ndarray
            Shape (100,) [-].
        """
        return sensitivities @ params

    def jacobian(params, index):
        """Return the constant sensitivities ``S`` [-], shape (100, 2).

        Parameters
        ----------
        params : numpy.ndarray
            ``[foo, bar]`` [-].
        index : numpy.ndarray
            Sample indices [-].

        Returns
        -------
        numpy.ndarray
            ``S``.
        """
        return sensitivities

    grid = np.arange(NEAR_THRESHOLD_ENTRIES, dtype=float)  # [-]
    estimator = ParameterEstimation(
        model, NEAR_THRESHOLD_PARAMS, grid,
        model(NEAR_THRESHOLD_PARAMS, grid), name_params=['foo', 'bar'],
        jac_fun=jacobian)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter('always')
        _, covariance, info = estimator.optimize_fn(verbose=False,
                                                    method='scipy')
    jac = info['jac']  # [-], (2, 100)
    np.testing.assert_array_equal(jac, sensitivities.T)
    # Guard: the fixture passes the conditioning test.
    unit_rows = jac / np.linalg.norm(jac, axis=1, keepdims=True)  # [-]
    singular = np.linalg.svd(unit_rows, compute_uv=False)  # [-]
    assert (singular[0] / singular[-1]) ** 2 * EPS < 1

    try:
        variances = np.diag(np.linalg.inv(np.dot(jac, jac.T)))  # [-]
        invalid = ~np.isfinite(variances) | (variances < 0)
    except np.linalg.LinAlgError:
        invalid = np.ones(2, dtype=bool)
    caught = undefined_warnings(record)
    if invalid.any():
        assert np.isnan(covariance).all()
        assert len(caught) == 1
        message = str(caught[0].message)
        assert 'passed the conditioning test' in message
        named = [name for name, bad in zip(['foo', 'bar'], invalid) if bad]
        assert f"for parameters {', '.join(named)}." in message
        assert 'Least-determined parameter direction' in message
    else:
        assert caught == []
        assert np.all(np.diag(covariance) >= 0)
