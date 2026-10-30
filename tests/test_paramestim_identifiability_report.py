"""Parameter uncertainty and practical identifiability report (#352, AC4).

Scope: ``ParameterEstimation.get_identifiability`` and
``StatsModule.IdentifiabilityReport``: the explicit noise assumption built
from the estimator state (declared uncertainties, a supplied
``weight_matrix``, including the identity, or no weighting; relative or
absolute scale; unit caveats), the covariance, standard errors and
Student-t intervals of ``get_covariance`` and ``get_confidence_intervals``
reproduced bit for bit without changing the estimator, and withheld when
the variance scale cannot be estimated; the singular values, condition
number and collinearity index of the Jacobi-scaled Jacobian; the
two-term sensitivity error model (row-relative ``jacobian_rtol``; the
finite-difference rounding term with ``output_rtol``, whitened exactly for
diagonal and by an operator-norm bound for correlated weights;
``sensitivity_error_bounds``, ``unresolved_parameters``,
``resolution_floor``), its invariance to output offsets and robustness to
magnitudes near the floating-point limits, and the #532 criterion behind
the unresolved directions; active bounds; the convergence outcome;
``MultipleCurveResolution`` inheritance; pickling; and input validation.

Fixtures: the synthetic two-experiment first-order decay of
``test_records_calibration`` (hot/cold runs, c_A [mol/L] and T [K]), with
the closed-form models of ``test_paramestim_rank_deficient_covariance``
(no-effect parameter, product ``a * b``, sums ``k + w * b``, near-collinear
pair), variants that fit the adiabatic rise [K*L/mol] and an
initial-concentration factor [-] next to the rate [1/s] (unequal parameter
scales), the reviewers' Jacobian-accuracy probes, and the staggered MCR case
of ``test_paramestim_mcr_observation_masks``. Expected values are derived
from the closed-form sensitivities of these models or by hand, not from the
report code. They are verification fixtures, not empirical calibrations.
Core lane; every fit takes milliseconds.
"""

import copy
import dataclasses
import pickle
import re
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import test_paramestim_mcr_observation_masks as mcr_case
import test_paramestim_observation_masks as masks
import test_paramestim_rank_deficient_covariance as deficient
import test_records_calibration as case
from PharmaPy.ParamEstim import (Experiment, Measurement,
                                 MultipleCurveResolution, ParameterEstimation)
from PharmaPy.StatsModule import IdentifiabilityReport

pytestmark = pytest.mark.unit

EPS = np.finfo(float).eps  # [-], float64 machine epsilon
# Forward-difference relative step of the default dx_finitediff for
# |p| >= 1e-6 (ParameterEstimation docstring, jac_module.dx_jac_p).
DEFAULT_RELATIVE_STEP = 1e-3  # [-]
# Default output accuracy of finite-difference Jacobians: the default rtol
# of Assimulo's CVode (1e-6 in Assimulo 3.4), the accuracy the default
# relative step sqrt(1e-6) = 1e-3 is designed for.
OUTPUT_RTOL = 1e-6  # [-]
# The production steps 1e-3 * |p| divided back by |p| differ from 1e-3 by
# a few ulps, far below this tolerance.
STEP_RTOL = 1e-12  # [-]
# Analytic-Jacobian quantities against closed forms evaluated here: both
# evaluate the same exponentials in a different order (a few ulps); the
# derived covariances, norms and singular values of these well-conditioned
# (cond <= 1e3) matrices stay within 1e-12 relative.
ROUNDOFF_RTOL = 1e-12  # [-]
# Eigenvectors of the closed-form 3 x 3 scaled Gram matrix against the
# production SVD of the analytic Jacobian: both are accurate to about
# eps / (eigenvalue gap) with gaps above 1e-2 here, so 1e-10 bounds them.
EIGENVECTOR_ATOL = 1e-10  # [-]
# Components of a unit direction whose two Jacobi-scaled rows coincide:
# (1, -1) / sqrt(2) [-].
HALF_ROOT_TWO = np.sqrt(0.5)  # [-]
# Two Jacobi-scaled rows always give the Gram matrix [[1, c], [c, 1]],
# whose eigenvectors are exactly (1, +/-1) / sqrt(2) for any c; the SVD
# returns them to about eps / (s_1 - s_2), i.e. 1e-15 for these gaps above
# 1, so 1e-12 is a round-off tolerance. Exactly proportional rows (the
# product and no-effect cases) are equally accurate.
DIRECTION_ATOL = 1e-12  # [-]
# Measurement-error covariance of the weight_matrix case: the hot-run
# standard deviations of the fixture, 0.02 mol/L and 0.5 K, squared.
WEIGHT_STD = {'c_A': 0.02, 'T': 0.5}  # [mol/L], [K]
WEIGHT_MATRIX = pd.DataFrame(
    np.diag([WEIGHT_STD['c_A'] ** 2, WEIGHT_STD['T'] ** 2]),
    index=['c_A', 'T'], columns=['c_A', 'T'])  # [mol**2/L**2], [K**2]
# A supplied identity weight_matrix: errors of standard deviation 1 mol/L
# and 1 K, a statement about the data rather than an absence of weights.
IDENTITY_WEIGHT_MATRIX = pd.DataFrame(np.eye(2), index=['c_A', 'T'],
                                      columns=['c_A', 'T'])
UNCONVERGED_EVALUATIONS = 1  # [-], LM max_fun_eval that stops the fit
ALPHA_90 = 0.9  # [-], a non-default confidence level
FIT_ATTRIBUTES = ParameterEstimation._FIT_ATTRIBUTES
# Weights of b in the structurally rank-one rates k + w * b: the #532 sum
# and the reviewers' 1e-12 variant, both fitted with forward differences.
SUM_WEIGHTS = (deficient.SUM_WEIGHT, 1e-12)  # [-]
# Unit of the rescaled rate [1e-10 1/s], as in the #532 scale-invariance
# test: the rate's Jacobian row shrinks by 1e10 and its value grows by
# 1e10, which must not change the scaled analysis.
RATE_UNIT_SCALE = deficient.RATE_UNIT_SCALE  # [-]
# Rate units of the rescaling test: 1e10 and 1e-3 per (1/s), both against
# the fit in 1/s. The rescaled rate stays above the 1e-6 magnitude floor of
# the default step, so the relative step is 1e-3 in every unit.
RATE_SCALES = (RATE_UNIT_SCALE, 1e-3)  # [-]
# The unscaled and rescaled SciPy fits stop at points that agree to about
# the 1e-8 xtol/ftol thresholds, so their forward-difference Jacobians, and
# the scaled quantities derived from them, agree to well within 1e-6.
RESCALE_RTOL = 1e-6  # [-]
# Seed of the initial-concentration factor of the four-parameter case: the
# nominal initial concentrations.
CONC_FACTOR_SEED = 1.0  # [-]
# Parameter magnitude below which the default finite-difference step stops
# shrinking (jac_module.PARAMETER_MAGNITUDE_FLOOR) [parameter units].
ZERO_PARAMETER_FLOOR = 1e-6
# Effect of b [-] per unit b in the tiny-effect case: 1e-7 is ten times
# below the 1e-6 output accuracy times the 1e-3 relative step, so the
# default bound of b is about 2e4.
TINY_EFFECT = 1e-7  # [-]
# The tiny-effect fit converges to a = b = 1 within LM's 1e-8 relative
# step test; 1e-6 covers it and the derived norms and bounds.
FIT_RTOL = 1e-6  # [-]
# Declared row-relative accuracy of the probe3 analytic Jacobian.
PROBE_RTOL = 1e-3  # [-]
# Reviewer probe (probe3): sensitivity rows 1e6 * t and 1e-6 * (t + 0.05
# t**2) of a model linear in (a, b); the quadratic term is the
# perturbation that makes the rows exactly collinear. The SVD returns the
# small singular value near 8e-3 to about eps / 8e-3 relative; 1e-9 bounds
# that against sqrt(1 - c).
PROBE_RELATION_RTOL = 1e-9  # [-]
PROBE_TIMES = np.linspace(0.1, 1.0, 6)  # [s]
PROBE_LARGE = 1e6  # [output unit / (a unit * s)]
PROBE_SMALL = 1e-6  # [output unit / (b unit * s)]
PROBE_CURVATURE = 0.05  # [1/s]
# Hand-built three-parameter fit of the from_fit sign test: orthogonal
# Jacobian rows of norm sqrt(2), so inv(J J^T) = 0.5 I, and one residual of
# 0.3 over 6 observed entries (dof 3): s2 = 0.09 / 3 = 0.03, variances
# 0.015 and standard errors sqrt(0.015) = 0.12247.
SIGN_JACOBIAN = np.array([[1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                          [0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
                          [0.0, 0.0, 0.0, 0.0, 1.0, 1.0]])  # [r/p]
SIGN_RESIDUALS = np.array([0.3, 0.0, 0.0, 0.0, 0.0, 0.0])  # [r]
SIGN_ESTIMATES = np.array([-1.0, 0.0, 0.3])  # [p]
SIGN_STANDARD_ERROR = np.sqrt(0.015)  # [p]
# Two-sided 95 % Student-t quantile for 3 degrees of freedom, from
# standard t tables (t_{0.975, 3} = 3.182).
T_TABLE_975_DOF3 = 3.182  # [-]
T_TABLE_ATOL = 5e-4  # [-], the table's three decimals
# A scalar forward-difference step of the rate [1/s] for the hand-computed
# default jacobian_rtol, and one so large (relative step near 2 at k*) that
# it is not a valid relative accuracy.
SCALAR_STEP = 1e-4  # [1/s]
OVERSIZED_STEP = 1.0  # [1/s]
# Overflowing covariance of the from_fit case: rows of 1e-150 give
# J J^T = 3e-300, inverted to 3.3e299, and a residual of 1e5 over dof 2
# gives s2 = 5e9, so s2 * inv(J J^T) overflows to inf.
TINY_SENSITIVITY = 1e-150  # [r/p]
LARGE_RESIDUAL = 1e5  # [r]
# A residual whose square overflows, s2 = inf.
HUGE_RESIDUAL = 1e200  # [r]
# Offset case: a rise a (1 - exp(-k t)) [K] with a = 10 K, k = 0.4 1/s,
# sampled at 15 times, plus a fixed +/-0.02 K perturbation, fitted on
# offsets of 26.85 K (degrees Celsius of 300 K), 300 K and 3000 K.
OFFSET_TIMES = np.linspace(0.2, 3.0, 15)  # [s]
OFFSET_TRUE = (10.0, 0.4)  # [K], [1/s]
OFFSET_SEED = [8.0, 0.5]  # [K], [1/s]
OFFSET_NOISE = 0.02 * np.sin(1.7 * np.arange(15))  # [K], arbitrary
OFFSETS = (26.85, 300.0, 3000.0)  # [K]
# With output_rtol = eps the offset enters the bounds only through
# 2 eps ||y|| / (h ||J_i||) < 1e-9 (||y|| <= 1.2e4 K, h ||J_i|| >= 1e-2 K)
# against 1e-3, and through the fits, which agree to about 1e-10; 1e-6
# relative bounds both.
OFFSET_FLOOR_RTOL = 1e-6  # [-]
# Heating-rate case: 300 K baseline, 0.02 K/s over 0-10 s (21 samples),
# a fixed +/-1 mK perturbation.
HEATING_BASELINE = 300.0  # [K]
HEATING_RATE = 0.02  # [K/s]
HEATING_SEED = 0.03  # [K/s]
HEATING_TIMES = np.linspace(0.0, 10.0, 21)  # [s]
HEATING_NOISE = 1e-3 * np.sin(1.3 * np.arange(21))  # [K], arbitrary
# The production row norm is that of the forward difference of the linear
# model, which has no truncation error: it differs from the exact ||t|| by
# the rounding of 300 K outputs over the step, about eps * ||y|| / (h ||t||)
# = 2.2e-16 * 1.4e3 / (2e-5 * 27) = 6e-10 relative; 1e-8 bounds it.
FD_ROW_RTOL = 1e-8  # [-]
# Correlated-weights case: two identical columns a t + 0.22 b over t = 1,
# 2, 3 with correlation 1 - 2**-16 (exact in binary), so ||W||_2 =
# 1 / sqrt(2**-16) = 256 exactly.
COMMON_EFFECT = 0.22  # [-]
CORRELATED_TIMES = (1.0, 2.0, 3.0)  # [-]
STRONG_CORRELATION = 1 - 2.0 ** -16  # [-]
# The Cholesky root of the nearly singular covariance (cond 1.3e5) is
# accurate to about cond * eps = 3e-11 relative; 1e-9 bounds the computed
# ||W||_2 against 256.
CORRELATED_RTOL = 1e-9  # [-]
# Uncertainty multipliers that bring the weighted values near 1e-200
# (squares underflow) and 1e160 (squares overflow).
EXTREME_FACTORS = (1e200, 1e-160)  # [-]
# Sensitivity of the underflow case: inv(J J^T) ~ 2e-301 times an s2 near
# 1e-200 underflows to zero.
HUGE_SENSITIVITY = 1e150  # [output unit / parameter unit]
# Heating rate of the injection test [K/s], where the factor 2 of the
# differenced output errors decides the verdict (bound 1.29, 0.64 without
# it).
INJECTION_RATE = 0.08  # [K/s]
# The rounding bound uses ||y(p)|| while the injected errors scale with
# |y(p)| + |y(p + h)|; they differ by h * t <= 8e-5 K/s * 10 s over at least
# 300 K, a relative 3e-6, plus the 6e-10 forward-difference rounding of the
# row norm; 1e-5 bounds both.
INJECTION_RTOL = 1e-5  # [-]
# Cancellation case: model outputs 1e-14 + 1e-20 p t against data 1e4.
TINY_LEVEL = 1e-14  # [-]
TINY_SLOPE = 1e-20  # [-]
LARGE_DATA = 1e4  # [-]
# The forward difference of that linear model is exact but for rounding
# of the 1e-14 outputs over the 1e-23 output change of the step:
# eps * 1e-14 / 1e-23 = 2e-7 relative; 1e-6 bounds it.
CANCELLATION_RTOL = 1e-6  # [-]
# Staggered case without fully observed rows: c_A and c_B sampling times
# [s], fixed measurement perturbations [mol/L] and a correlated covariance
# [(mol/L)**2] with correlation 0.9.
STAGGER_TIMES_A = (1.0, 2.0, 3.0)  # [s]
STAGGER_TIMES_B = (1.5, 2.5, 3.5)  # [s]
STAGGER_ERRORS_A = np.array([0.01, -0.01, 0.005])  # [mol/L], arbitrary
STAGGER_ERRORS_B = np.array([-0.005, 0.01, 0.0])  # [mol/L], arbitrary
STAGGER_COVARIANCE = np.array([[1.0, 0.9], [0.9, 1.0]])  # [(mol/L)**2]


def decay_jacobian(params, time_s, initial_mol_l, rise_k_l_mol=0.0,
                   temp0_k=300.0):
    """Return the exact sensitivities of ``case.decay`` to the rate.

    Parameters
    ----------
    params : numpy.ndarray
        Rate constant k, shape (1,) [1/s].
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
        Shape (2 * n_times, 1): d c_A / dk [mol/L*s] stacked over
        d T / dk [K*s].
    """
    d_conc = -time_s * initial_mol_l * np.exp(-params[0] * time_s)
    return np.concatenate((d_conc, -rise_k_l_mol * d_conc))[:, np.newaxis]


def rise_first_jacobian(params, time_s, initial_mol_l, temp0_k=300.0):
    """Return the exact sensitivities of ``case.decay_rise_first``.

    Parameters
    ----------
    params : numpy.ndarray
        Rise [K*L/mol], then rate constant k [1/s].
    time_s : numpy.ndarray
        Model grid [s].
    initial_mol_l : float
        Initial concentration of A [mol/L].
    temp0_k : float, optional
        Initial temperature [K]; it does not enter the derivatives.

    Returns
    -------
    numpy.ndarray
        Shape (2 * n_times, 2): d c_A / d[rise, k] ([mol/L / (K*L/mol)],
        [mol/L*s]) stacked over d T / d[rise, k] ([mol/L], [K*s]).
    """
    remaining = initial_mol_l * np.exp(-params[1] * time_s)  # [mol/L]
    d_conc = np.column_stack((np.zeros_like(time_s), -time_s * remaining))
    d_temp = np.column_stack((initial_mol_l - remaining,
                              params[0] * time_s * remaining))
    return np.vstack((d_conc, d_temp))


def entry_sigma(weighting, column, declared):
    """Return the weight divisor of one observed entry.

    Parameters
    ----------
    weighting : {'declared_uncertainty', 'weight_matrix', 'identity'}
        Weighting of the fit.
    column : int
        0 for c_A, 1 for T.
    declared : float
        Declared standard deviation of the entry [mol/L or K].

    Returns
    -------
    float
        Standard deviation that divides the residual [mol/L or K], or 1
        for the identity weighting.
    """
    if weighting == 'declared_uncertainty':
        return declared
    if weighting == 'weight_matrix':
        return WEIGHT_STD['c_A'] if column == 0 else WEIGHT_STD['T']
    return 1.0


def closed_form_rows(params, weighting, rise_free=False):
    """Return weighted residuals and Jacobian rows of the observed entries.

    Parameters
    ----------
    params : numpy.ndarray
        ``[k]`` [1/s], or ``[rise, k]`` ([K*L/mol], [1/s]) if
        ``rise_free``.
    weighting : str
        See ``entry_sigma``.
    rise_free : bool, optional
        If True, the rise is the first parameter, shared by both runs.

    Returns
    -------
    residuals : numpy.ndarray
        Weighted residuals (model - data) / sigma, shape (7,), in
        ``case.OBSERVED_ENTRIES`` order.
    jacobian : numpy.ndarray
        Their derivatives, shape (n_params, 7).
    """
    residuals, rows = [], []
    for experiment, column, time, value, declared in case.OBSERVED_ENTRIES:
        initial, rise, temp0 = case.CALLBACK[experiment]
        if rise_free:
            rise, rate = params
        else:
            rate = params[0]
        remaining = initial * np.exp(-rate * time)  # [mol/L]
        sigma = entry_sigma(weighting, column, declared)
        if column == 0:
            model = remaining
            derivative = [-time * remaining]  # [mol/L*s]
            if rise_free:
                derivative = [0.0] + derivative
        else:
            model = temp0 + rise * (initial - remaining)  # [K]
            derivative = [rise * time * remaining]  # [K*s]
            if rise_free:
                derivative = [initial - remaining] + derivative
        residuals.append((model - value) / sigma)
        rows.append(np.array(derivative) / sigma)
    return np.array(residuals), np.array(rows).T


def fit_decay(weighting, jac_fun=decay_jacobian):
    """Fit the one-parameter decay under one weighting.

    Parameters
    ----------
    weighting : str
        See ``entry_sigma``.
    jac_fun : callable or None, optional
        Jacobian callback; None uses finite differences.

    Returns
    -------
    ParameterEstimation
        The fitted estimator.
    """
    kwargs = {}
    if weighting == 'weight_matrix':
        kwargs['weight_matrix'] = WEIGHT_MATRIX
    experiments = case.make_experiments(
        uncertain=weighting == 'declared_uncertainty')
    estimator = ParameterEstimation(case.decay, [case.SEED_RATE],
                                    experiments, name_params=['k'],
                                    jac_fun=jac_fun, **kwargs)
    estimator.optimize_fn(verbose=False)
    return estimator


def fit_rise_and_rate(jac_fun):
    """Fit the rise [K*L/mol] and rate [1/s] together.

    Parameters
    ----------
    jac_fun : callable or None
        Jacobian callback; None uses finite differences.

    Returns
    -------
    ParameterEstimation
        The fitted estimator, declared uncertainties.
    """
    estimator = ParameterEstimation(
        case.decay_rise_first, [case.RISE_PARAMETER, case.SEED_RATE],
        case.experiments_without_rise(), name_params=['rise', 'k'],
        jac_fun=jac_fun)
    estimator.optimize_fn(verbose=False)
    return estimator


def fit_deficient(name, seed=None):
    """Fit one rank-deficient case and return its warnings.

    Parameters
    ----------
    name : str
        Key of ``deficient.CASES``.
    seed : float, optional
        Seed of the second parameter; the default is the case's seed.

    Returns
    -------
    estimator : ParameterEstimation
        The fitted estimator.
    messages : list of str
        Undefined-covariance warnings of ``optimize_fn``.
    """
    spec = deficient.CASES[name]
    estimator = ParameterEstimation(
        spec['model'],
        [case.SEED_RATE, spec['seed'] if seed is None else seed],
        case.make_experiments(), name_params=spec['names'],
        jac_fun=spec['jacobian'])
    with pytest.warns(RuntimeWarning, match=deficient.UNDEFINED) as record:
        estimator.optimize_fn(verbose=False)
    return estimator, [str(item.message) for item
                       in deficient.undefined_warnings(record)]


def snapshot(estimator):
    """Deep-copy the instance dictionary of an estimator.

    Parameters
    ----------
    estimator : ParameterEstimation
        Estimator to copy.

    Returns
    -------
    dict
        Deep copy of ``vars(estimator)``.
    """
    return copy.deepcopy(vars(estimator))


def assert_same(before, after, path='estimator'):
    """Assert two snapshots are equal bit for bit, NaN included.

    Parameters
    ----------
    before, after : object
        Snapshot values (dicts, sequences, arrays, frames or scalars).
    path : str, optional
        Location for the failure message.
    """
    if isinstance(before, dict):
        assert set(before) == set(after), path
        for key in before:
            assert_same(before[key], after[key], f"{path}[{key!r}]")
    elif isinstance(before, (list, tuple)):
        assert len(before) == len(after), path
        for ind, (old, new) in enumerate(zip(before, after)):
            assert_same(old, new, f"{path}[{ind}]")
    elif isinstance(before, np.ndarray):
        np.testing.assert_array_equal(before, after, err_msg=path)
        assert before.dtype == after.dtype, path
    elif isinstance(before, pd.DataFrame):
        pd.testing.assert_frame_equal(before, after, check_exact=True)
    elif isinstance(before, float) and np.isnan(before):
        assert np.isnan(after), path
    elif callable(before) or hasattr(before, '__dict__'):
        assert type(before) is type(after), path
    else:
        assert before == after, path




def four_parameter_model(params, time_s, initial_mol_l, temp0_k=300.0):
    """Return c_A [mol/L] and T [K] for ``[b, k, rise, factor]``.

    ``b`` has no effect; ``factor`` scales the initial concentration.

    Parameters
    ----------
    params : numpy.ndarray
        Unused b [-], rate k [1/s], rise [K*L/mol], initial-concentration
        factor [-].
    time_s : numpy.ndarray
        Model grid [s].
    initial_mol_l : float
        Nominal initial concentration of A [mol/L].
    temp0_k : float, optional
        Initial temperature [K].

    Returns
    -------
    numpy.ndarray
        Shape (n_times, 2).
    """
    return case.decay(params[1:2], time_s, params[3] * initial_mol_l,
                      params[2], temp0_k)


def four_parameter_jacobian(params, time_s, initial_mol_l, temp0_k=300.0):
    """Return the exact sensitivities of ``four_parameter_model``.

    Parameters
    ----------
    params : numpy.ndarray
        b [-], k [1/s], rise [K*L/mol], factor [-].
    time_s : numpy.ndarray
        Model grid [s].
    initial_mol_l : float
        Nominal initial concentration of A [mol/L].
    temp0_k : float, optional
        Initial temperature [K]; it does not enter the derivatives.

    Returns
    -------
    numpy.ndarray
        Shape (2 * n_times, 4): d c_A / d params stacked over d T /
        d params, in [output unit / parameter unit].
    """
    _, rate, rise, factor = params
    initial = factor * initial_mol_l  # [mol/L]
    remaining = initial * np.exp(-rate * time_s)  # [mol/L]
    zeros = np.zeros_like(time_s)
    d_conc = np.column_stack((zeros, -time_s * remaining, zeros,
                              remaining / factor))
    d_temp = np.column_stack((zeros, rise * time_s * remaining,
                              initial - remaining,
                              rise * (initial - remaining) / factor))
    return np.vstack((d_conc, d_temp))


def four_parameter_rows(params):
    """Return the closed-form weighted Jacobian rows of the observed data.

    Parameters
    ----------
    params : numpy.ndarray
        b [-], k [1/s], rise [K*L/mol], factor [-].

    Returns
    -------
    numpy.ndarray
        Shape (4, 7): derivatives of (model - data) / sigma over
        ``case.OBSERVED_ENTRIES``.
    """
    _, rate, rise, factor = params
    rows = []
    for experiment, column, time, _, sigma in case.OBSERVED_ENTRIES:
        initial = factor * case.CALLBACK[experiment][0]  # [mol/L]
        remaining = initial * np.exp(-rate * time)  # [mol/L]
        if column == 0:
            row = [0.0, -time * remaining, 0.0, remaining / factor]
        else:
            row = [0.0, rise * time * remaining, initial - remaining,
                   rise * (initial - remaining) / factor]
        rows.append(np.array(row) / sigma)
    return np.array(rows).T


def rescaled_rate_model(scale):
    """Return the rise/rate model with the rate in units of ``1/scale`` 1/s.

    Parameters
    ----------
    scale : float
        Rate unit factor [-]: the fitted value is ``scale`` times the rate
        in 1/s.

    Returns
    -------
    callable
        ``model(params, time_s, initial_mol_l, temp0_k=300.0)`` for
        ``[rise, k_scaled]`` returning c_A [mol/L] and T [K].
    """
    def model(params, time_s, initial_mol_l, temp0_k=300.0):
        """Return c_A [mol/L] and T [K] for ``[rise, k_scaled]``.

        Parameters
        ----------
        params : numpy.ndarray
            Rise [K*L/mol], then the rate in units of ``1/scale`` 1/s.
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
        rate = np.array([params[1] / scale])  # [1/s]
        return case.decay(rate, time_s, initial_mol_l, params[0], temp0_k)

    return model


def decay_with_sensitivities(params, time_s, initial_mol_l,
                             rise_k_l_mol=0.0, temp0_k=300.0):
    """Return the decay outputs together with their exact sensitivities.

    Parameters
    ----------
    params : numpy.ndarray
        Rate constant k, shape (1,) [1/s].
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
    tuple of numpy.ndarray
        ``case.decay`` output and ``decay_jacobian`` output, the layout
        ``ParameterEstimation`` accepts from models that integrate their
        own sensitivities.
    """
    return (case.decay(params, time_s, initial_mol_l, rise_k_l_mol,
                       temp0_k),
            decay_jacobian(params, time_s, initial_mol_l, rise_k_l_mol,
                           temp0_k))


def two_species(params, time_s, initial_mol_l):
    """Return c_A and c_B [mol/L] of A -> B.

    Parameters
    ----------
    params : numpy.ndarray
        Rate constant k, shape (1,) [1/s].
    time_s : numpy.ndarray
        Model grid [s].
    initial_mol_l : float
        Initial concentration of A [mol/L].

    Returns
    -------
    numpy.ndarray
        Shape (n_times, 2).
    """
    remaining = initial_mol_l * np.exp(-params[0] * time_s)  # [mol/L]
    return np.column_stack((remaining, initial_mol_l - remaining))


def hot_experiment(t_units):
    """Build the hot run with a chosen temperature unit label.

    Parameters
    ----------
    t_units : str or None
        Declared unit of the temperature measurement.

    Returns
    -------
    dict
        ``{'hot': Experiment}`` with c_A [mol/L] and T.
    """
    return {'hot': Experiment(
        {'c_A': Measurement(0, case.HOT_CA_TIME, case.HOT_CA,
                            units='mol/L'),
         'T': Measurement(1, case.HOT_T_TIME, case.HOT_T, units=t_units)},
        args=(case.CALLBACK['hot'][0],),
        kwargs={'rise_k_l_mol': case.CALLBACK['hot'][1]})}


def sign_report(**options):
    """Build the hand-made three-parameter report of the sign tests.

    Parameters
    ----------
    **options
        Overrides of the ``from_fit`` keywords.

    Returns
    -------
    IdentifiabilityReport
        Report of ``SIGN_JACOBIAN`` at ``SIGN_ESTIMATES``.
    """
    arguments = dict(noise_model='identity', measurement_units=None,
                     jacobian_source='jac_fun', jacobian_rtol=EPS)
    arguments.update(options)
    return IdentifiabilityReport.from_fit(
        SIGN_JACOBIAN, SIGN_RESIDUALS, SIGN_ESTIMATES, ['x', 'y', 'z'],
        SIGN_JACOBIAN.shape[1], **arguments)


def closed_form_prediction_norm(rate, weighting):
    """Return the norm of the weighted closed-form predictions.

    Parameters
    ----------
    rate : float
        Rate constant k [1/s].
    weighting : str
        See ``entry_sigma``.

    Returns
    -------
    float
        ``sqrt(sum((model / sigma)**2))`` over the observed entries
        [weighted residual unit].
    """
    weighted = [case.closed_form(rate, experiment, time)[column]
                / entry_sigma(weighting, column, declared)
                for experiment, column, time, _, declared
                in case.OBSERVED_ENTRIES]
    return float(np.linalg.norm(weighted))


@pytest.mark.parametrize('include_mse', [True, False])
@pytest.mark.parametrize('weighting', ['declared_uncertainty',
                                       'weight_matrix', 'identity'])
def test_one_parameter_uncertainty_matches_covariance_and_closed_form(
        weighting, include_mse):
    estimator = fit_decay(weighting)
    report = estimator.get_identifiability(alpha=ALPHA_90,
                                           include_mse=include_mse)

    assert report.noise_model == weighting
    assert report.parameter_names == ('k',)
    assert report.variance_scale_estimated is include_mse
    assert report.jacobian_source == 'jac_fun'
    np.testing.assert_array_equal(report.jacobian_rtol, [EPS])
    assert report.degrees_of_freedom == len(case.OBSERVED_ENTRIES) - 1
    np.testing.assert_array_equal(report.estimates, estimator.params_convg)
    assert report.active_bounds is None  # LM reports no active set

    # Bit for bit the estimator's own covariance and intervals.
    np.testing.assert_array_equal(
        report.covariance, estimator.get_covariance(include_mse=include_mse))
    np.testing.assert_array_equal(report.correlation,
                                  estimator.correl_params)
    intervals = estimator.get_confidence_intervals(
        alpha=ALPHA_90, include_mse=include_mse, verbose=False,
        set_self=False)
    np.testing.assert_array_equal(report.confidence_intervals[0],
                                  intervals['k'])

    # Independently, from the closed-form sensitivities at the same rate.
    rate = estimator.params_convg[0]  # [1/s]
    residuals, jacobian = closed_form_rows(estimator.params_convg, weighting)
    information = float((jacobian @ jacobian.T)[0, 0])  # [s**2/weight**2]
    mse = residuals @ residuals / (len(residuals) - 1)  # [weight**2]
    variance = (mse if include_mse else 1.0) / information  # [1/s**2]
    std_error = np.sqrt(variance)  # [1/s]
    quantile = stats.t.ppf((1 + ALPHA_90) / 2, len(residuals) - 1)  # [-]
    assert report.information_matrix[0, 0] == pytest.approx(
        information, rel=ROUNDOFF_RTOL)
    assert report.residual_mean_square == pytest.approx(mse,
                                                        rel=ROUNDOFF_RTOL)
    assert report.covariance[0, 0] == pytest.approx(variance,
                                                    rel=ROUNDOFF_RTOL)
    assert report.standard_errors[0] == pytest.approx(std_error,
                                                      rel=ROUNDOFF_RTOL)
    assert report.relative_standard_errors[0] == pytest.approx(
        std_error / rate, rel=ROUNDOFF_RTOL)
    np.testing.assert_allclose(
        report.confidence_intervals[0],
        [rate - quantile * std_error, rate + quantile * std_error],
        rtol=ROUNDOFF_RTOL)
    assert report.t_ratios[0] == pytest.approx(rate / std_error,
                                               rel=ROUNDOFF_RTOL)
    assert bool(report.significant[0]) is bool(rate / std_error > quantile)
    # [s**2 / weight**2] divided by the variance scale [weight**2] or 1
    expected_fisher = information / (mse if include_mse else 1.0)
    assert report.fisher_information[0, 0] == pytest.approx(
        expected_fisher, rel=ROUNDOFF_RTOL)

    # Error model: an analytic Jacobian row is exact to rounding, so its
    # bound is eps; diagonal weights whiten the output errors exactly.
    np.testing.assert_array_equal(report.sensitivity_error_bounds, [EPS])
    assert report.resolution_floor == pytest.approx(EPS, rel=ROUNDOFF_RTOL)
    assert report.finite_difference_steps is None
    assert report.output_rtol == OUTPUT_RTOL
    norm = closed_form_prediction_norm(rate, weighting)  # [weight]
    assert report.output_error_norm == pytest.approx(OUTPUT_RTOL * norm,
                                                     rel=ROUNDOFF_RTOL)
    assert report.to_frame()['active_bound'].isna().all()  # LM: no mask
    # A single informed parameter: the scaled Jacobian is a unit row.
    np.testing.assert_allclose(report.singular_values, [1.0],
                               rtol=ROUNDOFF_RTOL)
    assert report.identifiable is True
    assert report.unresolved_directions == ()
    assert report.weakest_direction.as_dict() == {'k': 1.0}


@pytest.mark.parametrize('weighting, include_mse, fragments', [
    ('declared_uncertainty', True,
     ['Measurement.uncertainty', 'relative weights', 'absolute_sigma=False',
      'reduced chi-square']),
    ('declared_uncertainty', False,
     ['Measurement.uncertainty', 'treated as absolute',
      'absolute_sigma=True', 'its expected value is 1']),
    ('weight_matrix', True,
     ['proportional to weight_matrix', 'independent between samples',
      'treated as relative']),
    ('weight_matrix', False,
     ['proportional to weight_matrix', 'absolute measurement-error '
      'covariance']),
    ('identity', True,
     ['No measurement uncertainty or weight_matrix', 'one common variance',
      'different declared units (c_A [mol/L], T [K])',
      'not a physical covariance']),
    ('identity', False,
     ['include_mse=False takes that variance as 1',
      'different declared units (c_A [mol/L], T [K])',
      'not a physical covariance']),
])
def test_noise_assumption_states_the_weighting(weighting, include_mse,
                                               fragments):
    report = fit_decay(weighting).get_identifiability(
        include_mse=include_mse)
    text = report.noise_assumption
    for fragment in fragments + ['Gauss-Newton', 'correct model',
                                 'interior optimum',
                                 'do not depend on the parameters',
                                 'user-supplied jac_fun',
                                 'relative jacobian_rtol = 2.22e-16']:
        assert fragment in text, (fragment, text)
    # The residual mean square is quoted with its value.
    assert f"s2 = {report.residual_mean_square:.4g}" in text
    if weighting != 'identity':
        assert 'physical covariance' not in text
    assert 'is the identity' not in text


def test_supplied_identity_weight_matrix_is_a_weight_matrix():
    estimator = ParameterEstimation(
        case.decay, [case.SEED_RATE], case.make_experiments(uncertain=False),
        name_params=['k'], jac_fun=decay_jacobian,
        weight_matrix=IDENTITY_WEIGHT_MATRIX)
    estimator.optimize_fn(verbose=False)
    report = estimator.get_identifiability()

    assert report.noise_model == 'weight_matrix'
    text = report.noise_assumption
    assert ('The supplied weight_matrix is the identity: independent '
            'errors with a standard deviation of 1 in each measured unit'
            in text)
    assert 'No measurement uncertainty or weight_matrix' not in text
    assert 'physical covariance' not in text
    # The weighting itself equals the unweighted fit's.
    unweighted = fit_decay('identity').get_identifiability()
    np.testing.assert_array_equal(report.covariance, unweighted.covariance)


def test_noise_assumption_unit_statements():
    time_s = np.array(case.HOT_CA_TIME)  # [s]
    initial, rise, _ = case.CALLBACK['hot']  # [mol/L], [K*L/mol]
    # c_A [mol/L] and T [K] of the hot run at the fixture's optimum rate,
    # as unlabelled legacy columns.
    observed = case.decay(np.array([deficient.K_STAR]), time_s, initial,
                          rise)
    legacy = ParameterEstimation(case.decay, [case.SEED_RATE], time_s,
                                 observed, args_fun=(initial, rise))
    legacy.optimize_fn(verbose=False)
    text = legacy.get_identifiability().noise_assumption
    assert 'units of y_0, y_1 are undeclared' in text
    assert 'not a physical covariance' in text

    single = ParameterEstimation(
        case.decay, [case.SEED_RATE],
        {'hot': Experiment(
            {'c_A': Measurement(0, case.HOT_CA_TIME, case.HOT_CA,
                                units='mol/L')},
            args=(case.CALLBACK['hot'][0],))})
    single.optimize_fn(verbose=False)
    text = single.get_identifiability().noise_assumption
    assert 'One measured field, c_A [mol/L].' in text
    assert 'physical covariance' not in text

    # c_A and c_B of the hot run, both declared in mol/L: summing their
    # residuals is dimensionally consistent.
    profile = two_species(np.array([deficient.K_STAR]), time_s, initial)
    same_unit = ParameterEstimation(
        two_species, [case.SEED_RATE],
        {'hot': Experiment(
            {'c_A': Measurement(0, time_s, profile[:, 0], units='mol/L'),
             'c_B': Measurement(1, time_s, profile[:, 1], units='mol/L')},
            args=(initial,))})
    same_unit.optimize_fn(verbose=False)
    text = same_unit.get_identifiability().noise_assumption
    assert 'Every measured field is in [mol/L].' in text
    assert 'physical covariance' not in text

    partly = ParameterEstimation(case.decay, [case.SEED_RATE],
                                 hot_experiment(None))
    partly.optimize_fn(verbose=False)
    text = partly.get_identifiability().noise_assumption
    assert 'The units of T are undeclared' in text
    assert 'not a physical covariance' in text


def test_two_parameter_scaled_sensitivity_matches_closed_form():
    estimator = fit_rise_and_rate(rise_first_jacobian)
    report = estimator.get_identifiability()

    _, jacobian = closed_form_rows(estimator.params_convg,
                                   'declared_uncertainty', rise_free=True)
    # [weighted residual / parameter unit], one norm per parameter row
    norms = np.linalg.norm(jacobian, axis=1)
    # Cosine of the two sensitivity rows; D J J^T D = [[1, c], [c, 1]] has
    # eigenvalues 1 +/- |c|, so the singular values of D J are their roots.
    cosine = jacobian[0] @ jacobian[1] / (norms[0] * norms[1])  # [-]
    assert cosine > 0  # T rises with both the rise and the rate
    np.testing.assert_allclose(
        report.singular_values, np.sqrt([1 + cosine, 1 - cosine]),
        rtol=ROUNDOFF_RTOL)
    assert report.condition_number == pytest.approx(
        (1 + cosine) / (1 - cosine), rel=ROUNDOFF_RTOL)
    assert report.collinearity_index == pytest.approx(
        1 / np.sqrt(1 - cosine), rel=ROUNDOFF_RTOL)
    # The eigenvector of 1 - c is (1, -1) / sqrt(2) in (rise, k) order.
    weakest = report.weakest_direction
    assert weakest.parameter_names == ('rise', 'k')
    np.testing.assert_allclose(weakest.components,
                               [HALF_ROOT_TWO, -HALF_ROOT_TWO],
                               rtol=ROUNDOFF_RTOL)
    assert weakest.singular_value == pytest.approx(np.sqrt(1 - cosine),
                                                   rel=ROUNDOFF_RTOL)
    # Analytic rows: bounds eps, floor sqrt(2) eps. Output-error norm from
    # the weighted closed-form predictions (the fitted rise is shared by
    # both runs).
    np.testing.assert_array_equal(report.sensitivity_error_bounds,
                                  [EPS, EPS])
    assert report.resolution_floor == pytest.approx(np.sqrt(2) * EPS,
                                                    rel=ROUNDOFF_RTOL)
    rise, rate = estimator.params_convg  # [K*L/mol], [1/s]
    predictions = []  # [-]
    for experiment, column, time, _, sigma in case.OBSERVED_ENTRIES:
        initial, _, temp0 = case.CALLBACK[experiment]
        remaining = initial * np.exp(-rate * time)  # [mol/L]
        value = (remaining if column == 0
                 else temp0 + rise * (initial - remaining))
        predictions.append(value / sigma)
    assert report.output_error_norm == pytest.approx(
        OUTPUT_RTOL * np.linalg.norm(predictions), rel=ROUNDOFF_RTOL)
    assert report.identifiable is True
    assert report.unresolved_directions == ()
    assert report.zero_information == ()
    np.testing.assert_array_equal(report.covariance,
                                  estimator.get_covariance())


def test_zero_information_parameter_first_and_three_informed_directions():
    estimator = ParameterEstimation(
        four_parameter_model,
        [deficient.SECOND_SEED, case.SEED_RATE, case.RISE_PARAMETER,
         CONC_FACTOR_SEED], case.experiments_without_rise(),
        name_params=['b_noeffect', 'k', 'rise', 'factor'],
        jac_fun=four_parameter_jacobian)
    with pytest.warns(RuntimeWarning, match=deficient.UNDEFINED):
        estimator.optimize_fn(verbose=False)
    nominal = estimator.get_identifiability()

    # Closed-form scaled Gram matrix of the informed rows (k, rise, factor).
    rows = four_parameter_rows(estimator.params_convg)[1:]
    scaled = rows / np.linalg.norm(rows, axis=1, keepdims=True)  # [-]
    eigenvalues, eigenvectors = np.linalg.eigh(scaled @ scaled.T)  # ascending
    np.testing.assert_allclose(
        nominal.singular_values,
        np.concatenate((np.sqrt(eigenvalues[::-1]), [0.0])),
        rtol=ROUNDOFF_RTOL, atol=EPS)
    assert nominal.zero_information == ('b_noeffect',)

    # Place the resolution floor between the two smallest informed singular
    # values (the floor is proportional to a uniform jacobian_rtol), so that
    # the weakest informed direction is reported with the no-effect one.
    target = np.sqrt(eigenvalues[0] * eigenvalues[1]) ** 0.5  # [-]
    factor = target / nominal.resolution_floor  # [-]
    report = estimator.get_identifiability(jacobian_rtol=EPS * factor)
    # The floor is linear in a uniform jacobian_rtol, up to rounding.
    assert report.resolution_floor == pytest.approx(target,
                                                    rel=ROUNDOFF_RTOL)
    assert len(report.unresolved_directions) == 2
    informed, no_effect = report.unresolved_directions
    assert no_effect.as_dict() == {'b_noeffect': 1.0, 'k': 0.0, 'rise': 0.0,
                                   'factor': 0.0}
    assert no_effect.singular_value == 0.0

    expected = eigenvectors[:, 0]  # [-], smallest eigenvalue
    first = np.flatnonzero(np.round(expected, 3))[0]
    expected = expected * np.sign(expected[first])  # sign convention
    components = informed.as_dict()
    assert list(components) == ['b_noeffect', 'k', 'rise', 'factor']
    assert components['b_noeffect'] == 0.0
    np.testing.assert_allclose(
        [components['k'], components['rise'], components['factor']],
        expected, atol=EIGENVECTOR_ATOL)
    assert not np.allclose(np.abs(expected[:2]), HALF_ROOT_TWO)
    assert informed.singular_value == pytest.approx(
        np.sqrt(eigenvalues[0]), rel=ROUNDOFF_RTOL)


@pytest.mark.parametrize('name, zero_information, components, line', [
    ('no_effect', ('b',), {'k': 0.0, 'b': 1.0},
     'Unresolved direction (zero information): 1.000*b'),
    ('product', (), {'a': HALF_ROOT_TWO, 'b': -HALF_ROOT_TWO},
     'Unresolved direction (singular value '),
])
def test_exactly_unidentifiable_parameters_are_not_reported_as_precise(
        name, zero_information, components, line):
    estimator, messages = fit_deficient(name)
    report = estimator.get_identifiability()

    assert report.identifiable is False
    assert np.isnan(report.covariance).all()
    assert np.isnan(report.correlation).all()
    assert np.isnan(report.standard_errors).all()
    assert np.isnan(report.relative_standard_errors).all()
    assert np.isnan(report.confidence_intervals).all()
    assert np.isnan(report.t_ratios).all()
    assert not report.significant.any()
    # The same diagnostic that optimize_fn warned with.
    assert messages == [report.covariance_diagnostic]
    assert report.zero_information == zero_information

    assert len(report.unresolved_directions) == 1
    direction = report.unresolved_directions[0]
    assert direction.parameter_names == tuple(components)
    np.testing.assert_allclose(direction.components,
                               list(components.values()),
                               atol=DIRECTION_ATOL)
    assert direction.singular_value <= np.sqrt(EPS) * (
        report.singular_values[0])
    assert report.weakest_direction == direction
    assert report.collinearity_index > 1 / np.sqrt(EPS)
    if zero_information:
        assert np.isnan(report.sensitivity_error_bounds[1])

    frame = report.to_frame()
    assert list(frame.index) == list(components)
    assert frame['significant'].isna().all()
    assert frame['zero_information'].tolist() == [
        parameter in zero_information for parameter in components]
    # Parameters carried by the unresolved direction (components that do
    # not round to zero) are flagged: b alone, or both of the product.
    assert frame['unresolved'].tolist() == [
        round(value, 3) != 0 for value in components.values()]
    assert frame.attrs == {'identifiable': False,
                           'covariance_diagnostic': messages[0]}
    summary = report.summary()
    assert 'NOT IDENTIFIABLE' in summary
    assert 'Covariance unavailable' in summary
    direction_lines = [text.strip() for text in summary.splitlines()
                       if text.startswith('  Unresolved direction')]
    assert len(direction_lines) == 1
    assert direction_lines[0].startswith(line)
    printed = '0.707*a - 0.707*b' if name == 'product' else '1.000*b'
    assert direction_lines[0].endswith(printed)
    weakest = [text for text in summary.splitlines()
               if text.startswith('Weakest direction (smallest singular '
                                  'value ')]
    assert len(weakest) == 1 and f": {printed};" in weakest[0]


def resolved_singular_values(jacobian):
    """Return the singular values of a Jacobian with unit-norm rows.

    Parameters
    ----------
    jacobian : numpy.ndarray
        Jacobian with nonzero rows, shape (n, m) [r / parameter unit].

    Returns
    -------
    numpy.ndarray
        Singular values [-] of the row-normalized matrix, descending.
    """
    rows = jacobian / np.linalg.norm(jacobian, axis=1, keepdims=True)
    return np.linalg.svd(rows, compute_uv=False)


@pytest.mark.parametrize('weight', SUM_WEIGHTS)
def test_inaccurate_small_sensitivity_row_is_not_resolved(weight):
    """The sums ``k + w * b`` with the default forward differences.

    The exact rows of ``b`` and ``k`` are proportional. The forward
    difference of ``b`` changes the rate by ``w * 1e-3`` 1/s, so its
    rounding separates the scaled rows (``s_min / s_max`` near 2e-4 for
    ``w = 1e-9`` and 0.16 for ``w = 1e-12`` on x86-64): the #532 test sees
    an invertible ``J J^T``. The rounding term of row ``b``,
    ``output_rtol * ||y_w|| / (h_b * ||J_b||)`` with ``||J_b||`` about ``w
    ||J_k||``, exceeds 1 by orders of magnitude, so ``b`` is unresolved and
    set aside; ``k`` alone is resolved (one unit row, ``s = 1``) above the
    floor of its own row.
    """
    model, _ = deficient.make_pair(weight, 0.0)
    estimator = ParameterEstimation(
        model, [case.SEED_RATE, deficient.SECOND_SEED],
        case.make_experiments(), name_params=['k', 'b'])
    estimator.optimize_fn(verbose=False)
    report = estimator.get_identifiability()

    assert report.jacobian_source == 'finite_difference'
    np.testing.assert_allclose(report.jacobian_rtol,
                               [DEFAULT_RELATIVE_STEP] * 2, rtol=STEP_RTOL)
    assert report.covariance_diagnostic is None  # #532 does not flag it
    jacobian = estimator.info_opt['jac']
    full = resolved_singular_values(jacobian)  # [-], both rows
    assert full[-1] / full[0] > np.sqrt(EPS)

    # Steps 1e-3 |p| and the output-error norm of the weighted closed-form
    # predictions at the effective rate k + w b (declared uncertainties are
    # diagonal weights).
    steps = DEFAULT_RELATIVE_STEP * np.abs(report.estimates)  # [1/s]
    np.testing.assert_allclose(report.finite_difference_steps, steps,
                               rtol=STEP_RTOL)
    rate = report.estimates[0] + weight * report.estimates[1]  # [1/s]
    error_norm = OUTPUT_RTOL * closed_form_prediction_norm(
        rate, 'declared_uncertainty')  # [-]
    assert report.output_error_norm == pytest.approx(error_norm,
                                                     rel=ROUNDOFF_RTOL)
    # Row bounds of the computed Jacobian, whose b row is w times the k row
    # up to its forward-difference error.
    norms = np.linalg.norm(jacobian, axis=1)  # [weighted residual * s]
    assert norms[1] < 2 * weight * norms[0]
    expected = (DEFAULT_RELATIVE_STEP
                + 2 * error_norm / (steps * norms))  # [-]
    np.testing.assert_allclose(report.sensitivity_error_bounds, expected,
                               rtol=STEP_RTOL)
    assert expected[0] < 1 <= expected[1]

    assert report.unresolved_parameters == ('b',)
    assert report.zero_information == ()
    assert report.identifiable is False
    # Only b is unresolved; k's analysis uses its own row.
    assert [direction.as_dict()
            for direction in report.unresolved_directions] == [
                {'k': 0.0, 'b': 1.0}]
    assert np.isnan(report.unresolved_directions[0].singular_value)
    assert report.resolution_floor == pytest.approx(expected[0],
                                                    rel=ROUNDOFF_RTOL)
    assert report.singular_values[0] == pytest.approx(1.0,
                                                      rel=ROUNDOFF_RTOL)
    assert np.isnan(report.singular_values[1])
    # NaN singular values compare unequal, so compare the components.
    assert report.weakest_direction.as_dict() == {'k': 0.0, 'b': 1.0}
    summary = report.summary()
    assert 'NOT IDENTIFIABLE' in summary
    assert 'Unresolved parameters' in summary
    assert 'Unresolved direction (unresolved parameter): 1.000*b' in summary
    assert ('Weakest direction (unresolved parameter, no singular value '
            'assigned): 1.000*b' in summary)
    assert 'Covariance unavailable' not in summary
    frame = report.to_frame()
    assert frame['unresolved'].tolist() == [False, True]
    assert frame['significant'].isna().tolist() == [False, True]
    assert frame.attrs == {'identifiable': False,
                           'covariance_diagnostic': None}

    # Outputs exact to rounding do not rescue a structurally rank-one pair:
    # either b stays unresolved or the weak direction (k - b) / sqrt(2) is.
    exact = estimator.get_identifiability(output_rtol=EPS)
    assert exact.identifiable is False
    assert exact.unresolved_parameters == ('b',) or any(
        np.allclose(direction.components, [HALF_ROOT_TWO, -HALF_ROOT_TWO],
                    atol=DIRECTION_ATOL)
        for direction in exact.unresolved_directions)


def test_tiny_effect_below_the_output_accuracy_is_not_resolved():
    """A parameter whose effect is smaller than the output errors.

    Model ``a * [1, 0, 0] + 1e-7 * b * [0, 1, 0]`` at ``a = b = 1``,
    forward differences with steps ``1e-3 |p|``. Hand derivation of the
    bounds ``h_rel + 2 output_rtol ||y|| / (h ||J_i||)`` with ``||y|| = 1``
    (to 1e-14): for a, ``1e-3 + 2 output_rtol / 1e-3``; for b, ``1e-3 + 2
    output_rtol / (1e-3 * 1e-7)``. With the default ``output_rtol = 1e-6``
    the bound of b is about 2e4: its sensitivity is buried in the output
    errors, so b is unresolved and a resolved. With outputs exact to
    rounding (``output_rtol = eps``) b is resolved.
    """
    def model(params, times):
        """Return ``a * [1, 0, 0] + 1e-7 * b * [0, 1, 0]`` [-].

        Parameters
        ----------
        params : numpy.ndarray
            ``[a, b]`` [-].
        times : numpy.ndarray
            Sample grid [-], unused.

        Returns
        -------
        numpy.ndarray
            Shape (3,).
        """
        return (params[0] * np.array([1.0, 0.0, 0.0])
                + TINY_EFFECT * params[1] * np.array([0.0, 1.0, 0.0]))

    observations = np.array([1.0, TINY_EFFECT, 0.0])  # [-], a = b = 1
    # b starts at its fitted value: its gradient, 1e-7 times smaller than
    # a's, does not move LM measurably.
    estimator = ParameterEstimation(model, [0.9, 1.0], np.arange(3.0),
                                    observations, name_params=['a', 'b'])
    estimator.optimize_fn(verbose=False)
    np.testing.assert_allclose(estimator.params_convg, [1.0, 1.0],
                               rtol=FIT_RTOL)
    report = estimator.get_identifiability(include_mse=False)

    assert report.output_error_norm == pytest.approx(OUTPUT_RTOL,
                                                     rel=FIT_RTOL)
    steps = DEFAULT_RELATIVE_STEP * np.abs(report.estimates)  # [-]
    expected = DEFAULT_RELATIVE_STEP + 2 * OUTPUT_RTOL / (
        steps * np.array([1.0, TINY_EFFECT]))  # [-]
    np.testing.assert_allclose(report.sensitivity_error_bounds, expected,
                               rtol=FIT_RTOL)
    assert report.identifiable is False
    assert report.unresolved_parameters == ('b',)
    assert [direction.as_dict()
            for direction in report.unresolved_directions] == [
                {'a': 0.0, 'b': 1.0}]
    assert report.resolution_floor == pytest.approx(expected[0],
                                                    rel=FIT_RTOL)
    assert report.singular_values[0] == pytest.approx(1.0,
                                                      rel=ROUNDOFF_RTOL)

    exact = estimator.get_identifiability(include_mse=False,
                                          output_rtol=EPS)
    assert exact.unresolved_parameters == ()
    assert exact.identifiable is True


def test_model_jacobian_rows_follow_their_row_relative_accuracy():
    """Reviewer probe3 under row-relative accuracy.

    Analytic rows ``1e6 t`` and ``1e-6 (t + 0.05 t**2)``. Hand derivation:
    each scaled row has the bound ``delta`` (``jacobian_rtol``), so the
    floor is ``sqrt(2) delta``; the Gram matrix ``[[1, c], [c, 1]]`` with
    ``c`` the cosine of the rows gives ``s_min = sqrt(1 - c)``, about 8e-3
    here, independent of the 1e12 ratio of the row magnitudes. With
    ``delta = 1e-3`` the pair is resolved. Removing the quadratic term
    changes row b by the relative amount ``rel = ||0.05 t**2|| / ||t +
    0.05 t**2||`` (about 0.035) and makes the rows collinear; declaring
    ``delta = rel`` puts that perturbation within the accuracy, and then
    the weak direction is not resolved, as Weyl's bound requires.
    """
    large = PROBE_LARGE * PROBE_TIMES  # [y / a]
    small = PROBE_SMALL * (PROBE_TIMES
                           + PROBE_CURVATURE * PROBE_TIMES ** 2)  # [y / b]
    jacobian = np.vstack((large, small))
    perturbation = np.vstack((np.zeros_like(small),
                              PROBE_SMALL * PROBE_TIMES - small))
    residuals = 0.01 * np.sin(np.arange(len(PROBE_TIMES)))  # [y], arbitrary

    def report_of(matrix, delta):
        """Return the report of one analytic Jacobian at ``a = b = 1``.

        Parameters
        ----------
        matrix : numpy.ndarray
            Jacobian, shape (2, 6) [y / parameter unit].
        delta : float
            Row-relative accuracy [-].

        Returns
        -------
        IdentifiabilityReport
            Report of the analytic Jacobian.
        """
        return IdentifiabilityReport.from_fit(
            matrix, residuals, [1.0, 1.0], ['a', 'b'], len(PROBE_TIMES),
            noise_model='identity', measurement_units=None,
            jacobian_source='jac_fun', jacobian_rtol=delta)

    cosine = large @ small / (np.linalg.norm(large)
                              * np.linalg.norm(small))  # [-]
    smallest = np.sqrt(1 - cosine)  # [-]
    resolved = report_of(jacobian, PROBE_RTOL)
    np.testing.assert_allclose(resolved.sensitivity_error_bounds,
                               [PROBE_RTOL, PROBE_RTOL])
    assert resolved.resolution_floor == pytest.approx(
        np.sqrt(2) * PROBE_RTOL, rel=ROUNDOFF_RTOL)
    assert resolved.singular_values[-1] == pytest.approx(
        smallest, rel=PROBE_RELATION_RTOL)
    assert np.sqrt(2) * PROBE_RTOL < smallest
    assert resolved.identifiable is True

    relative = (np.linalg.norm(perturbation[1])
                / np.linalg.norm(small))  # [-]
    nominal = report_of(jacobian, relative)
    with np.errstate(all='ignore'):  # exactly collinear: inv may warn
        collinear = report_of(jacobian + perturbation, relative)
    assert nominal.covariance_diagnostic is None
    assert nominal.identifiable is False
    assert collinear.identifiable is False
    assert nominal.weakest_direction in nominal.unresolved_directions
    shift = np.abs(nominal.singular_values - collinear.singular_values)
    assert (shift <= nominal.resolution_floor).all()


def test_near_collinear_pair_is_unresolved_by_the_532_criterion():
    estimator, _ = fit_deficient('near_collinear')
    report = estimator.get_identifiability()

    smallest, largest = (report.singular_values[-1],
                         report.singular_values[0])  # [-]
    assert EPS < smallest / largest <= np.sqrt(EPS)
    assert report.resolution_floor < smallest  # exact sensitivities
    assert len(report.unresolved_directions) == 1
    np.testing.assert_allclose(report.unresolved_directions[0].components,
                               [HALF_ROOT_TWO, -HALF_ROOT_TWO],
                               atol=DIRECTION_ATOL)
    assert report.identifiable is False


@pytest.mark.parametrize('scale', RATE_SCALES)
def test_rescaling_a_parameter_does_not_change_the_verdict(scale):
    estimator = ParameterEstimation(
        rescaled_rate_model(scale), [case.RISE_PARAMETER,
                                     case.SEED_RATE * scale],
        case.experiments_without_rise(), name_params=['rise', 'k'])
    # SciPy's Jacobian-based variable scaling makes the stopping tests
    # independent of the rate unit.
    estimator.optimize_fn(verbose=False, method='scipy',
                          optim_options={'x_scale': 'jac'})
    report = estimator.get_identifiability()
    reference = ParameterEstimation(
        case.decay_rise_first, [case.RISE_PARAMETER, case.SEED_RATE],
        case.experiments_without_rise(), name_params=['rise', 'k'])
    reference.optimize_fn(verbose=False, method='scipy',
                          optim_options={'x_scale': 'jac'})
    expected = reference.get_identifiability()

    assert report.jacobian_source == 'finite_difference'
    assert report.estimates[1] / scale == pytest.approx(
        expected.estimates[1], rel=RESCALE_RTOL)
    assert report.identifiable is True
    np.testing.assert_allclose(report.singular_values,
                               expected.singular_values, rtol=RESCALE_RTOL)
    np.testing.assert_allclose(report.sensitivity_error_bounds,
                               expected.sensitivity_error_bounds,
                               rtol=RESCALE_RTOL)
    assert report.resolution_floor == pytest.approx(
        expected.resolution_floor, rel=RESCALE_RTOL)
    assert report.resolution_floor < report.singular_values[-1]
    assert report.active_bounds == ()  # SciPy reports no active bound


def test_well_conditioned_finite_difference_fit_is_identifiable():
    estimator = fit_rise_and_rate(jac_fun=None)
    report = estimator.get_identifiability()

    assert report.jacobian_source == 'finite_difference'
    np.testing.assert_allclose(report.jacobian_rtol,
                               [DEFAULT_RELATIVE_STEP] * 2, rtol=STEP_RTOL)
    assert report.identifiable is True
    assert report.unresolved_directions == ()
    assert 'locally identifiable' in report.summary()


def test_model_returned_sensitivities_use_machine_precision():
    estimator = ParameterEstimation(decay_with_sensitivities,
                                    [case.SEED_RATE],
                                    case.make_experiments(),
                                    name_params=['k'])
    estimator.optimize_fn(verbose=False)
    report = estimator.get_identifiability()

    assert report.jacobian_source == 'model'
    np.testing.assert_array_equal(report.jacobian_rtol, [EPS])
    expected = fit_decay('declared_uncertainty').get_identifiability()
    np.testing.assert_allclose(report.covariance, expected.covariance,
                               rtol=ROUNDOFF_RTOL)


def test_default_rtol_needs_a_parameter_scale_at_zero():
    # b has no effect, so LM never moves it from an exact zero seed.
    estimator, _ = fit_deficient('no_effect', seed=0.0)
    assert estimator.params_convg[1] == 0.0
    with pytest.raises(ValueError, match=r'at zero or not finite \(b\); '
                       'pass param_scale.*alternatively pass jacobian_rtol'):
        estimator.get_identifiability()
    report = estimator.get_identifiability(param_scale=[1.0, 1.0])
    # Relative steps to the given scale 1: the default step of k is
    # 1e-3 * |k|, that of b at zero the absolute 1e-9 (1e-3 times the
    # magnitude floor 1e-6 of jac_module.dx_jac_p).
    rate = estimator.params_convg[0]  # [1/s]
    steps = np.array([DEFAULT_RELATIVE_STEP * abs(rate),
                      DEFAULT_RELATIVE_STEP * ZERO_PARAMETER_FLOOR])  # [-]
    np.testing.assert_allclose(report.finite_difference_steps, steps,
                               rtol=STEP_RTOL)
    np.testing.assert_allclose(report.jacobian_rtol, steps, rtol=STEP_RTOL)
    # An explicit jacobian_rtol needs no scale.
    explicit = estimator.get_identifiability(jacobian_rtol=PROBE_RTOL)
    np.testing.assert_array_equal(explicit.jacobian_rtol, [PROBE_RTOL] * 2)
    assert report.zero_information == ('b',)
    assert np.isnan(report.relative_standard_errors).all()


def test_default_rtol_branches_of_finite_differences():
    # A fixed parameter at exactly zero does not enter the default.
    fixed = ParameterEstimation(deficient.no_effect, [case.SEED_RATE, 0.0],
                                case.make_experiments(),
                                optimize_flags=[True, False],
                                name_params=['k', 'b'])
    fixed.optimize_fn(verbose=False)
    report = fixed.get_identifiability()
    assert report.parameter_names == ('k',)
    np.testing.assert_allclose(report.jacobian_rtol, [DEFAULT_RELATIVE_STEP],
                               rtol=STEP_RTOL)

    # A scalar absolute step: h = 1e-4 / |k|, hand-computed.
    scalar = ParameterEstimation(case.decay, [case.SEED_RATE],
                                 case.make_experiments(), name_params=['k'],
                                 dx_finitediff=SCALAR_STEP)
    scalar.optimize_fn(verbose=False)
    rate = scalar.params_convg[0]  # [1/s]
    relative = SCALAR_STEP / abs(rate)  # [-]
    np.testing.assert_allclose(
        scalar.get_identifiability().jacobian_rtol, [relative],
        rtol=STEP_RTOL)

    # A relative step near 2 is not a valid accuracy.
    oversized = ParameterEstimation(case.decay, [case.SEED_RATE],
                                    case.make_experiments(),
                                    name_params=['k'],
                                    dx_finitediff=OVERSIZED_STEP)
    oversized.optimize_fn(verbose=False)
    with pytest.raises(ValueError, match=r'not in \(0, 1\); pass '
                       'jacobian_rtol explicitly'):
        oversized.get_identifiability()


def test_signs_and_zero_estimates_against_tabulated_quantile():
    report = sign_report()

    assert report.degrees_of_freedom == 3
    assert stats.t.ppf(0.975, 3) == pytest.approx(T_TABLE_975_DOF3,
                                                  abs=T_TABLE_ATOL)
    np.testing.assert_allclose(report.standard_errors,
                               [SIGN_STANDARD_ERROR] * 3, rtol=ROUNDOFF_RTOL)
    np.testing.assert_allclose(
        report.t_ratios, np.abs(SIGN_ESTIMATES) / SIGN_STANDARD_ERROR,
        rtol=ROUNDOFF_RTOL)
    assert (report.t_ratios >= 0).all()
    # 8.16 > 3.182 > 2.45 > 0
    assert report.significant.tolist() == [True, False, False]
    relative = report.relative_standard_errors
    assert relative[0] == pytest.approx(SIGN_STANDARD_ERROR,
                                        rel=ROUNDOFF_RTOL)
    assert np.isnan(relative[1])
    assert relative[2] == pytest.approx(SIGN_STANDARD_ERROR / 0.3,
                                        rel=ROUNDOFF_RTOL)
    half_width = stats.t.ppf(0.975, 3) * SIGN_STANDARD_ERROR  # [p]
    np.testing.assert_allclose(report.confidence_intervals[0],
                               [-1.0 - half_width, -1.0 + half_width],
                               rtol=ROUNDOFF_RTOL)


def test_unestimable_variance_scale_withholds_the_uncertainty():
    """Noise-free data fitted exactly: ``s2 = 0`` with relative weights."""
    def constant(params, time_s):
        """Return the constant ``p`` [mol/L] at every sample.

        Parameters
        ----------
        params : numpy.ndarray
            ``[p]`` [mol/L].
        time_s : numpy.ndarray
            Sample times [s].

        Returns
        -------
        numpy.ndarray
            Shape (n_times,).
        """
        return params[0] * np.ones_like(time_s)

    time_s = np.arange(3.0)  # [s]
    level = 2.0  # [mol/L], noise-free data
    estimator = ParameterEstimation(constant, [1.0], time_s,
                                    np.full(3, level))
    # get_covariance keeps its legacy behaviour (s2 = 0 gives a zero
    # covariance and NaN correlations with NumPy warnings) during the fit.
    with np.errstate(all='ignore'):
        estimator.optimize_fn(verbose=False, method='scipy')
        legacy = estimator.get_covariance()
    assert estimator.params_convg[0] == level
    np.testing.assert_array_equal(legacy, [[0.0]])

    with warnings.catch_warnings():
        warnings.simplefilter('error')
        report = estimator.get_identifiability()
        fisher = report.fisher_information
    assert report.residual_mean_square == 0.0
    assert 'variance scale cannot be estimated' in (
        report.covariance_diagnostic)
    assert 'include_mse=False' in report.covariance_diagnostic
    for values in (report.covariance, report.correlation,
                   report.standard_errors, report.confidence_intervals,
                   report.t_ratios, fisher):
        assert np.isnan(values).all()
    assert not report.significant.any()
    assert report.identifiable is True  # J itself is fine
    assert 'Uncertainty unavailable' in report.summary()

    # Absolute weights do not need s2: J = [1, 1, 1], inv(J J^T) = 1/3.
    absolute = estimator.get_identifiability(include_mse=False)
    assert absolute.covariance_diagnostic is None
    assert absolute.covariance[0, 0] == pytest.approx(1 / 3,
                                                      rel=ROUNDOFF_RTOL)


@pytest.mark.parametrize('jacobian, residuals, fragment', [
    ([[1.0, 2.0, 3.0]], [HUGE_RESIDUAL, 0.0, 0.0],
     'variance scale cannot be estimated'),
    ([[TINY_SENSITIVITY] * 3], [LARGE_RESIDUAL, 0.0, 0.0],
     'covariance is not finite'),
])
def test_non_finite_covariance_is_withheld_without_warnings(
        jacobian, residuals, fragment):
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        report = IdentifiabilityReport.from_fit(
            jacobian, residuals, [1.0], ['p'], 3, noise_model='identity',
            measurement_units=None, jacobian_source='jac_fun',
            jacobian_rtol=EPS)
        fisher = report.fisher_information
    assert fragment in report.covariance_diagnostic
    assert np.isnan(report.covariance).all()
    assert np.isnan(report.standard_errors).all()
    assert np.isnan(report.confidence_intervals).all()
    assert not report.significant.any()
    if fragment.startswith('variance'):
        assert np.isnan(fisher).all()


def test_parameter_at_an_active_bound_has_no_interval():
    estimator = ParameterEstimation(
        case.decay_rise_first, [case.RISE_PARAMETER, 0.7],
        case.experiments_without_rise(), name_params=['rise', 'k'])
    lower = [0.0, 0.6]  # [K*L/mol], [1/s]; k* ~ 0.5 lies below 0.6
    upper = [100.0, 5.0]  # [K*L/mol], [1/s]
    estimator.optimize_fn(verbose=False, method='scipy',
                          bounds=(lower, upper))
    # SciPy returns the bound itself, up to the last ulp.
    assert estimator.params_convg[1] == pytest.approx(lower[1], rel=4 * EPS)
    report = estimator.get_identifiability()

    assert report.active_bounds == ('k',)
    assert np.isnan(report.confidence_intervals[1]).all()
    assert np.isnan(report.t_ratios[1])
    assert not report.significant[1]
    assert np.isfinite(report.confidence_intervals[0]).all()
    summary = report.summary()
    assert 'ACTIVE BOUNDS: k at a bound' in summary
    assert 'interior optimum' in summary
    frame = report.to_frame()
    assert frame['active_bound'].tolist() == [False, True]


@pytest.mark.parametrize('identifiable', [True, False])
def test_report_does_not_change_the_estimator(identifiable):
    if identifiable:
        estimator = fit_decay('declared_uncertainty', jac_fun=None)
    else:
        estimator, _ = fit_deficient('no_effect')
    info = estimator.info_opt
    before = snapshot(estimator)

    report = estimator.get_identifiability(include_mse=False)
    report.summary()
    report.to_frame()

    assert estimator.info_opt is info
    assert_same(before, snapshot(estimator))
    for name in FIT_ATTRIBUTES + ('covar_params', 'correl_params'):
        if name in before:
            assert_same(before[name], getattr(estimator, name), name)


def test_report_is_immutable_and_round_trips():
    report = fit_decay('declared_uncertainty').get_identifiability()
    with pytest.raises(dataclasses.FrozenInstanceError):
        report.identifiable = False
    with pytest.raises(ValueError, match='read-only'):
        report.covariance[0, 0] = 0.0
    with pytest.raises(TypeError):
        report.convergence['success'] = False
    with pytest.raises(TypeError):
        report.convergence['tolerances']['max_fun_eval'] = 1

    for restored in (pickle.loads(pickle.dumps(report)),
                     copy.deepcopy(report)):
        for field in dataclasses.fields(report):
            original = getattr(report, field.name)
            value = getattr(restored, field.name)
            if isinstance(original, np.ndarray):
                np.testing.assert_array_equal(value, original)
                assert not value.flags.writeable
            elif field.name == 'convergence':
                assert dict(value) == dict(original)
            else:
                assert value == original, field.name
        with pytest.raises(TypeError):
            restored.convergence['success'] = False
        assert restored.summary() == report.summary()


def test_unconverged_fit_is_reported_first():
    estimator = ParameterEstimation(case.decay, [case.SEED_RATE],
                                    case.make_experiments(),
                                    name_params=['k'])
    with pytest.warns(RuntimeWarning, match='stopped without converging'):
        estimator.optimize_fn(
            verbose=False,
            optim_options={'max_fun_eval': UNCONVERGED_EVALUATIONS})
    report = estimator.get_identifiability()

    assert report.convergence['success'] is False
    assert report.convergence['message'] == estimator.info_opt['message']
    assert report.convergence['tolerances'] == (
        estimator.info_opt['tolerances'])
    assert report.convergence['tolerances'] is not (
        estimator.info_opt['tolerances'])
    first, second = report.summary().splitlines()[:2]
    assert first.startswith('FIT DID NOT CONVERGE '
                            f"({estimator.info_opt['message']})")
    assert second.startswith('Noise assumption:')

    converged = fit_decay('declared_uncertainty').get_identifiability()
    assert converged.convergence['success'] is True
    lines = converged.summary().splitlines()
    assert lines[0].startswith('Noise assumption:')
    assert lines[1] == 'Convergence: converged (Small step).'


@pytest.mark.parametrize('model, source', [
    (mcr_case.model_with_sensitivities, 'model'),
    (mcr_case.model_states_only, 'finite_difference'),
])
@pytest.mark.parametrize('weight_key, noise_model', [
    (None, 'identity'), ('correlated', 'weight_matrix')])
def test_multiple_curve_resolution_inherits_the_report(
        model, source, weight_key, noise_model):
    estimator = mcr_case._estimator(model, weight_key)
    estimator.optimize_fn(verbose=False)
    report = estimator.get_identifiability()

    jacobian = estimator.info_opt['jac']  # data rows only
    assert jacobian.shape[1] == len(estimator.info_opt['fun'])
    np.testing.assert_array_equal(report.information_matrix,
                                  jacobian @ jacobian.T)
    np.testing.assert_array_equal(report.covariance,
                                  estimator.get_covariance())
    assert report.noise_model == noise_model
    assert report.jacobian_source == source
    if source == 'model':
        np.testing.assert_array_equal(report.jacobian_rtol, [EPS])
    # Output-error scale from the curve-resolved spectra and tracer
    # predictions at the observed entries (NaN elsewhere): their norm for
    # identity weights; for the correlated covariance C, ||W||_2 times it,
    # with ||W||_2 = max over the observed blocks C_oo of
    # 1 / sqrt(lambda_min(C_oo)) (residual sign flips do not change it).
    predictions = np.vstack(estimator.y_model)  # [-] and [mol/L]
    raw_norm = np.sqrt(np.nansum(predictions ** 2))  # [-]
    if noise_model == 'identity':
        assert 'not recorded for this data layout' in (
            report.noise_assumption)
        gain = 1.0  # [-]
    else:
        gain = max(whitening_gain(mcr_case.COVARIANCES[weight_key],
                                  pattern)
                   for pattern in observed_patterns(predictions))
    assert report.output_error_norm == pytest.approx(
        OUTPUT_RTOL * gain * raw_norm, rel=ROUNDOFF_RTOL)


def test_multiple_curve_resolution_ignores_jac_fun():
    """MCR differentiates its residuals itself and never calls jac_fun."""
    def unused_jacobian(*args, **kwargs):
        """Fail if called: MCR's get_gradient does not use jac_fun.

        Parameters
        ----------
        *args, **kwargs
            Callback arguments.

        Raises
        ------
        AssertionError
            Always.
        """
        raise AssertionError('MultipleCurveResolution called jac_fun')

    data = mcr_case._data()
    x_data = {name: {'spectra': time_s, 'non_spectra': time_s[rows]}
              for name, (time_s, rows) in mcr_case.SCHEDULES.items()}
    y_data = {name: {'spectra': spectra, 'non_spectra': tracer}
              for name, (spectra, tracer) in data.items()}
    estimator = MultipleCurveResolution(
        mcr_case.model_states_only, mcr_case.TRIAL_RATE, x_data, y_data,
        measured_ind={'spectra': [0, 1], 'non_spectra': [2]},
        jac_fun=unused_jacobian, name_states=['A', 'B', 'tracer'])
    estimator.optimize_fn(verbose=False)

    assert estimator.get_identifiability().jacobian_source == (
        'finite_difference')


def test_invalid_arguments_and_unfitted_estimator_are_rejected():
    unfitted = ParameterEstimation(case.decay, [case.SEED_RATE],
                                   case.make_experiments())
    with pytest.raises(AttributeError, match='info_opt'):
        unfitted.get_identifiability()

    estimator = fit_decay('declared_uncertainty')
    for alpha in (0.0, 1.0, -0.5, np.nan, np.inf):
        with pytest.raises(ValueError, match='alpha must be strictly'):
            estimator.get_identifiability(alpha=alpha)
    for alpha in (True, '0.9', 0.9 + 0j):
        with pytest.raises(TypeError, match='alpha must be a real number'):
            estimator.get_identifiability(alpha=alpha)
    for include_mse in (1, None, 'yes'):
        with pytest.raises(TypeError, match='include_mse must be a bool'):
            estimator.get_identifiability(include_mse=include_mse)
    for rtol in (0.0, 1.0, -1e-3, np.nan, np.inf, [0.0], [2.0]):
        with pytest.raises(ValueError, match='jacobian_rtol'):
            estimator.get_identifiability(jacobian_rtol=rtol)
    with pytest.raises(ValueError, match=r'shape \(1,\)'):
        estimator.get_identifiability(jacobian_rtol=[1e-3, 1e-3])
    with pytest.raises(TypeError, match='jacobian_rtol must be a real'):
        estimator.get_identifiability(jacobian_rtol='1e-6')
    for scale in (0.0, -1.0, np.inf, [1.0, 2.0]):
        with pytest.raises(ValueError, match='param_scale'):
            estimator.get_identifiability(param_scale=scale)
    with pytest.raises(TypeError, match='param_scale'):
        estimator.get_identifiability(param_scale='1')


def test_duplicate_parameter_names_are_rejected():
    estimator = ParameterEstimation(case.decay_rise_first,
                                    [case.RISE_PARAMETER, case.SEED_RATE],
                                    case.experiments_without_rise(),
                                    name_params=['k', 'k'])
    estimator.optimize_fn(verbose=False)
    with pytest.raises(ValueError, match='must be unique.*repeated: k'):
        estimator.get_identifiability()


def test_too_few_observations_are_rejected():
    def amplitude_decay(params, time_s):
        """Return ``a * exp(-k t)`` [mol/L] for ``[a, k]``.

        Parameters
        ----------
        params : numpy.ndarray
            Amplitude a [mol/L] and rate k [1/s].
        time_s : numpy.ndarray
            Sample times [s].

        Returns
        -------
        numpy.ndarray
            Concentration [mol/L], shape (n_times,).
        """
        return params[0] * np.exp(-params[1] * time_s)

    time_s = np.array([0.0, 1.0])  # [s], two samples for two parameters
    # Arbitrary generating values and seed: amplitude [mol/L], rate [1/s].
    data = amplitude_decay([2.0, 0.5], time_s)  # [mol/L]
    estimator = ParameterEstimation(amplitude_decay, [1.0, 0.3], time_s,
                                    data)
    # Zero degrees of freedom: get_covariance divides by zero inside the
    # fit (pre-existing behaviour, outside this report).
    with np.errstate(divide='ignore', invalid='ignore'):
        estimator.optimize_fn(verbose=False)
    with pytest.raises(ValueError, match=re.escape(
            'more observed entries than optimized parameters '
            '(num_data_total=2, num_params=2)')):
        estimator.get_identifiability()


def test_from_fit_validates_its_choices():
    with pytest.raises(ValueError, match='noise_model must be one of'):
        sign_report(noise_model='Identity')
    with pytest.raises(ValueError, match='jacobian_source must be one of'):
        sign_report(jacobian_source='analytic')
    with pytest.raises(ValueError, match='finite_difference_steps and '
                       'output_error_scale'):
        sign_report(jacobian_source='finite_difference')
    with pytest.raises(ValueError, match='finite_difference_steps must'):
        sign_report(jacobian_source='finite_difference',
                    output_error_scale=1.0,
                    finite_difference_steps=[1e-3, 0.0, 1e-3])
    with pytest.raises(ValueError, match='output_error_scale must be'):
        sign_report(output_error_scale=-1.0)
    for value in (0.0, 1.0, np.nan):
        with pytest.raises(ValueError, match='output_rtol must be strictly'):
            sign_report(output_rtol=value)
    with pytest.raises(ValueError, match='active_bounds names unknown '
                       'parameters: w'):
        sign_report(active_bounds=['w'])
    with pytest.raises(ValueError, match='repeated: x'):
        IdentifiabilityReport.from_fit(
            SIGN_JACOBIAN, SIGN_RESIDUALS, SIGN_ESTIMATES, ['x', 'x', 'z'],
            SIGN_JACOBIAN.shape[1], noise_model='identity',
            measurement_units=None, jacobian_source='jac_fun',
            jacobian_rtol=EPS)
    with pytest.raises(ValueError, match='more observed entries'):
        IdentifiabilityReport.from_fit(
            SIGN_JACOBIAN, SIGN_RESIDUALS, SIGN_ESTIMATES, ['x', 'y', 'z'],
            3, noise_model='identity', measurement_units=None,
            jacobian_source='jac_fun', jacobian_rtol=EPS)
    # Per-parameter accuracies are kept in parameter order.
    report = sign_report(jacobian_rtol=[1e-3, 1e-6, 1e-9])
    np.testing.assert_array_equal(report.jacobian_rtol, [1e-3, 1e-6, 1e-9])
    assert report.active_bounds is None
    assert np.isnan(report.output_error_norm)  # not supplied
    assert sign_report(active_bounds=['z', 'x']).active_bounds == ('x',
                                                                   'z')
    assert ('Convergence: not reported by the optimizer.'
            in report.summary().splitlines())


def whitening_gain(covariance, pattern):
    """Return the largest singular value of the whitening of a block.

    Parameters
    ----------
    covariance : numpy.ndarray
        Measurement-error covariance, shape (n, n) [u_i * u_j].
    pattern : tuple of bool
        Observed columns of a sample row.

    Returns
    -------
    float
        ``1 / sqrt(lambda_min(C_oo))`` [1/u]: any root ``S`` with ``S S^T =
        inv(C_oo)`` has ``||S||_2**2 = lambda_max(inv(C_oo))``.
    """
    observed = np.flatnonzero(pattern)
    block = covariance[np.ix_(observed, observed)]  # [u_i * u_j]
    return 1 / np.sqrt(np.linalg.eigvalsh(block).min())


def observed_patterns(predictions):
    """Return the observation patterns of the rows of a prediction table.

    Parameters
    ----------
    predictions : numpy.ndarray
        Shape (n_rows, n_columns), NaN where unobserved.

    Returns
    -------
    set of tuple of bool
        Distinct non-empty observed-column patterns.
    """
    return {tuple(row) for row in ~np.isnan(predictions) if row.any()}


def offset_model(offset):
    """Return the rising temperature ``T0 + a (1 - exp(-k t))``.

    Parameters
    ----------
    offset : float
        Constant output offset ``T0`` [K].

    Returns
    -------
    callable
        ``model(params, time_s)`` for ``[a, k]`` ([K], [1/s]) returning
        the temperature [K], shape (n_times,).
    """
    def model(params, time_s):
        """Return ``T0 + a (1 - exp(-k t))`` [K].

        Parameters
        ----------
        params : numpy.ndarray
            Amplitude a [K] and rate k [1/s].
        time_s : numpy.ndarray
            Sample times [s].

        Returns
        -------
        numpy.ndarray
            Temperature [K], shape (n_times,).
        """
        return offset + params[0] * (1 - np.exp(-params[1] * time_s))

    return model


def fit_offset_case(offset):
    """Fit the offset model with the default forward differences.

    Parameters
    ----------
    offset : float
        Output offset ``T0`` [K].

    Returns
    -------
    ParameterEstimation
        The fitted estimator; the data are the same rise plus a fixed
        perturbation for every offset.
    """
    model = offset_model(offset)
    data = model(OFFSET_TRUE, OFFSET_TIMES) + OFFSET_NOISE  # [K]
    estimator = ParameterEstimation(model, OFFSET_SEED, OFFSET_TIMES, data,
                                    name_params=['a', 'k'])
    estimator.optimize_fn(verbose=False)
    return estimator


def test_output_offsets_change_only_the_rounding_term():
    """Truncation errors do not depend on an additive output offset.

    The same rise is fitted with offsets 26.85, 300 and 3000 K (degrees
    Celsius and two Kelvin-like levels). With outputs exact to rounding
    (``output_rtol = eps``) the bounds are the relative steps 1e-3 up to
    ``2 eps ||y|| / (h ||J_i||)`` < 1e-9, so verdict and floor agree. With
    the default ``output_rtol`` the bounds follow ``h_rel + 2 output_rtol
    ||y|| / (h ||J_i||)`` with the closed-form ``||y||``, which grows with
    the offset: outputs accurate to 1e-6 of 3000 K carry 3 mK errors, and a
    forward difference of two of them up to 6 mK.
    """
    floors = []  # [-]
    for offset in OFFSETS:
        estimator = fit_offset_case(offset)
        exact = estimator.get_identifiability(output_rtol=EPS)
        assert exact.identifiable is True
        np.testing.assert_allclose(exact.jacobian_rtol,
                                   [DEFAULT_RELATIVE_STEP] * 2,
                                   rtol=STEP_RTOL)
        floors.append(exact.resolution_floor)

        report = estimator.get_identifiability()
        amplitude, rate = report.estimates  # [K], [1/s]
        predictions = offset + amplitude * (
            1 - np.exp(-rate * OFFSET_TIMES))  # [K], identity weights
        error_norm = OUTPUT_RTOL * np.linalg.norm(predictions)  # [K]
        assert report.output_error_norm == pytest.approx(error_norm,
                                                         rel=ROUNDOFF_RTOL)
        steps = DEFAULT_RELATIVE_STEP * np.abs(report.estimates)  # [K, 1/s]
        norms = np.linalg.norm(estimator.info_opt['jac'], axis=1)
        np.testing.assert_allclose(
            report.sensitivity_error_bounds,
            DEFAULT_RELATIVE_STEP + 2 * error_norm / (steps * norms),
            rtol=STEP_RTOL)
    np.testing.assert_allclose(floors, floors[0], rtol=OFFSET_FLOOR_RTOL)
    assert fit_offset_case(OFFSETS[0]).get_identifiability().identifiable


def test_heating_rate_on_a_300_kelvin_baseline():
    """``T = 300 K + rate * t`` with the default forward differences.

    The forward-difference row is exact to about 1e-10 (a closed-form
    model): with ``output_rtol = eps`` the bound is the relative step and
    the rate is resolved. Under the default assumption of outputs accurate
    to 1e-6 (0.3 mK at 300 K, so up to 0.6 mK for the difference of two
    outputs), the step changes the outputs by at most ``1e-3 * 0.02 K/s *
    10 s = 0.2 mK``, below that error: the bound ``1e-3 + 2e-6 ||y|| / (h
    ||t||)`` exceeds 1 and the rate is unresolved.
    """
    def heating(params, time_s):
        """Return ``300 K + rate * t`` [K].

        Parameters
        ----------
        params : numpy.ndarray
            Heating rate [K/s].
        time_s : numpy.ndarray
            Sample times [s].

        Returns
        -------
        numpy.ndarray
            Temperature [K], shape (n_times,).
        """
        return HEATING_BASELINE + params[0] * time_s

    data = heating([HEATING_RATE], HEATING_TIMES) + HEATING_NOISE  # [K]
    estimator = ParameterEstimation(heating, [HEATING_SEED], HEATING_TIMES,
                                    data, name_params=['rate'])
    estimator.optimize_fn(verbose=False)
    rate = estimator.params_convg[0]  # [K/s]
    predictions = HEATING_BASELINE + rate * HEATING_TIMES  # [K]

    def expected_bound(output_rtol):
        """Return ``h_rel + 2 output_rtol ||y|| / (h ||t||)`` [-].

        Parameters
        ----------
        output_rtol : float
            Relative output accuracy [-].

        Returns
        -------
        float
            The hand-derived bound with the exact sensitivity ``t``.
        """
        return DEFAULT_RELATIVE_STEP + 2 * output_rtol * np.linalg.norm(
            predictions) / (DEFAULT_RELATIVE_STEP * abs(rate)
                            * np.linalg.norm(HEATING_TIMES))

    exact = estimator.get_identifiability(output_rtol=EPS)
    assert exact.identifiable is True
    assert exact.sensitivity_error_bounds[0] == pytest.approx(
        expected_bound(EPS), rel=FD_ROW_RTOL)

    report = estimator.get_identifiability()
    bound = expected_bound(OUTPUT_RTOL)  # [-]
    assert report.sensitivity_error_bounds[0] == pytest.approx(
        bound, rel=FD_ROW_RTOL)
    assert bound > 1
    assert report.unresolved_parameters == ('rate',)


@pytest.mark.parametrize('correlation, resolved', [
    (0.0, True), (STRONG_CORRELATION, False)])
def test_correlated_weights_amplify_output_errors(correlation, resolved):
    """Whitening by a correlated covariance amplifies output errors.

    Both columns read ``a t + 0.22 b``: a and b move along the common
    direction, which whitening by ``[[1, rho], [rho, 1]]`` keeps (norm
    factor ``sqrt(2 / (1 + rho))``), while independent output errors may
    lie along the difference, amplified by ``1 / sqrt(1 - rho)``, 256 for
    ``rho = 1 - 2**-16``. The output-error norm is ``output_rtol *
    ||W||_2 * ||y||``; it leaves b (bound about 0.02 uncorrelated) above 1
    and a below it.
    """
    def model(params, time_s):
        """Return two identical columns ``a t + 0.22 b`` [-].

        Parameters
        ----------
        params : numpy.ndarray
            ``[a, b]`` [-].
        time_s : numpy.ndarray
            Sample times [-].

        Returns
        -------
        numpy.ndarray
            Shape (n_times, 2).
        """
        column = params[0] * time_s + COMMON_EFFECT * params[1]  # [-]
        return np.column_stack((column, column))

    times = np.array(CORRELATED_TIMES)  # [-]
    data = model([1.0, 1.0], times)  # [-], exact data
    covariance = np.array([[1.0, correlation],
                           [correlation, 1.0]])  # [-]
    estimator = ParameterEstimation(model, [0.9, 1.0], times, data,
                                    name_params=['a', 'b'],
                                    weight_matrix=covariance)
    estimator.optimize_fn(verbose=False)
    report = estimator.get_identifiability(include_mse=False)

    raw = model(report.estimates, times)  # [-]
    gain = 1 / np.sqrt(1 - correlation)  # [-], ||W||_2
    assert report.output_error_norm == pytest.approx(
        OUTPUT_RTOL * gain * np.linalg.norm(raw), rel=CORRELATED_RTOL)
    if resolved:
        assert report.identifiable is True
        assert report.sensitivity_error_bounds.max() < 1
    else:
        assert report.unresolved_parameters == ('b',)
        assert report.sensitivity_error_bounds[0] < 1
        assert [direction.as_dict()
                for direction in report.unresolved_directions] == [
                    {'a': 0.0, 'b': 1.0}]


@pytest.mark.parametrize('weight_key', ['diagonal', 'correlated'])
def test_output_error_norm_with_staggered_observations(weight_key):
    """Closed-form output-error norm of the staggered GLS fixture.

    Sample rows observe both states, c_A only or c_B only. Diagonal
    weights: ``output_rtol * sqrt(sum_o y_o^T inv(C_oo) y_o)``. Correlated
    weights: ``output_rtol * max_o ||W_o||_2 * ||y||`` with ``||W_o||_2 = 1 /
    sqrt(lambda_min(C_oo))`` over the three observed blocks.
    """
    estimator = masks._estimator(weight_key, 'finite_difference')
    estimator.optimize_fn(verbose=False)
    report = estimator.get_identifiability()

    covariance = masks._covariance_matrix(weight_key)  # [(mol/L)**2]
    _, y_data = masks._datasets()
    rows = masks._observed_rows(y_data)
    predictions = [regressors @ report.estimates
                   for _, regressors, _ in rows]  # [mol/L]
    if weight_key == 'diagonal':
        expected = np.sqrt(sum(
            values @ np.linalg.inv(covariance[np.ix_(observed, observed)])
            @ values for (observed, _, _), values
            in zip(rows, predictions)))  # [-]
    else:
        patterns = {tuple(state in observed for state in range(2))
                    for observed, _, _ in rows}
        gain = max(whitening_gain(covariance, pattern)
                   for pattern in patterns)  # [L/mol]
        expected = gain * np.linalg.norm(np.concatenate(predictions))
    assert len({len(observed) for observed, _, _ in rows}) == 2
    assert report.output_error_norm == pytest.approx(
        OUTPUT_RTOL * expected, rel=ROUNDOFF_RTOL)


def scaled_uncertainty_experiments(factor):
    """Build the decay experiments with every uncertainty times a factor.

    Parameters
    ----------
    factor : float
        Multiplier of the declared standard deviations [-].

    Returns
    -------
    dict
        Experiment name to ``Experiment``.
    """
    hot = Experiment(
        {'c_A': Measurement(0, case.HOT_CA_TIME, case.HOT_CA,
                            uncertainty=case.HOT_CA_STD * factor,
                            units='mol/L'),
         'T': Measurement(1, case.HOT_T_TIME, case.HOT_T,
                          uncertainty=case.HOT_T_STD * factor, units='K')},
        args=(case.CALLBACK['hot'][0],),
        kwargs={'rise_k_l_mol': case.CALLBACK['hot'][1]})
    cold = Experiment(
        {'c_A': Measurement(0, case.COLD_CA_TIME, case.COLD_CA,
                            uncertainty=np.array(case.COLD_CA_STD) * factor,
                            units='mol/L')},
        args=(case.CALLBACK['cold'][0],),
        kwargs={'rise_k_l_mol': case.CALLBACK['cold'][1],
                'temp0_k': case.CALLBACK['cold'][2]})
    return {'hot': hot, 'cold': cold}


@pytest.mark.parametrize('factor', EXTREME_FACTORS)
def test_bounds_survive_magnitudes_near_the_floating_point_limits(factor):
    """Weighted values near 1e-200 or 1e160 give the same bounds.

    Scaling every declared uncertainty by ``factor`` divides the weighted
    residuals, predictions and Jacobian by it. The bounds are ratios of
    norms, so they must not change, whereas plain sums of squares would
    underflow to 0 (``factor = 1e200``) or overflow to inf (``1e-160``).
    Every fit evaluates the seed only (``max_fun_eval = 0``), so all are
    analysed at the same parameters.
    """
    def evaluate(scale):
        """Return the report of the seed under scaled uncertainties.

        Parameters
        ----------
        scale : float
            Uncertainty multiplier [-].

        Returns
        -------
        IdentifiabilityReport
            Report with absolute weights.
        """
        estimator = ParameterEstimation(
            case.decay, [case.SEED_RATE],
            scaled_uncertainty_experiments(scale), name_params=['k'])
        with np.errstate(all='ignore'), warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            estimator.optimize_fn(verbose=False,
                                  optim_options={'max_fun_eval': 0})
        return estimator.get_identifiability(include_mse=False)

    nominal = evaluate(1.0)
    extreme = evaluate(factor)
    np.testing.assert_allclose(extreme.sensitivity_error_bounds,
                               nominal.sensitivity_error_bounds,
                               rtol=ROUNDOFF_RTOL)
    assert extreme.output_error_norm * factor == pytest.approx(
        nominal.output_error_norm, rel=ROUNDOFF_RTOL)
    assert extreme.resolution_floor == pytest.approx(
        nominal.resolution_floor, rel=ROUNDOFF_RTOL)
    assert extreme.unresolved_parameters == ()
    assert extreme.unresolved_directions == ()


def test_covariance_that_underflows_is_withheld():
    """Sensitivities of 1e150 make ``s2 * inv(J J^T)`` underflow to zero."""
    def linear(params, time_s):
        """Return ``p * 1e150 * t``.

        Parameters
        ----------
        params : numpy.ndarray
            ``[p]`` [1e-150 output unit / time unit].
        time_s : numpy.ndarray
            Sample times [-].

        Returns
        -------
        numpy.ndarray
            Shape (n_times,).
        """
        return params[0] * HUGE_SENSITIVITY * time_s

    def jacobian(params, time_s):
        """Return the exact sensitivities ``1e150 * t``.

        Parameters
        ----------
        params : numpy.ndarray
            ``[p]``, unused.
        time_s : numpy.ndarray
            Sample times [-].

        Returns
        -------
        numpy.ndarray
            Shape (n_times, 1).
        """
        return (HUGE_SENSITIVITY * time_s)[:, np.newaxis]

    estimator = ParameterEstimation(
        linear, [1 / HUGE_SENSITIVITY], np.arange(3.0),
        np.array([1e-100, 1.0, 2.0]), jac_fun=jacobian)
    with np.errstate(all='ignore'), warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        estimator.optimize_fn(verbose=False, method='scipy')
        legacy = estimator.get_covariance()
    assert legacy[0, 0] <= 0  # get_covariance keeps its earlier output
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        report = estimator.get_identifiability(jacobian_rtol=PROBE_RTOL)
    assert 'variances that are not positive' in report.covariance_diagnostic
    for values in (report.covariance, report.standard_errors,
                   report.confidence_intervals, report.t_ratios):
        assert np.isnan(values).all()
    assert not report.significant.any()


def test_non_finite_jacobian_has_no_direction_analysis():
    jacobian = np.array([[np.nan, 1.0, 2.0, 0.0],
                         [1.0, 0.0, 1.0, 1.0]])  # [r/p]
    report = IdentifiabilityReport.from_fit(
        jacobian, np.zeros(4), [1.0, 2.0], ['x', 'y'], 4,
        noise_model='identity', measurement_units=None,
        jacobian_source='jac_fun', jacobian_rtol=EPS, include_mse=False)

    assert 'non-finite entries for parameters x' in (
        report.covariance_diagnostic)
    assert report.identifiable is False
    assert np.isnan(report.singular_values).all()
    assert np.isnan(report.resolution_floor)
    assert np.isnan(report.condition_number)
    assert report.weakest_direction is None
    assert report.unresolved_directions == ()
    summary = report.summary()
    assert 'NOT IDENTIFIABLE' in summary
    assert 'Weakest direction' not in summary
    assert report.to_frame()['unresolved'].tolist() == [False, False]


def test_rounding_bound_covers_admissible_output_errors():
    """Opposite output errors at ``p`` and ``p + h`` are covered.

    ``T = 300 K + rate * t`` at ``rate = 0.08 K/s`` has no truncation
    error, so the forward-difference row error comes from output errors
    alone. Injecting the admissible errors ``-output_rtol |y(p)|`` and
    ``+output_rtol |y(p + h)|`` (the worst case of outputs accurate to
    ``output_rtol``) gives the realized row error ``output_rtol ||y(p) +
    y(p + h)|| / (h ||t||)``, computed here by hand. The reported bound
    must cover it, and its rounding part must match it up to ``|y(p +
    h)| - |y(p)| <= h * 10 s``, a relative 3e-6. That bound, 1.29, leaves
    the rate unresolved under the default output accuracy.
    """
    def heating(params, time_s):
        """Return ``300 K + rate * t`` [K].

        Parameters
        ----------
        params : numpy.ndarray
            Heating rate [K/s].
        time_s : numpy.ndarray
            Sample times [s].

        Returns
        -------
        numpy.ndarray
            Temperature [K], shape (n_times,).
        """
        return HEATING_BASELINE + params[0] * time_s

    data = heating([INJECTION_RATE], HEATING_TIMES) + HEATING_NOISE  # [K]
    estimator = ParameterEstimation(heating, [INJECTION_RATE],
                                    HEATING_TIMES, data,
                                    name_params=['rate'])
    estimator.optimize_fn(verbose=False)
    report = estimator.get_identifiability()

    rate = report.estimates[0]  # [K/s]
    step = DEFAULT_RELATIVE_STEP * abs(rate)  # [K/s]
    at_p = heating([rate], HEATING_TIMES) * (1 - OUTPUT_RTOL)  # [K]
    at_step = heating([rate + step], HEATING_TIMES) * (1 + OUTPUT_RTOL)
    perturbed_row = (at_step - at_p) / step  # [s]
    realized = (np.linalg.norm(perturbed_row - HEATING_TIMES)
                / np.linalg.norm(HEATING_TIMES))  # [-]
    bound = report.sensitivity_error_bounds[0]  # [-]
    assert realized <= bound
    assert bound - DEFAULT_RELATIVE_STEP == pytest.approx(
        realized, rel=INJECTION_RTOL)
    assert bound > 1
    assert report.unresolved_parameters == ('rate',)


def test_predictions_far_below_the_data_do_not_cancel():
    """Outputs near 1e-14 against data near 1e4.

    Residuals plus observations would rebuild the predictions as
    ``(y - 1e4) + 1e4``, which rounds them away; the report uses the
    accepted model outputs. Identity weights: ``output_error_norm =
    output_rtol ||1e-14 + 1e-20 p t||`` and the bound of p is ``1e-3 + 2
    output_error_norm / (1e-3 |p| * 1e-20 ||t||)``, hundreds.
    """
    def model(params, time_s):
        """Return ``1e-14 + 1e-20 * p * t``.

        Parameters
        ----------
        params : numpy.ndarray
            ``[p]`` [-].
        time_s : numpy.ndarray
            Sample locations [-].

        Returns
        -------
        numpy.ndarray
            Shape (n_times,).
        """
        return TINY_LEVEL + TINY_SLOPE * params[0] * time_s

    times = np.arange(1.0, 5.0)  # [-]
    estimator = ParameterEstimation(model, [1.0], times,
                                    np.full(4, LARGE_DATA))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        estimator.optimize_fn(verbose=False, method='scipy')
    report = estimator.get_identifiability(include_mse=False)

    value = report.estimates[0]  # [-]
    error_norm = OUTPUT_RTOL * np.linalg.norm(model([value], times))
    assert report.output_error_norm == pytest.approx(error_norm,
                                                     rel=ROUNDOFF_RTOL)
    bound = DEFAULT_RELATIVE_STEP + 2 * error_norm / (
        DEFAULT_RELATIVE_STEP * abs(value) * TINY_SLOPE
        * np.linalg.norm(times))  # [-]
    assert report.sensitivity_error_bounds[0] == pytest.approx(
        bound, rel=CANCELLATION_RTOL)
    assert report.identifiable is False
    assert report.unresolved_parameters == ('theta_1',)


def test_rows_observing_one_column_use_the_marginal_whitening_gain():
    """No sample row observes both columns under correlated weights.

    c_A is measured at 1, 2, 3 s and c_B at 1.5, 2.5, 3.5 s with the
    covariance ``[[1, 0.9], [0.9, 1]]`` (mol/L)**2: every row is whitened
    by a 1 x 1 marginal root of gain ``1 / sqrt(1)``, not by ``sigma_inv``
    of gain ``1 / sqrt(1 - 0.9) = 3.16``. The output-error norm is
    ``output_rtol * 1 * ||y||``.
    """
    initial = case.CALLBACK['hot'][0]  # [mol/L]
    times_a = np.array(STAGGER_TIMES_A)  # [s]
    times_b = np.array(STAGGER_TIMES_B)  # [s]
    true_rate = np.array([deficient.K_STAR])  # [1/s]
    observed_a = two_species(true_rate, times_a, initial)[:, 0] + (
        STAGGER_ERRORS_A)  # [mol/L]
    observed_b = two_species(true_rate, times_b, initial)[:, 1] + (
        STAGGER_ERRORS_B)  # [mol/L]
    experiments = {'run': Experiment(
        {'c_A': Measurement(0, times_a, observed_a, units='mol/L'),
         'c_B': Measurement(1, times_b, observed_b, units='mol/L')},
        args=(initial,))}
    covariance = pd.DataFrame(STAGGER_COVARIANCE, index=['c_A', 'c_B'],
                              columns=['c_A', 'c_B'])  # [(mol/L)**2]
    estimator = ParameterEstimation(two_species, [case.SEED_RATE],
                                    experiments, name_params=['k'],
                                    weight_matrix=covariance)
    estimator.optimize_fn(verbose=False)
    report = estimator.get_identifiability()

    rate = report.estimates  # [1/s]
    raw = np.concatenate((two_species(rate, times_a, initial)[:, 0],
                          two_species(rate, times_b, initial)[:, 1]))
    marginal = max(whitening_gain(STAGGER_COVARIANCE, pattern)
                   for pattern in ((True, False), (False, True)))
    full = whitening_gain(STAGGER_COVARIANCE, (True, True))  # [L/mol]
    assert marginal < full
    assert report.output_error_norm == pytest.approx(
        OUTPUT_RTOL * marginal * np.linalg.norm(raw), rel=ROUNDOFF_RTOL)
