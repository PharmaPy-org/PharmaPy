"""Independent derivative verification of the estimator Jacobian (#352, AC2).

Scope: ``ParameterEstimation.check_derivatives`` and ``DerivativeCheck``:
the comparison of ``get_gradient(p, out_array=True)`` (``jac_fun``,
model-returned sensitivities, or the estimator's forward differences) with
second-order finite differences of ``get_objective(p, out_array=True)``
over the optimized parameters only; error bounds derived from the
closed-form derivatives of the model; detection of a wrong sensitivity
row; the one-sided three-point formulas at bounds (near and far points),
with a domain-guarded model that raises outside its domain, and the
estimator's own forward steps (positive or negative) against bounds;
per-parameter pairs, arrays and ``scipy.optimize.Bounds``; zero parameters
and ``param_scale``; zero-width bounds; the unchanged fit state, also after
a failed evaluation and for callbacks that refill a preallocated
sensitivity array, and the final re-evaluation that leaves stateful
callbacks at the accepted point; fixed parameters; relative errors at
magnitudes near the float64 limits and for non-finite rows;
``MultipleCurveResolution`` inheritance; the result's immutability and
tables; and input validation. The SimulationExec reactor case of the same
re-evaluation is in ``test_paramestim_derivative_check_stateful``
(Assimulo lane).

Fixtures: the synthetic two-experiment adiabatic first-order decay of
``test_records_calibration`` with the temperature rise [K*L/mol] as the
first parameter and the rate constant [1/s] as the second
(``decay_rise_first``, declared uncertainties, staggered T and a missing
c_A sample, so the residual vector holds unobserved entries), its exact
Jacobian from ``test_paramestim_identifiability_report``, and the staggered
MCR case of ``test_paramestim_mcr_observation_masks``. The weighted
residuals are linear in the rise and exponential in the rate, so every
derivative is closed form: expected values and error bounds are derived
here from Taylor remainders, not from the production code. Core lane;
everything runs in milliseconds.
"""

import copy
import dataclasses
import pickle
import re
import warnings

import numpy as np
import pytest
from scipy.optimize import Bounds

import test_paramestim_identifiability_report as ident
import test_paramestim_mcr_observation_masks as mcr_case
import test_records_calibration as case
from PharmaPy.jac_module import (CENTRAL_DIFFERENCE_REL_STEP,
                                 DEFAULT_OUTPUT_RTOL)
from PharmaPy.ParamEstim import DerivativeCheck, ParameterEstimation

pytestmark = pytest.mark.unit

EPS = np.finfo(float).eps  # [-], float64 machine epsilon
# Rounding allowance per evaluated weighted residual, in units of EPS times
# the magnitudes it is computed from, (|model| + |data|) / sigma. Design
# assumption: each residual is a handful of floating-point operations
# (exp, products, a difference and a division); IEEE 754 rounds the basic
# operations to half an ulp and the exponential is accurate to a few ulp,
# so 8 ulp bounds their accumulated error with margin. The truncation
# assertions do not rest on it: at TRUNCATION_REL_STEP the truncation term
# exceeds this allowance by more than 100 times (asserted).
ROUNDING_ULPS = 8  # [-]
RISE = case.RISE_PARAMETER  # [K*L/mol], hot-run rise of the fixture
RATE = case.SEED_RATE  # [1/s], documented seed of k
SEED = (RISE, RATE)  # [K*L/mol], [1/s]
NAMES = ('rise', 'k')
# Relative step at which the reference's truncation error dominates its
# rounding: h**2 * |k**3 f'''/ f'| ~ 1e-6 * (k t)**2 against eps / h ~
# 2e-13 relative.
TRUNCATION_REL_STEP = 1e-3  # [-]
# Deliberate 10 % error injected into one sensitivity row.
WRONG_FACTOR = 1.1  # [-]
# Agreement threshold halfway to the injected error: a correct row agrees to
# about 1e-9 here (derived bounds below), a row wrong by WRONG_FACTOR
# differs by about WRONG_FACTOR - 1 = 0.1.
MIDWAY_RTOL = (WRONG_FACTOR - 1) / 2  # [-]
# Units: the weighted residuals of the fixture are dimensionless (declared
# uncertainties), so Jacobian and reference rows, their errors and bounds
# are in [1/p_i]: [mol/(K*L)] for the rise and [s] for k.
# Typical magnitude of the rise for the check at rise = 0: the hot-run rise
# of the fixture.
RISE_SCALE = RISE  # [K*L/mol]
# Observed entries of the residual layout, written from the fixture data:
# hot run c_A at 0, 1, 2, 4 s (all observed), hot T on the same grid (only
# at 2 s), cold run c_A at 0.5, 1.5, 3 s (1.5 s missing), cold T (never).
EXPECTED_OBSERVED = np.array([1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0],
                             dtype=bool)
NUM_ENTRIES = EXPECTED_OBSERVED.size  # [-]


class DomainViolation(Exception):
    """Raised by the guarded model callbacks outside their domain."""


def rise_first_with_sensitivities(params, time_s, initial_mol_l,
                                  temp0_k=300.0):
    """Return the decay outputs with their exact sensitivities.

    Parameters
    ----------
    params : numpy.ndarray
        Rise [K*L/mol], then rate constant k [1/s].
    time_s : numpy.ndarray
        Model grid [s].
    initial_mol_l : float
        Initial concentration of A [mol/L].
    temp0_k : float, optional
        Initial temperature [K].

    Returns
    -------
    tuple of numpy.ndarray
        ``case.decay_rise_first`` output and ``ident.rise_first_jacobian``
        output, the layout of models that integrate their sensitivities.
    """
    return (case.decay_rise_first(params, time_s, initial_mol_l, temp0_k),
            ident.rise_first_jacobian(params, time_s, initial_mol_l,
                                      temp0_k))


def scaled_jacobian(index):
    """Return a ``jac_fun`` whose row ``index`` is wrong by WRONG_FACTOR.

    Parameters
    ----------
    index : int
        Parameter (0 rise, 1 rate) whose sensitivities are scaled.

    Returns
    -------
    callable
        Jacobian callback with the signature of ``ident.rise_first_jacobian``.
    """
    def jacobian(params, time_s, initial_mol_l, temp0_k=300.0):
        """Return the exact Jacobian with one column scaled.

        Parameters
        ----------
        params : numpy.ndarray
            Rise [K*L/mol], then rate constant k [1/s].
        time_s : numpy.ndarray
            Model grid [s].
        initial_mol_l : float
            Initial concentration of A [mol/L].
        temp0_k : float, optional
            Initial temperature [K].

        Returns
        -------
        numpy.ndarray
            Shape (2 * n_times, 2), as ``ident.rise_first_jacobian``.
        """
        exact = ident.rise_first_jacobian(params, time_s, initial_mol_l,
                                          temp0_k)
        exact[:, index] *= WRONG_FACTOR
        return exact

    return jacobian


def guarded(lower, upper, calls, sensitivities=False,
            error=DomainViolation):
    """Return a model and Jacobian that raise outside a parameter box.

    Parameters
    ----------
    lower, upper : sequence of float
        Domain of the full parameter vector (rise [K*L/mol], k [1/s]). A
        float ``numpy.ndarray`` is held by reference, so a test can tighten
        the domain in place after a fit.
    calls : list
        Receives a copy of every parameter vector passed to either callback.
    sensitivities : bool, optional
        If True, the model returns its outputs together with the exact
        sensitivities, as ``rise_first_with_sensitivities``.
    error : type, optional
        Exception class raised outside the domain; the default is
        ``DomainViolation``.

    Returns
    -------
    model, jacobian : callable
        ``case.decay_rise_first`` (or ``rise_first_with_sensitivities``)
        and ``ident.rise_first_jacobian`` behind the domain guard.
    """
    lower = np.asarray(lower, dtype=float)  # [K*L/mol], [1/s]
    upper = np.asarray(upper, dtype=float)  # [K*L/mol], [1/s]

    def check(params):
        """Record ``params`` and raise outside the domain.

        Parameters
        ----------
        params : numpy.ndarray
            Full parameter vector [K*L/mol], [1/s].

        Raises
        ------
        DomainViolation
            Or ``error``, if a parameter lies outside ``[lower, upper]``.
        """
        calls.append(np.array(params, dtype=float, copy=True))
        if np.any(params < lower) or np.any(params > upper):
            raise error(f"evaluated outside the domain: {params}")

    def model(params, time_s, initial_mol_l, temp0_k=300.0):
        """Guarded ``case.decay_rise_first``; see that function."""
        check(params)
        if sensitivities:
            return rise_first_with_sensitivities(params, time_s,
                                                 initial_mol_l, temp0_k)
        return case.decay_rise_first(params, time_s, initial_mol_l, temp0_k)

    def jacobian(params, time_s, initial_mol_l, temp0_k=300.0):
        """Guarded ``ident.rise_first_jacobian``; see that function."""
        check(params)
        return ident.rise_first_jacobian(params, time_s, initial_mol_l,
                                         temp0_k)

    return model, jacobian


def make_estimator(model=case.decay_rise_first,
                   jac_fun=ident.rise_first_jacobian, seed=SEED, **kwargs):
    """Build the two-parameter decay estimator.

    Parameters
    ----------
    model : callable, optional
        Model callback with the signature of ``case.decay_rise_first``.
    jac_fun : callable or None, optional
        Jacobian callback; None uses the estimator's forward differences.
    seed : sequence of float, optional
        Rise [K*L/mol] and rate constant [1/s].
    **kwargs
        Further ``ParameterEstimation`` options.

    Returns
    -------
    ParameterEstimation
        Unfitted estimator with declared uncertainties.
    """
    return ParameterEstimation(model, list(seed),
                               case.experiments_without_rise(),
                               name_params=list(NAMES), jac_fun=jac_fun,
                               **kwargs)


def closed_form(params):
    """Return the observed weighted residuals and their exact Jacobian.

    Parameters
    ----------
    params : sequence of float
        Rise [K*L/mol] and rate constant [1/s].

    Returns
    -------
    residuals : numpy.ndarray
        ``(model - data) / sigma`` [-], shape (7,), in
        ``case.OBSERVED_ENTRIES`` order (the order of the observed entries
        of the residual layout).
    jacobian : numpy.ndarray
        Shape (2, 7): [1 / (K*L/mol)] and [s] rows.
    """
    return ident.closed_form_rows(np.asarray(params, dtype=float),
                                  'declared_uncertainty', rise_free=True)


def rate_derivative(params, order):
    """Return an exact derivative of the weighted residuals in the rate.

    Parameters
    ----------
    params : sequence of float
        Rise [K*L/mol] and rate constant k [1/s].
    order : int
        Derivative order n >= 1.

    Returns
    -------
    numpy.ndarray
        ``d^n r / dk^n`` [s**n], shape (7,), observed entries: ``c0
        (-t)**n exp(-k t) / sigma`` for c_A and ``-rise`` times that for
        T = T0 + rise (c0 - c_A). Its magnitude decreases with k for
        t >= 0, so its value at the smallest k of an interval bounds it
        there.
    """
    rise, rate = params
    values = []  # [s**n]
    for experiment, column, time, _, sigma in case.OBSERVED_ENTRIES:
        initial = case.CALLBACK[experiment][0]  # [mol/L]
        # [mol/L * s**n]
        base = initial * (-time) ** order * np.exp(-rate * time)
        values.append((base if column == 0 else -rise * base) / sigma)
    return np.array(values)


def rounding(params):
    """Return the rounding allowance of each observed weighted residual.

    Parameters
    ----------
    params : sequence of float
        Rise [K*L/mol] and rate constant [1/s].

    Returns
    -------
    numpy.ndarray
        ``ROUNDING_ULPS * EPS * (|model| + |data|) / sigma`` [-], shape (7,).
    """
    residuals, _ = closed_form(params)
    values = np.array([entry[3] for entry in case.OBSERVED_ENTRIES])
    sigmas = np.array([entry[4] for entry in case.OBSERVED_ENTRIES])
    models = residuals * sigmas + values  # [mol/L or K]
    return ROUNDING_ULPS * EPS * (np.abs(models) + np.abs(values)) / sigmas


def shifted(params, index, offset):
    """Return a copy of ``params`` with one entry shifted.

    Parameters
    ----------
    params : sequence of float
        Rise [K*L/mol] and rate constant [1/s].
    index : int
        Entry to shift.
    offset : float
        Shift [unit of that entry].

    Returns
    -------
    numpy.ndarray
        Shifted copy.
    """
    moved = np.array(params, dtype=float)
    moved[index] += offset
    return moved


def reference_bound(params, index, step, formula):
    """Bound the error of the finite-difference reference entry by entry.

    Parameters
    ----------
    params : sequence of float
        Point of the check: rise [K*L/mol] and rate constant [1/s].
    index : int
        Differentiated parameter, 0 (rise) or 1 (rate).
    step : float
        Positive step h [unit of the parameter].
    formula : {'central', 'forward', 'backward'}
        Stencil of the reference.

    Returns
    -------
    truncation : numpy.ndarray
        Taylor-remainder bound [weighted residual / parameter unit], shape
        (7,). Central: ``h**2 / 6 * max |f'''|`` over ``[p - h, p + h]``.
        One-sided three-point: the Lagrange remainders ``4 h**3 / 6 |f'''|``
        and ``8 h**3 / 6 |f'''|`` divided by ``2 h``, i.e. ``h**2 * max
        |f'''|`` over the stencil. Zero for the rise (linear residuals).
    rounding_term : numpy.ndarray
        Rounding of the evaluated residuals divided as in the formula, same
        shape and units.
    """
    signed = {'central': step, 'forward': step, 'backward': -step}[formula]
    if formula == 'central':
        points = [shifted(params, index, step), shifted(params, index, -step)]
        weights = [1.0, 1.0]  # [-], |coefficients| of the stencil
        denominator = 2 * step
        factor = step ** 2 / 6  # [parameter unit**2]
        smallest = params[index] - step
    else:
        points = [np.asarray(params, dtype=float),
                  shifted(params, index, signed),
                  shifted(params, index, 2 * signed)]
        weights = [3.0, 4.0, 1.0]  # [-]
        denominator = 2 * step
        factor = step ** 2  # [parameter unit**2]
        smallest = min(params[index], params[index] + 2 * signed)
    if index == 0:
        truncation = np.zeros(len(case.OBSERVED_ENTRIES))
    else:
        truncation = factor * np.abs(rate_derivative(
            (params[0], smallest), 3))
    rounding_term = sum(weight * rounding(point)
                        for weight, point in zip(weights, points)
                        ) / denominator
    return truncation, rounding_term


def observed_rows(check):
    """Return the observed columns of the Jacobian and the reference.

    Parameters
    ----------
    check : DerivativeCheck
        Result of ``check_derivatives``.

    Returns
    -------
    tuple of numpy.ndarray
        Jacobian and reference restricted to ``EXPECTED_OBSERVED``, shape
        (n_params, 7).
    """
    return (check.jacobian[:, EXPECTED_OBSERVED],
            check.reference[:, EXPECTED_OBSERVED])


def assert_steps(check, rel_step, scale):
    """Assert the steps are ``rel_step * scale`` up to their rounding.

    Parameters
    ----------
    check : DerivativeCheck
        Result of ``check_derivatives``.
    rel_step : float
        Relative step [-].
    scale : numpy.ndarray
        Parameter scales [parameter units].
    """
    nominal = rel_step * np.asarray(scale, dtype=float)
    # p + h is rounded to the spacing of p + h, at most 2 * EPS * |p + h|.
    spacing = 2 * EPS * (np.abs(check.params) + nominal)
    np.testing.assert_array_less(np.abs(check.steps - nominal),
                                 spacing + EPS * nominal)


@pytest.mark.parametrize('model, jac_fun, source', [
    (case.decay_rise_first, ident.rise_first_jacobian, 'jac_fun'),
    (rise_first_with_sensitivities, None, 'model')])
@pytest.mark.parametrize('rel_step', [None, TRUNCATION_REL_STEP])
def test_exact_jacobian_agrees_within_the_derived_reference_error(
        model, jac_fun, source, rel_step):
    estimator = make_estimator(model, jac_fun)
    check = estimator.check_derivatives(rel_step=rel_step)

    assert check.jacobian_source == source
    assert check.parameter_names == NAMES
    assert check.formulas == ('central', 'central')
    np.testing.assert_array_equal(check.params, SEED)
    used = CENTRAL_DIFFERENCE_REL_STEP if rel_step is None else rel_step
    np.testing.assert_array_equal(check.rel_step, [used, used])
    np.testing.assert_array_equal(check.param_scale, np.abs(SEED))
    assert_steps(check, used, SEED)
    np.testing.assert_array_equal(check.observed, EXPECTED_OBSERVED)
    assert check.jacobian.shape == check.reference.shape == (2, NUM_ENTRIES)
    # Unobserved entries are exactly zero in both.
    np.testing.assert_array_equal(check.jacobian[:, ~EXPECTED_OBSERVED], 0)
    np.testing.assert_array_equal(check.reference[:, ~EXPECTED_OBSERVED], 0)

    _, exact = closed_form(SEED)
    jacobian, reference = observed_rows(check)
    np.testing.assert_allclose(jacobian, exact, rtol=ident.ROUNDOFF_RTOL)
    for index in range(2):
        truncation, rounding_term = reference_bound(
            SEED, index, check.steps[index], 'central')
        # [1/p_i], plus the rounding of the exact row
        allowance = rounding_term + ROUNDING_ULPS * EPS * np.abs(exact[index])
        error = reference[index] - exact[index]  # [1/p_i]
        np.testing.assert_array_less(np.abs(error), truncation + allowance)
        if index == 1 and rel_step == TRUNCATION_REL_STEP:
            # The error is the O(h**2) central truncation term: it matches
            # h**2 / 6 * f'''(k) up to the next remainder, h**4 / 120 *
            # max |f^(5)|, and the rounding allowance, which is far smaller
            # wherever the truncation term is nonzero (t > 0).
            step = check.steps[1]  # [1/s]
            curved = truncation > 0  # entries with t > 0
            assert np.all(truncation[curved] > 100 * allowance[curved])
            leading = step ** 2 / 6 * rate_derivative(SEED, 3)  # [s]
            remainder = step ** 4 / 120 * np.abs(
                rate_derivative((RISE, RATE - step), 5))  # [s]
            np.testing.assert_array_less(np.abs(error - leading),
                                         remainder + allowance)
    np.testing.assert_array_equal(check.abs_error,
                                  np.abs(check.jacobian - check.reference))
    assert check.rtol is None and check.agrees is None


@pytest.mark.parametrize('wrong', [0, 1])
def test_wrong_sensitivity_row_is_detected_for_that_parameter_only(wrong):
    estimator = make_estimator(jac_fun=scaled_jacobian(wrong))
    _, exact = closed_form(SEED)
    errors, relative_errors = [], []
    for index in range(2):
        truncation, rounding_term = reference_bound(
            SEED, index, CENTRAL_DIFFERENCE_REL_STEP * SEED[index], 'central')
        # Bound on |J - R| of a correct row: reference error plus the
        # rounding of the evaluated exact Jacobian.
        errors.append(truncation + rounding_term
                      + ROUNDING_ULPS * EPS * np.abs(exact[index]))  # [1/p_i]
        # ||J - R|| / ||R|| of a correct row is at most ||E|| / (||J|| -
        # ||E||): the agreement level a correct row can reach.
        error_norm = np.linalg.norm(errors[-1])  # [1/p_i]
        relative_errors.append(error_norm / (np.linalg.norm(exact[index])
                                             - error_norm))  # [-]
    rtol = max(relative_errors)  # [-]

    check = estimator.check_derivatives(rtol=rtol)

    expected_agreement = [index != wrong for index in range(2)]
    np.testing.assert_array_equal(check.agrees, expected_agreement)
    assert check.rtol == rtol
    # ||(F - 1) J - E|| / ||J + E|| with F = WRONG_FACTOR and E the
    # reference error, bracketed by the triangle inequality.
    exact_norm = np.linalg.norm(exact[wrong])  # [1/p_i]
    error_norm = np.linalg.norm(errors[wrong])  # [1/p_i]
    low = ((WRONG_FACTOR - 1) * exact_norm - error_norm) / (
        exact_norm + error_norm)  # [-]
    high = ((WRONG_FACTOR - 1) * exact_norm + error_norm) / (
        exact_norm - error_norm)  # [-]
    assert low <= check.relative_error[wrong] <= high
    assert check.max_relative_error == check.relative_error[wrong]
    right = 1 - wrong
    assert check.relative_error[right] <= relative_errors[right]
    np.testing.assert_allclose(check.jacobian[wrong, EXPECTED_OBSERVED],
                               WRONG_FACTOR * exact[wrong],
                               rtol=ident.ROUNDOFF_RTOL)


def test_finite_difference_jacobian_differs_by_its_forward_truncation():
    estimator = make_estimator(jac_fun=None)
    check = estimator.check_derivatives()

    assert check.jacobian_source == 'finite_difference'
    _, exact = closed_form(SEED)
    jacobian, reference = observed_rows(check)
    # Default dx_finitediff: 1e-3 * |p| for |p| >= 1e-6 (forward).
    dx = ident.DEFAULT_RELATIVE_STEP * np.abs(SEED)  # [K*L/mol], [1/s]
    for index in range(2):
        truncation, rounding_term = reference_bound(
            SEED, index, check.steps[index], 'central')
        # Error of the reference against the exact derivative.
        reference_error = (truncation + rounding_term + ROUNDING_ULPS * EPS
                           * np.abs(exact[index]))  # [1/p_i]
        # Forward-difference rounding: two evaluated outputs over dx, plus
        # the representation of p + dx (at most 2 EPS |p + dx|) relative to
        # dx times the derivative.
        forward_rounding = ((rounding(SEED) + rounding(
            shifted(SEED, index, dx[index]))) / dx[index]
            + 2 * EPS * (SEED[index] + dx[index]) / dx[index]
            * np.abs(exact[index]))  # [1/p_i]
        difference = jacobian[index] - reference[index]  # [1/p_i]
        if index == 0:
            # Linear in the rise: the forward difference is exact.
            np.testing.assert_array_less(np.abs(difference),
                                         forward_rounding + reference_error)
            continue
        # (f(k + dx) - f(k)) / dx = f' + dx / 2 f'' + dx**2 / 6 f'''(xi).
        leading = dx[1] / 2 * rate_derivative(SEED, 2)  # [s]
        remainder = dx[1] ** 2 / 6 * np.abs(rate_derivative(SEED, 3))  # [s]
        bound = remainder + forward_rounding + reference_error  # [s]
        np.testing.assert_array_less(np.abs(difference - leading), bound)
        # Relative error O(dx): ||leading|| / ||f'|| up to those bounds.
        exact_norm = np.linalg.norm(exact[1])  # [s]
        reference_norm = np.linalg.norm(reference_error)  # [s]
        low = (np.linalg.norm(leading) - np.linalg.norm(bound)) / (
            exact_norm + reference_norm)  # [-]
        high = (np.linalg.norm(leading) + np.linalg.norm(bound)) / (
            exact_norm - reference_norm)  # [-]
        assert low <= check.relative_error[1] <= high
        # Consistent with forward truncation, not with a correct Jacobian.
        assert check.relative_error[1] > 100 * check.relative_error[0]


@pytest.mark.parametrize('rel_step', [None, TRUNCATION_REL_STEP])
@pytest.mark.parametrize('rate_offset_steps', [0.0, 0.5])
def test_bounds_select_one_sided_formulas_and_stay_in_the_domain(
        rate_offset_steps, rel_step):
    used = CENTRAL_DIFFERENCE_REL_STEP if rel_step is None else rel_step
    rate_step = used * RATE  # [1/s]
    # Lower bound of k at k (offset 0) or half a step below it, so that the
    # central point k - h leaves the domain in both cases.
    rate_lower = RATE - rate_offset_steps * rate_step  # [1/s]
    calls = []
    model, jacobian = guarded([-np.inf, rate_lower], [RISE, np.inf], calls)
    estimator = make_estimator(model, jacobian)
    bounds = [(None, RISE), (rate_lower, None)]

    check = estimator.check_derivatives(rel_step=rel_step, bounds=bounds)

    assert check.formulas == ('backward', 'forward')
    evaluated = np.array(calls)  # [K*L/mol], [1/s]
    assert np.all(evaluated[:, 0] <= RISE)
    assert np.all(evaluated[:, 1] >= rate_lower)
    # Both one-sided stencils were evaluated: 2 points each besides p.
    assert np.any(evaluated[:, 0] < RISE) and np.any(evaluated[:, 1] > RATE)
    _, exact = closed_form(SEED)
    _, reference = observed_rows(check)
    for index, formula in enumerate(check.formulas):
        truncation, rounding_term = reference_bound(
            SEED, index, check.steps[index], formula)
        # [1/p_i], plus the rounding of the exact row
        allowance = rounding_term + ROUNDING_ULPS * EPS * np.abs(exact[index])
        np.testing.assert_array_less(
            np.abs(reference[index] - exact[index]), truncation + allowance)
        if index == 1 and rel_step == TRUNCATION_REL_STEP:
            # A forward two-point difference would be wrong by h / 2 |f''|
            # (exact to first order), far above the three-point bound
            # wherever f'' is nonzero (t > 0).
            two_point = check.steps[1] / 2 * np.abs(
                rate_derivative(SEED, 2))  # [s]
            curved = two_point > 0  # entries with t > 0
            assert np.all(two_point[curved]
                          > 100 * (truncation + allowance)[curved])


def test_guard_is_effective_and_central_steps_would_leave_the_domain():
    calls = []
    model, jacobian = guarded([-np.inf, RATE], [RISE, np.inf], calls)
    estimator = make_estimator(model, jacobian)
    with pytest.raises(DomainViolation, match='outside the domain'):
        estimator.check_derivatives()


def test_parameters_outside_bounds_are_rejected_before_any_evaluation():
    calls = []
    model, jacobian = guarded([-np.inf, -np.inf], [np.inf, np.inf], calls)
    estimator = make_estimator(model, jacobian)
    with pytest.raises(ValueError, match=r'params lie outside bounds: k = '):
        estimator.check_derivatives(bounds=[(None, None), (RATE * 2, None)])
    with pytest.raises(ValueError, match=r'rise = .* not in'):
        estimator.check_derivatives(bounds=[(None, RISE / 2), (None, None)])
    assert calls == []


def test_parameter_without_room_for_any_stencil_is_named():
    calls = []
    model, jacobian = guarded([-np.inf, -np.inf], [np.inf, np.inf], calls)
    estimator = make_estimator(model, jacobian)
    rate_step = CENTRAL_DIFFERENCE_REL_STEP * RATE  # [1/s]
    narrow = (RATE - rate_step / 2, RATE + rate_step / 2)  # [1/s]
    with pytest.raises(ValueError, match=r"Neither a central nor a one-sided "
                       r"three-point step of 'k'"):
        estimator.check_derivatives(bounds=[(None, None), narrow])
    assert calls == []


# Domains of k in steps h = CENTRAL_DIFFERENCE_REL_STEP * k around k, with
# the formula expected: [k, k + 1.5 h] and [k - 1.5 h, k] leave room for
# the near point k +/- h of a one-sided stencil but not for its far point
# k +/- 2 h (nor for a central step); [k - 0.5 h, k + 2.5 h] just fits the
# forward stencil.
FAR_POINT_DOMAINS = [((0.0, 1.5), None), ((-1.5, 0.0), None),
                     ((-0.5, 2.5), 'forward')]  # [h], [h]


@pytest.mark.parametrize('domain, formula', FAR_POINT_DOMAINS)
def test_one_sided_stencils_need_their_far_point_inside(domain, formula):
    rate_step = CENTRAL_DIFFERENCE_REL_STEP * RATE  # [1/s]
    low, high = (RATE + offset * rate_step for offset in domain)  # [1/s]
    calls = []
    model, jacobian = guarded([-np.inf, low], [np.inf, high], calls)
    estimator = make_estimator(model, jacobian)
    bounds = [(None, None), (low, high)]
    if formula is None:
        with pytest.raises(ValueError, match="Neither a central nor a "
                           "one-sided three-point step of 'k'"):
            estimator.check_derivatives(bounds=bounds)
        assert calls == []
    else:
        check = estimator.check_derivatives(bounds=bounds)
        assert check.formulas == ('central', formula)
        evaluated = np.array(calls)  # [K*L/mol], [1/s]
        assert np.all((evaluated[:, 1] >= low) & (evaluated[:, 1] <= high))
        # The far point k + 2 h was evaluated.
        assert evaluated[:, 1].max() > RATE + 1.5 * rate_step


def negative_rate_step(params):
    """Return backward forward-difference steps, ``-1e-3 * |p|``.

    Parameters
    ----------
    params : numpy.ndarray
        Full parameter vector: rise [K*L/mol], k [1/s].

    Returns
    -------
    numpy.ndarray
        Steps [K*L/mol], [1/s], negative: the estimator's finite
        differences then evaluate the model below each parameter.
    """
    return -ident.DEFAULT_RELATIVE_STEP * np.abs(params)


# Forward-difference step and the bound it crosses: the default positive
# step against an upper bound of k at k, and a negative callable step
# against a lower bound of k at k.
STEPPING_OUT = [(None, (None, RATE), 'upper'),
                (negative_rate_step, (RATE, None), 'lower')]


@pytest.mark.parametrize('dx_finitediff, rate_bounds, side', STEPPING_OUT)
def test_forward_difference_jacobian_stepping_out_of_bounds_is_refused(
        dx_finitediff, rate_bounds, side):
    calls = []
    # [K*L/mol], [1/s]
    lower = [-np.inf, -np.inf if rate_bounds[0] is None else rate_bounds[0]]
    upper = [np.inf, np.inf if rate_bounds[1] is None else rate_bounds[1]]
    model, _ = guarded(lower, upper, calls)
    estimator = make_estimator(model, jac_fun=None,
                               dx_finitediff=dx_finitediff)
    with pytest.raises(ValueError, match="forward-difference Jacobian would "
                       "evaluate the model outside bounds at k = ") as error:
        estimator.check_derivatives(bounds=[(None, None), rate_bounds])
    evaluated = np.array(calls)  # [K*L/mol], [1/s]
    assert len(calls) > 0  # the point itself
    assert np.all((evaluated[:, 1] >= lower[1])
                  & (evaluated[:, 1] <= upper[1]))
    # The step crosses the named bound: k + 1e-3 k above, k - 1e-3 k below.
    sign = 1.0 if side == 'upper' else -1.0  # [-]
    crossing = RATE * (1 + sign * ident.DEFAULT_RELATIVE_STEP)  # [1/s]
    reported = float(re.search(r'at k = ([^,]+),',
                               str(error.value)).group(1))  # [1/s]
    assert reported == pytest.approx(crossing, rel=4 * EPS)


def test_zero_width_bounds_ask_for_optimize_flags():
    calls = []
    model, jacobian = guarded([-np.inf, -np.inf], [np.inf, np.inf], calls)
    estimator = make_estimator(model, jacobian)
    with pytest.raises(ValueError, match=r"bounds of 'k' have zero width .*"
                       "fix the parameter through optimize_flags"):
        estimator.check_derivatives(bounds=[(None, None), (RATE, RATE)])
    # At a zero parameter without param_scale, the zero width is reported,
    # not the missing scale.
    with pytest.raises(ValueError, match=r"bounds of 'rise' have zero "
                       "width"):
        estimator.check_derivatives(params=[0.0, RATE],
                                    bounds=[(0.0, 0.0), (None, None)])
    assert calls == []


def test_zero_parameter_needs_a_scale():
    estimator = make_estimator(seed=(0.0, RATE))
    with pytest.raises(ValueError, match=r'at zero or not finite \(rise\); '
                       r'pass param_scale.*step h = rel_step \* scale'):
        estimator.check_derivatives()

    check = estimator.check_derivatives(param_scale=[RISE_SCALE, RATE])

    np.testing.assert_array_equal(check.param_scale, [RISE_SCALE, RATE])
    assert_steps(check, CENTRAL_DIFFERENCE_REL_STEP, [RISE_SCALE, RATE])
    _, exact = closed_form((0.0, RATE))
    jacobian, reference = observed_rows(check)
    np.testing.assert_allclose(jacobian, exact, rtol=ident.ROUNDOFF_RTOL)
    truncation, rounding_term = reference_bound((0.0, RATE), 0,
                                                check.steps[0], 'central')
    np.testing.assert_array_less(
        np.abs(reference[0] - exact[0]),
        truncation + rounding_term + ROUNDING_ULPS * EPS * np.abs(exact[0]))
    # The rise row is not zero at rise = 0: c0 (1 - exp(-k t)) for T.
    assert np.linalg.norm(exact[0]) > 0


def fitted_estimator(jac_fun=ident.rise_first_jacobian, model=None):
    """Fit the two-parameter decay.

    Parameters
    ----------
    jac_fun : callable or None, optional
        Jacobian callback; None uses forward differences.
    model : callable, optional
        Model callback; the default is ``case.decay_rise_first``.

    Returns
    -------
    ParameterEstimation
        The fitted estimator.
    """
    estimator = make_estimator(model or case.decay_rise_first, jac_fun)
    estimator.optimize_fn(verbose=False)
    return estimator


@pytest.mark.parametrize('jac_fun', [ident.rise_first_jacobian, None])
def test_fit_state_is_unchanged_bit_for_bit(jac_fun):
    estimator = fitted_estimator(jac_fun)
    before = ident.snapshot(estimator)
    objects = {name: getattr(estimator, name)
               for name in ParameterEstimation._FIT_ATTRIBUTES
               if hasattr(estimator, name)}

    check = estimator.check_derivatives(
        rel_step=TRUNCATION_REL_STEP,
        bounds=[(None, None), (estimator.params_convg[1], None)])

    np.testing.assert_array_equal(check.params, estimator.params_convg)
    assert check.formulas == ('central', 'forward')
    ident.assert_same(before, ident.snapshot(estimator))
    for name, value in objects.items():
        assert getattr(estimator, name) is value, name


def test_fit_state_is_restored_when_an_evaluation_raises():
    calls = []
    # Domain held by reference and tightened in place after the fit.
    lower = np.array([-np.inf, -np.inf])  # [K*L/mol], [1/s]
    model, _ = guarded(lower, [np.inf, np.inf], calls, sensitivities=True)
    estimator = make_estimator(model, None)
    estimator.optimize_fn(verbose=False)
    # Model-returned sensitivities: the point stays valid, and every
    # perturbed evaluation stores its own sens_second before the central
    # step below k raises.
    lower[1] = estimator.params_convg[1]
    before = ident.snapshot(estimator)
    objects = {name: getattr(estimator, name)
               for name in ParameterEstimation._FIT_ATTRIBUTES
               if hasattr(estimator, name)}
    del calls[:]

    with pytest.raises(DomainViolation):
        estimator.check_derivatives()

    # The point, the rise steps and k + h, the failing k - h, then the
    # re-evaluation at params_convg.
    assert len(calls) > 3
    np.testing.assert_array_equal(calls[-1][1], estimator.params_convg[1])
    ident.assert_same(before, ident.snapshot(estimator))
    for name, value in objects.items():
        assert getattr(estimator, name) is value, name


def test_stateful_callback_ends_at_the_accepted_parameters():
    calls = []
    model, _ = guarded([-np.inf, -np.inf], [np.inf, np.inf], calls,
                       sensitivities=True)
    estimator = make_estimator(model, None)
    estimator.optimize_fn(verbose=False)
    del calls[:]

    estimator.check_derivatives(rel_step=TRUNCATION_REL_STEP)

    # The last evaluation, the state a unit operation keeps, is at the
    # accepted parameters again.
    np.testing.assert_array_equal(calls[-1], estimator.params_convg)
    assert any(not np.array_equal(call, estimator.params_convg)
               for call in calls)


def test_failed_reevaluation_warns_and_keeps_the_original_error():
    calls = []
    lower = np.array([-np.inf, -np.inf])  # [K*L/mol], [1/s]
    model, jacobian = guarded(lower, [np.inf, np.inf], calls)
    estimator = make_estimator(model, jacobian)
    estimator.optimize_fn(verbose=False)
    before = ident.snapshot(estimator)
    # The accepted point itself is now outside the guard's domain.
    lower[1] = estimator.params_convg[1] * 2

    with pytest.warns(RuntimeWarning, match='could not re-evaluate the '
                      'model'):
        with pytest.raises(DomainViolation):
            estimator.check_derivatives()
    ident.assert_same(before, ident.snapshot(estimator))

    # A successful check at a valid point whose re-evaluation at
    # params_convg fails reports that failure, with the state restored.
    inside = [estimator.params_convg[0], lower[1] * 2]  # [K*L/mol], [1/s]
    with pytest.raises(DomainViolation):
        estimator.check_derivatives(params=inside)
    ident.assert_same(before, ident.snapshot(estimator))


def test_interrupt_propagates_without_reevaluation_and_restores_state():
    calls = []
    lower = np.array([-np.inf, -np.inf])  # [K*L/mol], [1/s]
    model, _ = guarded(lower, [np.inf, np.inf], calls, sensitivities=True,
                       error=KeyboardInterrupt)
    estimator = make_estimator(model, None)
    estimator.optimize_fn(verbose=False)
    lower[1] = estimator.params_convg[1]
    before = ident.snapshot(estimator)
    del calls[:]

    with pytest.raises(KeyboardInterrupt):
        estimator.check_derivatives()

    # No re-evaluation after an interrupt: the last evaluation is the
    # perturbed point below k that raised.
    assert calls[-1][1] < estimator.params_convg[1]
    ident.assert_same(before, ident.snapshot(estimator))


def test_warnings_as_errors_do_not_mask_the_original_failure():
    calls = []
    lower = np.array([-np.inf, -np.inf])  # [K*L/mol], [1/s]
    model, jacobian = guarded(lower, [np.inf, np.inf], calls)
    estimator = make_estimator(model, jacobian)
    estimator.optimize_fn(verbose=False)
    before = ident.snapshot(estimator)
    # The check point and params_convg both fail.
    lower[1] = estimator.params_convg[1] * 2
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        with pytest.raises(DomainViolation):
            estimator.check_derivatives()
    ident.assert_same(before, ident.snapshot(estimator))


def test_bounds_must_contain_the_restoring_point():
    calls = []
    model, jacobian = guarded([-np.inf, -np.inf], [np.inf, np.inf], calls)
    estimator = make_estimator(model, jacobian)
    estimator.optimize_fn(verbose=False)
    accepted = estimator.params_convg  # [K*L/mol], [1/s]
    before = ident.snapshot(estimator)
    del calls[:]
    # A check point inside bounds that exclude the accepted rate.
    rate_bounds = (accepted[1] * 1.5, accepted[1] * 3)  # [1/s]
    inside = [accepted[0], accepted[1] * 2]  # [K*L/mol], [1/s]
    with pytest.raises(ValueError, match=r'params_convg lies outside bounds '
                       r'\(k = .*widen the bounds to include params_convg '
                       'or check without bounds'):
        estimator.check_derivatives(params=inside,
                                    bounds=[(None, None), rate_bounds])
    assert calls == []
    ident.assert_same(before, ident.snapshot(estimator))
    # Bounds that contain both points are accepted.
    check = estimator.check_derivatives(
        params=inside, bounds=[(None, None), (accepted[1], None)])
    assert check.formulas == ('central', 'central')


def readonly_view_decay():
    """Return an ``exp(-k t)`` model exposing a read-only view of its buffer.

    Returns
    -------
    callable
        ``model(params, time_s)`` returning ``exp(-k t)`` [-] and a
        read-only view of a writeable ``(n_times, 1)`` array [s] that every
        call refills with ``-t exp(-k t)``.
    """
    buffer = np.zeros((len(BUFFER_TIMES), 1))  # [s]
    view = buffer.view()
    view.setflags(write=False)

    def model(params, time_s):
        """Return the outputs and the read-only view.

        Parameters
        ----------
        params : numpy.ndarray
            Rate constant k, shape (1,) [1/s].
        time_s : numpy.ndarray
            Model grid [s].

        Returns
        -------
        tuple of numpy.ndarray
            ``exp(-k t)`` [-] and the read-only sensitivity view [s].
        """
        values = np.exp(-params[0] * time_s)  # [-]
        buffer[:, 0] = -time_s * values
        return values, view

    return model


def test_readonly_view_of_a_refilled_buffer_is_restored():
    estimator = ParameterEstimation(readonly_view_decay(),
                                    [BUFFER_RATE * 2 / 3],
                                    x_data=BUFFER_TIMES,
                                    y_data=np.exp(-BUFFER_RATE
                                                  * BUFFER_TIMES))
    estimator.get_objective([BUFFER_PROBE])
    # [s], identity weights: derivatives at the stored BUFFER_PROBE
    expected = -BUFFER_TIMES * np.exp(-BUFFER_PROBE * BUFFER_TIMES)
    np.testing.assert_array_equal(
        estimator.get_gradient([BUFFER_PROBE], out_array=True)[0], expected)

    estimator.check_derivatives()

    np.testing.assert_array_equal(estimator._params_sens_second,
                                  [BUFFER_PROBE])
    assert not estimator.sens_second[0].flags.writeable
    np.testing.assert_array_equal(
        estimator.get_gradient([BUFFER_PROBE], out_array=True)[0], expected)


# Decay exp(-k t) of the preallocated-sensitivity callback: grid [s],
# accepted-point rate [1/s] (data generated at it, so the fit returns it)
# and an evaluation point away from the seed [1/s].
BUFFER_TIMES = np.array([0.0, 1.0, 2.0])  # [s]
BUFFER_RATE = 0.3  # [1/s]
BUFFER_PROBE = 0.25  # [1/s]
# Relative step of the review reproduction: any step moves the perturbed
# evaluations away from the accepted rate, so they refill the shared array
# with other values.
BUFFER_REL_STEP = 1e-2  # [-]


def preallocated_decay():
    """Return an ``exp(-k t)`` model that refills one sensitivity array.

    Returns
    -------
    callable
        ``model(params, time_s)`` returning ``exp(-k t)`` [-] and its
        derivative ``-t exp(-k t)`` [s] written in place into the same
        preallocated ``(n_times, 1)`` array at every call, as compiled
        models with work arrays do.
    """
    buffer = np.zeros((len(BUFFER_TIMES), 1))  # [s]

    def model(params, time_s):
        """Return the outputs and the refilled sensitivity array.

        Parameters
        ----------
        params : numpy.ndarray
            Rate constant k, shape (1,) [1/s].
        time_s : numpy.ndarray
            Model grid [s].

        Returns
        -------
        tuple of numpy.ndarray
            ``exp(-k t)`` [-] and the shared sensitivity array [s].
        """
        values = np.exp(-params[0] * time_s)  # [-]
        buffer[:, 0] = -time_s * values
        return values, buffer

    return model


def buffer_estimator():
    """Build the estimator of the preallocated-sensitivity model.

    Returns
    -------
    ParameterEstimation
        Unfitted, seeded at 0.2 1/s, data ``exp(-0.3 t)`` [-].
    """
    return ParameterEstimation(preallocated_decay(), [BUFFER_RATE * 2 / 3],
                               x_data=BUFFER_TIMES,
                               y_data=np.exp(-BUFFER_RATE * BUFFER_TIMES))


def test_preallocated_sensitivities_keep_the_accepted_gradient():
    estimator = buffer_estimator()
    estimator.optimize_fn(verbose=False)
    accepted = estimator.params_convg  # [1/s]
    gradient = np.array(estimator.get_gradient(
        accepted, out_array=True))  # [s], identity weights
    before = ident.snapshot(estimator)

    estimator.check_derivatives(rel_step=BUFFER_REL_STEP)

    ident.assert_same(before, ident.snapshot(estimator))
    np.testing.assert_array_equal(
        estimator.get_gradient(accepted, out_array=True), gradient)


def test_preallocated_sensitivities_of_an_earlier_evaluation_are_kept():
    estimator = buffer_estimator()
    estimator.get_objective([BUFFER_PROBE])
    # [s], the stored sensitivities of the evaluation at BUFFER_PROBE
    expected = (-BUFFER_TIMES * np.exp(-BUFFER_PROBE * BUFFER_TIMES)
                )[:, np.newaxis]
    np.testing.assert_allclose(estimator.sens_second[0], expected,
                               rtol=ident.ROUNDOFF_RTOL)
    before = ident.snapshot(estimator)

    # Checked at the seed and re-evaluated there; the stored sensitivities
    # still describe BUFFER_PROBE.
    estimator.check_derivatives()

    ident.assert_same(before, ident.snapshot(estimator))
    np.testing.assert_array_equal(estimator._params_sens_second,
                                  [BUFFER_PROBE])


def test_unfitted_estimator_keeps_its_empty_histories():
    estimator = make_estimator()
    history = estimator.params_iter
    estimator.check_derivatives()
    assert estimator.params_iter is history and history == []
    assert estimator.objfun_iter == []
    assert estimator.params_residuals is None
    assert estimator.sens is None
    assert not hasattr(estimator, 'params_convg')


def test_fixed_parameters_are_not_perturbed():
    calls = []
    # The guard pins the fixed rise to its value exactly.
    model, jacobian = guarded([RISE, -np.inf], [RISE, np.inf], calls)
    estimator = make_estimator(model, jacobian,
                               optimize_flags=[False, True])

    check = estimator.check_derivatives()

    assert check.parameter_names == ('k',)
    assert check.jacobian.shape == (1, NUM_ENTRIES)
    np.testing.assert_array_equal(check.params, [RATE])
    evaluated = np.array(calls)  # [K*L/mol], [1/s]
    np.testing.assert_array_equal(evaluated[:, 0], RISE)
    assert np.any(evaluated[:, 1] != RATE)
    _, exact = closed_form(SEED)
    np.testing.assert_allclose(check.jacobian[0, EXPECTED_OBSERVED],
                               exact[1], rtol=ident.ROUNDOFF_RTOL)


def test_multiple_curve_resolution_masks_unobserved_tracer_entries():
    estimator = mcr_case._estimator(mcr_case.model_with_sensitivities,
                                    'correlated')
    check = estimator.check_derivatives(mcr_case.TRIAL_RATE)

    unobserved = mcr_case._unobserved_tracer_entries()
    num_data = unobserved.size  # [-]
    assert check.jacobian_source == 'model'
    # Data entries, then the always-counted absorptivity-penalty rows.
    np.testing.assert_array_equal(check.observed[:num_data], ~unobserved)
    assert check.observed[num_data:].all()
    np.testing.assert_array_equal(check.jacobian[0, :num_data][unobserved],
                                  0.0)
    np.testing.assert_array_equal(check.reference[0, :num_data][unobserved],
                                  0.0)
    # mcr_case's central-difference accuracy for its own step, 1e-7 of the
    # largest entry, bounds the default-step reference too (rounding eps |r|
    # / h near 1e-10 relative at h = 6e-6 k, truncation near 1e-11).
    assert check.relative_error[0] < mcr_case.DERIVATIVE_RTOL


def test_result_is_immutable_and_round_trips():
    check = make_estimator().check_derivatives(rtol=MIDWAY_RTOL)
    with pytest.raises(dataclasses.FrozenInstanceError):
        check.rtol = None
    with pytest.raises(ValueError, match='read-only'):
        check.jacobian[0, 0] = 0.0
    with pytest.raises(ValueError, match='read-only'):
        check.agrees[0] = False
    for restored in (pickle.loads(pickle.dumps(check)), copy.deepcopy(check)):
        for field in dataclasses.fields(check):
            value = getattr(restored, field.name)
            if isinstance(value, np.ndarray):
                np.testing.assert_array_equal(value,
                                              getattr(check, field.name))
                assert not value.flags.writeable
            else:
                assert value == getattr(check, field.name)


def test_table_and_summary_report_each_parameter():
    estimator = make_estimator(jac_fun=scaled_jacobian(1))
    check = estimator.check_derivatives(rtol=MIDWAY_RTOL)
    frame = check.to_frame()

    assert list(frame.index) == list(NAMES)
    assert list(frame.columns) == ['value', 'step', 'rel_step', 'formula',
                                   'max_abs_error', 'relative_error',
                                   'agrees']
    np.testing.assert_array_equal(frame['relative_error'],
                                  check.relative_error)
    assert list(frame['agrees']) == [True, False]
    assert frame.attrs == {'jacobian_source': 'jac_fun',
                           'rtol': MIDWAY_RTOL,
                           'max_relative_error': check.max_relative_error}
    observed_error = np.abs(check.jacobian - check.reference)[
        :, EXPECTED_OBSERVED]  # [1/p_i]
    np.testing.assert_array_equal(frame['max_abs_error'],
                                  observed_error.max(axis=1))
    text = check.summary()
    assert 'user-supplied jac_fun' in text
    assert 'over 7 observed entries' in text
    assert 'NO' in text.splitlines()[3] and 'yes' in text.splitlines()[2]
    assert 'solver_rtol**(1/3)' in text
    assert 'agrees' not in make_estimator().check_derivatives().to_frame()


def test_documented_ode_step_is_the_cube_root_of_cvode_rtol():
    # rtol**(1/3) at CVode's default rtol 1e-6 is 1e-2; the default is the
    # cube root of the machine epsilon.
    assert DEFAULT_OUTPUT_RTOL ** (1 / 3) == pytest.approx(1e-2, rel=EPS * 4)
    assert CENTRAL_DIFFERENCE_REL_STEP ** 3 == pytest.approx(EPS,
                                                             rel=EPS * 8)


@pytest.mark.parametrize('options, error, match', [
    ({'rel_step': 0.0}, ValueError, 'rel_step entries|rel_step must be'),
    ({'rel_step': 1.0}, ValueError, 'rel_step must be strictly between'),
    ({'rel_step': np.nan}, ValueError, 'rel_step must be strictly between'),
    ({'rel_step': True}, TypeError, 'rel_step must be a real number'),
    ({'rel_step': [1e-4, 1e-4, 1e-4]}, ValueError,
     r'rel_step must be a scalar or have shape \(2,\)'),
    ({'rel_step': [1e-4, -1e-4]}, ValueError, 'rel_step entries must be'),
    ({'param_scale': [1.0, 0.0]}, ValueError,
     'param_scale entries must be positive'),
    ({'param_scale': [1.0]}, ValueError,
     r'param_scale must be a scalar or have shape \(2,\)'),
    ({'param_scale': 'big'}, TypeError, 'param_scale must hold real'),
    ({'bounds': [(0.0, 50.0)]}, ValueError,
     r'one \(lower, upper\) pair per optimized parameter, or a '
     r'scipy.optimize.Bounds.*for 2 parameters \(rise, k\); got 1'),
    ({'bounds': ([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])}, ValueError,
     r"bounds of 'rise' must be a \(lower, upper\) pair.*"
     r"scipy.optimize.Bounds"),
    ({'bounds': [(0.0, 50.0, 1.0), (0.0, 1.0)]}, ValueError,
     r"bounds of 'rise' must be a \(lower, upper\) pair"),
    ({'bounds': [(np.nan, 50.0), (0.0, 1.0)]}, ValueError,
     "bounds of 'rise' must not be NaN"),
    ({'bounds': [(0.0, 50.0), (1.0, 0.0)]}, ValueError,
     r"lower bound of 'k' \(1.0\) exceeds its upper bound \(0.0\)"),
    ({'bounds': [(0.0, 50.0), (np.inf, None)]}, ValueError,
     "bounds of 'k' exclude every finite value"),
    ({'bounds': [(0.0, '50'), (0.0, 1.0)]}, TypeError,
     "bounds of 'rise' must be real numbers or None"),
    ({'bounds': (0.0, 50.0)}, ValueError, r"bounds of 'rise' must be a"),
    ({'bounds': 'wide'}, TypeError, 'bounds must be one'),
    ({'params': [RISE]}, ValueError, r'params must have shape \(2,\)'),
    ({'params': [RISE, np.nan]}, ValueError, 'params must be finite'),
    ({'params': [[RISE, RATE]]}, ValueError, 'one-dimensional'),
    ({'rtol': -1.0}, ValueError, 'rtol must be positive and finite'),
    ({'rtol': True}, TypeError, 'rtol must be a real number'),
])
def test_invalid_inputs_are_rejected_before_any_evaluation(options, error,
                                                           match):
    calls = []
    model, jacobian = guarded([-np.inf, -np.inf], [np.inf, np.inf], calls)
    estimator = make_estimator(model, jacobian)
    with pytest.raises(error, match=match):
        estimator.check_derivatives(**options)
    assert calls == []


# Unequal relative steps of the rise and k, so that swapped parameters
# would show; both are valid steps in (0, 1).
ARRAY_REL_STEPS = (1e-3, 1e-4)  # [-]
# A common scale for both parameters, different from |rise| and |k|.
COMMON_SCALE = 2.0  # [K*L/mol] for the rise, [1/s] for k


def test_array_valued_steps_scales_and_bounds_keep_parameter_order():
    estimator = make_estimator()
    steps = estimator.check_derivatives(rel_step=list(ARRAY_REL_STEPS))
    np.testing.assert_array_equal(steps.rel_step, ARRAY_REL_STEPS)
    # Ratios h / |p| equal rel_step up to the rounding of p + h, at most
    # 2 EPS |p + h| / h, i.e. 4 EPS / rel_step relative with margin.
    np.testing.assert_allclose(steps.steps / np.abs(SEED), ARRAY_REL_STEPS,
                               rtol=4 * EPS / min(ARRAY_REL_STEPS))
    scalar = estimator.check_derivatives(param_scale=COMMON_SCALE)
    np.testing.assert_array_equal(scalar.param_scale, [COMMON_SCALE] * 2)
    assert_steps(scalar, CENTRAL_DIFFERENCE_REL_STEP, [COMMON_SCALE] * 2)
    array_bounds = estimator.check_derivatives(
        bounds=np.array([[-np.inf, RISE], [RATE, np.inf]]))
    assert array_bounds.formulas == ('backward', 'forward')


def test_scipy_bounds_are_lower_and_upper_arrays():
    calls = []
    model, jacobian = guarded([-np.inf, RATE], [RISE, np.inf], calls)
    estimator = make_estimator(model, jacobian)
    check = estimator.check_derivatives(
        bounds=Bounds([-np.inf, RATE], [RISE, np.inf]))
    assert check.formulas == ('backward', 'forward')
    evaluated = np.array(calls)  # [K*L/mol], [1/s]
    assert np.all(evaluated[:, 0] <= RISE)
    assert np.all(evaluated[:, 1] >= RATE)
    # Scalar lb and ub apply to every parameter.
    unguarded = make_estimator()
    both = unguarded.check_derivatives(bounds=Bounds(RATE, np.inf))
    assert both.formulas == ('central', 'forward')
    with pytest.raises(ValueError, match=r'Bounds lb and ub must be '
                       r'scalars or have shape \(2,\)'):
        unguarded.check_derivatives(bounds=Bounds([0.0] * 3, [1.0] * 3))


# Extreme scales of the one-parameter line y = p * scale * t at p [-]:
# derivatives near the float64 maximum (1e308) and near its underflow
# (1e-300). The wrong Jacobian is the exact one times WRONG_FACTOR.
EXTREME_CASES = [(1e-308, 1e308), (1.0, 1e-300)]  # [-], [1/s]
EXTREME_TIMES = np.array([0.5, 1.0, 1.5])  # [s], keeps 1.1 * 1e308 * t finite


@pytest.mark.parametrize('value, scale', EXTREME_CASES)
def test_relative_error_is_exact_at_extreme_magnitudes(value, scale):
    def line(params, time_s):
        """Return ``p * scale * t`` [-].

        Parameters
        ----------
        params : numpy.ndarray
            Shape (1,) [-].
        time_s : numpy.ndarray
            Model grid [s].

        Returns
        -------
        numpy.ndarray
            Shape (n_times,) [-].
        """
        return params[0] * scale * time_s

    def wrong_jacobian(params, time_s):
        """Return ``WRONG_FACTOR * scale * t`` [1/(-)], shape (n, 1).

        Parameters
        ----------
        params : numpy.ndarray
            Shape (1,) [-].
        time_s : numpy.ndarray
            Model grid [s].

        Returns
        -------
        numpy.ndarray
            Shape (n_times, 1).
        """
        return (WRONG_FACTOR * scale * time_s)[:, np.newaxis]

    estimator = ParameterEstimation(line, [value], x_data=EXTREME_TIMES,
                                    y_data=value * scale * EXTREME_TIMES,
                                    jac_fun=wrong_jacobian)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        check = estimator.check_derivatives(rtol=MIDWAY_RTOL)

    exact = scale * EXTREME_TIMES  # [1/(-)]
    step = check.steps[0]  # [-]
    # Reference error: rounding of the two evaluated outputs (ROUNDING_ULPS
    # of |y|) over 2h, relative to the exact row; the line has no
    # truncation error.
    outputs = np.abs(value * scale * EXTREME_TIMES)  # [-]
    reference_error = np.linalg.norm(
        ROUNDING_ULPS * EPS * 2 * outputs / (2 * step) / exact)
    low = (WRONG_FACTOR - 1 - reference_error) / (1 + reference_error)
    high = (WRONG_FACTOR - 1 + reference_error) / (1 - reference_error)
    assert low <= check.relative_error[0] <= high
    np.testing.assert_array_equal(check.agrees, [False])


def test_step_below_the_floating_point_spacing_is_rejected():
    estimator = make_estimator()
    with pytest.raises(ValueError, match="The step of 'rise'.*vanishes"):
        estimator.check_derivatives(param_scale=[RISE * EPS / 8, RATE])


def test_from_derivatives_validates_its_inputs():
    arrays = dict(parameter_names=['a'], params=[1.0], rel_step=[1e-3],
                  param_scale=[1.0], steps=[1e-3], formulas=['central'],
                  jacobian_source='jac_fun', jacobian=[[1.0, 0.0]],
                  reference=[[1.0, 0.0]], observed=[True, False])
    check = DerivativeCheck.from_derivatives(**arrays)
    np.testing.assert_array_equal(check.relative_error, [0.0])
    with pytest.raises(ValueError, match='must have shape'):
        DerivativeCheck.from_derivatives(**{**arrays,
                                            'reference': [[1.0]]})
    with pytest.raises(ValueError, match='formulas must hold one of'):
        DerivativeCheck.from_derivatives(**{**arrays,
                                            'formulas': ['centered']})
    with pytest.raises(ValueError, match='Unknown jacobian_source'):
        DerivativeCheck.from_derivatives(**{**arrays,
                                            'jacobian_source': 'exact'})
    # Two zero rows agree; a nonzero row against a zero reference does not.
    zero = DerivativeCheck.from_derivatives(
        **{**arrays, 'jacobian': [[0.0, 0.0]], 'reference': [[0.0, 0.0]]})
    np.testing.assert_array_equal(zero.relative_error, [0.0])
    lone = DerivativeCheck.from_derivatives(
        **{**arrays, 'reference': [[0.0, 0.0]]})
    assert lone.relative_error[0] == 1.0 / np.finfo(float).tiny
    # A row of norm above 4 against a zero reference: ||d|| / tiny exceeds
    # the float64 range.
    tall = DerivativeCheck.from_derivatives(
        **{**arrays, 'jacobian': [[1e10, 0.0]], 'reference': [[0.0, 0.0]]})
    assert tall.relative_error[0] == np.inf
    # Overflowing differences and norms keep the true ratio: entries 5 %
    # below the float64 maximum, whose difference and squares overflow.
    near_max = 0.95 * np.finfo(float).max  # [1/p]
    huge = DerivativeCheck.from_derivatives(
        **{**arrays, 'jacobian': [[-near_max, near_max]],
           'reference': [[near_max, near_max]], 'observed': [True, True]})
    # ||(-2, 0)|| / ||(1, 1)|| after scaling by near_max.
    assert huge.relative_error[0] == pytest.approx(np.sqrt(2), rel=4 * EPS)
    # Non-finite entries give NaN without NumPy warnings.
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        for bad in (np.inf, np.nan):
            nonfinite = DerivativeCheck.from_derivatives(
                **{**arrays, 'jacobian': [[bad, 0.0]],
                   'reference': [[np.inf, 0.0]], 'rtol': 1.0})
            assert np.isnan(nonfinite.relative_error[0])
            np.testing.assert_array_equal(nonfinite.agrees, [False])
            assert np.isnan(nonfinite.max_abs_error[0])
    for label in ('params', 'rel_step', 'param_scale', 'steps'):
        with pytest.raises(ValueError, match=rf'{label} must have shape '
                           r'\(1,\)'):
            DerivativeCheck.from_derivatives(**{**arrays, label: [1.0, 2.0]})
    with pytest.raises(ValueError, match='formulas must hold one of'):
        DerivativeCheck.from_derivatives(
            **{**arrays, 'formulas': ['central', 'central']})
    # A difference far below the largest entry is not lost to squaring:
    # (1e-200)**2 underflows, the scaled norm does not. The threshold sits
    # below the true relative error 1e-200, so the row must not agree.
    small = 1e-200  # [1/p]
    underflow = DerivativeCheck.from_derivatives(
        **{**arrays, 'jacobian': [[1.0, small]], 'reference': [[1.0, 0.0]],
           'observed': [True, True], 'rtol': small / 2})
    assert underflow.relative_error[0] == pytest.approx(small, rel=4 * EPS)
    np.testing.assert_array_equal(underflow.agrees, [False])
    # An unobserved entry does not enter the norms.
    masked = DerivativeCheck.from_derivatives(
        **{**arrays, 'jacobian': [[1.0, 5.0]]})
    np.testing.assert_array_equal(masked.relative_error, [0.0])
    np.testing.assert_array_equal(masked.max_abs_error, [0.0])
