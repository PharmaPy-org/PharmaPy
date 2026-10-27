"""Finite-difference perturbations of ``PharmaPy.jac_module`` (issue #239).

Covers the helpers ``dx_jac_p``, ``numerical_jac``, ``numerical_jac_data``,
``numerical_jac_central`` and ``numerical_jacv`` at zero, integer, very
small, large and mixed-scale inputs, and their public callers: the default
``dx_finitediff`` of ``ParameterEstimation`` in a Levenberg-Marquardt fit
seeded at a zero parameter, and the finite-difference Jacobian of
``MultipleCurveResolution`` with an integer seed.

Expected derivatives are analytic. Forward-difference results are compared
with the error bound of Nocedal and Wright (Numerical Optimization, 2nd ed.,
Sec. 8.1): truncation ``h/2 |f''|`` plus rounding ``2 u |f| / h`` for a
step ``h`` and unit roundoff ``u``, plus the rounding of ``x + h`` itself.
Fitted parameters are compared with closed-form least-squares optima
(``numpy.linalg.lstsq``) or noise-free synthetic data. The fixtures are
synthetic contract cases, not calibrated process models; everything runs in
well under a second.
"""

import warnings

import numpy as np
import pytest

from PharmaPy.jac_module import (dx_jac_p, numerical_jac,
                                 numerical_jac_central, numerical_jac_data,
                                 numerical_jacv)
from PharmaPy.ParamEstim import MultipleCurveResolution, ParameterEstimation


pytestmark = pytest.mark.unit

UNIT_ROUNDOFF = np.finfo(float).eps / 2  # [-], float64 round-to-nearest
SQRT_EPS = np.sqrt(np.finfo(float).eps)  # [-], jac_module.eps
# Settings of the ParameterEstimation default step (dparam_fun): relative
# model accuracy [-] and parameter-magnitude floor [parameter units].
DEFAULT_REL_TOL = 1e-6
DEFAULT_ABS_TOL = 1e-6
# Previous default step |p| * sqrt(1e-6) as a relative step [-]; it must be
# kept exactly for every |p| >= DEFAULT_ABS_TOL.
PREVIOUS_RELATIVE_STEP = 1e-3
# Step of a parameter below the floor: 1e-3 * 1e-6 [parameter units].
FLOOR_STEP = 1e-9
# The steps are one square root and one product of correctly rounded
# float64 values, and 1e-3 has a representation error of at most one unit
# in the last place: three ulps bound the difference from the decimal
# expectation, so four machine epsilons suffice [-].
STEP_RTOL = 4 * np.finfo(float).eps
# Levenberg-Marquardt default gradient stop eps_1 [residual**2 / param].
LM_GRADIENT_TOL = 1e-8


def forward_difference_bound(step, x_abs, f_abs, f2_abs, df_abs):
    """Bound the error of a forward difference.

    Parameters
    ----------
    step : float or numpy.ndarray
        Forward step ``h`` [coordinate units], of either sign.
    x_abs : float or numpy.ndarray
        Magnitude of the differentiated coordinate [coordinate units].
    f_abs : float or numpy.ndarray
        Largest magnitude of the function at ``x`` and ``x + h``
        [output units].
    f2_abs : float or numpy.ndarray
        Largest magnitude of the second derivative on ``[x, x + h]``
        [output units / coordinate units**2].
    df_abs : float or numpy.ndarray
        Magnitude of the first derivative [output units / coordinate
        units].

    Returns
    -------
    float or numpy.ndarray
        ``h/2 f2 + 2 u f / h + u ((|x| + h) / h + 2) |f'|`` [output units
        / coordinate units]: truncation, rounding of the two function
        values (Nocedal and Wright, Sec. 8.1), rounding of ``x + h``, which
        changes the realized step by up to ``u (|x| + h)``, and rounding of
        the difference and the quotient, ``u |f'|`` each.
    """
    step = np.abs(step)
    return (step / 2 * f2_abs + 2 * UNIT_ROUNDOFF * f_abs / step
            + UNIT_ROUNDOFF * ((x_abs + step) / step + 2) * df_abs)


def contract_step(params):
    """Return the contracted default step, written independently.

    Parameters
    ----------
    params : numpy.ndarray
        Parameter values [parameter units].

    Returns
    -------
    numpy.ndarray
        ``1e-3 * |p|`` for ``|p| >= 1e-6``, otherwise 1e-9 away from zero
        (positive at zero) [parameter units].
    """
    params = np.asarray(params, dtype=float)
    steps = PREVIOUS_RELATIVE_STEP * np.abs(params)
    below = np.abs(params) < DEFAULT_ABS_TOL
    steps[below] = np.where(params[below] < 0, -FLOOR_STEP, FLOOR_STEP)
    return steps


def default_rule(params):
    """Apply ``dx_jac_p`` with the ParameterEstimation default settings.

    Parameters
    ----------
    params : numpy.ndarray
        Parameter values [parameter units].

    Returns
    -------
    numpy.ndarray
        Steps [parameter units].
    """
    return dx_jac_p(params, DEFAULT_ABS_TOL, DEFAULT_REL_TOL,
                    np.finfo(float).eps)


def three_outputs(params):
    """Evaluate an asymmetric test function of two coordinates.

    Parameters
    ----------
    params : numpy.ndarray
        Coordinates ``(p0, p1)`` [-].

    Returns
    -------
    numpy.ndarray
        ``(p0**2 + p1, exp(p1), p0 * p1)`` [-], shape (3,).
    """
    return np.array([params[0]**2 + params[1], np.exp(params[1]),
                     params[0] * params[1]])


def three_outputs_jacobian(params):
    """Return the analytic Jacobian of ``three_outputs``.

    Parameters
    ----------
    params : numpy.ndarray
        Coordinates ``(p0, p1)`` [-].

    Returns
    -------
    numpy.ndarray
        Shape (3, 2), rows outputs and columns coordinates [-].
    """
    p0, p1 = params
    return np.array([[2 * p0, 1.0], [0.0, np.exp(p1)], [p1, p0]])


def three_outputs_bound(params, steps):
    """Bound the forward-difference error of ``three_outputs``.

    Parameters
    ----------
    params : numpy.ndarray
        Coordinates ``(p0, p1)`` [-].
    steps : numpy.ndarray
        Forward steps of ``p0`` and ``p1`` [-].

    Returns
    -------
    numpy.ndarray
        Entry-wise bound, shape (3, 2) [-].
    """
    p0, p1 = np.asarray(params, dtype=float)
    h0, h1 = steps
    # Second derivatives along p0 and p1, maximized on the step interval.
    second = np.array([[2.0, 0.0], [0.0, np.exp(max(p1, p1 + h1))],
                       [0.0, 0.0]])
    f_abs = np.abs([[p0**2 + p1, (p0 + h0)**2 + p1],
                    [np.exp(p1), np.exp(p1 + h1)],
                    [p0 * p1, (p0 + h0) * (p1 + h1)]])
    f_max = np.column_stack((f_abs.max(axis=1), f_abs.max(axis=1)))
    x_abs = np.abs([p0, p1])
    return forward_difference_bound(
        np.array([h0, h1]), x_abs, f_max, second,
        np.abs(three_outputs_jacobian([p0, p1])))


def proportional(params, time_s):
    """Return ``k t``, a one-parameter model for step checks.

    Parameters
    ----------
    params : numpy.ndarray
        ``(k,)`` [mol/L/s].
    time_s : numpy.ndarray
        Times [s], shape (n,).

    Returns
    -------
    numpy.ndarray
        Concentration [mol/L], shape (n,).
    """
    return params[0] * time_s


def straight_line(params, time_s):
    """Return ``a + b t``.

    Parameters
    ----------
    params : numpy.ndarray
        ``(a [mol/L], b [mol/L/s])``.
    time_s : numpy.ndarray
        Times [s], shape (n,).

    Returns
    -------
    numpy.ndarray
        Concentration [mol/L], shape (n,).
    """
    return params[0] + params[1] * time_s


def unchanged(x):
    """Return the coordinates unchanged (the identity function).

    Parameters
    ----------
    x : numpy.ndarray
        Coordinates [-], shape (n,).

    Returns
    -------
    numpy.ndarray
        ``x`` [-], shape (n,).
    """
    return x


def squared_norm(x):
    """Return ``x . x`` as a Python float (a scalar-valued output).

    Parameters
    ----------
    x : numpy.ndarray
        Coordinates [-], shape (n,).

    Returns
    -------
    float
        ``x . x`` [-].
    """
    return float(x @ x)


POINTS = {
    'zero': np.array([0.0, 0.0]),
    'integer': np.array([2, 1]),
    'very_small': np.array([1e-10, -1e-12]),
    'large': np.array([3e5, 5.0]),
}


@pytest.mark.parametrize('params, expected', [
    (np.array([0.0, 1e-12, -5e-7, 1e-6, -1e-6, -2.5, 3e5]),
     np.array([FLOOR_STEP, FLOOR_STEP, -FLOOR_STEP, FLOOR_STEP, FLOOR_STEP,
               2.5e-3, 300.0])),
    (np.array([0, -3]), np.array([FLOOR_STEP, 3e-3])),
])
def test_dx_jac_p_is_nonzero_and_floating(params, expected):
    steps = default_rule(params)

    assert steps.dtype == np.float64
    np.testing.assert_allclose(steps, expected, rtol=STEP_RTOL, atol=0)


def test_dx_jac_p_keeps_the_previous_step_above_the_floor():
    """``|p| * 1e-3`` is unchanged for every ``|p| >= abstol``."""
    magnitudes = np.logspace(-6, 8, 29)  # [parameter units]
    params = np.concatenate((magnitudes, -magnitudes))

    np.testing.assert_allclose(default_rule(params),
                               PREVIOUS_RELATIVE_STEP * np.abs(params),
                               rtol=STEP_RTOL, atol=0)


def test_dx_jac_p_without_a_floor_keeps_the_relative_step():
    """``abstol=None``, the ``numerical_jac_central`` default."""
    params = np.array([0.0, -2.0, 3e5])  # [parameter units]

    steps = dx_jac_p(params, None, DEFAULT_REL_TOL, np.finfo(float).eps)

    np.testing.assert_allclose(steps, [0.0, 2e-3, 300.0], rtol=STEP_RTOL,
                               atol=0)


def test_numerical_jac_central_with_the_parameter_step_rule():
    """Default ``abs_tol=None`` reaches ``dx_jac_p`` as its ``abstol``."""
    point = np.array([3.0, -2.0])  # [-]

    def quadratic(x):
        """Evaluate ``(x0**2, x0 * x1)``.

        Parameters
        ----------
        x : numpy.ndarray
            Coordinates [-], shape (2,).

        Returns
        -------
        numpy.ndarray
            ``(x0**2, x0 * x1)`` [-], shape (2,).
        """
        return np.array([x[0]**2, x[0] * x[1]])

    jacobian = numerical_jac_central(quadratic, point, dx=dx_jac_p,
                                     rel_tol=DEFAULT_REL_TOL)

    expected = np.array([[6.0, 0.0], [-2.0, 3.0]])
    # Central differences are exact for quadratics: rounding of values of
    # magnitude <= f_max over 2h, plus the rounding of x +/- h.
    steps = PREVIOUS_RELATIVE_STEP * np.abs(point)  # [-]
    f_max = (np.abs(point).max() + steps.max())**2
    atol = (UNIT_ROUNDOFF * f_max / steps
            + UNIT_ROUNDOFF * (np.abs(point) + steps) / steps
            * np.abs(expected).max(axis=0))
    assert np.all(np.abs(jacobian - expected) <= atol)


def test_dx_jac_p_uses_the_noise_floor_below_reltol():
    # reltol 0 falls back to eps = 1e-10, so the factor is sqrt(1e-10).
    steps = dx_jac_p(np.array([0.0, 4.0]), 1e-6, 0.0, 1e-10)

    np.testing.assert_allclose(steps, [1e-5 * 1e-6, 1e-5 * 4.0],
                               rtol=STEP_RTOL, atol=0)


def test_parameter_estimation_default_step():
    time_s = np.linspace(0.0, 1.0, 3)  # [s]
    estimator = ParameterEstimation(proportional, [0.0], time_s, time_s)
    # [parameter units]
    params = np.array([0.0, 1e-12, -1e-12, 2.5, -3e5])

    np.testing.assert_allclose(
        estimator.dx_fd(params),
        [FLOOR_STEP, FLOOR_STEP, -FLOOR_STEP, 2.5e-3, 300.0],
        rtol=STEP_RTOL, atol=0)


@pytest.mark.parametrize('negative_param', [-1e-10, -5e-10])
def test_default_step_keeps_small_negative_parameters_negative(
        negative_param):
    """``log(-p)`` needs p < 0; the floor step must not cross zero."""
    time_s = np.linspace(0.1, 1.0, 4)  # [s]

    def log_model(params, t):
        """Evaluate ``log(-p0) + p1 t``.

        Parameters
        ----------
        params : numpy.ndarray
            ``(p0 [-], p1 [1/s])``, with ``p0 < 0``.
        t : numpy.ndarray
            Times [s], shape (n,).

        Returns
        -------
        numpy.ndarray
            Model output [-], shape (n,).
        """
        return np.log(-params[0]) + params[1] * t

    params = np.array([negative_param, 1.0])
    estimator = ParameterEstimation(log_model, params, time_s,
                                    log_model(params, time_s))

    jacobian = estimator.get_gradient(params, out_array=True)

    # The forward difference with the step the estimator uses is the
    # secant log(1 + h/p) / h of log(-p), here along the realized step
    # (p + h) - p; h must be negative (about -1e-9) so that p + h < 0.
    step = estimator.dx_fd(params)[0]  # [-]
    realized = (negative_param + step) - negative_param  # [-]
    secant = np.log1p(realized / negative_param) / step  # [-]
    # Rounding only: both log values (|f| <= f_abs) over |h|, and a few
    # ulp of the secant itself (log1p, division, and the difference).
    f_abs = np.abs(log_model(params, time_s)).max() + abs(np.log1p(
        realized / negative_param))
    tolerance = (2 * UNIT_ROUNDOFF * f_abs / abs(step)
                 + 4 * UNIT_ROUNDOFF * abs(secant))
    assert step < 0
    assert np.all(np.isfinite(jacobian))
    assert np.all(jacobian[0] < 0)
    assert np.all(np.abs(jacobian[0] - secant) <= tolerance)


@pytest.mark.parametrize('point', list(POINTS))
def test_numerical_jac_data_matches_analytic_jacobian(point):
    params = POINTS[point]
    float_params = params.astype(float)

    jacobian = numerical_jac_data(three_outputs, params, dx=default_rule)

    expected = three_outputs_jacobian(float_params)
    bound = three_outputs_bound(float_params, contract_step(float_params))
    assert jacobian.shape == (3, 2)
    assert np.all(np.abs(jacobian - expected) <= bound)


def test_numerical_jac_data_orders_columns_by_pick_x():
    params = np.array([2, 1])

    jacobian = numerical_jac_data(three_outputs, params, dx=default_rule,
                                  pick_x=[1, 0])

    expected = three_outputs_jacobian([2.0, 1.0])[:, [1, 0]]
    bound = three_outputs_bound([2.0, 1.0],
                                contract_step(np.array([2.0, 1.0])))[:, [1, 0]]
    assert np.all(np.abs(jacobian - expected) <= bound)


@pytest.mark.parametrize('params, expected', [
    (np.array([0.0, 0.0]), [4.0, 1.0]),
    (np.array([1, 2]), [16.0, 4.0]),
])
def test_numerical_jac_gradient_of_a_scalar(params, expected):
    """Gradient of ``(p0 + 1)**2 * (p1 + 2)`` with the default step rule."""
    def scalar(p):
        """Evaluate ``(p0 + 1)**2 * (p1 + 2)``.

        Parameters
        ----------
        p : numpy.ndarray
            Coordinates [-], shape (2,).

        Returns
        -------
        float
            Function value [-].
        """
        return (p[0] + 1)**2 * (p[1] + 2)

    gradient = numerical_jac(scalar, params, dx=default_rule)

    float_params = params.astype(float)
    steps = contract_step(float_params)
    p1 = float_params[1]
    bound = forward_difference_bound(
        steps, np.abs(float_params),
        np.abs(scalar(float_params + steps)),
        np.array([2 * (p1 + 2), 0.0]), np.abs(expected))
    assert gradient.shape == (1, 2)
    assert np.all(np.abs(gradient[0] - expected) <= bound)


def test_numerical_jac_identity_with_an_integer_seed():
    jacobian = numerical_jac(unchanged, np.array([1]))

    np.testing.assert_allclose(jacobian, [[1.0]], rtol=0,
                               atol=forward_difference_bound(
                                   SQRT_EPS, 1.0, 1.0, 0.0, 1.0))


def test_numerical_jac_central_with_an_integer_seed():
    """Central differences are exact for quadratics up to rounding."""
    step = 1e-3  # [-]

    def quadratic(x):
        """Evaluate ``(x0**2, x0 * x1)``.

        Parameters
        ----------
        x : numpy.ndarray
            Coordinates [-], shape (2,).

        Returns
        -------
        numpy.ndarray
            ``(x0**2, x0 * x1)`` [-], shape (2,).
        """
        return np.array([x[0]**2, x[0] * x[1]])

    point = np.array([3, 2])

    jacobian = numerical_jac_central(quadratic, point, dx=step)

    expected = np.array([[6.0, 0.0], [2.0, 3.0]])
    # Rounding of two values of magnitude <= f_max over 2h, plus the
    # rounding of x +/- h in the realized step, per column.
    f_max = (np.abs(point).max() + step)**2
    atol = (2 * UNIT_ROUNDOFF * f_max / (2 * step)
            + UNIT_ROUNDOFF * (np.abs(point) + step) / step
            * np.abs(expected).max(axis=0))
    assert np.all(np.abs(jacobian - expected) <= atol)


@pytest.mark.parametrize('jac_fun', [numerical_jac, numerical_jac_data])
@pytest.mark.parametrize('bad_step', [0.0, np.inf, -np.inf])
def test_forward_differences_reject_invalid_steps(jac_fun, bad_step):
    steps = np.array([1e-6, bad_step])

    with pytest.raises(ValueError, match=r"nonzero and finite.*\[1\]"):
        jac_fun(three_outputs, np.array([1.0, 2.0]), dx=steps)


def test_forward_differences_ignore_steps_of_unpicked_coordinates():
    steps = np.array([1e-6, 0.0])  # [-], zero step of an unpicked p1
    params = np.array([1.0, 2.0])

    jacobian = numerical_jac_data(three_outputs, params, dx=steps, pick_x=[0])

    bound = three_outputs_bound(params, steps + [0.0, 1e-6])[:, :1]
    expected = three_outputs_jacobian(params)[:, :1]
    assert np.all(np.abs(jacobian - expected) <= bound)


def directional_increment(x, direction):
    """Return the contracted directional increment, written independently.

    Parameters
    ----------
    x : numpy.ndarray
        State [state units].
    direction : numpy.ndarray
        ``v / max|v|`` [-].

    Returns
    -------
    float
        ``sqrt(eps) * max(|x . w|, ||w||_1) / ||w||**2``, at least
        ``sqrt(eps) * max_i(|w_i| * max(|x_i|, 1))``, signed as ``x . w``
        (positive at zero) [state units]; ``typx = 1``.
    """
    x = np.atleast_1d(x)
    direction = np.atleast_1d(direction)
    projection = np.sum(x * direction)  # [state units]
    size = max(abs(projection), np.sum(np.abs(direction)))  # [state units]
    increment = SQRT_EPS * size / np.sum(direction**2)  # [state units]
    # [state units], largest |w_i| max(|x_i|, 1)
    weighted = np.max(np.abs(direction) * np.maximum(np.abs(x), 1.0))
    increment = max(increment, SQRT_EPS * weighted)
    return -increment if projection < 0 else increment


def jacv_bound(func, gradient_abs, curvature_abs, x, v):
    """Bound the error of ``numerical_jacv`` for an analytic function.

    Parameters
    ----------
    func : callable
        Function of the state [output units], shape (m,).
    gradient_abs : callable
        ``gradient_abs(x)``: entry-wise magnitude of the Jacobian, shape
        (m, n) [output units / state units].
    curvature_abs : callable
        ``curvature_abs(x, w)``: bound on ``|w' H_i w|``, shape (m,)
        [output units / state units**2].
    x, v : numpy.ndarray
        State and direction [state units], shape (n,).

    Returns
    -------
    numpy.ndarray
        Bound on ``|jac_v - J v|`` [output units], shape (m,): the forward
        difference bound of ``J w`` (truncation, rounding of both function
        values, of ``x + e w`` and of the difference, quotient and final
        product), times ``max|v|``. Magnitudes are maximized over the two
        evaluation points, which bounds them on the segment for the
        monotone derivatives used here.
    """
    scale = np.max(np.abs(v))  # [state units]
    direction = v / scale  # [-]
    increment = directional_increment(x, direction)  # [state units]
    step = abs(increment)
    end = x + increment * direction
    f_abs = np.maximum(np.abs(func(x)), np.abs(func(end)))
    gradient = np.maximum(gradient_abs(x), gradient_abs(end))
    curvature = np.maximum(curvature_abs(x, direction),
                           curvature_abs(end, direction))
    displacement_rounding = UNIT_ROUNDOFF * (np.abs(x)
                                             + step * np.abs(direction))
    bound_w = (step / 2 * curvature + 2 * UNIT_ROUNDOFF * f_abs / step
               + gradient @ displacement_rounding / step
               + 3 * UNIT_ROUNDOFF * gradient @ np.abs(direction))
    return scale * bound_w


def directional_outputs(x):
    """Evaluate an asymmetric test function of two state components.

    Parameters
    ----------
    x : numpy.ndarray
        State [-], shape (2,).

    Returns
    -------
    numpy.ndarray
        ``(x0 * x1, x1**2, sin(x0))`` [-], shape (3,).
    """
    return np.array([x[0] * x[1], x[1]**2, np.sin(x[0])])


def square_and_identity(x):
    """Evaluate a quadratic and a linear output of a two-component state.

    Parameters
    ----------
    x : numpy.ndarray
        State [-], shape (2,).

    Returns
    -------
    numpy.ndarray
        ``(x0**2, x1)`` [-], shape (2,).
    """
    return np.array([x[0]**2, x[1]])


def identity(x):
    """Return a copy of a two-component state.

    Parameters
    ----------
    x : numpy.ndarray
        State [-], shape (2,).

    Returns
    -------
    numpy.ndarray
        ``(x0, x1)`` [-], shape (2,).
    """
    return np.array([x[0], x[1]])


def difference(x):
    """Evaluate the difference of the two state components.

    Parameters
    ----------
    x : numpy.ndarray
        State [-], shape (2,).

    Returns
    -------
    numpy.ndarray
        ``(x0 - x1,)`` [-], shape (1,).
    """
    return np.array([x[0] - x[1]])


def identity3(x):
    """Return a copy of a three-component state.

    Parameters
    ----------
    x : numpy.ndarray
        State [-], shape (3,).

    Returns
    -------
    numpy.ndarray
        ``(x0, x1, x2)`` [-], shape (3,).
    """
    return np.array([x[0], x[1], x[2]])


def difference3(x):
    """Evaluate the difference of the first two of three components.

    Parameters
    ----------
    x : numpy.ndarray
        State [-], shape (3,).

    Returns
    -------
    numpy.ndarray
        ``(x0 - x1,)`` [-], shape (1,).
    """
    return np.array([x[0] - x[1]])


def difference_last_two(x):
    """Evaluate the difference of the last two of three components.

    Parameters
    ----------
    x : numpy.ndarray
        State [-], shape (3,).

    Returns
    -------
    numpy.ndarray
        ``(x1 - x2,)`` [-], shape (1,).
    """
    return np.array([x[1] - x[2]])


def square(state):
    """Square a state elementwise.

    Parameters
    ----------
    state : numpy.ndarray
        State [-], shape (n,).

    Returns
    -------
    numpy.ndarray
        ``state**2`` [-], shape (n,).
    """
    return state**2


def square_gradient_abs(state):
    """Return the magnitude of the Jacobian of ``square`` for one component.

    Parameters
    ----------
    state : numpy.ndarray
        State [-], shape (1,).

    Returns
    -------
    numpy.ndarray
        ``|2 x|`` [-], shape (1, 1).
    """
    return np.abs([[2 * state[0]]])


def square_curvature_abs(state, direction):
    """Return ``|w' H w|`` of ``square`` for one component.

    Parameters
    ----------
    state : numpy.ndarray
        State [-], shape (1,); unused, the curvature is constant.
    direction : numpy.ndarray
        Direction ``w`` [-], shape (1,).

    Returns
    -------
    numpy.ndarray
        ``2 w**2`` [-], shape (1,).
    """
    return np.array([2 * direction[0]**2])


def log_and_identity(x):
    """Evaluate a logarithmic and a linear output of a two-component state.

    Parameters
    ----------
    x : numpy.ndarray
        State [-], shape (2,).

    Returns
    -------
    numpy.ndarray
        ``(log(x0), x1)`` [-], shape (2,); requires ``x0 > 0``.
    """
    return np.array([np.log(x[0]), x[1]])


# Name -> (function, |Jacobian|(x) of shape (m, n), bound on |w' H_i w|
# (x, w) of shape (m,), exact J(x) v of shape (m,)), all in [-] units;
# the lambdas are the analytic derivatives of the named function.
JACV_FUNCTIONS = {
    'directional_outputs': (
        directional_outputs,
        lambda x: np.abs([[x[1], x[0]], [0.0, 2 * x[1]],
                          [np.cos(x[0]), 0.0]]),
        lambda x, w: np.array([2 * abs(w[0] * w[1]), 2 * w[1]**2,
                               w[0]**2]),
        lambda x, v: np.array([x[1] * v[0] + x[0] * v[1], 2 * x[1] * v[1],
                               np.cos(x[0]) * v[0]])),
    'square_and_identity': (
        square_and_identity,
        lambda x: np.abs([[2 * x[0], 0.0], [0.0, 1.0]]),
        lambda x, w: np.array([2 * w[0]**2, 0.0]),
        lambda x, v: np.array([2 * x[0] * v[0], v[1]])),
    'identity': (
        identity,
        lambda x: np.eye(2),
        lambda x, w: np.zeros(2),
        lambda x, v: v),
    'difference': (
        difference,
        lambda x: np.array([[1.0, 1.0]]),
        lambda x, w: np.zeros(1),
        lambda x, v: np.array([v[0] - v[1]])),
    'identity3': (
        identity3,
        lambda x: np.eye(3),
        lambda x, w: np.zeros(3),
        lambda x, v: v),
    'difference3': (
        difference3,
        lambda x: np.array([[1.0, 1.0, 0.0]]),
        lambda x, w: np.zeros(1),
        lambda x, v: np.array([v[0] - v[1]])),
    'difference_last_two': (
        difference_last_two,
        lambda x: np.array([[0.0, 1.0, 1.0]]),
        lambda x, w: np.zeros(1),
        lambda x, v: np.array([v[1] - v[2]])),
    'log_and_identity': (
        log_and_identity,
        lambda x: np.abs([[1 / x[0], 0.0], [0.0, 1.0]]),
        lambda x, w: np.array([w[0]**2 / x[0]**2, 0.0]),
        lambda x, v: np.array([v[0] / x[0], v[1]])),
}


@pytest.mark.parametrize('name, x, v', [
    ('directional_outputs', np.array([3, -2]), np.array([1, 2])),
    ('directional_outputs', np.array([0.0, 0.0]), np.array([1.0, -1.0])),
    ('directional_outputs', np.array([1e-12, 0.0]), np.array([1e-9, 2e-9])),
    ('directional_outputs', np.array([1e4, -2e4]), np.array([1e6, 0.5])),
    # Mixed-scale states: the step follows the small component along v.
    ('square_and_identity', np.array([1e-4, 1e10]), np.array([1.0, 0.0])),
    ('square_and_identity', np.array([1e-3, 1e6]), np.array([1e-3, 0.0])),
    # The increment moves away from zero, so log(x0) stays defined.
    ('log_and_identity', np.array([1e-6, 1e3]), np.array([-1.0, 0.0])),
    # x . v cancels: the increment must still move x by more than its
    # floating-point spacing (1.2e-7 at 1e9).
    ('identity', np.array([1e9, 1e9]), np.array([1.0, -1.0])),
    ('difference', np.array([1e9, 1e9]), np.array([1.0, -1.0])),
    # The large components are touched with |w_i| < 1: the increment must
    # still move them (the largest touched component sets the floor).
    ('identity3', np.array([1e9, 1e9, 0.0]), np.array([0.5, -0.5, 1.0])),
    ('difference3', np.array([1e9, 1e9, 0.0]), np.array([0.5, -0.5, 1.0])),
    ('identity3', np.array([0.0, 1e9, 1e9]), np.array([1.0, 0.5, -0.5])),
    ('difference_last_two', np.array([0.0, 1e9, 1e9]),
     np.array([1.0, 0.5, -0.5])),
])
def test_numerical_jacv_matches_analytic_product(name, x, v):
    func, gradient_abs, curvature_abs, exact = JACV_FUNCTIONS[name]

    jac_v = numerical_jacv(func, x, v)

    x = x.astype(float)
    v = v.astype(float)
    bound = jacv_bound(func, gradient_abs, curvature_abs, x, v)
    assert np.all(np.abs(jac_v - exact(x, v)) <= bound)


def jacv_bound_for_steps(func, gradient_abs, curvature_abs, x, v,
                         step_min, step_max):
    """Bound the ``numerical_jacv`` error for an admissible step range.

    The bound does not use the production increment: it holds for any
    increment ``e`` with ``step_min <= |e| <= step_max`` along
    ``w = v / max|v|``, in either direction.

    Parameters
    ----------
    func : callable
        Function of the state [output units], shape (m,).
    gradient_abs : callable
        ``gradient_abs(x)``: entry-wise magnitude of the Jacobian, shape
        (m, n) [output units / state units].
    curvature_abs : callable
        ``curvature_abs(x, w)``: bound on ``|w' H_i w|``, shape (m,)
        [output units / state units**2].
    x, v : numpy.ndarray
        State and direction [state units], shape (n,).
    step_min, step_max : float
        Admissible increment magnitudes [state units].

    Returns
    -------
    numpy.ndarray
        Bound on ``|jac_v - J v|`` [output units], shape (m,): truncation
        at ``step_max``, rounding of both function values and of
        ``x + e w`` at ``step_min``, and of the difference, quotient and
        final product, times ``max|v|``. Magnitudes are maximized over
        ``x`` and ``x +/- step_max w``, which bounds them on the segment
        for the monotone derivatives used here.
    """
    scale = np.max(np.abs(v))  # [state units]
    direction = v / scale  # [-]
    points = [x, x + step_max * direction, x - step_max * direction]
    f_abs = np.max([np.abs(func(point)) for point in points], axis=0)
    gradient = np.max([gradient_abs(point) for point in points], axis=0)
    curvature = np.max([curvature_abs(point, direction) for point in points],
                       axis=0)
    displacement_rounding = UNIT_ROUNDOFF * (np.abs(x)
                                             + step_max * np.abs(direction))
    bound_w = (step_max / 2 * curvature + 2 * UNIT_ROUNDOFF * f_abs / step_min
               + gradient @ displacement_rounding / step_min
               + 3 * UNIT_ROUNDOFF * gradient @ np.abs(direction))
    return scale * bound_w


def sqrt_and_scaled(x):
    """Evaluate a dense scalar output ``sqrt(x0) + 1e-12 x1``.

    Parameters
    ----------
    x : numpy.ndarray
        State [-], shape (2,), with ``x0 > 0``.

    Returns
    -------
    numpy.ndarray
        ``(sqrt(x0) + 1e-12 x1,)`` [-], shape (1,).
    """
    return np.array([np.sqrt(x[0]) + 1e-12 * x[1]])


def sqrt_and_scaled_gradient_abs(x):
    """Return the magnitude of the gradient of ``sqrt_and_scaled``.

    Parameters
    ----------
    x : numpy.ndarray
        State [-], shape (2,), with ``x0 > 0``.

    Returns
    -------
    numpy.ndarray
        ``(0.5 / sqrt(x0), 1e-12)`` [-], shape (1, 2).
    """
    return np.array([[0.5 / np.sqrt(x[0]), 1e-12]])


def sqrt_and_scaled_curvature_abs(x, direction):
    """Return ``|w' H w|`` of ``sqrt_and_scaled``.

    Parameters
    ----------
    x : numpy.ndarray
        State [-], shape (2,), with ``x0 > 0``.
    direction : numpy.ndarray
        Direction ``w`` [-], shape (2,).

    Returns
    -------
    numpy.ndarray
        ``w0**2 / (4 x0**1.5)`` [-], shape (1,); the output is linear in
        ``x1``.
    """
    return np.array([direction[0]**2 / (4 * x[0]**1.5)])


# |x_1| = 1e12 is moved by |w_1| = 1e-300, 1e-20 or 1e-12, i.e. by at most
# 1 relative to |x_0| = 1, which the unit component w_0 moves: a local
# increment is a few eps * max(|x_0|, 1) = eps. A projection of at most 2
# allows |e| <= 2 eps; 4 eps is stated as the admissible maximum, and eps
# (the relative step of the unit component) as the minimum.
@pytest.mark.parametrize('v', [np.array([1.0, 1e-300]),
                               np.array([1.0, 1e-20]),
                               np.array([1.0, 1e-12])])
def test_numerical_jacv_ignores_barely_moved_large_components(v):
    x = np.array([1.0, 1e12])  # [-]

    jac_v = numerical_jacv(square_and_identity, x, v)

    func, gradient_abs, curvature_abs, exact = JACV_FUNCTIONS[
        'square_and_identity']
    bound = jacv_bound_for_steps(func, gradient_abs, curvature_abs, x, v,
                                 step_min=SQRT_EPS, step_max=4 * SQRT_EPS)
    expected = exact(x, v)  # (2, v_1)
    assert np.all(np.abs(jac_v - expected) <= bound)


def test_numerical_jacv_dense_direction_with_a_large_component():
    """``sqrt(x0) + 1e-12 x1`` at (0.5, 1e12) along (1, 1e-6).

    Both components are moved comparably relative to their size:
    ``|w_1| |x_1| = 1e6`` dominates, so a local increment is about
    ``eps * 1e6 = 1.5e-2`` (a relative eps on x1); at most 2e-2 is
    admissible, and at least ``eps`` (the unit component's own step).
    The exact product is ``0.5 / sqrt(0.5) + 1e-18``.
    """
    x = np.array([0.5, 1e12])  # [-]
    v = np.array([1.0, 1e-6])  # [-]

    jac_v = numerical_jacv(sqrt_and_scaled, x, v)

    expected = 0.5 / np.sqrt(0.5) + 1e-12 * 1e-6
    bound = jacv_bound_for_steps(
        sqrt_and_scaled, sqrt_and_scaled_gradient_abs,
        sqrt_and_scaled_curvature_abs, x, v, step_min=SQRT_EPS,
        step_max=2e-2)
    assert np.all(np.abs(jac_v - expected) <= bound)


@pytest.mark.parametrize('v', [np.array([1.0]), np.array([1e-3]),
                               np.array([1e-200])])
def test_numerical_jacv_is_local(v):
    """``d(x**2)/dx`` at 1 times v, for any length of v, even 1e-200."""
    x = np.array([1.0])

    jac_v = numerical_jacv(square, x, v)

    bound = jacv_bound(square, square_gradient_abs, square_curvature_abs, x,
                       v)
    assert np.all(np.abs(jac_v - 2 * v) <= bound)


def test_numerical_jacv_of_0d_inputs_is_a_scalar():
    """``d(z**2)/dz`` at 3 times 2, with 0-d state and direction."""
    jac_v = numerical_jacv(square, np.array(3.0), np.array(2.0))

    bound = jacv_bound(square, square_gradient_abs, square_curvature_abs,
                       np.array([3.0]), np.array([2.0]))
    assert np.ndim(jac_v) == 0
    assert abs(jac_v - 12.0) <= bound[0]


def test_numerical_jacv_of_a_zero_direction_and_scalar_output():
    jac_v = numerical_jacv(squared_norm,
                           np.array([1.0, 2.0]), np.zeros(2))

    assert not isinstance(jac_v, np.ndarray)
    assert jac_v == 0.0


@pytest.mark.parametrize('x', [np.zeros(3), np.array([-0.3, 0.1, 0.4])])
def test_numerical_jacv_uses_the_typical_size_floor(x):
    """``|x . w| < ||w||_1``: the increment is ``eps ||w||_1 / ||w||**2``.

    ``w = v = (1, 1, -0.5)``: ``||w||_1 = 2.5``, ``||w||**2 = 2.25``, which
    exceeds the representability floor ``eps * max(|x_j|, 1) = eps``. The
    sign follows ``x . w``: 0 (positive) and -0.4.
    """
    direction = np.array([1.0, 1.0, -0.5])  # [-]
    points = []

    def recording(state):
        """Store an evaluation point and return it.

        Parameters
        ----------
        state : numpy.ndarray
            Evaluation point [-], shape (3,).

        Returns
        -------
        numpy.ndarray
            ``state`` [-], shape (3,).
        """
        points.append(state.copy())
        return state

    numerical_jacv(recording, x, direction)

    sign = -1.0 if np.sum(x * direction) < 0 else 1.0
    increment = sign * SQRT_EPS * 2.5 / 2.25  # [state units]
    assert len(points) == 2
    np.testing.assert_array_equal(points[0], x)
    # x + e w is one rounded operation per component: within 2 ulp.
    np.testing.assert_allclose(points[1], x + increment * direction,
                               rtol=2 * np.finfo(float).eps, atol=0)


def test_numerical_jacv_of_a_zero_direction_is_zero():
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        jac_v = numerical_jacv(directional_outputs, np.array([1.0, 2.0]),
                               np.zeros(2))

    assert jac_v.shape == (3,)
    assert np.all(jac_v == 0.0)


@pytest.mark.parametrize('jac_fun', [numerical_jac, numerical_jac_data])
def test_forward_differences_with_no_picked_coordinates(jac_fun):
    jacobian = jac_fun(three_outputs, np.array([1.0, 2.0]), pick_x=[])

    assert jacobian.shape == (3, 0)


def test_default_rule_signed_step_through_numerical_jac():
    """Gradient of ``log(-p0) * p1`` at ``p0 = -5e-7``, below the floor.

    The default rule perturbs p0 by -1e-9 (away from zero) and p1 by
    1e-3 * p1; the analytic gradient is ``(p1 / p0, log(-p0))``.
    """
    def log_product(p):
        """Evaluate ``log(-p0) * p1``.

        Parameters
        ----------
        p : numpy.ndarray
            ``(p0, p1)`` [-], with ``p0 < 0``.

        Returns
        -------
        float
            Function value [-].
        """
        return np.log(-p[0]) * p[1]

    params = np.array([-5e-7, 1.0])  # [-]

    gradient = numerical_jac(log_product, params, dx=default_rule)

    steps = contract_step(params)  # [-], (-1e-9, 1e-3)
    expected = np.array([params[1] / params[0], np.log(-params[0])])
    f_abs = max(abs(log_product(params)),
                abs(log_product(params + [steps[0], 0.0])),
                abs(log_product(params + [0.0, steps[1]])))
    # |d2f/dp0**2| = p1 / p0**2 is largest at p0 (the step moves away from
    # zero); f is linear in p1.
    second = np.array([params[1] / params[0]**2, 0.0])
    bound = forward_difference_bound(steps, np.abs(params), f_abs, second,
                                     np.abs(expected))
    assert steps[0] < 0
    assert np.all(np.abs(gradient[0] - expected) <= bound)


def two_scale_outputs(x):
    """Evaluate ``(x0**2, x1)`` for the default-step tests.

    Parameters
    ----------
    x : numpy.ndarray
        Coordinates [-], shape (2,).

    Returns
    -------
    numpy.ndarray
        ``(x0**2, x1)`` [-], shape (2,).
    """
    return np.array([x[0]**2, x[1]])


@pytest.mark.parametrize('jac_fun', [numerical_jac, numerical_jac_data,
                                     numerical_jac_central])
@pytest.mark.parametrize('point', [np.array([1e9, 2e8]),
                                   np.array([10**9, 2 * 10**8])])
def test_default_step_scales_with_large_coordinates(jac_fun, point):
    """Default ``dx``: ``eps * max(|x|, 1)``, not a step lost in ``x + h``.

    A fixed 1.49e-8 step is below the spacing of 1e9 (1.2e-7), which gave
    an all-zero Jacobian. The forward-difference bound also bounds the
    central difference, which is exact for these outputs up to rounding.
    """
    jacobian = jac_fun(two_scale_outputs, point)

    x = point.astype(float)
    steps = SQRT_EPS * np.maximum(np.abs(x), 1.0)  # [-]
    expected = np.array([[2 * x[0], 0.0], [0.0, 1.0]])
    f_abs = np.array([[(x[0] + steps[0])**2], [x[1] + steps[1]]])
    second = np.array([[2.0, 0.0], [0.0, 0.0]])
    bound = forward_difference_bound(steps, np.abs(x), f_abs, second,
                                     np.abs(expected))
    assert np.all(np.abs(jacobian - expected) <= bound)


def recorded_points(jac_fun, x):
    """Record the points at which a Jacobian helper evaluates a function.

    Parameters
    ----------
    jac_fun : callable
        ``numerical_jac``, ``numerical_jac_data`` or
        ``numerical_jac_central``, called with its default ``dx``.
    x : numpy.ndarray
        Coordinates [-], shape (n,).

    Returns
    -------
    list of numpy.ndarray
        Evaluation points in call order [-], each shape (n,).
    """
    points = []

    def recording(state):
        """Store an evaluation point and return it.

        Parameters
        ----------
        state : numpy.ndarray
            Evaluation point [-], shape (n,).

        Returns
        -------
        numpy.ndarray
            A copy of ``state`` [-], shape (n,).
        """
        points.append(state.copy())
        return state.copy()

    jac_fun(recording, x)
    return points


# Earlier versions used the absolute step eps (about 1.49e-8) for every
# coordinate; for |x| <= 1 the default must still be exactly eps.
SMALL_COORDINATES = np.array([0.0, 0.5, -0.5])  # [-]


@pytest.mark.parametrize('jac_fun', [numerical_jac, numerical_jac_data])
def test_default_forward_step_is_the_old_step_for_small_coordinates(
        jac_fun):
    points = recorded_points(jac_fun, SMALL_COORDINATES)

    # f(x), then x + h_k e_k for each coordinate k.
    assert len(points) == 1 + len(SMALL_COORDINATES)
    for index, point in enumerate(points[1:]):
        moved = SMALL_COORDINATES.copy()
        moved[index] = SMALL_COORDINATES[index] + SQRT_EPS
        assert point[index] != SMALL_COORDINATES[index]
        np.testing.assert_allclose(point, moved,
                                   rtol=2 * np.finfo(float).eps, atol=0)


def test_default_central_step_is_the_old_step_for_small_coordinates():
    points = recorded_points(numerical_jac_central, SMALL_COORDINATES)

    # x + h_k e_k and x - h_k e_k for each coordinate k.
    assert len(points) == 2 * len(SMALL_COORDINATES)
    for index in range(len(SMALL_COORDINATES)):
        for point, sign in zip(points[2 * index:2 * index + 2], (1, -1)):
            moved = SMALL_COORDINATES.copy()
            moved[index] = SMALL_COORDINATES[index] + sign * SQRT_EPS
            assert point[index] != SMALL_COORDINATES[index]
            np.testing.assert_allclose(point, moved,
                                       rtol=2 * np.finfo(float).eps, atol=0)


@pytest.mark.parametrize('jac_fun', [numerical_jac, numerical_jac_data,
                                     numerical_jac_central])
def test_default_step_of_a_large_integer_identity(jac_fun):
    jacobian = jac_fun(unchanged, np.array([10**9]))

    step = SQRT_EPS * 1e9  # [-]
    np.testing.assert_allclose(jacobian, [[1.0]], rtol=0,
                               atol=forward_difference_bound(
                                   step, 1e9, 1e9 + step, 0.0, 1.0))


@pytest.mark.parametrize('jac_fun', [numerical_jac, numerical_jac_data,
                                     numerical_jac_central])
def test_float32_coordinates_are_differenced_in_float64(jac_fun):
    """A 1.49e-8 step is lost in float32 (spacing 1.2e-7 at 1)."""
    jacobian = jac_fun(unchanged, np.array([1.0], dtype=np.float32))

    np.testing.assert_allclose(jacobian, [[1.0]], rtol=0,
                               atol=forward_difference_bound(
                                   SQRT_EPS, 1.0, 1.0 + SQRT_EPS, 0.0,
                                   1.0))


def test_numerical_jacv_of_float32_inputs():
    x = np.array([1.0], dtype=np.float32)
    v = np.array([1.0], dtype=np.float32)

    jac_v = numerical_jacv(unchanged, x, v)

    increment = SQRT_EPS  # [-], eps * max(|x . w|, ||w||_1) / ||w||**2
    np.testing.assert_allclose(jac_v, [1.0], rtol=0,
                               atol=forward_difference_bound(
                                   increment, 1.0, 1.0 + increment, 0.0,
                                   1.0))


def test_numerical_jacv_sign_keeps_a_small_component_positive():
    """``log(x0)`` at ``x0 = 1e-9`` along ``-e0``: x0 must not cross zero.

    ``x . w = -1e-9 < 0``, so the increment is ``-eps`` and the point moves
    to ``x0 + eps``; the result is the secant along that realized point.
    """
    x = np.array([1e-9, 1e3])  # [-]
    v = np.array([-1.0, 0.0])  # [-]

    jac_v = numerical_jacv(log_and_identity, x, v)

    increment = -SQRT_EPS  # [-]
    moved = x[0] + SQRT_EPS  # [-], realized x0 + e w0
    secant = np.log1p((moved - x[0]) / x[0]) / increment  # [-]
    # Rounding only: two log values over |e|, and a few ulp of the secant.
    tolerance = (2 * UNIT_ROUNDOFF * abs(np.log(moved)) / SQRT_EPS
                 + 4 * UNIT_ROUNDOFF * abs(secant))
    assert np.all(np.isfinite(jac_v))
    assert abs(jac_v[0] - secant) <= tolerance
    assert jac_v[1] == 0.0


@pytest.mark.parametrize('v', [np.array([np.nan, 1.0]),
                               np.array([np.inf, 0.0]),
                               np.array([1.0, -np.inf])])
def test_numerical_jacv_of_a_non_finite_direction_is_nan(v):
    """NaN by a deliberate branch: no invalid-value arithmetic warnings."""
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        jac_v = numerical_jacv(directional_outputs, np.array([1.0, 2.0]), v)

    assert jac_v.shape == (3,)
    assert np.all(np.isnan(jac_v))


def test_numerical_jacv_of_a_non_finite_direction_and_scalar_output():
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        jac_v = numerical_jacv(squared_norm, np.array([1.0, 2.0]),
                               np.array([np.inf, 1.0]))

    assert not isinstance(jac_v, np.ndarray)
    assert np.isnan(jac_v)


# Synthetic first-order decay y = amplitude * exp(-rate * t) on 9 points.
DECAY_TIMES = np.linspace(0.0, 4.0, 9)  # [s]
DECAY_TRUE = np.array([0.4, 1.0])  # [1/s, mol/L], (rate, amplitude)
DECAY_SEED = np.array([0.0, 0.8])  # [1/s, mol/L], zero rate seed


def decay(params, time_s):
    """Return a first-order decay profile.

    Parameters
    ----------
    params : numpy.ndarray
        ``(rate [1/s], amplitude [mol/L])``.
    time_s : numpy.ndarray
        Times [s].

    Returns
    -------
    numpy.ndarray
        Concentration ``amplitude * exp(-rate * t)`` [mol/L].
    """
    return params[1] * np.exp(-params[0] * time_s)


def gradient_stop_bound(jacobian):
    """Bound the parameter error left by the LM gradient stop.

    Parameters
    ----------
    jacobian : numpy.ndarray
        Analytic residual Jacobian at the optimum, shape (n_data, n_params)
        [residual units / parameter units].

    Returns
    -------
    float
        ``||inv(J' J)||_2 * 2 eps_1`` [parameter units]. Near the optimum
        the parameter error is ``inv(J' J)`` times the gradient, which LM
        stops below ``eps_1``; the second ``eps_1`` bounds the difference
        between the finite-difference and the exact gradient,
        ``||dJ|| ||r||``, which is below 1e-10 for the fixtures here.
    """
    return (np.linalg.norm(np.linalg.inv(jacobian.T @ jacobian), 2)
            * 2 * LM_GRADIENT_TOL)


def test_optimize_fn_from_a_zero_rate_seed_reaches_the_optimum():
    """Noise-free decay data: the least-squares optimum is the truth."""
    observed = decay(DECAY_TRUE, DECAY_TIMES)  # [mol/L]
    estimator = ParameterEstimation(decay, DECAY_SEED, DECAY_TIMES, observed)

    params, covariance, info = estimator.optimize_fn(verbose=False)

    rate, amplitude = DECAY_TRUE
    profile = np.exp(-rate * DECAY_TIMES)  # [-]
    jacobian = np.column_stack((-amplitude * DECAY_TIMES * profile, profile))
    np.testing.assert_allclose(params, DECAY_TRUE, rtol=0,
                               atol=gradient_stop_bound(jacobian))
    assert info['stop_criterion'] == 'Small gradient'
    assert np.all(np.isfinite(covariance))


def test_optimize_fn_from_a_zero_intercept_matches_linear_least_squares():
    """Line ``a + b t`` with fixed perturbations, seeded at ``a = 0``."""
    time_s = DECAY_TIMES  # [s]
    # [mol/L], deterministic asymmetric scatter about 0.5 t
    observed = 0.5 * time_s + 0.01 * np.sin(7 * time_s)
    estimator = ParameterEstimation(
        straight_line, [0.0, 0.3], time_s,
        observed)

    params, covariance, info = estimator.optimize_fn(verbose=False)

    design = np.column_stack((np.ones_like(time_s), time_s))  # [-, s]
    expected, residual_ss, *_ = np.linalg.lstsq(design, observed, rcond=None)
    residual_variance = residual_ss[0] / (len(time_s) - 2)  # [(mol/L)**2]
    expected_cov = residual_variance * np.linalg.inv(design.T @ design)
    np.testing.assert_allclose(params, expected, rtol=0,
                               atol=gradient_stop_bound(design))
    assert info['stop_criterion'] == 'Small gradient'
    # The intercept column is differenced with h ~ 4.6e-7 at a ~ -4.6e-4,
    # so its rounding error is ~ 2 u * 2 / h ~ 1e-9 relative; 1e-7 bounds
    # the resulting covariance error with margin.
    np.testing.assert_allclose(covariance, expected_cov, rtol=1e-7)


def test_optimize_fn_rejects_a_zero_user_step():
    estimator = ParameterEstimation(decay, DECAY_SEED, DECAY_TIMES,
                                    decay(DECAY_TRUE, DECAY_TIMES),
                                    dx_finitediff=0.0)

    with pytest.raises(ValueError, match='nonzero and finite'):
        estimator.optimize_fn(verbose=False)


# Spectra of A -> B at two wavelengths, noise-free.
MCR_TIMES = np.array([0.5, 1.0, 2.0, 3.5, 5.0])  # [s]
MCR_TRUE_RATE = 0.4  # [1/s]
ABSORPTIVITY = np.array([[1.2, 0.4], [0.3, 0.9]])  # [L/mol], rows A, B


def first_order_species(params, time_s, reord_sens=False):
    """Return A and B concentrations of A -> B, unit initial A.

    Parameters
    ----------
    params : numpy.ndarray
        Rate k, shape (1,) [1/s].
    time_s : numpy.ndarray
        Times [s].
    reord_sens : bool, optional
        Keyword passed by ``MultipleCurveResolution``; unused.

    Returns
    -------
    numpy.ndarray
        Shape (n_times, 2) [mol/L].
    """
    reactant = np.exp(-params[0] * time_s)  # [mol/L]
    return np.column_stack((reactant, 1.0 - reactant))


def projected_residuals(rate):
    """Return spectra minus their projection on the model profiles.

    Parameters
    ----------
    rate : float
        Rate k [1/s].

    Returns
    -------
    numpy.ndarray
        Flattened residual absorbance [-].
    """
    spectra = first_order_species([MCR_TRUE_RATE], MCR_TIMES) @ ABSORPTIVITY
    profiles = first_order_species([rate], MCR_TIMES)
    fitted, *_ = np.linalg.lstsq(profiles, spectra, rcond=None)
    return (spectra - profiles @ fitted).ravel()


@pytest.mark.parametrize('seed', [[1], [1.0]])
def test_mcr_finite_difference_fit_accepts_an_integer_seed(seed):
    spectra = first_order_species([MCR_TRUE_RATE], MCR_TIMES) @ ABSORPTIVITY
    estimator = MultipleCurveResolution(first_order_species, seed, MCR_TIMES,
                                        spectra, measured_ind=[0, 1])

    params, covariance, info = estimator.optimize_fn(verbose=False)

    # Residual Jacobian at the truth by central differences of the
    # projection itself: relative step 1e-5 leaves O(1e-10) error.
    step = 1e-5 * MCR_TRUE_RATE  # [1/s]
    jacobian = ((projected_residuals(MCR_TRUE_RATE + step)
                 - projected_residuals(MCR_TRUE_RATE - step))
                / (2 * step))[:, np.newaxis]
    np.testing.assert_allclose(params, [MCR_TRUE_RATE], rtol=0,
                               atol=gradient_stop_bound(jacobian))
    assert info['stop_criterion'] == 'Small gradient'
