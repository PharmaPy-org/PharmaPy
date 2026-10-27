
# -*- coding: utf-8 -*-
"""Finite-difference derivatives of model functions.

Forward and central difference Jacobians, a Jacobian-vector product, and the
perturbation-size rules used by parameter estimation and sensitivity
equations. All differences are evaluated on floating-point copies of the
differentiated coordinates, so an integer input is never truncated.
"""

import numpy as np

#: Square root of the float64 machine epsilon [-], about 1.49e-8. It is the
#: forward-difference relative step that balances truncation and rounding
#: error for a function evaluated to machine precision (Nocedal and Wright,
#: Numerical Optimization, 2nd ed., Springer, 2006, Sec. 8.1).
eps = np.sqrt(np.finfo(float).eps)


def dx_jac_x(x, abstol, reltol, eps):
    sigma_zero = 1
    crit_1 = np.abs(x) * np.sqrt(eps)

    weights = 1 / (reltol * x + abstol)
    crit_2 = sigma_zero / weights

    dx = np.maximum(crit_1, crit_2)

    # print(dx)
    return dx


def _float_work_array(x):
    """Return differentiated coordinates as a floating-point array.

    Parameters
    ----------
    x : array-like
        Differentiated coordinates [coordinate-dependent units], shape (n,).

    Returns
    -------
    numpy.ndarray
        ``x`` with a dtype of at least float64 precision [same units], shape
        (n,). Integer and lower-precision inputs are promoted, so adding a
        finite-difference step to a copy of ``x`` never truncates it; a
        float64 or complex input is returned without a copy.
    """
    x = np.asarray(x)
    return x.astype(np.result_type(x.dtype, np.float64), copy=False)


def _default_steps(x):
    """Return the default forward-difference step of each coordinate.

    Parameters
    ----------
    x : numpy.ndarray
        Floating-point coordinates [coordinate units], shape (n,).

    Returns
    -------
    numpy.ndarray
        ``eps * max(|x|, 1)`` [coordinate units], shape (n,): the relative
        step ``eps`` (square root of the machine epsilon, see the module
        constant) for ``|x| > 1``, and the absolute step ``eps`` for
        ``|x| <= 1``, as earlier versions used for every coordinate. The
        typical size of one [coordinate units] is an assumption, as the
        callers carry no coordinate scales (Dennis and Schnabel, Sec. 5.4,
        Algorithm A5.4.1, use ``max(|x_j|, typx_j)``). A fixed absolute
        step below the floating-point spacing of a large coordinate would
        leave ``x + dx == x`` and a zero derivative.
    """
    typical_size = 1.0  # [coordinate units], typx assumption
    return eps * np.maximum(np.abs(x), typical_size)  # [coordinate units]


def _check_steps(dx, pick_x):
    """Reject zero or infinite forward-difference steps.

    Parameters
    ----------
    dx : numpy.ndarray
        Difference step of every coordinate [coordinate units], shape (n,).
    pick_x : numpy.ndarray
        Indices of the differentiated coordinates, shape (m,).

    Raises
    ------
    ValueError
        If the step of a differentiated coordinate is zero or infinite,
        which turns its Jacobian column into NaN, infinity or zero.

    Notes
    -----
    A NaN step is not rejected: a relative step rule returns NaN for a NaN
    coordinate, whose function values are NaN whatever the step, so the
    NaN Jacobian column reports the invalid point as before.
    """
    steps = dx[pick_x]  # [coordinate units]
    invalid = (steps == 0) | np.isinf(steps)
    if np.any(invalid):
        raise ValueError(
            "finite-difference steps must be nonzero and finite; got "
            f"{steps[invalid]!r} for coordinates {pick_x[invalid]!r}")


def dx_jac_p(p, abstol, reltol, eps):
    """Forward-difference perturbation of model parameters.

    Parameters
    ----------
    p : array-like
        Parameter values [parameter units], shape (n,).
    abstol : float, array-like or None
        Positive lower bound on the parameter magnitude that scales the step
        [parameter units], scalar or broadcastable to ``p``. Parameters
        with ``|p| < abstol``, including zero, are perturbed as if their
        magnitude were ``abstol``, away from zero. None, which
        ``numerical_jac_central`` passes to a callable ``dx`` by default,
        applies no floor: the step is ``|p| * sqrt(max(reltol, eps))``,
        zero for a zero parameter, as in earlier versions.
    reltol : float
        Relative accuracy of the model outputs being differentiated [-],
        e.g. the relative tolerance of the integrator that computes them.
    eps : float
        Lower bound on that relative accuracy [-].

    Returns
    -------
    numpy.ndarray
        Floating-point step of each parameter [parameter units], shape (n,):
        ``|p| * sqrt(max(reltol, eps))`` where ``|p| >= abstol``, and
        ``sign(p) * abstol * sqrt(max(reltol, eps))`` elsewhere, with
        ``sign(0) = +1``.

    Notes
    -----
    The relative factor ``sqrt(max(reltol, eps))`` is the square root of the
    relative noise of the differentiated function. Balancing the
    truncation error, proportional to the step, against the noise error,
    inversely proportional to it, gives this relative step (Nocedal and
    Wright, Numerical Optimization, 2nd ed., Sec. 8.1, for rounding noise;
    Dennis and Schnabel, Sec. 5.4, for a function with fewer accurate
    digits). It is the factor of the CVODES difference-quotient parameter
    increment used by ``Crystallizers.jac_params_numerical``,
    ``|pbar| * sqrt(max(rtol, eps))``, where ``pbar`` is a typical
    parameter magnitude.

    The magnitude ``max(|p|, abstol)`` follows Dennis and Schnabel,
    Numerical Methods for Unconstrained Optimization and Nonlinear
    Equations (Prentice-Hall, 1983; SIAM reprint 1996), Sec. 5.4 and
    Algorithm A5.4.1, whose step has magnitude
    ``sqrt(eta) * max(|x_j|, typx_j)`` with ``typx_j`` a typical magnitude
    and the sign of ``x_j``. Scaling by ``|p|`` alone gives a zero step,
    and a NaN derivative, at a zero parameter. PharmaPy applies the sign
    only below the floor, where a positive floor step could reach or cross
    zero from a small negative parameter, e.g. in
    ``log(-p)``; for ``|p| >= abstol`` the step stays exactly
    ``|p| * sqrt(max(reltol, eps))`` (positive), as in earlier versions,
    and cannot reach zero.
    """
    factor = np.sqrt(max(reltol, eps))  # [-]
    p_abs = np.abs(p)  # [parameter units]
    if abstol is None:
        return p_abs * factor  # [parameter units]

    # [-], +1 at zero and for positive parameters
    side = np.where(np.asarray(p) < 0, -1.0, 1.0)
    # [parameter units], signed only below the floor
    magnitude = np.where(p_abs < abstol, side * abstol, p_abs)
    dx = magnitude * factor  # [parameter units]
    return dx


def numerical_jac(func, x, args=(), dx=None, abs_tol=None, rel_tol=None,
                  pick_x=None):
    """Approximate a Jacobian with forward differences.

    Parameters
    ----------
    func : callable
        Function ``func(x, *args)`` returning a scalar or one-dimensional
        output [output units].
    x : array-like
        Point of differentiation [coordinate-dependent units], shape (n,).
        Integer values are promoted to float before ``func``, ``dx`` and
        the perturbations see them.
    args : tuple, optional
        Additional function arguments [argument-dependent units].
    dx : float, array-like, or callable, optional
        Difference step [coordinate units], scalar or shape (n,), or a
        callable ``dx(x)`` returning shape (n,). The default is None, which
        uses ``eps * max(|x|, 1)`` (``eps`` about 1.49e-8; see
        ``_default_steps``).
    abs_tol, rel_tol : optional
        Unused; kept for backward compatibility of the signature.
    pick_x : int or array-like of int, optional
        Indices of the coordinates to differentiate. The default is None,
        which differentiates all of them in order.

    Returns
    -------
    numpy.ndarray
        Jacobian [output units / coordinate units], shape
        (n_outputs, len(pick_x)); column ``k`` is the derivative with
        respect to ``x[pick_x[k]]``.

    Raises
    ------
    ValueError
        If the step of a differentiated coordinate is zero or infinite.
    """
    x = _float_work_array(x)  # [coordinate units]

    if dx is None:
        dx = _default_steps(x)
    elif callable(dx):
        dx = dx(x)
    else:
        dx = np.ones_like(x) * dx

    if pick_x is None:
        pick_x = np.arange(len(x))
    else:
        pick_x = np.atleast_1d(pick_x)
        if pick_x.size == 0:  # e.g. [], whose default dtype is float
            pick_x = pick_x.astype(np.intp)

    _check_steps(np.asarray(dx), pick_x)

    f_eval = func(x, *args)
    f_eval = np.atleast_1d(f_eval)

    num_x = len(pick_x)
    num_f = len(f_eval)

    jac = np.zeros((num_f, num_x))
    delx = np.zeros_like(x)

    for idx, i in enumerate(pick_x):
        delx[i] = dx[i]
        jac[:, idx] = (func(x + delx, *args) - f_eval)/dx[i]
        delx[i] = 0

    return jac


def numerical_jac_central(func, x, rel_tol=None, abs_tol=None, dx=None,
                          args=(), num_nonnegative=0):
    """Approximate a vector-valued Jacobian with second-order differences.

    Parameters
    ----------
    func : callable
        Function mapping ``x`` to a one-dimensional output vector.
    x : array-like
        Differentiated coordinates [coordinate-dependent units]. Integer
        values are promoted to float before differencing.
    rel_tol : float, optional
        Relative tolerance passed to a callable ``dx`` rule [-].
    abs_tol : float or array-like, optional
        Absolute tolerance passed to a callable ``dx`` rule [coordinate
        units].
    dx : float, array-like, or callable, optional
        Difference step [coordinate units], or a step-selection callable
        ``dx(x, abs_tol, rel_tol, eps)``. The default is None, which uses
        ``eps * max(|x|, 1)`` (see ``_default_steps``).
    args : tuple, optional
        Additional function arguments [argument-dependent units].
    num_nonnegative : int, optional
        Number of leading coordinates constrained to be nonnegative [-].

    Returns
    -------
    numpy.ndarray
        Jacobian [output units / coordinate units].

    Notes
    -----
    Interior coordinates use centered differences. A leading nonnegative
    coordinate uses the second-order forward stencil when a centered step
    would cross zero.
    """

    x = _float_work_array(x)  # [coordinate units]

    if dx is None:
        dx = _default_steps(x)
    elif callable(dx):
        dx = dx(x, abs_tol, rel_tol, eps)
    else:
        dx = dx * np.ones_like(x)

    num_x = len(x)
    jac = []
    delx = np.zeros_like(x)
    f_zero = None

    for j in range(num_x):
        delx[j] = dx[j]
        if j < num_nonnegative and x[j] < dx[j]:
            if f_zero is None:
                f_zero = np.atleast_1d(func(x, *args))
            jac.append((
                -3 * f_zero
                + 4 * np.atleast_1d(func(x + delx, *args))
                - np.atleast_1d(func(x + 2 * delx, *args))
            ) / 2. / dx[j])
        else:
            jac.append((
                np.atleast_1d(func(x + delx, *args))
                - np.atleast_1d(func(x - delx, *args))
            ) / 2. / dx[j])
        delx[j] = 0

    return np.column_stack(jac)


def numerical_jac_data(func, x, args=(), dx=None, pick_x=None):
    """Approximate a model-output Jacobian by forward differences.

    Parameters
    ----------
    func : callable
        Function ``func(x, *args)`` returning a one-dimensional output
        [output units], e.g. stacked model states at the data points.
    x : array-like
        Point of differentiation [coordinate-dependent units], shape (n,),
        e.g. the full parameter vector. Integer values are promoted to
        float before ``func``, ``dx`` and the perturbations see them.
    args : tuple, optional
        Additional function arguments [argument-dependent units].
    dx : float, array-like, or callable, optional
        Difference step [coordinate units], scalar or shape (n,), or a
        callable ``dx(x)`` returning shape (n,). The default is None, which
        uses ``eps * max(|x|, 1)`` (``eps`` about 1.49e-8; see
        ``_default_steps``).
    pick_x : int or array-like of int, optional
        Indices of the coordinates to differentiate. The default is None,
        which differentiates all of them in order.

    Returns
    -------
    numpy.ndarray
        Jacobian [output units / coordinate units], shape
        (n_outputs, len(pick_x)); column ``k`` is the derivative with
        respect to ``x[pick_x[k]]``.

    Raises
    ------
    ValueError
        If the step of a differentiated coordinate is zero or infinite.
    """
    x = _float_work_array(x)  # [coordinate units]

    if callable(dx):
        dx = dx(x)
    elif dx is None:
        dx = _default_steps(x)
    else:
        dx = np.ones_like(x) * dx

    if pick_x is None:
        pick_x = np.arange(len(x))
    else:
        pick_x = np.atleast_1d(pick_x)
        if pick_x.size == 0:  # e.g. [], whose default dtype is float
            pick_x = pick_x.astype(np.intp)

    _check_steps(np.asarray(dx), pick_x)

    f_eval = func(x, *args)
    num_data = len(f_eval)
    num_states = len(pick_x)

    jac = np.zeros((num_data, num_states))
    delx = np.zeros_like(x)

    for idx, i in enumerate(pick_x):
        delx[i] = dx[i]
        jac[:, idx] = (func(x + delx, *args) - f_eval)/dx[i]
        delx[i] = 0

    return jac


def numerical_jacv(func, x, v, args=()):
    """Approximate the Jacobian-vector product ``J(x) @ v`` by a forward
    difference along ``v``.

    Parameters
    ----------
    func : callable
        Function ``func(x, *args)`` returning a scalar or one-dimensional
        output [output units].
    x : array-like
        Real point of differentiation [state units], shape (n,) or 0-d.
        Integer values are promoted to float.
    v : array-like
        Real direction [state units], same shape as ``x``.
    args : tuple, optional
        Additional function arguments [argument-dependent units].

    Returns
    -------
    jac_v : numpy.ndarray or float
        Directional derivative ``J(x) @ v`` [output units], with the shape
        of ``func(x)``. It is exactly zero when every component of ``v`` is
        zero, and NaN in every entry when a component of ``v`` is NaN or
        infinite, which has no finite directional difference (earlier
        versions also returned NaN there).

    Notes
    -----
    By linearity ``J v = s J w`` with ``s = max_k |v_k|`` [state units] and
    the dimensionless direction ``w = v / s``, whose largest components
    have magnitude one, so ``||w||`` neither underflows nor overflows for
    very small or large ``v``. ``J w`` is the forward difference
    ``(func(x + e w) - func(x)) / e`` with the directional increment of
    Brown and Saad, as given by Knoll and Keyes ("Jacobian-free
    Newton-Krylov methods: a survey of approaches and applications",
    J. Comput. Phys. 193, 2004, 357-397, Sec. 2.3.1)::

        e_dir = b * max(|x . w|, typx * ||w||_1) / ||w||_2**2

    The displacement ``e_dir w`` has length ``b`` times the larger of the
    state's projection on ``w``, ``|x . w| / ||w||_2``, and the
    typical-size floor ``typx ||w||_1 / ||w||_2``: a relative perturbation
    of the state along the direction, as ``b`` is for a scalar forward
    difference. When ``x . w`` cancels, ``e_dir`` can fall below the
    floating-point spacing of large components, so that ``x + e w == x``;
    the increment is therefore at least ``b`` times the largest
    ``|w_i| * max(|x_i|, typx)``, so that the component that the direction
    moves most relative to its own size is moved by a relative ``b`` and
    its perturbation is representable::

        e = max(e_dir, b * max_i(|w_i| * max(|x_i|, typx))) * sign(x . w)

    Weighting by ``|w_i|`` keeps a component that the direction barely
    moves (``|w_i|`` tiny) from setting a large step for the others.

    The sign moves the point away from zero along the projection and is
    taken positive when ``x . w = 0``. ``b = eps``, the square root of the
    machine epsilon, balances truncation and rounding error for a function
    evaluated to machine precision (Nocedal and Wright, Numerical
    Optimization, 2nd ed., Sec. 8.1); Knoll and Keyes recommend ``b``
    within a few orders of magnitude of it. ``typx = 1`` [state units] is
    an assumption, as the signature carries no state scales. ``x`` and
    ``v`` keep their shape, so 0-d inputs reach ``func`` as 0-d arrays.

    The perturbation is local relative to the components weighted by how
    far the direction moves them, ``|w_i| * max(|x_i|, typx)``: a
    component that is small, or moved little, relative to the largest of
    these can be perturbed by much more than ``b`` relative to its own size,
    or not at all after rounding. For directions mixing components of very
    different magnitude, scale the variables first. A component smaller
    than ``b * typx`` can be crossed.
    """
    x = _float_work_array(x)  # [state units]
    v = _float_work_array(v)  # [state units]

    f_eval = func(x, *args)  # [output units]
    result_dtype = np.result_type(f_eval, float)
    abs_v = np.abs(v)  # [state units]

    if not np.all(np.isfinite(abs_v)):
        # [output units]; [()] keeps a scalar output a scalar
        return np.full_like(f_eval, np.nan, dtype=result_dtype)[()]

    v_scale = np.max(abs_v)  # [state units]
    if v_scale == 0:
        # [output units]; [()] keeps a scalar output a scalar
        return np.zeros_like(f_eval, dtype=result_dtype)[()]

    direction = v / v_scale  # [-], largest |component| is 1
    typical_size = 1.0  # [state units], typx assumption (see Notes)
    projection = np.vdot(x, direction)  # [state units]
    magnitude = max(abs(projection),
                    typical_size * np.sum(np.abs(direction)))  # [state units]
    # [state units]
    directional = eps * magnitude / np.vdot(direction, direction)
    # [state units], largest |w_i| * max(|x_i|, typx)
    weighted_size = np.max(np.abs(direction)
                           * np.maximum(np.abs(x), typical_size))
    representable = eps * weighted_size  # [state units]
    increment = max(directional, representable)  # [state units]
    if np.real(projection) < 0:
        increment = -increment

    # [output units / state units]
    jac_w = (func(x + increment*direction, *args) - f_eval) / increment
    jac_v = v_scale * jac_w  # [output units]

    return jac_v
