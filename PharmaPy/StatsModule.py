#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 15:10:19 2019

@author: dcasasor
"""

import copy
import dataclasses
import numbers
from types import MappingProxyType
from typing import Mapping, NamedTuple, Optional, Sequence, Union

import numpy as np
import numpy.typing as npt
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle
import pandas as pd
from itertools import combinations

from PharmaPy.jac_module import DEFAULT_OUTPUT_RTOL
import time
import warnings


# Opacity [-] of each sampled prediction line in
# StatisticsClass.bootstrap_confid; the value of the original
# implementation, low enough that many overlapping draws build a visible
# density band. Display choice only.
_ENSEMBLE_LINE_ALPHA = 0.02  # [-]
# Subplot columns per row in StatisticsClass.plot_boots; layout of the
# original implementation. Display choice only.
_BOOT_PLOT_COLUMNS = 2  # [-]


def _validated_count(value, name: str) -> int:
    """Return a positive integer count or raise an actionable error.

    Parameters
    ----------
    value : int
        Candidate count [-].
    name : str
        Argument name for the error message.

    Returns
    -------
    int
        ``value`` as a Python ``int`` [-].

    Raises
    ------
    ValueError
        If ``value`` is not an integer (booleans excluded) of at least 1.
    """
    if (isinstance(value, bool) or not isinstance(value, (int, np.integer))
            or value < 1):
        raise ValueError(f"{name} must be a positive integer; got {value!r}")

    return int(value)


# Machine epsilon of float64 [-], the spacing of floats at 1: eps = 2u with
# u the unit roundoff of Golub & Van Loan (Matrix Computations, 4th ed.,
# 2013). A computed inverse carries a relative error of order u * cond
# (ibid., Sec. 2.6), so cond * eps >= 1, i.e. u * cond >= 1/2, leaves an
# error of order one and no correct significant digit; this is the
# singularity test of _information_inverse.
_FLOAT_EPS = np.finfo(float).eps  # [-]
# Decimal places [-] of the unit-vector components that the
# undefined-covariance diagnostic prints for each poorly determined
# direction. Display choice only: components that round to zero at this
# precision are left out of the message; the test itself is not affected.
_DIRECTION_DECIMALS = 3  # [-]
# Remedy of the undefined-covariance warnings for unidentifiable parameters.
_UNIDENTIFIABLE_REMEDY = (
    "Fix or remove those parameters, reparameterize the model in "
    "identifiable combinations, or add data that distinguish them, then "
    "refit")


def _sign_normalized(vector: np.ndarray) -> np.ndarray:
    """Fix the sign of a parameter-space direction for stable reporting.

    Parameters
    ----------
    vector : numpy.ndarray
        Unit singular vector of the Jacobi-scaled Jacobian, shape
        ``(n,)`` [-] (see ``_information_inverse``).

    Returns
    -------
    numpy.ndarray
        ``vector`` or ``-vector``, shape ``(n,)`` [-], chosen so that the
        first component that does not round to zero at
        ``_DIRECTION_DECIMALS`` decimals is positive; ``vector`` itself when
        every component rounds to zero.

    Notes
    -----
    A singular vector is defined up to its sign. The sign is fixed by the
    first component shown in the diagnostics, which round-off cannot flip,
    so messages and reports do not change between platforms.
    """
    rounded = np.round(vector, _DIRECTION_DECIMALS)  # [-]
    nonzero = np.flatnonzero(rounded)
    if nonzero.size and rounded[nonzero[0]] < 0:
        return -vector
    return vector


def _format_direction(vector: np.ndarray, names: list) -> str:
    """Write a parameter-space direction as a linear combination of names.

    Parameters
    ----------
    vector : numpy.ndarray
        Unit eigenvector of the Jacobi-scaled information matrix, shape
        ``(len(names),)`` [-] (see ``_information_inverse``).
    names : list of str
        Parameter name of each component.

    Returns
    -------
    str
        Combination such as ``'0.707*k - 0.707*b'`` of the components
        rounded to ``_DIRECTION_DECIMALS`` decimals, with the sign chosen
        so that the first nonzero rounded component is positive; components
        that round to zero are omitted.
    """
    # [-]; round(-x) == -round(x), so this is the rounded vector with the
    # sign of _sign_normalized.
    rounded = np.round(_sign_normalized(vector), _DIRECTION_DECIMALS)
    terms = []
    for coefficient, name in zip(rounded, names):
        if coefficient == 0:
            continue
        magnitude = f"{abs(coefficient):.{_DIRECTION_DECIMALS}f}*{name}"
        if not terms:
            terms.append(magnitude if coefficient > 0 else f"-{magnitude}")
        else:
            terms.append(('+ ' if coefficient > 0 else '- ') + magnitude)

    return ' '.join(terms)


def _jacobi_scaled_rows(jac: np.ndarray) -> tuple:
    """Scale the informed rows of a weighted Jacobian to unit norm.

    Parameters
    ----------
    jac : numpy.ndarray
        Finite weighted residual Jacobian ``J``, shape ``(num_params,
        n_entries)``; entry ``(i, k)`` in [weighted residual k unit /
        parameter i unit].

    Returns
    -------
    no_effect : numpy.ndarray
        Boolean mask, shape ``(num_params,)``, of the exactly zero rows
        (parameters with zero information, ``A_ii = 0`` for ``A = J J^T``).
    informed : numpy.ndarray
        Indices of the other rows, in parameter order, shape
        ``(num_informed,)``.
    scaled_jac : numpy.ndarray
        ``D J`` restricted to the informed rows, ``D = diag(A_ii**-0.5)``:
        every row has unit Euclidean norm, so entries are [-]. Shape
        ``(num_informed, max(n_entries, num_informed))``: when ``J`` has
        fewer columns than informed rows, zero columns are appended so that
        the SVD returns the zero singular values of the short matrix.

    Notes
    -----
    Each row is divided by its largest magnitude before it is normalized,
    so the squares inside ``numpy.linalg.norm``, and the division by the
    norm, are free of overflow and underflow. This is the scaling of the
    #532 singularity test (``_information_inverse``) and of
    ``IdentifiabilityReport``.
    """
    no_effect = ~jac.any(axis=1)  # exactly zero rows, A_ii = 0
    informed = np.flatnonzero(~no_effect)
    num_informed = len(informed)  # [-]
    rows = jac[informed].astype(float)  # [r_k/p_i]
    if num_informed:
        # [-], largest magnitude 1 per row
        rows = rows / np.abs(rows).max(axis=1, keepdims=True)
        rows = rows / np.linalg.norm(rows, axis=1, keepdims=True)
    scaled_jac = rows  # [-], unit row norms
    if scaled_jac.shape[1] < num_informed:
        # Zero columns add the zero singular values of a short J.
        scaled_jac = np.hstack((scaled_jac, np.zeros(
            (num_informed, num_informed - scaled_jac.shape[1]))))

    return no_effect, informed, scaled_jac


def _information_inverse(jac, param_names) -> tuple:
    """Invert the Gauss-Newton information matrix, or report it undefined.

    Parameters
    ----------
    jac : array_like
        Weighted residual Jacobian ``J``, shape ``(num_params,
        n_entries)``; entry ``(i, k)`` in [weighted residual k unit /
        parameter i unit]. Unobserved entries are zero columns.
    param_names : sequence of str
        Names of the optimized parameters in the row order of ``J``, used
        in the diagnostic.

    Returns
    -------
    inverse : numpy.ndarray
        ``numpy.linalg.inv(np.dot(jac, jac.T))``, the inverse of the
        information matrix ``A = J @ J.T``, shape ``(num_params,
        num_params)``, entry ``(i, j)`` in [parameter i unit * parameter j
        unit / weighted residual unit**2], when the linearized covariance is
        defined; otherwise an array of that shape filled with NaN.
    diagnostic : str or None
        None when the inverse is defined. Otherwise an actionable message
        of one of three kinds: ``J`` has non-finite entries (naming those
        parameters); the parameters are not (locally) identifiable from
        the observed data (naming every parameter with zero information
        and every poorly determined direction, see Notes); or ``A`` passed
        the conditioning test but its inversion failed or gave non-finite
        or negative variances (naming those parameters and the
        least-determined direction).

    Notes
    -----
    The inverse is undefined when ``J`` has a non-finite entry, when a
    row of ``J`` is exactly zero (parameter ``i`` changes no observed
    weighted residual, ``A_ii = 0``), when the Jacobi-scaled matrix
    ``A_s = D A D`` with ``D = diag(A_ii**-0.5)`` is numerically singular,
    when ``numpy.linalg.inv`` raises ``LinAlgError``, or when the inverse
    has a non-finite or negative diagonal entry (a negative variance).

    ``A_s`` is numerically singular when ``cond(A_s) * eps >= 1``, with
    ``eps`` the float64 machine epsilon, ``eps = 2u`` for the unit
    roundoff ``u``: the relative error of a computed inverse is of order
    ``u * cond`` (Golub & Van Loan, *Matrix Computations*, 4th ed., 2013,
    Sec. 2.6), so ``cond * eps >= 1`` means an error of order one and no
    correct significant digit. ``A_s`` is unchanged when the parameter
    units change (a diagonal rescaling ``S A S`` of ``A``), so the test
    does not depend on those units, whereas ``cond(A)`` does. Jacobi
    scaling is also nearly optimal: for a symmetric positive-definite
    matrix it gives a condition number within a factor ``num_params`` of
    the smallest over all diagonal scalings (van der Sluis, A., 1969,
    "Condition numbers and equilibration of matrices", Numerische
    Mathematik 14(1), 14-23, doi:10.1007/BF02165096).

    ``cond(A_s)`` is evaluated as ``(s_max / s_min)**2`` from the singular
    values ``s`` of the scaled Jacobian ``D J`` (``A_s = (D J)(D J)^T``),
    not from the eigenvalues of the rounded product ``A_s``. Forming
    ``J J^T`` rounds every entry by a few ``eps`` relative to the largest,
    so the smallest computed eigenvalue of an exactly singular ``A_s`` can
    land a few ``eps`` above zero and pass ``cond(A_s) * eps < 1``; the
    singular values of ``D J`` resolve ratios down to about ``eps``, that
    is ``cond(A_s)`` up to about ``1/eps**2``. When ``J`` has fewer
    columns than rows, the missing singular values are zero. Each row of
    ``J`` is divided by its largest magnitude before it is normalized, so
    the row norms neither overflow nor underflow.

    Passing the test bounds the error of an inverse of the exact ``A_s``,
    not of the rounded ``A``: forming ``J @ J.T`` adds its own rounding, and
    entries near the overflow or underflow limits are not representable, so
    an ``A`` just below the threshold can still invert to invalid
    variances. That case is reported separately, with the least-determined
    direction (smallest singular value of ``D J``).

    The poorly determined directions are the left singular vectors of
    ``D J`` (the eigenvectors of ``A_s``) whose eigenvalue ``s**2`` meets
    the same criterion, ``s**2 <= eps * s_max**2``. Their components refer
    to the scaled parameters ``u_i = dp_i * sqrt(A_ii)``: each parameter
    change divided by that parameter's standard error when the others are
    held fixed, so the components are unit-free [-]. Parameters with
    ``A_ii = 0`` are reported separately and left out of ``D J``.

    When the inverse is defined it is ``numpy.linalg.inv`` of
    ``np.dot(jac, jac.T)`` itself, bit for bit, as computed before #532.
    """
    jac = np.asarray(jac)  # [r_k/p_i], shape (num_params, n_entries)
    names = list(param_names)
    num_params = jac.shape[0]  # [-]
    undefined = np.full((num_params, num_params), np.nan)  # [p_i*p_j/r**2]
    preamble = ("The linearized parameter covariance is undefined and is "
                "returned as NaN")
    if num_params == 0:
        return np.linalg.inv(np.dot(jac, jac.T)), None

    finite_rows = np.isfinite(jac).all(axis=1)
    if not finite_rows.all():
        bad = [names[ind] for ind in np.flatnonzero(~finite_rows)]
        return undefined, (
            f"{preamble}: the weighted residual Jacobian has non-finite "
            f"entries for parameters {', '.join(bad)}. Check the model and "
            "its sensitivities at the accepted parameters")

    a_matrix = np.dot(jac, jac.T)  # [r**2/(p_i*p_j)], information matrix
    no_effect, informed, scaled_jac = _jacobi_scaled_rows(jac)
    num_informed = len(informed)  # [-]
    if num_informed:
        # [-]; columns of directions are eigenvectors of A_s, descending
        directions, singular, _ = np.linalg.svd(scaled_jac,
                                                full_matrices=False)
        eigvals = singular ** 2  # [-], eigenvalues of A_s, descending
        weak = eigvals <= _FLOAT_EPS * eigvals[0]
        condition = (eigvals[0] / eigvals[-1] if eigvals[-1] > 0
                     else np.inf)  # [-], cond(A_s)
    else:
        weak = np.zeros(0, dtype=bool)
        condition = np.inf  # [-]

    if not no_effect.any() and not weak.any():
        try:
            inverse = np.linalg.inv(a_matrix)  # [p_i*p_j/r**2]
        except np.linalg.LinAlgError:
            inverse = None
        if inverse is None:
            failure = "numpy.linalg.inv found it singular"
            bad = names
        else:
            variances = np.diag(inverse)  # [p_i**2/r**2]
            invalid = ~np.isfinite(variances) | (variances < 0)
            if not invalid.any():
                return inverse, None
            failure = "the inverse has non-finite or negative variances"
            bad = [names[ind] for ind in np.flatnonzero(invalid)]
        least = _format_direction(directions[:, -1], names)
        return undefined, (
            f"{preamble}: J @ J.T passed the conditioning test (Jacobi-scaled "
            f"condition number {condition:.3g} < 1/eps = "
            f"{1 / _FLOAT_EPS:.3g}), but {failure}, for parameters "
            f"{', '.join(bad)}. Rounding in forming or inverting J @ J.T, or "
            "sensitivities near the floating-point overflow or underflow "
            "limits, dominate. Least-determined parameter direction (smallest "
            "singular value of the scaled Jacobian; components are changes "
            "in units of each parameter's standard error with the others held "
            f"fixed): {least}. Rescale the parameters to moderate units, or "
            "fix, remove or reparameterize the poorly determined ones, then "
            "refit")

    parts = [f"{preamble}: the parameters are not (locally) identifiable "
             "from the observed data."]
    if no_effect.any():
        dead = [names[ind] for ind in np.flatnonzero(no_effect)]
        parts.append("Parameters with no effect on any observed entry "
                     f"(zero information): {', '.join(dead)}.")
    if weak.any():
        informed_names = [names[ind] for ind in informed]
        described = [_format_direction(directions[:, ind], informed_names)
                     for ind in np.flatnonzero(weak)]
        parts.append(
            "Poorly determined parameter directions (Jacobi-scaled "
            f"condition number of J @ J.T {condition:.3g} >= 1/eps = "
            f"{1 / _FLOAT_EPS:.3g}; components are changes in units of each "
            "parameter's standard error with the others held fixed): "
            + '; '.join(described) + '.')
    parts.append(_UNIDENTIFIABLE_REMEDY + "; the fitted parameters are kept, "
                 "but their covariance, correlations and confidence "
                 "intervals are not available.")

    return undefined, ' '.join(parts)


def _linearized_covariance(jac, residuals, dof, param_names,
                           include_mse: bool) -> tuple:
    """Compute the linearized parameter covariance of an accepted fit.

    This is the computation of ``ParameterEstimation.get_covariance``,
    without its warning and without storing anything, so that
    ``IdentifiabilityReport`` reproduces it bit for bit.

    Parameters
    ----------
    jac : numpy.ndarray
        Weighted residual Jacobian ``J``, shape ``(num_params,
        n_entries)``; entry ``(i, k)`` in [weighted residual k unit /
        parameter i unit]. Unobserved entries are zero columns.
    residuals : numpy.ndarray
        Weighted residuals ``r`` at the accepted parameters, shape
        ``(n_entries,)`` [weighted residual unit]; zero where unobserved.
    dof : int
        Degrees of freedom ``num_data_total - num_params`` [-], with
        ``num_data_total`` the number of observed entries; positive.
    param_names : sequence of str
        Parameter names in the row order of ``J``, used in the diagnostic.
    include_mse : bool
        If True, scale ``inv(J @ J.T)`` by the residual mean square.

    Returns
    -------
    covariance : numpy.ndarray
        ``s2 * inv(J @ J.T)`` [p_i*p_j] if ``include_mse``, otherwise
        ``inv(J @ J.T)`` [p_i*p_j / weighted residual unit**2]; shape
        ``(num_params, num_params)``. All NaN when undefined.
    correlation : numpy.ndarray
        Matching correlation matrix [-], same shape; all NaN when the
        covariance is undefined.
    mse : float
        Residual mean square ``s2 = r @ r / dof`` [weighted residual
        unit**2].
    diagnostic : str or None
        None when the covariance is defined, otherwise the message of
        ``_information_inverse`` stating why it is not.
    """
    mse = 1 / dof * np.dot(residuals.T, residuals)  # [r**2]

    # inv(J @ J.T) [p_i*p_j/r**2], computed as before #532 when defined;
    # NaN when the covariance is undefined.
    inverse, diagnostic = _information_inverse(jac, param_names)
    if diagnostic is not None:
        # All NaN, in the unit of the requested form: [p_i*p_j] with
        # include_mse, [p_i*p_j/r**2] without.
        covar = inverse
        correlation = np.full_like(inverse, np.nan)  # [-]
    else:
        if include_mse:
            covar = mse * inverse  # [p_i*p_j], scaled by s2
        else:
            covar = inverse  # [p_i*p_j/r**2]
        sigma = np.sqrt(covar.diagonal())  # [p_i] or [p_i/r]
        d_matrix = np.diag(1/sigma)
        correlation = d_matrix.dot(covar).dot(d_matrix)  # [-]

    return covar, correlation, mse, diagnostic


def _require_defined_covariance(covar_params, consumer: str) -> None:
    """Raise an actionable error for an undefined (all-NaN) covariance.

    Parameters
    ----------
    covar_params : array_like
        Parameter covariance, shape ``(num_params, num_params)``, entry
        ``(i, j)`` in the product of the units of parameters ``i`` and
        ``j``, or the inverse information matrix ``inv(J @ J.T)``.
        ``ParameterEstimation.get_covariance`` returns it filled with NaN
        when it is undefined.
    consumer : str
        Name of the calling method, used in the message.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If ``covar_params`` is a non-empty floating-point array whose
        entries are all NaN, the marker of an undefined covariance.
        Partially non-finite matrices are left to the caller's own checks.

    Notes
    -----
    Apply it only to a covariance (or ``inv(J @ J.T)``) that the estimator
    produced: a NaN matrix given by the user says nothing about
    identifiability and keeps the caller's own validation message.
    """
    covar = np.asarray(covar_params)  # [p_i*p_j]
    if covar.size and covar.dtype.kind == 'f' and np.isnan(covar).all():
        raise ValueError(
            f"{consumer} needs the linearized parameter covariance, which is "
            "undefined (all NaN) for this fit. The RuntimeWarning of "
            "get_covariance (also issued by optimize_fn) states the cause, "
            "parameters that the observed data do not identify, a non-finite "
            "Jacobian, or an inversion of J @ J.T without valid variances, "
            "and names the parameters involved; remove that cause and refit")


def _covariance_factor(covar_params, num_params: int,
                       check_symmetry: bool = True) -> np.ndarray:
    """Validate a parameter covariance and return its Cholesky factor.

    Parameters
    ----------
    covar_params : array-like
        Parameter covariance, shape ``(num_params, num_params)``; entry
        ``(i, j)`` in the product of the units of parameters ``i`` and
        ``j``.
    num_params : int
        Number of optimized parameters [-].
    check_symmetry : bool, optional
        If True (user-supplied matrices), reject a matrix that is not
        symmetric on the correlation scale. If False (the estimator's own
        covariance, symmetric in exact arithmetic), only symmetrize it.
        The default is True.

    Returns
    -------
    numpy.ndarray
        Lower-triangular ``L`` with ``L @ L.T`` equal to the symmetric part
        of ``covar_params``, shape ``(num_params, num_params)``; row ``i``
        in the units of parameter ``i``.

    Raises
    ------
    ValueError
        If the matrix has the wrong shape, non-finite entries, a
        non-positive variance, is not symmetric (``check_symmetry``), or is
        not positive definite.

    Notes
    -----
    The symmetry check compares each pair on the correlation scale,
    ``|C_ij - C_ji| / sqrt(C_ii C_jj) <= sqrt(eps)``, so it does not depend
    on the parameter units. The ``sqrt(eps)`` tolerance is a design
    assumption: far above the round-off of a symmetric matrix written out
    by a user, far below any meaningful correlation. It is not applied to
    the estimator's ``s2 * inv(J J^T)``, whose LU inversion leaves an
    asymmetry growing with ``cond(J J^T)``. In both cases the symmetric part
    ``(C + C^T) / 2`` is factorized.
    """
    covar = np.asarray(covar_params, dtype=float)  # [param_i*param_j units]
    expected_shape = (num_params, num_params)
    if covar.shape != expected_shape:
        raise ValueError(
            f"covar_params has shape {covar.shape}; expected "
            f"{expected_shape}, one row and column per optimized parameter")
    if not np.all(np.isfinite(covar)):
        raise ValueError("covar_params must contain only finite values")

    variances = np.diag(covar)  # [param_i**2 units]
    if np.any(variances <= 0):
        raise ValueError(
            "covar_params must be positive definite to draw parameter "
            f"samples; its diagonal (variances) is {variances.tolist()}")

    if check_symmetry:
        # [-], pairwise asymmetry on the correlation scale
        std = np.sqrt(variances)  # [param units]
        asymmetry = np.abs(covar - covar.T) / np.outer(std, std)
        if asymmetry.max() > np.sqrt(np.finfo(float).eps):
            raise ValueError(
                "covar_params must be symmetric; largest asymmetry on the "
                f"correlation scale is {asymmetry.max():.3g}")

    try:
        return np.linalg.cholesky((covar + covar.T) / 2)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "covar_params must be positive definite to draw parameter "
            f"samples: {exc}") from exc


def _isolated_estimator(estimator):
    """Return a shallow copy of an estimator safe to evaluate and fit.

    Parameters
    ----------
    estimator : ParameterEstimation
        Fitted estimator; it is not modified.

    Returns
    -------
    ParameterEstimation
        Copy sharing the model callable, its arguments, the data layout
        and the weighting with ``estimator``, but owning every container
        that ``optimize_fn`` or the objective callback modify in place.

    Notes
    -----
    ``optimize_fn`` and its callbacks rebind estimator attributes
    (``y_data``, ``params_convg``, ``info_opt``, ``resid_runs``,
    ``y_model``, ``covar_params`` and others), so rebinding them on the
    copy leaves the original fit untouched. The exceptions, which are
    modified in place, are the iteration histories ``params_iter`` and
    ``objfun_iter`` (appended to while they are lists; ``objfun_iter``
    stays a list after ``optimize_fn(store_iter=False)``) and the cache
    ``_observed_roots``. The copy gets its own lists and cache. The solver
    options need no isolation: ``optimize_fn`` passes the solver a private
    copy and leaves the stored ``optim_options`` untouched (issue #476).
    Model callables and their arguments are shared, not copied, so models
    that close over external objects keep working.
    """
    isolated = copy.copy(estimator)
    for name in ('params_iter', 'objfun_iter'):
        history = getattr(isolated, name, None)
        if isinstance(history, list):
            setattr(isolated, name, list(history))

    roots = getattr(isolated, '_observed_roots', None)
    if isinstance(roots, dict):
        isolated._observed_roots = dict(roots)

    return isolated


def _fit_bootstrap_sample(estimator, y_sample: list) -> np.ndarray:
    """Fit one dataset without modifying the fitted estimator.

    Parameters
    ----------
    estimator : ParameterEstimation
        Fitted estimator whose model, weighting, solver method
        (``opt_method``) and solver options (``optim_options``) are reused;
        it is not modified.
    y_sample : list of numpy.ndarray
        One dataset per experiment, in ``x_data`` order, with the layout of
        ``StatisticsClass.get_bootsamples``: shape ``(n_times,
        n_measured)``, model-grid rows and measured-state columns in the
        states' units.

    Returns
    -------
    numpy.ndarray
        Optimized parameters, shape ``(num_params,)``, in the model's
        parameter units.

    Raises
    ------
    Exception
        Whatever ``optimize_fn`` raises for this dataset.

    Notes
    -----
    The fit runs on ``_isolated_estimator(estimator)`` with a copy of the
    fitted estimator's effective ``optim_options``. A stateful model
    callable is shared and is left at its last evaluation; see
    ``_restore_model_state``.
    """
    isolated = _isolated_estimator(estimator)
    isolated.y_data = y_sample  # [measured-state units]

    options = estimator.optim_options
    params, _, _ = isolated.optimize_fn(
        method=estimator.opt_method,
        verbose=False, store_iter=False,
        optim_options=None if options is None else dict(options))

    return params  # [model-parameter units]


def _restore_model_state(estimator) -> None:
    """Re-evaluate the model at the estimator's accepted parameters.

    Parameters
    ----------
    estimator : ParameterEstimation
        Fitted estimator; it is not modified.

    Returns
    -------
    None

    Raises
    ------
    Exception
        Whatever the model raises at the accepted parameters.

    Notes
    -----
    Refits and evaluations on isolated copies share the estimator's model
    callable. A stateful callable, such as a ``SimulationExec`` unit
    operation, is left at its last evaluation; this re-evaluates the model
    at the accepted ``params_convg``, matching the final refresh that
    ``optimize_fn`` performs after an LM fit. (After an IPOPT fit, whose
    last model calls are finite-difference perturbations, the callable ends
    at the accepted parameters rather than at that perturbed point.)

    The objective is evaluated on ``_isolated_estimator(estimator)``, so
    every attribute it rebinds or appends to (residuals, predictions,
    sensitivities, histories) belongs to the copy and the estimator stays
    unchanged. Nothing is evaluated when the estimator has no accepted
    parameters (``params_convg``).
    """
    accepted = getattr(estimator, 'params_convg', None)  # [param units]
    if accepted is None:
        return

    _isolated_estimator(estimator).get_objective(accepted, True)


def _restore_model_state_after_error(estimator, failed_step: str) -> None:
    """Restore the model after a failure without masking that failure.

    Parameters
    ----------
    estimator : ParameterEstimation
        Fitted estimator whose model callable is restored; it is not
        modified.
    failed_step : str
        Description of the failed operation for the warning message.

    Returns
    -------
    None

    Warns
    -----
    RuntimeWarning
        If the model fails at the accepted parameters. The report is
        dropped when warnings are turned into errors.

    Notes
    -----
    Call from an ``except`` block and re-raise afterwards: the caller's
    exception always takes priority over a restoration failure.
    """
    try:
        _restore_model_state(estimator)
    except Exception as refresh_error:
        try:
            warnings.warn(
                "Could not restore the model at the accepted parameters "
                "after {} failed: {}".format(failed_step, refresh_error),
                RuntimeWarning,
                stacklevel=3,
            )
        except Exception:
            # A warnings filter set to 'error' turns the report into an
            # exception; the original error still takes priority.
            pass


class ParallelProblem:
    """Refit a fitted estimator to one given dataset.

    The dataset replaces the observations of an isolated copy of the
    estimator, which is then refitted with the estimator's solver method
    and options, as ``StatisticsClass.bootstrap_params`` does for each
    bootstrap dataset. The estimator itself is never modified.
    """

    def __init__(self, instance):
        """Store the fitted estimator.

        Parameters
        ----------
        instance : ParameterEstimation
            Estimator on which ``optimize_fn`` has been run.
        """
        self.instance = instance

    def run_fit(self, sample):
        """Fit the given dataset and return its parameter estimates.

        Parameters
        ----------
        sample : list of array-like
            One dataset per experiment, in ``x_data`` order, in the layout
            of ``StatisticsClass.get_bootsamples`` (one entry of a
            bootstrap sample): shape ``(n_times, n_measured)`` on the
            estimator's model grid, measured-state columns in the states'
            units, NaN where unobserved. A 1-D array of length ``n_times``
            is accepted for an experiment with one measured state.

        Returns
        -------
        numpy.ndarray
            Optimized parameters for ``sample``, shape ``(num_params,)``, in
            the model's parameter units.

        Raises
        ------
        RuntimeError
            If the estimator has not been fitted (no ``optimize_fn`` run).
        ValueError
            If ``sample`` does not hold one array per experiment with the
            shape of that experiment's observations, holds NaN or infinite
            values at observed positions, or the estimator stores nested
            ``MultipleCurveResolution`` observations (#505).
        Exception
            Whatever ``optimize_fn`` raises for this dataset.

        Warns
        -----
        RuntimeWarning
            If the fit raises and the model then also fails at the accepted
            parameters; the fit's exception is still raised.

        Notes
        -----
        The fit runs on an isolated copy (see ``_fit_bootstrap_sample``), so
        the estimator keeps its observations, accepted parameters,
        residuals, predictions, solver information and histories. A stateful
        model callable is returned to the accepted parameters afterwards
        (``_restore_model_state``), also when the fit raises.
        """
        estimator = self.instance
        if getattr(estimator, 'params_convg', None) is None:
            raise RuntimeError(
                "ParallelProblem.run_fit needs a fitted estimator; run "
                "optimize_fn first")

        y_sample = _validated_sample(estimator, sample)

        try:
            params = _fit_bootstrap_sample(
                estimator, y_sample)  # [model-parameter units]
        except Exception:
            _restore_model_state_after_error(estimator, 'the sample fit')
            raise

        _restore_model_state(estimator)

        return params  # [model-parameter units]


def _validated_sample(estimator, sample) -> list:
    """Check a refit dataset against the estimator's observation layout.

    Parameters
    ----------
    estimator : ParameterEstimation
        Fitted estimator defining the number of experiments and the shape
        ``(n_times, n_measured)`` of each experiment's observations.
    sample : list of array-like
        One dataset per experiment, in ``x_data`` order [measured-state
        units].

    Returns
    -------
    list of numpy.ndarray
        Float arrays, one per experiment, with the observations' shapes
        [measured-state units]; 1-D single-state entries become columns.

    Raises
    ------
    ValueError
        If the estimator stores nested ``'spectra'``/``'non_spectra'``
        observations (``MultipleCurveResolution``, not supported, #505); if
        ``sample`` is not a list or tuple with one entry per experiment; if
        an entry's shape differs from the experiment's observations; or if
        an entry is NaN or infinite at an observed position.

    Notes
    -----
    Observed positions are those of the estimator's observation masks
    (``x_masks``; every entry when the mask is None). Unobserved entries,
    which ``get_bootsamples`` sets to NaN, may hold any value: their
    residuals are zeroed before the fit.
    """
    if any(isinstance(observed, dict) for observed in estimator.y_data):
        raise ValueError(
            "ParallelProblem.run_fit does not support nested 'spectra'/"
            "'non_spectra' observations (MultipleCurveResolution); bootstrap "
            "refits of that layout are tracked in #505")

    num_datasets = estimator.num_datasets
    if (not isinstance(sample, (list, tuple))
            or len(sample) != num_datasets):
        found = (f"a list of {len(sample)} entries"
                 if isinstance(sample, (list, tuple))
                 else type(sample).__name__)
        raise ValueError(
            f"sample must be a list with one array per experiment "
            f"({num_datasets}, in x_data order, as one entry of "
            f"get_bootsamples); got {found}")

    y_sample = []
    for ind, (values, observed, x_mask) in enumerate(zip(
            sample, estimator.y_data, estimator.x_masks)):
        name = estimator.experim_names[ind]
        expected_shape = np.shape(observed)  # (n_times, n_measured)
        values = np.asarray(values, dtype=float)  # [measured-state units]
        if values.ndim == 1 and expected_shape[1:] == (1,):
            values = values[:, np.newaxis]
        if values.shape != expected_shape:
            raise ValueError(
                f"sample entry {ind} ({name!r}) has shape "
                f"{np.shape(sample[ind])}; expected {expected_shape} "
                "(model-grid rows, measured-state columns)")

        observed_mask = (np.ones(expected_shape, dtype=bool) if x_mask is None
                         else np.asarray(x_mask, dtype=bool))
        invalid = observed_mask & ~np.isfinite(values)
        if invalid.any():
            positions = [tuple(pos) for pos in np.argwhere(invalid).tolist()]
            raise ValueError(
                f"sample entry {ind} ({name!r}) is NaN or infinite at "
                f"observed (row, column) positions {positions}; only "
                "unobserved entries may be non-finite")
        y_sample.append(values)

    return y_sample


class StatisticsClass:

    def __init__(self, estimation_instance, alpha=0.95):
        """Collect the fitted quantities used by the statistics methods.

        Parameters
        ----------
        estimation_instance : ParameterEstimation
            Fitted estimator (``optimize_fn`` already run): its
            ``info_opt`` (``'jac'``, ``'fun'``), ``params_convg``,
            ``param_seed``, ``map_fixed``, ``y_model``, ``resid_runs``,
            ``name_params``, ``num_data_total``, ``num_params`` and
            ``x_data`` are read.
            It is referenced as ``inst`` and not modified.
        alpha : float, optional
            Confidence level in (0, 1) [-] of the intervals and regions,
            e.g. 0.95 for 95 % Student-t intervals. The default is 0.95.

        Notes
        -----
        The attribute ``sigma_t`` (always None, never assigned) was removed
        together with the ``bootstrap_confid`` repair (#400).

        ``a_matrix`` is ``J @ J.T`` of the weighted residual Jacobian
        ``J = info_opt['jac']`` and ``a_inv`` its inverse, computed with the
        test of ``ParameterEstimation.get_covariance`` (see
        ``_information_inverse``): when the covariance is undefined (the
        parameters are not identifiable, ``J`` is not finite, or the
        inversion fails), ``a_inv`` is filled with NaN instead of raising
        ``numpy.linalg.LinAlgError`` or holding meaningless values, so the
        bootstrap methods, which refit the data, stay available, and
        ``confidence_regions`` raises an actionable ``ValueError``.
        """

        # Values from the instance
        inst = estimation_instance
        jac = inst.info_opt['jac']

        # [r**2/(p_i*p_j)], information matrix J J^T
        self.a_matrix = jac.dot(jac.T)
        # [p_i*p_j/r**2], NaN when the covariance is undefined
        self.a_inv, _ = _information_inverse(jac, inst.name_params)
        self.jac = jac

        self.params = inst.params_convg.copy()
        self.params_unactive = inst.param_seed[inst.map_fixed]

        self.y_nominal = inst.y_model

        self.residuals = inst.resid_runs

        self.param_names = inst.name_params

        # Calculations
        self.dof = inst.num_data_total - inst.num_params

        residuals = inst.info_opt['fun'].copy()
        resid_squared = np.dot(residuals.T, residuals)  # SSE
        self.var_fun = resid_squared / self.dof

        self.num_par = len(self.params)

        self.xgrid = inst.x_data

        self.inst = inst

        self.alpha = alpha

        # --------------- Pair-wise combinations
        # Indexes and names
        self.combs = tuple(combinations(range(self.num_par), 2))
        self.name_pairs = []
        for comb in self.combs:
            names = [self.param_names[ind] for ind in comb]
            name_pair = '/'.join(names)
            self.name_pairs.append(name_pair)

        # --------------- Output vals
        self.confid_intervals = None
        self.boot_params = None

    def get_intervals(self, covar_params=None, verbose=True, set_self=True):
        """Compute two-sided Student-t intervals of the fitted parameters.

        Parameters
        ----------
        covar_params : array_like, optional
            Parameter covariance, shape ``(num_params, num_params)``, entry
            ``(i, j)`` in the product of the units of parameters ``i`` and
            ``j``. The default None uses the estimator's ``covar_params``.
        verbose : bool, optional
            If True (the default), print the interval table.
        set_self : bool, optional
            If True (the default), store ``confid_intervals``,
            ``covar_params`` and ``intervals_array`` on this object.

        Returns
        -------
        dict
            Mapping from each parameter name in ``param_names`` to its
            two-element array ``[lower_bound, upper_bound]`` in that
            parameter's units: ``params -/+ t * sqrt(diag(covar_params))``
            with ``t`` the ``(1 + alpha) / 2`` quantile of Student's t
            distribution with ``dof`` degrees of freedom.

        Raises
        ------
        ValueError
            If ``covar_params`` is None and the estimator's covariance is
            undefined (all NaN, as ``ParameterEstimation.get_covariance``
            returns it; its warning states the cause). A given
            ``covar_params`` is used as is.

        Notes
        -----
        ``confid_intervals['perc_deviation']`` is the half-width relative to
        the parameter value [%].
        """
        if covar_params is None:
            covar_params = self.inst.covar_params
            _require_defined_covariance(covar_params,
                                        'StatisticsClass.get_intervals')

        sigma_params = np.sqrt(np.diag(covar_params))

        t_stat = stats.t.ppf((1 + self.alpha)/2, self.dof)

        delta_par = sigma_params * t_stat

        intervals = np.column_stack((self.params - delta_par,
                                     self.params + delta_par))

        confidence_int = dict(zip(self.param_names, intervals))

        perc_deviation = np.abs(delta_par / self.params * 100)

        if verbose:
            alperc = self.alpha * 100
            print('')
            print('{:<55}'.format('-'*100))
            print("{:<12} {:^15} {:^15} {:^15} {:^15} {:^15}".format(
                    'param', 'lb', 'mean', 'ub',
                    "%i%% CI (+/-)" % alperc, "%i%% CI (+/- %%)" % alperc))

            print('{:<55}'.format('-'*100))

            for ind, (key, val) in enumerate(confidence_int.items()):
                low, high = val
                print("{:<12} {:^15.3e} {:^15.3e} {:^15.3e} {:^15.3e} {:^15.5f}".format(
                        key, low, self.params[ind], high, delta_par[ind], perc_deviation[ind]))

            print('{:<55}'.format('-'*100))

        if set_self:
            self.confid_intervals = {'intervals': confidence_int,
                                     'delta_param': delta_par,
                                     'perc_deviation': perc_deviation}

            self.covar_params = covar_params
            self.intervals_array = intervals

        return confidence_int

    def confidence_regions(self, param_means, alphas=None):
        """Generate pairwise linearized confidence ellipses and boxes.

        For three or more parameters, the 2D projections are calculated as
        described in 'Nonlinear Regression' by Seber and Wild (1989).

        Parameters
        ----------
        param_means : pandas.Series
            Centre of each region, one entry per optimized parameter in
            ``param_names`` order and units, indexed by position.
        alphas : tuple or list, optional
            Confidence levels [-] of the regions to draw. If None, the alpha
            value specified when creating the instance is used.
            The default is None.

        Returns
        -------
        info_dict : dict of dicts
            Dictionary with keys as the parameter pair names, e.g.
            'theta_1/theta_2'. Each dictionary value is in turn a dictionary
            with the ellipse and rectangle patches and their respective
            dimensions (axes lengths or side lengths) in the units of the
            pair's parameters.

        Raises
        ------
        ValueError
            If the inverse information matrix ``a_inv`` is undefined (all
            NaN; see ``ParameterEstimation.get_covariance`` for the causes);
            the linearized regions do not exist then.
        """
        _require_defined_covariance(self.a_inv,
                                    'StatisticsClass.confidence_regions')

        info_dict = {}
        indexes = set(range(self.num_par))
        ident = np.eye(self.jac.shape[1])
        for comb, name in zip(self.combs, self.name_pairs):
            comb = list(comb)

            param_pair = param_means[comb].values
            diff = list(indexes.difference(comb))

            if len(diff) == 0:
                a_projec = self.a_matrix
                a_inv = self.a_inv
            else:
                f_1 = self.jac[diff].T
                f_2 = self.jac[comb].T

                projection = f_1.dot(np.linalg.inv(f_1.T.dot(f_1))).dot(f_1.T)
                a_projec = f_2.T.dot(ident - projection).dot(f_2)
                a_inv = np.linalg.inv(a_projec)

            el, rec, axlengs, recdims = self.__confidence_pair(
                a_projec, a_inv, param_pair, alphas=alphas)

            info_dict[name] = {'ellipses': el, 'rectangles': rec,
                               'ellipse_axes': axlengs,
                               'rectangle_dims': recdims}

        return info_dict

    def __confidence_pair(self, a_projected, a_inverse, param_pair,
                          alphas=None):

        # Select eigenvals, eigenvecs and params
        eig_vals, eig_vecs = np.linalg.eig(a_projected)

        # Statistic
        if alphas is None:
            alphas = [self.alpha]

        f_stat = stats.f.ppf(alphas, self.num_par, self.dof)  # TODO: I think self.num_par should be just 2
        contours = f_stat * self.num_par * self.var_fun

        # Order
        orders = np.argsort(eig_vals)[::-1]

        eig_vals = eig_vals[orders]
        eig_vecs = eig_vecs[:, orders]

        # Get rotation angle (angle of the largest eigenvector)
        angle = np.arctan2(eig_vecs[1, 0], eig_vecs[0, 0])
        angle = np.degrees(angle)

        ellipses = []
        axes_len = []

        rectangles = []
        width_height = []
        for contour in contours:
            axes = np.sqrt(contour / eig_vals)  # half width
            axes *= 2

            # Draw ellipse
            ellipse = Ellipse(param_pair, *axes, angle=angle,
                              alpha=0.2, edgecolor='k', linestyle='--',
                              zorder=5)

            # Draw rectangle
            side_len = np.sqrt(contour * np.diag(a_inverse))
            width = side_len[0] * 2
            height = side_len[1] * 2

            rectangle = Rectangle(param_pair - side_len, width, height,
                                  fill=False, zorder=5, alpha=0.4)

            # Store
            ellipses.append(ellipse)
            axes_len.append(axes)

            rectangles.append(rectangle)
            width_height.append(np.array([width, height]))

        return ellipses, rectangles, axes_len, width_height

    def plot_pairwise(self, boots=None, alphas=None, bounding_box=False,
                      interval_bounds=True, fig_size=(5, 5)):

        boot_params = boots

        if boot_params is None:
            boot_params = self.boot_params
            confid_interv = self.intervals_array

        else:
            covar_boots = np.cov(boot_params.T)
            confid_interv = self.get_intervals(covar_boots, verbose=False,
                                               set_self=False)

            confid_interv = np.vstack(list(confid_interv.values()))

        if boot_params is None:
            raise RuntimeError(
                "No data to plot. Run 'bootstrap_params' method first ")

        params = pd.DataFrame(boot_params)
        param_means = params.mean(axis=0)

        axes = pd.plotting.scatter_matrix(params, figsize=fig_size,
                                          hist_kwds={'bins': 50}, zorder=10,
                                          density_kwds={'s': 10})

        for ind in range(self.num_par):
            axes[ind, 0].set_ylabel(self.inst.name_params_plot[ind])
            axes[self.num_par - 1, ind].set_xlabel(
                self.inst.name_params_plot[ind])

        regions = self.confidence_regions(param_means, alphas=alphas)
        for ind, vals in enumerate(regions.values()):
            ovals = vals['ellipses']
            bounding = vals['rectangles']
            idx = self.combs[ind][::-1]  # lower triangular quadrant index

            for region, box in zip(ovals, bounding):

                axes[idx].add_patch(region)

                if interval_bounds:
                    # Show confidence intervals
                    intervals = confid_interv[idx[::-1], :]
                    for row in intervals.T:
                        axes[idx].axvline(row[0], ls='--', zorder=0)
                        axes[idx].axhline(row[1], ls='--', zorder=0)

                # axes[idx].autoscale()

                if bounding_box:
                    axes[idx].add_patch(box)

            param_pair = param_means[list(idx)[::-1]].values
            axes[idx].scatter(*param_pair, marker='+', color='k')

            width_height = vals['rectangle_dims']
            rect_array = np.array(width_height)

            max_dims = rect_array.max(axis=0) * 1.2

            fac = 0.5

            axes[idx].set_xlim(param_pair[0] - fac*max_dims[0],
                               param_pair[0] + fac*max_dims[0])

            axes[idx].set_ylim(param_pair[1] - fac*max_dims[1],
                               param_pair[1] + fac*max_dims[1])

        # Hide upper triangular
        mask = np.triu_indices_from(axes, 1)

        for row, col in zip(*mask):
            axes[row, col].set_visible(False)

        axes[0, 0].yaxis.set_ticks([])

        return axes

    def bootstrap_confid(self, num_samples, plot=True, covar_params=None):
        """Propagate parameter uncertainty into model predictions.

        Monte-Carlo propagation of the linearized parameter covariance:
        parameter vectors are drawn from the multivariate normal
        ``N(params_convg, covar_params)`` and the model is evaluated at
        each of them.

        Parameters
        ----------
        num_samples : int
            Number of parameter draws [-]; at least 1.
        plot : bool, optional
            If True, plot the nominal predictions and the sampled
            predictions of each experiment. The default is True.
        covar_params : numpy.ndarray, optional
            Covariance of the optimized parameters, shape
            ``(num_params, num_params)``, entry ``(i, j)`` in the product of
            the units of parameters ``i`` and ``j``; symmetric positive
            definite. The default is the estimator's ``covar_params``, the
            linearized covariance that ``get_intervals`` uses.

        Returns
        -------
        samples : numpy.ndarray
            Drawn parameters, shape ``(num_samples, num_params)``, in
            optimized-parameter (``name_params``) order and units.
        predictions : list of lists of numpy.ndarray
            One entry per sample, in ``samples`` order, holding one array
            per experiment (``x_data`` order) of the model predictions at
            the measured states, shape ``(n_times, n_measured)`` on that
            experiment's model grid (``x_model``), in the measured states'
            units. Unlike ``y_model``, entries are model values also where
            a state is not observed.
        fig : matplotlib.figure.Figure or None
            Figure, or None when ``plot`` is False.
        axis : numpy.ndarray of matplotlib.axes.Axes or None
            One axis per experiment, shape ``(num_datasets,)``, or None when
            ``plot`` is False.

        Raises
        ------
        ValueError
            If ``num_samples`` is not a positive integer, or the covariance
            is not a finite, positive-definite ``(num_params, num_params)``
            matrix; a given ``covar_params`` must also be symmetric on the
            correlation scale (see ``_covariance_factor``). The estimator's
            own covariance, when undefined (all NaN, see
            ``ParameterEstimation.get_covariance``), is rejected with a
            message saying so.
        Exception
            Whatever the model raises at a drawn parameter vector.

        Warns
        -----
        RuntimeWarning
            If a model evaluation raises and the model then also fails at
            the accepted parameters; the evaluation's exception is still
            raised.

        Notes
        -----
        Each draw is ``theta_s = params_convg + L z_s``, where ``L`` is the
        lower Cholesky factor of ``covar_params`` (``L L^T = covar_params``)
        and ``z_s`` holds ``num_params`` independent standard normal draws
        from NumPy's global generator (``numpy.random.standard_normal``,
        one ``(num_samples, num_params)`` call), so ``numpy.random.seed``
        reproduces the samples, as for ``get_bootsamples``. ``L`` factorizes
        the symmetric part of the covariance; a given ``covar_params`` is
        rejected unless symmetric to ``sqrt(eps)`` on the correlation scale,
        while the estimator's own covariance, symmetric in exact arithmetic,
        is symmetrized without that check. Earlier releases scaled the
        standard normals by the parameter variances instead of the standard
        deviations and never ran (#400); they read the attribute
        ``StatisticsClass.sigma_t``, which was never assigned (always None)
        and has been removed.

        The model is evaluated with ``get_objective`` on an isolated copy
        of the estimator, so the estimator keeps its observations, accepted
        parameters, residuals and predictions. A stateful model callable is
        returned to the accepted parameters afterwards, also when an
        evaluation raises. The covariance is a linearization around
        ``params_convg``; the sampled predictions inherit that
        approximation.
        """
        num_samples = _validated_count(num_samples, 'num_samples')
        # Lower Cholesky factor [param units]. The estimator's own
        # covariance is only symmetrized; a user matrix is also checked.
        if covar_params is None:
            _require_defined_covariance(self.inst.covar_params,
                                        'StatisticsClass.bootstrap_confid')
            chol_lower = _covariance_factor(self.inst.covar_params,
                                            self.num_par,
                                            check_symmetry=False)
        else:
            chol_lower = _covariance_factor(covar_params, self.num_par)

        # [-], independent standard normal draws
        std_normal = np.random.standard_normal((num_samples, self.num_par))
        samples = self.params + std_normal @ chol_lower.T  # [param units]

        estimator = _isolated_estimator(self.inst)
        try:
            estimator.get_objective(self.params, True)
            # [measured-state units], per experiment (n_times, n_measured)
            nominal = [np.array(y_run, dtype=float)
                       for y_run in estimator.y_runs]

            predictions = []
            for sample in samples:
                estimator.get_objective(sample, True)
                predictions.append([np.array(y_run, dtype=float)
                                    for y_run in estimator.y_runs])
        except Exception:
            _restore_model_state_after_error(self.inst,
                                             'a sampled model evaluation')
            raise

        _restore_model_state(self.inst)

        if not plot:
            return samples, predictions, None, None

        fig, axes = plt.subplots(1, self.inst.num_datasets, squeeze=False)
        axes = axes[0]
        for ind, axis in enumerate(axes):
            grid = self.inst.x_model[ind]  # [x units]
            nominal_lines = axis.plot(grid, nominal[ind])
            colors = [line.get_color() for line in nominal_lines]
            for prediction in predictions:
                for col, color in enumerate(colors):
                    axis.plot(grid, prediction[ind][:, col], color=color,
                              alpha=_ENSEMBLE_LINE_ALPHA)

            axis.set_title(self.inst.experim_names[ind])
            axis.spines['right'].set_visible(False)
            axis.spines['top'].set_visible(False)

        return samples, predictions, fig, axes

    def _observed_columns(self, ind):
        """Return the observation mask of one experiment's residual columns.

        Parameters
        ----------
        ind : int
            Experiment index in ``x_data`` order.

        Returns
        -------
        numpy.ndarray
            Boolean array with the shape of ``self.residuals[ind]``,
            ``(n_times, n_columns)``: True where the residual belongs to an
            observation, False at every unobserved entry (samples of
            staggered grids where a state is not measured, NaN-missing
            observations, and measurement columns the experiment does not
            include). For ``Experiment`` input the columns are the
            estimator's measurement columns in first-appearance order;
            otherwise they follow ``measured_ind``.

        Raises
        ------
        ValueError
            If a ``MultipleCurveResolution`` mask does not match the
            residual columns.

        Notes
        -----
        ``ParameterEstimation`` stores None (all observed) or a
        ``(n_times, n_measured)`` array. ``MultipleCurveResolution`` stores a
        dictionary of per-row masks keyed ``'spectra'`` and, optionally,
        ``'non_spectra'``; its residual columns hold every spectral channel
        followed by every non-spectral state (``get_global_analysis``), so
        each column takes the row mask of its group; the channel count comes
        from ``spectra_tot`` and the state count from ``measured_ind``.
        """
        residual = self.residuals[ind]
        x_mask = self.inst.x_masks[ind]
        if x_mask is None:
            return np.ones(residual.shape, dtype=bool)

        if not isinstance(x_mask, dict):
            return np.asarray(x_mask, dtype=bool)

        num_columns = {
            'spectra': self.inst.spectra_tot.shape[1],  # wavelengths
            'non_spectra': len(self.inst.measured_ind.get('non_spectra', [])),
        }
        columns = []
        for key in ('spectra', 'non_spectra'):
            if key in x_mask:
                columns += ([np.asarray(x_mask[key], dtype=bool)]
                            * num_columns[key])

        if len(columns) != residual.shape[1]:
            raise ValueError(
                f"Observation masks of experiment {ind} describe "
                f"{len(columns)} residual columns, but the residuals have "
                f"{residual.shape[1]}")

        return np.column_stack(columns)

    def get_bootsamples(self, num_samples, fix_initial=False):
        """Create residual-bootstrap datasets around the fitted responses.

        For each experiment and measured state, the residuals at observed
        model-grid entries are resampled with replacement and subtracted from
        the fitted responses at those entries. When the measurements declare
        standard deviations (``Measurement.uncertainty``), the standardized
        residuals are resampled instead and rescaled by the standard
        deviation of the entry they are assigned to. The datasets are fitted
        by ``bootstrap_params``.

        Parameters
        ----------
        num_samples : int
            Number of bootstrap datasets.
        fix_initial : bool, optional
            If True, the first model-grid row keeps its fitted value and is
            excluded from the resampled residual pool. The default is False.

        Returns
        -------
        y_boot : list of lists of numpy.ndarray
            One list per experiment, in ``x_data`` order, of ``num_samples``
            arrays with shape ``(n_times, n_measured)``: model-grid rows and
            measured-state (``measured_ind``) columns, in the measured
            states' units. Unobserved entries (staggered grids, missing
            observations, measurements an experiment does not include) are
            NaN.

        Raises
        ------
        ValueError
            If a ``MultipleCurveResolution`` observation mask does not match
            the residual columns (see ``_observed_columns``).

        Notes
        -----
        Each state's resampling pool holds only its observed residuals
        (model minus data, in that state's unit) of one experiment. Columns
        stored as data minus model, such as the spectral channels of
        ``MultipleCurveResolution``, are negated first (the estimator's
        ``_residual_signs``), so every generated entry is the fitted value
        minus a drawn model-minus-data residual.
        Unobserved entries have zero residual by construction; drawing them
        would add phantom zero errors and shrink the bootstrap spread. Each
        state is resampled independently, so correlations between
        measurement errors of different states are not reproduced. Without
        observation masks and without declared standard deviations the
        random draws equal those of earlier releases.
        ``_observed_columns`` documents the mask layouts, including those of
        ``MultipleCurveResolution``.

        Residual resampling assumes the errors in a pool are exchangeable.
        With declared standard deviations ``sigma`` [measured-state unit],
        which may differ from sample to sample, the raw residuals are not:
        a residual drawn from a precise sample and placed at an imprecise
        one would misstate that entry's error. The pool therefore holds the
        standardized residuals ``z = r / sigma`` [-] of the observed entries,
        and entry ``i`` of a generated dataset is
        ``y_fit[i] - sigma[i] * z_drawn`` [measured-state unit]. With one
        standard deviation per measurement this equals raw-residual
        resampling up to rounding. Without declared standard deviations the
        draws and generated values are unchanged.

        The generated errors follow the empirical residual scatter, like the
        default ``get_covariance(include_mse=True)`` (and
        ``scipy.optimize.curve_fit(..., absolute_sigma=False)``): the
        declared standard deviations set only the relative error size of
        each entry. Errors are not drawn with the declared absolute
        ``sigma``; that would require a parametric bootstrap, which is not
        provided.
        """
        # Per experiment, (n_times, n_measured) [measured-state units], NaN
        # where unobserved; None entries without declared uncertainties.
        std_runs = self.inst._residual_std_runs()

        y_boot = []
        for ind in range(self.inst.num_datasets):  # datasets
            # (n_measured, n_times) [measured-state units], model minus data
            signs = self.inst._residual_signs(ind)  # [-] or None
            residual = (self.residuals[ind] if signs is None
                        else self.residuals[ind] * signs).T
            y_nominal = self.y_nominal[ind].T
            observed_by_state = self._observed_columns(ind).T
            # (n_measured, n_times) [measured-state units] or None per state
            std_by_state = ([None] * len(residual) if std_runs[ind] is None
                            else std_runs[ind].T)

            y_states = []
            for res, y, observed, std in zip(residual, y_nominal,
                                             observed_by_state,
                                             std_by_state):  # states
                observed = observed[fix_initial:]
                # [measured-state unit], observed entries only
                resid = res[fix_initial:][observed]
                if std is None:
                    pool = resid  # [measured-state unit]
                else:
                    # [measured-state unit], observed entries only
                    scale = std[fix_initial:][observed]
                    pool = resid / scale  # [-], standardized residuals

                # [measured-state unit]; NaN where the state is unobserved
                y_generated = np.full((num_samples, len(observed)), np.nan)
                if pool.size > 0:
                    # Resampled observed residuals [measured-state unit], or
                    # standardized residuals [-] with declared uncertainties.
                    boots = np.random.choice(pool,
                                             size=(num_samples, len(pool)),
                                             replace=True)
                    if std is not None:
                        # [measured-state unit], rescaled to each entry
                        boots = boots * scale

                    y_generated[:, observed] = (
                        y[fix_initial:][observed] - boots)

                if fix_initial:
                    y_generated = np.insert(y_generated, 0, y[0], axis=1)

                y_states.append(y_generated)

            y_experiment = [np.column_stack(a) for a in list(zip(*y_states))]

            y_boot.append(y_experiment)

        return y_boot

    def bootstrap_params(self, num_samples: int = 100) -> np.ndarray:
        """Estimate parameters for residual-bootstrap response samples.

        Parameters
        ----------
        num_samples : int, optional
            Number of bootstrap datasets and optimization runs.

        Returns
        -------
        numpy.ndarray
            Estimated parameters with shape ``(num_samples, num_params)``:
            one row per bootstrap dataset in sampling order, columns in the
            order of the optimized parameters (``name_params``), in the
            physical units configured for each model parameter. A row is NaN
            when the optimizer encounters a singular linear system during the
            solve. A refit whose covariance is undefined, for example
            because its parameters are not identifiable, keeps its
            estimates as the row (see Notes).

        Warns
        -----
        RuntimeWarning
            If ``optimize_fn`` raises ``numpy.linalg.LinAlgError``; the warning
            includes the sample index and original diagnostic.
            Also if a sample's exception propagates and the model then fails
            at the accepted parameters; the sample's exception is still
            raised, also when warnings are turned into errors (the report is
            then dropped).
            Also, from ``optimize_fn``, if a sample refit stops without
            converging: an unsuccessful IPOPT termination, LM's
            ``'Maximum iterations exceeded'`` or an unsuccessful
            ``scipy.optimize.least_squares`` status; see Notes.
            Also, from ``ParameterEstimation.get_covariance``, if a sample
            refit's covariance is undefined, for example because its
            parameters are not identifiable.

        Raises
        ------
        Exception
            Any optimizer exception other than ``numpy.linalg.LinAlgError``,
            or a model error while restoring the accepted parameters after all
            samples were processed.

        Notes
        -----
        Each dataset is fitted on an isolated copy of the estimator (see
        ``_fit_bootstrap_sample``), so the estimator keeps its observations,
        accepted parameters, residuals, predictions, solver information and
        covariance, also when a sample fails or an exception propagates. The
        only state this method changes is ``boot_params`` and
        ``num_samples`` on this ``StatisticsClass``, after all samples are
        fitted. A stateful model callback, such as a ``SimulationExec`` unit
        operation, is shared with the copies; ``_restore_model_state`` returns
        it to the accepted parameters after the samples, also when a sample's
        exception propagates. A keyboard interrupt or other
        ``BaseException`` aborts at once without that evaluation.

        The package's Levenberg-Marquardt implementation uses
        ``numpy.linalg.solve`` for its steps; its final ``numpy.linalg.inv``
        and ``get_covariance`` return NaN instead of raising for a singular
        ``J @ J.T`` (#532), so a refit whose covariance is undefined keeps its
        estimates, which are all the bootstrap uses. The documented
        Exception type, rather than origin, defines recovery. Any
        ``LinAlgError`` escaping ``optimize_fn`` is recorded as a NaN row,
        including one raised inside a user model. Other programming and model
        errors propagate unchanged.

        This recovery is therefore scoped to ``opt_method='LM'``.
        ``ParameterEstimation.optimize_fn`` also accepts ``'IPOPT'``, whose
        numerical failures surface from cyipopt rather than as a
        ``LinAlgError``. Under that method a failed sample aborts the whole
        bootstrap run instead of recording a NaN row. A sample refit that
        stops without converging does not raise, whatever the method: an
        unsuccessful IPOPT termination (e.g. at ``max_iter``), LM at
        ``max_fun_eval`` (``'Maximum iterations exceeded'``) or an
        unsuccessful ``scipy.optimize.least_squares`` status (e.g. 0 at
        ``max_nfev``). ``optimize_fn`` emits its ``RuntimeWarning`` naming
        the stop reason, and the sample's last accepted iterate is stored as
        its row. When that warning is turned into an error, it propagates
        like any other sample exception (it is not a ``LinAlgError``) and
        aborts the run.
        """
        y_samples = self.get_bootsamples(
            num_samples)  # [response-dependent units]

        # Run parameter estimation
        boot_params = np.zeros(
            (num_samples, self.inst.num_params))  # [model-parameter units]

        tic = time.time()  # [s]
        try:
            for ind in range(num_samples):
                # Bootstrapped data for all the experimental runs
                y_sample = [y[ind] for y in y_samples]  # [response units]

                # Optimize
                try:
                    params = _fit_bootstrap_sample(
                        self.inst, y_sample)  # [model-parameter units]
                except np.linalg.LinAlgError as exc:
                    warnings.warn(
                        "Bootstrap optimization failed for sample {}: "
                        "{}".format(ind, exc),
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    params = [np.nan] * self.inst.num_params  # [param units]

                # Store
                boot_params[ind] = params
        except Exception as error:
            # Restore shared model state, but never mask the sample error.
            try:
                _restore_model_state(self.inst)
            except Exception as refresh_error:
                try:
                    warnings.warn(
                        "Could not restore the model at the accepted "
                        "parameters after bootstrap sample {} failed: "
                        "{}".format(ind, refresh_error),
                        RuntimeWarning,
                        stacklevel=2,
                    )
                except Exception:
                    # A warnings filter set to 'error' turns the report into
                    # an exception; the sample's error still takes priority.
                    pass
            raise error
        # Interrupts (BaseException) propagate without further model runs.

        _restore_model_state(self.inst)
        toc = time.time()  # [s]

        self.boot_params = boot_params  # [model-parameter units]
        self.num_samples = num_samples

        elapsed = toc - tic  # [s]

        print()
        print('Bootstrap time: {:.2f} s'.format(elapsed))
        print()

        return boot_params

    def plot_hist(self, fig_size=None):

        num_plots = self.inst.num_params
        num_cols = bool(num_plots // 2) + 1
        num_rows = num_plots // 2 + num_plots % 2

        if num_rows == 1:
            mult = 1.2
        else:
            mult = 1.0

        if fig_size is None:
            width = 5  # in
            height = width/num_cols/mult * num_rows
            fig_size = (width, height)

        fig, axes = plt.subplots(num_rows, num_cols, figsize=fig_size)
        for ind, ax in enumerate(axes.flatten()):
            ax.hist(self.boot_params.T[ind], bins=50)
            ax.set_xlabel(self.inst.param_names_plot[ind])
            ax.axvline(self.params[ind], linestyle='--', color='k')
            ylims = ax.get_ylim()

            ax.text(self.params[ind], ylims[1]*1.1,
                    '{:.3f}'.format(self.params[ind]), ha='center',
                    fontsize=8)

        fig.text(-0.02, 0.5,
                 r'Frequency (%i samples)' % self.num_samples,
                 rotation=90, va='center')
        fig.tight_layout()

        return fig, axes

    def confid_profiles(sens, hess_inv_params, num_times, variance_resid,
                        alpha, dof):

        sigma_resid = np.sqrt(variance_resid)
        sens_ord = []
        for ind in range(num_times):
            sens_time = sens[ind::num_times]
            sens_ord.append(sens_time)

        num_states = len(sens_ord[0])
        sigma_y = np.zeros((num_times, num_states))

        for ind in range(num_times):
            cov_y = (sens_ord[ind].dot(hess_inv_params)).dot(sens_ord[ind].T)
            sigma_y[ind] = np.sqrt(np.diag(cov_y)) * sigma_resid

        t_stat = stats.t.ppf((1 + alpha)/2, dof)

        sigma_y *= t_stat

        return sigma_y, sens_ord

    def plot_boots(self, num_samples, ind, fig_size=None):
        """Plot residual-bootstrap datasets of one experiment.

        Parameters
        ----------
        num_samples : int
            Number of bootstrap datasets drawn with ``get_bootsamples`` [-];
            at least 1.
        ind : int
            Experiment index in ``x_data`` order.
        fig_size : tuple of float, optional
            Figure size (width, height) [in].

        Returns
        -------
        fig : matplotlib.figure.Figure
            Figure with one subplot per measured column of experiment
            ``ind``; unused grid cells are removed.
        axes : numpy.ndarray of matplotlib.axes.Axes
            The subplots, shape ``(n_measured,)``, in measured-column
            order. Subplot ``col`` holds one line per dataset, in
            ``samples`` order: the observed entries of column ``col``
            against the experiment's model grid (``x_model[ind]``).
        samples : list of numpy.ndarray
            The plotted datasets: ``get_bootsamples(num_samples)[ind]``,
            ``num_samples`` arrays of shape ``(n_times, n_measured)`` in the
            measured states' units, NaN where unobserved.

        Raises
        ------
        ValueError
            If ``num_samples`` is not a positive integer or ``ind`` is not
            an experiment index.

        Notes
        -----
        Datasets are drawn from NumPy's global generator, as in
        ``get_bootsamples``; nothing on the estimator changes. Unobserved
        entries (staggered grids, missing values, measurements absent from
        the experiment) are omitted from the lines, so a column without
        observations shows an empty subplot.
        """
        num_samples = _validated_count(num_samples, 'num_samples')
        num_datasets = self.inst.num_datasets
        if (isinstance(ind, bool) or not isinstance(ind, (int, np.integer))
                or not 0 <= ind < num_datasets):
            raise ValueError(
                f"ind must be an experiment index from 0 to "
                f"{num_datasets - 1}; got {ind!r}")

        samples = self.get_bootsamples(num_samples)[ind]
        grid = self.inst.x_model[ind]  # [x units], model grid of the rows
        observed = self._observed_columns(ind)  # (n_times, n_measured)

        num_plots = samples[0].shape[1]
        num_cols = min(_BOOT_PLOT_COLUMNS, num_plots)
        num_rows = -(-num_plots // num_cols)  # ceiling division

        fig, axes = plt.subplots(num_rows, num_cols, figsize=fig_size,
                                 squeeze=False)
        axes_flat = axes.flatten()
        for col in range(num_plots):
            rows = observed[:, col]
            # (n_observed, num_samples) [measured-state unit]
            data = np.column_stack([sample[rows, col] for sample in samples])
            axes_flat[col].plot(grid[rows], data, '-o', mfc='None')
            axes_flat[col].set_title(
                f'{self.inst.experim_names[ind]}, column {col}')

        for axis in axes_flat[num_plots:]:
            fig.delaxes(axis)

        return fig, axes_flat[:num_plots], samples


# Finite choices of IdentifiabilityReport.noise_model; see its docstring.
_NOISE_MODELS = ('declared_uncertainty', 'weight_matrix', 'identity')
# Finite choices of IdentifiabilityReport.jacobian_source, with the
# description used in the noise assumption.
_JACOBIAN_SOURCES = {
    'model': 'sensitivities returned by the model callback',
    'jac_fun': 'the user-supplied jac_fun',
    'finite_difference': "the estimator's forward differences",
}


def _open_unit_fraction(value, name: str) -> float:
    """Return a real number strictly between 0 and 1, or raise.

    Parameters
    ----------
    value : float
        Candidate value [-].
    name : str
        Argument name for the error message.

    Returns
    -------
    float
        ``value`` as a Python ``float`` [-].

    Raises
    ------
    TypeError
        If ``value`` is not a real number (booleans and complex excluded).
    ValueError
        If ``value`` is not finite or not in the open interval ``(0, 1)``.
    """
    if (isinstance(value, bool) or not isinstance(value, numbers.Real)):
        raise TypeError(
            f"{name} must be a real number in (0, 1); got "
            f"{type(value).__name__}")
    number = float(value)  # [-]
    if not np.isfinite(number) or not 0.0 < number < 1.0:
        raise ValueError(
            f"{name} must be strictly between 0 and 1; got {value!r}")

    return number


def _per_parameter_fractions(value, name: str,
                             num_params: int) -> np.ndarray:
    """Return one fraction in ``(0, 1)`` per parameter, or raise.

    Parameters
    ----------
    value : float or array_like
        A real number for every parameter, or one per parameter, shape
        ``(num_params,)`` [-].
    name : str
        Argument name for the error message.
    num_params : int
        Number of parameters [-].

    Returns
    -------
    numpy.ndarray
        Shape ``(num_params,)`` [-], each entry finite in ``(0, 1)``.

    Raises
    ------
    TypeError
        If ``value`` is not real (booleans and complex excluded).
    ValueError
        If the shape is neither scalar nor ``(num_params,)`` or an entry is
        not finite or not strictly between 0 and 1.
    """
    if np.ndim(value) == 0:
        return np.full(num_params, _open_unit_fraction(value, name))
    array = np.asarray(value)
    if array.dtype.kind not in 'iuf':
        raise TypeError(f"{name} must hold real numbers in (0, 1); got "
                        f"dtype {array.dtype}")
    if array.shape != (num_params,):
        raise ValueError(f"{name} must be a scalar or have shape "
                         f"({num_params},), one entry per parameter; got "
                         f"{array.shape}")
    array = array.astype(float)  # [-]
    if not (np.isfinite(array).all() and (array > 0).all()
            and (array < 1).all()):
        raise ValueError(f"{name} entries must be strictly between 0 and "
                         f"1; got {array.tolist()}")

    return array


def _parameter_scale(estimates: np.ndarray, param_scale,
                     names: Sequence) -> np.ndarray:
    """Return the magnitude scale of each parameter, or raise.

    Parameters
    ----------
    estimates : numpy.ndarray
        Accepted parameters, shape ``(n,)``, each in its own unit.
    param_scale : float, array_like or None
        Positive finite scale of every parameter, or one per parameter,
        shape ``(n,)``, in the parameter units. None uses ``|estimates|``.
    names : sequence of str
        Parameter names, used in the messages.

    Returns
    -------
    numpy.ndarray
        Scale of each parameter, shape ``(n,)``, in its unit.

    Raises
    ------
    TypeError
        If ``param_scale`` is not real.
    ValueError
        If ``param_scale`` has the wrong shape or an entry that is not
        positive and finite, or, when it is None, an estimate is exactly
        zero or not finite: a magnitude is not invented for it.
    """
    num_params = len(estimates)  # [-]
    if param_scale is None:
        scale = np.abs(estimates)  # [p_i]
        bad = [name for name, value in zip(names, scale)
               if not (np.isfinite(value) and value > 0)]
        if bad:
            raise ValueError(
                "The parameter magnitude is undefined for the parameters at "
                f"zero or not finite ({', '.join(bad)}); pass param_scale, "
                "the typical magnitude of each parameter in its units")
        return scale
    if isinstance(param_scale, bool):
        raise TypeError("param_scale must be real; got bool")
    array = np.asarray(param_scale)
    if array.dtype.kind not in 'iuf':
        raise TypeError("param_scale must hold real numbers; got dtype "
                        f"{array.dtype}")
    if array.ndim == 0:
        array = np.full(num_params, float(array))
    if array.shape != (num_params,):
        raise ValueError(f"param_scale must be a scalar or have shape "
                         f"({num_params},); got {array.shape}")
    array = array.astype(float)  # [p_i]
    if not (np.isfinite(array).all() and (array > 0).all()):
        raise ValueError("param_scale entries must be positive and finite; "
                         f"got {array.tolist()}")

    return array


def _read_only(array) -> np.ndarray:
    """Return a read-only float copy of an array.

    Parameters
    ----------
    array : array_like
        Values in any unit; the unit is preserved.

    Returns
    -------
    numpy.ndarray
        Copy that cannot be written in place, so a frozen report cannot be
        changed through its arrays.
    """
    copied = np.array(array, dtype=float, copy=True)  # [unit of array]
    copied.setflags(write=False)
    return copied


def _read_only_bool(array) -> np.ndarray:
    """Return a read-only boolean copy of an array.

    Parameters
    ----------
    array : array_like
        Boolean flags.

    Returns
    -------
    numpy.ndarray
        Boolean copy that cannot be written in place.
    """
    copied = np.array(array, dtype=bool, copy=True)
    copied.setflags(write=False)
    return copied


def _read_only_mapping(value):
    """Return a deep read-only copy of a mapping, recursively.

    Parameters
    ----------
    value : object
        A mapping (copied into ``MappingProxyType``, values recursively) or
        any other value (deep-copied).

    Returns
    -------
    object
        The read-only copy.
    """
    if isinstance(value, Mapping):
        return MappingProxyType({key: _read_only_mapping(item)
                                 for key, item in value.items()})
    return copy.deepcopy(value)


def _plain(value):
    """Return a picklable copy of a report field value.

    Parameters
    ----------
    value : object
        Field value; read-only mappings become dicts, recursively.

    Returns
    -------
    object
        ``value`` with every ``Mapping`` replaced by a ``dict``.
    """
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    return value


def _scaled_norm(values) -> float:
    """Return the Euclidean norm of a vector without overflow or underflow.

    Parameters
    ----------
    values : array_like
        Finite values in one unit.

    Returns
    -------
    float
        ``||values||`` in that unit, computed as ``m * ||values / m||``
        with ``m`` the largest magnitude, so entries near the floating-point
        limits neither overflow nor underflow; 0 for an empty or all-zero
        vector.
    """
    array = np.ravel(np.asarray(values, dtype=float))  # [unit]
    largest = float(np.max(np.abs(array))) if array.size else 0.0  # [unit]
    if largest == 0 or not np.isfinite(largest):
        return largest
    return largest * float(np.linalg.norm(array / largest))


def _row_norms(jac: np.ndarray) -> np.ndarray:
    """Return the Euclidean norm of each row without overflow.

    Parameters
    ----------
    jac : numpy.ndarray
        Finite matrix, shape ``(n, m)``; row ``i`` in its own unit.

    Returns
    -------
    numpy.ndarray
        Norm of each row, shape ``(n,)``, in that row's unit; zero for zero
        rows.
    """
    largest = np.abs(jac).max(axis=1) if jac.shape[1] else np.zeros(
        jac.shape[0])  # [row unit]
    norms = np.zeros(jac.shape[0])  # [row unit]
    nonzero = largest > 0
    norms[nonzero] = largest[nonzero] * np.linalg.norm(
        jac[nonzero] / largest[nonzero, np.newaxis], axis=1)
    return norms


def _units_statement(measurement_units: Optional[Mapping]) -> str:
    """Describe the measurement units behind an identity weighting.

    Parameters
    ----------
    measurement_units : mapping or None
        Measurement column name -> declared unit (str) or None when that
        measurement declares none. None when the data layout records no
        measurement names (``MultipleCurveResolution``, nested
        ``'spectra'`` dictionaries).

    Returns
    -------
    str
        Sentence stating whether residuals in different units are summed
        and what that implies for the covariance.
    """
    caveat = ("the covariance depends on the unit chosen for each "
              "measurement and is not a physical covariance")
    if measurement_units is None:
        return (" The measurement units are not recorded for this data "
                "layout, so it cannot be checked whether residuals in "
                f"different units are summed; if they are, {caveat}.")
    names = list(measurement_units)
    declared = {name: unit for name, unit in measurement_units.items()
                if unit is not None}
    undeclared = [name for name in names if name not in declared]
    if len(names) == 1:
        unit = declared.get(names[0])
        label = f"[{unit}]" if unit is not None else "(units undeclared)"
        return f" One measured field, {names[0]} {label}."
    if len(set(declared.values())) > 1:
        listing = ', '.join(f"{name} [{unit}]"
                            for name, unit in declared.items())
        statement = (f" The measured fields have different declared units "
                     f"({listing}); identity weighting sums their residuals "
                     f"as plain numbers, so {caveat}. Declare measurement "
                     "uncertainties, or a weight_matrix of error variances, "
                     "for a dimensionless weighting.")
        if undeclared:
            statement += (" Undeclared units: "
                          f"{', '.join(undeclared)}.")
        return statement
    if undeclared:
        return (f" The units of {', '.join(undeclared)} are undeclared, so "
                "it cannot be checked whether residuals in different units "
                f"are summed; if they are, {caveat}.")
    unit = next(iter(declared.values()))
    return f" Every measured field is in [{unit}]."


def _format_per_parameter(values: np.ndarray, names: Sequence) -> str:
    """Write per-parameter values compactly.

    Parameters
    ----------
    values : numpy.ndarray
        One value per parameter, shape ``(n,)``.
    names : sequence of str
        Parameter names.

    Returns
    -------
    str
        The common value if all are equal, otherwise ``'name: value'``
        pairs.
    """
    if np.all(values == values[0]):
        return f"{values[0]:.3g}"
    return '(' + ', '.join(f"{name}: {value:.3g}"
                           for name, value in zip(names, values)) + ')'


def _noise_assumption(noise_model: str, include_mse: bool, mse: float,
                      measurement_units: Optional[Mapping],
                      jacobian_source: str, jacobian_rtol: np.ndarray,
                      names: Sequence, weight_matrix_is_identity: bool,
                      output_rtol: float) -> str:
    """Write the explicit noise and linearization assumptions of a report.

    Parameters
    ----------
    noise_model : {'declared_uncertainty', 'weight_matrix', 'identity'}
        Weighting of the fit (see ``IdentifiabilityReport``).
    include_mse : bool
        Whether the common variance scale is re-estimated from ``mse``.
    mse : float
        Residual mean square [weighted residual unit**2].
    measurement_units : mapping or None
        Measurement name -> declared unit or None (see
        ``_units_statement``); read only for ``'identity'``.
    jacobian_source : {'model', 'jac_fun', 'finite_difference'}
        Origin of the weighted residual Jacobian.
    jacobian_rtol : numpy.ndarray
        Row-relative accuracy [-] of the sensitivities, per parameter,
        shape ``(n,)``.
    names : sequence of str
        Parameter names.
    weight_matrix_is_identity : bool
        Whether a supplied ``weight_matrix`` is the identity; read only for
        ``'weight_matrix'``.
    output_rtol : float
        Relative accuracy [-] of the model outputs; stated only for
        finite-difference Jacobians.

    Returns
    -------
    str
        The assumptions, in sentences, built from the arguments.
    """
    check = (f"the residual mean square s2 = {mse:.4g} is reported as a "
             "check: its expected value is 1 under this assumption")
    if noise_model == 'declared_uncertainty':
        text = ("Measurement errors are assumed independent, zero-mean and "
                "Gaussian, with the standard deviations declared by the "
                "measurements (Measurement.uncertainty); the weighted "
                "residuals (model - data) / sigma are dimensionless [-].")
        if include_mse:
            text += (" The declared standard deviations are treated as "
                     "relative weights: their common scale is re-estimated "
                     f"from the residual mean square s2 = {mse:.4g} [-] "
                     "(reduced chi-square) and the covariance is "
                     "s2 * inv(J J^T), like scipy.optimize.curve_fit with "
                     "absolute_sigma=False.")
        else:
            text += (" The declared standard deviations are treated as "
                     "absolute: the covariance is inv(J J^T), like "
                     "scipy.optimize.curve_fit with absolute_sigma=True; "
                     f"{check}.")
    elif noise_model == 'weight_matrix':
        text = ("The measurement errors of the measured columns of a sample "
                "are assumed zero-mean and Gaussian, correlated with a "
                "covariance proportional to weight_matrix (its marginal "
                "block for partially observed samples), and independent "
                "between samples.")
        if weight_matrix_is_identity:
            text += (" The supplied weight_matrix is the identity: "
                     "independent errors with a standard deviation of 1 in "
                     "each measured unit.")
        if include_mse:
            text += (" weight_matrix is treated as relative: the common "
                     "scale is re-estimated from the residual mean square "
                     f"s2 = {mse:.4g} (dimensionless when weight_matrix "
                     "holds error variances in squared measurement units) "
                     "and the covariance is s2 * inv(J J^T).")
        else:
            text += (" weight_matrix is treated as the absolute "
                     "measurement-error covariance: the covariance is "
                     f"inv(J J^T); {check}.")
    else:
        text = ("No measurement uncertainty or weight_matrix was given: "
                "the measurement errors are assumed independent, zero-mean "
                "and Gaussian, with one common variance for every observed "
                "entry in its measurement unit.")
        if include_mse:
            text += (" That variance is estimated by the residual mean "
                     f"square s2 = {mse:.4g} [measurement unit**2] and the "
                     "covariance is s2 * inv(J J^T).")
        else:
            text += (" include_mse=False takes that variance as 1 in the "
                     "measurement units: the covariance inv(J J^T) is "
                     "meaningful only if the errors have unit variance in "
                     f"those units; {check}.")
        text += _units_statement(measurement_units)

    text += (" This is a local linearized (Gauss-Newton) analysis at the "
             "accepted parameters: it assumes a correct model, an interior "
             "optimum (no active parameter bounds) and weights that do not "
             "depend on the parameters. The Jacobian comes from "
             f"{_JACOBIAN_SOURCES[jacobian_source]}; each sensitivity row is "
             "taken as accurate to a relative jacobian_rtol = "
             f"{_format_per_parameter(jacobian_rtol, names)} of itself"
             + (", plus the difference of two model outputs, each accurate "
                f"to a relative output_rtol = {output_rtol:.3g}, over the "
                "forward-difference step"
                if jacobian_source == 'finite_difference' else "")
             + " (see sensitivity_error_bounds).")
    return text


class ParameterDirection(NamedTuple):
    """A direction in the Jacobi-scaled parameter space of a fit.

    Attributes
    ----------
    parameter_names : tuple of str
        Parameter names, in the order of ``components``.
    components : tuple of float
        Components [-] of a unit vector. They refer to the scaled
        parameters ``u_i = dp_i * sqrt(A_ii)`` with ``A = J J^T``: each
        parameter change in units of that parameter's standard error when
        the others are held fixed (see ``_information_inverse``). The sign
        is fixed so that the first component that does not round to zero
        at three decimals is positive.
    singular_value : float
        Singular value [-] of the Jacobi-scaled Jacobian ``D J`` for this
        direction: the norm of the change of the weighted residuals,
        relative to the per-parameter norms, along the direction.
    """

    parameter_names: tuple
    components: tuple
    singular_value: float

    def as_dict(self) -> dict:
        """Return the components by parameter name.

        Returns
        -------
        dict
            Parameter name -> component [-], in parameter order.
        """
        return dict(zip(self.parameter_names, self.components))


@dataclasses.dataclass(frozen=True, eq=False)
class IdentifiabilityReport:
    """Parameter uncertainty and practical identifiability of a fit.

    Built by ``ParameterEstimation.get_identifiability`` from an accepted
    fit only: the weighted residual Jacobian ``J = info_opt['jac']``, the
    weighted residuals ``r = info_opt['fun']``, a bound on the whitened
    output errors, the finite-difference steps, the accepted parameters
    and the number of observed entries. Nothing is refitted and the
    estimator is not modified. All quantities are local and linearized
    (Gauss-Newton) at the accepted parameters; ``noise_assumption`` states
    the assumptions that give them a statistical meaning. The report is
    immutable (its arrays are read-only, its mappings read-only views) and
    can be pickled and deep copied.

    ``information_matrix`` and the sensitivity quantities
    (``singular_values``, ``unresolved_directions``) are the documented
    hand-off to model-based experiment design (issue #355).

    Attributes
    ----------
    parameter_names : tuple of str
        Optimized parameter names (unique), in ``name_params`` order; the
        order of every per-parameter array and of the matrix rows and
        columns.
    estimates : numpy.ndarray
        Accepted parameters, shape ``(n,)``, each in its own unit.
    noise_model : {'declared_uncertainty', 'weight_matrix', 'identity'}
        Weighting of the fit: measurements declared uncertainties (residuals
        divided by their standard deviations), a supplied ``weight_matrix``
        (correlated measurement-error covariance, possibly the identity),
        or no weighting.
    noise_assumption : str
        Explicit statement, built from the estimator state, of the assumed
        measurement-error model, whether the declared uncertainties are
        relative or absolute (``include_mse``), the unit caveat of an
        identity weighting, and the linearization assumptions.
    variance_scale_estimated : bool
        ``include_mse``: True if the common variance scale is re-estimated
        from ``residual_mean_square`` (relative weights).
    residual_mean_square : float
        ``s2 = r @ r / degrees_of_freedom`` [weighted residual unit**2]:
        [-] (reduced chi-square) with declared uncertainties, or with a
        ``weight_matrix`` of error variances in squared measurement units;
        the measurement unit squared with identity weighting of one unit.
    degrees_of_freedom : int
        ``num_data_total - num_params`` [-], counting observed entries only.
    information_matrix : numpy.ndarray
        ``J @ J.T`` of the weighted Jacobian, shape ``(n, n)``; entry
        ``(i, j)`` in [weighted residual unit**2 / (p_i * p_j)]. Under the
        noise assumption it is the Gauss-Newton (Fisher) information
        multiplied by the error variance scale (``residual_mean_square``
        when ``variance_scale_estimated``, 1 for absolute weights);
        ``fisher_information`` removes that scale.
    covariance : numpy.ndarray
        Linearized parameter covariance, shape ``(n, n)``:
        ``s2 * inv(J J^T)`` [p_i * p_j] if ``variance_scale_estimated``,
        else ``inv(J J^T)`` [p_i * p_j / weighted residual unit**2], which
        is [p_i * p_j] for dimensionless weighted residuals. Bit for bit
        ``ParameterEstimation.get_covariance(include_mse)`` when that is
        finite; all NaN when it is undefined, and also, unlike
        ``get_covariance``, when ``s2`` is zero or not finite with
        ``variance_scale_estimated`` or the result is not finite (see
        ``covariance_diagnostic``).
    covariance_diagnostic : str or None
        None when the covariance is available. Otherwise why it is not:
        the message ``get_covariance`` warns with (undefined covariance),
        or that the variance scale cannot be estimated (``s2`` zero or not
        finite), or that the covariance overflows.
    standard_errors : numpy.ndarray
        ``sqrt(diag(covariance))``, shape ``(n,)``, in the unit of each
        parameter (divided by the weighted residual unit for the absolute
        form with unit-bearing residuals). NaN when unavailable.
    relative_standard_errors : numpy.ndarray
        ``standard_errors / |estimates|`` [-], shape ``(n,)``; NaN where
        the estimate is exactly zero or the covariance is unavailable.
    correlation : numpy.ndarray
        Parameter correlation matrix [-], shape ``(n, n)``; NaN when the
        covariance is unavailable.
    alpha : float
        Confidence level [-] of ``confidence_intervals`` and
        ``significant``.
    confidence_intervals : numpy.ndarray
        Two-sided Student-t intervals, shape ``(n, 2)`` (lower, upper) in
        each parameter's unit: ``estimates -/+ t * standard_errors`` with
        ``t`` the ``(1 + alpha) / 2`` quantile of Student's t with
        ``degrees_of_freedom``, as ``get_confidence_intervals``. NaN when
        the covariance is unavailable and for parameters at an active
        bound.
    t_ratios : numpy.ndarray
        ``|estimates| / standard_errors`` [-], shape ``(n,)``; NaN when
        unavailable and for parameters at an active bound.
    significant : numpy.ndarray
        Boolean, shape ``(n,)``: ``t_ratios`` above that Student-t quantile,
        i.e. the estimate is distinguishable from zero at confidence
        ``alpha``. This is a significance test against zero, not an
        identifiability test, and it is not meaningful for parameters whose
        zero is not a natural reference (for example log-transformed
        parameters). False where ``t_ratios`` is NaN.
    active_bounds : tuple of str or None
        Parameters at an active bound (``info_opt['active_mask']`` of
        ``scipy.optimize.least_squares``), for which the linearized
        intervals do not hold; empty when none is active; None when the
        optimizer does not report it (Levenberg-Marquardt ignores bounds,
        #533; IPOPT fits store no active set).
    jacobian_source : {'model', 'jac_fun', 'finite_difference'}
        Origin of ``J``: sensitivities returned by the model callback, the
        user's ``jac_fun``, or the estimator's forward differences.
    jacobian_rtol : numpy.ndarray
        Row-relative accuracy [-] of the sensitivities, shape ``(n,)``:
        ``||dJ_i|| <= jacobian_rtol_i * ||J_i||`` (see Notes and
        ``ParameterEstimation.get_identifiability``).
    output_rtol : float
        Relative accuracy [-] of the model outputs, entering the bounds
        only for finite-difference Jacobians (rounding term, see Notes).
    output_error_norm : float
        Bound [weighted residual unit] on the norm of the whitened output
        errors when each raw output ``y_j`` is accurate to ``output_rtol *
        |y_j|``: ``output_rtol * ||W |y|||`` (exact) for diagonal weights,
        ``output_rtol * ||W||_2 * ||y||`` for a correlated
        ``weight_matrix`` (see ``ParameterEstimation._output_error_scale``).
        NaN when not supplied (Jacobians other than finite differences).
    finite_difference_steps : numpy.ndarray or None
        Forward-difference step ``h_i`` of each parameter, shape ``(n,)``,
        in its unit; None unless ``jacobian_source`` is
        ``'finite_difference'``.
    sensitivity_error_bounds : numpy.ndarray
        Error bound ``||dJ_i|| / ||J_i||`` [-] of each row of the
        Jacobi-scaled Jacobian ``D J``, shape ``(n,)``: ``jacobian_rtol_i``,
        plus ``2 * output_error_norm / (h_i * ||J_i||)`` for finite
        differences; NaN for zero-information rows. A bound of at least 1
        makes the parameter unresolved (see ``unresolved_parameters``).
    resolution_floor : float
        ``sqrt(sum(sensitivity_error_bounds**2))`` [-] over the resolved
        rows (neither zero-information nor unresolved parameters): a bound
        on ``||D E||_2`` for the Jacobian error ``E`` of those rows;
        singular values at or below it are not resolved. 0 when no row
        remains, NaN when ``J`` is not finite.
    singular_values : numpy.ndarray
        Shape ``(n,)`` [-]: the singular values of ``D J``, ``D =
        diag(A_ii**-0.5)`` (unit row norms, the scaling of the #532
        singularity test), restricted to the resolved rows, in descending
        order; then NaN for each of ``unresolved_parameters`` (their rows
        are indistinguishable from zero, so no singular value is assigned)
        and 0 for each of ``zero_information``. All NaN when ``J`` is not
        finite.
    condition_number : float
        ``(s_max / s_min)**2 = cond(D J J^T D)`` [-] of the resolved rows;
        infinite when ``s_min = 0`` or any parameter is unresolved or has
        zero information, NaN when ``J`` is not finite.
    collinearity_index : float
        ``1 / s_min`` [-] of the resolved rows, the collinearity index of
        Brun, Reichert and Kuensch (2001); infinite in the same cases as
        ``condition_number``, NaN when ``J`` is not finite. No flagging
        threshold is applied.
    unresolved_directions : tuple of ParameterDirection
        Directions of the resolved rows of ``D J`` whose singular value
        ``s`` satisfies ``s <= resolution_floor``, or ``s**2 <= eps *
        s_max**2`` (the #532 criterion that makes the covariance
        undefined), in descending order of ``s``; then the unit vector of
        each unresolved parameter (``s`` NaN) and of each zero-information
        parameter (``s = 0``). Empty when ``J`` is not finite.
    weakest_direction : ParameterDirection or None
        The last direction in the order of ``singular_values``: that of the
        smallest resolved singular value, or of the last unresolved or
        zero-information parameter when there is one. Reported even when
        the parameters are identifiable; None when ``J`` is not finite.
    unresolved_parameters : tuple of str
        Parameters with a nonzero row of ``J`` whose error bound is at
        least the row's norm (``sensitivity_error_bounds >= 1``): under the
        declared accuracy the row is consistent with a zero row, so the
        parameter is not resolved by the data and is set aside from the
        direction analysis of the others.
    zero_information : tuple of str
        Parameters whose row of ``J`` is exactly zero: they change no
        observed weighted residual.
    identifiable : bool
        True only if ``J @ J.T`` passes the #532 test (the covariance is
        not undefined) and ``unresolved_directions`` is empty (so also no
        unresolved or zero-information parameter). It does not
        depend on ``s2``: with an unestimable variance scale the parameters
        may be identifiable while their uncertainty is unavailable. When
        False and the covariance is defined (only the resolution floor
        fired), the standard errors and intervals are kept but are not
        precise along the unresolved directions.
    convergence : collections.abc.Mapping or None
        Read-only ``{'success', 'message', 'tolerances'}`` copied from
        ``info_opt`` (None entries where absent; ``'tolerances'`` is a
        read-only copy), or None when the optimizer reported no
        ``'success'``.

    Notes
    -----
    Sensitivity error model, in the row-normalized coordinates of ``D J``
    (row ``i`` divided by ``||J_i||``)::

        e_i = jacobian_rtol_i
              + [finite differences] 2 * output_error_norm / (h_i * ||J_i||)

    The first term is the accuracy of the sensitivity row relative to
    itself: machine epsilon for model-returned sensitivities and
    ``jac_fun`` by default (an ODE solver's sensitivity ``rtol`` when
    passed), and for finite differences the truncation error ``(h_i / 2)
    |d2y/dp_i2|`` of Nocedal and Wright (*Numerical Optimization*, 2nd
    ed., 2006, Sec. 8.1, eq. 8.5, first term), at most about the relative
    step ``h_i / scale_i`` under the well-scaled-curvature assumption
    ``scale_i |d2y/dp_i2| <~ |dy/dp_i|``. It does not depend on additive
    offsets of the outputs. The second term is the rounding term of the
    same equation, ``2 u L_f / h``: the forward difference subtracts two
    outputs, at ``p`` and ``p + h_i``, each accurate to ``output_rtol *
    |y|`` (whitened as in ``output_error_norm``), so the difference may be
    wrong by twice that, divided by the step ``h_i``. Row
    scaling amplifies the errors of small rows, which a normwise bound on
    ``J`` would miss.

    Unresolved parameters: a row with ``e_i >= ||J_i||`` may be entirely
    error; a zero row is consistent with it under the declared accuracy.
    Such a parameter is reported as unresolved and its row is set aside,
    like a zero-information row, so that its error does not enter the
    bound of the other rows.

    Singular-value criterion: by Weyl's inequality for singular values
    (Golub and Van Loan, *Matrix Computations*, 4th ed., 2013, Cor. 8.6.2),
    a perturbation ``D E`` of the resolved rows changes every singular
    value of their ``D J`` by at most ``||D E||_2 <= ||D E||_F =
    resolution_floor``. A singular value at or below that floor is
    therefore not distinguishable from zero, so the data, seen through
    this Jacobian, do not resolve its direction. The bound is still
    conservative: the Frobenius norm of all resolved rows bounds every
    direction.

    The collinearity index is that of Brun, R., Reichert, P. and Kuensch,
    H. R. (2001), "Practical identifiability analysis of large
    environmental simulation models", Water Resources Research 37(4),
    1015-1030, doi:10.1029/2000WR900350: ``1 / sqrt(lambda_min)`` of the
    cross-product of the sensitivity vectors normalized to unit length.
    The parameters' sensitivity vectors are the rows of ``D J``, so it is
    ``1 / s_min``. It is reported for comparison only; no threshold is
    adopted.

    The Jacobi scaling makes ``singular_values``, ``condition_number``,
    ``collinearity_index``, ``sensitivity_error_bounds``,
    ``resolution_floor`` and the directions independent of the units
    chosen for the parameters (the products ``h_i * ||J_i||`` and
    ``h_i / scale_i`` are unit-free).
    """

    parameter_names: tuple
    estimates: np.ndarray
    noise_model: str
    noise_assumption: str
    variance_scale_estimated: bool
    residual_mean_square: float
    degrees_of_freedom: int
    information_matrix: np.ndarray
    covariance: np.ndarray
    covariance_diagnostic: Optional[str]
    standard_errors: np.ndarray
    relative_standard_errors: np.ndarray
    correlation: np.ndarray
    alpha: float
    confidence_intervals: np.ndarray
    t_ratios: np.ndarray
    significant: np.ndarray
    active_bounds: Optional[tuple]
    jacobian_source: str
    jacobian_rtol: np.ndarray
    output_rtol: float
    output_error_norm: float
    finite_difference_steps: Optional[np.ndarray]
    sensitivity_error_bounds: np.ndarray
    resolution_floor: float
    singular_values: np.ndarray
    condition_number: float
    collinearity_index: float
    unresolved_directions: tuple
    weakest_direction: Optional[ParameterDirection]
    unresolved_parameters: tuple
    zero_information: tuple
    identifiable: bool
    convergence: Optional[Mapping]

    #: Float array fields, frozen as read-only copies by __post_init__.
    _ARRAY_FIELDS = ('estimates', 'information_matrix',
                     'covariance', 'standard_errors',
                     'relative_standard_errors', 'correlation',
                     'confidence_intervals', 't_ratios', 'jacobian_rtol',
                     'sensitivity_error_bounds', 'singular_values')

    def __post_init__(self) -> None:
        """Freeze the array and mapping fields.

        Returns
        -------
        None

        Notes
        -----
        Arrays are replaced by read-only copies and ``convergence`` by a
        deep read-only copy, also after unpickling (see ``__reduce__``).
        """
        for name in self._ARRAY_FIELDS:
            object.__setattr__(self, name, _read_only(getattr(self, name)))
        object.__setattr__(self, 'significant',
                           _read_only_bool(self.significant))
        if self.finite_difference_steps is not None:
            object.__setattr__(self, 'finite_difference_steps',
                               _read_only(self.finite_difference_steps))
        object.__setattr__(self, 'parameter_names',
                           tuple(self.parameter_names))
        if self.convergence is not None:
            object.__setattr__(self, 'convergence',
                               _read_only_mapping(self.convergence))

    def __reduce__(self) -> tuple:
        """Support pickling and deep copying.

        Returns
        -------
        tuple
            ``(IdentifiabilityReport, field values)`` with read-only
            mappings as dicts; the constructor freezes them again.
        """
        return (type(self), tuple(_plain(getattr(self, field.name))
                                  for field in dataclasses.fields(self)))

    @classmethod
    def from_fit(cls, jac: npt.ArrayLike, residuals: npt.ArrayLike,
                 estimates: npt.ArrayLike, parameter_names: Sequence[str],
                 num_data_total: int, *, noise_model: str,
                 measurement_units: Optional[Mapping[str, Optional[str]]],
                 jacobian_source: str,
                 jacobian_rtol: Union[float, npt.ArrayLike],
                 output_rtol: float = DEFAULT_OUTPUT_RTOL,
                 output_error_scale: Optional[float] = None,
                 finite_difference_steps: Optional[npt.ArrayLike] = None,
                 alpha: float = 0.95, include_mse: bool = True,
                 active_bounds: Optional[Sequence[str]] = None,
                 weight_matrix_is_identity: bool = False,
                 convergence: Optional[Mapping] = None
                 ) -> 'IdentifiabilityReport':
        """Build the report from the quantities of an accepted fit.

        Parameters
        ----------
        jac : array_like
            Weighted residual Jacobian ``J``, shape ``(n, n_entries)``;
            entry ``(i, k)`` in [weighted residual k unit / parameter i
            unit]. Unobserved entries are zero columns.
        residuals : array_like
            Weighted residuals at the accepted parameters, shape
            ``(n_entries,)`` [weighted residual unit].
        estimates : array_like
            Accepted parameters, shape ``(n,)``, each in its own unit.
        parameter_names : sequence of str
            Unique parameter names in the row order of ``J``.
        num_data_total : int
            Number of observed entries [-].
        noise_model : {'declared_uncertainty', 'weight_matrix', 'identity'}
            Weighting of the fit.
        measurement_units : mapping or None
            Measurement column name -> declared unit or None; None when the
            data layout records no measurement names.
        jacobian_source : {'model', 'jac_fun', 'finite_difference'}
            Origin of ``J``.
        jacobian_rtol : float or array_like
            Row-relative accuracy [-] of the sensitivities, in ``(0, 1)``:
            one value for every parameter or one per parameter, shape
            ``(n,)``.
        output_rtol : float, optional
            Relative accuracy [-] of the model outputs, in ``(0, 1)``; read
            for finite-difference Jacobians. The default is
            ``jac_module.DEFAULT_OUTPUT_RTOL``.
        output_error_scale : float, optional
            Bound [weighted residual unit] on the whitened output errors per
            unit relative output error (``ParameterEstimation.
            _output_error_scale``), finite and non-negative. Required for
            finite-difference Jacobians; the default None stores NaN.
        finite_difference_steps : array_like, optional
            Forward-difference step of each parameter, shape ``(n,)``,
            positive and finite, in the parameter units. Required for
            finite-difference Jacobians, ignored otherwise.
        alpha : float, optional
            Confidence level [-] in ``(0, 1)``. The default is 0.95.
        include_mse : bool, optional
            If True (the default), the covariance is scaled by the residual
            mean square (relative weights); if False the weights are
            absolute.
        active_bounds : sequence of str, optional
            Names of the parameters at an active bound; the default None
            means not reported.
        weight_matrix_is_identity : bool, optional
            Whether the supplied ``weight_matrix`` is the identity (read
            for ``noise_model='weight_matrix'``). The default is False.
        convergence : mapping, optional
            ``{'success', 'message', 'tolerances'}`` of the optimizer; the
            default None means not reported.

        Returns
        -------
        IdentifiabilityReport
            The immutable report.

        Raises
        ------
        TypeError
            If ``alpha``, ``jacobian_rtol`` or ``output_rtol`` is not real,
            or ``include_mse`` is not a bool.
        ValueError
            If ``alpha``, ``output_rtol`` or an entry of ``jacobian_rtol``
            is not finite and strictly between 0 and 1; ``noise_model`` or
            ``jacobian_source`` is not one of its choices; there are no
            parameters, or duplicate or unknown (``active_bounds``)
            parameter names; the shapes of ``jac``, ``residuals``,
            ``estimates``, ``parameter_names``, ``jacobian_rtol`` or
            ``finite_difference_steps`` disagree; a finite-difference
            Jacobian lacks positive finite steps or a finite non-negative
            ``output_error_scale``; or ``num_data_total`` does not exceed
            the number of parameters.
        """
        alpha = _open_unit_fraction(alpha, 'alpha')  # [-]
        output_rtol = _open_unit_fraction(output_rtol, 'output_rtol')  # [-]
        if not isinstance(include_mse, bool):
            raise TypeError("include_mse must be a bool; got "
                            f"{type(include_mse).__name__}")
        if noise_model not in _NOISE_MODELS:
            raise ValueError(f"noise_model must be one of {_NOISE_MODELS}; "
                             f"got {noise_model!r}")
        if jacobian_source not in _JACOBIAN_SOURCES:
            raise ValueError(
                f"jacobian_source must be one of {tuple(_JACOBIAN_SOURCES)};"
                f" got {jacobian_source!r}")

        jac = np.asarray(jac, dtype=float)  # [r_k/p_i], shape (n, m)
        residuals = np.asarray(residuals, dtype=float)  # [r], shape (m,)
        estimates = np.asarray(estimates, dtype=float)  # [p_i], shape (n,)
        names = tuple(parameter_names)
        num_params = len(names)  # [-]
        if num_params == 0:
            raise ValueError("The fit has no optimized parameters")
        duplicates = sorted({name for name in names
                             if names.count(name) > 1})
        if duplicates:
            raise ValueError(
                "Parameter names must be unique to label directions and "
                f"table rows; repeated: {', '.join(map(str, duplicates))}")
        if (jac.ndim != 2 or jac.shape[0] != num_params
                or estimates.shape != (num_params,)
                or residuals.shape != (jac.shape[1],)):
            raise ValueError(
                "jac must have shape (n, n_entries), residuals "
                "(n_entries,) and estimates (n,) for the n = "
                f"{num_params} parameters; got {jac.shape}, "
                f"{residuals.shape} and {estimates.shape}")
        dof = int(num_data_total) - num_params  # [-]
        if dof <= 0:
            raise ValueError(
                "The identifiability report requires more observed entries "
                f"than optimized parameters (num_data_total="
                f"{int(num_data_total)}, num_params={num_params})")
        rtol = _per_parameter_fractions(jacobian_rtol, 'jacobian_rtol',
                                        num_params)  # [-]
        output_error_norm = np.nan  # [r], not supplied
        if output_error_scale is not None:
            error_scale = float(output_error_scale)  # [r]
            if not (np.isfinite(error_scale) and error_scale >= 0):
                raise ValueError("output_error_scale must be finite and "
                                 f"non-negative; got {output_error_scale!r}")
            output_error_norm = output_rtol * error_scale  # [r]
        steps = None  # [p_i]
        if jacobian_source == 'finite_difference':
            if finite_difference_steps is None or output_error_scale is None:
                raise ValueError(
                    "A finite-difference Jacobian needs "
                    "finite_difference_steps and output_error_scale for its "
                    "rounding error bound")
            steps = np.abs(np.asarray(finite_difference_steps,
                                      dtype=float))  # [p_i]
            if steps.shape != (num_params,) or not (
                    np.isfinite(steps).all() and (steps > 0).all()):
                raise ValueError(
                    "finite_difference_steps must hold one positive finite "
                    f"step per parameter, shape ({num_params},); got "
                    f"{np.asarray(finite_difference_steps).tolist()}")
        if active_bounds is not None:
            unknown = [name for name in active_bounds if name not in names]
            if unknown:
                raise ValueError("active_bounds names unknown parameters: "
                                 f"{', '.join(map(str, unknown))}")
            active_bounds = tuple(name for name in names
                                  if name in set(active_bounds))

        # Overflow, underflow, zero or non-finite s2 are handled below; no
        # stray NumPy warnings.
        with np.errstate(all='ignore'):
            # [p_i*p_j] or [p_i*p_j/r**2]; [-]; [r**2]
            covariance, correlation, mse, diagnostic = (
                _linearized_covariance(jac, residuals, dof, list(names),
                                       include_mse))
            information = np.dot(jac, jac.T)  # [r**2/(p_i*p_j)]
        undefined = diagnostic is not None  # #532: J J^T not invertible
        if not undefined:
            if include_mse and not (np.isfinite(mse) and mse > 0):
                diagnostic = (
                    "The variance scale cannot be estimated: the residual "
                    f"mean square s2 = {float(mse)!r} is zero or not finite "
                    "(for example noise-free synthetic data, or residuals "
                    "that overflow), so the covariance, standard errors, "
                    "intervals and t ratios are not available. Declare the "
                    "absolute measurement uncertainties (Measurement."
                    "uncertainty or weight_matrix) and pass "
                    "include_mse=False, or check the residuals")
            elif not np.isfinite(covariance).all():
                diagnostic = (
                    "The linearized covariance is not finite (overflow in "
                    "forming it), so the covariance, standard errors, "
                    "intervals and t ratios are not available. Rescale the "
                    "parameters or the measurements to moderate units")
            elif not (np.diag(covariance) > 0).all():
                diagnostic = (
                    "The linearized covariance has variances that are not "
                    "positive (underflow in forming it, e.g. sensitivities "
                    "near the floating-point limits), so the covariance, "
                    "standard errors, intervals and t ratios are not "
                    "available. Rescale the parameters or the measurements "
                    "to moderate units")
            if diagnostic is not None:
                covariance = np.full((num_params, num_params), np.nan)
                correlation = np.full((num_params, num_params), np.nan)

        standard_errors = np.sqrt(np.diag(covariance))  # [p_i]
        # Same expression as StatisticsClass.get_intervals.
        t_quantile = stats.t.ppf((1 + alpha)/2, dof)  # [-]
        half_width = standard_errors * t_quantile  # [p_i]
        intervals = np.column_stack((estimates - half_width,
                                     estimates + half_width))  # [p_i]
        relative = np.full(num_params, np.nan)  # [-]
        nonzero = estimates != 0
        relative[nonzero] = (standard_errors[nonzero]
                             / np.abs(estimates[nonzero]))
        with np.errstate(divide='ignore', invalid='ignore'):
            t_ratios = np.abs(estimates) / standard_errors  # [-]
        if active_bounds:
            # The linearized intervals assume an interior optimum.
            at_bound = np.array([name in active_bounds for name in names])
            intervals[at_bound] = np.nan
            t_ratios[at_bound] = np.nan
        significant = t_ratios > t_quantile  # False where NaN

        sensitivity = _scaled_sensitivity(jac, names, rtol,
                                          output_error_norm, steps)

        return cls(
            parameter_names=names,
            estimates=estimates,
            noise_model=noise_model,
            noise_assumption=_noise_assumption(
                noise_model, include_mse, mse, measurement_units,
                jacobian_source, rtol, names, weight_matrix_is_identity,
                output_rtol),
            variance_scale_estimated=include_mse,
            residual_mean_square=float(mse),
            degrees_of_freedom=dof,
            information_matrix=information,
            covariance=covariance,
            covariance_diagnostic=diagnostic,
            standard_errors=standard_errors,
            relative_standard_errors=relative,
            correlation=correlation,
            alpha=alpha,
            confidence_intervals=intervals,
            t_ratios=t_ratios,
            significant=significant,
            active_bounds=active_bounds,
            jacobian_source=jacobian_source,
            jacobian_rtol=rtol,
            output_rtol=output_rtol,
            output_error_norm=float(output_error_norm),
            finite_difference_steps=steps,
            sensitivity_error_bounds=sensitivity['error_bounds'],
            resolution_floor=sensitivity['floor'],
            singular_values=sensitivity['singular_values'],
            condition_number=sensitivity['condition_number'],
            collinearity_index=sensitivity['collinearity_index'],
            unresolved_directions=sensitivity['unresolved'],
            weakest_direction=sensitivity['weakest'],
            unresolved_parameters=sensitivity['unresolved_parameters'],
            zero_information=sensitivity['zero_information'],
            identifiable=not undefined and not sensitivity['unresolved'],
            convergence=None if convergence is None else {
                'success': convergence.get('success'),
                'message': convergence.get('message'),
                'tolerances': convergence.get('tolerances')},
        )

    @property
    def fisher_information(self) -> np.ndarray:
        """Return the Fisher information under the noise assumption.

        Returns
        -------
        numpy.ndarray
            ``information_matrix / residual_mean_square`` when
            ``variance_scale_estimated`` (the error variance scale is
            ``s2``), all NaN if ``s2`` is zero or not finite; else
            ``information_matrix`` itself (absolute weights). Shape ``(n,
            n)``, entry ``(i, j)`` in [1 / (p_i * p_j)] for dimensionless
            weighted residuals or the estimated-scale form. When the
            covariance is available it equals ``inv(covariance)`` up to
            rounding. A new read-only array.
        """
        if not self.variance_scale_estimated:
            return _read_only(self.information_matrix)
        scale = self.residual_mean_square  # [r**2]
        if not (np.isfinite(scale) and scale > 0):
            return _read_only(np.full(self.information_matrix.shape,
                                      np.nan))
        with np.errstate(over='ignore'):
            return _read_only(self.information_matrix / scale)

    def summary(self) -> str:
        """Return a human-readable account of the report.

        Returns
        -------
        str
            Lines in this order: a non-convergence warning first when the
            fit did not converge; the noise assumption; the convergence
            status; parameters at active bounds; the identifiability verdict
            with the unavailable-covariance diagnostic, the unresolved
            directions and the zero-information parameters; the weakest
            direction with the collinearity index, condition number and
            resolution floor; then a parameter table (estimate, standard
            error, relative standard error, interval, t ratio,
            significance).
        """
        lines = []
        success = (None if self.convergence is None
                   else self.convergence['success'])
        message = (None if self.convergence is None
                   else self.convergence['message'])
        if success is False:
            lines.append(
                f"FIT DID NOT CONVERGE ({message}): every quantity below "
                "describes the last accepted iterate, not a converged "
                "estimate.")
        lines.append(f"Noise assumption: {self.noise_assumption}")
        if success is None:
            lines.append("Convergence: not reported by the optimizer.")
        else:
            state = 'converged' if success else 'NOT converged'
            lines.append(f"Convergence: {state} ({message}).")
        if self.active_bounds:
            lines.append(
                "ACTIVE BOUNDS: "
                f"{', '.join(self.active_bounds)} at a bound. The "
                "linearized analysis assumes an interior optimum, so their "
                "intervals and significance are not reported (NaN / "
                "False) and their standard errors are not valid.")

        names = list(self.parameter_names)
        floor = (f"resolution floor {self.resolution_floor:.3g}"
                 if np.isfinite(self.resolution_floor)
                 else "resolution floor unavailable")
        if self.identifiable:
            lines.append(
                "Identifiability: the parameters are locally identifiable "
                f"from the observed data ({floor}).")
            if self.covariance_diagnostic is not None:
                lines.append("  Uncertainty unavailable: "
                             f"{self.covariance_diagnostic}")
        else:
            lines.append(
                "Identifiability: NOT IDENTIFIABLE. The standard errors and "
                "intervals below are not precise estimates of the "
                "parameter uncertainty.")
            if self.covariance_diagnostic is not None:
                lines.append(f"  Covariance unavailable: "
                             f"{self.covariance_diagnostic}")
            for direction in self.unresolved_directions:
                vector = np.array(direction.components)  # [-]
                if np.isnan(direction.singular_value):
                    reason = "unresolved parameter"
                elif direction.singular_value == 0 and np.count_nonzero(
                        vector) == 1:
                    reason = "zero information"
                else:
                    reason = (f"singular value "
                              f"{direction.singular_value:.3g} <= {floor}, "
                              "or s**2 <= eps * s_max**2")
                lines.append(
                    f"  Unresolved direction ({reason}): "
                    f"{_format_direction(vector, names)}")
        if self.unresolved_parameters:
            lines.append(
                "  Unresolved parameters (sensitivity error bound >= the "
                "sensitivity itself, consistent with no effect): "
                f"{', '.join(self.unresolved_parameters)}")
        if self.zero_information:
            lines.append("  Zero information (no effect on any observed "
                         f"entry): {', '.join(self.zero_information)}")
        if self.weakest_direction is not None:
            weakest = self.weakest_direction
            vector = np.array(weakest.components)  # [-]
            value = ("unresolved parameter, no singular value assigned"
                     if np.isnan(weakest.singular_value) else
                     f"smallest singular value {weakest.singular_value:.3g} "
                     "of the scaled Jacobian")
            lines.append(
                f"Weakest direction ({value}): "
                f"{_format_direction(vector, names)}; collinearity index "
                f"(Brun et al., 2001) {self.collinearity_index:.3g}, scaled "
                f"condition number {self.condition_number:.3g}, {floor}.")

        alpha_pct = 100 * self.alpha  # [%]
        lines.append(
            f"{'parameter':<14} {'estimate':>12} {'std error':>12} "
            f"{'rel. SE':>10} {f'{alpha_pct:g}% CI lower':>14} "
            f"{f'{alpha_pct:g}% CI upper':>14} {'t ratio':>10} "
            f"{'signif.':>8}")
        for ind, name in enumerate(names):
            mark = ('n/a' if np.isnan(self.t_ratios[ind])
                    else 'yes' if self.significant[ind] else 'no')
            lines.append(
                f"{name:<14} {self.estimates[ind]:>12.5g} "
                f"{self.standard_errors[ind]:>12.4g} "
                f"{self.relative_standard_errors[ind]:>10.3g} "
                f"{self.confidence_intervals[ind, 0]:>14.5g} "
                f"{self.confidence_intervals[ind, 1]:>14.5g} "
                f"{self.t_ratios[ind]:>10.3g} {mark:>8}")

        return '\n'.join(lines)

    def unresolved_flags(self) -> np.ndarray:
        """Return which parameters the data do not resolve.

        Returns
        -------
        numpy.ndarray
            Boolean, shape ``(n,)``, in ``parameter_names`` order: True for
            the parameters in ``unresolved_parameters`` or
            ``zero_information`` and for every parameter whose component in
            an unresolved direction does not round to zero at the three
            decimals of the printed directions; False otherwise. All False
            when ``J`` is not finite (no direction analysis).
        """
        flags = np.array([name in self.unresolved_parameters
                          or name in self.zero_information
                          for name in self.parameter_names])
        for direction in self.unresolved_directions:
            flags |= np.round(np.array(direction.components),
                              _DIRECTION_DECIMALS) != 0
        return flags

    def to_frame(self) -> pd.DataFrame:
        """Return the per-parameter quantities as a table.

        Returns
        -------
        pandas.DataFrame
            Indexed by parameter name, in ``parameter_names`` order, with
            columns ``estimate`` and ``standard_error`` (parameter units),
            ``relative_standard_error`` [-], ``ci_lower`` and ``ci_upper``
            (parameter units, confidence ``alpha``), ``t_ratio`` [-],
            ``significant`` (nullable ``boolean``: missing where
            ``t_ratio`` is NaN and for unresolved parameters),
            ``unresolved`` (bool, ``unresolved_flags``),
            ``zero_information`` (bool) and ``active_bound`` (nullable
            ``boolean``: missing when the optimizer does not report active
            bounds). ``DataFrame.attrs`` holds ``'identifiable'`` (bool) and
            ``'covariance_diagnostic'`` (str or None).
        """
        unresolved = self.unresolved_flags()
        significant = pd.array(
            [None if np.isnan(ratio) or loose else bool(flag)
             for ratio, flag, loose in zip(self.t_ratios, self.significant,
                                           unresolved)],
            dtype='boolean')
        active = pd.array(
            [None if self.active_bounds is None
             else name in self.active_bounds
             for name in self.parameter_names], dtype='boolean')
        frame = pd.DataFrame(
            {'estimate': self.estimates,
             'standard_error': self.standard_errors,
             'relative_standard_error': self.relative_standard_errors,
             'ci_lower': self.confidence_intervals[:, 0],
             'ci_upper': self.confidence_intervals[:, 1],
             't_ratio': self.t_ratios,
             'significant': significant,
             'unresolved': unresolved,
             'zero_information': [name in self.zero_information
                                  for name in self.parameter_names],
             'active_bound': active},
            index=pd.Index(self.parameter_names, name='parameter'))
        frame.attrs['identifiable'] = bool(self.identifiable)
        frame.attrs['covariance_diagnostic'] = self.covariance_diagnostic
        return frame


def _scaled_sensitivity(jac: np.ndarray, names: tuple,
                        jacobian_rtol: np.ndarray, output_error_norm: float,
                        steps: Optional[np.ndarray]) -> dict:
    """Analyze the Jacobi-scaled Jacobian of an accepted fit.

    Parameters
    ----------
    jac : numpy.ndarray
        Weighted residual Jacobian ``J``, shape ``(n, n_entries)``; entry
        ``(i, k)`` in [weighted residual k unit / parameter i unit].
    names : tuple of str
        Unique parameter names in the row order of ``J``.
    jacobian_rtol : numpy.ndarray
        Row-relative accuracy [-] of each sensitivity row, shape ``(n,)``.
    output_error_norm : float
        Bound on the whitened output errors [weighted residual unit]; read
        when ``steps`` is given.
    steps : numpy.ndarray or None
        Forward-difference steps, shape ``(n,)`` [parameter units], for a
        finite-difference Jacobian; None otherwise.

    Returns
    -------
    dict
        ``'error_bounds'``: ``jacobian_rtol_i`` plus, with ``steps``,
        ``2 * output_error_norm / (h_i * ||J_i||)`` [-] for the nonzero rows,
        NaN for zero rows; ``'unresolved_parameters'``: names of the nonzero
        rows with ``e_i / ||J_i|| >= 1``; ``'zero_information'``: names of
        the exactly zero rows; ``'singular_values'``, shape ``(n,)`` [-]:
        those of ``D J`` restricted to the remaining (resolved) rows,
        descending, then NaN for each unresolved parameter and 0 for each
        zero-information parameter (all NaN when ``J`` is not finite);
        ``'floor'``: root sum of squares of the remaining rows' bounds [-];
        ``'condition_number'`` ``(s_max / s_min)**2`` and
        ``'collinearity_index'`` ``1 / s_min`` [-] (infinite when any
        parameter is unresolved or has zero information); ``'unresolved'``:
        tuple of ``ParameterDirection``, the directions of the remaining
        rows with ``s <= floor`` or ``s**2 <= eps * s_max**2``, then the
        unit directions of the unresolved and zero-information parameters;
        ``'weakest'``: the last direction in that order (None when ``J`` is
        not finite).

    Notes
    -----
    A row whose error bound is at least its own norm is consistent with a
    zero row under the declared accuracy, so it is set aside like a
    zero-information row: each such parameter is its own unit direction,
    and the remaining rows are decomposed and bounded by Weyl's inequality
    without it, so its error does not raise the floor of the others.
    """
    num_params = len(names)  # [-]
    zero_rows = ~jac.any(axis=1)  # NaN entries count as nonzero
    zero_information = tuple(names[ind] for ind in np.flatnonzero(zero_rows))
    if not np.isfinite(jac).all():
        return {'singular_values': np.full(num_params, np.nan),
                'condition_number': np.nan, 'collinearity_index': np.nan,
                'error_bounds': np.full(num_params, np.nan),
                'floor': np.nan, 'unresolved': (), 'weakest': None,
                'unresolved_parameters': (),
                'zero_information': zero_information}

    nonzero = np.flatnonzero(~zero_rows)
    error_bounds = np.full(num_params, np.nan)  # [-]
    error_bounds[nonzero] = jacobian_rtol[nonzero]  # [-], first term
    if steps is not None and nonzero.size:
        # [-]; two outputs, each with an error up to output_error_norm,
        # are differenced (Nocedal & Wright eq. 8.5, 2 u L_f / h). Divide
        # the norms before the steps so that magnitudes near the
        # floating-point limits cancel first (inf means unresolved).
        with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
            rounding = (2 * (output_error_norm / _row_norms(jac[nonzero]))
                        / steps[nonzero])
        error_bounds[nonzero] += np.nan_to_num(rounding, nan=np.inf)
    # Rows whose error may be as large as the row itself [-]
    swamped = np.zeros(num_params, dtype=bool)
    swamped[nonzero] = error_bounds[nonzero] >= 1
    kept = ~zero_rows & ~swamped
    unresolved_parameters = tuple(names[ind]
                                  for ind in np.flatnonzero(swamped))

    # The kept rows, with the set-aside rows zeroed, so that
    # _jacobi_scaled_rows treats both like zero rows.
    kept_jac = np.where(kept[:, np.newaxis], jac, 0.0)  # [r_k/p_i]
    _, informed, scaled_jac = _jacobi_scaled_rows(kept_jac)
    num_informed = len(informed)  # [-]
    floor = _scaled_norm(error_bounds[informed])  # [-], 0 when empty
    directions = np.zeros((num_params, num_params))  # [-], columns
    singular = np.zeros(num_params)  # [-]
    if num_informed:
        # [-]; left singular vectors of D J are parameter-space directions
        left, singular_informed, _ = np.linalg.svd(scaled_jac,
                                                   full_matrices=False)
        directions[np.ix_(informed, np.arange(num_informed))] = left
        singular[:num_informed] = singular_informed
    set_aside = np.concatenate((np.flatnonzero(swamped),
                                np.flatnonzero(zero_rows)))
    for offset, ind in enumerate(set_aside):
        directions[ind, num_informed + offset] = 1.0
    num_swamped = int(swamped.sum())  # [-]
    singular[num_informed:num_informed + num_swamped] = np.nan

    largest = singular[0] if num_informed else 0.0  # [-]
    resolved = singular[:num_informed]  # [-]
    if set_aside.size or not num_informed or resolved[-1] == 0:
        condition, collinearity = np.inf, np.inf  # [-]
    else:
        condition = (largest / resolved[-1]) ** 2  # [-]
        collinearity = 1 / resolved[-1]  # [-]
    unresolved_mask = np.ones(num_params, dtype=bool)
    unresolved_mask[:num_informed] = (
        (resolved <= floor) | (resolved ** 2 <= _FLOAT_EPS * largest ** 2))

    def direction(column: int) -> ParameterDirection:
        """Return the sign-normalized direction of one column.

        Parameters
        ----------
        column : int
            Column of ``directions``: resolved rows in descending
            singular-value order, then the set-aside parameters.

        Returns
        -------
        ParameterDirection
            Components [-] in parameter order and the singular value [-]
            (NaN for an unresolved parameter, 0 for zero information).
        """
        vector = _sign_normalized(directions[:, column])  # [-]
        return ParameterDirection(tuple(names), tuple(vector.tolist()),
                                  float(singular[column]))

    return {'singular_values': singular,
            'condition_number': float(condition),
            'collinearity_index': float(collinearity),
            'error_bounds': error_bounds, 'floor': floor,
            'unresolved': tuple(direction(column) for column
                                in np.flatnonzero(unresolved_mask)),
            'weakest': direction(num_params - 1),
            'unresolved_parameters': unresolved_parameters,
            'zero_information': zero_information}
