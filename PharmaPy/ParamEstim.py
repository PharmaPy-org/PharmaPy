#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 15:35:48 2019

@author: casas100
"""

import copy
import dataclasses
import inspect
import numbers
import warnings

import numpy as np
import numpy.typing as npt
from scipy.linalg import cholesky, solve_triangular
from scipy.optimize import Bounds, least_squares
from collections.abc import Mapping
from itertools import cycle
from types import MappingProxyType
from typing import Callable, NamedTuple, Optional, Sequence, Union

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable

import pandas as pd
from PharmaPy.jac_module import (numerical_jac_data, dx_jac_p, numerical_jac,
                                 CENTRAL_DIFFERENCE_REL_STEP,
                                 DEFAULT_OUTPUT_RTOL,
                                 PARAMETER_MAGNITUDE_FLOOR)
from PharmaPy import Gaussians as gs

from PharmaPy.LevMarq import levenberg_marquardt
from PharmaPy.Commons import plot_sens, reorder_sens
from PharmaPy.StatsModule import (IdentifiabilityReport, StatisticsClass,
                                  _JACOBIAN_SOURCES, _linearized_covariance,
                                  _parameter_scale, _per_parameter_fractions,
                                  _read_only, _read_only_bool,
                                  _require_defined_covariance, _scaled_norm)

try:
    from cyipopt import minimize_ipopt
    have_cyipopt = True
except ImportError:
    have_cyipopt = False

#: Values of ``method`` accepted by ``ParameterEstimation.optimize_fn``;
#: ``'scipy'`` and ``'least_squares'`` both select
#: ``scipy.optimize.least_squares``.
OPTIMIZATION_METHODS = ('LM', 'IPOPT', 'scipy', 'least_squares')
#: Marks an estimator attribute that did not exist when ``optimize_fn``
#: saved the previous fit.
_ABSENT = object()
#: Ipopt ``print_level`` [-] that ``optimize_fn`` requests when ``verbose``
#: is True: Ipopt's own default output level (``print_level`` accepts 0 to
#: 12, default 5; Ipopt options reference,
#: https://coin-or.github.io/Ipopt/OPTIONS.html). ``verbose=False`` gives 0.
IPOPT_VERBOSE_PRINT_LEVEL = 5
#: ``stop_criterion`` values of ``levenberg_marquardt`` (PharmaPy/LevMarq.py)
#: that mean a stopping test was met: a step below ``eps_2``, a gradient
#: below ``eps_1`` or a residual norm below ``tol_fun``. Its only other
#: value, ``'Maximum iterations exceeded'``, means that ``max_fun_eval``
#: trials were spent without meeting any of them.
_LM_CONVERGED_STOPS = (
    'Small step', 'Small gradient',
    'Value of the objective function lower than tolerance')
#: ``stop_criterion`` of ``levenberg_marquardt`` when its evaluation budget
#: ``max_fun_eval`` was spent.
_LM_BUDGET_STOP = 'Maximum iterations exceeded'
#: ``status`` of ``scipy.optimize.least_squares`` when its evaluation budget
#: ``max_nfev`` was exceeded (SciPy documentation of ``least_squares``,
#: ``OptimizeResult.status``); other unsuccessful statuses, -1 (improper
#: MINPACK input) and -2 (callback stop), have other causes.
_SCIPY_BUDGET_STATUS = 0
#: Stopping criteria of ``levenberg_marquardt`` recorded in
#: ``info['tolerances']``, in its signature order: ``max_fun_eval`` [-]
#: trial evaluations, ``eps_1`` [residual units**2 / parameter units] on
#: the gradient norm, ``eps_2`` [-] on the step relative to the parameter
#: norm and ``tol_fun`` [residual units] on the residual norm.
_LM_STOPPING_OPTIONS = ('max_fun_eval', 'eps_1', 'eps_2', 'tol_fun')
#: Stopping criteria of ``scipy.optimize.least_squares`` recorded in
#: ``info['tolerances']``, in its signature order: ``ftol`` [-] on the
#: relative cost change, ``xtol`` [-] on the relative step, ``gtol`` on the
#: method-scaled gradient norm (see the SciPy documentation) and
#: ``max_nfev`` [-] residual evaluations.
_SCIPY_STOPPING_OPTIONS = ('ftol', 'xtol', 'gtol', 'max_nfev')
#: Ipopt options of the "Termination" section of the Ipopt options
#: reference (https://coin-or.github.io/Ipopt/OPTIONS.html), the stopping
#: criteria recorded in ``info['tolerances']`` when they are in the options
#: of an IPOPT fit: tolerances on the scaled and unscaled optimality
#: error and its components, the iteration and time limits [-] and [s],
#: the "acceptable" termination thresholds and counts, the divergence
#: threshold and the complementarity target.
_IPOPT_STOPPING_OPTIONS = (
    'tol', 's_max', 'max_iter', 'max_wall_time', 'max_cpu_time',
    'dual_inf_tol', 'constr_viol_tol', 'compl_inf_tol', 'acceptable_tol',
    'acceptable_iter', 'acceptable_dual_inf_tol',
    'acceptable_constr_viol_tol', 'acceptable_compl_inf_tol',
    'acceptable_obj_change_tol', 'diverging_iterates_tol', 'mu_target')

linestyles = cycle(['-', '--', '-.', ':'])

eps = np.finfo(float).eps


def _user_stacklevel() -> int:
    """Return the ``warnings.warn`` stacklevel of the nearest user frame.

    Public estimation methods reach the same warning through call chains of
    different depth (``get_covariance`` directly, through ``optimize_fn``,
    ``estimate_parameters``, ``SimulationExec.EstimateParams``,
    ``CalibrationRecord.from_estimation`` or the bootstrap refits), so a
    fixed ``stacklevel`` cannot point at the user's call.

    Returns
    -------
    int
        Stacklevel [-] of the first frame, counted from the caller of this
        helper (stacklevel 1), whose module is not ``PharmaPy`` or one of
        its submodules. Falls back to 1 when every frame belongs to
        PharmaPy.
    """
    frame = inspect.currentframe()
    if frame is None:  # pragma: no cover - no frame support on this runtime
        return 1

    frame = frame.f_back  # caller of this helper, i.e. stacklevel 1
    level = 1
    while frame is not None:
        module = frame.f_globals.get('__name__', '')
        if module != 'PharmaPy' and not module.startswith('PharmaPy.'):
            return level
        frame = frame.f_back
        level += 1

    return 1


def _stopping_tolerances(method: str, options: Mapping) -> dict:
    """Return the stopping criteria that governed a fit, with their values.

    Parameters
    ----------
    method : {'LM', 'IPOPT', 'scipy', 'least_squares'}
        Optimization method of the fit.
    options : Mapping
        Effective options of the fit as the solver left them
        (``optimize_fn``'s private copy). For IPOPT, cyipopt has converted
        its keys to bytes and added its own entries, such as ``tol``.

    Returns
    -------
    dict
        For LM, ``_LM_STOPPING_OPTIONS`` and for ``'scipy'`` and
        ``'least_squares'``, ``_SCIPY_STOPPING_OPTIONS``, each with the
        value in ``options`` or, when not given, the keyword default of
        ``levenberg_marquardt`` or ``scipy.optimize.least_squares`` read
        from its signature, so that it cannot drift from the code that ran
        the fit (as ``Records._default_options`` reads the LM defaults).
        SciPy's ``max_nfev`` default is None, which selects SciPy's
        automatic, version-dependent evaluation limit (``100 * n`` for
        ``n`` parameters in SciPy 1.16 and later); None is recorded rather
        than a number SciPy may not use. For IPOPT, the entries of
        ``_IPOPT_STOPPING_OPTIONS`` present in ``options``, with bytes keys
        decoded as UTF-8: the caller's options and cyipopt's ``tol``
        default. Ipopt's native defaults, e.g. ``max_iter`` and
        ``acceptable_tol``, are not available there and are not recorded,
        as in ``Records.SolverSettings.default_options``. Units are those
        documented with each tuple of option names.
    """
    if method == 'IPOPT':
        decoded = {(key.decode('utf-8', errors='replace')
                    if isinstance(key, bytes) else key): value
                   for key, value in options.items()}
        return {name: decoded[name] for name in _IPOPT_STOPPING_OPTIONS
                if name in decoded}

    if method == 'LM':
        function, names = levenberg_marquardt, _LM_STOPPING_OPTIONS
    else:  # 'scipy' or 'least_squares'
        function, names = least_squares, _SCIPY_STOPPING_OPTIONS
    parameters = inspect.signature(function).parameters
    return {name: options.get(name, parameters[name].default)
            for name in names}


def _warn_unconverged(solver: str, reason: str, evaluations: str,
                      budget_option: Optional[str] = None,
                      budget_value: Optional[int] = None) -> None:
    """Warn that a fit stopped before meeting a convergence criterion.

    Parameters
    ----------
    solver : str
        Description of the optimizer and the ``method`` that selected it.
    reason : str
        The optimizer's stop reason (status and message).
    evaluations : str
        Description of the evaluation counts of the fit.
    budget_option : str, optional
        Name of the option whose evaluation budget stopped the fit
        (``max_fun_eval`` or ``max_nfev``). None (the default) when the fit
        stopped for another reason, e.g. a SciPy callback; the warning then
        names no budget and does not suggest raising one.
    budget_value : int, optional
        Effective value of ``budget_option`` [-]; None for SciPy's
        automatic limit. Ignored without ``budget_option``.

    Warns
    -----
    RuntimeWarning
        Always, pointing at the first caller outside PharmaPy
        (``_user_stacklevel``).
    """
    if budget_option is None:
        cause = ''
        advice = 'Review the solver status and message.'
    else:
        limit = (f"{budget_option}=None, SciPy's automatic limit"
                 if budget_value is None
                 else f"{budget_option}={budget_value!r}")
        cause = f', bounded by {limit}'
        advice = (f'Raise {budget_option} or review the seed, scaling and '
                  "tolerances (info['tolerances']).")
    warnings.warn(
        f"{solver} stopped without converging ({reason}) after "
        f"{evaluations}{cause}; the returned parameters, covariance and "
        "statistics describe the last accepted iterate, not a converged "
        f"estimate. {advice}",
        RuntimeWarning, stacklevel=_user_stacklevel())


def pseudo_inv(states, dstates_dparam=None):
    # Pseudo-inverse
    U, di, VT = np.linalg.svd(states)
    D_plus = np.zeros_like(states).T
    np.fill_diagonal(D_plus, 1 / di)
    pseudo_inv = np.dot(VT.T, D_plus).dot(U.T)

    if dstates_dparam is None:
        return pseudo_inv, di
    else:
        # Derivative
        first_term = -pseudo_inv @ dstates_dparam @ pseudo_inv
        second_term = pseudo_inv @ pseudo_inv.T @ dstates_dparam.T @ \
            (1 - states @ pseudo_inv)
        third_term = (1 - pseudo_inv @ states) @ (pseudo_inv.T @ pseudo_inv)

        dplus_dtheta = first_term + second_term + third_term

        return pseudo_inv, di, dplus_dtheta


def mcr_spectra(conc, spectra):
    conc_plus, _ = pseudo_inv(conc)

    absortivity_pred = np.dot(conc_plus, spectra)
    absorbance_pred = np.dot(conc, absortivity_pred)

    return conc_plus, absortivity_pred, absorbance_pred


def enforce_two_d(ar):
    if isinstance(ar, np.ndarray) and ar.ndim == 1:
        ar = ar.reshape(-1, 1)

    return ar


def convert_types(data, two_d=False):
    if isinstance(data, dict):
        data = list(data.values())
    elif isinstance(data, np.ndarray):
        data = [data]

    out = []
    for val in data:
        if isinstance(val, dict):
            proc = convert_types(val, two_d)
            val = dict(zip(val.keys(), proc))

        elif isinstance(val, (list, tuple)):
            val = convert_types(val, two_d)
        else:
            if two_d:
                val = enforce_two_d(val)

        out.append(val)

    return out


def _ordered_experiment_values(data: dict, names: list, label: str) -> list:
    """Read an experiment mapping in the declared independent-data order.

    Parameters
    ----------
    data : dict
        Experiment-keyed values; array shapes and physical units are retained.
    names : list
        Experiment keys in the insertion order of ``x_data``.
    label : str
        Input name used in validation errors.

    Returns
    -------
    list
        Original values selected by key, without modifying caller data.

    Raises
    ------
    ValueError
        If experiment keys differ from those declared by ``x_data``.
    """
    missing = [name for name in names if name not in data]
    unexpected = [name for name in data if name not in names]
    if missing or unexpected:
        raise ValueError(
            f"{label} experiment keys must match x_data; "
            f"missing={missing!r}, unexpected={unexpected!r}")
    return [data[name] for name in names]


def _experiment_arguments(data, count: int, names: Optional[list],
                          label: str, keyword: bool) -> list:
    """Normalize callback arguments once at the estimator boundary.

    Parameters
    ----------
    data : tuple, list, dict or None
        Callback arguments, preserving their model-defined units and shapes.
    count : int
        Number of experiments.
    names : list or None
        Explicit experiment keys, or None for positional datasets.
    label : str
        Input name used in validation errors.
    keyword : bool
        Whether values contain keyword dictionaries instead of argument tuples.

    Returns
    -------
    list
        One argument container per experiment in ``x_data`` order.

    Raises
    ------
    ValueError
        If keys or the number of argument containers do not match the data,
        or multiple experiment keys cannot be aligned with unnamed data.
    TypeError
        If positional arguments are not iterable or keyword arguments are not
        a dictionary. Positional errors identify experiment keys or, for
        unnamed experiments, zero-based positions.

    Notes
    -----
    For one experiment, a tuple contains positional callback arguments and a
    dictionary contains callback keywords. A dictionary keyed by the sole
    experiment name with a dictionary value instead denotes keyed keywords;
    use a one-element list to pass that same structure as callback keywords.
    Experiment mappings with multiple entries require named datasets;
    otherwise use a list in the positional dataset order.
    """
    if data is None:
        return [{} if keyword else () for _ in range(count)]

    single_keywords = keyword and count == 1 and isinstance(data, dict)
    keyed_keywords = (single_keywords and names is not None
                      and set(data) == set(names)
                      and isinstance(data[names[0]], dict))
    if single_keywords and not keyed_keywords:
        values = [data]
    elif isinstance(data, dict):
        if names is None and len(data) > 1:
            raise ValueError(
                f"{label} experiment keys {list(data)!r} cannot be aligned "
                "with unnamed x_data; pass x_data as a dictionary with the "
                f"same keys, or {label} as a list in x_data order")
        values = (list(data.values()) if names is None else
                  _ordered_experiment_values(data, names, label))
    elif (count == 1 and not keyword
          and not (isinstance(data, list) and data
                   and all(isinstance(value, tuple) for value in data))):
        values = [data]
    else:
        values = list(data)

    if len(values) != count:
        raise ValueError(
            f"{label} must contain one entry per experiment; "
            f"expected {count}, got {len(values)}")
    if keyword and any(not isinstance(value, dict) for value in values):
        raise TypeError(f"Each {label} entry must be a dictionary")
    if not keyword:
        invalid = []
        for name, value in zip(names if names is not None else range(count),
                               values):
            try:
                iter(value)
            except TypeError:
                invalid.append(name)
        if invalid:
            raise TypeError(
                f"Each {label} entry must be an iterable of positional "
                f"arguments; offending experiments: {invalid!r}. "
                "Use (value,) for a single positional argument.")
    return values


def _measurement_precision_root(weight_matrix) -> np.ndarray:
    """Build the root that applies the measurement precision to residuals.

    Parameters
    ----------
    weight_matrix : array_like
        Measurement-error covariance of the measured states, shape
        ``(n_measured, n_measured)``, with rows and columns in measured-state
        (``measured_ind`` and ``y_data`` column) order. Entry ``(i, j)`` has
        the unit ``u_i * u_j``, where ``u_i`` is the unit of measured state
        ``i`` (the standard-deviation unit of its measurement error). It must
        be symmetric positive definite; only its lower triangle is read.

    Returns
    -------
    precision_root : numpy.ndarray
        Lower-triangular Cholesky factor ``S`` of the precision
        ``P = inv(weight_matrix)``, shape ``(n_measured, n_measured)``, with
        ``S @ S.T == P`` in the same measured-state order and a positive
        diagonal, which makes it unique. Row ``i`` carries unit ``1/u_i`` and
        columns are dimensionless, so entry ``(i, j)`` has unit ``1/u_i``.
        Raw residual rows ``r``, shape ``(n_times, n_measured)``, are weighted
        as ``r @ S``: whitened component ``j`` combines measured states
        ``i >= j`` and is dimensionless, and the sum of squared weighted
        residuals equals ``sum_k r_k @ P @ r_k`` over sample rows ``k``.

    Raises
    ------
    ValueError
        If ``weight_matrix`` is not a square two-dimensional array or contains
        non-finite entries.
    numpy.linalg.LinAlgError
        If ``weight_matrix`` is not positive definite. ``LinAlgError`` is a
        ``ValueError`` subclass.

    Notes
    -----
    The factor of ``P`` is obtained without forming ``P``. With the exchange
    (order-reversal) matrix ``J``, the Cholesky factorization
    ``J @ weight_matrix @ J = R.T @ R`` (``R`` upper triangular) gives the
    reverse Cholesky factorization ``weight_matrix = U @ U.T`` with
    ``U = (J @ R @ J).T`` upper triangular. Then
    ``P = inv(U).T @ inv(U)``, so ``S = inv(U).T`` is lower triangular with
    ``S @ S.T = P``; a triangular solve supplies ``inv(U)``. The weighted
    rows ``r @ S = (inv(U) @ r.T).T`` are whitened residuals, with identity
    covariance when ``weight_matrix`` is the measurement-error covariance.
    A diagonal matrix of variances gives reciprocal standard deviations on
    the diagonal of ``S``.

    This equals the root ``L @ sqrt(D)`` of an unpivoted LDL factorization
    ``P = L @ D @ L.T``, which earlier releases stored whenever that
    factorization needed no pivoting, so those per-entry weighted residuals
    and Jacobian rows are preserved. Cholesky needs no pivoting for a
    symmetric positive-definite matrix; indexing a pivoted LDL factor by its
    permutation instead replaced ``P`` by ``P[perm][:, perm]`` (issue #237).
    """
    weight_matrix = np.asarray(weight_matrix, dtype=float)
    if weight_matrix.ndim != 2 or (
            weight_matrix.shape[0] != weight_matrix.shape[1]):
        raise ValueError(
            "weight_matrix must be a square two-dimensional measurement-error "
            "covariance with one row and column per measured state; got "
            f"shape {weight_matrix.shape}")
    if not np.all(np.isfinite(weight_matrix)):
        raise ValueError("weight_matrix entries must be finite")

    # Entry (i, j) in [u_(n-1-i) * u_(n-1-j)]: states in reversed order.
    reversed_covariance = weight_matrix[::-1, ::-1]
    try:
        # Upper triangular; reversed_covariance = reversed_chol.T @
        # reversed_chol. Column j carries [u_(n-1-j)]. The upper triangle of
        # reversed_covariance is the lower triangle of weight_matrix.
        reversed_chol = cholesky(reversed_covariance, lower=False,
                                 check_finite=False)
    except np.linalg.LinAlgError as error:
        raise np.linalg.LinAlgError(
            "weight_matrix must be a symmetric positive-definite "
            "measurement-error covariance; its Cholesky factorization failed "
            f"({error})") from error

    # Upper triangular; weight_matrix = chol_upper @ chol_upper.T. Row i
    # carries [u_i]; columns are dimensionless.
    chol_upper = reversed_chol[::-1, ::-1].T
    # inv(chol_upper), upper triangular; column i carries [1/u_i].
    chol_upper_inv = solve_triangular(
        chol_upper, np.eye(weight_matrix.shape[0]), lower=False,
        check_finite=False)

    return chol_upper_inv.T


def get_masked_ydata(y_list, masks, assign_missing=None, merge=True):
    y_out = []

    for y, mask in zip(y_list, masks):
        nrow = len(mask)
        ncol = y.shape[1]

        y_ar = np.zeros((nrow, ncol))

        y_ar[mask] = y

        if assign_missing is not None:

            if isinstance(assign_missing, (np.ndarray, list)):
                y_ar[~mask] = assign_missing[~mask]
            else:
                y_ar[~mask] = assign_missing

        y_out.append(y_ar)

    if merge:
        y_out = np.hstack(y_out)

    return y_out


def get_array_list(values):
    out = []
    for val in values:
        if isinstance(val, list):
            out += val
        elif isinstance(val, tuple):
            out += list(val)
        elif isinstance(val, np.ndarray):
            out.append(val)

    return out


def analyze_data(x, y, merge_y=True):

    def get_di(keys, data):
        return dict(zip(keys, data))

    x_model = []
    x_mask = []

    y_masked = []
    for x_data, y_data in zip(x, y):  # experiment loop
        if isinstance(y_data, dict) and isinstance(y_data, dict):
            pass_x = get_array_list(x_data.values())
            pass_y = list(y_data.values())

            x_unif, x_ma, y_ma = analyze_data([pass_x], [pass_y],
                                              merge_y=False)

            # Save data
            x_model.append(x_unif[0])

            keys = y_data.keys()
            x_masks = [row for row in x_ma[0].T]
            x_mask.append(get_di(keys, x_masks))

            y_masked.append(get_di(keys, y_ma[0]))

        elif isinstance(x_data, (list, tuple)):
            x_all = np.sort(np.hstack(x_data))
            x_unique = np.unique(x_all)

            x_common = [np.isin(x_unique, data) for data in x_data]
            # x_common = np.column_stack(x_common)

            x_model.append(x_unique)
            x_mask.append(np.column_stack(x_common))

            y_mask = get_masked_ydata(y_data, x_common, np.nan, merge=merge_y)
            y_masked.append(y_mask)
        else:
            x_model.append(x_data)
            x_mask.append(None)

            y_masked.append(y_data)

    return x_model, x_mask, y_masked


def _validate_field(field):
    """Validate the model-output identity of a measurement.

    Parameters
    ----------
    field : int or str
        Zero-based column of the model callback output, or the name of that
        column in the estimator's ``output_names``.

    Returns
    -------
    int or str
        ``field`` as a built-in ``int`` or ``str``.

    Raises
    ------
    TypeError
        If ``field`` is a bool or neither an integer nor a str.
    ValueError
        If an integer ``field`` is negative or a str ``field`` is empty.
    """
    if isinstance(field, (bool, np.bool_)):
        raise TypeError(
            "Measurement field must be a non-negative int column index or a "
            "str output name; got a bool")
    if isinstance(field, (int, np.integer)):
        if field < 0:
            raise ValueError(
                "Measurement field must be a non-negative column index of "
                f"the model output; got {field}")
        return int(field)
    if isinstance(field, str):
        if not field:
            raise ValueError("Measurement field name must be a non-empty str")
        return field
    raise TypeError(
        "Measurement field must be a non-negative int column index or a str "
        f"output name; got {type(field).__name__}")


def _float_vector(values, label: str) -> np.ndarray:
    """Return a private one-dimensional float copy of sample values.

    Parameters
    ----------
    values : array_like
        Scalar or one-dimensional samples; units are preserved.
    label : str
        Input name used in error messages.

    Returns
    -------
    numpy.ndarray
        New float array of shape ``(n,)``; a scalar becomes shape ``(1,)``.

    Raises
    ------
    TypeError
        If ``values`` is complex or cannot be converted to floats.
    ValueError
        If ``values`` has more than one dimension.
    """
    try:
        is_complex = np.iscomplexobj(np.asarray(values))
    except (TypeError, ValueError):
        is_complex = False  # reported by the float conversion below
    if is_complex:
        raise TypeError(
            f"{label} must be real; got complex values, whose imaginary "
            "part would otherwise be discarded")
    try:
        array = np.array(values, dtype=float)  # [units of values]
    except (TypeError, ValueError) as error:
        raise TypeError(f"{label} must be numeric; {error}") from error
    if array.ndim == 0:
        array = array.reshape(1)  # [units of values]
    if array.ndim != 1:
        raise ValueError(
            f"{label} must be one-dimensional (a scalar is one sample); got "
            f"shape {array.shape}")
    return array


def _frozen(array: np.ndarray) -> np.ndarray:
    """Return an immutable copy of a numeric array.

    Parameters
    ----------
    array : numpy.ndarray
        Values to freeze; units and shape are preserved.

    Returns
    -------
    numpy.ndarray
        Read-only copy backed by an immutable ``bytes`` buffer, so its
        ``WRITEABLE`` flag cannot be set to True again.
    """
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(
        array.shape)


_MISSING = object()


def _optional_label(value, label: str) -> Optional[str]:
    """Validate optional unit or basis metadata.

    Parameters
    ----------
    value : str or None
        Metadata label; None means undeclared.
    label : str
        Input name used in error messages.

    Returns
    -------
    str or None
        ``value`` unchanged.

    Raises
    ------
    TypeError
        If ``value`` is neither None nor a str.
    """
    if value is not None and not isinstance(value, str):
        raise TypeError(
            f"Measurement {label} must be a str or None; got "
            f"{type(value).__name__}")
    return value


def _optional_experiment_text(value, label: str) -> Optional[str]:
    """Validate optional ``Experiment`` metadata text.

    Parameters
    ----------
    value : str or None
        Metadata label; None means undeclared.
    label : str
        Attribute name used in error messages.

    Returns
    -------
    str or None
        ``value`` unchanged.

    Raises
    ------
    TypeError
        If ``value`` is neither None nor a str.
    ValueError
        If ``value`` is an empty str.
    """
    if value is not None:
        if not isinstance(value, str):
            raise TypeError(
                f"Experiment {label} must be a str or None; got "
                f"{type(value).__name__}")
        if not value:
            raise ValueError(f"Experiment {label} must be non-empty")
    return value


class Measurement:
    """One observed model-output field sampled within one experiment.

    A measurement pairs independent-variable samples ``x`` (typically time
    [s]) with observations of one column of the model callback output.
    Arrays are stored as private immutable float copies (their read-only
    flag cannot be re-enabled), so later changes to
    the caller's arrays do not affect the measurement and fitting never
    modifies them.

    Parameters
    ----------
    field : int, str or None, optional
        Zero-based column of the model callback output, or its name in the
        ``output_names`` given to ``ParameterEstimation``. May be omitted
        (None) when the measurement is passed in an ``Experiment`` mapping,
        where ``Experiment`` defaults ``field`` to that mapping's key.
    x : array_like
        Independent-variable samples, shape ``(n,)``, in ``x_units``. A
        scalar is a single sampled point. Samples must be finite and
        non-decreasing; a repeated value denotes replicate observations at
        that x.
    values : array_like, optional
        Observations, shape ``(n,)``, already in the units and physical
        basis of the model output column ``field``. NaN marks a missing
        observation, which is excluded from residuals, data counts and
        degrees of freedom; nothing is interpolated or invented. Either
        ``values`` or its keyword alias ``y`` must be given, not both.
    y : array_like, optional
        Keyword alias for ``values``; normalized to ``values``.
    uncertainty : float or array_like, optional
        Standard deviation of the measurement error, a positive scalar or
        shape ``(n,)``, in the units and basis of ``values``. It must be
        finite and positive at every observed sample; entries at missing
        samples are ignored. The default None declares no uncertainty.
    units : str, optional
        Label of the observation units, e.g. ``'mol/L'``. No conversion is
        performed; the label documents ``values`` and is checked for
        consistency with other measurements of the same name or field.
    unit_y : str, optional
        Keyword alias for ``units``; normalized to ``units``. Cannot be
        combined with ``units``.
    basis : str, optional
        Label of the observation basis, e.g. ``'molar concentration'``; no
        conversion is performed.
    x_units : str or None, optional
        Label of the independent-variable units. The default is ``'s'``
        (time); None leaves them undeclared.
    unit_x : str or None, optional
        Keyword alias for ``x_units``; normalized to ``x_units``. Cannot be
        combined with ``x_units``.

    Raises
    ------
    TypeError
        If ``field`` or a label has an invalid type, ``x`` or observation
        values are omitted, or samples are not numeric.
    ValueError
        If duplicate alias keywords (``values`` and ``y``, ``units`` and
        ``unit_y``, or ``x_units`` and ``unit_x``) are given together, ``x``
        is empty, non-finite or decreasing, ``values`` does not
        match ``x`` in shape, contains infinities or has no observation, or
        ``uncertainty`` is not finite and positive at an observed sample.

    Notes
    -----
    Replicates may be given either as repeated x values within one
    measurement or as separate measurements of the same field. Within an
    ``Experiment`` the model grid holds each distinct x as many times as the
    measurement with most replicates at that x, and the model callback then
    receives repeated x values (as with the legacy array input, for example
    the replicate samples of the Ziegler 723 K workshop data).
    """

    def __init__(self, field=None, x=_MISSING, values=_MISSING, *,
                 y=_MISSING, uncertainty=None, units=_MISSING,
                 unit_y=_MISSING, basis=None, x_units=_MISSING,
                 unit_x=_MISSING) -> None:
        """Validate and store a private copy of one measurement.

        Parameters
        ----------
        field, x, values, y, uncertainty, units, unit_y, basis, x_units, unit_x
            See the class docstring.

        Raises
        ------
        TypeError, ValueError
            See the class docstring.
        """
        self._field = None if field is None else _validate_field(field)

        if x is _MISSING:
            raise TypeError(
                "Measurement requires independent-variable samples 'x'")
        if values is not _MISSING and y is not _MISSING:
            raise ValueError(
                "Measurement received both 'values' and alias 'y'; pass "
                "only one")
        if values is _MISSING and y is _MISSING:
            raise TypeError(
                "Measurement requires observation values; pass 'values' or "
                "'y'")
        if values is _MISSING:
            values = y

        if units is not _MISSING and unit_y is not _MISSING:
            raise ValueError(
                "Measurement received both 'units' and alias 'unit_y'; pass "
                "only one")
        if unit_y is not _MISSING:
            units = unit_y
        elif units is _MISSING:
            units = None

        if x_units is not _MISSING and unit_x is not _MISSING:
            raise ValueError(
                "Measurement received both 'x_units' and alias 'unit_x'; "
                "pass only one")
        if unit_x is not _MISSING:
            x_units = unit_x
        elif x_units is _MISSING:
            x_units = 's'

        x = _float_vector(x, 'Measurement x')  # [x_units]
        if x.size == 0:
            raise ValueError("Measurement x must contain at least one sample")
        nonfinite = np.flatnonzero(~np.isfinite(x))
        if nonfinite.size:
            index = nonfinite[0]
            raise ValueError(
                "Measurement x must be finite; "
                f"x[{index}] = {float(x[index])!r}")
        decreasing = np.flatnonzero(x[1:] < x[:-1])
        if decreasing.size:
            index = decreasing[0] + 1
            raise ValueError(
                f"Measurement x must be non-decreasing; x[{index}] = "
                f"{float(x[index])!r} is smaller than x[{index - 1}] = "
                f"{float(x[index - 1])!r}. Sort the samples together with "
                "their values; repeated x values denote replicate "
                "observations")

        values = _float_vector(values, 'Measurement values')  # [units]
        if values.shape != x.shape:
            raise ValueError(
                "Measurement values must have one entry per x sample; got "
                f"values shape {values.shape} and x shape {x.shape}")
        infinite = np.flatnonzero(np.isinf(values))
        if infinite.size:
            index = infinite[0]
            raise ValueError(
                f"Measurement values must be finite or NaN (missing); "
                f"values[{index}] = {float(values[index])!r}")
        observed = ~np.isnan(values)
        if not observed.any():
            raise ValueError(
                "Measurement values are all NaN; a measurement needs at "
                "least one observation (NaN marks a missing observation)")

        std = None  # [units]
        if uncertainty is not None:
            if np.ndim(uncertainty) == 0:
                scalar = _float_vector(uncertainty,
                                       'Measurement uncertainty')  # [units]
                std = np.full(x.shape, scalar[0])  # [units]
            else:
                std = _float_vector(uncertainty,
                                    'Measurement uncertainty')  # [units]
            if std.shape != x.shape:
                raise ValueError(
                    "Measurement uncertainty must be a scalar or have one "
                    f"entry per x sample; got shape {std.shape} and x shape "
                    f"{x.shape}")
            invalid = np.flatnonzero(
                observed & ~(np.isfinite(std) & (std > 0)))
            if invalid.size:
                index = invalid[0]
                raise ValueError(
                    "Measurement uncertainty must be a finite, positive "
                    "standard deviation at every observed sample; "
                    f"uncertainty[{index}] = {float(std[index])!r}")
            std = _frozen(std)  # [units]

        self._x = _frozen(x)  # [x_units]
        self._values = _frozen(values)  # [units]
        self._uncertainty = std  # [units]
        self._units = _optional_label(units, 'units')
        self._basis = _optional_label(basis, 'basis')
        self._x_units = _optional_label(x_units, 'x_units')

    def _with_field(self, field) -> 'Measurement':
        """Return a copy of this measurement with ``field`` set.

        Parameters
        ----------
        field : int or str
            Zero-based model-output column or output name.

        Returns
        -------
        Measurement
            New measurement sharing the immutable arrays and metadata of
            ``self``, with ``field`` set to ``field``.
        """
        clone = object.__new__(Measurement)
        clone._field = _validate_field(field)
        clone._x = self._x
        clone._values = self._values
        clone._uncertainty = self._uncertainty
        clone._units = self._units
        clone._basis = self._basis
        clone._x_units = self._x_units
        return clone

    @property
    def field(self):
        """Model-output identity.

        Returns
        -------
        int, str or None
            Zero-based output column, output name, or None when undeclared
            prior to ``Experiment`` mapping normalization.
        """
        return self._field

    @property
    def x(self) -> np.ndarray:
        """Independent-variable samples.

        Returns
        -------
        numpy.ndarray
            Read-only, shape ``(n,)``, non-decreasing, in ``x_units``.
        """
        return self._x

    @property
    def values(self) -> np.ndarray:
        """Observations.

        Returns
        -------
        numpy.ndarray
            Read-only, shape ``(n,)``, in ``units``; NaN marks a missing
            observation.
        """
        return self._values

    @property
    def y(self) -> np.ndarray:
        """Alias for ``values``.

        Returns
        -------
        numpy.ndarray
            Read-only, shape ``(n,)``, in ``units``; NaN marks a missing
            observation.
        """
        return self._values

    @property
    def uncertainty(self) -> Optional[np.ndarray]:
        """Standard deviation of the measurement error.

        Returns
        -------
        numpy.ndarray or None
            Read-only, shape ``(n,)``, in ``units``; None if undeclared.
        """
        return self._uncertainty

    @property
    def units(self) -> Optional[str]:
        """Observation unit label.

        Returns
        -------
        str or None
            Label, or None if undeclared.
        """
        return self._units

    @property
    def unit_y(self) -> Optional[str]:
        """Alias for ``units``.

        Returns
        -------
        str or None
            Label, or None if undeclared.
        """
        return self._units

    @property
    def basis(self) -> Optional[str]:
        """Observation basis label.

        Returns
        -------
        str or None
            Label, or None if undeclared.
        """
        return self._basis

    @property
    def x_units(self) -> Optional[str]:
        """Independent-variable unit label.

        Returns
        -------
        str or None
            Label, or None if undeclared.
        """
        return self._x_units

    @property
    def unit_x(self) -> Optional[str]:
        """Alias for ``x_units``.

        Returns
        -------
        str or None
            Label, or None if undeclared.
        """
        return self._x_units


class Experiment:
    """One experimental run: named measurements and its callback association.

    Parameters
    ----------
    measurements : Mapping[str, Measurement] or sequence of Measurement
        Named measurements; insertion order is the declared order. In a
        mapping, any ``Measurement`` whose ``field`` is None defaults its
        ``field`` to its mapping key (stored on a normalized copy without
        modifying the caller's ``Measurement``). In sequence form each
        measurement must declare a ``field`` and is named by it: the str
        field itself, or ``f'field_{field}'`` for an int field; replicate
        measurements of one field then need a mapping with distinct names.
    args : tuple or list, optional
        Positional callback arguments for this experiment, passed as
        ``func(params, x_model, *args, **kwargs)`` in model-defined units.
        Stored as a tuple; the values themselves are not copied.
    kwargs : Mapping, optional
        Callback keywords in model-defined units; a shallow copy is stored.
    name : str or None, optional
        Identifier of this experiment. Used by ``ParameterEstimation`` when
        a single ``Experiment`` or a sequence of ``Experiment`` objects is
        passed; when a mapping of experiments is passed, any non-None
        ``name`` must match its mapping key. The default is None.
    author : str or None, optional
        Author or source attribution for this experiment. The default is
        None.
    date : str or None, optional
        Date or timestamp label for this experiment. The default is None.

    Raises
    ------
    TypeError
        If ``measurements`` is not a mapping or sequence of
        ``Measurement`` objects, a name, ``author`` or ``date`` is not a
        str, ``args`` is not a tuple or list, or ``kwargs`` is not a
        mapping with str keys.
    ValueError
        If there are no measurements, a measurement in sequence form has
        ``field=None``, a name, ``author`` or ``date`` is empty, the
        sequence form repeats a name, or measurements declare different
        ``x_units``.

    Notes
    -----
    The experiment model grid is the multiset union of the measurement
    grids: each distinct x appears as many times as the largest number of
    replicates any measurement has at that x, in sorted order. The k-th
    replicate of a measurement at x maps to the k-th grid row with that x;
    rows a measurement does not sample are unobserved for it. Undeclared
    (None) ``x_units`` are compatible with any declared value.
    """

    def __init__(self, measurements, args=(), kwargs=None,
                 name: Optional[str] = None,
                 author: Optional[str] = None,
                 date: Optional[str] = None) -> None:
        """Validate and store the measurements, callback arguments and metadata.

        Parameters
        ----------
        measurements, args, kwargs, name, author, date
            See the class docstring.

        Raises
        ------
        TypeError, ValueError
            See the class docstring.
        """
        self._name = _optional_experiment_text(name, 'name')
        self._author = _optional_experiment_text(author, 'author')
        self._date = _optional_experiment_text(date, 'date')

        if isinstance(measurements, Measurement):
            raise TypeError(
                "Experiment measurements must be a mapping of names to "
                "Measurement objects or a sequence of Measurement objects; "
                "wrap a single Measurement in a list")
        if isinstance(measurements, Mapping):
            items = list(measurements.items())
        elif isinstance(measurements, (list, tuple)):
            items = []
            for index, measurement in enumerate(measurements):
                if not isinstance(measurement, Measurement):
                    raise TypeError(
                        f"Experiment measurements[{index}] must be a "
                        f"Measurement; got {type(measurement).__name__}")
                field = measurement.field
                if field is None:
                    raise ValueError(
                        f"Experiment measurements[{index}] has field=None; "
                        "specify field on each Measurement or pass "
                        "measurements as a mapping keyed by field name")
                meas_name = (field if isinstance(field, str)
                             else f'field_{field}')
                items.append((meas_name, measurement))
            names = [meas_name for meas_name, _ in items]
            repeated = sorted({meas_name for meas_name in names
                               if names.count(meas_name) > 1})
            if repeated:
                raise ValueError(
                    f"Experiment measurements repeat the names {repeated!r}; "
                    "pass a mapping with distinct names for replicate "
                    "measurements of the same field")
        else:
            raise TypeError(
                "Experiment measurements must be a mapping of names to "
                "Measurement objects or a sequence of Measurement objects; "
                f"got {type(measurements).__name__}")

        if not items:
            raise ValueError("Experiment needs at least one Measurement")

        checked = {}
        for meas_name, measurement in items:
            if not isinstance(meas_name, str):
                raise TypeError(
                    "Experiment measurement names must be str; got "
                    f"{type(meas_name).__name__} {meas_name!r}")
            if not meas_name:
                raise ValueError(
                    "Experiment measurement names must be non-empty")
            if not isinstance(measurement, Measurement):
                raise TypeError(
                    f"Experiment measurement {meas_name!r} must be a "
                    f"Measurement; got {type(measurement).__name__}")
            if measurement.field is None:
                measurement = measurement._with_field(meas_name)
            checked[meas_name] = measurement

        declared_x_units = {meas_name: measurement.x_units
                            for meas_name, measurement in checked.items()
                            if measurement.x_units is not None}
        if len(set(declared_x_units.values())) > 1:
            raise ValueError(
                "All measurements of one experiment must share x_units; got "
                f"{declared_x_units!r}")

        if not isinstance(args, (tuple, list)):
            raise TypeError(
                "Experiment args must be a tuple or list of positional "
                f"callback arguments; got {type(args).__name__}")
        if kwargs is None:
            kwargs = {}
        if not isinstance(kwargs, Mapping):
            raise TypeError(
                "Experiment kwargs must be a mapping of callback keywords; "
                f"got {type(kwargs).__name__}")
        for key in kwargs:
            if not isinstance(key, str):
                raise TypeError(
                    "Experiment kwargs keys must be str callback keyword "
                    f"names; got {type(key).__name__} {key!r}")

        self._measurements = checked
        self._x_units = next(iter(declared_x_units.values()), None)
        self._args = tuple(args)
        self._kwargs = dict(kwargs)

    @property
    def name(self) -> Optional[str]:
        """Experiment identifier.

        Returns
        -------
        str or None
            Declared experiment name, or None if undeclared.
        """
        return self._name

    @property
    def author(self) -> Optional[str]:
        """Experiment author or source attribution.

        Returns
        -------
        str or None
            Declared author, or None if undeclared.
        """
        return self._author

    @property
    def date(self) -> Optional[str]:
        """Experiment date or timestamp label.

        Returns
        -------
        str or None
            Declared date label, or None if undeclared.
        """
        return self._date

    @property
    def measurements(self) -> Mapping:
        """Named measurements in declared order.

        Returns
        -------
        types.MappingProxyType
            Read-only view mapping names to ``Measurement`` objects.
        """
        return MappingProxyType(self._measurements)

    @property
    def measurement_names(self) -> tuple:
        """Measurement names in declared order.

        Returns
        -------
        tuple of str
            Names of ``measurements``.
        """
        return tuple(self._measurements)

    @property
    def args(self) -> tuple:
        """Positional callback arguments.

        Returns
        -------
        tuple
            Arguments in model-defined units.
        """
        return self._args

    @property
    def kwargs(self) -> Mapping:
        """Callback keywords.

        Returns
        -------
        types.MappingProxyType
            Read-only view of the stored keywords, in model-defined units.
        """
        return MappingProxyType(self._kwargs)

    @property
    def x_units(self) -> Optional[str]:
        """Independent-variable unit label shared by the measurements.

        Returns
        -------
        str or None
            Declared label, or None if no measurement declares one.
        """
        return self._x_units


class _AlignedColumn(NamedTuple):
    """One measurement column of one experiment, aligned to its model grid.

    Attributes
    ----------
    name : str
        Measurement (residual column) name.
    field : int or str
        Declared model-output field: a column index (negative legacy
        indices count from the last output column) or an output name.
    units, basis, x_units : str or None
        Declared labels; None means undeclared.
    rows : numpy.ndarray
        Integer model-grid row of each sample, shape ``(n_m,)``.
    values : numpy.ndarray
        Observations, shape ``(n_m,)``, in the measurement units; NaN marks
        a missing observation.
    uncertainty : numpy.ndarray or None
        Measurement-error standard deviations, shape ``(n_m,)``, in the
        measurement units, or None if undeclared.
    """

    name: str
    field: object
    units: Optional[str]
    basis: Optional[str]
    x_units: Optional[str]
    rows: np.ndarray
    values: np.ndarray
    uncertainty: Optional[np.ndarray]


class _AlignedExperiment(NamedTuple):
    """Experiment data aligned to the model grid passed to the callback.

    Attributes
    ----------
    grid : object
        Model grid passed to the callback, in the model's x units. For
        ``Experiment`` input and staggered legacy grids, the multiset union
        of the measurement grids; for a shared legacy grid, the caller's
        object unchanged (dtype, shape, order and repeats preserved).
    num_rows : int
        Number of model-grid samples (first-axis length of ``grid``).
    columns : list of _AlignedColumn
        Measurement columns in declared order.
    args : object
        Positional callback arguments.
    kwargs : object
        Callback keyword dictionary.
    staggered : bool
        True for legacy staggered (per-state) grids, which keep the legacy
        layout: an ndarray observation mask even when every entry is
        observed, and float64 observations.
    """

    grid: object
    num_rows: int
    columns: list
    args: object
    kwargs: object
    staggered: bool = False


class _CompiledExperiments(NamedTuple):
    """Internal estimation layout compiled from aligned experiments.

    Attributes
    ----------
    measurement_names : list of str
        Residual (measurement) column names, first-appearance order.
    fields : list of int
        Model-output column of each measurement column, resolved through
        ``output_names``; legacy negative indices are kept.
    x_model : list
        Per-experiment model grid passed to the callback, in x units.
    x_masks : list of numpy.ndarray or None
        Per-experiment observation mask, shape ``(n_times, n_columns)``;
        None when every entry is observed.
    y_data : list of numpy.ndarray
        Per-experiment observations, shape ``(n_times, n_columns)``, in the
        measurement units; NaN where unobserved.
    residual_std : list of numpy.ndarray or None
        Per-experiment measurement standard deviations, shape
        ``(n_times, n_columns)``, in the measurement units, NaN where
        unobserved; None when no uncertainty is declared.
    args_fun : list
        Positional callback arguments per experiment.
    kwargs_fun : list
        Callback keywords per experiment.
    num_data : list of int
        Observed entries per experiment.
    measurement_units : list of str or None
        Declared unit of each measurement column (``Measurement.units``),
        None where no measurement of that name declares one.
    """

    measurement_names: list
    fields: list
    x_model: list
    x_masks: list
    y_data: list
    residual_std: Optional[list]
    args_fun: list
    kwargs_fun: list
    num_data: list
    measurement_units: list


def _validate_output_names(output_names) -> Optional[tuple]:
    """Validate the names of the model callback output columns.

    Parameters
    ----------
    output_names : iterable of str or None
        Unique, non-empty names in output-column order.

    Returns
    -------
    tuple of str or None
        The names, or None if not given.

    Raises
    ------
    TypeError
        If ``output_names`` is a str or contains a non-str entry.
    ValueError
        If ``output_names`` is empty, contains an empty name or repeats a
        name.
    """
    if output_names is None:
        return None
    if isinstance(output_names, str):
        raise TypeError(
            "output_names must be a sequence of str, one per model output "
            "column, not a single str")
    names = tuple(output_names)
    if not names:
        raise ValueError("output_names must name at least one output column")
    for name in names:
        if not isinstance(name, str):
            raise TypeError(
                f"output_names entries must be str; got {name!r}")
        if not name:
            raise ValueError("output_names entries must be non-empty")
    repeated = sorted({name for name in names if names.count(name) > 1})
    if repeated:
        raise ValueError(
            f"output_names must be unique; repeated {repeated!r}")
    return names


def _resolve_field(field, measurement_name, experiment_name,
                   output_names: Optional[tuple]) -> int:
    """Resolve a measurement field to a model-output column index.

    Parameters
    ----------
    field : int or str
        Validated field: a column index (negative legacy indices count from
        the last output column) or an output name.
    measurement_name, experiment_name : str
        Identity used in error messages.
    output_names : tuple of str or None
        Names of the model output columns.

    Returns
    -------
    int
        Model-output column; a negative legacy index is returned unchanged
        and checked against the callback output at evaluation.

    Raises
    ------
    ValueError
        If a str field is given without ``output_names`` or is not one of
        them, or an int field lies outside the declared output columns.
    """
    location = (f"Measurement {measurement_name!r} of experiment "
                f"{experiment_name!r}")
    if isinstance(field, str):
        if output_names is None:
            raise ValueError(
                f"{location} names model output {field!r}, but output_names "
                "was not given; pass output_names (the model output column "
                "names in column order) or use an int column index")
        if field not in output_names:
            raise ValueError(
                f"{location} names unknown model output {field!r}; "
                f"available output_names: {list(output_names)!r}")
        return output_names.index(field)
    if output_names is not None and not (
            -len(output_names) <= field < len(output_names)):
        raise ValueError(
            f"{location} selects model output column {field}, but "
            f"output_names declares {len(output_names)} output columns "
            f"{list(output_names)!r}")
    return field


def _model_grid(grids: list) -> tuple:
    """Build the multiset union of non-decreasing measurement grids.

    Parameters
    ----------
    grids : list of numpy.ndarray
        Non-decreasing one-dimensional grids, each shape ``(n_m,)``, in one
        x unit.

    Returns
    -------
    x_model : numpy.ndarray
        Sorted grid, shape ``(n_times,)``, in the promoted dtype of
        ``grids``: each distinct x repeated as many times as its largest
        multiplicity in any grid.
    distinct : numpy.ndarray
        Sorted distinct x values.
    first_rows : numpy.ndarray
        Row of ``x_model`` holding the first copy of each distinct value.

    Notes
    -----
    For strictly increasing grids every multiplicity is one and ``x_model``
    equals ``numpy.unique`` of the concatenated grids, the legacy staggered
    model grid.
    """
    distinct = np.unique(np.concatenate(grids))  # [x units]
    multiplicity = np.zeros(distinct.size, dtype=int)  # [-], copies per value
    for grid in grids:
        # [x units], [-] copies of each value in this grid.
        values, counts = np.unique(grid, return_counts=True)
        positions = np.searchsorted(distinct, values)
        multiplicity[positions] = np.maximum(multiplicity[positions], counts)

    x_model = np.repeat(distinct, multiplicity)  # [x units]
    first_rows = np.cumsum(multiplicity) - multiplicity
    return x_model, distinct, first_rows


def _grid_rows(grid: np.ndarray, distinct: np.ndarray,
               first_rows: np.ndarray) -> np.ndarray:
    """Map the samples of one measurement grid to model-grid rows.

    Parameters
    ----------
    grid : numpy.ndarray
        Non-decreasing measurement grid, shape ``(n_m,)``, in x units.
    distinct, first_rows : numpy.ndarray
        Outputs of ``_model_grid`` for the experiment.

    Returns
    -------
    numpy.ndarray
        Row index of each sample, shape ``(n_m,)``: the k-th replicate at a
        value maps to the k-th model-grid row holding that value.
    """
    # [-], zero-based replicate index among equal x values.
    replicate_rank = (np.arange(grid.size)
                      - np.searchsorted(grid, grid, side='left'))
    return first_rows[np.searchsorted(distinct, grid)] + replicate_rank


def _align_experiment(experiment: 'Experiment') -> _AlignedExperiment:
    """Align the measurements of one ``Experiment`` to its model grid.

    Parameters
    ----------
    experiment : Experiment
        Validated experiment.

    Returns
    -------
    _AlignedExperiment
        Multiset-union model grid [x units], one aligned column per
        measurement, the callback arguments and a fresh keyword copy.
    """
    measurements = experiment.measurements
    grid, distinct, first_rows = _model_grid(
        [measurement.x for measurement in measurements.values()])
    columns = [
        _AlignedColumn(name, measurement.field, measurement.units,
                       measurement.basis, measurement.x_units,
                       _grid_rows(measurement.x, distinct, first_rows),
                       measurement.values, measurement.uncertainty)
        for name, measurement in measurements.items()]
    return _AlignedExperiment(grid, grid.size, columns, experiment.args,
                              dict(experiment.kwargs))


def _check_declared(declared: dict, key: tuple, value, measurement_name,
                    experiment_name, subject: str) -> None:
    """Require declared metadata to agree; None (undeclared) always agrees.

    Parameters
    ----------
    declared : dict
        First declaration per key: ``(value, measurement, experiment)``.
        Updated in place.
    key : tuple
        Identity of the declaration, e.g. ``('units', 'field', 0)``.
    value : str or None
        Declared label.
    measurement_name, experiment_name : str
        Identity of the declaring measurement.
    subject : str
        Description of what must agree, used in the error message.

    Raises
    ------
    ValueError
        If ``value`` differs from an earlier declared value.
    """
    if value is None:
        return
    reference = declared.setdefault(
        key, (value, measurement_name, experiment_name))
    if reference[0] != value:
        raise ValueError(
            f"Inconsistent {key[0]} for {subject}: measurement "
            f"{reference[1]!r} of experiment {reference[2]!r} declares "
            f"{reference[0]!r}, but measurement {measurement_name!r} of "
            f"experiment {experiment_name!r} declares {value!r}. Values are "
            "not converted, so they must already be in one unit and basis")


def _compile_experiments(experiments: dict,
                         output_names: Optional[tuple]
                         ) -> _CompiledExperiments:
    """Compile experiments into the estimator's aligned data layout.

    This is the only alignment path for ``ParameterEstimation`` data:
    ``Experiment`` input and adapted legacy arrays both pass through it.

    Parameters
    ----------
    experiments : dict
        Experiment name -> ``Experiment`` or ``_AlignedExperiment`` (the
        legacy adapter's output), in experiment order.
    output_names : tuple of str or None
        Names of the model output columns, used to resolve str fields.

    Returns
    -------
    _CompiledExperiments
        Measurement columns in first-appearance order (experiment order,
        then each experiment's declared order); per experiment the model
        grid, observations and mask of shape ``(n_times, n_columns)``
        (columns an experiment does not measure are unobserved), standard
        deviations, callback arguments and observed-entry counts.

    Raises
    ------
    ValueError
        If a measurement name resolves to different fields in different
        experiments, declared ``x_units``, ``units`` or ``basis`` disagree
        (per measurement name and per resolved field; ``x_units`` across
        all experiments), only some measurements declare an uncertainty,
        or a field cannot be resolved.

    Notes
    -----
    Observations keep the dtype of the supplied values when every entry is
    covered (a shared legacy grid); otherwise they are promoted to at least
    float64 so that unobserved entries can hold NaN. Legacy staggered grids
    keep the earlier layout: float64 observations and an ndarray mask even
    when every entry is observed.
    """
    aligned = {name: (_align_experiment(experiment)
                      if isinstance(experiment, Experiment) else experiment)
               for name, experiment in experiments.items()}

    column_fields = {}  # name -> (field, first experiment)
    declared = {}
    with_std, without_std = [], []
    for experiment_name, spec in aligned.items():
        for column in spec.columns:
            name = column.name
            field = _resolve_field(column.field, name, experiment_name,
                                   output_names)
            if name in column_fields:
                reference_field, reference_experiment = column_fields[name]
                if field != reference_field:
                    raise ValueError(
                        f"Measurement {name!r} resolves to model output "
                        f"column {reference_field} in experiment "
                        f"{reference_experiment!r} but to column {field} in "
                        f"experiment {experiment_name!r}; a measurement name "
                        "must denote the same model output in every "
                        "experiment")
            else:
                column_fields[name] = (field, experiment_name)

            _check_declared(declared, ('x_units',), column.x_units, name,
                            experiment_name,
                            'all experiments, because one model callback '
                            'receives every grid')
            for attribute in ('units', 'basis'):
                value = getattr(column, attribute)
                _check_declared(declared, (attribute, 'name', name), value,
                                name, experiment_name,
                                f'measurement name {name!r}')
                _check_declared(declared, (attribute, 'field', field), value,
                                name, experiment_name,
                                f'model output column {field}')

            if column.uncertainty is None:
                without_std.append((experiment_name, name))
            else:
                with_std.append((experiment_name, name))

    if with_std and without_std:
        raise ValueError(
            "Either every Measurement declares an uncertainty or none does: "
            f"measurement {with_std[0][1]!r} of experiment "
            f"{with_std[0][0]!r} declares one, but measurement "
            f"{without_std[0][1]!r} of experiment {without_std[0][0]!r} does "
            "not. Mixing standard-deviation-scaled (dimensionless) and "
            "unscaled (unit-bearing) residuals is dimensionally inconsistent")

    measurement_names = list(column_fields)
    column_of = {name: column
                 for column, name in enumerate(measurement_names)}
    num_columns = len(measurement_names)  # [-]

    x_model, x_masks, y_data, residual_std = [], [], [], []
    args_fun, kwargs_fun, num_data = [], [], []
    for spec in aligned.values():
        shape = (spec.num_rows, num_columns)
        covered = np.zeros(shape, dtype=bool)
        for column in spec.columns:
            covered[column.rows, column_of[column.name]] = True
        dtype = np.result_type(*[column.values for column in spec.columns])
        if spec.staggered:
            dtype = np.dtype(np.float64)  # legacy staggered layout
        elif not covered.all():
            dtype = np.result_type(np.float64, dtype)
        # [measurement units]; NaN where unobserved.
        observations = (np.full(shape, np.nan, dtype=dtype)
                        if np.issubdtype(dtype, np.inexact)
                        else np.empty(shape, dtype=dtype))
        observed = np.zeros(shape, dtype=bool)
        std = np.full(shape, np.nan) if with_std else None  # [meas. units]
        for column in spec.columns:
            index = column_of[column.name]
            values = column.values  # [measurement units]
            sampled = (~np.isnan(values)
                       if np.issubdtype(values.dtype, np.inexact)
                       else np.ones(values.shape, dtype=bool))
            observations[column.rows, index] = values
            observed[column.rows, index] = sampled
            if std is not None:
                std[column.rows[sampled], index] = (
                    column.uncertainty[sampled])

        x_model.append(spec.grid)
        x_masks.append(None if observed.all() and not spec.staggered
                       else observed)
        y_data.append(observations)
        residual_std.append(std)
        args_fun.append(spec.args)
        kwargs_fun.append(spec.kwargs)
        num_data.append(int(observed.sum()))

    return _CompiledExperiments(
        measurement_names=measurement_names,
        fields=[column_fields[name][0] for name in measurement_names],
        x_model=x_model, x_masks=x_masks, y_data=y_data,
        residual_std=residual_std if with_std else None,
        args_fun=args_fun, kwargs_fun=kwargs_fun, num_data=num_data,
        measurement_units=[declared.get(('units', 'name', name), (None,))[0]
                           for name in measurement_names])


def _holds_measurements(x_data) -> bool:
    """Tell whether independent data contain bare ``Measurement`` objects.

    Parameters
    ----------
    x_data : object
        ``x_data`` argument of an estimator.

    Returns
    -------
    bool
        True for a ``Measurement``, or a list, tuple or mapping whose
        entries (or the entries of whose list or tuple values) include one.
    """
    def is_or_holds(value):
        """Tell whether one entry is or directly holds a Measurement.

        Parameters
        ----------
        value : object
            One ``x_data`` entry.

        Returns
        -------
        bool
            True for a ``Measurement`` or a list or tuple containing one.
        """
        return isinstance(value, Measurement) or (
            isinstance(value, (list, tuple))
            and any(isinstance(item, Measurement) for item in value))

    if is_or_holds(x_data):
        return True
    if isinstance(x_data, Mapping):
        return any(is_or_holds(value) for value in x_data.values())
    if isinstance(x_data, (list, tuple)):
        return any(is_or_holds(value) for value in x_data)
    return False


def _holds_experiments(x_data) -> bool:
    """Tell whether independent data contain ``Experiment`` objects.

    Parameters
    ----------
    x_data : object
        ``x_data`` argument of an estimator.

    Returns
    -------
    bool
        True for an ``Experiment`` or a mapping, list or tuple containing at
        least one.
    """
    if isinstance(x_data, Experiment):
        return True
    if isinstance(x_data, Mapping):
        return any(isinstance(value, Experiment) for value in x_data.values())
    if isinstance(x_data, (list, tuple)):
        return any(isinstance(value, Experiment) for value in x_data)
    return False


def _experiment_collection(x_data, label: str = 'x_data') -> Optional[dict]:
    """Normalize ``Experiment`` input to an ordered name mapping.

    Parameters
    ----------
    x_data : object
        ``x_data`` or ``experiments`` argument of ``ParameterEstimation``.
    label : str, optional
        Argument name used in error messages. The default is ``'x_data'``.

    Returns
    -------
    dict or None
        Experiment name -> ``Experiment`` in experiment order: mapping keys,
        ``experiment.name`` when declared on a single ``Experiment`` or in a
        sequence, or ``'exp_1'``, ``'exp_2'``, ... when ``experiment.name``
        is None. None if ``x_data`` holds no ``Experiment`` (legacy input).

    Raises
    ------
    TypeError
        If ``Experiment`` objects are mixed with other entries or a mapping
        key is not a str.
    ValueError
        If a mapping key is empty, a mapping key disagrees with a non-None
        ``experiment.name``, or a sequence repeats an experiment name.
    """
    if not _holds_experiments(x_data):
        return None
    if isinstance(x_data, Experiment):
        exp_name = x_data.name if x_data.name is not None else 'exp_1'
        return {exp_name: x_data}
    if isinstance(x_data, Mapping):
        items = list(x_data.items())
        from_mapping = True
    else:
        items = [
            (value.name if isinstance(value, Experiment)
             and value.name is not None else f'exp_{index + 1}', value)
            for index, value in enumerate(x_data)]
        from_mapping = False

    mixed = [name for name, value in items
             if not isinstance(value, Experiment)]
    if mixed:
        raise TypeError(
            f"{label} mixes Experiment objects with other entries "
            f"{mixed!r}; pass only Experiment objects, or only legacy "
            "arrays")
    for name, experiment in items:
        if not isinstance(name, str):
            raise TypeError(
                f"Experiment names must be str; got {type(name).__name__} "
                f"{name!r}")
        if not name:
            raise ValueError("Experiment names must be non-empty")
        if (from_mapping and experiment.name is not None
                and experiment.name != name):
            raise ValueError(
                f"Experiment mapping key {name!r} disagrees with "
                f"experiment.name {experiment.name!r}; use matching names "
                "or omit experiment.name")
    if not from_mapping:
        names = [name for name, _ in items]
        repeated = sorted({name for name in names if names.count(name) > 1})
        if repeated:
            raise ValueError(
                f"Experiment sequence repeats the names {repeated!r}; give "
                "each Experiment a distinct name")
    return dict(items)


def _labelled_weight_matrix(weight_matrix, names: list) -> np.ndarray:
    """Order a labelled measurement-error covariance by measurement column.

    Parameters
    ----------
    weight_matrix : pandas.DataFrame
        Covariance with index and columns labelled by measurement name, in
        any order; entry ``(i, j)`` in ``u_i * u_j``.
    names : list of str
        Measurement column names in internal column order.

    Returns
    -------
    numpy.ndarray
        Symmetric covariance, shape ``(n_columns, n_columns)``, rows and
        columns in ``names`` order, in ``u_i * u_j``.

    Raises
    ------
    TypeError
        If ``weight_matrix`` is not a DataFrame or holds complex values.
    ValueError
        If labels repeat, index and columns differ, or the labels are not
        exactly ``names``.

    Notes
    -----
    Columns are first aligned to the index order, and only the lower
    triangle in that caller order is read (as for array weights); the
    symmetric matrix it defines is then reordered by name. The weighting is
    therefore independent of the measurement column order even when the
    upper triangle differs or is left empty.
    """
    if not isinstance(weight_matrix, pd.DataFrame):
        raise TypeError(
            "With Experiment input, weight_matrix must be a pandas DataFrame "
            f"whose index and columns are the measurement names {names!r}, "
            "so that rows are matched by name rather than position; got "
            f"{type(weight_matrix).__name__}")
    index = list(weight_matrix.index)
    columns = list(weight_matrix.columns)
    for axis, labels in (('index', index), ('columns', columns)):
        repeated = sorted({str(label) for label in labels
                           if labels.count(label) > 1})
        if repeated:
            raise ValueError(
                f"weight_matrix {axis} repeats the labels {repeated!r}")
    if set(index) != set(columns):
        raise ValueError(
            "weight_matrix index and columns must hold the same measurement "
            f"names; index only: {[i for i in index if i not in columns]!r}, "
            f"columns only: {[c for c in columns if c not in index]!r}")
    missing = [name for name in names if name not in index]
    unexpected = [label for label in index if label not in names]
    if missing or unexpected:
        raise ValueError(
            "weight_matrix labels must be exactly the measurement names; "
            f"missing={missing!r}, unexpected={unexpected!r}")
    # Caller order: columns aligned to the index labels.
    aligned = weight_matrix.loc[index, index].to_numpy()
    if np.iscomplexobj(aligned):
        raise TypeError(
            "weight_matrix must hold real covariances; got complex values")
    aligned = np.asarray(aligned, dtype=float)  # [u_i * u_j]
    # [u_i * u_j], symmetric from the lower triangle in caller order.
    symmetric = np.tril(aligned) + np.tril(aligned, -1).T
    positions = [index.index(name) for name in names]
    # [u_i * u_j], rows and columns in measurement column order.
    return symmetric[np.ix_(positions, positions)]


def _validate_legacy_fields(measured_ind) -> list:
    """Validate legacy ``measured_ind`` entries.

    Parameters
    ----------
    measured_ind : sequence of int or str
        Model-output column per y column; negative ints count from the last
        output column (NumPy indexing), str entries are output names.

    Returns
    -------
    list
        Entries as built-in ``int`` or ``str``.

    Raises
    ------
    TypeError
        If ``measured_ind`` is not a sequence or an entry is a bool or
        neither an integer nor a str.
    """
    if isinstance(measured_ind, (str, bytes)) or not hasattr(
            measured_ind, '__len__'):
        raise TypeError(
            "measured_ind must be a sequence of int model-output columns or "
            f"output names; got {measured_ind!r}")
    fields = []
    for field in measured_ind:
        if isinstance(field, (bool, np.bool_)) or not isinstance(
                field, (int, np.integer, str)):
            raise TypeError(
                "measured_ind entries must be int columns or str output "
                f"names; got {field!r}")
        fields.append(field if isinstance(field, str) else int(field))
    return fields


def _legacy_grid(grid, name, label: str) -> np.ndarray:
    """Validate one staggered legacy grid without changing its dtype.

    Parameters
    ----------
    grid : array_like
        Per-state sample grid in the model's x units.
    name : str
        Experiment name used in error messages.
    label : str
        Grid position used in error messages.

    Returns
    -------
    numpy.ndarray
        The grid as a one-dimensional array: a scalar becomes one sample,
        and a grid with at most one non-singleton axis, such as ``(1, n)``
        or ``(n, 1)``, is flattened as earlier releases did.

    Raises
    ------
    ValueError
        If the grid has more than one non-singleton axis, is non-finite or
        decreasing.
    """
    grid = np.asarray(grid)  # [x units]
    if sum(length != 1 for length in grid.shape) <= 1:
        grid = grid.reshape(-1)  # [x units]
    location = f"x_data grid {label} of experiment {name!r}"
    if grid.ndim != 1:
        raise ValueError(
            f"{location} must be one-dimensional; got shape {grid.shape}")
    if np.issubdtype(grid.dtype, np.inexact) and not np.all(
            np.isfinite(grid)):
        raise ValueError(f"{location} must be finite")
    # Direct comparison; np.diff wraps or overflows for integer dtypes.
    decreasing = np.flatnonzero(grid[1:] < grid[:-1])
    if decreasing.size:
        index = decreasing[0] + 1
        raise ValueError(
            f"{location} must be non-decreasing; x[{index}] = "
            f"{grid[index].item()!r} is smaller than x[{index - 1}] = "
            f"{grid[index - 1].item()!r}. Sort each grid together with its "
            "observations")
    return grid


def _legacy_alignment(x_experiment, y_experiment, name) -> tuple:
    """Align one legacy experiment's observations to its model grid.

    Parameters
    ----------
    x_experiment : array_like or list of array_like
        Shared grid (any shape; row i belongs to observation row i), or one
        grid per observation array (staggered input), in the model's
        independent-variable units.
    y_experiment : numpy.ndarray or list of numpy.ndarray
        Observations of shape ``(n, n_columns)`` (or ``(n,)``), or one such
        array per staggered grid, in the model's state units.
    name : str
        Experiment name used in error messages.

    Returns
    -------
    grid : object
        Model grid: the caller's shared grid unchanged, or the multiset
        union of staggered grids in their promoted dtype.
    num_rows : int
        Number of model-grid samples.
    columns : list of tuple
        ``(rows, values)`` per y column, in column order: model-grid rows
        and observations (caller dtype).
    staggered : bool
        True for per-state (staggered) grids.

    Raises
    ------
    ValueError
        If staggered grids and observation arrays do not pair up, a grid is
        invalid, or the row counts of grids and observations differ.
    """
    def as_two_d(observations, label):
        """Return observations as a two-dimensional array.

        Parameters
        ----------
        observations : array_like
            Observations in the model's state units.
        label : str
            Position used in error messages.

        Returns
        -------
        numpy.ndarray
            Shape ``(n, n_columns)``, dtype preserved.

        Raises
        ------
        ValueError
            If ``observations`` has more than two dimensions.
        """
        observations = np.asarray(observations)  # [state units]
        if observations.ndim < 2:
            # [state units]
            observations = observations.reshape(-1, 1)
        if observations.ndim != 2:
            raise ValueError(
                f"y_data {label}of experiment {name!r} must be one- or "
                f"two-dimensional; got shape {observations.shape}")
        return observations

    if not isinstance(x_experiment, (list, tuple)):
        # Shared grid: row i of y_data is observed at x_data[i] (identity).
        observations = as_two_d(y_experiment, '')
        num_rows = 1 if np.ndim(x_experiment) == 0 else len(x_experiment)
        if observations.shape[0] != num_rows:
            raise ValueError(
                f"Experiment {name!r}: y_data has {observations.shape[0]} "
                f"rows, but x_data has {num_rows} samples; row i of y_data "
                "must hold the observations at x_data[i]")
        rows = np.arange(num_rows)
        return x_experiment, num_rows, [(rows, column)
                                        for column in observations.T], False

    if (not isinstance(y_experiment, (list, tuple))
            or len(y_experiment) != len(x_experiment)):
        count = (len(y_experiment)
                 if isinstance(y_experiment, (list, tuple)) else 'one')
        raise ValueError(
            f"Experiment {name!r} lists {len(x_experiment)} x_data "
            "grids, so y_data must list one observation array per grid; "
            f"got {count}")
    grids = [_legacy_grid(grid, name, str(index))
             for index, grid in enumerate(x_experiment)]
    grid, distinct, first_rows = _model_grid(grids)
    columns = []
    for index, (samples, observations) in enumerate(zip(grids,
                                                        y_experiment)):
        observations = as_two_d(observations, f'array {index} ')
        if observations.shape[0] != samples.size:
            raise ValueError(
                f"Experiment {name!r}: y_data array {index} has "
                f"{observations.shape[0]} rows, but its x_data grid has "
                f"{samples.size} samples")
        rows = _grid_rows(samples, distinct, first_rows)
        columns += [(rows, column) for column in observations.T]
    return grid, grid.size, columns, True


def _legacy_values(values: np.ndarray, column: int, name,
                   allow_empty: bool) -> np.ndarray:
    """Validate one legacy observation column with Measurement's rules.

    Parameters
    ----------
    values : numpy.ndarray
        Observations of one y column, shape ``(n,)``, in the model's state
        units.
    column : int
        Zero-based y column, used in error messages.
    name : str
        Experiment name, used in error messages.
    allow_empty : bool
        Accept a column without samples, which a staggered experiment uses
        to leave a state unmeasured (a fully unobserved column).

    Returns
    -------
    numpy.ndarray
        ``values`` unchanged for integer, boolean and floating dtypes; an
        object array whose elements are all real numbers (for example a
        pandas nullable column exported with ``na_value=numpy.nan``) is
        converted to float64, so its NaN entries mark missing observations.

    Raises
    ------
    TypeError
        If the observations are complex, or of another non-numeric dtype or
        object elements that are not real numbers.
    ValueError
        If the column is empty (unless ``allow_empty``), has no non-NaN
        observation, or holds an infinite value.

    Notes
    -----
    Integer and boolean dtypes cannot hold NaN or infinity and are accepted
    as they are.
    """
    location = f"y_data column {column} of experiment {name!r}"
    if values.dtype == object:
        invalid = [value for value in values.ravel()
                   if not isinstance(value, (numbers.Real, np.number))
                   or np.iscomplexobj(value)]
        if invalid:
            raise TypeError(
                f"{location}: values must be real numbers; got "
                f"{invalid[0]!r}")
        values = values.astype(np.float64)  # [state units]
    if np.iscomplexobj(values):
        raise TypeError(
            f"{location}: values must be real; got complex values, whose "
            "imaginary part would otherwise be discarded")
    if not (np.issubdtype(values.dtype, np.number)
            or np.issubdtype(values.dtype, np.bool_)):
        raise TypeError(
            f"{location}: values must be real numbers; got dtype "
            f"{values.dtype}")
    if values.size == 0:
        if allow_empty:
            return values
        raise ValueError(f"{location} has no observations")
    if not np.issubdtype(values.dtype, np.inexact):
        return values
    infinite = np.flatnonzero(np.isinf(values))
    if infinite.size:
        index = infinite[0]
        raise ValueError(
            f"{location}: values must be finite or NaN (missing); "
            f"values[{index}] = {float(values[index])!r}")
    if np.all(np.isnan(values)):
        raise ValueError(
            f"{location}: values are all NaN; a measurement needs at least "
            "one observation (NaN marks a missing observation)")
    return values


def _legacy_experiments(alignments: list, fields: list, args_fun: list,
                        kwargs_fun: list, names: list) -> dict:
    """Adapt aligned legacy data to compile-ready experiments.

    Parameters
    ----------
    alignments : list of tuple
        ``_legacy_alignment`` output per experiment.
    fields : list
        Validated ``measured_ind`` entries, one per y column.
    args_fun, kwargs_fun : list
        Aligned callback containers per experiment, stored unchanged.
    names : list
        Experiment names in experiment order.

    Returns
    -------
    dict
        Name -> ``_AlignedExperiment`` with one column ``f'y_{j}'`` per y
        column ``j`` (field ``fields[j]``; undeclared units, basis and x
        units).

    Raises
    ------
    ValueError
        If an experiment's column count differs from ``fields`` or a column
        is invalid (see ``_legacy_values``; empty columns are accepted only
        in staggered experiments).
    TypeError
        If a column holds complex or non-numeric values.
    """
    experiments = {}
    for name, (grid, num_rows, columns, staggered), args, kwargs in zip(
            names, alignments, args_fun, kwargs_fun):
        if len(columns) != len(fields):
            raise ValueError(
                f"Experiment {name!r} has {len(columns)} y_data "
                f"columns, but measured_ind lists {len(fields)} model "
                "outputs; supply one y_data column per measured_ind entry")
        columns = [(rows, _legacy_values(values, index, name, staggered))
                   for index, (rows, values) in enumerate(columns)]
        aligned_columns = [
            _AlignedColumn(f'y_{index}', field, None, None, None, rows,
                           values, None)
            for index, (field, (rows, values)) in enumerate(
                zip(fields, columns))]
        experiments[name] = _AlignedExperiment(grid, num_rows,
                                               aligned_columns, args, kwargs,
                                               staggered)
    return experiments


def flatten_spectral_sens(sens):
    """
    Flatten 3D spectral sensitivity into 2D array consistent with parameter
    estimation

    Parameters
    ----------
    sens : 3D numpy array
        .

    Returns
    -------
    out : 2D numpy array
        .

    """
    n_par, n_times, n_lambda = sens.shape
    out = sens.T.reshape(n_lambda * n_times, n_par)

    return out


#: Finite-difference formulas of ``DerivativeCheck.formulas``: the central
#: difference and the one-sided second-order three-point differences toward
#: larger and smaller parameter values (``_three_point_difference``).
DIFFERENCE_FORMULAS = ('central', 'forward', 'backward')


def _parameter_bounds(bounds, names: Sequence[str]) -> tuple:
    """Return the lower and upper bounds of the optimized parameters.

    Parameters
    ----------
    bounds : sequence of pairs, scipy.optimize.Bounds or None
        One ``(lower, upper)`` pair per optimized parameter, in
        ``name_params`` order and parameter units, the convention of the
        ``bounds`` of an IPOPT fit (``optimize_fn``) and of
        ``CalibrationRecord.optimizer_inputs()``; None or an infinity marks
        an unbounded side. A ``(num_params, 2)`` array is accepted too.
        Alternatively a ``scipy.optimize.Bounds`` whose ``lb`` and ``ub``
        are scalars or have shape ``(num_params,)`` (infinities unbounded),
        the way to pass the ``(lower_array, upper_array)`` bounds of
        ``scipy.optimize.least_squares``: a plain tuple of two arrays is
        read as per-parameter pairs, and is rejected only when its shape
        cannot be one pair per parameter. None leaves every parameter
        unbounded.
    names : sequence of str
        Optimized parameter names, used in the messages.

    Returns
    -------
    lower : numpy.ndarray
        Lower bounds, shape ``(num_params,)`` [parameter units], ``-inf``
        where unbounded.
    upper : numpy.ndarray
        Upper bounds, shape ``(num_params,)`` [parameter units], ``+inf``
        where unbounded.

    Raises
    ------
    TypeError
        If ``bounds`` is neither a ``scipy.optimize.Bounds`` nor a sequence
        of pairs, or a bound is neither None nor a real number (booleans
        and complex excluded).
    ValueError
        If ``bounds`` does not hold one pair per optimized parameter, a pair
        does not have two entries, the ``lb`` or ``ub`` of a
        ``scipy.optimize.Bounds`` does not broadcast to ``(num_params,)``,
        a bound is NaN, a lower bound is ``+inf``, an upper bound is
        ``-inf``, or a lower bound exceeds its upper bound.
    """
    num_params = len(names)  # [-]
    lower = np.full(num_params, -np.inf)  # [parameter units]
    upper = np.full(num_params, np.inf)  # [parameter units]
    forms = ("one (lower, upper) pair per optimized parameter, or a "
             "scipy.optimize.Bounds(lower_array, upper_array) (a plain "
             "(lower_array, upper_array) tuple is read as pairs)")
    if bounds is None:
        return lower, upper
    if isinstance(bounds, Bounds):
        try:
            # [parameter units], scalars broadcast to every parameter
            sides = [np.broadcast_to(np.asarray(side, dtype=float),
                                     (num_params,))
                     for side in (bounds.lb, bounds.ub)]
        except (TypeError, ValueError) as error:
            raise ValueError(
                "scipy.optimize.Bounds lb and ub must be scalars or have "
                f"shape ({num_params},), one entry per optimized parameter "
                f"({', '.join(names)}); {error}") from error
        bounds = list(zip(sides[0].tolist(), sides[1].tolist()))
    if isinstance(bounds, np.ndarray):
        bounds = bounds.tolist()
    if isinstance(bounds, (str, bytes)) or not isinstance(bounds,
                                                          (list, tuple)):
        raise TypeError(f"bounds must be {forms}; got "
                        f"{type(bounds).__name__}")
    if len(bounds) != num_params:
        raise ValueError(
            f"bounds must be {forms}, for {num_params} parameters "
            f"({', '.join(names)}); got {len(bounds)} entries")
    for ind, (name, pair) in enumerate(zip(names, bounds)):
        if isinstance(pair, np.ndarray):
            pair = pair.tolist()
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError(f"bounds of {name!r} must be a (lower, upper) "
                             f"pair; got {pair!r}. bounds must be {forms}")
        for side, value in enumerate(pair):
            if value is None:
                continue
            if isinstance(value, (bool, np.bool_)) or not isinstance(
                    value, numbers.Real):
                raise TypeError(
                    f"bounds of {name!r} must be real numbers or None; got "
                    f"{type(value).__name__}")
            if np.isnan(value):
                raise ValueError(f"bounds of {name!r} must not be NaN")
            (lower if side == 0 else upper)[ind] = float(value)
        if lower[ind] == np.inf or upper[ind] == -np.inf:
            raise ValueError(
                f"bounds of {name!r} exclude every finite value: lower "
                f"{float(lower[ind])!r}, upper {float(upper[ind])!r}")
        if lower[ind] > upper[ind]:
            raise ValueError(
                f"The lower bound of {name!r} ({float(lower[ind])!r}) "
                f"exceeds its upper bound ({float(upper[ind])!r})")
    return lower, upper


def _array_contents(values) -> list:
    """Copy the contents of the arrays among saved values.

    Parameters
    ----------
    values : iterable
        Saved attribute values; arrays held directly or in lists and tuples
        are copied, read-only ones included: a read-only view of a
        writeable buffer changes when the buffer is refilled.

    Returns
    -------
    list of tuple
        ``(array, copy)`` pairs, units of each array preserved, for
        ``_restore_array_contents``: e.g. a preallocated sensitivity array
        returned by the model, or a read-only view of one, and stored in
        ``sens_second``.
    """
    pairs = []
    for value in values:
        items = value if isinstance(value, (list, tuple)) else (value,)
        for item in items:
            if isinstance(item, np.ndarray):
                pairs.append((item, item.copy()))
    return pairs


def _restore_array_contents(pairs: list) -> None:
    """Write saved contents back into arrays changed in place.

    Parameters
    ----------
    pairs : list of tuple
        Result of ``_array_contents``.

    Notes
    -----
    Writeable arrays are restored directly. A read-only view whose memory
    is writeable through its base is made writeable for the copy and
    read-only again afterwards, so its flags are unchanged. Arrays whose
    memory cannot be written at all (NumPy refuses to set the flag) cannot
    have been changed in place and are skipped.
    """
    for array, saved in pairs:
        if array.flags.writeable:
            np.copyto(array, saved)
            continue
        try:
            array.setflags(write=True)
        except ValueError:
            continue
        try:
            np.copyto(array, saved)
        finally:
            array.setflags(write=False)


def _agreement_threshold(rtol) -> Optional[float]:
    """Validate the optional agreement threshold of a derivative check.

    Parameters
    ----------
    rtol : float or None
        Threshold [-] on relative errors, or None for no threshold.

    Returns
    -------
    float or None
        ``rtol`` as a Python float [-], or None.

    Raises
    ------
    TypeError
        If ``rtol`` is not a real number (booleans excluded).
    ValueError
        If ``rtol`` is not positive and finite.
    """
    if rtol is None:
        return None
    if isinstance(rtol, (bool, np.bool_)) or not isinstance(rtol,
                                                            numbers.Real):
        raise TypeError(f"rtol must be a real number; got "
                        f"{type(rtol).__name__}")
    threshold = float(rtol)  # [-]
    if not (np.isfinite(threshold) and threshold > 0):
        raise ValueError(f"rtol must be positive and finite; got {rtol!r}")
    return threshold


def _relative_row_error(row: np.ndarray,
                        reference_row: np.ndarray) -> float:
    """Return ``||row - reference|| / max(||reference||, tiny)`` safely.

    Parameters
    ----------
    row : numpy.ndarray
        Jacobian entries of one parameter, shape ``(k,)`` [weighted
        residual unit / parameter unit].
    reference_row : numpy.ndarray
        Reference entries, same shape and units.

    Returns
    -------
    float
        Relative error [-]: 0 when both rows are zero (or empty), ``inf``
        or a huge finite value when only the reference is zero, NaN when
        either row holds a non-finite entry.

    Notes
    -----
    Both rows are divided by their common largest magnitude before the
    difference is formed, and the norms are computed by
    ``StatsModule._scaled_norm``, so no intermediate overflows or
    underflows: entries near the float64 limits give the true ratio, not
    ``inf / inf`` or ``0 / 0``, and a difference far below the largest
    entry is not lost to squaring. ``tiny`` is the smallest normal
    float64, scaled the same way.
    """
    if not (np.isfinite(row).all() and np.isfinite(reference_row).all()):
        return float('nan')
    largest = max(float(np.max(np.abs(row), initial=0.0)),
                  float(np.max(np.abs(reference_row),
                               initial=0.0)))  # [r / p]
    if largest == 0:
        return 0.0
    # [-]; _scaled_norm rescales again, so squares of entries far below
    # the largest one do not underflow to zero
    error_norm = _scaled_norm(row / largest - reference_row / largest)
    reference_norm = _scaled_norm(reference_row / largest)  # [-]
    # [-], tiny in the scaled units; may underflow to 0 for large rows
    floor = np.finfo(float).tiny / largest
    denominator = max(reference_norm, floor)  # [-]
    if denominator == 0:
        return float('inf')
    with np.errstate(over='ignore'):
        return float(error_norm / denominator)


def _three_point_difference(f_center: np.ndarray, f_near: np.ndarray,
                            f_far: np.ndarray, step: float) -> np.ndarray:
    """Return the one-sided second-order three-point derivative.

    Parameters
    ----------
    f_center : numpy.ndarray
        Function values at ``p`` [output units], shape ``(m,)``.
    f_near : numpy.ndarray
        Function values at ``p + step`` [output units], shape ``(m,)``.
    f_far : numpy.ndarray
        Function values at ``p + 2 * step`` [output units], shape ``(m,)``.
    step : float
        Signed step [parameter units]: positive for the forward formula,
        negative for the backward one.

    Returns
    -------
    numpy.ndarray
        ``(-3 f(p) + 4 f(p + s) - f(p + 2 s)) / (2 s)`` [output units /
        parameter units], shape ``(m,)``.

    Notes
    -----
    Taylor expansion, ``f(p + s) = f + s f' + s**2 f''/2 + s**3 f'''/6``
    and ``f(p + 2 s) = f + 2 s f' + 2 s**2 f'' + 4 s**3 f'''/3`` (plus
    higher-order terms), gives ``-3 f + 4 f(p + s) - f(p + 2 s) = 2 s f' -
    2 s**3 f'''/3``: the ``f''`` terms cancel and the truncation error is
    ``-s**2 f'''(p) / 3``, of the same order as the ``s**2 f'''(p) / 6`` of
    the central difference. Only points on one side of ``p`` are
    evaluated.
    """
    return (-3 * f_center + 4 * f_near - f_far) / (2 * step)


@dataclasses.dataclass(frozen=True, eq=False)
class DerivativeCheck:
    """Independent finite-difference check of the estimator's Jacobian.

    Built by ``ParameterEstimation.check_derivatives``. It compares the
    weighted residual Jacobian the optimizer and the covariance use,
    ``get_gradient(p, out_array=True)``, with second-order finite
    differences of the weighted residuals ``get_objective(p,
    out_array=True)`` at the same point, perturbing only the optimized
    parameters. Both arrays follow ``get_residual_layout``: unobserved
    entries are exactly zero in both and are excluded from the norms. No
    pass/fail threshold is applied unless the caller supplies ``rtol``.

    Attributes
    ----------
    parameter_names : tuple of str
        Optimized parameter names (``name_params``), in parameter order.
    params : numpy.ndarray
        Point of the check, shape ``(n,)`` [parameter units].
    rel_step : numpy.ndarray
        Relative steps [-], shape ``(n,)``.
    param_scale : numpy.ndarray
        Parameter scales the relative steps refer to, shape ``(n,)``
        [parameter units]: ``|params|`` or the caller's ``param_scale``.
    steps : numpy.ndarray
        Steps ``h_i`` actually used, shape ``(n,)`` [parameter units],
        positive: ``rel_step * param_scale`` rounded so that ``p + h`` (or
        ``p - h``) is the floating-point number evaluated.
    formulas : tuple of str
        ``'central'``, ``'forward'`` or ``'backward'`` per parameter (see
        ``DIFFERENCE_FORMULAS``); a one-sided formula is used where a
        central step would leave the bounds.
    jacobian_source : {'model', 'jac_fun', 'finite_difference'}
        Origin of ``jacobian`` (``ParameterEstimation._jacobian_source``):
        sensitivities returned by the model callback, ``jac_fun``, or the
        estimator's forward differences with ``dx_finitediff``.
    jacobian : numpy.ndarray
        Estimator's weighted residual Jacobian, shape ``(n, m)``; row ``i``
        in weighted residual units per unit of parameter ``i``.
    reference : numpy.ndarray
        Finite-difference reference, same shape and units.
    abs_error : numpy.ndarray
        ``|jacobian - reference|``, same shape and units.
    observed : numpy.ndarray
        Boolean, shape ``(m,)``: True for the observed entries of the
        residual layout (and for every penalty row of
        ``MultipleCurveResolution``).
    max_abs_error : numpy.ndarray
        Largest ``abs_error`` of each row over the observed entries, shape
        ``(n,)`` [weighted residual unit / parameter unit]; 0 without
        observed entries.
    relative_error : numpy.ndarray
        ``||J_i - R_i|| / max(||R_i||, tiny)`` over the observed entries,
        shape ``(n,)`` [-], with ``tiny`` the smallest normal float64 so
        that two zero rows give 0, computed after scaling both rows by
        their largest magnitude so that no norm overflows or underflows;
        NaN when either row holds a non-finite observed entry.
    rtol : float or None
        Agreement threshold [-] supplied by the caller, or None.
    agrees : numpy.ndarray or None
        Boolean, shape ``(n,)``: ``relative_error <= rtol`` (False for NaN);
        None without ``rtol``.

    Notes
    -----
    The reference is itself approximate. Its truncation error is about
    ``h**2 |f'''| / 6`` (central) or ``h**2 |f'''| / 3`` (one-sided), and
    its rounding error about ``u |f| / h`` for outputs of relative accuracy
    ``u``; see ``ParameterEstimation.check_derivatives`` for the step
    choice. A relative error far above that level for one parameter points
    at that parameter's sensitivities.
    """

    parameter_names: tuple
    params: np.ndarray
    rel_step: np.ndarray
    param_scale: np.ndarray
    steps: np.ndarray
    formulas: tuple
    jacobian_source: str
    jacobian: np.ndarray
    reference: np.ndarray
    abs_error: np.ndarray
    observed: np.ndarray
    max_abs_error: np.ndarray
    relative_error: np.ndarray
    rtol: Optional[float]
    agrees: Optional[np.ndarray]

    #: Float array fields, frozen as read-only copies by __post_init__.
    _ARRAY_FIELDS = ('params', 'rel_step', 'param_scale', 'steps',
                     'jacobian', 'reference', 'abs_error', 'max_abs_error',
                     'relative_error')

    def __post_init__(self) -> None:
        """Freeze the array fields.

        Returns
        -------
        None

        Notes
        -----
        Arrays are replaced by read-only copies, also after unpickling
        (see ``__reduce__``).
        """
        for name in self._ARRAY_FIELDS:
            object.__setattr__(self, name, _read_only(getattr(self, name)))
        object.__setattr__(self, 'observed', _read_only_bool(self.observed))
        if self.agrees is not None:
            object.__setattr__(self, 'agrees', _read_only_bool(self.agrees))
        object.__setattr__(self, 'parameter_names',
                           tuple(self.parameter_names))
        object.__setattr__(self, 'formulas', tuple(self.formulas))

    def __reduce__(self) -> tuple:
        """Support pickling and deep copying.

        Returns
        -------
        tuple
            ``(DerivativeCheck, field values)``; the constructor freezes the
            arrays again.
        """
        return (type(self), tuple(getattr(self, field.name)
                                  for field in dataclasses.fields(self)))

    @classmethod
    def from_derivatives(cls, parameter_names: Sequence[str],
                         params: np.ndarray, rel_step: np.ndarray,
                         param_scale: np.ndarray, steps: np.ndarray,
                         formulas: Sequence[str], jacobian_source: str,
                         jacobian: np.ndarray, reference: np.ndarray,
                         observed: np.ndarray,
                         rtol: Optional[float] = None
                         ) -> 'DerivativeCheck':
        """Compare a Jacobian with its finite-difference reference.

        Parameters
        ----------
        parameter_names : sequence of str
            Optimized parameter names, ``n`` entries.
        params : numpy.ndarray
            Point of the check, shape ``(n,)`` [parameter units].
        rel_step : numpy.ndarray
            Relative steps, shape ``(n,)`` [-].
        param_scale : numpy.ndarray
            Parameter scales, shape ``(n,)`` [parameter units].
        steps : numpy.ndarray
            Positive steps used, shape ``(n,)`` [parameter units].
        formulas : sequence of str
            One of ``DIFFERENCE_FORMULAS`` per parameter.
        jacobian_source : {'model', 'jac_fun', 'finite_difference'}
            Origin of ``jacobian``.
        jacobian : numpy.ndarray
            Estimator's Jacobian, shape ``(n, m)`` [weighted residual unit
            / parameter unit].
        reference : numpy.ndarray
            Finite-difference reference, shape ``(n, m)``, same units.
        observed : numpy.ndarray
            Boolean, shape ``(m,)``: entries entering the norms.
        rtol : float, optional
            Agreement threshold [-], positive and finite. The default None
            computes no ``agrees``.

        Returns
        -------
        DerivativeCheck
            The comparison; ``abs_error``, ``max_abs_error``,
            ``relative_error`` and ``agrees`` are computed here. A row with
            a non-finite observed entry in ``jacobian`` or ``reference``
            has NaN ``relative_error`` (and ``agrees`` False); no NumPy
            floating-point warning is emitted.

        Raises
        ------
        ValueError
            If ``params``, ``rel_step``, ``param_scale``, ``steps`` or
            ``formulas`` does not have ``n`` entries, the Jacobian shapes
            disagree, a formula or ``jacobian_source`` is unknown, or
            ``rtol`` is not positive and finite.
        TypeError
            If ``rtol`` is not a real number.
        """
        names = tuple(parameter_names)
        num_params = len(names)  # [-]
        vectors = {'params': params, 'rel_step': rel_step,
                   'param_scale': param_scale, 'steps': steps}
        for label, values in vectors.items():
            if np.shape(values) != (num_params,):
                raise ValueError(
                    f"{label} must have shape ({num_params},), one entry "
                    f"per parameter; got {np.shape(values)}")
        jac = np.array(jacobian, dtype=float)  # [r / p_i]
        ref = np.array(reference, dtype=float)  # [r / p_i]
        mask = np.array(observed, dtype=bool)
        if jac.ndim != 2 or jac.shape[0] != num_params or (
                ref.shape != jac.shape) or mask.shape != (jac.shape[1],):
            raise ValueError(
                f"jacobian {jac.shape} and reference {ref.shape} must have "
                f"shape ({num_params}, m) and observed {mask.shape} shape "
                "(m,)")
        unknown = [name for name in formulas
                   if name not in DIFFERENCE_FORMULAS]
        if unknown or len(formulas) != num_params:
            raise ValueError(f"formulas must hold one of "
                             f"{DIFFERENCE_FORMULAS} per parameter; got "
                             f"{list(formulas)}")
        if jacobian_source not in _JACOBIAN_SOURCES:
            raise ValueError(f"Unknown jacobian_source {jacobian_source!r}; "
                             f"expected one of {tuple(_JACOBIAN_SOURCES)}")
        rtol = _agreement_threshold(rtol)  # [-]

        # [r / p_i]; inf where the true difference exceeds the float64
        # range, NaN where an entry is not finite
        with np.errstate(over='ignore', invalid='ignore'):
            abs_error = np.abs(jac - ref)
        max_abs_error = np.zeros(num_params)  # [r / p_i]
        relative_error = np.zeros(num_params)  # [-]
        for ind in range(num_params):
            observed_error = abs_error[ind, mask]  # [r / p_i]
            if observed_error.size:
                max_abs_error[ind] = (np.nan if np.isnan(observed_error).any()
                                      else np.max(observed_error))
            relative_error[ind] = _relative_row_error(jac[ind, mask],
                                                      ref[ind, mask])
        agrees = None if rtol is None else relative_error <= rtol

        return cls(parameter_names=names, params=params, rel_step=rel_step,
                   param_scale=param_scale, steps=steps,
                   formulas=tuple(formulas),
                   jacobian_source=jacobian_source, jacobian=jac,
                   reference=ref, abs_error=abs_error, observed=mask,
                   max_abs_error=max_abs_error,
                   relative_error=relative_error, rtol=rtol, agrees=agrees)

    @property
    def max_relative_error(self) -> float:
        """Return the largest relative error over the parameters.

        Returns
        -------
        float
            ``max(relative_error)`` [-]; NaN if any entry is NaN, 0 without
            parameters.
        """
        if self.relative_error.size == 0:
            return 0.0
        return float(np.max(self.relative_error))

    def to_frame(self) -> pd.DataFrame:
        """Return the per-parameter comparison as a table.

        Returns
        -------
        pandas.DataFrame
            Indexed by parameter name, in ``parameter_names`` order, with
            columns ``value`` and ``step`` [parameter units], ``rel_step``
            [-], ``formula`` (str), ``max_abs_error`` [weighted residual
            unit / parameter unit] and ``relative_error`` [-], plus
            ``agrees`` (bool) when ``rtol`` was given.
            ``DataFrame.attrs`` holds ``'jacobian_source'``, ``'rtol'`` and
            ``'max_relative_error'``.
        """
        columns = {'value': self.params, 'step': self.steps,
                   'rel_step': self.rel_step, 'formula': list(self.formulas),
                   'max_abs_error': self.max_abs_error,
                   'relative_error': self.relative_error}
        if self.agrees is not None:
            columns['agrees'] = self.agrees
        frame = pd.DataFrame(columns, index=pd.Index(self.parameter_names,
                                                     name='parameter'))
        frame.attrs['jacobian_source'] = self.jacobian_source
        frame.attrs['rtol'] = self.rtol
        frame.attrs['max_relative_error'] = self.max_relative_error
        return frame

    def summary(self) -> str:
        """Return a human-readable account of the check.

        Returns
        -------
        str
            The Jacobian source and the number of observed entries; a table
            with each parameter's value, step, formula, largest absolute
            error and relative error (and agreement with ``rtol`` when it
            was given); the largest relative error; and a reminder of the
            accuracy of the reference itself.
        """
        lines = [
            f"Derivative check of {_JACOBIAN_SOURCES[self.jacobian_source]} "
            "against second-order finite differences of the weighted "
            f"residuals over {int(self.observed.sum())} observed entries.",
            f"{'parameter':<14} {'value':>12} {'step':>11} "
            f"{'formula':>9} {'max abs err':>12} {'rel. error':>11}"
            + ('' if self.agrees is None else f" {'agrees':>7}")]
        for ind, name in enumerate(self.parameter_names):
            line = (f"{name:<14} {self.params[ind]:>12.5g} "
                    f"{self.steps[ind]:>11.3g} {self.formulas[ind]:>9} "
                    f"{self.max_abs_error[ind]:>12.3g} "
                    f"{self.relative_error[ind]:>11.3g}")
            if self.agrees is not None:
                line += f" {'yes' if self.agrees[ind] else 'NO':>7}"
            lines.append(line)
        lines.append(f"Largest relative error: {self.max_relative_error:.3g}"
                     + ("" if self.rtol is None else
                        f" (rtol {self.rtol:.3g})") + ".")
        lines.append(
            "The reference is accurate to about rel_step**2 (truncation) "
            "plus the output accuracy divided by rel_step (rounding); for "
            "ODE outputs computed with the integrator relative tolerance "
            "solver_rtol (e.g. CVode rtol) use rel_step ~ "
            "solver_rtol**(1/3).")
        return '\n'.join(lines)


class ParameterEstimation:
    """Fit model parameters to weighted observations from one or more experiments.

    Data are given either as ``Experiment`` objects holding named
    ``Measurement`` objects, or as legacy arrays, lists and dictionaries.
    Both are compiled by one alignment path into measurement columns, a
    per-experiment model grid and observation masks. Experiment identity
    follows ``x_data`` keys when provided; state columns, physical bases and
    callback units remain those declared by the model; no unit conversion is
    performed.

    Weighted residuals are packed in experiment order, then measurement
    column order, then model-grid sample order (``get_residual_layout``).
    Unobserved entries keep their positions and are exactly zero.
    """

    #: Whether data are compiled through ``_compile_experiments``. Subclasses
    #: with a different residual definition (``MultipleCurveResolution``)
    #: keep the legacy ``analyze_data`` layout.
    _compiles_experiments = True
    #: Attributes that describe the last completed ``optimize_fn`` fit; a
    #: call that raises restores them (see ``_restore_fit_state``). The fit
    #: results and iteration histories are followed by the evaluation
    #: buffers that ``get_objective`` and ``get_gradient`` overwrite at
    #: every trial and that consumers such as ``StatisticsClass`` read.
    _FIT_ATTRIBUTES = ('opt_method', 'optim_options', 'optimize_flag',
                       'params_convg', 'info_opt', 'cond_number',
                       'params_iter', 'objfun_iter', 'paramest_df',
                       'y_model', 'covar_params', 'correl_params',
                       'y_runs', 'resid_runs', 'residuals',
                       'params_residuals', 'weighted_residuals', 'sens',
                       'sens_second', '_params_sens_second',
                       'num_model_states')

    def __init__(self, func: Callable, param_seed, x_data=None, y_data=None,
                 measured_ind=None,
                 args_fun=None, kwargs_fun=None,
                 optimize_flags=None,
                 jac_fun=None, dx_finitediff=None,
                 weight_matrix=None,
                 name_params=None, name_states=None,
                 output_names=None,
                 experiments=None) -> None:
        """ Create a ParameterEstimation object

        Parameters
        ----------
        func : callable
            Model function with signature func(params, x_data, *args, **kwargs).
            It must return either an array of size len(x_i) in the one-state
            case, and an array of size len(x_i) x num_states for models
            describing multiple states. State units/bases must match ``y_data``;
            parameter units must match ``param_seed``. See ``x_data`` for x_i.
        param_seed : array-like
            Parameter seed values in the units required by ``func``.
        x_data : Experiment, mapping or sequence of them, or legacy arrays, optional
            Either ``Experiment`` objects or legacy independent-variable
            data. ``Experiment`` input is a single ``Experiment`` (named
            ``experiment.name`` or ``'exp_1'``), a mapping of experiment
            names to ``Experiment`` (insertion order is experiment order),
            or a list or tuple of ``Experiment`` (named by each
            ``experiment.name`` or ``'exp_1'``, ``'exp_2'``, ...); then
            ``y_data``, ``measured_ind``, ``args_fun`` and ``kwargs_fun``
            must be None because each ``Experiment`` holds them, and
            ``self.x_data`` stores the per-experiment model grids.
            Legacy input is an array with experimental values for the
            independent variable x. If several datasets Ne are passed, either
            a list of arrays
                x_data = [x_1, ..., x_i, ..., x_Ne]
            or a dictionary of arrays:
                x_data = {'name_exp_1': x_1, ..., 'name_exp_i': x_i, ...,
                          'name_exp_N': x_Ne}
            can be specified; an experiment may also be a list of per-state
            grids (staggered sampling). Units follow the model independent
            variable (typically time [s]). A shared array is passed to
            ``func`` unchanged (any shape, dtype, order or repeated values),
            with row i of y belonging to its first-axis entry i. Staggered
            grids must be one-dimensional, finite and non-decreasing;
            repeated values are replicate samples, passed to ``func``
            repeatedly. Dictionary insertion order declares experiment
            order; other experiment mappings are aligned by those keys.
            Either ``x_data`` or ``experiments`` must be supplied, not both.
        y_data : numpy array, list of arrays or dict
            Legacy experimental values for the dependent variable(s) y, in
            the model's state units and physical bases.
            Array y is of dimension len(x_i) x N_meas, where N_meas is less
            than or equal to the number of states returned by func (Ny).
            It supports same data structures as ``x_data``. If ``y_data`` is a
            dictionary, its keys must match those of ``x_data``. A dictionary
            with more than one experiment requires named ``x_data``; with
            unnamed ``x_data``, pass a list in ``x_data`` order instead.
            NaN marks a missing observation. Observations are required for
            legacy input; ``None`` raises ``TypeError``.
        measured_ind : list of int or str, optional
            Indexes of the states returned by func that are measured and
            passed in each dataset contained in 'y_data', one per y column.
            Negative indexes count from the last output column; str entries
            name columns through ``output_names``. A one-dimensional output
            has one column, and with a single measured column its index is
            ignored for such outputs. If None, the first N_meas states are
            assumed to be measured. It is read at construction: the caller's
            object is stored unchanged as ``measured_ind``, and later changes
            to it have no effect. The default is None.
        args_fun : tuple, list of tuples or dict of tuples, optional
            Positional callback arguments in model-defined units. A tuple is
            passed directly for one experiment. A list contains one tuple per
            experiment in ``x_data`` order, including for one experiment:
            ``[(initial,)]`` passes ``initial``, not ``(initial,)``. Use
            ``((initial,),)`` when the callback argument is itself a tuple.
            With named experiments, a mapping must have exactly the
            ``x_data`` keys. Mappings with multiple entries require named
            ``x_data``; otherwise use a list in ``x_data`` order. Other
            iterable argument containers, such as lists and one-dimensional
            arrays, retain their callback meaning.
            The default is None.
        kwargs_fun : dict or list of dicts, optional
            Callback keywords in model-defined units. For multiple experiments,
            use a list in ``x_data`` order or a mapping with exactly its keys.
            Experiment mappings with multiple entries require named
            ``x_data``; otherwise use a list in ``x_data`` order.
            For one experiment, a dictionary contains callback keywords, except
            that ``{experiment_name: keyword_dict}`` denotes keyed keywords.
            To pass that reserved structure to the callback itself, wrap it in
            a one-element list. The default is None.
        optimize_flags : list of bools, optional
            list with dimension len(param_seed). If a given parameter is to
            be optimized, its corresponding flag is True. Otherwise, the flag
            must be False. If not provided, all the flags are set to
            True (all the parameters are used for optimization).
            The default is None.
        jac_fun : callable, optional
            jacobian function with the same signature as func. It must return
            an array with elements
                dy_j(t) / dparam_p (j = 1, ..., Ny, p = 1, ..., Np).

            The resulting array must be of size [sum_i len(x_i)] x num_params,
            which is formed by stacking jacobian matrices for each state
            vertically, with units of state / parameter. If None, it is
            computed using finite differences. The default is None.
        dx_finitediff : float, array-like or callable, optional
            perturbation in the parameter space used to estimate the
            parametric jacobian, in the corresponding parameter units, or
            a callable mapping the full parameter vector to the steps.
            Steps of estimated parameters must be nonzero and not
            infinite. The default is None, which uses forward-difference
            steps of magnitude ``1e-3 * max(|param|, 1e-6)`` [parameter
            units]: a relative step of 1e-3 [-] (positive) for
            ``|param| >= 1e-6``, and an absolute step of 1e-9 [parameter
            units] away from zero for smaller magnitudes (positive at
            zero), so a small negative parameter stays negative (see
            ``jac_module.dx_jac_p``).
        weight_matrix : numpy array, optional
            Symmetric positive-definite measurement-error covariance with
            dimension N_meas x N_meas. Rows and columns follow the measured
            states in ``measured_ind`` (``y_data`` column) order, and entry
            (i, j) has the product of the units of measured states i and j.
            Residuals are weighted by its inverse (the precision), so the
            objective is ``1/2 * sum_k r_k @ inv(weight_matrix) @ r_k`` over
            sample rows ``r_k`` of model minus data. A typical choice is a
            diagonal matrix of experimental state variances; correlated
            covariances are supported. Only the lower triangle is read.
            Sample rows with unobserved states (staggered ``x_data``
            grids) are weighted by the marginal precision
            ``inv(weight_matrix[o][:, o])`` of their observed states ``o``.
            With ``Experiment`` input it must be a ``pandas.DataFrame``
            whose index and columns are the measurement names (in any
            order), so rows are matched by name and reordering experiments
            cannot re-weight the fit; a plain array raises ``TypeError``.
            It must be None when measurements declare an ``uncertainty``.
            The default is None, which uses the dimensionless identity.
        name_params : list of str, optional
            Parameter names, either one per parameter of ``param_seed`` (in
            callback order; ``name_params`` then keeps the optimized ones)
            or one per optimized parameter (``name_params_total`` then holds
            only these). The default is None, which names parameter ``i``
            of ``param_seed`` ``'theta_<i+1>'`` in ``name_params_total``
            and keeps the same name for it in ``name_params``,
            ``name_params_plot`` and the ``paramest_df`` columns when it is
            optimized.
        name_states : list of str, optional
            list with state names. The default is None, which uses the
            measurement names for ``Experiment`` input.
        output_names : sequence of str, optional
            Unique names of the model callback output columns, in column
            order. Str ``Measurement.field`` values resolve through them, and
            the callback output must then have exactly this many columns.
            The default is None.
        experiments : Experiment, mapping or sequence of Experiment, optional
            Explicit keyword entry point for ``Experiment`` input: a single
            ``Experiment``, a mapping of experiment names to ``Experiment``,
            or a list or tuple of ``Experiment`` objects, normalized through
            ``_compile_experiments``. Cannot be passed simultaneously with
            ``x_data``. The default is None.

        Returns
        -------
        ParameterEstimation
            Estimator with one aligned input and callback entry per experiment.

        Raises
        ------
        ValueError
            Raised when:

            * both ``x_data`` and ``experiments`` are supplied;
            * legacy experiment mappings (``y_data``, ``args_fun``,
              ``kwargs_fun``) have keys other than those of ``x_data``, or
              several keys while ``x_data`` is unnamed;
            * no experiments are given, or ``x_data``, ``y_data``,
              ``args_fun`` or ``kwargs_fun`` hold different experiment
              counts;
            * ``y_data``, ``measured_ind``, ``args_fun`` or ``kwargs_fun``
              accompany ``Experiment`` input, an experiment name is empty,
              a mapping key disagrees with ``experiment.name``, or a
              sequence of ``Experiment`` objects repeats a name;
            * ``output_names`` is empty, has an empty or repeated name, or is
              given with nested state observation dictionaries;
            * a field names no ``output_names`` entry, a str field is given
              without ``output_names``, an int field lies outside them, or a
              measurement name resolves to different fields in different
              experiments;
            * declared units or bases disagree for one measurement name or
              model-output field, or declared x units across experiments;
            * only some measurements declare an uncertainty, or
              uncertainties are combined with ``weight_matrix``;
            * ``weight_matrix`` is not a finite square two-dimensional array
              with one row and column per measurement column, or a labelled
              ``weight_matrix`` repeats labels, has different index and
              column labels, or labels other than the measurement names;
            * a legacy shared grid and its observations differ in row count,
              observations have more than two dimensions, or an
              experiment's column count differs from ``measured_ind``;
            * legacy staggered grids do not pair with one observation array
              each or differ from it in length, or a grid has more than one
              non-singleton axis, is non-finite or decreasing;
            * a legacy observation column is empty (allowed only in a
              staggered experiment, where it leaves the state unmeasured),
              has no non-NaN observation, or holds an infinite value.
        TypeError
            Raised when:

            * both ``x_data`` and ``experiments`` are None, or ``y_data`` is
              None for legacy input;
            * ``Experiment`` objects are mixed with other entries,
              ``experiments`` does not hold ``Experiment`` objects, an
              experiment name is not a str, or ``x_data`` or ``experiments``
              holds bare ``Measurement`` objects not wrapped in an
              ``Experiment``;
            * ``output_names`` is a single str or holds a non-str entry;
            * with ``Experiment`` input, ``weight_matrix`` is not a
              ``pandas.DataFrame``, or a labelled ``weight_matrix`` holds
              complex values;
            * ``measured_ind`` is not a sequence or holds an entry that is
              a bool or neither an int nor a str;
            * a legacy observation column holds complex or non-numeric
              values (object arrays must hold real numbers only);
            * positional callback arguments are not iterable or keyword
              arguments are not a dictionary. Positional errors identify
              experiment keys or, for unnamed experiments, zero-based
              positions.
        numpy.linalg.LinAlgError
            If ``weight_matrix`` is not positive definite.

        Notes
        -----
        Legacy arrays, lists and dictionaries are adapted to one
        measurement column per y column (named ``'y_0'``, ``'y_1'``, ...;
        field ``measured_ind[j]``) and compiled by the same alignment path
        as ``Experiment`` input. A shared grid is aligned by identity (row
        i of y to entry i of x) and passed to ``func`` unchanged; the
        observations keep their dtype and the callback argument containers
        are stored as given. Staggered grids keep the earlier layout:
        float64 observations and an ndarray mask; an empty grid with an
        empty observation array leaves that state unmeasured. Valid legacy
        input therefore keeps its model grid, masks (None when a shared
        grid is fully observed), objective, residuals, Jacobian and fit.
        Intentional changes for legacy input:

        * A NaN observation marks a missing observation, excluded from the
          residuals and the data count; it previously made the objective
          NaN.
        * Object-dtype observations whose elements are all real numbers
          (for example pandas nullable columns exported with
          ``na_value=numpy.nan``) are converted to float64, so NaN entries
          are missing observations; previously they made the objective
          NaN. Other object or non-numeric observations raise
          ``TypeError``.
        * A column that is empty (outside staggered grids), has no non-NaN
          observation or holds an infinite value raises ``ValueError``, and
          complex observations raise ``TypeError``; previously they gave a
          non-finite or complex objective.
        * Raw residuals (``resid_runs``) of unobserved entries are exactly
          zero for every observation dtype; with float32 observations they
          were previously rounding residues of the model values.
        * Decreasing staggered grids raise ``ValueError``; they were
          previously paired with the wrong observations. Staggered grids
          with repeated values are now accepted as replicates.
        * Observation arrays whose row count differs from the shared grid,
          or whose column count differs from ``measured_ind``, raise
          ``ValueError``; they were previously broadcast.
        * Callback outputs must be one- or two-dimensional with one row per
          model-grid sample, and selected columns must exist; violations
          raise ``ValueError`` naming the experiment instead of being
          broadcast or raising ``IndexError``.
        * ``measured_ind`` is read at construction; later changes to the
          caller's object do not affect residuals or sensitivities.
        * For a one-dimensional callback output with a ``measured_ind``
          listing column 0 more than once (e.g. ``[0, 0]``), ``y_runs``
          holds one column per entry; it was previously one column
          broadcast against the observations, with equal residuals.

        Nested state observation dictionaries (the ``'spectra'`` /
        ``'non_spectra'`` layout of ``MultipleCurveResolution``) keep the
        legacy ``analyze_data`` layout, because their residuals are spectra
        resolved by curve resolution rather than model outputs minus
        observations; they are passed through without treating state names
        as experiment names.

        Lists and arrays retain positional ordering. Observation and callback
        mappings with multiple experiment keys cannot be aligned with
        unnamed ``x_data`` and are rejected; use lists in ``x_data`` order
        instead. A single experiment's direct keyword dictionary may still
        contain multiple callback keywords.

        Measurement columns are the union of measurement names in
        first-appearance order. A column an experiment does not measure is
        unobserved for it. Each experiment's model grid is the multiset
        union of its measurement grids (see ``Experiment``), so the model
        callback may receive repeated x values for replicate samples.
        Declared units and bases must agree per measurement name and per
        model-output field, and declared x units across experiments; None
        (undeclared) agrees with anything.

        If measurements declare uncertainties, each observed residual and
        its sensitivities are divided by their standard deviation, so the
        weighted residuals ``(model - data) / sigma`` are dimensionless and
        quantities in different units can be summed. The standard
        deviations are stored in ``_residual_std``. ``get_covariance``
        describes whether they are treated as relative or absolute, and
        ``StatisticsClass.get_bootsamples`` resamples the standardized
        residuals.
        No physical-unit or state-column conversion is performed.

        """

        self.function = func

        self.jac_fun = jac_fun

        self.fit_spectra = False

        param_seed = np.asarray(param_seed)
        if param_seed.ndim == 0:
            param_seed = param_seed[np.newaxis]

        if dx_finitediff is None:
            # Relative accuracy assumed for the model outputs [-]; the
            # relative step is its square root, 1e-3 (rationale at
            # jac_module.DEFAULT_OUTPUT_RTOL).
            rel_tol = DEFAULT_OUTPUT_RTOL
            # Parameter magnitude below which the step stops shrinking
            # [parameter units], 1e-6; its rationale is documented at
            # jac_module.PARAMETER_MAGNITUDE_FLOOR. Every |p| >= 1e-6 keeps
            # its previous step |p| * 1e-3. Below that the step is 1e-9 away
            # from zero (positive at zero): larger than the previous
            # 1e-3 * |p| for 0 < |p| < 1e-6, and nonzero at zero, where the
            # zero step made the Jacobian column NaN. Pass dx_finitediff for
            # parameters whose natural scale is smaller.
            abs_tol = PARAMETER_MAGNITUDE_FLOOR

            def dparam_fun(param):
                """Default finite-difference step of the parameters.

                Parameters
                ----------
                param : numpy.ndarray
                    Parameter values [parameter units], shape (n,).

                Returns
                -------
                numpy.ndarray
                    Steps of magnitude ``1e-3 * max(|param|, 1e-6)``
                    [parameter units], shape (n,), negative only for
                    ``-1e-6 < param < 0``; see ``jac_module.dx_jac_p``.
                """
                dp = dx_jac_p(param, abs_tol, rel_tol, eps)
                return dp

            dx_finitediff = dparam_fun

        self.dx_fd = dx_finitediff

        # --------------- Data
        self.experim_names = None
        self.output_names = _validate_output_names(output_names)

        if x_data is not None and experiments is not None:
            raise ValueError("Pass either x_data or experiments, not both")
        if x_data is None and experiments is None:
            raise TypeError(
                "x_data or experiments is required; pass Experiment objects "
                "through experiments= (or x_data=) or legacy x_data and "
                "y_data")

        if experiments is not None:
            if not self._compiles_experiments:
                raise TypeError(
                    "MultipleCurveResolution does not accept Experiment or "
                    "Measurement objects; pass spectral observations through "
                    "time_data and y_spectra")
            if _holds_measurements(experiments):
                raise TypeError(
                    "experiments holds Measurement objects; wrap each "
                    "experiment's measurements in Experiment(...), e.g. "
                    "experiments=Experiment({'c_A': measurement}) or "
                    "{'run_1': Experiment([...]), "
                    "'run_2': Experiment([...])}")
            experiment_source = 'experiments'
            experiments = _experiment_collection(experiments,
                                                 label='experiments')
            if experiments is None:
                raise TypeError(
                    "experiments must be an Experiment or a non-empty "
                    "mapping, list or tuple of Experiment objects")
        else:
            experiment_source = 'x_data'
            experiments = (_experiment_collection(x_data, label='x_data')
                           if self._compiles_experiments else None)

        if experiments is not None:
            supplied = [label for label, value in (
                ('y_data', y_data), ('measured_ind', measured_ind),
                ('args_fun', args_fun), ('kwargs_fun', kwargs_fun))
                if value is not None]
            if supplied:
                raise ValueError(
                    f"{', '.join(supplied)} must be None when "
                    f"{experiment_source} holds Experiment objects; "
                    "observations, measured fields and callback arguments "
                    "live in each Experiment and its Measurement objects")
            self._experiment_metadata = {
                name: {
                    'name': exp.name,
                    'author': exp.author,
                    'date': exp.date,
                }
                for name, exp in experiments.items()
            }
            self.experim_names = list(experiments)
            compiled = _compile_experiments(experiments, self.output_names)
            measured_ind = list(compiled.fields)
            x_data = list(compiled.x_model)  # [x units], model grids
            if weight_matrix is not None and compiled.residual_std is None:
                # [u_i * u_j], reordered by name to the column order.
                weight_matrix = _labelled_weight_matrix(
                    weight_matrix, compiled.measurement_names)
            legacy_fields = None
        else:
            self._experiment_metadata = None
            if self._compiles_experiments and _holds_measurements(x_data):
                raise TypeError(
                    "x_data holds Measurement objects; wrap each "
                    "experiment's measurements in Experiment(...), e.g. "
                    "x_data=Experiment({'c_A': measurement}) or "
                    "{'run_1': Experiment([...]), "
                    "'run_2': Experiment([...])}")
            if isinstance(x_data, dict):
                self.experim_names = list(x_data.keys())

            if y_data is None:
                raise TypeError(
                    "y_data is required; pass observations for each "
                    "experiment in x_data")

            if isinstance(y_data, dict) and self.experim_names is not None:
                y_data = _ordered_experiment_values(
                    y_data, self.experim_names, 'y_data')
            elif isinstance(y_data, dict) and len(y_data) > 1:
                raise ValueError(
                    f"y_data experiment keys {list(y_data)!r} cannot be "
                    "aligned with unnamed x_data; pass x_data as a "
                    "dictionary with the same keys, or y_data as a list in "
                    "x_data order")

            x_data = convert_types(x_data)
            y_data = convert_types(y_data, two_d=True)
            if not x_data or len(x_data) != len(y_data):
                raise ValueError(
                    "x_data and y_data must contain the same nonzero number "
                    f"of experiments; got {len(x_data)} and {len(y_data)}")

            args_fun = _experiment_arguments(
                args_fun, len(x_data), self.experim_names, 'args_fun',
                keyword=False)
            kwargs_fun = _experiment_arguments(
                kwargs_fun, len(x_data), self.experim_names, 'kwargs_fun',
                keyword=True)

            nested = any(isinstance(y, dict) for y in y_data)
            legacy_fields = None
            if self._compiles_experiments and not nested:
                # Legacy adapter: one column per y column, compiled by the
                # same path as Experiment input. A shared grid is aligned by
                # identity and passed to the callback unchanged; staggered
                # grids use the multiset union.
                names = (self.experim_names if self.experim_names is not None
                         else ['exp_%i' % (ind + 1)
                               for ind in range(len(x_data))])
                alignments = [_legacy_alignment(x, y, name)
                              for x, y, name in zip(x_data, y_data, names)]
                if measured_ind is None:
                    measured_ind = list(range(len(alignments[0][2])))
                legacy_fields = _validate_legacy_fields(measured_ind)
                compiled = _compile_experiments(
                    _legacy_experiments(alignments, legacy_fields, args_fun,
                                        kwargs_fun, names),
                    self.output_names)
            else:
                # Nested 'spectra'/'non_spectra' dictionaries and
                # MultipleCurveResolution keep the analyze_data layout: their
                # spectral residuals are not model outputs minus data.
                if self.output_names is not None:
                    raise ValueError(
                        "output_names is not supported with nested state "
                        "observation dictionaries or MultipleCurveResolution")
                compiled = None

        if compiled is None:
            x_model, x_masks, y_data = analyze_data(x_data, y_data)
        else:
            x_model, x_masks, y_data = (compiled.x_model, compiled.x_masks,
                                        compiled.y_data)
            args_fun, kwargs_fun = compiled.args_fun, compiled.kwargs_fun

        # [x units]; per experiment, model grid with replicate repeats.
        self.x_model = x_model
        self.x_masks = x_masks

        self.x_data = x_data
        self.y_data = y_data  # [measured-state units]
        self.num_datasets = len(self.y_data)

        if self.experim_names is None:
            self.experim_names = ['exp_%i' % (ind + 1)
                                  for ind in range(self.num_datasets)]

        if measured_ind is None:
            measured_ind = list(range(y_data[0].shape[1]))

        # Caller's object for legacy input, read once here; the private
        # resolved copy below selects model outputs.
        self.measured_ind = measured_ind
        self.num_model_states = None
        self.sens_second = None  # sensitivities returned along with obj fun
        # [parameter units], parameters at which sens_second was evaluated.
        self._params_sens_second = None

        # Compiled layout; None for the analyze_data layout.
        self._measurement_names = (None if compiled is None
                                   else list(compiled.measurement_names))
        # Declared unit (str) or None per measurement column, read by
        # get_identifiability; None for the analyze_data layout.
        self._measurement_units = (None if compiled is None
                                   else list(compiled.measurement_units))
        self._measured_fields = (None if compiled is None
                                 else list(compiled.fields))
        # Legacy input with one measured column ignores its field for
        # one-dimensional callback outputs, as earlier releases did.
        self._legacy_single_output = (legacy_fields is not None
                                      and len(legacy_fields) == 1)
        # Non-negative output columns selected at the last evaluation.
        self._output_columns = None
        # Per experiment, measurement-error standard deviations [measurement
        # units], shape (n_times, n_columns), NaN where unobserved; None
        # unless measurements declare an uncertainty.
        self._residual_std = (None if compiled is None
                              else compiled.residual_std)

        # ---------- Arguments
        self.args_fun = args_fun
        self.kwargs_fun = kwargs_fun

        if compiled is None:
            num_data = []
            for ind in range(self.num_datasets):
                if self.x_masks[ind] is None:
                    num_data.append(self.y_data[ind].size)
                else:
                    if isinstance(self.x_masks[ind], dict):
                        num = [sum(mask)
                               for mask in self.x_masks[ind].values()]
                        num_data.append(sum(num))
                    else:
                        num_data.append(self.x_masks[ind].sum())
        else:
            num_data = list(compiled.num_data)  # observed entries only

        self.num_data_total = sum(num_data)
        self.num_data = num_data

        if self._residual_std is not None and weight_matrix is not None:
            raise ValueError(
                "weight_matrix must be None when measurements declare an "
                "uncertainty: a correlated covariance (weight_matrix) and "
                "per-sample standard deviations are alternative weightings")

        num_columns = (len(self.measured_ind) if compiled is None
                       else len(self._measured_fields))  # [-]
        # Whether the caller supplied weight_matrix (even the identity);
        # read by get_identifiability to state the noise model.
        self._weight_matrix_given = weight_matrix is not None
        if weight_matrix is None:
            weight_matrix = np.eye(num_columns)  # [-]

        # Lower triangular; row i in [1/u_i] for measured state unit u_i,
        # columns [-]. sigma_inv @ sigma_inv.T equals inv(weight_matrix) in
        # measured-state order.
        self.sigma_inv = _measurement_precision_root(weight_matrix)
        if compiled is not None and self.sigma_inv.shape[0] != num_columns:
            raise ValueError(
                "weight_matrix must have one row and column per measurement "
                f"column ({num_columns}: {self._measurement_names!r}); got "
                f"shape {self.sigma_inv.shape}")
        # [u_i * u_j], validated above; source of the marginal precision
        # roots of partially observed sample rows. A private copy, because
        # those roots are built lazily and must match sigma_inv even if the
        # caller later reuses or mutates its array.
        self._measurement_covariance = np.array(weight_matrix, dtype=float,
                                                copy=True)
        # Observed-state pattern (tuple of bool) -> lower root of
        # inv(weight_matrix[observed][:, observed]): row i [1/u_i] for the
        # i-th observed state, columns [-]; filled on first use.
        self._observed_roots = {}

        # --------------- Parameters
        self.num_params_total = len(param_seed)
        if optimize_flags is None:
            self.map_fixed = []
            self.map_variable = np.array([True]*self.num_params_total)
        else:
            self.map_variable = np.array(optimize_flags)
            self.map_fixed = ~self.map_variable

        self.param_seed = param_seed

        self.num_params = self.map_variable.sum()

        # --------------- Names
        # Parameters
        if name_params is None:
            # Number over the full parameter vector, so an optimized
            # parameter keeps its full-vector name when others are fixed.
            optimized_ind = np.flatnonzero(self.map_variable)
            self.name_params_total = ['theta_{}'.format(ind + 1)
                                      for ind in range(self.num_params_total)]

            self.name_params = [self.name_params_total[ind]
                                for ind in optimized_ind]

            # Braces keep multi-digit indices in the subscript.
            self.name_params_plot = [r'$\theta_{{{}}}$'.format(ind + 1)
                                     for ind in optimized_ind]
        else:
            self.name_params_total = name_params
            if len(name_params) > sum(self.map_variable):
                self.name_params = [name_params[ind]
                                    for ind in range(len(name_params))
                                    if self.map_variable[ind]]
            else:
                self.name_params = name_params

            self.name_params_plot = [r'$' + name + '$'
                                     for name in self.name_params]

        # ---------- States
        if name_states is None and experiments is not None:
            self.name_states = list(self._measurement_names)
        elif name_states is None and legacy_fields is not None:
            # Output names label str fields.
            self.name_states = [field if isinstance(field, str)
                                else r'$y_{}$'.format(field + 1)
                                for field in legacy_fields]
        elif name_states is None:
            self.name_states = [r'$y_{}$'.format(ind + 1)
                                for ind in self.measured_ind]
        else:
            self.name_states = name_states

        # --------------- Outputs
        self.params_iter = []
        self.objfun_iter = []
        self.cond_number = []

        self.optimize_flag = True  # [-]
        self.resid_runs = None
        self.params_residuals = None  # parameter units
        # [measured-state unit / weight_matrix unit**0.5]
        self.weighted_residuals = None
        self.y_runs = None
        self.sens = None
        self.sens_runs = None
        self.y_model = []

        self.method = None

    def _weight_sample_rows(self, values: np.ndarray,
                            x_mask: Optional[np.ndarray],
                            residual_std: Optional[np.ndarray] = None
                            ) -> np.ndarray:
        """Whiten residual-like values sample row by sample row.

        Parameters
        ----------
        values : numpy.ndarray
            Raw residuals, shape ``(n_times, n_measured)``, or their parameter
            sensitivities, shape ``(n_times, n_params, n_measured)``. The last
            axis follows measured-state (``measured_ind``) order; entries have
            the measured state's unit, divided by the parameter unit for
            sensitivities.
        x_mask : numpy.ndarray or None
            Observation mask, shape ``(n_times, n_measured)``, True where the
            state was measured at that model time. None means every state was
            measured at every model time.
        residual_std : numpy.ndarray or None, optional
            Measurement-error standard deviation of each entry, shape
            ``(n_times, n_measured)``, in the measured state's unit; read
            only at observed entries. None (the default) applies the
            ``weight_matrix`` weighting instead.

        Returns
        -------
        numpy.ndarray
            Weighted values with the shape of ``values``. With
            ``residual_std``, observed entries are ``values / residual_std``
            (dimensionless residuals, sensitivities per parameter unit).
            Otherwise, for each sample row
            with observed states ``o``, entries ``o`` are
            ``values[k, ..., o] @ S_o``, where ``S_o`` is the lower Cholesky
            factor of ``inv(weight_matrix[o][:, o])``; unobserved entries are
            exactly zero. Units are those of ``values`` divided by the
            measured state's unit (dimensionless for residuals weighted by a
            covariance in squared state units).

        Notes
        -----
        Residuals and sensitivities pass through the same weighting, so the
        weighted Jacobian is the derivative of the weighted residuals.

        Whitening the zero-filled full row with ``sigma_inv`` would instead
        apply the conditional precision ``P[o][:, o]`` of the full precision
        ``P = inv(weight_matrix)``, which exceeds the marginal precision of
        the observed states when they are correlated with unobserved ones,
        and would leave sensitivities of unobserved entries in the Jacobian.
        With a diagonal ``weight_matrix`` both precisions coincide and the
        weighted values equal ``values @ sigma_inv`` with unobserved entries
        zeroed. Fully observed rows use ``sigma_inv`` itself, so estimates
        without staggered grids are unchanged.
        """
        num_measured = values.shape[-1]
        if residual_std is not None:
            observed = (np.ones(residual_std.shape, dtype=bool)
                        if x_mask is None else x_mask)
            std = residual_std  # [u_i]; NaN where unobserved
            if values.ndim == 3:
                std = std[:, np.newaxis, :]  # [u_i]
                observed = observed[:, np.newaxis, :]
            # [value unit / u_i]; divided directly, so tiny standard
            # deviations cannot overflow a reciprocal. Unobserved entries
            # are exactly zero.
            weighted = np.zeros(values.shape)
            np.divide(values, std, out=weighted, where=observed)
            return weighted

        if x_mask is None:
            # [value unit / u_i]: whitened values of every row.
            weighted = np.dot(values.reshape(-1, num_measured), self.sigma_inv)
            return weighted.reshape(values.shape)

        # [value unit / u_i]; rows without observations stay zero.
        weighted = np.zeros_like(values, dtype=float)
        patterns, pattern_of_row = np.unique(x_mask, axis=0,
                                             return_inverse=True)
        for pattern_index, observed in enumerate(patterns):
            if not observed.any():
                continue

            if observed.all():
                root = self.sigma_inv  # rows [1/u_i], columns [-]
            else:
                key = tuple(observed.tolist())
                if key not in self._observed_roots:
                    marginal_covariance = self._measurement_covariance[
                        np.ix_(observed, observed)]  # [u_i * u_j]
                    self._observed_roots[key] = _measurement_precision_root(
                        marginal_covariance)
                # Rows [1/u_i] for the observed states, columns [-].
                root = self._observed_roots[key]

            rows = np.flatnonzero(pattern_of_row.ravel() == pattern_index)
            observed_values = values[rows][..., observed]  # [value unit]
            # [value unit / u_i], whitened observed entries.
            block = np.dot(observed_values.reshape(-1, observed.sum()), root)

            weighted_rows = weighted[rows]
            weighted_rows[..., observed] = block.reshape(
                observed_values.shape)
            weighted[rows] = weighted_rows

        return weighted

    def _residual_std_runs(self) -> list:
        """Return the per-experiment standard deviations used for weighting.

        Returns
        -------
        list
            ``_residual_std`` entries, shape ``(n_times, n_columns)`` in the
            measurement units, or one None per experiment when no
            uncertainty is declared.
        """
        if self._residual_std is None:
            return [None] * self.num_datasets
        return self._residual_std

    def _check_model_output(self, ind: int, output,
                            two_dimensional: bool = False) -> None:
        """Check the shape of one experiment's callback output.

        Parameters
        ----------
        ind : int
            Experiment index in ``experim_names`` order.
        output : numpy.ndarray
            Model states returned by the callback, in model units.
        two_dimensional : bool, optional
            Require a two-dimensional output, as curve resolution selects
            state columns. The default False also accepts one dimension.

        Raises
        ------
        ValueError
            If ``output`` does not have the allowed number of dimensions or
            one row per model-grid sample (the first-axis length of
            ``x_model[ind]``).
        """
        grid = self.x_model[ind]  # [x units]
        num_rows = 1 if np.ndim(grid) == 0 else np.shape(grid)[0]  # [-]
        shape = np.shape(output)
        allowed = (2,) if two_dimensional else (1, 2)
        if len(shape) not in allowed or shape[0] != num_rows:
            expected = ('a two-dimensional' if two_dimensional
                        else 'a one- or two-dimensional')
            raise ValueError(
                "The model callback returned an output of shape "
                f"{shape} for experiment {self.experim_names[ind]!r}; "
                f"expected {expected} array with {num_rows} rows, one per "
                "model-grid sample")

    def _select_output_columns(self, ind: int, output) -> list:
        """Resolve the model-output columns selected by the measurements.

        Parameters
        ----------
        ind : int
            Experiment index in ``experim_names`` order.
        output : numpy.ndarray
            Callback output of shape ``(n_times,)`` or
            ``(n_times, n_outputs)``, in model units.

        Returns
        -------
        list of int
            Non-negative output column per measurement column, also stored
            as ``_output_columns`` for the sensitivity selection.

        Raises
        ------
        ValueError
            If the column count differs from ``output_names`` or a
            measurement selects a column the output does not have.

        Notes
        -----
        The selector is resolved from the private copy of the fields taken
        at construction, so later changes to a caller's ``measured_ind``
        cannot make residuals and sensitivities select different columns.
        Negative legacy indices count from the last output column. Legacy
        input with a single measured column ignores its field when the
        output is one-dimensional, as earlier releases did.
        """
        experiment = self.experim_names[ind]
        # [-], model output columns; a 1-D output has one.
        num_columns = 1 if np.ndim(output) == 1 else np.shape(output)[1]
        if (self.output_names is not None
                and num_columns != len(self.output_names)):
            raise ValueError(
                f"The model callback returned {num_columns} output columns "
                f"for experiment {experiment!r}, but output_names declares "
                f"{len(self.output_names)}: {list(self.output_names)!r}")
        if np.ndim(output) == 1 and self._legacy_single_output:
            columns = [0]
        else:
            columns = []
            for name, field in zip(self._measurement_names,
                                   self._measured_fields):
                if not -num_columns <= field < num_columns:
                    raise ValueError(
                        f"Measurement {name!r} selects model output column "
                        f"{field}, but the model callback returned "
                        f"{num_columns} output column(s) for experiment "
                        f"{experiment!r}")
                columns.append(field % num_columns)
        self._output_columns = columns
        return columns

    @property
    def experiment_metadata(self) -> Optional[Mapping]:
        """Return a read-only mapping of experiment name to metadata dict.

        Returns
        -------
        collections.abc.Mapping or None
            Experiment name -> ``{'name', 'author', 'date'}`` mapping in
            ``experim_names`` order when constructed from ``Experiment``
            objects, or None for legacy array input.
        """
        if self._experiment_metadata is None:
            return None
        return MappingProxyType({
            name: MappingProxyType(meta)
            for name, meta in self._experiment_metadata.items()
        })

    def get_residual_layout(self) -> pd.DataFrame:
        """Describe each entry of the weighted residual vector.

        Returns
        -------
        pandas.DataFrame
            One row per weighted residual, in the order of
            ``get_objective(..., out_array=True)``: experiment order, then
            measurement column order, then model-grid sample order. Columns:
            ``'experiment'`` (name), ``'measurement'`` (column name),
            ``'field'`` (model-output column: after an evaluation, the
            non-negative column actually selected, which is 0 for legacy
            single-column input with a one-dimensional callback output;
            before any evaluation, the declared column resolved through
            ``output_names``, with negative legacy indices as given),
            ``'x'`` (model-grid entry in the model's x units; a row of a
            multi-dimensional legacy grid) and ``'observed'`` (bool;
            unobserved entries are exactly zero).
            Columns of ``info['jac']`` and rows of ``sens`` follow the same
            layout.

        Raises
        ------
        NotImplementedError
            For the nested ``'spectra'``/``'non_spectra'`` layout, whose
            residuals are not one entry per measurement and sample.
        """
        if self._measurement_names is None:
            raise NotImplementedError(
                "get_residual_layout is not available for nested state "
                "observation dictionaries or MultipleCurveResolution")

        fields = (self._measured_fields if self._output_columns is None
                  else self._output_columns)
        layout = {'experiment': [], 'measurement': [], 'field': [], 'x': [],
                  'observed': []}
        for name, x_model, x_mask in zip(self.experim_names, self.x_model,
                                         self.x_masks):
            # [x units], one entry (or legacy grid row) per sample.
            samples = [x_model] if np.ndim(x_model) == 0 else list(x_model)
            num_times = len(samples)  # [-]
            for column, (measurement, field) in enumerate(
                    zip(self._measurement_names, fields)):
                layout['experiment'] += [name] * num_times
                layout['measurement'] += [measurement] * num_times
                layout['field'] += [field] * num_times
                layout['x'] += samples
                layout['observed'] += (
                    [True] * num_times if x_mask is None
                    else x_mask[:, column].tolist())

        return pd.DataFrame(layout).astype({'field': int, 'observed': bool})

    def select_sens(self, sens_ordered, num_states, times=None):
        """Select the sensitivities of the measured model outputs.

        Parameters
        ----------
        sens_ordered : numpy.ndarray
            State-major sensitivities, shape ``(num_states * n_times,
            n_params)``, in state unit per parameter unit.
        num_states : int
            Number of model output columns.
        times : sequence of array_like, optional
            Row selection per measured output. The default None keeps all
            rows.

        Returns
        -------
        list of numpy.ndarray
            One ``(n_times, n_params)`` block per measurement column, in
            measurement column order.

        Notes
        -----
        Compiled layouts use the same private selector as the residuals
        (``_select_output_columns``), so the Jacobian is the derivative of
        the residuals even if a caller later changes its ``measured_ind``.
        """
        parts = np.split(sens_ordered, num_states, axis=0)

        if self._measured_fields is None:
            selector = self.measured_ind
        elif self._output_columns is not None:
            selector = self._output_columns
        else:
            selector = self._measured_fields

        if times is None:
            selected = [parts[ind] for ind in selector]
        else:
            selected = [parts[ind][times[count]]
                        for count, ind in enumerate(selector)]

        return selected

    def reconstruct_params(self, params):
        params_reconstr = np.zeros(self.num_params_total)
        params_reconstr[self.map_fixed] = self.param_seed[self.map_fixed]
        params_reconstr[self.map_variable] = params

        return params_reconstr

    def func_aux(self, params, x_vals, args, kwargs):
        """Evaluate model states for finite-difference sensitivities.

        Parameters
        ----------
        params : array_like
            Full parameter vector, including fixed and optimized entries.
            Units follow the model callback's parameter contract.
        x_vals : array_like
            Independent-variable samples passed to the model callback. Units
            follow the callback, typically time [s].
        args : tuple
            Positional arguments forwarded to the model callback.
        kwargs : dict
            Keyword arguments forwarded to the model callback.

        Returns
        -------
        states_flat : numpy.ndarray
            Model states flattened in state-major order. Units follow the model
            callback's state outputs.

        """
        states = self.function(params, x_vals, *args, **kwargs)

        return states.T.ravel()

    def get_objective(self, params, out_array=False, set_self=True):
        """Evaluate weighted residuals or their least-squares objective.

        Parameters
        ----------
        params : array_like
            Optimized parameter vector. Fixed parameters are reconstructed from
            ``param_seed`` before evaluating the model. Units follow the model
            callback's parameter contract.
        out_array : bool, optional
            If True, return weighted residuals flattened in state-major order.
            If False, return ``1/2 * weighted_residuals.T @
            weighted_residuals``. The default is False.
        set_self : bool, optional
            If True, store the latest model outputs, raw residuals in model
            units as ``residuals`` with shape ``(sum_times, n_measured)`` in
            data-major order, and the row-whitened residuals flattened
            state-major as ``weighted_residuals``: fully observed sample rows
            are weighted by ``sigma_inv``, partially observed rows by the root
            of ``inv(weight_matrix[o][:, o])`` for their observed states
            ``o``, and unobserved entries are zero (see
            ``_weight_sample_rows``). The default is True.

        Returns
        -------
        objective_or_residuals : float or numpy.ndarray
            Weighted scalar objective ``1/2 * r.T @ r`` when ``out_array`` is
            False, or the weighted residual vector ``r`` when ``out_array`` is
            True. Each entry of ``r`` has its measured state's unit divided by
            the square root of the matching ``weight_matrix`` unit: the
            measured-state unit with the default identity weights, and
            dimensionless [-] when ``weight_matrix`` holds measurement
            variances in squared state units. The objective has the squared
            unit of ``r``.

        Raises
        ------
        ValueError
            If the model output is not one- or two-dimensional with one row
            per model-grid sample, lacks a measured column, or its column
            count differs from ``output_names``.

        Notes
        -----
        ``r`` is packed by experiment, then measurement column, then
        model-grid sample; ``get_residual_layout`` lists the entries.
        With staggered observation grids or missing observations, unobserved
        model-grid entries keep their positions in ``r`` but are exactly
        zero, and each partially
        observed sample row is weighted by the marginal precision of its
        observed states (see ``_weight_sample_rows``), so the objective is
        ``1/2 * sum_k r_ok @ inv(weight_matrix[o_k][:, o_k]) @ r_ok`` over the
        observed residuals ``r_ok`` of each sample row. When measurements
        declare uncertainties, each observed entry is instead
        ``(model - data) / sigma`` [-], and the objective is half the sum of
        their squares.

        While ``objfun_iter`` is a list, each evaluation appends a sum of
        squares to it (the source of ``paramest_df['obj_fun']``). When
        measurements declare uncertainties this is the weighted sum of
        squares ``r @ r`` [-], twice the returned objective, because the raw
        residuals may mix physical units. Otherwise it remains the unweighted
        sum of squared raw residuals in squared measured-state units, as in
        earlier releases, also when ``weight_matrix`` is given.

        """
        # Store parameter values
        params_in = np.asarray(params)  # parameter units
        if type(self.params_iter) is list:
            self.params_iter.append(params)

        # Reconstruct parameter set with fixed and non-fixed indexes
        params = self.reconstruct_params(params)

        # --------------- Solve
        y_runs = []
        resid_runs = []
        sens_second = []

        for ind in range(self.num_datasets):
            # Solve model
            result = self.function(params, self.x_model[ind],
                                   *self.args_fun[ind], **self.kwargs_fun[ind])

            if isinstance(result, (tuple, list)):  # func also returns the jacobian
                y_prof, sens = result

                sens_second.append(sens)

            else:  # call a separate function for jacobian
                y_prof = result

            self._check_model_output(ind, y_prof)
            if self._measured_fields is not None:
                columns = self._select_output_columns(ind, y_prof)
                # [model state units]; one column for 1-D outputs.
                y_run = y_prof.reshape(len(y_prof), -1)[:, columns]
                self.num_model_states = 1 if y_prof.ndim == 1 else (
                    y_prof.shape[1])

            elif y_prof.ndim == 1:
                y_run = y_prof.reshape(-1, 1)
                self.num_model_states = 1

            else:
                y_run = y_prof[:, self.measured_ind]
                self.num_model_states = y_prof.shape[1]

            # [model state units]; residuals of unobserved entries are set
            # to exactly zero, whatever the observation dtype (writing model
            # values into a float32 copy would round them).
            x_mask = self.x_masks[ind]
            resid_run = y_run - self.y_data[ind]
            if x_mask is not None:
                resid_run[~x_mask] = 0

            # Store
            y_runs.append(y_run)
            resid_runs.append(resid_run)

        # [measured-state unit / weight_matrix unit**0.5]; [-] only with
        # state-variance weights. Unobserved entries are exactly zero.
        weighted_residuals = [
            self._weight_sample_rows(resid, x_mask, residual_std)
            for resid, x_mask, residual_std in zip(
                resid_runs, self.x_masks, self._residual_std_runs())]

        if len(sens_second) > 0:
            self.sens_second = sens_second
            # [parameter units], parameters at which sens_second was evaluated.
            self._params_sens_second = np.array(params_in, copy=True)
        else:
            self.sens_second = None
            self._params_sens_second = None

        # Weighted-residual units as above, flattened state-major.
        residual_out = np.concatenate([ar.T.ravel()
                                       for ar in weighted_residuals])

        if type(self.objfun_iter) is list:
            if self._residual_std is None:
                # [(measured-state unit)**2], raw SSE kept for legacy history
                objfun_val = np.linalg.norm(np.concatenate(resid_runs))**2
            else:
                # [-], weighted SSE; raw residuals may mix physical units.
                objfun_val = np.dot(residual_out, residual_out)
            self.objfun_iter.append(objfun_val)

        residuals = self.optimize_flag * np.concatenate(resid_runs)

        if set_self:
            self.y_runs = y_runs
            self.resid_runs = resid_runs

            self.residuals = residuals
            self.params_residuals = np.array(params_in, copy=True)
            self.weighted_residuals = self.optimize_flag * residual_out

        # Return objective
        if out_array:
            return residual_out
        else:
            residual_out = 1/2 * np.dot(residual_out, residual_out)

        return residual_out

    def get_gradient(self, params, out_array=False, set_self=True):
        """Assemble residual Jacobians or objective gradients.

        Parameters
        ----------
        params : array_like
            Optimized parameter vector. Fixed parameters are reconstructed from
            ``param_seed`` before evaluating ``jac_fun`` or numerical finite
            differences. Units follow the parameter definitions supplied to
            the model.
        out_array : bool, optional
            If True, return the weighted residual Jacobian transposed as
            ``(n_params, n_data)``. If False, return the scalar-objective
            gradient used by IPOPT. The default is False.
        set_self : bool, optional
            Reserved for compatibility with optimizer callback signatures. The
            current implementation stores the assembled weighted Jacobian on
            ``self.sens`` regardless of this value.

        Returns
        -------
        jacobian_or_gradient : numpy.ndarray
            Weighted residual Jacobian, shape ``(n_params, n_data)``, when
            ``out_array`` is True: row ``p`` holds derivatives, each in its
            weighted residual's unit divided by the unit of optimized
            parameter ``p``. Objective gradient, shape ``(n_params,)``, when
            ``out_array`` is False: entry ``p`` has the objective unit divided
            by the unit of parameter ``p``. Both reduce to reciprocal-parameter
            units only when ``weight_matrix`` holds measurement variances in
            squared state units; ``get_objective`` documents the residual
            units.

        Raises
        ------
        ValueError
            If the finite-difference path is used (no ``jac_fun`` and no
            model-returned sensitivities) and ``dx_finitediff`` gives a
            zero or infinite step for an optimized parameter.

        Notes
        -----
        Analytical (``jac_fun`` or model-returned) and finite-difference
        model sensitivities are weighted row by row exactly like the
        residuals, including division by declared measurement standard
        deviations. Columns follow ``get_residual_layout``. Columns of
        unobserved model-grid entries are therefore
        zero and contribute no information to ``jac @ jac.T`` or to the
        parameter covariance.

        """
        sens_stale = (
            self.sens_second is not None and (
                self._params_sens_second is None or
                not np.array_equal(params, self._params_sens_second))
        )
        if not out_array and (
                self.params_residuals is None or
                not np.array_equal(params, self.params_residuals) or
                sens_stale):
            self.get_objective(params)
        elif out_array and (self.num_model_states is None or sens_stale):
            self.get_objective(params, out_array=True, set_self=set_self)

        if self.sens_second is None:
            raw_sens = []
            # [parameter units], full parameter vector with fixed entries.
            params_full = self.reconstruct_params(params)
            pick_p = np.where(self.map_variable)[0]
            for ind in range(self.num_datasets):
                if self.jac_fun is None:
                    pass_to_fun = (self.x_model[ind], self.args_fun[ind],
                                   self.kwargs_fun[ind])

                    sens = numerical_jac_data(self.func_aux, params_full,
                                              pass_to_fun, dx=self.dx_fd,
                                              pick_x=pick_p)
                else:
                    sens = self.jac_fun(params_full, self.x_model[ind],
                                        *self.args_fun[ind],
                                        **self.kwargs_fun[ind])

                raw_sens.append(sens)

        else:
            raw_sens = self.sens_second

        weighted_sens = []
        for sensit, x_model, x_mask, residual_std in zip(
                raw_sens, self.x_model, self.x_masks,
                self._residual_std_runs()):
            sensit = self.select_sens(sensit, self.num_model_states)

            # (n_params * n_times, n_measured), parameter-major rows; this
            # also accepts the legacy one-state (1, n_times) jac_fun layout.
            sens_by_y = reorder_sens(sensit)
            num_measured = sens_by_y.shape[1]
            # (n_times, n_params, n_measured) [state unit / parameter unit]
            sens_by_row = sens_by_y.reshape(
                -1, len(x_model), num_measured).transpose(1, 0, 2)
            # Same row weighting as the residuals, so unobserved entries have
            # zero sensitivity and the result is the Jacobian of the
            # weighted residuals returned by get_objective.
            # (n_times, n_params, n_measured) [weighted residual unit /
            # parameter unit]
            weighted = self._weight_sample_rows(sens_by_row, x_mask,
                                                residual_std)
            # (n_measured * n_times, n_params), same units; state-major rows
            # (state, time) matching the residual vector.
            weighted = weighted.transpose(2, 0, 1).reshape(
                -1, weighted.shape[1])

            weighted_sens.append(weighted)

        concat_sens = np.vstack(weighted_sens)
        if not self.fit_spectra:

            if len(self.map_variable) == concat_sens.shape[1]:
                concat_sens = concat_sens[:, self.map_variable]

        self.sens = concat_sens
        jacobian = concat_sens

        if out_array:
            return jacobian.T  # LM doesn't require (y - y_e)^T J
        else:
            res = self.weighted_residuals
            gradient = jacobian.T.dot(res)  # 1D
            return gradient

    def get_cond_number(self, sens_matrix):
        _, sing_vals, _ = np.linalg.svd(sens_matrix)

        cond_number = max(sing_vals) / min(sing_vals)

        return cond_number

    def assemble_solver_info(self, opt_par):
        """Assemble weighted residual and Jacobian data at a solved point.

        Parameters
        ----------
        opt_par : numpy.ndarray
            Solved parameter vector. Each entry uses the unit declared by its
            corresponding model parameter.

        Returns
        -------
        dict
            ``fun`` contains the row-whitened residuals of
            ``get_objective``, model output minus data, in the basis of the
            measured states: ``sigma_inv`` weights fully observed sample
            rows, the root of ``inv(weight_matrix[o][:, o])`` weights rows
            observing only states ``o``, and unobserved entries are zero
            (see ``_weight_sample_rows``). Each
            entry's unit is its measured state's unit divided by the square
            root of the matching ``weight_matrix`` unit. With the default
            identity ``weight_matrix`` each entry keeps its measured state's
            unit, for example [mol/L]; entries are dimensionless [-] only when
            ``weight_matrix`` holds measurement variances in squared state
            units. ``jac`` has shape ``(n_params, n_data)``, and row ``p``
            holds the derivatives of ``fun`` with respect to optimized
            parameter ``p``, each in its ``fun`` entry's unit divided by that
            parameter's unit.

        Notes
        -----
        Evaluating the objective with ``set_self=False`` preserves the raw
        residuals stored by the final solver callback (issue #78).
        """
        # [measured-state unit / weight_matrix unit**0.5]; the measured-state
        # unit with identity weights, [-] with state-variance weights.
        residuals = self.get_objective(
            opt_par, out_array=True, set_self=False
        )
        # [residuals unit / optimized-parameter unit] for each row.
        jacobian = self.get_gradient(opt_par, out_array=True)
        return {'jac': jacobian, 'fun': residuals}

    def _data_solver_info(self, info: dict) -> dict:
        """Return the solver information reported for the measured data.

        Parameters
        ----------
        info : dict
            Solver information at the accepted parameters, with ``fun`` the
            residual vector the solver minimized and ``jac`` its Jacobian,
            shape ``(num_params, len(fun))``, in the units documented by
            ``assemble_solver_info``.

        Returns
        -------
        dict
            ``info`` itself: every residual of ``get_objective`` is a data
            residual. ``optimize_fn`` stores the result as ``info_opt``, the
            input of ``get_covariance`` and ``StatisticsClass``.

        Notes
        -----
        Subclasses whose solver residual vector appends non-data rows, such
        as the absorptivity penalty of ``MultipleCurveResolution``, override
        this to report the data rows only.
        """
        return info

    def _model_predictions(self) -> list:
        """Return the model predictions of the last stored evaluation.

        Returns
        -------
        list of numpy.ndarray
            One array per experiment in ``x_data`` order, shape
            ``(n_times, n_measured)``: model-grid rows and measured-state
            columns in the model state units, ``resid_runs + y_data`` because
            the residuals are model minus data. Entries without an
            observation are NaN, as in ``y_data``.

        Notes
        -----
        ``optimize_fn`` stores the result as ``y_model``, the nominal
        response of ``StatisticsClass``. Subclasses whose residuals are not
        model minus data, such as ``MultipleCurveResolution``, override this.
        """
        predictions = []  # [model state units], one array per experiment
        for ind in range(self.num_datasets):
            # [model state units], NaN where unobserved
            y_data = self.y_data[ind]
            if isinstance(y_data, dict):
                y_data = np.hstack(list(y_data.values()))

            y_model = self.resid_runs[ind] + y_data  # [model state units]
            predictions.append(y_model)

        return predictions

    def _residual_signs(self, ind: int) -> Optional[np.ndarray]:
        """Return the sign convention of one experiment's residual columns.

        Parameters
        ----------
        ind : int
            Experiment index in ``x_data`` order.

        Returns
        -------
        numpy.ndarray or None
            None: every column of ``resid_runs[ind]`` is model minus data.
            Subclasses return one sign per column [-], +1 for model minus
            data and -1 for data minus model; ``StatisticsClass``
            multiplies the residuals by it to pool model-minus-data
            residuals.
        """
        return None

    def _fit_state(self) -> dict:
        """Save the attributes that describe the last completed fit.

        Returns
        -------
        dict
            The object bound to each name in ``_FIT_ATTRIBUTES``, or
            ``_ABSENT`` if the attribute does not exist. Values keep their
            units (``params_convg`` in parameter units, ``covar_params`` in
            their products, and so on).

        Notes
        -----
        Saving references suffices: ``optimize_fn`` binds new iteration
        history lists after saving, and evaluations bind new buffers
        instead of modifying the saved objects in place.
        """
        return {name: getattr(self, name, _ABSENT)
                for name in self._FIT_ATTRIBUTES}

    def _restore_fit_state(self, state: dict) -> None:
        """Restore the attributes saved by ``_fit_state``.

        Parameters
        ----------
        state : dict
            Result of ``_fit_state``. Each saved object is rebound, and
            attributes saved as ``_ABSENT`` are deleted if they exist now.
        """
        for name, value in state.items():
            if value is not _ABSENT:
                setattr(self, name, value)
            elif hasattr(self, name):
                delattr(self, name)

    def optimize_fn(self, optim_options: Optional[dict] = None,
                    simulate: bool = False, verbose: int | bool = True,
                    store_iter: bool = True, method: str = 'LM',
                    bounds: Optional[Sequence] = None
                    ) -> tuple[np.ndarray, np.ndarray, dict]:
        """Optimize variable parameters and assemble fit statistics.

        Parameters
        ----------
        optim_options : dict, optional
            Solver options passed to the selected optimization method. The
            mapping and its values are not modified: ``optimize_fn`` passes
            a copy, with deep copies of the non-callable values, to the
            solver, with the entries it sets itself (``full_output`` and
            ``verbose`` for LM, ``print_level`` for IPOPT, ``verbose`` and,
            if given, ``bounds`` for ``'scipy'``/``'least_squares'``).
            After a successful call that copy, as the solver left it, is
            stored as ``optim_options``, the effective options of the last
            fit; for IPOPT, cyipopt has converted its keys to bytes and
            added its own defaults, such as ``mu_strategy``.
        simulate : bool, optional
            If True, evaluate residuals in simulation mode by disabling the
            optimization residual contribution.
        verbose : int or bool, optional
            If True (or a positive int for ``'scipy'``/``'least_squares'``),
            request verbose solver output when supported.
        store_iter : bool, optional
            If True, store unique parameter/objective iterates in
            ``params_iter``, ``objfun_iter`` and the ``paramest_df`` table,
            whose ``'obj_fun'`` column is the weighted sum of squares [-]
            (twice the objective) when measurements declare uncertainties,
            and otherwise the unweighted sum of squared raw residuals in
            squared measured-state units, also with ``weight_matrix`` (see
            ``get_objective``). Every call starts new histories, so they
            describe this fit only. If False, ``params_iter`` and
            ``objfun_iter`` keep every evaluation of this call, in order,
            and no ``paramest_df`` is stored.
        method : {'LM', 'IPOPT', 'scipy', 'least_squares'}, optional
            Optimization method used for fitting. ``'scipy'`` and
            ``'least_squares'`` both select
            ``scipy.optimize.least_squares``. The default is ``'LM'``.
        bounds : sequence, optional
            Parameter bounds in the model parameter units, passed to IPOPT or
            ``scipy.optimize.least_squares``.

        Returns
        -------
        opt_par : numpy.ndarray
            Accepted variable parameters, shape ``(num_params,)``, in the
            model callback's parameter units.
        covar_params : numpy.ndarray
            Estimated covariance, shape ``(num_params, num_params)``. Entry
            ``(i, j)`` has the product of parameter i and parameter j units.
            Filled with NaN, with a ``RuntimeWarning`` stating the cause,
            when it is undefined, for example because the parameters are not
            (locally) identifiable from the observed data (see
            ``get_covariance``); the fit is kept.
        info : dict
            Solver information at the accepted parameters. ``info['fun']`` is
            the weighted residual vector, ordered by experiment then
            measurement column then model-grid sample (see
            ``get_residual_layout``), in the units documented by
            ``assemble_solver_info``: measured-state units with the default
            identity weights and [-] with state-variance weights or declared
            measurement uncertainties. ``info['jac']`` has shape
            ``(num_params, len(info['fun']))`` across all optimizers; each
            entry has its ``fun`` entry's unit divided by the matching
            parameter unit.
            With staggered measurement grids or missing observations, columns
            include unobserved model-grid entries, so their count can exceed
            ``num_data_total``;
            those entries of ``info['fun']`` and columns of ``info['jac']``
            are exactly zero, so they add no residual or information.
            LM and ``scipy.optimize.least_squares`` additionally supply their
            accepted ``x`` and native solver diagnostics (with ``info['jac']``
            stored in the canonical ``(num_params, n_data)`` orientation and,
            for ``'scipy'``/``'least_squares'``, the raw
            ``scipy.optimize.OptimizeResult`` stored in
            ``info['scipy_result']``). ``info['fun']`` and ``info['jac']``
            hold data residuals only (see ``_data_solver_info``):
            ``MultipleCurveResolution`` moves its absorptivity-penalty rows to
            ``info['fun_penalty']`` and ``info['jac_penalty']``, while native
            solver diagnostics such as SciPy's ``cost`` and ``grad`` describe
            the full residual vector.
            IPOPT fits add cyipopt's termination outcome: ``info['success']``
            (bool, True only for Ipopt status 0, ``Solve_Succeeded``),
            ``info['status']`` (int, Ipopt's return status),
            ``info['message']`` (str, Ipopt's status message) and, when
            cyipopt reports it, ``info['nit']`` (int, iterations [-]).
            Every method reports ``info['success']`` (bool) and
            ``info['message']`` (str). For LM, ``success`` is True when
            ``info['stop_criterion']`` is a convergence test
            (``'Small step'``, ``'Small gradient'`` or ``'Value of the
            objective function lower than tolerance'``) and False for
            ``'Maximum iterations exceeded'``; ``message`` repeats the stop
            criterion. With ``max_fun_eval=0`` LM evaluates the seed only,
            reports no ``stop_criterion``, and ``success`` is False with a
            message saying that no iterations were requested. For
            ``'scipy'``/``'least_squares'``, ``success``, ``status`` and
            ``message`` are SciPy's (``success`` True for ``status > 0``).
            ``info['tolerances']`` (dict) holds the stopping criteria that
            governed the fit with their effective values, the given options
            merged over the optimizer defaults: ``max_fun_eval`` [-],
            ``eps_1`` [residual units**2 / parameter units], ``eps_2`` [-]
            and ``tol_fun`` [residual units] for LM (defaults read from
            ``levenberg_marquardt``'s signature); ``ftol`` [-], ``xtol``
            [-], ``gtol`` and ``max_nfev`` [-] for SciPy (defaults read from
            ``scipy.optimize.least_squares``'s signature; ``max_nfev=None``
            means SciPy's automatic limit); and, for IPOPT, only the
            termination options present in the solver's options, with
            ``str`` keys: the caller's and cyipopt's ``tol`` default, not
            Ipopt's native defaults (see ``_stopping_tolerances``).

        Raises
        ------
        ImportError
            If IPOPT is selected without the optional cyipopt dependency.
        ValueError
            If ``method`` is not one of ``'LM'``, ``'IPOPT'``, ``'scipy'`` or
            ``'least_squares'``, if ``bounds`` is supplied both through
            ``bounds=`` and ``optim_options``, or if ``loss`` is not
            ``'linear'`` when using ``'scipy'``/``'least_squares'``, or if
            finite-difference sensitivities are used and ``dx_finitediff``
            gives a zero or infinite step for an optimized parameter (see
            ``get_gradient``).

        Warns
        -----
        RuntimeWarning
            If IPOPT terminates unsuccessfully (``info['success']`` False),
            for example at its iteration limit. The warning names the status
            and message; the returned parameters, covariance and statistics
            then describe the last iterate, not a converged estimate.
            Likewise if LM stops with ``'Maximum iterations exceeded'`` or
            ``scipy.optimize.least_squares`` without success (e.g. status
            0 at ``max_nfev``, or status -2 after a callback stop): the
            warning names the method, the stop reason (SciPy's status and
            message) and the evaluation counts, and states that the results
            describe the last accepted iterate, not a converged estimate.
            Only when the evaluation budget stopped the fit (LM's
            ``'Maximum iterations exceeded'``, SciPy status 0) does it name
            the limiting option with its effective value (``max_fun_eval``
            for LM, ``max_nfev`` for SciPy) and suggest raising it.
            The returned parameters, covariance and statistics are those of
            the solver, unchanged. No warning is issued for the
            evaluation-only LM call ``max_fun_eval=0``. These warnings point
            at the caller outside PharmaPy, also through
            ``estimate_parameters`` or ``SimulationExec.EstimateParams``.
            Also, from ``get_covariance``, if the covariance at the accepted
            parameters is undefined. The warning gives one of three causes:
            a non-finite weighted residual Jacobian (naming those
            parameters); a singular or numerically singular ``J @ J.T``,
            i.e. parameters that the observed data do not (locally)
            identify (naming them and the undetermined parameter
            directions); or an inversion of ``J @ J.T`` that failed or gave
            non-finite or negative variances although it passed the
            conditioning test (naming those parameters and the
            least-determined direction). The fit is stored and returned
            with an all-NaN ``covar_params`` and ``correl_params``. Earlier
            releases raised ``numpy.linalg.LinAlgError`` for an exactly
            singular ``J @ J.T`` and kept the previous fit, or returned
            meaningless values such as negative variances (#532).

        Notes
        -----
        After LM or ``scipy.optimize.least_squares`` finishes, the model is
        evaluated once per experiment at the accepted parameters to refresh
        ``y_runs``, ``resid_runs``, ``residuals``, and ``weighted_residuals``
        before assembling ``y_model`` (``_model_predictions``). This
        includes termination after a rejected trial; it does not imply
        convergence. Predictions and raw residuals retain the model state
        units and measured-state order. Repeated calls replace the previous
        ``y_model`` list.

        The reporting evaluation does not change the solver's returned ``x``,
        ``fun``, ``jac``, or solver counters. It uses the usual objective
        callback and its history recording; with ``store_iter=True`` duplicate
        parameter entries are removed as usual. Stateful callbacks must
        support another evaluation at the same parameters, as during
        optimization.

        ``method`` and the availability of cyipopt are validated before any
        attribute changes. ``opt_method``, ``optim_options``,
        ``params_convg`` and ``info_opt`` are assigned together after the
        solver returns, so they always describe the same fit.
        ``optimize_flag`` is set to ``not simulate`` during the solve,
        because ``get_objective`` reads it. A call that raises, during the
        solve or afterwards (for example when a ``get_covariance`` warning
        is turned into an error by a warnings filter), restores every
        attribute named in the class's
        ``_FIT_ATTRIBUTES`` to its value before the call, or removes it if a
        first fit had not yet created it: the fit results (``opt_method``,
        ``optim_options``, ``optimize_flag``, ``params_convg``,
        ``info_opt``, ``y_model``, ``covar_params``, ``correl_params``), the
        iteration histories (``params_iter``, ``objfun_iter``,
        ``cond_number``, ``paramest_df``), the evaluation buffers that
        ``get_objective`` and ``get_gradient`` overwrite (``y_runs``,
        ``resid_runs``, ``residuals``, ``weighted_residuals``, ``sens`` and
        the others listed there), and the attributes subclasses add, such as
        ``epsilon_mcr`` and the projections of
        ``MultipleCurveResolution``, which also clears its evaluation cache
        (see ``_restore_fit_state``).

        """
        if method not in OPTIMIZATION_METHODS:
            raise ValueError(
                f"Unknown optimization method {method!r}; expected 'LM', "
                "'IPOPT', 'scipy' or 'least_squares'")
        if method == 'IPOPT' and not have_cyipopt:
            raise ImportError(
                'cyipopt is an optional import. Please install cyipopt to '
                'use IPOPT as a solver for parameter estimation. conda '
                'install -c conda-forge cyipopt')

        params_var = self.param_seed[self.map_variable]  # [model parameter units]

        # Effective options of this fit: a private deep copy, so the entries
        # set below and cyipopt's in-place rewrite never reach the caller,
        # and later in-place edits of the caller's values (e.g. a d_diag
        # array) never reach the stored options or info_opt (#476).
        # Callable values are kept as given: deepcopy would copy the object
        # bound to a method.
        given_options = {} if optim_options is None else dict(optim_options)
        effective_options = {
            key: value if callable(value) else copy.deepcopy(value)
            for key, value in given_options.items()}

        previous_fit = self._fit_state()
        # This fit's iteration history (#479): get_objective appends to these
        # lists only, so arrays left by an earlier fit would keep its history.
        self.params_iter = []
        self.objfun_iter = []
        self.cond_number = []
        if hasattr(self, 'paramest_df'):
            del self.paramest_df
        self.optimize_flag = not simulate
        try:
            if method == 'LM':
                effective_options['full_output'] = True
                effective_options['verbose'] = verbose

                opt_par, inv_hessian, info = levenberg_marquardt(
                    params_var,
                    self.get_objective,
                    self.get_gradient,
                    args=(True,),
                    **effective_options)

                # Rejected trial callbacks overwrite model buffers. Refresh
                # them at the accepted point while retaining native LM result
                # metadata.
                self.get_objective(opt_par, out_array=True)

                if 'stop_criterion' not in info:
                    # max_fun_eval=0: an explicit evaluation at the seed,
                    # not a fit, so it is reported but not warned about.
                    info['success'] = False
                    info['message'] = (
                        'No iterations were requested (max_fun_eval=0); '
                        'the parameters are the seed, evaluated only')
                else:
                    info['success'] = (info['stop_criterion']
                                       in _LM_CONVERGED_STOPS)
                    info['message'] = str(info['stop_criterion'])
                    if not info['success']:
                        budget = (
                            ('max_fun_eval', _stopping_tolerances(
                                'LM', effective_options)['max_fun_eval'])
                            if info['stop_criterion'] == _LM_BUDGET_STOP
                            else (None, None))
                        _warn_unconverged(
                            "Levenberg-Marquardt (method='LM')",
                            info['message'],
                            f"{info['num_fun_eval']} trial evaluations and "
                            f"{info['num_iter']} accepted steps",
                            *budget)

            elif method == 'IPOPT':
                effective_options['print_level'] = (
                    int(verbose) * IPOPT_VERBOSE_PRINT_LEVEL)  # [-]

                kwargs_fun = {'out_array': False}
                result = minimize_ipopt(self.get_objective, params_var,
                                        jac=self.get_gradient,
                                        bounds=bounds,
                                        options=effective_options,
                                        kwargs=kwargs_fun)

                opt_par = result['x']

                info = self.assemble_solver_info(opt_par)
                # Keep the termination outcome (#475); cyipopt returns the
                # message as bytes.
                message = result['message']
                if isinstance(message, bytes):
                    message = message.decode('utf-8', errors='replace')
                info['success'] = bool(result['success'])
                info['status'] = int(result['status'])
                info['message'] = message
                if 'nit' in result:
                    info['nit'] = int(result['nit'])  # [-]
                if not info['success']:
                    warnings.warn(
                        f"IPOPT terminated unsuccessfully (status "
                        f"{info['status']}: {message}); the returned "
                        "parameters, covariance and statistics describe its "
                        "last iterate, not a converged estimate",
                        RuntimeWarning, stacklevel=_user_stacklevel())

            else:  # 'scipy' or 'least_squares'
                scipy_options = effective_options
                if 'bounds' in scipy_options and bounds is not None:
                    raise ValueError(
                        "Pass bounds either through bounds= or optim_options, "
                        "not both")
                loss = scipy_options.get('loss', 'linear')
                if loss != 'linear':
                    raise ValueError(
                        "Only loss='linear' is supported for least-squares "
                        f"parameter estimation and covariance; got {loss!r}")
                if bounds is not None:
                    scipy_options['bounds'] = copy.deepcopy(bounds)
                if 'verbose' in scipy_options and verbose is True:
                    scipy_options['verbose'] = int(scipy_options['verbose'])
                else:
                    scipy_options['verbose'] = int(verbose)

                def _scipy_jac(params_eval: np.ndarray,
                               *args_eval) -> np.ndarray:
                    """Return weighted residual Jacobian in SciPy orientation.

                    Parameters
                    ----------
                    params_eval : numpy.ndarray
                        Variable parameter vector, shape ``(num_params,)``,
                        in the model callback's parameter units.
                    *args_eval
                        Positional arguments forwarded to ``get_gradient``
                        (``out_array=True``).

                    Returns
                    -------
                    numpy.ndarray
                        Weighted residual Jacobian, shape
                        ``(n_data, num_params)``. Entry ``(k, i)`` is in the
                        unit of weighted residual ``k`` divided by the unit of
                        variable parameter ``i``.
                    """
                    return self.get_gradient(params_eval, *args_eval).T

                result = least_squares(
                    self.get_objective,
                    params_var,
                    jac=_scipy_jac,
                    args=(True,),
                    **scipy_options)

                opt_par = np.asarray(result.x)  # [model parameter units]

                # Rejected trial callbacks overwrite model buffers. Refresh
                # them at the accepted point while retaining native solver
                # metadata and canonical (num_params, n_data) Jacobian
                # orientation.
                self.get_objective(opt_par, out_array=True)
                info = dict(result)
                info['scipy_result'] = result
                # [residual unit / parameter unit], shape (num_params, n_data).
                info['jac'] = np.asarray(result.jac).T
                info['success'] = bool(result.success)
                info['message'] = str(result.message)
                if not info['success']:
                    budget = (
                        ('max_nfev', _stopping_tolerances(
                            method, effective_options)['max_nfev'])
                        if result.status == _SCIPY_BUDGET_STATUS
                        else (None, None))
                    _warn_unconverged(
                        "scipy.optimize.least_squares "
                        f"(method={method!r})",
                        f"status {int(result.status)}: {info['message']}",
                        f"{result.nfev} residual and {result.njev} Jacobian "
                        "evaluations",
                        *budget)

            info['tolerances'] = _stopping_tolerances(method,
                                                      effective_options)

            # Statistics use the data residuals; see _data_solver_info.
            info = self._data_solver_info(info)
            if method in ('scipy', 'least_squares'):
                # (n_data, num_params) [residual unit / parameter unit]
                self.sens = info['jac'].T

            # Store
            self.opt_method = method
            self.optim_options = effective_options
            self.params_convg = opt_par  # [model parameter units]
            self.info_opt = info

            self.cond_number = np.array(self.cond_number)
            self.params_iter = np.array(self.params_iter)
            _, idx = np.unique(self.params_iter, axis=0, return_index=True)

            if store_iter:
                self.params_iter = self.params_iter[np.sort(idx)]
                self.objfun_iter = np.array(self.objfun_iter)[np.sort(idx)]

                col_names = ['obj_fun'] + self.name_params
                self.paramest_df = pd.DataFrame(
                    np.column_stack((self.objfun_iter, self.params_iter)),
                    columns=col_names)

            # Model prediction with final parameters, [model state units], one
            # array per experiment; see _model_predictions.
            self.y_model = self._model_predictions()

            covar_params = self.get_covariance()
        except BaseException:
            # Keep the previous fit, also when the post-solve assembly fails,
            # e.g. an undefined-covariance warning raised as an error
            # (#479, #532).
            self._restore_fit_state(previous_fit)
            raise

        return opt_par, covar_params, info

    def estimate_parameters(
            self, optim_options: Optional[dict] = None,
            simulate: bool = False, verbose: int | bool = True,
            store_iter: bool = True, method: str = 'scipy',
            bounds: Optional[Sequence] = None, **solver_options
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        """Estimate variable parameters with ``scipy.optimize.least_squares``.

        Parameters
        ----------
        optim_options : dict, optional
            Solver options passed to ``optimize_fn``. Additional keyword
            arguments in ``**solver_options`` are merged into this mapping.
        simulate : bool, optional
            If True, evaluate residuals in simulation mode by disabling the
            optimization residual contribution. The default is False.
        verbose : int or bool, optional
            Verbosity forwarded to the selected optimizer (``0``, ``1`` or
            ``2`` for ``scipy.optimize.least_squares``; bool for ``'LM'`` and
            ``'IPOPT'``). The default is True.
        store_iter : bool, optional
            If True, store unique parameter/objective iterates in
            ``params_iter``, ``objfun_iter`` and ``paramest_df``. The default
            is True.
        method : {'scipy', 'least_squares', 'LM', 'IPOPT'}, optional
            Optimization method forwarded to ``optimize_fn``. The default is
            ``'scipy'`` (``scipy.optimize.least_squares``).
        bounds : sequence, optional
            Parameter bounds in the model parameter units, forwarded to
            ``optimize_fn``.
        **solver_options
            Additional keyword options forwarded to the optimizer (for
            example ``max_nfev=100``, ``ftol=1e-10``, ``xtol=1e-10``).

        Returns
        -------
        opt_par : numpy.ndarray
            Accepted variable parameters, shape ``(num_params,)``, in the
            model callback's parameter units.
        covar_params : numpy.ndarray
            Estimated covariance, shape ``(num_params, num_params)``. Entry
            ``(i, j)`` has the product of parameter i and parameter j units.
            All NaN, with a ``RuntimeWarning``, when it is undefined, for
            example because the parameters are not identifiable (see
            ``optimize_fn``).
        info : dict
            Solver information at the accepted parameters, with ``info['jac']``
            in the canonical ``(num_params, len(info['fun']))`` orientation
            documented by ``optimize_fn``, the termination outcome
            ``info['success']`` and ``info['message']``, and the effective
            stopping criteria ``info['tolerances']``, which include the
            options given here through ``optim_options`` or
            ``**solver_options``.

        Raises
        ------
        TypeError
            If ``optim_options`` is neither a mapping nor None.
        ValueError
            If the same option is given in both ``optim_options`` and
            ``**solver_options``, or if ``optimize_fn`` rejects ``method`` or
            ``bounds``.

        Warns
        -----
        RuntimeWarning
            If the optimizer stops without converging, or the covariance is
            undefined; see ``optimize_fn``.
        """
        if optim_options is not None and not isinstance(optim_options, Mapping):
            raise TypeError(
                f"optim_options must be a mapping or None; got "
                f"{type(optim_options).__name__}")
        if optim_options is not None and solver_options:
            duplicates = sorted(set(optim_options).intersection(solver_options))
            if duplicates:
                raise ValueError(
                    "Duplicate solver options in optim_options and keyword "
                    f"arguments: {duplicates!r}")
            merged_options = {**optim_options, **solver_options}
        elif optim_options is not None:
            merged_options = dict(optim_options)
        elif solver_options:
            merged_options = dict(solver_options)
        else:
            merged_options = None

        return self.optimize_fn(
            optim_options=merged_options, simulate=simulate, verbose=verbose,
            store_iter=store_iter, method=method, bounds=bounds)

    def get_confidence_intervals(
            self, alpha: float = 0.95,
            covar_params: Optional[np.ndarray] = None,
            include_mse: Optional[bool] = None, verbose: bool = True,
            set_self: bool = True) -> dict:
        """Compute Student-t parameter confidence intervals at the fitted point.

        Delegates to ``StatisticsClass(self, alpha=alpha).get_intervals(...)``
        so the intervals use the same linearized covariance and degrees of
        freedom ``num_data_total - num_params`` as ``get_covariance`` and
        ``StatisticsClass``.

        Parameters
        ----------
        alpha : float, optional
            Two-sided confidence level [-] in the open interval ``(0, 1)``.
            The default is 0.95 (95% confidence intervals).
        covar_params : numpy.ndarray, optional
            Parameter covariance matrix, shape ``(num_params, num_params)``,
            with entry ``(i, j)`` in the product of parameter i and parameter
            j units. If None (the default), ``self.covar_params`` is reused
            when ``include_mse`` is None and a covariance has already been
            computed; otherwise ``get_covariance(include_mse=...)`` is used
            (defaulting to ``include_mse=True``).
        include_mse : bool, optional
            If given while ``covar_params`` is None, recompute the covariance
            with ``get_covariance(include_mse=include_mse)``. Cannot be
            combined with ``covar_params``. The default is None, which reuses
            ``self.covar_params`` if present and otherwise computes
            ``get_covariance(include_mse=True)``.
        verbose : bool, optional
            If True (the default), print the formatted parameter interval
            table.
        set_self : bool, optional
            If True (the default), store ``confid_intervals``,
            ``intervals_array`` and ``stats`` on ``self`` (and on ``stats``).

        Returns
        -------
        dict
            Mapping from each optimized parameter name in ``name_params`` to
            its two-element ``numpy.ndarray`` ``[lower_bound, upper_bound]``
            in that parameter's physical unit.

        Raises
        ------
        AttributeError
            If no fit has been run, so ``info_opt`` does not exist.
        TypeError
            If ``alpha`` is not a real number, ``include_mse`` is neither a
            ``bool`` nor None, or ``covar_params`` is not a real numeric
            array.
        ValueError
            If ``alpha`` is not in the open interval ``(0, 1)``, both
            ``covar_params`` and ``include_mse`` are supplied, the number of
            observed data entries does not exceed ``num_params``, the
            stored or recomputed covariance (``covar_params`` None) is
            undefined (all NaN, as ``get_covariance`` returns it; the message
            says so and points to its warning for the cause), or
            ``covar_params`` does not have finite
            entries of shape ``(num_params, num_params)`` with
            non-negative diagonal variances.
        """
        if (isinstance(alpha, bool)
                or not isinstance(alpha, (int, float, np.number))
                or np.iscomplexobj(alpha)):
            raise TypeError(
                "alpha must be a real number in (0, 1); got "
                f"{type(alpha).__name__}")
        alpha_val = float(alpha)  # [-]
        if not np.isfinite(alpha_val) or not (0.0 < alpha_val < 1.0):
            raise ValueError(
                f"alpha must be strictly between 0 and 1; got {alpha!r}")
        if include_mse is not None and not isinstance(include_mse, bool):
            raise TypeError(
                "include_mse must be a bool or None; got "
                f"{type(include_mse).__name__}")
        if covar_params is not None and include_mse is not None:
            raise ValueError(
                "Pass either covar_params or include_mse, not both")

        _ = self.info_opt
        if self.num_data_total <= self.num_params:
            raise ValueError(
                "Student-t confidence intervals require more observed entries "
                "than optimized parameters "
                f"(num_data_total={int(self.num_data_total)}, "
                f"num_params={int(self.num_params)})")

        given_covar = covar_params is not None
        if covar_params is None:
            had_covar = hasattr(self, 'covar_params')
            prev_covar = getattr(self, 'covar_params', None)
            had_correl = hasattr(self, 'correl_params')
            prev_correl = getattr(self, 'correl_params', None)
            if include_mse is not None:
                # [param_i unit * param_j unit], shape (num_params, num_params).
                covar_params = self.get_covariance(include_mse=include_mse)
            elif prev_covar is not None:
                covar_params = prev_covar
            else:
                covar_params = self.get_covariance(include_mse=True)
            if not set_self:
                if had_covar:
                    self.covar_params = prev_covar
                elif hasattr(self, 'covar_params'):
                    delattr(self, 'covar_params')
                if had_correl:
                    self.correl_params = prev_correl
                elif hasattr(self, 'correl_params'):
                    delattr(self, 'correl_params')

        covar_raw = np.asarray(covar_params)
        if covar_raw.dtype.kind not in ('i', 'u', 'f'):
            raise TypeError("covar_params must be a real numeric 2D array")
        # [param_i unit * param_j unit], shape (num_params, num_params).
        covar_array = np.asarray(covar_raw, dtype=float)
        expected_shape = (int(self.num_params), int(self.num_params))
        if covar_array.shape != expected_shape:
            raise ValueError(
                f"covar_params must have shape {expected_shape}; got "
                f"{covar_array.shape}")
        if not given_covar:
            _require_defined_covariance(
                covar_array, 'ParameterEstimation.get_confidence_intervals')
        if not np.all(np.isfinite(covar_array)):
            raise ValueError("covar_params entries must be finite")
        if np.any(np.diag(covar_array) < 0.0):
            raise ValueError(
                "covar_params diagonal variances must be non-negative")

        stats_obj = StatisticsClass(self, alpha=alpha_val)
        intervals = stats_obj.get_intervals(
            covar_params=covar_array, verbose=verbose, set_self=set_self)
        if set_self:
            self.confid_intervals = stats_obj.confid_intervals
            # [parameter units], shape (num_params, 2).
            self.intervals_array = stats_obj.intervals_array
            self.stats = stats_obj

        return intervals

    def get_covariance(self, include_mse: bool = True) -> np.ndarray:
        """Estimate the parameter covariance at the accepted parameters.

        The linearized (Gauss-Newton) covariance is built from the weighted
        Jacobian ``J = info_opt['jac']``, shape ``(num_params, n_entries)``,
        and the weighted residuals ``r = info_opt['fun']`` stored by
        ``optimize_fn``.

        Parameters
        ----------
        include_mse : bool, optional
            If True (the default), return ``s2 * inv(J @ J.T)`` with the
            residual mean square ``s2 = r @ r / (num_data_total -
            num_params)``. If False, return ``inv(J @ J.T)``.

        Returns
        -------
        numpy.ndarray
            Covariance, shape ``(num_params, num_params)``, in optimized
            parameter (``name_params``) order. Entry ``(i, j)`` has the
            product of the units of parameters ``i`` and ``j`` when the
            weighted residuals are dimensionless (declared measurement
            uncertainties, or a ``weight_matrix`` holding measurement-error
            covariances in squared measurement units), for both values of
            ``include_mse``. With the default identity weighting this holds
            for ``include_mse=True`` only if every residual has the same
            physical unit; ``include_mse=False`` is then additionally
            divided by that unit squared. With identity weighting of
            residuals in different units (e.g. [mol/L] and [K]) neither
            form is a physical covariance: it depends on the units chosen
            for each measurement. Also stored as ``covar_params``; the
            matching correlation matrix [-] is stored as ``correl_params``.
            Both are filled with NaN, in the unit of the requested form,
            when the covariance is undefined (see Warns and Notes).

        Raises
        ------
        AttributeError
            If no fit has been run, so ``info_opt`` does not exist.

        Warns
        -----
        RuntimeWarning
            If the covariance is undefined, with one of three diagnostics:
            the weighted residual Jacobian has non-finite entries (naming
            those parameters); the parameters are not (locally)
            identifiable from the observed data (naming every parameter
            with no effect on any observed entry and every poorly
            determined direction as a combination of parameter names, e.g.
            ``0.707*k - 0.707*b``); or ``J @ J.T`` passed the conditioning
            test but its inversion failed or gave non-finite or negative
            variances (naming those parameters and the least-determined
            direction).

        Notes
        -----
        The linearized covariance is undefined when ``J`` is not finite or
        ``J @ J.T`` is singular or numerically singular: a parameter has
        zero information (it changes no observed entry), the Jacobi-scaled
        matrix ``D J J^T D``, ``D = diag((J J^T)_ii**-0.5)``, has a
        condition number of at least ``1/eps`` with ``eps`` the machine
        epsilon (``2u`` for the unit roundoff ``u``; the computed inverse
        would have an error of order one and no correct significant digit,
        Golub & Van Loan, *Matrix Computations*, 4th ed., 2013, Sec. 2.6;
        evaluated from the singular values of ``D J``), the inversion
        fails, or it yields a non-finite or negative variance. The Jacobi
        scaled matrix does not change with the units chosen for the
        parameters, so neither does the test; ``_information_inverse``
        gives the details. Earlier releases raised ``numpy.linalg.LinAlgError``
        for an exactly singular ``J @ J.T`` and otherwise returned
        meaningless values, including negative variances (#532). A defined
        covariance is computed exactly as before.

        Only observed entries count as data: unobserved entries have zero
        weighted residual and zero Jacobian column, and the degrees of
        freedom are ``num_data_total - num_params`` with ``num_data_total``
        the number of observed entries. The residual mean square requires
        more observed entries than optimized parameters.

        ``s2`` is the reduced chi-square of the weighted residuals [-] when
        they are dimensionless, and the residual variance in the
        measurement unit squared with identity weighting of a single
        physical unit. The default
        ``include_mse=True`` therefore treats declared standard deviations
        (or ``weight_matrix``) as relative weights whose common scale is
        re-estimated from the residual scatter, like
        ``scipy.optimize.curve_fit(..., absolute_sigma=False)``. When the
        declared standard deviations are known in absolute terms,
        ``include_mse=False`` gives the covariance that treats them as
        absolute (``absolute_sigma=True``); both coincide when ``s2 = 1``.
        """
        jac = self.info_opt['jac']
        resid = self.info_opt['fun']

        dof = self.num_data_total - self.num_params  # [-]
        # [p_i*p_j] with include_mse, [p_i*p_j/r**2] without; correlation
        # [-]. Both all NaN, with a diagnostic, when undefined.
        covar, correlation, _, diagnostic = _linearized_covariance(
            jac, resid, dof, self.name_params, include_mse)
        if diagnostic is not None:
            warnings.warn(diagnostic, RuntimeWarning,
                          stacklevel=_user_stacklevel())

        self.covar_params = covar
        self.correl_params = correlation

        return covar

    def _noise_model(self) -> str:
        """Return the measurement-error weighting of this estimator.

        Returns
        -------
        {'declared_uncertainty', 'weight_matrix', 'identity'}
            ``'declared_uncertainty'`` when measurements declare standard
            deviations (``_residual_std`` is not None), ``'weight_matrix'``
            when ``weight_matrix`` was supplied, also the identity (a
            statement that the errors have standard deviation 1 in each
            measured unit), and ``'identity'`` for the unweighted fit.

        Notes
        -----
        Reads ``_weight_matrix_given``, which every estimator built by this
        version stores at construction.
        """
        if self._residual_std is not None:
            return 'declared_uncertainty'
        return 'weight_matrix' if self._weight_matrix_given else 'identity'

    def _jacobian_source(self) -> str:
        """Return the origin of the weighted residual Jacobian.

        Returns
        -------
        {'model', 'jac_fun', 'finite_difference'}
            ``'model'`` when the model callback returns sensitivities with
            its outputs (``sens_second`` is stored), ``'jac_fun'`` when a
            Jacobian callback was given, ``'finite_difference'`` otherwise;
            the precedence of ``get_gradient``.
        """
        if self.sens_second is not None:
            return 'model'
        if self.jac_fun is not None:
            return 'jac_fun'
        return 'finite_difference'

    def _finite_difference_steps(self) -> np.ndarray:
        """Return the forward-difference steps of the optimized parameters.

        Returns
        -------
        numpy.ndarray
            ``|dx_finitediff|`` evaluated, as ``get_gradient`` does, at the
            full parameter vector of the accepted fit, for the optimized
            parameters, shape ``(num_params,)`` [parameter units].
        """
        return np.abs(self._signed_finite_difference_steps(
            self.params_convg))  # [parameter units]

    def _signed_finite_difference_steps(self, params) -> np.ndarray:
        """Return the signed forward-difference steps at a point.

        Parameters
        ----------
        params : array_like
            Optimized parameters, shape ``(num_params,)`` [parameter
            units].

        Returns
        -------
        numpy.ndarray
            ``dx_finitediff`` evaluated, as ``get_gradient`` does, at the
            full parameter vector of ``params``, for the optimized
            parameters, shape ``(num_params,)`` [parameter units]: the
            finite-difference Jacobian evaluates the model at ``params +
            steps``, one parameter at a time.
        """
        # [parameter units], the vector at which get_gradient evaluates
        # dx_finitediff
        params_full = np.asarray(self.reconstruct_params(params),
                                 dtype=float)
        # [parameter units], one step per entry of params_full
        steps = (self.dx_fd(params_full) if callable(self.dx_fd)
                 else np.ones_like(params_full) * self.dx_fd)
        # [parameter units], steps of the optimized parameters
        return np.asarray(steps, dtype=float)[self.map_variable]

    def _default_jacobian_rtol(self, source: str,
                               steps: Optional[np.ndarray],
                               param_scale) -> np.ndarray:
        """Return the default row-relative accuracy of the sensitivities.

        Parameters
        ----------
        source : {'model', 'jac_fun', 'finite_difference'}
            Origin of the Jacobian (``_jacobian_source``).
        steps : numpy.ndarray or None
            Forward-difference steps ``h_i`` of the optimized parameters,
            shape ``(num_params,)`` [parameter units]; read for
            ``'finite_difference'``.
        param_scale : float, array_like or None
            Typical magnitude of the parameters [parameter units]; None uses
            ``|params_convg|``.

        Returns
        -------
        numpy.ndarray
            Shape ``(num_params,)`` [-]: machine epsilon for ``'model'`` and
            ``'jac_fun'`` (derivatives exact to rounding); for
            ``'finite_difference'`` the relative steps ``h_i / scale_i``,
            the truncation error of a forward difference relative to the
            derivative (see Notes).

        Raises
        ------
        ValueError
            If a finite-difference relative step is undefined (a parameter
            at zero without ``param_scale``), not finite or not in ``(0,
            1)``; pass ``param_scale`` or ``jacobian_rtol`` explicitly then.

        Notes
        -----
        The forward difference ``(f(p + h) - f(p)) / h`` has the truncation
        error ``(h / 2) |f''|`` (Nocedal and Wright, *Numerical
        Optimization*, 2nd ed., Springer, 2006, Sec. 8.1, eq. 8.5, first
        term). Under the well-scaled-curvature assumption ``scale |f''| <~
        |f'|``, that error relative to the derivative is at most about
        ``h / scale``, independent of any additive offset of ``f``. The
        rounding term of eq. 8.5 enters ``IdentifiabilityReport`` through
        ``output_rtol``.
        """
        num_params = int(self.num_params)  # [-]
        if source != 'finite_difference':
            return np.full(num_params, float(np.finfo(float).eps))
        try:
            scale = _parameter_scale(
                np.asarray(self.params_convg, dtype=float), param_scale,
                list(self.name_params))  # [parameter units]
        except ValueError as error:
            raise ValueError(
                f"{error}. It sets the relative step |dx| / scale, the "
                "default jacobian_rtol of a finite-difference Jacobian; "
                "alternatively pass jacobian_rtol") from error
        relative_steps = steps / scale  # [-]
        if not (np.isfinite(relative_steps).all()
                and (relative_steps > 0).all() and (relative_steps < 1).all()):
            raise ValueError(
                "The default jacobian_rtol of the finite-difference "
                "Jacobian, the relative steps |dx| / scale = "
                f"{relative_steps.tolist()} of {list(self.name_params)}, is "
                "not in (0, 1); pass jacobian_rtol explicitly")
        return relative_steps

    def _weighted_predictions(self) -> np.ndarray:
        """Return the accepted model outputs weighted like the residuals.

        Returns
        -------
        numpy.ndarray
            Shape ``(len(info_opt['fun']),)`` [weighted residual unit]: the
            model outputs of each experiment at the accepted parameters
            (``y_runs``, which ``optimize_fn`` refreshes there for every
            method) passed through ``_weight_sample_rows`` (declared
            standard deviations or the ``weight_matrix`` roots), zero where
            unobserved, in the residual layout (experiment, measurement
            column, sample).

        Notes
        -----
        The raw outputs are used directly rather than residuals plus
        observations, whose sum cancels when the predictions are far below
        the data. ``y_runs`` describes the last ``get_objective``
        evaluation with ``set_self=True``: an explicit evaluation after the
        fit replaces it.
        """
        parts = []
        for outputs, x_mask, std in zip(self.y_runs, self.x_masks,
                                        self._residual_std_runs()):
            # [model state units]; unobserved entries are not read
            predictions = np.asarray(outputs, dtype=float)
            weighted = self._weight_sample_rows(predictions, x_mask, std)
            parts.append(weighted.T.ravel())
        return np.concatenate(parts)

    def _residual_blocks(self, values: np.ndarray) -> list:
        """Split a vector in the residual layout into sample-row blocks.

        Parameters
        ----------
        values : numpy.ndarray
            Shape ``(len(info_opt['fun']),)``, in the layout of the weighted
            residuals [weighted residual unit].

        Returns
        -------
        list of tuple
            Per experiment ``(rows, x_mask)``: ``rows`` of shape ``(n_times,
            n_columns)`` (sample rows, measurement columns) and the
            observation mask (None when every entry is observed).
        """
        blocks = []
        start = 0  # [-]
        for observations, x_mask in zip(self.y_data, self.x_masks):
            num_times, num_columns = np.shape(observations)  # [-]
            stop = start + num_times * num_columns  # [-]
            # [weighted residual unit], (n_times, n_columns)
            blocks.append((values[start:stop].reshape(num_columns,
                                                      num_times).T, x_mask))
            start = stop
        return blocks

    def _diagonal_weighting(self) -> bool:
        """Return whether the residual whitening is entry by entry.

        Returns
        -------
        bool
            True for declared standard deviations and for a diagonal
            ``weight_matrix`` (including the identity default); False for a
            correlated ``weight_matrix``.
        """
        if self._residual_std is not None:
            return True
        covariance = self._measurement_covariance  # [u_i * u_j]
        return not np.count_nonzero(covariance - np.diag(np.diag(covariance)))

    def _output_error_scale(self, weighted_predictions: np.ndarray) -> float:
        """Bound the whitened effect of a relative output error.

        Parameters
        ----------
        weighted_predictions : numpy.ndarray
            Weighted model predictions in the residual layout, shape
            ``(len(info_opt['fun']),)`` [weighted residual unit].

        Returns
        -------
        float
            Upper bound [weighted residual unit] on the norm of the whitened
            output errors per unit relative output error: errors of at most
            ``r * |y_j|`` in each raw prediction ``y_j`` give whitened errors
            of norm at most ``r`` times this value. For diagonal weighting it
            is ``||W |y|||``, which equals ``||y_w||`` (exact). For a
            correlated ``weight_matrix`` it is ``||W||_2 * ||y||`` over the
            observed entries, with ``||W||_2`` the largest singular value of
            the whitening factors used (``sigma_inv`` for fully observed
            sample rows, the marginal roots for partially observed ones),
            because whitening may amplify errors in directions that the
            predictions themselves do not occupy.

        Notes
        -----
        Norms are computed scaled, free of overflow and underflow
        (``StatsModule._scaled_norm``). No state is modified: marginal roots
        are computed afresh rather than cached.
        """
        if self._diagonal_weighting():
            return _scaled_norm(weighted_predictions)

        largest_gain = 0.0  # [1/u], largest singular value of the roots
        raw_parts = []  # [measurement units], observed raw predictions
        for rows, x_mask in self._residual_blocks(weighted_predictions):
            observed = (np.ones(rows.shape, dtype=bool) if x_mask is None
                        else np.asarray(x_mask, dtype=bool))
            patterns, pattern_of_row = np.unique(observed, axis=0,
                                                 return_inverse=True)
            for pattern_index, pattern in enumerate(patterns):
                if not pattern.any():
                    continue
                root = (self.sigma_inv if pattern.all()
                        else _measurement_precision_root(
                            self._measurement_covariance[
                                np.ix_(pattern, pattern)]))  # [1/u_i]
                largest_gain = max(largest_gain,
                                   float(np.linalg.norm(root, 2)))
                selected = pattern_of_row.ravel() == pattern_index
                # [measurement units]: rows_w = raw @ root, solved for raw
                raw = np.linalg.solve(root.T, rows[selected][:, pattern].T)
                raw_parts.append(raw.ravel())
        return largest_gain * _scaled_norm(np.concatenate(raw_parts))

    def get_identifiability(
            self, alpha: float = 0.95, include_mse: bool = True,
            jacobian_rtol: Optional[Union[float, Sequence[float],
                                          np.ndarray]] = None,
            output_rtol: float = DEFAULT_OUTPUT_RTOL,
            param_scale: Optional[Union[float, Sequence[float],
                                        np.ndarray]] = None
    ) -> IdentifiabilityReport:
        """Report parameter uncertainty and practical identifiability.

        Builds an ``IdentifiabilityReport`` from the accepted fit only
        (``info_opt['jac']``, ``info_opt['fun']``, ``params_convg``,
        ``name_params``, ``num_data_total``, the observations and the
        weights): the explicit noise assumption, the information matrix
        (the hand-off to experiment design, #355), the covariance, standard
        errors, correlations and Student-t intervals of ``get_covariance``
        and ``get_confidence_intervals``, the singular values, condition
        number and collinearity index of the Jacobi-scaled Jacobian, the
        parameters and directions the data do not resolve given the
        accuracy of the sensitivities, the parameters at active bounds and
        the optimizer's convergence outcome. Nothing is refitted, no model
        is evaluated, and the estimator (``covar_params``,
        ``correl_params``, ``params_convg``, ``info_opt``, ...) is not
        modified.

        Parameters
        ----------
        alpha : float, optional
            Confidence level [-] in ``(0, 1)`` of the intervals and of the
            significance test. The default is 0.95.
        include_mse : bool, optional
            If True (the default), declared uncertainties or the
            ``weight_matrix`` are relative weights whose common scale is
            re-estimated from the residual mean square; if False they are
            absolute (see ``get_covariance``).
        jacobian_rtol : float or array_like, optional
            Row-relative accuracy [-] of the sensitivities, in ``(0, 1)``,
            for every parameter or one per optimized parameter:
            ``||dJ_i|| <= jacobian_rtol_i * ||J_i||``. The default None uses
            machine epsilon for model-returned sensitivities and
            ``jac_fun`` (exact to rounding; for sensitivities integrated by
            an ODE solver pass its ``rtol``, which is how CVODES controls
            the sensitivity error) and the relative forward-difference step
            ``h_i / scale_i`` for finite differences (truncation term).
        output_rtol : float, optional
            Relative accuracy [-] of the model outputs, in ``(0, 1)``,
            used only for finite-difference Jacobians (rounding term). The
            default ``jac_module.DEFAULT_OUTPUT_RTOL`` (1e-6, CVode's default
            rtol) is the accuracy the default step ``sqrt(1e-6) * |p|`` is
            designed for; closed-form models can pass machine epsilon and
            ODE models their solver ``rtol``.
        param_scale : float or array_like, optional
            Positive finite typical magnitude of every optimized parameter,
            or one per parameter, in the parameter units, for the default
            finite-difference ``jacobian_rtol``. The default None uses
            ``|params_convg|``; it is required when an estimate is exactly
            zero and that default is used.

        Returns
        -------
        IdentifiabilityReport
            Immutable report; its attributes document units, shapes and
            meaning. ``identifiable`` is False when the covariance is
            undefined (its numbers are then NaN), any parameter or
            direction is unresolved (its numbers are then kept but not
            precise).

        Raises
        ------
        AttributeError
            If no fit has been run, so ``info_opt`` does not exist.
        TypeError
            If ``alpha``, ``jacobian_rtol``, ``output_rtol`` or
            ``param_scale`` is not real, or ``include_mse`` is not a bool.
        ValueError
            If ``alpha``, ``output_rtol`` or ``jacobian_rtol`` is not finite
            and strictly between 0 and 1 or has the wrong shape;
            ``param_scale`` is not positive and finite or has the wrong
            shape; the default finite-difference ``jacobian_rtol`` is
            undefined (an estimate at exactly zero without ``param_scale``)
            or not in ``(0, 1)`` (a relative step of at least 1); parameter
            names repeat; or the number of observed entries does not exceed
            ``num_params``.

        Notes
        -----
        The weighted predictions are the accepted model outputs
        (``y_runs``) weighted like the residuals
        (``_weighted_predictions``); ``MultipleCurveResolution`` weights its
        curve-resolved spectra and non-spectral predictions in its residual
        sign convention, which leaves the norms unchanged.
        ``_output_error_scale`` turns them into the whitened output-error
        bound (exact for diagonal weights, an operator-norm bound for
        correlated ones), and the forward difference of two such outputs
        is wrong by at most twice it.

        Active bounds are read from ``info_opt['active_mask']`` of
        ``scipy.optimize.least_squares``; Levenberg-Marquardt ignores
        bounds (#533) and IPOPT fits store no active set, so for them
        ``active_bounds`` is None.

        No warning is issued for an unavailable covariance: the report
        carries the reason in ``covariance_diagnostic``. Unlike
        ``get_covariance``, which keeps its earlier behaviour, the report
        does not present a covariance computed with a zero or non-finite
        residual mean square (``include_mse=True``), one that is not
        finite, or one with a variance that is not positive.
        """
        info = self.info_opt
        names = list(self.name_params)
        estimates = np.asarray(self.params_convg, dtype=float)  # [p_i]
        if param_scale is not None:
            _parameter_scale(estimates, param_scale, names)  # validation
        source = self._jacobian_source()
        steps = (self._finite_difference_steps()
                 if source == 'finite_difference' else None)  # [p_i]
        if jacobian_rtol is None:
            jacobian_rtol = self._default_jacobian_rtol(
                source, steps, param_scale)  # [-]
        # [weighted residual unit], weighted model predictions
        predictions = self._weighted_predictions()
        # [weighted residual unit] per unit relative output error
        error_scale = self._output_error_scale(predictions)
        units = self._measurement_units
        columns = self._measurement_names
        measurement_units = (None if columns is None or units is None
                             else dict(zip(columns, units)))
        noise_model = self._noise_model()
        weight_identity = noise_model == 'weight_matrix' and np.array_equal(
            self._measurement_covariance,
            np.eye(self._measurement_covariance.shape[0]))
        active_mask = info.get('active_mask')
        active_bounds = (None if active_mask is None else
                         [name for name, flag in zip(names, active_mask)
                          if flag != 0])
        convergence = ({'success': info.get('success'),
                        'message': info.get('message'),
                        'tolerances': info.get('tolerances')}
                       if 'success' in info else None)
        return IdentifiabilityReport.from_fit(
            info['jac'], info['fun'], estimates, names, self.num_data_total,
            noise_model=noise_model, measurement_units=measurement_units,
            jacobian_source=source, jacobian_rtol=jacobian_rtol,
            output_rtol=output_rtol, output_error_scale=error_scale,
            finite_difference_steps=steps, alpha=alpha,
            include_mse=include_mse, active_bounds=active_bounds,
            weight_matrix_is_identity=weight_identity,
            convergence=convergence)

    def _observed_residual_entries(self, num_entries: int) -> np.ndarray:
        """Return which entries of the weighted residual vector are observed.

        Parameters
        ----------
        num_entries : int
            Length of ``get_objective(..., out_array=True)`` [-].

        Returns
        -------
        numpy.ndarray
            Boolean, shape ``(num_entries,)``: the ``'observed'`` column of
            ``get_residual_layout``. Every entry is marked observed for
            nested state observation dictionaries, which have no residual
            layout; their unobserved entries are zero in the residuals and
            the Jacobian alike, so norms are unchanged.

        Raises
        ------
        ValueError
            If the residual layout does not have ``num_entries`` rows.
        """
        try:
            layout = self.get_residual_layout()
        except NotImplementedError:
            return np.ones(num_entries, dtype=bool)
        observed = layout['observed'].to_numpy(dtype=bool)
        if observed.size != num_entries:
            raise ValueError(
                f"The residual layout has {observed.size} entries, but the "
                f"weighted residual vector has {num_entries}")
        return observed

    def _end_derivative_check(self, state: dict, contents: list,
                              restore_point: np.ndarray,
                              failure: Optional[BaseException]) -> None:
        """Re-evaluate the model and restore the fit after a check.

        Parameters
        ----------
        state : dict
            Result of ``_fit_state`` before the check.
        contents : list of tuple
            ``_array_contents`` of the saved attribute values: ``(array,
            copy)`` pairs whose contents are copied back.
        restore_point : numpy.ndarray
            Optimized parameters [parameter units] at which the model is
            evaluated once more: ``params_convg``, or the checked point
            before a fit.
        failure : BaseException or None
            Exception raised by the check, or None after success.

        Raises
        ------
        Exception
            Whatever the re-evaluation raises when ``failure`` is None,
            after the state has been restored.

        Warns
        -----
        RuntimeWarning
            If the re-evaluation raises while ``failure`` is an
            ``Exception``; the caller re-raises ``failure``. The report is
            dropped when warnings are turned into errors, so that it never
            replaces ``failure``.

        Notes
        -----
        No re-evaluation is attempted when ``failure`` is not an
        ``Exception`` (e.g. ``KeyboardInterrupt``).
        """
        try:
            if failure is None or isinstance(failure, Exception):
                try:
                    self.get_objective(
                        np.array(restore_point, dtype=float, copy=True),
                        out_array=True, set_self=False)
                except Exception as refresh_error:
                    if failure is None:
                        raise
                    try:
                        warnings.warn(
                            "check_derivatives could not re-evaluate the "
                            f"model at {np.asarray(restore_point).tolist()} "
                            f"after an evaluation failed ({refresh_error!r});"
                            " a stateful model callback may be left at the "
                            "last evaluated parameters",
                            RuntimeWarning, stacklevel=_user_stacklevel())
                    except Exception:
                        # A warnings filter set to 'error' turns the report
                        # into an exception; the original failure, which
                        # the caller re-raises, takes priority.
                        pass
        finally:
            self._restore_fit_state(state)
            _restore_array_contents(contents)

    def check_derivatives(
            self, params: Optional[npt.ArrayLike] = None,
            rel_step: Optional[Union[float, npt.ArrayLike]] = None,
            param_scale: Optional[Union[float, npt.ArrayLike]] = None,
            bounds: Optional[Union[Sequence, Bounds]] = None,
            rtol: Optional[float] = None) -> DerivativeCheck:
        """Verify the estimator's Jacobian against finite differences.

        Compares the weighted residual Jacobian that the optimizer and the
        covariance use, ``get_gradient(p, out_array=True)`` (model-returned
        sensitivities, ``jac_fun`` or the estimator's forward differences
        with ``dx_finitediff``), with an independent second-order
        finite-difference derivative of the weighted residuals
        ``get_objective(p, out_array=True)`` at the same point ``p``. Only
        the optimized parameters are perturbed, one at a time; fixed
        parameters keep their ``param_seed`` values. The estimator's fit is
        not modified (see Notes).

        Parameters
        ----------
        params : array_like, optional
            Optimized parameters at which to check, shape ``(num_params,)``
            [parameter units], finite. The default is ``params_convg`` after
            a fit and the optimized entries of ``param_seed`` before one.
        rel_step : float or array_like, optional
            Relative step [-] in ``(0, 1)``, for every parameter or one per
            optimized parameter. The step of parameter ``i`` is ``h_i =
            rel_step_i * scale_i``. The default is
            ``jac_module.CENTRAL_DIFFERENCE_REL_STEP``, the cube root of the
            machine epsilon (about 6.06e-6), which is right only for
            outputs accurate to machine precision, such as closed-form
            models (see Notes).
        param_scale : float or array_like, optional
            Positive finite typical magnitude ``scale_i`` of every
            optimized parameter, or one per parameter [parameter units].
            The default None uses ``|params|`` and requires every parameter
            to be nonzero: no magnitude is invented for a parameter at zero.
        bounds : sequence of pairs or scipy.optimize.Bounds, optional
            Valid domain of the optimized parameters, in ``name_params``
            order and parameter units: one ``(lower, upper)`` pair per
            optimized parameter, as for an IPOPT fit (``optimize_fn``) and
            in ``CalibrationRecord.optimizer_inputs()``, with None or an
            infinity for an unbounded side; or a ``scipy.optimize.Bounds``
            (``lb`` and ``ub`` scalars or one entry per parameter). The
            ``(lower_array, upper_array)`` form of
            ``scipy.optimize.least_squares`` must be wrapped in
            ``scipy.optimize.Bounds``: a plain tuple of two arrays is read
            as per-parameter pairs (for two parameters it is accepted with
            that meaning; otherwise its shape is rejected). ``params`` must
            lie inside, and after a fit ``params_convg`` too, where the
            model is re-evaluated at the end (see Notes); the model is
            never evaluated outside: a
            parameter whose central points ``p +/- h`` leave the domain
            uses the one-sided three-point formula toward the interior
            (needs ``p + 2h`` or ``p - 2h`` inside). The default None is
            unbounded.
        rtol : float, optional
            Agreement threshold [-] on the relative errors, positive and
            finite. Supplied, it fills ``DerivativeCheck.agrees``; the
            default None applies no threshold.

        Returns
        -------
        DerivativeCheck
            Immutable comparison: the estimator's ``jacobian`` and the
            ``reference``, shape ``(num_params, n_entries)`` [weighted
            residual unit / parameter unit] in the ``get_residual_layout``
            order, their absolute errors, and per parameter the step,
            formula, largest absolute error and relative error ``||J_i -
            R_i|| / ||R_i||`` over the observed entries.

        Raises
        ------
        TypeError
            If ``params``, ``rel_step``, ``param_scale``, a bound or
            ``rtol`` is not real.
        ValueError
            If ``params`` does not have shape ``(num_params,)``, is not
            finite or lies outside ``bounds``; ``bounds`` is given after
            a fit and ``params_convg``, where the model is re-evaluated at
            the end, lies outside it; ``rel_step`` is not in
            ``(0, 1)`` or has the wrong shape; ``param_scale`` is not
            positive and finite or has the wrong shape; a parameter is
            exactly zero without ``param_scale``; ``bounds`` is malformed
            (see ``_parameter_bounds``); a parameter's bounds have zero
            width (fix it through ``optimize_flags`` instead); a step
            vanishes at the floating-point spacing of its parameter;
            neither a central nor a one-sided step of a parameter fits
            inside its bounds; or the Jacobian is the estimator's forward
            difference and ``dx_finitediff`` would step outside ``bounds``.
        Exception
            Whatever the model callback or ``jac_fun`` raises; the model is
            re-evaluated and the fit state restored first (see Notes).

        Warns
        -----
        RuntimeWarning
            If an evaluation raised and the model then also fails at the
            restoring re-evaluation (see Notes); the original exception
            propagates.

        Notes
        -----
        *Formulas.* The central difference ``(f(p + h) - f(p - h)) / (2h)``
        has the truncation error ``h**2 f'''(p) / 6``; the one-sided
        three-point difference ``(-3 f(p) + 4 f(p + s) - f(p + 2s)) /
        (2s)``, ``s = +h`` (forward) or ``-h`` (backward), has
        ``-s**2 f'''(p) / 3`` (derivation in ``_three_point_difference``).
        The steps are rounded so that ``p + h`` is the evaluated number.

        *Step choice.* If each output is accurate to a relative ``u``, the
        rounding error of either formula is of order ``u |f| / h``. The
        total, of order ``h**2 |f'''| + u |f| / h``, is smallest for ``h``
        proportional to ``u**(1/3)`` (Nocedal and Wright, *Numerical
        Optimization*, 2nd ed., Springer, 2006, Sec. 8.1, for ``u`` the
        machine epsilon), with an error of order ``u**(2/3)``. The default
        relative step assumes machine-precision outputs. For outputs of an
        ODE integrator with relative tolerance ``solver_rtol`` (e.g. the
        CVode ``rtol``), ``u`` is about ``solver_rtol``: pass ``rel_step``
        of about ``solver_rtol**(1/3)``, e.g.
        ``jac_module.DEFAULT_OUTPUT_RTOL**(1/3)`` = 1e-2 at CVode's default
        ``rtol`` of 1e-6, and tighten the solver tolerances when a sharper
        check is needed. (The ``rtol`` argument of this method is the
        agreement threshold, unrelated to the solver tolerance.) A too
        small step for noisy outputs reports rounding noise as a
        derivative error.

        *Reading the result.* The relative errors include the error of the
        reference itself, so compare them with its expected accuracy,
        about ``rel_step**2`` plus ``u / rel_step`` for well-scaled
        parameters, rather than with zero. A forward-difference Jacobian
        (``jacobian_source='finite_difference'``) differs from the
        reference by its own truncation error, about ``|dx| |f''| / 2``,
        relative ``|dx| / scale`` for well-scaled curvature.

        *Domain.* The estimator's forward-difference Jacobian is evaluated
        exactly as the optimizer evaluates it, at ``p + dx_finitediff``;
        it is checked against ``bounds`` before any of those evaluations.

        *State.* ``get_objective`` and ``get_gradient`` overwrite the
        evaluation buffers (``y_runs``, ``resid_runs``,
        ``weighted_residuals``, ``sens``, ``sens_second``, ...) and append
        to the iteration histories, and stateful model callbacks, such as
        the unit operations behind a ``SimulationExec`` estimator, keep the
        parameters and results of their last evaluation. After the check,
        also when an evaluation raises, the model is therefore evaluated
        once more at ``params_convg`` (at the checked point before a fit),
        as ``optimize_fn`` refreshes its accepted point, so such callbacks
        end at the parameters they started from. Then the attributes in
        ``_FIT_ATTRIBUTES`` saved before the evaluations are rebound, and
        the contents of the writeable arrays they hold (directly or in
        lists and tuples) are copied back, which also undoes callbacks
        that refill a preallocated sensitivity array in place. The
        histories are detached during the check. So ``params_convg``,
        ``info_opt``, ``covar_params``, ``params_iter`` and the buffers
        that ``StatisticsClass`` reads keep their objects and values. If
        that re-evaluation fails after a successful check, its exception
        propagates (the attributes are still restored); after a failed
        evaluation, the original exception propagates and a
        ``RuntimeWarning`` reports the second failure. Only internal caches
        change: the marginal precision roots of ``_weight_sample_rows`` may
        gain entries, and ``MultipleCurveResolution`` clears its evaluation
        cache (see its ``_restore_fit_state``).
        """
        names = list(self.name_params)
        num_params = int(self.num_params)  # [-]
        rtol = _agreement_threshold(rtol)  # [-]
        if params is None:
            params = getattr(self, 'params_convg', None)
            if params is None:
                params = self.param_seed[self.map_variable]
        point = _float_vector(params, 'params')  # [parameter units]
        if point.shape != (num_params,):
            raise ValueError(
                f"params must have shape ({num_params},), one value per "
                f"optimized parameter ({', '.join(names)}); got "
                f"{point.shape}")
        if not np.isfinite(point).all():
            raise ValueError(f"params must be finite; got {point.tolist()}")
        lower, upper = _parameter_bounds(bounds, names)  # [parameter units]
        outside = [f"{name} = {float(value)!r} not in [{float(low)!r}, "
                   f"{float(high)!r}]"
                   for name, value, low, high in zip(names, point, lower,
                                                     upper)
                   if not low <= value <= high]
        if outside:
            raise ValueError("params lie outside bounds: "
                             + '; '.join(outside))
        for name, low, high in zip(names, lower, upper):
            if low == high:
                raise ValueError(
                    f"The bounds of {name!r} have zero width "
                    f"([{float(low)!r}, {float(high)!r}]), so it cannot be "
                    "perturbed: fix the parameter through optimize_flags "
                    "instead of bounds")
        accepted = getattr(self, 'params_convg', None)  # [parameter units]
        if accepted is not None:
            accepted = np.asarray(accepted, dtype=float)
            stray = [f"{name} = {float(value)!r} not in [{float(low)!r}, "
                     f"{float(high)!r}]"
                     for name, value, low, high in zip(names, accepted,
                                                       lower, upper)
                     if not low <= value <= high]
            if stray:
                raise ValueError(
                    "params_convg lies outside bounds ("
                    + '; '.join(stray) + "), and check_derivatives "
                    "re-evaluates the model there at the end to leave "
                    "stateful callbacks at the accepted parameters; widen "
                    "the bounds to include params_convg or check without "
                    "bounds")
        rel_steps = _per_parameter_fractions(
            CENTRAL_DIFFERENCE_REL_STEP if rel_step is None else rel_step,
            'rel_step', num_params)  # [-]
        try:
            scale = _parameter_scale(point, param_scale,
                                     names)  # [parameter units]
        except ValueError as error:
            raise ValueError(
                f"{error}. It sets the step h = rel_step * scale of the "
                "derivative check") from error

        # Per parameter: formula, signed step [parameter units] and the
        # evaluated parameter values [parameter units], all inside bounds.
        plan = []
        for ind, name in enumerate(names):
            value = point[ind]  # [parameter units]
            nominal = rel_steps[ind] * scale[ind]  # [parameter units]
            above = value + nominal  # [parameter units]
            step = above - value  # [parameter units], as represented
            below = value - step  # [parameter units]
            if not step > 0:
                raise ValueError(
                    f"The step of {name!r}, rel_step * scale = "
                    f"{float(nominal)!r}, vanishes at the floating-point "
                    f"spacing of {float(value)!r}; "
                    "pass a larger rel_step or param_scale")
            if lower[ind] <= below and above <= upper[ind]:
                plan.append(('central', step, (above, below)))
                continue
            far = value + 2 * step  # [parameter units]
            if far <= upper[ind]:
                plan.append(('forward', step, (above, far)))
                continue
            near = value - nominal  # [parameter units]
            back = near - value  # [parameter units], negative
            far = value + 2 * back  # [parameter units]
            if back < 0 and lower[ind] <= far:
                plan.append(('backward', back, (near, far)))
                continue
            raise ValueError(
                f"Neither a central nor a one-sided three-point step of "
                f"{name!r} (h = {float(step)!r} at {float(value)!r}) fits "
                f"inside its bounds [{float(lower[ind])!r}, "
                f"{float(upper[ind])!r}]; pass a smaller rel_step or "
                "param_scale")

        state = self._fit_state()
        contents = _array_contents(state.values())
        # Detached histories: get_objective appends only to lists.
        self.params_iter = None
        self.objfun_iter = None
        # [parameter units], where stateful callbacks are left afterwards
        restore_point = state['params_convg']
        if restore_point is _ABSENT or restore_point is None:
            restore_point = point
        try:
            # [weighted residual unit], also refreshes model-returned
            # sensitivities at the point
            center = np.array(self.get_objective(point.copy(), out_array=True,
                                                 set_self=False), dtype=float)
            source = self._jacobian_source()
            if source == 'finite_difference':
                # [parameter units], points of the forward differences
                fd_points = point + self._signed_finite_difference_steps(
                    point)
                stepping_out = [
                    f"{name} = {float(value)!r}"
                    for name, value, low, high in zip(names, fd_points,
                                                      lower, upper)
                    if value < low or value > high]
                if stepping_out:
                    raise ValueError(
                        "The estimator's forward-difference Jacobian would "
                        "evaluate the model outside bounds at "
                        + ', '.join(stepping_out) + ", as the optimizer "
                        "would at this point; pass dx_finitediff steps that "
                        "point into the domain")
            # [weighted residual unit / parameter unit], (n, n_entries)
            jacobian = np.array(self.get_gradient(point.copy(),
                                                  out_array=True),
                                dtype=float)
            reference = np.zeros((num_params, center.size))  # same units

            for ind, (formula, step, values) in enumerate(plan):
                outputs = []  # [weighted residual unit]
                for value in values:
                    trial = point.copy()  # [parameter units]
                    trial[ind] = value
                    outputs.append(np.array(
                        self.get_objective(trial, out_array=True,
                                           set_self=False), dtype=float))
                if formula == 'central':
                    reference[ind] = ((outputs[0] - outputs[1])
                                      / (values[0] - values[1]))
                else:
                    reference[ind] = _three_point_difference(
                        center, outputs[0], outputs[1], step)
            observed = self._observed_residual_entries(center.size)
        except BaseException as failure:
            self._end_derivative_check(state, contents, restore_point,
                                       failure)
            raise
        self._end_derivative_check(state, contents, restore_point, None)

        return DerivativeCheck.from_derivatives(
            names, point, rel_steps, scale,
            np.abs([step for _, step, _ in plan]),
            [formula for formula, _, _ in plan], source, jacobian, reference,
            observed, rtol=rtol)

    def plot_data_model(self, **fig_kwargs):

        num_plots = self.num_datasets

        if 'ncols' not in fig_kwargs and 'nrows' not in fig_kwargs:
            num_cols = bool(num_plots // 2) + 1
            num_rows = num_plots // 2 + num_plots % 2

            fig_kwargs.update({'nrows': num_rows, 'ncols': num_cols})

        fig, axes = plt.subplots(**fig_kwargs)

        if num_plots == 1:
            axes = np.asarray(axes)[np.newaxis]

        ax_kwargs = {'mfc': 'None', 'ls': '', 'ms': 4}

        ax_flatten = axes.flatten()

        x_data = self.x_model
        y_data = self.y_data

        for ind in range(self.num_datasets):  # experiment loop
            mask_nan = np.isfinite(self.y_model[ind])
            y_model = self.y_model[ind]

            # Model prediction
            for col, mask in enumerate(mask_nan.T):
                ax_flatten[ind].plot(x_data[ind][mask], y_model[mask, col])

            lines = ax_flatten[ind].lines
            colors = [line.get_color() for line in lines]

            markers = cycle(['o', 's', '^', '*', 'P', 'X'])
            for color, y in zip(colors, y_data[ind].T):
                ax_flatten[ind].plot(x_data[ind], y, color=color,
                                     marker=next(markers),
                                     **ax_kwargs)

            # Edit
            ax_flatten[ind].spines['right'].set_visible(False)
            ax_flatten[ind].spines['top'].set_visible(False)

            # ax_flatten[ind].set_xlabel('$x$')
            ax_flatten[ind].set_ylabel(r'$\mathbf{y}$')

            ax_flatten[ind].xaxis.set_minor_locator(AutoMinorLocator(2))
            ax_flatten[ind].yaxis.set_minor_locator(AutoMinorLocator(2))

        if len(ax_flatten) > self.num_datasets:
            fig.delaxes(ax_flatten[-1])

        if len(axes) == 1:
            axes = axes[0]
            axes.set_xlabel('$x$')
        else:
            fig.text(0.5, 0, '$x$', ha='center')

        fig.tight_layout()

        return fig, axes

    def plot_data_model_sep(self, fig_size=None, fig_kwargs=None, dataset=0):

        xdata = self.x_model[dataset]

        ydata = self.y_data[dataset].T
        ymodel = self.y_model[dataset].T  # TODO: change this

        num_plots = ydata.shape[0]

        num_col = 2
        num_row = num_plots // num_col + num_plots % num_col

        num_row = 2
        num_col = num_plots // num_row + num_plots % num_row

        fig, axes = plt.subplots(num_row, num_col, figsize=fig_size)

        for ind in range(num_plots):  # for every state
            axes.flatten()[ind].plot(xdata, ymodel[ind])

            line = axes.flatten()[ind].lines[0]
            color = line.get_color()
            axes.flatten()[ind].plot(xdata, ydata[ind], 'o',
                                     color=color, mfc='None')

            axes.flatten()[ind].set_ylabel(self.name_states[ind])

        fig.tight_layout()
        return fig, axes

    def plot_sens_param(self):
        figs = []
        axes = []
        for times, sens in zip(self.x_data, self.sens_runs):
            fig, ax = plot_sens(times, sens)

            figs.append(fig)
            axes.append(ax)

        return figs, axes

    def plot_parity(self, fig_size=(4.5, 4.0), **fig_kwargs):
        if len(fig_kwargs) == 0:
            fig_kwargs['alpha'] = 0.70

        fig, axis = plt.subplots(figsize=fig_size)

        y_model = self.y_model

        if self.experim_names is None:
            experim_names = ['experiment {}'.format(ind + 1)
                             for ind in range(self.num_datasets)]
        else:
            experim_names = self.experim_names

        markers = cycle(['o', 's', '^', '*', 'P', 'X'])
        for ind, y_model in enumerate(y_model):
            y_data = self.y_data[ind]

            if isinstance(y_data, dict):
                y_data = np.hstack(list(y_data.values()))

            axis.scatter(y_model.T.flatten(), y_data.T.flatten(),
                         label=experim_names[ind], marker=next(markers),
                         **fig_kwargs)

        axis.set_xlabel('Model')
        axis.set_ylabel('Data')
        axis.legend(loc='best')

        plot_min, plot_max = axis.get_xlim()

        offset = 0.05*plot_max
        x_central = [plot_min - offset, plot_max + offset]

        axis.plot(x_central, x_central, 'k')
        axis.set_xlim(x_central)
        axis.set_ylim(x_central)

        return fig, axis

    def plot_correlation(self):
        """Plot the parameter correlation matrix as a heatmap.

        Returns
        -------
        fig_heat : matplotlib.figure.Figure
            Figure holding the heatmap.
        axis_heat : matplotlib.axes.Axes
            Axis of the heatmap: the lower triangle of ``correl_params``
            [-], diagonal included, on a -1 to 1 colour scale, with the
            optimized parameters' plot labels (``name_params_plot``).

        Raises
        ------
        AttributeError
            If no covariance has been computed, so ``correl_params`` does
            not exist.
        ValueError
            If the correlations are undefined (all NaN, as stored by
            ``get_covariance`` when the covariance is undefined; its
            warning states the cause). Earlier releases drew an empty
            heatmap.
        """
        _require_defined_covariance(self.correl_params,
                                    'ParameterEstimation.plot_correlation')

        # Mask
        mask = np.tri(self.num_params, k=-1).T
        corr_masked = np.ma.masked_array(self.correl_params, mask=mask)

        # Plot
        fig_heat, axis_heat = plt.subplots()

        heatmap = axis_heat.imshow(corr_masked, cmap='RdBu', aspect='equal',
                                   vmin=-1, vmax=1)

        divider = make_axes_locatable(axis_heat)
        cax = divider.append_axes("right", size="5%", pad=0.05)

        cbar = fig_heat.colorbar(heatmap, ax=axis_heat, cax=cax)
        cbar.outline.set_visible(False)

        axis_heat.set_xticks(range(self.num_params))
        axis_heat.set_yticks(range(self.num_params))

        axis_heat.set_xticklabels(self.name_params_plot, rotation=90)
        axis_heat.set_yticklabels(self.name_params_plot)

        return fig_heat, axis_heat


class Deconvolution:
    def __init__(self, mu, sigma, ampl, x_data, y_data):

        self.mu = mu
        self.sigma = sigma
        self.ampl = ampl

        self.x_data = x_data
        self.y_data = y_data

    def concat_params(self):
        if isinstance(self.mu, float) or isinstance(self.mu, int):
            params_concat = np.array([self.mu, self.sigma, self.ampl])
        else:
            params_concat = np.concatenate((self.mu, self.sigma, self.ampl))

        return params_concat

    def fun_wrapper(self, params, x_data):

        grouped_params = np.split(params, 3)
        gaussian = gs.multiple_gaussian(self.x_data, *grouped_params)

        return gaussian

    def dparam_wrapper(self, params, x_data):
        grouped_params = np.split(params, 3)
        der_params = gs.gauss_dparam_mult(x_data, *grouped_params)

        return der_params

    def dx_wrapper(self, params, x_data):
        grouped_params = np.split(params, 3)
        der_x = gs.gauss_dx_mult(x_data, *grouped_params)

        return der_x

    def inspect_data(self):
        gaussian_pred = gs.multiple_gaussian(self.x_data, self.mu, self.sigma,
                                             self.ampl)

        fig, axis = plt.subplots()

        axis.plot(self.x_data, gaussian_pred)
        axis.plot(self.x_data, self.y_data, 'o', mfc='None')

        axis.legend(('prediction with seed params', 'experimental data'))
        axis.set_xlabel('$x$')
        axis.set_ylabel('signal')

        axis.spines['right'].set_visible(False)
        axis.spines['top'].set_visible(False)

        return fig, axis, gaussian_pred

    # def estimate_params(self, optim_opt=None):
    #     seed = self.concat_params()

    #     paramest = ParameterEstimation(self.fun_wrapper, seed, self.x_data,
    #                                    self.y_data,
    #                                    df_dtheta=self.dparam_wrapper,
    #                                    df_dy=self.dx_wrapper)

    #     result = paramest.optimize_fn(optim_options=optim_opt)

    #     optim_params = np.split(result[0], 3)

    #     self.param_obj = paramest
    #     self.optim_params = optim_params

    #     return optim_params, result[1:], paramest

    def plot_results(self, fig_size=None, plot_initial=False,
                     plot_individual=False):

        fig, axis = self.param_obj.plot_data_model(fig_size=fig_size,
                                                   plot_initial=plot_initial)

        axis.legend(('best fit', 'experimental data'))
        axis.set_ylabel('signal')

        axis.xaxis.set_minor_locator(AutoMinorLocator(2))
        axis.yaxis.set_minor_locator(AutoMinorLocator(2))

        if plot_individual:
            gaussian_vals = gs.multiple_gaussian(
                self.x_data, *self.optim_params, separated=True)

            for row in gaussian_vals.T:
                axis.plot(self.x_data, row, '--')
                color = axis.lines[-1].get_color()

                axis.fill_between(self.x_data, row.min(), row, fc=color,
                                  alpha=0.2)

        return fig, axis

    def plot_deriv(self, which='both', plot_mu=False):
        fig, axis = plt.subplots()

        # fun = gs.multiple_gaussian(self.x_data, *self.optim_params)

        if which == 'both':
            first = gs.gauss_dx_mult(self.x_data, *self.optim_params)
            second = gs.gauss_dxdx_mult(self.x_data, *self.optim_params)

            axis.plot(self.x_data, first)
            axis.set_ylabel('$\partial f / \partial x$')

            axis_sec = axis.twinx()
            axis_sec.plot(self.x_data, second, '--')
            axis_sec.set_ylabel('$\partial^2 f / \partial x^2$')

            fig.legend(('first', 'second'), bbox_to_anchor=(1, 1),
                       bbox_transform=axis.transAxes)

            axis.spines['top'].set_visible(False)
            axis_sec.spines['top'].set_visible(False)

        elif which == 'first':
            first = gs.gauss_dx_mult(self.x_data, *self.optim_params)
            axis.plot(self.x_data, first)
            axis.set_ylabel('$\partial f / \partial x$')

            axis.spines['right'].set_visible(False)
            axis.spines['top'].set_visible(False)

        elif which == 'second':
            second = gs.gauss_dxdx_mult(self.x_data, *self.optim_params)
            axis.plot(self.x_data, second)
            axis.set_ylabel('$\partial^2 f / \partial x^2$')

            axis.spines['right'].set_visible(False)
            axis.spines['top'].set_visible(False)

        axis.xaxis.set_minor_locator(AutoMinorLocator(2))
        axis.yaxis.set_minor_locator(AutoMinorLocator(2))

        axis.set_xlabel('$x$')

        if plot_mu:
            mu_opt = self.optim_params[0]

            for mu in mu_opt:
                axis.axvline(mu, ls='--', alpha=0.4)

        return fig, axis


class MultipleCurveResolution(ParameterEstimation):
    """Estimate kinetic parameters from spectra by curve resolution (MCR).

    Concentration profiles from the model are resolved against measured
    absorbance spectra with least-squares pure-component absorptivities;
    optional non-spectral states are fitted as model minus data.

    The absorptivities are eliminated by variable projection (they are not
    optimized parameters), assuming the Beer-Lambert law; analytic
    derivatives assume concentrations of constant rank. Negative
    absorptivities are penalized by ``mult_penalty * sum(max(-eps,
    0)**2)``; least-squares solvers see the penalty as extra residual rows,
    while fit statistics use the data residuals only (see
    ``get_objective``).
    """

    # Spectral residuals are not model outputs minus observations, so data
    # keep the legacy analyze_data layout (see __init__ Notes).
    _compiles_experiments = False
    #: Also restore the resolved absorptivities and the projections of the
    #: last evaluation after a failed ``optimize_fn`` call.
    _FIT_ATTRIBUTES = ParameterEstimation._FIT_ATTRIBUTES + (
        'epsilon_mcr', 'projection_kwargs', '_projection_blocks')

    def __init__(self, func, param_seed, time_data, y_spectra, mult_penalty=1,
                 global_analysis=True,
                 args_fun=None, kwargs_fun=None,
                 optimize_flags=None,
                 jac_fun=None, dx_finitediff=None,
                 measured_ind=None, non_spectral_ind=None, weight_matrix=None,
                 name_params=None, name_states=None):
        """Create a multivariate curve resolution (MCR) estimator.

        Parameters
        ----------
        func : callable
            Model ``func(params, x, reord_sens=False, *args, **kwargs)``
            returning states, shape ``(n_times, n_states)`` in model units,
            or ``(states, sensitivities)`` with sensitivities of shape
            ``(n_params, n_times, n_states)``.
        param_seed : array-like
            Parameter seed in the units required by ``func``.
        time_data : numpy.ndarray, list or dict
            Times per experiment [s]; for non-spectral states measured at
            other times, a dictionary per experiment with ``'spectra'`` and
            ``'non_spectra'`` time arrays, as for ``ParameterEstimation``.
        y_spectra : numpy.ndarray, list or dict
            Absorbance spectra, shape ``(n_times, n_lambda)`` [-], or per
            experiment a dictionary with ``'spectra'`` and
            ``'non_spectra'`` observations (model state units).
        mult_penalty : float, optional
            Nonnegative weight of the penalty ``mult_penalty * sum(min(eps,
            0)**2)`` on negative pure-component absorptivities ``eps``, added
            to the objective and represented in the least-squares residual
            vector by the rows ``sqrt(2 * mult_penalty) * max(-eps, 0)``;
            units of the objective per squared absorptivity unit (absorbance
            per concentration unit). The default 1 is a scaling assumption,
            not a calibrated value; 0 disables the penalty.
        global_analysis : bool, optional
            If True (the default), resolve one set of absorptivities for all
            experiments. If False, each experiment resolves its own
            pure-component spectra (``get_local_analysis``).
        args_fun, kwargs_fun, optimize_flags, jac_fun, dx_finitediff,
        name_params, name_states : optional
            As for ``ParameterEstimation``.
        measured_ind : list or dict
            States resolved from the spectra, or a dictionary with
            ``'spectra'`` and optional ``'non_spectra'`` state indices.
        non_spectral_ind : optional
            Unused; kept for call compatibility.
        weight_matrix : numpy.ndarray, optional
            Measurement-error covariance (data minus truth) of the residual
            columns: every spectral channel followed by every non-spectral
            state, shape ``(n_lambda + n_non, n_lambda + n_non)``. Entry
            ``(i, j)`` has the product of the units of columns ``i`` and
            ``j``: absorbance [-] for spectral channels, the model-state
            unit for non-spectral states. The default is the identity.

        Raises
        ------
        ValueError
            If ``mult_penalty`` is negative or not finite, a spectra
            dictionary lacks ``'spectra'`` or ``weight_matrix`` does not
            match the residual columns, besides the ``ParameterEstimation``
            input errors.
        TypeError
            If ``time_data`` holds ``Experiment`` or ``Measurement``
            objects, or ``mult_penalty`` is not a real number.

        Notes
        -----
        Spectral residuals are data minus prediction and non-spectral ones
        model minus data, so the residual covariance is
        ``D @ weight_matrix @ D`` with ``D = diag(+1 spectral, -1
        non-spectral``); ``sigma_inv`` and the marginal roots of partially
        observed rows are built for it. Block-diagonal weights are
        unaffected.

        Data keep the legacy ``analyze_data`` layout instead of the
        ``Experiment``/``Measurement`` compilation of ``ParameterEstimation``:
        spectral residuals are absorbances resolved by curve resolution, with
        one column per wavelength, not model-output columns minus data.
        """
        if _holds_experiments(time_data) or _holds_measurements(time_data):
            raise TypeError(
                "MultipleCurveResolution does not accept Experiment or "
                "Measurement objects; pass spectral observations through "
                "time_data and y_spectra")
        if (isinstance(mult_penalty, bool)
                or not isinstance(mult_penalty, numbers.Real)):
            raise TypeError("mult_penalty must be a real number; got "
                            f"{type(mult_penalty).__name__}")
        if not np.isfinite(mult_penalty) or mult_penalty < 0:
            raise ValueError("mult_penalty must be finite and nonnegative; "
                             f"got {mult_penalty!r}")

        super().__init__(func, param_seed, time_data, y_spectra, measured_ind,
                         args_fun, kwargs_fun, optimize_flags, jac_fun,
                         dx_finitediff, weight_matrix,
                         name_params, name_states)

        self.fit_spectra = True

        y_spectral = []
        y_data = []
        for y in self.y_data:
            if isinstance(y, dict):
                if 'spectra' not in y:
                    raise ValueError(
                        "Data dictionary must have the 'spectra' key")

                y_spectral.append(y)

            elif isinstance(y, np.ndarray):
                y_di = {'spectra': y}
                y_spectral.append(y_di)

        self.len_spectra = [data['spectra'].shape[0] for data in y_spectral]
        self.size_spectra = [data['spectra'].size for data in y_spectral]

        keys = list(set().union(*[di.keys() for di in y_spectral]))

        y_concat = {}
        for key in keys:
            li = [di[key] for di in y_spectral if key in di]
            y_concat[key] = np.vstack(li)

        self.spectra_tot = y_concat['spectra']
        self.y_concat = np.hstack(list(y_concat.values()))

        self.global_analysis = global_analysis

        if isinstance(measured_ind, (tuple, list, range)):
            self.measured_ind = {'spectra': measured_ind}

        self.has_non = False
        if 'non_spectra' in self.measured_ind:
            self.has_non = True

        num_lambda = y_spectral[0]['spectra'].shape[1]  # [-], channels
        num_non = len(self.measured_ind.get('non_spectra', []))  # [-]
        size_sigma = num_lambda + num_non  # [-], residual columns
        if weight_matrix is None:
            self.sigma_inv = np.eye(size_sigma)
            self._measurement_covariance = np.eye(size_sigma)  # [-]
        else:
            if self._measurement_covariance.shape != (size_sigma,
                                                      size_sigma):
                raise ValueError(
                    "weight_matrix must have one row and column per spectral "
                    f"channel ({num_lambda}) followed by one per non-spectral "
                    f"state ({num_non}); got shape "
                    f"{self._measurement_covariance.shape}")
            # Spectral residuals are data minus prediction, non-spectral
            # ones model minus data. weight_matrix is the covariance of the
            # measurement errors, so the residual covariance is
            # D @ weight_matrix @ D with D = diag(+1 spectral, -1
            # non-spectral); D @ sigma_inv @ D is its lower precision root.
            # Block-diagonal weights are unchanged.
            residual_signs = np.concatenate(
                (np.ones(num_lambda), -np.ones(num_non)))  # [-]
            sign_flip = np.outer(residual_signs, residual_signs)  # [-]
            self.sigma_inv = self.sigma_inv * sign_flip
            self._measurement_covariance = (
                self._measurement_covariance * sign_flip)
        # Marginal roots must follow the residual-column covariance above.
        self._observed_roots = {}

        # Legacy: projection of the last global analysis, no longer read.
        self.projection_kwargs = {}
        # (first row [-], end row [-], C [conc unit], pinv(C) [1/conc
        # unit], absorptivities [-/conc unit]) per resolution block of the
        # last evaluation; see _projection_sensitivities.
        self._projection_blocks = []
        # Optimized parameters [parameter units] and results of the last
        # objective evaluation, whatever set_self was.
        self._params_mcr_eval = None
        self._mcr_evaluation = None
        self.mult_penalty = mult_penalty  # [objective unit / eps unit**2]

    def _jacobian_source(self) -> str:
        """Return the origin of the weighted residual Jacobian.

        Returns
        -------
        {'model', 'finite_difference'}
            ``'model'`` when the model callback returns sensitivities with
            its outputs (``sens_second`` is stored), otherwise
            ``'finite_difference'``: ``get_gradient`` of this class does not
            call ``jac_fun``.
        """
        if self.sens_second is not None:
            return 'model'
        return 'finite_difference'

    def _weighted_predictions(self) -> np.ndarray:
        """Return the accepted MCR predictions weighted like the residuals.

        Returns
        -------
        numpy.ndarray
            Shape ``(len(info_opt['fun']),)`` [weighted residual unit]: the
            stacked predictions of ``_model_predictions`` (curve-resolved
            spectra ``C eps`` [-] from ``y_runs``, then the non-spectral
            model values [model units]; zero where unobserved), each column
            multiplied by its residual sign (``_residual_signs``: -1
            spectral, +1 non-spectral) and whitened by
            ``_weight_sample_rows``, flattened column-major like
            ``_assemble_residuals``.

        Notes
        -----
        The spectral predictions are the raw ``y_runs``. The non-spectral
        model values are not stored separately: ``_model_predictions``
        rebuilds them as residuals plus observations, which loses accuracy
        when they are far below the data.
        """
        # [-] spectral, [model units] non-spectral; NaN where unobserved
        predictions = np.vstack(self._model_predictions())
        signed = (np.nan_to_num(predictions)
                  * self._residual_signs(0))  # [-] / [model units]
        # [weighted residual unit]
        weighted = self._weight_sample_rows(
            signed, self._residual_observation_mask())
        return weighted.T.ravel()

    def _residual_blocks(self, values: np.ndarray) -> list:
        """Split a vector in the MCR residual layout into sample rows.

        Parameters
        ----------
        values : numpy.ndarray
            Shape ``(len(info_opt['fun']),)``, flattened column-major over
            the stacked model-grid rows of all experiments, as in
            ``_assemble_residuals`` [weighted residual unit].

        Returns
        -------
        list of tuple
            One ``(rows, mask)``: ``rows`` of shape ``(sum(len_spectra),
            n_lambda + n_non)`` and ``_residual_observation_mask()``.
        """
        num_rows = int(sum(self.len_spectra))  # [-]
        rows = values.reshape(-1, num_rows).T  # [weighted residual unit]
        return [(rows, self._residual_observation_mask())]

    def _residual_observation_mask(self) -> Optional[np.ndarray]:
        """Return the observation mask of the stacked MCR residual columns.

        Returns
        -------
        numpy.ndarray or None
            Boolean array, shape ``(sum(len_spectra), n_lambda + n_non)``:
            rows are model-grid times of all experiments in ``x_data``
            order, columns every spectral channel followed by every
            non-spectral state, True where the entry was measured. None when
            every entry was measured.

        Notes
        -----
        A dictionary mask stores one row mask per group (``'spectra'``,
        ``'non_spectra'``), shared by the group's columns. Unobserved
        non-spectral entries have zero residual (``get_global_analysis``)
        and, through this mask, zero weighted residual and sensitivity.
        """
        num_lambda = self.spectra_tot.shape[1]  # [-]
        num_non = len(self.measured_ind.get('non_spectra', []))  # [-]
        blocks = []
        for num_rows, x_mask in zip(self.len_spectra, self.x_masks):
            if isinstance(x_mask, dict):
                columns = [np.asarray(x_mask['spectra'], dtype=bool)]
                columns *= num_lambda
                if num_non:
                    columns += ([np.asarray(x_mask['non_spectra'],
                                            dtype=bool)] * num_non)
                blocks.append(np.column_stack(columns))
            else:
                blocks.append(np.ones((num_rows, num_lambda + num_non),
                                      dtype=bool))

        mask = np.vstack(blocks)
        return None if mask.all() else mask

    def _observed_residual_entries(self, num_entries: int) -> np.ndarray:
        """Return which entries of the MCR residual vector are observed.

        Parameters
        ----------
        num_entries : int
            Length of ``get_objective(..., out_array=True)``: the data
            residuals followed by the absorptivity-penalty rows [-].

        Returns
        -------
        numpy.ndarray
            Boolean, shape ``(num_entries,)``: ``_residual_observation_mask``
            flattened column-major over the stacked model-grid rows, as the
            data residuals are (``_assemble_residuals``), then True for
            every penalty row.

        Raises
        ------
        ValueError
            If ``num_entries`` is smaller than the number of data residuals.
        """
        num_rows = int(sum(self.len_spectra))  # [-]
        num_columns = (self.spectra_tot.shape[1]
                       + len(self.measured_ind.get('non_spectra', [])))  # [-]
        mask = self._residual_observation_mask()
        data = (np.ones(num_rows * num_columns, dtype=bool) if mask is None
                else mask.T.ravel())
        if data.size > num_entries:
            raise ValueError(
                f"The MCR residual vector has {num_entries} entries, fewer "
                f"than its {data.size} data residuals")
        return np.concatenate((data, np.ones(num_entries - data.size,
                                             dtype=bool)))

    def get_sens_projection(self, c_target: np.ndarray, c_plus: np.ndarray,
                            sens_states: np.ndarray,
                            spectra: Optional[np.ndarray] = None
                            ) -> np.ndarray:
        """Differentiate the curve-resolved spectra of one projection block.

        Parameters
        ----------
        c_target : numpy.ndarray
            Concentrations of the spectrally active states, shape
            ``(n_rows, n_spectral)``, in the model's concentration unit; their
            rank is assumed constant near the evaluated parameters.
        c_plus : numpy.ndarray
            Pseudo-inverse of ``c_target``, shape ``(n_spectral, n_rows)``,
            in reciprocal concentration units.
        sens_states : numpy.ndarray
            Sensitivities of ``c_target``, shape ``(n_params_total, n_rows,
            n_spectral)`` or ``(n_params, n_rows, n_spectral)`` with only the
            optimized parameters, in concentration unit per parameter unit.
        spectra : numpy.ndarray, optional
            Measured absorbance of the block, shape ``(n_rows, n_lambda)``
            [-]. The default is ``spectra_tot``, the global-analysis block.

        Returns
        -------
        numpy.ndarray
            Derivative of the predicted absorbance ``c_target @ c_plus @
            spectra`` with respect to each optimized parameter, shape
            ``(n_params, n_rows, n_lambda)``, in absorbance [-] per parameter
            unit.

        Notes
        -----
        For ``C`` of constant rank the projector ``C @ C+`` has the
        derivative ``P dC C+ + (P dC C+)^T`` with ``P = I - C @ C+``
        (G. H. Golub and V. Pereyra, "The differentiation of pseudo-inverses
        and nonlinear least squares problems whose variables separate",
        SIAM J. Numer. Anal. 10(2):413-432, 1973); it is undefined where the
        rank changes.
        """
        if spectra is None:
            spectra = self.spectra_tot  # [-]

        eye = np.eye(c_target.shape[0])  # [-], identity over the block rows
        proj_orthogonal = eye - np.dot(c_target, c_plus)  # [-], P = I - C C+

        if sens_states.shape[0] == sum(self.map_variable):
            # [conc unit / param unit], optimized parameters
            sens_pick = sens_states
        else:
            # [conc unit / param unit], optimized parameters
            sens_pick = sens_states[self.map_variable]

        # [1/param unit], (n_params, n_rows, n_rows)
        first_term = proj_orthogonal @ sens_pick @ c_plus
        second_term = first_term.transpose((0, 2, 1))  # [1/param unit]
        sens_an = (first_term + second_term) @ spectra  # [-/param unit]

        return sens_an

    def _projection_sensitivities(self, sens_spectral: np.ndarray
                                  ) -> tuple[np.ndarray, np.ndarray]:
        """Differentiate the resolved spectra and absorptivities.

        Parameters
        ----------
        sens_spectral : numpy.ndarray
            Sensitivities of the spectrally active states of all experiments
            stacked by row, shape ``(n_params_total, sum(len_spectra),
            n_spectral)`` or with only the optimized parameters, in
            concentration unit per parameter unit.

        Returns
        -------
        sens_spectra : numpy.ndarray
            Derivative of the predicted absorbance, shape ``(n_params,
            sum(len_spectra), n_lambda)``, in absorbance [-] per parameter
            unit.
        sens_absorptivity : numpy.ndarray
            Derivative of ``epsilon_mcr``, shape ``(n_params,) +
            epsilon_mcr.shape``, in absorptivity unit per parameter unit.

        Notes
        -----
        Uses the projection blocks of the last objective evaluation: one for
        global analysis, one per experiment for local analysis. For a block
        with concentrations ``C`` of constant rank near the evaluated
        parameters, spectra ``S``, absorptivities ``eps = C+ S`` and
        spectral residual ``R = S - C eps``, the pseudoinverse derivative
        is (G. H. Golub and V. Pereyra, "The differentiation of
        pseudo-inverses and nonlinear least squares problems whose
        variables separate", SIAM J. Numer. Anal. 10(2):413-432, 1973)::

            dC+ = -C+ dC C+ + C+ C+^T dC^T (I - C C+)
                  + (I - C+ C) dC^T C+^T C+

        so that::

            deps = dC+ S = -C+ dC eps + C+ C+^T dC^T R
                           + (I - C+ C) dC^T C+^T eps

        The first two terms follow from differentiating the normal
        equations ``C^T C eps = C^T S``. The third is the component of
        ``dC+`` in the null space of ``C``: differentiating the identity
        ``C+ = C^T C+^T C+`` and multiplying by ``I - C+ C``, which
        annihilates ``C^T``, gives ``(I - C+ C) dC+ = (I - C+ C) dC^T
        C+^T C+``. It vanishes for full column rank, where ``C+ C = I``.
        The projector derivative ``d(C C+) = P dC C+ + (P dC C+)^T`` with
        ``P = I - C C+`` (``get_sens_projection``) also needs only constant
        rank. Both derivatives are undefined where the rank of ``C``
        changes; the central-difference regressions in
        ``tests/test_paramestim_mcr_objective_consistency.py`` cover the
        data and penalty rows of full-rank and constant-rank-deficient
        blocks.
        """
        if sens_spectral.shape[0] == sum(self.map_variable):
            # [conc unit / param unit], (n_params, n_rows, n_spectral)
            sens_pick = sens_spectral
        else:
            # [conc unit / param unit], optimized parameters
            sens_pick = sens_spectral[self.map_variable]

        # [-/param unit], one (n_params, n_rows, n_lambda) per block
        spectra_blocks = []
        absorptivity_blocks = []  # [absorptivity / param unit], one per block
        for start, stop, c_target, c_plus, absorptivity in (
                self._projection_blocks):
            # (n_params, n_rows, n_spectral) [conc unit / param unit]
            sens_block = sens_pick[:, start:stop]
            spectra = self.spectra_tot[start:stop]  # [-]
            spectra_blocks.append(self.get_sens_projection(
                c_target, c_plus, sens_block, spectra=spectra))

            resid = spectra - c_target @ absorptivity  # [-]
            # C+ C+^T [1/conc unit**2]; inv(C^T C) for full column rank.
            gram_plus = c_plus @ c_plus.T
            # I - C+ C [-], projector onto the null space of C.
            null_proj = np.eye(c_plus.shape[0]) - c_plus @ c_target
            # dC^T [conc unit / param unit], (n_params, n_spectral, n_rows)
            sens_t = sens_block.transpose(0, 2, 1)
            # (n_params, n_spectral, n_lambda) [absorptivity / param unit]
            absorptivity_blocks.append(
                gram_plus @ sens_t @ resid
                - c_plus @ sens_block @ absorptivity
                + null_proj @ sens_t @ c_plus.T @ absorptivity)

        # [-/param unit], (n_params, sum(len_spectra), n_lambda)
        sens_spectra = np.concatenate(spectra_blocks, axis=1)
        if self.global_analysis:
            # [absorptivity / param unit]
            sens_absorptivity = absorptivity_blocks[0]
        else:
            # [absorptivity / param unit]
            sens_absorptivity = np.stack(absorptivity_blocks, axis=1)

        return sens_spectra, sens_absorptivity

    def _penalty_residuals(self, absorptivity: np.ndarray) -> np.ndarray:
        """Return the residual rows of the absorptivity penalty.

        Parameters
        ----------
        absorptivity : numpy.ndarray
            Resolved absorptivities, shape ``(n_spectral, n_lambda)`` for
            global analysis or ``(n_experiments, n_spectral, n_lambda)`` for
            local analysis, in absorbance per model concentration unit.

        Returns
        -------
        numpy.ndarray
            ``sqrt(2 * mult_penalty) * max(-absorptivity, 0)`` flattened in
            C order (experiment, species, wavelength), shape
            ``(absorptivity.size,)``, in the unit of the weighted residuals.

        Notes
        -----
        The factor 2 matches the ``1/2 r @ r`` objective convention, so half
        the sum of squares of these rows is the penalty term
        ``mult_penalty * sum(max(-absorptivity, 0)**2)`` of
        ``get_objective``. Nonnegative absorptivities give zero rows.
        """
        violation = np.maximum(-absorptivity, 0)  # [absorptivity unit]
        return np.sqrt(2 * self.mult_penalty) * violation.ravel()

    def _penalty_sensitivities(self, absorptivity: np.ndarray,
                               sens_absorptivity: np.ndarray) -> np.ndarray:
        """Differentiate the absorptivity-penalty residual rows.

        Parameters
        ----------
        absorptivity : numpy.ndarray
            Resolved absorptivities, as for ``_penalty_residuals``.
        sens_absorptivity : numpy.ndarray
            Their derivatives, shape ``(n_params,) + absorptivity.shape``,
            in absorptivity unit per parameter unit.

        Returns
        -------
        numpy.ndarray
            Derivatives of ``_penalty_residuals(absorptivity)``, shape
            ``(absorptivity.size, n_params)``, in the weighted-residual unit
            per parameter unit. Rows of nonnegative absorptivities are zero
            (the one-sided derivative of ``max(-eps, 0)`` at ``eps = 0``).
        """
        num_params = sens_absorptivity.shape[0]  # [-]
        active = absorptivity < 0  # bool, same shape as absorptivity
        # [weighted residual / param unit]
        sens = -np.sqrt(2 * self.mult_penalty) * sens_absorptivity * active
        return sens.reshape(num_params, -1).T

    def _evaluate_runs(self, params: np.ndarray
                       ) -> tuple[list, Union[np.ndarray, list]]:
        """Evaluate the model for every experiment.

        Parameters
        ----------
        params : numpy.ndarray
            Full parameter vector in the model's units.

        Returns
        -------
        c_runs : list of numpy.ndarray
            Spectrally active concentrations per experiment, shape
            ``(n_times, n_spectral)``, in model concentration units.
        resid_non : numpy.ndarray or list
            Non-spectral residuals, model minus data, of all experiments
            stacked by row, shape ``(sum(len_spectra), n_non)`` in model
            units and zero where unobserved; an empty list without
            non-spectral states.

        Raises
        ------
        ValueError
            If a callback output is not two-dimensional with one row per
            model-grid sample of its experiment.

        Notes
        -----
        The model is called with ``reord_sens=False``. When it returns
        sensitivities, shape ``(n_params_total, n_times, n_states)``, they
        are stored as ``sens_second``. The cached evaluation of
        ``get_objective`` is invalidated, so a direct call of
        ``get_global_analysis`` or ``get_local_analysis`` makes
        ``get_gradient`` evaluate the model again.
        """
        # This evaluation replaces the projection and sensitivities that
        # get_gradient reuses; get_objective records its parameters after.
        self._params_mcr_eval = None
        self._mcr_evaluation = None

        c_runs = []  # [conc unit], one (n_times, n_spectral) per run
        states_non = []  # [model units], one (n_times, n_non) per run
        sens_states = []  # [state unit / param unit], one per run

        y_data_non = []  # [model units], one (n_times, n_non) per run
        for ind in range(self.num_datasets):
            result = self.function(params, self.x_model[ind],
                                   reord_sens=False,
                                   *self.args_fun[ind], **self.kwargs_fun[ind])

            if isinstance(result, tuple):
                # [model units], [state unit / param unit]
                states, sens_st = result
                sens_states.append(sens_st)
            else:
                states = result  # [model units]

            self._check_model_output(ind, states, two_dimensional=True)
            # [conc unit]
            conc_target = states[:, self.measured_ind['spectra']]

            if self.has_non:
                # [model units]
                non_spectra = states[:, self.measured_ind['non_spectra']]

                # [model units]
                y_non = self.y_data[ind]['non_spectra'].copy()
                x_mask = self.x_masks[ind]['non_spectra']
                if x_mask is not None:
                    y_non[~x_mask] = non_spectra[~x_mask]

                states_non.append(non_spectra)
                y_data_non.append(y_non)

            c_runs.append(conc_target)

        if self.has_non:
            states_non = np.vstack(states_non)  # [model units], stacked rows
            y_data_non = np.vstack(y_data_non)  # [model units], stacked rows

            # [model units], model minus data
            resid_non = states_non - y_data_non

        else:
            resid_non = []

        if len(sens_states) > 0:
            self.sens_second = sens_states

        return c_runs, resid_non

    def _assemble_residuals(self, spectra_pred: np.ndarray,
                            resid_non: Union[np.ndarray, list]
                            ) -> tuple[list, list, np.ndarray]:
        """Assemble raw and weighted residuals from predicted spectra.

        Parameters
        ----------
        spectra_pred : numpy.ndarray
            Predicted absorbance of all experiments stacked by row, shape
            ``(sum(len_spectra), n_lambda)`` [-].
        resid_non : numpy.ndarray or list
            Non-spectral residuals from ``_evaluate_runs``.

        Returns
        -------
        y_runs : list of numpy.ndarray
            Predicted spectra per experiment, ``(n_times, n_lambda)`` [-].
        resid_runs : list of numpy.ndarray
            Raw residuals per experiment, ``(n_times, n_lambda + n_non)``:
            spectral data minus prediction [-], then non-spectral model
            minus data (zero where unobserved) in model units.
        weighted_resid : numpy.ndarray
            Row-whitened residuals flattened column-major, shape
            ``(n_data,)`` (see ``_weight_sample_rows``); unobserved entries
            are zero.
        """
        # [-], spectral data minus prediction
        residuals = self.spectra_tot - spectra_pred
        if self.has_non:
            # [-] spectral, [model units] non-spectral
            residuals = np.hstack((residuals, resid_non))

        # Row-wise whitening: sigma_inv for fully observed rows, the
        # marginal precision root for partially observed rows, zero for
        # unobserved entries (ParameterEstimation._weight_sample_rows).
        # [weighted residual unit]
        weighted_resid = self._weight_sample_rows(
            residuals, self._residual_observation_mask())

        # [-], row indices splitting the experiments
        trim_y = np.cumsum(self.len_spectra)[:-1]

        y_runs = np.split(spectra_pred, trim_y, axis=0)  # [-], per experiment
        # [-] spectral, [model units] non-spectral
        resid_runs = np.split(residuals, trim_y, axis=0)

        # [weighted residual unit], column-major
        weighted_resid = weighted_resid.T.ravel()

        return y_runs, resid_runs, weighted_resid

    def get_gradient(self, params: np.ndarray, out_array: bool = False,
                     set_self: bool = True) -> np.ndarray:
        """Return the Jacobian of the MCR residuals or the objective gradient.

        Parameters
        ----------
        params : array_like
            Optimized parameters in the model's units.
        out_array : bool, optional
            If True, return the Jacobian of the residual vector
            ``get_objective(params, out_array=True)`` for least-squares
            solvers; otherwise the gradient of the scalar objective for
            IPOPT. The default is False.
        set_self : bool, optional
            Forwarded to ``get_objective`` when the model has to be
            evaluated at ``params`` first (model-returned sensitivities of
            another point, or no evaluation yet). The default is True.

        Returns
        -------
        numpy.ndarray
            With ``out_array``, shape ``(n_params, n_data + n_penalty)``:
            derivatives of the weighted data residuals (column-major over
            residual columns and stacked times; unobserved non-spectral
            entries are zero) followed by those of the penalty rows, per
            parameter unit. Otherwise the gradient of ``get_objective``,
            shape ``(n_params,)``, in objective unit per parameter unit.

        Raises
        ------
        ValueError
            If the finite-difference path is used (no model-returned
            sensitivities) and ``dx_finitediff`` gives a zero or infinite
            step for an optimized parameter.

        Notes
        -----
        Without model-returned sensitivities both results are forward
        differences of ``get_objective``, so they include the penalty. With
        them, the data rows come from the variable-projection derivative of
        the spectra (``get_sens_projection``), the penalty rows from the
        absorptivity derivative (``_projection_sensitivities``), and the
        gradient is ``J @ r`` with ``J`` the Jacobian above and ``r`` the
        augmented residual vector of ``get_objective``: the exact derivative
        of ``f = 1/2 r_data @ r_data + mult_penalty * sum(max(-eps, 0)**2) =
        1/2 r @ r``.
        """
        stale = (self._params_mcr_eval is None or
                 not np.array_equal(params, self._params_mcr_eval))
        if stale and (self.sens_second is not None or
                      self._params_mcr_eval is None):
            self.get_objective(params, out_array=out_array,
                               set_self=set_self)

        if self.sens_second is None:
            pick_p = np.where(self.map_variable)[0]

            if out_array:
                # Finite differences of the residual vector are already its
                # Jacobian, (n_data + n_penalty, n_params).
                # [weighted residual / param unit], (n_rows, n_params)
                jacobian = numerical_jac_data(
                    self.get_objective, params, args=(True, False),
                    dx=self.dx_fd, pick_x=pick_p)
                return jacobian.T

            # [objective unit / param unit], (1, n_params)
            gradient = numerical_jac(self.get_objective, params,
                                     args=(False, False), dx=self.dx_fd,
                                     pick_x=pick_p)
            return gradient[0]

        raw_sens = self.sens_second  # [state unit / param unit], one per run
        # [state unit / param unit], (n_par, n_times, n_states)
        sens_tot = np.concatenate(raw_sens, axis=1)

        # [conc unit / param unit]
        sens_mcr = sens_tot[:, :, self.measured_ind['spectra']]

        # Variable projection derivatives: n_par x n_times x n_lambda for the
        # spectra and n_par x epsilon_mcr.shape for the absorptivities.
        sens_spectra, sens_absorptivity = self._projection_sensitivities(
            sens_mcr)

        x_mask = self._residual_observation_mask()
        if self.has_non:
            # [model unit / param unit]
            sens_regular = sens_tot[:, :, self.measured_ind['non_spectra']]

            # weighted_sens holds minus the residual derivative (it is
            # negated below): the spectral residual is data minus
            # prediction, the non-spectral one model minus data.
            # [-/param unit] spectral, [model unit / param unit] non-spectral
            all_sens = np.concatenate((sens_spectra, -sens_regular),
                                      axis=2)

            n_lambda = sens_spectra.shape[2]  # [-], wavelengths
            # (n_par, n_times, n_columns); rows whitened like the
            # residuals, unobserved entries zero.
            # [weighted residual / param unit]
            weighted_all = self._weight_sample_rows(
                all_sens.transpose(1, 0, 2), x_mask).transpose(1, 0, 2)

            # [weighted residual / param unit], (n_rows, n_params)
            weighted_sp = flatten_spectral_sens(weighted_all[:, :, :n_lambda])
            # [weighted residual / param unit], (n_rows, n_params)
            weighted_reg = flatten_spectral_sens(weighted_all[:, :, n_lambda:])

            # [weighted residual / param unit], data rows
            weighted_sens = np.vstack((weighted_sp, weighted_reg))
        else:
            # [weighted residual / param unit]
            weighted_sens = self._weight_sample_rows(
                sens_spectra.transpose(1, 0, 2), x_mask).transpose(1, 0, 2)
            # [weighted residual / param unit], (n_data, n_params)
            weighted_sens = flatten_spectral_sens(weighted_sens)

        # [absorptivity unit]
        absorptivity = self._mcr_evaluation['absorptivity']
        # [weighted residual / param unit], (n_penalty, n_params)
        penalty_sens = self._penalty_sensitivities(absorptivity,
                                                   sens_absorptivity)
        # [weighted residual / param unit], data then penalty rows
        weighted_sens = np.vstack((weighted_sens, -penalty_sens))

        # [weighted residual / param unit], (n_params, n_rows)
        jacobian = -weighted_sens.T
        if out_array:
            return jacobian
        else:
            # [objective unit / parameter unit]
            return jacobian @ self._mcr_evaluation['residuals']

    def get_global_analysis(self, params: np.ndarray
                            ) -> tuple[list, list, np.ndarray, np.ndarray]:
        """Resolve all experiments with one set of absorptivities.

        Parameters
        ----------
        params : numpy.ndarray
            Full parameter vector in the model's units.

        Returns
        -------
        y_runs : list of numpy.ndarray
            Predicted spectra per experiment, ``(n_times, n_lambda)`` [-].
        resid_runs : list of numpy.ndarray
            Raw residuals per experiment, ``(n_times, n_lambda + n_non)``:
            spectral data minus prediction [-], then non-spectral model
            minus data (zero where unobserved) in model units.
        weighted_resid : numpy.ndarray
            Row-whitened residuals flattened column-major, shape
            ``(n_data,)`` (see ``_weight_sample_rows``); unobserved entries
            are zero.
        absorptivity_pure : numpy.ndarray
            Least-squares absorptivities, ``(n_spectral_states, n_lambda)``,
            in absorbance per model concentration unit.

        Raises
        ------
        ValueError
            If a callback output is not two-dimensional with one row per
            model-grid sample of its experiment.

        Notes
        -----
        Stores the projection of the stacked concentrations in
        ``_projection_blocks`` for the model-sensitivity Jacobian, and as
        ``projection_kwargs`` (``c_target`` [concentration unit] and its
        pseudo-inverse ``c_plus``), a legacy attribute that ``get_gradient``
        no longer reads.
        """
        c_runs, resid_non = self._evaluate_runs(params)

        conc_tot = np.vstack(c_runs)  # [conc unit], stacked rows

        # MCR
        conc_plus = np.linalg.pinv(conc_tot)  # [1/conc unit]
        # [-/conc unit], absorbance per concentration
        absorptivity_pure = np.dot(conc_plus, self.spectra_tot)

        spectra_pred = np.dot(conc_tot, absorptivity_pure)  # [-]

        self.projection_kwargs = {'c_target': conc_tot, 'c_plus': conc_plus}
        self._projection_blocks = [
            (0, conc_tot.shape[0], conc_tot, conc_plus, absorptivity_pure)]

        y_runs, resid_runs, weighted_resid = self._assemble_residuals(
            spectra_pred, resid_non)

        return y_runs, resid_runs, weighted_resid, absorptivity_pure

    def get_local_analysis(self, params: np.ndarray
                           ) -> tuple[list, list, np.ndarray, np.ndarray]:
        """Resolve every experiment with its own absorptivities.

        Parameters
        ----------
        params : numpy.ndarray
            Full parameter vector in the model's units.

        Returns
        -------
        y_runs : list of numpy.ndarray
            Predicted spectra per experiment, ``(n_times, n_lambda)`` [-].
        resid_runs : list of numpy.ndarray
            Raw residuals per experiment, ``(n_times, n_lambda + n_non)``,
            as for ``get_global_analysis``.
        weighted_resid : numpy.ndarray
            Row-whitened residuals, shape ``(n_data,)``, in the layout,
            weighting and observation masking of ``get_global_analysis``.
        absorptivity_pure : numpy.ndarray
            Least-squares absorptivities of each experiment, shape
            ``(n_experiments, n_spectral_states, n_lambda)`` in ``x_data``
            order, in absorbance per model concentration unit.

        Raises
        ------
        ValueError
            If a callback output is not two-dimensional with one row per
            model-grid sample of its experiment.

        Notes
        -----
        Each experiment resolves its own pure-component spectra: its
        absorptivities are the least-squares solution for its own spectra
        and concentrations only, for example when the optical path or
        instrument differs between experiments. Non-spectral residuals,
        weights and masks are those of the global analysis, so only the
        spectral prediction differs.
        """
        c_runs, resid_non = self._evaluate_runs(params)

        # [-], first row of each experiment
        row_bounds = np.concatenate(([0], np.cumsum(self.len_spectra)))
        pred_runs = []  # [-], projected spectra per experiment
        absorptivities = []  # [-/conc unit], per experiment
        # Projection blocks, entries as in _projection_blocks.
        blocks = []
        for ind, conc_target in enumerate(c_runs):
            start, stop = row_bounds[ind], row_bounds[ind + 1]
            spectra = self.spectra_tot[start:stop]  # [-]

            # MCR of this experiment
            conc_plus = np.linalg.pinv(conc_target)  # [1/conc unit]
            absorptivity = np.dot(conc_plus, spectra)  # [-/conc unit]
            pred_runs.append(np.dot(conc_target, absorptivity))

            absorptivities.append(absorptivity)
            blocks.append((start, stop, conc_target, conc_plus,
                           absorptivity))

        self._projection_blocks = blocks

        y_runs, resid_runs, weighted_resid = self._assemble_residuals(
            np.vstack(pred_runs), resid_non)

        return y_runs, resid_runs, weighted_resid, np.stack(absorptivities)

    def get_objective(self, params: np.ndarray, out_array: bool = False,
                      set_self: Optional[bool] = None, *,
                      update_self: Optional[bool] = None
                      ) -> Union[float, np.ndarray]:
        """Evaluate the MCR residual vector or the scalar objective.

        Parameters
        ----------
        params : array_like
            Optimized parameters in the model's units; fixed parameters are
            taken from ``param_seed``.
        out_array : bool, optional
            If True, return the residual vector minimized by least-squares
            solvers; otherwise the scalar objective. The default is False.
        set_self : bool, optional
            If True, store the reporting state of this evaluation:
            ``residuals``, ``y_runs``, ``resid_runs``, ``epsilon_mcr`` and
            the ``params_iter``/``objfun_iter`` histories. The default is
            True (None means True). ``assemble_solver_info`` passes False
            to keep the state of the final solver callback.
        update_self : bool, optional
            Deprecated keyword alias of ``set_self``, the name of this
            argument before #240.

        Returns
        -------
        float or numpy.ndarray
            With ``out_array``, ``r = [r_data, r_pen]``: the row-whitened
            data residuals of ``get_global_analysis`` or
            ``get_local_analysis``, shape ``(n_data,)``, followed by the
            penalty rows ``sqrt(2 * mult_penalty) * max(-eps, 0)``, one per
            resolved absorptivity ``eps`` in ``epsilon_mcr`` C order, shape
            ``(epsilon_mcr.size,)``. Otherwise the objective ``f = 1/2
            r_data @ r_data + mult_penalty * sum(max(-eps, 0)**2)``, which
            equals ``1/2 r @ r``. Spectral residuals are absorbance [-] and
            non-spectral ones model units, both divided by the root of the
            matching ``weight_matrix`` unit; ``f`` has their squared unit.

        Raises
        ------
        ValueError
            If a callback output is not two-dimensional with one row per
            model-grid sample of its experiment.
        TypeError
            If ``set_self`` and ``update_self`` are both given and differ.

        Warns
        -----
        DeprecationWarning
            If ``update_self`` is given without a conflicting ``set_self``.

        Notes
        -----
        The penalty rows exist only in the solver residual vector and its
        Jacobian. Reported quantities use the data residuals only:
        ``residuals`` (and ``objfun_iter``, the data sum of squares) are
        the weighted data residuals; ``y_runs``, ``resid_runs`` and the
        observation masks cover data entries only; ``optimize_fn`` moves the
        penalty rows of the solver's ``fun`` and ``jac`` to
        ``info['fun_penalty']`` and ``info['jac_penalty']``, so
        ``info['fun']``, ``info['jac']``, the covariance and
        ``StatisticsClass`` use ``n_data`` entries and ``num_data_total``
        degrees of freedom. With nonnegative absorptivities the penalty rows
        are zero.
        """
        if update_self is not None:
            if set_self is not None and bool(set_self) != bool(update_self):
                raise TypeError("set_self and update_self disagree; pass "
                                "only set_self")
            warnings.warn(
                "update_self is deprecated; use set_self, its new name",
                DeprecationWarning, stacklevel=2)
            set_self = update_self
        if set_self is None:
            set_self = True

        params_in = np.array(params, copy=True)  # [parameter units]

        # Reconstruct parameter set with fixed and non-fixed indexes
        params = self.reconstruct_params(params)

        if self.global_analysis:
            out = self.get_global_analysis(params)

        else:
            out = self.get_local_analysis(params)

        y_runs, resid, weighted_resid, molar_abs = out
        # [weighted residual unit], data then penalty rows
        residuals_all = np.concatenate(
            (weighted_resid, self._penalty_residuals(molar_abs)))

        # Evaluation consumed by get_gradient's model-sensitivity branch.
        self._params_mcr_eval = params_in
        self._mcr_evaluation = {'residuals': residuals_all,
                                'absorptivity': molar_abs}

        if set_self:
            if type(self.objfun_iter) is list:
                # [weighted residual unit**2], data rows only
                objfun_val = np.linalg.norm(weighted_resid)**2
                self.objfun_iter.append(objfun_val)

            if type(self.params_iter) is list:
                self.params_iter.append(params)

            self.residuals = weighted_resid
            self.y_runs = y_runs
            self.epsilon_mcr = molar_abs
            self.resid_runs = resid

        # Return objective
        if out_array:
            return residuals_all
        else:
            # [objective unit]
            residual = 1/2 * np.dot(weighted_resid, weighted_resid)
            # [objective unit]
            penalty = self.mult_penalty*(np.maximum(-molar_abs, 0)**2).sum()
            return residual + penalty

    def _restore_fit_state(self, state: dict) -> None:
        """Restore the saved fit and invalidate the evaluation cache.

        Parameters
        ----------
        state : dict
            Result of ``_fit_state``, including ``epsilon_mcr`` [absorptivity
            units] and the projection buffers.

        Notes
        -----
        The cache ``_params_mcr_eval``/``_mcr_evaluation`` that
        ``get_gradient`` reuses describes the last trial evaluation, not the
        restored fit, so it is cleared and the next gradient evaluates the
        model again.
        """
        super()._restore_fit_state(state)
        self._params_mcr_eval = None
        self._mcr_evaluation = None

    def _data_solver_info(self, info: dict) -> dict:
        """Move the absorptivity-penalty rows out of the solver information.

        Parameters
        ----------
        info : dict
            Solver information at the accepted parameters, with ``fun`` the
            residual vector of ``get_objective(out_array=True)`` and ``jac``
            its Jacobian, shape ``(num_params, n_data + n_penalty)``.

        Returns
        -------
        dict
            Copy of ``info`` in which ``fun`` and ``jac`` hold the data rows
            only, shapes ``(n_data,)`` and ``(num_params, n_data)``, and
            ``fun_penalty`` and ``jac_penalty`` the penalty rows, shapes
            ``(n_penalty,)`` and ``(num_params, n_penalty)``. Other entries,
            such as the native SciPy diagnostics, still describe the full
            residual vector.
        """
        # [-], one row per absorptivity
        num_penalty = (len(self.measured_ind['spectra'])
                       * self.spectra_tot.shape[1])  # [-]
        if not self.global_analysis:
            num_penalty *= self.num_datasets  # [-]

        # [weighted residual unit], data then penalty rows
        fun = np.asarray(info['fun'])
        jac = np.asarray(info['jac'])  # [weighted residual / param unit]
        num_data = fun.size - num_penalty  # [-]

        reported = dict(info)
        reported['fun'] = fun[:num_data]
        reported['jac'] = jac[:, :num_data]
        reported['fun_penalty'] = fun[num_data:]
        reported['jac_penalty'] = jac[:, num_data:]
        return reported

    def _model_predictions(self) -> list:
        """Return the curve-resolved predictions of the last evaluation.

        Returns
        -------
        list of numpy.ndarray
            One array per experiment in ``x_data`` order, shape ``(n_times,
            n_lambda + n_non)``, in the column layout of ``resid_runs``: the
            predicted absorbance ``C eps`` [-] of every spectral channel
            (``y_runs``), then the model value of every non-spectral state
            in its model unit. Entries without an observation are NaN, as
            in ``ParameterEstimation``.

        Notes
        -----
        Spectral residuals are data minus prediction, so ``resid_runs +
        y_data`` would give ``2 data - prediction`` for them. Non-spectral
        residuals are model minus data, so their predictions are
        ``resid_runs + y_data`` as in ``ParameterEstimation``.
        """
        # [-] spectral, [model state units] non-spectral; one array per
        # experiment
        predictions = []
        for ind in range(self.num_datasets):
            # [-], (n_times, n_lambda), unobserved rows set to NaN below
            spectra_pred = np.array(self.y_runs[ind], dtype=float)
            x_mask = self.x_masks[ind]
            if isinstance(x_mask, dict):
                spectra_pred[~np.asarray(x_mask['spectra'], dtype=bool)] = (
                    np.nan)
            # Blocks: spectral [-], then non-spectral [model state units]
            columns = [spectra_pred]
            if self.has_non:
                num_lambda = spectra_pred.shape[1]  # [-]
                # [model units], model minus data plus data; NaN where the
                # observation is NaN-padded (unobserved).
                columns.append(self.resid_runs[ind][:, num_lambda:]
                               + self.y_data[ind]['non_spectra'])
            predictions.append(np.hstack(columns))

        return predictions

    def _residual_signs(self, ind: int) -> np.ndarray:
        """Return the sign convention of one experiment's residual columns.

        Parameters
        ----------
        ind : int
            Experiment index in ``x_data`` order.

        Returns
        -------
        numpy.ndarray
            Shape ``(n_lambda + n_non,)`` [-]: -1 for every spectral channel
            (data minus prediction), +1 for every non-spectral state (model
            minus data), in the column order of ``resid_runs``.
        """
        num_lambda = self.spectra_tot.shape[1]  # [-]
        num_non = len(self.measured_ind.get('non_spectra', []))  # [-]
        return np.concatenate((-np.ones(num_lambda), np.ones(num_non)))
