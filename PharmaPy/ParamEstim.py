#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 15:35:48 2019

@author: casas100
"""

import numbers

import numpy as np
from scipy.linalg import cholesky, solve_triangular
from collections.abc import Mapping
from itertools import cycle
from types import MappingProxyType
from typing import Callable, NamedTuple, Optional, Sequence

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable

import pandas as pd
from PharmaPy.jac_module import numerical_jac_data, dx_jac_p, numerical_jac
from PharmaPy import Gaussians as gs

from PharmaPy.LevMarq import levenberg_marquardt
from PharmaPy.Commons import plot_sens, reorder_sens

try:
    from cyipopt import minimize_ipopt
    have_cyipopt = True
except ImportError:
    have_cyipopt = False

linestyles = cycle(['-', '--', '-.', ':'])

eps = np.finfo(float).eps


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
    field : int or str
        Zero-based column of the model callback output, or its name in the
        ``output_names`` given to ``ParameterEstimation``.
    x : array_like
        Independent-variable samples, shape ``(n,)``, in ``x_units``. A
        scalar is a single sampled point. Samples must be finite and
        non-decreasing; a repeated value denotes replicate observations at
        that x.
    values : array_like
        Observations, shape ``(n,)``, already in the units and physical
        basis of the model output column ``field``. NaN marks a missing
        observation, which is excluded from residuals, data counts and
        degrees of freedom; nothing is interpolated or invented.
    uncertainty : float or array_like, optional
        Standard deviation of the measurement error, a positive scalar or
        shape ``(n,)``, in the units and basis of ``values``. It must be
        finite and positive at every observed sample; entries at missing
        samples are ignored. The default None declares no uncertainty.
    units : str, optional
        Label of the observation units, e.g. ``'mol/L'``. No conversion is
        performed; the label documents ``values`` and is checked for
        consistency with other measurements of the same name or field.
    basis : str, optional
        Label of the observation basis, e.g. ``'molar concentration'``; no
        conversion is performed.
    x_units : str or None, optional
        Label of the independent-variable units. The default is ``'s'``
        (time); None leaves them undeclared.

    Raises
    ------
    TypeError
        If ``field`` or a label has an invalid type, or samples are not
        numeric.
    ValueError
        If ``x`` is empty, non-finite or decreasing, ``values`` does not
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

    def __init__(self, field, x, values, *, uncertainty=None, units=None,
                 basis=None, x_units='s') -> None:
        """Validate and store a private copy of one measurement.

        Parameters
        ----------
        field, x, values, uncertainty, units, basis, x_units
            See the class docstring.

        Raises
        ------
        TypeError, ValueError
            See the class docstring.
        """
        self._field = _validate_field(field)

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

    @property
    def field(self):
        """Model-output identity.

        Returns
        -------
        int or str
            Zero-based output column or output name.
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


class Experiment:
    """One experimental run: named measurements and its callback association.

    Parameters
    ----------
    measurements : Mapping[str, Measurement] or sequence of Measurement
        Named measurements; insertion order is the declared order. In
        sequence form each measurement is named by its field: the str field
        itself, or ``f'field_{field}'`` for an int field; replicate
        measurements of one field then need a mapping with distinct names.
    args : tuple or list, optional
        Positional callback arguments for this experiment, passed as
        ``func(params, x_model, *args, **kwargs)`` in model-defined units.
        Stored as a tuple; the values themselves are not copied.
    kwargs : Mapping, optional
        Callback keywords in model-defined units; a shallow copy is stored.

    Raises
    ------
    TypeError
        If ``measurements`` is not a mapping or sequence of
        ``Measurement`` objects, a name is not a str, ``args`` is not a
        tuple or list, or ``kwargs`` is not a mapping with str keys.
    ValueError
        If there are no measurements, a name is empty, the sequence form
        repeats a name, or measurements declare different ``x_units``.

    Notes
    -----
    The experiment model grid is the multiset union of the measurement
    grids: each distinct x appears as many times as the largest number of
    replicates any measurement has at that x, in sorted order. The k-th
    replicate of a measurement at x maps to the k-th grid row with that x;
    rows a measurement does not sample are unobserved for it. Undeclared
    (None) ``x_units`` are compatible with any declared value.
    """

    def __init__(self, measurements, args=(), kwargs=None) -> None:
        """Validate and store the measurements and callback arguments.

        Parameters
        ----------
        measurements, args, kwargs
            See the class docstring.

        Raises
        ------
        TypeError, ValueError
            See the class docstring.
        """
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
                name = field if isinstance(field, str) else f'field_{field}'
                items.append((name, measurement))
            names = [name for name, _ in items]
            repeated = sorted({name for name in names
                               if names.count(name) > 1})
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
        for name, measurement in items:
            if not isinstance(name, str):
                raise TypeError(
                    "Experiment measurement names must be str; got "
                    f"{type(name).__name__} {name!r}")
            if not name:
                raise ValueError(
                    "Experiment measurement names must be non-empty")
            if not isinstance(measurement, Measurement):
                raise TypeError(
                    f"Experiment measurement {name!r} must be a Measurement; "
                    f"got {type(measurement).__name__}")
            checked[name] = measurement

        declared_x_units = {name: measurement.x_units
                            for name, measurement in checked.items()
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
        args_fun=args_fun, kwargs_fun=kwargs_fun, num_data=num_data)


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


def _experiment_collection(x_data) -> Optional[dict]:
    """Normalize ``Experiment`` input to an ordered name mapping.

    Parameters
    ----------
    x_data : object
        ``x_data`` argument of ``ParameterEstimation``.

    Returns
    -------
    dict or None
        Experiment name -> ``Experiment`` in experiment order: mapping keys,
        or ``'exp_1'``, ``'exp_2'``, ... for a single experiment or a list or
        tuple. None if ``x_data`` holds no ``Experiment`` (legacy input).

    Raises
    ------
    TypeError
        If ``Experiment`` objects are mixed with other entries or a mapping
        key is not a str.
    ValueError
        If a mapping key is empty.
    """
    if not _holds_experiments(x_data):
        return None
    if isinstance(x_data, Experiment):
        return {'exp_1': x_data}
    if isinstance(x_data, Mapping):
        items = list(x_data.items())
    else:
        items = [(f'exp_{index + 1}', value)
                 for index, value in enumerate(x_data)]

    mixed = [name for name, value in items
             if not isinstance(value, Experiment)]
    if mixed:
        raise TypeError(
            "x_data mixes Experiment objects with other entries "
            f"{mixed!r}; pass only Experiment objects, or only legacy "
            "arrays")
    for name, _ in items:
        if not isinstance(name, str):
            raise TypeError(
                f"Experiment names must be str; got {type(name).__name__} "
                f"{name!r}")
        if not name:
            raise ValueError("Experiment names must be non-empty")
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

    def __init__(self, func: Callable, param_seed, x_data, y_data=None,
                 measured_ind=None,
                 args_fun=None, kwargs_fun=None,
                 optimize_flags=None,
                 jac_fun=None, dx_finitediff=None,
                 weight_matrix=None,
                 name_params=None, name_states=None,
                 output_names=None) -> None:
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
        x_data : Experiment, mapping or sequence of them, or legacy arrays
            Either ``Experiment`` objects or legacy independent-variable
            data. ``Experiment`` input is a single ``Experiment`` (named
            ``'exp_1'``), a mapping of experiment names to ``Experiment``
            (insertion order is experiment order), or a list or tuple of
            ``Experiment`` (named ``'exp_1'``, ``'exp_2'``, ...); then
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
        dx_finitediff : float, optional
            perturbation in the parameter space used to estimate the
            parametric jacobian, in the corresponding parameter units.
            The default is None.
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
            list with parameter names. The default is None.
        name_states : list of str, optional
            list with state names. The default is None, which uses the
            measurement names for ``Experiment`` input.
        output_names : sequence of str, optional
            Unique names of the model callback output columns, in column
            order. Str ``Measurement.field`` values resolve through them, and
            the callback output must then have exactly this many columns.
            The default is None.

        Returns
        -------
        ParameterEstimation
            Estimator with one aligned input and callback entry per experiment.

        Raises
        ------
        ValueError
            Raised when:

            * legacy experiment mappings (``y_data``, ``args_fun``,
              ``kwargs_fun``) have keys other than those of ``x_data``, or
              several keys while ``x_data`` is unnamed;
            * no experiments are given, or ``x_data``, ``y_data``,
              ``args_fun`` or ``kwargs_fun`` hold different experiment
              counts;
            * ``y_data``, ``measured_ind``, ``args_fun`` or ``kwargs_fun``
              accompany ``Experiment`` input, or an experiment name is empty;
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

            * ``y_data`` is None for legacy input;
            * ``Experiment`` objects are mixed with other entries, an
              experiment name is not a str, or ``x_data`` holds bare
              ``Measurement`` objects not wrapped in an ``Experiment``;
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
            abs_tol = 1e-6 * np.ones_like(param_seed)
            rel_tol = 1e-6

            def dparam_fun(param):
                dp = dx_jac_p(param, abs_tol, rel_tol, eps)
                return dp

            dx_finitediff = dparam_fun

        self.dx_fd = dx_finitediff

        # --------------- Data
        self.experim_names = None
        self.output_names = _validate_output_names(output_names)

        experiments = (_experiment_collection(x_data)
                       if self._compiles_experiments else None)
        if experiments is not None:
            supplied = [label for label, value in (
                ('y_data', y_data), ('measured_ind', measured_ind),
                ('args_fun', args_fun), ('kwargs_fun', kwargs_fun))
                if value is not None]
            if supplied:
                raise ValueError(
                    f"{', '.join(supplied)} must be None when x_data holds "
                    "Experiment objects; observations, measured fields and "
                    "callback arguments live in each Experiment and its "
                    "Measurement objects")
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

        # Compiled layout; None for the analyze_data layout.
        self._measurement_names = (None if compiled is None
                                   else list(compiled.measurement_names))
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
            self.name_params = ['theta_{}'.format(ind + 1)
                                for ind in range(self.num_params)]

            self.name_params_total = ['theta_{}'.format(ind + 1)
                                      for ind in range(self.num_params_total)]

            self.name_params_plot = [r'$\theta_{}$'.format(ind + 1)
                                     for ind in range(self.num_params)]
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
            ``param_seed`` before numerical finite differences are evaluated.
            Units follow the parameter definitions supplied to the model.
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

        if not out_array and (
                self.params_residuals is None or
                not np.array_equal(params, self.params_residuals)):
            self.get_objective(params)

        if self.sens_second is None:
            raw_sens = []
            for ind in range(self.num_datasets):
                if self.jac_fun is None:
                    pass_to_fun = (self.x_model[ind], self.args_fun[ind],
                                   self.kwargs_fun[ind])

                    pick_p = np.where(self.map_variable)[0]
                    params_full = self.reconstruct_params(params)
                    sens = numerical_jac_data(self.func_aux, params_full,
                                              pass_to_fun, dx=self.dx_fd,
                                              pick_x=pick_p)
                else:
                    sens = self.jac_fun(params, self.x_model[ind],
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

    def optimize_fn(self, optim_options: Optional[dict] = None,
                    simulate: bool = False, verbose: bool = True,
                    store_iter: bool = True, method: str = 'LM',
                    bounds: Optional[Sequence] = None
                    ) -> tuple[np.ndarray, np.ndarray, dict]:
        """Optimize variable parameters and assemble fit statistics.

        Parameters
        ----------
        optim_options : dict, optional
            Solver options passed to the selected optimization method.
        simulate : bool, optional
            If True, evaluate residuals in simulation mode by disabling the
            optimization residual contribution.
        verbose : bool, optional
            If True, request verbose solver output when supported.
        store_iter : bool, optional
            If True, store unique parameter/objective iterates in
            ``params_iter``, ``objfun_iter`` and the ``paramest_df`` table,
            whose ``'obj_fun'`` column is the weighted sum of squares [-]
            (twice the objective) when measurements declare uncertainties,
            and otherwise the unweighted sum of squared raw residuals in
            squared measured-state units, also with ``weight_matrix`` (see
            ``get_objective``).
        method : {'LM', 'IPOPT'}, optional
            Optimization method used for fitting.
        bounds : sequence, optional
            Parameter bounds passed to IPOPT, in the model parameter units.

        Returns
        -------
        opt_par : numpy.ndarray
            Accepted variable parameters, shape ``(num_params,)``, in the
            model callback's parameter units.
        covar_params : numpy.ndarray
            Estimated covariance, shape ``(num_params, num_params)``. Entry
            ``(i, j)`` has the product of parameter i and parameter j units.
        info : dict
            Solver information at the accepted parameters. ``info['fun']`` is
            the weighted residual vector, ordered by experiment then
            measurement column then model-grid sample (see
            ``get_residual_layout``), in the units documented by
            ``assemble_solver_info``: measured-state units with the default
            identity weights and [-] with state-variance weights or declared
            measurement uncertainties. ``info['jac']`` has shape
            ``(num_params, len(info['fun']))``; each entry has its ``fun``
            entry's unit divided by the matching parameter unit.
            With staggered measurement grids or missing observations, columns
            include unobserved model-grid entries, so their count can exceed
            ``num_data_total``;
            those entries of ``info['fun']`` and columns of ``info['jac']``
            are exactly zero, so they add no residual or information.
            LM additionally supplies its accepted ``x`` and solver diagnostics.

        Raises
        ------
        ImportError
            If IPOPT is selected without the optional cyipopt dependency.
        numpy.linalg.LinAlgError
            If the accepted Jacobian yields a singular covariance matrix.

        Notes
        -----
        After LM finishes, the model is evaluated once per experiment at the
        accepted parameters to refresh ``y_runs``, ``resid_runs``,
        ``residuals``, and ``weighted_residuals`` before assembling
        ``y_model``. This includes termination after a rejected trial; it does
        not imply convergence. Predictions and raw residuals retain the model
        state units and measured-state order. Repeated calls replace the
        previous ``y_model`` list.

        The reporting evaluation does not change LM's returned ``x``, ``fun``,
        ``jac``, or solver counters. It uses the usual objective callback and
        its history recording; with ``store_iter=True`` duplicate parameter
        entries are removed as usual. Stateful callbacks must support another
        evaluation at the same parameters, as during optimization.

        """

        self.optimize_flag = not simulate
        self.opt_method = method

        params_var = self.param_seed[self.map_variable]  # [model parameter units]

        if method == 'LM':
            if optim_options is None:
                optim_options = {'full_output': True, 'verbose': verbose}
            else:
                optim_options['full_output'] = True
                optim_options['verbose'] = verbose

            opt_par, inv_hessian, info = levenberg_marquardt(
                params_var,
                self.get_objective,
                self.get_gradient,
                args=(True,),
                **optim_options)

            # Rejected trial callbacks overwrite model buffers. Refresh them
            # at the accepted point while retaining native LM result metadata.
            self.get_objective(opt_par, out_array=True)

        elif method == 'IPOPT':
            if not have_cyipopt:
                raise ImportError('cyipopt is an optional import. Please install cyipopt to use IPOPT as a solver for parameter estimation. conda install -c conda-forge cyipopt')
            if optim_options is None:
                optim_options = {'print_level': int(verbose) * 5}
            else:
                optim_options['print_level'] = int(verbose) * 5

            kwargs_fun = {'out_array': False}
            result = minimize_ipopt(self.get_objective, params_var,
                                    jac=self.get_gradient,
                                    bounds=bounds, options=optim_options,
                                    kwargs=kwargs_fun)

            opt_par = result['x']

            info = self.assemble_solver_info(opt_par)

        self.optim_options = optim_options

        # Store
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

        # Model prediction with final parameters
        self.y_model = []  # [model state units], one array per experiment
        for ind in range(self.num_datasets):
            y_data = self.y_data[ind]
            if isinstance(y_data, dict):
                y_data = np.hstack(list(y_data.values()))

            y_model = self.resid_runs[ind] + y_data  # [model state units]
            self.y_model.append(y_model)

        covar_params = self.get_covariance()

        return opt_par, covar_params, info

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

        Raises
        ------
        AttributeError
            If no fit has been run, so ``info_opt`` does not exist.
        numpy.linalg.LinAlgError
            If ``J @ J.T`` is singular, for example when a parameter does
            not affect any observed entry.

        Notes
        -----
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

        hessian_approx = np.dot(jac, jac.T)

        dof = self.num_data_total - self.num_params
        mse = 1 / dof * np.dot(resid.T, resid)

        if include_mse:
            covar = mse * np.linalg.inv(hessian_approx)

        else:
            covar = np.linalg.inv(hessian_approx)
        # Correlation matrix
        sigma = np.sqrt(covar.diagonal())
        d_matrix = np.diag(1/sigma)
        correlation = d_matrix.dot(covar).dot(d_matrix)

        self.covar_params = covar
        self.correl_params = correlation

        return covar

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
    """

    # Spectral residuals are not model outputs minus observations, so data
    # keep the legacy analyze_data layout (see __init__ Notes).
    _compiles_experiments = False

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
            Weight of the penalty ``mult_penalty * sum(min(eps, 0)**2)`` on
            negative pure-component absorptivities ``eps``, added to the
            objective; units of the objective per squared absorptivity unit
            (absorbance per concentration unit). The default 1 is a scaling
            assumption, not a calibrated value.
        global_analysis : bool, optional
            Resolve one set of absorptivities for all experiments.
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
            If a spectra dictionary lacks ``'spectra'`` or ``weight_matrix``
            does not match the residual columns, besides the
            ``ParameterEstimation`` input errors.
        TypeError
            If ``time_data`` holds ``Experiment`` or ``Measurement``
            objects.

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

        self.projection_kwargs = {}
        self.mult_penalty = mult_penalty

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

    def get_sens_projection(self, c_target, c_plus, sens_states):
        eye = np.eye(c_target.shape[0])
        proj_orthogonal = eye - np.dot(c_target, c_plus)

        if sens_states.shape[0] == sum(self.map_variable):
            sens_pick = sens_states
        else:
            sens_pick = sens_states[self.map_variable]

        first_term = proj_orthogonal @ sens_pick @ c_plus
        second_term = first_term.transpose((0, 2, 1))
        sens_an = (first_term + second_term) @ self.spectra_tot

        return sens_an

    def func_aux(self, params, x_vals, spectra, *args):
        states = self.function(params, x_vals, *args)

        _, epsilon, absorbance = mcr_spectra(
            states[:, self.spectral_ind], spectra)

        return absorbance.T.ravel()

    def get_gradient(self, params, out_array=False):
        """Return the Jacobian of the weighted MCR residuals or a gradient.

        Parameters
        ----------
        params : array_like
            Optimized parameters in the model's units.
        out_array : bool, optional
            If True, return the weighted-residual Jacobian for LM;
            otherwise the scalar-objective gradient. The default is False.

        Returns
        -------
        numpy.ndarray
            With ``out_array``, shape ``(n_params, n_data)``: derivatives of
            the weighted residuals of ``get_objective`` (column-major over
            residual columns and stacked times) per parameter unit, from
            finite differences or from model-returned sensitivities.
            Unobserved non-spectral entries are zero. Otherwise the
            objective gradient from finite differences, shape
            ``(n_params,)``.
        """
        raw_sens = []
        if self.sens_second is None:
            pick_p = np.where(self.map_variable)[0]

            if out_array:
                args = (True, False)  # out_array, update_self
                jac_fun = numerical_jac_data
            else:
                args = (False, False)
                jac_fun = numerical_jac

            weighted_sens = jac_fun(self.get_objective, params, args=args,
                                    dx=self.dx_fd, pick_x=pick_p)

            if out_array:
                # Finite differences of the weighted residual are already
                # its Jacobian, (n_data, n_params); LM needs no sign change.
                return weighted_sens.T

        else:
            raw_sens = self.sens_second
            sens_tot = np.concatenate(raw_sens, axis=1)  # (n_par x n_times x n_states)

            sens_mcr = sens_tot[:, :, self.measured_ind['spectra']]

            # Variable projection derivative: n_par x n_times x n_lambda
            sens_spectra = self.get_sens_projection(sens_states=sens_mcr,
                                                    **self.projection_kwargs)

            n_par, n_times, n_lambda = sens_spectra.shape
            x_mask = self._residual_observation_mask()
            if self.has_non:
                sens_regular = sens_tot[:, :, self.measured_ind['non_spectra']]

                # weighted_sens holds minus the residual derivative (it is
                # negated below): the spectral residual is data minus
                # prediction, the non-spectral one model minus data.
                all_sens = np.concatenate((sens_spectra, -sens_regular),
                                          axis=2)

                # (n_par, n_times, n_columns); rows whitened like the
                # residuals, unobserved entries zero.
                weighted_all = self._weight_sample_rows(
                    all_sens.transpose(1, 0, 2), x_mask).transpose(1, 0, 2)

                weighted_sp = flatten_spectral_sens(weighted_all[:, :, :n_lambda])
                weighted_reg = flatten_spectral_sens(weighted_all[:, :, n_lambda:])

                weighted_sens = np.vstack((weighted_sp, weighted_reg))
            else:
                sens = sens_spectra

                weighted_sens = self._weight_sample_rows(
                    sens.transpose(1, 0, 2), x_mask).transpose(1, 0, 2)
                weighted_sens = flatten_spectral_sens(weighted_sens)

        if out_array:
            return -weighted_sens.T
        else:
            return weighted_sens[0]

    def get_global_analysis(self, params):
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
            Least-squares absorptivities, ``(n_spectral_states, n_lambda)``.

        Raises
        ------
        ValueError
            If a callback output is not two-dimensional with one row per
            model-grid sample of its experiment.
        """
        c_runs = []
        states_non = []
        sens_states = []

        y_data_non = []
        for ind in range(self.num_datasets):
            result = self.function(params, self.x_model[ind],
                                   reord_sens=False,
                                   *self.args_fun[ind], **self.kwargs_fun[ind])

            if isinstance(result, tuple):
                states, sens_st = result
                sens_states.append(sens_st)
            else:
                states = result

            self._check_model_output(ind, states, two_dimensional=True)
            conc_target = states[:, self.measured_ind['spectra']]

            if self.has_non:
                non_spectra = states[:, self.measured_ind['non_spectra']]

                y_non = self.y_data[ind]['non_spectra'].copy()
                x_mask = self.x_masks[ind]['non_spectra']
                if x_mask is not None:
                    y_non[~x_mask] = non_spectra[~x_mask]

                states_non.append(non_spectra)
                y_data_non.append(y_non)

            c_runs.append(conc_target)

        conc_tot = np.vstack(c_runs)

        if self.has_non:
            states_non = np.vstack(states_non)
            y_data_non = np.vstack(y_data_non)

            resid_non = states_non - y_data_non

        else:
            resid_non = []

        # MCR
        conc_plus = np.linalg.pinv(conc_tot)
        absorptivity_pure = np.dot(conc_plus, self.spectra_tot)

        spectra_pred = np.dot(conc_tot, absorptivity_pure)

        self.projection_kwargs = {'c_target': conc_tot, 'c_plus': conc_plus}

        if len(sens_states) > 0:
            self.sens_second = sens_states

        residuals = self.spectra_tot - spectra_pred
        if self.has_non:
            residuals = np.hstack((residuals, resid_non))

        # Row-wise whitening: sigma_inv for fully observed rows, the
        # marginal precision root for partially observed rows, zero for
        # unobserved entries (ParameterEstimation._weight_sample_rows).
        weighted_resid = self._weight_sample_rows(
            residuals, self._residual_observation_mask())

        trim_y = np.cumsum(self.len_spectra)[:-1]

        y_runs = np.split(spectra_pred, trim_y, axis=0)
        resid_runs = np.split(residuals, trim_y, axis=0)

        weighted_resid = weighted_resid.T.ravel()

        return y_runs, resid_runs, weighted_resid, absorptivity_pure

    def get_local_analysis(self, params):
        y_runs = []
        resid_runs = []
        sens_runs = []
        c_runs = []
        epsilon_mcr = []

        weighted_resid = []

        for ind in range(self.num_datasets):
            result = self.function(params, self.x_fit[ind],
                                   reorder=False,
                                   *self.args_fun[ind], **self.kwargs_fun[ind])

            if isinstance(result, tuple):
                conc_prof, sens_states = result
            else:
                conc_prof = result
                sens_states = None

            conc_target = conc_prof[:, self.spectral_ind]

            # MCR
            conc_plus = np.linalg.pinv(conc_target)
            absorptivity_pure = np.dot(conc_plus, self.spectra[ind])
            spectra_pred = np.dot(conc_target, absorptivity_pure)

            epsilon_mcr.append(absorptivity_pure)

            if sens_states is None:
                args_merged = [self.x_data[ind],
                               self.args_fun[ind], self.spectra[ind]]

                sens = numerical_jac_data(self.func_aux, params, args_merged,
                                          dx=self.dx_fd)[:, self.map_variable]
            else:
                sens = self.get_sens_projection(conc_target, conc_plus,
                                                sens_states, spectra_pred)

            resid_run = spectra_pred - self.spectra[ind]
            weighted_resid = np.dot(resid_run, self.sigma_inv)

            y_runs.append(spectra_pred.T.ravel())
            resid_runs.append(resid_run)
            sens_runs.append(sens)

        self.resid_runs = resid_runs
        self.y_runs = y_runs
        self.sens_runs = sens_runs

        self.epsilon_mcr = epsilon_mcr

        weighted_resid = [elem.T.ravel() for elem in weighted_resid]
        weighted_resid = np.concatenate(weighted_resid)

        return y_runs, weighted_resid, absorptivity_pure  # TODO: I didn't work on this method

    def get_objective(self, params, out_array=False, update_self=True):

        # if type(self.params_iter) is list:
        #     self.params_iter.append(params)

        # Reconstruct parameter set with fixed and non-fixed indexes
        params = self.reconstruct_params(params)

        if self.global_analysis:
            out = self.get_global_analysis(params)

        else:
            out = self.get_local_analysis(params)

        y_runs, resid, weighted_resid, molar_abs = out

        if update_self:
            if type(self.objfun_iter) is list:
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
            return weighted_resid
        else:
            residual = 1/2 * np.dot(weighted_resid, weighted_resid)
            penalty = self.mult_penalty*(np.maximum(-molar_abs, 0)**2).sum()
            return residual + penalty
