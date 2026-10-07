#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 15:35:48 2019

@author: casas100
"""

import numpy as np
from scipy.linalg import cholesky, solve_triangular
from itertools import cycle
from typing import Callable, Optional, Sequence

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


class Experiment:
    def __init__(self):
        pass


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

    Experiment identity follows ``x_data`` keys when provided; state columns,
    physical bases and callback units remain those declared by the model.
    """

    def __init__(self, func: Callable, param_seed, x_data, y_data=None,
                 measured_ind=None,
                 args_fun=None, kwargs_fun=None,
                 optimize_flags=None,
                 jac_fun=None, dx_finitediff=None,
                 weight_matrix=None,
                 name_params=None, name_states=None) -> None:
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
        x_data : numpy array, list of arrays or dict
            array with experimental values for the independent variable x. If
            several datasets Ne are passed, either a list of arrays
                x_data = [x_1, ..., x_i, ..., x_Ne]
            or a dictionary of arrays:
                x_data = {'name_exp_1': x_1, ..., 'name_exp_i': x_i, ...,
                          'name_exp_N': x_Ne}
            can be specified. Units follow the model independent variable
            (typically time [s]). Dictionary insertion order declares experiment
            order; other experiment mappings are aligned by those keys.
        y_data : numpy array, list of arrays or dict
            Experimental values for the dependent variable(s) y, in the model's
            state units and physical bases.
            Array y is of dimension len(x_i) x N_meas, where N_meas is less
            than or equal to the number of states returned by func (Ny).
            It supports same data structures as ``x_data``. If ``y_data`` is a
            dictionary, its keys must match those of ``x_data``. A dictionary
            with more than one experiment requires named ``x_data``; with
            unnamed ``x_data``, pass a list in ``x_data`` order instead.
            Observations are required; ``None`` raises ``TypeError``.
        measured_ind : list of int, optional
            Indexes of the states returned by func that are measured and
            passed in each dataset contained in 'y_data'.
            If None, it is assumed that all the states are measured.
            The default is None.
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
            The default is None, which uses the dimensionless identity.
        name_params : list of str, optional
            list with parameter names. The default is None.
        name_states : list of str, optional
            list with state names. The default is None.

        Returns
        -------
        ParameterEstimation
            Estimator with one aligned input and callback entry per experiment.

        Raises
        ------
        ValueError
            If experiment mappings disagree, experiment counts differ, no
            experiments are supplied, an observation/callback mapping
            has multiple experiment keys while ``x_data`` is unnamed, or
            ``weight_matrix`` is not a finite square two-dimensional array.
        TypeError
            If ``y_data`` is None, positional arguments are not iterable, or
            keyword arguments are not a dictionary. Positional errors identify
            experiment keys or, for unnamed experiments, zero-based positions.
        numpy.linalg.LinAlgError
            If ``weight_matrix`` is not positive definite.

        Notes
        -----
        Lists and arrays retain positional ordering. Nested state observation
        dictionaries are passed through without treating state names as
        experiment names. Observation and callback mappings with multiple
        experiment keys cannot be aligned with unnamed ``x_data`` and are
        rejected; use lists in ``x_data`` order instead. A single
        experiment's direct keyword dictionary may still contain multiple
        callback keywords.
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

        if isinstance(x_data, dict):
            self.experim_names = list(x_data.keys())

        if y_data is None:
            raise TypeError(
                "y_data is required; pass observations for each experiment "
                "in x_data")

        if isinstance(y_data, dict) and self.experim_names is not None:
            y_data = _ordered_experiment_values(
                y_data, self.experim_names, 'y_data')
        elif isinstance(y_data, dict) and len(y_data) > 1:
            raise ValueError(
                f"y_data experiment keys {list(y_data)!r} cannot be aligned "
                "with unnamed x_data; pass x_data as a dictionary with the "
                "same keys, or y_data as a list in x_data order")

        x_data = convert_types(x_data)
        y_data = convert_types(y_data, two_d=True)
        if not x_data or len(x_data) != len(y_data):
            raise ValueError(
                "x_data and y_data must contain the same nonzero number of "
                f"experiments; got {len(x_data)} and {len(y_data)}")

        args_fun = _experiment_arguments(
            args_fun, len(x_data), self.experim_names, 'args_fun', keyword=False)
        kwargs_fun = _experiment_arguments(
            kwargs_fun, len(x_data), self.experim_names, 'kwargs_fun',
            keyword=True)

        x_model, x_masks, y_data = analyze_data(x_data, y_data)

        self.x_model = x_model
        self.x_masks = x_masks

        self.x_data = x_data
        self.y_data = y_data
        self.num_datasets = len(self.y_data)

        if self.experim_names is None:
            self.experim_names = ['exp_%i' % (ind + 1)
                                  for ind in range(self.num_datasets)]

        if measured_ind is None:
            measured_ind = list(range(y_data[0].shape[1]))

        self.measured_ind = measured_ind
        self.num_model_states = None
        self.sens_second = None  # sensitivities returned along with obj fun

        # ---------- Arguments
        self.args_fun = args_fun
        self.kwargs_fun = kwargs_fun

        num_data = []
        for ind in range(self.num_datasets):
            if self.x_masks[ind] is None:
                num_data.append(self.y_data[ind].size)
            else:
                if isinstance(self.x_masks[ind], dict):
                    num = [sum(mask) for mask in self.x_masks[ind].values()]
                    num_data.append(sum(num))
                else:
                    num_data.append(self.x_masks[ind].sum())

        self.num_data_total = sum(num_data)
        self.num_data = num_data

        if weight_matrix is None:
            weight_matrix = np.eye(len(self.measured_ind))  # [-]

        # Lower triangular; row i in [1/u_i] for measured state unit u_i,
        # columns [-]. sigma_inv @ sigma_inv.T equals inv(weight_matrix) in
        # measured-state order.
        self.sigma_inv = _measurement_precision_root(weight_matrix)
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
        if name_states is None:
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
                            x_mask: Optional[np.ndarray]) -> np.ndarray:
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

        Returns
        -------
        numpy.ndarray
            Weighted values with the shape of ``values``. For each sample row
            with observed states ``o``, entries ``o`` are
            ``values[k, ..., o] @ S_o``, where ``S_o`` is the lower Cholesky
            factor of ``inv(weight_matrix[o][:, o])``; unobserved entries are
            exactly zero. Units are those of ``values`` divided by the
            measured state's unit (dimensionless for residuals weighted by a
            covariance in squared state units).

        Notes
        -----
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

    def select_sens(self, sens_ordered, num_states, times=None):

        parts = np.split(sens_ordered, num_states, axis=0)

        if times is None:
            selected = [parts[ind] for ind in self.measured_ind]
        else:
            selected = [parts[ind][times[count]]
                        for count, ind in enumerate(self.measured_ind)]

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

        Notes
        -----
        With staggered observation grids, unobserved model-grid entries keep
        their positions in ``r`` but are exactly zero, and each partially
        observed sample row is weighted by the marginal precision of its
        observed states (see ``_weight_sample_rows``), so the objective is
        ``1/2 * sum_k r_ok @ inv(weight_matrix[o_k][:, o_k]) @ r_ok`` over the
        observed residuals ``r_ok`` of each sample row.

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

            if y_prof.ndim == 1:
                y_run = y_prof.reshape(-1, 1)
                self.num_model_states = 1

            else:
                y_run = y_prof[:, self.measured_ind]
                self.num_model_states = y_prof.shape[1]

            # Replace missing data with model values so as residuals are zero
            # for those entries
            y_data = self.y_data[ind].copy()
            x_mask = self.x_masks[ind]
            if x_mask is not None:
                y_data[~x_mask] = y_run[~x_mask]

            resid_run = y_run - y_data

            # Store
            y_runs.append(y_run)
            resid_runs.append(resid_run)

        # [measured-state unit / weight_matrix unit**0.5]; [-] only with
        # state-variance weights. Unobserved entries are exactly zero.
        weighted_residuals = [self._weight_sample_rows(resid, x_mask)
                              for resid, x_mask in zip(resid_runs,
                                                       self.x_masks)]

        if len(sens_second) > 0:
            self.sens_second = sens_second

        if type(self.objfun_iter) is list:
            objfun_val = np.linalg.norm(np.concatenate(resid_runs))**2
            self.objfun_iter.append(objfun_val)

        residuals = self.optimize_flag * np.concatenate(resid_runs)

        # Weighted-residual units as above, flattened state-major.
        residual_out = np.concatenate([ar.T.ravel()
                                       for ar in weighted_residuals])

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
        residuals. Columns of unobserved model-grid entries are therefore
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
        for sensit, x_model, x_mask in zip(raw_sens, self.x_model,
                                           self.x_masks):
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
            weighted = self._weight_sample_rows(sens_by_row, x_mask)
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
            If True, store unique parameter/objective iterates.
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
            the weighted residual vector, ordered by experiment then state
            then sample, in the units documented by ``assemble_solver_info``:
            measured-state units with the default identity weights and [-]
            with state-variance weights. ``info['jac']`` has shape
            ``(num_params, len(info['fun']))``; each entry has its ``fun``
            entry's unit divided by the matching parameter unit.
            With staggered measurement grids, columns include unobserved
            model-grid entries, so their count can exceed ``num_data_total``;
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

    def get_covariance(self, include_mse=True):
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
            Weight of the penalty on negative pure-component absorptivities.
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
            state, shape ``(n_lambda + n_non, n_lambda + n_non)``. The
            default is the identity.

        Raises
        ------
        ValueError
            If a spectra dictionary lacks ``'spectra'`` or ``weight_matrix``
            does not match the residual columns, besides the
            ``ParameterEstimation`` input errors.

        Notes
        -----
        Spectral residuals are data minus prediction and non-spectral ones
        model minus data, so the residual covariance is
        ``D @ weight_matrix @ D`` with ``D = diag(+1 spectral, -1
        non-spectral``); ``sigma_inv`` and the marginal roots of partially
        observed rows are built for it. Block-diagonal weights are
        unaffected.
        """

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
