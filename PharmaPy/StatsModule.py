#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 15:10:19 2019

@author: dcasasor
"""

import copy

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle
import pandas as pd
from itertools import combinations
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
        """

        # Values from the instance
        inst = estimation_instance
        jac = inst.info_opt['jac']

        self.a_matrix = jac.dot(jac.T)
        self.a_inv = np.linalg.inv(self.a_matrix)
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

        if covar_params is None:
            covar_params = self.inst.covar_params

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
        """
        Generate pairwise confidence ellipses. For three or more parameters,
        the 2D projections are calculated as described in 'Nonlinear
        Regression' by Seber and Wilde (1989).

        Parameters
        ----------
        alphas : tuple or list, optional
            confidence levels to draw ellipses. If None, the alpha value
            specified when creating the instance is used.
            The default is None.

        Returns
        -------
        info_dict : dict of dicts
            dictionary with keys as the paramter pair names, e.g.
            'theta_1/theta_2'. Each dictionary value is in turn a dictionary
            with the ellipse and rectangle patches and their respective
            dimensions (axes lenghts or side lengths)

        """

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
            correlation scale (see ``_covariance_factor``).
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
            when the optimizer encounters a singular linear system.

        Warns
        -----
        RuntimeWarning
            If ``optimize_fn`` raises ``numpy.linalg.LinAlgError``; the warning
            includes the sample index and original diagnostic.
            Also if a sample's exception propagates and the model then fails
            at the accepted parameters; the sample's exception is still
            raised, also when warnings are turned into errors (the report is
            then dropped).
            Also, from ``optimize_fn``, if an IPOPT sample refit terminates
            unsuccessfully; see Notes.

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
        ``numpy.linalg.solve`` and ``numpy.linalg.inv``. Their documented
        Exception type, rather than origin, defines recovery. Any
        ``LinAlgError`` escaping ``optimize_fn`` is recorded as a NaN row,
        including one raised inside a user model. Other programming and model
        errors propagate unchanged.

        This recovery is therefore scoped to ``opt_method='LM'``.
        ``ParameterEstimation.optimize_fn`` also accepts ``'IPOPT'``, whose
        numerical failures surface from cyipopt rather than as a
        ``LinAlgError``. Under that method a failed sample aborts the whole
        bootstrap run instead of recording a NaN row. An unsuccessful IPOPT
        termination of a sample refit, for example at ``max_iter``, does not
        raise: ``optimize_fn`` emits its ``RuntimeWarning`` naming the Ipopt
        status, and the sample's last iterate is stored as its row. When
        that warning is turned into an error, it propagates like any other
        sample exception and aborts the run.
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
