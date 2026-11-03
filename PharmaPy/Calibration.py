#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 11:29:05 2020

@author: dcasasor
"""

import numpy as np
import itertools
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator


class PCR_calibration:
    """Principal-component regression (PCR) calibration of spectra.

    Predictor rows are spectra (one row per sample, one column per
    wavelength) in absorbance or other spectral-intensity units; responses
    are the calibrated quantities, for example concentrations [g/L].

    The predictor preprocessing is fixed by the calibration data:

    1. Optional standard normal variate (SNV, ``snv=True``): each spectrum
       is centered by its own mean across wavelengths and divided by its own
       sample standard deviation across wavelengths (denominator
       ``n_wavelengths - 1``, ``ddof=1``), as defined by Barnes, Dhanoa and
       Lister, Appl. Spectrosc. 43(5), 772-777 (1989). The transform of a
       spectrum does not depend on the other spectra in the batch, and
       removes additive offsets and multiplicative scaling of a spectrum.
       Spectra are converted to float64 first.
    2. Column centering with the training column means, and, when
       ``standardize=True``, column scaling with the training column
       population standard deviations, both computed from the (SNV-
       transformed, if requested) calibration spectra.

    ``predict`` applies the same steps with the stored training column
    statistics, so a single new spectrum is projected on the fitted
    principal-component basis.

    Warnings
    --------
    Before #138, ``snv=True`` did not apply SNV: it column-centered and
    column-scaled the calibration spectra, and ``predict`` recomputed those
    column statistics from the new batch (``NaN`` for a single spectrum).
    Calibrations and predictions with ``snv=True`` therefore differ from
    earlier releases. ``snv=False`` is unchanged.
    """

    def __init__(self, data, num_comp=None, standardize=True, snv=False,
                 y_name=None, y_suffixes=None):
        """Build the principal-component basis of the calibration spectra.

        Parameters
        ----------
        data : numpy.ndarray
            Calibration spectra, shape ``(n_samples, n_wavelengths)``, in
            spectral-intensity units such as absorbance [AU].
        num_comp : int, optional
            Number of principal components retained for regression [-].
            Default is all singular values.
        standardize : bool, optional
            If True, scale each preprocessed column by its training
            population standard deviation. Default is True.
        snv : bool, optional
            If True, apply the per-spectrum standard normal variate before
            column centering. Default is False.
        y_name : str, optional
            Response-label prefix used in plots. Default is ``'y_'``.
        y_suffixes : list of str, optional
            Response-label suffixes used in plots.

        Raises
        ------
        ValueError
            If ``snv=True`` and a calibration spectrum has zero standard
            deviation across wavelengths or fewer than two wavelengths; or
            if ``snv=True`` and ``standardize=True`` and an SNV column is
            constant across the calibration spectra, so that column scaling
            is undefined.

        Notes
        -----
        ``data_mean`` and ``data_std`` store the training column statistics
        (population, ``ddof=0``) of the spectra entering the column
        centering: raw spectra [AU] when ``snv=False`` and SNV-transformed
        spectra [-] when ``snv=True``.

        SNV can make a column constant even when the raw column varies, for
        example when one wavelength equals the row mean of every spectrum.
        Such a column carries no information after SNV; with
        ``standardize=True`` it is rejected rather than divided by zero.
        Use ``standardize=False`` for these spectra. The non-SNV path does
        not check its columns (#517).
        """
        self.data = data
        self.standardize = standardize
        self.snv = snv

        if snv:
            # Per-spectrum normalized spectra [-] and the bound on the SNV
            # round-off of each spectrum's entries [-].
            data_preprocessed, snv_roundoff = self.__snv_transform(data)
        else:
            # Raw spectra [AU].
            data_preprocessed = data

        # Training column statistics of the preprocessed spectra; [AU]
        # without SNV, [-] with SNV.
        data_mean = data_preprocessed.mean(axis=0)
        data_std = data_preprocessed.std(axis=0)

        if snv and standardize:
            # A column that is constant in exact arithmetic is computed as
            # that constant plus each row's SNV round-off, bounded per row by
            # snv_roundoff (see __snv_transform), so its spread, and hence
            # its std, is at most the largest of those bounds. Computing the
            # column std adds the round-off of the column mean, bounded like
            # the row check in __snv_transform by
            # n_spectra * eps * max|column|. A column std within the sum of
            # both bounds carries no information beyond round-off [-].
            num_spectra = data_preprocessed.shape[0]
            roundoff_col_std = (snv_roundoff.max()
                                + num_spectra * np.finfo(np.float64).eps
                                * np.abs(data_preprocessed).max(axis=0))
            constant_cols = np.flatnonzero(data_std <= roundoff_col_std)
            if constant_cols.size > 0:
                raise ValueError(
                    "SNV made calibration column(s) "
                    f"{constant_cols.tolist()} constant across the spectra, "
                    "so standardize=True would divide by a zero standard "
                    "deviation; use standardize=False with snv=True")

        self.data_centered = self.__center_data(data_preprocessed, data_mean,
                                                data_std)

        self.data_mean = data_mean
        self.data_std = data_std

        (self.projections, self.explained_variance,
         self.svd_dict) = self.__get_projections()

        if num_comp is None:
            self.num_comp = len(self.svd_dict['sv'])
        else:
            self.num_comp = num_comp

        if y_name is None:
            y_name = 'y_'

        self.y_name = y_name
        self.y_suffixes = y_suffixes

    def __snv_transform(self, data):
        """Apply the standard normal variate (SNV) to each spectrum.

        Parameters
        ----------
        data : numpy.ndarray
            Spectra, shape ``(n_samples, n_wavelengths)``, in
            spectral-intensity units such as absorbance [AU].

        Returns
        -------
        data_snv : numpy.ndarray
            float64 spectra with zero mean and unit sample standard
            deviation across wavelengths, shape
            ``(n_samples, n_wavelengths)`` [-].
        snv_roundoff : numpy.ndarray
            Bound on the floating-point round-off of each spectrum's SNV
            values, ``(n_wavelengths + 4) * eps * max|x_i| / s_i``, shape
            ``(n_samples,)`` [-].

        Raises
        ------
        ValueError
            If the spectra have fewer than two wavelengths, or a spectrum
            has zero standard deviation across wavelengths (a constant
            spectrum), for which SNV is undefined. A standard deviation
            within the floating-point round-off of the row mean counts as
            zero.

        Notes
        -----
        ``x_snv[i, j] = (x[i, j] - mean_j(x[i])) / s_j(x[i])`` with the
        sample standard deviation ``s`` (denominator ``n_wavelengths - 1``,
        ``ddof=1``), following Barnes, Dhanoa and Lister, Appl. Spectrosc.
        43(5), 772-777 (1989). Each row is transformed independently of the
        other rows. Spectra are converted to float64 first, so the
        round-off bound below, which uses the float64 machine epsilon,
        applies to every input precision.
        """
        # [AU], float64 so the float64 round-off bound below applies.
        data = np.atleast_2d(np.asarray(data, dtype=np.float64))
        num_wavelengths = data.shape[1]
        if num_wavelengths < 2:
            raise ValueError(
                "SNV needs at least two wavelengths per spectrum; got "
                f"{num_wavelengths}")

        row_mean = data.mean(axis=1, keepdims=True)  # [AU]
        row_std = data.std(axis=1, ddof=1, keepdims=True)  # [AU]

        # A constant spectrum can return a tiny nonzero std made only of
        # round-off in the row mean. Higham, Accuracy and Stability of
        # Numerical Algorithms, 2nd ed. (SIAM, 2002), eq. (4.4), bounds the
        # recursive-summation error of n terms by gamma_(n-1) * sum|x_i|;
        # divided by n, the computed mean of a constant row errs by at most
        # (n - 1) * u * max|x| with u = eps / 2. Every deviation of a
        # constant row equals that error, so its sample std is at most
        # sqrt(n / (n - 1)) * (n - 1) * u * max|x| < n * u * max|x|, and
        # n * eps * max|x| bounds it with a margin of at least 2.
        roundoff_std = (num_wavelengths * np.finfo(np.float64).eps
                        * np.abs(data).max(axis=1, keepdims=True))  # [AU]
        constant_rows = np.flatnonzero(row_std <= roundoff_std)
        if constant_rows.size > 0:
            raise ValueError(
                "SNV is undefined for spectra with zero standard deviation "
                "across wavelengths; constant spectrum row(s) "
                f"{constant_rows.tolist()}")

        data_snv = (data - row_mean) / row_std  # [-]

        # Round-off of z_ij = (x_ij - m_i) / s_i [-]. The computed mean errs
        # by at most (n - 1) * u * max|x_i| (above), a shift of every
        # deviation that leaves s_i unchanged to first order because the
        # deviations sum to zero. The subtraction adds a relative error u
        # to each deviation; the two-pass std about (n + 4) * u / 2 (3 u
        # from the rounded deviations and squares, (n - 1) u from the sum,
        # u from the division, halved by the root, plus u / 2 for the root
        # itself); the division u. With |z_ij| <= 2 * max|x_i| / s_i, to
        # first order in u, |dz_ij| <= (n - 1) u max|x_i| / s_i
        # + 2 max|x_i| / s_i * (2 u + (n + 4) u / 2)
        # = (2 n + 7) u max|x_i| / s_i < (n + 4) * eps * max|x_i| / s_i.
        snv_roundoff = ((num_wavelengths + 4) * np.finfo(np.float64).eps
                        * np.abs(data).max(axis=1)
                        / row_std[:, 0])  # [-]

        return data_snv, snv_roundoff

    def __center_data(self, data, mean, std):
        """Center and optionally standardize predictor data.

        Parameters
        ----------
        data : numpy.ndarray
            Predictor rows, shape ``(n_samples, n_wavelengths)``, after any
            SNV transform: [AU] without SNV, [-] with SNV.
        mean : numpy.ndarray
            Training column means, shape ``(n_wavelengths,)``, in the units
            of ``data``.
        std : numpy.ndarray
            Training column population standard deviations, shape
            ``(n_wavelengths,)``, in the units of ``data``. Used only when
            ``standardize`` is True.

        Returns
        -------
        data_centered : numpy.ndarray
            Centered predictors, shape of ``data``. Values are
            dimensionless [-] when ``standardize`` is True, otherwise in the
            units of ``data``.

        """
        data_centered = data - mean

        if self.standardize:
            data_centered *= 1 / std

        return data_centered

    def __get_projections(self, data=None, n_comp=None):

        if data is None:
            data = self.data_centered

        # Perform SVD
        u_m, sv, v_nt = np.linalg.svd(data)
        v_n = v_nt.T

        # Percent of explained variance
        explained_var = sv**2 / (sv**2).sum() * 100

        # Store SVD in a dict
        v_trunc = v_nt[:len(sv)].T

        svd_dict = {'U': u_m, 'sv': sv, 'V': v_n, 'V_trunc': v_trunc}

        # Projections
        projections = np.dot(data, v_n)

        if n_comp is not None:
            projections = projections[:, :n_comp]

        return projections, explained_var, svd_dict

    def plot_projections(self, fig_size=None, num_comp=None):

        if num_comp is None:
            num_comp = self.num_comp

        comb = itertools.combinations(range(num_comp), 2)

        combs = [item for item in comb]
        num_plots = len(combs)

        if num_plots == 1:
            fig, axes = plt.subplots(figsize=fig_size)
            axes = np.atleast_1d(axes)
        else:
            ncols = 2
            nrows = num_plots // ncols + num_plots % ncols

            fig, axes = plt.subplots(nrows, ncols, figsize=fig_size)

        axes_flat = axes.flatten()
        num_axes = len(axes_flat)
        for ind in range(num_plots):
            axis = axes_flat[ind]
            pc_one, pc_two = combs[ind]

            my_map = plt.get_cmap('Reds')
            data_plot = self.projections[:, combs[ind]].T
            axis.scatter(data_plot[0], data_plot[1],
                         # 'o', mfc='None',
                         s=30/(num_axes/2), c=range(data_plot.shape[1]),
                         cmap=my_map,
                         marker='o', edgecolor='k')

            axis.text(0.5, -0.08, 'PC%i' % (pc_one + 1),
                      transform=axis.transAxes, ha='center')

            axis.text(-0.08, 0.5, 'PC%i' % (pc_two + 1), rotation=90,
                      transform=axis.transAxes, va='center')

            axis.spines['bottom'].set_position('zero')
            axis.spines['left'].set_position('zero')

            axis.spines['top'].set_visible(False)
            axis.spines['right'].set_visible(False)

        if num_plots < num_axes:
            fig.delaxes(axes_flat[-1])

        fig.tight_layout()

        return fig, axes

    def get_regression(self, y_data, num_comp=None, update_instance=True):

        if self.y_suffixes is None:
            self.y_suffixes = ['%i' % num for num in range(1, len(y_data))]

        self.y_labels = [r'$' + self.y_name + ('{%s}' % suffix) + '$'
                         for suffix in self.y_suffixes]

        if num_comp is None:
            num_comp = self.num_comp

        # scores = self.projections[:, :num_comp]
        scores = self.projections[:, :num_comp]

        scores_inv = np.linalg.pinv(scores)  # pseudoinverse of scores

        # Regression coefficients w.r.t. principal components
        if y_data.ndim == 1:
            y_data = y_data[..., np.newaxis]

        y_means = y_data.mean(axis=0)
        y_center = y_data - y_means
        q_coeff = np.dot(scores_inv, y_center)

        regression_coeff = q_coeff

        self.y_data = y_data
        self.y_center = y_center
        self.y_means = y_means

        if update_instance:
            self.regression_coeff = regression_coeff
            self.num_comp = num_comp

            y_pred = self.predict(self.data)
            residuals = y_data - y_pred
            self.residuals = residuals

        # # Regression coefficients w.r.t. original X
        # regression_coeff = np.dot(self.svd_dict['V'][:, :num_comp],
        #                           q_coeff)


        # if update_instance:
        #     self.regression_coeff = regression_coeff

        #     y_pred = self.predict(self.data)
        #     residuals = y_data - y_pred
        #     self.residuals = residuals

        return regression_coeff

    def predict(self, inputs, num_comp=None, regression_coeff=None,
                full_output=False, y_observed=None):
        """Predict responses for new predictor rows.

        Parameters
        ----------
        inputs : array-like
            New predictor spectra, shape ``(n_rows, n_wavelengths)`` or
            ``(n_wavelengths,)`` for one spectrum, in the same units as the
            calibration data (e.g. absorbance [AU]).
        num_comp : int, optional
            Number of principal components used for prediction [-].
        regression_coeff : numpy.ndarray, optional
            Regression coefficients with respect to principal-component
            scores, shape ``(num_comp, n_responses)`` [response units per
            score unit]: [response units] when the scores are
            dimensionless (``standardize=True``, or ``snv=True``), and
            [response units/AU] for ``snv=False, standardize=False``.
        full_output : bool, optional
            If True, return projections, predictions, and the MSE
            diagnostic.
        y_observed : array-like, optional
            Observed responses of the rows of ``inputs``, in the same order
            and in the units of the calibration responses (e.g. [g/L]),
            shape ``(n_rows, n_responses)`` or ``(n_rows,)`` for a single
            response. Only used with ``full_output=True``, to compute
            ``'MSE'``.

        Returns
        -------
        response : numpy.ndarray or dict
            Predicted responses, shape ``(n_rows, n_responses)``, in the
            units of the responses used to fit ``regression_coeff``, when
            ``full_output`` is False. If ``full_output`` is True, a
            dictionary with ``'x_projected'`` (scores, shape
            ``(n_rows, num_comp)``), ``'y_pred'`` (the predicted responses)
            and ``'MSE'``: ``resid.T @ resid / resid.size`` with
            ``resid = y_pred - y_observed``, shape
            ``(n_responses, n_responses)`` [response units**2], whose trace
            is the mean squared error over all entries; ``None`` when
            ``y_observed`` is not given.

        Raises
        ------
        ValueError
            If ``snv=True`` and a row of ``inputs`` has zero standard
            deviation across wavelengths; if ``y_observed`` does not have
            one row per input row and one column per fitted response; or if
            ``y_observed`` is given without ``full_output=True``.

        Notes
        -----
        New predictors follow the training preprocessing: with ``snv=True``
        each row is first SNV-transformed with its own across-wavelength mean
        and sample standard deviation (``ddof=1``, denominator
        ``n_wavelengths - 1``), as in training; then every row is centered with
        the training column means (and scaled with the training column
        standard deviations when ``standardize`` is True), so the scores
        remain on the fitted principal-component basis and a row's
        prediction does not depend on the other rows in ``inputs``.

        Warnings
        --------
        ``snv=True`` predictions differ from releases before #138, which
        recomputed column statistics from ``inputs`` instead of applying
        SNV.

        """
        inputs = np.atleast_2d(inputs)

        if self.snv:
            inputs, _ = self.__snv_transform(inputs)  # [-]

        inputs_centered = self.__center_data(inputs, self.data_mean,
                                             self.data_std)

        if regression_coeff is None:
            coeff = self.regression_coeff
        else:
            coeff = regression_coeff

        if num_comp is None:
            num_comp = self.num_comp

        if y_observed is not None and not full_output:
            raise ValueError("y_observed is only used with full_output=True")

        p_matrix = self.svd_dict['V'][:, :num_comp]
        new_projections = np.dot(inputs_centered, p_matrix)

        response = np.dot(new_projections, coeff) + self.y_means

        if full_output:
            if y_observed is None:
                # No observations of the new rows: no prediction error can
                # be computed (#253); the key is kept for compatibility.
                mse = None
            else:
                y_obs = self.__check_observed(y_observed, response.shape)
                resid = response - y_obs  # [response units]
                # [response units**2]
                mse = np.dot(resid.T, resid) / resid.size

            info_out = {'x_projected': new_projections, 'y_pred': response,
                        'MSE': mse}

            return info_out
        else:

            return response

    @staticmethod
    def __check_observed(y_observed, expected_shape):
        """Validate observed responses against the predicted responses.

        Parameters
        ----------
        y_observed : array-like
            Observed responses, shape ``(n_rows,)`` or
            ``(n_rows, n_responses)`` [response units].
        expected_shape : tuple of int
            Shape ``(n_rows, n_responses)`` of the predicted responses.

        Returns
        -------
        numpy.ndarray
            Observed responses as a float array of ``expected_shape``
            [response units].

        Raises
        ------
        ValueError
            If the observed and expected shapes are incompatible.
        """
        y_obs = np.asarray(y_observed, dtype=float)
        num_rows, num_responses = expected_shape

        if y_obs.ndim == 1 and num_responses == 1:
            y_obs = y_obs[:, np.newaxis]

        if y_obs.shape != tuple(expected_shape):
            accepted = f"{tuple(expected_shape)}"
            if num_responses == 1:
                accepted += f" or ({num_rows},)"
            raise ValueError(
                f"y_observed has shape {np.shape(y_observed)}; expected "
                f"{accepted}: one row per input spectrum and one column per "
                f"fitted response")

        return y_obs

    def evaluate_mse(self, num_comp=None):

        if num_comp is None:
            pc_counter = range(len(self.svd_dict['sv']))
        else:
            pc_counter = range(num_comp)

        mse = []
        residuals = []

        n_data = np.prod(self.y_data.shape)
        for n_component in pc_counter:
            coeff = self.get_regression(self.y_data, n_component + 1,
                                        update_instance=False)

            pred = self.predict(self.data, regression_coeff=coeff,
                                num_comp=n_component + 1)

            resid = self.y_data - pred

            mse_val = 1 / n_data * (resid**2).sum()

            mse.append(mse_val)
            residuals.append(resid)

        return mse, residuals

    def plot_parity(self, figsize=None):
        if figsize is None:
            figsize = (4, 3.5)
        fig, axis = plt.subplots(figsize=figsize)

        markers = ['o', 's', 'd', '*']

        y_pred = self.y_data - self.residuals

        minim = (np.minimum(self.y_data, y_pred)).min()
        maxim = (np.maximum(self.y_data, y_pred)).max()

        range_vals = maxim - minim

        left_bottom = [minim - range_vals*0.01]*2
        right_top = [maxim + range_vals*0.01]*2

        axis.plot(*zip(left_bottom, right_top), '--k', alpha=0.5)

        for ind in range(self.y_data.shape[1]):
            axis.plot(self.y_data[:, ind], y_pred[:, ind],
                      marker=markers[ind], mfc='None', ls='',
                      label=self.y_labels[ind])

        axis.legend()

        axis.set_xlabel('$%s{data}$' % self.y_name)
        axis.set_ylabel('$%s{model}$' % self.y_name)

        axis.xaxis.set_minor_locator(AutoMinorLocator(2))
        axis.yaxis.set_minor_locator(AutoMinorLocator(2))

        axis.text(1, 1.04, 'num_components = %i' % self.num_comp,
                  transform=axis.transAxes, ha='right')

        return fig, axis

    def cross_validation(self, num_groups=10):
        perm = np.random.permutation(self.data.shape[0])
