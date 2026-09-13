#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 22:06:49 2020

@author: casas100
"""

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.special import comb


def local_newton_interpolation(time, t_data, y_data, num_points=3):
    idx_time = np.argmin(abs(time - t_data))

    idx_lower = max(0, min(idx_time - 1, len(t_data) - num_points))
    idx_upper = idx_lower + num_points

    t_interp = t_data[idx_lower:idx_upper]
    y_interp = y_data[idx_lower:idx_upper]

    # Newton interpolation
    interp = NewtonInterpolation(t_interp, y_interp)
    y_target = interp.evalPolynomial(time)

    return y_target


class NewtonInterpolation:
    def __init__(self, x_data, y_data):

        self.x_data = x_data
        self.y_data = y_data

        self.coeff = self.__getCoefficients()

    def __getCoefficients(self):
        """
        x: list or np array contanining x data points
        y: list or np array contanining y data points
        """

        n = len(self.x_data)
        x = self.x_data

        a = np.copy(self.y_data)

        if a.ndim == 1:
            for k in range(1, n):
                a[k:n] = (a[k:n] - a[k - 1]) / (x[k:n] - x[k - 1])

        else:
            for k in range(1, n):
                coeff = (a[k:n] - a[k - 1]).T / (x[k:n] - x[k - 1])
                a[k:n] = coeff.T

        return a

    def evalPolynomial(self, x):
        """
        x_data: data points at x
        y_data: data points at y
        x: evaluation point(s)
        """

        a = self.coeff
        n = len(self.x_data) - 1  # Degree of polynomial
        p = a[n]

        if isinstance(x, float) or isinstance(x, int):
            x_eval = x
        elif a.ndim == 1:
            x_eval = x
        else:
            x_eval = x[..., np.newaxis]

        for k in range(1, n + 1):
            p = a[n - k] + (x_eval - self.x_data[n - k])*p

        return p


def smoothstep(x, x_min=0, x_max=1, N=1):
    x = np.clip((x - x_min) / (x_max - x_min), 0, 1)

    result = 0
    for n in range(0, N + 1):
        result += comb(N + n, n) * comb(2 * N + 1, N - n) * (-x)**n

    result *= x ** (N + 1)

    return result


# class SplineInterpolation:
#     def __init__(self, x_data, y_data):

#         self.Spline = CubicSpline(x_data, y_data)

#     def evalSpline(self, x):

#         y_interp = self.Spline(x)

#         return y_interp


class PiecewiseLagrange:

    def __init__(self, time_final, y_vals, order=2, time_k=None, time_zero=0):
        """ Create a piecewise Lagrange interpolation object

        Parameters
        ----------
        time_final : float
            final time
        y_vals : numpy array
            2D array with dimensions num_intervals x order. Each row contains
            the values of the function at the collocation point within a given
            finite element
        order : int
            order of the Lagrange polynomial (1 for piecewise constant,
            2 for linear...)
        time_k : array-like (optional)
            If None, the time horizon is divided in num_intervals equally
            spaced intervals. Otherwise, time_k represents the limits of the
            finite elements.
        time_zero : float (optional)
            initial time (default is zero)


        The interpolation algorithm is based on the work of Vassiliadis et al
        (Ind. Eng. Chem. Res., 1994, 33, 2111-2122)

        """

        y_vals = np.atleast_1d(y_vals)

        if y_vals.ndim == 1:
            y_vals = y_vals.reshape(-1, order)

        if time_k is None:
            num_interv = len(y_vals)
            time_k = np.linspace(time_zero, time_final, num_interv + 1)
        else:
            num_interv = len(time_k) - 1
            time_k = np.asarray(time_k)

            if len(y_vals) != num_interv:
                raise ValueError("The number of rows in 'y_vals' must match "
                                 "the number of intervals given by the length "
                                 "of 'time_k'")

        delta_t = np.diff(time_k)
        if (delta_t < 0).any():
            raise ValueError("Values in 'time_k' must be strictly increasing.")
        equal = np.isclose(delta_t[1:], delta_t[:-1]).all()
        self.equal_dt = equal

        self.time_k = time_k

        if equal:
            self.dt = delta_t[0]
        else:
            self.dt = delta_t

        self.order = order

        self.num_interv = num_interv
        self.y_vals = y_vals

    def evaluate_poly(self, time_eval, y_init=None):
        if y_init is not None:
            self.y_vals[0, 0] = y_init

        time_k = self.time_k

        if self.equal_dt:
            k = np.ceil((time_eval - time_k[0]) / self.dt).astype(int)
            k = np.maximum(1, k)
        else:
            k = np.searchsorted(self.time_k, time_eval, side='right')
            k = np.minimum(self.num_interv, k)

        k = np.clip(k, 1, self.num_interv)

        # ---------- Time normalization
        tau_k = (time_eval - time_k[k - 1]) / (time_k[k] - time_k[k - 1])
        tau_k = tau_k[..., np.newaxis]

        # ---------- Build Lagrange polynomials
        # Intermediate collocation points
        colloc = np.linspace(0, 1, self.order)

        # Lagrange polynomials for k = 1, ..., K (all intervals)
        i_set = set(range(self.order))

        if isinstance(time_eval, np.ndarray):
            ntimes = len(time_eval)

            poly = np.zeros((ntimes, self.order))

            for i in i_set:
                i_pr = list(i_set.difference([i]))
                poly_indiv = (tau_k - colloc[i_pr]) / (colloc[i] - colloc[i_pr])

                poly[:, i] = poly_indiv.prod(axis=1)

            u_time = np.zeros_like(time_eval, dtype=float)

            for ind in np.unique(k):
                row_map = k == ind
                poly_k = poly[row_map]
                u_time[row_map] = np.dot(poly_k,
                                         self.y_vals[ind - 1]).flatten()

        else:
            poly = np.zeros(self.order)
            for i in i_set:
                i_pr = list(i_set.difference([i]))
                poly_indiv = (tau_k - colloc[i_pr]) / (colloc[i] - colloc[i_pr])

                poly[i] = poly_indiv.prod()

            u_time = np.dot(poly, self.y_vals[k - 1])

        return u_time
