# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 18:21:24 2020

@author: dcasasor
"""

import numpy as np


def gaussian(x, mu=0, sigma=1, ampl=1):
    func = ampl/sigma/np.sqrt(2*np.pi) * np.exp(-0.5*((x - mu) / sigma)**2)

    return func


def multiple_gaussian(x, mus, sigmas, ampls, separated=False):

    num_mu = len(mus)
    num_sigma = len(sigmas)
    num_ampl = len(ampls)

    num_x = len(x)

    equal = num_mu == num_sigma == num_ampl
    if not equal:
        raise RuntimeError("'mus', 'sigmas' and 'ampls' must have the same "
                           "number of elements")

    gaussians = np.zeros((num_mu, num_x))
    for ind in range(num_mu):
        gaussians[ind] = gaussian(x, mus[ind], sigmas[ind], ampls[ind])

    gaussians = gaussians.T

    if separated:
        return gaussians
    else:
        return gaussians.sum(axis=1)


def gaussian_dparam(x, mu=0, sigma=1, ampl=1):
    """Return the parameter derivatives of a normalized Gaussian peak.

    The peak is
    ``f(x) = ampl / (sigma * sqrt(2 pi)) * exp(-((x - mu) / sigma)**2 / 2)``,
    so ``ampl`` is the peak area and ``f`` has units of [ampl]/[x].

    Parameters
    ----------
    x : float or numpy.ndarray
        Abscissa values at which the derivatives are evaluated [x], e.g.
        wavelength [nm] or particle size [m]. Any shape.
    mu : float, optional
        Peak center, in the units of ``x`` [x]. Default is 0.
    sigma : float, optional
        Peak standard deviation, in the units of ``x`` [x]; must be nonzero.
        Default is 1.
    ampl : float, optional
        Peak area [ampl] = [signal]*[x]. Any finite value, including zero.
        Default is 1.

    Returns
    -------
    df_dmu : float or numpy.ndarray
        Derivative with respect to ``mu`` [ampl/x**2], shape of ``x``.
    df_dsigma : float or numpy.ndarray
        Derivative with respect to ``sigma`` [ampl/x**2], shape of ``x``.
    df_dampl : float or numpy.ndarray
        Derivative with respect to ``ampl`` [1/x], shape of ``x``.

    Notes
    -----
    The peak is linear in ``ampl``, so ``df_dampl`` is the unit-area
    Gaussian ``gaussian(x, mu, sigma, 1)``. Evaluating it directly, rather
    than as ``f / ampl``, keeps the derivative finite at ``ampl = 0`` (#254).
    """
    gauss = gaussian(x, mu, sigma, ampl)  # [ampl/x]

    df_dmu = gauss * (x - mu) / sigma**2  # [ampl/x**2]
    df_dsigma = gauss / sigma * (((x - mu) / sigma)**2 - 1)  # [ampl/x**2]
    df_dampl = gaussian(x, mu, sigma, 1)  # [1/x]

    return df_dmu, df_dsigma, df_dampl


def gaussian_dx(x, mu=0, sigma=1, ampl=1):
    gauss = gaussian(x, mu, sigma, ampl)

    df_dx = -gauss * (x - mu) / sigma**2

    return df_dx


def gaussian_dxdx(x, mu=0, sigma=1, ampl=1):
    gauss = gaussian(x, mu, sigma, ampl)

    df_dxdx = gauss / sigma**2 * (((x - mu) / sigma)**2 - 1)

    return df_dxdx


def gauss_dparam_mult(x, mus, sigmas, ampls):
    num_mu = len(mus)
    num_sigma = len(sigmas)
    num_ampl = len(ampls)

    num_x = len(x)

    equal = num_mu == num_sigma == num_ampl
    if not equal:
        raise RuntimeError("'mus', 'sigmas' and 'ampls' must have the same "
                           "number of elements")

    dgauss_dmu = np.zeros((num_x, num_mu))
    dgauss_dsigma = np.zeros_like(dgauss_dmu)
    dgauss_dampl = np.zeros_like(dgauss_dmu)

    for ind in range(num_mu):
        df_dmu, df_dsigma, df_dampl = gaussian_dparam(x, mus[ind], sigmas[ind],
                                                      ampls[ind])

        dgauss_dmu[:, ind] = df_dmu
        dgauss_dsigma[:, ind] = df_dsigma
        dgauss_dampl[:, ind] = df_dampl

    dgauss_dparam = np.hstack((dgauss_dmu, dgauss_dsigma, dgauss_dampl))

    return dgauss_dparam


def gauss_dx_mult(x, mus, sigmas, ampls):
    num_mu = len(mus)
    num_sigma = len(sigmas)
    num_ampl = len(ampls)

    num_x = len(x)

    equal = num_mu == num_sigma == num_ampl
    if not equal:
        raise RuntimeError("'mus', 'sigmas' and 'ampls' must have the same "
                           "number of elements")

    dgauss_dx = np.zeros((num_mu, num_x))

    for ind in range(num_mu):
        dgauss_dx[ind] = gaussian_dx(x, mus[ind], sigmas[ind], ampls[ind])

    dgauss_dx = dgauss_dx.sum(axis=0)

    return dgauss_dx


def gauss_dxdx_mult(x, mus, sigmas, ampls):
    num_mu = len(mus)
    num_sigma = len(sigmas)
    num_ampl = len(ampls)

    num_x = len(x)

    equal = num_mu == num_sigma == num_ampl
    if not equal:
        raise RuntimeError("'mus', 'sigmas' and 'ampls' must have the same "
                           "number of elements")

    dgauss_dxdx = np.zeros((num_mu, num_x))

    for ind in range(num_mu):
        dgauss_dxdx[ind] = gaussian_dxdx(x, mus[ind], sigmas[ind], ampls[ind])

    dgauss_dxdx = dgauss_dxdx.sum(axis=0)

    return dgauss_dxdx
