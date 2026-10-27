"""Regression tests for the Gaussian-peak amplitude derivative (#254).

Scope: ``Gaussians.gaussian_dparam``, the multi-peak parameter Jacobian
``Gaussians.gauss_dparam_mult``, and their public caller
``ParamEstim.Deconvolution.dparam_wrapper`` used as ``jac_fun`` by
``ParameterEstimation``. The fixtures are synthetic peak shapes on a
dimensionless abscissa [x] (for example a scaled wavelength); amplitudes are
peak areas [signal*x]. Expected values come from ``scipy.stats.norm.pdf`` and
from central finite differences, not from the production expressions.
"""

import numpy as np
import pytest
from scipy.stats import norm

from PharmaPy import Gaussians as gs
from PharmaPy.ParamEstim import Deconvolution, ParameterEstimation

pytestmark = pytest.mark.unit

# Seven abscissa points against two peaks, so a (row, column) transposition
# of the (n_x, 3 * n_peaks) Jacobian cannot pass a shape check.
X_GRID = np.linspace(-3.0, 3.0, 7)  # [x]

# Central-difference step on a parameter [parameter units]. The peak is
# linear in the amplitude, so the central difference has no truncation error
# in that direction and its round-off error is about
# machine_eps * max|f| / step ~ 2.2e-16 * 2 / 1e-4 ~ 4e-12 [signal/x]. For
# mu and sigma the truncation error is step**2 / 6 * |f'''|. The largest
# third derivative in the fixtures (areas <= 1, sigma >= 0.5) is the sigma
# one of the unit-area peak with sigma = 0.5: at x = mu it is
# d3f/dsigma3 = -6 / (sigma**4 * sqrt(2 pi)) ~ -38.3, and its maximum over x
# (near |x - mu| ~ sigma) is ~38.8 (the mu derivative peaks at ~8.8). The
# bound is therefore 1e-8 * 38.8 / 6 ~ 6.5e-8.
FD_STEP = 1e-4  # [parameter units]
FD_ATOL_LINEAR = 1e-10  # [1/x], > 4e-12 round-off bound above
FD_ATOL_NONLINEAR = 1e-6  # [signal/x**2], > 10x the 6.5e-8 bound above
# Agreement with scipy's closed-form pdf: both evaluate the same expression
# in double precision, so a few ulps of the O(1) values suffice.
PDF_RTOL = 1e-12  # [-]


def _central_difference(fun, params, index, step=FD_STEP):
    """Return the central difference of ``fun`` along one parameter.

    Parameters
    ----------
    fun : callable
        ``fun(params) -> numpy.ndarray`` of shape ``(n_x,)`` [signal/x].
    params : numpy.ndarray
        Parameter vector, shape ``(n_params,)`` [parameter units].
    index : int
        Position of the perturbed parameter.
    step : float, optional
        Perturbation [parameter units].

    Returns
    -------
    numpy.ndarray
        Derivative estimate, shape ``(n_x,)`` [signal/x per parameter unit].
    """
    delta = np.zeros_like(params)  # [parameter units]
    delta[index] = step
    return (fun(params + delta) - fun(params - delta)) / (2.0 * step)


@pytest.mark.parametrize("ampl", [0.0, 2.0])  # [signal*x]
def test_amplitude_derivative_is_unit_area_gaussian(ampl):
    mu = 0.4  # [x]
    sigma = 0.7  # [x]

    _, _, df_dampl = gs.gaussian_dparam(X_GRID, mu, sigma, ampl)

    expected = norm.pdf(X_GRID, loc=mu, scale=sigma)  # [1/x]
    assert np.all(np.isfinite(df_dampl))
    np.testing.assert_allclose(df_dampl, expected, rtol=PDF_RTOL, atol=0.0)

    params = np.array([mu, sigma, ampl])  # [x], [x], [signal*x]

    def peak(theta):
        return gs.gaussian(X_GRID, *theta)

    fd_ampl = _central_difference(peak, params, 2)  # [1/x]
    np.testing.assert_allclose(df_dampl, fd_ampl, rtol=0.0,
                               atol=FD_ATOL_LINEAR)


def test_multi_peak_jacobian_with_zero_amplitude_component():
    mus = np.array([-1.0, 1.0])  # [x]
    sigmas = np.array([0.5, 0.8])  # [x]
    ampls = np.array([1.0, 0.0])  # [signal*x]; second peak switched off

    jac = gs.gauss_dparam_mult(X_GRID, mus, sigmas, ampls)

    # Columns are [mu_1, mu_2, sigma_1, sigma_2, ampl_1, ampl_2].
    assert jac.shape == (len(X_GRID), 6)
    assert np.all(np.isfinite(jac))

    unit_pdfs = np.column_stack([
        norm.pdf(X_GRID, loc=mu, scale=sigma)
        for mu, sigma in zip(mus, sigmas)])  # [1/x]
    np.testing.assert_allclose(jac[:, 4:], unit_pdfs, rtol=PDF_RTOL,
                               atol=0.0)
    # A zero-area peak has no shape sensitivity.
    np.testing.assert_array_equal(jac[:, 1], 0.0)
    np.testing.assert_array_equal(jac[:, 3], 0.0)
    # Guard against a vacuous all-zero mu block: the live peak is sensitive.
    assert np.any(jac[:, 0] != 0)

    params = np.concatenate((mus, sigmas, ampls))

    def peaks(theta):
        return gs.multiple_gaussian(X_GRID, *np.split(theta, 3))

    for column in range(6):
        atol = FD_ATOL_LINEAR if column >= 4 else FD_ATOL_NONLINEAR
        fd_column = _central_difference(peaks, params, column)
        np.testing.assert_allclose(jac[:, column], fd_column, rtol=0.0,
                                   atol=atol, err_msg=f"column {column}")


def test_deconvolution_fit_from_zero_amplitude_seed_uses_finite_jacobian():
    """A zero amplitude seed must not poison the analytic ``jac_fun``.

    Before #254 the NaN amplitude column made the Levenberg-Marquardt normal
    matrix singular (``LinAlgError``). The data are noise-free, so the fit
    must recover the generating parameters.
    """
    x_data = np.linspace(-3.0, 3.0, 25)  # [x]
    true_params = np.array([-1.0, 1.0,  # mu [x]
                            0.5, 0.8,  # sigma [x]
                            1.0, 0.6])  # area [signal*x]
    y_data = gs.multiple_gaussian(x_data, *np.split(true_params, 3))

    deconv = Deconvolution(np.array([-0.9, 1.1]),  # [x]
                           np.array([0.6, 0.7]),  # [x]
                           np.array([1.0, 0.0]),  # [signal*x]
                           x_data, y_data)
    seed = deconv.concat_params()

    estimator = ParameterEstimation(deconv.fun_wrapper, seed, x_data,
                                    y_data, jac_fun=deconv.dparam_wrapper)
    params, _, _ = estimator.optimize_fn(verbose=False)

    jac_seed = deconv.dparam_wrapper(seed, x_data)
    assert np.all(np.isfinite(jac_seed))

    # Noise-free data: the converged parameters reproduce the generating set
    # to the default Levenberg-Marquardt stopping tolerances
    # (eps_1 = eps_2 = 1e-8 in LevMarq.levenberg_marquardt); the bound keeps
    # two orders of magnitude of margin.
    param_atol = 1e-6  # [parameter units]
    np.testing.assert_allclose(params, true_params, rtol=0.0,
                               atol=param_atol)
