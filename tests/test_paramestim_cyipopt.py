"""Real IPOPT fits for GLS, observation masks and bootstrap state (#397).

Reuse the independently derived linear-model expectations from the core tests.
These small fits require cyipopt, but not Assimulo. MCR's scalar-gradient and
post-solve failures remain owned by #240; this module does not treat either
failure as a passing optimization contract.
"""

import numpy as np
import pytest

pytestmark = pytest.mark.integration
pytest.importorskip('cyipopt')

from PharmaPy.ParamEstim import ParameterEstimation
from PharmaPy.StatsModule import StatisticsClass
import test_paramestim_gls_weighting as gls
import test_paramestim_observation_masks as masks
import test_statistics_bootstrap_state as bootstrap


# Tight termination for the synthetic linear models, whose true gradient is
# affine. A hundred iterations is a generous ceiling for these tiny fits.
IPOPT_OPTIONS = {'tol': 1e-10, 'max_iter': 100}  # [-], [-]
# The fixtures have O(1) parameters and normal-matrix condition numbers below
# 1e3. This budget covers the solver stop and finite-difference roundoff while
# remaining far below the O(1e-2) errors caused by the old weighting/masking.
FIT_RTOL = 1e-6  # [-]
FIT_ATOL = 1e-8  # parameter units: [mol/L/s], [mol/L], [mol/L/s**2]
# Jacobian roundoff is below 1e-8 relative for these affine fixtures; allow
# 1e-6 for its products and for the returned residual sum of squares.
STATISTICS_RTOL = 1e-6  # [-]
# Absolute floor for near-zero whitened residuals, with the same numerical
# scale rationale as FIT_ATOL but in the residual's dimensionless units.
WHITENED_ATOL = 1e-8  # [-]


def test_ipopt_correlated_gls_matches_closed_form():
    """Fit correlated data with real IPOPT and check returned statistics."""
    estimator = gls._estimator(gls._observations(), gls.COVARIANCE_MOL2_L2)
    expected, expected_covariance, normal = gls._closed_form_gls(
        gls.PRECISION_L2_MOL2)  # parameter units, their products, inverse products

    # Parameter units, their products, and weighted residual/Jacobian outputs.
    fitted, covariance, info = estimator.optimize_fn(
        method='IPOPT', optim_options=dict(IPOPT_OPTIONS), verbose=False)

    np.testing.assert_allclose(fitted, expected, rtol=FIT_RTOL, atol=FIT_ATOL)
    np.testing.assert_allclose(covariance, expected_covariance,
                               rtol=STATISTICS_RTOL, atol=0.0)
    # Independently whiten the known residuals in the original state order.
    residual = gls.two_state_model(expected, gls.TIME_S) - gls._observations()
    # [mol/L]
    weighted = residual @ np.linalg.cholesky(gls.PRECISION_L2_MOL2)  # [-]
    np.testing.assert_allclose(info['fun'], weighted.T.ravel(),
                               rtol=STATISTICS_RTOL, atol=WHITENED_ATOL)
    np.testing.assert_allclose(info['jac'] @ info['jac'].T, normal,
                               rtol=STATISTICS_RTOL, atol=0.0)


@pytest.mark.parametrize('sensitivity',
                         ['jac_fun', 'model_output', 'finite_difference'])
def test_ipopt_staggered_gls_uses_only_observed_information(sensitivity):
    """Fit sparse observations and check their actual information content.

    Parameters
    ----------
    sensitivity : str
        Analytic callback, model-returned or finite-difference sensitivities.
    """
    estimator = masks._estimator('correlated', sensitivity)
    _, observations = masks._datasets()  # per-state times [s], values [mol/L]
    expected = masks._closed_form('correlated', masks.SEED_PARAMS, observations)
    # Parameters in model units; covariance and information in their products
    # and inverse products; objective [-].
    at_solution = masks._closed_form(
        'correlated', expected['estimate'], observations)

    # Parameter units, their products, and weighted residual/Jacobian outputs.
    fitted, covariance, info = estimator.optimize_fn(
        method='IPOPT', optim_options=dict(IPOPT_OPTIONS), verbose=False)

    np.testing.assert_allclose(fitted, expected['estimate'],
                               rtol=FIT_RTOL, atol=FIT_ATOL)
    np.testing.assert_allclose(covariance, expected['covariance'],
                               rtol=STATISTICS_RTOL, atol=0.0)
    np.testing.assert_allclose(info['jac'] @ info['jac'].T, expected['normal'],
                               rtol=STATISTICS_RTOL, atol=0.0)
    assert 0.5 * info['fun'] @ info['fun'] == pytest.approx(
        at_solution['objective'], rel=STATISTICS_RTOL)
    # The fixture's unequal grids contain real missing entries. Keep their
    # state-major positions but contribute neither residual nor information.
    unobserved = ~np.concatenate([mask.T.ravel() for mask in estimator.x_masks])
    assert unobserved.any() and (~unobserved).any()
    np.testing.assert_array_equal(info['fun'][unobserved], 0.0)
    np.testing.assert_array_equal(info['jac'][:, unobserved], 0.0)


def test_ipopt_bootstrap_preserves_fit_and_matches_sample_optima():
    """Refit real IPOPT bootstrap samples without overwriting the fit."""
    last_call = {}
    estimator = ParameterEstimation(
        bootstrap._guarded_model(-np.inf, ValueError, last_call),
        bootstrap.SEED_RATE, bootstrap.TIMES_S, bootstrap._observations(),
        jac_fun=bootstrap._linear_jacobian, name_params=['rate_mol_l_s'])
    estimator.optimize_fn(method='IPOPT', optim_options=dict(IPOPT_OPTIONS),
                          verbose=False)
    statistics = StatisticsClass(estimator)
    before = bootstrap._state(estimator)
    expected = bootstrap._sample_estimates(statistics)  # [mol/L/s]
    assert np.all(expected != estimator.params_convg[0])

    fitted = bootstrap._seeded(
        statistics.bootstrap_params, bootstrap.NUM_SAMPLES)  # [mol/L/s]

    assert fitted.shape == (bootstrap.NUM_SAMPLES, 1)
    np.testing.assert_allclose(fitted[:, 0], expected,
                               rtol=bootstrap.REFIT_RTOL, atol=0.0)
    bootstrap._assert_same(before, bootstrap._state(estimator))
    # The shared model is restored without a user evaluation after bootstrap.
    assert last_call['rate'] == pytest.approx(estimator.params_convg[0],
                                            rel=FIT_RTOL, abs=0.0)
