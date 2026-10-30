"""Real IPOPT fits for GLS, observation masks and bootstrap state (#397).

Reuse the independently derived linear-model expectations from the core tests.
Also check that IPOPT fits leave the caller's ``optim_options`` unchanged
(#476), with the proportional fixture of
``test_paramestim_optimize_fn_contract``, and that they report Ipopt's
termination outcome and warn when it is unsuccessful (#475), also through
``estimate_parameters`` with the warning pointing at the caller, and record
the termination options in ``info['tolerances']`` (#352), with the
synthetic first-order decay of the #475 reproduction.
These small fits require cyipopt, but not Assimulo. MCR IPOPT fits are
covered by ``test_paramestim_mcr_cyipopt`` (#240).
"""

import warnings

import numpy as np
import pytest
from scipy.optimize import least_squares

pytestmark = pytest.mark.integration
pytest.importorskip('cyipopt')

from PharmaPy.ParamEstim import ParameterEstimation
from PharmaPy.StatsModule import StatisticsClass
import test_paramestim_gls_weighting as gls
import test_paramestim_observation_masks as masks
import test_statistics_bootstrap_state as bootstrap
import test_paramestim_optimize_fn_contract as contract


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


def test_ipopt_leaves_caller_options_unchanged():
    """Reuse one options dict for two IPOPT fits of the same data (#476)."""
    data = contract.proportional([contract.TRUE_RATE],
                                 contract.TIME_S)  # [mol/L]
    options = dict(IPOPT_OPTIONS)
    fitted = []  # [mol/L/s]
    for _ in range(2):
        estimator = ParameterEstimation(
            contract.proportional, [contract.SEED_RATE], contract.TIME_S,
            data)
        fitted.append(estimator.optimize_fn(
            method='IPOPT', optim_options=options, verbose=False)[0])

        assert options == IPOPT_OPTIONS
        # optim_options is optimize_fn's private copy as cyipopt left it:
        # bytes keys, print_level 0 for verbose=False.
        stored = estimator.optim_options
        assert stored is not options
        assert {key: stored[key.encode()] for key in IPOPT_OPTIONS} == (
            IPOPT_OPTIONS)
        assert stored[b'print_level'] == 0

    np.testing.assert_array_equal(fitted[1], fitted[0])
    np.testing.assert_allclose(fitted[0], [contract.TRUE_RATE],
                               rtol=FIT_RTOL, atol=FIT_ATOL)


# Synthetic first-order decay c = c0 * exp(-k t) of the #475 reproduction,
# with a constant offset so the data are not exactly representable.
DECAY_TIME_S = np.linspace(0.0, 4.0, 5)  # [s]
DECAY_INITIAL = 2.0  # [mol/L], c0
DECAY_TRUE_RATE = 0.5  # [1/s], rate that generates the data
DECAY_OFFSET = 0.01  # [mol/L], constant offset added to the data
# A seed an order of magnitude below the optimum: one IPOPT iteration from
# it ends far from convergence (k about 1.0 1/s in the reproduction).
DECAY_SEED_RATE = 0.05  # [1/s]
DECAY_BOUNDS = [(0.01, 2.0)]  # [1/s], synthetic range containing the optimum
ONE_ITERATION = {'max_iter': 1}  # [-]
# Ipopt ApplicationReturnStatus values (IpReturnCodes_inc.h).
IPOPT_SOLVE_SUCCEEDED = 0  # [-]
IPOPT_MAXIMUM_ITERATIONS_EXCEEDED = -1  # [-]
UNSUCCESSFUL_IPOPT = 'IPOPT terminated unsuccessfully'
# Ipopt tol [-] that cyipopt 1.7's minimize_ipopt writes into the options
# when the caller gives none (cyipopt/scipy_interface.py:
# ``options[b'tol'] = tol or 1e-8``), the value Ipopt then stops on.
CYIPOPT_DEFAULT_TOL = 1e-8  # [-]
# Termination tolerances of the independent SciPy reference fit, a few
# machine epsilons (2.2e-16), so the reference optimum is resolved far
# below DECAY_FIT_RTOL.
REFERENCE_TOL = 1e-15  # [-]
# Relative error budget of the converged nonlinear decay fit, derived at the
# optimum k* = 0.4923 1/s, where J @ J.T = sum((c0 t exp(-k t))**2) = 6.85
# [(mol/L)**2 s**2] and sum(r * d2r/dk2) = -2.2e-3 [(mol/L)**2 s**2]
# (r = model - data). The estimator differentiates by forward differences
# with the default relative step h = sqrt(1e-6) * k, whose gradient bias
# (h / 2) * sum(r * d2r/dk2) shifts k by about 7.8e-8 1/s, 1.6e-7 relative;
# IPOPT's tol = 1e-10 on the gradient adds only about tol / (J @ J.T) / k =
# 3e-11. 1e-6 bounds the sum with a sixfold margin.
DECAY_FIT_RTOL = 1e-6  # [-]


def decay(params, time_s):
    """Return the concentration of a first-order decay.

    Parameters
    ----------
    params : array_like
        ``[k]``, rate constant [1/s].
    time_s : numpy.ndarray
        Sample times [s], shape ``(n_times,)``.

    Returns
    -------
    numpy.ndarray
        Concentration ``c0 * exp(-k t)`` [mol/L], shape ``(n_times,)``.
    """
    return DECAY_INITIAL * np.exp(-params[0] * time_s)


def decay_data():
    """Return the offset decay data.

    Returns
    -------
    numpy.ndarray
        ``decay([DECAY_TRUE_RATE], DECAY_TIME_S) + DECAY_OFFSET`` [mol/L].
    """
    return decay([DECAY_TRUE_RATE], DECAY_TIME_S) + DECAY_OFFSET


def decay_estimator():
    """Return an estimator of k for the offset decay data.

    Returns
    -------
    ParameterEstimation
        Unfitted estimator seeded at ``DECAY_SEED_RATE`` [1/s].
    """
    return ParameterEstimation(decay, [DECAY_SEED_RATE], DECAY_TIME_S,
                               decay_data())


def test_ipopt_iteration_limit_is_reported_and_warned():
    estimator = decay_estimator()

    with pytest.warns(RuntimeWarning, match=(
            rf"{UNSUCCESSFUL_IPOPT} \(status "
            rf"{IPOPT_MAXIMUM_ITERATIONS_EXCEEDED}: Maximum number of "
            "iterations exceeded")):
        _, _, info = estimator.optimize_fn(
            method='IPOPT', optim_options=dict(ONE_ITERATION),
            bounds=DECAY_BOUNDS, verbose=False)

    assert info['success'] is False
    assert info['status'] == IPOPT_MAXIMUM_ITERATIONS_EXCEEDED
    assert info['message'].startswith('Maximum number of iterations')
    assert info['nit'] == ONE_ITERATION['max_iter']
    assert estimator.info_opt is info


def test_converged_ipopt_fit_reports_success_without_warning():
    estimator = decay_estimator()
    data = decay_data()  # [mol/L]
    # Independent least-squares optimum of the same residuals with SciPy.
    expected = least_squares(
        lambda rate: decay(rate, DECAY_TIME_S) - data, [DECAY_SEED_RATE],
        bounds=tuple(np.array(DECAY_BOUNDS).T), xtol=REFERENCE_TOL,
        ftol=REFERENCE_TOL, gtol=REFERENCE_TOL).x  # [1/s]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        fitted, _, info = estimator.optimize_fn(
            method='IPOPT', optim_options=dict(IPOPT_OPTIONS),
            bounds=DECAY_BOUNDS, verbose=False)

    assert not [item for item in caught
                if UNSUCCESSFUL_IPOPT in str(item.message)]
    assert info['success'] is True
    assert info['status'] == IPOPT_SOLVE_SUCCEEDED
    assert isinstance(info['message'], str)
    assert 1 <= info['nit'] <= IPOPT_OPTIONS['max_iter']
    np.testing.assert_allclose(fitted, expected, rtol=DECAY_FIT_RTOL,
                               atol=0.0)


def test_ipopt_outcome_through_estimate_parameters_points_at_caller():
    """The #475 warning and the recorded tolerances through the wrapper.

    ``max_iter`` is given as a keyword of ``estimate_parameters``; the
    warning must point at this file, not at ``ParamEstim.py``, and
    ``info['tolerances']`` holds that limit and cyipopt's ``tol`` default
    (#352).
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        _, _, info = decay_estimator().estimate_parameters(
            method='IPOPT', bounds=DECAY_BOUNDS, verbose=False,
            **ONE_ITERATION)

    unsuccessful = [item for item in caught
                    if UNSUCCESSFUL_IPOPT in str(item.message)]
    assert len(unsuccessful) == 1
    assert unsuccessful[0].filename == __file__
    assert info['success'] is False
    assert info['tolerances'] == {'tol': CYIPOPT_DEFAULT_TOL,
                                  'max_iter': ONE_ITERATION['max_iter']}
