"""Real IPOPT fits of multivariate curve resolution (MCR), issue #240.

``optimize_fn(method='IPOPT')`` drives ``MultipleCurveResolution`` through
its scalar objective and gradient, then assembles the result with
``get_objective(..., set_self=False)``. These fits use the staggered-tracer
fixture of ``test_paramestim_mcr_observation_masks`` (positive
absorptivities, no penalty) and the negative-absorptivity fixture of
``test_paramestim_mcr_objective_consistency`` (active penalty), with model
sensitivities and with finite differences. Expected optima, objectives,
residual ordering and information come from the independent least-squares
and marginal-GLS helpers of those modules and from Levenberg-Marquardt
reference fits. Requires cyipopt, but not Assimulo; each fit takes at most a
few seconds.
"""

import numpy as np
import pytest

pytestmark = pytest.mark.integration
pytest.importorskip('cyipopt')

import test_paramestim_mcr_objective_consistency as consistency
import test_paramestim_mcr_observation_masks as case


# IPOPT's default tolerance (1e-8) is far below FIT_RTOL for these
# one-parameter fits; a hundred iterations is a generous ceiling.
IPOPT_OPTIONS = {'max_iter': 100}  # [-]
# With model sensitivities both IPOPT and LM reach the minimum of the same
# smooth objective; LM's 1e-8 relative step stop dominates their difference.
FIT_RTOL = 1e-6  # [-]
# With finite differences the IPOPT gradient is a forward difference of step
# h = dx_finitediff(k), whose bias (h / 2) f'' moves the stationary point by
# about h / 2; one full step bounds it.
FORWARD_STEPS = 1.0  # [-]
# Information J @ J.T against the central-difference information of
# case._marginal_gls: the budget of the core tests for model sensitivities
# (case.DERIVATIVE_RTOL); forward-difference Jacobians match to
# case.FORWARD_DIFFERENCE_RTOL of their largest entry, their products to
# twice that.
INFORMATION_RTOL = {'model_sensitivities': case.DERIVATIVE_RTOL,
                    'finite_difference': 2 * case.FORWARD_DIFFERENCE_RTOL}
# (model, weight key). Finite differences are fitted with identity weights.
# With the correlated weights the curvature f'' is about 5e4 s**2, so the
# forward-difference gradient is off by (h / 2) f'', about 9 s, near the
# optimum; IPOPT then cannot meet its tolerance and stops at max_iter after
# thousands of evaluations (still within one step h of the optimum).
FIT_CASES = [(case.model_with_sensitivities, None),
             (case.model_with_sensitivities, 'correlated'),
             (case.model_states_only, None)]
FIT_IDS = ['model_sensitivities-identity', 'model_sensitivities-correlated',
           'finite_difference-identity']
# A first-order rate constant is nonnegative. The penalized fixture also has
# a lower, unphysical minimum at a negative rate (growing reactant), which an
# unbounded IPOPT step can reach across the singular point k = 0, where the
# concentrations of A and B are collinear.
RATE_BOUNDS = [(0.0, None)]  # [1/s]


def _check_fit(estimator, fitted, reference, objective):
    """Compare an IPOPT optimum with an LM reference optimum.

    Parameters
    ----------
    estimator : MultipleCurveResolution
        The fitted estimator, for its forward-difference step.
    fitted : numpy.ndarray
        IPOPT rate, shape ``(1,)`` [1/s].
    reference : numpy.ndarray
        LM rate with model sensitivities, shape ``(1,)`` [1/s].
    objective : callable
        Independent objective [-] of a rate, shape ``(1,)`` [1/s].

    Notes
    -----
    With model sensitivities the IPOPT rate must also be a local minimum
    of ``objective`` at a relative offset of ``case.MINIMUM_PROBE``. The
    forward-difference bias h / 2 is of the order of that offset, so the
    finite-difference fit is checked against the LM optimum only.
    """
    if estimator.sens_second is None:
        step = estimator.dx_fd(reference)[0]  # [1/s]
        assert abs(fitted[0] - reference[0]) <= FORWARD_STEPS * step
    else:
        assert fitted[0] == pytest.approx(reference[0], rel=FIT_RTOL)
        at_fit = objective(fitted)  # [-]
        for factor in (1 - case.MINIMUM_PROBE, 1 + case.MINIMUM_PROBE):
            assert objective(fitted * factor) > at_fit


def _expected_identity_residuals(rate):
    """Return the identity-weighted residual vector of the staggered case.

    Parameters
    ----------
    rate : numpy.ndarray
        Rate k, shape ``(1,)`` [1/s].

    Returns
    -------
    numpy.ndarray
        Column-major over (wavelength 1 [-], wavelength 2 [-], tracer
        [mol/L]) and the stacked rows of both experiments: spectral data
        minus projection, tracer model minus data, zero where unobserved.
    """
    rows = case._measurement_errors(rate)
    columns = np.zeros((len(rows), 3))
    for index, (observed, values) in enumerate(rows):
        columns[index, :2] = values[:2]
        if 2 in observed:
            columns[index, 2] = -values[2]
    return columns.T.ravel()


@pytest.mark.parametrize(('model', 'weight_key'), FIT_CASES, ids=FIT_IDS)
def test_ipopt_mcr_fit_returns_parameters_covariance_and_info(model,
                                                              weight_key):
    estimator = case._estimator(model, weight_key)

    fitted, covariance, info = estimator.optimize_fn(
        method='IPOPT', optim_options=dict(IPOPT_OPTIONS), verbose=False)

    weights = (np.eye(3) if weight_key is None
               else case.COVARIANCES[weight_key])
    reference = case._estimator(case.model_with_sensitivities, weight_key)
    reference_fit, _, _ = reference.optimize_fn(verbose=False)  # [1/s]
    _check_fit(estimator, fitted, reference_fit,
               lambda rate: case._marginal_gls(rate, weights)[0])

    objective, information = case._marginal_gls(fitted, weights)

    unobserved = case._unobserved_tracer_entries()
    assert info['fun'].shape == unobserved.shape
    assert info['jac'].shape == (1, unobserved.size)
    np.testing.assert_array_equal(info['fun'][unobserved], 0.0)
    np.testing.assert_array_equal(info['jac'][:, unobserved], 0.0)
    # One zero penalty row per absorptivity (species x wavelengths).
    num_penalty = consistency.NUM_SPECIES * consistency.NUM_WAVELENGTHS
    np.testing.assert_array_equal(info['fun_penalty'],
                                  np.zeros(num_penalty))
    np.testing.assert_array_equal(info['jac_penalty'],
                                  np.zeros((1, num_penalty)))
    assert 0.5 * info['fun'] @ info['fun'] == pytest.approx(
        objective, rel=case.ROUNDOFF_RTOL)
    if weight_key is None:
        np.testing.assert_allclose(
            info['fun'], _expected_identity_residuals(fitted),
            rtol=case.ROUNDOFF_RTOL, atol=case.ROUNDOFF_RTOL)
    model_name = ('model_sensitivities' if estimator.sens_second is not None
                  else 'finite_difference')
    rtol = INFORMATION_RTOL[model_name]  # [-]
    assert (info['jac'] @ info['jac'].T)[0, 0] == pytest.approx(
        information, rel=rtol)

    dof = estimator.num_data_total - 1  # [-], one optimized rate
    assert covariance.shape == (1, 1)
    assert covariance[0, 0] == pytest.approx(
        2 * objective / dof / information, rel=rtol)

    # Reporting state belongs to the accepted rate.
    fresh = case._estimator(model, weight_key)
    at_fit = fresh.get_objective(fitted, True)[:unobserved.size]
    np.testing.assert_allclose(estimator.residuals, at_fit,
                               rtol=case.ROUNDOFF_RTOL,
                               atol=case.ROUNDOFF_RTOL)
    np.testing.assert_allclose(estimator.epsilon_mcr, fresh.epsilon_mcr,
                               rtol=case.ROUNDOFF_RTOL, atol=0.0)
    np.testing.assert_array_equal(estimator.params_convg, fitted)


@pytest.mark.parametrize('model', consistency.MODELS,
                         ids=consistency.MODEL_IDS)
def test_ipopt_mcr_fit_minimizes_penalized_objective(model):
    estimator = consistency._negative_estimator(model)

    fitted, covariance, info = estimator.optimize_fn(
        method='IPOPT', optim_options=dict(IPOPT_OPTIONS), verbose=False,
        bounds=RATE_BOUNDS)

    reference = consistency._negative_estimator(case.model_with_sensitivities)
    reference_fit, _, _ = reference.optimize_fn(verbose=False)  # [1/s]
    assert reference_fit[0] > 0
    _check_fit(estimator, fitted, reference_fit,
               lambda rate: consistency._negative_reference(rate)[2])

    data_residual, absorptivity, objective = consistency._negative_reference(
        fitted)
    assert np.any(absorptivity < 0)

    np.testing.assert_allclose(info['fun'], data_residual.T.ravel(),
                               rtol=case.ROUNDOFF_RTOL, atol=0.0)
    assert info['jac'].shape == (1, data_residual.size)
    assert info['jac_penalty'].shape == (1, absorptivity.size)
    active = absorptivity.ravel() < 0
    np.testing.assert_array_equal(info['fun_penalty'][~active], 0.0)
    np.testing.assert_allclose(
        0.5 * info['fun_penalty'][active]**2,
        consistency.PENALTY_WEIGHT * absorptivity.ravel()[active]**2,
        rtol=case.ROUNDOFF_RTOL, atol=0.0)
    assert 0.5 * (info['fun'] @ info['fun'] + info['fun_penalty']
                  @ info['fun_penalty']) == pytest.approx(
        objective, rel=case.ROUNDOFF_RTOL)
    assert covariance.shape == (1, 1) and covariance[0, 0] > 0
