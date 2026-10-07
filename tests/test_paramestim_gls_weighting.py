"""Regression tests for correlated GLS weighting in ParameterEstimation.

Issue #237: ``weight_matrix`` is the measurement-error covariance of the
measured states, and the stored root ``sigma_inv`` must reproduce its inverse
(the precision) in measured-state order. The fixtures use precisions whose
Bunch-Kaufman LDL factorization pivots, which previously permuted the
precision. Concentrations are [mol/L] over time [s]; the two-state model is
linear in three parameters, so the generalized least-squares (GLS) objective,
gradient, estimate, and parameter covariance have closed forms written here
directly from the precision literal rather than from the estimator's root.
A multivariate curve resolution (MCR) case checks that
``MultipleCurveResolution`` applies the same precision to its projected
spectral residuals. All fits use PharmaPy's Levenberg-Marquardt solver on
five samples and run in milliseconds without optional solver backends; the
``SimulationExec`` handoff is covered in ``test_simexec_gls_weighting.py``.
"""

import numpy as np
import pytest
from scipy.linalg import ldl

from PharmaPy import ParamEstim


pytestmark = pytest.mark.unit

# Synthetic sampling times; unequal spacing and a count (5) distinct from the
# number of states (2) and parameters (3) expose axis-ordering mistakes.
TIME_S = np.array([0.5, 1.0, 2.0, 3.5, 5.0])  # [s]

# Variance scale 1/16 [(mol/L)**2] is a power of two, so the covariance and
# precision literals below are exact inverses in binary floating point.
VARIANCE_SCALE_MOL2_L2 = 0.0625  # [(mol/L)**2]
# Correlated precision P of the states (c_A, c_B); det(P / 16) = 1. Its
# Bunch-Kaufman LDL factorization pivots (asserted in the tests).
PRECISION_L2_MOL2 = np.array(
    [[1.0, 2.0], [2.0, 5.0]]) / VARIANCE_SCALE_MOL2_L2  # [(L/mol)**2]
# Covariance = inv(P): standard deviations 0.56 and 0.25 mol/L, correlation
# -2/sqrt(5) = -0.89 [-].
COVARIANCE_MOL2_L2 = np.array(
    [[5.0, -2.0], [-2.0, 1.0]]) * VARIANCE_SCALE_MOL2_L2  # [(mol/L)**2]

# Parameter order: rate k [mol/L/s], shared offset c0 [mol/L], quadratic
# coefficient a [mol/L/s**2].
TRUE_PARAMS = np.array([0.8, 0.5, 0.12])  # [mol/L/s], [mol/L], [mol/L/s**2]
SEED_PARAMS = np.array([1.0, 0.2, 0.1])  # [mol/L/s], [mol/L], [mol/L/s**2]
# Fixed synthetic measurement errors, columns (c_A, c_B), chosen asymmetric so
# GLS, ordinary least squares, and the misweighted fit all differ.
MEASUREMENT_ERRORS_MOL_L = np.array([  # [mol/L]
    [0.30, -0.10],
    [-0.45, 0.25],
    [0.20, -0.30],
    [0.55, 0.05],
    [-0.35, 0.40],
])
NUM_DATA = MEASUREMENT_ERRORS_MOL_L.size  # [-]
NUM_PARAMS = TRUE_PARAMS.size  # [-]

# LM gradient-norm stop; eps_2 = 0 disables the step-size stop so the
# returned point satisfies ||J_w^T r_w|| < LM_GRADIENT_TOL (see _lm_options).
# Design choice: LM reaches about 4e-11 on these fixtures, but with the
# pre-#237 misweighting it stalled near 1e-8, where objective decreases fall
# below roundoff and trial steps are rejected. 1e-6 keeps two orders of
# headroom above that stall, so the red run fails on the estimate rather than
# the stop reason, while bounding the parameter error near 2e-8, six orders
# below the 2e-2 shift caused by misweighting.
LM_GRADIENT_TOL = 1e-6  # [weighted objective unit / parameter unit]


def two_state_model(params, time_s):
    """Return concentrations of two states for the GLS fixture.

    Parameters
    ----------
    params : numpy.ndarray
        Shape ``(3,)``: rate k [mol/L/s], offset c0 [mol/L], and quadratic
        coefficient a [mol/L/s**2].
    time_s : numpy.ndarray
        Sampling times, shape ``(n_times,)`` [s].

    Returns
    -------
    numpy.ndarray
        Shape ``(n_times, 2)``, columns ``c_A = k t + c0`` and
        ``c_B = c0 + a t**2`` [mol/L].
    """
    rate, offset, quadratic = params  # [mol/L/s], [mol/L], [mol/L/s**2]
    return np.column_stack(
        (rate * time_s + offset, offset + quadratic * time_s**2))


def two_state_jacobian(params, time_s):
    """Return analytic state-major sensitivities of ``two_state_model``.

    Parameters
    ----------
    params : numpy.ndarray
        Shape ``(3,)``, as in ``two_state_model``. Unused because the model
        is linear in its parameters.
    time_s : numpy.ndarray
        Sampling times, shape ``(n_times,)`` [s].

    Returns
    -------
    numpy.ndarray
        Shape ``(2 * n_times, 3)``: rows of ``c_A`` then ``c_B`` over time,
        columns d/dk [s], d/dc0 [-], d/da [s**2].
    """
    zeros = np.zeros_like(time_s)  # [-]
    ones = np.ones_like(time_s)  # [-]
    sens_c_a = np.column_stack((time_s, ones, zeros))  # [s], [-], [s**2]
    sens_c_b = np.column_stack((zeros, ones, time_s**2))  # [s], [-], [s**2]
    return np.vstack((sens_c_a, sens_c_b))


def _observations():
    """Return synthetic observations of ``two_state_model``.

    Returns
    -------
    numpy.ndarray
        Shape ``(5, 2)``, model at ``TRUE_PARAMS`` plus
        ``MEASUREMENT_ERRORS_MOL_L``, columns (c_A, c_B) [mol/L].
    """
    return two_state_model(TRUE_PARAMS, TIME_S) + MEASUREMENT_ERRORS_MOL_L


def _design_blocks():
    """Return per-sample regressor blocks of the linear model.

    Returns
    -------
    list of numpy.ndarray
        One ``(2, 3)`` block per time ``t``: rows ``[t, 1, 0]`` for c_A and
        ``[0, 1, t**2]`` for c_B; columns have units [s], [-], [s**2].
    """
    return [np.array([[time, 1.0, 0.0], [0.0, 1.0, time**2]])
            for time in TIME_S]


def _closed_form_gls(precision):
    """Solve the GLS normal equations for the fixture.

    Parameters
    ----------
    precision : numpy.ndarray
        Measurement precision, shape ``(2, 2)`` in (c_A, c_B) order
        [(L/mol)**2].

    Returns
    -------
    params : numpy.ndarray
        GLS estimate ``(X^T P X)^-1 X^T P y``, shape ``(3,)``
        [mol/L/s], [mol/L], [mol/L/s**2].
    covariance : numpy.ndarray
        ``s2 * (X^T P X)^-1`` with ``s2 = r^T P r / (N - p)``, shape
        ``(3, 3)``; entry (i, j) has the product of parameter units.
    normal_matrix : numpy.ndarray
        ``X^T P X``, shape ``(3, 3)``; entry (i, j) has units
        [(L/mol)**2] times the regressor units of parameters i and j.
    """
    observations = _observations()  # [mol/L]
    blocks = _design_blocks()
    normal_matrix = sum(block.T @ precision @ block for block in blocks)
    rhs = sum(block.T @ precision @ obs
              for block, obs in zip(blocks, observations))
    params = np.linalg.solve(normal_matrix, rhs)
    residuals = two_state_model(params, TIME_S) - observations  # [mol/L]
    # [-]: precision-weighted sum of squared residuals.
    weighted_ssr = np.einsum('ki,ij,kj->', residuals, precision, residuals)
    residual_variance = weighted_ssr / (NUM_DATA - NUM_PARAMS)  # [-]
    covariance = residual_variance * np.linalg.inv(normal_matrix)
    return params, covariance, normal_matrix


def _lm_options():
    """Return LM options whose only convergence stop is the gradient norm.

    Returns
    -------
    dict
        ``eps_1`` is ``LM_GRADIENT_TOL``; ``eps_2 = 0`` disables the
        relative step-size stop, whose distance to the optimum is unbounded
        by the gradient criterion.
    """
    return {'eps_1': LM_GRADIENT_TOL, 'eps_2': 0.0}


def _assert_matches_closed_form_gls(estimate, covariance, info, precision):
    """Compare an LM fit with the closed-form GLS solution.

    Parameters
    ----------
    estimate : numpy.ndarray
        Fitted parameters, shape ``(3,)`` [mol/L/s], [mol/L], [mol/L/s**2].
    covariance : numpy.ndarray
        Estimated parameter covariance, shape ``(3, 3)``.
    info : dict
        LM output from ``optimize_fn``.
    precision : numpy.ndarray
        Precision in (c_A, c_B) order, shape ``(2, 2)`` [(L/mol)**2].

    Notes
    -----
    For a linear model the weighted gradient is ``A (theta - theta*)`` with
    ``A = X^T P X``, so the gradient stop bounds the parameter error by
    ``LM_GRADIENT_TOL / lambda_min(A)`` (about 1e-8 here). The factor 2
    covers roundoff in the evaluated gradient and in the closed-form solve
    (cond(A) is about 6e2, so both are near 1e-13). The covariance differs
    only through that offset, which changes the weighted SSR by at most
    ``LM_GRADIENT_TOL**2 / lambda_min(A)`` (relative 1e-15), and through
    inverting ``A`` (relative 1e-13); ``covariance_rtol`` bounds both with
    margin. A misweighted fit differs by more than 1e-2 in both.
    """
    expected_params, expected_covariance, normal_matrix = (
        _closed_form_gls(precision))
    assert info['stop_criterion'] == 'Small gradient'
    param_atol = (2 * LM_GRADIENT_TOL
                  / np.linalg.eigvalsh(normal_matrix)[0])  # parameter units
    covariance_rtol = 1e-10  # [-]
    np.testing.assert_allclose(estimate, expected_params, rtol=0.0,
                               atol=param_atol)
    np.testing.assert_allclose(covariance, expected_covariance,
                               rtol=covariance_rtol, atol=0.0)


def _estimator(observations, covariance, measured_ind=None):
    """Build the fixture estimator with analytic sensitivities.

    Parameters
    ----------
    observations : numpy.ndarray
        Shape ``(5, 2)`` [mol/L], columns in ``measured_ind`` order.
    covariance : numpy.ndarray
        Measurement-error covariance, shape ``(2, 2)`` [(mol/L)**2], in
        ``measured_ind`` order.
    measured_ind : list of int, optional
        Model state columns matching ``observations``.

    Returns
    -------
    ParamEstim.ParameterEstimation
        Estimator seeded at ``SEED_PARAMS``.
    """
    return ParamEstim.ParameterEstimation(
        two_state_model,
        param_seed=SEED_PARAMS,
        x_data=TIME_S,
        y_data=observations,
        measured_ind=measured_ind,
        jac_fun=two_state_jacobian,
        weight_matrix=covariance,
        name_params=['rate_mol_l_s', 'offset_mol_l', 'quadratic_mol_l_s2'],
    )


def _ldl_permutation(precision):
    """Return the Bunch-Kaufman LDL row permutation of ``precision``.

    Parameters
    ----------
    precision : numpy.ndarray
        Symmetric matrix, shape ``(n, n)``.

    Returns
    -------
    numpy.ndarray
        Integer permutation, shape ``(n,)``, from ``scipy.linalg.ldl``.
    """
    return ldl(precision)[2]


# Three-state precision L @ L.T with L = [[1, 0, 0], [3, 1, 0], [1, 2, 1]];
# det(L) = 1, so its inverse, the covariance below, is an exact integer
# matrix [(mol/L)**2]. cond(covariance) is about 5.7e2.
THREE_STATE_PRECISION = np.array(
    [[1.0, 3.0, 1.0], [3.0, 10.0, 5.0], [1.0, 5.0, 6.0]])  # [(L/mol)**2]
THREE_STATE_COVARIANCE = np.array(
    [[35.0, -13.0, 5.0], [-13.0, 5.0, -2.0], [5.0, -2.0, 1.0]])  # [(mol/L)**2]


@pytest.mark.parametrize(
    'precision, covariance',
    [
        (PRECISION_L2_MOL2, COVARIANCE_MOL2_L2),
        (THREE_STATE_PRECISION, THREE_STATE_COVARIANCE),
    ],
    ids=['two_states', 'three_states'],
)
def test_weight_root_reproduces_precision_in_measured_state_order(
        precision, covariance):
    num_states = precision.shape[0]
    assert not np.array_equal(_ldl_permutation(precision),
                              np.arange(num_states))
    observations = np.outer(TIME_S, np.arange(1, num_states + 1))  # [mol/L]

    def linear_states(params, time_s):
        """Return ``params[0] * t * j`` for states j = 1..n [mol/L]."""
        return params[0] * np.outer(time_s, np.arange(1, num_states + 1))

    estimator = ParamEstim.ParameterEstimation(
        linear_states, param_seed=[1.0], x_data=TIME_S,  # [mol/L/s]
        y_data=observations, weight_matrix=covariance)

    root = estimator.sigma_inv  # [L/mol]
    # cond(covariance) <= 5.7e2 times eps 2.2e-16 is about 1.3e-13; 1e-11
    # leaves headroom for dimension-dependent factorization constants. All
    # precision entries are nonzero, so a relative tolerance suffices.
    root_rtol = 1e-11  # [-]
    np.testing.assert_allclose(root @ root.T, precision, rtol=root_rtol,
                               atol=0.0)


def test_diagonal_variances_keep_reciprocal_standard_deviation_root():
    """Uncorrelated variances keep the established diagonal weighting."""
    std_mol_l = np.array([3.0, 0.5])  # [mol/L]
    estimator = ParamEstim.ParameterEstimation(
        two_state_model, param_seed=SEED_PARAMS, x_data=TIME_S,
        y_data=_observations(), weight_matrix=np.diag(std_mol_l**2))

    # A square root and a reciprocal are each correctly rounded, so one ulp
    # of relative error bounds the diagonal; off-diagonals stay exactly 0.
    np.testing.assert_allclose(estimator.sigma_inv, np.diag(1 / std_mol_l),
                               rtol=np.finfo(float).eps, atol=0.0)


# Correlated case whose precision LDL factorization does not pivot: the
# covariance [[1, 2], [2, 5]] / 16 has the exact inverse below (det = 1).
UNPIVOTED_COVARIANCE = np.array(
    [[1.0, 2.0], [2.0, 5.0]]) * VARIANCE_SCALE_MOL2_L2  # [(mol/L)**2]
UNPIVOTED_PRECISION = np.array(
    [[5.0, -2.0], [-2.0, 1.0]]) / VARIANCE_SCALE_MOL2_L2  # [(L/mol)**2]


@pytest.mark.parametrize(
    'precision, covariance, pivots',
    [
        (UNPIVOTED_PRECISION, UNPIVOTED_COVARIANCE, False),
        (PRECISION_L2_MOL2, COVARIANCE_MOL2_L2, True),
        (THREE_STATE_PRECISION, THREE_STATE_COVARIANCE, True),
    ],
    ids=['two_states_unpivoted', 'two_states_pivoted', 'three_states'],
)
def test_weight_root_is_lower_cholesky_factor_of_precision(
        precision, covariance, pivots):
    """``sigma_inv`` is the unique lower Cholesky factor of the precision.

    Without LDL pivoting the previous root ``L @ sqrt(D)`` already was this
    factor, so the unpivoted case pins the per-entry weighted residuals and
    Jacobian rows of established correlated fits; the pivoted cases are the
    #237 defect. The expectation factors the precision literal directly.
    """
    num_states = precision.shape[0]
    assert pivots != np.array_equal(_ldl_permutation(precision),
                                    np.arange(num_states))
    observations = np.outer(TIME_S, np.arange(1, num_states + 1))  # [mol/L]

    def linear_states(params, time_s):
        """Return ``params[0] * t * j`` for states j = 1..n [mol/L]."""
        return params[0] * np.outer(time_s, np.arange(1, num_states + 1))

    estimator = ParamEstim.ParameterEstimation(
        linear_states, param_seed=[1.0], x_data=TIME_S,  # [mol/L/s]
        y_data=observations, weight_matrix=covariance)

    expected_root = np.linalg.cholesky(precision)  # [L/mol], lower
    # Same roundoff budget as the root-product test; zeros above the
    # diagonal need an absolute floor scaled to the largest entry.
    root_rtol = 1e-11  # [-]
    root_atol = root_rtol * np.abs(expected_root).max()  # [L/mol]
    np.testing.assert_allclose(estimator.sigma_inv, expected_root,
                               rtol=root_rtol, atol=root_atol)


def test_objective_and_gradient_match_closed_form_gls_at_trial_params():
    """Objective and IPOPT gradient callbacks weight by the precision."""
    assert not np.array_equal(_ldl_permutation(PRECISION_L2_MOL2), [0, 1])
    estimator = _estimator(_observations(), COVARIANCE_MOL2_L2)
    residuals = (two_state_model(SEED_PARAMS, TIME_S)
                 - _observations())  # [mol/L]

    # 1/2 sum_k r_k^T P r_k [-], from the precision literal.
    expected_objective = 0.5 * sum(
        resid @ PRECISION_L2_MOL2 @ resid for resid in residuals)
    # Gradient of that objective, sum_k X_k^T P r_k [1/parameter unit].
    expected_gradient = sum(
        block.T @ PRECISION_L2_MOL2 @ resid
        for block, resid in zip(_design_blocks(), residuals))

    objective = estimator.get_objective(SEED_PARAMS)  # [-]
    gradient = estimator.get_gradient(SEED_PARAMS)  # [1/parameter unit]

    # Ten residual quadratic forms and their gradients are summed from O(1)
    # to O(1e3) terms; roundoff is near 1e-14 relative and the entries are
    # far from zero, so 1e-12 bounds it while a permuted precision differs
    # at O(1).
    callback_rtol = 1e-12  # [-]
    assert objective == pytest.approx(expected_objective, rel=callback_rtol)
    np.testing.assert_allclose(gradient, expected_gradient,
                               rtol=callback_rtol, atol=0.0)


def test_lm_fit_matches_closed_form_gls_estimate_and_covariance():
    estimator = _estimator(_observations(), COVARIANCE_MOL2_L2)
    estimate, covariance, info = estimator.optimize_fn(
        optim_options=_lm_options(), verbose=False)

    _assert_matches_closed_form_gls(estimate, covariance, info,
                                    PRECISION_L2_MOL2)


def test_consistent_state_permutation_preserves_estimate_and_covariance():
    """Reordering measured states, data columns and covariance is neutral.

    The reversed precision factorizes without LDL pivoting while the original
    order pivots, so the two orders exercise both factorization branches.
    """
    reversed_order = [1, 0]
    reversed_precision = PRECISION_L2_MOL2[np.ix_(reversed_order,
                                                  reversed_order)]
    assert np.array_equal(_ldl_permutation(reversed_precision), [0, 1])
    assert not np.array_equal(_ldl_permutation(PRECISION_L2_MOL2), [0, 1])

    original = _estimator(_observations(), COVARIANCE_MOL2_L2)
    reordered = _estimator(
        _observations()[:, reversed_order],
        COVARIANCE_MOL2_L2[np.ix_(reversed_order, reversed_order)],
        measured_ind=reversed_order)

    original_fit = original.optimize_fn(optim_options=_lm_options(),
                                        verbose=False)
    reordered_fit = reordered.optimize_fn(optim_options=_lm_options(),
                                          verbose=False)

    # Both orders must reach the same closed-form GLS solution, written in
    # the original (c_A, c_B) order.
    for estimate, covariance, info in (original_fit, reordered_fit):
        _assert_matches_closed_form_gls(estimate, covariance, info,
                                        PRECISION_L2_MOL2)
    # Fitted profiles keep the requested measured-state column order. Each
    # fit lies within param_atol of the GLS optimum (see
    # _assert_matches_closed_form_gls), so the profiles differ by at most
    # twice that bound times the largest regressor row sum (1 + t**2).
    _, _, normal_matrix = _closed_form_gls(PRECISION_L2_MOL2)
    param_atol = (2 * LM_GRADIENT_TOL
                  / np.linalg.eigvalsh(normal_matrix)[0])  # parameter units
    # Numeric bound over regressor magnitudes in [s], [-], [s**2].
    max_regressor_sum = max(np.abs(block).sum(axis=1).max()
                            for block in _design_blocks())
    profile_atol = 2 * param_atol * max_regressor_sum  # [mol/L]
    np.testing.assert_allclose(reordered.y_model[0],
                               original.y_model[0][:, reversed_order],
                               rtol=0.0, atol=profile_atol)


# Multivariate curve resolution (MCR) fixture: absorbance [-] of species A
# and B at two wavelengths plus a separately measured tracer [mol/L].
MCR_TRUE_RATE = np.array([0.4])  # [1/s]
MCR_TRIAL_RATE = np.array([0.3])  # [1/s]
# Pure-species absorptivities, rows (A, B), columns wavelengths; all positive
# so the fitted absorptivities stay nonnegative and MCR adds no penalty.
MCR_ABSORPTIVITY = np.array([[1.2, 0.4], [0.3, 0.9]])  # [L/mol]
MCR_SPECTRA_ERRORS = np.array([  # [-], fixed asymmetric absorbance errors
    [0.020, -0.010],
    [-0.030, 0.025],
    [0.010, -0.020],
    [0.035, 0.005],
    [-0.020, 0.030],
])
MCR_TRACER_ERRORS = np.array(
    [[0.010], [-0.020], [0.015], [0.0], [-0.010]])  # [mol/L]
# Wavelength covariance and its precision reuse the pivoting 2x2 literals,
# now in absorbance units [-]; the tracer is uncorrelated with them.
MCR_TRACER_VARIANCE = 0.0004  # [(mol/L)**2], 0.02 mol/L standard deviation


def mcr_model(params, time_s, reord_sens=False):
    """Return species and tracer concentrations for the MCR fixture.

    Parameters
    ----------
    params : numpy.ndarray
        Shape ``(1,)``: first-order A -> B rate constant [1/s].
    time_s : numpy.ndarray
        Sampling times, shape ``(n_times,)`` [s].
    reord_sens : bool, optional
        Keyword passed by ``MultipleCurveResolution``; no sensitivities are
        returned, so it is unused.

    Returns
    -------
    numpy.ndarray
        Shape ``(n_times, 3)``: columns A, B (unit initial concentration)
        and a linearly accumulating tracer ``0.3 + 0.1 t`` [mol/L].
    """
    reactant = np.exp(-params[0] * time_s)  # [mol/L]
    return np.column_stack(
        (reactant, 1.0 - reactant, 0.3 + 0.1 * time_s))


@pytest.mark.parametrize('with_tracer', [False, True],
                         ids=['spectra_only', 'spectra_and_tracer'])
def test_mcr_objective_weights_projected_spectra_by_precision(with_tracer):
    """MCR forwards correlated weights to its projected spectral residuals.

    The expected residuals use a least-squares solve, not the production
    pseudo-inverse. With the tracer, the weight matrix is block diagonal, so
    the check covers the spectra-then-tracer column order without depending
    on the sign convention of either residual block.
    """
    concentrations = mcr_model(MCR_TRUE_RATE, TIME_S)  # [mol/L]
    spectra = (concentrations[:, :2] @ MCR_ABSORPTIVITY
               + MCR_SPECTRA_ERRORS)  # [-]
    tracer = concentrations[:, [2]] + MCR_TRACER_ERRORS  # [mol/L]

    trial_conc = mcr_model(MCR_TRIAL_RATE, TIME_S)  # [mol/L]
    absorptivity, *_ = np.linalg.lstsq(trial_conc[:, :2], spectra,
                                       rcond=None)  # [L/mol]
    assert np.all(absorptivity > 0)  # MCR nonnegativity penalty is zero.
    residuals = spectra - trial_conc[:, :2] @ absorptivity  # [-]
    precision = PRECISION_L2_MOL2  # [-], absorbance precision literal
    covariance = COVARIANCE_MOL2_L2  # [-], absorbance covariance

    if with_tracer:
        tracer_residuals = trial_conc[:, [2]] - tracer  # [mol/L]
        residuals = np.hstack((residuals, tracer_residuals))
        # Rows/columns: two wavelengths [-], tracer [mol/L].
        precision = np.zeros((3, 3))
        precision[:2, :2] = PRECISION_L2_MOL2
        precision[2, 2] = 1 / MCR_TRACER_VARIANCE  # [(L/mol)**2]
        covariance = np.zeros((3, 3))
        covariance[:2, :2] = COVARIANCE_MOL2_L2
        covariance[2, 2] = MCR_TRACER_VARIANCE  # [(mol/L)**2]
        estimator = ParamEstim.MultipleCurveResolution(
            mcr_model, MCR_TRIAL_RATE,
            {'run': {'spectra': TIME_S, 'non_spectra': TIME_S}},
            {'run': {'spectra': spectra, 'non_spectra': tracer}},
            measured_ind={'spectra': [0, 1], 'non_spectra': [2]},
            weight_matrix=covariance, name_states=['A', 'B', 'tracer'])
    else:
        estimator = ParamEstim.MultipleCurveResolution(
            mcr_model, MCR_TRIAL_RATE, TIME_S, spectra,
            measured_ind=[0, 1], weight_matrix=covariance)
    assert not np.array_equal(_ldl_permutation(precision),
                              np.arange(precision.shape[0]))

    # [-]: 1/2 sum_k r_k^T P r_k over sample rows.
    expected_objective = 0.5 * np.einsum(
        'ki,ij,kj->', residuals, precision, residuals)
    # Projection and quadratic-form roundoff relative to O(1e-2) to O(1)
    # objectives is near 1e-14; 1e-10 adds headroom for the
    # pseudo-inverse/least-squares difference (cond(C) is about 2), far below
    # the O(1) error of identity or permuted weights.
    objective_rtol = 1e-10  # [-]
    assert estimator.get_objective(MCR_TRIAL_RATE) == pytest.approx(
        expected_objective, rel=objective_rtol)


def test_rejects_indefinite_covariance():
    # Symmetric and invertible, eigenvalues 3 and -1 [(mol/L)**2].
    indefinite = np.array([[1.0, 2.0], [2.0, 1.0]])  # [(mol/L)**2]
    with pytest.raises(np.linalg.LinAlgError,
                       match='symmetric positive-definite'):
        _estimator(_observations(), indefinite)


@pytest.mark.parametrize(
    'weight_matrix',
    [np.ones((2, 3)), np.array([4.0, 9.0])],  # [(mol/L)**2]
    ids=['rectangular', 'variance_vector'],
)
def test_rejects_non_square_weight_matrix(weight_matrix):
    with pytest.raises(ValueError, match='square two-dimensional'):
        _estimator(_observations(), weight_matrix)


def test_rejects_non_finite_weight_matrix():
    # An unknown variance entered as NaN [(mol/L)**2].
    weight_matrix = np.array([[0.25, 0.0], [0.0, np.nan]])  # [(mol/L)**2]
    with pytest.raises(ValueError,
                       match='weight_matrix entries must be finite'):
        _estimator(_observations(), weight_matrix)
