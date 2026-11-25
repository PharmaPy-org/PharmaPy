"""Calibration validation of the Ziegler reactor stand-in (#352, AC2-AC5).

Scope: the shared public estimation contract on the workshop's
multi-temperature Ziegler fit, the solver-backed reactor stand-in of #352
while the December demonstrators (#358 crystallization, #361 bioreactor)
are not on master. The fit, its fixtures and constants are those of
``test_records_ziegler_case`` (stage-2 ``SimulationExec`` fit of five
``Experiment`` objects, eight reformulated Arrhenius parameters, CVODES
forward sensitivities at CVode's default tolerances). This module adds,
on that fit: the derivative check of the CVODES sensitivities against
central differences of complete solves over every fitted parameter (AC2),
observations left unchanged by the analyses (AC3), the identifiability
report with ``jacobian_rtol`` established by that derivative check (the
solver ``rtol`` understates the CVODES sensitivity error) and its verdict
derived from the floor and the singular values (AC4), and the convergence
outcome and tolerances (AC5). Repeatability and record capture, reload and
rerun (AC3, AC6) are covered by ``test_records_ziegler_case``, whose two
rerun paths refit from the recorded seed and must reproduce the fit to
four units in the last place; they are not repeated here.

Stand-in status: the workshop data are of unrecorded origin (issue #478);
the fit verifies the estimation contract and is not evidence of an
empirical calibration. Noise assumption: the measurements declare no
uncertainty, so the fit uses identity weighting; both measured fields are
molar concentrations [mol/L], so the residuals share one unit and the
scaled covariance is a covariance in the parameters' units (one common,
re-estimated error variance). Assimulo lane; the module-scoped fixture fits
stage 1 and stage 2 once (a few seconds), and each derivative check adds
two five-experiment solves per parameter; under 10 s in total.
"""

import numpy as np
import pytest

# The imported Ziegler fixtures need the optional Assimulo backend; skip at
# collection without it.
pytest.importorskip('assimulo')

import test_calibration_validation_decay as validation
import test_records_ziegler_case as zcase

pytestmark = [pytest.mark.assimulo, pytest.mark.integration, pytest.mark.slow]

EPS = np.finfo(float).eps  # [-], float64 machine epsilon
NUM_PARAMS = 2 * zcase.NUM_REACTIONS  # [-], phi_1,1..4 then phi_2,1..4
# CVode relative tolerance of the fit (Assimulo default, verified by
# test_records_ziegler_case.test_model_solver_entry_matches_cvode_defaults).
SOLVER_RTOL = zcase.MODEL_SOLVER['rtol']  # [-]
# Central-difference step. The parameters are logarithms (phi_1 = ln(k /
# (1/s)), phi_2 = ln((E_a/R) / K)), so a unit scale makes the step h a
# relative change h of the rate constant or of E_a/R; with outputs accurate
# to the CVode rtol r, the reference error h**2 + r / h (Nocedal and
# Wright, Numerical Optimization, 2nd ed., 2006, Sec. 8.1) is smallest at
# h = r**(1/3) = 1e-2 (check_derivatives Notes).
PARAM_SCALE = 1.0  # [-]
REL_STEP = SOLVER_RTOL ** (1 / 3)  # [-]
# Agreement bound of the CVODES sensitivities with that reference: the
# error model h**2 + r / h with unit constants, the natural scale of the
# logarithmic parametrization (a unit change of phi is an e-fold change of
# the rate), times validation.SOLVER_CHECK_SAFETY: 2e-3 at h = 1e-2.
# Design assumption: the CVODES sensitivities, whose error CVODES controls
# relative to r, add an error of order r, negligible against r / h.
DERIVATIVE_RTOL = validation.SOLVER_CHECK_SAFETY * (
    REL_STEP ** 2 + SOLVER_RTOL / REL_STEP)  # [-]
# Row-relative accuracy of the CVODES sensitivities for the identifiability
# report, established by the derivative check rather than assumed: a row
# that agrees with the reference to DERIVATIVE_RTOL, whose own error is at
# most DERIVATIVE_RTOL by the error model, is within 2 * DERIVATIVE_RTOL =
# 4e-3 of the exact row (triangle inequality, to first order in these
# bounds). The solver rtol would understate it: CVODES controls the
# sensitivity error relative to the sensitivities only to a modest multiple
# of rtol, and an independent matrix-exponential Frechet-derivative check of
# this fit (#352 review) found row errors up to 8.3e-6, above rtol = 1e-6.
SENSITIVITY_RTOL = 2 * DERIVATIVE_RTOL  # [-]
# Stopping criteria of the fit: the LM defaults of test_records_calibration
# overridden by the notebook's eps_1 and eps_2.
# max_fun_eval [-]; eps_1, eps_2 [-] (log parameters, dimensionless
# residual gradient scale as in OPTIM_OPTIONS); tol_fun [(mol/L)**2].
TOLERANCES = dict(validation.LM_TOLERANCES, **zcase.OPTIM_OPTIONS)
# The module-scoped stage-1/stage-2 fit and record of the record test,
# fitted once more for this module.
ziegler = zcase.ziegler


def test_cvodes_sensitivities_agree_with_central_differences(ziegler):
    estimator = ziegler['simulation'].ParamInst
    accepted = np.array(estimator.params_convg, copy=True)  # [-]

    check = estimator.check_derivatives(rel_step=REL_STEP,
                                        param_scale=PARAM_SCALE,
                                        rtol=DERIVATIVE_RTOL)

    assert check.jacobian_source == 'model'
    assert check.parameter_names == tuple(estimator.name_params)
    assert len(check.parameter_names) == NUM_PARAMS
    assert check.formulas == ('central',) * NUM_PARAMS
    validation.deriv.assert_steps(check, REL_STEP,
                                  np.full(NUM_PARAMS, PARAM_SCALE))
    assert check.observed.all()  # no missing observation in these data
    assert check.agrees.all(), check.summary()
    # A sensitivity row wrong by 10 % would be flagged (about 50 times the
    # bound here).
    assert (validation.MIN_POWER_MARGIN * DERIVATIVE_RTOL
            <= validation.deriv.WRONG_FACTOR - 1)
    validation.case.assert_bits_equal(estimator.params_convg, accepted)


def test_observations_are_unchanged_by_the_analyses(ziegler):
    estimator = ziegler['simulation'].ParamInst
    experiments = ziegler['experiments']
    before = validation.snapshot(estimator, experiments)

    estimator.get_confidence_intervals(verbose=False)
    estimator.get_identifiability(jacobian_rtol=SENSITIVITY_RTOL)
    estimator.check_derivatives(rel_step=REL_STEP, param_scale=PARAM_SCALE)

    validation.assert_same_snapshot(
        validation.snapshot(estimator, experiments), before)


def test_identifiability_with_the_checked_sensitivity_accuracy(ziegler):
    estimator = ziegler['simulation'].ParamInst
    # The accuracy rests on the derivative check passing at DERIVATIVE_RTOL.
    check = estimator.check_derivatives(rel_step=REL_STEP,
                                        param_scale=PARAM_SCALE,
                                        rtol=DERIVATIVE_RTOL)
    assert check.agrees.all(), check.summary()
    report = estimator.get_identifiability(jacobian_rtol=SENSITIVITY_RTOL)

    assert report.noise_model == 'identity'
    assert report.jacobian_source == 'model'
    text = report.noise_assumption
    for fragment in ('No measurement uncertainty or weight_matrix',
                     'Every measured field is in [mol/L].',
                     f'relative jacobian_rtol = {SENSITIVITY_RTOL:.3g}'):
        assert fragment in text, (fragment, text)
    np.testing.assert_array_equal(report.covariance,
                                  estimator.get_covariance())

    # Independent route to the scaled singular values: the eigenvalues of
    # D A D, A = information matrix, D = diag(A_ii**-0.5).
    information = report.information_matrix  # [(mol/L)**2 / (p_i p_j)]
    scale = 1 / np.sqrt(np.diag(information))  # [p_i / (mol/L)]
    gram = scale[:, None] * information * scale[None, :]  # [-]
    eigenvalues = np.linalg.eigvalsh(gram)[::-1]  # [-], descending
    singular = np.sqrt(eigenvalues)  # [-]
    # The eigenvalues of this well-conditioned Gram matrix (cond ~ 1e2) and
    # the production SVD agree to about cond * eps; 1e-10 bounds it.
    np.testing.assert_allclose(report.singular_values, singular,
                               rtol=1e-10)
    # Verdict by the report's own criterion: every row has the bound
    # jacobian_rtol, so the floor is sqrt(n) * jacobian_rtol (1.1e-2); the
    # smallest singular value (about 0.19) exceeds it and passes the #532
    # test s**2 > eps * s_max**2.
    floor = np.sqrt(NUM_PARAMS) * SENSITIVITY_RTOL  # [-]
    assert report.resolution_floor == pytest.approx(floor, rel=1e-12)
    assert singular[-1] > floor
    assert singular[-1] ** 2 > EPS * singular[0] ** 2
    assert report.identifiable is True
    assert report.unresolved_directions == ()
    assert report.collinearity_index == pytest.approx(1 / singular[-1],
                                                      rel=1e-10)
    # Strongest pairwise correlation (0.936), reported but not a verdict.
    off_diagonal = np.abs(report.correlation
                          - np.eye(NUM_PARAMS))  # [-]
    assert 0 < off_diagonal.max() < 1


def test_fit_converged_with_the_notebook_tolerances(ziegler):
    _, _, info = ziegler['fit']
    assert info['success'] is True
    assert info['message'] in validation.LM_CONVERGED
    assert info['tolerances'] == TOLERANCES
    report = ziegler['simulation'].ParamInst.get_identifiability(
        jacobian_rtol=SENSITIVITY_RTOL)
    assert report.convergence['success'] is True
    assert report.summary().splitlines()[1].startswith(
        'Convergence: converged')
