"""One objective for multivariate curve resolution (MCR), issue #240.

``MultipleCurveResolution`` minimizes ``f = 1/2 r_data @ r_data +
mult_penalty * sum(max(-eps, 0)**2)`` over the variable-projection
absorptivities ``eps``. Least-squares solvers see the penalty as residual
rows appended after the data residuals, so ``f = 1/2 r @ r`` for the
augmented vector ``r``; the analytic scalar gradient differentiates ``f``
itself; local analysis resolves one set of absorptivities per experiment;
and fit statistics use the data residuals only.

Fixtures reuse the first-order A -> B reaction of
``test_paramestim_mcr_observation_masks`` (absorbance [-] at two wavelengths
and a staggered tracer [mol/L]) and add spectra built from absorptivities
with a negative entry, which forces a negative least-squares absorptivity and
an active penalty. Further synthetic fixtures cover rank-deficient
concentration blocks of constant rank, three species at four wavelengths
with two optimized parameters and one fixed one, SciPy least-squares fits,
the cached evaluation behind the analytic gradient and the deprecated
``update_self`` keyword. Expected values come from ``numpy.linalg.lstsq``
projections, ``numpy.linalg.inv`` of the observed covariance blocks and
central differences written here. Runs in a few seconds.
"""

import numpy as np
import pytest

from PharmaPy.LevMarq import levenberg_marquardt
from PharmaPy.ParamEstim import MultipleCurveResolution
from PharmaPy.StatsModule import StatisticsClass
import test_paramestim_mcr_observation_masks as case


pytestmark = pytest.mark.unit

TIME_S = case.SCHEDULES['first'][0]  # [s], five samples of one experiment
# Pure-species absorptivities, rows (A, B), columns wavelengths. The
# negative entry of A is unphysical on purpose: the least-squares
# absorptivity of A at wavelength 2 stays negative near the fitted rate, so
# the nonnegativity penalty is active.
NEGATIVE_ABSORPTIVITY = np.array([[1.2, -0.4], [0.3, 0.9]])  # [L/mol]
# Local-analysis case: each experiment has its own pure spectra (as for
# another optical path or instrument) with one negative entry, species B at
# the first wavelength. Its flat penalty-row index in the first experiment
# is 2 in (experiment, species, wavelength) order but 4 in (species,
# experiment, wavelength) order, so the row order is pinned.
FIRST_ABSORPTIVITY = np.array([[1.2, 0.4], [-0.25, 0.9]])  # [L/mol]
SECOND_ABSORPTIVITY = np.array([[0.9, 0.5], [-0.3, 1.4]])  # [L/mol]
# Penalty weight [(mol/L)**2] with identity weights (objective [-] per
# absorptivity [L/mol] squared). Not 1, so using mult_penalty instead of
# its square root in the rows sqrt(2 * mult_penalty) * max(-eps, 0) shows.
PENALTY_WEIGHT = 2.5  # [(mol/L)**2]
# Weights of the LM penalty sweep, from no penalty to a dominant one.
PENALTY_SWEEP = (0.0, 1.0, 100.0)  # [(mol/L)**2]
OTHER_RATE = np.array([0.45])  # [1/s], a second evaluation point
# Central differences with a 1e-5 relative step (h = 3e-6 1/s): truncation
# h**2 |f'''| / (6 |f'|) and roundoff eps |f| / (h |f'|) stay below 1e-10
# here (|f''' / f'| below 1 s**2 and |f / f'| below 1 1/s on these
# fixtures), so 1e-6 bounds both with margin; the old analytic scalar
# gradient (a residual-sensitivity row) is off by O(1).
FD_RELATIVE_STEP = 1e-5  # [-]
DERIVATIVE_RTOL = 1e-6  # [-]
# The finite-difference scalar gradient is a forward difference with step h
# = dx_finitediff(k); its leading error is (h / 2) f''. A bound of h |f''|
# leaves a factor 2 for the higher-order terms.
FORWARD_BIAS_FACTOR = 2.0  # [-]
# Projection and quadratic-form roundoff relative to O(1e-3) to O(1e2)
# objectives is near 1e-13; 1e-10 adds headroom for the pseudo-inverse
# versus least-squares difference (cond(C) below 1e2 here).
ROUNDOFF_RTOL = 1e-10  # [-]
# Relative rate offset probing a fitted minimum, as in
# test_paramestim_mcr_observation_masks: far above LM's 1e-8 stops.
MINIMUM_PROBE = 1e-3  # [-]
# One penalty row per resolved absorptivity: species x wavelengths for
# global analysis, times the number of experiments for local analysis.
NUM_SPECIES = case.ABSORPTIVITY.shape[0]  # [-], A and B
NUM_WAVELENGTHS = case.ABSORPTIVITY.shape[1]  # [-]
LOCAL_PENALTY_ROWS = (len(case.SCHEDULES) * NUM_SPECIES
                      * NUM_WAVELENGTHS)  # [-]
MODELS = [case.model_with_sensitivities, case.model_states_only]
MODEL_IDS = ['model_sensitivities', 'finite_difference']


def _negative_spectra():
    """Return absorbance with a negative pure-component absorptivity.

    Returns
    -------
    numpy.ndarray
        Shape ``(5, 2)`` [-]: ``NEGATIVE_ABSORPTIVITY`` at the true rate
        plus the fixed errors of the first experiment of ``case``.
    """
    concentrations = case._states(case.TRUE_RATE, TIME_S)[:, :2]  # [mol/L]
    return (concentrations @ NEGATIVE_ABSORPTIVITY
            + case.SPECTRA_ERRORS['first'])


def _negative_estimator(model, mult_penalty=PENALTY_WEIGHT):
    """Build a one-experiment, spectra-only MCR estimator with a penalty.

    Parameters
    ----------
    model : callable
        ``case.model_with_sensitivities`` or ``case.model_states_only``.
    mult_penalty : float, optional
        Penalty weight [(mol/L)**2].

    Returns
    -------
    MultipleCurveResolution
        Estimator seeded at ``case.TRIAL_RATE`` with identity weights.
    """
    return MultipleCurveResolution(
        model, case.TRIAL_RATE, TIME_S, _negative_spectra(),
        measured_ind=[0, 1], mult_penalty=mult_penalty)


def _negative_reference(rate, mult_penalty=PENALTY_WEIGHT):
    """Resolve the negative-absorptivity case independently.

    Parameters
    ----------
    rate : numpy.ndarray
        Rate k, shape ``(1,)`` [1/s].
    mult_penalty : float, optional
        Penalty weight [(mol/L)**2].

    Returns
    -------
    residuals : numpy.ndarray
        Data minus projection, shape ``(5, 2)`` [-].
    absorptivity : numpy.ndarray
        Least-squares absorptivities, shape ``(2, 2)`` [L/mol].
    objective : float
        ``1/2 sum(residuals**2) + mult_penalty * sum(min(eps, 0)**2)`` [-].
    """
    spectra = _negative_spectra()  # [-]
    concentrations = case._states(rate, TIME_S)[:, :2]  # [mol/L]
    absorptivity, *_ = np.linalg.lstsq(concentrations, spectra, rcond=None)
    residuals = spectra - concentrations @ absorptivity  # [-]
    objective = (0.5 * np.sum(residuals**2)
                 + mult_penalty * np.sum(np.minimum(absorptivity, 0)**2))
    return residuals, absorptivity, objective


def _local_data():
    """Return the observations of the two-experiment local-analysis case.

    Returns
    -------
    dict
        Experiment name -> (absorbance ``(n_times, 2)`` [-], tracer
        ``(n_tracer,)`` [mol/L]); ``'first'`` uses ``FIRST_ABSORPTIVITY``,
        ``'second'`` uses ``SECOND_ABSORPTIVITY``.
    """
    pure = {'first': FIRST_ABSORPTIVITY, 'second': SECOND_ABSORPTIVITY}
    data = {}
    for name, (time_s, tracer_rows) in case.SCHEDULES.items():
        states = case._states(case.TRUE_RATE, time_s)  # [mol/L]
        spectra = states[:, :2] @ pure[name] + case.SPECTRA_ERRORS[name]
        tracer = states[tracer_rows, 2] + case.TRACER_ERRORS[name]
        data[name] = (spectra, tracer)
    return data


def _local_estimator(model):
    """Build the staggered two-experiment estimator with local analysis.

    Parameters
    ----------
    model : callable
        ``case.model_with_sensitivities`` or ``case.model_states_only``.

    Returns
    -------
    MultipleCurveResolution
        Estimator with ``global_analysis=False``, correlated weights and
        ``PENALTY_WEIGHT``, seeded at ``case.TRIAL_RATE``.
    """
    x_data = {name: {'spectra': time_s, 'non_spectra': time_s[rows]}
              for name, (time_s, rows) in case.SCHEDULES.items()}
    y_data = {name: {'spectra': spectra, 'non_spectra': tracer}
              for name, (spectra, tracer) in _local_data().items()}
    return MultipleCurveResolution(
        model, case.TRIAL_RATE, x_data, y_data,
        measured_ind={'spectra': [0, 1], 'non_spectra': [2]},
        weight_matrix=case.COVARIANCES['correlated'],
        mult_penalty=PENALTY_WEIGHT, global_analysis=False,
        name_states=['A', 'B', 'tracer'])


def _local_reference(rate):
    """Resolve each experiment of the local case independently.

    Parameters
    ----------
    rate : numpy.ndarray
        Rate k, shape ``(1,)`` [1/s].

    Returns
    -------
    absorptivities : list of numpy.ndarray
        Least-squares absorptivities per experiment, shape ``(2, 2)``
        [L/mol], in ``case.SCHEDULES`` order.
    predictions : list of numpy.ndarray
        Projected spectra per experiment, shape ``(n_times, 2)`` [-].
    objective : float
        ``1/2 sum e_o^T inv(S_oo) e_o`` over the observed entries ``o`` of
        each row (spectral data minus projection [-], tracer data minus
        model [mol/L]) plus ``PENALTY_WEIGHT * sum(min(eps, 0)**2)`` [-].
    """
    covariance = case.COVARIANCES['correlated']
    absorptivities = []
    predictions = []
    objective = 0.0  # [-]
    for name, (spectra, tracer) in _local_data().items():
        time_s, tracer_rows = case.SCHEDULES[name]
        states = case._states(rate, time_s)  # [mol/L]
        absorptivity, *_ = np.linalg.lstsq(states[:, :2], spectra,
                                           rcond=None)  # [L/mol]
        prediction = states[:, :2] @ absorptivity  # [-]
        for row in range(time_s.size):
            columns = [0, 1]
            values = list(spectra[row] - prediction[row])
            if row in tracer_rows:
                columns.append(2)
                values.append(tracer[tracer_rows.index(row)]
                              - states[row, 2])
            values = np.array(values)
            precision = np.linalg.inv(covariance[np.ix_(columns, columns)])
            objective += 0.5 * values @ precision @ values
        objective += PENALTY_WEIGHT * np.sum(np.minimum(absorptivity, 0)**2)
        absorptivities.append(absorptivity)
        predictions.append(prediction)
    return absorptivities, predictions, objective


def _central_difference(function, rate):
    """Differentiate ``function`` with a central difference.

    Parameters
    ----------
    function : callable
        Maps a rate, shape ``(1,)`` [1/s], to a float or array.
    rate : numpy.ndarray
        Rate k, shape ``(1,)`` [1/s].

    Returns
    -------
    float or numpy.ndarray
        Derivative with respect to k, in the output unit times [s].
    """
    step = FD_RELATIVE_STEP * rate  # [1/s]
    return (function(rate + step) - function(rate - step)) / (2 * step[0])


def _fresh_objective(build, *args):
    """Return ``get_objective`` of a fresh estimator for each evaluation.

    Parameters
    ----------
    build : callable
        Estimator factory, called as ``build(*args)``.
    *args
        Arguments of ``build``.

    Returns
    -------
    callable
        ``f(rate, out_array=False)`` evaluated on a new estimator, so the
        estimator under test keeps its own evaluation state.
    """
    def objective(rate, out_array=False):
        """Evaluate a new estimator's objective or residual vector.

        Parameters
        ----------
        rate : numpy.ndarray
            Rate k, shape ``(1,)`` [1/s].
        out_array : bool, optional
            Return the residual vector instead of the objective.

        Returns
        -------
        float or numpy.ndarray
            ``get_objective(rate, out_array)`` [-].
        """
        return build(*args).get_objective(rate, out_array)
    return objective


GRADIENT_CASES = {
    # Staggered tracer and correlated weights; positive absorptivities.
    'no_penalty': lambda model: case._estimator(model, 'correlated'),
    # Spectra only, identity weights; active absorptivity penalty.
    'active_penalty': _negative_estimator,
}


@pytest.mark.parametrize('case_name', list(GRADIENT_CASES))
def test_analytic_scalar_gradient_is_objective_derivative(case_name):
    build = GRADIENT_CASES[case_name]
    estimator = build(case.model_with_sensitivities)
    estimator.get_objective(case.TRIAL_RATE)

    gradient = estimator.get_gradient(case.TRIAL_RATE)  # [s]

    expected = _central_difference(
        _fresh_objective(build, case.model_with_sensitivities),
        case.TRIAL_RATE)
    assert gradient.shape == (1,)
    assert gradient[0] == pytest.approx(expected, rel=DERIVATIVE_RTOL)


def test_penalty_case_keeps_its_sign_pattern_within_the_step():
    """The central differences above do not straddle the penalty kink."""
    step = FD_RELATIVE_STEP * case.TRIAL_RATE  # [1/s]
    signs = [np.sign(_negative_reference(rate)[1])
             for rate in (case.TRIAL_RATE - step, case.TRIAL_RATE + step)]
    np.testing.assert_array_equal(signs[0], signs[1])
    assert (signs[0] < 0).sum() == 1


@pytest.mark.parametrize('case_name', list(GRADIENT_CASES))
def test_finite_difference_scalar_gradient_includes_penalty(case_name):
    """Forward differences of f differ from f' by about (h / 2) f''."""
    build = GRADIENT_CASES[case_name]
    estimator = build(case.model_states_only)
    estimator.get_objective(case.TRIAL_RATE)

    gradient = estimator.get_gradient(case.TRIAL_RATE)  # [s]

    objective = _fresh_objective(build, case.model_states_only)
    derivative = _central_difference(objective, case.TRIAL_RATE)  # [s]
    step = FD_RELATIVE_STEP * case.TRIAL_RATE  # [1/s]
    curvature = (objective(case.TRIAL_RATE + step)
                 - 2 * objective(case.TRIAL_RATE)
                 + objective(case.TRIAL_RATE - step)) / step[0]**2  # [s**2]
    forward_step = estimator.dx_fd(case.TRIAL_RATE)[0]  # [1/s]
    assert abs(gradient[0] - derivative) <= (
        FORWARD_BIAS_FACTOR * forward_step / 2 * abs(curvature))


@pytest.mark.parametrize('model', MODELS, ids=MODEL_IDS)
def test_residual_vector_appends_penalty_rows(model):
    estimator = _negative_estimator(model)

    residual = estimator.get_objective(case.TRIAL_RATE, out_array=True)
    objective = estimator.get_objective(case.TRIAL_RATE)  # [-]

    expected_data, absorptivity, expected_objective = _negative_reference(
        case.TRIAL_RATE)
    num_data = expected_data.size  # [-], 5 times x 2 wavelengths
    assert residual.shape == (num_data + absorptivity.size,)
    # Column-major data residuals, then one row per absorptivity in
    # (species, wavelength) order.
    np.testing.assert_allclose(residual[:num_data], expected_data.T.ravel(),
                               rtol=ROUNDOFF_RTOL, atol=0.0)
    penalty = residual[num_data:]  # [-]
    assert np.all(penalty >= 0)
    np.testing.assert_allclose(
        0.5 * penalty**2,
        PENALTY_WEIGHT * np.minimum(absorptivity, 0).ravel()**2,
        rtol=ROUNDOFF_RTOL, atol=0.0)
    np.testing.assert_array_equal(penalty[absorptivity.ravel() > 0], 0.0)
    assert objective == pytest.approx(expected_objective, rel=ROUNDOFF_RTOL)
    assert 0.5 * residual @ residual == pytest.approx(objective,
                                                      rel=ROUNDOFF_RTOL)
    np.testing.assert_array_equal(estimator.residuals, residual[:num_data])

    jacobian = estimator.get_gradient(case.TRIAL_RATE, out_array=True)[0]
    expected_jacobian = _central_difference(
        lambda rate: _fresh_objective(_negative_estimator, model)(rate, True),
        case.TRIAL_RATE)  # [s]
    if model is case.model_with_sensitivities:
        np.testing.assert_allclose(
            jacobian, expected_jacobian, rtol=0.0,
            atol=DERIVATIVE_RTOL * np.abs(expected_jacobian).max())
    else:
        np.testing.assert_allclose(
            jacobian, expected_jacobian, rtol=0.0,
            atol=case.FORWARD_DIFFERENCE_RTOL
            * np.abs(expected_jacobian).max())
    penalty_jacobian = jacobian[num_data:]  # [s]
    active = absorptivity.ravel() < 0
    assert np.all(penalty_jacobian[active] != 0)
    np.testing.assert_array_equal(penalty_jacobian[~active], 0.0)


@pytest.mark.parametrize('model', MODELS, ids=MODEL_IDS)
def test_penalty_moves_lm_fit_toward_nonnegative_absorptivities(model):
    """A heavier penalty raises the most negative fitted absorptivity."""
    lowest = []
    for mult_penalty in PENALTY_SWEEP:
        estimator = _negative_estimator(model, mult_penalty)
        fitted, _, info = estimator.optimize_fn(verbose=False)

        _, absorptivity, at_fit = _negative_reference(fitted, mult_penalty)
        assert estimator.get_objective(fitted) == pytest.approx(
            at_fit, rel=ROUNDOFF_RTOL)
        for factor in (1 - MINIMUM_PROBE, 1 + MINIMUM_PROBE):
            assert _negative_reference(fitted * factor,
                                       mult_penalty)[2] > at_fit
        # The solver minimized 1/2 |[fun; fun_penalty]|**2 = f.
        assert info['fun'].shape == (TIME_S.size * 2,)
        assert 0.5 * (info['fun'] @ info['fun'] + info['fun_penalty']
                      @ info['fun_penalty']) == pytest.approx(
            at_fit, rel=ROUNDOFF_RTOL)
        np.testing.assert_allclose(estimator.epsilon_mcr, absorptivity,
                                   rtol=ROUNDOFF_RTOL, atol=0.0)
        lowest.append(absorptivity.min())  # [L/mol]

    assert lowest[0] < lowest[1] < lowest[2]


def test_statistics_use_data_residuals_with_active_penalty():
    estimator = _negative_estimator(case.model_with_sensitivities)
    fitted, covariance, info = estimator.optimize_fn(verbose=False)
    assert np.any(info['fun_penalty'] > 0)

    data_residual = _negative_reference(fitted)[0].T.ravel()  # [-]
    data_jacobian = _central_difference(
        lambda rate: _negative_reference(rate)[0].T.ravel(), fitted)  # [-*s]
    num_data = data_residual.size  # [-]
    dof = num_data - 1  # [-], one optimized rate
    np.testing.assert_allclose(info['fun'], data_residual,
                               rtol=ROUNDOFF_RTOL, atol=0.0)
    expected = (data_residual @ data_residual / dof
                / (data_jacobian @ data_jacobian))  # [1/s**2]
    assert covariance.shape == (1, 1)
    assert covariance[0, 0] == pytest.approx(expected, rel=DERIVATIVE_RTOL)

    statistics = StatisticsClass(estimator)
    assert statistics.dof == dof
    assert statistics.var_fun == pytest.approx(
        data_residual @ data_residual / dof, rel=ROUNDOFF_RTOL)


@pytest.mark.parametrize('model', MODELS, ids=MODEL_IDS)
def test_zero_penalty_rows_leave_lm_iterates_unchanged(model):
    """With positive absorptivities the fit equals a data-only LM fit.

    The reference runs the package's Levenberg-Marquardt solver on the data
    rows alone, the residual contract before #240. The penalty rows are
    exact zeros appended after the data rows; they can only reassociate the
    BLAS sums, so the iterates, residuals and Jacobian are unchanged to
    roundoff and the iteration count is identical.
    """
    estimator = case._estimator(model, 'correlated')
    fitted, covariance, info = estimator.optimize_fn(verbose=False)

    num_data = case._unobserved_tracer_entries().size  # [-]
    reference = case._estimator(model, 'correlated')
    expected, _, expected_info = levenberg_marquardt(
        case.TRIAL_RATE,
        lambda rate: reference.get_objective(rate, True)[:num_data],
        lambda rate: reference.get_gradient(rate, True)[:, :num_data],
        full_output=True)

    np.testing.assert_array_equal(info['fun_penalty'], 0.0)
    np.testing.assert_array_equal(info['jac_penalty'], 0.0)
    # Reassociated sums differ by a few ulps (1e-16 relative) per
    # operation; the one-parameter normal equations are well conditioned
    # (condition number 1), so 1e-10 bounds the propagated difference while
    # a changed data row would differ at O(1e-3).
    np.testing.assert_allclose(fitted, expected, rtol=ROUNDOFF_RTOL, atol=0.0)
    np.testing.assert_allclose(info['fun'], expected_info['fun'],
                               rtol=ROUNDOFF_RTOL, atol=0.0)
    np.testing.assert_allclose(info['jac'], expected_info['jac'],
                               rtol=ROUNDOFF_RTOL, atol=0.0)
    assert info['num_iter'] == expected_info['num_iter']
    np.testing.assert_array_equal(estimator.residuals, info['fun'])
    # The penalty rows do not count as data: the degrees of freedom are the
    # estimator's observed-entry count minus the one optimized rate.
    dof = estimator.num_data_total - 1  # [-]
    jac = expected_info['jac']  # [weighted residual * s]
    np.testing.assert_allclose(
        covariance, expected_info['fun'] @ expected_info['fun'] / dof
        * np.linalg.inv(jac @ jac.T), rtol=ROUNDOFF_RTOL, atol=0.0)


@pytest.mark.parametrize('model', MODELS, ids=MODEL_IDS)
def test_local_analysis_resolves_each_experiment(model):
    estimator = _local_estimator(model)

    objective = estimator.get_objective(case.TRIAL_RATE)  # [-]
    residual = estimator.get_objective(case.TRIAL_RATE, out_array=True)

    absorptivities, predictions, expected = _local_reference(case.TRIAL_RATE)
    assert objective == pytest.approx(expected, rel=ROUNDOFF_RTOL)
    assert 0.5 * residual @ residual == pytest.approx(expected,
                                                      rel=ROUNDOFF_RTOL)
    # (experiment, species, wavelength), experiments in x_data order.
    assert estimator.epsilon_mcr.shape == (2, 2, 2)
    np.testing.assert_allclose(estimator.epsilon_mcr, absorptivities,
                               rtol=ROUNDOFF_RTOL, atol=0.0)
    for run, prediction in zip(estimator.y_runs, predictions):
        np.testing.assert_allclose(run, prediction, rtol=ROUNDOFF_RTOL,
                                   atol=0.0)

    unobserved = case._unobserved_tracer_entries()
    num_data = unobserved.size  # [-], same data layout as global analysis
    assert residual.shape == (num_data + LOCAL_PENALTY_ROWS,)
    np.testing.assert_array_equal(residual[:num_data][unobserved], 0.0)
    # One negative absorptivity per experiment; penalty rows in C order
    # (experiment, species, wavelength).
    assert np.all(np.sum(np.less(absorptivities, 0), axis=(1, 2)) == 1)
    expected_penalty = np.sqrt(2 * PENALTY_WEIGHT) * np.maximum(
        -np.stack(absorptivities), 0).ravel()  # [-]
    np.testing.assert_allclose(residual[num_data:], expected_penalty,
                               rtol=ROUNDOFF_RTOL, atol=0.0)


def test_local_analysis_derivatives_match_central_differences():
    estimator = _local_estimator(case.model_with_sensitivities)
    estimator.get_objective(case.TRIAL_RATE)

    jacobian = estimator.get_gradient(case.TRIAL_RATE, out_array=True)[0]
    gradient = estimator.get_gradient(case.TRIAL_RATE)  # [s]

    objective = _fresh_objective(_local_estimator,
                                 case.model_with_sensitivities)
    expected_jacobian = _central_difference(
        lambda rate: objective(rate, True), case.TRIAL_RATE)
    np.testing.assert_allclose(
        jacobian, expected_jacobian, rtol=0.0,
        atol=DERIVATIVE_RTOL * np.abs(expected_jacobian).max())
    assert np.any(jacobian[case._unobserved_tracer_entries().size:] != 0)
    assert gradient[0] == pytest.approx(
        _central_difference(objective, case.TRIAL_RATE),
        rel=DERIVATIVE_RTOL)


@pytest.mark.parametrize('model', MODELS, ids=MODEL_IDS)
def test_local_analysis_lm_fit_minimizes_independent_objective(model):
    estimator = _local_estimator(model)

    fitted, covariance, info = estimator.optimize_fn(verbose=False)

    assert info['num_iter'] > 0
    at_fit = _local_reference(fitted)[2]  # [-]
    for factor in (1 - MINIMUM_PROBE, 1 + MINIMUM_PROBE):
        assert _local_reference(fitted * factor)[2] > at_fit
    assert info['fun'].shape == (case._unobserved_tracer_entries().size,)
    absorptivities = np.stack(_local_reference(fitted)[0])  # [L/mol]
    assert np.any(absorptivities[0] < 0) and np.any(absorptivities[1] < 0)
    np.testing.assert_allclose(
        info['fun_penalty'],
        np.sqrt(2 * PENALTY_WEIGHT) * np.maximum(-absorptivities, 0).ravel(),
        rtol=ROUNDOFF_RTOL, atol=0.0)
    assert info['fun_penalty'].shape == (LOCAL_PENALTY_ROWS,)
    assert covariance.shape == (1, 1) and covariance[0, 0] > 0


def test_result_assembly_accepts_set_self_and_keeps_reporting_state():
    """``assemble_solver_info`` evaluates without replacing fit results."""
    estimator = _negative_estimator(case.model_with_sensitivities)
    fitted, _, _ = estimator.optimize_fn(verbose=False)
    residuals = estimator.residuals.copy()  # weighted [-]
    absorptivity = estimator.epsilon_mcr.copy()  # [L/mol]
    num_history = len(estimator.params_iter)  # [-]

    info = estimator.assemble_solver_info(OTHER_RATE)

    expected = _negative_estimator(case.model_with_sensitivities)
    np.testing.assert_array_equal(
        info['fun'], expected.get_objective(OTHER_RATE, out_array=True))
    np.testing.assert_array_equal(
        info['jac'], expected.get_gradient(OTHER_RATE, out_array=True))
    np.testing.assert_array_equal(estimator.residuals, residuals)
    np.testing.assert_array_equal(estimator.epsilon_mcr, absorptivity)
    assert len(estimator.params_iter) == num_history
    np.testing.assert_array_equal(estimator.params_convg, fitted)


@pytest.mark.parametrize('out_array', [False, True])
def test_model_sensitivity_gradient_reevaluates_another_point(out_array):
    """Stored sensitivities of another rate are not reused."""
    estimator = _negative_estimator(case.model_with_sensitivities)
    estimator.get_objective(case.TRIAL_RATE)

    derivative = estimator.get_gradient(OTHER_RATE, out_array=out_array)

    expected = _central_difference(
        lambda rate: _fresh_objective(
            _negative_estimator, case.model_with_sensitivities)(
                rate, out_array),
        OTHER_RATE)
    np.testing.assert_allclose(
        np.ravel(derivative), np.ravel(expected), rtol=0.0,
        atol=DERIVATIVE_RTOL * np.abs(expected).max())


@pytest.mark.parametrize('mult_penalty', [-1.0, np.nan, np.inf])
def test_penalty_weight_must_be_finite_and_nonnegative(mult_penalty):
    with pytest.raises(ValueError, match='mult_penalty must be finite and '
                       'nonnegative'):
        _negative_estimator(case.model_states_only, mult_penalty)


@pytest.mark.parametrize('mult_penalty', ['1', True, None])
def test_penalty_weight_must_be_real(mult_penalty):
    with pytest.raises(TypeError, match='mult_penalty must be a real number'):
        _negative_estimator(case.model_states_only, mult_penalty)


def test_zero_penalty_weight_disables_penalty():
    estimator = _negative_estimator(case.model_states_only, 0)

    residual = estimator.get_objective(case.TRIAL_RATE, out_array=True)

    expected_data, absorptivity, expected = _negative_reference(
        case.TRIAL_RATE, 0.0)
    assert np.any(absorptivity < 0)
    assert residual.shape == (expected_data.size + absorptivity.size,)
    np.testing.assert_array_equal(residual[expected_data.size:], 0.0)
    assert estimator.get_objective(case.TRIAL_RATE) == pytest.approx(
        expected, rel=ROUNDOFF_RTOL)


# ------------------------------------------- rank-deficient concentrations
# Synthetic contract fixtures from review of #240: the third column of C is a
# parameter-dependent combination of the first two, so C has constant rank
# 2 for every parameter value and the pseudoinverse is differentiable.
RAMP_TIMES = (np.arange(5.0), np.arange(5.0) + 0.5)  # [s], two experiments
RAMP_SLOPE = np.array([0.3])  # [-], combination coefficient p
DECAY_TIMES = (np.linspace(0.0, 2.0, 5), np.linspace(0.25, 2.5, 6))  # [s]
DECAY_WEIGHT = np.array([0.5])  # [-], combination coefficient k
RANK = 2  # [-], rank of every concentration block


def ramp_model(params, time_s, reord_sens=False):
    """Return ``C = [u, 1, p u + 1]`` with ``u = t + 1`` and dC/dp.

    Parameters
    ----------
    params : numpy.ndarray
        Coefficient p, shape ``(1,)`` [-].
    time_s : numpy.ndarray
        Times, shape ``(n_times,)`` [s].
    reord_sens : bool, optional
        Keyword passed by ``MultipleCurveResolution``; unused.

    Returns
    -------
    states : numpy.ndarray
        Shape ``(n_times, 3)`` [mol/L], rank 2.
    sensitivities : numpy.ndarray
        Shape ``(1, n_times, 3)`` [mol/L].
    """
    ramp = time_s + 1.0  # [mol/L]
    ones = np.ones_like(ramp)  # [mol/L]
    states = np.column_stack((ramp, ones, params[0] * ramp + 1.0))
    sens = np.column_stack((0 * ramp, 0 * ramp, ramp))  # [mol/L]
    return states, sens[np.newaxis]


def ramp_spectra(time_s):
    """Return the one-wavelength absorbance of the ramp case.

    Parameters
    ----------
    time_s : numpy.ndarray
        Times, shape ``(n_times,)`` [s].

    Returns
    -------
    numpy.ndarray
        ``2 u - 1`` with ``u = t + 1``, shape ``(n_times, 1)`` [-].
    """
    return (2 * (time_s + 1.0) - 1.0)[:, np.newaxis]


def decay_model(params, time_s, reord_sens=False):
    """Return ``C = [a, b, k a + b]`` with ``a = exp(-t)``, ``b = 1 - a``.

    Parameters
    ----------
    params : numpy.ndarray
        Coefficient k, shape ``(1,)`` [-].
    time_s : numpy.ndarray
        Times, shape ``(n_times,)`` [s].
    reord_sens : bool, optional
        Keyword passed by ``MultipleCurveResolution``; unused.

    Returns
    -------
    states : numpy.ndarray
        Shape ``(n_times, 3)`` [mol/L], rank 2.
    sensitivities : numpy.ndarray
        Shape ``(1, n_times, 3)`` [mol/L].
    """
    decayed = np.exp(-time_s)  # [mol/L]
    formed = 1.0 - decayed  # [mol/L]
    states = np.column_stack((decayed, formed, params[0] * decayed + formed))
    sens = np.column_stack((0 * decayed, 0 * decayed, decayed))  # [mol/L]
    return states, sens[np.newaxis]


def decay_spectra(time_s):
    """Return the one-wavelength absorbance of the decay case.

    Parameters
    ----------
    time_s : numpy.ndarray
        Times, shape ``(n_times,)`` [s].

    Returns
    -------
    numpy.ndarray
        ``a + 0.1 b`` with ``a = exp(-t)``, ``b = 1 - a``, shape
        ``(n_times, 1)`` [-].
    """
    decayed = np.exp(-time_s)  # [mol/L]
    return (decayed + 0.1 * (1.0 - decayed))[:, np.newaxis]


# Rank-deficient case with nonzero data residuals and Jacobian: C = [e, 1,
# 2 e - 0.5] with e = exp(-k t) has rank 2, the spectra lie outside its
# column space and dC/dk = [-t e, 0, -2 t e] does too. Each experiment has
# its own grid, so the local blocks differ.
OFFSET_TIMES = (np.linspace(0.2, 4.0, 7), np.linspace(0.5, 5.0, 6))  # [s]
OFFSET_RATE = np.array([0.4])  # [1/s]
OFFSET_COMBINATION = (2.0, -0.5)  # [-], [mol/L]: C3 = 2 C1 - 0.5
# Data rows and Jacobian entries here are O(1e-2) to O(1e-1); 1e-3 is far
# above roundoff (1e-15) yet below them, so the check is not vacuous.
NONZERO_FLOOR = 1e-3  # [-]


def offset_model(params, time_s, reord_sens=False):
    """Return ``C = [e, 1, 2 e - 0.5]`` with ``e = exp(-k t)`` and dC/dk.

    Parameters
    ----------
    params : numpy.ndarray
        Rate k, shape ``(1,)`` [1/s].
    time_s : numpy.ndarray
        Times, shape ``(n_times,)`` [s].
    reord_sens : bool, optional
        Keyword passed by ``MultipleCurveResolution``; unused.

    Returns
    -------
    states : numpy.ndarray
        Shape ``(n_times, 3)`` [mol/L], rank 2.
    sensitivities : numpy.ndarray
        Shape ``(1, n_times, 3)`` [mol/L*s].
    """
    slope, offset = OFFSET_COMBINATION
    decayed = np.exp(-params[0] * time_s)  # [mol/L]
    states = np.column_stack((decayed, np.ones_like(decayed),
                              slope * decayed + offset))
    sens = np.column_stack((-time_s * decayed, 0 * decayed,
                            -slope * time_s * decayed))  # [mol/L*s]
    return states, sens[np.newaxis]


def offset_spectra(time_s):
    """Return two-wavelength absorbance outside the column space of C.

    Parameters
    ----------
    time_s : numpy.ndarray
        Times, shape ``(n_times,)`` [s].

    Returns
    -------
    numpy.ndarray
        Columns ``1 - 1.5 exp(-0.7 t) + 0.1 t`` and ``0.5 + 0.3 sin(t)``,
        shape ``(n_times, 2)`` [-]; synthetic, chosen so that a
        least-squares absorptivity is negative.
    """
    return np.column_stack((1.0 - 1.5 * np.exp(-0.7 * time_s)
                            + 0.1 * time_s, 0.5 + 0.3 * np.sin(time_s)))


RANK_CASES = {
    'ramp': (ramp_model, ramp_spectra, RAMP_TIMES, RAMP_SLOPE, 1.0),
    'decay': (decay_model, decay_spectra, DECAY_TIMES, DECAY_WEIGHT,
              PENALTY_WEIGHT),
    'offset': (offset_model, offset_spectra, OFFSET_TIMES, OFFSET_RATE,
               PENALTY_WEIGHT),
}
# Cases whose spectra and dC lie in the column space of C, so their data
# residuals and data Jacobian are zero to roundoff: they test the penalty
# rows only. 'offset' tests the data rows (get_sens_projection) too.
ZERO_DATA_CASES = ('ramp', 'decay')


def _rank_estimator(name, global_analysis):
    """Build an estimator of a rank-deficient case.

    Parameters
    ----------
    name : str
        Key of ``RANK_CASES``.
    global_analysis : bool
        Global (first experiment only) or local (both experiments).

    Returns
    -------
    MultipleCurveResolution
        Estimator seeded at the case's coefficient.
    """
    model, spectra, times, seed, mult_penalty = RANK_CASES[name]
    times = times[:1] if global_analysis else times
    return MultipleCurveResolution(
        model, seed, list(times), [spectra(time_s) for time_s in times],
        measured_ind=[0, 1, 2], mult_penalty=mult_penalty,
        global_analysis=global_analysis)


@pytest.mark.parametrize('global_analysis', [True, False],
                         ids=['global', 'local'])
@pytest.mark.parametrize('name', list(RANK_CASES))
def test_rank_deficient_derivatives_match_central_differences(
        name, global_analysis):
    """Constant-rank C: projector and pseudoinverse derivatives hold."""
    model, _, times, seed, _ = RANK_CASES[name]
    for time_s in times:
        assert np.linalg.matrix_rank(model(seed, time_s)[0]) == RANK
    estimator = _rank_estimator(name, global_analysis)
    residual = estimator.get_objective(seed, out_array=True)
    assert np.any(estimator.epsilon_mcr < 0)  # active penalty

    jacobian = estimator.get_gradient(seed, out_array=True)[0]
    gradient = estimator.get_gradient(seed)  # [-]

    objective = _fresh_objective(_rank_estimator, name, global_analysis)
    expected = _central_difference(lambda value: objective(value, True),
                                   seed)
    num_data = residual.size - estimator.epsilon_mcr.size  # [-]
    scale = np.abs(expected).max()
    if name not in ZERO_DATA_CASES:
        assert np.abs(residual[:num_data]).max() > NONZERO_FLOOR
        assert np.abs(jacobian[:num_data]).max() > NONZERO_FLOOR
    # Data rows test get_sens_projection (nonzero only for 'offset'),
    # penalty rows the pseudoinverse derivative.
    np.testing.assert_allclose(jacobian[:num_data], expected[:num_data],
                               rtol=0.0, atol=DERIVATIVE_RTOL * scale)
    np.testing.assert_allclose(jacobian[num_data:], expected[num_data:],
                               rtol=0.0, atol=DERIVATIVE_RTOL * scale)
    assert gradient[0] == pytest.approx(_central_difference(objective, seed),
                                        rel=DERIVATIVE_RTOL)


# ------------------------------------------------ several parameters
# Synthetic three-species fixture: A decays with k_A, B forms with k_B and C
# follows t / (1 + t); all scale with the fixed initial concentration c0.
# Parameters in the order (k_A [1/s], c0 [mol/L], k_B [1/s]) with c0 fixed
# in the middle, so a parameter-selection error shows. Three species and
# four wavelengths keep n_species != n_lambda.
MULTI_TRUE = np.array([0.5, 1.5, 0.3])  # [1/s], [mol/L], [1/s]
MULTI_SEED = np.array([0.6, 1.5, 0.35])  # [1/s], [mol/L], [1/s]
MULTI_FLAGS = [True, False, True]
MULTI_ABSORPTIVITY = np.array([[1.0, 0.6, -0.3, 0.2],
                               [0.2, 0.9, 0.5, -0.4],
                               [0.4, -0.2, 0.7, 1.1]])  # [L/mol]
MULTI_TIMES = (np.array([0.3, 0.7, 1.2, 2.0, 3.0, 4.5, 6.0]),
               np.array([0.5, 1.0, 1.8, 2.7, 3.9, 5.5]))  # [s]
# Fixed deterministic absorbance errors [-], amplitude 0.02, no pattern
# aligned with the rows or wavelengths.
MULTI_ERROR_AMPLITUDE = 0.02  # [-]
MULTI_ERROR_FREQUENCY = 1.7  # [-], radians per flattened entry


def three_species_model(params, time_s, reord_sens=False):
    """Return three species and their sensitivities to all parameters.

    Parameters
    ----------
    params : numpy.ndarray
        ``(k_A [1/s], c0 [mol/L], k_B [1/s])``.
    time_s : numpy.ndarray
        Times, shape ``(n_times,)`` [s].
    reord_sens : bool, optional
        Keyword passed by ``MultipleCurveResolution``; unused.

    Returns
    -------
    states : numpy.ndarray
        Shape ``(n_times, 3)`` [mol/L]: ``c0 exp(-k_A t)``, ``c0 (1 -
        exp(-k_B t))`` and ``c0 t / (1 + t)``.
    sensitivities : numpy.ndarray
        Shape ``(3, n_times, 3)``: derivatives with respect to k_A
        [mol/L*s], c0 [-] and k_B [mol/L*s].
    """
    rate_a, conc_zero, rate_b = params
    decay_a = np.exp(-rate_a * time_s)  # [-]
    decay_b = np.exp(-rate_b * time_s)  # [-]
    shape_c = time_s / (1.0 + time_s)  # [-]
    states = conc_zero * np.column_stack((decay_a, 1.0 - decay_b, shape_c))
    zeros = np.zeros_like(time_s)
    sens = np.stack((
        np.column_stack((-time_s * conc_zero * decay_a, zeros, zeros)),
        states / conc_zero,
        np.column_stack((zeros, time_s * conc_zero * decay_b, zeros))))
    return states, sens


def _multi_spectra(time_s):
    """Return the absorbance of the three-species fixture.

    Parameters
    ----------
    time_s : numpy.ndarray
        Times, shape ``(n_times,)`` [s].

    Returns
    -------
    numpy.ndarray
        ``states @ MULTI_ABSORPTIVITY`` at ``MULTI_TRUE`` plus fixed
        errors, shape ``(n_times, 4)`` [-].
    """
    states = three_species_model(MULTI_TRUE, time_s)[0]  # [mol/L]
    num_wavelengths = MULTI_ABSORPTIVITY.shape[1]  # [-]
    errors = MULTI_ERROR_AMPLITUDE * np.sin(
        MULTI_ERROR_FREQUENCY * np.arange(states.shape[0] * num_wavelengths)
    ).reshape(-1, num_wavelengths)  # [-]
    return states @ MULTI_ABSORPTIVITY + errors


def _multi_estimator(global_analysis):
    """Build the three-species estimator with c0 fixed.

    Parameters
    ----------
    global_analysis : bool
        Global (first experiment) or local (both experiments) analysis.

    Returns
    -------
    MultipleCurveResolution
        Estimator with model sensitivities and ``PENALTY_WEIGHT``.
    """
    times = MULTI_TIMES[:1] if global_analysis else MULTI_TIMES
    return MultipleCurveResolution(
        three_species_model, MULTI_SEED, list(times),
        [_multi_spectra(time_s) for time_s in times],
        measured_ind=[0, 1, 2], optimize_flags=MULTI_FLAGS,
        mult_penalty=PENALTY_WEIGHT, global_analysis=global_analysis)


@pytest.mark.parametrize('global_analysis', [True, False],
                         ids=['global', 'local'])
def test_several_parameters_with_fixed_one_match_central_differences(
        global_analysis):
    optimized = MULTI_SEED[MULTI_FLAGS]  # [1/s], (k_A, k_B)
    estimator = _multi_estimator(global_analysis)
    residual = estimator.get_objective(optimized, out_array=True)
    assert np.any(estimator.epsilon_mcr < 0)  # active penalty

    jacobian = estimator.get_gradient(optimized, out_array=True)
    gradient = estimator.get_gradient(optimized)  # [s]

    objective = _fresh_objective(_multi_estimator, global_analysis)
    assert jacobian.shape == (2, residual.size)
    for index in range(optimized.size):
        step = np.zeros_like(optimized)  # [1/s]
        step[index] = FD_RELATIVE_STEP * optimized[index]
        expected_row = (objective(optimized + step, True)
                        - objective(optimized - step, True)) / (
                            2 * step[index])  # [weighted residual * s]
        np.testing.assert_allclose(
            jacobian[index], expected_row, rtol=0.0,
            atol=DERIVATIVE_RTOL * np.abs(expected_row).max())
        expected_gradient = (objective(optimized + step)
                             - objective(optimized - step)) / (
                                 2 * step[index])  # [s]
        assert gradient[index] == pytest.approx(expected_gradient,
                                                rel=DERIVATIVE_RTOL)


# ------------------------------------------------- cache and solvers


@pytest.mark.parametrize('global_analysis', [True, False],
                         ids=['global', 'local'])
def test_direct_analysis_call_does_not_corrupt_gradient(global_analysis):
    """A public analysis at another point invalidates the cached state."""
    if global_analysis:
        build = _negative_estimator
        analysis = 'get_global_analysis'
    else:
        build = _local_estimator
        analysis = 'get_local_analysis'
    estimator = build(case.model_with_sensitivities)
    estimator.get_objective(case.TRIAL_RATE)
    getattr(estimator, analysis)(OTHER_RATE)

    jacobian = estimator.get_gradient(case.TRIAL_RATE, out_array=True)
    gradient = estimator.get_gradient(case.TRIAL_RATE)  # [s]

    fresh = build(case.model_with_sensitivities)
    fresh.get_objective(case.TRIAL_RATE)
    np.testing.assert_allclose(
        jacobian, fresh.get_gradient(case.TRIAL_RATE, out_array=True),
        rtol=ROUNDOFF_RTOL, atol=0.0)
    np.testing.assert_allclose(gradient, fresh.get_gradient(case.TRIAL_RATE),
                               rtol=ROUNDOFF_RTOL, atol=0.0)


# SciPy's least-squares stops on 1e-12 changes, so its optimum agrees with
# LM's (1e-8 relative step stop) far inside the 1e-6 budget.
SCIPY_OPTIONS = {'xtol': 1e-12, 'ftol': 1e-12, 'gtol': 1e-12}  # [-]


@pytest.mark.parametrize('method', ['scipy', 'least_squares'])
def test_scipy_fit_minimizes_penalized_objective(method):
    estimator = _negative_estimator(case.model_with_sensitivities)

    fitted, covariance, info = estimator.optimize_fn(
        method=method, optim_options=dict(SCIPY_OPTIONS), verbose=False)

    reference = _negative_estimator(case.model_with_sensitivities)
    reference_fit, _, _ = reference.optimize_fn(verbose=False)  # [1/s]
    assert fitted[0] == pytest.approx(reference_fit[0], rel=DERIVATIVE_RTOL)
    data_residual, absorptivity, at_fit = _negative_reference(fitted)
    for factor in (1 - MINIMUM_PROBE, 1 + MINIMUM_PROBE):
        assert _negative_reference(fitted * factor)[2] > at_fit
    # Native SciPy diagnostics describe the full residual vector.
    assert info['cost'] == pytest.approx(at_fit, rel=ROUNDOFF_RTOL)

    data = data_residual.T.ravel()  # [-]
    data_jacobian = _central_difference(
        lambda rate: _negative_reference(rate)[0].T.ravel(), fitted)
    np.testing.assert_allclose(info['fun'], data, rtol=ROUNDOFF_RTOL,
                               atol=0.0)
    assert info['jac'].shape == (1, data.size)
    np.testing.assert_allclose(
        info['jac'][0], data_jacobian, rtol=0.0,
        atol=DERIVATIVE_RTOL * np.abs(data_jacobian).max())
    np.testing.assert_array_equal(estimator.sens, info['jac'].T)
    active = absorptivity.ravel() < 0
    assert active.any()
    np.testing.assert_array_equal(info['fun_penalty'][~active], 0.0)
    np.testing.assert_allclose(
        0.5 * info['fun_penalty'][active]**2,
        PENALTY_WEIGHT * absorptivity.ravel()[active]**2,
        rtol=ROUNDOFF_RTOL, atol=0.0)
    dof = data.size - 1  # [-], one optimized rate
    assert covariance[0, 0] == pytest.approx(
        data @ data / dof / (data_jacobian @ data_jacobian),
        rel=DERIVATIVE_RTOL)


def test_update_self_is_a_deprecated_alias_of_set_self():
    estimator = _negative_estimator(case.model_with_sensitivities)
    estimator.get_objective(case.TRIAL_RATE)
    residuals = estimator.residuals.copy()  # weighted [-]

    with pytest.warns(DeprecationWarning, match='update_self is deprecated'):
        other = estimator.get_objective(OTHER_RATE, True, update_self=False)

    np.testing.assert_array_equal(estimator.residuals, residuals)
    np.testing.assert_array_equal(
        other, _negative_estimator(case.model_with_sensitivities)
        .get_objective(OTHER_RATE, True))
    with pytest.warns(DeprecationWarning, match='update_self is deprecated'):
        estimator.get_objective(OTHER_RATE, update_self=True)
    assert not np.array_equal(estimator.residuals, residuals)
    with pytest.raises(TypeError, match='set_self and update_self disagree'):
        estimator.get_objective(OTHER_RATE, set_self=True,
                                update_self=False)


def test_gradient_set_self_false_keeps_reporting_state():
    """Re-evaluating at another point with set_self=False stores nothing."""
    estimator = _negative_estimator(case.model_with_sensitivities)
    estimator.get_objective(case.TRIAL_RATE)
    residuals = estimator.residuals.copy()  # weighted [-]
    absorptivity = estimator.epsilon_mcr.copy()  # [L/mol]
    num_history = len(estimator.params_iter)  # [-]

    gradient = estimator.get_gradient(OTHER_RATE, set_self=False)  # [s]

    np.testing.assert_array_equal(estimator.residuals, residuals)
    np.testing.assert_array_equal(estimator.epsilon_mcr, absorptivity)
    assert len(estimator.params_iter) == num_history
    assert gradient[0] == pytest.approx(
        _central_difference(_fresh_objective(
            _negative_estimator, case.model_with_sensitivities), OTHER_RATE),
        rel=DERIVATIVE_RTOL)
