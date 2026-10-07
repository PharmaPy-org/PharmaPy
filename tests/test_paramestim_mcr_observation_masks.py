"""Staggered non-spectral observations in multivariate curve resolution.

Issue #236 for ``MultipleCurveResolution`` (MCR): spectra are recorded on
the full model grid of each experiment, while a non-spectral tracer is
measured at fewer, experiment-specific times. Unobserved tracer entries must
have zero weighted residual and zero sensitivity for model-returned and
finite-difference Jacobians, partially observed rows must be weighted by the
marginal precision of their observed columns, and the weight matrix is the
covariance of the measurement errors (data minus truth), also when it
correlates spectral and tracer columns.

The fixture is a first-order A -> B reaction with absorbance [-] at two
wavelengths and a tracer [mol/L] that depends on the rate. Expected values
are computed here from ``numpy.linalg.lstsq`` projections, measurement
errors and ``numpy.linalg.inv`` of each observed covariance block, not from
the estimator's residuals, masks or weight roots. Runs in milliseconds.
"""

import numpy as np
import pytest

from PharmaPy.ParamEstim import MultipleCurveResolution


pytestmark = pytest.mark.unit

# Experiment name -> (model-grid times [s], tracer row indices).
SCHEDULES = {
    'first': (np.array([0.5, 1.0, 2.0, 3.5, 5.0]), [0, 1, 4]),
    'second': (np.array([0.8, 1.6, 2.5, 4.0]), [1, 3]),
}
TRUE_RATE = np.array([0.4])  # [1/s]
TRIAL_RATE = np.array([0.3])  # [1/s]
TRACER_OFFSET = 0.3  # [mol/L]
TRACER_GAIN = 0.5  # [mol/L], tracer = offset + gain * k * t
# Pure-species absorptivities, rows (A, B), columns wavelengths; positive,
# so the fitted absorptivities stay positive and MCR adds no penalty.
ABSORPTIVITY = np.array([[1.2, 0.4], [0.3, 0.9]])  # [L/mol]
SPECTRA_ERRORS = {  # [-], fixed asymmetric absorbance errors
    'first': np.array([[0.020, -0.010], [-0.030, 0.025], [0.010, -0.020],
                       [0.035, 0.005], [-0.020, 0.030]]),
    'second': np.array([[-0.015, 0.020], [0.025, -0.005], [-0.010, 0.015],
                        [0.030, -0.025]]),
}
TRACER_ERRORS = {'first': np.array([0.010, -0.020, 0.015]),
                 'second': np.array([-0.012, 0.018])}  # [mol/L]
# Columns (wavelength 1 [-], wavelength 2 [-], tracer [mol/L]); entries in
# the products of those units. The correlated case couples the tracer with
# both wavelengths; both matrices are positive definite.
COVARIANCES = {
    'block_diagonal': np.array([[4.0, -1.0, 0.0],
                                [-1.0, 2.0, 0.0],
                                [0.0, 0.0, 3.0]]) * 1e-4,
    'correlated': np.array([[4.0, -1.0, 1.5],
                            [-1.0, 2.0, -0.8],
                            [1.5, -0.8, 3.0]]) * 1e-4,
}
# Central differences with a 1e-5 relative step: truncation O(h**2) near
# 1e-11 and roundoff eps * |r| / h near 1e-11 relative to the O(1e-1)
# derivatives, so 1e-7 relative to the largest entry bounds both.
FD_RELATIVE_STEP = 1e-5  # [-]
DERIVATIVE_RTOL = 1e-7  # [-]
# PharmaPy's own forward differences use a step of |k| * sqrt(1e-6);
# truncation (dx / 2) |r'' / r'| stays below 1e-3 of the largest entry.
FORWARD_DIFFERENCE_RTOL = 1e-3  # [-]
# Quadratic forms of O(1e2) terms summed over 9 rows; roundoff near 1e-13.
ROUNDOFF_RTOL = 1e-10  # [-]
# Relative rate offset probing the fitted minimum: LM's default 1e-8 step
# and gradient stops place the fit far closer to the minimum than 1e-3,
# while the objective change at 1e-3 (curvature times 1e-6 k**2) stays far
# above roundoff.
MINIMUM_PROBE = 1e-3  # [-]


def _states(params, time_s):
    """Return species and tracer concentrations.

    Parameters
    ----------
    params : numpy.ndarray
        Rate k, shape ``(1,)`` [1/s].
    time_s : numpy.ndarray
        Times, shape ``(n_times,)`` [s].

    Returns
    -------
    numpy.ndarray
        Shape ``(n_times, 3)``: A and B (unit initial concentration) and the
        tracer ``TRACER_OFFSET + TRACER_GAIN k t`` [mol/L].
    """
    reactant = np.exp(-params[0] * time_s)  # [mol/L]
    return np.column_stack(
        (reactant, 1.0 - reactant,
         TRACER_OFFSET + TRACER_GAIN * params[0] * time_s))


def model_with_sensitivities(params, time_s, reord_sens=False):
    """Return states and their rate sensitivities, as MCR accepts them.

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
        Shape ``(n_times, 3)`` [mol/L], see ``_states``.
    sensitivities : numpy.ndarray
        Shape ``(1, n_times, 3)``: d(state)/dk [mol/L*s], evaluated on the
        full model grid, including unobserved tracer times.
    """
    reactant = np.exp(-params[0] * time_s)  # [mol/L]
    sens = np.column_stack((-time_s * reactant, time_s * reactant,
                            TRACER_GAIN * time_s))  # [mol/L*s]
    return _states(params, time_s), sens[np.newaxis]


def model_states_only(params, time_s, reord_sens=False):
    """Return the states without sensitivities (finite-difference path).

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
    numpy.ndarray
        Shape ``(n_times, 3)`` [mol/L], see ``_states``.
    """
    return _states(params, time_s)


def _data():
    """Return the spectra and tracer observations of each experiment.

    Returns
    -------
    dict
        Experiment name -> (absorbance ``(n_times, 2)`` [-], tracer
        ``(n_tracer,)`` [mol/L]).
    """
    data = {}
    for name, (time_s, tracer_rows) in SCHEDULES.items():
        states = _states(TRUE_RATE, time_s)  # [mol/L]
        spectra = states[:, :2] @ ABSORPTIVITY + SPECTRA_ERRORS[name]
        tracer = states[tracer_rows, 2] + TRACER_ERRORS[name]
        data[name] = (spectra, tracer)
    return data


def _estimator(model, weight_key):
    """Build the staggered MCR estimator.

    Parameters
    ----------
    model : callable
        ``model_with_sensitivities`` or ``model_states_only``.
    weight_key : str or None
        Key of ``COVARIANCES``; None for identity weights.

    Returns
    -------
    MultipleCurveResolution
        Estimator seeded at ``TRIAL_RATE``.
    """
    data = _data()
    x_data = {name: {'spectra': time_s, 'non_spectra': time_s[rows]}
              for name, (time_s, rows) in SCHEDULES.items()}
    y_data = {name: {'spectra': spectra, 'non_spectra': tracer}
              for name, (spectra, tracer) in data.items()}
    return MultipleCurveResolution(
        model, TRIAL_RATE, x_data, y_data,
        measured_ind={'spectra': [0, 1], 'non_spectra': [2]},
        weight_matrix=None if weight_key is None else COVARIANCES[weight_key],
        name_states=['A', 'B', 'tracer'])


def _measurement_errors(params):
    """Return per-row residuals in the data-minus-model convention.

    Parameters
    ----------
    params : numpy.ndarray
        Rate k, shape ``(1,)`` [1/s].

    Returns
    -------
    list of tuple
        ``(observed_columns, values)`` per stacked model-grid row of both
        experiments: spectral residuals after the least-squares projection
        on the concentrations of A and B [-], and the tracer's data minus
        model [mol/L] where measured.
    """
    data = _data()
    conc = np.vstack([_states(params, time_s)[:, :2]
                      for time_s, _ in SCHEDULES.values()])  # [mol/L]
    spectra = np.vstack([spectra for spectra, _ in data.values()])  # [-]
    absorptivity, *_ = np.linalg.lstsq(conc, spectra, rcond=None)
    assert np.all(absorptivity > 0)  # no nonnegativity penalty
    spectral = spectra - conc @ absorptivity  # [-]

    rows = []
    offset = 0
    for name, (time_s, tracer_rows) in SCHEDULES.items():
        tracer_model = _states(params, time_s)[:, 2]  # [mol/L]
        for row in range(time_s.size):
            values = list(spectral[offset + row])
            columns = [0, 1]
            if row in tracer_rows:
                columns.append(2)
                values.append(data[name][1][tracer_rows.index(row)]
                              - tracer_model[row])
            rows.append((columns, np.array(values)))
        offset += time_s.size
    return rows


def _marginal_gls(params, covariance):
    """Return the marginal GLS objective and rate information.

    Parameters
    ----------
    params : numpy.ndarray
        Rate k, shape ``(1,)`` [1/s].
    covariance : numpy.ndarray
        Measurement-error covariance, shape ``(3, 3)``.

    Returns
    -------
    objective : float
        ``1/2 sum e_o^T inv(S_oo) e_o`` [-].
    information : float
        ``sum de_o^T inv(S_oo) de_o`` [s**2], with ``de`` the central
        difference of the residuals with respect to the rate.
    """
    step = FD_RELATIVE_STEP * params  # [1/s]
    rows = _measurement_errors(params)
    upper = _measurement_errors(params + step)
    lower = _measurement_errors(params - step)
    objective = 0.0  # [-]
    information = 0.0  # [s**2]
    for (columns, values), (_, high), (_, low) in zip(rows, upper, lower):
        precision = np.linalg.inv(covariance[np.ix_(columns, columns)])
        derivative = (high - low) / (2 * step[0])
        objective += 0.5 * values @ precision @ values
        information += derivative @ precision @ derivative
    return objective, information


def _unobserved_tracer_entries():
    """Return a mask of unobserved tracer entries in the residual vector.

    Returns
    -------
    numpy.ndarray
        Boolean, length ``3 * n_rows``: the column-major residual layout
        (wavelength 1, wavelength 2, tracer) over the stacked rows.
    """
    tracer_observed = np.concatenate([
        np.isin(np.arange(time_s.size), rows)
        for time_s, rows in SCHEDULES.values()])
    num_rows = tracer_observed.size
    return np.concatenate((np.zeros(2 * num_rows, dtype=bool),
                           ~tracer_observed))


@pytest.mark.parametrize('weight_key', [None, *COVARIANCES])
def test_model_jacobian_is_derivative_of_weighted_residual(weight_key):
    estimator = _estimator(model_with_sensitivities, weight_key)
    weighted = estimator.get_objective(TRIAL_RATE, True)
    jacobian = estimator.get_gradient(TRIAL_RATE, out_array=True)[0]

    step = FD_RELATIVE_STEP * TRIAL_RATE  # [1/s]
    difference = (estimator.get_objective(TRIAL_RATE + step, True, False)
                  - estimator.get_objective(TRIAL_RATE - step, True, False)
                  ) / (2 * step[0])
    np.testing.assert_allclose(
        jacobian, difference, rtol=0.0,
        atol=DERIVATIVE_RTOL * np.abs(difference).max())

    unobserved = _unobserved_tracer_entries()
    np.testing.assert_array_equal(weighted[unobserved], 0.0)
    np.testing.assert_array_equal(jacobian[unobserved], 0.0)
    assert np.all(jacobian[~unobserved] != 0.0)


@pytest.mark.parametrize('weight_key', list(COVARIANCES))
def test_finite_difference_jacobian_matches_model_jacobian(weight_key):
    analytic = _estimator(model_with_sensitivities, weight_key)
    analytic.get_objective(TRIAL_RATE)
    expected = analytic.get_gradient(TRIAL_RATE, out_array=True)

    numerical = _estimator(model_states_only, weight_key)
    numerical.get_objective(TRIAL_RATE)
    jacobian = numerical.get_gradient(TRIAL_RATE, out_array=True)

    np.testing.assert_array_equal(jacobian[0][_unobserved_tracer_entries()],
                                  0.0)
    np.testing.assert_allclose(
        jacobian, expected, rtol=0.0,
        atol=FORWARD_DIFFERENCE_RTOL * np.abs(expected).max())


@pytest.mark.parametrize('weight_key', list(COVARIANCES))
def test_objective_and_information_match_marginal_gls(weight_key):
    """Weighting uses inv(Sigma_oo) of the measurement-error covariance."""
    estimator = _estimator(model_with_sensitivities, weight_key)
    objective = estimator.get_objective(TRIAL_RATE)  # [-]
    jacobian = estimator.get_gradient(TRIAL_RATE, out_array=True)

    expected_objective, expected_information = _marginal_gls(
        TRIAL_RATE, COVARIANCES[weight_key])

    assert objective == pytest.approx(expected_objective, rel=ROUNDOFF_RTOL)
    assert (jacobian @ jacobian.T)[0, 0] == pytest.approx(
        expected_information, rel=DERIVATIVE_RTOL)


@pytest.mark.parametrize('model',
                         [model_with_sensitivities, model_states_only],
                         ids=['model_jacobian', 'finite_difference'])
@pytest.mark.parametrize('weight_key', list(COVARIANCES))
def test_lm_fit_minimizes_marginal_gls(model, weight_key):
    """LM reaches a local minimum of the independent marginal objective.

    The objective is evaluated at the fitted rate and at a relative offset
    of ``MINIMUM_PROBE`` on either side; both neighbours must be higher.
    """
    estimator = _estimator(model, weight_key)
    fitted, _, info = estimator.optimize_fn(verbose=False)

    assert info['num_iter'] > 0
    covariance = COVARIANCES[weight_key]
    at_fit, _ = _marginal_gls(fitted, covariance)
    for factor in (1 - MINIMUM_PROBE, 1 + MINIMUM_PROBE):
        neighbour, _ = _marginal_gls(fitted * factor, covariance)
        assert neighbour > at_fit
