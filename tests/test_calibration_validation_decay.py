"""Calibration validation of the analytic decay stand-in (#352, AC2-AC6).

Scope: the shared public estimation contract exercised end to end on one
case, the first of the three #352 validation modules (the Ziegler and
crystallizer modules, ``test_calibration_validation_ziegler`` and
``test_calibration_validation_crystallizer``, run the same contract in the
Assimulo lane). On the fitted decay it checks: derivative verification with
``check_derivatives`` for the default forward-difference Jacobian, a
``jac_fun`` and model-returned sensitivities, with a fixed parameter and
near a lower bound (AC2); experiment identity and alignment, i.e. fits and
named residual rows that do not depend on experiment or measurement order,
residuals and accepted outputs equal to the closed form placed by
``get_residual_layout``, bitwise repeatability, and observations left
unchanged by the uncertainty, identifiability, derivative and bootstrap
analyses (AC3); the identifiability report with its declared-uncertainty
noise assumption in both covariance forms (AC4); the convergence outcome
and stopping tolerances of converged and truncated fits (AC5); and the
``estimate_parameters(method='LM')`` -> ``CalibrationRecord`` -> JSON ->
``rerun`` path (AC6). Detailed unit coverage of each feature is in
``test_paramestim_derivative_check``,
``test_paramestim_identifiability_report``,
``test_paramestim_convergence_reporting`` and ``test_records_calibration``;
this module does not repeat it, it exercises the features together at the
accepted fit.

Fixture: the synthetic two-experiment first-order decay A -> B with an
adiabatic temperature rise of ``test_records_calibration`` (hot/cold runs,
c_A [mol/L] and T [K], a missing cold c_A sample and unobserved entries).
It is a stand-in verification fixture, not an empirical calibration: the
data were written for the user guide. Noise assumption: the measurements
declare their standard deviations (``Measurement.uncertainty``), so the
weighted residuals are dimensionless; the fits are Levenberg-Marquardt
(LM), because SciPy fits cannot yet be recorded (#534). Only the T sample
of the hot run is observed, where the rise is 20 K*L/mol, the value
``decay_rise_first`` takes as its fixed first parameter, so the
one-parameter fit and the fixed-rise fit see the same residuals. Expected
values come from the closed-form model and its derivatives
(``test_paramestim_derivative_check``), never from the production code.
Core lane; every fit takes milliseconds.
"""

import warnings

import numpy as np
import pytest

import test_paramestim_derivative_check as deriv
import test_paramestim_identifiability_report as ident
import test_records_calibration as case
from PharmaPy.jac_module import CENTRAL_DIFFERENCE_REL_STEP
from PharmaPy.ParamEstim import ParameterEstimation
from PharmaPy.Records import CalibrationRecord
from PharmaPy.StatsModule import StatisticsClass

pytestmark = pytest.mark.unit

EPS = np.finfo(float).eps  # [-], float64 machine epsilon
RISE = case.RISE_PARAMETER  # [K*L/mol], rise of the only observed T sample
NUM_OBSERVED = len(case.OBSERVED_ENTRIES)  # [-], observed entries
# Stopping criteria a fit without options must report: the LM keyword
# defaults of test_records_calibration that bound the iterations.
# max_fun_eval [-]; eps_1 [gradient unit], eps_2 [-] relative step and
# tol_fun [weighted residual unit], as in case.LM_DEFAULT_OPTIONS.
LM_TOLERANCES = {key: case.LM_DEFAULT_OPTIONS[key]
                 for key in ('max_fun_eval', 'eps_1', 'eps_2', 'tol_fun')}
# LM stop criteria that are convergence tests (doc/online_docs/
# advanced_usage.rst, convergence section).
LM_CONVERGED = ('Small step', 'Small gradient',
                'Value of the objective function lower than tolerance')
# One LM evaluation: fewer than the nine the documented fit needs, so the
# fit stops at the evaluation limit.
TRUNCATED_EVALUATIONS = 1  # [-]
# Experiment and measurement orders of the alignment test; the first is the
# documented order the others are compared with.
ORDERS = [(('hot', 'cold'), ('c_A', 'T')), (('cold', 'hot'), ('c_A', 'T')),
          (('hot', 'cold'), ('T', 'c_A')), (('cold', 'hot'), ('T', 'c_A'))]
# A reordered fit runs the same LM iterations on a permuted residual vector:
# only the summation order of J @ J.T, J @ r and r @ r changes, a few ulps
# per iterate, and the one-parameter Gauss-Newton step does not amplify
# them. 1e-12 relative bounds that with margin; on the reference platform
# the fits are bitwise equal.
REORDER_RTOL = case.ROUNDOFF_RTOL  # [-]
# Residual-bootstrap samples of the data-preservation check; any positive
# count exercises the refits.
BOOTSTRAP_SAMPLES = 4  # [-]
BOOTSTRAP_SEED = 352  # [-], arbitrary fixed NumPy seed
# Safety factor of the derivative-check bounds of the solver-backed
# validation modules (Ziegler, crystallizer). Their bound is the reference
# error model C_T h**2 + r / h (check_derivatives Notes), whose constants
# are model-dependent and known only in order of magnitude. Design
# assumption: a factor 10 allows for constants up to an order of magnitude
# above the stated ones. The resulting bounds stay below the 0.1 relative
# error of a sensitivity row wrong by 10 % (deriv.WRONG_FACTOR) by about 50
# times (Ziegler, 2e-3), 6 times (crystallizer at CVode's default rtol,
# 1.6e-2) and 140 times (crystallizer at the tightened rtol, 7.2e-4).
SOLVER_CHECK_SAFETY = 10.0  # [-]
# Smallest admissible ratio of that 0.1 error to a derivative-check bound.
# The relative error of a row wrong by 10 % is at least 0.1 minus the
# reference error, itself at most the bound, so at a ratio of 5 it is at
# least 4 times the bound: such a row is always flagged.
MIN_POWER_MARGIN = 5.0  # [-]


def fit(model=case.decay, seed=(case.SEED_RATE,), experiments=None,
        **options):
    """Fit the decay case with Levenberg-Marquardt.

    Parameters
    ----------
    model : callable, optional
        Model callback; ``case.decay`` (rate k [1/s]) by default.
    seed : sequence of float, optional
        Parameter seed in the callback's order and units.
    experiments : dict, optional
        Experiments; the documented ones by default.
    **options
        Further ``ParameterEstimation`` keywords (``jac_fun``,
        ``optimize_flags``, ``name_params``).

    Returns
    -------
    ParameterEstimation
        The fitted estimator.
    """
    options.setdefault('name_params', ['k'])
    experiments = case.make_experiments() if experiments is None \
        else experiments
    estimator = ParameterEstimation(model, list(seed), experiments,
                                    **options)
    estimator.optimize_fn(verbose=False)
    return estimator


def rise_and_rate(rate):
    """Return the closed-form parameter point of the observed residuals.

    Parameters
    ----------
    rate : float
        Rate constant k [1/s].

    Returns
    -------
    tuple of float
        ``(RISE [K*L/mol], rate [1/s])``, the argument of the closed-form
        helpers of ``test_paramestim_derivative_check``.
    """
    return (RISE, float(rate))


def exact_rate_row(rate):
    """Return the exact rate sensitivities of the observed residuals.

    Parameters
    ----------
    rate : float
        Rate constant k [1/s].

    Returns
    -------
    numpy.ndarray
        ``d r / d k`` [s], shape (7,), in ``case.OBSERVED_ENTRIES`` order.
    """
    return deriv.closed_form(rise_and_rate(rate))[1][1]


def reference_error_bound(check, formula='central'):
    """Bound the reference error of a rate derivative check entry by entry.

    Parameters
    ----------
    check : DerivativeCheck
        One-parameter (rate) check.
    formula : str, optional
        Stencil of the reference.

    Returns
    -------
    numpy.ndarray
        Taylor remainder plus rounding of the reference and of the exact
        row [s], shape (7,) (``deriv.reference_bound``).
    """
    point = rise_and_rate(check.params[0])
    truncation, rounding_term = deriv.reference_bound(
        point, 1, check.steps[0], formula)  # [s], [s]
    exact = exact_rate_row(check.params[0])  # [s]
    return truncation + rounding_term + deriv.ROUNDING_ULPS * EPS * np.abs(
        exact)


def forward_difference_error(rate):
    """Return the error of the default forward-difference rate row.

    Parameters
    ----------
    rate : float
        Rate constant k [1/s] at which the Jacobian is evaluated.

    Returns
    -------
    leading : numpy.ndarray
        Leading truncation term ``dx / 2 * f''(k)`` [s], shape (7,), with
        the default step ``dx = 1e-3 * k``.
    bound : numpy.ndarray
        Bound on the rest [s]: the next Taylor remainder ``dx**2 / 6 *
        |f'''(k)|`` (``|f'''|`` decreases with k, so its value at k bounds
        it on ``[k, k + dx]``), the rounding of the two outputs over ``dx``
        and that of representing ``k + dx`` (``2 eps |k + dx| / dx`` times
        the row).
    """
    point = rise_and_rate(rate)
    step = ident.DEFAULT_RELATIVE_STEP * rate  # [1/s]
    leading = step / 2 * deriv.rate_derivative(point, 2)  # [s]
    remainder = step ** 2 / 6 * np.abs(
        deriv.rate_derivative(point, 3))  # [s]
    rounding_term = (deriv.rounding(point) + deriv.rounding(
        deriv.shifted(point, 1, step))) / step  # [s]
    representation = 2 * EPS * (rate + step) / step * np.abs(
        exact_rate_row(rate))  # [s]
    return leading, remainder + rounding_term + representation


def forward_row_norm_bracket(rate):
    """Bracket the norm of the forward-difference rate row.

    Parameters
    ----------
    rate : float
        Rate constant k [1/s].

    Returns
    -------
    tuple of float
        Lower and upper bounds [s] of ``||J_fd||`` over the observed
        entries: ``||f' + leading|| -/+ ||bound||`` (triangle inequality).
    """
    leading, bound = forward_difference_error(rate)  # [s], [s]
    center = np.linalg.norm(exact_rate_row(rate) + leading)  # [s]
    spread = np.linalg.norm(bound)  # [s]
    return center - spread, center + spread


def layout_keys(estimator):
    """Return the named rows of the residual layout.

    Parameters
    ----------
    estimator : ParameterEstimation
        Fitted estimator.

    Returns
    -------
    list of tuple
        ``(experiment, measurement, x [s], field, observed)`` per residual
        row, in layout order.
    """
    layout = estimator.get_residual_layout()
    return list(zip(layout['experiment'], layout['measurement'],
                    layout['x'], layout['field'], layout['observed']))


def predictions(estimator):
    """Return the accepted model outputs in residual-layout order.

    Parameters
    ----------
    estimator : ParameterEstimation
        Fitted estimator.

    Returns
    -------
    numpy.ndarray
        ``y_model`` [mol/L or K], each experiment's (sample, measurement)
        array flattened measurement-major like the layout; NaN where
        unobserved.
    """
    return np.concatenate([np.asarray(values).T.ravel()
                           for values in estimator.y_model])


def closed_form_by_layout(estimator, rate):
    """Place the closed-form residuals and outputs by the residual layout.

    Parameters
    ----------
    estimator : ParameterEstimation
        Fitted estimator whose layout places the entries.
    rate : float
        Rate constant k [1/s].

    Returns
    -------
    residuals : numpy.ndarray
        ``(model - data) / sigma`` [-] of each observed row (from
        ``case.OBSERVED_ENTRIES``), 0 elsewhere.
    outputs : numpy.ndarray
        Closed-form c_A [mol/L] or T [K] of each observed row, NaN
        elsewhere.
    allowance : numpy.ndarray
        Rounding allowance of each residual [-] (``deriv.rounding``), 0
        elsewhere.
    sensitivity : numpy.ndarray
        Exact rate sensitivity ``d r / d k`` [s] of each observed row, 0
        elsewhere.
    """
    columns = {'c_A': 0, 'T': 1}
    observed = {(experiment, column, time): position
                for position, (experiment, column, time, _, _)
                in enumerate(case.OBSERVED_ENTRIES)}
    closed, jacobian = deriv.closed_form(rise_and_rate(rate))  # [-], [s]
    rounding = deriv.rounding(rise_and_rate(rate))  # [-]
    keys = layout_keys(estimator)
    residuals, outputs, allowance, sensitivity = (
        np.zeros(len(keys)), np.full(len(keys), np.nan),
        np.zeros(len(keys)), np.zeros(len(keys)))
    for row, (experiment, measurement, time, field, flag) in enumerate(keys):
        assert field == columns[measurement]
        position = observed.get((experiment, columns[measurement], time))
        assert flag is (position is not None)
        if position is None:
            continue
        residuals[row] = closed[position]
        allowance[row] = rounding[position]
        sensitivity[row] = jacobian[1, position]  # [s]
        outputs[row] = case.closed_form(rate, experiment, time)[
            columns[measurement]]
    return residuals, outputs, allowance, sensitivity


def snapshot(estimator, experiments):
    """Copy the observations an analysis must not change.

    Parameters
    ----------
    estimator : ParameterEstimation
        Fitted estimator.
    experiments : dict
        Its ``Experiment`` objects.

    Returns
    -------
    dict
        Copies of ``x_data`` [s], ``y_data`` [mol/L or K], and of every
        measurement's ``x``, ``values`` and ``uncertainty`` and every
        experiment's callback arguments.
    """
    measurements = {}
    for name, experiment in experiments.items():
        for label, measurement in experiment.measurements.items():
            measurements[name, label] = tuple(
                np.array(getattr(measurement, entry), copy=True)
                for entry in ('x', 'values', 'uncertainty'))
    return {'x_data': [np.array(item, copy=True)
                       for item in estimator.x_data],
            'y_data': [np.array(item, copy=True)
                       for item in estimator.y_data],
            'measurements': measurements,
            'callback': {name: (experiment.args, dict(experiment.kwargs))
                         for name, experiment in experiments.items()},
            'names': {name: experiment.measurement_names
                      for name, experiment in experiments.items()},
            'params_convg': np.array(estimator.params_convg, copy=True)}


def assert_same_snapshot(actual, expected):
    """Assert two ``snapshot`` results are equal bit for bit.

    Parameters
    ----------
    actual, expected : dict
        ``snapshot`` results.
    """
    for key in ('x_data', 'y_data'):
        assert len(actual[key]) == len(expected[key])
        for value, reference in zip(actual[key], expected[key]):
            case.assert_bits_equal(value, reference)
    assert list(actual['measurements']) == list(expected['measurements'])
    for key, arrays in expected['measurements'].items():
        for value, reference in zip(actual['measurements'][key], arrays):
            case.assert_bits_equal(value, reference)
    assert actual['callback'] == expected['callback']
    assert actual['names'] == expected['names']
    case.assert_bits_equal(actual['params_convg'], expected['params_convg'])


# ------------------------------------------------------- AC2: derivatives


@pytest.mark.parametrize('model, options, source', [
    (case.decay, {'jac_fun': case.rate_jacobian}, 'jac_fun'),
    (case.decay_with_sensitivities, {}, 'model')])
def test_exact_sensitivities_of_the_fit_agree_with_the_reference(
        model, options, source):
    estimator = fit(model, **options)
    check = estimator.check_derivatives()

    assert check.jacobian_source == source
    assert check.parameter_names == ('k',)
    case.assert_bits_equal(check.params, estimator.params_convg)
    assert check.formulas == ('central',)
    np.testing.assert_array_equal(check.observed, deriv.EXPECTED_OBSERVED)
    exact = exact_rate_row(check.params[0])  # [s]
    jacobian = check.jacobian[0, deriv.EXPECTED_OBSERVED]  # [s]
    reference = check.reference[0, deriv.EXPECTED_OBSERVED]  # [s]
    np.testing.assert_allclose(jacobian, exact, rtol=ident.ROUNDOFF_RTOL)
    bound = reference_error_bound(check)  # [s]
    np.testing.assert_array_less(np.abs(reference - exact), bound)
    # ||J - R|| / ||R|| <= ||E|| / (||f'|| - ||E||) for a reference within
    # E of the exact row f' and a Jacobian exact to rounding.
    error = np.linalg.norm(
        bound + ident.ROUNDOFF_RTOL * np.abs(exact))  # [s]
    assert check.relative_error[0] <= error / (np.linalg.norm(exact) - error)


def test_default_forward_differences_show_their_truncation_at_the_fit():
    estimator = fit()
    check = estimator.check_derivatives()

    assert check.jacobian_source == 'finite_difference'
    assert check.parameter_names == ('k',)
    rate = check.params[0]  # [1/s]
    exact = exact_rate_row(rate)  # [s]
    leading, bound = forward_difference_error(rate)  # [s]
    reference_error = reference_error_bound(check)  # [s]
    difference = (check.jacobian[0, deriv.EXPECTED_OBSERVED]
                  - check.reference[0, deriv.EXPECTED_OBSERVED])  # [s]
    np.testing.assert_array_less(np.abs(difference - leading),
                                 bound + reference_error)
    # Relative error of order dx / k = 1e-3, bracketed by the triangle
    # inequality around ||leading|| / ||f'||.
    spread = np.linalg.norm(bound + reference_error)  # [s]
    exact_norm = np.linalg.norm(exact)  # [s]
    reference_norm = np.linalg.norm(reference_error)  # [s]
    low = (np.linalg.norm(leading) - spread) / (
        exact_norm + reference_norm)  # [-]
    high = (np.linalg.norm(leading) + spread) / (
        exact_norm - reference_norm)  # [-]
    assert low <= check.relative_error[0] <= high
    assert high < ident.DEFAULT_RELATIVE_STEP  # O(dx / k), not O(1)


@pytest.mark.parametrize('bounded', [False, True])
def test_fixed_rise_is_not_perturbed_and_bounds_keep_the_domain(bounded):
    """Fixed-parameter case, unbounded and with k half a step from a bound.

    The rise is fixed through ``optimize_flags``; ``deriv.guarded`` raises
    if it is ever moved. After the fit the guard's lower rate bound is
    raised to half a central step below the fitted rate, so the central
    point ``k - h`` would leave the domain and the forward three-point
    stencil must be used.
    """
    calls = []
    lower = np.array([RISE, -np.inf])  # [K*L/mol], [1/s]
    upper = np.array([RISE, np.inf])  # [K*L/mol], [1/s]
    model, jacobian = deriv.guarded(lower, upper, calls)
    estimator = fit(model, seed=(RISE, case.SEED_RATE),
                    experiments=case.experiments_without_rise(),
                    jac_fun=jacobian, optimize_flags=[False, True],
                    name_params=['rise', 'k'])
    rate = estimator.params_convg[0]  # [1/s]
    options, formula = {}, 'central'
    if bounded:
        step = CENTRAL_DIFFERENCE_REL_STEP * rate  # [1/s]
        lower[1] = rate - step / 2  # [1/s], tightens the guard in place
        options, formula = {'bounds': [(lower[1], None)]}, 'forward'
    del calls[:]

    check = estimator.check_derivatives(**options)

    assert check.parameter_names == ('k',)
    assert check.formulas == (formula,)
    evaluated = np.array(calls)  # [K*L/mol], [1/s]
    np.testing.assert_array_equal(evaluated[:, 0], RISE)
    assert np.all(evaluated[:, 1] >= lower[1])
    exact = exact_rate_row(rate)  # [s]
    np.testing.assert_allclose(check.jacobian[0, deriv.EXPECTED_OBSERVED],
                               exact, rtol=ident.ROUNDOFF_RTOL)
    np.testing.assert_array_less(
        np.abs(check.reference[0, deriv.EXPECTED_OBSERVED] - exact),
        reference_error_bound(check, formula))


# ----------------------------------------- AC3: experiment identity, data


@pytest.mark.parametrize('order, hot_order', ORDERS[1:])
def test_reordered_experiments_and_measurements_give_the_same_fit(
        order, hot_order):
    documented = fit()
    reordered = fit(experiments=case.make_experiments(order, hot_order))

    assert list(reordered.get_residual_layout()['experiment'].unique()) == \
        list(order)
    np.testing.assert_allclose(reordered.params_convg,
                               documented.params_convg, rtol=REORDER_RTOL,
                               atol=0)
    np.testing.assert_allclose(reordered.covar_params,
                               documented.covar_params, rtol=REORDER_RTOL,
                               atol=0)
    # The same named rows, each with the same residual and output.
    rows = {key: (residual, output) for key, residual, output in zip(
        layout_keys(documented), documented.info_opt['fun'],
        predictions(documented))}
    keys = layout_keys(reordered)
    assert sorted(keys) == sorted(rows)
    # Each fit's residuals carry their rounding allowance, and any round-off
    # difference of the fitted rates moves a residual by |dr/dk| |dk|.
    _, _, allowance, sensitivity = closed_form_by_layout(
        reordered, reordered.params_convg[0])
    rate_shift = abs(reordered.params_convg[0]
                     - documented.params_convg[0])  # [1/s]
    for key, residual, output, slack, slope in zip(
            keys, reordered.info_opt['fun'], predictions(reordered),
            allowance, sensitivity):
        expected_residual, expected_output = rows[key]
        assert abs(residual - expected_residual) <= (
            2 * slack + abs(slope) * rate_shift)
        np.testing.assert_allclose(output, expected_output,
                                   rtol=REORDER_RTOL)


def test_residuals_and_outputs_are_the_closed_form_placed_by_the_layout():
    estimator = fit()
    rate = estimator.params_convg[0]  # [1/s]
    residuals, outputs, allowance, _ = closed_form_by_layout(estimator,
                                                             rate)

    fun = estimator.info_opt['fun']  # [-]
    observed = estimator.get_residual_layout()['observed'].to_numpy()
    assert observed.sum() == NUM_OBSERVED
    np.testing.assert_array_equal(fun[~observed], 0.0)
    np.testing.assert_array_less(np.abs(fun - residuals)[observed],
                                 allowance[observed])
    accepted = predictions(estimator)  # [mol/L or K]
    np.testing.assert_array_equal(np.isnan(accepted), ~observed)
    np.testing.assert_allclose(accepted[observed], outputs[observed],
                               rtol=case.ROUNDOFF_RTOL,
                               atol=case.ROUNDOFF_ATOL)


def test_repeated_fits_are_bitwise_identical():
    first, second = fit(), fit()
    for name in ('params_convg', 'covar_params'):
        case.assert_bits_equal(getattr(second, name), getattr(first, name))
    for key in ('fun', 'jac'):
        case.assert_bits_equal(second.info_opt[key], first.info_opt[key])
    assert second.info_opt['num_fun_eval'] == first.info_opt['num_fun_eval']


def test_observations_are_unchanged_by_the_analyses():
    experiments = case.make_experiments()
    estimator = fit(experiments=experiments)
    before = snapshot(estimator, experiments)

    estimator.get_confidence_intervals(verbose=False)
    estimator.get_identifiability()
    estimator.check_derivatives()
    statistics = StatisticsClass(estimator)
    previous_state = np.random.get_state()
    try:
        np.random.seed(BOOTSTRAP_SEED)
        boot_params = statistics.bootstrap_params(BOOTSTRAP_SAMPLES)  # [1/s]
    finally:
        np.random.set_state(previous_state)

    assert boot_params.shape == (BOOTSTRAP_SAMPLES, 1)
    assert np.all(np.isfinite(boot_params))
    assert_same_snapshot(snapshot(estimator, experiments), before)


# ------------------------------------------------------ AC4: uncertainty


@pytest.mark.parametrize('include_mse', [True, False])
def test_identifiability_states_the_declared_noise_and_resolves_k(
        include_mse):
    estimator = fit()  # default forward-difference Jacobian
    report = estimator.get_identifiability(include_mse=include_mse)
    rate = estimator.params_convg[0]  # [1/s]

    assert report.noise_model == 'declared_uncertainty'
    assert report.variance_scale_estimated is include_mse
    text = report.noise_assumption
    for fragment in ['Measurement.uncertainty', 'dimensionless [-]',
                     'relative weights' if include_mse
                     else 'treated as absolute', 'Gauss-Newton',
                     "estimator's forward differences"]:
        assert fragment in text, (fragment, text)
    np.testing.assert_array_equal(
        report.covariance, estimator.get_covariance(include_mse=include_mse))

    # Closed form: s2 from the residuals at the accepted k, the information
    # from the forward-difference row bracketed around the exact one.
    residuals, _ = deriv.closed_form(rise_and_rate(rate))  # [-]
    mse = residuals @ residuals / (NUM_OBSERVED - 1)  # [-]
    assert report.residual_mean_square == pytest.approx(
        mse, rel=case.ROUNDOFF_RTOL)
    low, high = forward_row_norm_bracket(rate)  # [s]
    information = report.information_matrix[0, 0]  # [s**2]
    assert low ** 2 <= information <= high ** 2
    scale = mse if include_mse else 1.0  # [-]
    assert report.fisher_information[0, 0] == pytest.approx(
        information / scale, rel=case.ROUNDOFF_RTOL)
    assert scale / high ** 2 <= report.covariance[0, 0] <= scale / low ** 2

    # Error model of the default forward differences: jacobian_rtol = dx /
    # k = 1e-3 plus the rounding term 2 * output_rtol * ||y / sigma|| /
    # (dx ||J||), with the closed-form predictions of the observed entries.
    step = ident.DEFAULT_RELATIVE_STEP * rate  # [1/s]
    np.testing.assert_allclose(report.finite_difference_steps, [step],
                               rtol=ident.STEP_RTOL)
    np.testing.assert_allclose(report.jacobian_rtol,
                               [ident.DEFAULT_RELATIVE_STEP],
                               rtol=ident.STEP_RTOL)
    weighted = [case.closed_form(rate, experiment, time)[column] / sigma
                for experiment, column, time, _, sigma
                in case.OBSERVED_ENTRIES]  # [-]
    error_norm = ident.OUTPUT_RTOL * np.linalg.norm(weighted)  # [-]
    assert report.output_error_norm == pytest.approx(
        error_norm, rel=case.ROUNDOFF_RTOL)
    floor_low = (ident.DEFAULT_RELATIVE_STEP
                 + 2 * error_norm / (step * high))  # [-]
    floor_high = (ident.DEFAULT_RELATIVE_STEP
                  + 2 * error_norm / (step * low))  # [-]
    # Bracket widened by the 1e-12 rounding of the steps and norms.
    assert (floor_low * (1 - ident.STEP_RTOL) <= report.resolution_floor
            <= floor_high * (1 + ident.STEP_RTOL))
    # Verdict: one informed parameter has the unit scaled row, singular
    # value 1, above the floor (< 0.03 here), so k is resolved.
    np.testing.assert_allclose(report.singular_values, [1.0],
                               rtol=case.ROUNDOFF_RTOL)
    assert floor_high < 1.0
    assert report.identifiable is True
    assert report.unresolved_directions == ()
    assert report.unresolved_parameters == ()


# ------------------------------------------------------ AC5: convergence


def test_converged_fit_reports_success_and_its_tolerances():
    estimator = ParameterEstimation(case.decay, [case.SEED_RATE],
                                    case.make_experiments(),
                                    name_params=['k'])
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        _, _, info = estimator.estimate_parameters(method='LM',
                                                   verbose=False)

    assert info['success'] is True
    assert info['message'] in LM_CONVERGED
    assert info['tolerances'] == LM_TOLERANCES
    report = estimator.get_identifiability()
    assert report.convergence['success'] is True
    assert report.convergence['tolerances'] == LM_TOLERANCES
    assert report.summary().splitlines()[1] == (
        f"Convergence: converged ({info['message']}).")


def test_truncated_fit_warns_and_is_reported_as_not_converged():
    experiments = case.make_experiments()
    estimator = ParameterEstimation(case.decay, [case.SEED_RATE],
                                    experiments, name_params=['k'])
    with pytest.warns(RuntimeWarning, match=(
            r"Levenberg-Marquardt \(method='LM'\) stopped without "
            r"converging \(Maximum iterations exceeded\)")):
        _, _, info = estimator.estimate_parameters(
            method='LM', verbose=False,
            max_fun_eval=TRUNCATED_EVALUATIONS)

    assert info['success'] is False
    assert info['message'] == 'Maximum iterations exceeded'
    assert info['tolerances'] == dict(
        LM_TOLERANCES, max_fun_eval=TRUNCATED_EVALUATIONS)
    report = estimator.get_identifiability()
    assert report.convergence['success'] is False
    assert report.summary().splitlines()[0].startswith(
        'FIT DID NOT CONVERGE (Maximum iterations exceeded)')
    # The record keeps the stop criterion, so the outcome stays visible.
    record = case.capture(estimator, experiments)
    assert record.solver.outcome['stop_criterion'] == (
        'Maximum iterations exceeded')
    assert record.solver.options == {'max_fun_eval': TRUNCATED_EVALUATIONS}


# ------------------------------------------------- AC6: record and rerun


def test_estimate_parameters_fit_is_recorded_reloaded_and_rerun():
    experiments = case.make_experiments()
    estimator = ParameterEstimation(case.decay, [case.SEED_RATE],
                                    experiments, name_params=['k'])
    params, covariance, _ = estimator.estimate_parameters(method='LM',
                                                          verbose=False)

    record = case.capture(estimator, experiments)
    loaded = CalibrationRecord.from_json(record.to_json())

    assert loaded == record
    assert loaded.solver.method == 'LM'
    case.assert_bits_equal(loaded.fitted_values, params)
    case.assert_bits_equal(loaded.covariance, covariance)
    for name, experiment in experiments.items():
        reloaded = loaded.experiments[name]
        assert reloaded.measurement_names == experiment.measurement_names
        for label, measurement in experiment.measurements.items():
            case.assert_bits_equal(reloaded.measurements[label].values,
                                   measurement.values)
    rerun = loaded.rerun(case.decay, rtol=case.REPLAY_RTOL,
                         atol=case.REPLAY_ATOL)
    np.testing.assert_allclose(rerun.params_convg, params,
                               rtol=case.REPLAY_RTOL, atol=case.REPLAY_ATOL)
    assert rerun.info_opt['success'] is True
