"""Rank and explain next experiments on the analytic calibration stand-in.

The synthetic hot/cold decay fit has an asymmetric two-parameter Jacobian,
mixed measurement units, and known independent or correlated prospective
errors. Small numerical boundary cases use the public report constructor.
All tests run in the core lane without an external design or ODE solver.
"""

import numpy as np
import pytest

import test_paramestim_derivative_check as deriv
import test_paramestim_rank_deficient_covariance as deficient
import test_records_calibration as case
from PharmaPy.ExperimentDesign import (
    CandidateExperiment, pydex_prior, rank_experiments)
from PharmaPy.ParamEstim import Experiment, ParameterEstimation
from PharmaPy.StatsModule import IdentifiabilityReport

pytestmark = pytest.mark.unit

# Reverse calibration order in the input mapping to exercise keyed units.
UNITS = {'k': '1/s', 'rise': 'K*L/mol'}
INITIAL = case.CALLBACK['hot'][0]  # [mol/L], documented hot decay condition
TIMES = np.array([1.0, 2.0, 4.0])  # [s], three nonzero hot-run times
CONC_STD = case.HOT_CA_STD  # [mol/L]
TEMP_STD = case.HOT_T_STD  # [K]
# A shared concentration readout error correlates the two concentration
# samples. Use 1/2 to keep the determinant away from zero.
CORRELATION = 0.5  # [-]
CORRELATED_NOISE = np.array([
    [CONC_STD**2, CORRELATION * CONC_STD**2, 0],
    [CORRELATION * CONC_STD**2, CONC_STD**2, 0],
    [0, 0, TEMP_STD**2],
])  # [observation_i * observation_j], (c_A, c_A, T)
# Stability of the small full-rank Gram matrices: the reference and
# whitening/SVD methods incur several float64 operations per entry and
# the matrices are well conditioned in their Jacobi-scaled coordinates.
RTOL = case.ROUNDOFF_RTOL  # [-]


def mixed_sensitivities(point):
    """Evaluate exact sensitivities on the three-observation plan.

    Parameters
    ----------
    point : mapping
        Named fitted rise [K*L/mol] and rate k [1/s].

    Returns
    -------
    dict
        Three entries per parameter for c_A at 1 and 2 s and T at 4 s.
        Columns are in [observation unit / parameter unit].
    """
    derivatives = deriv.ident.rise_first_jacobian(
        [point['rise'], point['k']], TIMES, INITIAL)  # [observation unit / p]
    # Jacobian blocks are c_A then T. Pick the first two c_A and last T.
    selected = derivatives[[0, 1, -1]]  # [observation unit / p]
    return {'k': selected[:, 1], 'rise': selected[:, 0]}


def independent_information(fit):
    """Derive baseline, candidate and scale using closed-form decay.

    Parameters
    ----------
    fit : IdentifiabilityReport
        Accepted two-parameter decay fit in (rise [K*L/mol], k [1/s]) order.

    Returns
    -------
    baseline, increment : numpy.ndarray
        Two by two information matrices before variance scaling,
        with entries [1 / (p_i * p_j)].
    mse : float
        Residual mean square [-] from the documented calibration samples.
    """
    residuals, rows = deriv.closed_form(fit.estimates)  # [-], [1 / p_i]
    mse = np.mean(residuals**2) * len(residuals) / (len(residuals) - len(UNITS))  # [-]
    # Closed form for the inverse of the correlated 2x2 concentration block.
    # Only k affects concentration. Temperature contributes an independent
    # outer product of the rise and k derivatives.
    rise, rate = fit.estimates  # [K*L/mol], [1/s]
    remaining = INITIAL * np.exp(-rate * TIMES)  # [mol/L]
    early, later, temp_rate_term = TIMES * remaining  # [mol/L*s]
    concentration_info = (
        (early**2 + later**2 - 2 * CORRELATION * early * later)
        / (CONC_STD**2 * (1 - CORRELATION**2)))  # [s**2]
    temp_rise = (INITIAL - remaining[-1]) / TEMP_STD  # [mol/(K*L)]
    temp_rate = rise * temp_rate_term / TEMP_STD  # [s]
    increment = np.array([
        [temp_rise**2, temp_rise * temp_rate],
        [temp_rise * temp_rate, temp_rate**2 + concentration_info],
    ])  # [1 / (p_i * p_j)]
    return rows @ rows.T, increment, mse


@pytest.mark.parametrize('include_mse', [False, True])
def test_ranking_uses_pooled_information_and_preserves_parameter_coordinates(
        include_mse):
    fit = deriv.fitted_estimator().get_identifiability(include_mse=include_mse)
    baseline, increment, mse = independent_information(fit)  # [1/(p_i*p_j)], same, [-]
    baseline /= mse if include_mse else 1  # [1 / (p_i * p_j)]
    # Doubling each measurement's standard deviation quarters its Fisher
    # contribution, with no change to the candidate's raw sensitivities.
    lower_precision = 4 * CORRELATED_NOISE  # [observation_i * observation_j]
    candidates = [
        CandidateExperiment('z-tied', mixed_sensitivities, CORRELATED_NOISE),
        CandidateExperiment('less precise', mixed_sensitivities, lower_precision),
        CandidateExperiment('a-tied', mixed_sensitivities, CORRELATED_NOISE),
    ]
    # Reorder both observations and covariance axes, and swap returned keys.
    order = [2, 0, 1]

    def permuted(point):
        """Return the same derivatives [observation unit / p] reordered."""
        original = mixed_sensitivities(point)  # [observation unit / p]
        return {name: original[name][order] for name in ('rise', 'k')}

    reordered = CandidateExperiment('permuted', permuted,
                                    CORRELATED_NOISE[np.ix_(order, order)])
    design = rank_experiments(fit, candidates, parameter_units=UNITS)
    replay = rank_experiments(fit, candidates[::-1], parameter_units=UNITS)
    reordered_design = rank_experiments(fit, [reordered], parameter_units=UNITS)
    assert design.fit.parameter_names == ('rise', 'k')
    assert design.parameter_units == ('K*L/mol', '1/s')
    assert [result.name for result in design.candidates] == [
        'a-tied', 'z-tied', 'less precise']
    assert [result.name for result in replay.candidates] == [
        result.name for result in design.candidates]
    np.testing.assert_array_equal([r.score for r in replay.candidates],
                                  [r.score for r in design.candidates])

    for result, expected in zip(
            (*design.candidates, reordered_design.candidates[0]),
            (increment, increment, increment / 4, increment)):
        assert result.status == 'ranked'
        assert result.candidate_rank == result.combined_rank == 2
        np.testing.assert_allclose(result.added_information, expected,
                                   rtol=RTOL, atol=0)
        # The determinant of a 2x2 matrix is the product of its diagonal
        # minus the squared off-diagonal. NumPy's determinant here uses LU.
        pooled_det = np.linalg.det(baseline + expected)  # [1 / (rise * k)**2]
        assert result.score == pytest.approx(np.log(pooled_det), rel=RTOL)
        assert result.log_information_gain == pytest.approx(
            np.log(pooled_det / np.linalg.det(baseline)), rel=RTOL)


def test_relative_candidate_uses_calibration_scale_without_rescaling_absolute():
    estimator = deriv.fitted_estimator()
    fit = estimator.get_identifiability(include_mse=True)
    _, increment, mse = independent_information(fit)  # [1/(p_i*p_j)], [-]
    assert 0 < mse < 1  # Measured spread of the synthetic calibration.
    candidates = [
        CandidateExperiment('absolute', mixed_sensitivities, CORRELATED_NOISE),
        CandidateExperiment('relative', mixed_sensitivities, CORRELATED_NOISE,
                            relative_noise=True),
    ]
    design = rank_experiments(fit, candidates, parameter_units=UNITS)
    relative, absolute = design.candidates
    assert (relative.name, absolute.name) == ('relative', 'absolute')
    np.testing.assert_allclose(relative.added_information, increment / mse,
                               rtol=RTOL, atol=0)
    np.testing.assert_allclose(absolute.added_information, increment,
                               rtol=RTOL, atol=0)
    absolute_fit = estimator.get_identifiability(include_mse=False)
    invalid = rank_experiments(absolute_fit, candidates, parameter_units=UNITS)
    assert invalid.candidates[-1].status == 'failed'
    assert 'requires an estimated calibration variance scale' in (
        invalid.candidates[-1].diagnostic)


def test_candidate_rank_one_can_complete_a_rank_one_calibration():
    # Omit temperature from the calibration. Concentration depends on k
    # alone, so a new T observation at t > 0 resolves the unfitted direction.
    experiments = {
        name: Experiment({'c_A': exp.measurements['c_A']},
                         args=exp.args, kwargs=exp.kwargs)
        for name, exp in case.experiments_without_rise().items()
    }
    estimator = ParameterEstimation(
        case.decay_rise_first, list(deriv.SEED), experiments,
        name_params=list(deriv.NAMES), jac_fun=deriv.ident.rise_first_jacobian)
    with pytest.warns(RuntimeWarning, match=deficient.UNDEFINED):
        estimator.estimate_parameters(method='LM', verbose=False)
    fit = estimator.get_identifiability(include_mse=False)

    def concentration(point):
        """Return one c_A derivative row [mol/L / p] from the mixed plan."""
        return {name: values[:1]
                for name, values in mixed_sensitivities(point).items()}

    def temperature(point):
        """Return one T derivative row [K / p] from the mixed plan."""
        return {name: values[-1:]
                for name, values in mixed_sensitivities(point).items()}

    candidates = [
        CandidateExperiment('concentration', concentration, [[CONC_STD**2]]),
        CandidateExperiment('temperature', temperature, [[TEMP_STD**2]]),
    ]
    design = rank_experiments(fit, candidates, parameter_units=UNITS)
    temperature_result, concentration_result = design.candidates
    assert design.baseline_rank == 1
    assert temperature_result.name == 'temperature'
    assert temperature_result.status == 'ranked'
    assert temperature_result.candidate_rank == 1
    assert temperature_result.combined_rank == 2
    assert temperature_result.log_information_gain is None
    # For diag(0,B) plus h h^T, det = B*h_rise^2 (all mixed terms cancel).
    _, rate = fit.estimates  # [K*L/mol], [1/s]
    baseline_k = sum(
        (case.CALLBACK[run][0] * time * np.exp(-rate * time) / sigma)**2
        for run, column, time, _, sigma in case.OBSERVED_ENTRIES if column == 0
    )  # [s**2]
    h_rise = INITIAL * (1 - np.exp(-rate * TIMES[-1])) / TEMP_STD  # [mol/(K*L)]
    assert temperature_result.score == pytest.approx(
        np.log(baseline_k * h_rise**2), rel=RTOL)
    assert concentration_result.status == 'rank_deficient'
    assert concentration_result.combined_rank == 1
    assert concentration_result.score is None


def test_failed_candidates_have_context_and_do_not_interrupt_ranking():
    fit = deriv.fitted_estimator().get_identifiability(include_mse=False)

    def outside_domain(point):
        """Signal a failed plan at the named fit [1/s], [K*L/mol]."""
        raise RuntimeError(f"no feasible trajectory at k={point['k']:.3f} 1/s")

    def incomplete(point):
        """Return only the rate column [observation unit * s]."""
        return {'k': mixed_sensitivities(point)['k']}

    bad_covariance = CORRELATED_NOISE.copy()  # [observation_i * observation_j]
    # Negative temperature variance violates positive definiteness.
    bad_covariance[-1, -1] *= -1
    candidates = [
        CandidateExperiment('domain', outside_domain, CORRELATED_NOISE),
        CandidateExperiment('missing', incomplete, CORRELATED_NOISE),
        CandidateExperiment('noise', mixed_sensitivities, bad_covariance),
        CandidateExperiment('valid', mixed_sensitivities, CORRELATED_NOISE),
    ]
    design = rank_experiments(fit, candidates, parameter_units=UNITS)
    assert [result.name for result in design.candidates] == [
        'valid', 'domain', 'missing', 'noise']
    domain, missing, noise = design.candidates[1:]
    assert domain.diagnostic == (
        f"Sensitivity evaluation: RuntimeError: no feasible trajectory "
        f"at k={fit.estimates[1]:.3f} 1/s")
    assert 'keyed by exactly the fitted parameters' in missing.diagnostic
    assert 'Noise covariance must be positive definite' in noise.diagnostic
    for failure in (domain, missing, noise):
        assert failure.status == 'failed'
        assert failure.added_information is failure.combined_rank is failure.score is None
    assert design.candidates[0].log_information_gain > 0


def orthogonal_fit(sensitivity=1.0, *, estimate=1.0, include_mse=False,
                   success=True):
    """Build a two-direction report for isolated numerical edge cases.

    Parameters
    ----------
    sensitivity : float or array_like
        Weighted derivative [1 / p_i], common or one per independent direction.
    estimate : float
        Nominal value [p_i] in each coordinate.
    include_mse : bool
        Request variance scaling on the exact, zero-residual case.
    success : bool or numpy.bool_
        Fit convergence flag.

    Returns
    -------
    IdentifiabilityReport
        Report on three weighted observations, one for each parameter and
        one at a point insensitive to both. The Fisher diagonal is the
        squared sensitivity in the absolute noise case [1 / p_i**2].
    """
    jacobian = np.asarray(sensitivity).reshape(-1, 1) * np.eye(2, 3)  # [1 / p_i]
    return IdentifiabilityReport.from_fit(
        jacobian, np.zeros(3), [estimate, estimate], ('rise', 'k'), 3,
        noise_model='declared_uncertainty', measurement_units={'y': '[-]'},
        jacobian_source='jac_fun', jacobian_rtol=np.finfo(float).eps,
        include_mse=include_mse, convergence={'success': success})


@pytest.mark.parametrize('baseline_sensitivity, added_sensitivity', [
    # The tiny increment resolves positive gain even if log det rounds
    # to the baseline score. The large increment must retain both ranks
    # even if entries of the explicitly added Gram matrix lose F0.
    (1e75, 1e68), (1.0, 1e9),  # [1 / p_i], [1 / p_i]
])
def test_gain_survives_rounded_determinants(baseline_sensitivity,
                                            added_sensitivity):
    fit = orthogonal_fit(baseline_sensitivity, success=np.bool_(True))

    def collinear(point):
        """Return a one-observation sensitivity [1 / p_i] in both directions."""
        return {name: [added_sensitivity] for name in point}

    def no_information(point):
        """Return a zero derivative [1 / p_i] for each calibrated direction."""
        return {name: [0.0] for name in point}

    candidates = [
        CandidateExperiment('alphabetically first', no_information, [[1.0]]),
        CandidateExperiment('informative', collinear, [[1.0]]),
    ]  # Unit variance for one standardized observation [-].
    best, no_gain = rank_experiments(fit, candidates, parameter_units=UNITS).candidates
    assert best.name == 'informative' and best.status == 'ranked'
    assert best.candidate_rank == 1 and best.combined_rank == 2
    assert best.log_information_gain == pytest.approx(
        np.log1p(2 * (added_sensitivity / baseline_sensitivity)**2),
        rel=RTOL, abs=0)
    assert no_gain.log_information_gain == pytest.approx(0, abs=0)


def test_rank_repair_retains_directions_resolved_by_a_weaker_full_rank_plan():
    """Pool the same two sources in either order without losing a direction."""
    large = 1e8  # [1 / p_i], large enough to hide I under the pooled rank cutoff
    repeated = [[large, large]]  # [1 / p_i], one observed linear combination
    independent = np.eye(2)  # [1 / p_i], one observation per coordinate
    expected = np.log1p(2 * large**2)  # [-], log determinant of I + h h.T
    for baseline, prospective in ((repeated, independent),
                                  (independent, repeated)):
        # Pad to three observed rows with zero sensitivities at insensitive
        # sample points, giving a positive residual degrees of freedom.
        rows = np.asarray(baseline)  # [1 / p_i]
        jacobian = np.pad(rows.T, ((0, 0), (0, 3 - len(rows))))  # [1 / p_i]
        fit = IdentifiabilityReport.from_fit(
            jacobian, np.zeros(3), [1.0, 1.0], ('rise', 'k'), 3,
            noise_model='declared_uncertainty', measurement_units={'y': '[-]'},
            jacobian_source='jac_fun', jacobian_rtol=np.finfo(float).eps,
            include_mse=False, convergence={'success': True})
        prospective_rows = np.asarray(prospective)  # [1 / p_i]

        def sensitivities(point):
            """Return the standardized sensitivity columns [1 / p_i]."""
            return dict(zip(point, prospective_rows.T))

        candidate = CandidateExperiment('next', sensitivities,
                                        np.eye(len(prospective_rows)))  # noise [-]
        result, = rank_experiments(fit, [candidate],
                                   parameter_units=UNITS).candidates
        assert result.status == 'ranked'
        assert result.combined_rank == 2
        assert result.score == pytest.approx(expected, rel=RTOL)

        if len(rows) == 1:
            # More samples in the already informed direction do not repair
            # rank. Use a large value to reveal spurious projection leakage.
            same_direction = 1e16  # [1 / p_i]

            def collinear(point):
                """Return another derivative [1 / p_i] along the prior row."""
                return {name: [same_direction] for name in point}

            redundant = CandidateExperiment('redundant', collinear, [[1.0]])
            unresolved, = rank_experiments(
                fit, [redundant], parameter_units=UNITS).candidates
            assert unresolved.status == 'rank_deficient'
            assert unresolved.combined_rank == 1


def test_roundoff_in_repeated_observations_does_not_inflate_gain():
    """Use the exact rank-one determinants to disambiguate two forecasts."""
    fit = orthogonal_fit()
    large = 1e16  # [1 / p_i], null singular-value roundoff is comparable to I
    competing_norm_squared = 5  # [-], between two repeats' factor 4 and a false gain

    def repeated(point):
        """Return identical sensitivity rows [1 / p_i] for both observations."""
        return {name: [large, large] for name in point}

    def single(point):
        """Return one larger sensitivity [1 / p_i] on the rise coordinate."""
        return {'rise': [np.sqrt(competing_norm_squared) * large], 'k': [0.0]}

    candidates = [
        CandidateExperiment('repeated', repeated, np.eye(2)),
        CandidateExperiment('single', single, [[1.0]]),
    ]  # Unit noise variances [-]
    best, other = rank_experiments(fit, candidates, parameter_units=UNITS).candidates
    assert (best.name, other.name) == ('single', 'repeated')
    assert best.candidate_rank == other.candidate_rank == 1
    assert best.log_information_gain == pytest.approx(
        np.log1p(competing_norm_squared * large**2), rel=RTOL)
    # Two independent repeats along (1, 1) give an eigenvalue 4*large**2.
    assert other.log_information_gain == pytest.approx(
        np.log1p(4 * large**2), rel=RTOL)


def test_pooled_ranking_prefers_the_weaker_direction_independently_of_units():
    # Information diag(16, 1). A larger candidate sensitivity in the
    # better-known direction has smaller gain than the weaker direction.
    baseline = np.array([4.0, 1.0])  # [1 / p_i]
    prospective = {'rise only': np.array([3.0, 0.0]),
                   'rate only': np.array([0.0, 2.0])}  # [1 / p_i]
    # Rescale k from reciprocal seconds to reciprocal minutes. The
    # derivative per 1/min is 1/60 of the derivative per 1/s.
    seconds_per_minute = 60  # [s/min]
    converted_units = dict(UNITS, k='1/min')
    for rate_factor, units in ((1.0, UNITS),
                               (1 / seconds_per_minute, converted_units)):
        coordinate_factor = np.array([1.0, rate_factor])  # [old p_i / new p_i]
        fit = orthogonal_fit(baseline * coordinate_factor)

        def evaluate_direction(direction):
            """Bind a standardized sensitivity vector [1 / p_i]."""
            def evaluate(point):
                """Return one derivative per named fitted coordinate [1 / p_i]."""
                return {name: [value] for name, value in zip(point, direction)}

            return evaluate

        candidates = [
            CandidateExperiment(name, evaluate_direction(values * coordinate_factor),
                                [[1.0]])  # Standardized observation variance [-].
            for name, values in prospective.items()
        ]
        best, other = rank_experiments(fit, candidates, parameter_units=units).candidates
        assert (best.name, other.name) == ('rate only', 'rise only')
        assert best.candidate_rank == other.candidate_rank == 1
        assert best.combined_rank == other.combined_rank == 2
        assert best.log_information_gain == pytest.approx(np.log(5), rel=RTOL)
        assert other.log_information_gain == pytest.approx(np.log(25 / 16), rel=RTOL)


def test_complex_object_values_do_not_silently_become_real():
    fit = orthogonal_fit()

    def invalid(point):
        """Construct a complex derivative [1 / p_i] in an object array."""
        value = np.array([np.complex128(1 + 1j)], dtype=object)  # [1 / p_i]
        return dict.fromkeys(point, value)

    candidate = CandidateExperiment('complex', invalid, [[1.0]])  # variance [-]
    result, = rank_experiments(fit, [candidate], parameter_units=UNITS).candidates
    assert result.status == 'failed'
    assert 'must be a real observation vector' in result.diagnostic


def test_pydex_prior_preserves_parameter_order_and_absolute_fisher_basis():
    fit = deriv.fitted_estimator().get_identifiability(include_mse=True)
    baseline, _, mse = independent_information(fit)  # [1/(p_i*p_j)], [-]
    normalized, theta = pydex_prior(fit)  # [-], [K*L/mol], [1/s]
    rise, rate = fit.estimates  # [K*L/mol], [1/s]
    expected = np.array([
        [baseline[0, 0] * rise**2, baseline[0, 1] * rise * rate],
        [baseline[1, 0] * rate * rise, baseline[1, 1] * rate**2],
    ]) / mse  # [-]
    np.testing.assert_allclose(normalized, expected, rtol=RTOL, atol=0)
    np.testing.assert_array_equal(theta, fit.estimates)
    assert not np.shares_memory(theta, fit.estimates)


def test_missing_fisher_information_and_nonconvergence_are_distinct():
    candidate = CandidateExperiment('valid', mixed_sensitivities, CORRELATED_NOISE)
    with pytest.raises(ValueError, match='Fisher information is unavailable'):
        rank_experiments(orthogonal_fit(include_mse=True), [candidate],
                         parameter_units=UNITS)
    with pytest.raises(ValueError, match='requires a fit with reported convergence'):
        rank_experiments(orthogonal_fit(success=False), [candidate],
                         parameter_units=UNITS)
    with pytest.raises(ValueError, match='requires nonzero estimates'):
        pydex_prior(orthogonal_fit(estimate=0.0))
