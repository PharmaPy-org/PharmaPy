"""Reported predictions of multivariate curve resolution (MCR), issue #500.

``optimize_fn`` stores ``y_model``, the predictions that ``StatisticsClass``
uses as the nominal response of its bootstrap samples. MCR spectral residuals
are data minus prediction, so predictions cannot be rebuilt as residuals plus
data; MCR reports the curve-resolved spectra ``C(k) eps`` and the model value
of non-spectral states instead, with NaN where nothing was observed. Bootstrap
samples add drawn spectral residuals (data minus prediction) to the
prediction, as the data do.

Fixtures are the staggered-tracer case of
``test_paramestim_mcr_observation_masks`` (global analysis), the
two-experiment local-analysis case of
``test_paramestim_mcr_objective_consistency`` and, for ordinary estimation,
the staggered two-state case of ``test_paramestim_observation_masks``.
Expected predictions are computed here from ``numpy.linalg.lstsq``
projections and the model functions at the fitted parameters. Runs in about
a second.
"""

import numpy as np
import pytest

from PharmaPy.StatsModule import StatisticsClass
import test_paramestim_mcr_objective_consistency as consistency
import test_paramestim_mcr_observation_masks as case
import test_paramestim_observation_masks as masks


pytestmark = pytest.mark.unit

# Predictions are O(1) [-] and [mol/L]; the estimator's pseudo-inverse and
# the least-squares solve here differ by roundoff of order cond(C) * 1e-16
# with cond(C) below 1e2, so 1e-10 bounds them while the old 2 data -
# prediction differs by twice the O(1e-2) residuals.
ROUNDOFF_RTOL = 1e-10  # [-]
ROUNDOFF_ATOL = 1e-12  # [-] and [mol/L], for near-zero predictions
BOOTSTRAP_SAMPLES = 5  # [-], enough to exercise the sample layout
BOOTSTRAP_SEED = 500  # [-], fixed draws


def _expected_predictions(rate, global_analysis):
    """Return independent MCR predictions at a rate.

    Parameters
    ----------
    rate : numpy.ndarray
        Rate k, shape ``(1,)`` [1/s].
    global_analysis : bool
        One least-squares absorptivity matrix for all experiments, or one
        per experiment.

    Returns
    -------
    list of numpy.ndarray
        Per experiment of ``case.SCHEDULES``, shape ``(n_times, 3)``:
        projected absorbance at two wavelengths [-] and the tracer model
        value [mol/L], NaN at rows where the tracer was not measured.
    """
    data = (case._data() if global_analysis else consistency._local_data())
    states = [case._states(rate, time_s)
              for time_s, _ in case.SCHEDULES.values()]  # [mol/L]
    spectra = [data[name][0] for name in case.SCHEDULES]  # [-]
    if global_analysis:
        absorptivity, *_ = np.linalg.lstsq(
            np.vstack([state[:, :2] for state in states]),
            np.vstack(spectra), rcond=None)  # [L/mol]
        absorptivities = [absorptivity] * len(states)
    else:
        absorptivities = [
            np.linalg.lstsq(state[:, :2], spectrum, rcond=None)[0]
            for state, spectrum in zip(states, spectra)]  # [L/mol]

    expected = []
    for state, absorptivity, (_, rows) in zip(
            states, absorptivities, case.SCHEDULES.values()):
        tracer = np.full(state.shape[0], np.nan)  # [mol/L]
        tracer[rows] = state[rows, 2]
        expected.append(np.column_stack((state[:, :2] @ absorptivity,
                                         tracer)))
    return expected


def _build(global_analysis, model):
    """Build the global (staggered) or local MCR estimator.

    Parameters
    ----------
    global_analysis : bool
        Select ``case._estimator`` or ``consistency._local_estimator``.
    model : callable
        Model with sensitivities or states only.

    Returns
    -------
    MultipleCurveResolution
        Estimator seeded at ``case.TRIAL_RATE``.
    """
    if global_analysis:
        return case._estimator(model, 'correlated')
    return consistency._local_estimator(model)


@pytest.mark.parametrize('model', consistency.MODELS,
                         ids=consistency.MODEL_IDS)
@pytest.mark.parametrize('global_analysis', [True, False],
                         ids=['global', 'local'])
def test_mcr_y_model_is_the_curve_resolved_prediction(global_analysis,
                                                      model):
    estimator = _build(global_analysis, model)

    fitted, _, _ = estimator.optimize_fn(verbose=False)

    expected = _expected_predictions(fitted, global_analysis)
    assert len(estimator.y_model) == len(expected)
    for prediction, reference in zip(estimator.y_model, expected):
        assert prediction.shape == reference.shape
        np.testing.assert_array_equal(np.isnan(prediction),
                                      np.isnan(reference))
        np.testing.assert_allclose(prediction, reference,
                                   rtol=ROUNDOFF_RTOL, atol=ROUNDOFF_ATOL)


def test_bootstrap_nominal_is_the_mcr_prediction():
    estimator = _build(True, case.model_with_sensitivities)
    fitted, _, _ = estimator.optimize_fn(verbose=False)
    expected = _expected_predictions(fitted, True)

    statistics = StatisticsClass(estimator)
    np.random.seed(BOOTSTRAP_SEED)
    samples = statistics.get_bootsamples(BOOTSTRAP_SAMPLES,
                                         fix_initial=True)

    data = case._data()
    for nominal, reference, runs, name in zip(
            statistics.y_nominal, expected, samples, case.SCHEDULES):
        np.testing.assert_allclose(nominal, reference, rtol=ROUNDOFF_RTOL,
                                   atol=ROUNDOFF_ATOL)
        # Spectral data = prediction + residual, so a sample is the
        # prediction plus a residual drawn from the same channel.
        spectral_residual = data[name][0] - reference[:, :2]  # [-]
        for sample in runs:
            # fix_initial keeps the nominal first row in every sample.
            np.testing.assert_allclose(sample[0], reference[0],
                                       rtol=ROUNDOFF_RTOL,
                                       atol=ROUNDOFF_ATOL)
            for channel in range(2):
                drawn = sample[1:, channel] - reference[1:, channel]  # [-]
                pool = spectral_residual[1:, channel]  # [-]
                distance = np.abs(drawn[:, np.newaxis] - pool).min(axis=1)
                assert distance.max() <= ROUNDOFF_ATOL


def test_ordinary_y_model_is_the_model_output():
    """Ordinary estimation keeps ``resid_runs + y_data`` exactly."""
    estimator = masks._estimator('correlated', 'jac_fun')

    fitted, _, _ = estimator.optimize_fn(verbose=False)

    for ind, prediction in enumerate(estimator.y_model):
        np.testing.assert_array_equal(
            prediction, estimator.resid_runs[ind] + estimator.y_data[ind])
        model = masks.two_state_model(fitted, estimator.x_model[ind])
        observed = estimator.x_masks[ind]  # [-], bool
        assert observed is not None and not observed.all()
        np.testing.assert_allclose(prediction[observed], model[observed],
                                   rtol=ROUNDOFF_RTOL, atol=ROUNDOFF_ATOL)
        assert np.all(np.isnan(prediction[~observed]))
