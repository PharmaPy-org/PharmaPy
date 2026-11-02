"""Regression tests for PCR prediction diagnostics (#253).

Scope: ``PCR_calibration.predict(..., full_output=True, y_observed=...)``.
The ``'MSE'`` diagnostic must pair each new prediction with the observed
response of the same new row, never with the training responses, and
``predict`` must not print. Fixtures are synthetic linear absorbance models
(absorbance [AU] = concentration [g/L] x pure-component spectrum [AU*L/g]
plus fixed noise [AU]) on four wavelengths, with one or two calibrated
responses; expected MSE values are computed independently with ``np.mean``.
"""

import numpy as np
import pytest

from PharmaPy.Calibration import PCR_calibration

pytestmark = pytest.mark.unit

PURE_A = np.array([0.8, 0.5, 0.2, 0.1])  # [AU*L/g]
PURE_B = np.array([0.1, 0.3, 0.6, 0.4])  # [AU*L/g]
# Fixed deterministic training noise, shape (6, 4) [AU].
TRAIN_NOISE = np.array([
    [0.01, -0.02, 0.005, 0.0], [-0.01, 0.0, 0.01, -0.005],
    [0.0, 0.01, -0.01, 0.002], [0.02, -0.01, 0.0, 0.004],
    [-0.015, 0.005, 0.0, -0.002], [0.005, 0.0, 0.003, 0.001]])
CONC_A_TRAIN = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])  # [g/L]
CONC_B_TRAIN = np.array([1.2, 0.4, 0.9, 0.2, 0.7, 1.5])  # [g/L]

# Six training rows: equal-row-count predictions previously broadcast
# against the training responses without error.
NUM_TRAIN = len(CONC_A_TRAIN)  # [-]

# MSE agreement: the same squared residuals summed in a different order;
# a relative bound far above double-precision round-off.
MSE_RTOL = 1e-10  # [-]


def _single_response_calibration():
    """Return a one-component PCR calibration of species A.

    Returns
    -------
    PCR_calibration
        Fitted to six spectra [AU] and concentrations ``CONC_A_TRAIN``
        [g/L].
    """
    spectra = np.outer(CONC_A_TRAIN, PURE_A) + TRAIN_NOISE  # [AU]
    calibration = PCR_calibration(spectra, num_comp=1)
    calibration.get_regression(CONC_A_TRAIN)
    return calibration


def _two_response_calibration():
    """Return a two-component PCR calibration of species A and B.

    Returns
    -------
    PCR_calibration
        Fitted to six spectra [AU] and concentrations of A and B, shape
        ``(6, 2)`` [g/L].
    """
    spectra = (np.outer(CONC_A_TRAIN, PURE_A)
               + np.outer(CONC_B_TRAIN, PURE_B) + TRAIN_NOISE)  # [AU]
    calibration = PCR_calibration(spectra, num_comp=2)
    calibration.get_regression(
        np.column_stack((CONC_A_TRAIN, CONC_B_TRAIN)), num_comp=2)
    return calibration


@pytest.mark.parametrize("conc_new", [
    np.array([0.7, 1.7, 2.7]),  # fewer rows than training [g/L]
    np.linspace(0.5, 3.0, 8),  # more rows than training [g/L]
    CONC_A_TRAIN[::-1].copy(),  # same rows as training, reversed [g/L]
    np.full(NUM_TRAIN, 1.0),  # equal row count, other values [g/L]
], ids=["fewer", "more", "reversed", "equal-count"])
def test_mse_pairs_predictions_with_new_observations(conc_new):
    calibration = _single_response_calibration()
    # Noise-free new spectra with measured concentrations offset by a known
    # amount, so the MSE is dominated by a value the test controls.
    spectra_new = np.outer(conc_new, PURE_A)  # [AU]
    observed_offset = np.linspace(-0.05, 0.05, len(conc_new))  # [g/L]
    observed = conc_new + observed_offset  # [g/L]

    info = calibration.predict(spectra_new, full_output=True,
                               y_observed=observed)

    y_pred = info["y_pred"]
    assert y_pred.shape == (len(conc_new), 1)
    expected_mse = np.mean((y_pred.ravel() - observed) ** 2)  # [(g/L)**2]
    assert info["MSE"].shape == (1, 1)
    assert info["MSE"].item() == pytest.approx(expected_mse, rel=MSE_RTOL)

    # A 1-D and a column-shaped observation vector are equivalent.
    info_col = calibration.predict(spectra_new, full_output=True,
                                   y_observed=observed[:, np.newaxis])
    np.testing.assert_allclose(info_col["MSE"], info["MSE"], rtol=0.0,
                               atol=0.0)


def test_mse_is_invariant_to_consistent_row_reordering():
    calibration = _single_response_calibration()
    conc_new = np.array([0.6, 1.4, 2.1, 2.9, 1.1, 2.4])  # [g/L], 6 rows
    observed = conc_new + np.array(
        [0.01, -0.02, 0.03, -0.01, 0.02, 0.0])  # [g/L]
    spectra_new = np.outer(conc_new, PURE_A)  # [AU]
    order = np.array([3, 0, 5, 1, 4, 2])

    forward = calibration.predict(spectra_new, full_output=True,
                                  y_observed=observed)
    permuted = calibration.predict(spectra_new[order], full_output=True,
                                   y_observed=observed[order])

    np.testing.assert_allclose(permuted["y_pred"], forward["y_pred"][order],
                               rtol=MSE_RTOL)
    assert permuted["MSE"].item() == pytest.approx(forward["MSE"].item(),
                                                   rel=MSE_RTOL)
    expected = np.mean((forward["y_pred"].ravel() - observed) ** 2)
    assert forward["MSE"].item() == pytest.approx(expected, rel=MSE_RTOL)


def test_mse_matrix_for_two_responses():
    calibration = _two_response_calibration()
    conc_new = np.array([[0.9, 0.3], [2.2, 1.1], [1.6, 0.6]])  # [g/L]
    spectra_new = conc_new @ np.vstack((PURE_A, PURE_B))  # [AU]
    observed = conc_new + np.array(
        [[0.02, -0.01], [-0.03, 0.02], [0.01, 0.04]])  # [g/L]

    info = calibration.predict(spectra_new, full_output=True,
                               y_observed=observed)

    resid = info["y_pred"] - observed  # [g/L]
    num_entries = resid.size  # [-]
    expected = np.array([
        [np.sum(resid[:, i] * resid[:, j]) / num_entries for j in range(2)]
        for i in range(2)])  # [(g/L)**2]
    np.testing.assert_allclose(info["MSE"], expected, rtol=MSE_RTOL)
    assert np.trace(info["MSE"]) == pytest.approx(np.mean(resid ** 2),
                                                  rel=MSE_RTOL)


def test_training_rows_reproduce_training_diagnostics():
    calibration = _single_response_calibration()
    spectra = np.outer(CONC_A_TRAIN, PURE_A) + TRAIN_NOISE  # [AU]

    info = calibration.predict(spectra, full_output=True,
                               y_observed=CONC_A_TRAIN)
    mse_by_comp, resid_by_comp = calibration.evaluate_mse(num_comp=1)

    np.testing.assert_allclose(info["y_pred"],
                               CONC_A_TRAIN[:, None] - calibration.residuals,
                               rtol=MSE_RTOL)
    np.testing.assert_allclose(resid_by_comp[0], calibration.residuals,
                               rtol=MSE_RTOL)
    assert info["MSE"].item() == pytest.approx(mse_by_comp[0], rel=MSE_RTOL)


def test_mse_is_none_without_observations():
    calibration = _single_response_calibration()
    spectra_new = np.outer(CONC_A_TRAIN, PURE_A)  # [AU], 6 rows

    info = calibration.predict(spectra_new, full_output=True)

    assert set(info) == {"x_projected", "y_pred", "MSE"}
    assert info["MSE"] is None
    np.testing.assert_allclose(info["y_pred"],
                               calibration.predict(spectra_new),
                               rtol=0.0, atol=0.0)


def test_observed_shape_errors_name_observed_and_expected_shapes():
    single = _single_response_calibration()
    spectra_new = np.outer([0.7, 1.7, 2.7], PURE_A)  # [AU], 3 rows

    with pytest.raises(ValueError,
                       match=r"shape \(4,\); expected \(3, 1\) or \(3,\)"):
        single.predict(spectra_new, full_output=True,
                       y_observed=np.ones(4))
    with pytest.raises(ValueError,
                       match=r"shape \(3, 2\); expected \(3, 1\) or \(3,\)"):
        single.predict(spectra_new, full_output=True,
                       y_observed=np.ones((3, 2)))
    with pytest.raises(ValueError, match="only used with full_output=True"):
        single.predict(spectra_new, y_observed=np.ones(3))

    double = _two_response_calibration()
    with pytest.raises(ValueError, match=r"shape \(3,\); expected \(3, 2\):"):
        double.predict(spectra_new, full_output=True, y_observed=np.ones(3))


def test_predict_does_not_print(capsys):
    calibration = _single_response_calibration()
    capsys.readouterr()

    calibration.predict(np.outer([0.7, 1.7], PURE_A))
    calibration.predict(np.outer([0.7, 1.7], PURE_A), full_output=True,
                        y_observed=[0.7, 1.7])

    captured = capsys.readouterr()
    assert captured.out == ""
