"""Regression tests for standard-normal-variate PCR calibration (#138).

Scope: ``PCR_calibration(..., snv=True)`` training and ``predict``. The
fixture is a synthetic three-component absorbance model on six wavelengths:
each spectrum is ``scatter * (c * analyte + d * interferent + background) +
offset`` [AU] with analyte concentration ``c`` [g/L] and an uncalibrated
interferent concentration ``d`` [g/L]. The interferent makes the SNV spectra
span three directions, so the two retained components truncate the basis and
the column scaling choice (``standardize``) changes the predictions. The
per-sample multiplicative scatter [-] and additive baseline offset [AU] are
the artifacts SNV is designed to remove (Barnes, Dhanoa and Lister, Appl.
Spectrosc. 43(5), 772-777, 1989).
Expected values come from an independent NumPy implementation of
SNV -> column centering/scaling -> SVD -> least-squares score regression.
"""

import numpy as np
import pytest

from PharmaPy.Calibration import PCR_calibration

pytestmark = pytest.mark.unit

# Synthetic pure-component spectra on six wavelengths.
ANALYTE_SPECTRUM = np.array(
    [0.10, 0.35, 0.80, 0.55, 0.20, 0.05])  # [AU*L/g]
INTERFERENT_SPECTRUM = np.array(
    [0.50, 0.30, 0.10, 0.05, 0.10, 0.25])  # [AU*L/g]
BACKGROUND_SPECTRUM = np.array(
    [0.30, 0.25, 0.20, 0.30, 0.45, 0.60])  # [AU]
# Seven calibration samples (unequal to the six wavelengths).
CONC_TRAIN = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5])  # [g/L]
INTERFERENT_TRAIN = np.array([0.3, 0.1, 0.4, 0.2, 0.5, 0.15, 0.35])  # [g/L]
SCATTER_TRAIN = np.array([1.0, 1.2, 0.9, 1.1, 0.8, 1.3, 1.05])  # [-]
OFFSET_TRAIN = np.array([0.0, 0.05, -0.03, 0.1, 0.02, -0.05, 0.08])  # [AU]

CONC_NEW = np.array([0.8, 2.2, 3.1])  # [g/L]
INTERFERENT_NEW = np.array([0.25, 0.45, 0.05])  # [g/L]
SCATTER_NEW = np.array([1.15, 0.95, 1.25])  # [-]
OFFSET_NEW = np.array([0.04, -0.02, 0.06])  # [AU]

NUM_COMP = 2  # retained principal components [-]

# Both sides evaluate the same double-precision linear algebra in a
# different order (SVD sign/order conventions, pinv versus lstsq); a
# relative bound well above ~1e-15 round-off but far below any physically
# meaningful concentration difference.
PRED_RTOL = 1e-9  # [-]


def _spectra(conc, interferent, scatter, offset):
    """Build synthetic spectra with scatter and baseline offset.

    Parameters
    ----------
    conc : numpy.ndarray
        Analyte concentrations, shape ``(n_samples,)`` [g/L].
    interferent : numpy.ndarray
        Interferent concentrations, shape ``(n_samples,)`` [g/L].
    scatter : numpy.ndarray
        Multiplicative scatter factors, shape ``(n_samples,)`` [-].
    offset : numpy.ndarray
        Additive baseline offsets, shape ``(n_samples,)`` [AU].

    Returns
    -------
    numpy.ndarray
        Spectra, shape ``(n_samples, 6)`` [AU].
    """
    clean = (np.outer(conc, ANALYTE_SPECTRUM)
             + np.outer(interferent, INTERFERENT_SPECTRUM)
             + BACKGROUND_SPECTRUM)  # [AU]
    return scatter[:, None] * clean + offset[:, None]


def _independent_snv(spectra):
    """Return the row-wise standard normal variate of spectra.

    Parameters
    ----------
    spectra : numpy.ndarray
        Spectra, shape ``(n_samples, n_wavelengths)`` [AU].

    Returns
    -------
    numpy.ndarray
        Spectra centered and scaled by their own mean and sample standard
        deviation (denominator ``n_wavelengths - 1``, Barnes et al. 1989)
        across wavelengths, same shape [-].
    """
    num_wavelengths = spectra.shape[1]  # [-]
    mean = spectra.mean(axis=1)[:, None]  # [AU]
    std = np.sqrt(((spectra - mean) ** 2).sum(axis=1)
                  / (num_wavelengths - 1))[:, None]  # [AU]
    return (spectra - mean) / std


def _independent_pcr(train_spectra, train_conc, new_spectra, standardize):
    """Fit and apply SNV-PCR without using ``PCR_calibration``.

    Parameters
    ----------
    train_spectra : numpy.ndarray
        Calibration spectra, shape ``(n_train, n_wavelengths)`` [AU].
    train_conc : numpy.ndarray
        Calibration concentrations, shape ``(n_train,)`` [g/L].
    new_spectra : numpy.ndarray
        Spectra to predict, shape ``(n_new, n_wavelengths)`` [AU].
    standardize : bool
        Whether to divide by the training column standard deviation.

    Returns
    -------
    train_pred : numpy.ndarray
        Fitted training concentrations, shape ``(n_train,)`` [g/L].
    new_pred : numpy.ndarray
        Predicted concentrations, shape ``(n_new,)`` [g/L].
    col_mean : numpy.ndarray
        Training column means of the SNV spectra [-].
    col_std : numpy.ndarray
        Training column standard deviations of the SNV spectra [-].
    """
    train_snv = _independent_snv(train_spectra)  # [-]
    col_mean = train_snv.mean(axis=0)  # [-]
    col_std = np.sqrt(((train_snv - col_mean) ** 2).mean(axis=0))  # [-]
    col_scale = col_std if standardize else np.ones_like(col_std)  # [-]

    def preprocess(spectra):
        return (_independent_snv(spectra) - col_mean) / col_scale

    train_x = preprocess(train_spectra)  # [-]
    _, _, v_t = np.linalg.svd(train_x)
    loadings = v_t[:NUM_COMP].T  # [-]
    conc_mean = train_conc.mean()  # [g/L]
    coeff, *_ = np.linalg.lstsq(train_x @ loadings, train_conc - conc_mean,
                                rcond=None)  # [g/L]

    def apply(spectra):
        return preprocess(spectra) @ loadings @ coeff + conc_mean

    return apply(train_spectra), apply(new_spectra), col_mean, col_std


def _fit(train_spectra, standardize=True):
    """Fit an SNV PCR calibration to the training concentrations.

    Parameters
    ----------
    train_spectra : numpy.ndarray
        Calibration spectra, shape ``(7, 6)`` [AU].
    standardize : bool, optional
        Forwarded to ``PCR_calibration``.

    Returns
    -------
    PCR_calibration
        Calibration regressed on ``CONC_TRAIN`` [g/L] with ``NUM_COMP``
        components.
    """
    calibration = PCR_calibration(train_spectra, num_comp=NUM_COMP,
                                  standardize=standardize, snv=True)
    calibration.get_regression(CONC_TRAIN, num_comp=NUM_COMP)
    return calibration


@pytest.mark.parametrize("standardize", [True, False])
def test_snv_training_fit_matches_independent_pipeline(standardize):
    train = _spectra(CONC_TRAIN, INTERFERENT_TRAIN, SCATTER_TRAIN,
                     OFFSET_TRAIN)  # [AU]
    calibration = _fit(train, standardize)

    train_pred, _, col_mean, col_std = _independent_pcr(
        train, CONC_TRAIN, train[:1], standardize)

    # SNV column means can be near zero, so pair the relative bound with an
    # absolute one at the same level on the O(1) normalized scale [-].
    np.testing.assert_allclose(calibration.data_mean, col_mean,
                               rtol=PRED_RTOL, atol=PRED_RTOL)
    np.testing.assert_allclose(calibration.data_std, col_std,
                               rtol=PRED_RTOL)
    np.testing.assert_allclose(calibration.residuals.ravel(),
                               CONC_TRAIN - train_pred, rtol=0.0,
                               atol=PRED_RTOL * CONC_TRAIN.max())


def test_snv_single_spectrum_prediction_is_finite_and_batch_independent():
    train = _spectra(CONC_TRAIN, INTERFERENT_TRAIN, SCATTER_TRAIN,
                     OFFSET_TRAIN)  # [AU]
    new = _spectra(CONC_NEW, INTERFERENT_NEW, SCATTER_NEW,
                   OFFSET_NEW)  # [AU]
    calibration = _fit(train)

    batch_pred = calibration.predict(new).ravel()  # [g/L]
    single_pred = np.array([
        calibration.predict(new[row:row + 1]).item()
        for row in range(len(new))])  # [g/L]
    # A 1-D spectrum is promoted to one row.
    one_d_pred = calibration.predict(new[1]).item()  # [g/L]

    _, expected, _, _ = _independent_pcr(train, CONC_TRAIN, new, True)

    assert np.all(np.isfinite(single_pred))
    np.testing.assert_allclose(single_pred, batch_pred, rtol=PRED_RTOL)
    np.testing.assert_allclose(single_pred, expected, rtol=PRED_RTOL)
    assert one_d_pred == pytest.approx(expected[1], rel=PRED_RTOL)


def test_snv_issue_reproduction_single_spectrum_is_finite():
    # #138 reproduction: absorbance [AU] at three wavelengths, response [g/L].
    spectra_au = np.array([[0.20, 1.10, 2.40], [0.45, 1.35, 2.95],
                           [0.80, 1.85, 3.45], [1.10, 2.10, 4.05]])
    conc_g_l = np.array([1.2, 1.8, 2.6, 3.1])
    new_au = np.array([[0.70, 1.70, 3.20]])

    calibration = PCR_calibration(spectra_au, num_comp=2, standardize=True,
                                  snv=True)
    calibration.get_regression(conc_g_l, num_comp=2)
    prediction = calibration.predict(new_au)

    assert prediction.shape == (1, 1)
    assert np.all(np.isfinite(prediction))


def test_snv_prediction_is_invariant_to_spectrum_offset_and_scale():
    train = _spectra(CONC_TRAIN, INTERFERENT_TRAIN, SCATTER_TRAIN,
                     OFFSET_TRAIN)  # [AU]
    clean_new = _spectra(CONC_NEW, INTERFERENT_NEW, np.ones(3),
                         np.zeros(3))  # [AU]
    distorted_new = _spectra(CONC_NEW, INTERFERENT_NEW, SCATTER_NEW,
                             OFFSET_NEW)  # [AU]
    calibration = _fit(train)

    np.testing.assert_allclose(calibration.predict(distorted_new),
                               calibration.predict(clean_new),
                               rtol=PRED_RTOL)

    # The calibration itself is invariant: training on artifact-free
    # spectra gives the same model.
    clean_train = _spectra(CONC_TRAIN, INTERFERENT_TRAIN, np.ones(7),
                           np.zeros(7))  # [AU]
    clean_calibration = _fit(clean_train)
    np.testing.assert_allclose(clean_calibration.predict(distorted_new),
                               calibration.predict(clean_new),
                               rtol=PRED_RTOL)


def test_snv_zero_variance_spectrum_raises_naming_rows():
    train = _spectra(CONC_TRAIN, INTERFERENT_TRAIN, SCATTER_TRAIN,
                     OFFSET_TRAIN)  # [AU]
    calibration = _fit(train)
    new = _spectra(CONC_NEW, INTERFERENT_NEW, SCATTER_NEW,
                   OFFSET_NEW)  # [AU]
    new[1] = 0.4  # constant spectrum [AU]

    with pytest.raises(ValueError, match=r"constant spectrum row\(s\) \[1\]"):
        calibration.predict(new)

    # A small but genuine variation is far above round-off and is accepted.
    nearly_flat = 0.4 + 1e-6 * np.arange(6.0)  # [AU]
    assert np.isfinite(calibration.predict(nearly_flat).item())

    flat_train = train.copy()
    flat_train[[2, 5]] = 0.3  # constant spectra [AU]
    with pytest.raises(ValueError,
                       match=r"constant spectrum row\(s\) \[2, 5\]"):
        PCR_calibration(flat_train, num_comp=NUM_COMP, snv=True)


def test_snv_rejects_float32_constant_spectra():
    """Constant float32 spectra are rejected like their float64 values.

    Their float32 row std is round-off far above the float64 bound unless
    the spectra are converted to float64 first.
    """
    train = _spectra(CONC_TRAIN, INTERFERENT_TRAIN, SCATTER_TRAIN,
                     OFFSET_TRAIN)  # [AU]
    calibration = _fit(train)
    flat_f32 = np.full(6, 0.3, dtype=np.float32)  # constant spectrum [AU]
    assert flat_f32.std() > 0  # the float32 round-off this test targets

    with pytest.raises(ValueError, match=r"constant spectrum row\(s\) \[0\]"):
        calibration.predict(flat_f32)

    train_f32 = train.astype(np.float32)
    train_f32[3] = np.float32(0.3)
    with pytest.raises(ValueError, match=r"constant spectrum row\(s\) \[3\]"):
        PCR_calibration(train_f32, num_comp=NUM_COMP, snv=True)


def test_snv_rejects_single_wavelength_spectra():
    spectra = np.array([[0.2], [0.5], [0.9]])  # one wavelength [AU]

    with pytest.raises(ValueError,
                       match="at least two wavelengths per spectrum; got 1"):
        PCR_calibration(spectra, num_comp=1, snv=True)


# Spectra whose middle wavelength equals every row mean, so that SNV makes
# column 2 identically zero although the raw column varies [AU].
CONSTANT_SNV_COLUMN_SPECTRA = [
    np.array([[1.0, 2.0, 3.0, 4.0, 5.0], [2.0, 3.0, 5.0, 7.0, 8.0],
              [2.0, 4.0, 6.0, 8.0, 10.0], [3.0, 7.0, 8.0, 9.0, 13.0]]),
    np.array([[4.0, 3.0, 2.0, 1.0, 0.0], [6.0, 7.0, 4.0, 1.0, 2.0],
              [9.0, 7.0, 6.0, 5.0, 3.0], [9.0, 10.0, 8.0, 6.0, 7.0]]),
]


# Scale factors [-] that make the integer spectra non-integer, so the SNV of
# the middle column is round-off (~1e-16) instead of exact zeros.
SPECTRA_SCALES = [1.0, 0.1, 0.3, 0.37, 1e-3]  # [-]


@pytest.mark.parametrize("scale", SPECTRA_SCALES)
@pytest.mark.parametrize("spectra", CONSTANT_SNV_COLUMN_SPECTRA)
def test_snv_constant_column_rejected_with_standardize(spectra, scale):
    with pytest.raises(ValueError,
                       match=r"column\(s\) \[2\] constant.*"
                             r"use standardize=False"):
        PCR_calibration(scale * spectra, num_comp=NUM_COMP, snv=True,
                        standardize=True)


# Random calibration sets [AU]: 6 spectra x 5 wavelengths, uniform in
# [0.1, 1.0) AU; the middle wavelength is set to the mean of the other four,
# hence equal to each row mean in exact arithmetic.
RANDOM_SEEDS = range(20)  # [-]
RANDOM_SHAPE = (6, 5)  # (spectra, wavelengths)
CONSTANT_COLUMN = 2  # [-], column index


def _row_mean_column_spectra(seed):
    """Return random spectra whose middle wavelength equals each row mean.

    Parameters
    ----------
    seed : int
        Seed of ``numpy.random.default_rng``.

    Returns
    -------
    numpy.ndarray
        Spectra, shape ``RANDOM_SHAPE`` [AU].
    """
    rng = np.random.default_rng(seed)
    spectra = rng.uniform(0.1, 1.0, RANDOM_SHAPE)  # [AU]
    spectra[:, CONSTANT_COLUMN] = np.delete(
        spectra, CONSTANT_COLUMN, axis=1).mean(axis=1)
    return spectra


@pytest.mark.parametrize("seed", RANDOM_SEEDS)
def test_snv_round_off_constant_column_rejected(seed):
    spectra = _row_mean_column_spectra(seed)  # [AU]

    with pytest.raises(ValueError, match=r"column\(s\) \[2\] constant"):
        PCR_calibration(spectra, num_comp=NUM_COMP, snv=True,
                        standardize=True)


@pytest.mark.parametrize("seed", RANDOM_SEEDS)
def test_snv_small_real_column_variation_is_accepted(seed):
    """A column variation of 1e-8 relative is far above SNV round-off.

    The round-off bound is ~(n + 4) * eps * max|x| / s ~ 1e-14 here (row
    stds ~ 0.2-0.3 AU), so a 1e-8 AU variation (a few 1e-8 on the SNV
    scale) is genuine information.
    """
    spectra = _row_mean_column_spectra(seed)  # [AU]
    variation = 1e-8 * np.array([1.0, -1.0, 2.0, 0.0, -2.0, 0.5])  # [AU]
    spectra[:, CONSTANT_COLUMN] += variation

    calibration = PCR_calibration(spectra, num_comp=NUM_COMP, snv=True,
                                  standardize=True)

    assert calibration.data_std[CONSTANT_COLUMN] > 0
    assert np.all(np.isfinite(calibration.data_centered))


@pytest.mark.parametrize("spectra", CONSTANT_SNV_COLUMN_SPECTRA)
def test_snv_constant_column_fits_without_standardize(spectra):
    conc = np.array([1.0, 2.0, 3.0, 4.0])  # [g/L]
    calibration = PCR_calibration(spectra, num_comp=NUM_COMP, snv=True,
                                  standardize=False)
    calibration.get_regression(conc, num_comp=NUM_COMP)

    train_pred, new_pred, _, _ = _independent_pcr(spectra, conc,
                                                  spectra[1:2], False)

    np.testing.assert_allclose(calibration.residuals.ravel(),
                               conc - train_pred, rtol=0.0,
                               atol=PRED_RTOL * conc.max())
    np.testing.assert_allclose(calibration.predict(spectra[1]).ravel(),
                               new_pred, rtol=PRED_RTOL)
