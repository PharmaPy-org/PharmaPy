"""Whole-spectrum score of ``examples/pat/PATSignals.py`` on constructed spectra.

The spectra are sums of Gaussian bands of a solvent and a solute on a decreasing
axis, as an IR file stores it, with a known temperature effect and known changes
of probe contact. Every expected value is the constructed one.
"""
import numpy as np
import pytest

from PharmaPy import PATSignals

pytestmark = pytest.mark.unit

AXIS = np.linspace(1800.0, 900.0, 46)  # [cm-1]
REFERENCE_TEMPERATURE = 310.15  # [K]
ROUND_OFF = 1e-10  # [-] the decomposition is exact for constructed spectra


def _band(center: float, width: float, height: float) -> np.ndarray:
    return height * np.exp(-0.5 * ((AXIS - center) / width) ** 2)


SOLVENT = _band(1740.0, 20.0, 1.7) + _band(1240.0, 25.0, 2.0) + _band(1045.0, 18.0, 1.6) + 0.05  # [-]
SOLUTE = _band(1000.0, 30.0, 0.35) + _band(1170.0, 15.0, 0.30) + _band(1460.0, 22.0, 0.12)  # [-] per [kg/kg]
PER_KELVIN = 1e-3 * (_band(1700.0, 40.0, -1.0) + _band(1100.0, 60.0, 0.6)) + 2e-5  # [1/K]
DISSOLVED_RATIO, SLURRY_RATIO = 0.45, 0.20  # [kg/kg]
REFERENCE = SOLVENT + DISSOLVED_RATIO * SOLUTE  # [-] dissolved load at the reference temperature
DIRECTION = (DISSOLVED_RATIO - SLURRY_RATIO) * SOLUTE  # [-] dissolved minus slurry


def test_temperature_effect_recovers_the_line_of_every_point():
    temperature = np.array([293.2, 296.0, 301.5, 304.1, 308.9, 309.0, 312.3])  # [K]
    spectra = REFERENCE + np.outer(temperature - REFERENCE_TEMPERATURE, PER_KELVIN)

    at_reference, per_kelvin, residual = PATSignals.temperature_effect(spectra, temperature, REFERENCE_TEMPERATURE)

    np.testing.assert_allclose(at_reference, REFERENCE, atol=ROUND_OFF)
    np.testing.assert_allclose(per_kelvin, PER_KELVIN, atol=ROUND_OFF)
    assert residual < ROUND_OFF
    assert at_reference.shape == per_kelvin.shape == (AXIS.size,)


def test_spectra_are_moved_to_the_reference_temperature():
    temperature = np.array([293.15, 318.15, REFERENCE_TEMPERATURE])  # [K]
    spectra = REFERENCE + np.outer(temperature - REFERENCE_TEMPERATURE, PER_KELVIN)

    moved = PATSignals.at_reference_temperature(spectra, temperature, PER_KELVIN, REFERENCE_TEMPERATURE)

    np.testing.assert_allclose(moved, np.tile(REFERENCE, (3, 1)), atol=ROUND_OFF)
    assert np.abs(spectra[0] - REFERENCE).max() > 1e-3  # the cold spectrum did differ before the move


def test_score_separates_the_solute_change_from_a_change_of_probe_contact():
    position = (AXIS - AXIS.mean()) / np.ptp(AXIS)
    expected = {"score": np.array([0.0, -1.0, -0.4, 0.3, -0.75]), "scale": np.array([0.0, 0.0, 0.012, -0.02, 0.005]),
                "offset": np.array([0.0, 0.0, 0.004, -0.003, 0.0]), "tilt": np.array([0.0, 0.0, -0.006, 0.0, 0.009])}
    spectra = (REFERENCE + np.outer(expected["score"], DIRECTION) + np.outer(expected["scale"], REFERENCE)
               + expected["offset"][:, None] + np.outer(expected["tilt"], position))

    terms = PATSignals.score_terms(AXIS, spectra, REFERENCE, DIRECTION)

    for name, values in expected.items():
        np.testing.assert_allclose(getattr(terms, name), values, atol=ROUND_OFF, err_msg=name)
    np.testing.assert_allclose(terms.residual, 0.0, atol=ROUND_OFF)
    # The second spectrum is the slurry: one whole direction below the dissolved load.
    np.testing.assert_allclose(spectra[1], SOLVENT + SLURRY_RATIO * SOLUTE, atol=ROUND_OFF)
    # A plain projection on the direction would read the contact change as solute.
    plain = (spectra - REFERENCE) @ DIRECTION / (DIRECTION @ DIRECTION)
    assert np.abs(plain - expected["score"])[[2, 3]].min() > 0.1
    assert 1.0 < terms.condition < 100.0


def test_a_gain_change_goes_to_the_scale_and_multiplies_the_score():
    gain, drop = 0.03, -0.6  # [-]

    terms = PATSignals.score_terms(AXIS, ((1.0 + gain) * (REFERENCE + drop * DIRECTION))[None, :], REFERENCE, DIRECTION)

    np.testing.assert_allclose(terms.scale, [gain], atol=ROUND_OFF)
    np.testing.assert_allclose(terms.score, [drop * (1.0 + gain)], atol=ROUND_OFF)
    np.testing.assert_allclose(terms.residual, [0.0], atol=ROUND_OFF)


def test_score_reports_what_the_four_terms_do_not_explain():
    position = (AXIS - AXIS.mean()) / np.ptp(AXIS)
    components = np.column_stack((DIRECTION, REFERENCE, np.ones(AXIS.size), position))
    other = _band(1600.0, 12.0, 0.01)  # [-] a band of something else
    unexplained = other - components @ np.linalg.lstsq(components, other, rcond=None)[0]

    terms = PATSignals.score_terms(AXIS, (REFERENCE - 0.5 * DIRECTION + other)[None, :], REFERENCE, DIRECTION)

    np.testing.assert_allclose(terms.residual, [np.sqrt(np.mean(unexplained ** 2))], rtol=1e-9)
    assert terms.residual[0] > 1e-4
    assert terms.score.shape == terms.residual.shape == (1,)


def test_score_refuses_a_direction_that_is_a_contact_change():
    with pytest.raises(ValueError, match="cannot be separated"):
        PATSignals.score_terms(AXIS, REFERENCE[None, :], REFERENCE, 0.3 * REFERENCE + 0.01)


def test_usable_points_keep_the_window_and_drop_strong_bands():
    keep = PATSignals.usable_points(AXIS, REFERENCE, 1000.0, 1500.0, 1.3)

    inside = (AXIS >= 1000.0) & (AXIS <= 1500.0)
    assert inside.sum() == 26
    assert not keep[~inside].any()
    # On the 20 cm-1 grid the solvent bands at 1240 and 1045 cm-1 exceed 1.3 at these points only:
    # 2.0 * exp(-0.5 * (20 / 25)**2) = 1.45 beside the first, and 1.6 * exp(-0.5 * (5 / 18)**2) = 1.54 on the second.
    np.testing.assert_array_equal(AXIS[inside & ~keep], [1260.0, 1240.0, 1220.0, 1040.0])
    with pytest.raises(ValueError, match="No spectral point"):
        PATSignals.usable_points(AXIS, REFERENCE, 1230.0, 1250.0, 1.3)
    with pytest.raises(ValueError, match="must increase"):
        PATSignals.usable_points(AXIS, REFERENCE, 1500.0, 1000.0, 1.3)


def test_shapes_and_temperatures_are_checked():
    spectra = np.tile(REFERENCE, (3, 1))
    with pytest.raises(ValueError, match="span at least two temperatures"):
        PATSignals.temperature_effect(spectra, np.full(3, 300.0), REFERENCE_TEMPERATURE)
    with pytest.raises(ValueError, match="one temperature each"):
        PATSignals.temperature_effect(spectra, np.array([300.0, 301.0]), REFERENCE_TEMPERATURE)
    with pytest.raises(ValueError, match="finite temperature"):
        PATSignals.at_reference_temperature(spectra, np.array([300.0, np.nan, 302.0]), PER_KELVIN, REFERENCE_TEMPERATURE)
    with pytest.raises(ValueError, match="Shapes disagree"):
        PATSignals.score_terms(AXIS, spectra, REFERENCE[:-1], DIRECTION)


def _run(ratio, temperature, per_kelvin=PER_KELVIN):
    """Constructed spectra of a solution with a dissolved ratio [kg/kg] at a temperature [K]."""
    ratio, temperature = np.asarray(ratio, dtype=float), np.asarray(temperature, dtype=float)
    return SOLVENT + np.outer(ratio, SOLUTE) + np.outer(temperature - REFERENCE_TEMPERATURE, per_kelvin)


# Reference run: three slurry spectra, six of the dissolved charge at several temperatures, one partly crystallized.
RUN_RATIO = np.array([SLURRY_RATIO] * 3 + [DISSOLVED_RATIO] * 6 + [0.33])  # [kg/kg]
RUN_TEMPERATURE = np.array([293.0, 293.2, 293.1, 318.0, 316.0, 312.0, 310.4, 310.1, 309.9, 300.0])  # [K]
SLURRY = np.arange(10) < 3
CLEAR = (np.arange(10) >= 3) & (np.arange(10) < 9)
DISSOLVED = (np.arange(10) >= 6) & (np.arange(10) < 9)


def _fitted(per_kelvin=PER_KELVIN, window=(950.0, 1700.0), maximum_absorbance=1.3):
    return PATSignals.SpectralScore.fit(
        AXIS, _run(RUN_RATIO, RUN_TEMPERATURE, per_kelvin), RUN_TEMPERATURE, clear_solution=CLEAR, dissolved=DISSOLVED, slurry=SLURRY,
        reference_temperature=REFERENCE_TEMPERATURE, window=window, maximum_absorbance=maximum_absorbance)


def test_fitted_score_holds_the_temperature_effect_and_the_direction_of_the_reference_run():
    fitted = _fitted()

    # 38 grid points lie between 950 and 1700 cm-1; four are on solvent bands above 1.3 (see the usable-points test).
    usable = (AXIS >= 950.0) & (AXIS <= 1700.0) & ~np.isin(AXIS, [1260.0, 1240.0, 1220.0, 1040.0])
    np.testing.assert_array_equal(fitted.usable, usable)
    assert fitted.usable.sum() == 34
    np.testing.assert_array_equal(fitted.wavenumber, AXIS)
    np.testing.assert_allclose(fitted.per_kelvin, PER_KELVIN, atol=ROUND_OFF)
    np.testing.assert_allclose(fitted.direction, DIRECTION[usable], atol=ROUND_OFF)
    assert fitted.reference_temperature == REFERENCE_TEMPERATURE
    assert fitted.temperature_range == (309.9, 318.0)
    assert fitted.temperature_residual < ROUND_OFF


def test_fitted_score_counts_from_the_dissolved_charge_of_each_run():
    fitted = _fitted()

    reference_run = fitted.score(AXIS, _run(RUN_RATIO, RUN_TEMPERATURE), RUN_TEMPERATURE, DISSOLVED)
    # Another run with a charge of 0.40 kg/kg, which its two dissolved spectra read as 0.39 and 0.41; its last
    # spectrum has 2 % more signal from a change of probe contact.
    ratio, temperature = np.array([0.39, 0.41, 0.30, 0.25]), np.array([310.0, 311.0, 300.0, REFERENCE_TEMPERATURE])  # [kg/kg], [K]
    spectra = _run(ratio, temperature) * (1.0 + np.array([0.0, 0.0, 0.0, 0.02]))[:, None]
    other_run = fitted.score(AXIS, spectra, temperature, np.array([True, True, False, False]))

    drop = DISSOLVED_RATIO - SLURRY_RATIO  # [kg/kg] one unit of the score
    np.testing.assert_allclose(reference_run.score, (RUN_RATIO - DISSOLVED_RATIO) / drop, atol=ROUND_OFF)
    np.testing.assert_allclose(reference_run.residual, 0.0, atol=ROUND_OFF)
    # The gain multiplies the solute change as well, so the last score reads 2 % larger.
    expected = [(0.39 - 0.40) / drop, (0.41 - 0.40) / drop, (0.30 - 0.40) / drop, 1.02 * (0.25 - 0.40) / drop]
    np.testing.assert_allclose(other_run.score, expected, atol=ROUND_OFF)
    np.testing.assert_allclose(other_run.scale, [0.0, 0.0, 0.0, 0.02], atol=ROUND_OFF)
    assert other_run.score.shape == (4,) and reference_run.score.shape == (10,)
    moved = fitted.at_reference(AXIS, spectra[:3], temperature[:3])
    np.testing.assert_allclose(moved, (SOLVENT + np.outer(ratio[:3], SOLUTE))[:, fitted.usable], atol=ROUND_OFF)


def test_fitted_score_reports_what_lines_in_temperature_leave_of_the_clear_solution():
    # Three clear spectra 10 K apart are shifted by +d, -2d and +d. No line in temperature follows that pattern, so
    # the lines stay where they were and the whole shift is left over: sqrt((1 + 4 + 1) / 3) d.
    shift = 1e-3  # [-] d
    temperature = REFERENCE_TEMPERATURE + np.array([-17.0, -10.0, 0.0, 10.0])  # [K]
    spectra = _run([SLURRY_RATIO] + [DISSOLVED_RATIO] * 3, temperature)
    spectra[1:] += shift * np.array([1.0, -2.0, 1.0])[:, None]
    clear = np.array([False, True, True, True])

    fitted = PATSignals.SpectralScore.fit(AXIS, spectra, temperature, clear_solution=clear, dissolved=clear, slurry=~clear,
                                          reference_temperature=REFERENCE_TEMPERATURE, window=(950.0, 1700.0), maximum_absorbance=1.3)

    assert fitted.temperature_residual == pytest.approx(shift * np.sqrt(2.0), rel=1e-9)
    np.testing.assert_allclose(fitted.per_kelvin, PER_KELVIN, atol=ROUND_OFF)
    np.testing.assert_allclose(fitted.direction, DIRECTION[fitted.usable], atol=ROUND_OFF)


def test_score_per_kelvin_is_the_score_of_one_kelvin_of_temperature_effect():
    position = (AXIS - AXIS.mean()) / np.ptp(AXIS)
    per_kelvin_score = 0.004  # [1/K] the part of the temperature effect that looks like the direction
    per_kelvin = per_kelvin_score * DIRECTION - 3e-4 * REFERENCE + 2e-5 + 1e-4 * position  # [1/K]

    fitted = _fitted(per_kelvin, window=(900.0, 1800.0), maximum_absorbance=10.0)

    assert fitted.usable.all()
    assert fitted.score_per_kelvin == pytest.approx(per_kelvin_score, abs=ROUND_OFF)
    # Its meaning: a spectrum of the dissolved charge 7 K above the reference temperature, read without the move.
    warm = _run([DISSOLVED_RATIO], [REFERENCE_TEMPERATURE + 7.0], per_kelvin)
    assert PATSignals.score_terms(AXIS, warm, REFERENCE, DIRECTION).score[0] == pytest.approx(7.0 * per_kelvin_score, abs=ROUND_OFF)
    assert fitted.score(AXIS, warm, np.array([REFERENCE_TEMPERATURE + 7.0]), np.array([True])).score[0] == pytest.approx(0.0, abs=ROUND_OFF)


def test_fitted_score_refuses_selections_that_are_not_masks_and_another_axis():
    spectra = _run(RUN_RATIO, RUN_TEMPERATURE)
    settings = {"reference_temperature": REFERENCE_TEMPERATURE, "window": (950.0, 1700.0), "maximum_absorbance": 1.3}
    with pytest.raises(ValueError, match=r"slurry selection must be a boolean mask with one entry per spectrum \(10\); got int"):
        PATSignals.SpectralScore.fit(AXIS, spectra, RUN_TEMPERATURE, clear_solution=CLEAR, dissolved=DISSOLVED, slurry=SLURRY.astype(int), **settings)
    with pytest.raises(ValueError, match="dissolved selection must be a boolean mask"):
        PATSignals.SpectralScore.fit(AXIS, spectra, RUN_TEMPERATURE, clear_solution=CLEAR, dissolved=DISSOLVED[:-1], slurry=SLURRY, **settings)
    with pytest.raises(ValueError, match="clear-solution selection holds no spectrum"):
        PATSignals.SpectralScore.fit(AXIS, spectra, RUN_TEMPERATURE, clear_solution=np.zeros(10, dtype=bool), dissolved=DISSOLVED, slurry=SLURRY, **settings)
    with pytest.raises(ValueError, match="Shapes disagree: axis"):
        PATSignals.SpectralScore.fit(AXIS[:-1], spectra, RUN_TEMPERATURE, clear_solution=CLEAR, dissolved=DISSOLVED, slurry=SLURRY, **settings)
    fitted = _fitted()
    with pytest.raises(ValueError, match="not on the spectral axis the score was fitted on"):
        fitted.score(AXIS + 4.0, spectra, RUN_TEMPERATURE, DISSOLVED)
    with pytest.raises(ValueError, match="dissolved selection holds no spectrum"):
        fitted.score(AXIS, spectra, RUN_TEMPERATURE, np.zeros(10, dtype=bool))
