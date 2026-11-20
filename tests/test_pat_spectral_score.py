"""Whole-spectrum score of ``examples/pat/spectral_score.py`` on constructed spectra.

The spectra are sums of Gaussian bands of a solvent and a solute on a decreasing
axis, as an IR file stores it, with a known temperature effect and known changes
of probe contact. Every expected value is the constructed one.
"""
import numpy as np
import pytest

from examples.pat import spectral_score

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

    at_reference, per_kelvin, residual = spectral_score.temperature_effect(spectra, temperature, REFERENCE_TEMPERATURE)

    np.testing.assert_allclose(at_reference, REFERENCE, atol=ROUND_OFF)
    np.testing.assert_allclose(per_kelvin, PER_KELVIN, atol=ROUND_OFF)
    assert residual < ROUND_OFF
    assert at_reference.shape == per_kelvin.shape == (AXIS.size,)


def test_spectra_are_moved_to_the_reference_temperature():
    temperature = np.array([293.15, 318.15, REFERENCE_TEMPERATURE])  # [K]
    spectra = REFERENCE + np.outer(temperature - REFERENCE_TEMPERATURE, PER_KELVIN)

    moved = spectral_score.at_reference_temperature(spectra, temperature, PER_KELVIN, REFERENCE_TEMPERATURE)

    np.testing.assert_allclose(moved, np.tile(REFERENCE, (3, 1)), atol=ROUND_OFF)
    assert np.abs(spectra[0] - REFERENCE).max() > 1e-3  # the cold spectrum did differ before the move


def test_score_separates_the_solute_change_from_a_change_of_probe_contact():
    position = (AXIS - AXIS.mean()) / np.ptp(AXIS)
    expected = {"score": np.array([0.0, -1.0, -0.4, 0.3, -0.75]), "scale": np.array([0.0, 0.0, 0.012, -0.02, 0.005]),
                "offset": np.array([0.0, 0.0, 0.004, -0.003, 0.0]), "tilt": np.array([0.0, 0.0, -0.006, 0.0, 0.009])}
    spectra = (REFERENCE + np.outer(expected["score"], DIRECTION) + np.outer(expected["scale"], REFERENCE)
               + expected["offset"][:, None] + np.outer(expected["tilt"], position))

    terms = spectral_score.score(AXIS, spectra, REFERENCE, DIRECTION)

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

    terms = spectral_score.score(AXIS, ((1.0 + gain) * (REFERENCE + drop * DIRECTION))[None, :], REFERENCE, DIRECTION)

    np.testing.assert_allclose(terms.scale, [gain], atol=ROUND_OFF)
    np.testing.assert_allclose(terms.score, [drop * (1.0 + gain)], atol=ROUND_OFF)
    np.testing.assert_allclose(terms.residual, [0.0], atol=ROUND_OFF)


def test_score_reports_what_the_four_terms_do_not_explain():
    position = (AXIS - AXIS.mean()) / np.ptp(AXIS)
    components = np.column_stack((DIRECTION, REFERENCE, np.ones(AXIS.size), position))
    other = _band(1600.0, 12.0, 0.01)  # [-] a band of something else
    unexplained = other - components @ np.linalg.lstsq(components, other, rcond=None)[0]

    terms = spectral_score.score(AXIS, (REFERENCE - 0.5 * DIRECTION + other)[None, :], REFERENCE, DIRECTION)

    np.testing.assert_allclose(terms.residual, [np.sqrt(np.mean(unexplained ** 2))], rtol=1e-9)
    assert terms.residual[0] > 1e-4
    assert terms.score.shape == terms.residual.shape == (1,)


def test_score_refuses_a_direction_that_is_a_contact_change():
    with pytest.raises(ValueError, match="cannot be separated"):
        spectral_score.score(AXIS, REFERENCE[None, :], REFERENCE, 0.3 * REFERENCE + 0.01)


def test_usable_points_keep_the_window_and_drop_strong_bands():
    keep = spectral_score.usable_points(AXIS, REFERENCE, 1000.0, 1500.0, 1.3)

    inside = (AXIS >= 1000.0) & (AXIS <= 1500.0)
    assert inside.sum() == 26
    assert not keep[~inside].any()
    # On the 20 cm-1 grid the solvent bands at 1240 and 1045 cm-1 exceed 1.3 at these points only:
    # 2.0 * exp(-0.5 * (20 / 25)**2) = 1.45 beside the first, and 1.6 * exp(-0.5 * (5 / 18)**2) = 1.54 on the second.
    np.testing.assert_array_equal(AXIS[inside & ~keep], [1260.0, 1240.0, 1220.0, 1040.0])
    with pytest.raises(ValueError, match="No spectral point"):
        spectral_score.usable_points(AXIS, REFERENCE, 1230.0, 1250.0, 1.3)
    with pytest.raises(ValueError, match="must increase"):
        spectral_score.usable_points(AXIS, REFERENCE, 1500.0, 1000.0, 1.3)


def test_shapes_and_temperatures_are_checked():
    spectra = np.tile(REFERENCE, (3, 1))
    with pytest.raises(ValueError, match="span at least two temperatures"):
        spectral_score.temperature_effect(spectra, np.full(3, 300.0), REFERENCE_TEMPERATURE)
    with pytest.raises(ValueError, match="one temperature each"):
        spectral_score.temperature_effect(spectra, np.array([300.0, 301.0]), REFERENCE_TEMPERATURE)
    with pytest.raises(ValueError, match="finite temperature"):
        spectral_score.at_reference_temperature(spectra, np.array([300.0, np.nan, 302.0]), PER_KELVIN, REFERENCE_TEMPERATURE)
    with pytest.raises(ValueError, match="Shapes disagree"):
        spectral_score.score(AXIS, spectra, REFERENCE[:-1], DIRECTION)
