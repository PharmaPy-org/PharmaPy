"""Score calibration of ``PharmaPy.PATCalibration`` on a constructed pair of runs.

The first run holds a charge of 0.45 kg/kg and every score unit is 0.25 kg/kg.
A later run holds 0.41 kg/kg and starts as a slurry with 0.20 kg/kg dissolved.
Every expected value follows from these four numbers.
"""
import numpy as np
import pytest

from PharmaPy import PATCalibration

pytestmark = pytest.mark.unit

SENSITIVITY = 0.25  # [kg/kg] per score unit
FIRST_CHARGE, LATER_CHARGE = 0.45, 0.41  # [kg/kg] dissolved charges of the two runs
SLURRY_RATIO = 0.20  # [kg/kg] dissolved in the slurry of the later run
LATER_AGAINST_FIRST = (LATER_CHARGE - FIRST_CHARGE) / SENSITIVITY  # [-] -0.16: dissolved charge of the later run on the first run's zero
SLURRY_SCORE = (SLURRY_RATIO - LATER_CHARGE) / SENSITIVITY  # [-] -0.84: the slurry on the later run's own zero


def test_sensitivity_follows_from_two_states_on_one_zero():
    # On the first run's zero the slurry of the later run sits at -0.16 - 0.84 = -1.
    slurry_on_first_zero = LATER_AGAINST_FIRST + SLURRY_SCORE
    assert slurry_on_first_zero == pytest.approx(-1.0)

    sensitivity = PATCalibration.fit_score_sensitivity([FIRST_CHARGE, SLURRY_RATIO], [0.0, slurry_on_first_zero])

    assert sensitivity == pytest.approx(SENSITIVITY)
    assert isinstance(sensitivity, float)
    # The order of the two states does not matter.
    assert PATCalibration.fit_score_sensitivity([SLURRY_RATIO, FIRST_CHARGE], [slurry_on_first_zero, 0.0]) == pytest.approx(SENSITIVITY)


def test_score_and_ratio_convert_into_each_other_from_the_charge_of_the_run():
    scores = np.array([0.0, -0.4, SLURRY_SCORE])  # [-]

    ratios = PATCalibration.ratio_from_score(scores, LATER_CHARGE, SENSITIVITY)

    np.testing.assert_allclose(ratios, [LATER_CHARGE, 0.31, SLURRY_RATIO])
    np.testing.assert_allclose(PATCalibration.score_from_ratio(ratios, LATER_CHARGE, SENSITIVITY), scores, atol=1e-12)
    assert ratios.shape == (3,)
    assert np.shape(PATCalibration.ratio_from_score(-0.4, LATER_CHARGE, SENSITIVITY)) == ()
    # The same score is another composition in the first run, which holds more.
    assert PATCalibration.ratio_from_score(-0.4, FIRST_CHARGE, SENSITIVITY) == pytest.approx(0.35)


def test_charge_follows_from_one_known_state_of_the_run():
    charge = PATCalibration.charge_from_score(SLURRY_RATIO, SLURRY_SCORE, SENSITIVITY)

    assert charge == pytest.approx(LATER_CHARGE)
    assert isinstance(charge, float)
    # A charge compared with the first run's is the first charge moved by that score.
    assert PATCalibration.ratio_from_score(LATER_AGAINST_FIRST, FIRST_CHARGE, SENSITIVITY) == pytest.approx(LATER_CHARGE)


def test_score_calibration_refuses_states_it_cannot_solve():
    with pytest.raises(ValueError, match="Exactly two states are needed; got 3 ratios and 2 scores"):
        PATCalibration.fit_score_sensitivity([0.45, 0.30, 0.20], [0.0, -1.0])
    with pytest.raises(ValueError, match=r"same score \(-1\)"):
        PATCalibration.fit_score_sensitivity([0.45, 0.20], [-1.0, -1.0])
    with pytest.raises(ValueError, match="sensitivity of -0.25 kg/kg per score unit"):
        PATCalibration.fit_score_sensitivity([0.20, 0.45], [0.0, -1.0])
    with pytest.raises(ValueError, match="more dissolved solute must have the higher score"):
        PATCalibration.fit_score_sensitivity([0.45, 0.45], [0.0, -1.0])
    with pytest.raises(ValueError, match="more dissolved solute must have the higher score"):
        PATCalibration.fit_score_sensitivity([0.45, np.nan], [0.0, -1.0])


@pytest.mark.parametrize("sensitivity", [0.0, -0.25, np.nan])
def test_conversions_refuse_a_sensitivity_that_is_not_positive(sensitivity):
    for convert in (PATCalibration.ratio_from_score, PATCalibration.score_from_ratio, PATCalibration.charge_from_score):
        with pytest.raises(ValueError, match="score sensitivity must be positive"):
            convert(0.3, LATER_CHARGE, sensitivity)
