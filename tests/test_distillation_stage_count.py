"""Shortcut-design stage counts of the distillation columns (issue #523).

``num_plates=None`` (the documented default), zero or a negative value ask
the shortcut design to estimate the number of equilibrium stages: the
Gilliland correlation in Molokanov's form, or the Fenske minimum times
``abs(num_plates)``, both rounded up. The estimate was a NumPy float, which
``DistillationColumn.calc_plates`` cannot use as an array dimension or a
range bound, so those documented options failed with a ``TypeError``. The
tests check that both estimates are whole stage counts (ints) equal to the
correlations evaluated here, that a positive ``num_plates`` is kept, and
that both columns use them; ``DynamicDistillation`` already converted the
count in ``column_startup`` and is pinned through it.

Fixture: a synthetic ideal binary whose Antoine constants differ only in
A, by log10(ALPHA), so the relative volatility is exactly ALPHA at every
temperature. Fenske's minimum stage count and the binary Underwood minimum
reflux then have closed forms, evaluated in this module independently of
PharmaPy. Core tests only; no integrator is involved.
"""

import json

import numpy as np
import pytest

from PharmaPy.Distillation import DistillationColumn, DynamicDistillation
from PharmaPy.Phases import LiquidPhase
from PharmaPy.Streams import LiquidStream

pytestmark = pytest.mark.unit

# Relative volatility [-] of the binary, set through Antoine A [-] with
# equal B [K] and C [K] (log10(p/[Pa]) = A - B/(T + C)); mw [g/mol],
# rho_liq [kg/m**3], cp_liq [J/mol/K] and t_crit [K] (far above the
# column temperatures, about 465 K at 1 atm) are synthetic constants.
ALPHA = 3.0  # [-]
HEAVY_ANTOINE = [9.0, 1700.0, -40.0]
RECORD = {"mw": 100.0, "rho_liq": 1000.0, "cp_liq": [100.0],
          "t_crit": 1000.0}
PRESSURE = 101325.0  # [Pa]
FEED = np.array([0.5, 0.5])  # [-], light, heavy
FEED_FLOW = 25.0  # [mol/s]
FEED_TEMP = 350.0  # [K]
Q_FEED = 1.0  # [-], saturated liquid
LK_RECOVERY = 95.0  # [%], light key to the distillate
HK_RECOVERY = 5.0  # [%], heavy key to the distillate
REFLUX_TO_MINIMUM = 1.5  # [-], DEFAULT_REFLUX_TO_MINIMUM_RATIO, reflux=None
FENSKE_MULTIPLIER = 2  # [-], num_plates = -2
GIVEN_PLATES = 8  # [-]
HOLDUP = 50.0  # [mol], dynamic column holdup
# Feed stage [-] of the plate-to-plate solves, within every stage count
# here. Left to the operating-line switch, the stepping of this fixture
# reaches dew points above t_crit, where the database has no Henry
# constants; that limitation of the steady plate-to-plate calculation is
# not part of issue #523.
SOLVE_FEED_STAGE = 4  # [-]
DESIGN_TIME = 1.0  # [s], startup design time of the dynamic column
# The design minimizes the squared Underwood residual with a tolerance of
# 1e-10 [-], so its minimum reflux can differ from the closed form by about
# the square root of that, relative; 1e-6 allows it (observed: 8e-9).
UNDERWOOD_RTOL = 1e-6  # [-]
# Fenske uses the key ratios at two bubble points, where the fixture's
# volatility is ALPHA up to float64 round-off of a few operations; 1e-9
# [-] is far above that (observed: 4e-16) and far below the fractional
# part of the stage counts.
FENSKE_RTOL = 1e-9  # [-]


def _database(tmp_path):
    """Write the constant-volatility binary database.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Directory for the file.

    Returns
    -------
    str
        Path of the database.
    """
    light = [HEAVY_ANTOINE[0] + np.log10(ALPHA), *HEAVY_ANTOINE[1:]]
    path = tmp_path / "binary.json"
    path.write_text(json.dumps({"light": dict(RECORD, p_vap=light),
                                "heavy": dict(RECORD, p_vap=HEAVY_ANTOINE)}))
    return str(path)


def _fenske_minimum_stages():
    """Return Fenske's minimum stage count for the fixture.

    Returns
    -------
    float
        ``ln(r_L/(1 - r_L) * (1 - r_H)/r_H) / ln(ALPHA)`` [-], with the
        recoveries ``r`` of the keys to the distillate.
    """
    light, heavy = LK_RECOVERY / 100, HK_RECOVERY / 100  # [-]
    return np.log(light / (1 - light) * (1 - heavy) / heavy) / np.log(ALPHA)


def _underwood_minimum_reflux():
    """Return the binary Underwood minimum reflux of the fixture.

    Returns
    -------
    float
        ``R_min`` [-]: the root ``phi`` between 1 and ALPHA of
        ``ALPHA z_L/(ALPHA - phi) + z_H/(1 - phi) = 1 - q``, a root of the
        quadratic obtained by multiplying out the denominators, gives
        ``V_min = ALPHA d_L/(ALPHA - phi) + d_H/(1 - phi)`` with the
        distillate component flows ``d_i`` [mol/s], and
        ``R_min = V_min/D - 1``.
    """
    vapor_fraction = 1 - Q_FEED  # [-]
    roots = np.roots([
        vapor_fraction,
        ALPHA * FEED[0] + FEED[1] - vapor_fraction * (ALPHA + 1),
        vapor_fraction * ALPHA - ALPHA * FEED.sum()])  # [-]
    phi = roots[(roots > 1) & (roots < ALPHA)].item()  # [-]
    distillate = FEED_FLOW * FEED * np.array([LK_RECOVERY,
                                              HK_RECOVERY]) / 100  # [mol/s]
    vapor_min = (ALPHA * distillate[0] / (ALPHA - phi)
                 + distillate[1] / (1 - phi))  # [mol/s]
    return vapor_min / distillate.sum() - 1


def _gilliland_stages():
    """Return the Gilliland stage count, Molokanov's form, rounded up.

    Returns
    -------
    int
        ``ceil((Y + N_min)/(1 - Y))`` [-] with ``Y = 1 - exp((1 + 54.4 X)
        / (11 + 117.2 X) (X - 1)/sqrt(X))`` and ``X = (R - R_min)/(R + 1)``
        (Molokanov et al.; as ``molokanov_equation``), at the default reflux
        ``R = REFLUX_TO_MINIMUM R_min``.
    """
    min_reflux = _underwood_minimum_reflux()  # [-]
    reflux = REFLUX_TO_MINIMUM * min_reflux  # [-]
    x_val = (reflux - min_reflux) / (reflux + 1)  # [-]
    y_val = 1 - np.exp((1 + 54.4 * x_val) / (11 + 117.2 * x_val)
                       * (x_val - 1) / np.sqrt(x_val))  # [-]
    return int(np.ceil((y_val + _fenske_minimum_stages()) / (1 - y_val)))


EXPECTED_STAGES = {
    "estimated": (None, _gilliland_stages()),
    "fenske-multiple": (-FENSKE_MULTIPLIER,
                        int(np.ceil(FENSKE_MULTIPLIER
                                    * _fenske_minimum_stages()))),
    "given": (GIVEN_PLATES, GIVEN_PLATES),
}  # num_plates option, expected stage count [-]


def _column(cls, tmp_path, num_plates, num_feed=None):
    """Build a column on the binary fixture.

    Parameters
    ----------
    cls : type
        ``DistillationColumn`` or ``DynamicDistillation``.
    tmp_path : pathlib.Path
        Directory for the database.
    num_plates : int or None
        Stage-count option.
    num_feed : int, optional
        Feed stage counted from the top [-]; None leaves it to Kirkbride
        (shortcut design) and to the operating-line switch (steady
        plate-to-plate calculation).

    Returns
    -------
    DistillationColumn or DynamicDistillation
        Column with its feed (and, for the dynamic column, a holdup at the
        feed composition) attached and the default reflux
        (``reflux=None``).
    """
    path = _database(tmp_path)
    column = cls(pres=PRESSURE, q_feed=Q_FEED, LK="light", HK="heavy",
                 perc_LK=LK_RECOVERY, perc_HK=HK_RECOVERY,
                 num_plates=num_plates, num_feed=num_feed)
    column.Inlet = LiquidStream(path, temp=FEED_TEMP, pres=PRESSURE,
                                mole_flow=FEED_FLOW, mole_frac=FEED,
                                verbose=False)
    if cls is DynamicDistillation:
        column.Phases = LiquidPhase(path, temp=FEED_TEMP, pres=PRESSURE,
                                    moles=HOLDUP, mole_frac=FEED,
                                    verbose=False)
    return column


def test_fixture_estimates_are_not_already_whole():
    """The correlations' raw values are fractional for this fixture.

    A fractional raw value is what made the rounded-up estimate a float;
    were it whole, the tests below could not tell rounding from truncation.
    """
    min_reflux = _underwood_minimum_reflux()  # [-]
    assert min_reflux > 0
    assert _fenske_minimum_stages() % 1 != 0
    assert (FENSKE_MULTIPLIER * _fenske_minimum_stages()) % 1 != 0
    assert _gilliland_stages() > FENSKE_MULTIPLIER * _fenske_minimum_stages()


@pytest.mark.parametrize("option", sorted(EXPECTED_STAGES))
def test_shortcut_design_gives_a_whole_stage_count(tmp_path, option):
    """The design's stage count is the expected int.

    Its minimum reflux and stage count match Underwood and Fenske, which
    the estimates are built from.
    """
    num_plates, expected = EXPECTED_STAGES[option]
    column = _column(DistillationColumn, tmp_path, num_plates)

    design = column.calculate_shortcut_design()

    assert design["num_plates"] == expected
    assert isinstance(design["num_plates"], int)
    assert isinstance(design["num_feed"], int)
    assert 1 <= design["num_feed"] <= expected
    assert design["min_reflux"] == pytest.approx(_underwood_minimum_reflux(),
                                                 rel=UNDERWOOD_RTOL)
    assert design["num_min"] == pytest.approx(_fenske_minimum_stages(),
                                              rel=FENSKE_RTOL)


@pytest.mark.parametrize("option", sorted(EXPECTED_STAGES))
def test_steady_column_solves_with_every_stage_count_option(tmp_path,
                                                            option):
    """``solve_unit`` builds the plates for every stage-count option.

    The estimated and the Fenske-multiple counts raised ``TypeError`` in
    ``calc_plates``; the plate profiles now have one column per plate and
    one for the reboiler.
    """
    num_plates, expected = EXPECTED_STAGES[option]
    column = _column(DistillationColumn, tmp_path, num_plates,
                     SOLVE_FEED_STAGE)

    column.solve_unit(solve_ss=True)

    assert column.num_plates == expected
    assert column.result.num_plates == expected
    assert np.shape(column.result.x) == (len(FEED), expected + 1)
    assert np.shape(column.result.T) == (expected + 1,)


@pytest.mark.parametrize("option", sorted(EXPECTED_STAGES))
def test_dynamic_column_startup_uses_the_same_stage_count(tmp_path, option):
    """The dynamic column's startup design gives the same whole count."""
    num_plates, expected = EXPECTED_STAGES[option]
    column = _column(DynamicDistillation, tmp_path, num_plates)

    column.column_startup(DESIGN_TIME)

    assert column.num_plates == expected
    assert isinstance(column.num_plates, int)
    assert isinstance(column.num_feed, int)
