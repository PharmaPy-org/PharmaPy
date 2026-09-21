"""Distillation VLE calls must honor the configured activity model.

``_BaseDistillation`` and ``DynamicDistillation`` accept a ``gamma_model``
option, but several vapor-liquid equilibrium calls omitted it and so fell back
to the ideal default of ``getKeqVLE`` and ``getBubblePoint``. A column
configured with a non-ideal model then ordered feed volatilities, closed its
dynamic material balances, initialized its plate temperatures, and reported
vapor profiles from ideal K-values.

The self-contained two-species fixture needs no repository data. It is a
close-boiling pair in which the heavier-boiling species is dilute and shows a
strong positive deviation from ideality, so its activity coefficient is large
enough to reverse the volatility ordering. An ideal fallback is therefore
visible in the model outputs instead of hiding in the fourth decimal.

The shortcut flow split needs a third species. ``global_material_bce`` assigns
every non-key component wholly to one product, so in a binary -- where the two
declared keys are the only species -- the split follows from the recovery
specifications alone and cannot depend on the ranking. A separate three-species
fixture adds one non-key component that crosses the keys between the two models
while the declared key ordering itself is preserved, so the flows move for a
physical reason and both product streams stay realizable.

Expected K-values are assembled in this module from the fixture Antoine
constants and ``getActivityCoeff`` rather than from ``getKeqVLE``, so the
assertions do not restate the production expression under test.
"""

import json
import re

import numpy as np
import pytest
from scipy.optimize import brentq

import PharmaPy.Distillation as distillation
from PharmaPy.Phases import LiquidPhase
from PharmaPy.Streams import LiquidStream


pytestmark = pytest.mark.unit


ACTIVITY_MODEL = 'UNIQUAC'

# Two-species UNIQUAC fixture. The pair is close-boiling: the Antoine B
# constants differ by 25 [K], which alone gives a relative volatility of only
# about 1.14 [-] near the bubble point. Both species carry identical UNIQUAC
# size and surface parameters, so for a binary mixture the combinatorial
# contribution to ln(gamma) cancels identically and the entire deviation from
# ideality comes from the residual (interaction) term. These values are a
# self-consistent modeling fixture, not measured data for any real pair.
# Antoine form: log10(P/[Pa]) = A - B/(T + C), with T [K].
THERMO_NONIDEAL_BINARY = {
    "light": {
        "mw": 32.0,  # [g/mol]
        "t_crit": 650.0,  # [K]
        "rho_liq": 800.0,  # [kg/m**3]
        "cp_liq": [75.0],  # [J/mol/K]
        "p_vap": [9.0, 1700.0, -40.0],  # Antoine A [-], B [K], C [K]
        "delta_hvap": {"value": 40000.0, "temp_ref": 350.0},  # [J/mol], [K]
        "ri": 1.4311,  # [-], UNIQUAC volume parameter
        "qi": 1.4320,  # [-], UNIQUAC surface parameter
        "qip": 1.4320,  # [-], UNIQUAC residual surface parameter
    },
    "heavy": {
        "mw": 100.0,  # [g/mol]
        "t_crit": 700.0,  # [K]
        "rho_liq": 1250.0,  # [kg/m**3]
        "cp_liq": [150.0],  # [J/mol/K]
        "p_vap": [9.0, 1725.0, -40.0],  # Antoine A [-], B [K], C [K]
        "delta_hvap": {"value": 60000.0, "temp_ref": 350.0},  # [J/mol], [K]
        "ri": 1.4311,  # [-], UNIQUAC volume parameter
        "qi": 1.4320,  # [-], UNIQUAC surface parameter
        "qip": 1.4320,  # [-], UNIQUAC residual surface parameter
    },
    # UNIQUAC binary interaction energies a_mk [J/mol], ordered
    # (light, heavy). The off-diagonal pair is positive and asymmetric, which
    # is the signature of a positive-deviation mixture.
    "interaction": {"amk": [[0.0, 2500.0], [500.0, 0.0]]},
}

SPECIES = ("light", "heavy")
ANTOINE = np.array(
    [THERMO_NONIDEAL_BINARY[name]["p_vap"] for name in SPECIES]
)  # columns: A [-], B [K], C [K]

COLUMN_PRESSURE = 101325.0  # [Pa]
FEED_TEMPERATURE = 350.0  # [K]
FEED_MOLE_FRAC = np.array([0.75, 0.25])  # [-], the heavy species is dilute
FEED_MOLE_FLOW = 3.0  # [mol/s]
PLATE_HOLDUP = 50.0  # [mol], liquid moles retained on one stage

# Bubble-point bracket for this fixture at COLUMN_PRESSURE; both pure-species
# boiling points lie near 470 [K].
TEMP_BRACKET = (300.0, 600.0)  # [K]

# Minimum separation the fixture must keep between the ideal and the
# configured non-ideal result. These are not tolerances: each assertion below
# uses one to confirm the two models really are distinguishable in that
# quantity, so no contrast can pass on rounding noise. The fixture delivers
# about 9.2 [K], 0.29 [-], and 0.22 [-] respectively, so each threshold sits
# several times below the separation it guards.
MIN_TEMP_SEPARATION = 1.0  # [K]
MIN_RESIDUAL_SEPARATION = 0.1  # [-]
MIN_VAPOR_FRAC_SEPARATION = 0.05  # [-]

# Three-species extension of the fixture above, used only by the shortcut
# flow-split test. ``global_material_bce`` builds the bottoms flow from four
# index sets -- the two declared keys, the species ranked above the light key,
# and the species ranked below the heavy key -- so those sets must partition the
# species for the balance to mean anything. A binary cannot exercise that: its
# two keys are the only species, the outer sets are always empty, and the split
# is fixed by the recovery specifications no matter how the feed is ranked.
#
# In this fixture the two keys carry zero mutual interaction energy and
# identical interaction energies with the non-key, so rows and columns 0 and 1
# of ``amk`` are exchange-symmetric and the UNIQUAC residual term returns
# gamma_light == gamma_heavy at every composition. The light key therefore stays
# above the heavy key and adjacent to it under both models -- the partition
# holds -- while the dilute non-key's own activity coefficient of about 2.3 [-]
# lifts it from the bottom of the ideal ranking to the top of the non-ideal one.
NONKEY_SPECIES = "nonkey"
TERNARY_SPECIES = ("light", "heavy", NONKEY_SPECIES)
NONKEY_INDEX = TERNARY_SPECIES.index(NONKEY_SPECIES)

THERMO_NONIDEAL_TERNARY = {
    # The two keys are the same species as the binary fixture above.
    "light": THERMO_NONIDEAL_BINARY["light"],
    "heavy": THERMO_NONIDEAL_BINARY["heavy"],
    NONKEY_SPECIES: {
        "mw": 60.0,  # [g/mol]
        "t_crit": 675.0,  # [K]
        "rho_liq": 1000.0,  # [kg/m**3]
        "cp_liq": [110.0],  # [J/mol/K]
        # Antoine B lies 35 [K] above the heavy key's, so under ideal K-values
        # the non-key is the least volatile of the three.
        "p_vap": [9.0, 1760.0, -40.0],  # Antoine A [-], B [K], C [K]
        "delta_hvap": {"value": 50000.0, "temp_ref": 350.0},  # [J/mol], [K]
        "ri": 1.4311,  # [-], UNIQUAC volume parameter
        "qi": 1.4320,  # [-], UNIQUAC surface parameter
        "qip": 1.4320,  # [-], UNIQUAC residual surface parameter
    },
    # UNIQUAC binary interaction energies a_mk [J/mol], ordered
    # (light, heavy, nonkey). The key-key entries are zero, which is what makes
    # the two keys exchange-symmetric; the key/non-key pair reuses the positive,
    # asymmetric energies of the binary fixture above.
    "interaction": {
        "amk": [
            [0.0, 0.0, 2500.0],
            [0.0, 0.0, 2500.0],
            [500.0, 500.0, 0.0],
        ],
    },
}

# The non-key is dilute, so it perturbs the key pair only weakly, and it is the
# single species that changes product between the two rankings: its feed flow is
# exactly the difference between the two shortcut splits.
TERNARY_FEED_MOLE_FRAC = np.array([0.55, 0.35, 0.10])  # [-]


def _thermo_file(tmp_path, database=None, name="thermo_nonideal_binary.json"):
    """Write a fixture thermodynamic database to a temporary file.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Directory provided by the pytest ``tmp_path`` fixture.
    database : dict, optional
        Thermodynamic database to serialize. The default is the two-species
        ``THERMO_NONIDEAL_BINARY`` fixture.
    name : str, optional
        File name to write inside ``tmp_path``. The default is the two-species
        database file name.

    Returns
    -------
    str
        Path to the database file.
    """
    if database is None:
        database = THERMO_NONIDEAL_BINARY

    path = tmp_path / name
    path.write_text(json.dumps(database))

    return str(path)


def _vapor_pressure(temp):
    """Pure-component saturation pressures from the fixture Antoine constants.

    Parameters
    ----------
    temp : float or ndarray
        Temperature [K]. A 1-D array is broadcast over the species axis.

    Returns
    -------
    ndarray
        Saturation pressures with the species along the last axis [Pa].
    """
    a_ct, b_ct, c_ct = ANTOINE.T  # [-], [K], [K]

    temp = np.asarray(temp, dtype=float)
    if temp.ndim == 1:
        temp = temp[:, np.newaxis]  # [K]

    return 10**(a_ct - b_ct / (temp + c_ct))  # [Pa]


def _expected_k_values(phase, temp, mole_frac, model=ACTIVITY_MODEL):
    """Modified Raoult's law K-values for the fixture mixture.

    Parameters
    ----------
    phase : ThermoPhysicalManager
        Phase or stream carrying the fixture activity-model parameters.
    temp : float or ndarray
        Temperature [K].
    mole_frac : ndarray
        Liquid mole fractions [-], with the species along the last axis.
    model : str, optional
        Activity-coefficient model name. The default is ``'UNIQUAC'``.

    Returns
    -------
    ndarray
        Vapor-liquid equilibrium ratios ``y/x`` [-].

    Notes
    -----
    ``K_i = gamma_i * Psat_i / P`` is assembled here from the fixture Antoine
    constants and ``getActivityCoeff``, deliberately bypassing ``getKeqVLE``,
    which is the call under test.
    """
    gamma = phase.getActivityCoeff(
        method=model, mole_frac=np.asarray(mole_frac), temp=temp)  # [-]

    return gamma * _vapor_pressure(temp) / COLUMN_PRESSURE  # [-]


def _expected_bubble_temp(phase, mole_frac, model=ACTIVITY_MODEL):
    """Bubble temperature of the fixture mixture at ``COLUMN_PRESSURE``.

    Parameters
    ----------
    phase : ThermoPhysicalManager
        Phase or stream carrying the fixture activity-model parameters.
    mole_frac : ndarray
        Liquid mole fractions [-].
    model : str, optional
        Activity-coefficient model name. The default is ``'UNIQUAC'``.

    Returns
    -------
    float
        Bubble temperature [K].

    Notes
    -----
    The bubble condition ``sum(x_i * K_i(T, x)) = 1`` is solved here with a
    bracketing method on the independently assembled K-values, so the result
    does not depend on ``getBubblePoint``, which is the call under test.
    """
    def bubble_residual(temp):
        """Bubble-point objective ``sum(x*K) - 1`` [-]."""
        k_vals = _expected_k_values(phase, temp, mole_frac, model=model)  # [-]

        return np.dot(mole_frac, k_vals) - 1  # [-]

    return brentq(bubble_residual, *TEMP_BRACKET, xtol=1e-10)  # [K]


def _feed_stream(thermo_path):
    """Build the fixture feed stream.

    Parameters
    ----------
    thermo_path : str
        Path to the two-species database file.

    Returns
    -------
    LiquidStream
        Feed at ``COLUMN_PRESSURE`` [Pa] and ``FEED_TEMPERATURE`` [K].
    """
    return LiquidStream(
        thermo_path,
        temp=FEED_TEMPERATURE,  # [K]
        pres=COLUMN_PRESSURE,  # [Pa]
        mole_frac=FEED_MOLE_FRAC,  # [-]
        mole_flow=FEED_MOLE_FLOW,  # [mol/s]
    )


def _holdup_phase(thermo_path):
    """Build the per-plate liquid holdup phase.

    Parameters
    ----------
    thermo_path : str
        Path to the two-species database file.

    Returns
    -------
    LiquidPhase
        Holdup of ``PLATE_HOLDUP`` [mol] at the feed composition.
    """
    return LiquidPhase(
        thermo_path,
        temp=FEED_TEMPERATURE,  # [K]
        pres=COLUMN_PRESSURE,  # [Pa]
        mole_frac=FEED_MOLE_FRAC,  # [-]
        moles=PLATE_HOLDUP,  # [mol]
    )


def _shortcut_column(thermo_path, gamma_model):
    """Build a shortcut-design column with the fixture feed.

    Parameters
    ----------
    thermo_path : str
        Path to the two-species database file.
    gamma_model : str
        Activity-coefficient model name.

    Returns
    -------
    DistillationColumn
        Column whose ``Inlet`` is the fixture feed stream.
    """
    column = distillation.DistillationColumn(
        pres=COLUMN_PRESSURE,  # [Pa]
        q_feed=1.0,  # [-], saturated-liquid feed
        LK="light",
        HK="heavy",
        perc_LK=95.0,  # [%]
        perc_HK=5.0,  # [%]
        gamma_model=gamma_model,
    )
    column.Inlet = _feed_stream(thermo_path)

    return column


def test_fixture_activity_model_reverses_ideal_volatility_order(tmp_path):
    """The fixture separates the two activity models by construction.

    Every other test in this module contrasts a non-ideal column with the ideal
    fallback, so those contrasts are only meaningful while the fixture still
    distinguishes the models. This pins that premise directly: the close-boiling
    pair is more volatile in the light species under ideal K-values and in the
    heavy species once the UNIQUAC residual term is applied.
    """
    stream = _feed_stream(_thermo_file(tmp_path))

    temp_bubble = _expected_bubble_temp(stream, FEED_MOLE_FRAC)  # [K]
    gamma = stream.getActivityCoeff(
        method=ACTIVITY_MODEL, mole_frac=FEED_MOLE_FRAC,
        temp=temp_bubble)  # [-]

    # Both deviations are positive and the dilute heavy species carries the
    # larger one, which is what reverses the ordering.
    assert gamma[1] > gamma[0] > 1.0

    k_ideal = _expected_k_values(
        stream, _expected_bubble_temp(stream, FEED_MOLE_FRAC, model='ideal'),
        FEED_MOLE_FRAC, model='ideal')  # [-]
    k_nonideal = _expected_k_values(stream, temp_bubble, FEED_MOLE_FRAC)  # [-]

    assert k_ideal[0] > k_ideal[1]
    assert k_nonideal[1] > k_nonideal[0]


def test_feed_volatility_order_uses_configured_activity_model(tmp_path):
    """Shortcut key ordering follows the configured activity model.

    ``global_material_bce`` ranks the feed by K-value to classify non-key
    components. Ranking a non-ideal feed ideally reports the wrong component as
    the most volatile, and that ordering is what the rest of the shortcut design
    is built on.
    """
    thermo_path = _thermo_file(tmp_path)

    nonideal = _shortcut_column(thermo_path, ACTIVITY_MODEL)
    nonideal.global_material_bce()

    ideal = _shortcut_column(thermo_path, 'ideal')
    ideal.global_material_bce()

    assert nonideal.sorted_by_volatility == ["heavy", "light"]
    assert ideal.sorted_by_volatility == ["light", "heavy"]


def test_volatility_diagnostic_reports_model_bubble_point(tmp_path, capsys):
    """The reported ordering temperature is the one the ordering was made at.

    When the declared keys are not adjacent in the computed ranking,
    ``global_material_bce`` prints the ranking together with the feed bubble
    temperature so the user can re-check their key selection. Computing that
    bubble point ideally while ranking non-ideally reports a temperature the
    printed ranking was never evaluated at, which is exactly the diagnostic a
    user consults after an unexpected ordering.
    """
    thermo_path = _thermo_file(tmp_path)
    column = _shortcut_column(thermo_path, ACTIVITY_MODEL)

    column.global_material_bce()

    # The fixture reverses the ordering, so the non-adjacent-keys diagnostic is
    # the branch that runs here.
    printed = capsys.readouterr().out
    assert "not adjacent" in printed

    match = re.search(r"T_bubble = ([0-9.]+) \[K\]", printed)
    assert match is not None, printed
    reported_temp = float(match.group(1))  # [K]

    expected_temp = _expected_bubble_temp(column.Inlet, FEED_MOLE_FRAC)  # [K]
    ideal_temp = _expected_bubble_temp(
        column.Inlet, FEED_MOLE_FRAC, model='ideal')  # [K]

    # The message prints one decimal, so compare at that resolution.
    assert reported_temp == pytest.approx(expected_temp, abs=0.05)  # [K]
    assert abs(expected_temp - ideal_temp) > MIN_TEMP_SEPARATION  # [K]


def _ternary_shortcut_column(thermo_path, gamma_model):
    """Build a shortcut-design column on the three-species fixture.

    Parameters
    ----------
    thermo_path : str
        Path to the three-species database file.
    gamma_model : str
        Activity-coefficient model name.

    Returns
    -------
    DistillationColumn
        Column whose ``Inlet`` is the three-species feed stream.
    """
    column = distillation.DistillationColumn(
        pres=COLUMN_PRESSURE,  # [Pa]
        q_feed=1.0,  # [-], saturated-liquid feed
        LK="light",
        HK="heavy",
        perc_LK=95.0,  # [%]
        perc_HK=5.0,  # [%]
        gamma_model=gamma_model,
    )
    column.Inlet = LiquidStream(
        thermo_path,
        temp=FEED_TEMPERATURE,  # [K]
        pres=COLUMN_PRESSURE,  # [Pa]
        mole_frac=TERNARY_FEED_MOLE_FRAC,  # [-]
        mole_flow=FEED_MOLE_FLOW,  # [mol/s]
    )

    return column


def test_feed_volatility_order_changes_shortcut_flow_split(tmp_path):
    """The activity model reaches the shortcut distillate and bottoms flows.

    ``global_material_bce`` sends every non-key component wholly to the
    distillate or wholly to the bottoms according to where the volatility
    ranking places it relative to the keys, so an ideal ranking of a non-ideal
    feed does not merely mislabel the order: it moves the shortcut material
    balance itself.

    The three-species fixture keeps the declared light key above the heavy key
    and adjacent to it under both models, so the index sets the balance is built
    from remain a partition of the species and both product streams stay
    physically realizable. Only the non-key crosses the keys, and it takes its
    whole feed flow with it.
    """
    thermo_path = _thermo_file(
        tmp_path, THERMO_NONIDEAL_TERNARY, "thermo_nonideal_ternary.json")

    nonideal = _ternary_shortcut_column(thermo_path, ACTIVITY_MODEL)
    (x_dist_nonideal, x_bot_nonideal,
     dist_nonideal, bot_nonideal) = nonideal.global_material_bce()

    ideal = _ternary_shortcut_column(thermo_path, 'ideal')
    (x_dist_ideal, x_bot_ideal,
     dist_ideal, bot_ideal) = ideal.global_material_bce()

    # The declared keys stay adjacent in both rankings. The non-key sits above
    # them under the configured model and below them under the ideal fallback.
    assert nonideal.sorted_by_volatility == ["nonkey", "light", "heavy"]
    assert ideal.sorted_by_volatility == ["light", "heavy", "nonkey"]

    for x_dist, x_bot, dist_flow, bot_flow in (
            (x_dist_nonideal, x_bot_nonideal, dist_nonideal, bot_nonideal),
            (x_dist_ideal, x_bot_ideal, dist_ideal, bot_ideal)):
        # Each product must carry a positive flow of a nonnegative, normalized
        # composition. These are the assertions that a ranking which breaks the
        # index partition fails, and it breaks it silently: the shortcut
        # balance then reports a negative flow and unnormalized fractions
        # rather than raising.
        assert dist_flow > 0  # [mol/s]
        assert bot_flow > 0  # [mol/s]

        assert np.all(x_dist >= 0)  # [-]
        assert np.all(x_bot >= 0)  # [-]
        assert x_dist.sum() == pytest.approx(1.0)  # [-]
        assert x_bot.sum() == pytest.approx(1.0)  # [-]

    # The non-key leaves entirely in one product, and which one is the only
    # thing the activity model changes here, so the two splits differ by its
    # whole feed flow.
    nonkey_flow = (
        FEED_MOLE_FLOW * TERNARY_FEED_MOLE_FRAC[NONKEY_INDEX])  # [mol/s]

    assert dist_nonideal * x_dist_nonideal[NONKEY_INDEX] == pytest.approx(
        nonkey_flow)  # [mol/s]
    assert bot_ideal * x_bot_ideal[NONKEY_INDEX] == pytest.approx(
        nonkey_flow)  # [mol/s]
    assert dist_nonideal - dist_ideal == pytest.approx(nonkey_flow)  # [mol/s]


def test_relative_volatility_uses_activity_model_bubble_point(tmp_path):
    """``get_alpha`` evaluates its bubble temperature with the same model.

    The K-values in ``get_alpha`` were already non-ideal, but the bubble
    temperature they were evaluated at was not, so the relative volatilities
    came from a temperature that is not an equilibrium state of the configured
    model.
    """
    thermo_path = _thermo_file(tmp_path)
    column = _shortcut_column(thermo_path, ACTIVITY_MODEL)

    temp_bubble = _expected_bubble_temp(column.Inlet, FEED_MOLE_FRAC)  # [K]
    k_vals = _expected_k_values(
        column.Inlet, temp_bubble, FEED_MOLE_FRAC)  # [-]
    expected_alpha = k_vals / k_vals[column.HK_index]  # [-]

    alpha = column.get_alpha(COLUMN_PRESSURE, FEED_MOLE_FRAC)  # [-]

    np.testing.assert_allclose(alpha, expected_alpha, rtol=1e-8)

    # The ideal bubble temperature is several kelvin away for this fixture, so
    # the same non-ideal K-values evaluated there give different volatilities.
    temp_ideal = _expected_bubble_temp(
        column.Inlet, FEED_MOLE_FRAC, model='ideal')  # [K]
    assert abs(temp_ideal - temp_bubble) > MIN_TEMP_SEPARATION  # [K]

    k_offpoint = _expected_k_values(
        column.Inlet, temp_ideal, FEED_MOLE_FRAC)  # [-]
    alpha_offpoint = k_offpoint / k_offpoint[column.HK_index]  # [-]
    assert alpha[column.LK_index] != pytest.approx(
        alpha_offpoint[column.LK_index], rel=1e-4)


def _dynamic_column(thermo_path, gamma_model, num_plates=4, num_feed=2):
    """Build a dynamic column with the shortcut-design results supplied.

    Parameters
    ----------
    thermo_path : str
        Path to the two-species database file.
    gamma_model : str
        Activity-coefficient model name.
    num_plates : int, optional
        Equilibrium-stage count [-]. The default is 4.
    num_feed : int, optional
        Feed tray counted from the top [-]. The default is 2.

    Returns
    -------
    DynamicDistillation
        Column with feed, holdup phase, and operating point assigned, ready to
        evaluate ``material_balances`` without a solver.
    """
    column = distillation.DynamicDistillation(
        pres=COLUMN_PRESSURE,  # [Pa]
        q_feed=1.0,  # [-], saturated-liquid feed
        LK="light",
        HK="heavy",
        perc_LK=95.0,  # [%]
        perc_HK=5.0,  # [%]
        gamma_model=gamma_model,
    )
    column.Inlet = _feed_stream(thermo_path)
    column.Phases = _holdup_phase(thermo_path)

    # Operating point that ``column_startup`` would otherwise take from the
    # shortcut design. The split is a self-consistent overall balance on the
    # fixture feed of FEED_MOLE_FLOW [mol/s].
    column.num_plates = num_plates  # [-]
    column.num_feed = num_feed  # [-]
    column.dist_flowrate = 1.0  # [mol/s]
    column.bot_flowrate = FEED_MOLE_FLOW - 1.0  # [mol/s]
    column.reflux = 2.0  # [-], L/D

    return column


def _plate_profiles(num_plates):
    """Non-uniform plate temperatures and compositions for a dynamic column.

    Parameters
    ----------
    num_plates : int
        Equilibrium-stage count [-]; the state carries ``num_plates + 1`` rows.

    Returns
    -------
    temp : ndarray
        Plate temperatures [K], top to bottom.
    x_liq : ndarray
        Plate liquid mole fractions [-] with shape ``(num_plates + 1, 2)``.

    Notes
    -----
    The profiles vary from plate to plate, so a K-value evaluated per plate
    cannot be confused with a single mixture-average value, and the light
    fraction decreases down the column as it does in a rectification profile.
    """
    num_rows = num_plates + 1  # [-]

    temp = np.linspace(455.0, 475.0, num_rows)  # [K]
    x_light = np.linspace(0.85, 0.35, num_rows)  # [-]
    x_liq = np.column_stack((x_light, 1 - x_light))  # [-]

    return temp, x_liq


def test_dynamic_material_balances_use_configured_activity_model(tmp_path):
    """Dynamic residuals and vapor compositions use the configured model.

    The algebraic temperature residual is ``sum(x * (K - 1))`` and the plate
    coupling uses ``y = K * x``, so an ideal K-value here drives the whole DAE
    towards an equilibrium state the configured model does not predict.
    """
    thermo_path = _thermo_file(tmp_path)
    column = _dynamic_column(thermo_path, ACTIVITY_MODEL)

    temp, x_liq = _plate_profiles(column.num_plates)  # [K], [-]

    residuals = column.material_balances(
        time=0.0, temp=temp, x_liq=x_liq)  # [-], [1/s]

    k_expected = _expected_k_values(column.Liquid_1, temp, x_liq)  # [-]
    expected_temp_residual = (x_liq * (k_expected - 1)).sum(axis=1)  # [-]

    np.testing.assert_allclose(
        residuals[:, 0], expected_temp_residual, rtol=1e-8)

    # The ideal fallback the fix removes is far from this residual, so the
    # assertion above cannot be satisfied by both models at once.
    k_ideal = _expected_k_values(
        column.Liquid_1, temp, x_liq, model='ideal')  # [-]
    ideal_temp_residual = (x_liq * (k_ideal - 1)).sum(axis=1)  # [-]
    residual_gap = np.max(
        np.abs(ideal_temp_residual - expected_temp_residual))  # [-]
    assert residual_gap > MIN_RESIDUAL_SEPARATION


def test_dynamic_material_balances_track_the_selected_model(tmp_path):
    """Selecting the ideal model still produces ideal residuals.

    Together with the non-ideal case, this shows ``material_balances`` reads
    ``gamma_model`` rather than hard-coding either branch.
    """
    thermo_path = _thermo_file(tmp_path)
    column = _dynamic_column(thermo_path, 'ideal')

    temp, x_liq = _plate_profiles(column.num_plates)  # [K], [-]

    residuals = column.material_balances(
        time=0.0, temp=temp, x_liq=x_liq)  # [-], [1/s]

    k_ideal = _expected_k_values(
        column.Liquid_1, temp, x_liq, model='ideal')  # [-]
    expected_temp_residual = (x_liq * (k_ideal - 1)).sum(axis=1)  # [-]

    np.testing.assert_allclose(
        residuals[:, 0], expected_temp_residual, rtol=1e-8)


def test_dynamic_vapor_profiles_use_configured_activity_model(tmp_path):
    """Reported vapor profiles use the configured activity model.

    ``retrieve_results`` recomputes ``y = K * x`` for every stored time and
    plate. An ideal K-value there leaves the solved liquid profiles non-ideal
    while the reported vapor profiles are not, so the two disagree inside one
    result object.
    """
    thermo_path = _thermo_file(tmp_path)
    column = _dynamic_column(thermo_path, ACTIVITY_MODEL)

    temp, x_liq = _plate_profiles(column.num_plates)  # [K], [-]
    plate_states = np.column_stack((temp, x_liq))  # [K], [-]

    # Two stored time points, flattened the way the DAE solver returns them.
    states = np.vstack([plate_states.ravel(), plate_states.ravel()])
    time = np.array([0.0, 10.0])  # [s]

    column.retrieve_results(time, states)

    k_expected = _expected_k_values(column.Liquid_1, temp, x_liq)  # [-]
    expected_y = k_expected * x_liq  # [-]

    for idx, name in enumerate(SPECIES):
        np.testing.assert_allclose(
            column.result.y_vap[name][-1], expected_y[:, idx], rtol=1e-8)

    k_ideal = _expected_k_values(
        column.Liquid_1, temp, x_liq, model='ideal')  # [-]
    ideal_y = k_ideal * x_liq  # [-]
    assert np.max(np.abs(ideal_y - expected_y)) > MIN_VAPOR_FRAC_SEPARATION


def test_dynamic_startup_temperature_uses_configured_activity_model(
        tmp_path, monkeypatch):
    """Initial plate temperatures use the configured activity model.

    ``solve_unit`` seeds every plate at the bubble point of the initial liquid
    holdup. Seeding a non-ideal column at its ideal bubble point starts the DAE
    off the equilibrium manifold its own algebraic residuals enforce, so this is
    a consistent-initialization failure rather than a small offset.

    ``PharmaPy._assimulo`` exports lazy factories, so importing ``Distillation``
    no longer imports Assimulo. ``solve_unit`` still calls ``Implicit_Problem``
    and ``IDA``, though, and each call imports it, so patching those two names
    is what keeps this test in the Assimulo-free core lane. The doubles also
    capture the initial state handed to the DAE, which is the quantity under
    test and is not recoverable from a completed solve. This is the same seam
    ``tests/test_deliquoring_particle_size_units.py`` patches.
    """
    thermo_path = _thermo_file(tmp_path)

    captured = {}

    class FakeProblem:
        """Implicit-problem double that records the initial state."""

        def __init__(self, residual, y0, yd0, t0=0.0, sw0=None):
            """Record the initial state handed to the DAE problem.

            Parameters
            ----------
            residual : callable
                Residual function of the DAE.
            y0 : ndarray
                Flattened initial state; temperatures are [K] and mole
                fractions are [-].
            yd0 : ndarray
                Flattened initial state derivative, [K/s] and [1/s].
            t0 : float, optional
                Initial time [s].
            sw0 : list of bool, optional
                Initial state-event switches.
            """
            captured['y0'] = np.asarray(y0)

    class FakeSolver:
        """Solver double that replays the initial state instead of solving."""

        def __init__(self, problem):
            """Store the problem handed to the solver.

            Parameters
            ----------
            problem : FakeProblem
                Implicit-problem double carrying the initial state.
            """
            self.problem = problem

        def make_consistent(self, mode):
            """Accept the consistency mode without solving.

            Parameters
            ----------
            mode : str
                Assimulo initialization mode name.

            Returns
            -------
            None
            """

        def simulate(self, final_time, ncp_list=None):
            """Return the recorded initial state as two stored time points.

            Parameters
            ----------
            final_time : float
                Final integration time [s].
            ncp_list : array_like, optional
                Requested output times [s].

            Returns
            -------
            tuple
                Times [s], states, and state derivatives, [K/s] and [1/s].
            """
            states = np.vstack([captured['y0'], captured['y0']])

            return (np.array([0.0, final_time]), states,
                    np.zeros_like(states))

    class StartupColumn(distillation.DynamicDistillation):
        """Dynamic column double with a deterministic shortcut design."""

        def calculate_shortcut_design(self, time=None):
            """Return a fixed shortcut design for the fixture feed.

            Parameters
            ----------
            time : float, optional
                Shortcut-design time [s].

            Returns
            -------
            dict
                Shortcut-design result. Mole fractions are [-], molar flows are
                [mol/s], reflux is [-], and stage counts are [-].
            """
            return {
                "material_balances": {
                    "bottom_flow": FEED_MOLE_FLOW - 1.0,  # [mol/s]
                    "dist_flow": 1.0,  # [mol/s]
                    "x_dist": np.array([0.95, 0.05]),  # [-]
                    "x_bottom": np.array([0.65, 0.35]),  # [-]
                },
                "min_reflux": 1.2,  # [-]
                "num_min": 3.0,  # [-]
                "reflux": 2.0,  # [-]
                "num_plates": 4.0,  # [-]
                "num_feed": 2.0,  # [-]
            }

        def retrieve_results(self, time, states):
            """Skip result retrieval, which this test does not exercise.

            Parameters
            ----------
            time : array_like
                Simulated time points [s].
            states : ndarray
                Flattened state history.

            Returns
            -------
            None
            """

    column = StartupColumn(
        pres=COLUMN_PRESSURE,  # [Pa]
        q_feed=1.0,  # [-], saturated-liquid feed
        LK="light",
        HK="heavy",
        perc_LK=95.0,  # [%]
        perc_HK=5.0,  # [%]
        gamma_model=ACTIVITY_MODEL,
    )
    column.Inlet = _feed_stream(thermo_path)
    column.Phases = _holdup_phase(thermo_path)

    monkeypatch.setattr(distillation, 'Implicit_Problem', FakeProblem)
    monkeypatch.setattr(distillation, 'IDA', FakeSolver)

    column.solve_unit(runtime=10.0)  # [s]

    holdup_frac = column.Liquid_1.mole_frac  # [-]
    expected_temp = _expected_bubble_temp(column.Liquid_1, holdup_frac)  # [K]
    ideal_temp = _expected_bubble_temp(
        column.Liquid_1, holdup_frac, model='ideal')  # [K]

    init_states = captured['y0'].reshape(-1, column.len_states)
    temp_init = init_states[:, 0]  # [K]

    np.testing.assert_allclose(temp_init, expected_temp, rtol=1e-8)
    assert abs(expected_temp - ideal_temp) > MIN_TEMP_SEPARATION  # [K]
