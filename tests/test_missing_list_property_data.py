"""Missing list-valued property data: parser sentinel and consumers (#414).

``ParseDatabase`` stores a NaN row for a species that supplies no
coefficients for a list-valued property (``cp_liq``, ``cp_vapor``,
``visc_liq``, ``p_vap``, ``diffusivity``...), as it stores NaN for a missing
scalar. Earlier releases stored a row of zeros, which evaluated silently as a
zero heat capacity, a 1 mPa*s viscosity or a 1 Pa vapor pressure.

The tests cover the exact parsed arrays and their species ordering, then each
consumer: a result that needs a missing row raises ``MissingPropertyError``
(an ``AttributeError``) naming the species and the property, and a result that
does not need it (zero fraction, species in no reaction, supercritical species
routed to Henry's law, non-volatile drying carrier, species absent from a
reactor) keeps its value, with no NaN leaking through ``NaN * 0``. Expected
values are closed-form evaluations of the synthetic correlations below, written
out independently of PharmaPy, or the same run with arbitrary data for the
unused species, which must agree exactly.

All databases are synthetic, rounded test data, not measured properties, except
the shipped evaporator, nitrogen and PFR databases. Core tests use real phases,
a real ``Drying`` unit and a real ``PlugFlowReactor.energy_steady`` evaluation
without a solver. The ``assimulo`` tests run CVode reactors (Batch, CSTR,
Semibatch, transient and steady PFR, including feed pulses and dynamic feed
overrides) and the shipped evaporator database merged with the shipped nitrogen
record through IDA.
"""

import copy
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from PharmaPy.Phases import LiquidPhase, SolidPhase, VaporPhase
from PharmaPy.ThermoModule import (MissingPropertyError, ParseDatabase,
                                   ThermoPhysicalManager)


REPO_ROOT = Path(__file__).resolve().parents[1]

TEMP_REF = 298.15  # [K], default lower limit of the enthalpy integrals
# Watson exponent: K. M. Watson, Ind. Eng. Chem. 1943, 35, 398-406; Poling,
# Prausnitz & O'Connell, The Properties of Gases and Liquids, 5th ed.,
# Eq. (7-11.1).
WATSON_EXPONENT = 0.38  # [-]
# Closed-form expectations differ from PharmaPy's sums only by roundoff.
CLOSED_FORM_RTOL = 1e-12  # [-]
# Newton-root tolerance (bubble and dew points, Mixer outlet temperature):
# scipy.optimize.newton stops when a step is below its default tol = 1.48e-8
# in the root's units, here [K]; the closed-form expectations are exact, so
# 1e-6 K leaves margin for the last step.
BUBBLE_POINT_ATOL = 1e-6  # [K]
# Same stopping rule for bubble pressures: newton's tol = 1.48e-8 applies to
# the secant step in [Pa]; 1e-6 Pa keeps the same margin and stays far above
# the float spacing of the 1e3-1e5 Pa roots compared (about 1e-11 Pa).
BUBBLE_PRESSURE_ATOL = 1e-6  # [Pa]

# --------------------------------------------------------------------------
# Parser fixture: three species with mixed present and missing properties.
# "solvent" supplies the longest cp_liq list (3 terms); "solute" supplies a
# constant cp_liq (1 term), an empty cp_vapor list and a null p_vap; "gas"
# supplies cp_liq in the structured {"value": ...} form (2 terms) and omits
# visc_liq and t_crit. Units: mw [g/mol], t_crit [K], cp coefficients
# [J/mol/K/K**k], Antoine A [-], B [K], C [K], visc_liq log10(mPa*s) terms.
PARSER_DATABASE = {
    "solvent": {
        "mw": 72.0,
        "t_crit": 540.0,
        "cp_liq": [60.0, 0.4, -1.25e-3],
        "p_vap": [9.25, 1250.0, -40.5],
        "visc_liq": [-2.75, 476.5, 4.5e-3, -6.5e-6],
    },
    "solute": {
        "mw": 150.0,
        "t_crit": 615.0,
        "cp_liq": [120.0],
        "cp_vapor": [],
        "p_vap": None,
        "visc_liq": [-4.5, 1027.0, 6.0e-3, -5.75e-6],
    },
    "gas": {
        "mw": 28.0,
        "cp_liq": {"value": [30.0, 0.125]},
        "cp_vapor": [29.0, 2.0e-3],
        "p_vap": [8.75, 264.5, -6.75],
    },
}

# --------------------------------------------------------------------------
# Liquid fixture. "inert" is a nitrogen-like gas (mw, t_crit and Henry
# constant copied from data/evaporator/props_nitrogen.json; liquid density of
# nitrogen at its normal boiling point, rounded) that supplies no cp_liq,
# visc_liq, p_vap or diffusivity, while the other two species supply them.
# Units: mw [g/mol], t_crit [K], rho_liq [kg/m**3], henry_constant [Pa] on a
# liquid mole-fraction basis, cp_liq [J/mol/K, J/mol/K**2], Antoine A [-],
# B [K], C [K] for log10(p/[Pa]), visc_liq terms of log10(mu/[mPa*s]) =
# A + B/T + C*T + D*T**2, diffusivity [m**2/s] with respect to each species.
SOLVENT_CP = (60.0, 0.15)  # [J/mol/K], [J/mol/K**2]
SOLUTE_CP = (300.0,)  # [J/mol/K]
SOLVENT_ANTOINE = (10.24, 1595.8, -46.7)  # [-], [K], [K]; ethanol-like, ~1 bar at 351 K
SOLUTE_ANTOINE = (9.0, 3000.0, -50.0)  # [-], [K], [K]
SOLVENT_VISC = (-2.0, 600.0, 0.0, 0.0)  # [-], [K], [1/K], [1/K**2]
SOLUTE_VISC = (-1.5, 700.0, 0.0, 0.0)  # [-], [K], [1/K], [1/K**2]
INERT_HENRY = 7.0e9  # [Pa]
LIQUID_DATABASE = {
    "solvent": {
        "mw": 46.0,
        "t_crit": 514.0,
        "rho_liq": 789.0,
        "cp_liq": list(SOLVENT_CP),
        "p_vap": list(SOLVENT_ANTOINE),
        "visc_liq": list(SOLVENT_VISC),
        "diffusivity": [1.0e-9, 1.25e-9, 2.0e-9],
    },
    "solute": {
        "mw": 180.0,
        "t_crit": 700.0,
        "rho_liq": 1200.0,
        "cp_liq": list(SOLUTE_CP),
        "p_vap": list(SOLUTE_ANTOINE),
        "visc_liq": list(SOLUTE_VISC),
        "diffusivity": [0.75e-9, 1.0e-9, 1.5e-9],
    },
    "inert": {
        "mw": 28.0134,
        "t_crit": 126.14,
        "rho_liq": 806.0,
        "henry_constant": INERT_HENRY,
    },
}
LIQUID_TEMP = 350.0  # [K], above the inert's t_crit, below the others'
BELOW_INERT_T_CRIT = 100.0  # [K], below the inert's 126.14 K t_crit
PRESSURE = 101325.0  # [Pa], standard atmosphere
# Asymmetric liquid compositions [-] in database order (solvent, solute,
# inert): the inert is either absent or present.
INERT_ABSENT = (0.25, 0.75, 0.0)  # [-]
INERT_PRESENT = (0.25, 0.70, 0.05)  # [-]

# --------------------------------------------------------------------------
# Vapor fixture. "butane_like" (t_crit 425 K) has no cp_vapor while the other
# species have it; it is subcritical at 400 K and supercritical at 450 K.
# "light_gas" is supercritical at both, "water_like" subcritical at both.
# Units: mw [g/mol], t_crit and tref_hvap [K], delta_hvap [J/mol] at
# tref_hvap, cp coefficients [J/mol/K, J/mol/K**2].
LIGHT_CP_VAPOR = (29.0, 2.0e-3)  # [J/mol/K], [J/mol/K**2]
VAPOR_DATABASE = {
    "light_gas": {
        "mw": 28.0, "t_crit": 150.0, "cp_liq": [30.0],
        "cp_vapor": list(LIGHT_CP_VAPOR),
        "delta_hvap": 5600.0, "tref_hvap": 77.0,
    },
    "butane_like": {
        "mw": 58.0, "t_crit": 425.0, "cp_liq": [140.0],
        "delta_hvap": 22400.0, "tref_hvap": 272.5,
    },
    "water_like": {
        "mw": 18.0, "t_crit": 647.0, "cp_liq": [75.0],
        "cp_vapor": [33.5],
        "delta_hvap": 40650.0, "tref_hvap": 373.15,
    },
}
SUBCRITICAL_TEMP = 400.0  # [K], below butane_like's t_crit
SUPERCRITICAL_TEMP = 450.0  # [K], above butane_like's t_crit
VAPOR_FRACTIONS = np.array([0.2, 0.3, 0.5])  # [-], asymmetric, all present
BUTANE_ABSENT = np.array([0.4, 0.0, 0.6])  # [-], butane_like absent


def _write(tmp_path, database, name="thermo.json"):
    """Write a property database as JSON.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Directory for the file.
    database : dict
        Species records with the units documented at their definition.
    name : str, optional
        File name.

    Returns
    -------
    str
        Path of the written file.
    """
    path = tmp_path / name
    path.write_text(json.dumps(database))
    return str(path)


def _poly_enthalpy(coefficients, temp, temp_ref=TEMP_REF):
    """Integrate an ascending cp polynomial in closed form.

    Parameters
    ----------
    coefficients : sequence of float
        ``c_k`` of ``cp = sum_k c_k * T**k`` [J/mol/K/K**k].
    temp, temp_ref : float
        Upper and lower integration limits [K].

    Returns
    -------
    float
        ``sum_k c_k / (k + 1) * (temp**(k + 1) - temp_ref**(k + 1))``
        [J/mol].
    """
    return sum(coef / (power + 1) * (temp**(power + 1) - temp_ref**(power + 1))
               for power, coef in enumerate(coefficients))


def _watson(species, temp):
    """Watson latent heat of a VAPOR_DATABASE species.

    Parameters
    ----------
    species : dict
        Record with ``t_crit`` and ``tref_hvap`` [K], ``delta_hvap`` [J/mol].
    temp : float
        Temperature at or below ``t_crit`` [K].

    Returns
    -------
    float
        Latent heat of vaporization [J/mol].
    """
    ratio = ((species["t_crit"] - temp)
             / (species["t_crit"] - species["tref_hvap"]))  # [-]
    return species["delta_hvap"] * ratio**WATSON_EXPONENT


def _antoine_pressure(coefficients, temp):
    """Antoine saturation pressure, ``log10(p/[Pa]) = A - B/(T + C)``.

    Parameters
    ----------
    coefficients : sequence of float
        ``A`` [-], ``B`` [K], ``C`` [K].
    temp : float
        Temperature [K].

    Returns
    -------
    float
        Saturation pressure [Pa].
    """
    a_ct, b_ct, c_ct = coefficients
    return 10**(a_ct - b_ct / (temp + c_ct))


def _liquid_viscosity(coefficients, temp):
    """Liquid viscosity, ``log10(mu/[mPa*s]) = A + B/T + C*T + D*T**2``.

    Parameters
    ----------
    coefficients : sequence of float
        ``A`` [-], ``B`` [K], ``C`` [1/K], ``D`` [1/K**2].
    temp : float
        Temperature [K].

    Returns
    -------
    float
        Viscosity [Pa*s].
    """
    a_ct, b_ct, c_ct, d_ct = coefficients
    return 10**(a_ct + b_ct / temp + c_ct * temp + d_ct * temp**2) / 1000


@pytest.fixture
def liquid_path(tmp_path):
    """Write LIQUID_DATABASE.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Test directory.

    Returns
    -------
    str
        JSON database path.
    """
    return _write(tmp_path, LIQUID_DATABASE)


@pytest.fixture
def vapor_path(tmp_path):
    """Write VAPOR_DATABASE.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Test directory.

    Returns
    -------
    str
        JSON database path.
    """
    return _write(tmp_path, VAPOR_DATABASE)


def _liquid(path, mole_frac):
    """Build a one-mole liquid at LIQUID_TEMP.

    Parameters
    ----------
    path : str
        Property database path.
    mole_frac : array-like
        Mole fractions [-], shape (3,).

    Returns
    -------
    LiquidPhase
        Real liquid phase.
    """
    return LiquidPhase(path, temp=LIQUID_TEMP, moles=1.0,
                       mole_frac=np.asarray(mole_frac, dtype=float))


# ------------------------------------------------------------------ parser
@pytest.mark.unit
def test_missing_property_error_is_an_attribute_error():
    """Handlers written for the pre-#414 AttributeError still catch it."""
    assert issubclass(MissingPropertyError, AttributeError)


@pytest.mark.unit
def test_parser_marks_missing_list_and_scalar_data_with_nan(tmp_path):
    """Exact arrays, species order, padding and the NaN sentinel."""
    parsed = ParseDatabase(_write(tmp_path, PARSER_DATABASE))
    nan = np.nan

    assert parsed["name_species"] == ["solvent", "solute", "gas"]
    np.testing.assert_array_equal(parsed["mw"], [72.0, 150.0, 28.0])
    np.testing.assert_array_equal(parsed["t_crit"], [540.0, 615.0, nan])
    # Shorter lists are padded with trailing zeros to the longest list.
    np.testing.assert_array_equal(
        parsed["cp_liq"],
        [[60.0, 0.4, -1.25e-3], [120.0, 0.0, 0.0], [30.0, 0.125, 0.0]])
    # An empty list, a null and an absent key all give a NaN row.
    np.testing.assert_array_equal(
        parsed["cp_vapor"], [[nan, nan], [nan, nan], [29.0, 2.0e-3]])
    np.testing.assert_array_equal(
        parsed["p_vap"],
        [[9.25, 1250.0, -40.5], [nan, nan, nan], [8.75, 264.5, -6.75]])
    np.testing.assert_array_equal(
        parsed["visc_liq"],
        [[-2.75, 476.5, 4.5e-3, -6.5e-6], [-4.5, 1027.0, 6.0e-3, -5.75e-6],
         [nan, nan, nan, nan]])
    for key in ("cp_liq", "cp_vapor", "p_vap", "visc_liq"):
        assert parsed[key].dtype == np.float64


@pytest.mark.unit
def test_parser_rows_follow_species_order_of_reordered_and_merged_files(
        tmp_path):
    """Reordering species or merging files moves whole rows with them."""
    reordered = {name: PARSER_DATABASE[name]
                 for name in ("gas", "solvent", "solute")}
    parsed = ParseDatabase(_write(tmp_path, reordered))
    nan = np.nan

    assert parsed["name_species"] == ["gas", "solvent", "solute"]
    np.testing.assert_array_equal(
        parsed["cp_vapor"], [[29.0, 2.0e-3], [nan, nan], [nan, nan]])
    np.testing.assert_array_equal(parsed["t_crit"], [nan, 540.0, 615.0])

    # The Evaporator nitrogen merge: the second file's species come last.
    first = {name: PARSER_DATABASE[name] for name in ("solute", "solvent")}
    second = {"gas": PARSER_DATABASE["gas"]}
    merged = ParseDatabase([_write(tmp_path, first, "first.json"),
                            _write(tmp_path, second, "second.json")])

    assert merged["name_species"] == ["solute", "solvent", "gas"]
    np.testing.assert_array_equal(
        merged["visc_liq"],
        [[-4.5, 1027.0, 6.0e-3, -5.75e-6], [-2.75, 476.5, 4.5e-3, -6.5e-6],
         [nan, nan, nan, nan]])
    np.testing.assert_array_equal(
        merged["cp_liq"],
        [[120.0, 0.0, 0.0], [60.0, 0.4, -1.25e-3], [30.0, 0.125, 0.0]])


@pytest.mark.unit
def test_parser_without_arrays_keeps_missing_values_as_supplied(tmp_path):
    """``to_arrays=False`` returns the raw per-species values."""
    parsed = ParseDatabase(_write(tmp_path, PARSER_DATABASE), to_arrays=False)

    assert parsed["cp_vapor"] == [None, [], [29.0, 2.0e-3]]
    assert parsed["p_vap"] == [[9.25, 1250.0, -40.5], None,
                               [8.75, 264.5, -6.75]]


@pytest.mark.unit
@pytest.mark.parametrize("species", ["solvent", "solute"])
def test_parser_rejects_scalar_for_list_valued_property(tmp_path, species):
    """A scalar where other species give a list is malformed, not padded.

    The scalar is placed first ("solvent") or after a list ("solute").
    """
    database = copy.deepcopy(PARSER_DATABASE)
    database[species]["cp_liq"] = 120.0  # [J/mol/K], scalar, not a list

    with pytest.raises(TypeError) as error:
        ParseDatabase(_write(tmp_path, database))

    assert str(error.value) == (
        f"Property 'cp_liq' of species '{species}' must be a list, as other "
        "species supply a list for it; got 120.0")


@pytest.mark.unit
@pytest.mark.parametrize("values", [
    ([], [], []), (None, None, None), ([], None, []), (None, [], None)],
    ids=["all_empty", "all_null", "empty_null_empty", "null_empty_null"])
def test_parser_drops_list_property_that_no_species_supplies(tmp_path,
                                                             values):
    """All-empty or all-null coefficient lists behave like an omitted key."""
    database = copy.deepcopy(PARSER_DATABASE)
    for name, value in zip(database, values):
        database[name]["cp_vapor"] = value

    parsed = ParseDatabase(_write(tmp_path, database))

    assert "cp_vapor" not in parsed
    # The other properties keep their arrays.
    np.testing.assert_array_equal(parsed["t_crit"], [540.0, 615.0, np.nan])


@pytest.mark.unit
def test_parser_keeps_all_null_scalar_property_as_nan(tmp_path):
    """An all-null scalar key keeps the established NaN representation."""
    database = copy.deepcopy(PARSER_DATABASE)
    for record in database.values():
        record["t_crit"] = None

    parsed = ParseDatabase(_write(tmp_path, database))

    np.testing.assert_array_equal(parsed["t_crit"], [np.nan] * 3)


# ------------------------------------------------------------ vapor phase
@pytest.mark.unit
@pytest.mark.parametrize("total_h", [True, False])
def test_vapor_enthalpy_rejects_supercritical_species_without_cp_vapor(
        vapor_path, total_h):
    """The mixed-database case of #414 raises instead of returning 0 J/mol."""
    vapor = VaporPhase(vapor_path, temp=SUPERCRITICAL_TEMP,
                       mole_frac=VAPOR_FRACTIONS, moles=1.0, verbose=False)  # [mol]

    with pytest.raises(MissingPropertyError) as error:
        vapor.getEnthalpy(SUPERCRITICAL_TEMP, basis="mole", total_h=total_h)

    # butane_like is the second supercritical column, after light_gas.
    message = str(error.value)
    assert message.startswith(
        "Species ['butane_like'] are supercritical (temp > t_crit)")
    assert ("vapor heat-capacity coefficients 'cp_vapor' "
            "(cp [J/mol/K] = sum_k c_k * T**k)") in message
    assert message.endswith(
        "add 'cp_vapor' for these species or evaluate at or below t_crit")


@pytest.mark.unit
@pytest.mark.parametrize("empty", [[], None], ids=["empty_lists", "nulls"])
def test_vapor_enthalpy_rejects_cp_vapor_that_no_species_supplies(tmp_path,
                                                                  empty):
    """All-empty cp_vapor lists no longer integrate to 0 J/mol."""
    database = copy.deepcopy(VAPOR_DATABASE)
    for record in database.values():
        record["cp_vapor"] = empty
    vapor = VaporPhase(_write(tmp_path, database), temp=SUPERCRITICAL_TEMP,
                       mole_frac=VAPOR_FRACTIONS, moles=1.0, verbose=False)  # [mol]

    with pytest.raises(MissingPropertyError) as error:
        vapor.getEnthalpy(SUPERCRITICAL_TEMP, basis="mole")

    assert str(error.value).startswith(
        "Species ['light_gas', 'butane_like'] are supercritical")
    assert "'cp_vapor'" in str(error.value)


@pytest.mark.unit
def test_vapor_enthalpy_needs_cp_liq_only_for_present_subcritical_species(
        tmp_path):
    """The liquid branch has the same zero-fraction rule as the vapor one."""
    database = copy.deepcopy(VAPOR_DATABASE)
    del database["water_like"]["cp_liq"]
    vapor = VaporPhase(_write(tmp_path, database), temp=SUBCRITICAL_TEMP,
                       mole_frac=VAPOR_FRACTIONS, moles=1.0, verbose=False)  # [mol]

    with pytest.raises(MissingPropertyError) as error:
        vapor.getEnthalpy(SUBCRITICAL_TEMP, basis="mole")
    assert str(error.value).startswith(
        "Species ['water_like'] are at or below t_crit (or have no t_crit) "
        "at a requested temperature, so their enthalpy needs liquid "
        "heat-capacity coefficients 'cp_liq'")

    water_absent = np.array([0.4, 0.6, 0.0])  # [-]
    butane = VAPOR_DATABASE["butane_like"]
    expected = (
        water_absent[0] * _poly_enthalpy(LIGHT_CP_VAPOR, SUBCRITICAL_TEMP)
        + water_absent[1] * (_poly_enthalpy(butane["cp_liq"],
                                            SUBCRITICAL_TEMP)
                             + _watson(butane, SUBCRITICAL_TEMP)))  # [J/mol]
    total = vapor.getEnthalpy(SUBCRITICAL_TEMP, basis="mole",
                              mole_frac=water_absent)  # [J/mol]
    assert total == pytest.approx(expected, rel=CLOSED_FORM_RTOL)


@pytest.mark.unit
def test_vapor_enthalpy_below_critical_temperature_keeps_its_value(
        vapor_path):
    """Below its t_crit the species uses cp_liq and Watson, as before."""
    vapor = VaporPhase(vapor_path, temp=SUBCRITICAL_TEMP,
                       mole_frac=VAPOR_FRACTIONS, moles=1.0, verbose=False)  # [mol]
    light, butane, water = (VAPOR_DATABASE[name] for name in
                            ("light_gas", "butane_like", "water_like"))
    expected_species = np.array([
        _poly_enthalpy(LIGHT_CP_VAPOR, SUBCRITICAL_TEMP),
        _poly_enthalpy(butane["cp_liq"], SUBCRITICAL_TEMP)
        + _watson(butane, SUBCRITICAL_TEMP),
        _poly_enthalpy(water["cp_liq"], SUBCRITICAL_TEMP)
        + _watson(water, SUBCRITICAL_TEMP),
    ])  # [J/mol]
    assert light["t_crit"] < SUBCRITICAL_TEMP < butane["t_crit"]

    species = vapor.getEnthalpy(SUBCRITICAL_TEMP, basis="mole", total_h=False)
    total = vapor.getEnthalpy(SUBCRITICAL_TEMP, basis="mole")

    np.testing.assert_allclose(species, [expected_species],
                               rtol=CLOSED_FORM_RTOL)
    assert total == pytest.approx(expected_species @ VAPOR_FRACTIONS,
                                  rel=CLOSED_FORM_RTOL)


@pytest.mark.unit
def test_vapor_enthalpy_ignores_missing_cp_vapor_at_zero_fraction(vapor_path):
    """An absent species needs no data; it adds zero, not NaN * 0."""
    vapor = VaporPhase(vapor_path, temp=SUPERCRITICAL_TEMP,
                       mole_frac=VAPOR_FRACTIONS, moles=1.0, verbose=False)  # [mol]
    absent_butane = BUTANE_ABSENT  # [-]
    water = VAPOR_DATABASE["water_like"]
    expected = (
        absent_butane[0] * _poly_enthalpy(LIGHT_CP_VAPOR, SUPERCRITICAL_TEMP)
        + absent_butane[2] * (_poly_enthalpy(water["cp_liq"],
                                             SUPERCRITICAL_TEMP)
                              + _watson(water, SUPERCRITICAL_TEMP)))  # [J/mol]

    total = vapor.getEnthalpy(SUPERCRITICAL_TEMP, basis="mole",
                              mole_frac=absent_butane)

    assert total == pytest.approx(expected, rel=CLOSED_FORM_RTOL)


@pytest.mark.unit
def test_vapor_enthalpy_profile_needs_cp_vapor_only_where_present(vapor_path):
    """Each (temperature, species) entry is needed only at nonzero fraction."""
    vapor = VaporPhase(vapor_path, temp=SUBCRITICAL_TEMP,
                       mole_frac=VAPOR_FRACTIONS, moles=1.0, verbose=False)  # [mol]
    temps = np.array([SUBCRITICAL_TEMP, SUPERCRITICAL_TEMP])  # [K]
    # Row 1 (400 K) contains butane_like below its t_crit; row 2 (450 K),
    # where it is supercritical, does not contain it.
    profile = np.array([VAPOR_FRACTIONS, BUTANE_ABSENT])  # [-]
    butane, water = (VAPOR_DATABASE[name]
                     for name in ("butane_like", "water_like"))
    row_species = np.array([
        [_poly_enthalpy(LIGHT_CP_VAPOR, SUBCRITICAL_TEMP),
         _poly_enthalpy(butane["cp_liq"], SUBCRITICAL_TEMP)
         + _watson(butane, SUBCRITICAL_TEMP),
         _poly_enthalpy(water["cp_liq"], SUBCRITICAL_TEMP)
         + _watson(water, SUBCRITICAL_TEMP)],
        # butane_like is absent here; its term is zero whatever its value.
        [_poly_enthalpy(LIGHT_CP_VAPOR, SUPERCRITICAL_TEMP), 0.0,
         _poly_enthalpy(water["cp_liq"], SUPERCRITICAL_TEMP)
         + _watson(water, SUPERCRITICAL_TEMP)],
    ])  # [J/mol]
    expected = (row_species * profile).sum(axis=1)  # [J/mol]

    rows = vapor.getEnthalpy(temps, basis="mole", mole_frac=profile)

    np.testing.assert_allclose(rows, expected, rtol=CLOSED_FORM_RTOL)
    assert np.isfinite(rows).all()
    with pytest.raises(MissingPropertyError, match=r"\['butane_like'\].*'cp_vapor'"):
        vapor.getEnthalpy(temps, basis="mole", mole_frac=profile[::-1])


# ----------------------------------------------- heat capacity and enthalpy
@pytest.mark.unit
def test_liquid_cp_and_enthalpy_ignore_missing_cp_liq_at_zero_fraction(
        liquid_path):
    """Mixture Cp and enthalpy skip an absent species without data."""
    mole_frac = np.array(INERT_ABSENT)  # [-]
    liquid = _liquid(liquid_path, mole_frac)
    cp_solvent = SOLVENT_CP[0] + SOLVENT_CP[1] * LIQUID_TEMP  # [J/mol/K]
    cp_solute = SOLUTE_CP[0]  # [J/mol/K]
    expected_cp = mole_frac[0] * cp_solvent + mole_frac[1] * cp_solute  # [J/mol/K]
    expected_h = (mole_frac[0] * _poly_enthalpy(SOLVENT_CP, LIQUID_TEMP)
                  + mole_frac[1] * _poly_enthalpy(SOLUTE_CP, LIQUID_TEMP))  # [J/mol]

    cp_mix = liquid.getCp(basis="mole")  # [J/mol/K]
    h_mix = liquid.getEnthalpy(basis="mole")  # [J/mol]

    assert cp_mix == pytest.approx(expected_cp, rel=CLOSED_FORM_RTOL)
    assert h_mix == pytest.approx(expected_h, rel=CLOSED_FORM_RTOL)


@pytest.mark.unit
@pytest.mark.parametrize("call", [
    lambda liquid: liquid.getCp(basis="mole"),
    lambda liquid: liquid.getEnthalpy(basis="mole"),
    lambda liquid: liquid.getEnthalpy(basis="mole", total_h=False),
    lambda liquid: liquid.getCpPure(LIQUID_TEMP),
], ids=["cp_mix", "enthalpy_mix", "enthalpy_species", "cp_pure"])
def test_liquid_cp_consumers_reject_missing_cp_liq_when_needed(liquid_path,
                                                               call):
    """A present species or a per-species result needs the coefficients."""
    liquid = _liquid(liquid_path, INERT_PRESENT)

    with pytest.raises(MissingPropertyError) as error:
        call(liquid)

    assert str(error.value).startswith("Species ['inert'] ")
    assert ("liquid heat-capacity coefficients 'cp_liq' "
            "(cp [J/mol/K] = sum_k c_k * T**k)") in str(error.value)
    assert liquid_path in str(error.value)


def _mix_cp(manager, profile):
    """Mixture molar heat capacity of a composition profile at LIQUID_TEMP.

    Parameters
    ----------
    manager : ThermoPhysicalManager
        Property object.
    profile : ndarray
        Mole fractions [-], shape (2, 3).

    Returns
    -------
    ndarray
        Heat capacity [J/mol/K], shape (2,).
    """
    return manager.getCpMix(LIQUID_TEMP, mole_frac=profile, basis="mole")


def _mix_enthalpy(manager, profile):
    """Mixture molar enthalpy of a composition profile at LIQUID_TEMP.

    Parameters
    ----------
    manager : ThermoPhysicalManager
        Property object.
    profile : ndarray
        Mole fractions [-], shape (2, 3).

    Returns
    -------
    ndarray
        Enthalpy [J/mol], shape (2,).
    """
    return manager.getEnthalpy(LIQUID_TEMP, mole_frac=profile, basis="mole")


def _mix_viscosity(manager, profile):
    """Mixture liquid viscosity of a composition profile at LIQUID_TEMP.

    Parameters
    ----------
    manager : ThermoPhysicalManager
        Property object.
    profile : ndarray
        Mole fractions [-], shape (2, 3).

    Returns
    -------
    ndarray
        Viscosity [Pa*s], shape (2,).
    """
    return manager.getViscosityMix(LIQUID_TEMP, mole_frac=profile)


def _closed_form_rows(kind, profile):
    """Closed-form mixture values of the solvent/solute rows of a profile.

    Parameters
    ----------
    kind : {'cp', 'enthalpy', 'viscosity'}
        Mixture property.
    profile : ndarray
        Mole fractions [-], shape (2, 3); the inert column is zero.

    Returns
    -------
    ndarray
        [J/mol/K], [J/mol] or [Pa*s], shape (2,).
    """
    if kind == "cp":
        pure = np.array([SOLVENT_CP[0] + SOLVENT_CP[1] * LIQUID_TEMP,
                         SOLUTE_CP[0]])  # [J/mol/K]
        return profile[:, :2] @ pure
    if kind == "enthalpy":
        pure = np.array([_poly_enthalpy(SOLVENT_CP, LIQUID_TEMP),
                         _poly_enthalpy(SOLUTE_CP, LIQUID_TEMP)])  # [J/mol]
        return profile[:, :2] @ pure
    pure = np.array([_liquid_viscosity(SOLVENT_VISC, LIQUID_TEMP),
                     _liquid_viscosity(SOLUTE_VISC, LIQUID_TEMP)])  # [Pa*s]
    return np.exp(profile[:, :2] @ np.log(pure))


@pytest.mark.unit
@pytest.mark.parametrize("kind, method, prop", [
    ("cp", _mix_cp, "cp_liq"),
    ("enthalpy", _mix_enthalpy, "cp_liq"),
    ("viscosity", _mix_viscosity, "visc_liq"),
], ids=["getCpMix", "getEnthalpy", "getViscosityMix"])
def test_profile_needs_data_if_species_is_present_in_any_row(liquid_path,
                                                             kind, method,
                                                             prop):
    """Two-row profiles: nonzero in some row, not in every row, is needed."""
    manager = ThermoPhysicalManager(liquid_path)
    present_in_one_row = np.array([INERT_ABSENT, INERT_PRESENT])  # [-]
    absent_in_both = np.array([INERT_ABSENT, [0.5, 0.5, 0.0]])  # [-]

    with pytest.raises(MissingPropertyError) as error:
        method(manager, present_in_one_row)
    assert str(error.value).startswith("Species ['inert'] carry a nonzero")
    assert f"'{prop}'" in str(error.value)

    np.testing.assert_allclose(method(manager, absent_in_both),
                               _closed_form_rows(kind, absent_in_both),
                               rtol=CLOSED_FORM_RTOL)


@pytest.mark.unit
@pytest.mark.parametrize("call", [
    lambda manager: manager.getCpPure(LIQUID_TEMP, phase="gas"),
    lambda manager: manager.getCpMix(LIQUID_TEMP, phase="gas",
                                     mole_frac=np.array(INERT_ABSENT)),
    lambda manager: manager.getEnthalpy(LIQUID_TEMP, phase="gas",
                                        mole_frac=np.array(INERT_ABSENT)),
    lambda manager: manager.getEnthalpy(LIQUID_TEMP, phase="gas",
                                        total_h=False),
], ids=["getCpPure", "getCpMix", "getEnthalpy_total", "getEnthalpy_species"])
def test_unknown_heat_capacity_phase_is_rejected(liquid_path, call):
    """An unknown phase raises ValueError, not UnboundLocalError/KeyError."""
    manager = ThermoPhysicalManager(liquid_path)

    with pytest.raises(ValueError) as error:
        call(manager)

    assert str(error.value) == (
        "phase must be one of ('liquid', 'solid', 'vapor'), got 'gas'")


@pytest.mark.unit
def test_weighted_per_species_values_need_data_only_at_nonzero_weight(
        liquid_path):
    """Reactor-style weights: zero-weight species without data give 0."""
    liquid = _liquid(liquid_path, INERT_ABSENT)
    conc = np.array([[2.0, 0.5, 0.0], [1.5, 1.0, 0.0]])  # [mol/L]
    cp_solvent = SOLVENT_CP[0] + SOLVENT_CP[1] * LIQUID_TEMP  # [J/mol/K]
    h_pure = [_poly_enthalpy(SOLVENT_CP, LIQUID_TEMP),
              _poly_enthalpy(SOLUTE_CP, LIQUID_TEMP), 0.0]  # [J/mol]

    _, cp_mole = liquid.getCpPure(LIQUID_TEMP, weights=conc)  # [J/mol/K]
    h_species = liquid.getEnthalpy(LIQUID_TEMP, total_h=False, basis="mole",
                                   weights=conc)  # [J/mol]

    np.testing.assert_allclose(cp_mole, [cp_solvent, SOLUTE_CP[0], 0.0],  # [J/mol/K]
                               rtol=CLOSED_FORM_RTOL)
    np.testing.assert_allclose(h_species, [h_pure], rtol=CLOSED_FORM_RTOL)
    np.testing.assert_allclose(conc @ cp_mole,
                               conc[:, :2] @ [cp_solvent, SOLUTE_CP[0]],
                               rtol=CLOSED_FORM_RTOL)
    present = conc + np.array([0.0, 0.0, 0.25])  # [mol/L], inert present
    with pytest.raises(MissingPropertyError, match=r"^Species \['inert'\] carry "
                       r"a nonzero weight.*'cp_liq'"):
        liquid.getCpPure(LIQUID_TEMP, weights=present)
    with pytest.raises(MissingPropertyError, match=r"^Species \['inert'\] carry "
                       r"a nonzero weight.*'cp_liq'"):
        liquid.getEnthalpy(LIQUID_TEMP, total_h=False, weights=present)


@pytest.mark.unit
def test_solid_cp_mix_needs_cp_solid_only_for_present_species(tmp_path):
    """The solid branch of the shared mixing method follows the same rule."""
    database = copy.deepcopy(LIQUID_DATABASE)
    solid_cp = (1.5, 0.25)  # [J/mol/K], [J/mol/K**2], synthetic
    database["solute"]["cp_solid"] = list(solid_cp)
    manager = ThermoPhysicalManager(_write(tmp_path, database))
    temp = 300.0  # [K]
    pure_solute = np.array([0.0, 1.0, 0.0])  # [-]

    cp_mass = manager.getCpMix(temp, mass_frac=pure_solute, phase="solid")  # [J/kg/K]

    expected = ((solid_cp[0] + solid_cp[1] * temp)
                * 1000 / database["solute"]["mw"])  # [J/kg/K]
    assert cp_mass == pytest.approx(expected, rel=CLOSED_FORM_RTOL)
    with pytest.raises(MissingPropertyError, match=r"\['solvent'\].*'cp_solid'"):
        manager.getCpMix(temp, mass_frac=np.array([0.5, 0.5, 0.0]),  # [-]
                         phase="solid")


@pytest.mark.unit
def test_heat_of_reaction_needs_cp_liq_only_for_reacting_species(liquid_path):
    """Species outside every reaction contribute zero, not NaN."""
    liquid = _liquid(liquid_path, INERT_PRESENT)
    mask = np.ones(3, dtype=bool)  # every species participates in the list
    heat_ref = -5.0e4  # [J/mol of reaction], synthetic exothermic value
    tref = 300.0  # [K]
    solvent_to_solute = np.array([[-1.0, 1.0, 0.0]])  # [-]

    heat = liquid.getHeatOfRxn(solvent_to_solute, LIQUID_TEMP, mask,
                               heat_ref, tref)  # [J/mol]

    expected = (heat_ref + _poly_enthalpy(SOLUTE_CP, LIQUID_TEMP, tref)
                - _poly_enthalpy(SOLVENT_CP, LIQUID_TEMP, tref))  # [J/mol]
    np.testing.assert_allclose(heat, [expected], rtol=CLOSED_FORM_RTOL)
    with pytest.raises(MissingPropertyError, match=r"\['inert'\].*'cp_liq'"):
        liquid.getHeatOfRxn(np.array([[-1.0, 0.0, 1.0]]), LIQUID_TEMP, mask,  # [-]
                            heat_ref, tref)

    # Partial mask: the stoichiometric columns are solvent and inert.
    partial = np.array([True, False, True])
    heat = liquid.getHeatOfRxn(np.array([[-1.0, 0.0]]), LIQUID_TEMP, partial,  # [-]
                               heat_ref, tref)
    expected = heat_ref - _poly_enthalpy(SOLVENT_CP, LIQUID_TEMP, tref)  # [J/mol]
    np.testing.assert_allclose(heat, [expected], rtol=CLOSED_FORM_RTOL)
    with pytest.raises(MissingPropertyError) as error:
        liquid.getHeatOfRxn(np.array([[-1.0, 1.0]]), LIQUID_TEMP, partial,  # [-]
                            heat_ref, tref)
    assert str(error.value).startswith(
        "Species ['inert'] take part in a reaction")


# -------------------------------------------------------------- viscosity
@pytest.mark.unit
def test_liquid_viscosity_needs_visc_liq_only_for_present_species(
        liquid_path):
    """Logarithmic mixing skips an absent species without coefficients."""
    mole_frac = np.array(INERT_ABSENT)  # [-]
    liquid = _liquid(liquid_path, mole_frac)
    expected = np.exp(
        mole_frac[0] * np.log(_liquid_viscosity(SOLVENT_VISC, LIQUID_TEMP))
        + mole_frac[1] * np.log(_liquid_viscosity(SOLUTE_VISC, LIQUID_TEMP))
    )  # [Pa*s]

    assert liquid.getViscosity() == pytest.approx(expected,
                                                  rel=CLOSED_FORM_RTOL)
    with pytest.raises(MissingPropertyError, match=r"\['inert'\].*'visc_liq'"):
        liquid.getViscosity(mole_frac=np.array(INERT_PRESENT))
    with pytest.raises(MissingPropertyError, match=r"\['inert'\].*'visc_liq'"):
        liquid.getViscosityPure()


INERT_VISC = (-1.0, 100.0, 0.0, 0.0)  # [-], [K], [1/K], [1/K**2], synthetic


def _log_mixing(pure, frac):
    """Closed-form logarithmic liquid-viscosity mixing.

    Parameters
    ----------
    pure : numpy.ndarray
        Pure viscosities [Pa*s], shape (num_temps, num_species).
    frac : numpy.ndarray
        Mole fractions [-], shape (num_species,) or (num_temps,
        num_species).

    Returns
    -------
    numpy.ndarray
        ``exp(sum_i x_i * ln(mu_i))`` [Pa*s], shape (num_temps,).
    """
    return np.exp((frac * np.log(pure)).sum(axis=-1))


@pytest.mark.unit
def test_liquid_viscosity_accepts_temperature_arrays(tmp_path):
    """Array temperatures give one row per temperature, like getEnthalpy.

    Pure and mixture viscosities equal the closed-form correlation and
    logarithmic mixing rule at each temperature, for one composition and
    for a profile paired row by row.
    """
    database = copy.deepcopy(LIQUID_DATABASE)
    database["inert"]["visc_liq"] = list(INERT_VISC)
    liquid = _liquid(_write(tmp_path, database), INERT_PRESENT)
    temps = np.array([300.0, LIQUID_TEMP, 380.0])  # [K], asymmetric spacing
    composition = np.array(INERT_PRESENT)  # [-]
    profile = np.array([INERT_PRESENT, [0.5, 0.25, 0.25],
                        [0.75, 0.25, 0.0]])  # [-], paired with temps
    closed_form = np.array([[_liquid_viscosity(coefs, temp)
                             for coefs in (SOLVENT_VISC, SOLUTE_VISC,
                                           INERT_VISC)]
                            for temp in temps])  # [Pa*s]

    pure = liquid.getViscosityPure("liquid", temps)  # [Pa*s]
    mixed = liquid.getViscosity(temps, mole_frac=composition)  # [Pa*s]
    paired = liquid.getViscosity(temps, mole_frac=profile)  # [Pa*s]

    np.testing.assert_allclose(pure, closed_form, rtol=CLOSED_FORM_RTOL)
    np.testing.assert_allclose(mixed, _log_mixing(closed_form, composition),
                               rtol=CLOSED_FORM_RTOL)
    np.testing.assert_allclose(paired, _log_mixing(closed_form, profile),
                               rtol=CLOSED_FORM_RTOL)


@pytest.mark.unit
def test_liquid_viscosity_arrays_need_visc_liq_only_where_present(
        liquid_path):
    """The inert lacks visc_liq: absent rows evaluate, a present row raises."""
    liquid = _liquid(liquid_path, INERT_ABSENT)
    temps = np.array([300.0, LIQUID_TEMP])  # [K]
    absent = np.array([INERT_ABSENT, [0.5, 0.5, 0.0]])  # [-]
    closed_form = np.array([[_liquid_viscosity(coefs, temp)
                             for coefs in (SOLVENT_VISC, SOLUTE_VISC)]
                            for temp in temps])  # [Pa*s]

    paired = liquid.getViscosity(temps, mole_frac=absent)  # [Pa*s]

    assert np.isfinite(paired).all()
    np.testing.assert_allclose(paired,
                               _log_mixing(closed_form, absent[:, :2]),
                               rtol=CLOSED_FORM_RTOL)
    with pytest.raises(MissingPropertyError, match=r"^Species \['inert'\] "
                       r"carry a nonzero fraction.*'visc_liq'"):
        liquid.getViscosity(temps, mole_frac=np.array([INERT_ABSENT,
                                                       INERT_PRESENT]))


@pytest.mark.unit
def test_vle_ratio_rejects_unpaired_rows(liquid_path):
    """Temperatures and composition rows must pair one to one."""
    liquid = _liquid(liquid_path, INERT_ABSENT)
    temps = np.array([300.0, LIQUID_TEMP])  # [K], two rows
    three_rows = np.array([INERT_ABSENT] * 3)  # [-], three rows

    with pytest.raises(ValueError, match=r"temp has 2 entries but x_liq "
                       r"has 3 rows"):
        liquid.getKeqVLE(temp=temps, pres=PRESSURE, x_liq=three_rows)


@pytest.mark.unit
def test_viscosity_without_any_visc_liq_names_species_and_property(tmp_path):
    """A wholly absent correlation gives the named error, not a shape error."""
    database = copy.deepcopy(LIQUID_DATABASE)
    for record in database.values():
        record.pop("visc_liq", None)
    liquid = _liquid(_write(tmp_path, database), INERT_ABSENT)

    with pytest.raises(MissingPropertyError) as error:
        liquid.getViscosity()

    assert str(error.value).startswith(
        "Species ['solvent', 'solute'] carry a nonzero fraction, so the "
        "mixture viscosity needs liquid-viscosity coefficients 'visc_liq' "
        "(log10(mu / [mPa*s]) = A + B/T + C*T + D*T**2)")


@pytest.mark.unit
def test_diffusivity_rejects_species_without_data(liquid_path):
    """Washing needs a diffusivity for every species."""
    liquid = _liquid(liquid_path, INERT_ABSENT)

    with pytest.raises(MissingPropertyError, match=r"\['inert'\].*'diffusivity'"):
        liquid.getDiffusivityPure(wrt=0)


# ------------------------------------------------------- Antoine and VLE
@pytest.mark.unit
def test_antoine_needs_p_vap_for_every_selected_species(liquid_path):
    """Pure saturation values are returned per species, in idx order."""
    liquid = _liquid(liquid_path, INERT_ABSENT)

    with pytest.raises(MissingPropertyError) as error:
        liquid.AntoineEquation(temp=LIQUID_TEMP)
    assert str(error.value).startswith("Species ['inert'] ")
    assert "Antoine vapor-pressure coefficients 'p_vap'" in str(error.value)

    expected = [_antoine_pressure(SOLUTE_ANTOINE, LIQUID_TEMP),
                _antoine_pressure(SOLVENT_ANTOINE, LIQUID_TEMP)]  # [Pa]
    for idx in ([1, 0], (1, 0), np.array([1, 0])):
        selected = liquid.AntoineEquation(temp=LIQUID_TEMP, idx=idx)  # [Pa]
        np.testing.assert_allclose(selected, expected, rtol=CLOSED_FORM_RTOL)

    with pytest.raises(MissingPropertyError) as error:
        liquid.AntoineEquation(temp=LIQUID_TEMP, idx=(2, 0))
    assert str(error.value).startswith("Species ['inert'] ")
    with pytest.raises(TypeError, match="integer species indices"):
        liquid.AntoineEquation(temp=LIQUID_TEMP,
                               idx=[True, False, True])


@pytest.mark.unit
def test_vle_ratio_uses_henry_for_supercritical_species_without_p_vap(
        liquid_path):
    """Above t_crit the Henry constant replaces the absent Antoine data."""
    pressure = PRESSURE  # [Pa]
    liquid = _liquid(liquid_path, INERT_PRESENT)

    k_values = liquid.getKeqVLE(temp=LIQUID_TEMP, pres=pressure)  # [-]

    expected = np.array([
        _antoine_pressure(SOLVENT_ANTOINE, LIQUID_TEMP),
        _antoine_pressure(SOLUTE_ANTOINE, LIQUID_TEMP),
        INERT_HENRY]) / pressure  # [-]
    np.testing.assert_allclose(k_values, expected, rtol=CLOSED_FORM_RTOL)
    # Below its t_crit the present inert would use Antoine data it lacks.
    temps = np.array([LIQUID_TEMP, BELOW_INERT_T_CRIT])  # [K]
    with pytest.raises(MissingPropertyError, match=r"\['inert'\] carry a nonzero "
                       r"fraction and are at or below t_crit.*'p_vap'"):
        liquid.getKeqVLE(temp=temps, pres=pressure)

    # Absent inert: no data needed; its ratio is the finite zero-coefficient
    # placeholder, 10**(0 - 0/T) = 1 Pa over the pressure, so K * x = 0.
    absent = np.array(INERT_ABSENT)  # [-]
    k_profile = liquid.getKeqVLE(temp=temps, pres=pressure,
                                 x_liq=absent)  # [-]
    np.testing.assert_allclose(k_profile[:, 2], [INERT_HENRY / pressure,
                                                 1.0 / pressure],  # [-]
                               rtol=CLOSED_FORM_RTOL)
    np.testing.assert_array_equal((k_profile * absent)[:, 2], [0.0, 0.0])  # [-]


@pytest.mark.unit
def test_vle_ratio_profile_pairs_temperature_and_composition_rows(
        liquid_path):
    """Each row needs Antoine data only where the species is present there.

    Row 1 is below the inert's critical temperature with no inert; row 2 is
    above it with the inert present, so Henry's law applies and no row needs
    the inert's Antoine data. The profile equals the stacked single-row
    evaluations. Swapping the rows puts the inert in the subcritical row.
    """
    liquid = _liquid(liquid_path, INERT_PRESENT)
    temps = np.array([BELOW_INERT_T_CRIT, LIQUID_TEMP])  # [K]
    profile = np.array([INERT_ABSENT, INERT_PRESENT])  # [-]

    paired = liquid.getKeqVLE(temp=temps, pres=PRESSURE,
                              x_liq=profile)  # [-]

    stacked = np.vstack([
        liquid.getKeqVLE(temp=temp, pres=PRESSURE, x_liq=row)
        for temp, row in zip(temps, profile)])  # [-]
    np.testing.assert_array_equal(paired, stacked)
    with pytest.raises(MissingPropertyError, match=r"^Species \['inert'\] "
                       r"carry a nonzero fraction.*'p_vap'"):
        liquid.getKeqVLE(temp=temps, pres=PRESSURE, x_liq=profile[::-1])


@pytest.mark.unit
def test_bubble_point_with_henry_species_without_p_vap(liquid_path):
    """The seed evaluates the inert's missing row with zero coefficients.

    That is the seed of releases before #414, so the Newton path is
    unchanged; the Henry K-value enters the converged root.
    """
    pressure = PRESSURE  # [Pa]
    # Trace dissolved inert: its Henry partial pressure, 7e9 Pa * 1e-6 =
    # 7e3 Pa, stays below the total pressure. The solute is absent.
    inert_frac = 1.0e-6  # [-]
    mole_frac = np.array([1.0 - inert_frac, 0.0, inert_frac])  # [-]
    liquid = _liquid(liquid_path, mole_frac)

    temp_bubble = liquid.getBubblePoint(pres=pressure)  # [K]

    # x_s * psat_s(T) + x_i * H = P, solved for T with the Antoine form.
    psat_solvent = (pressure - inert_frac * INERT_HENRY) / mole_frac[0]  # [Pa]
    a_ct, b_ct, c_ct = SOLVENT_ANTOINE
    expected = b_ct / (a_ct - np.log10(psat_solvent)) - c_ct  # [K]
    assert temp_bubble == pytest.approx(expected, abs=BUBBLE_POINT_ATOL)
    assert temp_bubble > LIQUID_DATABASE["inert"]["t_crit"]


@pytest.mark.unit
def test_bubble_pressure_and_dew_point_with_henry_species_without_p_vap(
        liquid_path):
    """The other Antoine-seeded solvers keep their pre-#414 seeds too.

    Their seeds evaluate the inert's missing row with zero coefficients;
    the Henry K-value enters the converged root.
    """
    inert_frac = 1.0e-6  # [-], trace dissolved inert, as above
    liquid_frac = np.array([1.0 - inert_frac, 0.0, inert_frac])  # [-]
    liquid = _liquid(liquid_path, liquid_frac)

    pres_bubble = liquid.getBubblePressure(temp=LIQUID_TEMP)  # [Pa]

    # P = x_s * psat_s(T) + x_i * H for the ideal liquid.
    expected_pres = (liquid_frac[0]
                     * _antoine_pressure(SOLVENT_ANTOINE, LIQUID_TEMP)
                     + inert_frac * INERT_HENRY)  # [Pa]
    assert pres_bubble == pytest.approx(expected_pres, rel=0,
                                        abs=BUBBLE_PRESSURE_ATOL)

    pressure = PRESSURE  # [Pa]
    vapor_frac = np.array([0.5, 0.0, 0.5])  # [-], equimolar solvent/inert
    vapor = VaporPhase(liquid_path, temp=LIQUID_TEMP, pres=pressure,
                       moles=1.0, mole_frac=vapor_frac, verbose=False)  # [mol]

    temp_dew = vapor.getDewPoint()  # [K]

    # y_s * P / psat_s(T) + y_i * P / H = 1, solved for psat_s, then T.
    psat_solvent = (vapor_frac[0] * pressure
                    / (1 - vapor_frac[2] * pressure / INERT_HENRY))  # [Pa]
    a_ct, b_ct, c_ct = SOLVENT_ANTOINE
    expected_temp = b_ct / (a_ct - np.log10(psat_solvent)) - c_ct  # [K]
    assert temp_dew == pytest.approx(expected_temp, abs=BUBBLE_POINT_ATOL)
    assert temp_dew > LIQUID_DATABASE["inert"]["t_crit"]


@pytest.mark.unit
def test_dilute_bubble_pressure_keeps_the_henry_raoult_root(tmp_path):
    """A low-volatility solvent with a trace Henry inert lacking p_vap.

    Raoult plus Henry, ``P = x_s * psat_s + x_i * H``, has one root. The
    Antoine B is chosen so that psat_s(350 K) = 1e-8 Pa exactly
    (A = 10, C = 0): ``B = T * (A - log10(psat))``. The bubble pressure is
    dominated by the inert's Henry term, 1e-6 * 7e9 Pa = 7000 Pa.
    """
    temp = 350.0  # [K]
    psat_solvent = 1.0e-8  # [Pa], low-volatility solvent at temp
    antoine_a = 10.0  # [-]
    antoine_b = temp * (antoine_a - np.log10(psat_solvent))  # [K], with C = 0
    database = {
        "low_volatility_solvent": {
            "mw": 100.0, "rho_liq": 1000.0, "cp_liq": [100.0],
            "t_crit": 700.0, "p_vap": [antoine_a, antoine_b, 0.0]},
        "dilute_inert": {
            "mw": 28.0, "rho_liq": 806.0, "cp_liq": [30.0],
            "t_crit": 126.14, "henry_constant": INERT_HENRY},
    }  # mw [g/mol], rho_liq [kg/m**3], cp_liq [J/mol/K], t_crit [K]
    inert_frac = 1.0e-6  # [-]
    mole_frac = np.array([1.0 - inert_frac, inert_frac])  # [-]
    liquid = LiquidPhase(_write(tmp_path, database), temp=temp,
                         mole_frac=mole_frac, moles=1.0)  # [mol]

    pres_bubble = liquid.getBubblePressure()  # [Pa]

    expected = (mole_frac[0] * psat_solvent
                + inert_frac * INERT_HENRY)  # [Pa]
    assert pres_bubble == pytest.approx(expected, rel=0,
                                        abs=BUBBLE_PRESSURE_ATOL)


@pytest.mark.unit
def test_dew_point_with_henry_carrier_subcritical_at_the_seed(tmp_path):
    """The seed lies below the carrier's t_crit; the root lies above it.

    The equimolar seed, ``0.5 * Tsat_solvent(P)``, is about 175 K, below
    the carrier's 200 K critical temperature, where the carrier would need
    Antoine data. Newton iterates are not results: only the converged root,
    where the carrier is supercritical and uses Henry's law, is checked.
    """
    carrier_t_crit = 200.0  # [K], between the seed and the dew point
    database = {
        "solvent": {"mw": 46.0, "t_crit": 514.0,
                    "p_vap": list(SOLVENT_ANTOINE)},
        "carrier": {"mw": 28.0, "t_crit": carrier_t_crit,
                    "henry_constant": INERT_HENRY},
    }  # mw [g/mol], t_crit [K], Antoine [-], [K], [K], Henry [Pa]
    vapor_frac = np.array([0.5, 0.5])  # [-]
    vapor = VaporPhase(_write(tmp_path, database), temp=LIQUID_TEMP,
                       pres=PRESSURE, moles=1.0, mole_frac=vapor_frac,  # [mol]
                       verbose=False)
    a_ct, b_ct, c_ct = SOLVENT_ANTOINE
    seed = vapor_frac[0] * (b_ct / (a_ct - np.log10(PRESSURE)) - c_ct)  # [K]
    assert seed < carrier_t_crit

    temp_dew = vapor.getDewPoint()  # [K]

    # y_s * P / psat_s(T) + y_c * P / H = 1, solved for psat_s, then T.
    psat_solvent = (vapor_frac[0] * PRESSURE
                    / (1 - vapor_frac[1] * PRESSURE / INERT_HENRY))  # [Pa]
    expected = b_ct / (a_ct - np.log10(psat_solvent)) - c_ct  # [K]
    assert temp_dew == pytest.approx(expected, abs=BUBBLE_POINT_ATOL)
    assert temp_dew > carrier_t_crit


@pytest.mark.unit
def test_bubble_point_needs_p_vap_only_for_present_species(tmp_path):
    """A nonvolatile solute without Antoine data matters only when present.

    At zero fraction the root equals the root with arbitrary Antoine data
    for the solute, since its term is zero; at a nonzero fraction the
    root check raises.
    """
    database = copy.deepcopy(LIQUID_DATABASE)
    del database["solute"]["p_vap"]
    with_data = copy.deepcopy(database)
    with_data["solute"]["p_vap"] = [5.0, 4000.0, 0.0]  # arbitrary [-], [K], [K]
    solute_absent = np.array([0.6, 0.0, 0.4e-6])  # [-], before normalizing
    solute_absent = solute_absent / solute_absent.sum()  # [-]
    roots = []
    for name, data in (("missing.json", database), ("data.json", with_data)):
        liquid = _liquid(_write(tmp_path, data, name), solute_absent)
        roots.append(liquid.getBubblePoint(pres=PRESSURE))  # [K]

    assert roots[0] == roots[1]
    # Ten percent solute: Newton converges with the zero-coefficient
    # iterates, as it did before #414; the converged root is then rejected.
    solute_present = np.array([0.9, 0.1 - 1e-6, 1e-6])  # [-]
    liquid = _liquid(_write(tmp_path, database, "present.json"),
                     solute_present)
    with pytest.raises(MissingPropertyError, match=r"^Species \['solute'\] carry "
                       r"a nonzero fraction.*'p_vap'"):
        liquid.getBubblePoint(pres=PRESSURE)


@pytest.mark.unit
def test_bubble_pressure_and_dew_point_check_their_converged_roots(tmp_path):
    """Newton converges on the placeholder; the root check names the gap.

    The solute has no p_vap but is present and subcritical, so the
    converged root needs its Antoine data. Without the check, both solvers
    would return roots computed with the 1 Pa placeholder.
    """
    database = copy.deepcopy(LIQUID_DATABASE)
    del database["solute"]["p_vap"]
    path = _write(tmp_path, database)
    message = (r"^Species \['solute'\] carry a nonzero fraction and are "
               r"at or below t_crit.*'p_vap'")

    equimolar = np.array([0.5, 0.5, 0.0])  # [-], solvent and solute
    liquid = _liquid(path, equimolar)
    with pytest.raises(MissingPropertyError, match=message):
        liquid.getBubblePressure(temp=LIQUID_TEMP)

    trace_solute = 1.0e-6  # [-], small enough for Newton to converge
    vapor_frac = np.array([1.0 - trace_solute, trace_solute, 0.0])  # [-]
    vapor = VaporPhase(path, temp=LIQUID_TEMP, pres=PRESSURE, moles=1.0,  # [mol]
                       mole_frac=vapor_frac, verbose=False)
    with pytest.raises(MissingPropertyError, match=message):
        vapor.getDewPoint()


# Solid fixture: "A" has a constant cp_solid, "B" none. Units: mw [g/mol],
# rho_solid [kg/m**3], cp_solid [J/mol/K].
SOLID_DATABASE = {
    "A": {"mw": 100.0, "rho_solid": 1200.0, "cp_solid": [10.0]},
    "B": {"mw": 50.0, "rho_solid": 1000.0},
}
SOLID_TEMP = 320.0  # [K]


SOLID_ZERO_SENTINEL = np.finfo(float).eps  # [-], what SolidPhase stores for 0


def _solid(path, mass_frac, stream=False):
    """Build a one-kilogram solid phase, or a 1 kg/s solid stream.

    Parameters
    ----------
    path : str
        Property database path.
    mass_frac : array-like
        Mass fractions [-].
    stream : bool, optional
        Build a ``SolidStream`` instead of a ``SolidPhase``.

    Returns
    -------
    SolidPhase
        The phase or stream at SOLID_TEMP.
    """
    from PharmaPy.Streams import SolidStream

    if stream:
        return SolidStream(path, temp=SOLID_TEMP, mass_flow=1.0,
                           mass_frac=mass_frac)  # [kg/s]
    return SolidPhase(path, temp=SOLID_TEMP, mass=1.0,
                      mass_frac=mass_frac)  # [kg]


@pytest.mark.unit
@pytest.mark.parametrize("stream", [False, True], ids=["phase", "stream"])
def test_solid_zero_sentinel_species_needs_no_cp_solid(tmp_path, stream):
    """A zero solid fraction, stored as eps, stays absent.

    The constructor replaces the zero fraction of "B" by machine epsilon;
    the default Cp and enthalpy still treat "B" as absent, giving the
    closed-form values of pure "A", also for a solid rebuilt from that
    stored composition. Reassigning a real composition, or a genuine trace
    above the sentinel, makes "B" present, so it needs its data.
    """
    path = _write(tmp_path, SOLID_DATABASE)
    original = _solid(path, [1.0, 0.0])
    rebuilt = _solid(path, original.mass_frac, stream=stream)
    cp_a = SOLID_DATABASE["A"]["cp_solid"][0]  # [J/mol/K]
    mw_a = SOLID_DATABASE["A"]["mw"]  # [g/mol]
    h_a = cp_a * (SOLID_TEMP - TEMP_REF)  # [J/mol]

    for solid in (_solid(path, [1.0, 0.0], stream=stream), rebuilt):
        assert solid.getCp() == pytest.approx(
            cp_a * 1000 / mw_a, rel=CLOSED_FORM_RTOL)  # [J/kg/K]
        assert solid.getCp(basis="mole") == pytest.approx(
            cp_a, rel=CLOSED_FORM_RTOL)  # [J/mol/K]
        assert solid.getEnthalpy() == pytest.approx(
            h_a * 1000 / mw_a, rel=CLOSED_FORM_RTOL)  # [J/kg]
        assert solid.getEnthalpy(basis="mole") == pytest.approx(
            h_a, rel=CLOSED_FORM_RTOL)  # [J/mol]

    message = r"^Species \['B'\] carry a nonzero fraction.*'cp_solid'"
    reassigned_frac = np.array([0.6, 0.4])  # [-]
    original.mass_frac = reassigned_frac
    original.mole_frac = original.frac_to_frac(mass_frac=reassigned_frac)
    with pytest.raises(MissingPropertyError, match=message):
        original.getCp()
    with pytest.raises(MissingPropertyError, match=message):
        original.getEnthalpy()

    trace = 1.0e-12  # [-], small but above the sentinel
    with pytest.raises(MissingPropertyError, match=message):
        _solid(path, [1.0 - trace, trace], stream=stream).getCp()


@pytest.mark.unit
def test_solid_sentinel_species_with_data_keeps_its_epsilon_weight(tmp_path):
    """With data, the sentinel weight still enters, as before #414.

    B's cp_solid is synthetic and large, so its eps-weighted term stands far
    above round-off on both bases; the tolerance is a quarter of that term.
    """
    database = copy.deepcopy(SOLID_DATABASE)
    cp_b = 1.0e6  # [J/mol/K], synthetic, makes the eps term resolvable
    database["B"]["cp_solid"] = [cp_b]
    solid = _solid(_write(tmp_path, database), [1.0, 0.0])
    cp_mole = np.array([SOLID_DATABASE["A"]["cp_solid"][0], cp_b])  # [J/mol/K]
    mw = np.array([SOLID_DATABASE["A"]["mw"], database["B"]["mw"]])  # [g/mol]
    mass_frac = np.array([1.0, SOLID_ZERO_SENTINEL])  # [-], as stored
    mole_frac = (mass_frac / mw) / (mass_frac / mw).sum()  # [-]

    for basis, frac, cp_pure in (("mass", mass_frac, cp_mole * 1000 / mw),
                                 ("mole", mole_frac, cp_mole)):
        eps_term = frac[1] * cp_pure[1]  # [J/kg/K] or [J/mol/K]
        tolerance = 0.25 * eps_term  # [J/kg/K] or [J/mol/K]
        observed = solid.getCp(basis=basis)  # [J/kg/K] or [J/mol/K]
        assert observed == pytest.approx(frac @ cp_pure, rel=0,
                                         abs=tolerance)
        assert abs(observed - frac[0] * cp_pure[0]) > tolerance


@pytest.mark.unit
def test_mixer_with_absent_solid_species_without_cp_solid(tmp_path):
    """Mixed solids built from stored compositions keep B absent.

    Two slurries at 300 K and 320 K, the second with three times the
    particle number density, so their solid and liquid masses differ. With
    constant heat capacities the adiabatic outlet temperature is the
    capacity-weighted mean ``sum_i C_i T_i / sum_i C_i``, where
    ``C_i = m_liq,i * cp_liq,B + m_sol,i * cp_sol,A`` [J/K]. The mixer
    builds its outlet solid from the inlets' stored (eps-filled) mass
    fractions, so B, without cp_solid, must stay absent; A's solid heat
    capacity must enter with its full weight.
    """
    from PharmaPy.Containers import Mixer
    from PharmaPy.MixedPhases import Slurry

    database = {
        "A": {"mw": 100.0, "rho_solid": 1200.0, "cp_solid": [100.0],
              "rho_liq": 1000.0, "cp_liq": [150.0]},
        "B": {"mw": 50.0, "rho_solid": 1000.0, "rho_liq": 900.0,
              "cp_liq": [120.0]},
    }  # mw [g/mol], rho [kg/m**3], cp [J/mol/K]; B has no cp_solid
    path = _write(tmp_path, database)
    sizes = np.array([0.0, 100.0, 200.0, 300.0])  # [um]
    number = np.array([0.0, 1.0e9, 1.25e8, 0.0])  # [#/um], synthetic
    density_scales = (1.0, 3.0)  # [-], second inlet carries more solid
    temps = (300.0, 320.0)  # [K]

    inlets = []
    for temp, scale in zip(temps, density_scales):
        liquid = LiquidPhase(path, mass_frac=[0.0, 1.0], temp=temp)
        solid = SolidPhase(path, mass_frac=[1.0, 0.0], kv=1.0, temp=temp)
        slurry = Slurry(vol=1.0e-3, x_distrib=sizes,
                        distrib=scale * number)  # [m**3], [#/um]
        slurry.Phases = (liquid, solid)
        inlets.append(slurry)
    mixer = Mixer()
    mixer.Inlets = inlets

    mixer.solve_unit()

    cp_liquid_b = (database["B"]["cp_liq"][0] * 1000
                   / database["B"]["mw"])  # [J/kg/K]
    cp_solid_a = (database["A"]["cp_solid"][0] * 1000
                  / database["A"]["mw"])  # [J/kg/K]
    capacities = np.array([
        inlet.Liquid_1.mass * cp_liquid_b + inlet.Solid_1.mass * cp_solid_a
        for inlet in inlets])  # [J/K]
    expected = capacities @ np.array(temps) / capacities.sum()  # [K]
    masses = [(inlet.Liquid_1.mass, inlet.Solid_1.mass)
              for inlet in inlets]  # [kg]
    assert masses[0] != masses[1]
    assert abs(expected - np.mean(temps)) > BUBBLE_POINT_ATOL  # [K]
    assert mixer.Outlet.Solid_1.temp == pytest.approx(
        expected, abs=BUBBLE_POINT_ATOL)  # [K], Newton root


# ------------------------------------------------------- phase construction
@pytest.mark.unit
def test_phases_build_without_heat_capacity_or_antoine_data(tmp_path):
    """Constructors no longer require cp_liq, p_vap or cp_solid.

    A missing property is reported by the method that needs it.
    """
    density = 997.0  # [kg/m**3], same for both synthetic species
    database = {
        "water_like": {"mw": 18.0, "rho_liq": density, "rho_solid": density},
        "heavy": {"mw": 180.0, "rho_liq": density, "rho_solid": density},
    }  # mw [g/mol]
    path = _write(tmp_path, database)
    mass_frac = np.array([0.25, 0.75])  # [-]
    liquid = LiquidPhase(path, temp=LIQUID_TEMP, mass=1.0,  # [kg]
                         mass_frac=mass_frac)  # [K], [kg]
    solid = SolidPhase(path, temp=LIQUID_TEMP, mass=1.0,  # [kg]
                       mass_frac=mass_frac)  # [K], [kg]

    assert liquid.getDensity() == pytest.approx(density, rel=CLOSED_FORM_RTOL)
    assert solid.getDensity() == pytest.approx(density, rel=CLOSED_FORM_RTOL)
    with pytest.raises(MissingPropertyError, match=r"^Species \['water_like', "
                       r"'heavy'\] carry a nonzero fraction.*'cp_liq'"):
        liquid.getEnthalpy()
    with pytest.raises(MissingPropertyError, match=r"^Species \['water_like', "
                       r"'heavy'\] carry a nonzero fraction.*'cp_solid'"):
        solid.getEnthalpy()


# --------------------------------------------------------------- reactors
# PFR case: the shipped PFR database (A, B, C, solvent "solv") plus, in the
# solver tests, an "imp" species without cp_liq; second-order A + B --> C.
PFR_IMPURITY = {"mw": 120.0, "rho_liq": 1000.0}  # [g/mol], [kg/m**3]
PFR_BASE_CONC = [0.15, 0.10, 0.02, 10.0]  # [mol/L], A, B, C, solvent
PFR_TEMP = 320.0  # [K]
PFR_DIAMETER = 0.0254  # [m]
PFR_NODES = 3  # [-]
PFR_HOLDUP = 0.002  # [m**3]
PFR_RESIDENCE_TIME = 1800.0  # [s]
PFR_RATE_CONSTANT = 1e-3  # [L/mol/s]
PFR_HEAT_OF_REACTION = -1e4  # [J/mol]
PFR_RUNTIME = 60.0  # [s]
PFR_IMPURITY_CONC = 0.5  # [mol/L], when "imp" is charged or fed


def _pfr_steady_derivative(path, solvent_conc=0.0):
    """Steady PFR temperature derivative for a reacting ternary.

    Parameters
    ----------
    path : str
        Property database: tests/integration/data/pfr_test_pure_comp.json,
        possibly edited.
    solvent_conc : float, optional
        Concentration of the non-reacting library solvent [mol/L].

    Returns
    -------
    float
        ``dT/dV`` [K/m**3] at the inlet, at PFR_TEMP.
    """
    from PharmaPy.Kinetics import RxnKinetics
    from PharmaPy.Reactors import PlugFlowReactor
    from PharmaPy.Streams import LiquidStream

    # A, B, C react; the library solvent does not react.
    conc = np.array(PFR_BASE_CONC[:3] + [solvent_conc])  # [mol/L]
    unit = PlugFlowReactor(PFR_DIAMETER, PFR_NODES, adiabatic=True)
    unit.Phases = LiquidPhase(path, temp=PFR_TEMP, mole_conc=conc,
                              vol=PFR_HOLDUP)
    unit.Kinetics = RxnKinetics(
        path, k_params=[PFR_RATE_CONSTANT], ea_params=[0.0],
        rxn_list=["A + B --> C"], delta_hrxn=PFR_HEAT_OF_REACTION,
        tref_hrxn=TEMP_REF)
    unit.Inlet = LiquidStream(path, temp=PFR_TEMP, mole_conc=conc,
                              vol_flow=PFR_HOLDUP / PFR_RESIDENCE_TIME)
    unit.set_names()
    unit.c_inert = unit.Inlet.mole_conc[~unit.mask_species]  # [mol/L]
    return unit.energy_steady(conc[unit.mask_species], PFR_TEMP)


@pytest.mark.unit
def test_pfr_energy_balance_ignores_absent_species_without_cp_liq(tmp_path):
    """The absent solvent's cp never enters the volumetric heat capacity."""
    database = json.loads(
        (REPO_ROOT / "tests/integration/data/pfr_test_pure_comp.json")
        .read_text())
    missing = copy.deepcopy(database)
    del missing["solv"]["cp_liq"]
    arbitrary = copy.deepcopy(database)
    arbitrary["solv"]["cp_liq"] = [123.0]  # [J/mol/K], arbitrary value

    derivative = _pfr_steady_derivative(_write(tmp_path, missing, "m.json"))
    reference = _pfr_steady_derivative(_write(tmp_path, arbitrary, "a.json"))

    assert np.isfinite(derivative)
    assert derivative == reference

    # Present in the holdup and feed, the solvent needs its cp_liq.
    present_conc = 0.5  # [mol/L]
    with pytest.raises(MissingPropertyError, match=r"^Species \['solv'\] carry a "
                       r"nonzero weight.*'cp_liq'"):
        _pfr_steady_derivative(_write(tmp_path, missing, "p.json"),
                               present_conc)


# ----------------------------------------------------------------- drying
# Cake pressure drop for ``Drying.initialize_states``: the value of the
# existing drying gas-balance tests. It sets the gas flow, which is the same
# in the two databases each test compares, so the comparisons do not depend
# on it.
DRYING_PRESSURE_DROP = 5.0e4  # [Pa]


def _dryer(drying_unit_factory, path):
    """Build the conftest drying unit on a given property database.

    Parameters
    ----------
    drying_unit_factory : callable
        The conftest factory.
    path : str
        Database derived from ``conftest.DRYING_THERMO_DATA``.

    Returns
    -------
    tuple
        The ``Drying`` unit and its initial packed state (units of
        ``Drying.initialize_states``).
    """
    dryer = drying_unit_factory(thermo_path=path)
    states = dryer.initialize_states(deltaP=DRYING_PRESSURE_DROP).ravel()
    return dryer, states


@pytest.mark.unit
def test_drying_equilibrium_needs_no_p_vap_for_the_carrier_gas(
        tmp_path, drying_unit_factory):
    """The supercritical drying carrier may omit Antoine coefficients."""
    from conftest import DRYING_THERMO_DATA

    database = copy.deepcopy(DRYING_THERMO_DATA)
    del database["nitrogen"]["p_vap"]
    dryer, _ = _dryer(drying_unit_factory, _write(tmp_path, database))
    assert dryer.Liquid_1.name_species[2] == "nitrogen"
    assert np.isnan(dryer.Liquid_1.p_vap[2]).all()

    temp_cond = np.array([300.0, 310.0])  # [K], two nodes
    x_liq = np.array([[0.5, 0.5], [0.25, 0.75]])  # [-], water/ethanol mass
    p_gas = 1.0e5  # [Pa]
    y_equil = dryer.get_y_equilib(temp_cond, x_liq, p_gas)  # [-]

    mw = np.array([database["water"]["mw"], database["ethanol"]["mw"]])  # [g/mol]
    moles = x_liq / mw  # [mol/g]
    x_mole = moles / moles.sum(axis=1, keepdims=True)  # [-]
    psat = np.array([[_antoine_pressure(database[name]["p_vap"], temp)
                      for name in ("water", "ethanol")]
                     for temp in temp_cond])  # [Pa]
    # get_y_equilib returns shape (num_volatiles, num_nodes).
    np.testing.assert_allclose(y_equil, (x_mole * psat / p_gas).T,
                               rtol=CLOSED_FORM_RTOL)


@pytest.mark.unit
def test_drying_model_ignores_the_carrier_antoine_data(tmp_path,
                                                       drying_unit_factory):
    """The full drying right-hand side never reads the carrier's p_vap."""
    from conftest import DRYING_THERMO_DATA

    without = copy.deepcopy(DRYING_THERMO_DATA)
    del without["nitrogen"]["p_vap"]
    arbitrary = copy.deepcopy(DRYING_THERMO_DATA)
    arbitrary["nitrogen"]["p_vap"] = [5.0, 100.0, 0.0]  # [-], [K], [K]
    derivatives = []
    for name, data in (("without.json", without), ("arbitrary.json",
                                                   arbitrary)):
        dryer, states = _dryer(drying_unit_factory,
                               _write(tmp_path, data, name))
        derivatives.append(np.asarray(dryer.unit_model(0.0, states)))  # [s]

    assert np.isfinite(derivatives[0]).all()
    np.testing.assert_array_equal(derivatives[0], derivatives[1])


# Tank-reactor case: first-order, exothermic A --> B in a solvent, about one
# reaction time constant long, cooled by a 300 K utility. "impurity" has no
# cp_liq; it is absent unless a test charges or feeds it.
# Units: mw [g/mol], cp_liq [J/mol/K], rho_liq [kg/m**3]. Reactors read no
# Antoine data, so the fixture has none.
REACTOR_DATABASE = {
    "A": {"mw": 100.0, "cp_liq": [200.0], "rho_liq": 1000.0},
    "B": {"mw": 100.0, "cp_liq": [210.0], "rho_liq": 1000.0},
    "solv": {"mw": 50.0, "cp_liq": [100.0], "rho_liq": 800.0},
    "impurity": {"mw": 120.0, "rho_liq": 1000.0},
}
REACTOR_CONC = [1.0, 0.0, 12.0, 0.0]  # [mol/L], A, B, solvent, impurity
IMPURITY_DOSE = [0.0, 0.0, 0.0, 0.5]  # [mol/L], impurity added to a stream
RATE_CONSTANT = 1e-2  # [1/s], first order
HEAT_OF_REACTION = -5e4  # [J/mol], exothermic
TANK_TEMP = 320.0  # [K], initial liquid
FEED_TEMP = 300.0  # [K]
UTILITY_TEMP = 300.0  # [K]
UTILITY_FLOW = 0.1  # [kg/s]
TANK_HOLDUP = 1.0e-3  # [m**3]
TANK_FEED_FLOW = 1.0e-5  # [m**3/s], continuous tanks
SEMIBATCH_FEED_FLOW = 1.0e-6  # [m**3/s], keeps the 2 L tank below capacity
TANK_RUNTIME = 100.0  # [s], one time constant at RATE_CONSTANT
SHORT_RUNTIME = 10.0  # [s]
PULSE_START, PULSE_END = 20.0, 40.0  # [s], impurity feed window
STEP_CAP = 5.0  # [s], maximum CVode step, so the solver samples the pulse
ARBITRARY_CP = [123.0]  # [J/mol/K], data for a species that must not matter


def _kinetics(path):
    """First-order A --> B kinetics on a database.

    Parameters
    ----------
    path : str
        Property database path.

    Returns
    -------
    RxnKinetics
        Rate constant RATE_CONSTANT, heat of reaction HEAT_OF_REACTION.
    """
    from PharmaPy.Kinetics import RxnKinetics

    return RxnKinetics(path, rxn_list=["A --> B"], k_params=[RATE_CONSTANT],
                       ea_params=[0.0],
                       delta_hrxn=[HEAT_OF_REACTION])  # [J/mol]


def _impurity_pulse(start=PULSE_START, end=PULSE_END):
    """Feed controller that adds the impurity between two times.

    Parameters
    ----------
    start, end : float, optional
        Window [s] in which IMPURITY_DOSE is added to REACTOR_CONC; ``end``
        may be ``numpy.inf``.

    Returns
    -------
    DynamicInput
        Controls only ``mole_conc`` [mol/L].
    """
    from PharmaPy.ProcessControl import DynamicInput

    base_feed = np.array(REACTOR_CONC)  # [mol/L]

    def feed_conc(time):
        """Feed concentrations with the impurity window.

        Parameters
        ----------
        time : float or numpy.ndarray
            Absolute time [s].

        Returns
        -------
        numpy.ndarray
            Concentrations [mol/L], shape (4,) or (num_times, 4).
        """
        window = ((np.asarray(time) > start)
                  & (np.asarray(time) < end)).astype(float)  # [-]
        return base_feed + np.multiply.outer(window, IMPURITY_DOSE)  # [mol/L]

    controller = DynamicInput()
    controller.add_variable("mole_conc", feed_conc)
    return controller


def _tank(kind, path, conc=REACTOR_CONC, inlet_path=None, controller=None,
          feed=REACTOR_CONC):
    """Configure a tank reactor of the fixture case.

    Parameters
    ----------
    kind : {'batch_iso', 'batch', 'cstr_iso', 'cstr', 'semibatch'}
        Reactor class and energy mode.
    path : str
        Reactor property database.
    conc : sequence of float, optional
        Initial holdup concentrations [mol/L].
    inlet_path : str, optional
        Separate inlet database; ``path`` when None.
    controller : DynamicInput, optional
        Dynamic inlet of the feed.
    feed : sequence of float, optional
        Static feed concentrations [mol/L].

    Returns
    -------
    _BaseReactor
        Unit with kinetics, phase, inlet and utility as applicable.
    """
    from PharmaPy.Reactors import BatchReactor, CSTR, SemibatchReactor
    from PharmaPy.Streams import LiquidStream
    from PharmaPy.Utilities import CoolingWater

    isothermal = kind.endswith("_iso")
    if kind.startswith("batch"):
        unit = BatchReactor(isothermal=isothermal)
    elif kind.startswith("cstr"):
        unit = CSTR(isothermal=isothermal)
    else:
        unit = SemibatchReactor(vol_tank=2 * TANK_HOLDUP, isothermal=False)
    unit.Kinetics = _kinetics(path)
    unit.Phases = LiquidPhase(path, temp=TANK_TEMP, vol=TANK_HOLDUP,
                              mole_conc=conc, name_solv="solv")
    if not kind.startswith("batch"):
        flow = (SEMIBATCH_FEED_FLOW if kind == "semibatch"
                else TANK_FEED_FLOW)  # [m**3/s]
        inlet = LiquidStream(inlet_path or path, temp=FEED_TEMP,
                             mole_conc=feed, vol_flow=flow, name_solv="solv")
        if controller is not None:
            inlet.DynamicInlet = controller
        unit.Inlet = inlet
    if not isothermal:
        unit.Utility = CoolingWater(mass_flow=UTILITY_FLOW,
                                    temp_in=UTILITY_TEMP)
    return unit


def _solve(unit, runtime=TANK_RUNTIME, **kwargs):
    """Solve a configured reactor quietly.

    Parameters
    ----------
    unit : _BaseReactor
        Configured unit.
    runtime : float, optional
        Integration span [s].
    **kwargs
        Extra ``solve_unit`` options.

    Returns
    -------
    tuple of numpy.ndarray
        Times [s] and states (units of the reactor's ``states_di``).
    """
    time, states = unit.solve_unit(runtime=runtime, verbose=False, **kwargs)
    return np.asarray(time), np.asarray(states)


def _with_impurity_cp(cp=ARBITRARY_CP):
    """REACTOR_DATABASE with cp_liq data for the impurity.

    Parameters
    ----------
    cp : list of float, optional
        Impurity cp_liq coefficients [J/mol/K].

    Returns
    -------
    dict
        Species records.
    """
    database = copy.deepcopy(REACTOR_DATABASE)
    database["impurity"]["cp_liq"] = list(cp)
    return database


@pytest.mark.integration
@pytest.mark.assimulo
@pytest.mark.parametrize("kind", ["batch_iso", "batch", "cstr", "semibatch"])
def test_reactors_ignore_absent_species_without_cp_liq(tmp_path, kind):
    """An absent species needs no cp_liq in any reactor energy balance.

    The run with the impurity's cp_liq omitted must equal the run with an
    arbitrary value: a never-present species cannot change the arithmetic,
    so equality is exact. Presence is structural (see
    ``_BaseReactor._present_species``), so solver perturbations of the
    impurity's zero concentration state do not make it present. With the
    kernel change alone (strict per-species heat capacities) these runs
    raised; on the pre-#414 parser the zero fill hid the missing data.
    """
    pytest.importorskip("assimulo")
    runs = []
    for name, data in (("missing.json", REACTOR_DATABASE),
                       ("arbitrary.json", _with_impurity_cp())):
        unit = _tank(kind, _write(tmp_path, data, name))
        time, states = _solve(unit)
        runs.append((time, states, np.asarray(unit.result.q_rxn)))  # [W]

    assert np.isfinite(runs[0][1]).all()
    for observed, expected in zip(*runs):
        np.testing.assert_array_equal(observed, expected)


@pytest.mark.integration
@pytest.mark.assimulo
def test_reactor_presence_latches_species_fed_by_a_stopped_feed(tmp_path):
    """A species fed only for a while stays present after its feed stops.

    Isothermal CSTR: the impurity is fed between two reported times. The
    heat-profile reconstruction samples the feed only where it is zero, yet
    the impurity is in the tank there, so the named cp_liq error must be
    raised instead of a silently zero impurity enthalpy. The two runs with
    cp_liq data show that the impurity's enthalpy changes ``q_flow``.
    """
    pytest.importorskip("assimulo")
    report_times = [0.0, 10.0, 50.0, 100.0]  # [s], outside the pulse
    runs = []
    for value in ([0.0], ARBITRARY_CP):  # [J/mol/K], zero (old fill), other
        unit = _tank("cstr_iso", _write(tmp_path, _with_impurity_cp(value),
                                        f"cp_{value[0]}.json"),
                     controller=_impurity_pulse())
        _, states = _solve(unit, time_grid=report_times,
                           sundials_opts={"maxh": STEP_CAP})
        runs.append((states, np.asarray(unit.result.q_flow)))  # [W]
    impurity_conc = runs[0][0][:, 3]  # [mol/L]
    assert impurity_conc[0] == 0.0 and impurity_conc[-1] > 0.0  # [mol/L]
    assert runs[0][1][-1] != runs[1][1][-1]

    unit = _tank("cstr_iso", _write(tmp_path, REACTOR_DATABASE, "m.json"),
                 controller=_impurity_pulse())
    with pytest.raises(MissingPropertyError, match=r"^Species "
                       r"\['impurity'\] carry a nonzero weight.*'cp_liq'"):
        _solve(unit, time_grid=report_times,
               sundials_opts={"maxh": STEP_CAP})


@pytest.mark.integration
@pytest.mark.assimulo
def test_batch_reactor_needs_cp_liq_for_a_species_only_in_the_holdup(
        tmp_path):
    """A non-reacting species in the initial holdup is present."""
    pytest.importorskip("assimulo")
    charged = np.add(REACTOR_CONC, IMPURITY_DOSE)  # [mol/L]
    unit = _tank("batch", _write(tmp_path, REACTOR_DATABASE), conc=charged)

    with pytest.raises(MissingPropertyError, match=r"^Species "
                       r"\['impurity'\] carry a nonzero weight.*'cp_liq'"):
        _solve(unit)


def _pfr_database(imp_cp=None):
    """Shipped PFR database with an extra species "imp".

    Parameters
    ----------
    imp_cp : list of float, optional
        cp_liq coefficients of "imp" [J/mol/K]; omitted when None.

    Returns
    -------
    dict
        Species records in the units of the shipped file.
    """
    database = json.loads(
        (REPO_ROOT / "tests/integration/data/pfr_test_pure_comp.json")
        .read_text())
    database["imp"] = dict(PFR_IMPURITY)
    if imp_cp is not None:
        database["imp"]["cp_liq"] = imp_cp
    return database


def _pfr(path, inlet_imp=0.0, holdup_imp=0.0, imp_participates=False,
         unit=None):
    """Configure a small non-isothermal PFR with an optional "imp".

    Parameters
    ----------
    path : str
        Database from ``_pfr_database``.
    inlet_imp, holdup_imp : float, optional
        "imp" concentration [mol/L] in the inlet and in the initial holdup.
    imp_participates : bool, optional
        List "imp" as a participating species with a zero stoichiometric
        coefficient.
    unit : PlugFlowReactor, optional
        Unit to reconfigure; a new one when None.

    Returns
    -------
    PlugFlowReactor
        Configured unit.
    """
    from PharmaPy.Kinetics import RxnKinetics
    from PharmaPy.Reactors import PlugFlowReactor
    from PharmaPy.Streams import LiquidStream
    from PharmaPy.Utilities import CoolingWater

    if unit is None:
        unit = PlugFlowReactor(PFR_DIAMETER, PFR_NODES, isothermal=False)
    unit.Phases = LiquidPhase(path, temp=PFR_TEMP, vol=PFR_HOLDUP,
                              mole_conc=PFR_BASE_CONC + [holdup_imp])
    if imp_participates:
        kinetics = RxnKinetics(
            path, k_params=[PFR_RATE_CONSTANT], ea_params=[0.0],
            stoich_matrix=[[-1, -1, 1, 0]],
            partic_species=["A", "B", "C", "imp"],
            delta_hrxn=PFR_HEAT_OF_REACTION, tref_hrxn=TEMP_REF)
    else:
        kinetics = RxnKinetics(
            path, k_params=[PFR_RATE_CONSTANT], ea_params=[0.0],
            rxn_list=["A + B --> C"], delta_hrxn=PFR_HEAT_OF_REACTION,
            tref_hrxn=TEMP_REF)
    unit.Kinetics = kinetics
    unit.Inlet = LiquidStream(path, temp=PFR_TEMP,
                              mole_conc=PFR_BASE_CONC + [inlet_imp],
                              vol_flow=PFR_HOLDUP / PFR_RESIDENCE_TIME)
    unit.Utility = CoolingWater(mass_flow=UTILITY_FLOW, temp_in=UTILITY_TEMP)
    return unit


def _run_pfr(path, steady=False, **case):
    """Run the PFR case transiently or at steady state.

    Parameters
    ----------
    path : str
        Database from ``_pfr_database``.
    steady : bool, optional
        Run ``solve_steady`` (adiabatic) instead of ``solve_unit``.
    **case
        Options of ``_pfr``.

    Returns
    -------
    numpy.ndarray
        States: transient [mol/L, K] per node and time, or the steady
        profile along volume.
    """
    unit = _pfr(path, **case)
    if steady:
        return np.asarray(unit.solve_steady(PFR_HOLDUP, adiabatic=True)[1])
    return _solve(unit, runtime=PFR_RUNTIME)[1]


@pytest.mark.integration
@pytest.mark.assimulo
@pytest.mark.parametrize("case", [
    dict(),
    dict(steady=True),
    dict(steady=True, imp_participates=True),
    dict(steady=True, holdup_imp=PFR_IMPURITY_CONC),
], ids=["transient", "steady", "steady_zero_stoichiometry",
        "steady_unused_holdup"])
def test_pfr_ignores_absent_species_without_cp_liq(tmp_path, case):
    """PFR runs with an absent "imp" equal the runs with arbitrary data.

    The steady solve builds everything from the inlet: a zero-stoichiometry
    participant that the solver perturbs and a species only in the unused
    transient holdup are both absent.
    """
    pytest.importorskip("assimulo")
    missing = _run_pfr(_write(tmp_path, _pfr_database(), "m.json"), **case)
    reference = _run_pfr(
        _write(tmp_path, _pfr_database(ARBITRARY_CP), "a.json"), **case)

    assert np.isfinite(missing).all()
    np.testing.assert_array_equal(missing, reference)


@pytest.mark.integration
@pytest.mark.assimulo
@pytest.mark.parametrize("steady", [False, True], ids=["transient", "steady"])
def test_pfr_raises_named_error_for_a_fed_species_without_cp_liq(tmp_path,
                                                                 steady):
    """A fed species needs cp_liq; the error names it on the public path.

    Both solves evaluate the right-hand side once before building CVode,
    so the error is raised directly, not chained from a solver error.
    """
    pytest.importorskip("assimulo")
    with pytest.raises(MissingPropertyError, match=r"^Species \['imp'\] "
                       r"carry a nonzero weight.*'cp_liq'") as error:
        _run_pfr(_write(tmp_path, _pfr_database()), steady=steady,
                 inlet_imp=PFR_IMPURITY_CONC)
    assert error.value.__cause__ is None


def _failing_and_clean(kind, path):
    """Build a unit that fails for the impurity, and its clean re-setup.

    Parameters
    ----------
    kind : {'batch', 'cstr', 'semibatch', 'pfr'}
        Reactor case.
    path : str
        Database without impurity cp_liq (PFR: ``_pfr_database()``).

    Returns
    -------
    tuple
        The failing unit, a function that reconfigures a given unit (or a
        new one, when passed None) for an impurity-free run, and the solve
        options [s] of the case.
    """
    if kind == "pfr":
        unit = _pfr(path, inlet_imp=PFR_IMPURITY_CONC)

        def clean(target):
            """Reconfigure a PFR without "imp".

            Parameters
            ----------
            target : PlugFlowReactor or None
                Unit to reconfigure; a new one when None.

            Returns
            -------
            PlugFlowReactor
                Impurity-free unit.
            """
            return _pfr(path, unit=target)

        return unit, clean, dict(runtime=PFR_RUNTIME)

    if kind == "batch":
        unit = _tank("batch", path, conc=np.add(REACTOR_CONC, IMPURITY_DOSE))
    else:
        unit = _tank(kind, path, controller=_impurity_pulse())

    def clean(target):
        """Give a tank an impurity-free charge and static feed.

        Parameters
        ----------
        target : _BaseReactor or None
            Unit to reconfigure; a new one when None.

        Returns
        -------
        _BaseReactor
            Impurity-free unit.
        """
        from PharmaPy.Streams import LiquidStream

        fresh = _tank(kind, path)
        if target is None:
            return fresh
        target.Phases = LiquidPhase(path, temp=TANK_TEMP, vol=TANK_HOLDUP,
                                    mole_conc=REACTOR_CONC, name_solv="solv")
        if kind != "batch":
            target.Inlet = LiquidStream(path, temp=FEED_TEMP,
                                        mole_conc=REACTOR_CONC,
                                        vol_flow=fresh.Inlet.vol_flow,
                                        name_solv="solv")
        return target

    return unit, clean, dict(runtime=TANK_RUNTIME,
                             sundials_opts={"maxh": STEP_CAP})


@pytest.mark.integration
@pytest.mark.assimulo
@pytest.mark.parametrize("kind", ["batch", "cstr", "semibatch", "pfr"])
def test_named_failure_then_clean_resolve_matches_fresh_unit(tmp_path, kind):
    """After the named error, the same unit re-solves like a fresh one.

    Batch charges the impurity, CSTR and semibatch feed it from 20 s to
    40 s (a mid-run failure inside CVode, re-raised chained from the CVode
    error), and the PFR feeds it from the start. The unit then gets an
    impurity-free charge and feed: the latched presence and the recorded
    error of the failed solve must not leak into the next solve.
    """
    pytest.importorskip("assimulo")
    database = _pfr_database() if kind == "pfr" else REACTOR_DATABASE
    path = _write(tmp_path, database)
    unit, clean, options = _failing_and_clean(kind, path)

    with pytest.raises(MissingPropertyError, match=r"carry a nonzero "
                       r"weight.*'cp_liq'") as error:
        _solve(unit, **options)
    if kind in ("cstr", "semibatch"):
        assert type(error.value.__cause__).__name__ == "CVodeError"

    reused = _solve(clean(unit), **options)
    fresh = _solve(clean(None), **options)
    for observed, expected in zip(reused, fresh):
        np.testing.assert_array_equal(observed, expected)


@pytest.mark.integration
@pytest.mark.assimulo
def test_later_unrelated_solver_failure_is_not_relabeled(tmp_path):
    """A recorded missing-data error does not label a later failure."""
    pytest.importorskip("assimulo")
    path = _write(tmp_path, REACTOR_DATABASE)
    unit, clean, options = _failing_and_clean("cstr", path)
    with pytest.raises(MissingPropertyError):
        _solve(unit, **options)

    from assimulo.exception import AssimuloException

    negative_runtime = -50.0  # [s], before the start time
    with pytest.raises(AssimuloException, match="Final time") as error:
        clean(unit).solve_unit(runtime=negative_runtime, verbose=False)
    assert not isinstance(error.value, MissingPropertyError)


@pytest.mark.unit
def test_inlet_enthalpy_keeps_requiring_a_species_after_its_feed_stops(
        tmp_path):
    """The fed-species latch, without a solver.

    A separate inlet database lacks the impurity's cp_liq; the pulse feeds
    it between 20 s and 40 s. Before any feed of it (10 s) the energy
    balance needs no impurity data. After it was fed (input evaluated at
    30 s), the balance at 50 s, where the current feed is impurity-free,
    still needs it from the inlet database.
    """
    reactor_path = _write(tmp_path, _with_impurity_cp(), "reactor.json")
    inlet_path = _write(tmp_path, REACTOR_DATABASE, "inlet.json")
    unit = _tank("cstr_iso", reactor_path, inlet_path=inlet_path,
                 controller=_impurity_pulse())
    unit.set_names()
    conc = np.atleast_2d(REACTOR_CONC)  # [mol/L], holdup without impurity
    temp = np.array([TANK_TEMP])  # [K]
    before, during, after = 10.0, 30.0, 50.0  # [s], around the pulse

    heat = unit.energy_balances(before, conc, TANK_HOLDUP, temp, None,
                                unit.get_inputs(before), heat_prof=True)  # [W]
    assert np.isfinite(heat).all()

    unit.get_inputs(during)
    inputs_after = unit.get_inputs(after)
    assert inputs_after["Inlet"]["mole_conc"][3] == 0.0  # [mol/L]
    with pytest.raises(MissingPropertyError, match=r"^Species "
                       r"\['impurity'\] carry a nonzero weight, so their "
                       r"enthalpy.*'cp_liq'") as error:
        unit.energy_balances(after, conc, TANK_HOLDUP, temp, None,
                             inputs_after, heat_prof=True)
    assert inlet_path in str(error.value)


@pytest.mark.integration
@pytest.mark.assimulo
@pytest.mark.parametrize("kind", ["batch", "cstr"])
def test_unit_stays_copyable_after_a_pre_solve_missing_data_error(tmp_path,
                                                                  kind):
    """An error raised before CVode runs leaves nothing on the unit.

    The first right-hand-side evaluation, before the solver is built,
    raises the named error directly; recording it (with its traceback)
    would make ``copy.deepcopy`` of the unit fail.
    """
    pytest.importorskip("assimulo")
    path = _write(tmp_path, REACTOR_DATABASE)
    charged = np.add(REACTOR_CONC, IMPURITY_DOSE)  # [mol/L]
    if kind == "batch":
        unit = _tank("batch", path, conc=charged)
    else:
        unit = _tank("cstr", path, feed=charged)

    with pytest.raises(MissingPropertyError) as error:
        _solve(unit)
    assert error.value.__cause__ is None

    duplicate = copy.deepcopy(unit)
    assert duplicate.Liquid_1.name_species == unit.Liquid_1.name_species


# Child-process script for the termination test: solve the separate-inlet
# case and print the exception type, its cause type and the CVode flag.
_SEPARATE_INLET_SCRIPT = """
import sys
import warnings
from pathlib import Path
import numpy as np
sys.path.insert(0, {tests_dir!r})
warnings.simplefilter("ignore")
import test_missing_list_property_data as case
reactor_path = case._write(Path({tmp_dir!r}), case._with_impurity_cp(),
                           "r.json")
inlet_path = case._write(Path({tmp_dir!r}), case.REACTOR_DATABASE, "i.json")
unit = case._tank("cstr", reactor_path, inlet_path=inlet_path,
                  controller=case._impurity_pulse(end=np.inf))
try:
    case._solve(unit)
    print("NO ERROR")
except Exception as error:
    cause = error.__cause__
    print(type(error).__name__, type(cause).__name__,
          getattr(cause, "args", (None,))[0])
"""
# A healthy run fails within a second; the defect stalled CVode
# indefinitely at t = 20 s, so a generous wall-clock bound separates them.
SEPARATE_INLET_TIMEOUT = 120.0  # [s]
# CVode flag CV_REPTD_RHSFUNC_ERR: the right-hand side failed repeatedly.
CV_REPEATED_RHS_FAILURE = -10  # [-]


@pytest.mark.integration
@pytest.mark.assimulo
def test_feed_that_starts_mid_run_from_a_separate_database_terminates(
        tmp_path):
    """A cp-less species fed from 20 s through a separate inlet database.

    The inlet enthalpy check is latched and the recorded error is sticky,
    so CVode cannot step around the failure: the solve ends with the named
    error chained from a repeated right-hand-side failure. The case runs in
    a child process with a wall-clock bound, because the defect made CVode
    stall without returning (``maxsteps`` does not bound that). Either
    defense alone already ends the run; the test is red with both removed.
    """
    pytest.importorskip("assimulo")
    script = _SEPARATE_INLET_SCRIPT.format(
        tests_dir=str(Path(__file__).resolve().parent),
        tmp_dir=str(tmp_path))

    result = subprocess.run(
        [sys.executable, "-c", script], cwd=REPO_ROOT, capture_output=True,
        text=True, check=False, timeout=SEPARATE_INLET_TIMEOUT)

    assert result.stdout.split() == [
        "MissingPropertyError", "CVodeError", str(CV_REPEATED_RHS_FAILURE)]


def _override_states(path, controller):
    """Isothermal CSTR whose static feed carries an unused impurity.

    Parameters
    ----------
    path : str
        REACTOR_DATABASE path, possibly with an impurity cp_liq.
    controller : object
        Dynamic inlet set on the feed stream.

    Returns
    -------
    numpy.ndarray
        States [mol/L] after SHORT_RUNTIME.
    """
    static_feed = np.add(REACTOR_CONC, IMPURITY_DOSE)  # [mol/L], overridden
    unit = _tank("cstr_iso", path, controller=controller, feed=static_feed)
    return _solve(unit, runtime=SHORT_RUNTIME)[1]


def _impurity_free_feed():
    """Dynamic inlet that controls ``mole_conc`` with an impurity-free feed.

    Returns
    -------
    DynamicInput
        Constant REACTOR_CONC [mol/L].
    """
    from PharmaPy.ProcessControl import DynamicInput

    controller = DynamicInput()
    controller.add_variable(
        "mole_conc",
        lambda time: np.broadcast_to(REACTOR_CONC,
                                     np.shape(time) + (4,)).copy())
    return controller


@pytest.mark.integration
@pytest.mark.assimulo
def test_dynamic_feed_overrides_static_inlet_composition(tmp_path):
    """Only the feed the reactor evaluates decides presence."""
    pytest.importorskip("assimulo")
    missing = _override_states(_write(tmp_path, REACTOR_DATABASE, "m"),
                               _impurity_free_feed())
    reference = _override_states(_write(tmp_path, _with_impurity_cp(), "a"),
                                 _impurity_free_feed())

    np.testing.assert_array_equal(missing, reference)


def _connected_inlet_states(path, temperature_control):
    """CSTR whose stream carries an upstream profile with the impurity.

    Parameters
    ----------
    path : str
        REACTOR_DATABASE path, possibly with an impurity cp_liq.
    temperature_control : bool
        Also give the stream a ``DynamicInlet`` that controls only its
        temperature; ``get_inputs`` then ignores the upstream profile.

    Returns
    -------
    numpy.ndarray
        States [mol/L, K] after SHORT_RUNTIME.
    """
    from PharmaPy.ProcessControl import DynamicInput

    unit = _tank("cstr", path)
    profile_times = np.array([0.0, TANK_RUNTIME])  # [s]
    profile = np.tile(np.add(REACTOR_CONC, IMPURITY_DOSE), (2, 1))  # [mol/L]
    unit.Inlet.time_upstream = profile_times
    unit.Inlet.y_inlet = {"mole_conc": profile}
    unit.Inlet.y_upstream = unit.Inlet.y_inlet
    if temperature_control:
        controller = DynamicInput()
        controller.add_variable(
            "temp", lambda time: FEED_TEMP + np.zeros_like(time))  # [K]
        unit.Inlet.DynamicInlet = controller
    return _solve(unit, runtime=SHORT_RUNTIME)[1]


@pytest.mark.integration
@pytest.mark.assimulo
def test_temperature_control_makes_the_upstream_profile_irrelevant(
        tmp_path):
    """With any DynamicInlet, the ignored upstream profile adds nothing.

    The control sets only the temperature, so the feed concentration is
    the static, impurity-free one, as ``get_inputs`` evaluates it.
    """
    pytest.importorskip("assimulo")
    missing = _connected_inlet_states(
        _write(tmp_path, REACTOR_DATABASE, "m.json"), True)
    reference = _connected_inlet_states(
        _write(tmp_path, _with_impurity_cp(), "a.json"), True)

    np.testing.assert_array_equal(missing, reference)


@pytest.mark.integration
@pytest.mark.assimulo
def test_connected_upstream_profile_species_needs_cp_liq(tmp_path):
    """Without a DynamicInlet, the upstream profile is the feed.

    The impurity is only in the profile, not in the static composition;
    it is fed, so it needs cp_liq.
    """
    pytest.importorskip("assimulo")
    with pytest.raises(MissingPropertyError, match=r"^Species "
                       r"\['impurity'\] carry a nonzero weight.*'cp_liq'"):
        _connected_inlet_states(_write(tmp_path, REACTOR_DATABASE), False)


@pytest.mark.integration
@pytest.mark.assimulo
def test_inlet_enthalpy_needs_data_only_for_fed_species(tmp_path):
    """The inlet's own database needs cp_liq only for what it feeds.

    B is formed in the tank (so the reactor database has its cp_liq) but
    is not fed; the inlet's separate database omits B's cp_liq.
    """
    pytest.importorskip("assimulo")
    from PharmaPy.Reactors import CSTR
    from PharmaPy.Streams import LiquidStream

    reactor_db = {name: dict(REACTOR_DATABASE[name])
                  for name in ("A", "B", "solv")}
    reactor_path = _write(tmp_path, reactor_db, "reactor.json")
    feed = REACTOR_CONC[:3]  # [mol/L], no B in the feed
    results = []
    for b_cp in (None, ARBITRARY_CP):  # [J/mol/K], omitted or arbitrary
        feed_db = copy.deepcopy(reactor_db)
        del feed_db["B"]["cp_liq"]
        if b_cp is not None:
            feed_db["B"]["cp_liq"] = b_cp
        feed_path = _write(tmp_path, feed_db, f"feed_{b_cp}.json")
        unit = CSTR(isothermal=True)
        unit.Kinetics = _kinetics(reactor_path)
        unit.Phases = LiquidPhase(reactor_path, temp=TANK_TEMP,
                                  vol=TANK_HOLDUP, mole_conc=feed,
                                  name_solv="solv")
        unit.Inlet = LiquidStream(feed_path, temp=FEED_TEMP, mole_conc=feed,
                                  vol_flow=TANK_FEED_FLOW, name_solv="solv")
        _, states = _solve(unit, runtime=SHORT_RUNTIME)
        results.append((states, np.asarray(unit.result.q_flow)))  # [W]

    for observed, expected in zip(*results):
        np.testing.assert_array_equal(observed, expected)


# ----------------------------------------------------- evaporator (solver)
@pytest.mark.integration
@pytest.mark.assimulo
def test_nitrogen_evaporator_on_shipped_database_ignores_missing_rows(
        tmp_path):
    """The shipped nitrogen merge keeps its trajectory exactly.

    ``Evaporator(include_nitrogen=True)`` merges the shipped
    ``tests/Flowsheet/data/compound_database.json`` with
    ``data/evaporator/props_nitrogen.json``. The user species have no
    ``cp_vapor`` and nitrogen has no ``visc_liq``, so the merged phases carry
    NaN rows. The same run with explicit ``cp_vapor`` rows for the user
    species, either zero (the representation earlier releases parsed) or an
    arbitrary constant, must give an identical trajectory: rows that are
    never read cannot change any arithmetic, so equality is exact.

    Out of band, during the #414 review, the same 600 s run was also compared
    with the pre-#414 code: all 145 x 23 stored states were bit-identical.
    That comparison needs an old checkout and is not repeated here.
    """
    pytest.importorskip("assimulo")
    from PharmaPy.Evaporators import Evaporator
    from PharmaPy.Utilities import CoolingWater

    shipped = REPO_ROOT / "tests/Flowsheet/data/compound_database.json"
    database = json.loads(shipped.read_text())
    assert all("cp_vapor" not in record for record in database.values())
    zero_rows = copy.deepcopy(database)
    other_rows = copy.deepcopy(database)
    for name in database:
        zero_rows[name]["cp_vapor"] = [0.0]  # [J/mol/K], former zero fill
        other_rows[name]["cp_vapor"] = [123.0]  # [J/mol/K], arbitrary value
    paths = [str(shipped), _write(tmp_path, zero_rows, "zero.json"),
             _write(tmp_path, other_rows, "other.json")]

    # Synthetic operating point: a partly filled drum, a subcooled charge
    # of mostly solvent (asymmetric composition) and a hotter utility, so
    # the run heats and boils the charge well below every t_crit (>= 540 K).
    pressure = PRESSURE  # [Pa]
    drum_volume = 1.0  # [m**3]
    charge_volume = 0.2  # [m**3]
    charge_temp = 320.0  # [K], below the charge's bubble point
    mole_frac = np.array([0.05, 0.05, 0.02, 0.03, 0.85])  # [-]
    utility_flow = 1.0  # [kg/s]
    utility_temp = 400.0  # [K]
    duration = 600.0  # [s], long enough to reach boiling
    trajectories = []
    for path in paths:
        unit = Evaporator(drum_volume, pressure=pressure,
                          include_nitrogen=True)
        unit.Phases = LiquidPhase(path, temp=charge_temp, pres=pressure,
                                  vol=charge_volume, mole_frac=mole_frac)
        unit.Utility = CoolingWater(mass_flow=utility_flow,
                                    temp_in=utility_temp)
        time, states = unit.solve_unit(duration, verbose=False)
        trajectories.append((np.asarray(time), states))
        if path == paths[0]:
            merged = ParseDatabase(unit.paths)  # the files the unit merged
            final_temp = unit.result.temp[-1]  # [K]

    assert merged["name_species"] == list(database) + ["nitrogen"]
    assert np.isnan(merged["cp_vapor"][:-1]).all()
    assert np.isfinite(merged["cp_vapor"][-1]).all()
    assert np.isnan(merged["visc_liq"][-1]).all()
    assert final_temp > charge_temp  # [K], the charge was heated
    reference_time, reference_states = trajectories[0]
    assert np.isfinite(reference_states).all()
    for time, states in trajectories[1:]:
        np.testing.assert_array_equal(time, reference_time)
        np.testing.assert_array_equal(states, reference_states)
