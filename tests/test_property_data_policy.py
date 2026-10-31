"""Property-database key checks and the missing-property exception policy.

Issue #424. ``MissingPropertyError`` is the one exception for missing or
invalid property data and subclasses both ``AttributeError`` and
``ValueError``, the types that earlier handlers catch. These tests cover:

- scalar data that a present species lacks: ``henry_constant`` of a
  supercritical species in ``getKeqVLE``, ``rho_liq``/``rho_solid`` in the
  density methods, phase construction and concentration conversions
  (including the solvent of a solvent closure, and the stored zero-fraction
  sentinel of solids, which counts as absent), and ``surf_tension``; absent species need none and contribute exactly zero
  (or, for a Henry constant, keep the documented NaN ratio);
- keys that no species supplies (``t_crit``, ``henry_constant``), missing
  ``visc_gas`` and invalid Watson latent-heat data, now species-named
  ``MissingPropertyError``;
- getters that do not modify stored database arrays;
- coefficient-list properties given as one scalar per species (maintainer
  decision on #424, option A): they load without error or warning and every
  evaluation raises, whatever the fractions, including the shipped
  ``cp_solid: 1600`` and ``visc_liq: 1`` workshop placeholders;
- the Henry-constant rule in the Newton iterates of the bubble- and
  dew-point solvers;
- the zero-fraction exemption for Watson latent-heat data in
  ``VaporPhase.getEnthalpy(total_h=True)``, which equals the enthalpy with
  that species removed, while ``total_h=False`` and
  ``getHeatVaporization`` stay strict;
- ``PropertyKeyWarning`` for near-miss keys (``cp_vap``, ``Tc``,
  ``temp_crit``, ``Pc``, ``surface_tension``, case and separator variants),
  silence for metadata keys, and the exact warnings of the shipped
  databases.

All databases are synthetic, rounded test data (water-, ethanol-, nitrogen-
and oxygen-like where noted), not measured properties, written to temporary
JSON files and loaded by real phases; the shipped databases are read as
shipped. Expected values are closed-form evaluations of the fixture data
(Antoine and Henry ratios, additive volumes, constant heat capacities and
the Watson correlation), or the same calculation on a database without the
absent species, which must agree exactly. Core lane, no solver.
"""

import copy
import json
import re
import warnings
from pathlib import Path

import numpy as np
import pytest

from PharmaPy.Phases import LiquidPhase, SolidPhase, VaporPhase
from PharmaPy.Streams import SolidStream
from PharmaPy.ThermoModule import (MissingPropertyError, ParseDatabase,
                                   PropertyKeyWarning, ThermoPhysicalManager)


pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[1]

PRESSURE = 101325.0  # [Pa], standard atmosphere
GRAMS_PER_KILOGRAM = 1000.0  # [g/kg], exact unit conversion
LITERS_PER_CUBIC_METER = 1000.0  # [L/m**3], exact unit conversion
TEMP_REF = 298.15  # [K], default lower limit of the enthalpy integrals
# Watson exponent: K. M. Watson, Ind. Eng. Chem. 1943, 35, 398-406; Poling,
# Prausnitz & O'Connell, The Properties of Gases and Liquids, 5th ed.,
# Eq. (7-11.1).
WATSON_EXPONENT = 0.38  # [-]
# Closed-form expectations differ from PharmaPy's sums only by roundoff.
CLOSED_FORM_RTOL = 1e-12  # [-]

# --------------------------------------------------------------------------
# Liquid fixture, database order solvent, solute, gas, gas_without_henry.
# Units: mw [g/mol], t_crit [K], rho_liq and rho_solid [kg/m**3], Antoine
# p_vap A [-], B [K], C [K] of log10(p/[Pa]) = A - B/(T + C), henry_constant
# [Pa] on a liquid mole-fraction basis, surf_tension [N/m]. "gas" is
# nitrogen-like (Henry constant of data/evaporator/props_nitrogen.json) and
# "gas_without_henry" oxygen-like; both are supercritical at VLE_TEMP.
ANTOINE = {
    "solvent": (10.24, 1595.8, -46.7),
    "solute": (9.0, 3000.0, -50.0),
    "gas": (8.75, 264.5, -6.75),
    "gas_without_henry": (9.0, 340.0, -7.0),
}
GAS_HENRY = 7.0e9  # [Pa]
LIQUID_DATABASE = {
    "solvent": {"mw": 46.0, "t_crit": 514.0, "rho_liq": 789.0,
                "rho_solid": 950.0, "p_vap": list(ANTOINE["solvent"]),
                "surf_tension": 0.022},
    "solute": {"mw": 180.0, "t_crit": 700.0, "rho_liq": 1200.0,
               "rho_solid": 1400.0, "p_vap": list(ANTOINE["solute"]),
               "surf_tension": 0.050},
    "gas": {"mw": 28.0, "t_crit": 126.2, "rho_liq": 806.0,
            "rho_solid": 1030.0, "p_vap": list(ANTOINE["gas"]),
            "henry_constant": GAS_HENRY, "surf_tension": 0.009},
    "gas_without_henry": {"mw": 32.0, "t_crit": 154.6, "rho_liq": 1141.0,
                          "rho_solid": 1350.0,
                          "p_vap": list(ANTOINE["gas_without_henry"]),
                          "surf_tension": 0.013},
}
LIQUID_NAMES = list(LIQUID_DATABASE)
LIQUID_MW = np.array([LIQUID_DATABASE[name]["mw"]
                      for name in LIQUID_NAMES])  # [g/mol]
LIQUID_RHO = np.array([LIQUID_DATABASE[name]["rho_liq"]
                       for name in LIQUID_NAMES])  # [kg/m**3]
LIQUID_TEMP = 300.0  # [K], phase temperature of the density tests
VLE_TEMP = 350.0  # [K], above both gases' t_crit, below the others'
# [K], below the t_crit of gas_without_henry (154.6 K), above the gas's
LOW_TEMP = 140.0
BELOW_BOTH_T_CRIT = 120.0  # [K], below the t_crit of both gases
GAS_PRESENT = np.array([0.5, 0.3, 0.2, 0.0])  # [-], gas_without_henry absent
BOTH_GASES_PRESENT = np.array([0.5, 0.3, 0.1, 0.1])  # [-]
# [-], unequal mass fractions with the solute absent
SOLUTE_ABSENT = np.array([0.6, 0.0, 0.3, 0.1])
SOLUTE_PRESENT = np.array([0.5, 0.1, 0.3, 0.1])  # [-]

# --------------------------------------------------------------------------
# Vapor fixture: water-, ethanol- and nitrogen-like species. Ethanol has
# critical data but no Watson data (delta_hvap, tref_hvap); nitrogen has no
# cp_liq. Units: mw [g/mol], t_crit [K], cp coefficients [J/mol/K] (constant),
# delta_hvap [J/mol], tref_hvap [K].
WATER_MW = 18.0  # [g/mol]
WATER_T_CRIT = 647.0  # [K]
WATER_CP_LIQ = 75.0  # [J/mol/K]
WATER_DELTA_HVAP = 40650.0  # [J/mol]
WATER_TREF_HVAP = 373.15  # [K]
NITROGEN_MW = 28.0  # [g/mol]
NITROGEN_CP_VAPOR = 29.1  # [J/mol/K]
VAPOR_DATABASE = {
    "water": {"mw": WATER_MW, "t_crit": WATER_T_CRIT, "cp_liq": [WATER_CP_LIQ],
              "cp_vapor": [33.6], "delta_hvap": WATER_DELTA_HVAP,
              "tref_hvap": WATER_TREF_HVAP},
    "ethanol": {"mw": 46.0, "t_crit": 514.0, "cp_liq": [112.0],
                "cp_vapor": [65.0]},
    "nitrogen": {"mw": NITROGEN_MW, "t_crit": 126.2,
                 "cp_vapor": [NITROGEN_CP_VAPOR]},
}
VAPOR_TEMPS = np.array([350.0, 365.0])  # [K], water and ethanol subcritical
VAPOR_MOLE_FRAC = np.array([0.1, 0.0, 0.9])  # [-], ethanol absent


def _write(tmp_path, database, name="database.json"):
    """Write a property database and return its path.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Directory for the JSON file.
    database : dict
        Component table in the PharmaPy JSON layout.
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


def _without(database, species, *fields):
    """Copy a database with fields removed from one or every species.

    Parameters
    ----------
    database : dict
        Component table in the PharmaPy JSON layout.
    species : str or None
        Species to strip, or ``None`` for every species.
    *fields : str
        Property keys to remove.

    Returns
    -------
    dict
        Modified deep copy.
    """
    copied = copy.deepcopy(database)
    for name, record in copied.items():
        if species is None or name == species:
            for field in fields:
                record.pop(field, None)
    return copied


def _antoine_ratio(name, temp):
    """Return the closed-form Raoult ratio of a fixture species.

    Parameters
    ----------
    name : str
        Liquid-fixture species.
    temp : float
        Temperature [K].

    Returns
    -------
    float
        ``10**(A - B/(T + C)) / PRESSURE`` [-].
    """
    a_ct, b_ct, c_ct = ANTOINE[name]
    return 10**(a_ct - b_ct / (temp + c_ct)) / PRESSURE


def _liquid(path, mole_frac=None, mass_frac=None):
    """Build a one-mole or one-kilogram liquid phase.

    Parameters
    ----------
    path : str
        Property database.
    mole_frac, mass_frac : array-like, optional
        Composition [-]; exactly one is given.

    Returns
    -------
    LiquidPhase
        Phase at ``LIQUID_TEMP`` and ``PRESSURE``: 1 mol for a molar
        composition, 1 kg for a mass composition.
    """
    if mole_frac is not None:
        return LiquidPhase(path, temp=LIQUID_TEMP, pres=PRESSURE, moles=1.0,
                           mole_frac=mole_frac)
    return LiquidPhase(path, temp=LIQUID_TEMP, pres=PRESSURE, mass=1.0,
                       mass_frac=mass_frac)


# ----------------------------------------------------------- exception type
def test_missing_property_error_is_attribute_and_value_error():
    """Handlers for either earlier exception type still catch it."""
    assert issubclass(MissingPropertyError, AttributeError)
    assert issubclass(MissingPropertyError, ValueError)


# ------------------------------------------------------------ Henry, t_crit
def test_present_supercritical_species_without_henry_constant_raises(
        tmp_path):
    """A supercritical species in the liquid needs its Henry constant."""
    liquid = _liquid(_write(tmp_path, LIQUID_DATABASE),
                     mole_frac=BOTH_GASES_PRESENT)

    with pytest.raises(MissingPropertyError, match=(
            r"^Species \['gas_without_henry'\] carry a nonzero fraction and "
            r"are above t_crit at a requested temperature, so their VLE "
            r"ratio needs Henry constant 'henry_constant' \(\[Pa\], liquid "
            r"mole-fraction basis\)")):
        liquid.getKeqVLE(VLE_TEMP, PRESSURE, BOTH_GASES_PRESENT)


def test_absent_or_subcritical_species_need_no_henry_constant(tmp_path):
    """Rows pair temperature with composition; absent ratios stay NaN.

    Row 0 (350 K) has gas_without_henry absent and supercritical, so its
    ratio is the documented NaN. Row 1 (140 K) has it present but below its
    critical temperature, where it uses Antoine and needs no Henry constant.
    The gas is supercritical in both rows and uses its Henry constant.
    """
    liquid = _liquid(_write(tmp_path, LIQUID_DATABASE), mole_frac=GAS_PRESENT)
    temps = np.array([VLE_TEMP, LOW_TEMP])  # [K]
    x_liq = np.array([GAS_PRESENT, BOTH_GASES_PRESENT])  # [-]
    henry_ratio = GAS_HENRY / PRESSURE  # [-]
    expected = np.array([
        [_antoine_ratio("solvent", VLE_TEMP),
         _antoine_ratio("solute", VLE_TEMP), henry_ratio, np.nan],
        [_antoine_ratio("solvent", LOW_TEMP),
         _antoine_ratio("solute", LOW_TEMP), henry_ratio,
         _antoine_ratio("gas_without_henry", LOW_TEMP)]])  # [-]

    k_rows = liquid.getKeqVLE(temps, PRESSURE, x_liq)  # [-]
    k_single = liquid.getKeqVLE(VLE_TEMP, PRESSURE, GAS_PRESENT)  # [-]

    assert k_rows.shape == (2, len(LIQUID_NAMES))
    np.testing.assert_allclose(k_rows, expected, rtol=CLOSED_FORM_RTOL)
    np.testing.assert_allclose(k_single, expected[0], rtol=CLOSED_FORM_RTOL)


def test_weights_decide_which_ratios_need_henry_data(tmp_path):
    """A caller that never uses a ratio passes it a zero weight.

    The batch evaporator weights by liquid fraction with its excluded
    supercritical species set to zero, while ``x_liq`` still includes them.
    """
    liquid = _liquid(_write(tmp_path, LIQUID_DATABASE),
                     mole_frac=BOTH_GASES_PRESENT)
    used = BOTH_GASES_PRESENT * np.array([1.0, 1.0, 1.0, 0.0])  # [-]

    k_values = liquid.getKeqVLE(VLE_TEMP, PRESSURE, BOTH_GASES_PRESENT,
                                weights=used)  # [-]

    np.testing.assert_allclose(
        k_values[:3], [_antoine_ratio("solvent", VLE_TEMP),
                       _antoine_ratio("solute", VLE_TEMP),
                       GAS_HENRY / PRESSURE], rtol=CLOSED_FORM_RTOL)
    assert np.isnan(k_values[3])
    with pytest.raises(MissingPropertyError, match=r"\['gas_without_henry'\]"):
        liquid.getKeqVLE(VLE_TEMP, PRESSURE, GAS_PRESENT,
                         weights=BOTH_GASES_PRESENT)


@pytest.mark.parametrize("weights", [
    0.0, np.array([0.0]), np.array([1.0, 1.0]),
    np.array([[0.5, 0.3, 0.1, 0.1]])],
    ids=["scalar", "length_one", "wrong_length", "extra_row"])
def test_weights_must_have_the_shape_of_the_composition(tmp_path, weights):
    """Mis-shaped weights are rejected instead of switching checks off.

    A scalar zero once broadcast to every species and silently disabled
    both missing-data checks for the present gas_without_henry.
    """
    liquid = _liquid(_write(tmp_path, LIQUID_DATABASE),
                     mole_frac=BOTH_GASES_PRESENT)

    with pytest.raises(ValueError) as error:
        liquid.getKeqVLE(VLE_TEMP, PRESSURE, BOTH_GASES_PRESENT,
                         weights=weights)

    assert not isinstance(error.value, MissingPropertyError)
    assert str(error.value) == (
        "weights must have the shape of x_liq, (4,), one entry per species "
        "(4 species) and, for a composition profile, one row per row of "
        f"x_liq; got shape {np.shape(weights)}")


def test_database_without_henry_constant_names_supercritical_species(
        tmp_path):
    """No species supplies the key: name every supercritical species.

    Earlier releases raised a bare ``AttributeError`` here, whatever the
    fractions; the trigger is unchanged.
    """
    path = _write(tmp_path, _without(LIQUID_DATABASE, None, "henry_constant"))
    liquid = _liquid(path, mole_frac=GAS_PRESENT)

    with pytest.raises(MissingPropertyError, match=(
            r"^Species \['gas', 'gas_without_henry'\] are above t_crit at a "
            r"requested temperature, so their VLE ratio needs Henry constant "
            r"'henry_constant'")):
        liquid.getKeqVLE(VLE_TEMP, PRESSURE, GAS_PRESENT)

    k_low = liquid.getKeqVLE(BELOW_BOTH_T_CRIT, PRESSURE, GAS_PRESENT)  # [-]
    np.testing.assert_allclose(
        k_low, [_antoine_ratio(name, BELOW_BOTH_T_CRIT)
                for name in LIQUID_NAMES], rtol=CLOSED_FORM_RTOL)


def test_database_without_critical_temperatures_names_every_species(
        tmp_path):
    """A missing ``t_crit`` key is a species-named missing-property error."""
    liquid = _liquid(_write(tmp_path, _without(LIQUID_DATABASE, None,
                                               "t_crit")),
                     mole_frac=GAS_PRESENT)
    vapor_path = _write(tmp_path, _without(VAPOR_DATABASE, None, "t_crit"),
                        "vapor.json")
    vapor = VaporPhase(vapor_path, temp=VAPOR_TEMPS[0], moles=1.0,
                       mole_frac=VAPOR_MOLE_FRAC, check_input=False)
    liquid_names = r"\['solvent', 'solute', 'gas', 'gas_without_henry'\]"
    vapor_names = r"\['water', 'ethanol', 'nitrogen'\]"

    with pytest.raises(MissingPropertyError,
                       match=rf"^Species {liquid_names} are classified.*"
                             r"'t_crit' \[K\]"):
        liquid.getKeqVLE(VLE_TEMP, PRESSURE, GAS_PRESENT)
    with pytest.raises(MissingPropertyError,
                       match=rf"^Species {vapor_names} are classified.*"
                             r"'t_crit' \[K\]"):
        vapor.getEnthalpy(VAPOR_TEMPS[0], basis="mole")
    with pytest.raises(MissingPropertyError, match=rf"^Species {vapor_names}"):
        vapor.getHeatVaporization(VAPOR_TEMPS[0], basis="mole")


# ------------------------------------------------------------------ visc_gas
def test_missing_vapor_viscosity_names_the_species(tmp_path):
    """The earlier message text is kept and the species are named."""
    database = copy.deepcopy(VAPOR_DATABASE)
    database["water"]["visc_gas"] = 1.2e-5  # [Pa*s], synthetic constant
    manager = ThermoPhysicalManager(_write(tmp_path, database))
    message = (r"^Vapor viscosity requires the missing 'visc_gas' property "
               r"\[Pa\*s\] of species \['ethanol', 'nitrogen'\]\. .*JSON")

    with pytest.raises(MissingPropertyError, match=message):
        manager.getViscosityPure(phase="vapor")
    with pytest.raises(MissingPropertyError, match=message):
        manager.getViscosityMix(mole_frac=VAPOR_MOLE_FRAC, phase="vapor")


# ------------------------------------------------------------------- Watson
def _vapor(tmp_path, database, mole_frac=VAPOR_MOLE_FRAC, name="vapor.json"):
    """Build a one-mole vapor phase.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Directory for the JSON file.
    database : dict
        Component table.
    mole_frac : array-like, optional
        Vapor mole fractions [-].
    name : str, optional
        File name.

    Returns
    -------
    VaporPhase
        Phase at the first ``VAPOR_TEMPS`` entry and ``PRESSURE``.
    """
    return VaporPhase(_write(tmp_path, database, name), temp=VAPOR_TEMPS[0],
                      pres=PRESSURE, moles=1.0, mole_frac=mole_frac,
                      check_input=False)


def test_invalid_watson_data_raise_missing_property_error(tmp_path):
    """Strict latent-heat uses report missing Watson data with the class."""
    vapor = _vapor(tmp_path, VAPOR_DATABASE)

    with pytest.raises(MissingPropertyError,
                       match=r"needs finite t_crit.*\['ethanol'\]"):
        vapor.getHeatVaporization(VAPOR_TEMPS[0], basis="mole")
    with pytest.raises(MissingPropertyError,
                       match=r"needs finite t_crit.*\['ethanol'\]"):
        vapor.getEnthalpy(VAPOR_TEMPS[0], total_h=False, basis="mole")
    with pytest.raises(MissingPropertyError,
                       match=r"needs finite t_crit.*\['ethanol'\]"):
        vapor.getEnthalpy(VAPOR_TEMPS[0], basis="mole",
                          mole_frac=np.array([0.1, 0.05, 0.85]))


@pytest.mark.parametrize("temp", [VAPOR_TEMPS[0], VAPOR_TEMPS],
                         ids=["scalar", "array"])
@pytest.mark.parametrize("basis", ["mole", "mass"])
def test_zero_fraction_species_needs_no_watson_data(tmp_path, temp, basis):
    """An absent species adds nothing, exactly as if it were removed.

    Water is subcritical (liquid sensible heat plus Watson latent heat),
    nitrogen supercritical (vapor sensible heat); ethanol is absent.
    """
    vapor = _vapor(tmp_path, VAPOR_DATABASE)
    reduced = {name: VAPOR_DATABASE[name] for name in ("water", "nitrogen")}
    reference = _vapor(tmp_path, reduced, VAPOR_MOLE_FRAC[[0, 2]],
                       "reduced.json")
    temps = np.atleast_1d(temp)  # [K]
    h_water = (WATER_CP_LIQ * (temps - TEMP_REF)
               + WATER_DELTA_HVAP * ((WATER_T_CRIT - temps)
                                     / (WATER_T_CRIT - WATER_TREF_HVAP))
               ** WATSON_EXPONENT)  # [J/mol]
    h_nitrogen = NITROGEN_CP_VAPOR * (temps - TEMP_REF)  # [J/mol]
    mw_present = np.array([WATER_MW, NITROGEN_MW])  # [g/mol]
    y_present = VAPOR_MOLE_FRAC[[0, 2]]  # [-]
    if basis == "mole":
        expected = y_present[0] * h_water + y_present[1] * h_nitrogen  # [J/mol]
    else:
        # Mass fractions and specific enthalpies of one mole of mixture.
        mass_present = y_present * mw_present  # [g]
        w_present = mass_present / mass_present.sum()  # [-]
        expected = (w_present[0] * h_water * GRAMS_PER_KILOGRAM
                    / mw_present[0]
                    + w_present[1] * h_nitrogen * GRAMS_PER_KILOGRAM
                    / mw_present[1])  # [J/kg]

    h_total = vapor.getEnthalpy(temp, mole_frac=VAPOR_MOLE_FRAC,
                                basis=basis)  # [J/mol] or [J/kg]
    h_reference = reference.getEnthalpy(temp, mole_frac=y_present,
                                        basis=basis)  # [J/mol] or [J/kg]

    assert np.shape(h_total) == np.shape(h_reference) == np.shape(temp)
    np.testing.assert_array_equal(h_total, h_reference)
    np.testing.assert_allclose(h_total, expected if np.ndim(temp) else
                               expected[0], rtol=CLOSED_FORM_RTOL)


# ---------------------------------------------------------------- densities
def test_absent_species_needs_no_liquid_density(tmp_path):
    """Volumes and concentrations exclude an absent species without data.

    One kilogram of liquid occupies the sum of the pure-species volumes; the
    concentrations are the species moles over that volume.
    """
    path = _write(tmp_path, _without(LIQUID_DATABASE, "solute", "rho_liq"))
    liquid = _liquid(path, mass_frac=SOLUTE_ABSENT)
    present = SOLUTE_ABSENT > 0  # [-]
    species_mass = SOLUTE_ABSENT  # [kg], of the 1 kg phase
    volume = np.sum(species_mass[present] / LIQUID_RHO[present])  # [m**3]
    moles = species_mass * GRAMS_PER_KILOGRAM / LIQUID_MW  # [mol]
    expected_conc = moles / (volume * LITERS_PER_CUBIC_METER)  # [mol/L]

    assert liquid.vol == pytest.approx(volume, rel=CLOSED_FORM_RTOL)
    assert liquid.getDensity() == pytest.approx(1.0 / volume,
                                                rel=CLOSED_FORM_RTOL)
    np.testing.assert_allclose(liquid.mole_conc, expected_conc,
                               rtol=CLOSED_FORM_RTOL)
    assert liquid.mole_conc[1] == 0.0


def test_present_species_without_liquid_density_raises(tmp_path):
    """A present species without ``rho_liq`` is named, not a NaN volume."""
    path = _write(tmp_path, _without(LIQUID_DATABASE, "solute", "rho_liq"))
    manager = ThermoPhysicalManager(path)
    message = r"^Species \['solute'\] .*'rho_liq' \[kg/m\*\*3\]"

    with pytest.raises(MissingPropertyError, match=message):
        _liquid(path, mass_frac=SOLUTE_PRESENT)
    with pytest.raises(MissingPropertyError,
                       match=r"carry a nonzero fraction, so the mixture "
                             r"density needs pure liquid density 'rho_liq'"):
        manager.getDensityMix(mass_frac=SOLUTE_PRESENT)
    with pytest.raises(MissingPropertyError, match=message):
        manager.getDensityPure()


def test_absent_species_needs_no_solid_density(tmp_path):
    """The solid branch follows the same rule."""
    path = _write(tmp_path, _without(LIQUID_DATABASE, "solute", "rho_solid"))
    manager = ThermoPhysicalManager(path)
    rho_solid = np.array([LIQUID_DATABASE[name]["rho_solid"]
                          for name in LIQUID_NAMES])  # [kg/m**3]
    present = SOLUTE_ABSENT > 0  # [-]
    expected = 1.0 / np.sum(SOLUTE_ABSENT[present]
                            / rho_solid[present])  # [kg/m**3]

    density = manager.getDensityMix(mass_frac=SOLUTE_ABSENT, phase="solid")

    assert density == pytest.approx(expected, rel=CLOSED_FORM_RTOL)
    with pytest.raises(MissingPropertyError,
                       match=r"\['solute'\].*'rho_solid'"):
        manager.getDensityMix(mass_frac=SOLUTE_PRESENT, phase="solid")


def test_solvent_closure_needs_density_of_present_species_only(tmp_path):
    """The solvent concentration closes the additive volume balance.

    Molar volumes are ``mw / rho`` [L/mol]; mass volumes ``1 / rho``
    [m**3/kg]. The absent solute has no density.
    """
    path = _write(tmp_path, _without(LIQUID_DATABASE, "solute", "rho_liq"))
    manager = ThermoPhysicalManager(path)
    mole_conc = np.array([0.0, 0.0, 0.5, 0.25])  # [mol/L], solvent ignored
    mass_conc = np.array([0.0, 0.0, 14.0, 8.0])  # [kg/m**3], solvent ignored
    molar_volume = LIQUID_MW / LIQUID_RHO  # [L/mol], NaN-free for used ones
    solvent_mole = (1 - mole_conc[2:] @ molar_volume[2:]) / molar_volume[0]  # [mol/L]
    solvent_mass = (1 - mass_conc[2:] @ (1 / LIQUID_RHO[2:])) * LIQUID_RHO[0]  # [kg/m**3]
    full_mole = np.array([solvent_mole, 0.0, 0.5, 0.25])  # [mol/L]
    full_mass = np.array([solvent_mass, 0.0, 14.0, 8.0])  # [kg/m**3]

    mole_frac, filled_mole = manager.conc_to_frac(
        mole_conc.copy(), solvent_ind=0, basis="mole")  # [-], [mol/L]
    mass_frac, filled_mass = manager.mass_conc_to_frac(
        mass_conc.copy(), solvent_ind=0, basis="mass")  # [-], [kg/m**3]

    np.testing.assert_allclose(filled_mole, full_mole, rtol=CLOSED_FORM_RTOL)
    np.testing.assert_allclose(mole_frac, full_mole / full_mole.sum(),
                               rtol=CLOSED_FORM_RTOL)
    np.testing.assert_allclose(filled_mass, full_mass, rtol=CLOSED_FORM_RTOL)
    np.testing.assert_allclose(mass_frac, full_mass / full_mass.sum(),
                               rtol=CLOSED_FORM_RTOL)
    with pytest.raises(MissingPropertyError, match=r"\['solute'\].*'rho_liq'"):
        manager.conc_to_frac(np.array([0.0, 0.1, 0.5, 0.25]), solvent_ind=0)


def test_solvent_without_liquid_density_cannot_close_the_balance(tmp_path):
    """The solvent closes the volume balance, so it always needs ``rho_liq``.

    Its input concentration is ignored (zero here), so the presence rule
    alone would not ask for its density; without it the closure would
    divide by a missing molar volume.
    """
    path = _write(tmp_path, _without(LIQUID_DATABASE, "solvent", "rho_liq"))
    manager = ThermoPhysicalManager(path)
    message = r"^Species \['solvent'\] .*'rho_liq' \[kg/m\*\*3\]"

    with pytest.raises(MissingPropertyError, match=message):
        manager.conc_to_frac(np.array([0.0, 0.1, 0.5, 0.25]), solvent_ind=0)
    with pytest.raises(MissingPropertyError, match=message):
        manager.mass_conc_to_frac(np.array([0.0, 18.0, 14.0, 8.0]),
                                  solvent_ind=0)


# --------------------------------------------------------------- solid phase
# Solid fixture: an API-like species with solid data and a solvent-like
# species without rho_solid or cp_solid, declared at zero solid fraction.
# Units: mw [g/mol], rho_solid [kg/m**3], cp_solid [J/mol/K] (constant).
SOLID_API = {"mw": 300.0, "rho_liq": 1200.0, "rho_solid": 1300.0,
             "cp_liq": [300.0], "cp_solid": [250.0]}
SOLID_DATABASE = {"api": SOLID_API,
                  "solvent": {"mw": 46.0, "rho_liq": 789.0, "cp_liq": [110.0]}}
SOLID_SIZES = np.linspace(10.0, 500.0, 20)  # [um], crystal size grid
SOLID_VOLUME_SHARES = np.ones_like(SOLID_SIZES)  # [-], uniform vol_perc bins
SOLID_MASS = 0.5  # [kg] (per second for the stream)
SOLID_TEMP = 300.0  # [K]
# Synthetic solvent solid data: cp_solid [J/mol/K] (constant), rho_solid
# [kg/m**3]; each asymmetric case supplies only one of them.
SOLVENT_SOLID_DATA = {"cp_solid": [120.0], "rho_solid": 950.0}


def _solid(cls, path, mass_frac, **distribution):
    """Build a solid phase or stream at ``SOLID_TEMP``.

    Parameters
    ----------
    cls : type
        ``SolidPhase`` or ``SolidStream``.
    path : str
        Property database.
    mass_frac : array-like
        Solid mass fractions [-].
    **distribution
        Optional ``x_distrib`` [um] and ``distrib`` [-] keywords.

    Returns
    -------
    SolidPhase or SolidStream
        Solid with ``SOLID_MASS`` [kg] or [kg/s].
    """
    amount = {"mass": SOLID_MASS} if cls is SolidPhase else {
        "mass_flow": SOLID_MASS}
    return cls(path, temp=SOLID_TEMP, mass_frac=mass_frac, **amount,
               **distribution)


@pytest.mark.parametrize("cls", [SolidPhase, SolidStream])
@pytest.mark.parametrize("with_distribution", [False, True],
                         ids=["no_distribution", "distribution"])
def test_zero_fraction_solid_species_needs_no_solid_density(
        tmp_path, cls, with_distribution):
    """A declared zero solid fraction is absent for ``rho_solid``.

    The constructor stores the zero as the sentinel ``eps``; the stored
    composition treats it as absent, as ``getCp`` does for ``cp_solid``, so
    the density, the volume of a distribution-based solid and the heat
    capacity equal those of a database without that species.
    """
    distribution = ({"x_distrib": SOLID_SIZES,
                     "distrib": SOLID_VOLUME_SHARES}
                    if with_distribution else {})
    solid = _solid(cls, _write(tmp_path, SOLID_DATABASE),
                   [1.0, 0.0], **distribution)
    reference = _solid(cls, _write(tmp_path, {"api": SOLID_API},
                                   "reduced.json"), [1.0], **distribution)
    cp_mass = (SOLID_API["cp_solid"][0] * GRAMS_PER_KILOGRAM
               / SOLID_API["mw"])  # [J/kg/K]

    assert solid.getDensity() == reference.getDensity()
    assert solid.getDensity() == SOLID_API["rho_solid"]
    assert solid.getCp() == reference.getCp()
    assert solid.getCp() == pytest.approx(cp_mass, rel=CLOSED_FORM_RTOL)
    if with_distribution:
        assert solid.vol == reference.vol
        assert solid.vol == pytest.approx(SOLID_MASS / SOLID_API["rho_solid"],
                                          rel=CLOSED_FORM_RTOL)
    with pytest.raises(MissingPropertyError, match=r"\['solvent'\].*'rho_solid'"):
        solid.getDensity(mass_frac=np.array([0.9, 0.1]))


@pytest.mark.parametrize("present_key, missing_key", [
    ("cp_solid", "rho_solid"), ("rho_solid", "cp_solid")],
    ids=["heat_capacity_only", "density_only"])
def test_solid_sentinel_rule_keys_on_the_evaluated_property(
        tmp_path, present_key, missing_key):
    """Each getter treats the zero-fraction species by its own property.

    The zero-fraction solvent supplies one of ``cp_solid`` and
    ``rho_solid``. The getter of the missing property excludes it exactly
    (equal to the database without it); the getter of the supplied property
    keeps its ``eps`` weight, so it agrees with that database to roundoff.
    """
    database = copy.deepcopy(SOLID_DATABASE)
    database["solvent"][present_key] = SOLVENT_SOLID_DATA[present_key]
    solid = _solid(SolidPhase, _write(tmp_path, database), [1.0, 0.0])
    reference = _solid(SolidPhase, _write(tmp_path, {"api": SOLID_API},
                                          "reduced.json"), [1.0])
    getters = {"cp_solid": "getCp", "rho_solid": "getDensity"}

    excluded = getattr(solid, getters[missing_key])()
    weighted = getattr(solid, getters[present_key])()

    assert excluded == getattr(reference, getters[missing_key])()
    assert weighted == pytest.approx(
        getattr(reference, getters[present_key])(), rel=CLOSED_FORM_RTOL)


@pytest.mark.parametrize("cls", [SolidPhase, SolidStream])
def test_solid_without_distribution_does_not_evaluate_its_density(tmp_path,
                                                                   cls):
    """Mass alone needs no density, so construction does not compute it.

    The solvent is present without ``rho_solid``: the solid builds, as in
    earlier releases, and only a density request reports the missing data.
    """
    database = copy.deepcopy(SOLID_DATABASE)
    database["solvent"]["cp_solid"] = [120.0]  # [J/mol/K], synthetic
    solid = _solid(cls, _write(tmp_path, database), [0.75, 0.25])

    assert not hasattr(solid, "vol")
    with pytest.raises(MissingPropertyError, match=(
            r"^Species \['solvent'\] carry a nonzero fraction, so the "
            r"mixture density needs pure solid density 'rho_solid'")):
        solid.getDensity()


# ----------------------------------------------------------- surface tension
def test_surface_tension_needs_data_of_present_species_only(tmp_path):
    """An absent species without data adds zero; a present one is named."""
    database = _without(LIQUID_DATABASE, "solute", "surf_tension")
    liquid = _liquid(_write(tmp_path, database), mass_frac=SOLUTE_ABSENT)
    # [N/m]; the absent solute's entry is never weighted
    sigma = np.array([database[name].get("surf_tension", 0.0)
                      for name in LIQUID_NAMES])
    second_row = np.array([0.2, 0.0, 0.5, 0.3])  # [-], solute absent

    surface = liquid.getSurfTension()  # [N/m]
    profile = liquid.getSurfTension(
        mass_frac=np.array([SOLUTE_ABSENT, second_row]))  # [N/m]

    assert surface == pytest.approx(SOLUTE_ABSENT @ sigma,
                                    rel=CLOSED_FORM_RTOL)
    assert profile.shape == (2,)
    np.testing.assert_allclose(profile, [SOLUTE_ABSENT @ sigma,
                                         second_row @ sigma],
                               rtol=CLOSED_FORM_RTOL)
    with pytest.raises(MissingPropertyError, match=(
            r"^Species \['solute'\] carry a nonzero mass fraction, so the "
            r"mixture surface tension needs surface tension 'surf_tension' "
            r"\[N/m\]")):
        liquid.getSurfTension(mass_frac=SOLUTE_PRESENT)
    with pytest.raises(MissingPropertyError, match=r"\['solute'\]"):
        liquid.getSurfTensionPure()


def test_surface_tension_getters_leave_stored_data_unchanged(tmp_path):
    """Neither getter writes into the stored ``surf_tension`` array."""
    database = _without(LIQUID_DATABASE, "solute", "surf_tension")
    liquid = _liquid(_write(tmp_path, database), mass_frac=SOLUTE_ABSENT)
    complete = _liquid(_write(tmp_path, LIQUID_DATABASE, "complete.json"),
                       mass_frac=SOLUTE_ABSENT)
    stored = liquid.surf_tension.copy()  # [N/m], NaN for the solute

    liquid.getSurfTension()
    returned = complete.getSurfTensionPure()  # [N/m]
    returned[:] = 0.0  # [N/m], caller-side modification

    np.testing.assert_array_equal(liquid.surf_tension, stored)
    assert np.isnan(liquid.surf_tension[1])
    np.testing.assert_array_equal(
        complete.surf_tension,
        [LIQUID_DATABASE[name]["surf_tension"] for name in LIQUID_NAMES])


# ------------------------------------------- list properties given as scalars
# Each case gives a coefficient-list property as scalars (issue #424, option
# A): (database, property, species that supply scalars, their values,
# evaluation). The species counts are chosen to coincide with valid
# coefficient counts where that hid the error: one species (any
# polynomial), three species (Antoine A, B, C) and four species
# (visc_liq A, B, C, D).
def _scalar_case(prop, values, extra=None):
    """Build a database whose species give ``prop`` as the given values.

    Parameters
    ----------
    prop : str
        Coefficient-list property key.
    values : sequence
        One entry per species: a number, ``{"value": number}``, or ``None``
        to omit the key for that species.
    extra : dict, optional
        Additional keys for every species.

    Returns
    -------
    dict
        Component table; species are named ``s0``, ``s1``, ... with molar
        masses 20, 30, ... [g/mol] and liquid density 1000 kg/m**3.
    """
    database = {}
    for ind, value in enumerate(values):
        record = {"mw": 20.0 + 10.0 * ind, "rho_liq": 1000.0,
                  "t_crit": 600.0}
        record.update(extra or {})
        if value is not None:
            record[prop] = value
        database[f"s{ind}"] = record
    return database


SCALAR_CASES = {
    "cp_liq_single_species": (
        _scalar_case("cp_liq", [75.0]), "cp_liq", ["s0"], [75.0],
        lambda manager: manager.getCpPure(300.0)),
    "visc_liq_four_species": (
        _scalar_case("visc_liq", [1.0, 1.0, 0.5, 2.0]), "visc_liq",
        ["s0", "s1", "s2", "s3"], [1.0, 1.0, 0.5, 2.0],
        lambda manager: manager.getViscosityPure(temp=300.0)),
    "p_vap_three_species": (
        _scalar_case("p_vap", [10.0, 1000.0, -40.0]), "p_vap",
        ["s0", "s1", "s2"], [10.0, 1000.0, -40.0],
        lambda manager: manager.AntoineEquation(300.0)),
    "cp_solid_mixed_with_omitted": (
        _scalar_case("cp_solid", [1600.0, None, 1600.0]), "cp_solid",
        ["s0", "s2"], [1600.0, 1600.0],
        lambda manager: manager.getCpPure(300.0, phase="solid")),
    "cp_vapor_structured": (
        _scalar_case("cp_vapor", [{"value": 29.0}, {"value": 33.0}]),
        "cp_vapor", ["s0", "s1"], [29.0, 33.0],
        lambda manager: manager.getCpPure(300.0, phase="vapor")),
    "diffusivity_two_species": (
        _scalar_case("diffusivity", [1.0e-9, 2.0e-9]), "diffusivity",
        ["s0", "s1"], [1.0e-9, 2.0e-9],
        lambda manager: manager.getDiffusivityPure(0)),
}
SCALAR_UNITS = {
    "cp_liq": "(cp [J/mol/K] = sum_k c_k * T**k)",
    "cp_solid": "(cp [J/mol/K] = sum_k c_k * T**k)",
    "cp_vapor": "(cp [J/mol/K] = sum_k c_k * T**k)",
    "visc_liq": "(log10(mu / [mPa*s]) = A + B/T + C*T + D*T**2)",
    "p_vap": "(log10(p / [Pa]) = A - B/(T + C))",
    "diffusivity": "([m**2/s], one column per reference species)",
}
SCALAR_DESCRIPTIONS = {
    "cp_liq": "liquid heat-capacity coefficients",
    "cp_solid": "solid heat-capacity coefficients",
    "cp_vapor": "vapor heat-capacity coefficients",
    "visc_liq": "liquid-viscosity coefficients",
    "p_vap": "Antoine vapor-pressure coefficients",
    "diffusivity": "diffusivities",
}


def _scalar_message(prop, names, values, path):
    """Return the expected message for a scalar list property.

    Parameters
    ----------
    prop : str
        Property key.
    names : list of str
        Species that supply scalars.
    values : list of float
        Their values, in the property's units as supplied.
    path : str
        Database path.

    Returns
    -------
    str
        Full ``MissingPropertyError`` message.
    """
    return (f"Property '{prop}' ({SCALAR_DESCRIPTIONS[prop]}) of species "
            f"{names} in the property database {path!r} is given as the "
            f"scalar value(s) {values}, but it must be a list of "
            f"coefficients per species {SCALAR_UNITS[prop]}. A scalar is not "
            "read as a constant correlation. Confirm the units and physical "
            "basis of these values and supply the coefficient list that the "
            "correlation requires for each species")


@pytest.mark.parametrize("case", list(SCALAR_CASES))
def test_scalar_list_property_loads_and_is_rejected_when_used(tmp_path, case):
    """Load as one value per species, silently; reject every evaluation."""
    database, prop, names, values, evaluate = SCALAR_CASES[case]
    path = _write(tmp_path, database)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        parsed = ParseDatabase(path)
        manager = ThermoPhysicalManager(path)

    assert parsed[prop].shape == (len(database),)
    with pytest.raises(MissingPropertyError) as error:
        evaluate(manager)
    assert str(error.value) == _scalar_message(prop, names, values, path)
    # The message must not suggest wrapping the unverified literal.
    assert not re.search(r"(e\.g\.|for example|such as) \[",
                         str(error.value))


def test_placeholder_strings_are_listed_by_species(tmp_path):
    """``"N/A"`` is not a number, so ``ParseDatabase`` keeps the strings."""
    strings = _write(tmp_path, _scalar_case("cp_solid", ["N/A", "N/A"]),
                     "strings.json")

    with pytest.raises(MissingPropertyError) as error:
        ThermoPhysicalManager(strings).getCpPure(300.0, phase="solid")
    assert str(error.value) == _scalar_message(
        "cp_solid", ["s0", "s1"], ["N/A", "N/A"], strings)


def test_scalar_column_that_no_species_supplies_is_missing(tmp_path):
    """A JSON ``NaN`` for every species: the property is missing for all."""
    empty = _write(tmp_path, _scalar_case("cp_solid", [np.nan, np.nan]),
                   "empty.json")

    with pytest.raises(MissingPropertyError) as error:
        ThermoPhysicalManager(empty).getCpPure(300.0, phase="solid")
    assert str(error.value) == (
        "Species ['s0', 's1'] give no value for it, so any calculation that "
        "evaluates it needs solid heat-capacity coefficients 'cp_solid' "
        f"{SCALAR_UNITS['cp_solid']}, which the property database "
        f"{empty!r} does not supply for them; add 'cp_solid' for these "
        "species")


def test_scalar_list_property_is_rejected_at_zero_weight(tmp_path):
    """No zero-weight exemption: the property itself is malformed.

    Species s0 supplies the scalar but has zero fraction and weight.
    """
    database = _scalar_case("cp_liq", [75.0, None])
    path = _write(tmp_path, database)
    manager = ThermoPhysicalManager(path)
    absent = np.array([0.0, 1.0])  # [-], s0 absent
    expected = _scalar_message("cp_liq", ["s0"], [75.0], path)

    for evaluate in (
            lambda: manager.getCpMix(300.0, mass_frac=absent),
            lambda: manager.getEnthalpy(300.0, mass_frac=absent),
            lambda: manager.getCpPure(300.0, weights=absent),
            lambda: manager.getEnthalpy(300.0, total_h=False,
                                        weights=absent)):
        with pytest.raises(MissingPropertyError) as error:
            evaluate()
        assert str(error.value) == expected


def test_shipped_scalar_list_properties_are_rejected_when_used():
    """The workshop placeholders load for liquid use and fail when used.

    ``pure_components_ziegler.json`` gives ``cp_solid: 1600`` for its five
    species and ``compounds_mom.json`` ``visc_liq: 1`` for its two. Loading
    emits only the separate near-miss warning for ``cp_vap``.
    """
    ziegler = str(REPO_ROOT / "workshop_data/Case_study_1/"
                  "pure_components_ziegler.json")
    mom = str(REPO_ROOT / "workshop_data/Case_study_2/compounds_mom.json")
    fractions = np.array([0.2, 0.2, 0.2, 0.2, 0.2])  # [-]
    message = _scalar_message("cp_solid", ["A", "B", "C", "D", "solvent"],
                              [1600.0] * 5, ziegler)

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        manager = ThermoPhysicalManager(ziegler)
        liquid = LiquidPhase(ziegler, temp=300.0, mass=1.0,
                             mass_frac=fractions)
        solid = SolidPhase(ziegler, temp=300.0, mass=1.0,
                           mass_frac=fractions)
    assert [item.category for item in record] == [PropertyKeyWarning] * 3
    assert all("'cp_vap'" in str(item.message) for item in record)
    assert np.isfinite(liquid.getDensity())

    for evaluate in (
            lambda: manager.getCpPure(300.0, phase="solid"),
            lambda: manager.getCpMix(300.0, mass_frac=fractions,
                                     phase="solid"),
            lambda: manager.getEnthalpy(310.0, mass_frac=fractions,
                                        phase="solid"),
            solid.getCp, lambda: solid.getEnthalpy(310.0)):
        with pytest.raises(MissingPropertyError) as error:
            evaluate()
        assert str(error.value) == message

    with pytest.warns(PropertyKeyWarning, match="'surface_tension'"):
        mom_liquid = LiquidPhase(mom, temp=300.0, mass=1.0,
                                 mass_frac=np.array([0.3, 0.7]))
    with pytest.raises(MissingPropertyError) as error:
        mom_liquid.getViscosity()
    assert str(error.value) == _scalar_message(
        "visc_liq", ["solute", "solvent"], [1.0, 1.0], mom)


def test_scalar_antoine_data_give_the_named_error_in_vle(tmp_path):
    """Scalar p_vap in a VLE evaluation gives the named error.

    Three species, so the scalars once looked like one Antoine row, and one
    species has a Henry constant. The scalar Antoine data are rejected when
    the Antoine rows are read, before any Henry check, with the named error
    instead of the IndexError of the review fixture.
    """
    database = {
        "A": {"mw": 18.0, "p_vap": 10.0, "t_crit": 647.0},
        "B": {"mw": 28.0, "p_vap": 1000.0, "t_crit": 126.0,
              "henry_constant": 7.0e9},
        "C": {"mw": 32.0, "p_vap": -40.0, "t_crit": 155.0},
    }
    path = _write(tmp_path, database)
    manager = ThermoPhysicalManager(path)
    temps = np.array([300.0, 350.0])  # [K]
    fractions = np.array([0.5, 0.3, 0.2])  # [-]

    with pytest.raises(MissingPropertyError) as error:
        manager.getKeqVLE(temps, PRESSURE, fractions)
    assert str(error.value) == _scalar_message(
        "p_vap", ["A", "B", "C"], [10.0, 1000.0, -40.0], path)


# ------------------------------------------------- Henry data in bubble/dew
def test_bubble_and_dew_points_name_the_missing_henry_constant(tmp_path):
    """A supercritical present species without Henry data is named.

    gas_without_henry (t_crit 154.6 K) is present; every Newton iterate of
    these mixtures lies above its critical temperature, so the iterates,
    not only the converged root, need its Henry constant.
    """
    path = _write(tmp_path, LIQUID_DATABASE)
    liquid = _liquid(path, mole_frac=BOTH_GASES_PRESENT)
    vapor = VaporPhase(path, temp=VLE_TEMP, pres=PRESSURE, moles=1.0,
                       mole_frac=BOTH_GASES_PRESENT, check_input=False)
    message = (r"^Species \['gas_without_henry'\] carry a nonzero fraction "
               r"and are above t_crit at a requested temperature, so their "
               r"VLE ratio needs Henry constant 'henry_constant'")

    with pytest.raises(MissingPropertyError, match=message):
        liquid.getBubblePoint(PRESSURE, mole_frac=BOTH_GASES_PRESENT)
    with pytest.raises(MissingPropertyError, match=message):
        liquid.getBubblePressure(VLE_TEMP, mole_frac=BOTH_GASES_PRESENT)
    with pytest.raises(MissingPropertyError, match=message):
        vapor.getDewPoint(PRESSURE, mole_frac=BOTH_GASES_PRESENT)


# ------------------------------------------------------------ database keys
NEAR_MISS_DATABASE = {
    "first": {"mw": 46.0, "cp_vap": [30.0], "Tc": 514.0, "Formula": "C2H6O",
              "CAS": "64-17-5", "source": "synthetic", "units": "SI"},
    "second": {"mw": 180.0, "temp_crit": 700.0, "Pc": 3.0e6, "cas": "x",
               "formula": "y", "rho_vap": 1.2, "normal_state": "solid",
               "surface_tension": 0.05},
    "third": {"MW": 28.0, "T_crit": 126.2, "kv_solid": 0.5,
              "heat_form_std": -1.0},
}


def test_near_miss_keys_warn_and_metadata_keys_stay_silent(tmp_path):
    """One warning per near-miss key, naming its species and the target.

    The near-miss value is never read as the recognized property: the
    critical temperatures given as ``Tc``, ``temp_crit`` and ``T_crit``
    leave ``t_crit`` unparsed.
    """
    path = _write(tmp_path, NEAR_MISS_DATABASE)
    expected = [  # (near-miss key, species, recognized key), first seen
        ("cp_vap", ["first"], "cp_vapor"),
        ("Tc", ["first"], "t_crit"),
        ("temp_crit", ["second"], "t_crit"),
        ("Pc", ["second"], "p_crit"),
        ("surface_tension", ["second"], "surf_tension"),
        ("MW", ["third"], "mw"),
        ("T_crit", ["third"], "t_crit"),
    ]

    with pytest.warns(PropertyKeyWarning) as record:
        parsed = ParseDatabase(path)

    messages = [str(item.message) for item in record]
    assert all(issubclass(item.category, PropertyKeyWarning)
               for item in record)
    assert messages == [
        f"Property key {key!r} of species {names} in {path!r} is not read "
        f"by PharmaPy and looks like the recognized key {target!r}; its "
        "value is ignored, so a calculation that needs the property reports "
        "it as missing. Rename the key if it holds that property (see "
        "PharmaPy.ThermoModule.RECOGNIZED_PROPERTY_KEYS)."
        for key, names, target in expected]
    assert "t_crit" not in parsed and "cp_vapor" not in parsed


def test_metadata_keys_do_not_warn(tmp_path):
    """Keys that resemble no recognized key are silent metadata."""
    database = {name: {key: value for key, value in record.items()
                       if key in ("mw", "Formula", "CAS", "cas", "formula",
                                  "source", "units", "rho_vap",
                                  "normal_state", "kv_solid",
                                  "heat_form_std")}
                for name, record in NEAR_MISS_DATABASE.items()}
    database["third"]["mw"] = 28.0  # [g/mol]

    with warnings.catch_warnings():
        warnings.simplefilter("error", PropertyKeyWarning)
        parsed = ParseDatabase(_write(tmp_path, database))

    assert parsed["name_species"] == ["first", "second", "third"]


@pytest.mark.parametrize("relative_path, expected", [
    ("data/evaporator/props_nitrogen.json", []),
    ("tests/Flowsheet/data/compound_database.json", []),
    ("tests/integration/data/pfr_test_pure_comp.json", []),
    ("workshop_data/Case_study_1/compounds_ziegler.json", []),
    ("workshop_data/Case_study_3/compound_database.json", []),
    ("workshop_data/Case_study_3/compounds_lomustine.json", []),
    ("workshop_data/Case_study_1/pure_components_ziegler.json",
     [("cp_vap", "cp_vapor")]),
    ("workshop_data/Case_study_1/pure_component_database.json",
     [("Tc", "t_crit"), ("Pc", "p_crit"), ("cp_vap", "cp_vapor")]),
    ("workshop_data/Case_study_2/compounds_mom.json",
     [("surface_tension", "surf_tension")]),
])
def test_shipped_databases_warn_exactly_for_their_near_miss_keys(
        relative_path, expected):
    """Shipped data: the workshop near misses warn, nothing else does."""
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        ParseDatabase(str(REPO_ROOT / relative_path))

    found = [re.match(r"^Property key '([^']+)' .* recognized key '([^']+)'",
                      str(item.message)).groups()
             for item in record
             if issubclass(item.category, PropertyKeyWarning)]
    assert found == expected
