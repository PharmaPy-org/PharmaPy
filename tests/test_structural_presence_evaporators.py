"""Structural species presence in the evaporators and flash units (#425).

A solver-based unit carries every species as a state, so a species that is
neither charged nor fed picks up round-off fractions from solver iterates.
The evaporators and flash units weight the property kernels with a
structural presence mask (``PharmaPy.Presence.SpeciesPresence``) and give
an absent species a unit VLE ratio: an absent species needs no property
data, the unit's result equals the result with arbitrary data supplied for
it, and published phases hold it at exactly zero. A species counts as
present if it is charged, fed with a nonzero effective flow (a feed without
flow, or shut off by the liquid-volume event, adds nothing), or is the
evaporator's nitrogen; the record follows the charge as it is at the first
initialization. Only the evaluated feed counts (a controller's
composition, not the stored inlet's), and a failed solve keeps none of its
latches and restores the attached phases. A charged or fed species without
data still raises the species-named ``MissingPropertyError``; a species fed
later is present from its first feed evaluation on. The module also
covers the kernel ``weights`` paths that the units use: per-entry
exclusion, shape validation, the Henry-constant pre-check and the
bubble-point seed.

Fixture: the two-species evaporator database of ``tests/conftest.py``
(synthetic, rounded) plus a third species ``api`` that is never charged or
fed unless a test says so. ``API_FULL`` gives it arbitrary synthetic data
for every property the units use; ``API_BARE`` gives only its molar mass,
which an absent species still needs (phase and stream constructors compute
mass fractions from it, and a NaN would count as nonzero);
``API_CONSTRUCTIBLE`` adds the liquid density that streams and phases
holding it need at construction; ``API_SUPERCRITICAL`` makes it
supercritical at every drum temperature without a Henry constant;
``API_INFINITE_TSAT`` gives it Antoine data with an infinite saturation
temperature at the drum pressure. The
kernel tests use an asymmetric order (light, api, heavy) so that axis or
ordering mistakes show. Core tests evaluate the flash solves (no ODE
backend) and the evaporator residuals, initialization and result
publication directly; the ``assimulo`` tests integrate the evaporators with
IDA, as in the issue's reproduction (a few seconds each).
"""

import json

import numpy as np
import pytest

from conftest import THERMO_TWO_SPECIES
from PharmaPy.Commons import unpack_states
from PharmaPy.Evaporators import (AdiabaticFlash, ContinuousEvaporator,
                                  Evaporator, IsothermalFlash)
from PharmaPy.Phases import LiquidPhase, VaporPhase
from PharmaPy.Presence import SpeciesPresence
from PharmaPy.ProcessControl import DynamicInput
from PharmaPy.Streams import LiquidStream
from PharmaPy.ThermoModule import MissingPropertyError
from PharmaPy.Utilities import CoolingWater

# Arbitrary synthetic data for the absent species, in the units of the
# database template: mw [g/mol], rho_liq [kg/m**3], t_crit [K], cp
# coefficients [J/mol/K], Antoine A [-], B [K], C [K], delta_hvap [J/mol],
# tref_hvap [K]. None of it may change a result.
API_FULL = {"mw": 300.0, "rho_liq": 1300.0, "t_crit": 900.0,
            "cp_liq": [500.0], "cp_vapor": [400.0],
            "p_vap": [9.0, 4000.0, -50.0], "delta_hvap": 1.0e5,
            "tref_hvap": 500.0}
# [g/mol], no other property data; mw itself is needed for mass fractions
API_BARE = {"mw": 300.0}
# A stream or phase that holds api computes its density at construction,
# so a charged or fed api needs mw [g/mol] and rho_liq [kg/m**3]; it still
# lacks the VLE and enthalpy data the units use.
API_CONSTRUCTIBLE = {"mw": 300.0, "rho_liq": 1300.0}
# As API_CONSTRUCTIBLE plus VLE data, but without cp_liq [J/mol/K].
API_WITHOUT_CP = {"mw": 300.0, "rho_liq": 1300.0, "t_crit": 900.0,
                  "p_vap": [9.0, 4000.0, -50.0], "delta_hvap": 1.0e5,
                  "tref_hvap": 500.0}
# t_crit [K] below every drum and flash temperature here, no Henry constant.
API_SUPERCRITICAL = {"mw": 300.0, "t_crit": 300.0}
# t_crit [K] above the initial bubble point (about 361 K at DRUM_PRESSURE)
# but below CROSSING_TEMP: api becomes supercritical only after the start.
API_CROSSING = {"mw": 300.0, "t_crit": 370.0}
# Antoine A = log10(DRUM_PRESSURE / [Pa]) = 3 makes the saturation
# temperature B / (A - log10 p) - C at DRUM_PRESSURE infinite; otherwise as
# API_FULL (A [-], B [K], C [K]).
API_INFINITE_TSAT = dict(API_FULL, p_vap=[3.0, 4000.0, -50.0])
CROSSING_TEMP = 380.0  # [K], drum temperature of a later residual
# Synthetic Henry constant of the light species, which only makes the
# database carry the henry_constant key: light (t_crit 650 K) stays
# subcritical here, so the value itself is never used.
LIGHT_HENRY = 1.0e9  # [Pa]

CHARGE = np.array([0.4, 0.6, 0.0])  # [-], light, heavy, absent api
WITH_API = np.array([0.4, 0.5, 0.1])  # [-], api charged or fed
DRUM_PRESSURE = 1000.0  # [Pa], conftest evaporator pressure, subcritical
DRUM_TEMP = 350.0  # [K], initial liquid temperature
DRUM_VOLUME = 1.0  # [m**3]
LIQUID_VOLUME = 0.2  # [m**3], partly filled drum
UTILITY_TEMP = 450.0  # [K], heating utility above both bubble points
FEED_FLOW = 2.0  # [mol/s]
REFLUX_RATIO = 0.3  # [-], a partial reflux exercising the bubble-point path
# Flash: at 400 K the light species' Antoine pressure is about 6.8 kPa and
# the heavy one's 1.0 kPa, so a 3 kPa drum splits the feed into two phases.
FLASH_TEMP = 400.0  # [K]
FLASH_PRESSURE = 3000.0  # [Pa]
# Round-off of an absent species, of the size IDA leaves (issue #425 shows
# 1e-25 to 1e-21), injected into states to evaluate the residuals.
ROUND_OFF = 1.0e-20  # [-]
# The flash solves stop at a scaled residual of sqrt(float64 eps), the
# default residual_tolerance of solve_unit, so two solves that reach the
# same root by different iterates agree within that allowance.
FLASH_TOL = np.sqrt(np.finfo(float).eps)  # [-]
# Two databases differ only in data that enters through ROUND_OFF times a
# property of order one relative to the other species' properties, which is
# below one unit in the last place of every term. A computed quantity can
# then differ only by the rounding of the few operations that form it;
# eight units of float64 round-off bound that (assumption: an allowance for
# the handful of additions and products per quantity, not a derived count).
ROUNDING = 8 * np.finfo(float).eps  # [-]
# IDA tolerances of the two integrations, Assimulo's defaults set
# explicitly so that the comparison tolerance is tied to them.
IDA_RTOL = 1.0e-6  # [-]
IDA_ATOL = 1.0e-6  # [units of each state]
# A temperature step off the initial equilibrium, so that the closures are
# nonzero and depend on every species' enthalpy and ratio.
TEMP_STEP = 5.0  # [K]


def _assert_allclose(actual, desired, **tolerances):
    """``numpy.testing.assert_allclose`` in which NaN never matches.

    Parameters
    ----------
    actual, desired : array-like
        Values to compare, in the same units.
    **tolerances
        ``rtol`` [-] and ``atol`` [units of the values].
    """
    np.testing.assert_allclose(actual, desired, equal_nan=False,
                               **tolerances)


def _database(tmp_path, api, name, light_henry=False):
    """Write the evaporator fixture with a third species ``api``.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Directory for the JSON file.
    api : dict or None
        Property record of ``api``; None omits the species.
    name : str
        File name.
    light_henry : bool, optional
        Give the light species ``LIGHT_HENRY`` [Pa].

    Returns
    -------
    str
        Database path.
    """
    database = {key: dict(record)
                for key, record in THERMO_TWO_SPECIES.items()}
    if light_henry:
        database["light"]["henry_constant"] = LIGHT_HENRY
    if api is not None:
        database["api"] = api
    path = tmp_path / name
    path.write_text(json.dumps(database))
    return str(path)


@pytest.fixture
def paths(tmp_path):
    """Return the databases of the fixture records.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Per-test directory.

    Returns
    -------
    dict
        Database paths by record: ``full``, ``bare``, ``constructible``,
        ``no_cp``, ``supercritical`` (no Henry constant in the database),
        ``supercritical_henry_key`` (only light has one), ``crossing``
        (likewise), ``infinite_tsat`` and ``two_species`` (no api).
    """
    return {
        "full": _database(tmp_path, API_FULL, "full.json"),
        "bare": _database(tmp_path, API_BARE, "bare.json"),
        "constructible": _database(tmp_path, API_CONSTRUCTIBLE,
                                   "constructible.json"),
        "no_cp": _database(tmp_path, API_WITHOUT_CP, "no_cp.json"),
        "supercritical": _database(tmp_path, API_SUPERCRITICAL,
                                   "supercritical.json"),
        "supercritical_henry_key": _database(
            tmp_path, API_SUPERCRITICAL, "supercritical_henry_key.json",
            light_henry=True),
        "crossing": _database(tmp_path, API_CROSSING, "crossing.json",
                              light_henry=True),
        "infinite_tsat": _database(tmp_path, API_INFINITE_TSAT,
                                   "infinite_tsat.json"),
        "two_species": _database(tmp_path, None, "two_species.json")}


# ------------------------------------------------------------ the record
@pytest.mark.unit
def test_presence_record_latches_and_shapes_weights():
    """Exact zeros are absent; recorded species stay present."""
    presence = SpeciesPresence(3)
    first = presence.record(np.array([0.4, 0.6, 0.0]))
    second = presence.record(np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1e-30]]))
    weights = presence.weights(np.zeros((2, 3)), used=[True, False, True])

    np.testing.assert_array_equal(first, [True, True, False])
    np.testing.assert_array_equal(second, [True, True, True])
    np.testing.assert_array_equal(weights, [[1.0, 0.0, 1.0]] * 2)
    np.testing.assert_array_equal(
        SpeciesPresence(3).project(np.array([0.5, 0.5, 1e-25])),
        [0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match=r"one entry per species \(3\)"):
        presence.record(np.ones(2))


# ------------------------------------------------------------- flash units
def _flash(cls, path, fractions, seed_basedon_input=True):
    """Solve a flash of a 1 mol/s feed at ``FLASH_TEMP``.

    Parameters
    ----------
    cls : type
        ``IsothermalFlash`` or ``AdiabaticFlash``.
    path : str
        Database path.
    fractions : array-like
        Feed mole fractions [-].
    seed_basedon_input : bool, optional
        Seed option of ``AdiabaticFlash``.

    Returns
    -------
    tuple
        The flash, its solution vector and its heat duty [W] (zero for the
        adiabatic flash).
    """
    feed = LiquidStream(path, temp=FLASH_TEMP, pres=101325.0, mole_flow=1.0,
                        mole_frac=fractions)
    if cls is IsothermalFlash:
        flash = IsothermalFlash(temp_drum=FLASH_TEMP,
                                pres_drum=FLASH_PRESSURE)
        flash.Inlet = feed
        solution, heat = flash.solve_unit()
    else:
        flash = AdiabaticFlash(FLASH_PRESSURE,
                               seed_basedon_input=seed_basedon_input)
        flash.Inlet = feed
        solution, heat = flash.solve_unit(), 0.0
    return flash, solution, heat


def _flash_products(flash):
    """Return the published products of a solved flash.

    Parameters
    ----------
    flash : IsothermalFlash or AdiabaticFlash
        Solved flash.

    Returns
    -------
    numpy.ndarray
        Liquid and vapor flows [mol/s], drum temperature [K], then the
        liquid and vapor mole fractions [-] of light and heavy.
    """
    liquid, vapor = flash.LiquidOut, flash.VaporOut
    return np.concatenate(([liquid.mole_flow, vapor.mole_flow, liquid.temp],
                           liquid.mole_frac[:2], vapor.mole_frac[:2]))


@pytest.mark.unit
@pytest.mark.parametrize("cls", [IsothermalFlash, AdiabaticFlash])
def test_flash_needs_no_data_for_an_absent_species(paths, cls):
    """Bare and full ``api`` data give the same flash; outlets hold zero."""
    full, full_solution, full_heat = _flash(cls, paths["full"], CHARGE)
    bare, bare_solution, bare_heat = _flash(cls, paths["bare"], CHARGE)

    _assert_allclose(bare_solution, full_solution, rtol=FLASH_TOL,
                     atol=FLASH_TOL)
    assert bare_heat == pytest.approx(full_heat, rel=FLASH_TOL)
    for flash in (full, bare):
        assert flash.LiquidOut.mole_frac[2] == 0.0
        assert flash.VaporOut.mole_frac[2] == 0.0
    _assert_allclose(bare.LiquidOut.mole_frac,
                     full.LiquidOut.mole_frac, atol=FLASH_TOL)
    # The two-phase split is real: both outlets carry material.
    assert 0 < bare_solution[0] < 1


@pytest.mark.unit
@pytest.mark.parametrize("cls", [IsothermalFlash, AdiabaticFlash])
@pytest.mark.parametrize("record", ["supercritical",
                                    "supercritical_henry_key"])
def test_flash_ignores_an_absent_supercritical_species(paths, cls, record):
    """An absent api above its t_crit without Henry data changes nothing.

    The reference is the same feed in a database without api. Earlier,
    the isothermal flash raised for a missing Henry constant (or failed to
    converge on the NaN ratio), and the adiabatic flash classified api as
    supercritical and returned a NaN temperature.
    """
    flash, _, heat = _flash(cls, paths[record], CHARGE)
    reference, _, reference_heat = _flash(cls, paths["two_species"],
                                          CHARGE[:2])

    _assert_allclose(_flash_products(flash),
                     _flash_products(reference), rtol=FLASH_TOL,
                     atol=FLASH_TOL)
    assert heat == pytest.approx(reference_heat, rel=FLASH_TOL)
    assert flash.LiquidOut.mole_frac[2] == 0.0
    assert flash.VaporOut.mole_frac[2] == 0.0


@pytest.mark.unit
def test_adiabatic_flash_needs_no_henry_data_of_a_fed_supercritical_species(
        tmp_path):
    """A fed supercritical species' ratio is unused, so is its Henry data.

    Its equilibrium is removed, as before structural presence: the flash
    of a database without its Henry constant equals the flash with an
    arbitrary one. The light species' Henry constant only makes the
    database carry the key.
    """
    # Synthetic record in the units of API_FULL; t_crit < FLASH_TEMP.
    gas = {"mw": 44.0, "rho_liq": 800.0, "t_crit": 300.0,
           "cp_liq": [100.0], "cp_vapor": [40.0],
           "p_vap": [9.0, 1000.0, -10.0], "delta_hvap": 2.0e4,
           "tref_hvap": 250.0}
    gas_henry = 5.0e8  # [Pa], arbitrary
    feed = np.array([0.35, 0.55, 0.10])  # [-], light, heavy, gas
    flashes = []
    for name, record in (("without", gas),
                         ("with", dict(gas, henry_constant=gas_henry))):
        flash, _, _ = _flash(
            AdiabaticFlash, _database(tmp_path, record, f"{name}.json",
                                      light_henry=True), feed,
            seed_basedon_input=False)
        flashes.append(flash)

    without, with_henry = flashes
    _assert_allclose(_flash_products(without),
                     _flash_products(with_henry), rtol=FLASH_TOL,
                     atol=FLASH_TOL)
    # The gas leaves entirely with the vapor.
    assert without.LiquidOut.mole_frac[2] == 0.0
    assert without.VaporOut.mole_frac[2] > 0.0


@pytest.mark.unit
@pytest.mark.parametrize("cls", [IsothermalFlash, AdiabaticFlash])
@pytest.mark.parametrize("record, prop", [("constructible", "p_vap"),
                                          ("no_cp", "cp_liq")])
def test_flash_still_requires_data_of_a_fed_species(paths, cls, record,
                                                    prop):
    """A fed species without data is reported by name and property.

    api passes stream construction, so the error comes from the flash's
    VLE ratio or enthalpy kernel.
    """
    with pytest.raises(MissingPropertyError,
                       match=rf"\['api'\].*'{prop}'"):
        _flash(cls, paths[record], WITH_API, seed_basedon_input=False)


# -------------------------------------------------------------- evaporators
def _evaporator(cls, path, charge=CHARGE, inlet=None, reflux_ratio=0.0,
                include_nitrogen=False):
    """Build a partly filled drum, optionally fed.

    Parameters
    ----------
    cls : type
        ``Evaporator`` or ``ContinuousEvaporator``.
    path : str
        Database path.
    charge : array-like, optional
        Liquid charge mole fractions [-].
    inlet : LiquidStream, optional
        Feed; the continuous evaporator gets a feed of the charge
        composition when none is given.
    reflux_ratio : float, optional
        Reflux ratio [-] of the continuous evaporator.
    include_nitrogen : bool, optional
        Nitrogen option of the batch evaporator.

    Returns
    -------
    Evaporator or ContinuousEvaporator
        Unit with ``LIQUID_VOLUME`` of liquid and a heating utility.
    """
    kwargs = {"pressure": DRUM_PRESSURE}
    if cls is ContinuousEvaporator:
        kwargs["frac_liq"] = LIQUID_VOLUME / DRUM_VOLUME  # [-]
        kwargs["reflux_ratio"] = reflux_ratio  # [-]
    else:
        kwargs["include_nitrogen"] = include_nitrogen
    unit = cls(DRUM_VOLUME, **kwargs)
    unit.Phases = LiquidPhase(path, temp=DRUM_TEMP, pres=DRUM_PRESSURE,
                              vol=LIQUID_VOLUME, mole_frac=charge)
    unit.Utility = CoolingWater(mass_flow=1.0, temp_in=UTILITY_TEMP)
    if inlet is None and cls is ContinuousEvaporator:
        inlet = LiquidStream(path, temp=DRUM_TEMP, pres=DRUM_PRESSURE,
                             mole_flow=FEED_FLOW, mole_frac=charge)
    if inlet is not None:
        unit.Inlet = inlet
    return unit


def _initialize(unit):
    """Initialize a unit as ``solve_unit`` does before integrating.

    Parameters
    ----------
    unit : Evaporator or ContinuousEvaporator
        Unit with phases, utility and (for the continuous one) inlet.

    Returns
    -------
    tuple
        Packed initial states and derivatives (see ``init_unit``).

    Notes
    -----
    ``Evaporator.solve_unit`` classifies the species as supercritical at
    the initial temperature before integrating; this helper repeats that
    step so that the residuals can be evaluated without IDA.
    """
    states, sdot = unit.init_unit()
    if isinstance(unit, Evaporator):
        unit.is_supercritic = states[-1] > unit.Liquid_1.t_crit
    return states, sdot


def _with_round_off(unit, states):
    """Give the absent species round-off fractions, as IDA does.

    Parameters
    ----------
    unit : Evaporator or ContinuousEvaporator
        Unit whose state layout is used.
    states : numpy.ndarray
        Packed initial states.

    Returns
    -------
    numpy.ndarray
        Copy with ``ROUND_OFF`` liquid and vapor fractions of ``api``.
    """
    perturbed = states.copy()
    num = len(unit.name_species)
    perturbed[num + 2] = ROUND_OFF  # [-], x_api
    perturbed[2 * num + 2] = ROUND_OFF  # [-], y_api
    return perturbed


def _residual_atol(unit, states, feed_flow):
    """Absolute rounding allowance of each residual entry.

    Parameters
    ----------
    unit : Evaporator or ContinuousEvaporator
        Initialized unit.
    states : numpy.ndarray
        Packed state at which the residual is evaluated.
    feed_flow : float
        Feed flow [mol/s] of the unit.

    Returns
    -------
    numpy.ndarray
        ``ROUNDING`` times the magnitude of the largest term of each
        closure, in its units: component rates [mol/s] (feed and outlet
        flows), component and total holdups [mol] (the total holdup),
        equilibria [-] (fractions and ``K * x`` of order one), volume
        [m**3] (the drum), pressure [Pa], energy rate [J/s] (heat transfer
        plus flow times the specific internal energy [J/mol]) and internal
        energy [J].
    """
    values = unpack_states(states, unit.dim_states, unit.name_states)
    num = len(unit.name_species)
    holdup = np.sum(values["mol_i"])  # [mol]
    if isinstance(unit, Evaporator):
        (flow_vap, _), heat = unit.unit_model(0.0, states, None, None)
        flow_scale = max(feed_flow, flow_vap)  # [mol/s]
    else:
        (flow_liq, flow_vap, _), (_, heat) = unit.unit_model(
            0.0, states, None, None, enrgy_bce=True)
        flow_scale = max(feed_flow, flow_liq, flow_vap)  # [mol/s]
    specific_energy = abs(values["u_int"]) / holdup  # [J/mol]
    scales = np.concatenate((
        np.full(num, flow_scale),  # [mol/s]
        np.full(num, holdup),  # [mol]
        np.ones(num),  # [-]
        [holdup, DRUM_VOLUME, values["pres"],  # [mol], [m**3], [Pa]
         abs(heat) + flow_scale * specific_energy,  # [J/s]
         abs(values["u_int"])]))  # [J]
    return ROUNDING * scales


def _evaluate(unit, feed_flow):
    """Initialize, evaluate with round-off api and publish a unit.

    Parameters
    ----------
    unit : Evaporator or ContinuousEvaporator
        Unit to evaluate.
    feed_flow : float
        Feed flow [mol/s] of the unit.

    Returns
    -------
    dict
        Initial ``states`` and derivatives ``sdot``, ``residual`` at the
        round-off state and its ``residual_atol``, ``heat_profile`` [J/s],
        the published liquid mole fractions ``published`` [-], and the
        ``continued_states`` and ``continued_sdot`` of a second
        initialization after publication (the continuation path of
        ``Evaporator``, a fresh start from the published liquid for
        ``ContinuousEvaporator``). Units as in ``states_di``.
    """
    states, sdot = _initialize(unit)
    perturbed = _with_round_off(unit, states)
    residual = unit.unit_model(0.0, perturbed, sdot, None)
    atol = _residual_atol(unit, perturbed, feed_flow)
    unit.retrieve_results(np.array([0.0, 1.0]),
                          np.tile(perturbed, (2, 1)))  # [s], states
    published = unit.Liquid_1.mole_frac.copy()  # [-]
    continued_states, continued_sdot = unit.init_unit()
    return {"states": states, "sdot": sdot, "residual": residual,
            "residual_atol": atol, "heat_profile": unit.heat_profile.copy(),
            "published": published, "continued_states": continued_states,
            "continued_sdot": continued_sdot}


def _assert_close_entrywise(actual, desired, atol):
    """Assert ``|actual - desired| <= atol + ROUNDING * |desired|``.

    Parameters
    ----------
    actual, desired : numpy.ndarray
        Residuals, shape (num_entries,), in the units of each entry.
    atol : numpy.ndarray
        Absolute allowance of each entry (see ``_residual_atol``).
    """
    excess = np.abs(actual - desired) - (atol + ROUNDING * np.abs(desired))
    bad = np.flatnonzero(~(excess <= 0))  # NaN fails too
    assert bad.size == 0, (
        f"entries {bad}: actual {actual[bad]}, desired {desired[bad]}, "
        f"atol {atol[bad]}")


def _assert_same_evaluation(result, reference):
    """Compare two ``_evaluate`` results to rounding.

    Parameters
    ----------
    result, reference : dict
        Results of ``_evaluate``.
    """
    _assert_allclose(result["states"], reference["states"],
                     rtol=ROUNDING, atol=0)
    _assert_close_entrywise(result["residual"], reference["residual"],
                            reference["residual_atol"])
    # The derivative seeds sit at the rate rows of the residual layout
    # ([mol/s], [J/s]) and are zero elsewhere.
    for key in ("sdot", "continued_sdot"):
        _assert_close_entrywise(result[key], reference[key],
                                reference["residual_atol"])
    _assert_allclose(result["continued_states"],
                     reference["continued_states"], rtol=ROUNDING, atol=0)
    _assert_allclose(result["heat_profile"],
                     reference["heat_profile"], rtol=ROUNDING,
                     atol=0)
    _assert_allclose(result["published"], reference["published"],
                     rtol=ROUNDING, atol=0)


EVAPORATOR_CASES = [
    pytest.param(Evaporator, 0.0, False, id="Evaporator"),
    pytest.param(Evaporator, 0.0, True, id="Evaporator-nitrogen"),
    pytest.param(ContinuousEvaporator, 0.0, False,
                 id="ContinuousEvaporator"),
    pytest.param(ContinuousEvaporator, REFLUX_RATIO, False,
                 id="ContinuousEvaporator-reflux")]


@pytest.mark.unit
@pytest.mark.parametrize("cls, reflux_ratio, include_nitrogen",
                         EVAPORATOR_CASES)
@pytest.mark.parametrize("record", ["bare", "supercritical",
                                    "supercritical_henry_key",
                                    "infinite_tsat"])
def test_evaporator_needs_no_data_for_an_absent_species(paths, cls,
                                                        reflux_ratio,
                                                        include_nitrogen,
                                                        record):
    """Initialization, residuals and publication ignore an absent species.

    The initial states and derivatives, the residuals at a state with
    round-off ``api`` fractions, the heat duty, the published liquid and a
    second initialization from the published results (with nitrogen, the
    continuation path that rebuilds both phases) equal those of the full
    database. This holds also when api is supercritical without a Henry
    constant (earlier the continuous evaporator reported its NaN ratio,
    and a database without any Henry constant failed in both units), and
    when its Antoine data give an infinite saturation temperature at the
    drum pressure (earlier a NaN bubble-point seed failed the
    initialization).
    """
    feed_flow = FEED_FLOW if cls is ContinuousEvaporator else 0.0  # [mol/s]
    results = {}
    for kind in ("full", record):
        unit = _evaporator(cls, paths[kind], reflux_ratio=reflux_ratio,
                           include_nitrogen=include_nitrogen)
        results[kind] = _evaluate(unit, feed_flow)

    _assert_same_evaluation(results[record], results["full"])
    assert results[record]["published"][2] == 0.0
    states = unpack_states(results[record]["states"], unit.dim_states,
                           unit.name_states)
    assert states["x_liq"][2] == 0.0 and states["y_vap"][2] == 0.0
    if cls is ContinuousEvaporator:
        np.testing.assert_array_equal(
            unit.profiles_runs[-1]["mole_frac"][:, 2], 0.0)


@pytest.mark.unit
@pytest.mark.parametrize("cls, reflux_ratio, include_nitrogen",
                         EVAPORATOR_CASES)
def test_published_vapor_needs_no_data_for_an_absent_species(
        paths, cls, reflux_ratio, include_nitrogen):
    """After a segment, the attached vapor's own methods need no api data.

    The residual is evaluated at, and the segment ends in, a state with
    round-off api, as IDA leaves it. The vapor's enthalpy on both bases,
    without weights, equals that of the full database; earlier the batch
    evaporator left the last iterate in its vapor, and the fraction rule
    then needed api's ``cp_liq``.
    """
    enthalpies = {}
    for kind in ("full", "constructible"):
        unit = _evaporator(cls, paths[kind], reflux_ratio=reflux_ratio,
                           include_nitrogen=include_nitrogen)
        states, sdot = _initialize(unit)
        iterate = _with_round_off(unit, states)
        unit.unit_model(0.0, iterate, sdot, None)
        unit.retrieve_results(np.array([0.0, 1.0]),
                              np.tile(iterate, (2, 1)))  # [s], states
        vapor = unit.Vapor_1
        enthalpies[kind] = np.array([vapor.getEnthalpy(),
                                     vapor.getEnthalpy(basis="mole")])

    _assert_allclose(enthalpies["constructible"], enthalpies["full"],
                     rtol=ROUNDING, atol=0)
    assert vapor.mole_frac[2] == 0.0


@pytest.mark.unit
def test_evaporator_ignores_an_absent_species_turning_supercritical(paths):
    """api crossing its t_crit after the start needs no Henry data.

    The supercritical classification is fixed at the initial temperature,
    where api is subcritical; at ``CROSSING_TEMP`` its ratio would be a
    NaN Henry ratio (earlier making the residual NaN).
    """
    residuals = {}
    for kind in ("full", "crossing"):
        unit = _evaporator(Evaporator, paths[kind])
        states, sdot = _initialize(unit)
        hot = _with_round_off(unit, states)
        hot[-1] = CROSSING_TEMP  # [K]
        residuals[kind] = unit.unit_model(0.0, hot, sdot, None)
        atol = _residual_atol(unit, hot, 0.0)
    assert not unit.is_supercritic[2]

    _assert_close_entrywise(residuals["crossing"], residuals["full"], atol)


@pytest.mark.unit
@pytest.mark.parametrize("cls", [Evaporator, ContinuousEvaporator])
@pytest.mark.parametrize("record, prop", [("constructible", "p_vap"),
                                          ("no_cp", "cp_liq")])
def test_evaporator_still_requires_data_of_a_charged_species(paths, cls,
                                                             record, prop):
    """A charged species without data is reported by name and property.

    api passes phase construction, so the error comes from the unit's
    bubble-point (VLE) or enthalpy kernel.
    """
    unit = _evaporator(cls, paths[record], charge=WITH_API)
    with pytest.raises(MissingPropertyError,
                       match=rf"\['api'\].*'{prop}'"):
        unit.init_unit()


@pytest.mark.unit
@pytest.mark.parametrize("cls", [Evaporator, ContinuousEvaporator])
def test_species_fed_later_is_present_from_its_first_feed(paths, cls):
    """A feed that adds ``api`` at 1 s makes it present from then on.

    Before the feed changes, residuals with round-off ``api`` fractions
    evaluate; once a feed evaluation has supplied ``api``, the same
    residual needs its data, also back at time zero (presence is latched).
    """
    feed = LiquidStream(paths["bare"], temp=DRUM_TEMP, pres=DRUM_PRESSURE,
                        mole_flow=FEED_FLOW, mole_frac=CHARGE)
    control = DynamicInput()
    control.add_variable(
        "mole_frac", lambda time: CHARGE if time < 1.0 else WITH_API)
    feed.DynamicInlet = control
    unit = _evaporator(cls, paths["bare"], inlet=feed)
    states, sdot = _initialize(unit)
    perturbed = _with_round_off(unit, states)

    unit.unit_model(0.5, perturbed, sdot, None)
    with pytest.raises(MissingPropertyError, match=r"\['api'\]"):
        unit.unit_model(2.0, perturbed, sdot, None)
    with pytest.raises(MissingPropertyError, match=r"\['api'\]"):
        unit.unit_model(0.5, perturbed, sdot, None)


@pytest.mark.unit
@pytest.mark.parametrize("cls", [Evaporator, ContinuousEvaporator])
def test_feed_without_flow_adds_no_species(paths, cls):
    """A zero-flow feed that contains api leaves api absent.

    The charge has no api and the inlet carries 10 % api at zero flow, so
    api is never fed. The result equals that of the full database;
    earlier, the inlet composition alone made api present and required its
    VLE and heat-capacity data.
    """
    results = {}
    for kind in ("full", "constructible"):
        inlet = LiquidStream(paths[kind], temp=DRUM_TEMP, pres=DRUM_PRESSURE,
                             mole_flow=0.0, mole_frac=WITH_API)
        unit = _evaporator(cls, paths[kind], inlet=inlet)
        results[kind] = _evaluate(unit, 0.0)

    _assert_same_evaluation(results["constructible"], results["full"])
    assert results["constructible"]["published"][2] == 0.0


@pytest.mark.unit
@pytest.mark.parametrize("cls", [Evaporator, ContinuousEvaporator])
def test_initial_feed_uses_the_evaluated_composition(paths, cls):
    """The evaluated feed, not the stored inlet, decides the initial state.

    The stored inlet carries 10 % api, but its controller always feeds the
    api-free charge composition, so api is never fed. The initial
    derivatives (feed enthalpy) and the evaluation equal those of the full
    database; earlier the continuous evaporator's initial feed enthalpy
    used the stored composition, needed api's ``cp_liq`` and, with data,
    depended on it.
    """
    results = {}
    for kind in ("full", "constructible"):
        inlet = LiquidStream(paths[kind], temp=DRUM_TEMP, pres=DRUM_PRESSURE,
                             mole_flow=FEED_FLOW, mole_frac=WITH_API)
        control = DynamicInput()
        control.add_variable("mole_frac", lambda time: CHARGE)
        inlet.DynamicInlet = control
        unit = _evaporator(cls, paths[kind], inlet=inlet)
        results[kind] = _evaluate(unit, FEED_FLOW)

    _assert_same_evaluation(results["constructible"], results["full"])
    assert results["constructible"]["published"][2] == 0.0


@pytest.mark.unit
@pytest.mark.parametrize("cls", [Evaporator, ContinuousEvaporator])
def test_charge_edited_in_place_is_honored(paths, cls):
    """A charge changed with ``updatePhase`` after assignment counts.

    The first initialization rebuilds the presence from the liquid as it
    is then: with full data the added api is integrated and published at
    its nonzero fraction (earlier silently published as zero); without its
    data the species-named error is raised.
    """
    unit = _evaporator(cls, paths["full"])
    unit.Liquid_1.updatePhase(mole_frac=WITH_API)
    states, _ = _initialize(unit)
    unit.retrieve_results(np.array([0.0, 1.0]), np.tile(states, (2, 1)))

    values = unpack_states(states, unit.dim_states, unit.name_states)
    _assert_allclose(values["x_liq"], WITH_API, rtol=ROUNDING)
    _assert_allclose(unit.Liquid_1.mole_frac, WITH_API,
                     rtol=ROUNDING)

    unit = _evaporator(cls, paths["constructible"])
    unit.Liquid_1.updatePhase(mole_frac=WITH_API)
    with pytest.raises(MissingPropertyError, match=r"\['api'\].*'p_vap'"):
        unit.init_unit()


@pytest.mark.unit
@pytest.mark.parametrize("cls, include_nitrogen", [
    pytest.param(Evaporator, False, id="Evaporator"),
    pytest.param(Evaporator, True, id="Evaporator-nitrogen"),
    pytest.param(ContinuousEvaporator, False, id="ContinuousEvaporator")])
def test_all_present_charge_matches_the_fraction_rule(paths, cls,
                                                      include_nitrogen):
    """With every species present, the weights change nothing.

    The internal energy of ``init_unit`` and, after a temperature step,
    the energy closure and the equilibria of ``unit_model`` equal the
    values of the fraction-rule kernels (no ``weights``). Nitrogen fills
    the headspace, so leaving it out of the presence would drop its vapor
    enthalpy.
    """
    unit = _evaporator(cls, paths["two_species"], charge=CHARGE[:2],
                       include_nitrogen=include_nitrogen)
    states, sdot = _initialize(unit)
    values = unpack_states(states, unit.dim_states, unit.name_states)
    temp, pres = values["temp"], values["pres"]  # [K], [Pa]
    x_liq, y_vap = values["x_liq"], values["y_vap"]  # [-]
    mol_liq, mol_vap = values["mol_liq"], values["mol_vap"]  # [mol]
    if include_nitrogen:
        assert y_vap[-1] > 0.0

    def internal_energy(temp):
        """Fraction-rule internal energy [J] of the initial holdups."""
        h_liq = unit.Liquid_1.getEnthalpy(temp, mole_frac=x_liq,
                                          basis="mole")  # [J/mol]
        h_vap = unit.Vapor_1.getEnthalpy(temp, mole_frac=y_vap,
                                         basis="mole")  # [J/mol]
        return mol_liq * h_liq + mol_vap * h_vap - pres * DRUM_VOLUME

    u_int = values["u_int"]  # [J]
    assert u_int == pytest.approx(internal_energy(temp), rel=ROUNDING)

    stepped = states.copy()
    stepped[-1] = temp + TEMP_STEP  # [K]
    residual = unit.unit_model(0.0, stepped, sdot, None)
    num = len(unit.name_species)
    equilibria = residual[2 * num:3 * num]  # [-]
    ratios = unit.Liquid_1.getKeqVLE(temp + TEMP_STEP, pres, x_liq)  # [-]
    condensable = ~getattr(unit, "is_supercritic", np.zeros(num, bool))
    expected = np.where(condensable, y_vap - ratios * x_liq, x_liq)  # [-]

    _assert_allclose(equilibria, expected, rtol=ROUNDING,
                     atol=ROUNDING)
    assert residual[-1] == pytest.approx(
        internal_energy(temp + TEMP_STEP) - u_int,
        abs=ROUNDING * abs(u_int))


# ------------------------------------------------------- kernel weights
KERNEL_ORDER = ("light", "api", "heavy")
KERNEL_X = np.array([0.3, 0.0, 0.7])  # [-], api absent, asymmetric
KERNEL_X_ROUND_OFF = np.array([0.3, ROUND_OFF, 0.7])  # [-]
KERNEL_WEIGHTS = np.array([1.0, 0.0, 1.0])  # [-], api marked absent
KERNEL_PROFILE = np.array([[0.3, ROUND_OFF, 0.7],
                           [0.8, 0.0, 0.2]])  # [-], two unequal rows
KERNEL_PROFILE_EXACT = np.array([[0.3, 0.0, 0.7], [0.8, 0.0, 0.2]])  # [-]
KERNEL_TEMPS = np.array([340.0, 360.0])  # [K], paired with the rows
KERNEL_TEMP = 350.0  # [K]


def _kernel_database(tmp_path, api, name):
    """Write a database in the asymmetric order light, api, heavy.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Directory for the JSON file.
    api : dict
        Property record of ``api``.
    name : str
        File name.

    Returns
    -------
    str
        Database path.
    """
    records = dict(THERMO_TWO_SPECIES, api=api)
    path = tmp_path / name
    path.write_text(json.dumps({key: records[key] for key in KERNEL_ORDER}))
    return str(path)


@pytest.fixture
def kernel_phases(tmp_path):
    """Return liquid and vapor phases with full and bare ``api`` data.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Per-test directory.

    Returns
    -------
    dict
        ``{('liquid' | 'vapor', 'full' | 'bare'): phase}`` at
        ``KERNEL_TEMP`` [K] and ``DRUM_PRESSURE`` [Pa], composition
        ``KERNEL_X`` [-].
    """
    phases = {}
    for kind, api in (("full", API_FULL), ("bare", API_BARE)):
        path = _kernel_database(tmp_path, api, f"{kind}.json")
        phases["liquid", kind] = LiquidPhase(
            path, temp=KERNEL_TEMP, pres=DRUM_PRESSURE, moles=1.0,
            mole_frac=KERNEL_X)
        phases["vapor", kind] = VaporPhase(
            path, temp=KERNEL_TEMP, pres=DRUM_PRESSURE, moles=1.0,
            mole_frac=KERNEL_X)
    return phases


KERNEL_CALLS = {
    "density": lambda phase, frac, weights: phase.getDensity(
        mole_frac=frac, basis="mole", weights=weights),
    "density_profile": lambda phase, frac, weights: phase.getDensity(
        mole_frac=frac, basis="mole", weights=weights),
    "liquid_enthalpy": lambda phase, frac, weights: phase.getEnthalpy(
        KERNEL_TEMP, mole_frac=frac, basis="mole", weights=weights),
    "liquid_enthalpy_profile": lambda phase, frac, weights:
        phase.getEnthalpy(KERNEL_TEMPS, mole_frac=frac, basis="mole",
                          weights=weights),
    "vapor_enthalpy": lambda phase, frac, weights: phase.getEnthalpy(
        KERNEL_TEMP, mole_frac=frac, basis="mole", weights=weights),
    "vapor_enthalpy_profile": lambda phase, frac, weights:
        phase.getEnthalpy(KERNEL_TEMPS, mole_frac=frac, basis="mole",
                          weights=weights),
    "bubble_point": lambda phase, frac, weights: np.concatenate(
        [np.atleast_1d(value) for value in phase.getBubblePoint(
            pres=DRUM_PRESSURE, mole_frac=frac, y_vap=True,
            weights=weights)]),
}
KERNEL_CASES = [
    ("density", "liquid", False), ("density_profile", "liquid", True),
    ("liquid_enthalpy", "liquid", False),
    ("liquid_enthalpy_profile", "liquid", True),
    ("vapor_enthalpy", "vapor", False),
    ("vapor_enthalpy_profile", "vapor", True),
    ("bubble_point", "liquid", False)]
# Property each kernel needs from a weighted species (api has no t_crit,
# so the vapor enthalpy uses its liquid heat capacity).
KERNEL_PROPERTY = {"density": "rho_liq", "density_profile": "rho_liq",
                   "liquid_enthalpy": "cp_liq",
                   "liquid_enthalpy_profile": "cp_liq",
                   "vapor_enthalpy": "cp_liq",
                   "vapor_enthalpy_profile": "cp_liq",
                   "bubble_point": "p_vap"}


@pytest.mark.unit
@pytest.mark.parametrize("kernel, phase, profile", KERNEL_CASES)
def test_weighted_kernel_equals_the_data_complete_kernel(kernel_phases,
                                                         kernel, phase,
                                                         profile):
    """Zero-weight round-off of a species without data changes nothing.

    The bare database with round-off api and zero api weight gives the
    full database's fraction-rule value with api exactly zero.
    """
    call = KERNEL_CALLS[kernel]
    frac = KERNEL_PROFILE if profile else KERNEL_X_ROUND_OFF  # [-]
    exact = KERNEL_PROFILE_EXACT if profile else KERNEL_X  # [-]
    weights = np.broadcast_to(KERNEL_WEIGHTS, frac.shape).copy()  # [-]

    weighted = call(kernel_phases[phase, "bare"], frac, weights)
    reference = call(kernel_phases[phase, "full"], exact, None)

    _assert_allclose(weighted, reference, rtol=ROUNDING, atol=0)


@pytest.mark.unit
@pytest.mark.parametrize("kernel, phase, profile", KERNEL_CASES)
def test_nonzero_weight_on_a_species_without_data_raises(kernel_phases,
                                                         kernel, phase,
                                                         profile):
    """A weighted species without data is reported, whatever its fraction.

    The fractions hold api at exactly zero, so only the weight requires
    its data.
    """
    frac = KERNEL_PROFILE_EXACT if profile else KERNEL_X  # [-]
    weights = np.ones_like(frac)  # [-]

    with pytest.raises(MissingPropertyError,
                       match=rf"\['api'\].*'{KERNEL_PROPERTY[kernel]}'"):
        KERNEL_CALLS[kernel](kernel_phases[phase, "bare"], frac, weights)


@pytest.mark.unit
@pytest.mark.parametrize("kernel, phase", [
    ("density", "liquid"), ("liquid_enthalpy", "liquid"),
    ("vapor_enthalpy", "vapor"), ("bubble_point", "liquid")])
@pytest.mark.parametrize("weights", [
    0.0, np.array([0.0]), np.array([1.0, 1.0]), np.ones(4),
    np.ones((2, 3))],
    ids=["scalar", "length_one", "short", "long", "extra_row"])
def test_mis_shaped_weights_are_rejected(kernel_phases, kernel, phase,
                                         weights):
    """Weights must have the composition's shape; others are ValueError.

    A scalar zero or a one-entry array once broadcast and silently
    switched the missing-data check off (bare api, 10 % of the liquid,
    gave a density of 15.93 instead of raising); others gave IndexError.
    """
    frac = np.array([0.3, 0.1, 0.6])  # [-], api present without data

    with pytest.raises(ValueError, match="weights must have") as error:
        KERNEL_CALLS[kernel](kernel_phases[phase, "bare"], frac, weights)
    assert not isinstance(error.value, MissingPropertyError)


@pytest.mark.unit
def test_vapor_enthalpy_weights_require_the_mixture_enthalpy(kernel_phases):
    """Species enthalpies (``total_h=False``) take no weights."""
    with pytest.raises(ValueError, match="total_h=True"):
        kernel_phases["vapor", "full"].getEnthalpy(
            KERNEL_TEMP, basis="mole", total_h=False,
            weights=KERNEL_WEIGHTS)


@pytest.mark.unit
def test_vapor_enthalpy_excludes_zero_weight_entries_row_by_row(tmp_path):
    """A species weighted in one row adds nothing in a row without weight.

    Light is used only at 350 K and heavy only at 400 K; each row equals
    its single used species' enthalpy times its fraction, evaluated alone
    (earlier every row also held the other species' sensible heat).
    """
    path = _database(tmp_path, None, "two.json")
    vapor = VaporPhase(path, temp=KERNEL_TEMP, pres=DRUM_PRESSURE, moles=1.0,
                       mole_frac=CHARGE[:2])
    temps = np.array([350.0, 400.0])  # [K]
    fractions = np.tile(CHARGE[:2], (2, 1))  # [-]
    weights = np.array([[1.0, 0.0], [0.0, 1.0]])  # [-]

    mixture = vapor.getEnthalpy(temps, mole_frac=fractions, basis="mole",
                                weights=weights)  # [J/mol]
    light = vapor.getEnthalpy(temps[0], basis="mole", total_h=False)[0, 0]
    heavy = vapor.getEnthalpy(temps[1], basis="mole", total_h=False)[0, 1]

    _assert_allclose(mixture, [CHARGE[0] * light, CHARGE[1] * heavy],
                     rtol=ROUNDING)


@pytest.mark.unit
def test_henry_pre_check_ignores_zero_weight_species(tmp_path):
    """Without any Henry constant, only weighted supercritical species count.

    Unweighted, the earlier rule stays: every supercritical species is
    named. With weights, an absent supercritical api needs nothing and
    gets a NaN placeholder ratio, while a weighted one is still named.
    """
    path = _kernel_database(tmp_path, API_SUPERCRITICAL, "super.json")
    liquid = LiquidPhase(path, temp=KERNEL_TEMP, pres=DRUM_PRESSURE,
                         moles=1.0, mole_frac=KERNEL_X)

    ratios = liquid.getKeqVLE(KERNEL_TEMP, DRUM_PRESSURE, KERNEL_X,
                              weights=KERNEL_WEIGHTS)  # [-]
    assert np.isnan(ratios[1]) and np.all(np.isfinite(ratios[[0, 2]]))
    for weights in (None, np.ones(3)):
        with pytest.raises(MissingPropertyError,
                           match=r"\['api'\].*'henry_constant'"):
            liquid.getKeqVLE(KERNEL_TEMP, DRUM_PRESSURE, KERNEL_X,
                             weights=weights)


@pytest.mark.unit
@pytest.mark.parametrize("kernel", ["density_profile",
                                    "liquid_enthalpy_profile",
                                    "vapor_enthalpy_profile"])
@pytest.mark.parametrize("weights", [
    np.ones(3), np.ones((1, 3)), np.ones((3, 3))],
    ids=["one_dimensional", "one_row", "three_rows"])
def test_profile_weights_must_have_the_profile_shape(kernel_phases, kernel,
                                                     weights):
    """With a two-row profile, weights need exactly two rows.

    The vapor enthalpy once accepted ``(num_species,)`` weights for a
    ``(num_temperatures, num_species)`` profile and vice versa.
    """
    phase = "vapor" if kernel.startswith("vapor") else "liquid"

    with pytest.raises(ValueError, match="weights must have") as error:
        KERNEL_CALLS[kernel](kernel_phases[phase, "full"],
                             KERNEL_PROFILE_EXACT, weights)
    assert not isinstance(error.value, MissingPropertyError)


@pytest.mark.unit
def test_bubble_point_seed_leaves_absent_species_out(tmp_path):
    """Absent api with an infinite saturation temperature changes nothing.

    With ``API_INFINITE_TSAT`` at ``DRUM_PRESSURE``, api's seed term was
    zero times infinity, a NaN seed that Newton could not recover from.
    The weighted bubble point equals that of the two-species database.
    """
    path = _kernel_database(tmp_path, API_INFINITE_TSAT, "infinite.json")
    liquid = LiquidPhase(path, temp=KERNEL_TEMP, pres=DRUM_PRESSURE,
                         moles=1.0, mole_frac=KERNEL_X)
    reference = LiquidPhase(_database(tmp_path, None, "two.json"),
                            temp=KERNEL_TEMP, pres=DRUM_PRESSURE, moles=1.0,
                            mole_frac=KERNEL_X[[0, 2]])

    temp, vapor = liquid.getBubblePoint(pres=DRUM_PRESSURE, mole_frac=KERNEL_X,
                                        y_vap=True, weights=KERNEL_WEIGHTS)
    reference_temp, reference_vapor = reference.getBubblePoint(
        pres=DRUM_PRESSURE, y_vap=True)

    assert temp == pytest.approx(reference_temp, rel=ROUNDING)
    _assert_allclose(vapor[[0, 2]], reference_vapor, rtol=ROUNDING)
    assert vapor[1] == 0.0


@pytest.mark.unit
def test_bubble_point_accepts_lists(kernel_phases):
    """List fractions and weights give the array result.

    Earlier, indexing a list fraction with the weight mask raised
    ``TypeError``.
    """
    liquid = kernel_phases["liquid", "bare"]
    expected = liquid.getBubblePoint(pres=DRUM_PRESSURE, mole_frac=KERNEL_X,
                                     y_vap=True, weights=KERNEL_WEIGHTS)

    result = liquid.getBubblePoint(pres=DRUM_PRESSURE,
                                   mole_frac=KERNEL_X.tolist(), y_vap=True,
                                   weights=KERNEL_WEIGHTS.tolist())

    assert result[0] == pytest.approx(expected[0], rel=ROUNDING)
    _assert_allclose(result[1], expected[1], rtol=ROUNDING, atol=0)


# Two function evaluations cannot converge the steady solve.
TOO_FEW_EVALUATIONS = {"maxfev": 2}


@pytest.mark.unit
def test_failed_steady_solve_is_undone(paths):
    """A failed steady solve leaves the charge and presence as they were.

    The attached liquid and vapor are the same objects with their state
    and derived fields before the solve (``init_unit`` updates the vapor's
    amounts and fractions), and a following initialization equals a fresh
    unit's.
    """
    unit = _evaporator(ContinuousEvaporator, paths["constructible"])
    phases = (unit.Liquid_1, unit.Vapor_1)
    before = [_phase_fingerprint(phase) for phase in phases]
    with pytest.raises(RuntimeError, match="ier="):
        unit.solve_unit(1.0, steady_state=True,
                        fsolve_opts=TOO_FEW_EVALUATIONS)

    assert unit.Liquid_1 is phases[0] and unit.Vapor_1 is phases[1]
    for phase, fingerprint in zip(phases, before):
        _assert_allclose(_phase_fingerprint(phase), fingerprint,
                         rtol=ROUNDING, atol=0)
    states, sdot = unit.init_unit()
    fresh_states, fresh_sdot = _evaporator(
        ContinuousEvaporator, paths["constructible"]).init_unit()
    _assert_allclose(states, fresh_states, rtol=ROUNDING, atol=0)
    _assert_allclose(sdot, fresh_sdot, rtol=ROUNDING, atol=0)


@pytest.mark.unit
def test_vle_messages_name_the_weight(kernel_phases):
    """With weights, the missing-data message says the weight required it.

    The fractions hold api at zero; only its weight makes it needed.
    """
    liquid = kernel_phases["liquid", "bare"]
    with pytest.raises(MissingPropertyError,
                       match=r"^Species \['api'\] carry a nonzero weight "
                             r"and are at or below t_crit.*'p_vap'"):
        liquid.getKeqVLE(KERNEL_TEMP, DRUM_PRESSURE, KERNEL_X,
                         weights=np.ones(3))
    with pytest.raises(MissingPropertyError,
                       match=r"^Species \['api'\] carry a nonzero weight"):
        liquid.getBubblePoint(pres=DRUM_PRESSURE, mole_frac=KERNEL_X,
                              weights=np.ones(3))


@pytest.mark.unit
def test_henry_message_names_the_weight(tmp_path):
    """A weighted supercritical species without Henry data: same wording.

    Light carries ``LIGHT_HENRY`` so that the database has the key and the
    per-species Henry rule, not the database-wide pre-check, applies.
    """
    records = dict(THERMO_TWO_SPECIES, api=API_SUPERCRITICAL)
    records["light"] = dict(records["light"], henry_constant=LIGHT_HENRY)
    path = tmp_path / "henry_key.json"
    path.write_text(json.dumps({key: records[key] for key in KERNEL_ORDER}))
    liquid = LiquidPhase(str(path), temp=KERNEL_TEMP, pres=DRUM_PRESSURE,
                         moles=1.0, mole_frac=KERNEL_X)

    with pytest.raises(MissingPropertyError,
                       match=r"^Species \['api'\] carry a nonzero weight "
                             r"and are above t_crit.*'henry_constant'"):
        liquid.getKeqVLE(KERNEL_TEMP, DRUM_PRESSURE, KERNEL_X,
                         weights=np.ones(3))


# ------------------------------------------------------------ IDA solves
@pytest.mark.assimulo
@pytest.mark.integration
@pytest.mark.parametrize("cls", [Evaporator, ContinuousEvaporator])
def test_evaporator_solve_needs_no_data_for_an_absent_species(paths, cls):
    """The issue's reproduction: the IDA solve is the same with bare data.

    After the segment, the attached vapor's enthalpy (both bases, no
    weights) evaluates without api data and equals the full database's.

    Earlier, the bare database failed at t = 0 with IDAError after a
    missing-``p_vap`` error for ``api``.
    """
    pytest.importorskip("assimulo")
    profiles = {}
    vapor_enthalpies = {}
    vapors = []
    for kind in ("full", "bare"):
        unit = _evaporator(cls, paths[kind])
        time, states = unit.solve_unit(
            1.0, verbose=False,
            sundials_opts={"rtol": IDA_RTOL, "atol": IDA_ATOL})  # [s]
        profiles[kind] = (np.asarray(time), np.asarray(states))
        vapor_enthalpies[kind] = np.array(
            [unit.Vapor_1.getEnthalpy(),
             unit.Vapor_1.getEnthalpy(basis="mole")])  # [J/kg], [J/mol]
        vapors.append((unit.Liquid_1, unit.Vapor_1))

    # The residuals of the two databases agree to rounding, so the
    # integrations agree within IDA's own error control.
    _assert_allclose(profiles["bare"][0], profiles["full"][0],
                     rtol=IDA_RTOL, atol=IDA_ATOL)
    _assert_allclose(profiles["bare"][1], profiles["full"][1],
                     rtol=IDA_RTOL, atol=IDA_ATOL)
    _assert_allclose(vapor_enthalpies["bare"], vapor_enthalpies["full"],
                     rtol=IDA_RTOL)
    for liquid, vapor in vapors:
        assert liquid.mole_frac[2] == 0.0 and vapor.mole_frac[2] == 0.0


@pytest.mark.assimulo
@pytest.mark.integration
def test_feed_switched_after_the_inlet_shut_off_adds_no_species(paths):
    """api offered only after the volume event shut the inlet stays absent.

    The feed fills the drum to the 95 % event (with
    ``stop_at_maxvol=False``, the inlet then shuts) before its composition
    switches to include api at ``switch_time``. Neither the rest of the
    solve nor a continuation needs api data; earlier the evaluated feed
    composition alone made api present.
    """
    pytest.importorskip("assimulo")
    fill_flow = 2000.0  # [mol/s], fills 0.75 m**3 of liquid within seconds
    switch_time = 20.0  # [s], well after the event
    runtime = 30.0  # [s]
    feed = LiquidStream(paths["bare"], temp=DRUM_TEMP, pres=DRUM_PRESSURE,
                        mole_flow=fill_flow, mole_frac=CHARGE)
    control = DynamicInput()
    control.add_variable(
        "mole_frac",
        lambda time: CHARGE if time < switch_time else WITH_API)
    feed.DynamicInlet = control
    unit = Evaporator(DRUM_VOLUME, pressure=DRUM_PRESSURE,
                      stop_at_maxvol=False)
    unit.Phases = LiquidPhase(paths["bare"], temp=DRUM_TEMP,
                              pres=DRUM_PRESSURE, vol=LIQUID_VOLUME,
                              mole_frac=CHARGE)
    unit.Utility = CoolingWater(mass_flow=1.0, temp_in=UTILITY_TEMP)
    unit.Inlet = feed

    unit.solve_unit(runtime, verbose=False)
    assert unit.inlet_stop_time is not None
    assert unit.inlet_stop_time < switch_time
    time, _ = unit.solve_unit(runtime, verbose=False)  # continuation

    assert time[-1] == pytest.approx(2 * runtime)
    assert unit.Liquid_1.mole_frac[2] == 0.0



def _late_api_fractions(time):
    """Feed composition that adds api from ``LATE_API_TIME`` on.

    Parameters
    ----------
    time : float or numpy.ndarray
        Absolute time [s], scalar or shape (num_times,).

    Returns
    -------
    numpy.ndarray
        ``CHARGE`` before and ``WITH_API`` from ``LATE_API_TIME`` [-],
        shape (num_species,) or (num_times, num_species).
    """
    late = np.asarray(time)[..., np.newaxis] >= LATE_API_TIME  # [-]
    return np.where(late, WITH_API, CHARGE)


def _late_api_evaporator(cls, path, include_nitrogen=False):
    """Build a fed unit whose feed adds api from ``LATE_API_TIME`` on.

    Parameters
    ----------
    cls : type
        ``Evaporator`` or ``ContinuousEvaporator``.
    path : str
        Database path; api must be constructible.
    include_nitrogen : bool, optional
        Nitrogen option of the batch evaporator.

    Returns
    -------
    Evaporator or ContinuousEvaporator
        Unit fed ``FEED_FLOW`` [mol/s] of the charge composition, switching
        to ``WITH_API`` at ``LATE_API_TIME`` [s].
    """
    feed = LiquidStream(path, temp=DRUM_TEMP, pres=DRUM_PRESSURE,
                        mole_flow=FEED_FLOW, mole_frac=CHARGE)
    control = DynamicInput()
    control.add_variable("mole_frac", _late_api_fractions)
    feed.DynamicInlet = control
    return _evaporator(cls, path, inlet=feed,
                       include_nitrogen=include_nitrogen)


def _phase_fingerprint(phase):
    """Collect a phase's state and derived fields.

    Parameters
    ----------
    phase : LiquidPhase, VaporPhase or a stream
        Attached phase.

    Returns
    -------
    numpy.ndarray
        Temperature [K], amount [mol], mass [kg], mean molar mass [g/mol],
        enthalpy on the mass [J/kg] and mole [J/mol] bases, then the mole
        and mass fractions [-].

    Notes
    -----
    The enthalpies weight api (index 2) out: it is absent in every case
    that uses this helper, but a phase refreshed by a solve can hold its
    round-off, which the fraction rule would count as present.
    """
    weights = np.ones(len(phase.mole_frac))  # [-]
    weights[2] = 0.0  # [-], api
    return np.concatenate((
        [phase.temp, phase.moles, phase.mass, phase.mw_av,
         phase.getEnthalpy(basis="mass", weights=weights),
         phase.getEnthalpy(basis="mole", weights=weights)],
        phase.mole_frac, phase.mass_frac))


LATE_API_TIME = 0.5  # [s], api enters the feed
SHORT_SEGMENT = 0.1  # [s], ends before LATE_API_TIME
LONG_SEGMENT = 1.0  # [s], reaches LATE_API_TIME and fails without api data


@pytest.mark.assimulo
@pytest.mark.integration
@pytest.mark.parametrize("cls, first_segments, include_nitrogen", [
    pytest.param(Evaporator, 0, False, id="Evaporator-first-segment"),
    pytest.param(Evaporator, 0, True, id="Evaporator-nitrogen-first-segment"),
    pytest.param(Evaporator, 1, False, id="Evaporator-continuation"),
    pytest.param(ContinuousEvaporator, 0, False, id="ContinuousEvaporator")])
def test_failed_solve_is_undone(paths, cls, first_segments,
                                include_nitrogen):
    """A solve that fails leaves presence and phases as they were.

    The long segment latches api once IDA evaluates the feed after
    ``LATE_API_TIME``, and fails for its missing data. The attached phases
    are the same objects with the same state and derived fields (mass
    fractions, molar mass, amounts, enthalpies on both bases); with
    nitrogen, the condensable charge rather than the flash outlets. A
    retried short segment, which never feeds api, then ends like the same
    short segment of a fresh unit. Earlier, api stayed latched, and the
    attached liquid kept the last iterate (with round-off api, from which a
    first segment or the continuous evaporator initializes), so the retry
    failed too.
    """
    sundials = pytest.importorskip("assimulo.solvers.sundials")
    path = paths["constructible"]
    unit = _late_api_evaporator(cls, path, include_nitrogen)
    fresh = _late_api_evaporator(cls, path, include_nitrogen)
    for _ in range(first_segments):
        unit.solve_unit(SHORT_SEGMENT, verbose=False)
        fresh.solve_unit(SHORT_SEGMENT, verbose=False)
    phases = (unit.Liquid_1, unit.Vapor_1)
    before = [_phase_fingerprint(phase) for phase in phases]
    with pytest.raises(sundials.IDAError):
        unit.solve_unit(LONG_SEGMENT, verbose=False)

    assert unit.Liquid_1 is phases[0] and unit.Vapor_1 is phases[1]
    assert len(unit.Liquid_1.mole_frac) == unit.num_species
    for phase, fingerprint in zip(phases, before):
        _assert_allclose(_phase_fingerprint(phase), fingerprint,
                         rtol=ROUNDING, atol=0)
    _, states = unit.solve_unit(SHORT_SEGMENT, verbose=False)
    _, fresh_states = fresh.solve_unit(SHORT_SEGMENT, verbose=False)

    _assert_allclose(np.asarray(states)[-1], np.asarray(fresh_states)[-1],
                     rtol=IDA_RTOL, atol=IDA_ATOL)
    assert unit.Liquid_1.mole_frac[2] == 0.0


@pytest.mark.assimulo
@pytest.mark.integration
def test_failed_continuation_initialization_is_undone(paths):
    """A continuation that fails in ``init_unit`` keeps none of its latches.

    The continuation is fed api, which lacks data, so its initialization
    raises after the feed evaluation latched api. With the feed reverted,
    the next continuation equals a fresh unit's; earlier api stayed
    present and every later continuation demanded its data.
    """
    pytest.importorskip("assimulo")
    path = paths["constructible"]

    def feed(fractions):
        """Return a static feed of ``FEED_FLOW`` [mol/s]."""
        return LiquidStream(path, temp=DRUM_TEMP, pres=DRUM_PRESSURE,
                            mole_flow=FEED_FLOW, mole_frac=fractions)

    unit = _evaporator(Evaporator, path, inlet=feed(CHARGE))
    fresh = _evaporator(Evaporator, path, inlet=feed(CHARGE))
    unit.solve_unit(SHORT_SEGMENT, verbose=False)
    fresh.solve_unit(SHORT_SEGMENT, verbose=False)
    unit.Inlet = feed(WITH_API)
    with pytest.raises(MissingPropertyError, match=r"\['api'\].*'p_vap'"):
        unit.solve_unit(SHORT_SEGMENT, verbose=False)
    unit.Inlet = feed(CHARGE)

    _, states = unit.solve_unit(SHORT_SEGMENT, verbose=False)
    _, fresh_states = fresh.solve_unit(SHORT_SEGMENT, verbose=False)

    _assert_allclose(np.asarray(states)[-1], np.asarray(fresh_states)[-1],
                     rtol=IDA_RTOL, atol=IDA_ATOL)
