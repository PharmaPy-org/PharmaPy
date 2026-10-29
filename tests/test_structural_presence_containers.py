"""Structural species presence in the container units (#425, part B1).

``ContinuousHoldup`` integrates mass fractions, so a difference-quotient
Jacobian (CVode's, or SciPy's ``BDF``) perturbs the fraction of a species
that is neither charged nor fed, and its heat capacity and enthalpy then
needed the species' ``cp_liq``. The holdup now weights those kernels with
a structural presence record (``PharmaPy.Presence.StructuralPresenceMixin``):
an absent species needs no property data other than ``mw``, the result
equals the result with arbitrary data supplied for it, and the published
liquid, outlet and outputs hold it at exactly zero. A charged or fed species
without ``cp_liq`` still raises the species-named ``MissingPropertyError``,
also from ``solve_unit`` before CVode starts; a species fed later is present
from its first feed evaluation on, a feed without flow adds nothing, and a
failed solve keeps none of its latches. The module also covers the
``weights`` keyword that ``getCpMix`` and ``LiquidPhase.getCp`` gained.

``Mixer`` and the liquid ``DynamicCollector`` evaluate their kernels only on
exact inputs (mass-weighted mixes of the inlet compositions, and a
right-hand side that depends on the feed alone), so the fraction rule
already needs no data for an absent species; two pin tests record that.

Fixture: the two-species database of ``tests/conftest.py`` (synthetic,
rounded, constant ``cp_liq``) plus a third species ``api``, in the
asymmetric order light, api, heavy so that axis or ordering mistakes show.
``API_FULL`` gives api arbitrary synthetic data, ``API_BARE`` only its
molar mass (which phase and stream constructors need for mass fractions),
``API_CONSTRUCTIBLE`` adds the liquid density that a phase or stream
holding it needs at construction, without ``cp_liq``. Expected derivatives
are computed independently from the constant heat capacities. Core tests
evaluate the right-hand sides, integrate them with SciPy's ``BDF`` (whose
numerical Jacobian perturbs every state, as CVode's does) and publish
through ``retrieve_results``; the ``assimulo`` tests integrate the
holdup and the collector with CVode (well under a second each).
"""

import json

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from conftest import THERMO_TWO_SPECIES
from test_container_liquid_continuation import (COLD_FRAC, COLD_TEMP,
                                                HOT_FRAC, HOT_TEMP,
                                                expected_mixture,
                                                profile_source)
from PharmaPy.Connections import Connection
from PharmaPy.Containers import ContinuousHoldup, DynamicCollector, Mixer
from PharmaPy.MixedPhases import SlurryStream
from PharmaPy.Phases import LiquidPhase
from PharmaPy.ProcessControl import DynamicInput
from PharmaPy.Streams import LiquidStream, SolidStream
from PharmaPy.ThermoModule import MissingPropertyError

SPECIES_ORDER = ("light", "api", "heavy")
API_INDEX = 1  # [-], position of api in SPECIES_ORDER
# Arbitrary synthetic data for api: mw [g/mol], rho_liq [kg/m**3], cp_liq
# coefficient [J/mol/K]. None of it may change a result where api is absent.
API_FULL = {"mw": 300.0, "rho_liq": 1300.0, "cp_liq": [500.0]}
API_BARE = {"mw": 300.0}  # [g/mol], no other property data
API_CONSTRUCTIBLE = {"mw": 300.0, "rho_liq": 1300.0}  # no cp_liq

CHARGE = np.array([0.4, 0.0, 0.6])  # [-], mass fractions, api absent
FEED = np.array([0.7, 0.0, 0.3])  # [-], mass fractions, api absent
WITH_API = np.array([0.4, 0.1, 0.5])  # [-], mass fractions, api present
CHARGE_TEMP = 300.0  # [K]
FEED_TEMP = 350.0  # [K]
HOLDUP_MASS = 10.0  # [kg], constant inventory
FEED_FLOW = 1.0  # [kg/s], a ten-second residence time with HOLDUP_MASS
# Established default reference temperature of LiquidPhase.getEnthalpy.
TEMP_REF = 298.15  # [K]
# Round-off of an absent species, of the size the solvers leave (issue
# #425 shows 1e-25 to 1e-21), injected into states.
ROUND_OFF = 1.0e-20  # [-]
# A computed derivative may differ from the independent one by the
# rounding of the handful of sums and products that form it, and the full
# database adds ROUND_OFF times an order-one property, far below one unit
# in the last place; eight units of float64 round-off bound that
# (assumption: an allowance for a few operations, not a derived count).
ROUNDING = 8 * np.finfo(float).eps  # [-]
# Integration accuracy of both backends, as in
# tests/test_container_liquid_continuation.py: relative [-] and absolute
# ([-] for fractions, [K] for temperature) local error limits.
ODE_RTOL = 1e-10  # [-]
ODE_ATOL = 1e-11  # [-] or [K]
# Comparison allowance of two integrated trajectories or of a trajectory
# against the closed-form material balance: well above the local error
# limits above, accumulated over a few hundred steps.
TRAJECTORY_RTOL = 1e-5  # [-]
RUN_TIME = 20.0  # [s], two residence times
SWITCH_TIME = 1.0  # [s], when the switched feed starts to carry api
FEED_WINDOW = (1.0, 2.5)  # [s], a feed of api (or a stop) that starts and ends
# A temperature-dependent cp of the light species for the mixer cases, so
# that their energy balance is not linear in temperature (fsolve warns on
# an exactly linear residual): cp = 75 + 0.05 T [J/mol/K], T in [K];
# synthetic, of the order of water's cp near 300 K.
LIGHT_CP_SLOPE = [75.0, 0.05]  # [J/mol/K], [J/mol/K**2]


def _database(tmp_path, api, name):
    """Write the fixture database in the order light, api, heavy.

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
    path.write_text(json.dumps({key: records[key] for key in SPECIES_ORDER}))
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
        Database paths by record: ``full``, ``bare`` and ``constructible``.
    """
    return {"full": _database(tmp_path, API_FULL, "full.json"),
            "bare": _database(tmp_path, API_BARE, "bare.json"),
            "constructible": _database(tmp_path, API_CONSTRUCTIBLE,
                                       "constructible.json")}


def _mass_heat_capacities(api):
    """Return the independent mass-basis heat capacities of the fixture.

    Parameters
    ----------
    api : dict
        Property record of ``api``; without ``cp_liq`` its entry is NaN.

    Returns
    -------
    numpy.ndarray
        Constant heat capacities [J/kg/K] in SPECIES_ORDER: the molar
        coefficient [J/mol/K] times 1000 g/kg over mw [g/mol].
    """
    records = dict(THERMO_TWO_SPECIES, api=api)
    return np.array([records[name]["cp_liq"][0] * 1000 / records[name]["mw"]
                     if "cp_liq" in records[name] else np.nan
                     for name in SPECIES_ORDER])


def _expected_derivatives(fractions, temp, feed_fractions, api):
    """Evaluate the holdup equations independently, without absent species.

    Parameters
    ----------
    fractions : numpy.ndarray
        Holdup mass fractions [-] in SPECIES_ORDER.
    temp : float
        Holdup temperature [K].
    feed_fractions : numpy.ndarray
        Feed mass fractions [-] in SPECIES_ORDER, at FEED_FLOW [kg/s] and
        FEED_TEMP [K].
    api : dict
        Property record of ``api``.

    Returns
    -------
    numpy.ndarray
        dw/dt = F/m (w_in - w) [1/s] and dT/dt = F/(m cp) (h_in - h) [K/s],
        with h = sum_i w_i cp_i (T - TEMP_REF) [J/kg], summed over the
        species with an exactly nonzero charge or feed fraction only.
    """
    used = (CHARGE != 0) | (feed_fractions != 0)  # [-]
    cp_mass = _mass_heat_capacities(api)[used]  # [J/kg/K]
    holdup = fractions[used]  # [-]
    feed = feed_fractions[used]  # [-]
    cp_mix = holdup @ cp_mass  # [J/kg/K]
    h_in = feed @ cp_mass * (FEED_TEMP - TEMP_REF)  # [J/kg]
    h_holdup = holdup @ cp_mass * (temp - TEMP_REF)  # [J/kg]
    dw_dt = FEED_FLOW / HOLDUP_MASS * (feed_fractions - fractions)  # [1/s]
    dtemp_dt = FEED_FLOW / HOLDUP_MASS / cp_mix * (h_in - h_holdup)  # [K/s]
    return np.append(dw_dt, dtemp_dt)


def _holdup(path, charge=CHARGE, feed=FEED, feed_flow=FEED_FLOW,
            feed_path=None):
    """Build a charged and fed ``ContinuousHoldup``.

    Parameters
    ----------
    path : str
        Database of the charge.
    charge, feed : numpy.ndarray, optional
        Charge and feed mass fractions [-] in SPECIES_ORDER.
    feed_flow : float, optional
        Feed mass flow [kg/s].
    feed_path : str, optional
        Database of the feed; ``path`` when None.

    Returns
    -------
    ContinuousHoldup
        HOLDUP_MASS [kg] at CHARGE_TEMP [K], fed at FEED_TEMP [K].
    """
    unit = ContinuousHoldup()
    unit.Phases = LiquidPhase(path, temp=CHARGE_TEMP, mass=HOLDUP_MASS,
                              mass_frac=charge)
    unit.Inlet = LiquidStream(path if feed_path is None else feed_path,
                              temp=FEED_TEMP, mass_flow=feed_flow,
                              mass_frac=feed)
    return unit


def _with_round_off(fractions):
    """Return holdup states with round-off on the absent api.

    Parameters
    ----------
    fractions : numpy.ndarray
        Mass fractions [-] in SPECIES_ORDER with api at exactly zero.

    Returns
    -------
    numpy.ndarray
        Mass fractions [-] with api at ROUND_OFF, followed by CHARGE_TEMP
        [K].
    """
    perturbed = fractions.copy()  # [-]
    perturbed[API_INDEX] = ROUND_OFF
    return np.append(perturbed, CHARGE_TEMP)


# ------------------------------------------------------ heat-capacity kernel
@pytest.mark.unit
@pytest.mark.parametrize("basis", ["mass", "mole"])
@pytest.mark.parametrize("profile", [False, True])
def test_weighted_heat_capacity_equals_the_data_complete_value(
        paths, basis, profile):
    """Zero-weight round-off of a species without cp_liq changes nothing.

    The bare database with round-off api and zero api weight gives the
    independent heat capacity of the mixture without api.
    """
    phase = LiquidPhase(paths["bare"], temp=CHARGE_TEMP, mass=1.0,
                        mass_frac=CHARGE)
    frac = np.array([[0.3, ROUND_OFF, 0.7], [0.8, 0.0, 0.2]])  # [-]
    temps = np.array([340.0, 360.0])  # [K], paired with the rows
    if not profile:
        frac, temps = frac[0], temps[0]
    weights = np.broadcast_to([1.0, 0.0, 1.0], frac.shape).copy()  # [-]
    cp_mass = _mass_heat_capacities(API_FULL)  # [J/kg/K]
    mw = np.array([THERMO_TWO_SPECIES["light"]["mw"], API_FULL["mw"],
                   THERMO_TWO_SPECIES["heavy"]["mw"]])  # [g/mol]
    # Basis of the fractions: the cp per unit of that basis [J/kg/K] or
    # [J/mol/K] (1 kg = 1000 g).
    cp_basis = cp_mass if basis == "mass" else cp_mass * mw / 1000
    kwargs = {"mass_frac" if basis == "mass" else "mole_frac": frac}

    weighted = phase.getCp(temps, basis=basis, weights=weights, **kwargs)
    expected = (np.where(weights != 0, frac, 0.0) * cp_basis).sum(axis=-1)

    np.testing.assert_allclose(weighted, expected, rtol=ROUNDING, atol=0)


@pytest.mark.unit
def test_nonzero_cp_weight_on_a_species_without_data_raises(paths):
    """A weighted species without cp_liq is reported at zero fraction."""
    phase = LiquidPhase(paths["bare"], temp=CHARGE_TEMP, mass=1.0,
                        mass_frac=CHARGE)

    with pytest.raises(MissingPropertyError,
                       match=r"\['api'\] carry a nonzero weight.*'cp_liq'"):
        phase.getCp(CHARGE_TEMP, mass_frac=CHARGE, weights=np.ones(3))


@pytest.mark.unit
@pytest.mark.parametrize("weights", [
    0.0, np.array([0.0]), np.array([1.0, 1.0]), np.ones(4),
    np.ones((2, 3))],
    ids=["scalar", "length_one", "short", "long", "extra_row"])
def test_mis_shaped_cp_weights_are_rejected(paths, weights):
    """Weights must have the composition's shape; others are ValueError."""
    phase = LiquidPhase(paths["constructible"], temp=CHARGE_TEMP, mass=1.0,
                        mass_frac=WITH_API)

    with pytest.raises(ValueError, match="weights must have") as error:
        phase.getCp(CHARGE_TEMP, mass_frac=WITH_API, weights=weights)
    assert not isinstance(error.value, MissingPropertyError)


# ----------------------------------------------------------- ContinuousHoldup
@pytest.mark.unit
@pytest.mark.parametrize("record", ["bare", "full"])
def test_holdup_rhs_needs_no_data_for_an_absent_species(paths, record):
    """Round-off api in the states needs no data and changes nothing.

    Earlier the bare database raised MissingPropertyError for api's
    ``cp_liq`` from the holdup heat capacity.
    """
    unit = _holdup(paths[record])
    states = _with_round_off(CHARGE)

    derivatives = unit.unit_model(0.0, states)

    expected = _expected_derivatives(states[:-1], CHARGE_TEMP, FEED,
                                     API_FULL)
    np.testing.assert_allclose(derivatives, expected, rtol=ROUNDING, atol=0)


@pytest.mark.unit
def test_all_present_holdup_rhs_matches_the_independent_balance(paths):
    """With api charged and fed, its data enter the balance as before."""
    feed = np.array([0.6, 0.2, 0.2])  # [-], api fed
    unit = _holdup(paths["full"], charge=WITH_API, feed=feed)
    temp = 320.0  # [K], off the charge temperature
    derivatives = unit.unit_model(0.0, np.append(WITH_API, temp))

    cp_mass = _mass_heat_capacities(API_FULL)  # [J/kg/K]
    h_in = feed @ cp_mass * (FEED_TEMP - TEMP_REF)  # [J/kg]
    h_holdup = WITH_API @ cp_mass * (temp - TEMP_REF)  # [J/kg]
    expected = np.append(
        FEED_FLOW / HOLDUP_MASS * (feed - WITH_API),
        FEED_FLOW / HOLDUP_MASS / (WITH_API @ cp_mass) * (h_in - h_holdup))
    np.testing.assert_allclose(derivatives, expected, rtol=ROUNDING, atol=0)


@pytest.mark.unit
def test_holdup_integration_needs_no_data_for_an_absent_species(paths):
    """A BDF integration, whose Jacobian perturbs api, needs no api data.

    Both databases give the same trajectory, which follows the closed-form
    material balance w(t) = w_in + (w0 - w_in) exp(-F t / m); the published
    liquid, outlet and outputs hold api at exactly zero.
    """
    results = {}
    for record in ("full", "bare"):
        unit = _holdup(paths[record])
        initial = np.append(CHARGE, CHARGE_TEMP)  # [-], [K]
        solution = solve_ivp(unit.unit_model, (0.0, RUN_TIME), initial,
                             method="BDF", rtol=ODE_RTOL, atol=ODE_ATOL)
        assert solution.success, solution.message
        unit.retrieve_results(solution.t, solution.y.T)
        results[record] = unit

    full, bare = results["full"], results["bare"]
    np.testing.assert_allclose(bare.outputs["temp"][-1],
                               full.outputs["temp"][-1],
                               rtol=TRAJECTORY_RTOL)
    expected = FEED + (CHARGE - FEED) * np.exp(
        -FEED_FLOW / HOLDUP_MASS * RUN_TIME)  # [-]
    np.testing.assert_allclose(bare.Liquid_1.mass_frac, expected,
                               rtol=TRAJECTORY_RTOL)
    assert bare.Liquid_1.mass_frac[API_INDEX] == 0.0
    assert bare.Outlet.mass_frac[API_INDEX] == 0.0
    assert np.all(bare.outputs["mass_frac"][:, API_INDEX] == 0.0)


@pytest.mark.unit
def test_holdup_publishes_an_absent_species_at_exactly_zero(paths):
    """Solver round-off of api is not published.

    The retrieved states carry api at ROUND_OFF; the liquid, outlet and
    outputs hold it at zero. Earlier the published liquid and outlet kept
    the round-off and, without api's ``rho_liq``, their density raised.
    """
    unit = _holdup(paths["bare"])
    states = np.tile(_with_round_off(CHARGE), (2, 1))  # [-], [K]

    unit.retrieve_results(np.array([0.0, 1.0]), states)

    assert unit.Liquid_1.mass_frac[API_INDEX] == 0.0
    assert unit.Outlet.mass_frac[API_INDEX] == 0.0
    assert np.all(unit.outputs["mass_frac"][:, API_INDEX] == 0.0)
    np.testing.assert_allclose(unit.Liquid_1.mass_frac, CHARGE,
                               rtol=ROUNDING)


@pytest.mark.unit
@pytest.mark.parametrize("source", ["charge", "feed"])
def test_holdup_still_requires_data_of_a_present_species(paths, source):
    """A charged or fed api without cp_liq raises by name before CVode.

    ``solve_unit`` evaluates the derivatives at the initial state first, so
    the error is the species-named one (without Assimulo, the solver would
    not even be reached; inside CVode the error would be replaced).
    """
    charge, feed = (WITH_API, FEED) if source == "charge" else (CHARGE,
                                                                WITH_API)
    unit = _holdup(paths["constructible"], charge=charge, feed=feed)

    with pytest.raises(MissingPropertyError, match=r"\['api'\].*'cp_liq'"):
        unit.unit_model(0.0, np.append(charge, CHARGE_TEMP))
    with pytest.raises(MissingPropertyError, match=r"\['api'\].*'cp_liq'"):
        unit.solve_unit(RUN_TIME, verbose=False)


@pytest.mark.unit
def test_charge_edited_in_place_is_honored(paths):
    """A charge changed with ``updatePhase`` after assignment counts.

    ``solve_unit`` rebuilds the presence from the liquid as it is at the
    start, so the added api needs its ``cp_liq``; a record kept from the
    assignment would treat api as absent and silently drop its heat
    capacity.
    """
    unit = _holdup(paths["constructible"])
    unit.Liquid_1.updatePhase(mass_frac=WITH_API)

    with pytest.raises(MissingPropertyError, match=r"\['api'\].*'cp_liq'"):
        unit.solve_unit(RUN_TIME, verbose=False)


def _windowed_feed(unit, inside, outside=FEED, flow=None):
    """Give a unit's inlet a composition that changes inside FEED_WINDOW.

    Parameters
    ----------
    unit : ContinuousHoldup or DynamicCollector
        Unit whose ``Inlet`` gets the controller.
    inside, outside : numpy.ndarray
        Feed mass fractions [-] inside and outside FEED_WINDOW [s].
    flow : tuple of float, optional
        Feed mass flows [kg/s] inside and outside the window; the stream's
        flow when None.
    """
    def pick(time, values):
        """Return ``values[0]`` inside FEED_WINDOW, else ``values[1]``.

        Parameters
        ----------
        time : float or array-like
            Absolute time [s], scalar or shape ``(num_times,)``; the result
            retrieval requests a time array.
        values : tuple
            The value inside and the value outside the window: mass flows
            [kg/s] (scalars) or mass fractions [-], shape
            ``(num_species,)``.

        Returns
        -------
        numpy.ndarray
            One value for a scalar time (0-d for a flow, ``(num_species,)``
            for fractions), or one per time, shape ``(num_times,)`` or
            ``(num_times, num_species)``, in the units of ``values``.
        """
        time = np.asarray(time, dtype=float)  # [s]
        inside = (FEED_WINDOW[0] <= time) & (time < FEED_WINDOW[1])  # [-]
        if np.ndim(values[0]) > 0:
            inside = inside[..., np.newaxis]  # [-], one flag per row
        return np.where(inside, values[0], values[1])

    control = DynamicInput()
    control.add_variable("mass_frac", lambda time: pick(time,
                                                        (inside, outside)))
    if flow is not None:
        control.add_variable("mass_flow", lambda time: pick(time, flow))
    unit.Inlet.DynamicInlet = control


@pytest.mark.unit
def test_species_fed_later_is_present_from_its_first_feed(paths):
    """A feed of api in FEED_WINDOW makes it present from its feed on.

    Before api is fed, derivatives with round-off ``api`` evaluate without
    its data. Once a feed evaluation at 2 s has supplied api, derivatives at
    and after that time need its ``cp_liq``, also at 3 s, after its feed
    stopped (presence is latched); derivatives at 0.5 s, before api was
    fed, still need none and equal the independent balance without api.
    """
    unit = _holdup(paths["bare"])
    _windowed_feed(unit, inside=WITH_API)
    states = _with_round_off(CHARGE)

    unit.unit_model(0.5, states)
    with pytest.raises(MissingPropertyError, match=r"\['api'\].*'cp_liq'"):
        unit.unit_model(2.0, states)
    with pytest.raises(MissingPropertyError,
                       match=r"\['api'\] carry a nonzero weight.*'cp_liq'"):
        unit.unit_model(3.0, states)
    derivatives = unit.unit_model(0.5, states)

    expected = _expected_derivatives(states[:-1], CHARGE_TEMP, FEED,
                                     API_FULL)
    np.testing.assert_allclose(derivatives, expected, rtol=ROUNDING, atol=0)


@pytest.mark.unit
def test_publication_uses_the_presence_of_each_row(paths):
    """Rows before api was fed publish it at zero, later rows keep it.

    api is fed from 2 s on; the retrieved rows at 0.5 s and 3 s carry
    round-off and a real api fraction. The published outputs hold the
    first at exactly zero and the second unchanged; the presence of the
    whole run would have kept the round-off in the earlier row.
    """
    unit = _holdup(paths["full"])
    _windowed_feed(unit, inside=WITH_API)
    unit.unit_model(2.0, np.append(CHARGE, CHARGE_TEMP))
    later = np.array([0.4, 0.05, 0.55])  # [-], api fed in by 3 s
    states = np.vstack((_with_round_off(CHARGE),
                        np.append(later, CHARGE_TEMP)))  # [-], [K]

    unit.retrieve_results(np.array([0.5, 3.0]), states)

    published = unit.outputs["mass_frac"]  # [-]
    assert published[0, API_INDEX] == 0.0
    np.testing.assert_array_equal(published[1], later)
    np.testing.assert_array_equal(unit.Liquid_1.mass_frac, later)


@pytest.mark.unit
@pytest.mark.parametrize("flow", [np.array([FEED_FLOW]), np.array(FEED_FLOW)],
                         ids=["one_entry", "zero_dimensional"])
@pytest.mark.parametrize("controller", [False, True],
                         ids=["static", "controller"])
def test_array_flow_of_one_entry_is_a_scalar_flow(paths, flow, controller):
    """A (1,)-shaped or 0-d feed flow gives the derivatives of a scalar.

    The flow comes from a stream built with it or from a controller
    returning it. Earlier its feed weights took the shape (1, num_species),
    which the feed enthalpy rejected against the (num_species,) feed
    fractions; master integrated it.
    """
    unit = _holdup(paths["bare"], feed_flow=FEED_FLOW if controller else flow)
    if controller:
        control = DynamicInput()
        control.add_variable("mass_flow", lambda time: flow)
        unit.Inlet.DynamicInlet = control
    states = _with_round_off(CHARGE)

    derivatives = unit.unit_model(0.0, states)

    expected = _expected_derivatives(states[:-1], CHARGE_TEMP, FEED,
                                     API_FULL)
    np.testing.assert_allclose(np.concatenate(
        [np.ravel(value) for value in derivatives]), expected,
        rtol=ROUNDING, atol=0)


@pytest.mark.unit
def test_feed_without_flow_adds_no_species(paths):
    """A zero-flow feed that contains api leaves api absent.

    The derivatives are exactly zero and need no api data, also with
    round-off api in the states; earlier the feed composition alone
    required api's ``cp_liq`` for the feed enthalpy.
    """
    with pytest.warns(RuntimeWarning, match="all set to zero"):
        unit = _holdup(paths["constructible"], feed=WITH_API, feed_flow=0.0)
    states = _with_round_off(CHARGE)

    derivatives = unit.unit_model(0.0, states)
    again = unit.unit_model(0.0, states)

    np.testing.assert_array_equal(derivatives, np.zeros(4))
    np.testing.assert_array_equal(again, np.zeros(4))


@pytest.mark.unit
def test_failed_solve_keeps_no_latch_and_leaves_the_liquid(paths):
    """A solve that fails on a fed species keeps none of its latches.

    The feed supplies api without ``cp_liq``, so the initial evaluation
    latches api and raises. After the failure, with an api-free feed
    attached, round-off api again needs no data, and the liquid is
    unchanged. Earlier api stayed latched and the evaluation raised.
    """
    unit = _holdup(paths["constructible"], feed=WITH_API)
    liquid = unit.Liquid_1
    fields = ("mass_frac", "mole_frac", "temp", "mass", "moles", "vol")
    before = {name: np.copy(getattr(liquid, name)) for name in fields}

    with pytest.raises(MissingPropertyError, match=r"\['api'\].*'cp_liq'"):
        unit.solve_unit(RUN_TIME, verbose=False)
    unit.Inlet = LiquidStream(paths["constructible"], temp=FEED_TEMP,
                              mass_flow=FEED_FLOW, mass_frac=FEED)
    derivatives = unit.unit_model(0.0, _with_round_off(CHARGE))

    expected = _expected_derivatives(_with_round_off(CHARGE)[:-1],
                                     CHARGE_TEMP, FEED, API_FULL)
    np.testing.assert_allclose(derivatives, expected, rtol=ROUNDING, atol=0)
    assert unit.Liquid_1 is liquid
    for name, value in before.items():
        np.testing.assert_array_equal(getattr(liquid, name), value,
                                      err_msg=name)


@pytest.mark.assimulo
@pytest.mark.integration
def test_holdup_solve_needs_no_data_for_an_absent_species(paths):
    """CVode, whose Jacobian perturbs api, needs no api data.

    Earlier the bare database failed inside CVode. Both databases give the
    same trajectory, and api is published at exactly zero.
    """
    pytest.importorskip("assimulo")
    options = {"rtol": ODE_RTOL, "atol": ODE_ATOL}
    results = {}
    for record in ("full", "bare"):
        unit = _holdup(paths[record])
        _, states = unit.solve_unit(RUN_TIME, verbose=False,
                                    sundials_opts=options)
        results[record] = (unit, states)

    full_states = results["full"][1]
    bare, bare_states = results["bare"]
    np.testing.assert_allclose(bare_states[-1], full_states[-1],
                               rtol=TRAJECTORY_RTOL)
    expected = FEED + (CHARGE - FEED) * np.exp(
        -FEED_FLOW / HOLDUP_MASS * RUN_TIME)  # [-]
    np.testing.assert_allclose(bare.Liquid_1.mass_frac, expected,
                               rtol=TRAJECTORY_RTOL)
    assert bare.Liquid_1.mass_frac[API_INDEX] == 0.0
    assert bare.Outlet.mass_frac[API_INDEX] == 0.0
    assert np.all(bare.outputs["mass_frac"][:, API_INDEX] == 0.0)


@pytest.mark.assimulo
@pytest.mark.integration
def test_failed_publication_publishes_nothing(tmp_path):
    """A solve whose result retrieval raises keeps no results.

    The charge database gives api ``cp_liq`` but no ``rho_liq``, the feed
    database both: the integration of the fed api succeeds, but publishing
    the liquid that holds it raises. Afterwards the unit is as before the
    call (time, outputs, result, outlet, presence, liquid), and a solve
    with an api-free feed equals a fresh unit's; earlier the outputs and
    result of the failed segment and api's latch stayed. Publication writes
    the liquid last, after the only step that can fail here, so the liquid
    assertion is a guard, not a regression.
    """
    pytest.importorskip("assimulo")
    charge_path = _database(tmp_path, {"mw": 300.0, "cp_liq": [500.0]},
                            "charge.json")
    feed_path = _database(tmp_path, API_FULL, "feed.json")
    unit, fresh = (_holdup(charge_path, feed=WITH_API, feed_path=feed_path)
                   for _ in range(2))
    liquid = unit.Liquid_1
    fields = ("mass_frac", "mole_frac", "temp", "mass", "moles", "vol")
    before = {name: np.copy(getattr(liquid, name)) for name in fields}
    presence_before = unit._presence.mask

    with pytest.raises(MissingPropertyError, match=r"\['api'\].*'rho_liq'"):
        unit.solve_unit(RUN_TIME, verbose=False)

    assert unit.elapsed_time == 0
    for name in ("outputs", "output", "result", "Outlet"):
        assert not hasattr(unit, name), name
    np.testing.assert_array_equal(unit._presence.mask, presence_before)
    assert unit.Liquid_1 is liquid
    for name, value in before.items():
        np.testing.assert_array_equal(getattr(liquid, name), value,
                                      err_msg=name)
    for each in (unit, fresh):
        each.Inlet = LiquidStream(charge_path, temp=FEED_TEMP,
                                  mass_flow=FEED_FLOW, mass_frac=FEED)
    _, states = unit.solve_unit(RUN_TIME, verbose=False)
    _, fresh_states = fresh.solve_unit(RUN_TIME, verbose=False)
    np.testing.assert_array_equal(states[-1], fresh_states[-1])


# ------------------------------------ units that need no structural presence
@pytest.mark.unit
def test_mixer_needs_no_data_for_an_absent_species(paths):
    """Mixed fractions of an absent species are exact zeros.

    The mixed composition is a mass-weighted sum of exact inlet fractions,
    and only the outlet temperature is iterated, so the fraction rule
    already needs no api data (no structural presence is needed).
    """
    results = {}
    for record in ("full", "bare"):
        mixer = Mixer()
        mixer.Inlets = [
            LiquidPhase(paths[record], temp=CHARGE_TEMP, mass=HOLDUP_MASS,
                        mass_frac=CHARGE),
            LiquidPhase(paths[record], temp=FEED_TEMP,
                        mass=2 * HOLDUP_MASS, mass_frac=FEED)]
        mixer.solve_unit()
        results[record] = mixer.Outlet

    bare, full = results["bare"], results["full"]
    # Independent balance over the species present: constant cp makes the
    # enthalpy linear in temperature, so the mixed temperature is the
    # heat-capacity-weighted mean of the inlet temperatures [K].
    cp_mass = _mass_heat_capacities(API_BARE)[[0, 2]]  # [J/kg/K]
    capacities = HOLDUP_MASS * np.array(
        [CHARGE[[0, 2]] @ cp_mass, 2 * FEED[[0, 2]] @ cp_mass])  # [J/K]
    expected_temp = (capacities @ [CHARGE_TEMP, FEED_TEMP]
                     / capacities.sum())  # [K]
    assert bare.mass_frac[API_INDEX] == 0.0
    np.testing.assert_allclose(bare.mass_frac, (CHARGE + 2 * FEED) / 3,
                               rtol=ROUNDING, atol=0)
    np.testing.assert_allclose(bare.mass_frac, full.mass_frac,
                               rtol=ROUNDING, atol=0)
    assert bare.temp == pytest.approx(expected_temp, rel=ROUNDING)
    assert full.temp == pytest.approx(expected_temp, rel=ROUNDING)


@pytest.mark.assimulo
@pytest.mark.integration
def test_liquid_collector_needs_no_data_for_an_absent_species(paths):
    """The liquid collector's states never carry round-off of api.

    Its right-hand side depends on the feed only, so an absent species'
    inventory stays exactly zero and its temperature inversion runs on
    exact zeros: no structural presence is needed (issue #467 made the
    collector integrate inventories).
    """
    pytest.importorskip("assimulo")
    results = {}
    for record in ("full", "bare"):
        collector = DynamicCollector()
        collector.Inlet = LiquidStream(paths[record], temp=FEED_TEMP,
                                       mass_flow=FEED_FLOW, mass_frac=FEED)
        _, states = collector.solve_unit(RUN_TIME, verbose=False)
        results[record] = (collector, states)

    full_states = results["full"][1]
    bare, bare_states = results["bare"]
    assert np.all(bare_states[:, API_INDEX] == 0.0)
    np.testing.assert_allclose(bare_states, full_states, rtol=ROUNDING,
                               atol=0)
    assert bare.Outlet.mass_frac[API_INDEX] == 0.0


def _mixer_database(tmp_path, api, name):
    """Write the fixture database with LIGHT_CP_SLOPE for the light species.

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
    records["light"] = dict(records["light"], cp_liq=LIGHT_CP_SLOPE)
    path = tmp_path / name
    path.write_text(json.dumps({key: records[key] for key in SPECIES_ORDER}))
    return str(path)


def _expected_mix_temperature(amounts):
    """Solve the adiabatic mix of CHARGE and FEED independently.

    Parameters
    ----------
    amounts : sequence of float
        Masses [kg] or mass flows [kg/s] of the CHARGE_TEMP charge
        composition and the FEED_TEMP feed composition.

    Returns
    -------
    float
        Outlet temperature [K] at which the mixture's sensible enthalpy
        equals the inlets', with the cp of LIGHT_CP_SLOPE and the heavy
        species' constant cp (api absent).
    """
    from scipy.optimize import brentq

    mw = {name: THERMO_TWO_SPECIES[name]["mw"] for name in ("light", "heavy")}

    def enthalpy(fractions, temp):
        """Return the mixture's sensible enthalpy without api.

        Parameters
        ----------
        fractions : numpy.ndarray
            Mass fractions [-] in SPECIES_ORDER; api's entry is not used.
        temp : float
            Temperature [K].

        Returns
        -------
        float
            Mass-basis enthalpy [J/kg] relative to TEMP_REF [K], which
            cancels in the balance.
        """
        light = (LIGHT_CP_SLOPE[0] * (temp - TEMP_REF)
                 + LIGHT_CP_SLOPE[1] / 2 * (temp**2 - TEMP_REF**2))  # [J/mol]
        heavy = THERMO_TWO_SPECIES["heavy"]["cp_liq"][0] * (
            temp - TEMP_REF)  # [J/mol]
        # 1000 g/kg over mw [g/mol] converts J/mol to J/kg.
        return 1000 * (fractions[0] * light / mw["light"]
                       + fractions[2] * heavy / mw["heavy"])

    total = sum(amounts)  # [kg] or [kg/s]
    mixed = (amounts[0] * CHARGE + amounts[1] * FEED) / total  # [-]
    energy = (amounts[0] * enthalpy(CHARGE, CHARGE_TEMP)
              + amounts[1] * enthalpy(FEED, FEED_TEMP))  # [J] or [J/s]
    return brentq(lambda temp: total * enthalpy(mixed, temp) - energy,
                  CHARGE_TEMP, FEED_TEMP)


@pytest.mark.unit
@pytest.mark.parametrize("continuous", [False, True],
                         ids=["batch", "continuous"])
def test_mixer_inlet_without_flow_needs_no_data(tmp_path, continuous):
    """An inlet without mass or flow adds nothing and needs no data.

    A third inlet holds 10 % api, which has no ``cp_liq``, at zero mass
    (batch) or zero flow (continuous). The mix equals the independent mix
    of the other two inlets; earlier the inlet's enthalpy required api's
    ``cp_liq``.
    """
    path = _mixer_database(tmp_path, API_CONSTRUCTIBLE, "mixer.json")
    amounts = (HOLDUP_MASS, 2 * HOLDUP_MASS)  # [kg] or, read so, [kg/s]
    cls, key = ((LiquidStream, "mass_flow") if continuous
                else (LiquidPhase, "mass"))
    mixer = Mixer()
    with pytest.warns(RuntimeWarning, match="all set to zero"):
        empty = cls(path, temp=FEED_TEMP, mass_frac=WITH_API, **{key: 0.0})
    mixer.Inlets = [cls(path, temp=CHARGE_TEMP, mass_frac=CHARGE,
                        **{key: amounts[0]}),
                    cls(path, temp=FEED_TEMP, mass_frac=FEED,
                        **{key: amounts[1]}),
                    empty]

    mixer.solve_unit()

    outlet = mixer.Outlet
    expected = (amounts[0] * CHARGE + amounts[1] * FEED) / sum(amounts)
    np.testing.assert_allclose(outlet.mass_frac, expected, rtol=ROUNDING,
                               atol=0)
    assert outlet.mass_frac[API_INDEX] == 0.0
    assert np.ravel(outlet.temp)[0] == pytest.approx(
        _expected_mix_temperature(amounts), rel=TRAJECTORY_RTOL)


@pytest.mark.unit
def test_collector_feed_without_flow_needs_no_data(paths):
    """A collector feed row without flow adds nothing and needs no data.

    Inside FEED_WINDOW the feed carries 10 % api, which has no ``cp_liq``,
    at zero flow. Its enthalpy and species rates there are exactly zero;
    outside the window the rates are those of the api-free feed. Earlier
    the window's feed enthalpy required api's ``cp_liq``.
    """
    collector = DynamicCollector()
    collector.Inlet = LiquidStream(paths["constructible"], temp=FEED_TEMP,
                                   mass_flow=FEED_FLOW, mass_frac=FEED)
    _windowed_feed(collector, inside=WITH_API, flow=(0.0, FEED_FLOW))
    cp_mass = _mass_heat_capacities(API_FULL)  # [J/kg/K]

    inside = collector.get_inputs_new(2.0)["Inlet"]
    outside = collector.get_inputs_new(3.0)["Inlet"]

    np.testing.assert_array_equal(collector.material_balances(inside),
                                  np.zeros(3))
    assert collector.energy_balance(inside) == 0.0
    np.testing.assert_allclose(collector.material_balances(outside),
                               FEED_FLOW * FEED, rtol=ROUNDING, atol=0)
    assert collector.energy_balance(outside) == pytest.approx(
        FEED_FLOW * (FEED @ cp_mass) * (FEED_TEMP - TEMP_REF), rel=ROUNDING)


@pytest.mark.assimulo
@pytest.mark.integration
def test_collector_solve_with_a_stopped_feed_needs_no_data(paths):
    """A collector run whose feed stops in FEED_WINDOW needs no api data.

    While the flow is zero the feed composition holds api; the solve
    keeps api's inventory at exactly zero and equals the run with full
    api data.
    """
    pytest.importorskip("assimulo")
    results = {}
    for record in ("full", "constructible"):
        collector = DynamicCollector()
        collector.Inlet = LiquidStream(paths[record], temp=FEED_TEMP,
                                       mass_flow=FEED_FLOW, mass_frac=FEED)
        _windowed_feed(collector, inside=WITH_API, flow=(0.0, FEED_FLOW))
        _, states = collector.solve_unit(RUN_TIME, verbose=False)
        results[record] = states

    assert np.all(results["constructible"][:, API_INDEX] == 0.0)
    np.testing.assert_allclose(results["constructible"], results["full"],
                               rtol=ROUNDING, atol=0)


# Profile times [s] of the connected inlet in the time-grid mixer cases.
GRID_TIMES = np.array([0.0, 1.0, 2.0])
# Algebraic-balance agreement of the mixer with the independent mixture
# (float64 property and root error), as RTOL of
# tests/test_container_liquid_continuation.py.
MIX_RTOL = 1e-10  # [-]


@pytest.mark.unit
@pytest.mark.parametrize("control", ["composition", "flow"])
def test_controlled_inlet_next_to_a_time_grid_mixes(data_path, control):
    """A time-independent control next to a time-grid inlet mixes.

    The first inlet is a connected cold profile on GRID_TIMES at 1 kg/s;
    the second is a hot stream whose controller sets only its composition
    (HOT_FRAC) or only its flow (2 kg/s). The mixer stores the controlled
    composition as one row, which the earlier feed weights broadcast to
    the grid and the enthalpy rejected. Flows, fractions and temperatures
    equal the independent mixture of the two feeds on the grid.
    """
    path = str(data_path["flowsheet"] / "compound_database.json")
    cold = {"mass_flow": np.full(len(GRID_TIMES), FEED_FLOW),  # [kg/s]
            "temp": np.full(len(GRID_TIMES), COLD_TEMP),  # [K]
            "mass_frac": np.tile(COLD_FRAC, (len(GRID_TIMES), 1))}  # [-]
    hot_flow = 2 * FEED_FLOW if control == "flow" else FEED_FLOW  # [kg/s]
    hot = {"mass_flow": np.full(len(GRID_TIMES), hot_flow),  # [kg/s]
           "temp": np.full(len(GRID_TIMES), HOT_TEMP),  # [K]
           "mass_frac": np.tile(HOT_FRAC, (len(GRID_TIMES), 1))}  # [-]
    mixer = Mixer()
    Connection(profile_source(path, cold, GRID_TIMES), mixer).transfer_data()
    stream = LiquidStream(path, mass_flow=FEED_FLOW, temp=HOT_TEMP,
                          mass_frac=COLD_FRAC if control == "composition"
                          else HOT_FRAC)
    stream.DynamicInlet = DynamicInput()
    if control == "composition":
        stream.DynamicInlet.add_variable("mass_frac", lambda time: HOT_FRAC)
    else:
        stream.DynamicInlet.add_variable("mass_flow", lambda time: hot_flow)
    mixer.Inlets = stream

    flow, fractions, temps = mixer.solve_unit()

    expected = expected_mixture(path, (cold, hot))
    np.testing.assert_allclose(flow, expected[0], rtol=MIX_RTOL)
    np.testing.assert_allclose(fractions, expected[1], rtol=MIX_RTOL)
    np.testing.assert_allclose(temps, expected[2], rtol=MIX_RTOL)


@pytest.mark.unit
def test_one_composition_row_is_fed_if_any_flow_row_flows():
    """One composition row against a flow profile keeps its shape.

    A species of the row is fed if the flow is nonzero in any row, and not
    fed if every row is zero; a 1-D composition keeps one row per flow.
    """
    from PharmaPy.Presence import feed_weights

    row = np.array([[0.5, 0.0, 0.5]])  # [-], one stored row
    flows = np.array([0.0, 2.0, 0.0])  # [kg/s]

    np.testing.assert_array_equal(feed_weights(flows, row),
                                  [[1.0, 0.0, 1.0]])
    np.testing.assert_array_equal(feed_weights(np.zeros(3), row),
                                  [[0.0, 0.0, 0.0]])
    np.testing.assert_array_equal(feed_weights(flows, row[0]),
                                  [[0.0, 0.0, 0.0], [1.0, 0.0, 1.0],
                                   [0.0, 0.0, 0.0]])


# Slurry-stream mixer of tests/test_mixer_slurry_energy_balance.py on the
# shipped five species plus api (last, without cp_liq): liquid and solid
# mass fractions [-], crystal grid [um], volume-specific number density
# [#/m**3/um] (third moment 0.2 m**3/m**3), shape factor [-], slurry flows
# [m**3/s] and temperatures [K].
SLURRY_LIQUID_FRAC = [0.1, 0.1, 0.1, 0.1, 0.6, 0.0]
SLURRY_SOLID_FRAC = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
SLURRY_GRID = np.array([0.0, 100.0, 200.0, 300.0])  # [um]
SLURRY_DISTRIB = np.array([0.0, 1.0e9, 1.25e8, 0.0])  # [#/m**3/um]
SLURRY_KV = 1.0  # [-]
SLURRY_FLOWS = (1.0e-3, 2.0e-3)  # [m**3/s]
SLURRY_TEMPS = (350.0, 300.0)  # [K]
# A liquid inlet without flow holding 10 % api.
EMPTY_INLET_FRAC = [0.1, 0.1, 0.1, 0.1, 0.5, 0.1]  # [-]


def _slurry_database(data_path, tmp_path, api, name):
    """Write the shipped five-species database plus ``api``.

    Parameters
    ----------
    data_path : dict
        Repository data directories.
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
    shipped = data_path["flowsheet"] / "compound_database.json"
    records = json.loads(shipped.read_text())
    records["api"] = api
    path = tmp_path / name
    path.write_text(json.dumps(records))
    return str(path)


def _slurry_mix_temperature(path):
    """Solve the slurry-stream mix with a zero-flow liquid inlet.

    Parameters
    ----------
    path : str
        Database path (five species plus api).

    Returns
    -------
    float
        Adiabatic outlet temperature [K] of ``Mixer.energy_balance``, as
        tests/test_mixer_slurry_energy_balance.py calls it, with the
        zero-flow EMPTY_INLET_FRAC liquid inlet first.
    """
    def slurry(vol_flow, temp):
        """Return a slurry stream of ``vol_flow`` [m**3/s] at ``temp`` [K]."""
        stream = SlurryStream(vol_flow=vol_flow, x_distrib=SLURRY_GRID,
                              distrib=SLURRY_DISTRIB)
        stream.Phases = (
            LiquidStream(path, mass_frac=SLURRY_LIQUID_FRAC, temp=temp),
            SolidStream(path, mass_frac=SLURRY_SOLID_FRAC, kv=SLURRY_KV,
                        temp=temp))
        return stream

    first, second = (slurry(flow, temp)
                     for flow, temp in zip(SLURRY_FLOWS, SLURRY_TEMPS))
    empty = LiquidStream(path, temp=SLURRY_TEMPS[0], mass_flow=0.0,
                         mass_frac=EMPTY_INLET_FRAC, verbose=False)
    mixer = Mixer()
    mixer.Inlets = [empty, first, second]
    u_inputs = {
        "temp": np.array([empty.temp, first.Liquid_1.temp,
                          second.Liquid_1.temp]),  # [K]
        "mass_frac": np.array([empty.mass_frac, first.Liquid_1.mass_frac,
                               second.Liquid_1.mass_frac]),  # [-]
        "mass_liq": np.array([0.0, first.Liquid_1.mass_flow,
                              second.Liquid_1.mass_flow]),  # [kg/s]
        "mass_solid": np.array([0.0, first.Solid_1.mass_flow,
                                second.Solid_1.mass_flow]),  # [kg/s]
        "num_distrib": np.vstack((np.zeros_like(first.Solid_1.distrib),
                                  first.Solid_1.distrib,
                                  second.Solid_1.distrib))}  # [#/um/s]
    outlet = SlurryStream(vol_flow=sum(SLURRY_FLOWS), x_distrib=SLURRY_GRID,
                          distrib=SLURRY_DISTRIB)
    outlet.Phases = (
        LiquidStream(path, mass_frac=SLURRY_LIQUID_FRAC),
        SolidStream(path, mass_frac=SLURRY_SOLID_FRAC, kv=SLURRY_KV))
    mixer.Outlet = outlet
    return mixer.energy_balance(u_inputs)


@pytest.mark.unit
def test_slurry_mixer_liquid_inlet_without_flow_needs_no_data(data_path,
                                                              tmp_path):
    """A zero-flow liquid inlet of a slurry mix needs no api cp_liq.

    The slurry-path energy balance with a zero-flow liquid inlet holding
    api equals the balance with api's full data; earlier that inlet's
    enthalpy required api's ``cp_liq``.
    """
    bare = _slurry_database(data_path, tmp_path, API_CONSTRUCTIBLE,
                            "slurry_bare.json")
    full = _slurry_database(data_path, tmp_path, API_FULL,
                            "slurry_full.json")

    # The slurry phases are built without amounts, which the slurry
    # streams then set from their volume flows; construction warns.
    with pytest.warns(RuntimeWarning, match="all set to zero"):
        temp_bare = _slurry_mix_temperature(bare)  # [K]
    with pytest.warns(RuntimeWarning, match="all set to zero"):
        temp_full = _slurry_mix_temperature(full)  # [K]

    assert temp_bare == pytest.approx(temp_full, rel=ROUNDING)
    assert SLURRY_TEMPS[1] < temp_bare < SLURRY_TEMPS[0]
