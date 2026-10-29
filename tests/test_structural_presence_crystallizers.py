"""Structural species presence in the crystallizers (#425, part B2).

The crystallizers integrate liquid mass concentrations and copy them into
the liquid phase whose density, heat capacity and enthalpy the balances
read (also through ``Slurry``), so a difference-quotient Jacobian (CVode's,
or SciPy's ``BDF``) that perturbs the concentration of a species that is
neither charged nor fed made the balances need its ``rho_liq`` and
``cp_liq``. ``BatchCryst``, ``MSMPR`` and ``SemibatchCryst``, with moment
and 1D-FVM populations, now keep a structural presence record
(``PharmaPy.Presence.StructuralPresenceMixin``): a species is present if it
is in the liquid charge, is the crystallizing target (dissolution returns
it to the liquid), is the liquid's named solvent (``name_solv``, which the
liquid recomputes from its volume closure), or is fed with a nonzero flow,
from the time of that feed evaluation on. The liquid phase that the kernels read, and the
published concentrations, hold absent species at exactly zero; an absent
species needs no data other than ``mw``, and the result equals the result
with arbitrary data supplied for it. A present species without data still
raises the species-named error, and a failed solve keeps no results. The
slurry-fed ``DynamicCollector`` delegates to ``SemibatchCryst`` and is
covered through it.

Fixture: the shipped five-species database
(tests/Flowsheet/data/compound_database.json, species A, B, C, D and
solvent) plus a sixth species ``api``. ``API_FULL`` gives api arbitrary
synthetic data, ``API_BARE`` only its molar mass, ``API_CONSTRUCTIBLE``
adds the liquid density that a phase holding it needs at construction,
without ``cp_liq``. The vessels are seeded with pure-A crystals and use
synthetic growth kinetics (solubility 50 kg/m**3 against about 93 kg/m**3
of A) and a cooling-water utility, so the population, the concentrations
and the temperature all change. Core tests evaluate the right-hand sides
and integrate them with SciPy's ``BDF`` (whose numerical Jacobian perturbs
every state, as CVode's does) before publishing through
``retrieve_results``; the ``assimulo`` tests integrate with CVode (about a
second each).
"""

import json

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from PharmaPy.Containers import DynamicCollector
from PharmaPy.Crystallizers import BatchCryst, MSMPR, SemibatchCryst
from PharmaPy.Kinetics import CrystKinetics
from PharmaPy.MixedPhases import Slurry, SlurryStream
from PharmaPy.Phases import LiquidPhase, SolidPhase
from PharmaPy.ProcessControl import DynamicInput
from PharmaPy.Streams import LiquidStream, SolidStream
from PharmaPy.ThermoModule import MissingPropertyError
from PharmaPy.Utilities import CoolingWater

API_INDEX = 5  # [-], api follows the five shipped species
# Arbitrary synthetic data for api: mw [g/mol], rho_liq and rho_solid
# [kg/m**3], cp_liq and cp_solid coefficients [J/mol/K]. None of it may
# change a result where api is absent.
API_FULL = {"mw": 300.0, "rho_liq": 1300.0, "rho_solid": 1400.0,
            "cp_liq": [500.0], "cp_solid": [400.0]}
API_BARE = {"mw": 300.0}  # [g/mol], no other property data
API_CONSTRUCTIBLE = {"mw": 300.0, "rho_liq": 1300.0}  # no cp_liq

LIQUID_FRAC = [0.1, 0.1, 0.1, 0.1, 0.6, 0.0]  # [-], api absent
WITH_API = [0.1, 0.1, 0.1, 0.1, 0.55, 0.05]  # [-], api charged
SOLID_FRAC = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # [-], pure A crystals
TEMPERATURE = 300.0  # [K], charge and feed
FEED_TEMP = 305.0  # [K]
LIQUID_VOLUME = 1.0e-3  # [m**3]
SEMIBATCH_TANK = 2.0e-3  # [m**3], room for the feed
FEED_FLOW = 2.0e-6  # [m**3/s]
# Volume-specific seed moments [m**n/m**3], n = 0..3, of a synthetic,
# polydisperse seed: 1e8 crystals per m**3 with a number-mean size
# mu_1/mu_0 = 100 um, and mu_3 = 0.05 m**3/m**3, i.e. a solids volume
# fraction kv mu_3 = 2.5 % (kv = 0.5). The moments are not those of a
# monodisperse population; they only need to be positive and realizable.
SEED_MOMENTS = np.array([1e8, 1e4, 1.0, 0.05])  # [m**n/m**3]
SEED_GRID = np.array([10.0, 20.0, 40.0, 80.0])  # [um]
SEED_DISTRIB = np.array([1e7, 2e7, 1e7, 5e6])  # [#/um], asymmetric
KV = 0.5  # [-]
# Synthetic growth kinetics: solubility [kg/m**3] below the A charge
# (about 93 kg/m**3), growth prefactor [um/s], activation energy [J/mol],
# order [-] on the absolute supersaturation.
SOLUBILITY = [50.0]  # [kg/m**3]
GROWTH = [1.0e-3, 0.0, 1.0]  # [um/s], [J/mol], [-]
# Dissolution of api crystals into an api-free liquid: solubility
# [kg/m**3], dissolution prefactor [um/s], activation energy [J/mol],
# order [-].
API_SOLUBILITY = [50.0]  # [kg/m**3]
DISSOLUTION = (1.0, 0.0, 1.0)  # [um/s], [J/mol], [-]
UTILITY_FLOW = 2.0e-5  # [m**3/s], cooling water
UTILITY_TEMP = 285.0  # [K]
RUN_TIME = 10.0  # [s]
# Slurry-fed collector, as in tests/test_dynamic_collector_slurry_sources.py:
# the issue #416/#417 slurry liquid flow, an undersaturated solubility far
# above every A concentration (about 93 kg/m**3) so that the default zero
# rates leave the feed uncrystallized, and an absolute CVode tolerance four
# orders below the collector's smallest state, its seed liquid volume of
# about 1e-8 m**3.
COLLECTOR_LIQUID_FLOW = 0.5  # [kg/s]
COLLECTOR_SOLUBILITY = [2000.0]  # [kg/m**3]
COLLECTOR_ATOL = 1e-12  # [state units]
# api data for the presence checks: a dissolving api target needs its
# crystal density and heat capacity but lacks cp_liq; units as API_FULL.
API_TARGET_NO_CP = {"mw": 300.0, "rho_liq": 1300.0, "rho_solid": 1400.0,
                    "cp_solid": [400.0]}
API_FEED_CONC = 10.0  # [kg/m**3], api in a controlled feed
ZERO_FLOW_UNTIL = 5.0  # [s], a feed flow controller is zero before this
# The cooled vessels change their tank temperature by 0.27 K to 0.71 K over
# RUN_TIME in these fixtures; a change above 0.1 K shows that the energy
# balance was integrated, far above the integration tolerances.
TEMP_CHANGE_FLOOR = 0.1  # [K]
FEED_WINDOW = (1.0, 2.5)  # [s], a feed of api that starts and stops
# Round-off of an absent species, of the size the solvers leave.
ROUND_OFF = 1.0e-20  # [kg/m**3]
# Equal arithmetic on equal inputs; eight units of float64 round-off allow
# for reordered sums (assumption, not a derived count).
ROUNDING = 8 * np.finfo(float).eps  # [-]
# BDF and CVode local error limits of the integrations, relative [-] and
# absolute [state units], and the allowance for comparing two integrated
# trajectories, well above them.
ODE_RTOL = 1e-8  # [-]
ODE_ATOL = 1e-10  # [state units]
TRAJECTORY_RTOL = 1e-5  # [-]
CASES = [(BatchCryst, "moments"), (BatchCryst, "1D-FVM"),
         (MSMPR, "moments"), (MSMPR, "1D-FVM"),
         (SemibatchCryst, "moments"), (SemibatchCryst, "1D-FVM")]
CASE_IDS = [f"{cls.__name__}-{method}" for cls, method in CASES]


def _database(data_path, tmp_path, api, name):
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
    records = dict(json.loads(shipped.read_text()), api=api)
    path = tmp_path / name
    path.write_text(json.dumps(records))
    return str(path)


@pytest.fixture
def paths(data_path, tmp_path):
    """Return the databases of the fixture records.

    Parameters
    ----------
    data_path : dict
        Repository data directories.
    tmp_path : pathlib.Path
        Per-test directory.

    Returns
    -------
    dict
        Database paths by record: ``full``, ``bare``, ``constructible`` and
        ``target_no_cp``.
    """
    return {kind: _database(data_path, tmp_path, api, f"{kind}.json")
            for kind, api in (("full", API_FULL), ("bare", API_BARE),
                              ("constructible", API_CONSTRUCTIBLE),
                              ("target_no_cp", API_TARGET_NO_CP))}


def _crystallizer(cls, method, path, liquid_frac=LIQUID_FRAC,
                  solid_frac=SOLID_FRAC, target="A", kinetics=None,
                  name_solv=None, controls=None):
    """Build a seeded, cooled crystallizer.

    Parameters
    ----------
    cls : type
        ``BatchCryst``, ``MSMPR`` or ``SemibatchCryst``.
    method : {'moments', '1D-FVM'}
        Population model.
    path : str
        Database path.
    liquid_frac, solid_frac : list of float, optional
        Liquid charge and crystal mass fractions [-].
    target : str, optional
        Crystallizing species.
    kinetics : CrystKinetics, optional
        Kinetics; the growth kinetics of the module when None.
    name_solv : str, optional
        Named solvent of the liquid charge, completed by its volume closure.
    controls : dict, optional
        Crystallizer controls, for example a prescribed temperature [K].

    Returns
    -------
    crystallizer
        LIQUID_VOLUME [m**3] of liquid at TEMPERATURE [K] with KV seeds; the
        continuous units get a FEED_FLOW [m**3/s] liquid feed of the charge
        composition at FEED_TEMP [K].
    """
    liquid = LiquidPhase(path, temp=TEMPERATURE, vol=LIQUID_VOLUME,
                         mass_frac=liquid_frac, name_solv=name_solv)
    kwargs = {"method": method}
    if controls is not None:
        kwargs["controls"] = controls
    if cls is SemibatchCryst:
        kwargs["vol_tank"] = SEMIBATCH_TANK  # [m**3]
    unit = cls(target, **kwargs)
    if method == "moments":
        solid = SolidPhase(path, temp=TEMPERATURE, kv=KV,
                           moments=SEED_MOMENTS * LIQUID_VOLUME,
                           mass_frac=solid_frac)
        slurry = Slurry(vol=LIQUID_VOLUME, moments=SEED_MOMENTS)
        slurry.Phases = [liquid, solid]
        unit.Phases = slurry
    else:
        solid = SolidPhase(path, temp=TEMPERATURE, kv=KV,
                           x_distrib=SEED_GRID, distrib=SEED_DISTRIB,
                           mass_frac=solid_frac)
        unit.Phases = (liquid, solid)
    unit.Utility = CoolingWater(vol_flow=UTILITY_FLOW, temp_in=UTILITY_TEMP)
    unit.Kinetics = kinetics or CrystKinetics(
        coeff_solub=SOLUBILITY, growth=GROWTH, sup_sat_type="absolute")
    if cls is not BatchCryst:
        unit.Inlet = LiquidStream(path, temp=FEED_TEMP, vol_flow=FEED_FLOW,
                                  mass_frac=liquid_frac)
    return unit


def _api_state(unit):
    """Return the index of api's concentration in the solver states.

    Parameters
    ----------
    unit : crystallizer
        Initialized crystallizer.

    Returns
    -------
    int
        Position [-] after the population states.
    """
    return unit.num_distr + API_INDEX


def _with_round_off(unit, states):
    """Return a copy of states with api's concentration at ROUND_OFF.

    Parameters
    ----------
    unit : crystallizer
        Initialized crystallizer.
    states : numpy.ndarray
        Solver states in ``name_states`` order.

    Returns
    -------
    numpy.ndarray
        The perturbed copy.
    """
    perturbed = states.copy()
    perturbed[_api_state(unit)] = ROUND_OFF
    return perturbed


def _integrate(unit):
    """Integrate a crystallizer with BDF and publish the result.

    Parameters
    ----------
    unit : crystallizer
        Crystallizer to initialize and integrate over RUN_TIME [s].

    Returns
    -------
    numpy.ndarray
        Solver states, shape (num_times, num_states).
    """
    initial, final_time = unit.initialize_states(runtime=RUN_TIME)
    solution = solve_ivp(unit.unit_model, (0.0, final_time), initial,
                         method="BDF", rtol=ODE_RTOL, atol=ODE_ATOL)
    assert solution.success, solution.message
    states = solution.y.T.copy()
    unit.retrieve_results(solution.t, solution.y.T.copy())
    return states


# ------------------------------------------------------------- core tests
@pytest.mark.unit
@pytest.mark.parametrize("cls, method", CASES, ids=CASE_IDS)
def test_rhs_needs_no_data_for_an_absent_species(paths, cls, method):
    """Round-off api in the states needs no data and changes nothing.

    The derivatives with the bare database equal those with full api data
    at the same perturbed states; earlier the bare database raised for
    api's ``rho_liq``.
    """
    derivatives = {}
    for kind in ("full", "bare"):
        unit = _crystallizer(cls, method, paths[kind])
        initial, _ = unit.initialize_states(runtime=RUN_TIME)
        derivatives[kind] = unit.unit_model(0.0,
                                            _with_round_off(unit, initial))

    np.testing.assert_allclose(derivatives["bare"], derivatives["full"],
                               rtol=ROUNDING, atol=0)


@pytest.mark.unit
@pytest.mark.parametrize("cls, method", [CASES[0], CASES[3], CASES[4]],
                         ids=[CASE_IDS[0], CASE_IDS[3], CASE_IDS[4]])
def test_integration_needs_no_data_and_publishes_zero(paths, cls, method):
    """A BDF integration, whose Jacobian perturbs api, needs no api data.

    Both databases give the same trajectory; the published concentrations,
    liquid and outlet hold api at exactly zero.
    """
    results = {}
    for kind in ("full", "bare"):
        unit = _crystallizer(cls, method, paths[kind])
        results[kind] = (unit, _integrate(unit))

    full_states = results["full"][1]
    bare, bare_states = results["bare"]
    np.testing.assert_allclose(bare_states[-1], full_states[-1],
                               rtol=TRAJECTORY_RTOL)
    assert np.all(bare.outputs["mass_conc"][:, API_INDEX] == 0.0)
    assert bare.Liquid_1.mass_conc[API_INDEX] == 0.0
    assert bare.Outlet.Liquid_1.mass_conc[API_INDEX] == 0.0
    temp_col = sum(bare.dim_states[:bare.name_states.index("temp")])  # [-]
    assert abs(bare_states[-1, temp_col] - TEMPERATURE) > TEMP_CHANGE_FLOOR


@pytest.mark.unit
@pytest.mark.parametrize("cls", [BatchCryst, MSMPR, SemibatchCryst])
def test_published_round_off_is_zero(paths, cls):
    """Solver round-off of api is not published.

    The retrieved rows carry api at ROUND_OFF; the outputs and the liquid
    hold it at zero. Earlier the published liquid kept the round-off and,
    without api's ``rho_liq``, its update raised.
    """
    unit = _crystallizer(cls, "moments", paths["bare"])
    initial, _ = unit.initialize_states(runtime=RUN_TIME)
    rows = np.tile(_with_round_off(unit, initial), (2, 1))

    unit.retrieve_results(np.array([0.0, 1.0]), rows)

    assert np.all(unit.outputs["mass_conc"][:, API_INDEX] == 0.0)
    assert unit.Liquid_1.mass_conc[API_INDEX] == 0.0


@pytest.mark.unit
def test_presence_starts_at_assignment(paths):
    """Assigning the phases records the charge and the target.

    The charge holds A to solvent; A is also the target. api, neither
    charged nor the target, is absent.
    """
    unit = _crystallizer(BatchCryst, "moments", paths["bare"])

    np.testing.assert_array_equal(unit._presence.mask,
                                  [True] * 5 + [False])


@pytest.mark.unit
@pytest.mark.parametrize("cls", [BatchCryst, MSMPR, SemibatchCryst])
def test_named_solvent_is_present(paths, cls):
    """A named solvent counts as present, even at zero concentration.

    The liquid completes its named solvent from its volume closure on every
    update, so a projected solvent would reappear in the stored liquid
    while the outputs report zero. api is the named solvent at zero
    concentration and lacks ``cp_liq``: the initialization raises the
    species-named error, instead of the solve failing later.
    """
    with pytest.raises(MissingPropertyError, match=r"\['api'\].*'cp_liq'"):
        unit = _crystallizer(cls, "moments", paths["constructible"],
                             name_solv="api")
        unit.solve_unit(RUN_TIME, verbose=False)


def _windowed_feed(unit, inside):
    """Make the feed carry other concentrations inside FEED_WINDOW.

    Parameters
    ----------
    unit : MSMPR or SemibatchCryst
        Crystallizer whose ``Inlet`` gets the controller.
    inside : numpy.ndarray
        Feed mass concentrations [kg/m**3] inside FEED_WINDOW [s]; outside
        the stream's own.
    """
    outside = unit.Inlet.mass_conc.copy()  # [kg/m**3]

    def pick(time):
        """Return the feed concentrations at some times.

        Parameters
        ----------
        time : float or array-like
            Absolute time [s], scalar or shape ``(num_times,)``.

        Returns
        -------
        numpy.ndarray
            Mass concentrations [kg/m**3], shape ``(num_species,)`` or
            ``(num_times, num_species)``.
        """
        time = np.asarray(time, dtype=float)  # [s]
        within = (FEED_WINDOW[0] <= time) & (time < FEED_WINDOW[1])  # [-]
        return np.where(within[..., np.newaxis], inside, outside)

    control = DynamicInput()
    control.add_variable("mass_conc", pick)
    unit.Inlet.DynamicInlet = control


@pytest.mark.unit
def test_species_fed_later_is_present_from_its_first_feed(paths):
    """A feed of api in FEED_WINDOW makes it present from its feed on.

    The feed's controller adds api only inside the window, so the stream's
    own composition (and enthalpy) holds none. Derivatives with round-off
    api need no data at 0.5 s; once a feed evaluation at 2 s has supplied
    api, they need its ``cp_liq`` at 2 s and 3 s (latched), and still none
    at 0.5 s.
    """
    unit = _crystallizer(MSMPR, "moments", paths["constructible"])
    inside = unit.Inlet.mass_conc.copy()  # [kg/m**3]
    inside[API_INDEX] = 10.0  # [kg/m**3], api fed in the window
    _windowed_feed(unit, inside)
    initial, _ = unit.initialize_states(runtime=RUN_TIME)
    states = _with_round_off(unit, initial)

    early = unit.unit_model(0.5, states)
    for time in (2.0, 3.0):  # [s], fed, then latched after the feed
        with pytest.raises(MissingPropertyError,
                           match=r"\['api'\].*'cp_liq'"):
            unit.unit_model(time, states)
    again = unit.unit_model(0.5, states)

    np.testing.assert_array_equal(again, early)


@pytest.mark.unit
def test_dissolving_target_is_present(paths):
    """A target dissolving into a liquid that holds none is present.

    The liquid charge holds no api; the crystals are api and dissolve.
    The published api concentration is the integrated one, not a projected
    zero, and it grows from zero.
    """
    kinetics = CrystKinetics(coeff_solub=API_SOLUBILITY,
                             dissolution=DISSOLUTION,
                             sup_sat_type="absolute")
    unit = _crystallizer(BatchCryst, "moments", paths["full"],
                         solid_frac=[0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                         target="api", kinetics=kinetics)

    states = _integrate(unit)

    api_state = states[:, _api_state(unit)]  # [kg/m**3]
    assert api_state[0] == 0.0 and api_state[-1] > 0.0
    np.testing.assert_array_equal(unit.outputs["mass_conc"][:, API_INDEX],
                                  api_state)
    assert unit.Liquid_1.mass_conc[API_INDEX] == api_state[-1]


@pytest.mark.unit
@pytest.mark.parametrize("cls, isothermal", [
    (BatchCryst, False), (MSMPR, False), (SemibatchCryst, False),
    (SemibatchCryst, True)],
    ids=["BatchCryst", "MSMPR", "SemibatchCryst",
         "SemibatchCryst-isothermal"])
def test_dissolving_target_without_data_is_reported_by_name(paths, cls,
                                                            isothermal):
    """A target at zero concentration without cp_liq is reported first.

    The liquid holds no api, the crystals are api and could dissolve, and
    api lacks ``cp_liq``. The initialization, which ``solve_unit`` runs
    before CVode, raises the species-named error; earlier the initial
    derivatives passed (api at zero) and CVode failed without naming it.
    An isothermal
    SemibatchCryst needs ``cp_liq`` too: its published slurry evaluates the
    liquid enthalpy, which earlier failed after the integration.
    """
    kinetics = CrystKinetics(coeff_solub=API_SOLUBILITY,
                             dissolution=DISSOLUTION,
                             sup_sat_type="absolute")
    controls = ({"temp": lambda time: TEMPERATURE + 0 * np.asarray(time)}
                if isothermal else None)  # [K]
    unit = _crystallizer(cls, "moments", paths["target_no_cp"],
                         solid_frac=[0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                         target="api", kinetics=kinetics, controls=controls)

    # The initialization that solve_unit runs first raises, before any
    # solver exists.
    with pytest.raises(MissingPropertyError, match=r"\['api'\].*'cp_liq'"):
        unit.initialize_states(runtime=RUN_TIME)


def _controlled_api_feed(unit, flow=None):
    """Make a crystallizer's feed carry api through its controller only.

    Parameters
    ----------
    unit : MSMPR or SemibatchCryst
        Crystallizer whose ``Inlet`` gets the controller.
    flow : callable, optional
        Feed volume flow [m**3/s] as a function of time [s]; the stream's
        flow when None.

    Notes
    -----
    The stream's own composition, which its density and enthalpy use,
    holds no api; the controller adds API_FEED_CONC [kg/m**3] of api at
    every time.
    """
    conc = unit.Inlet.mass_conc.copy()  # [kg/m**3]
    conc[API_INDEX] = API_FEED_CONC  # [kg/m**3]

    def feed_conc(time):
        """Return the feed concentrations [kg/m**3] at ``time`` [s].

        Parameters
        ----------
        time : float or array-like
            Absolute time [s], scalar or shape ``(num_times,)``.

        Returns
        -------
        numpy.ndarray
            Mass concentrations [kg/m**3], shape ``(num_species,)`` or
            ``(num_times, num_species)``.
        """
        return np.broadcast_to(conc, np.shape(time) + conc.shape).copy()

    control = DynamicInput()
    control.add_variable("mass_conc", feed_conc)
    if flow is not None:
        control.add_variable("vol_flow", flow)
    unit.Inlet.DynamicInlet = control


@pytest.mark.unit
@pytest.mark.parametrize("cls", [MSMPR, SemibatchCryst])
def test_species_fed_from_the_start_without_data_is_reported(paths, cls):
    """A species fed from t = 0 without cp_liq is reported first.

    The controlled feed supplies api from the start while the tank holds
    none, and api lacks ``cp_liq``. ``solve_unit`` raises the
    species-named error in its initialization, before CVode; earlier the
    initial derivatives passed and CVode failed without naming it.
    """
    unit = _crystallizer(cls, "moments", paths["constructible"])
    _controlled_api_feed(unit)

    with pytest.raises(MissingPropertyError, match=r"\['api'\].*'cp_liq'"):
        unit.solve_unit(RUN_TIME, verbose=False)


@pytest.mark.unit
@pytest.mark.parametrize("cls", [MSMPR, SemibatchCryst])
def test_feed_without_flow_needs_no_data(paths, cls):
    """A feed without flow adds nothing and needs no data.

    The feed stream holds api, which lacks ``cp_liq``, and its controlled
    flow is zero until ZERO_FLOW_UNTIL. The initialization and the
    derivatives before then need no api data; once the feed flows, its
    enthalpy needs it. Earlier the zero-flow feed's enthalpy required
    api's ``cp_liq`` at once.
    """
    unit = _crystallizer(cls, "moments", paths["constructible"])
    unit.Inlet = LiquidStream(paths["constructible"], temp=FEED_TEMP,
                              vol_flow=FEED_FLOW, mass_frac=WITH_API)
    control = DynamicInput()
    control.add_variable(
        "vol_flow", lambda time: np.where(np.asarray(time) < ZERO_FLOW_UNTIL,
                                          0.0, FEED_FLOW))
    unit.Inlet.DynamicInlet = control

    initial, _ = unit.initialize_states(runtime=RUN_TIME)
    derivatives = unit.unit_model(1.0, initial)

    assert np.all(np.isfinite(derivatives))
    with pytest.raises(MissingPropertyError, match=r"\['api'\].*'cp_liq'"):
        unit.unit_model(ZERO_FLOW_UNTIL + 1.0, initial)


@pytest.mark.unit
@pytest.mark.parametrize("cls", [MSMPR, SemibatchCryst])
def test_publication_uses_the_presence_of_each_row(paths, cls):
    """Rows before api was fed publish it at zero, later rows keep it.

    api is fed inside FEED_WINDOW, latched by a feed evaluation at 2 s;
    the retrieved rows at 0.5 s and 3 s carry round-off and a real api
    concentration. The published outputs hold the first at exactly zero
    and the second unchanged.
    """
    unit = _crystallizer(cls, "moments", paths["full"])
    inside = unit.Inlet.mass_conc.copy()  # [kg/m**3]
    inside[API_INDEX] = API_FEED_CONC  # [kg/m**3]
    _windowed_feed(unit, inside)
    initial, _ = unit.initialize_states(runtime=RUN_TIME)
    unit.unit_model(2.0, initial)
    later = initial.copy()
    later[_api_state(unit)] = 0.05  # [kg/m**3], api fed in by 3 s
    rows = np.vstack((_with_round_off(unit, initial), later))

    unit.retrieve_results(np.array([0.5, 3.0]), rows)

    published = unit.outputs["mass_conc"][:, API_INDEX]  # [kg/m**3]
    assert published[0] == 0.0
    assert published[1] == 0.05


@pytest.mark.unit
def test_each_initialization_rebuilds_the_presence(paths):
    """A feed latch of one run does not carry into the next.

    A first initialization and a feed evaluation at 2 s latch api (the
    tank still holds none, so nothing needs its data yet); the second
    initialization, after the api feed is removed, starts from the charge
    again, so round-off api needs no data at 3 s. A record kept across
    initializations would still count api as fed.
    """
    unit = _crystallizer(MSMPR, "moments", paths["constructible"])
    inside = unit.Inlet.mass_conc.copy()  # [kg/m**3]
    inside[API_INDEX] = 10.0  # [kg/m**3]
    _windowed_feed(unit, inside)
    initial, _ = unit.initialize_states(runtime=RUN_TIME)
    unit.unit_model(2.0, initial)

    unit.Inlet.DynamicInlet = None
    initial, _ = unit.initialize_states(runtime=RUN_TIME)
    derivatives = unit.unit_model(3.0, _with_round_off(unit, initial))

    assert np.all(np.isfinite(derivatives))


# --------------------------------------------------------- assimulo tests
@pytest.mark.assimulo
@pytest.mark.integration
@pytest.mark.parametrize("cls, method", [CASES[0], CASES[3], CASES[4]],
                         ids=[CASE_IDS[0], CASE_IDS[3], CASE_IDS[4]])
def test_solve_needs_no_data_for_an_absent_species(paths, cls, method):
    """CVode, whose Jacobian perturbs api, needs no api data.

    With dense Newton iterations (moment cases), the bare database failed
    inside CVode before; the 1D-FVM case uses SPGMR, whose products never
    perturbed api alone, and is a regression guard only. Both databases
    give the same trajectory, and api is published at exactly zero.
    """
    pytest.importorskip("assimulo")
    options = {"rtol": ODE_RTOL, "atol": ODE_ATOL}
    results = {}
    for kind in ("full", "bare"):
        unit = _crystallizer(cls, method, paths[kind])
        _, states = unit.solve_unit(RUN_TIME, verbose=False,
                                    sundials_opts=dict(options))
        results[kind] = (unit, np.asarray(states))

    full_states = results["full"][1]
    bare, bare_states = results["bare"]
    np.testing.assert_allclose(bare_states[-1], full_states[-1],
                               rtol=TRAJECTORY_RTOL)
    assert np.all(bare.outputs["mass_conc"][:, API_INDEX] == 0.0)
    assert bare.Liquid_1.mass_conc[API_INDEX] == 0.0


@pytest.mark.assimulo
@pytest.mark.integration
def test_failed_solve_keeps_no_results(paths):
    """A solve that fails on a species fed mid-run keeps nothing.

    api without ``cp_liq`` is fed only inside FEED_WINDOW, so CVode fails
    after the start. Afterwards the unit is as before the call (time,
    profiles, results, presence, liquid), and a solve that ends before the
    window equals a fresh unit's. Earlier the attempt's last iterate stayed
    in the liquid and api stayed latched.
    """
    pytest.importorskip("assimulo")
    from assimulo.solvers.sundials import CVodeError

    units = []
    for _ in range(2):
        unit = _crystallizer(MSMPR, "moments", paths["constructible"])
        inside = unit.Inlet.mass_conc.copy()  # [kg/m**3]
        inside[API_INDEX] = 10.0  # [kg/m**3]
        _windowed_feed(unit, inside)
        units.append(unit)
    unit, fresh = units

    def fingerprint(unit):
        """Return the solve-sensitive state of a unit's phases [mixed units].

        Parameters
        ----------
        unit : MSMPR
            Crystallizer.

        Returns
        -------
        dict
            Liquid concentrations [kg/m**3], fractions [-], temperature [K]
            and volume [m**3]; solid temperature [K] and moments [m**n];
            and every array attribute of the slurry, among them its volume
            [m**3], moments [m**n/m**3] and the volume fractions [-] and
            densities [kg/m**3] that its heat capacity stores, which the
            energy balance of every derivative evaluation rewrites.
        """
        liquid, solid, slurry = unit.Liquid_1, unit.Solid_1, unit.Slurry
        slurry_arrays = {f"slurry_{name}": np.copy(value)
                         for name, value in vars(slurry).items()
                         if isinstance(value, (np.ndarray, float))}
        return {**slurry_arrays,
                "liquid_conc": np.copy(liquid.mass_conc),
                "liquid_frac": np.copy(liquid.mass_frac),
                "liquid_temp": np.copy(liquid.temp),
                "liquid_vol": np.copy(liquid.vol),
                "solid_temp": np.copy(solid.temp),
                "solid_moments": np.copy(solid.moments),
                "slurry_vol": np.copy(slurry.vol),
                "slurry_moments": np.copy(slurry.moments),
                "slurry_names": sorted(vars(slurry))}

    before = fingerprint(unit)
    presence_before = unit._presence.mask

    # CVode reports the right-hand side's missing-data error as repeated
    # recoverable failures, without naming api (issue #435).
    with pytest.raises(CVodeError, match="repeated recoverable errors"):
        unit.solve_unit(RUN_TIME, verbose=False)

    assert unit.elapsed_time == 0 and unit.profiles_runs == []
    assert not hasattr(unit, "result")
    np.testing.assert_array_equal(unit._presence.mask, presence_before)
    after = fingerprint(unit)
    assert set(after) == set(before)
    for name, value in after.items():
        np.testing.assert_array_equal(value, before[name], err_msg=name)
    short = 0.5 * FEED_WINDOW[0]  # [s], before api is fed
    _, states = unit.solve_unit(short, verbose=False)
    _, fresh_states = fresh.solve_unit(short, verbose=False)
    np.testing.assert_array_equal(np.asarray(states)[-1],
                                  np.asarray(fresh_states)[-1])


@pytest.mark.assimulo
@pytest.mark.integration
@pytest.mark.parametrize("population", ["fvm", "moments"])
def test_slurry_collector_needs_no_data_for_an_absent_species(paths,
                                                              population):
    """A slurry-fed collector, through its SemibatchCryst, needs no api data.

    The feed slurry's liquid holds no api; the collected concentrations
    equal those with full api data and hold api at exactly zero. The
    moment population runs the delegate with dense Newton iterations,
    which perturbed api and failed before; the 1D-FVM population uses
    SPGMR and is a regression guard only.
    """
    pytest.importorskip("assimulo")
    results = {}
    for kind in ("full", "bare"):
        liquid = LiquidStream(paths[kind], mass_frac=LIQUID_FRAC,
                              temp=TEMPERATURE,
                              mass_flow=COLLECTOR_LIQUID_FLOW)
        if population == "fvm":
            solid = SolidStream(paths[kind], mass_frac=SOLID_FRAC,
                                temp=TEMPERATURE, kv=KV,
                                x_distrib=SEED_GRID.copy(),
                                distrib=SEED_DISTRIB.copy())
            slurry = SlurryStream()
        else:
            solid = SolidStream(paths[kind], mass_frac=SOLID_FRAC,
                                temp=TEMPERATURE, kv=KV)
            slurry = SlurryStream(vol_flow=liquid.vol_flow,
                                  moments=SEED_MOMENTS)
        slurry.Phases = [liquid, solid]
        collector = DynamicCollector()
        collector.KinCryst = CrystKinetics(coeff_solub=COLLECTOR_SOLUBILITY)
        collector.kwargs_cryst = {"target_ind": 0, "target_comp": "A"}
        collector.Inlet = slurry
        collector.solve_unit(runtime=RUN_TIME, verbose=False,
                             sundials_opts={"rtol": ODE_RTOL,
                                            "atol": COLLECTOR_ATOL})
        results[kind] = collector

    bare, full = results["bare"], results["full"]
    assert np.all(bare.outputs["mass_conc"][:, API_INDEX] == 0.0)
    np.testing.assert_allclose(bare.outputs["mass_conc"],
                               full.outputs["mass_conc"],
                               rtol=TRAJECTORY_RTOL)

