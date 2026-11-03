"""Structural species presence in distillation and extraction (#425, B3).

``DynamicDistillation`` integrates the plate liquid mole fractions with IDA
and ``DynamicExtractor`` the stage mole fractions of both liquids (after an
equilibrium correction whose ``root`` iterates perturb every fraction), so
a species that is neither charged nor fed picked up round-off fractions,
and the VLE ratios, activity coefficients and enthalpies evaluated on them
needed its ``p_vap``, UNIQUAC parameters and ``cp_liq``. Both units now keep
a structural presence record (``PharmaPy.Presence.StructuralPresenceMixin``),
give an absent species the unit equilibrium ratio, publish it at exactly
zero and roll back a failed solve. The shortcut design shared by both
columns and the steady ``DistillationColumn`` work on exact compositions,
whose presence is their own nonzero entries plus the keys; a species
without feed then needs no Henry constant and no activity parameters, and
it no longer moves in ``sorted_by_volatility``. ``ContinuousExtractor`` and
``BatchExtractor`` evaluate the default activity ratio on the feed's species
only. ``VaporPhase.getDewPoint`` gained the ``weights`` keyword of
``getBubblePoint``. ``ThreePhaseSettler`` evaluates its kernels only on
stored phases, never on solver states, and is unchanged.

Fixtures: the shipped four-species database
(tests/integration/data/pfr_test_pure_comp.json: A, B, C, solv) for the
ideal columns; the synthetic UNIQUAC fixtures of
tests/test_distillation_activity_model.py for the non-ideal columns; the
synthetic constant-property liquids of
tests/test_dynamic_extractor_inlet_temperatures.py for the dynamic
extractor; and a synthetic partially miscible UNIQUAC ternary for the
steady extractors. Each adds a species ``api``: ``API_FULL`` gives it
arbitrary synthetic data, ``API_BARE`` only its molar mass,
``API_CONSTRUCTIBLE`` adds the liquid density that a phase or stream
holding it needs at construction, ``API_GAS`` a critical temperature below
every temperature here, without Henry data. Expected values come from the
same unit built on the database without ``api`` (an absent species must
not change the present species) and from the same unit with ``API_FULL``
(its data must not change anything). Core tests evaluate residuals,
initializations, shortcut designs and steady solves; the ``assimulo``
tests integrate the dynamic column and extractor with IDA (about a second
each).
"""

import contextlib
import json

import numpy as np
import pytest
from scipy.optimize import brentq

from test_distillation_activity_model import (THERMO_NONIDEAL_BINARY,
                                              THERMO_NONIDEAL_TERNARY)
from test_dynamic_extractor_inlet_temperatures import (
    HEAVY_COMPOSITION, HEAVY_FLOW, INITIAL_TEMPERATURE, LIGHT_COMPOSITION,
    LIGHT_FLOW, PHASE_HOLDUP, SPECIES as EXTRACTION_SPECIES)

from PharmaPy.Distillation import DistillationColumn, DynamicDistillation
from PharmaPy.DynamicExtraction import DynamicExtractor
from PharmaPy.Extractors import BatchExtractor, ContinuousExtractor
from PharmaPy.Phases import LiquidPhase, VaporPhase
from PharmaPy.ProcessControl import DynamicInput
from PharmaPy.Streams import LiquidStream
from PharmaPy.ThermoModule import MissingPropertyError

# Arbitrary synthetic data for api: mw [g/mol], rho_liq [kg/m**3], t_crit
# [K], Antoine A [-], B [K], C [K] (log10(p/[Pa]) = A - B/(T + C); about
# 6e6 Pa at 350 K, so api would be by far the most volatile species if it
# were present), cp_liq [J/mol/K] and UNIQUAC ri, qi, qip [-]. None of it
# may change a result where api is absent.
API_FULL = {"mw": 300.0, "rho_liq": 1300.0, "t_crit": 600.0,
            "p_vap": [10.0, 1000.0, -40.0], "cp_liq": [500.0],
            "ri": 5.0, "qi": 4.0, "qip": 4.0}
API_BARE = {"mw": 300.0}  # [g/mol], no other property data
API_CONSTRUCTIBLE = {"mw": 300.0, "rho_liq": 1300.0}  # no VLE data or cp_liq
# A light gas: critical temperature [K] below every temperature here, so
# its VLE ratio would need a Henry constant, which the databases lack.
API_GAS = {"mw": 300.0, "t_crit": 100.0}
API_VARIANTS = {"bare": API_BARE, "gas": API_GAS}
# Arbitrary UNIQUAC interaction energies [J/mol] of api with every other
# species (row) and of every other species with api (column), for
# API_FULL; the absent api's bare variants carry zeros, the neutral
# entries that the interaction matrix needs for its shape.
API_INTERACTION_ROW = 700.0  # [J/mol]
API_INTERACTION_COLUMN = 900.0  # [J/mol]

PFR_SPECIES = ("A", "B", "C", "solv")
COLUMN_PRESSURE = 101325.0  # [Pa]
FEED_TEMPERATURE = 350.0  # [K]
FEED_FLOW = 3.0  # [mol/s]
# A, B and solv fed, C (with data) and api (when added) not.
PFR_FEED = np.array([0.3, 0.4, 0.0, 0.3])  # [-], A, B, C, solv
TERNARY_FEED = np.array([0.55, 0.35, 0.10])  # [-], light, heavy, nonkey
BINARY_FEED = np.array([0.75, 0.25])  # [-], light, heavy
# Shortcut specifications, as in tests/test_distillation.py: recoveries of
# the keys to the distillate [%], reflux ratio [-], stage counts [-].
LK_RECOVERY = 95.0  # [%]
HK_RECOVERY = 5.0  # [%]
STEADY_REFLUX = 1.0  # [-]
STEADY_PLATES = 8  # [-]
STEADY_FEED_PLATE = 4  # [-]
# Operating point of the dynamic columns, a self-consistent overall
# balance on FEED_FLOW, as in tests/test_distillation_activity_model.py.
DYNAMIC_PLATES = 4  # [-]
DYNAMIC_FEED_PLATE = 2  # [-]
DISTILLATE_FLOW = 1.0  # [mol/s]
DYNAMIC_REFLUX = 2.0  # [-]
PLATE_HOLDUP = 50.0  # [mol]
# Plate temperatures [K] and light-key fractions [-] of a rectification
# profile; the other present species share the rest in fixed proportions.
PLATE_TEMPS = np.linspace(340.0, 380.0, DYNAMIC_PLATES + 1)  # [K]
PLATE_LK = np.linspace(0.6, 0.2, DYNAMIC_PLATES + 1)  # [-]
NONIDEAL_PLATE_TEMPS = np.linspace(455.0, 475.0, DYNAMIC_PLATES + 1)  # [K]
VAPOR_MOLES = 1.0  # [mol], any amount; the dew point is intensive
# Independent ideal bubble point of the shipped feed at COLUMN_PRESSURE:
# bracket [K] around its boiling range and root tolerance [K]. getBubblePoint
# stops its Newton iteration at SciPy's default 1.48e-8 K, which moves the
# ratio by under 1e-9 relative (d ln K / dT is below 0.06 1/K here), well
# inside ALPHA_RTOL.
BUBBLE_BRACKET = (300.0, 450.0)  # [K]
BUBBLE_XTOL = 1e-12  # [K]
ALPHA_RTOL = 1e-8  # [-]

# Round-off of an absent species, of the size the solvers leave (issue
# #425 shows 1e-25 to 1e-21), injected into states.
ROUND_OFF = 1.0e-20  # [-]
# Two databases differ only in data that enters through ROUND_OFF times a
# property of order one, below one unit in the last place of every term,
# or in an absent species that adds exact zeros; a computed quantity can
# then differ only by the rounding of the few operations that form it.
# Eight units of float64 round-off bound that (assumption: an allowance
# for a handful of additions and products, not a derived count).
ROUNDING = 8 * np.finfo(float).eps  # [-]
# Feed controller of the time-resolved case: the feed carrying api flows
# from FEED_ON on; BEFORE_FEED is an earlier evaluation time.
FEED_ON = 5.0  # [s]
BEFORE_FEED = 1.0  # [s]
# IDA local error limits of the integrations, relative [-] and absolute
# ([K] or [-]; [J] for the extractor's energy), and the allowance for two
# trajectories of different state dimension, which take different steps:
# well above the local limits, accumulated over a few hundred steps.
IDA_RTOL = 1e-10  # [-]
IDA_ATOL = 1e-10  # [state units]
TRAJECTORY_RTOL = 1e-6  # [-]
TRAJECTORY_ATOL = 1e-8  # [state units]
RUN_TIME = 5.0  # [s]
NUM_OUTPUTS = 11  # [-], output grid points

# Dynamic extractor: api sits between a and b, so the closure species c
# stays last, as tests/test_dynamic_extractor_inlet_temperatures.py has it.
EXTRACTION_ORDER = ("a", "api", "b", "c")
EXTRACTION_API = 1  # [-]
# Prescribed distribution ratios [-] of the synthetic liquids (heavy over
# light fractions of the fixture, 0.2/0.4, 0.4/0.2 and 0.4/0.4), with an
# arbitrary user ratio for api, which a user k_fun may return.
EXTRACTION_RATIOS = {"a": 0.5, "api": 3.0, "b": 2.0, "c": 1.0}  # [-]
FEED_TEMP = 330.0  # [K], warm light feed
SOLVENT_TEMP = 290.0  # [K], cool heavy solvent
# Inlets off the equilibrium compositions, so that the stage compositions
# change, as in tests/test_dynamic_extractor_inlet_temperatures.py.
LIGHT_INLET = {"a": 0.45, "b": 0.15, "c": 0.4}  # [-]
HEAVY_INLET = {"a": 0.25, "b": 0.35, "c": 0.4}  # [-]
# A solvent that also carries api, and api's share of a light feed [-].
API_SOLVENT = dict(HEAVY_INLET, api=0.1, c=0.3)  # [-]
API_IN_FEED = 0.02  # [-], taken from a
# A Murphree stage efficiency below one [-], as in
# tests/test_dynamic_extractor_stage_efficiency.py's partial-stage cases.
PARTIAL_EFFICIENCY = 0.75  # [-]

# Synthetic partially miscible UNIQUAC ternary for the steady extractors: a
# small polar species w, a larger solvent-like species s and a solute c,
# with surface and volume parameters [-] of the size of small molecules and
# a w-s interaction energy [J/mol] large enough that a w-s-rich feed splits
# into two liquids at LLE_TEMP (probed: two phases of about 48 % and 52 %
# of the feed). Not fitted to any real system.
LLE_SPECIES = {
    "w": {"mw": 18.0, "rho_liq": 1000.0, "cp_liq": [75.0], "ri": 0.92,
          "qi": 1.4, "qip": 1.4},
    "s": {"mw": 74.0, "rho_liq": 810.0, "cp_liq": [180.0], "ri": 3.45,
          "qi": 3.05, "qip": 3.05},
    "c": {"mw": 60.0, "rho_liq": 900.0, "cp_liq": [120.0], "ri": 2.0,
          "qi": 1.9, "qip": 1.9},
}  # mw [g/mol], rho_liq [kg/m**3], constant cp_liq [J/mol/K], ri, qi, qip [-]
LLE_INTERACTION = 3000.0  # [J/mol], a_ws = a_sw
LLE_ORDER = ("w", "api", "s", "c")
LLE_FEED = {"w": 0.5, "s": 0.4, "c": 0.1}  # [-]
LLE_TEMP = 300.0  # [K]
LLE_AMOUNT = 10.0  # [mol], batch charge, or [mol/s], continuous feed
# The successive substitution starts from random seeds and stops when the
# compositions change by less than LLE_TOL [-]; seeds that reach the same
# root agree within about 6e-12 [-] (probed), and the material balance and
# isoactivity hold to the same order at that root, well inside LLE_ATOL.
LLE_TOL = 1e-12  # [-]
LLE_MAX_ITER = 500  # [-]
LLE_ATOL = 1e-9  # [-] for fractions, [mol] or [mol/s] for amounts
# The fixture's two liquids differ by about 0.85 in the fraction of w
# (probed: 0.94 against 0.10), so a split that differs by less than this
# in every fraction would be the trivial one-phase root.
MIN_LLE_SEPARATION = 0.5  # [-]
# Seed of NumPy's global generator for the steady extractors' random
# initial split. The successive substitution does not converge from every
# seed (issue #453): seeded before each solve, seed 324 ends in a Newton
# RuntimeError for the databases with api and seed 90 for the one without
# (probed over seeds 0 to 399). The tests that solve therefore seed each
# solve with one that converges for every database and extractor variant
# here (verified for the ternary without api and with bare and full api,
# for both extractor classes), and restore the generator's state
# afterwards.
LLE_SEED = 0
# Deterministic seeds of a direct material_balance call: a phase fraction
# [-] and the mole fractions [-] of the two liquids, w-rich and s-rich, in
# LLE_ORDER (api absent).
LLE_PHI_SEED = 0.5  # [-]
LLE_X1_SEED = np.array([0.9, 0.0, 0.07, 0.03])  # [-]
LLE_X2_SEED = np.array([0.1, 0.0, 0.8, 0.1])  # [-]

# Underwood fixtures: two species with equal Antoine B and C, so their
# relative volatility is exactly 10**(A_light - A_heavy) at every
# temperature, in an equimolar saturated-vapor feed (q = 0). For alpha = 3
# the Underwood root is 2 and for alpha = 2 it is 1.5, the midpoint of the
# volatilities, where an absent species' placeholder relative volatility
# 1 / K_HK also fell (issue #425 review). Pressures [Pa] as in the review
# reproductions; Antoine A [-], B [K], C [K]; mw [g/mol], rho_liq
# [kg/m**3], t_crit [K] far above the column temperatures.
UNDERWOOD_HEAVY_ANTOINE = [9.0, 1700.0, -40.0]
UNDERWOOD_CASES = {"alpha3-0.5bar": (3.0, 50000.0),
                   "alpha2-2bar": (2.0, 200000.0)}  # [-], [Pa]
UNDERWOOD_RECORD = {"mw": 100.0, "rho_liq": 1000.0, "t_crit": 1000.0}
UNDERWOOD_FEED = np.array([0.5, 0.5])  # [-], light, heavy
# The shortcut design minimizes the squared Underwood residual with a
# tolerance of 1e-10 [-], so the residual is about 1e-5 [-] at most; its
# slope in phi is at least 2
# [-] for these fixtures, so phi is within 5e-6 [-], and the minimum
# reflux changes by at most about 8 [-] per unit of phi, below 4e-5 [-].
UNDERWOOD_ATOL = 1e-4  # [-]


@contextlib.contextmanager
def _seeded_rng(seed):
    """Seed NumPy's global generator and restore its state on exit.

    Parameters
    ----------
    seed : int
        Seed [-] for the block.

    Yields
    ------
    None
        The block runs with the generator seeded; its earlier state is
        restored afterwards, also when the block raises.
    """
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def _write(tmp_path, database, name):
    """Write a property database to a temporary JSON file.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Directory for the file.
    database : dict
        Species records, and optionally an ``interaction`` record.
    name : str
        File name stem.

    Returns
    -------
    str
        Path of the written file.
    """
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps(database))
    return str(path)


def _pfr_database(data_path, tmp_path, api=None, name="pfr"):
    """Write the shipped four-species database, optionally with ``api``.

    Parameters
    ----------
    data_path : dict
        Repository data directories.
    tmp_path : pathlib.Path
        Directory for the file.
    api : dict, optional
        Property record of ``api``, appended last; None leaves it out.
    name : str, optional
        File name stem.

    Returns
    -------
    str
        Path of the database.
    """
    database = json.loads(
        (data_path["integration"] / "pfr_test_pure_comp.json").read_text())
    if api is not None:
        database["api"] = api
    return _write(tmp_path, database, name)


def _uniquac_database(tmp_path, fixture, api=None, name="uniquac",
                      index=None):
    """Write a UNIQUAC fixture database, optionally with ``api`` inserted.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Directory for the file.
    fixture : dict
        Species records and ``interaction`` record with ``amk`` [J/mol].
    api : dict, optional
        Property record of ``api``; None leaves it out. With UNIQUAC
        parameters it gets the arbitrary interaction energies, otherwise
        zeros.
    name : str, optional
        File name stem.
    index : int, optional
        Position of api among the species; last when None.

    Returns
    -------
    str
        Path of the database.
    """
    species = [key for key in fixture if key != "interaction"]
    amk = np.asarray(fixture["interaction"]["amk"], dtype=float)  # [J/mol]
    if api is not None:
        index = len(species) if index is None else index
        species.insert(index, "api")
        amk = np.insert(np.insert(amk, index, 0.0, axis=0), index, 0.0,
                        axis=1)  # [J/mol]
        if "ri" in api:
            amk[index, :] = API_INTERACTION_ROW
            amk[:, index] = API_INTERACTION_COLUMN
            amk[index, index] = 0.0
    database = {key: (api if key == "api" else fixture[key])
                for key in species}
    database["interaction"] = {"amk": amk.tolist()}
    return _write(tmp_path, database, name)


def _with_api(fractions, index=None):
    """Insert a zero api fraction into a composition.

    Parameters
    ----------
    fractions : array-like
        Mole fractions [-] without api, shape ``(num_species,)`` or
        ``(num_rows, num_species)``.
    index : int, optional
        Position of api; last when None.

    Returns
    -------
    numpy.ndarray
        Mole fractions [-] with an api entry of exactly zero.
    """
    fractions = np.asarray(fractions, dtype=float)  # [-]
    index = fractions.shape[-1] if index is None else index
    return np.insert(fractions, index, 0.0, axis=-1)


def _steady_column(path, feed, light_key="A", heavy_key="B",
                   gamma_model="ideal"):
    """Build a steady column with the shortcut specifications.

    Parameters
    ----------
    path : str
        Property database.
    feed : array-like
        Feed mole fractions [-].
    light_key, heavy_key : str, optional
        Key species.
    gamma_model : str, optional
        Activity model.

    Returns
    -------
    DistillationColumn
        Column with its feed at FEED_TEMPERATURE [K] and FEED_FLOW [mol/s].
    """
    column = DistillationColumn(
        pres=COLUMN_PRESSURE, q_feed=1.0, LK=light_key, HK=heavy_key,
        perc_LK=LK_RECOVERY, perc_HK=HK_RECOVERY, reflux=STEADY_REFLUX,
        num_plates=STEADY_PLATES, num_feed=STEADY_FEED_PLATE,
        gamma_model=gamma_model)  # saturated-liquid feed, q = 1 [-]
    column.Inlet = LiquidStream(path, temp=FEED_TEMPERATURE,
                                pres=COLUMN_PRESSURE, mole_flow=FEED_FLOW,
                                mole_frac=feed, verbose=False)
    return column


def _steady_results(column):
    """Collect the published results of a solved steady column.

    Parameters
    ----------
    column : DistillationColumn
        Solved column.

    Returns
    -------
    dict
        Plate temperatures [K], liquid and vapor plate fractions [-]
        (species by plate), product fractions [-], product flows [mol/s],
        minimum reflux [-] and minimum stage count [-].
    """
    result = column.result
    return {"T": np.asarray(result.T), "x": np.asarray(result.x),
            "y": np.asarray(result.y), "x_dist": result.x_dist,
            "x_bot": result.x_bot,
            "flows": np.array([result.dist_flowrate, result.bot_flowrate]),
            "design": np.array([result.min_reflux, result.N_min])}


@pytest.mark.unit
@pytest.mark.parametrize("variant", sorted(API_VARIANTS))
def test_steady_column_needs_no_data_for_a_species_without_feed(
        data_path, tmp_path, variant):
    """A species without feed changes nothing in a steady column.

    ``API_GAS`` is above its critical temperature everywhere and the
    database has no Henry constants, so earlier releases required one for
    it in the shortcut bubble point even at zero fraction.
    """
    reference = _steady_column(_pfr_database(data_path, tmp_path), PFR_FEED)
    reference.solve_unit()
    expected = _steady_results(reference)

    column = _steady_column(
        _pfr_database(data_path, tmp_path, API_VARIANTS[variant], variant),
        _with_api(PFR_FEED))
    column.solve_unit()
    results = _steady_results(column)

    api = len(PFR_SPECIES)  # [-]
    for key in ("x", "y"):
        np.testing.assert_array_equal(results[key][api], 0.0)
        np.testing.assert_allclose(np.delete(results[key], api, axis=0),
                                   expected[key], rtol=ROUNDING, atol=0)
    for key in ("x_dist", "x_bot"):
        assert results[key][api] == 0.0
        np.testing.assert_allclose(np.delete(results[key], api),
                                   expected[key], rtol=ROUNDING, atol=0)
    for key in ("T", "flows", "design"):
        np.testing.assert_allclose(results[key], expected[key],
                                   rtol=ROUNDING, atol=0)
    assert column.OutletDistillate.mole_frac[api] == 0.0
    assert column.OutletBottom.mole_frac[api] == 0.0


@pytest.mark.unit
def test_steady_uniquac_column_needs_no_activity_data_for_absent_species(
        tmp_path):
    """UNIQUAC evaluates the present species only, in every VLE call.

    Earlier releases evaluated the activity model on every species, so the
    bare api's missing ``ri`` and ``qi`` made every coefficient NaN.
    """
    reference = _steady_column(
        _uniquac_database(tmp_path, THERMO_NONIDEAL_TERNARY), TERNARY_FEED,
        "light", "heavy", "UNIQUAC")
    reference.solve_unit()
    expected = _steady_results(reference)

    results = {}
    for name, api in (("bare", API_BARE), ("full", API_FULL)):
        column = _steady_column(
            _uniquac_database(tmp_path, THERMO_NONIDEAL_TERNARY, api, name),
            _with_api(TERNARY_FEED), "light", "heavy", "UNIQUAC")
        column.solve_unit()
        results[name] = _steady_results(column)
        assert column.sorted_by_volatility == (
            reference.sorted_by_volatility + ["api"])

    api = len(TERNARY_FEED)  # [-]
    for name, values in results.items():
        np.testing.assert_array_equal(values["x"][api], 0.0)
        np.testing.assert_allclose(np.delete(values["x"], api, axis=0),
                                   expected["x"], rtol=ROUNDING, atol=0)
        np.testing.assert_allclose(values["T"], expected["T"],
                                   rtol=ROUNDING, atol=0)
        np.testing.assert_allclose(values["design"], expected["design"],
                                   rtol=ROUNDING, atol=0)
    for key in ("T", "x", "y", "flows", "design"):
        np.testing.assert_allclose(results["bare"][key], results["full"][key],
                                   rtol=ROUNDING, atol=0)


@pytest.mark.unit
def test_volatility_order_lists_absent_species_last_whatever_their_data(
        data_path, tmp_path):
    """Absent species follow the present ones, in database order.

    With ``API_FULL`` api would be the most volatile species, and earlier
    releases ranked it first; without data its placeholder ratio ranked it
    last. C, absent too, follows the present species although it has data.
    """
    orders = []
    for name, api in (("bare", API_BARE), ("full", API_FULL)):
        column = _steady_column(
            _pfr_database(data_path, tmp_path, api, name),
            _with_api(PFR_FEED))
        column.global_material_bce()
        orders.append(column.sorted_by_volatility)

    reference = _steady_column(_pfr_database(data_path, tmp_path), PFR_FEED)
    reference.global_material_bce()
    present_order = [name for name in reference.sorted_by_volatility
                     if name != "C"]

    assert orders[0] == orders[1] == present_order + ["C", "api"]


def _binary_underwood_min_reflux(alpha, feed, q_feed):
    """Solve the binary Underwood equations independently of PharmaPy.

    Parameters
    ----------
    alpha : float
        Relative volatility of the light key to the heavy key [-].
    feed : numpy.ndarray
        Feed mole fractions [-] of the light and heavy keys.
    q_feed : float
        Liquid fraction of the feed [-].

    Returns
    -------
    float
        Minimum reflux ratio [-] for the recoveries LK_RECOVERY and
        HK_RECOVERY [%] to the distillate.

    Notes
    -----
    The root ``phi`` of ``alpha z_L / (alpha - phi) + z_H / (1 - phi) =
    1 - q`` between 1 and ``alpha``, a root of the quadratic obtained by
    multiplying out the denominators, gives ``V_min / D = (alpha d_L /
    (alpha - phi) + d_H / (1 - phi)) / D`` with the distillate component
    flows ``d_i = F z_i r_i`` [mol/s], and ``R_min = V_min / D - 1``.
    """
    vapor_fraction = 1 - q_feed  # [-]
    roots = np.roots([
        vapor_fraction,
        alpha * feed[0] + feed[1] - vapor_fraction * (alpha + 1),
        vapor_fraction * alpha - alpha * feed.sum()])  # [-]
    phi = roots[(roots > 1) & (roots < alpha)].item()  # [-]
    distillate = FEED_FLOW * feed * np.array([LK_RECOVERY,
                                              HK_RECOVERY]) / 100  # [mol/s]
    vapor_min = (alpha * distillate[0] / (alpha - phi)
                 + distillate[1] / (1 - phi))  # [mol/s]
    return vapor_min / distillate.sum() - 1


@pytest.mark.unit
@pytest.mark.parametrize("case", sorted(UNDERWOOD_CASES))
def test_underwood_ignores_the_volatility_of_absent_species(tmp_path, case):
    """Absent species stay out of both Underwood sums.

    An absent species gets the unit ratio, so its relative volatility is
    ``1 / K_HK``; in these saturated-vapor feeds that equals the Underwood
    root, and its zero term became ``0 / 0``: the minimum and the selected
    reflux were NaN. The design must match the binary Underwood solution.
    """
    alpha, pressure = UNDERWOOD_CASES[case]  # [-], [Pa]
    light_antoine = [UNDERWOOD_HEAVY_ANTOINE[0] + np.log10(alpha),
                     *UNDERWOOD_HEAVY_ANTOINE[1:]]
    database = {"light": dict(UNDERWOOD_RECORD, p_vap=light_antoine),
                "heavy": dict(UNDERWOOD_RECORD,
                              p_vap=UNDERWOOD_HEAVY_ANTOINE),
                "api": API_BARE}
    path = _write(tmp_path, database, case)
    column = DistillationColumn(
        pres=pressure, q_feed=0.0, LK="light", HK="heavy",
        perc_LK=LK_RECOVERY, perc_HK=HK_RECOVERY,
        num_plates=STEADY_PLATES, num_feed=STEADY_FEED_PLATE)
    column.Inlet = LiquidStream(path, temp=FEED_TEMPERATURE, pres=pressure,
                                mole_flow=FEED_FLOW,
                                mole_frac=_with_api(UNDERWOOD_FEED),
                                verbose=False)

    design = column.calculate_shortcut_design()

    expected = _binary_underwood_min_reflux(alpha, UNDERWOOD_FEED, 0.0)
    assert design["min_reflux"] == pytest.approx(expected, abs=UNDERWOOD_ATOL)
    assert design["reflux"] == pytest.approx(
        column.reflux_to_minimum_ratio * expected,
        abs=column.reflux_to_minimum_ratio * UNDERWOOD_ATOL)


@pytest.mark.unit
def test_shortcut_key_without_vle_data_is_named(data_path, tmp_path):
    """The keys are present by construction, whatever their fractions.

    ``get_alpha`` normalizes by the heavy key's ratio and Fenske uses the
    light key's, so a key without ``p_vap`` raises the species-named error;
    earlier releases used a placeholder ratio when its fraction was zero.
    """
    path = _pfr_database(data_path, tmp_path, API_CONSTRUCTIBLE)
    column = DistillationColumn(
        pres=COLUMN_PRESSURE, q_feed=1.0, LK="api", HK="B",
        perc_LK=LK_RECOVERY, perc_HK=HK_RECOVERY)
    column.Inlet = LiquidStream(path, temp=FEED_TEMPERATURE,
                                pres=COLUMN_PRESSURE, mole_flow=FEED_FLOW,
                                mole_frac=_with_api(PFR_FEED), verbose=False)

    with pytest.raises(MissingPropertyError, match=r"\['api'\].*'p_vap'"):
        column.get_alpha(COLUMN_PRESSURE, _with_api(PFR_FEED))


@pytest.mark.unit
def test_relative_volatility_of_absent_species_is_the_unit_ratio(
        data_path, tmp_path):
    """Absent species get ``alpha = 1 / K_HK``, whatever their data.

    C (with data) and api have zero fraction. The expected heavy-key ratio
    is its Antoine pressure over the column pressure at the ideal bubble
    point of A, B and solv, solved here independently.
    """
    database = json.loads(
        (data_path["integration"] / "pfr_test_pure_comp.json").read_text())
    antoine = np.array([database[name]["p_vap"]
                        for name in PFR_SPECIES])  # [-], [K], [K]

    def saturation(temp):
        """Return the Antoine pressures of A, B, C and solv.

        Parameters
        ----------
        temp : float
            Temperature [K].

        Returns
        -------
        numpy.ndarray
            Saturation pressures [Pa], ``log10(p / [Pa]) = A - B/(T + C)``.
        """
        return 10**(antoine[:, 0] - antoine[:, 1] / (temp + antoine[:, 2]))

    temp_bubble = brentq(
        lambda temp: PFR_FEED @ saturation(temp) / COLUMN_PRESSURE - 1,
        *BUBBLE_BRACKET, xtol=BUBBLE_XTOL)  # [K]
    expected = COLUMN_PRESSURE / saturation(temp_bubble)[1]  # [-], 1/K_HK

    alphas = []
    for name, api in (("bare", API_BARE), ("full", API_FULL)):
        column = _steady_column(_pfr_database(data_path, tmp_path, api, name),
                                _with_api(PFR_FEED))
        alphas.append(column.get_alpha(COLUMN_PRESSURE, _with_api(PFR_FEED)))

    np.testing.assert_array_equal(alphas[0], alphas[1])
    absent = [PFR_SPECIES.index("C"), len(PFR_SPECIES)]  # [-], C and api
    np.testing.assert_allclose(alphas[0][absent], expected, rtol=ALPHA_RTOL,
                               atol=0)


def _dynamic_column(path, feed, light_key="A", heavy_key="B",
                    gamma_model="ideal", holdup=None):
    """Build a dynamic column with its operating point supplied.

    Parameters
    ----------
    path : str
        Property database.
    feed : array-like
        Feed mole fractions [-].
    light_key, heavy_key : str, optional
        Key species.
    gamma_model : str, optional
        Activity model.
    holdup : array-like, optional
        Holdup mole fractions [-]; the feed's when None.

    Returns
    -------
    DynamicDistillation
        Column ready to evaluate ``material_balances`` without the
        shortcut design.
    """
    column = DynamicDistillation(
        pres=COLUMN_PRESSURE, q_feed=1.0, LK=light_key, HK=heavy_key,
        perc_LK=LK_RECOVERY, perc_HK=HK_RECOVERY, gamma_model=gamma_model)
    column.Inlet = LiquidStream(path, temp=FEED_TEMPERATURE,
                                pres=COLUMN_PRESSURE, mole_flow=FEED_FLOW,
                                mole_frac=feed, verbose=False)
    column.Phases = LiquidPhase(path, temp=FEED_TEMPERATURE,
                                pres=COLUMN_PRESSURE, moles=PLATE_HOLDUP,
                                mole_frac=feed if holdup is None else holdup,
                                verbose=False)
    column.num_plates = DYNAMIC_PLATES  # [-]
    column.num_feed = DYNAMIC_FEED_PLATE  # [-]
    column.dist_flowrate = DISTILLATE_FLOW  # [mol/s]
    column.bot_flowrate = FEED_FLOW - DISTILLATE_FLOW  # [mol/s]
    column.reflux = DYNAMIC_REFLUX  # [-]
    return column


def _pfr_plates():
    """Return plate fractions of A, B, C and solv for the ideal columns.

    Returns
    -------
    numpy.ndarray
        Mole fractions [-], shape ``(DYNAMIC_PLATES + 1, 4)``: A follows
        ``PLATE_LK``, B and solv share the rest 4:3, and C is zero.
    """
    rest = 1 - PLATE_LK  # [-]
    return np.column_stack((PLATE_LK, rest * 4 / 7, np.zeros_like(rest),
                            rest * 3 / 7))


def _binary_plates():
    """Return plate fractions of the two-species UNIQUAC fixture.

    Returns
    -------
    numpy.ndarray
        Mole fractions [-] of light and heavy, shape
        ``(DYNAMIC_PLATES + 1, 2)``.
    """
    light = np.linspace(0.85, 0.35, DYNAMIC_PLATES + 1)  # [-]
    return np.column_stack((light, 1 - light))


def _round_off_api(fractions):
    """Append an api entry of solver round-off to plate fractions.

    Parameters
    ----------
    fractions : numpy.ndarray
        Plate mole fractions [-] without api.

    Returns
    -------
    numpy.ndarray
        Fractions [-] with api last at ``ROUND_OFF``.
    """
    return np.insert(fractions, fractions.shape[-1], ROUND_OFF, axis=-1)


@pytest.mark.unit
@pytest.mark.parametrize("case", ["ideal", "UNIQUAC"])
def test_dynamic_column_balances_ignore_round_off_of_absent_species(
        data_path, tmp_path, case):
    """Plate balances with round-off api need none of its data.

    Earlier releases weighted the VLE ratios by the state fractions:
    without ``p_vap`` the round-off api raised the species-named error
    (ideal), and without UNIQUAC parameters every ratio was NaN.
    """
    if case == "ideal":
        databases = {name: _pfr_database(data_path, tmp_path, api, name)
                     for name, api in (("free", None), ("bare", API_BARE),
                                       ("full", API_FULL))}
        feed, plates, temps = PFR_FEED, _pfr_plates(), PLATE_TEMPS
        keys, model = ("A", "B"), "ideal"
    else:
        databases = {name: _uniquac_database(tmp_path, THERMO_NONIDEAL_BINARY,
                                             api, name)
                     for name, api in (("free", None), ("bare", API_BARE),
                                       ("full", API_FULL))}
        feed, plates = BINARY_FEED, _binary_plates()
        temps, keys = NONIDEAL_PLATE_TEMPS, ("light", "heavy")
        model = "UNIQUAC"

    reference = _dynamic_column(databases["free"], feed, *keys, model)
    expected = reference.material_balances(0.0, temps, plates)  # [-], [1/s]

    residuals = {}
    for name in ("bare", "full"):
        column = _dynamic_column(databases[name], _with_api(feed), *keys,
                                 model)
        residuals[name] = column.material_balances(
            0.0, temps, _round_off_api(plates))  # [-], [1/s]

    api_column = 1 + len(feed)  # [-], after the temperature residual
    for values in residuals.values():
        np.testing.assert_allclose(np.delete(values, api_column, axis=1),
                                   expected, rtol=ROUNDING, atol=0)
    np.testing.assert_allclose(residuals["bare"], residuals["full"],
                               rtol=ROUNDING, atol=0)


@pytest.mark.unit
@pytest.mark.parametrize("variant", sorted(API_VARIANTS))
def test_dynamic_column_publishes_absent_species_at_zero(data_path,
                                                         tmp_path, variant):
    """Published liquid and vapor hold the absent species at exactly zero.

    The outlets are built from the published liquid, so the bare api needs
    no ``rho_liq`` either; earlier releases evaluated the vapor ratios and
    built the outlets with its round-off. The gas api is supercritical
    without a Henry constant, so its own ratio is NaN: its unit ratio keeps
    every published value finite.
    """
    reference = _dynamic_column(_pfr_database(data_path, tmp_path),
                                PFR_FEED)
    plate_states = np.column_stack((PLATE_TEMPS, _pfr_plates()))  # [K], [-]
    time = np.array([0.0, RUN_TIME])  # [s]
    reference.retrieve_results(time, np.vstack([plate_states.ravel()] * 2))

    column = _dynamic_column(
        _pfr_database(data_path, tmp_path, API_VARIANTS[variant], variant),
        _with_api(PFR_FEED))
    states = np.column_stack((PLATE_TEMPS, _round_off_api(_pfr_plates())))
    column.retrieve_results(time, np.vstack([states.ravel()] * 2))

    for field in ("x_liq", "y_vap"):
        published = getattr(column.result, field)
        assert all(np.all(np.isfinite(values))
                   for values in published.values())
        np.testing.assert_array_equal(published["api"], 0.0)
        for name in PFR_SPECIES:
            np.testing.assert_allclose(
                published[name], getattr(reference.result, field)[name],
                rtol=ROUNDING, atol=0)
    np.testing.assert_array_equal(column.outputs["mole_frac"][:, -1], 0.0)
    assert column.OutletBottom.mole_frac[-1] == 0.0
    assert column.OutletDistillate.mole_frac[-1] == 0.0
    np.testing.assert_allclose(column.OutletBottom.mole_frac[:-1],
                               reference.OutletBottom.mole_frac,
                               rtol=ROUNDING, atol=0)


def _controlled_feed_column(path, flow_after):
    """Build a dynamic column whose api-carrying feed starts at FEED_ON.

    Parameters
    ----------
    path : str
        Property database with api.
    flow_after : float
        Feed molar flow [mol/s] from FEED_ON on; zero before.

    Returns
    -------
    DynamicDistillation
        Column with an api-free holdup and a feed of api, B and solv.
    """
    feed = np.array([0.3, 0.4, 0.0, 0.2, 0.1])  # [-], A, B, C, solv, api
    column = _dynamic_column(path, feed, holdup=_with_api(PFR_FEED))
    control = DynamicInput()
    control.add_variable(
        "mole_flow",
        lambda time: flow_after if time >= FEED_ON else 0.0)  # [mol/s]
    column.Inlet.DynamicInlet = control
    return column


@pytest.mark.unit
def test_species_fed_later_is_present_only_from_its_feed(data_path,
                                                         tmp_path):
    """Presence is latched at the feed's time and resolved in time.

    The feed of api has no flow before FEED_ON, so it adds nothing there.
    From FEED_ON on api is present and its missing ``p_vap`` is named; an
    evaluation at an earlier time still needs none, as with full data.
    """
    column = _controlled_feed_column(
        _pfr_database(data_path, tmp_path, API_CONSTRUCTIBLE), FEED_FLOW)
    plates = _round_off_api(_pfr_plates())  # [-]

    with pytest.raises(MissingPropertyError, match=r"\['api'\].*'p_vap'"):
        column.material_balances(FEED_ON, PLATE_TEMPS, plates)

    earlier = column.material_balances(BEFORE_FEED, PLATE_TEMPS, plates)
    reference = _controlled_feed_column(
        _pfr_database(data_path, tmp_path, API_FULL, "full"), FEED_FLOW)
    expected = reference.material_balances(BEFORE_FEED, PLATE_TEMPS, plates)
    np.testing.assert_allclose(earlier, expected, rtol=ROUNDING, atol=0)


def _operating_point(column):
    """Collect a dynamic column's shortcut operating point.

    Parameters
    ----------
    column : DynamicDistillation
        Column after ``column_startup``.

    Returns
    -------
    numpy.ndarray
        Distillate and bottoms flows [mol/s], reflux and minimum reflux
        [-], minimum stage count [-], and the distillate mole fractions [-]
        of the species other than api.
    """
    x_dist = column.x_dist  # [-]
    if "api" in column.name_species:
        x_dist = np.delete(x_dist, column.name_species.index("api"))
    return np.concatenate(([column.dist_flowrate, column.bot_flowrate,
                            column.reflux, column.min_reflux,
                            column.num_min], x_dist))


@pytest.mark.unit
@pytest.mark.parametrize("variant", ["constructible", "full"])
def test_design_feed_without_flow_uses_the_holdup(data_path, tmp_path,
                                                  variant):
    """A feed that does not flow at the design time adds nothing to it.

    The feed carries api but never flows, so the startup design uses the
    holdup composition: api, which lacks ``p_vap`` in the constructible
    variant, needs no data, and its data cannot move the operating point.
    Earlier, the design took the feed composition without its flow.
    """
    api = {"constructible": API_CONSTRUCTIBLE, "full": API_FULL}[variant]
    column = _controlled_feed_column(
        _pfr_database(data_path, tmp_path, api, variant), 0.0)
    column.column_startup(RUN_TIME)

    reference = _dynamic_column(_pfr_database(data_path, tmp_path),
                                PFR_FEED)
    reference.column_startup(RUN_TIME)

    assert column.x_dist[-1] == 0.0
    np.testing.assert_allclose(_operating_point(column),
                               _operating_point(reference), rtol=ROUNDING,
                               atol=0)


@pytest.mark.assimulo
@pytest.mark.integration
def test_dynamic_column_solves_with_a_feed_that_never_flows(data_path,
                                                            tmp_path):
    """``solve_unit`` needs no data for a species of a feed without flow.

    Earlier the startup design required api's ``p_vap``, and with data its
    Antoine coefficients changed the operating point. Bare and full data
    must give the same run, with api published at zero.
    """
    pytest.importorskip("assimulo")
    grid = np.linspace(0.0, RUN_TIME, NUM_OUTPUTS)  # [s]
    options = {"rtol": IDA_RTOL, "atol": IDA_ATOL}
    runs = {}
    for name, api in (("constructible", API_CONSTRUCTIBLE),
                      ("full", API_FULL)):
        column = _controlled_feed_column(
            _pfr_database(data_path, tmp_path, api, name), 0.0)
        _, states, _ = column.solve_unit(time_grid=grid,
                                         sundials_opts=options,
                                         verbose=False)
        np.testing.assert_array_equal(column.result.x_liq["api"], 0.0)
        runs[name] = (states, _operating_point(column))

    np.testing.assert_allclose(runs["constructible"][1], runs["full"][1],
                               rtol=ROUNDING, atol=0)
    np.testing.assert_allclose(runs["constructible"][0], runs["full"][0],
                               rtol=ROUNDING, atol=ROUND_OFF)


@pytest.mark.assimulo
@pytest.mark.integration
def test_dynamic_column_solve_rebuilds_presence(data_path, tmp_path):
    """A species latched before the solve needs no data in the solve.

    A feed of api (without ``p_vap``) is evaluated once, which latches api
    and raises; the feed is then replaced by one without api. The solve
    rebuilds the record from the holdup, so it needs no data for api.
    """
    pytest.importorskip("assimulo")
    path = _pfr_database(data_path, tmp_path, API_CONSTRUCTIBLE)
    feed = np.array([0.3, 0.4, 0.0, 0.2, 0.1])  # [-], A, B, C, solv, api
    column = _dynamic_column(path, feed, holdup=_with_api(PFR_FEED))
    with pytest.raises(MissingPropertyError, match=r"\['api'\].*'p_vap'"):
        column.material_balances(0.0, PLATE_TEMPS,
                                 _round_off_api(_pfr_plates()))

    column.Inlet = LiquidStream(path, temp=FEED_TEMPERATURE,
                                pres=COLUMN_PRESSURE, mole_flow=FEED_FLOW,
                                mole_frac=_with_api(PFR_FEED), verbose=False)
    column.solve_unit(time_grid=np.linspace(0.0, RUN_TIME, NUM_OUTPUTS),
                      verbose=False)

    np.testing.assert_array_equal(column.result.x_liq["api"], 0.0)


@pytest.mark.unit
def test_dynamic_column_publishes_each_row_with_its_presence(data_path,
                                                             tmp_path):
    """Published rows use the presence of their own time.

    The feed latches api at FEED_ON; a row before it publishes api at
    exactly zero, a row at it publishes the state value.
    """
    column = _controlled_feed_column(
        _pfr_database(data_path, tmp_path, API_FULL), FEED_FLOW)
    plates = _round_off_api(_pfr_plates())  # [-]
    column.material_balances(FEED_ON, PLATE_TEMPS, plates)
    states = np.column_stack((PLATE_TEMPS, plates)).ravel()

    column.retrieve_results(np.array([BEFORE_FEED, FEED_ON]),
                            np.vstack([states, states]))

    published = column.result.x_liq["api"]  # [-], rows by time
    np.testing.assert_array_equal(published[0], 0.0)
    np.testing.assert_array_equal(published[1], plates[:, -1])


@pytest.mark.unit
def test_design_feed_without_flow_and_holdup_is_rejected(data_path):
    """A design feed without flow needs a holdup to design on.

    The column has no ``Liquid_1``, so there is no defined design
    composition: the startup raises and sets no operating point.
    """
    column = DynamicDistillation(
        pres=COLUMN_PRESSURE, q_feed=1.0, LK="A", HK="B",
        perc_LK=LK_RECOVERY, perc_HK=HK_RECOVERY)
    column.Inlet = LiquidStream(
        str(data_path["integration"] / "pfr_test_pure_comp.json"),
        temp=FEED_TEMPERATURE, pres=COLUMN_PRESSURE, mole_flow=FEED_FLOW,
        mole_frac=PFR_FEED, verbose=False)
    control = DynamicInput()
    control.add_variable("mole_flow", lambda time: 0.0)  # [mol/s]
    column.Inlet.DynamicInlet = control
    before = dict(vars(column))

    with pytest.raises(ValueError, match="has no flow at the design time"):
        column.column_startup(RUN_TIME)

    assert vars(column).keys() == before.keys()
    for name in ("reflux", "num_plates", "num_feed"):
        assert getattr(column, name) is before[name]


@pytest.mark.unit
def test_failed_dynamic_column_solve_keeps_no_results(data_path, tmp_path):
    """A charged species without data is named, and nothing is kept.

    The holdup holds api without ``p_vap``: the initialization raises the
    species-named error before IDA starts, and the operating point that
    the shortcut design had set is withdrawn, so a corrected retry starts
    from the user's settings.
    """
    path = _pfr_database(data_path, tmp_path, API_CONSTRUCTIBLE)
    column = DynamicDistillation(
        pres=COLUMN_PRESSURE, q_feed=1.0, LK="A", HK="B",
        perc_LK=LK_RECOVERY, perc_HK=HK_RECOVERY, num_plates=DYNAMIC_PLATES,
        num_feed=DYNAMIC_FEED_PLATE)
    column.Inlet = LiquidStream(path, temp=FEED_TEMPERATURE,
                                pres=COLUMN_PRESSURE, mole_flow=FEED_FLOW,
                                mole_frac=_with_api(PFR_FEED), verbose=False)
    charge = np.array([0.3, 0.4, 0.0, 0.2, 0.1])  # [-], api charged
    column.Phases = LiquidPhase(path, temp=FEED_TEMPERATURE,
                                pres=COLUMN_PRESSURE, moles=PLATE_HOLDUP,
                                mole_frac=charge, verbose=False)
    before = {name: getattr(column, name)
              for name in ("reflux", "num_plates", "num_feed",
                           "elapsed_time")}

    with pytest.raises(MissingPropertyError, match=r"\['api'\].*'p_vap'"):
        column.solve_unit(runtime=RUN_TIME, verbose=False)

    assert {name: getattr(column, name) for name in before} == before
    for name in ("shortcut_design", "x_dist", "result", "OutletBottom"):
        assert not hasattr(column, name)


@pytest.mark.assimulo
@pytest.mark.integration
def test_dynamic_column_integrates_without_data_for_absent_species(
        data_path, tmp_path):
    """IDA integrates the column with a bare absent species.

    Earlier releases failed at the start: IDA's difference-quotient
    Jacobian perturbs api, and its missing ``p_vap`` aborted the linear
    solver setup. Bare and full data give the same trajectory; the column
    without api gives the present species' trajectory.
    """
    pytest.importorskip("assimulo")
    grid = np.linspace(0.0, RUN_TIME, NUM_OUTPUTS)  # [s]
    options = {"rtol": IDA_RTOL, "atol": IDA_ATOL}
    runs = {}
    for name, api in (("free", None), ("bare", API_BARE), ("full", API_FULL)):
        path = _pfr_database(data_path, tmp_path, api, name)
        feed = PFR_FEED if api is None else _with_api(PFR_FEED)
        column = DynamicDistillation(
            pres=COLUMN_PRESSURE, q_feed=1.0, LK="A", HK="B",
            perc_LK=LK_RECOVERY, perc_HK=HK_RECOVERY, reflux=STEADY_REFLUX,
            num_plates=DYNAMIC_PLATES, num_feed=DYNAMIC_FEED_PLATE)
        column.Inlet = LiquidStream(path, temp=FEED_TEMPERATURE,
                                    pres=COLUMN_PRESSURE,
                                    mole_flow=FEED_FLOW, mole_frac=feed,
                                    verbose=False)
        column.Phases = LiquidPhase(path, temp=FEED_TEMPERATURE,
                                    pres=COLUMN_PRESSURE,
                                    moles=PLATE_HOLDUP, mole_frac=feed,
                                    verbose=False)
        time, states, _ = column.solve_unit(time_grid=grid,
                                            sundials_opts=options,
                                            verbose=False)
        np.testing.assert_allclose(time, grid, rtol=0, atol=IDA_ATOL)
        runs[name] = (states.reshape(len(time), DYNAMIC_PLATES + 1, -1),
                      column)

    for name in ("bare", "full"):
        states, column = runs[name]
        np.testing.assert_array_equal(column.result.x_liq["api"], 0.0)
        np.testing.assert_array_equal(column.result.y_vap["api"], 0.0)
        np.testing.assert_allclose(
            np.delete(states, -1, axis=-1), runs["free"][0],
            rtol=TRAJECTORY_RTOL, atol=TRAJECTORY_ATOL)
    np.testing.assert_allclose(runs["bare"][0], runs["full"][0],
                               rtol=ROUNDING, atol=ROUND_OFF)


@pytest.mark.unit
def test_dew_point_weights_leave_absent_species_out(data_path, tmp_path):
    """Weighted dew points need no data for zero-weight species.

    The gas api has no Henry constant; unweighted, its zero fraction still
    raised. The weighted root equals the dew point of the database without
    api, and the incipient liquid holds api at exactly zero.
    """
    vapor_frac = np.array([0.5, 0.3, 0.0, 0.2])  # [-], A, B, C, solv
    reference = VaporPhase(_pfr_database(data_path, tmp_path),
                           temp=FEED_TEMPERATURE, pres=COLUMN_PRESSURE,
                           moles=VAPOR_MOLES, mole_frac=vapor_frac,
                           verbose=False)
    expected_temp, expected_liquid = reference.getDewPoint(
        COLUMN_PRESSURE, mole_frac=vapor_frac, x_liq=True)  # [K], [-]

    vapor = VaporPhase(_pfr_database(data_path, tmp_path, API_GAS, "gas"),
                       temp=FEED_TEMPERATURE, pres=COLUMN_PRESSURE,
                       moles=VAPOR_MOLES, mole_frac=_with_api(vapor_frac),
                       verbose=False)
    weights = (_with_api(vapor_frac) != 0).astype(float)  # [-]
    temp_dew, liquid = vapor.getDewPoint(
        COLUMN_PRESSURE, mole_frac=_with_api(vapor_frac), x_liq=True,
        weights=weights)  # [K], [-]

    assert temp_dew == pytest.approx(expected_temp, rel=ROUNDING)
    assert liquid[-1] == 0.0
    np.testing.assert_allclose(liquid[:-1], expected_liquid, rtol=ROUNDING,
                               atol=0)


@pytest.mark.unit
@pytest.mark.parametrize("weights", [1.0, np.ones(4), np.ones((1, 5))],
                         ids=["scalar", "short", "two-dimensional"])
def test_dew_point_rejects_weights_of_another_shape(data_path, tmp_path,
                                                    weights):
    """Weights must have exactly the shape of the mole fractions."""
    vapor_frac = _with_api(np.array([0.5, 0.3, 0.0, 0.2]))  # [-]
    vapor = VaporPhase(_pfr_database(data_path, tmp_path, API_FULL),
                       temp=FEED_TEMPERATURE, pres=COLUMN_PRESSURE,
                       moles=VAPOR_MOLES, mole_frac=vapor_frac,
                       verbose=False)

    with pytest.raises(ValueError, match="weights must have the shape"):
        vapor.getDewPoint(COLUMN_PRESSURE, mole_frac=vapor_frac,
                          weights=weights)


def _extraction_database(tmp_path, api=None, order=EXTRACTION_ORDER,
                         name="extraction"):
    """Write the synthetic extraction liquids, optionally with ``api``.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Directory for the file.
    api : dict, optional
        Property record of ``api``; None leaves api out.
    order : tuple of str, optional
        Species order, with ``api`` among them.
    name : str, optional
        File name stem.

    Returns
    -------
    str
        Path of the database.
    """
    records = dict(EXTRACTION_SPECIES, api=api)
    database = {species: records[species] for species in order
                if species != "api" or api is not None}
    return _write(tmp_path, database, name)


def _species_vector(values, order):
    """Arrange per-species values in a species order.

    Parameters
    ----------
    values : dict
        Values by species name; a missing name gets 0.0.
    order : tuple of str
        Species order.

    Returns
    -------
    numpy.ndarray
        Values in ``order``, shape ``(len(order),)``.
    """
    return np.array([values.get(species, 0.0) for species in order])


def _composition(fractions, order):
    """Map the three-species fixture fractions to a species order.

    Parameters
    ----------
    fractions : array-like
        Mole fractions [-] of a, b and c.
    order : tuple of str
        Species order, possibly with api.

    Returns
    -------
    numpy.ndarray
        Mole fractions [-] in ``order``, api at exactly zero.
    """
    return _species_vector(dict(zip(("a", "b", "c"), fractions)), order)


def _extractor(path, order, solvent_flow=None, solvent=None, eff=1.0):
    """Build a two-stage extractor with a light feed and a heavy solvent.

    Parameters
    ----------
    path : str
        Property database.
    order : tuple of str
        Species order of the database.
    solvent_flow : float, optional
        Solvent molar flow [mol/s] returned by a controller of the solvent
        stream; None leaves the stream's HEAVY_FLOW without a controller.
    solvent : dict, optional
        Solvent mole fractions [-] by species; HEAVY_INLET when None.
    eff : float, optional
        Stage efficiency [-].

    Returns
    -------
    DynamicExtractor
        Extractor with a user ``k_fun`` of the prescribed ratios.
    """
    ratios = _species_vector(EXTRACTION_RATIOS, order)  # [-]

    def k_fun(x_light, x_heavy, temp):
        """Return the prescribed distribution ratios.

        Parameters
        ----------
        x_light, x_heavy : numpy.ndarray
            Light- and heavy-phase mole fractions [-]; not used.
        temp : float or numpy.ndarray
            Temperature [K]; not used.

        Returns
        -------
        numpy.ndarray
            ``EXTRACTION_RATIOS`` in the species order [-], shape
            ``(num_species,)``, broadcast over the stages by the extractor.
        """
        return ratios

    extractor = DynamicExtractor(num_stages=2, k_fun=k_fun, eff=eff,
                                 gamma_model="ideal")
    extractor.Phases = LiquidPhase(
        path, mole_frac=_composition(
            (LIGHT_COMPOSITION + HEAVY_COMPOSITION) / 2, order),
        moles=2 * PHASE_HOLDUP, temp=INITIAL_TEMPERATURE, verbose=False)
    extractor.Inlet = {
        "feed": LiquidStream(path, mole_frac=_species_vector(LIGHT_INLET,
                                                             order),
                             mole_flow=LIGHT_FLOW, temp=FEED_TEMP,
                             verbose=False),
        "solvent": LiquidStream(
            path, mole_frac=_species_vector(
                HEAVY_INLET if solvent is None else solvent, order),
            mole_flow=HEAVY_FLOW, temp=SOLVENT_TEMP, verbose=False)}
    if solvent_flow is not None:
        control = DynamicInput()
        control.add_variable("mole_flow",
                             lambda time: solvent_flow)  # [mol/s]
        extractor.Inlet["solvent"].DynamicInlet = control
    return extractor


def _initial_states(extractor, round_off=False):
    """Initialize an extractor and flatten its states for ``unit_model``.

    Parameters
    ----------
    extractor : DynamicExtractor
        Extractor to initialize.
    round_off : bool, optional
        Set api (at ``EXTRACTION_API``) to ``ROUND_OFF`` in both phases.

    Returns
    -------
    numpy.ndarray
        Stage states: light and heavy fractions [-], internal energy [J]
        and temperature [K], flattened stage by stage.
    """
    initial = extractor.initialize_model()
    if round_off:
        for key in ("x_i", "y_i"):
            initial[key] = initial[key].copy()
            initial[key][:, EXTRACTION_API] = ROUND_OFF
    return np.column_stack(
        [initial[key] for key in extractor.name_states]).ravel()


def _without_api(states, num_stages=2, num_species=4):
    """Drop the api entries of both phases from flattened stage values.

    Parameters
    ----------
    states : numpy.ndarray
        Flattened stage values with api at ``EXTRACTION_API`` in both
        phase blocks, shape ``(..., num_stages * (2 * num_species + 2))``.
    num_stages : int, optional
        Number of stages [-].
    num_species : int, optional
        Number of species with api [-].

    Returns
    -------
    numpy.ndarray
        The same values without the api entries, flattened alike.
    """
    stages = np.reshape(states, np.shape(states)[:-1] + (num_stages, -1))
    drop = [EXTRACTION_API, num_species + EXTRACTION_API]
    kept = np.delete(stages, drop, axis=-1)
    return kept.reshape(np.shape(states)[:-1] + (-1,))


@pytest.mark.unit
def test_extractor_balances_ignore_round_off_of_absent_species(tmp_path):
    """Stage residuals with round-off api need none of its data.

    Earlier releases weighted the stage enthalpies by the state fractions,
    so the round-off api raised the species-named error for its missing
    ``cp_liq``.
    """
    free_order = ("a", "b", "c")
    reference = _extractor(_extraction_database(tmp_path, order=free_order,
                                                name="free"), free_order)
    expected = reference.unit_model(0.0, _initial_states(reference))

    residuals = {}
    for name, api in (("bare", API_BARE), ("full", API_FULL)):
        extractor = _extractor(_extraction_database(tmp_path, api, name=name),
                               EXTRACTION_ORDER)
        initial = _initial_states(extractor)
        np.testing.assert_array_equal(
            initial.reshape(2, -1)[:, [EXTRACTION_API, 4 + EXTRACTION_API]],
            0.0)
        residuals[name] = extractor.unit_model(
            0.0, _initial_states(extractor, round_off=True))

    # The closure rows sum the fractions, which adds ROUND_OFF, and the
    # energy residuals are of order 1e4 J; ROUNDING relative and absolute
    # covers both (residuals in [1/s], [-], [J/s] or [J]).
    for values in residuals.values():
        np.testing.assert_allclose(_without_api(values), expected,
                                   rtol=ROUNDING, atol=ROUNDING)
    np.testing.assert_allclose(residuals["bare"], residuals["full"],
                               rtol=ROUNDING, atol=ROUNDING)


@pytest.mark.unit
@pytest.mark.parametrize("solvent_flow,raises", [(HEAVY_FLOW, True),
                                                 (0.0, False)],
                         ids=["flowing", "no-flow"])
def test_extractor_solvent_species_is_present_only_with_flow(
        tmp_path, solvent_flow, raises):
    """A solvent that flows makes its species present; one without does not.

    The solvent carries api, which lacks ``cp_liq``; with a nonzero flow
    the stage enthalpies need it and the error names it, already in the
    initialization, which records the inlets that flow at the start;
    without flow the solvent adds nothing and api stays absent.
    """
    path = _extraction_database(tmp_path, API_CONSTRUCTIBLE)
    extractor = _extractor(path, EXTRACTION_ORDER, solvent_flow,
                           API_SOLVENT)
    reference = _extractor(_extraction_database(tmp_path, API_FULL,
                                                name="full"),
                           EXTRACTION_ORDER, solvent_flow, API_SOLVENT)
    states = _initial_states(reference, round_off=True)

    if raises:
        with pytest.raises(MissingPropertyError,
                           match=r"\['api'\].*'cp_liq'"):
            extractor.initialize_model()
        with pytest.raises(MissingPropertyError,
                           match=r"\['api'\].*'cp_liq'"):
            extractor.unit_model(0.0, states)
    else:
        extractor.initialize_model()
        np.testing.assert_allclose(extractor.unit_model(0.0, states),
                                   reference.unit_model(0.0, states),
                                   rtol=ROUNDING, atol=ROUNDING)


@pytest.mark.unit
def test_extractor_publishes_absent_species_at_zero(tmp_path):
    """Published fractions and outlets hold the absent species at zero.

    The bare api has no ``rho_liq``, so outlets built from its round-off
    raised in earlier releases.
    """
    extractor = _extractor(_extraction_database(tmp_path, API_BARE),
                           EXTRACTION_ORDER)
    states = _initial_states(extractor, round_off=True)
    time = np.array([0.0, RUN_TIME])  # [s]

    extractor.retrieve_results(time, np.vstack([states, states]))

    for key in ("x_i", "y_i"):
        np.testing.assert_array_equal(
            getattr(extractor.result, key)["api"], 0.0)
        np.testing.assert_array_equal(
            extractor.profiles_runs[-1][key]["api"], 0.0)
    np.testing.assert_array_equal(
        extractor.outputs["mole_frac"][:, EXTRACTION_API], 0.0)
    for outlet in extractor.Outlet.values():
        assert outlet.mole_frac[EXTRACTION_API] == 0.0


@pytest.mark.unit
def test_extractor_closure_species_needs_its_data(tmp_path):
    """The last species closes both phases, so it is always present.

    api is last, neither charged nor fed, and lacks ``rho_liq``: the
    initialization names it before IDA starts and the solve keeps no
    results. Earlier releases started IDA, whose closure gave api nonzero
    fractions and then failed without naming it.
    """
    order = ("a", "b", "c", "api")
    extractor = _extractor(_extraction_database(tmp_path, API_BARE, order),
                           order)

    with pytest.raises(MissingPropertyError, match=r"\['api'\].*'rho_liq'"):
        extractor.solve_unit(RUN_TIME, verbose=False)

    assert extractor.fixed_vals == {}
    assert extractor.profiles_runs == []
    for name in ("target_states", "result", "outputs", "Outlet"):
        assert not hasattr(extractor, name)


@pytest.mark.unit
def test_extractor_checks_species_fed_at_the_start_before_ida(tmp_path):
    """A species fed from time 0 needs its density before IDA starts.

    A solvent controller supplies api, which lacks ``rho_liq``, from the
    start; the stored solvent holds none. The initialization records the
    inlets evaluated at time 0 before its density check, so it names api;
    earlier, IDA ran and the outlets failed only at publication.
    """
    api = dict(API_BARE, cp_liq=API_FULL["cp_liq"])  # no rho_liq
    extractor = _extractor(_extraction_database(tmp_path, api),
                           EXTRACTION_ORDER)
    control = DynamicInput()
    control.add_variable(
        "mole_frac",
        lambda time: _species_vector(API_SOLVENT, EXTRACTION_ORDER))  # [-]
    extractor.Inlet["solvent"].DynamicInlet = control

    with pytest.raises(MissingPropertyError, match=r"\['api'\].*'rho_liq'"):
        extractor.initialize_model()

    extractor = _extractor(_extraction_database(tmp_path, api, name="solve"),
                           EXTRACTION_ORDER)
    extractor.Inlet["solvent"].DynamicInlet = control
    with pytest.raises(MissingPropertyError, match=r"\['api'\].*'rho_liq'"):
        extractor.solve_unit(RUN_TIME, verbose=False)
    assert extractor.fixed_vals == {}
    assert extractor.profiles_runs == []


def _partial_efficiency_extractor(path):
    """Build an extractor with eff < 1 and a light feed that never flows.

    Parameters
    ----------
    path : str
        Property database with api in ``EXTRACTION_ORDER``.

    Returns
    -------
    DynamicExtractor
        Extractor with ``PARTIAL_EFFICIENCY`` [-] whose light feed holds
        api (API_IN_FEED [-], taken from a) under a zero-flow controller.
    """
    extractor = _extractor(path, EXTRACTION_ORDER, eff=PARTIAL_EFFICIENCY)
    feed = dict(LIGHT_INLET, api=API_IN_FEED,
                a=LIGHT_INLET["a"] - API_IN_FEED)  # [-]
    extractor.Inlet = {"feed": LiquidStream(
        path, mole_frac=_species_vector(feed, EXTRACTION_ORDER),
        mole_flow=LIGHT_FLOW, temp=FEED_TEMP, verbose=False)}
    control = DynamicInput()
    control.add_variable("mole_flow", lambda time: 0.0)  # [mol/s]
    extractor.Inlet["feed"].DynamicInlet = control
    return extractor


@pytest.mark.unit
def test_partial_efficiency_light_inlet_species_need_their_data(tmp_path):
    """With eff < 1 the light inlet's species are present, flow or not.

    The first stage's Murphree term carries the light inlet composition
    into the stages whatever the inlet's flow, so api of a light feed that
    never flows is present, and its missing ``cp_liq`` is named before IDA
    starts; earlier it was called absent and its enthalpy silently dropped.
    """
    extractor = _partial_efficiency_extractor(
        _extraction_database(tmp_path, API_CONSTRUCTIBLE))

    with pytest.raises(MissingPropertyError, match=r"\['api'\].*'cp_liq'"):
        extractor.initialize_model()


@pytest.mark.assimulo
@pytest.mark.integration
def test_partial_efficiency_light_inlet_species_are_published(tmp_path):
    """The Murphree-carried api keeps its integrated fractions.

    Earlier the published fractions of api were projected to zero although
    the stages held it (about 0.026 in the heavy phase).
    """
    pytest.importorskip("assimulo")
    extractor = _partial_efficiency_extractor(
        _extraction_database(tmp_path, API_FULL))
    time, states = extractor.solve_unit(
        RUN_TIME, sundials_opts={"rtol": IDA_RTOL, "atol": IDA_ATOL},
        verbose=False)
    stages = states.reshape(len(time), 2, -1)  # x_i, y_i [-], u_int [J], [K]
    api = EXTRACTION_API  # [-]

    assert np.any(stages[:, :, len(EXTRACTION_ORDER) + api] != 0.0)
    np.testing.assert_array_equal(extractor.result.x_i["api"],
                                  stages[:, :, api])
    np.testing.assert_array_equal(extractor.result.y_i["api"],
                                  stages[:, :, len(EXTRACTION_ORDER) + api])


@pytest.mark.assimulo
@pytest.mark.integration
def test_extractor_solve_rebuilds_presence(tmp_path):
    """A species latched before the solve needs no data in the solve.

    The solvent carries api without ``cp_liq``; one residual evaluation
    latches api and raises. With the solvent replaced by one without api,
    the solve rebuilds the record from the charge and needs no data for it.
    """
    pytest.importorskip("assimulo")
    path = _extraction_database(tmp_path, API_CONSTRUCTIBLE)
    extractor = _extractor(path, EXTRACTION_ORDER, solvent=API_SOLVENT)
    with pytest.raises(MissingPropertyError, match=r"\['api'\].*'cp_liq'"):
        extractor.unit_model(0.0, _initial_states(extractor, round_off=True))

    extractor.Inlet = {"solvent": LiquidStream(
        path, mole_frac=_species_vector(HEAVY_INLET, EXTRACTION_ORDER),
        mole_flow=HEAVY_FLOW, temp=SOLVENT_TEMP, verbose=False)}
    extractor.solve_unit(RUN_TIME, verbose=False)

    np.testing.assert_array_equal(extractor.result.x_i["api"], 0.0)


@pytest.mark.assimulo
@pytest.mark.integration
def test_extractor_publishes_the_closure_species_with_its_value(tmp_path):
    """The closure species keeps its integrated fractions when published.

    api is last, neither charged nor fed: its fractions are the
    normalization closures of both phases, which the constant-flow
    approximation (#195) leaves nonzero during the transient, so they are
    published as integrated, not projected to zero.
    """
    pytest.importorskip("assimulo")
    order = ("a", "b", "c", "api")
    extractor = _extractor(_extraction_database(tmp_path, API_FULL, order),
                           order)
    options = {"rtol": IDA_RTOL, "atol": IDA_ATOL}
    time, states = extractor.solve_unit(RUN_TIME, sundials_opts=options,
                                        verbose=False)
    stages = states.reshape(len(time), 2, -1)  # x_i, y_i [-], u_int [J], [K]
    api = order.index("api")  # [-]

    assert np.any(stages[:, :, api] != 0.0)
    np.testing.assert_array_equal(extractor.result.x_i["api"],
                                  stages[:, :, api])
    np.testing.assert_array_equal(extractor.result.y_i["api"],
                                  stages[:, :, len(order) + api])


@pytest.mark.assimulo
@pytest.mark.integration
def test_extractor_integrates_without_data_for_absent_species(tmp_path):
    """IDA integrates the extractor with a bare absent species.

    Earlier releases failed at the start, when IDA's Jacobian perturbed
    api and its missing ``cp_liq`` aborted the linear solver setup.
    """
    pytest.importorskip("assimulo")
    options = {"rtol": IDA_RTOL, "atol": IDA_ATOL}
    runs = {}
    for name, api, order in (("free", None, ("a", "b", "c")),
                             ("bare", API_BARE, EXTRACTION_ORDER),
                             ("full", API_FULL, EXTRACTION_ORDER)):
        extractor = _extractor(_extraction_database(tmp_path, api, order,
                                                    name), order)
        time, states = extractor.solve_unit(RUN_TIME, sundials_opts=options,
                                            verbose=False)
        runs[name] = (time, states, extractor)

    # IDA chooses its own output times, which round-off differences in
    # the residuals can move, so the runs are compared at their end.
    free_time, free_states, _ = runs["free"]
    for name in ("bare", "full"):
        time, states, extractor = runs[name]
        np.testing.assert_array_equal(extractor.result.x_i["api"], 0.0)
        np.testing.assert_array_equal(extractor.result.y_i["api"], 0.0)
        assert time[-1] == pytest.approx(free_time[-1], rel=ROUNDING)
        np.testing.assert_allclose(_without_api(states[-1]), free_states[-1],
                                   rtol=TRAJECTORY_RTOL,
                                   atol=TRAJECTORY_ATOL)
    np.testing.assert_allclose(runs["bare"][1][-1], runs["full"][1][-1],
                               rtol=TRAJECTORY_RTOL, atol=TRAJECTORY_ATOL)


def _uniquac_extractor(tmp_path, api, name):
    """Build a default-ratio UNIQUAC extractor on the ternary fixture.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Directory for the database.
    api : dict or None
        Property record of ``api``, inserted at ``EXTRACTION_API`` so that
        it is not the closure species; None leaves it out.
    name : str
        File name stem.

    Returns
    -------
    DynamicExtractor
        Extractor charged with ``TERNARY_FEED`` (api at zero) at
        INITIAL_TEMPERATURE [K].
    """
    path = _uniquac_database(tmp_path, THERMO_NONIDEAL_TERNARY, api, name,
                             index=EXTRACTION_API)
    unit = DynamicExtractor(num_stages=2, gamma_model="UNIQUAC")
    fractions = (TERNARY_FEED if api is None
                 else _with_api(TERNARY_FEED, EXTRACTION_API))  # [-]
    unit.Phases = LiquidPhase(path, mole_frac=fractions,
                              moles=2 * PHASE_HOLDUP,
                              temp=INITIAL_TEMPERATURE, verbose=False)
    return unit


@pytest.mark.unit
def test_default_extractor_ratio_evaluates_present_species_only(tmp_path):
    """The default UNIQUAC ``k_fun`` gives an absent species the ratio 1.

    Earlier releases evaluated UNIQUAC on every species, so the bare api's
    missing ``ri`` and ``qi`` made every ratio NaN.
    """
    stage_light = np.array([[0.5, 0.3, 0.2], [0.4, 0.4, 0.2]])  # [-]
    stage_heavy = np.array([[0.2, 0.6, 0.2], [0.3, 0.5, 0.2]])  # [-]
    temps = np.array([460.0, 470.0])  # [K]

    expected = _uniquac_extractor(tmp_path, None, "free").k_fun(
        stage_light, stage_heavy, temps)  # [-]
    for name, api in (("bare", API_BARE), ("full", API_FULL)):
        ratios = _uniquac_extractor(tmp_path, api, name).k_fun(
            np.insert(stage_light, EXTRACTION_API, ROUND_OFF, axis=-1),
            np.insert(stage_heavy, EXTRACTION_API, ROUND_OFF, axis=-1),
            temps)  # [-]
        np.testing.assert_array_equal(ratios[:, EXTRACTION_API], 1.0)
        np.testing.assert_allclose(np.delete(ratios, EXTRACTION_API, axis=1),
                                   expected, rtol=ROUNDING, atol=0)


def _uniquac_fed_extractor(tmp_path, api, name):
    """Build a default-ratio UNIQUAC extractor whose solvent starts later.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Directory for the database.
    api : dict
        Property record of ``api``, second in ``LLE_ORDER``.
    name : str
        File name stem.

    Returns
    -------
    unit : DynamicExtractor
        Initialized extractor on the partially miscible fixture, charged
        with ``LLE_FEED`` (api at zero) at LLE_TEMP [K], with a w-rich feed
        without api and an s-rich solvent holding api whose controller
        gives it a molar flow of HEAVY_FLOW [mol/s] from FEED_ON [s] on and
        none before.
    stages : dict
        Its initial stage states: fractions [-], energy [J], temperature
        [K].
    """
    path = _lle_database(tmp_path, api, name)
    unit = DynamicExtractor(num_stages=2, gamma_model="UNIQUAC")
    unit.Phases = LiquidPhase(
        path, mole_frac=_species_vector(LLE_FEED, LLE_ORDER),
        moles=LLE_AMOUNT, temp=LLE_TEMP, verbose=False)
    feed = _species_vector({"w": 0.8, "s": 0.1, "c": 0.1},
                           LLE_ORDER)  # [-]
    solvent = _species_vector({"w": 0.1, "api": 0.1, "s": 0.7, "c": 0.1},
                              LLE_ORDER)  # [-]
    unit.Inlet = {
        "feed": LiquidStream(path, mole_frac=feed, mole_flow=LIGHT_FLOW,
                             temp=LLE_TEMP, verbose=False),
        "solvent": LiquidStream(path, mole_frac=solvent,
                                mole_flow=HEAVY_FLOW, temp=LLE_TEMP,
                                verbose=False)}
    control = DynamicInput()
    control.add_variable(
        "mole_flow",
        lambda time: HEAVY_FLOW if time >= FEED_ON else 0.0)  # [mol/s]
    unit.Inlet["solvent"].DynamicInlet = control
    with _seeded_rng(LLE_SEED):
        stages = unit.initialize_model()
    return unit, stages


@pytest.mark.unit
def test_default_extractor_ratio_presence_is_time_resolved(tmp_path):
    """The default ratio uses the presence of each evaluation's time.

    api, in a solvent that flows from FEED_ON on, has ``cp_liq`` and
    ``rho_liq`` but no UNIQUAC parameters. From FEED_ON on its missing
    ``ri`` is named; an earlier evaluation, also after that latch, needs
    none and equals the evaluation with full data.
    """
    api = dict(API_CONSTRUCTIBLE, cp_liq=API_FULL["cp_liq"])  # no ri, qi
    unit, stages = _uniquac_fed_extractor(tmp_path, api, "no-uniquac")
    full, _ = _uniquac_fed_extractor(tmp_path, API_FULL, "full")
    api_index = LLE_ORDER.index("api")  # [-]
    stages["x_i"][:, api_index] = ROUND_OFF  # [-]
    stages["y_i"][:, api_index] = ROUND_OFF  # [-]
    states = np.column_stack([stages[key]
                              for key in unit.name_states]).ravel()

    with pytest.raises(MissingPropertyError, match=r"\['api'\].*'ri'"):
        unit.unit_model(FEED_ON, states)

    np.testing.assert_allclose(unit.unit_model(BEFORE_FEED, states),
                               full.unit_model(BEFORE_FEED, states),
                               rtol=ROUNDING, atol=ROUNDING)


@pytest.mark.unit
def test_extractor_reinitialization_uses_the_start_presence(tmp_path):
    """A re-initialization needs no data for species latched later.

    After an evaluation at FEED_ON latches api (and raises for its missing
    ``ri``), initializing again evaluates the batch split at time 0, where
    api is absent, so it needs no UNIQUAC parameters for it and repeats
    the first initialization.
    """
    api = dict(API_CONSTRUCTIBLE, cp_liq=API_FULL["cp_liq"])  # no ri, qi
    unit, stages = _uniquac_fed_extractor(tmp_path, api, "no-uniquac")
    states = np.column_stack([stages[key]
                              for key in unit.name_states]).ravel()
    with pytest.raises(MissingPropertyError, match=r"\['api'\].*'ri'"):
        unit.unit_model(FEED_ON, states)

    with _seeded_rng(LLE_SEED):
        again = unit.initialize_model()

    for key in unit.name_states:
        np.testing.assert_allclose(again[key], stages[key], rtol=ROUNDING,
                                   atol=0)


def _lle_database(tmp_path, api=None, name="lle"):
    """Write the synthetic partially miscible ternary, optionally with api.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Directory for the file.
    api : dict, optional
        Property record of ``api``, inserted second; None leaves it out.
    name : str, optional
        File name stem.

    Returns
    -------
    str
        Path of the database.
    """
    order = LLE_ORDER if api is not None else ("w", "s", "c")
    database = {species: (api if species == "api" else LLE_SPECIES[species])
                for species in order}
    amk = np.zeros((len(order), len(order)))  # [J/mol]
    w_index, s_index = order.index("w"), order.index("s")
    amk[w_index, s_index] = amk[s_index, w_index] = LLE_INTERACTION
    if api is not None and "ri" in api:
        api_index = order.index("api")
        amk[api_index, :] = API_INTERACTION_ROW
        amk[:, api_index] = API_INTERACTION_COLUMN
        amk[api_index, api_index] = 0.0
    database["interaction"] = {"amk": amk.tolist()}
    return _write(tmp_path, database, name)


def _steady_extractor(cls, path, order):
    """Build and solve a batch or continuous UNIQUAC extractor.

    Parameters
    ----------
    cls : type
        ``BatchExtractor`` or ``ContinuousExtractor``.
    path : str
        Property database.
    order : tuple of str
        Species order of the database.

    Returns
    -------
    ContinuousExtractor
        The solved extractor.

    Notes
    -----
    The random initial split is drawn with NumPy's global generator seeded
    with LLE_SEED, whose state is restored afterwards: the substitution
    does not converge from every seed (issue #453).
    """
    unit = cls(gamma_method="UNIQUAC")
    feed = _species_vector(LLE_FEED, order)  # [-]
    if cls is BatchExtractor:
        unit.Phases = LiquidPhase(path, mole_frac=feed, moles=LLE_AMOUNT,
                                  temp=LLE_TEMP, verbose=False)
    else:
        unit.Inlet = LiquidStream(path, mole_frac=feed, mole_flow=LLE_AMOUNT,
                                  temp=LLE_TEMP, verbose=False)
    with _seeded_rng(LLE_SEED):
        unit.solve_unit({"tol": LLE_TOL, "max_iter": LLE_MAX_ITER})
    return unit


@pytest.mark.unit
@pytest.mark.parametrize("cls", [BatchExtractor, ContinuousExtractor])
def test_steady_extractor_needs_no_activity_data_for_absent_species(
        tmp_path, cls):
    """The liquid split ignores a species without feed and its data.

    Earlier releases evaluated UNIQUAC on every species, so the bare api's
    missing parameters made the ratios NaN and the split failed. The
    products must be a real split of the feed that closes the material
    balance and the isoactivity of the present species, with activity
    coefficients from the ternary database without api, and bare and full
    data must give the same split. The solves use a fixed seed of the
    random initial split, which does not converge from every seed (issue
    #453).
    """
    free = LiquidPhase(_lle_database(tmp_path), mole_frac=[
        LLE_FEED[name] for name in ("w", "s", "c")], moles=LLE_AMOUNT,
        temp=LLE_TEMP, verbose=False)
    feed = _species_vector(LLE_FEED, LLE_ORDER)  # [-]
    api_index = LLE_ORDER.index("api")
    results = {}
    for name, api in (("bare", API_BARE), ("full", API_FULL)):
        result = _steady_extractor(cls, _lle_database(tmp_path, api, name),
                                   LLE_ORDER).result
        results[name] = result
        heavy, light = result.x_heavy, result.x_light  # [-]
        assert heavy[api_index] == 0.0
        assert light[api_index] == 0.0
        assert 0 < result.mol_light < LLE_AMOUNT  # [mol] or [mol/s]
        assert np.max(np.abs(heavy - light)) > MIN_LLE_SEPARATION  # [-]
        np.testing.assert_allclose(
            result.mol_heavy * heavy + result.mol_light * light,
            LLE_AMOUNT * feed, rtol=0, atol=LLE_ATOL)  # [mol] or [mol/s]
        present_heavy = np.delete(heavy, api_index)  # [-]
        present_light = np.delete(light, api_index)  # [-]
        activity_heavy = present_heavy * free.getActivityCoeff(
            "UNIQUAC", mole_frac=present_heavy, temp=LLE_TEMP)  # [-]
        activity_light = present_light * free.getActivityCoeff(
            "UNIQUAC", mole_frac=present_light, temp=LLE_TEMP)  # [-]
        np.testing.assert_allclose(activity_heavy, activity_light, rtol=0,
                                   atol=LLE_ATOL)

    for key in ("x_heavy", "x_light", "mol_heavy", "mol_light"):
        np.testing.assert_allclose(getattr(results["bare"], key),
                                   getattr(results["full"], key), rtol=0,
                                   atol=LLE_ATOL)


@pytest.mark.unit
def test_steady_extractor_names_a_fed_species_without_activity_data(
        tmp_path):
    """A fed species without UNIQUAC parameters raises the named error.

    Earlier releases propagated its NaN parameters into every ratio, and
    the split failed with a Newton convergence error.
    """
    api = dict(API_CONSTRUCTIBLE)  # no ri, qi or qip
    path = _lle_database(tmp_path, api)
    unit = BatchExtractor(gamma_method="UNIQUAC")
    feed = _species_vector(dict(LLE_FEED, api=0.05, w=0.45), LLE_ORDER)
    unit.Phases = LiquidPhase(path, mole_frac=feed, moles=LLE_AMOUNT,
                              temp=LLE_TEMP, verbose=False)

    with pytest.raises(MissingPropertyError, match=r"\['api'\].*'ri'"):
        unit.solve_unit({"tol": LLE_TOL, "max_iter": LLE_MAX_ITER})


@pytest.mark.unit
def test_steady_extractor_ratio_follows_the_composition_split(tmp_path):
    """The default ratio's presence comes from the composition split.

    After a solve on a feed holding api, the charge is replaced by one
    from a database whose api has no UNIQUAC parameters and is absent. A
    direct ``material_balance`` call on the new composition needs none of
    them and equals the split of the ternary without api; earlier the
    presence of the earlier solve stayed and the missing ``ri`` raised.
    """
    unit = BatchExtractor(gamma_method="UNIQUAC")
    unit.Phases = LiquidPhase(
        _lle_database(tmp_path, API_FULL, "full"),
        mole_frac=_species_vector(dict(LLE_FEED, api=0.05, w=0.45),
                                  LLE_ORDER),
        moles=LLE_AMOUNT, temp=LLE_TEMP, verbose=False)
    with _seeded_rng(LLE_SEED):
        unit.solve_unit({"tol": LLE_TOL, "max_iter": LLE_MAX_ITER})

    feed = _species_vector(LLE_FEED, LLE_ORDER)  # [-], api absent
    unit.Phases = LiquidPhase(_lle_database(tmp_path, API_BARE, "bare"),
                              mole_frac=feed, moles=LLE_AMOUNT,
                              temp=LLE_TEMP, verbose=False)
    phi, x_1, x_2, _ = unit.material_balance(
        LLE_PHI_SEED, LLE_X1_SEED, LLE_X2_SEED, feed, LLE_TEMP)

    reference = _steady_extractor(BatchExtractor, _lle_database(tmp_path),
                                  ("w", "s", "c"))
    api_index = LLE_ORDER.index("api")  # [-]
    expected = reference.material_balance(
        LLE_PHI_SEED, np.delete(LLE_X1_SEED, api_index),
        np.delete(LLE_X2_SEED, api_index), np.delete(feed, api_index),
        LLE_TEMP)
    assert x_1[api_index] == 0.0
    assert x_2[api_index] == 0.0
    assert phi == pytest.approx(expected[0], abs=LLE_ATOL)
    np.testing.assert_allclose(np.delete(x_1, api_index), expected[1],
                               rtol=0, atol=LLE_ATOL)
    np.testing.assert_allclose(np.delete(x_2, api_index), expected[2],
                               rtol=0, atol=LLE_ATOL)


@pytest.mark.unit
def test_extractor_publishes_each_row_with_its_presence(tmp_path):
    """Published rows use the presence of their own time.

    The solvent, which carries api, flows from FEED_ON on; an evaluation
    there latches api. A row before FEED_ON publishes api at exactly zero,
    a row at it publishes the state value.
    """
    extractor = _extractor(_extraction_database(tmp_path, API_FULL),
                           EXTRACTION_ORDER, solvent=API_SOLVENT)
    control = DynamicInput()
    control.add_variable(
        "mole_flow",
        lambda time: HEAVY_FLOW if time >= FEED_ON else 0.0)  # [mol/s]
    extractor.Inlet["solvent"].DynamicInlet = control
    states = _initial_states(extractor, round_off=True)
    extractor.unit_model(FEED_ON, states)

    extractor.retrieve_results(np.array([BEFORE_FEED, FEED_ON]),
                               np.vstack([states, states]))

    for key in ("x_i", "y_i"):
        published = getattr(extractor.result, key)["api"]  # [-], by time
        np.testing.assert_array_equal(published[0], 0.0)
        np.testing.assert_array_equal(published[1], ROUND_OFF)


@pytest.mark.unit
def test_dew_point_seed_leaves_absent_species_out(data_path, tmp_path):
    """An absent species with an infinite saturation temperature is ignored.

    api's Antoine A equals log10 of the pressure, so its saturation
    temperature at that pressure divides by zero; in the seed, zero times
    infinity would be NaN. Weighted, the dew point equals the one of the
    database without api, and the incipient liquid holds api at zero.
    """
    vapor_frac = np.array([0.5, 0.3, 0.0, 0.2])  # [-], A, B, C, solv
    reference = VaporPhase(_pfr_database(data_path, tmp_path),
                           temp=FEED_TEMPERATURE, pres=COLUMN_PRESSURE,
                           moles=VAPOR_MOLES, mole_frac=vapor_frac,
                           verbose=False)
    expected_temp, expected_liquid = reference.getDewPoint(
        COLUMN_PRESSURE, mole_frac=vapor_frac, x_liq=True)  # [K], [-]

    # Antoine A [-] = log10(pressure / [Pa]), B [K] and C [K] as API_FULL.
    api = dict(API_FULL, p_vap=[np.log10(COLUMN_PRESSURE),
                                *API_FULL["p_vap"][1:]])
    vapor = VaporPhase(_pfr_database(data_path, tmp_path, api, "infinite"),
                       temp=FEED_TEMPERATURE, pres=COLUMN_PRESSURE,
                       moles=VAPOR_MOLES, mole_frac=_with_api(vapor_frac),
                       verbose=False)
    temp_dew, liquid = vapor.getDewPoint(
        COLUMN_PRESSURE, mole_frac=_with_api(vapor_frac), x_liq=True,
        weights=(_with_api(vapor_frac) != 0).astype(float))  # [K], [-]

    assert temp_dew == pytest.approx(expected_temp, rel=ROUNDING)
    assert liquid[-1] == 0.0
    np.testing.assert_allclose(liquid[:-1], expected_liquid, rtol=ROUNDING,
                               atol=0)
