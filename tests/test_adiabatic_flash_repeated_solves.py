"""Repeated AdiabaticFlash solves with a supercritical species (issue #319).

``AdiabaticFlash.solve_unit`` removes species that are supercritical at the
feed temperature from its composition unknowns. It used to reduce
``num_comp`` on every call, so a second solve of the same feed failed with
``IndexError``; this is reachable through a flowsheet re-solve of
``Mixer --> AdiabaticFlash`` (issue #445). The active species are now
derived from the current feed on every solve: repeated solves, a
re-attached feed and a re-solved flowsheet give the same products, and
``num_comp`` stays the number of feed species. A flowsheet re-solve
re-attaches the transferred feed, which used to restore ``num_comp``
before the next solve, so there the defect showed as a reduced
``num_comp`` after each solve rather than as an ``IndexError``. A later solve also leaves
the products published by an earlier solve unchanged. Solves from other
valid vapor-fraction seeds, also after re-attaching the feed, agree with
the first within the solver's acceptance: twice the residual allowance
sqrt(float64 epsilon) per unit feed for amounts and mole fractions, and
the energy residual it allows (times ``div_energybce``) over the smallest
binary heat capacity, 75 J/mol/K, for the temperature.

Fixtures:

* The issue reproduction: a synthetic light/heavy binary (the numerical
  contract fixture of the issue, not calibrated data; units in
  ``BINARY``) with the packaged nitrogen (data/evaporator/props_nitrogen.json,
  critical temperature 126.14 K), 100 mol at 370 K and 1 atm with mole
  fractions 0.40/0.55/0.05, seeded from the feed. Its subcritical control
  is the binary alone at 350 K, flashed at 300 Pa.
* The flowsheet: the shipped five-species database with the packaged
  nitrogen appended, two feeds at 360 K and 1 atm carrying 1 % nitrogen,
  mixed by a continuous ``Mixer``; nitrogen is the only supercritical
  species, so the flash solves 2 * 5 + 3 unknowns.

Material closure is checked against the feed, liquid x + vapor y = feed z,
to the flash residual allowance sqrt(float64 epsilon) of
PharmaPy.Evaporators times the feed amount. No ODE backend is needed.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from PharmaPy.Containers import Mixer
from PharmaPy.Evaporators import AdiabaticFlash
from PharmaPy.Phases import LiquidPhase
from PharmaPy.SimExec import SimulationExec
from PharmaPy.Streams import LiquidStream

pytestmark = pytest.mark.unit

NITROGEN_PATH = (Path(__file__).resolve().parents[1] / 'data' / 'evaporator'
                 / 'props_nitrogen.json')
# Issue #319 synthetic binary: mw [g/mol], t_crit and tref_hvap [K],
# rho_liq [kg/m**3], constant cp_liq [J/mol/K], delta_hvap [J/mol], and
# p_vap = [A, B, C] of log10(P/Pa) = A - B/(T + C), B and C in K.
BINARY = {
    'light': {'mw': 18., 't_crit': 650., 'rho_liq': 1000., 'cp_liq': [75.],
              'p_vap': [8., 1500., -40.], 'delta_hvap': 40000.,
              'tref_hvap': 350.},
    'heavy': {'mw': 100., 't_crit': 700., 'rho_liq': 900., 'cp_liq': [150.],
              'p_vap': [8., 1800., -40.], 'delta_hvap': 60000.,
              'tref_hvap': 350.},
    }
ATMOSPHERE = 101325.0  # [Pa]
FEED_MOLES = 100.0  # [mol], issue #319 feed
NITROGEN_TEMP = 370.0  # [K], nitrogen case of issue #319
NITROGEN_FRACTIONS = np.array([.40, .55, .05])  # [-], light/heavy/nitrogen
CONTROL_TEMP = 350.0  # [K], subcritical control of issue #319
CONTROL_PRESSURE = 300.0  # [Pa], subcritical control of issue #319
CONTROL_FRACTIONS = np.array([.40, .60])  # [-], light/heavy
CHANGED_FRACTIONS = np.array([.45, .50, .05])  # [-], in-place feed change
REPEATS = 3  # [-], more than two so a single reset is not enough
FLOW_1 = 1.0  # [mol/s]
FLOW_2 = 0.5  # [mol/s]
FLOWSHEET_TEMP = 360.0  # [K]
FEED_1 = np.array([.30, .10, .29, .10, .20, .01])  # [-], 1 % nitrogen
FEED_2 = np.array([.10, .30, .10, .29, .20, .01])  # [-], 1 % nitrogen
NUM_ACTIVE = 5  # [-], the shipped species; nitrogen is supercritical
# [-], flash residual allowance of PharmaPy.Evaporators, relative to the
# feed amount, for the material closure.
CLOSURE_RTOL = np.sqrt(np.finfo(float).eps)
# Valid vapor-fraction seeds [-] in (0, 1): the default first, then lower
# and higher; the last is solved after re-attaching the feed.
SEEDS = (0.5, 0.2, 0.8)
REATTACHED_SEED = 0.3
# [-], divisor of the molar energy residual [J/mol]; the constructor
# default, passed explicitly because the temperature tolerance uses it.
ENERGY_SCALE = 1e3
# [J/mol/K], smallest liquid heat capacity of the binary (light species).
MIN_HEAT_CAPACITY = 75.0
# Each accepted solve has scaled residuals within CLOSURE_RTOL [-], so two
# solutions from different seeds may differ by twice that: in phase
# amounts per unit feed and in mole fractions [-], and in temperature by
# the energy residual [J/mol] it allows over the smallest heat capacity.
SEED_FRACTION_ATOL = 2 * CLOSURE_RTOL  # [-]
SEED_TEMP_ATOL = (2 * CLOSURE_RTOL * ENERGY_SCALE
                  / MIN_HEAT_CAPACITY)  # [K]


@pytest.fixture
def binary_path(tmp_path):
    """Write the issue's synthetic binary database.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary directory.

    Returns
    -------
    str
        Database path.
    """
    path = tmp_path / 'binary.json'
    path.write_text(json.dumps(BINARY))
    return str(path)


def _feed(binary_path, nitrogen=True):
    """Build the issue's batch feed.

    Parameters
    ----------
    binary_path : str
        Binary database path.
    nitrogen : bool, optional
        The nitrogen case (370 K, 1 atm) or the subcritical control
        (350 K, 300 Pa).

    Returns
    -------
    LiquidPhase
        FEED_MOLES [mol] of the case's composition.
    """
    if nitrogen:
        return LiquidPhase([binary_path, str(NITROGEN_PATH)],
                           temp=NITROGEN_TEMP, pres=ATMOSPHERE,
                           moles=FEED_MOLES, mole_frac=NITROGEN_FRACTIONS)
    return LiquidPhase(binary_path, temp=CONTROL_TEMP, pres=CONTROL_PRESSURE,
                       moles=FEED_MOLES, mole_frac=CONTROL_FRACTIONS)


def _products(flash):
    """Return the published products of a solved flash.

    Parameters
    ----------
    flash : AdiabaticFlash
        Solved flash.

    Returns
    -------
    tuple
        Liquid and vapor amounts [mol], their mole fractions [-] and the
        drum temperature [K].
    """
    liquid, vapor = flash.LiquidOut, flash.VaporOut
    return (liquid.moles, vapor.moles, liquid.mole_frac.copy(),
            vapor.mole_frac.copy(), liquid.temp)


def _assert_same(products, reference):
    """Assert two product tuples are bitwise identical.

    Parameters
    ----------
    products, reference : tuple
        Tuples returned by ``_products``.

    Notes
    -----
    Use only for solves of the same feed from the same seed: they repeat
    the same deterministic MINPACK iterations, so no tolerance applies.
    Solves from different seeds are compared by ``_assert_close``.
    """
    for value, expected in zip(products, reference):
        np.testing.assert_array_equal(value, expected)


@pytest.mark.parametrize('nitrogen', [True, False],
                         ids=['nitrogen', 'subcritical-control'])
def test_repeated_solves_give_the_same_products(binary_path, nitrogen):
    """Solve the same feed again, and after re-attaching it."""
    feed = _feed(binary_path, nitrogen)
    drum_pressure = ATMOSPHERE if nitrogen else CONTROL_PRESSURE  # [Pa]
    flash = AdiabaticFlash(drum_pressure, seed_basedon_input=True)
    flash.Inlet = feed
    flash.solve_unit()
    first = _products(flash)

    for _ in range(REPEATS - 1):
        flash.solve_unit()
        _assert_same(_products(flash), first)
    flash.Inlet = feed
    flash.solve_unit()
    _assert_same(_products(flash), first)

    assert flash.num_comp == len(feed.name_species)
    assert flash.LiquidOut.name_species == feed.name_species
    liquid, vapor, x_liq, y_vap, _ = first
    assert 0 < vapor < FEED_MOLES
    assert x_liq.sum() == pytest.approx(1, abs=CLOSURE_RTOL)
    assert y_vap.sum() == pytest.approx(1, abs=CLOSURE_RTOL)
    np.testing.assert_allclose(liquid * x_liq + vapor * y_vap,
                               FEED_MOLES * feed.mole_frac, rtol=0,
                               atol=CLOSURE_RTOL * FEED_MOLES)
    if nitrogen:
        assert x_liq[-1] == 0  # nitrogen only in the vapor


def _assert_close(products, reference):
    """Assert two product tuples agree within the solver's acceptance.

    Parameters
    ----------
    products, reference : tuple
        Tuples returned by ``_products`` for solves of the same feed of
        FEED_MOLES [mol] from different seeds.
    """
    liquid, vapor, x_liq, y_vap, temp = products
    ref_liquid, ref_vapor, ref_x, ref_y, ref_temp = reference
    assert liquid == pytest.approx(ref_liquid, rel=0,
                                   abs=SEED_FRACTION_ATOL * FEED_MOLES)
    assert vapor == pytest.approx(ref_vapor, rel=0,
                                  abs=SEED_FRACTION_ATOL * FEED_MOLES)
    np.testing.assert_allclose(x_liq, ref_x, rtol=0, atol=SEED_FRACTION_ATOL)
    np.testing.assert_allclose(y_vap, ref_y, rtol=0, atol=SEED_FRACTION_ATOL)
    assert temp == pytest.approx(ref_temp, rel=0, abs=SEED_TEMP_ATOL)


@pytest.mark.parametrize('nitrogen', [True, False],
                         ids=['nitrogen', 'subcritical-control'])
def test_changed_seeds_and_reattached_feed_agree(binary_path, nitrogen):
    """Solve one instance from several valid seeds and after re-attaching."""
    feed = _feed(binary_path, nitrogen)
    drum_pressure = ATMOSPHERE if nitrogen else CONTROL_PRESSURE  # [Pa]
    flash = AdiabaticFlash(drum_pressure, div_energybce=ENERGY_SCALE,
                           seed_basedon_input=True)
    flash.Inlet = feed
    solves = []
    for v_seed in SEEDS + (REATTACHED_SEED,):  # [-]
        if v_seed == REATTACHED_SEED:
            flash.Inlet = feed
        flash.solve_unit(v_seed=v_seed)
        solves.append(_products(flash))

        assert flash.LiquidOut.name_species == feed.name_species
        assert flash.VaporOut.name_species == feed.name_species
        liquid, vapor, x_liq, y_vap, _ = solves[-1]
        assert x_liq.sum() == pytest.approx(1, abs=CLOSURE_RTOL)
        assert y_vap.sum() == pytest.approx(1, abs=CLOSURE_RTOL)
        np.testing.assert_allclose(liquid * x_liq + vapor * y_vap,
                                   FEED_MOLES * feed.mole_frac, rtol=0,
                                   atol=CLOSURE_RTOL * FEED_MOLES)
        if nitrogen:
            assert x_liq[-1] == 0  # nitrogen only in the vapor

    # The last solve followed a re-attachment, which resets num_comp, so a
    # reduction by that solve still shows here.
    assert flash.num_comp == len(feed.name_species)
    for products in solves[1:]:
        _assert_close(products, solves[0])


def test_later_solve_keeps_earlier_products(binary_path):
    """Leave the first solve's vapor product as it was published."""
    feed = _feed(binary_path)
    flash = AdiabaticFlash(ATMOSPHERE, seed_basedon_input=True)  # [Pa]
    flash.Inlet = feed
    flash.solve_unit()
    first_vapor = flash.VaporOut
    first_y = first_vapor.mole_frac.copy()  # [-]

    feed.updatePhase(mole_frac=CHANGED_FRACTIONS)
    flash.solve_unit()

    assert flash.VaporOut is not first_vapor
    assert not np.allclose(flash.VaporOut.mole_frac, first_y)
    np.testing.assert_array_equal(first_vapor.mole_frac, first_y)


def _nitrogen_database(path, tmp_path):
    """Write the shipped database with the packaged nitrogen appended.

    Parameters
    ----------
    path : pathlib.Path
        Shipped five-species database path.
    tmp_path : pathlib.Path
        Temporary directory.

    Returns
    -------
    str
        Path of the six-species database (nitrogen last).
    """
    data = json.loads(Path(path).read_text())
    data.update(json.loads(NITROGEN_PATH.read_text()))
    target = tmp_path / 'with_nitrogen.json'
    target.write_text(json.dumps(data))
    return str(target)


def test_resolved_flowsheet_gives_the_same_products(data_path, tmp_path):
    """Re-solve Mixer --> AdiabaticFlash with nitrogen in the feeds."""
    path = _nitrogen_database(
        data_path['flowsheet'] / 'compound_database.json', tmp_path)
    sim = SimulationExec(path, {'M01': ['U01'], 'U01': []})
    sim.M01 = Mixer()
    sim.M01.Inlets = [
        LiquidStream(path, temp=FLOWSHEET_TEMP, pres=ATMOSPHERE,
                     mole_flow=FLOW_1, mole_frac=FEED_1),
        LiquidStream(path, temp=FLOWSHEET_TEMP, pres=ATMOSPHERE,
                     mole_flow=FLOW_2, mole_frac=FEED_2)]
    sim.U01 = AdiabaticFlash(pres_drum=ATMOSPHERE)
    results = []
    for _ in range(REPEATS):
        sim.SolveFlowsheet(verbose=False)
        # Each solve transfers a new Mixer outlet, which re-attaches the
        # feed; num_comp must still be the six feed species afterwards.
        assert sim.U01.num_comp == NUM_ACTIVE + 1
        row = sim.result.out_uos[4].split()
        assert row == ['U01', '0', str(2 * NUM_ACTIVE + 3), 'ALG',
                       'AdiabaticFlash']
        results.append((sim.U01.LiquidOut.mole_flow,
                        sim.U01.VaporOut.mole_flow,
                        sim.U01.VaporOut.mole_frac.copy(),
                        sim.U01.LiquidOut.temp))
    for products in results[1:]:
        _assert_same(products, results[0])

    total = FLOW_1 + FLOW_2  # [mol/s]
    liquid, vapor, y_vap, _ = results[0]
    x_liq = sim.U01.LiquidOut.mole_frac  # [-]
    np.testing.assert_allclose(liquid * x_liq + vapor * y_vap,
                               FLOW_1 * FEED_1 + FLOW_2 * FEED_2, rtol=0,
                               atol=CLOSURE_RTOL * total)
