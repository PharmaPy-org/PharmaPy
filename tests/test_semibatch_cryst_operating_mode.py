"""SemibatchCryst is a semibatch unit in flowsheets (issue #447).

``SemibatchCryst`` subclasses ``MSMPR`` but accumulates its feed, and its
``Outlet`` is its slurry holdup. It declares ``oper_mode = 'Semibatch'``
and ``is_continuous = False``, so a connection hands its final holdup to a
batch ``Mixer`` or ``BatchCryst``, rejects an ``MSMPR`` successor, installs
a connected ``MSMPR`` outlet as its feed, and ``GetLabor`` counts it as a
batch unit. As a semibatch destination of a batch ``Mixer``, a transferred
liquid completes the user's seed crystals, which leave the unit without a
reference copy of a complete charge, and is refused beside a liquid the
user charged (issue #438).
With the inherited continuous flags it was routed as a stream:
a downstream ``Mixer`` failed on its 33-sample profile, an ``MSMPR`` failed
on a missing ``DynamicInlet``, and ``GetLabor`` counted 2 instead of 3
operators.

Fixture (the triage reproduction of the issue): the shipped five-species
database (tests/Flowsheet/data/compound_database.json), a liquid charge of
mass fractions (0.1, 0.1, 0.1, 0.1, 0.6) at 310 K with crystals of A in
three size classes, ``CrystKinetics(coeff_solub=[2000.])`` and cooling
water, run for 100 s. The kinetics declare a solubility only, so no
nucleation, growth or dissolution mechanism is active and the crystal
population is constant; the feed has the charge's liquid composition, so
the liquid density is constant too. The liquid volume then obeys
dV/dt = Q for a solid-free feed of volume flow Q, so the expected volume
and liquid mass are V0 + Q t and the ideal-mixing density of the database
pure-component densities times that volume. The ``MSMPR`` source holds no
crystals, so its outlet is solid-free as well. Cases marked ``assimulo``
integrate the crystallizers with CVode; the unit case needs no solver.
"""

import json

import numpy as np
import pytest

from PharmaPy.Containers import DynamicCollector, Mixer
from PharmaPy.Crystallizers import BatchCryst, MSMPR, SemibatchCryst
from PharmaPy.Errors import PharmaPyValueError
from PharmaPy.Kinetics import CrystKinetics
from PharmaPy.Phases import LiquidPhase, SolidPhase
from PharmaPy.SimExec import SimulationExec
from PharmaPy.Streams import LiquidStream
from PharmaPy.Utilities import CoolingWater

COMPOSITION = np.array([0.1, 0.1, 0.1, 0.1, 0.6])  # [-], mass fractions
TEMP = 310.0  # [K], issue #447 triage charge and feed temperature
GRID = np.array([10., 20., 40.])  # [um], three size classes
SEED = np.array([1e7, 2e7, 1e7])  # [#/um], asymmetric crystal population
SOLID = [1, 0, 0, 0, 0]  # [-], crystals of species A
SOLUBILITY = [2000.]  # [kg/m**3], constant solubility of A; no mechanism
# [-], volumetric shape factor (crystal volume kv * x**3) of the triage
# seed; a synthetic cube-like value, not a calibrated crystal shape.
SHAPE_FACTOR = 1.0
CHARGE = 5e-4  # [m**3], semibatch liquid charge
TANK = 2e-3  # [m**3], semibatch vessel
MSMPR_CHARGE = 9e-4  # [m**3], MSMPR liquid charge
MSMPR_TANK = 1e-3  # [m**3]
FEED = 1e-5  # [m**3/s], liquid feed of the triage units
UTILITY_FLOW = 1e-5  # [m**3/s], cooling water
UTILITY_TEMP = 300.0  # [K], cooling water inlet
RUNTIME = 100.0  # [s], issue #447 triage run
# [-], CVode's default relative tolerance; the volume and inventories are
# linear in time, so they differ from the expectations by solver error only.
SOLVER_RTOL = 1e-6
RTOL = 1e-12  # [-], float64 roundoff of copied states and sums
RUN = {'runtime': RUNTIME, 'verbose': False}

HOLDUP_MESSAGE = (
    r"^Source 'S01' \(SemibatchCryst\) is not continuous: its outlet is a "
    r"holdup, an amount \[kg\] without a flow rate, but destination 'D01' "
    r"\({destination}\) takes a feed stream \[kg/s\] as its Inlet\. ")


@pytest.fixture
def path(data_path):
    """Return the shipped five-species property database.

    Parameters
    ----------
    data_path : dict
        Repository fixture directories.

    Returns
    -------
    str
        Database path.
    """
    return str(data_path['flowsheet'] / 'compound_database.json')


def _liquid_density(path):
    """Return the ideal-mixing density of the fixture liquid.

    Parameters
    ----------
    path : str
        Database path.

    Returns
    -------
    float
        ``1 / sum(w_i / rho_i)`` [kg/m**3] of the database pure liquid
        densities.
    """
    with open(path) as handle:
        data = json.load(handle)
    densities = np.array([value['rho_liq']
                          for value in data.values()])  # [kg/m**3]
    return 1 / np.sum(COMPOSITION / densities)  # [kg/m**3]


def _phases(path, vol, seed):
    """Build a liquid charge and its crystal population.

    Parameters
    ----------
    path : str
        Database path.
    vol : float
        Liquid volume [m**3].
    seed : numpy.ndarray
        Crystal population [#/um] on GRID.

    Returns
    -------
    tuple
        ``LiquidPhase`` and ``SolidPhase`` at TEMP [K].
    """
    return (LiquidPhase(path, temp=TEMP, vol=vol, mass_frac=COMPOSITION),
            SolidPhase(path, temp=TEMP, x_distrib=GRID.copy(),
                       distrib=seed.copy(), kv=SHAPE_FACTOR,
                       mass_frac=SOLID))


def _crystallizer(cls, **kwargs):
    """Attach the triage kinetics and cooling water to a crystallizer.

    Parameters
    ----------
    cls : type
        Crystallizer class.
    **kwargs : dict
        Constructor settings, such as ``vol_tank`` [m**3].

    Returns
    -------
    object
        Crystallizer of species A on the ``1D-FVM`` grid, without phases.
    """
    unit = cls('A', method='1D-FVM', **kwargs)
    unit.Kinetics = CrystKinetics(coeff_solub=SOLUBILITY)
    unit.Utility = CoolingWater(vol_flow=UTILITY_FLOW, temp_in=UTILITY_TEMP)
    return unit


def _semibatch(path, own_feed=True):
    """Build the triage SemibatchCryst.

    Parameters
    ----------
    path : str
        Database path.
    own_feed : bool, optional
        Attach its own solid-free liquid feed of FEED [m**3/s].

    Returns
    -------
    SemibatchCryst
        Charged unit.
    """
    unit = _crystallizer(SemibatchCryst, vol_tank=TANK)
    unit.Phases = _phases(path, CHARGE, SEED)
    if own_feed:
        unit.Inlet = LiquidStream(path, temp=TEMP, mass_frac=COMPOSITION,
                                  vol_flow=FEED)
    return unit


@pytest.mark.unit
def test_semibatch_crystallizer_declares_a_semibatch_mode():
    """Declare the accumulating unit's mode instead of MSMPR's."""
    unit = SemibatchCryst('A', method='1D-FVM', vol_tank=TANK)

    assert (unit.oper_mode, unit.is_continuous) == ('Semibatch', False)
    msmpr = MSMPR('A', method='1D-FVM', vol_tank=MSMPR_TANK)
    assert (msmpr.oper_mode, msmpr.is_continuous) == ('Continuous', True)


@pytest.mark.assimulo
@pytest.mark.integration
def test_holdup_is_mixed_as_a_batch(path):
    """Conserve the accumulated holdup in a batch Mixer."""
    pytest.importorskip('assimulo')
    sim = SimulationExec(path, {'S01': ['D01'], 'D01': []})
    sim.S01 = _semibatch(path)
    seed_mass = sim.S01.Solid_1.mass  # [kg], constant: no mechanism
    sim.D01 = Mixer()
    sim.SolveFlowsheet(kwargs_run={'S01': RUN}, verbose=False)

    mixed = sim.D01.Outlet
    assert sim.D01.oper_mode == 'Batch'
    liquid_mass = _liquid_density(path) * (CHARGE + FEED * RUNTIME)  # [kg]
    assert mixed.Liquid_1.mass == pytest.approx(liquid_mass, rel=SOLVER_RTOL)
    assert mixed.Solid_1.mass == pytest.approx(seed_mass, rel=SOLVER_RTOL)
    np.testing.assert_allclose(mixed.Liquid_1.mass_frac, COMPOSITION,
                               rtol=SOLVER_RTOL)


@pytest.mark.assimulo
@pytest.mark.integration
def test_holdup_charges_a_batch_crystallizer(path):
    """Start a BatchCryst from the accumulated holdup."""
    pytest.importorskip('assimulo')
    sim = SimulationExec(path, {'S01': ['D01'], 'D01': []})
    sim.S01 = _semibatch(path)
    sim.D01 = _crystallizer(BatchCryst)
    sim.SolveFlowsheet(kwargs_run={'S01': RUN, 'D01': RUN}, verbose=False)

    initial = sim.D01.result
    assert initial.vol[0] == pytest.approx(CHARGE + FEED * RUNTIME,
                                           rel=SOLVER_RTOL)
    np.testing.assert_allclose(initial.mass_conc[0],
                               COMPOSITION * _liquid_density(path),
                               rtol=SOLVER_RTOL)
    np.testing.assert_allclose(initial.distrib[0], SEED,
                               rtol=SOLVER_RTOL)


@pytest.mark.assimulo
@pytest.mark.integration
@pytest.mark.parametrize('destination', ['MSMPR', 'DynamicCollector'])
def test_holdup_is_rejected_by_a_continuous_successor(path, destination):
    """Reject the holdup instead of failing on a fake stream profile."""
    pytest.importorskip('assimulo')
    sim = SimulationExec(path, {'S01': ['D01'], 'D01': []})
    sim.S01 = _semibatch(path)
    if destination == 'MSMPR':
        sim.D01 = _crystallizer(MSMPR, vol_tank=MSMPR_TANK)
        sim.D01.Phases = _phases(path, MSMPR_CHARGE, SEED)
    else:
        sim.D01 = DynamicCollector()

    with pytest.raises(PharmaPyValueError, match=HOLDUP_MESSAGE.format(
            destination=destination)):
        sim.SolveFlowsheet(kwargs_run={'S01': RUN, 'D01': RUN}, verbose=False)

    assert sim.D01.Inlet is None


@pytest.mark.assimulo
@pytest.mark.integration
def test_msmpr_outlet_feeds_the_semibatch_crystallizer(path):
    """Install the MSMPR outlet as the feed and accumulate its volume."""
    pytest.importorskip('assimulo')
    sim = SimulationExec(path, {'S01': ['D01'], 'D01': []})
    sim.S01 = _crystallizer(MSMPR, vol_tank=MSMPR_TANK)
    sim.S01.Phases = _phases(path, MSMPR_CHARGE, np.zeros_like(SEED))
    sim.S01.Inlet = LiquidStream(path, temp=TEMP, mass_frac=COMPOSITION,
                                 vol_flow=FEED)
    sim.D01 = _semibatch(path, own_feed=False)
    sim.SolveFlowsheet(kwargs_run={'S01': RUN, 'D01': RUN}, verbose=False)

    feed = sim.D01.Inlet
    assert feed is sim.connections['CONN1'].transferred_matter
    np.testing.assert_allclose(feed.y_inlet['vol_flow'], FEED, rtol=RTOL)
    volume = sim.D01.result.vol  # [m**3], liquid
    assert volume[0] == pytest.approx(CHARGE, rel=RTOL)
    assert volume[-1] - volume[0] == pytest.approx(FEED * RUNTIME,
                                                   rel=SOLVER_RTOL)


@pytest.mark.assimulo
@pytest.mark.integration
def test_labor_counts_a_batch_solids_unit(path):
    """Staff the semibatch crystallizer as a batch solids unit."""
    pytest.importorskip('assimulo')
    sim = SimulationExec(path, {'S01': []})
    sim.S01 = _semibatch(path)
    sim.SolveFlowsheet(kwargs_run={'S01': RUN}, verbose=False)

    labor = sim.GetLabor()
    # Operators per shift: solids units take 2, plus 1 for batch operation.
    batch_solids_workers = 3  # [-]
    assert labor.loc['S01', 'has_solids'] == 1
    assert labor.loc['S01', 'is_batch'] == 1
    assert labor.loc['S01', 'num_workers'] == batch_solids_workers


def _seed(path):
    """Build the triage crystal seed alone.

    Parameters
    ----------
    path : str
        Database path.

    Returns
    -------
    SolidPhase
        Crystals of A on GRID [um] with population SEED [#/um].
    """
    return SolidPhase(path, temp=TEMP, x_distrib=GRID.copy(),
                      distrib=SEED.copy(), kv=SHAPE_FACTOR, mass_frac=SOLID)


def _liquid_mixer(path):
    """Build a batch Mixer holding the semibatch liquid charge.

    Parameters
    ----------
    path : str
        Database path.

    Returns
    -------
    Mixer
        Unsolved Mixer of a CHARGE [m**3] liquid at TEMP [K].
    """
    mixer = Mixer()
    mixer.Inlets = LiquidPhase(path, temp=TEMP, vol=CHARGE,
                               mass_frac=COMPOSITION)
    return mixer


@pytest.mark.assimulo
@pytest.mark.integration
def test_transferred_liquid_completes_a_seeded_charge(path):
    """Charge the user's seed with a liquid holdup, again on a re-solve.

    The crystallizer keeps no reference copy of its charge while it holds
    only the seed, so the connection decides by phase kind; the transfer
    then completes the charge, and the re-solve replaces the transferred
    liquid instead of stacking a second one.
    """
    pytest.importorskip('assimulo')
    sim = SimulationExec(path, {'M01': ['D01'], 'D01': []})
    sim.M01 = _liquid_mixer(path)
    sim.D01 = _crystallizer(SemibatchCryst, vol_tank=TANK)
    seed = _seed(path)
    sim.D01.Phases = seed
    sim.D01.Inlet = LiquidStream(path, temp=TEMP, mass_frac=COMPOSITION,
                                 vol_flow=FEED)
    assert not hasattr(sim.D01, '__original_phase__')

    for _ in range(2):
        sim.SolveFlowsheet(kwargs_run={'D01': RUN}, verbose=False)
        transferred = sim.connections['CONN1'].transferred_matter
        assert sim.D01.Phases == [seed, transferred]
        # The run starts from the transferred liquid and grows by the feed.
        assert sim.D01.result.vol[-1] == pytest.approx(
            CHARGE + FEED * RUNTIME, rel=SOLVER_RTOL)


@pytest.mark.unit
def test_transferred_liquid_is_refused_beside_a_user_liquid(path):
    """Refuse a second liquid charge rather than stacking two liquids."""
    sim = SimulationExec(path, {'M01': ['D01'], 'D01': []})
    sim.M01 = _liquid_mixer(path)
    sim.D01 = _crystallizer(SemibatchCryst, vol_tank=TANK)
    own_liquid = LiquidPhase(path, temp=TEMP, vol=CHARGE,
                             mass_frac=COMPOSITION)
    sim.D01.Phases = own_liquid
    message = (r"^Destination 'D01' \(SemibatchCryst\) already holds an "
               r"initial charge \(Phases\) assigned outside the flowsheet, "
               r"and the holdup \[kg\] of source 'M01' \(Mixer\), which is "
               r"not continuous, would be a second initial charge\. ")
    with pytest.raises(PharmaPyValueError, match=message):
        sim.SolveFlowsheet(kwargs_run={'D01': RUN}, verbose=False)

    assert sim.D01.Phases == [own_liquid]
    assert sim.M01.Outlet.y_upstream is None
