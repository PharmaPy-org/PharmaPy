"""Repeated flowsheet solves replace connected inputs (issue #441).

Each ``SolveFlowsheet`` transfer on a graph edge replaces the copy that the
previous transfer on that edge handed to the destination, while the
destination's own inputs are kept. A connected ``Mixer`` therefore holds its
raw inlets plus exactly one copy of each upstream outlet, in an unchanged
position, after repeated or partial (``pick_units``) solves; a
``BatchCryst`` or ``Drying`` holdup does not stack a second transferred
phase; and a ``DynamicExtractor`` takes the latest ``'feed'`` instead of
keeping the first one.

Unit cases solve real static continuous and batch ``Mixer`` units, which
need no ODE backend, on the shipped five-species database
(tests/Flowsheet/data/compound_database.json): source M1 mixes FEED_A and
FEED_B, and M2 adds its own raw feed of FEED_C, so the expected M2 outlet is
the sum of the three feeds on the flow [kg/s] or inventory [kg] basis. The
``Drying`` cases use ``Connection`` directly with a synthetic solved
``Filter``. An ``assimulo`` case also solves a real ``Filter -> Drying``
flowsheet twice on the synthetic drying database of tests/conftest.py,
which covers the phases of the earlier transfer that ``SimulationExec``
hands to ``Connection`` (``replaced_phases``). ``assimulo`` cases integrate the
issue reproduction (Mixer -> Mixer -> ``DynamicCollector``), a
``BatchCryst`` charged by a batch Mixer, and a ``DynamicExtractor`` fed by
a Mixer on the synthetic constant-K database of
tests/test_dynamic_extractor_inlet_temperatures.py.

Related issues: https://github.com/PharmaPy-org/PharmaPy/issues/441,
https://github.com/PharmaPy-org/PharmaPy/issues/342
"""

import json

import numpy as np
import pytest

from PharmaPy.Connections import Connection
from PharmaPy.Containers import Mixer
from PharmaPy.Drying_Model import Drying
from PharmaPy.MixedPhases import Cake
from PharmaPy.Phases import LiquidPhase, SolidPhase, VaporPhase
from PharmaPy.Results import DynamicResult
from PharmaPy.SimExec import SimulationExec
from PharmaPy.SolidLiquidSep import Filter
from PharmaPy.Streams import LiquidStream, VaporStream
from conftest import (DRYING_GAS_MASS_FRACTION, DRYING_LIQUID_MASS_FRACTION,
                      DRYING_SIZE_GRID_UM, DRYING_THERMO_DATA)

FEED_A = np.array([0.1, 0.1, 0.1, 0.1, 0.6])  # [-], issue #441 feed
FEED_B = np.array([0.0, 0.2, 0.0, 0.0, 0.8])  # [-], solvent-rich feed
FEED_C = np.array([0.3, 0.0, 0.2, 0.0, 0.5])  # [-], M2's own raw feed
FLOW_A = 1.0  # [kg/s], issue #441
FLOW_B = 0.5  # [kg/s], issue #441
FLOW_C = 0.25  # [kg/s], issue #441 raw feed of M2
FLOW_D = 0.125  # [kg/s], a second raw feed added to M2 by hand
CHANGED_FLOW_A = 2.0  # [kg/s], feed A of a replacement source M1
MASS_A = 1.0  # [kg], batch charges with the same roles as the flows
MASS_B = 2.0  # [kg]
MASS_C = 0.5  # [kg]
TEMP = 300.0  # [K], common feed temperature
RUNTIME = 10.0  # [s], issue #441 collector run
# DynamicCollector.solve_unit seeds a liquid collector with the feed of
# 0.1 s, a provisional seed tracked by issue #342; drop it from the
# expectations once #342 replaces the seed.
COLLECTOR_SEED_TIME = 0.1  # [s]
# [-], float64 roundoff of the Mixer's algebraic sums
RTOL = 1e-12
# [-], the default CVode rtol: the collector runs with default solver
# options, and a constant feed makes its mass balance linear in time.
FLOWSHEET_RTOL = 1e-6
REPEATED_SOLVES = 3  # [-], more than two so a single replacement is not enough
# Filter and dryer cases: a synthetic bench-scale cake, not a calibrated
# filtration or drying case; the values only need to give real, finite
# phases and a cake geometry.
STATION_DIAMETER = 0.1  # [m], common filter and dryer diameter
CAKE_GRID = np.array([0., 100., 200., 300.])  # [um], four crystal bins
# [#/um], asymmetric population of the inner bins, as in
# tests/test_container_connection_boundaries.py
CAKE_POPULATION = np.array([0., 1e6, 1.25e5, 0.])
FILTER_TIME = np.array([0., 1.])  # [s], one-second synthetic filter result
DRYER_NODES = 2  # [-], smallest axial grid with an interior face
GAS_MASS = 1e-4  # [kg], dryer gas holdup, small against the 1 kg cake liquid
GAS_FRACTION = [0., 0., 0., 0., 1.]  # [-], pure solvent vapor
# BatchCryst case of tests/test_batch_to_flow_connector.py (issue #423): an
# isothermal 1 L liquid charge with a 10-40 um seed population, run for
# 1 s far from saturation, so the solve is short and well conditioned.
CRYST_TEMP = 310.0  # [K]
SEED_GRID = np.array([10., 20., 40.])  # [um]
SEED_POPULATION = np.array([1e7, 2e7, 1e7])  # [#/um]
SEED_SHAPE_FACTOR = 0.5  # [-], volume shape factor kv
CRYST_LIQUID_VOL = 1e-3  # [m**3]
SOLUBILITY = [2000.]  # [kg/m**3], constant and undersaturated
CRYST_RUNTIME = 1.0  # [s]
# Extractor case: about half the 4 s thermal time scale of the two-stage
# constant-property extractor of tests/test_dynamic_extractor_inlet_
# temperatures.py, long enough for a real solve and short to run.
EXTRACTOR_RUNTIME = 2.0  # [s]


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


def _stream(path, fraction, flow):
    """Build a raw liquid stream at TEMP.

    Parameters
    ----------
    path : str
        Database path.
    fraction : numpy.ndarray
        Mass fractions [-], shape (5,).
    flow : float
        Mass flow [kg/s].

    Returns
    -------
    LiquidStream
        Raw stream.
    """
    return LiquidStream(path, temp=TEMP, mass_flow=flow, mass_frac=fraction)


def _phase(path, fraction, mass):
    """Build a raw liquid batch charge at TEMP.

    Parameters
    ----------
    path : str
        Database path.
    fraction : numpy.ndarray
        Mass fractions [-], shape (5,).
    mass : float
        Liquid mass [kg].

    Returns
    -------
    LiquidPhase
        Raw charge.
    """
    return LiquidPhase(path, temp=TEMP, mass=mass, mass_frac=fraction)


def _continuous_chain(path, flow_a=FLOW_A):
    """Build static continuous Mixers M1 -> M2, M2 with its own raw feed.

    Parameters
    ----------
    path : str
        Database path.
    flow_a : float, optional
        Mass flow [kg/s] of M1's feed A.

    Returns
    -------
    SimulationExec
        Unsolved flowsheet.
    """
    sim = SimulationExec(path, {'M1': ['M2'], 'M2': []})
    sim.M1 = _continuous_source(path, flow_a)
    sim.M2 = Mixer()
    sim.M2.Inlets = [_stream(path, FEED_C, FLOW_C)]
    return sim


def _continuous_source(path, flow_a=FLOW_A):
    """Build the static continuous source Mixer of feeds A and B.

    Parameters
    ----------
    path : str
        Database path.
    flow_a : float, optional
        Mass flow [kg/s] of feed A.

    Returns
    -------
    Mixer
        Unsolved Mixer of flow_a [kg/s] of FEED_A and FLOW_B of FEED_B.
    """
    mixer = Mixer()
    mixer.Inlets = [_stream(path, FEED_A, flow_a),
                    _stream(path, FEED_B, FLOW_B)]
    return mixer


def _assert_outlet_flow(mixer, species_flows):
    """Assert a Mixer outlet against independent species flows.

    Parameters
    ----------
    mixer : Mixer
        Solved continuous Mixer.
    species_flows : numpy.ndarray
        Expected species mass flows [kg/s], shape (5,).
    """
    outlet = mixer.Outlet
    assert outlet.mass_flow == pytest.approx(species_flows.sum(), rel=RTOL)
    np.testing.assert_allclose(outlet.mass_flow * outlet.mass_frac,
                               species_flows, rtol=RTOL)


@pytest.mark.unit
def test_repeated_continuous_solves_mix_the_upstream_flow_once(path):
    """Keep M2 at 1.75 kg/s on every solve instead of growing by 1.5 kg/s."""
    sim = _continuous_chain(path)
    raw_feed = sim.M2.Inlets[0]
    # [kg/s], 1.0 + 0.5 kg/s from M1 and M2's own 0.25 kg/s
    species_flows = FLOW_A * FEED_A + FLOW_B * FEED_B + FLOW_C * FEED_C

    for _ in range(REPEATED_SOLVES):
        sim.SolveFlowsheet(verbose=False)

        transferred = sim.connections['CONN1'].transferred_matter
        assert len(sim.M2.Inlets) == 2
        assert sim.M2.Inlets[0] is raw_feed
        assert sim.M2.Inlets[1] is transferred
        _assert_outlet_flow(sim.M2, species_flows)


@pytest.mark.unit
def test_partial_solves_keep_one_connected_copy(path):
    """Give the same M2 outlet across a sequence of pick_units solves."""
    sim = _continuous_chain(path)
    raw_feed = sim.M2.Inlets[0]
    species_flows = FLOW_A * FEED_A + FLOW_B * FEED_B + FLOW_C * FEED_C  # [kg/s]

    # Picking only M2 still transfers the already solved M1's outlet;
    # picking only M1 re-solves it without transferring to M2.
    for pick_units in (None, ['M2'], ['M1'], ['M2'], ['M1', 'M2']):
        sim.SolveFlowsheet(pick_units=pick_units, verbose=False)

        assert len(sim.M2.Inlets) == 2
        assert sim.M2.Inlets[0] is raw_feed
        if pick_units != ['M1']:
            transferred = sim.connections['CONN1'].transferred_matter
            assert sim.M2.Inlets[1] is transferred
            _assert_outlet_flow(sim.M2, species_flows)


@pytest.mark.unit
def test_replacement_keeps_the_inlet_position(path):
    """Replace the connected copy in place, before a feed added later."""
    sim = _continuous_chain(path)
    sim.SolveFlowsheet(verbose=False)
    raw_feed = sim.M2.Inlets[0]
    added_feed = _stream(path, FEED_B, FLOW_D)
    sim.M2.Inlets = added_feed  # appended after the connected copy

    sim.SolveFlowsheet(verbose=False)

    transferred = sim.connections['CONN1'].transferred_matter
    assert len(sim.M2.Inlets) == 3
    assert sim.M2.Inlets[0] is raw_feed
    assert sim.M2.Inlets[1] is transferred
    assert sim.M2.Inlets[2] is added_feed
    _assert_outlet_flow(sim.M2, FLOW_A * FEED_A + FLOW_B * FEED_B
                        + FLOW_C * FEED_C + FLOW_D * FEED_B)  # [kg/s]


@pytest.mark.unit
def test_a_replaced_source_feeds_its_new_outlet(path):
    """Mix the new source's outlet, not the stale copy of the old one."""
    sim = _continuous_chain(path)
    sim.SolveFlowsheet(verbose=False)
    sim.M1 = _continuous_source(path, CHANGED_FLOW_A)

    sim.SolveFlowsheet(verbose=False)

    assert len(sim.M2.Inlets) == 2
    _assert_outlet_flow(sim.M2, CHANGED_FLOW_A * FEED_A + FLOW_B * FEED_B
                        + FLOW_C * FEED_C)  # [kg/s]


@pytest.mark.unit
def test_rejected_replacement_leaves_the_mixer_unchanged(path):
    """Keep M2's inlets when the new transfer mixes inventories and flows."""
    sim = _continuous_chain(path)
    sim.SolveFlowsheet(verbose=False)
    held = list(sim.M2.Inlets)
    batch_source = Mixer()
    batch_source.Inlets = _phase(path, FEED_A, MASS_A)
    sim.M1 = batch_source

    with pytest.raises(ValueError, match=r"Mixer cannot replace a flow rate "
                                         r"\[kg/s\] with a batch inventory "
                                         r"\[kg\]"):
        sim.SolveFlowsheet(verbose=False)

    assert len(sim.M2.Inlets) == len(held)
    assert all(now is before for now, before in zip(sim.M2.Inlets, held))


@pytest.mark.unit
@pytest.mark.parametrize('first_batch', [True, False],
                         ids=['batch_to_flow', 'flow_to_batch'])
def test_basis_change_of_the_only_inlet_is_rejected(path, first_batch):
    """Reject a re-transfer on the other amount basis; change nothing.

    M2's only inlet is the connected copy. Its first assignment selected
    the flow or non-flow state names, so a replacement on the other basis
    is rejected instead of leaving those names stale.
    """
    batch_source = Mixer()
    batch_source.Inlets = [_phase(path, FEED_A, MASS_A),
                           _phase(path, FEED_B, MASS_B)]
    sources = [batch_source, _continuous_source(path)]
    if not first_batch:
        sources.reverse()
    sim = SimulationExec(path, {'M1': ['M2'], 'M2': []})
    sim.M1, sim.M2 = sources[0], Mixer()
    sim.SolveFlowsheet(verbose=False)
    held = list(sim.M2.Inlets)
    state = {name: getattr(sim.M2, name) for name in (
        'oper_mode', 'is_continuous', 'names_states_in', 'states_di',
        'name_states', 'dim_states')}
    names_in = list(sim.M2.names_states_in)
    sim.M1 = sources[1]

    old, new = (('a batch inventory \\[kg\\]', 'a flow rate \\[kg/s\\]')
                if first_batch else
                ('a flow rate \\[kg/s\\]', 'a batch inventory \\[kg\\]'))
    with pytest.raises(ValueError, match=rf"Mixer cannot replace {old} with "
                                         rf"{new}.*Assign a new Mixer"):
        sim.SolveFlowsheet(verbose=False)

    assert len(sim.M2.Inlets) == 1 and sim.M2.Inlets[0] is held[0]
    for name, value in state.items():
        assert getattr(sim.M2, name) is value, name
    assert list(sim.M2.names_states_in) == names_in


@pytest.mark.unit
def test_repeated_batch_solves_mix_the_upstream_charge_once(path):
    """Keep a batch M2 at 3.5 kg on every solve."""
    sim = SimulationExec(path, {'M1': ['M2'], 'M2': []})
    sim.M1 = Mixer()
    sim.M1.Inlets = [_phase(path, FEED_A, MASS_A), _phase(path, FEED_B, MASS_B)]
    sim.M2 = Mixer()
    sim.M2.Inlets = _phase(path, FEED_C, MASS_C)
    raw_charge = sim.M2.Inlets[0]
    # [kg], 1 + 2 kg from M1 and M2's own 0.5 kg
    species_mass = MASS_A * FEED_A + MASS_B * FEED_B + MASS_C * FEED_C

    for _ in range(REPEATED_SOLVES):
        sim.SolveFlowsheet(verbose=False)

        assert len(sim.M2.Inlets) == 2
        assert sim.M2.Inlets[0] is raw_charge
        outlet = sim.M2.Outlet
        assert outlet.mass == pytest.approx(species_mass.sum(), rel=RTOL)
        np.testing.assert_allclose(outlet.mass * outlet.mass_frac,
                                   species_mass, rtol=RTOL)


def _solved_filter(path):
    """Build a synthetic solved Filter whose outlet is a liquid-solid cake.

    Parameters
    ----------
    path : str
        Database path.

    Returns
    -------
    Filter
        Filter holding a 1 kg FEED_A liquid and a seed solid in its outlet
        ``Cake``, with a one-second batch result [s].
    """
    cake = Cake()
    cake.Phases = [_phase(path, FEED_A, MASS_A),
                   SolidPhase(path, mass_frac=[1, 0, 0, 0, 0],
                              x_distrib=CAKE_GRID, distrib=CAKE_POPULATION)]
    source = Filter(station_diam=STATION_DIAMETER)
    source.Outlet = cake
    source.outputs = {'mass': MASS_A}  # [kg], synthetic completed filter
    source.result = DynamicResult({}, time=FILTER_TIME)
    return source


def _dryer(supercrit_names=()):
    """Build an unconnected dryer of the filter's diameter.

    Parameters
    ----------
    supercrit_names : sequence of str, optional
        Species the dryer treats as supercritical.

    Returns
    -------
    Drying
        Dryer with DRYER_NODES nodes and no phases.
    """
    return Drying(number_nodes=DRYER_NODES,
                  supercrit_names=list(supercrit_names),
                  diam_unit=STATION_DIAMETER)


def _gas(path):
    """Build the dryer's own solvent-vapor gas phase.

    Parameters
    ----------
    path : str
        Database path.

    Returns
    -------
    VaporPhase
        GAS_MASS [kg] of GAS_FRACTION [-] at TEMP [K].
    """
    return VaporPhase(path, mass=GAS_MASS, mass_frac=GAS_FRACTION, temp=TEMP)


def _retransfer(source, dryer, first):
    """Transfer the source outlet again, replacing an earlier transfer.

    Parameters
    ----------
    source : Filter
        Solved source unit.
    dryer : Drying
        Destination holding the earlier transfer.
    first : Connection
        Earlier connection on the same edge.

    Returns
    -------
    Connection
        The new connection, after ``transfer_data``.
    """
    connection = Connection(source, dryer,
                            replaced_matter=first.transferred_matter,
                            replaced_phases=first.transferred_phases)
    connection.transfer_data()
    return connection


@pytest.mark.unit
def test_dryer_replaces_the_transferred_cake_phases(path):
    """Keep the dryer's gas and one copy of the cake across re-transfers."""
    source = _solved_filter(path)
    dryer = _dryer()
    gas = _gas(path)
    dryer.Phases = gas
    first = Connection(source, dryer)
    first.transfer_data()

    cake = _retransfer(source, dryer, first).transferred_matter

    assert len(dryer.Phases) == 3
    assert dryer.Phases[0] is gas
    assert dryer.Phases[1] is cake.Liquid_1
    assert dryer.Phases[2] is cake.Solid_1
    assert dryer.CakePhase is cake
    assert dryer.Liquid_1.mass == pytest.approx(MASS_A, rel=RTOL)


@pytest.mark.unit
def test_gas_added_after_the_transfer_survives_a_retransfer(path):
    """Keep a gas the dryer appended to the transferred cake's own list.

    With no phases, the dryer adopts the transferred cake's ``Phases``
    list, so a gas assigned afterwards lands in that list too; only the
    phases the cake carried when it was transferred may be removed.
    """
    source = _solved_filter(path)
    dryer = _dryer()
    first = Connection(source, dryer)
    first.transfer_data()
    gas = _gas(path)
    dryer.Phases = gas
    assert any(phase is gas for phase in first.transferred_matter.Phases)

    cake = _retransfer(source, dryer, first).transferred_matter

    assert len(dryer.Phases) == 3
    assert any(phase is gas for phase in dryer.Phases)
    assert dryer.Vapor_1 is gas and gas.name == 'Vapor_1'
    assert dryer.Liquid_1 is cake.Liquid_1
    assert dryer.Solid_1 is cake.Solid_1


@pytest.mark.unit
def test_rejected_dryer_transfer_restores_its_phases(path):
    """Restore the dryer's phases when the setter rejects the matter."""
    source = _solved_filter(path)
    dryer = _dryer()
    first = Connection(source, dryer)
    first.transfer_data()
    held = list(dryer.Phases)
    source.Outlet = _stream(path, FEED_A, FLOW_A)  # not a phase or cake

    with pytest.raises(RuntimeError, match='Please provide a list or tuple '
                                           'of phases objects'):
        Connection(source, dryer, replaced_matter=first.transferred_matter,
                   replaced_phases=first.transferred_phases).transfer_data()

    assert len(dryer.Phases) == len(held)
    assert all(now is before for now, before in zip(dryer.Phases, held))


@pytest.mark.unit
def test_late_rejected_dryer_transfer_restores_its_state(
        path, drying_cake_factory):
    """Restore every binding a setter that fails late has already changed.

    The dryer treats 'solvent' as supercritical; the replacing cake comes
    from the synthetic drying database, which has no 'solvent', so the
    setter raises only after re-classifying the phases, which renames the
    dryer's own raw solid, and resizing the cake grid.
    """
    source = _solved_filter(path)
    dryer = _dryer(supercrit_names=['solvent'])
    gas = _gas(path)
    dryer.Phases = gas
    first = Connection(source, dryer)
    first.transfer_data()
    # A raw solid after the cake is Solid_2; the failing setter renames it
    # Solid_1 once the transferred cake solid is removed. (A raw liquid
    # would become Liquid_1 and supply 'solvent', so nothing would fail.)
    raw = SolidPhase(path, mass_frac=[1, 0, 0, 0, 0], x_distrib=CAKE_GRID,
                     distrib=CAKE_POPULATION)
    dryer.Phases = raw
    assert raw.name == 'Solid_2' and dryer.Solid_2 is raw
    bindings = {name: getattr(dryer, name) for name in (
        'Liquid_1', 'Solid_1', 'Solid_2', 'Vapor_1', 'CakePhase', 'z_grid',
        'z_centers', 'dz', 'idx_supercrit', 'name_species', 'states_di')}
    height = dryer.cake_height  # [m]
    held = list(dryer.Phases)
    names = [phase.name for phase in held]
    source.Outlet = drying_cake_factory()

    with pytest.raises(ValueError, match="'solvent' is not in list"):
        _retransfer(source, dryer, first)

    assert all(now is before for now, before in zip(dryer.Phases, held))
    assert len(dryer.Phases) == len(held)
    assert [phase.name for phase in dryer.Phases] == names
    for name, value in bindings.items():
        assert getattr(dryer, name) is value, name
    assert raw.name == 'Solid_2' and dryer.Solid_2 is raw
    assert dryer.cake_height == height


@pytest.mark.assimulo
@pytest.mark.integration
def test_issue_reproduction_resolves_identically(path):
    """Re-solve Mixer -> Mixer -> collector with identical flows and mass."""
    pytest.importorskip('assimulo')
    from PharmaPy.Containers import DynamicCollector

    m1 = Mixer()
    m1.Inlets = [_stream(path, FEED_A, FLOW_A), _stream(path, FEED_A, FLOW_B)]
    m2 = Mixer()
    m2.Inlets = [_stream(path, FEED_A, FLOW_C)]
    sim = SimulationExec(path, {'M1': ['M2'], 'M2': ['C1'], 'C1': []})
    sim.M1, sim.M2, sim.C1 = m1, m2, DynamicCollector()
    total_flow = FLOW_A + FLOW_B + FLOW_C  # [kg/s], 1.75 kg/s
    # [kg], the run plus the provisional seed (issue #342)
    collected = total_flow * (RUNTIME + COLLECTOR_SEED_TIME)

    outcomes = []
    for _ in range(2):
        sim.SolveFlowsheet(kwargs_run={'C1': {'runtime': RUNTIME,
                                              'verbose': False}},
                           verbose=False)
        result = sim.C1.result
        outcomes.append((m2.Outlet.mass_flow, result.mass[-1],
                         result.time[-1] - result.time[0]))

    for outlet_flow, mass, span in outcomes:
        assert outlet_flow == pytest.approx(total_flow, rel=RTOL)
        assert mass == pytest.approx(collected, rel=FLOWSHEET_RTOL)
        assert span == pytest.approx(RUNTIME, rel=RTOL)
    assert len(m2.Inlets) == 2


@pytest.mark.assimulo
@pytest.mark.integration
def test_batch_crystallizer_does_not_stack_the_transferred_liquid(path):
    """Keep the seed and only the latest liquid charge on a re-solve."""
    pytest.importorskip('assimulo')
    from PharmaPy.Crystallizers import BatchCryst
    from PharmaPy.Kinetics import CrystKinetics

    seed = SolidPhase(path, temp=CRYST_TEMP, x_distrib=SEED_GRID,
                      distrib=SEED_POPULATION, kv=SEED_SHAPE_FACTOR,
                      mass_frac=[1., 0., 0., 0., 0.])  # [-], pure A
    crystallizer = BatchCryst('A', method='1D-FVM', controls={
        'temp': lambda time: CRYST_TEMP + np.zeros_like(time)})  # [K]
    crystallizer.Phases = seed
    crystallizer.Kinetics = CrystKinetics(coeff_solub=SOLUBILITY)
    sim = SimulationExec(path, {'M1': ['CR1'], 'CR1': []})
    sim.M1 = Mixer()
    sim.M1.Inlets = LiquidPhase(path, temp=CRYST_TEMP, vol=CRYST_LIQUID_VOL,
                                mass_frac=FEED_A)
    sim.CR1 = crystallizer
    run = {'CR1': {'runtime': CRYST_RUNTIME, 'verbose': False}}

    for _ in range(2):
        sim.SolveFlowsheet(kwargs_run=run, verbose=False)

        transferred = sim.connections['CONN1'].transferred_matter
        assert len(crystallizer.Phases) == 2
        assert crystallizer.Phases[0] is seed
        assert crystallizer.Phases[1] is transferred
        assert crystallizer.Liquid_1 is transferred
        assert not hasattr(crystallizer, 'Liquid_2')


@pytest.mark.assimulo
@pytest.mark.integration
def test_extractor_feed_is_the_latest_transfer(tmp_path):
    """Feed the extractor the new source outlet and keep its own solvent."""
    pytest.importorskip('assimulo')
    from PharmaPy.DynamicExtraction import DynamicExtractor
    from test_dynamic_extractor_inlet_temperatures import (
        HEAVY_COMPOSITION, HEAVY_FLOW, INITIAL_TEMPERATURE,
        LIGHT_COMPOSITION, LIGHT_FLOW, PHASE_HOLDUP, SPECIES,
        distribution_coefficients)

    thermo = tmp_path / 'extractor.json'
    thermo.write_text(json.dumps(SPECIES))
    thermo = str(thermo)

    def source(mole_flow):
        """Build a continuous Mixer of the heavy feed.

        Parameters
        ----------
        mole_flow : float
            Feed molar flow [mol/s].

        Returns
        -------
        Mixer
            Unsolved one-inlet Mixer at 300 K.
        """
        mixer = Mixer()
        mixer.Inlets = [LiquidStream(thermo, mole_frac=HEAVY_COMPOSITION,
                                     mole_flow=mole_flow, temp=TEMP,
                                     verbose=False)]
        return mixer

    extractor = DynamicExtractor(num_stages=2, k_fun=distribution_coefficients,
                                 gamma_model='ideal')
    extractor.Phases = LiquidPhase(
        thermo, mole_frac=(LIGHT_COMPOSITION + HEAVY_COMPOSITION) / 2,
        moles=2 * PHASE_HOLDUP, temp=INITIAL_TEMPERATURE, verbose=False)
    solvent = LiquidStream(thermo, mole_frac=LIGHT_COMPOSITION,
                           mole_flow=LIGHT_FLOW, temp=TEMP, verbose=False)
    extractor.Inlet = {'solvent': solvent}
    sim = SimulationExec(thermo, {'M1': ['E1'], 'E1': []})
    sim.E1 = extractor
    run = {'E1': {'runtime': EXTRACTOR_RUNTIME, 'verbose': False}}

    for feed_flow in (HEAVY_FLOW, 2 * HEAVY_FLOW):  # [mol/s]
        sim.M1 = source(feed_flow)
        sim.SolveFlowsheet(kwargs_run=run, verbose=False)

        transferred = sim.connections['CONN1'].transferred_matter
        assert set(extractor.Inlet) == {'feed', 'solvent'}
        assert extractor.Inlet['solvent'] is solvent
        assert extractor.Inlet['feed'] is transferred
        assert extractor.Inlet['feed'].mole_flow == pytest.approx(
            feed_flow, rel=RTOL)


# Filter -> Drying flowsheet on the synthetic drying database (water,
# ethanol, nitrogen) of tests/conftest.py: a 0.1 L liquid with ethanol
# crystals, a synthetic specific cake resistance, and a dryer with its own
# solvent-poor gas, solved for 1e-8 s, as tests/test_drying_gas_balance.py.
DRYER_LIQUID_VOLUME = 1e-4  # [m**3]
DRYER_POPULATION = np.array([1e5, 2e5, 1e5])  # [#/um] on DRYING_SIZE_GRID_UM
DRYER_CAKE_RESISTANCE = 1e11  # [m/kg], synthetic, no correlation applies
FILTER_PRESSURE_DROP = 1e5  # [Pa]
DRYER_PRESSURE_DROP = 1e4  # [Pa]
DRYER_RUNTIME = 1e-8  # [s]
DRYER_LIQUID_TEMP = 302.0  # [K]
DRYER_GAS_TEMP = 300.0  # [K]
DRYER_GAS_MASS = 1e-4  # [kg]
DRYER_GAS_FLOW = 1e-4  # [kg/s]
# [-], axial nodes of the dryer: the default num_discr of the Filter's
# outlet Cake (PharmaPy.MixedPhases.Cake), because Drying reads a
# multi-node saturation profile node for node.
CAKE_NODES = 50


@pytest.mark.assimulo
@pytest.mark.integration
def test_resolved_filter_replaces_the_dryer_cake(tmp_path):
    """Keep the dryer's gas and one copy of the latest Filter cake.

    ``SimulationExec`` hands ``Connection`` the phases of the previous
    transfer on the edge; a ``Drying`` holdup holds the cake's phases,
    not the cake, so without them the earlier cake would be stacked.
    """
    pytest.importorskip('assimulo')
    thermo = tmp_path / 'drying.json'
    thermo.write_text(json.dumps(DRYING_THERMO_DATA))
    thermo = str(thermo)
    sim = SimulationExec(thermo, {'F01': ['D01'], 'D01': []})
    sim.F01 = Filter(station_diam=STATION_DIAMETER,
                     alpha=DRYER_CAKE_RESISTANCE)
    sim.F01.Phases = [
        LiquidPhase(thermo, temp=DRYER_LIQUID_TEMP, vol=DRYER_LIQUID_VOLUME,
                    mass_frac=DRYING_LIQUID_MASS_FRACTION),
        SolidPhase(thermo, temp=DRYER_LIQUID_TEMP,
                   x_distrib=DRYING_SIZE_GRID_UM.copy(),
                   distrib=DRYER_POPULATION.copy(), mass_frac=[0, 1, 0])]
    sim.D01 = Drying(CAKE_NODES, supercrit_names=['nitrogen'],
                     diam_unit=STATION_DIAMETER)
    gas = VaporPhase(thermo, temp=DRYER_GAS_TEMP, mass=DRYER_GAS_MASS,
                     mass_frac=DRYING_GAS_MASS_FRACTION)
    sim.D01.Phases = gas
    sim.D01.Inlet = VaporStream(thermo, temp=DRYER_GAS_TEMP,
                                mass_flow=DRYER_GAS_FLOW,
                                mass_frac=DRYING_GAS_MASS_FRACTION)
    run = {'F01': {'deltaP': FILTER_PRESSURE_DROP, 'verbose': False},
           'D01': {'deltaP': DRYER_PRESSURE_DROP, 'runtime': DRYER_RUNTIME,
                   'verbose': False}}

    cakes = []
    for _ in range(2):
        sim.SolveFlowsheet(kwargs_run=run, verbose=False)
        cake = sim.connections['CONN1'].transferred_matter
        cakes.append(cake)
        held = sim.D01.Phases
        assert len(held) == 3
        assert held[0] is gas
        assert held[1] is cake.Liquid_1
        assert held[2] is cake.Solid_1
    assert cakes[1] is not cakes[0]
