"""Rejection of flowsheet fan-out by ``SimulationExec`` (issue #437).

``Connection.FeedConnection`` hands every successor of a unit a copy of its
whole outlet and PharmaPy has no split contract, so a unit connected to
several successors would duplicate its outlet mass in every branch.
``SimulationExec.SolveFlowsheet`` must therefore raise
``PharmaPyNonImplementedError`` naming the source and its successors before
any unit is solved or any stream is transferred, irrespective of
``pick_units``; a successor listed twice is a fan-out too. Building such a
graph stays legal, and a single successor still receives the whole outlet.

Unit cases solve real batch liquid ``Mixer`` units, which need no ODE
backend, on the shipped five-species database
(tests/Flowsheet/data/compound_database.json). Each source mixes two
liquid charges, FEED_MASSES = (1.0, 2.0) kg with compositions FEED_A and
FEED_B; the expected outlet amounts are their sums. The ``assimulo`` case
is the issue reproduction: a continuous ``Mixer`` of 1.0 kg/s and 0.5 kg/s
feeding two ``DynamicCollector`` units, which solve with CVode when the
guard is missing.

Related issues: https://github.com/PharmaPy-org/PharmaPy/issues/437,
https://github.com/PharmaPy-org/PharmaPy/issues/340
"""

import numpy as np
import pytest

from PharmaPy.Containers import Mixer
from PharmaPy.Errors import PharmaPyNonImplementedError
from PharmaPy.Phases import LiquidPhase
from PharmaPy.SimExec import SimulationExec
from PharmaPy.Streams import LiquidStream

FEED_A = np.array([0.1, 0.1, 0.1, 0.1, 0.6])  # [-], issue #437 feed A
FEED_B = np.array([0.0, 0.2, 0.0, 0.0, 0.8])  # [-], issue #437 feed B
FEED_MASSES = (1.0, 2.0)  # [kg], unequal batch charges of feeds A and B
FEED_TEMPS = (300.0, 320.0)  # [K], issue #437 feed temperatures
FEED_FLOWS = (1.0, 0.5)  # [kg/s], issue #437 continuous feeds
RUNTIME = 10.0  # [s], issue #437 collector run
# [-], float64 roundoff of the Mixer's algebraic sums of a few kilograms
RTOL = 1e-12


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


def _batch_source(path):
    """Build a batch liquid Mixer charged with feeds A and B.

    Parameters
    ----------
    path : str
        Database path.

    Returns
    -------
    Mixer
        Unsolved batch Mixer holding FEED_MASSES [kg] at FEED_TEMPS [K].
    """
    mixer = Mixer()
    mixer.Inlets = [
        LiquidPhase(path, temp=temp, mass=mass, mass_frac=fraction)
        for fraction, mass, temp in zip((FEED_A, FEED_B), FEED_MASSES,
                                        FEED_TEMPS)]
    return mixer


def _batch_flowsheet(path, graph):
    """Assign a batch source Mixer to every source and empty Mixers elsewhere.

    Parameters
    ----------
    path : str
        Database path.
    graph : dict of str to list of str
        Flowsheet adjacency.

    Returns
    -------
    SimulationExec
        Executor whose graph sources hold charged batch Mixers.
    """
    sim = SimulationExec(path, graph)
    destinations = {name for successors in graph.values()
                    for name in successors}
    for name in sim.execution_names:
        unit = Mixer() if name in destinations else _batch_source(path)
        setattr(sim, name, unit)
    return sim


def _assert_nothing_transferred(sim, destinations):
    """Assert that no flowsheet transfer or result was recorded.

    Parameters
    ----------
    sim : SimulationExec
        Executor after a rejected ``SolveFlowsheet`` call.
    destinations : sequence of str
        Names of the destination Mixers, which must hold no inlet.
    """
    assert sim.uos_instances == {}
    assert not hasattr(sim, 'connections')
    for name in destinations:
        assert getattr(sim, name).Inlets == []


@pytest.mark.unit
@pytest.mark.parametrize('container', [list, tuple, np.array],
                         ids=['list', 'tuple', 'array'])
def test_fan_out_is_rejected_before_any_unit_is_solved(path, container):
    """Name the source and both successors; solve and transfer nothing.

    Ordered successor containers accepted by ``topological_bfs`` give the
    same plain names in the message.
    """
    graph = {'A': container(['B', 'C']), 'B': container([]),
             'C': container([])}
    sim = _batch_flowsheet(path, graph)
    assert sim.execution_names == ['A', 'B', 'C']  # construction is legal

    with pytest.raises(PharmaPyNonImplementedError,
                       match=r"'A' feeds \['B', 'C'\]\. Every successor "
                             r"would receive a full copy of the outlet "
                             r"stream.*Connect each of these units to one "
                             r"successor only.*issue #340"):
        sim.SolveFlowsheet(verbose=False)

    assert not hasattr(sim.A, 'result')
    assert not hasattr(sim.A, 'Outlet')
    _assert_nothing_transferred(sim, ['B', 'C'])


@pytest.mark.unit
def test_fan_out_is_rejected_when_one_successor_is_picked(path):
    """Reject the graph when the picks reach one branch, call after call.

    Picking A with one successor B would transfer the whole outlet to B,
    and a later call picking the other successor C would transfer it again
    from the solved A, so both calls must fail before any transfer.
    """
    sim = _batch_flowsheet(path, {'A': ['B', 'C'], 'B': [], 'C': []})

    for pick_units in (['A', 'B'], ['C']):
        with pytest.raises(PharmaPyNonImplementedError,
                           match=r"'A' feeds \['B', 'C'\]"):
            sim.SolveFlowsheet(pick_units=pick_units, verbose=False)

        assert not hasattr(sim.A, 'result')
        _assert_nothing_transferred(sim, ['B', 'C'])


@pytest.mark.unit
def test_fan_out_is_rejected_for_an_already_solved_unpicked_source(path):
    """Reject a solved source although only one successor is picked."""
    sim = _batch_flowsheet(path, {'A': ['B', 'C'], 'B': [], 'C': []})
    sim.A.solve_unit()  # an already solved unit transfers when not picked

    with pytest.raises(PharmaPyNonImplementedError,
                       match=r"'A' feeds \['B', 'C'\]"):
        sim.SolveFlowsheet(pick_units=['C'], verbose=False)

    _assert_nothing_transferred(sim, ['B', 'C'])


@pytest.mark.unit
def test_duplicated_successor_entry_is_fan_out(path):
    """Reject a successor listed twice, which would get two full copies."""
    sim = _batch_flowsheet(path, {'A': ['B', 'B'], 'B': []})

    with pytest.raises(PharmaPyNonImplementedError,
                       match=r"'A' feeds \['B', 'B'\]"):
        sim.SolveFlowsheet(verbose=False)

    _assert_nothing_transferred(sim, ['B'])


@pytest.mark.unit
def test_every_fanned_out_unit_is_named(path):
    """List every offending source in graph declaration order."""
    graph = {'S2': ['X', 'Y'], 'S1': ['Z', 'W'], 'X': [], 'Y': [], 'Z': [],
             'W': []}
    sim = _batch_flowsheet(path, graph)

    with pytest.raises(PharmaPyNonImplementedError,
                       match=r"several successors: 'S2' feeds \['X', 'Y'\]; "
                             r"'S1' feeds \['Z', 'W'\]\. "):
        sim.SolveFlowsheet(verbose=False)

    _assert_nothing_transferred(sim, ['X', 'Y', 'Z', 'W'])


@pytest.mark.unit
def test_single_successor_receives_the_whole_outlet(path):
    """Keep one-successor handoff: the successor holds the fed charges."""
    sim = _batch_flowsheet(path, {'A': ['B'], 'B': []})

    sim.SolveFlowsheet(verbose=False)

    total_mass = FEED_MASSES[0] + FEED_MASSES[1]  # [kg], 1 + 2 kg
    # [kg], per-species charges of feeds A and B
    species_mass = FEED_MASSES[0] * FEED_A + FEED_MASSES[1] * FEED_B
    assert len(sim.connections) == 1
    assert len(sim.B.Inlets) == 1
    assert sim.B.Inlets[0].mass == pytest.approx(total_mass, rel=RTOL)
    assert sim.B.Outlet.mass == pytest.approx(total_mass, rel=RTOL)
    np.testing.assert_allclose(sim.B.Outlet.mass * sim.B.Outlet.mass_frac,
                               species_mass, rtol=RTOL)


@pytest.mark.assimulo
@pytest.mark.integration
def test_issue_reproduction_mixer_to_two_collectors_is_rejected(path):
    """Reject the issue #437 graph instead of collecting 30.3 kg of 15 kg."""
    pytest.importorskip('assimulo')
    from PharmaPy.Containers import DynamicCollector

    mixer = Mixer()
    mixer.Inlets = [
        LiquidStream(path, temp=temp, mass_flow=flow, mass_frac=fraction)
        for fraction, flow, temp in zip((FEED_A, FEED_B), FEED_FLOWS,
                                        FEED_TEMPS)]
    sim = SimulationExec(path, {'M01': ['C01', 'C02'], 'C01': [], 'C02': []})
    sim.M01 = mixer
    sim.C01, sim.C02 = DynamicCollector(), DynamicCollector()

    with pytest.raises(PharmaPyNonImplementedError,
                       match=r"'M01' feeds \['C01', 'C02'\]"):
        sim.SolveFlowsheet(
            kwargs_run={'C01': {'runtime': RUNTIME, 'verbose': False},
                        'C02': {'runtime': RUNTIME, 'verbose': False}},
            verbose=False)

    assert not hasattr(mixer, 'result')
    for collector in (sim.C01, sim.C02):
        assert collector.Inlet is None
        assert not hasattr(collector, 'result')
    assert sim.uos_instances == {}
