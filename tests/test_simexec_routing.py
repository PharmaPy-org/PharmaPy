"""Regression tests for graph-defined flowsheet stream routing.

The cases solve real batch mixers and transfer real liquid phases through real
``Connection`` and ``SimulationResult`` collaborators. A graph in which one
unit feeds several successors is rejected before any transfer (issue #437).
"""

import numpy as np
import pytest

from PharmaPy.Containers import Mixer
from PharmaPy.Errors import PharmaPyNonImplementedError
from PharmaPy.Phases import LiquidPhase
from PharmaPy.SimExec import SimulationExec


pytestmark = pytest.mark.unit

FEED_A_MASS_FRACTION = np.array([0.70, 0.20, 0.05, 0.03, 0.02])  # [-]
FEED_B_MASS_FRACTION = np.array([0.10, 0.20, 0.30, 0.20, 0.20])  # [-]


def _add_real_feed(mixer, database_path, temperature, mass_scale=1.0):
    """Assign two real liquid phases to a source mixer.

    Parameters
    ----------
    mixer : Mixer
        Source unit that receives batch liquid phases.
    database_path : str
        Path to the repository thermophysical database.
    temperature : float
        Temperature of the first liquid phase [K]. The second phase is 1 K
        warmer to exercise the real energy balance.
    mass_scale : float, optional
        Dimensionless source-specific mass multiplier [-].
    """
    mixer.Inlets = LiquidPhase(
        database_path,
        mass=1.0 * mass_scale,  # [kg]
        mass_frac=FEED_A_MASS_FRACTION,
        temp=temperature,
    )
    mixer.Inlets = LiquidPhase(
        database_path,
        mass=2.0 * mass_scale,  # [kg]
        mass_frac=FEED_B_MASS_FRACTION,
        temp=temperature + 1.0,  # [K]
    )


def _configured_flowsheet(graph, database_path):
    """Build a real mixer flowsheet and feed every graph source.

    Parameters
    ----------
    graph : dict of str to list of str
        Directed acyclic flowsheet adjacency mapping.
    database_path : str
        Path to the repository thermophysical database.

    Returns
    -------
    SimulationExec
        Executor containing real ``Mixer`` units and source phases.
    """
    flowsheet = SimulationExec(database_path, flowsheet=graph)
    destinations = {
        destination
        for successors in graph.values()
        for destination in successors
    }
    for index, name in enumerate(graph):
        mixer = Mixer()
        setattr(flowsheet, name, mixer)
        if name not in destinations:
            source_temperature = 300.0 + 5.0 * index  # [K]
            source_mass_scale = index + 1.0  # [-]
            _add_real_feed(
                mixer, database_path, source_temperature, source_mass_scale
            )

    return flowsheet


def _connection_edges(flowsheet):
    """Map real connection endpoints back to their graph unit names.

    Parameters
    ----------
    flowsheet : SimulationExec
        Solved flowsheet containing real connection objects.

    Returns
    -------
    list of tuple of str
        Source-destination names in connection creation order.
    """
    unit_names = {
        id(getattr(flowsheet, name)): name
        for name in flowsheet.graph
    }
    return [
        (
            unit_names[id(connection.source_uo)],
            unit_names[id(connection.destination_uo)],
        )
        for connection in flowsheet.connections.values()
    ]


def _assert_connections_follow_graph(graph, created):
    """Assert every real handoff is an edge in the supplied graph.

    Parameters
    ----------
    graph : dict of str to list of str
        Directed acyclic flowsheet adjacency mapping.
    created : sequence of tuple of str
        Observed source-destination handoffs.
    """
    misrouted = [
        (source, destination)
        for source, destination in created
        if destination not in graph[source]
    ]
    assert not misrouted


def test_stream_handoff_follows_graph_edges_not_execution_order(data_path):
    """Route real phase handoffs through graph edges, not adjacent units.

    Parameters
    ----------
    data_path : dict of pathlib.Path
        Repository test-data directories.
    """
    graph = {"A": ["C"], "B": ["C"], "C": ["D"], "D": []}
    database_path = str(data_path["flowsheet"] / "compound_database.json")
    flowsheet = _configured_flowsheet(graph, database_path)

    flowsheet.SolveFlowsheet(verbose=False)
    created = _connection_edges(flowsheet)

    _assert_connections_follow_graph(graph, created)
    assert created == [("A", "C"), ("B", "C"), ("C", "D")]
    assert len(flowsheet.C.Inlets) == 2
    assert len(flowsheet.D.Inlets) == 1
    source_masses = np.array([3.0, 6.0])  # [kg]
    # Sources A and B are both ready first; FIFO traversal runs them in
    # declaration order, so C receives A's inlet before B's (issue #309).
    destination_inlets = flowsheet.C.Inlets
    expected_sources = [flowsheet.A.Outlet, flowsheet.B.Outlet]
    np.testing.assert_allclose(
        [inlet.mass for inlet in destination_inlets], source_masses
    )
    for inlet, source in zip(destination_inlets, expected_sources):
        np.testing.assert_allclose(inlet.mass_frac, source.mass_frac)
        assert inlet.temp == pytest.approx(source.temp)
    assert flowsheet.D.Inlets[0].mass == pytest.approx(source_masses.sum())
    np.testing.assert_allclose(
        flowsheet.D.Inlets[0].mass_frac, flowsheet.C.Outlet.mass_frac
    )
    assert flowsheet.D.Inlets[0].temp == pytest.approx(
        flowsheet.C.Outlet.temp
    )


def test_stream_handoff_rejects_fanout_before_any_transfer(data_path):
    """Reject a real source phase routed to two graph successors.

    Copying the outlet to every successor would duplicate its mass, so the
    diamond is rejected before any unit is solved, although its downstream
    fan-in into Mixer D alone would be legal (issue #437).

    Parameters
    ----------
    data_path : dict of pathlib.Path
        Repository test-data directories.
    """
    graph = {"A": ["B", "C"], "B": ["D"], "C": ["D"], "D": []}
    database_path = str(data_path["flowsheet"] / "compound_database.json")
    flowsheet = _configured_flowsheet(graph, database_path)

    with pytest.raises(PharmaPyNonImplementedError,
                       match=r"'A' feeds \['B', 'C'\]\. Every successor "
                             r"would receive a full copy of the outlet"):
        flowsheet.SolveFlowsheet(verbose=False)

    assert not hasattr(flowsheet, "connections")
    assert not hasattr(flowsheet.A, "Outlet")
    assert flowsheet.B.Inlets == []
    assert flowsheet.C.Inlets == []
    assert flowsheet.D.Inlets == []


def test_already_solved_branch_honors_pick_units(data_path):
    """Filter a real pre-solved unit's handoffs to selected destinations.

    B and E are solved before the flowsheet and not picked: B still hands
    its phase to the picked destination C, while E's destination D is not
    picked, so D receives nothing.

    Parameters
    ----------
    data_path : dict of pathlib.Path
        Repository test-data directories.
    """
    graph = {"A": ["C"], "B": ["C"], "E": ["D"], "C": [], "D": []}
    database_path = str(data_path["flowsheet"] / "compound_database.json")
    flowsheet = _configured_flowsheet(graph, database_path)
    flowsheet.B.solve_unit()
    flowsheet.E.solve_unit()

    flowsheet.SolveFlowsheet(pick_units=["A", "C"], verbose=False)
    created = _connection_edges(flowsheet)

    _assert_connections_follow_graph(graph, created)
    assert created == [("A", "C"), ("B", "C")]
    # Sources A and B are scaled by 1 and 2: (1 + 2 kg) and 2 * (1 + 2 kg).
    source_masses = np.array([3.0, 6.0])  # [kg]
    np.testing.assert_allclose(
        [inlet.mass for inlet in flowsheet.C.Inlets], source_masses
    )
    assert flowsheet.C.Outlet.mass == pytest.approx(source_masses.sum())
    assert flowsheet.D.Inlets == []
