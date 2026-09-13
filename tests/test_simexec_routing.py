"""Regression tests for graph-defined flowsheet stream routing.

The cases solve real batch mixers and transfer real liquid phases through real
``Connection`` and ``SimulationResult`` collaborators.
"""

import numpy as np
import pytest

from PharmaPy.Containers import Mixer
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
    assert set(created) == {("A", "C"), ("B", "C"), ("C", "D")}
    assert len(flowsheet.C.Inlets) == 2
    assert len(flowsheet.D.Inlets) == 1
    source_masses = np.array([3.0, 6.0])  # [kg]
    destination_inlets = sorted(
        flowsheet.C.Inlets, key=lambda inlet: inlet.mass
    )
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


def test_stream_handoff_supports_fanout_to_multiple_successors(data_path):
    """Copy a real source phase to every graph successor.

    Parameters
    ----------
    data_path : dict of pathlib.Path
        Repository test-data directories.
    """
    graph = {"A": ["B", "C"], "B": ["D"], "C": ["D"], "D": []}
    database_path = str(data_path["flowsheet"] / "compound_database.json")
    flowsheet = _configured_flowsheet(graph, database_path)

    flowsheet.SolveFlowsheet(verbose=False)
    created = _connection_edges(flowsheet)

    _assert_connections_follow_graph(graph, created)
    assert set(created) == {
        ("A", "B"),
        ("A", "C"),
        ("B", "D"),
        ("C", "D"),
    }
    assert len(flowsheet.B.Inlets) == 1
    assert len(flowsheet.C.Inlets) == 1
    assert len(flowsheet.D.Inlets) == 2


def test_already_solved_branch_honors_pick_units(data_path):
    """Filter a real pre-solved unit's handoffs to selected destinations.

    Parameters
    ----------
    data_path : dict of pathlib.Path
        Repository test-data directories.
    """
    graph = {"A": ["B", "C"], "B": ["C", "D"], "C": [], "D": []}
    database_path = str(data_path["flowsheet"] / "compound_database.json")
    flowsheet = _configured_flowsheet(graph, database_path)
    _add_real_feed(
        flowsheet.B, database_path, temperature=307.0, mass_scale=2.0
    )
    flowsheet.B.solve_unit()

    flowsheet.SolveFlowsheet(pick_units=["A", "C"], verbose=False)
    created = _connection_edges(flowsheet)

    _assert_connections_follow_graph(graph, created)
    assert created == [("A", "C"), ("B", "C")]
    assert len(flowsheet.C.Inlets) == 2
    assert flowsheet.D.Inlets == []
