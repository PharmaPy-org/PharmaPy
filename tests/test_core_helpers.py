import numpy as np
import pytest

from PharmaPy.Connections import topological_bfs
from PharmaPy.Interpolation import local_newton_interpolation
from PharmaPy.NameAnalysis import get_dict_states


def test_topological_bfs_orders_linear_graph():
    graph = {
        "feed": ["reactor"],
        "reactor": ["filter"],
        "filter": [],
    }

    in_degree, path = topological_bfs(graph)

    assert path == ["feed", "reactor", "filter"]
    assert all(count == 0 for count in in_degree.values())


def test_topological_bfs_leaves_cycle_with_incoming_edges():
    graph = {
        "a": ["b"],
        "b": ["a"],
    }

    in_degree, path = topological_bfs(graph)

    assert path == []
    assert in_degree == {"a": 1, "b": 1}


def test_get_dict_states_splits_composition_distribution_and_scalar():
    states = np.array([
        [1.0, 2.0, 10.0, 20.0, 30.0, 300.0],
        [3.0, 4.0, 40.0, 50.0, 60.0, 310.0],
    ])

    result = get_dict_states(
        ["mole_conc", "mu_n", "temp"],
        num_species=2,
        num_distr=3,
        states=states,
    )

    np.testing.assert_allclose(result["mole_conc"], states[:, :2])
    np.testing.assert_allclose(result["mu_n"], states[:, 2:5])
    np.testing.assert_allclose(result["temp"], states[:, 5])


def test_local_newton_interpolation_matches_quadratic():
    time = np.array([0.0, 1.0, 2.0, 3.0])
    values = time**2

    interpolated = local_newton_interpolation(1.5, time, values)

    assert interpolated == pytest.approx(2.25)


def test_local_newton_interpolation_at_final_node_returns_exact_value():
    """Regression test for #77: querying exactly at the last data node
    used to collapse the interpolation window to a single point (the
    second-to-last node), returning the wrong value at the boundary."""
    time = np.linspace(0.0, 10.0, 11)
    values = time**2

    interpolated = local_newton_interpolation(10.0, time, values)

    assert interpolated == pytest.approx(100.0)


def test_local_newton_interpolation_just_inside_final_node():
    """Query just inside the final node -- still within the affected
    boundary window, but not exactly on the last data point."""
    time = np.linspace(0.0, 10.0, 11)
    values = time**2

    interpolated = local_newton_interpolation(9.5, time, values)

    assert interpolated == pytest.approx(90.25)


def test_local_newton_interpolation_away_from_boundary_unaffected():
    """Control case: a query comfortably away from the final node was
    never affected by the boundary bug, so this should pass both before
    and after the fix -- confirms the fix doesn't disturb interior
    interpolation."""
    time = np.linspace(0.0, 10.0, 11)
    values = time**2

    interpolated = local_newton_interpolation(5.0, time, values)

    assert interpolated == pytest.approx(25.0)
