"""Opt-in controlled finite-difference Jacobian for ScipyBackend.

Scope: the ``jacobian_relative_step`` / ``jacobian_state_scale`` options. With
the options absent, solve_ivp must receive no ``jac`` (upstream behavior is
unchanged). With them, solve_ivp must receive the fixed-perturbation Jacobian,
whose values are checked against independently known derivatives, and a real
batch reactor must reproduce its default-Jacobian trajectory. Fixtures reuse
``build_reactor`` from ``test_scipy_backend.py`` (five-species database, two
reactions). Cost: three short solves.
"""

import numpy as np
import pytest

import PharmaPy.IntegratorBackends as backends
from PharmaPy.IntegratorBackends import ScipyBackend, _controlled_difference_jacobian
from test_scipy_backend import GRID, build_reactor

pytestmark = pytest.mark.unit

JACOBIAN = {"jacobian_relative_step": 1e-6, "jacobian_state_scale": 1e-8}  # [-], [state units]


class RecordingSolveIvp:
    """Wrap ``solve_ivp`` and keep the keyword arguments of every call."""

    def __init__(self, original):
        """Store the wrapped solver.

        Parameters
        ----------
        original : callable
            The real ``scipy.integrate.solve_ivp``.
        """
        self.original = original
        self.calls = []

    def __call__(self, *args, **kwargs):
        """Record ``kwargs`` and delegate to the real solver.

        Parameters
        ----------
        *args, **kwargs
            Arguments forwarded unchanged to ``solve_ivp``.

        Returns
        -------
        OdeResult
            The real solver's result.
        """
        self.calls.append(dict(kwargs))
        return self.original(*args, **kwargs)


def capture_solve_ivp(monkeypatch):
    """Record the keyword arguments of every solve_ivp call.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fixture used to wrap ``solve_ivp`` inside the backends module.

    Returns
    -------
    list of dict
        One keyword mapping per call, filled during the solve.
    """
    recorder = RecordingSolveIvp(backends.solve_ivp)
    monkeypatch.setattr(backends, "solve_ivp", recorder)
    return recorder.calls


def test_linear_rhs_gives_its_exact_matrix():
    matrix = np.array([[-2.0, 0.5, 0.0], [1.0, -3.0, 0.25], [0.0, 4.0, -1.0]])  # [1/s]
    states = np.array([3.0, -0.2, 1e-9])  # [state units]; mixed signs and a tiny entry
    scale = np.array([1e-3, 1e-3, 1e-3])  # [state units]
    step = 1e-6  # [-]
    jacobian = _controlled_difference_jacobian(lambda t, y: matrix @ y, step, scale, 0.0, states)
    # A forward difference of a linear map has no truncation error; only
    # roundoff remains, about eps * |f| / h per entry. Allow four times that.
    increments = step * np.maximum(np.abs(states), scale)  # [state units]
    roundoff = 4 * np.finfo(float).eps * np.max(np.abs(matrix @ states)) / increments  # [1/s]
    np.testing.assert_array_less(np.abs(jacobian - matrix), np.broadcast_to(roundoff, matrix.shape))


def test_increment_uses_the_larger_of_state_and_scale():
    # rhs = y**2 has derivative 2y; a forward difference with increment h
    # returns 2y + h, so the error identifies the increment actually used.
    states = np.array([10.0, 1e-12])  # [state units]
    scale = np.array([1e-3, 1e-3])  # [state units]
    step = 1e-4  # [-]
    jacobian = _controlled_difference_jacobian(lambda t, y: y ** 2, step, scale, 0.0, states)
    expected_increment = step * np.maximum(np.abs(states), scale)  # [1e-3, 1e-7]
    np.testing.assert_allclose(np.diag(jacobian) - 2 * states, expected_increment, rtol=1e-6)
    assert jacobian[0, 1] == 0.0 and jacobian[1, 0] == 0.0


def test_absent_options_leave_scipy_jacobian_untouched(monkeypatch):
    calls = capture_solve_ivp(monkeypatch)
    build_reactor(ScipyBackend()).solve_unit(time_grid=GRID)
    assert calls and all("jac" not in kwargs for kwargs in calls)
    assert "jacobian_relative_step" not in ScipyBackend(JACOBIAN).translate_options()


def test_options_hand_the_controlled_jacobian_to_solve_ivp(monkeypatch):
    calls = capture_solve_ivp(monkeypatch)
    reactor = build_reactor(ScipyBackend(dict(JACOBIAN, rtol=1e-8, atol=1e-10)))
    reactor.solve_unit(time_grid=GRID)
    jac = calls[0]["jac"]
    assert jac.func is _controlled_difference_jacobian
    assert jac.args[1] == JACOBIAN["jacobian_relative_step"]
    # The scalar scale is broadcast to one floor per packed solver state.
    num_states = reactor.solver_state_collection.dim
    np.testing.assert_array_equal(jac.args[2], np.full(num_states, JACOBIAN["jacobian_state_scale"]))


def test_controlled_jacobian_reproduces_default_trajectory():
    default = build_reactor(ScipyBackend({"rtol": 1e-8, "atol": 1e-10}))
    controlled = build_reactor(ScipyBackend(dict(JACOBIAN, rtol=1e-8, atol=1e-10)))
    _, default_states = default.solve_unit(time_grid=GRID)
    _, controlled_states = controlled.solve_unit(time_grid=GRID)
    # The Jacobian only steers Newton iterations; both solutions meet the same
    # rtol 1e-8, so they agree to a small multiple of it.
    np.testing.assert_allclose(controlled_states, default_states, rtol=1e-6, atol=1e-9)


@pytest.mark.parametrize("options, message", [
    ({"jacobian_relative_step": 1e-6}, "together"),
    ({"jacobian_state_scale": 1e-8}, "together"),
    ({"jacobian_relative_step": 0.0, "jacobian_state_scale": 1e-8}, "finite and positive"),
    ({"jacobian_relative_step": 1e-6, "jacobian_state_scale": -1.0}, "finite and positive"),
    ({"jacobian_relative_step": float("nan"), "jacobian_state_scale": 1e-8}, "finite and positive"),
    ({"jacobian_relative_step": 1e-6, "jacobian_state_scale": [1e-8, 1e-8]}, "one value per state"),
    (dict(JACOBIAN, jac=lambda t, y: None), "not both"),
])
def test_invalid_controlled_jacobian_options_are_rejected(options, message):
    with pytest.raises(ValueError, match=message):
        build_reactor(ScipyBackend(options)).solve_unit(time_grid=GRID)
