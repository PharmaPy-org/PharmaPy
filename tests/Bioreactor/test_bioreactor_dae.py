"""Native ODE/DAE evaluation and optional IDA integration regressions."""

from pathlib import Path

import numpy as np
import pytest

from PharmaPy.DataClasses import IntraPhaseProcess, PhaseRef, StateEvent, StateKey, StateVariable, TransferResult
from PharmaPy.IntegratorBackends import AssimuloBackend, AssimuloDAEBackend, SciPyBackend, FixedStepBackend
from PharmaPy.Mechanisms import Mechanism
from PharmaPy.MultiPhaseVessel import MultiPhaseVessel
from PharmaPy.Phases_Refactored import LiquidPhase


class _CoupledMechanism(Mechanism):
    def __init__(self):
        super().__init__()
        self.constraint = 999.0
        self.solver_states = (StateVariable("constraint", 1, "kg", state_type="alg"),)
        self._update_exposed_attributes()
        self._timers = {}

    def get_overrides(self, name=None):
        return {} if name is None else None

    def get_solver_state_rates(self, process, phase, **kwargs):
        rate = phase.mass_j[0]
        return TransferResult({StateKey("mass_j", process.phaseref):
                               np.array([-rate, rate, 0., 0.])}, {}, 0.)

    def get_solver_state_residuals(self, completed_state, **kwargs):
        ref = self.solver_state_keys[0].phaseref
        return {StateKey("constraint", ref): completed_state[StateKey("constraint", ref)]
                - 2. * completed_state[StateKey("mass_j", ref)][0]}


@pytest.fixture
def vessel():
    thermo = Path(__file__).parents[2] / "examples/bioreactors/batch_ecoli_dfba/inputs/thermo.json"
    phase = LiquidPhase(thermo, mass=1., mass_frac=np.full(4, .25), verbose=False)
    mechanism = _CoupledMechanism()
    phase.mechanisms = mechanism
    unit = MultiPhaseVessel(isothermal=True)
    unit.Phases = phase
    unit.intraphase_processes = [IntraPhaseProcess(PhaseRef("liquid", 0), mechanism)]
    unit.compile_structure()
    return unit


def test_unit_model_dae_modes_and_ode_guard(vessel):
    states = vessel.create_solver_init_states()
    rates = vessel.create_solver_init_derivatives(states)
    np.testing.assert_allclose(rates, [-.25, .25, 0., 0., 0.])
    derivatives = np.array([1., 2., 3., 4., 100.])
    expected = [1.25, 1.75, 3., 4., 998.5]
    np.testing.assert_allclose(vessel.dae_residual(0., states, derivatives), expected)
    np.testing.assert_allclose(vessel.unit_model(0., states), [-.25, .25, 0., 0., 998.5])
    np.testing.assert_allclose(AssimuloDAEBackend().make_residual(vessel)(0., states, derivatives), expected)
    constraints = vessel.unit_model(0., states, alg_bce=True)
    np.testing.assert_allclose(constraints[StateKey("constraint", PhaseRef("liquid", 0))], 998.5)
    for backend in (AssimuloBackend(), SciPyBackend(), FixedStepBackend()):
        with pytest.raises(NotImplementedError, match="algebraic"):
            backend.compile_integrator(vessel)
    # Returned derivative guesses must survive later evaluations using the buffers.
    np.testing.assert_allclose(rates, [-.25, .25, 0., 0., 0.])


def test_initial_derivatives_do_not_require_algebraic_evaluation(vessel, monkeypatch):
    def unavailable(*args, **kwargs):
        raise AssertionError("Initialization must not evaluate algebraic constraints")
    monkeypatch.setattr(vessel, "algebraic_balances", unavailable)
    np.testing.assert_allclose(vessel.create_solver_init_derivatives(), [-.25, .25, 0., 0., 0.])


def test_dae_missing_equation_fails_explicitly(vessel, monkeypatch):
    states = vessel.create_solver_init_states()
    monkeypatch.setattr(vessel, "algebraic_balances", lambda *args: {})
    with pytest.raises(KeyError, match="No algebraic residual"):
        vessel.dae_residual(0., states, np.zeros_like(states))


def test_dae_global_energy_state_and_equation_guard(vessel, monkeypatch):
    vessel.isothermal = False
    key = StateKey("global_temp")
    vessel.solver_state_collection.add(StateVariable("global_temp", 1, "K", state_type="diff"))
    vessel.compile_structure()
    monkeypatch.setattr(vessel, "energy_balances", lambda *args: {key: 2.})
    states = vessel.create_solver_init_states()
    rates = vessel.create_solver_init_derivatives(states)
    index = vessel.solver_state_collection.slices[key]
    np.testing.assert_allclose(rates[index], 2.)
    residual = vessel.dae_residual(0., states, rates)
    np.testing.assert_allclose(residual[vessel.solver_state_collection.differential_mask], 0.)
    monkeypatch.setattr(vessel, "energy_balances", lambda *args: {})
    with pytest.raises(KeyError, match="No differential rate"):
        vessel.dae_residual(0., states, rates)


def test_ode_rates_unchanged_by_dae_evaluation(vessel):
    del vessel.solver_state_collection.states[StateKey("constraint", PhaseRef("liquid", 0))]
    vessel.compile_structure()
    states = vessel.create_solver_init_states()
    expected = [-.25, .25, 0., 0.]
    np.testing.assert_allclose(vessel.unit_model(0., states), expected)
    np.testing.assert_allclose(vessel.unit_model(0., states, mat_bce=True), expected)
    np.testing.assert_allclose(vessel.create_solver_init_derivatives(states), expected)
    np.testing.assert_allclose(vessel.unit_model(0., states), expected)


def test_ida_runs_through_solve_unit(vessel):
    pytest.importorskip("assimulo.solvers.sundials")
    vessel.integrator = AssimuloDAEBackend({"rtol": 1e-9, "atol": 1e-11})
    times, states = vessel.solve_unit(time_grid=np.linspace(0., 1., 11), verbose=False)
    expected = .25 * np.exp(-np.asarray(times))
    np.testing.assert_allclose(states[:, 0], expected, rtol=1e-7, atol=1e-9)
    np.testing.assert_allclose(states[:, 1], .5 - expected, rtol=1e-7, atol=1e-9)
    np.testing.assert_allclose(states[:, -1], 2. * expected, rtol=1e-7, atol=1e-9)
    np.testing.assert_allclose(vessel.result.mass_j_liquid0, states[:, :4])


def test_phase_owned_algebraic_equation_is_collected_once(vessel, monkeypatch):
    mechanism = vessel.intraphase_processes[0].mechanism
    original = mechanism.get_solver_state_residuals
    calls = []

    def residuals(**kwargs):
        calls.append(kwargs["time"])
        return original(**kwargs)

    monkeypatch.setattr(mechanism, "get_solver_state_residuals", residuals)
    vessel.unit_model(0., vessel.create_solver_init_states())
    assert calls == [0.]
    vessel.intraphase_processes = []
    vessel.compile_structure()
    values = vessel.unit_model(1., vessel.create_solver_init_states())
    np.testing.assert_allclose(values, [0., 0., 0., 0., 998.5])
    assert calls == [0., 1.]


def test_ida_continues_and_reinitializes_constraints(vessel):
    pytest.importorskip("assimulo.solvers.sundials")
    vessel.integrator = AssimuloDAEBackend({"rtol": 1e-9, "atol": 1e-11})
    vessel.solve_unit(runtime=.5, verbose=False)
    times, states = vessel.solve_unit(time_grid=np.linspace(.5, 1., 6), verbose=False)
    np.testing.assert_allclose(states[:, 0], .25 * np.exp(-np.asarray(times)), rtol=1e-7)
    np.testing.assert_allclose(states[:, -1], 2. * states[:, 0], atol=1e-9)


def test_ida_locates_parent_state_event(vessel):
    pytest.importorskip("assimulo.solvers.sundials")
    vessel.state_event_list = [StateEvent(
        "stop", lambda time, completed_state, unit: time - .4, terminal=True)]
    vessel.integrator = AssimuloDAEBackend({"rtol": 1e-9, "atol": 1e-11})
    times, states = vessel.solve_unit(runtime=1., verbose=False)
    assert times[-1] == pytest.approx(.4, abs=1e-8)
    np.testing.assert_allclose(states[-1, -1], 2. * states[-1, 0], atol=1e-9)
