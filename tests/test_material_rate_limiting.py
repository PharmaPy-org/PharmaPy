"""Conservation and stream consistency when native inventory limiting activates."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from PharmaPy.DataClasses import (MaterialContributionBuffer, PhaseRef,
                                  ResolvedPhaseTransfer, StateCollection,
                                  StateKey, StateVariable)
from PharmaPy.MultiPhaseVessel import MultiPhaseVessel
from PharmaPy.Streams_Refactored import LiquidStream
from PharmaPy.Phases_Refactored import LiquidPhase


def system(inventories):
    unit = MultiPhaseVessel(isothermal=True)
    unit.solver_state_collection = StateCollection()
    state = {}
    for index, inventory in enumerate(inventories):
        ref = PhaseRef("liquid", index)
        unit.solver_state_collection.add(StateVariable(
            "mass_j", len(inventory), "kg", state_type="diff", phaseref=ref))
        state[StateKey("mass_j", ref)] = np.array(inventory, dtype=float)
    unit._compile_positivity_layout()
    unit._material_slice_by_phase = {key.phaseref: value for key, value in
                                    unit.solver_state_collection.material_slices.items()}
    return unit, state, MaterialContributionBuffer(unit.solver_state_collection.material_dim)


def limit(unit, state, buffer, dt=1.):
    if np.any(buffer.contributions[buffer.CROSSPHASE]) and not buffer.aux[buffer.CROSSPHASE]:
        buffer.aux[buffer.CROSSPHASE].append({"material_writes": [
            (part, buffer.contributions[buffer.CROSSPHASE, part].copy())
            for part in unit.solver_state_collection.material_slices.values()]})
    return unit.apply_rate_scaling(buffer, state, dt)


def test_no_depletion_and_bypass_leave_all_rates_unchanged():
    unit, state, buffer = system([[1., 1.]])
    buffer.contributions[buffer.INTRAPHASE] = [-.2, .2]
    before = buffer.contributions.copy()
    limit(unit, state, buffer)
    np.testing.assert_array_equal(buffer.contributions, before)
    buffer.contributions *= 10.
    before = buffer.contributions.copy()
    limit(unit, state, buffer, dt=0.)
    np.testing.assert_array_equal(buffer.contributions, before)


def test_reaction_depletion_is_rejected_without_altering_stoichiometry():
    unit, state, buffer = system([[1., 0.]])
    buffer.contributions[buffer.INTRAPHASE] = [-2., 2.]
    before = buffer.contributions.copy()
    with pytest.raises(RuntimeError, match="reduce the integration step"):
        limit(unit, state, buffer)
    np.testing.assert_array_equal(buffer.contributions, before)
    # A shorter step is valid and keeps the complete reaction unchanged.
    limit(unit, state, buffer, dt=.5)
    np.testing.assert_array_equal(buffer.contributions, before)


def test_transfer_pairing_and_reaction_rates_are_preserved():
    unit, state, buffer = system([[1., 0.], [0., 0.]])
    buffer.contributions[buffer.INTRAPHASE] = [-.2, .2, 0., 0.]
    buffer.contributions[buffer.CROSSPHASE] = [-2., 0., 2., 0.]
    reaction = buffer.contributions[buffer.INTRAPHASE].copy()
    limit(unit, state, buffer)
    np.testing.assert_array_equal(buffer.contributions[buffer.INTRAPHASE], reaction)
    np.testing.assert_allclose(buffer.contributions[buffer.CROSSPHASE], [-.8, 0., .8, 0.])
    assert buffer.aux[buffer.CROSSPHASE][0]["transfer_scale"] == pytest.approx(.4)
    assert buffer.contributions.sum() == pytest.approx(0.)


def test_receiver_depletion_cannot_be_hidden_by_scaling_donor():
    unit, state, buffer = system([[1.], [0.]])
    buffer.contributions[buffer.INTRAPHASE] = [0., -1.5]
    buffer.contributions[buffer.CROSSPHASE] = [-2., 2.]
    before = buffer.contributions.copy()
    with pytest.raises(RuntimeError, match="Inventory depletion"):
        limit(unit, state, buffer)
    np.testing.assert_array_equal(buffer.contributions, before)


def test_independent_transfers_keep_their_own_material_and_heat_factors():
    unit, state, buffer = system([[.5], [0.], [10.], [0.]])
    buffer.contributions[buffer.CROSSPHASE] = [-2., 2., -2., 2.]
    for offset, heat in ((0, 100.), (2, 200.)):
        mechanism = SimpleNamespace(get_heat_generation=lambda heat=heat, **kw: heat)
        buffer.aux[buffer.CROSSPHASE].append({
            "connection": SimpleNamespace(mechanism=mechanism),
            "material_writes": [(slice(offset, offset + 1), np.array([-2.])),
                                (slice(offset + 1, offset + 2), np.array([2.]))],
        })
    limit(unit, state, buffer)
    np.testing.assert_allclose(buffer.contributions[buffer.CROSSPHASE], [-.5, .5, -2., 2.])
    energy = {"crossphase": 0.}
    unit.add_crossphase_energy_terms(energy, buffer.aux[buffer.CROSSPHASE], 0., state)
    assert energy["crossphase"] == pytest.approx(225.)


@pytest.mark.parametrize("available", [0., .25])
def test_outlet_species_volume_composition_and_energy_stay_consistent(available):
    unit, state, buffer = system([[available] * 4])
    thermo = Path(__file__).parents[1] / "examples/bioreactors/batch_ecoli_dfba/inputs/thermo.json"
    stream = LiquidStream(thermo, mass_flow=4., mass_frac=np.full(4, .25), verbose=False)
    transfer = ResolvedPhaseTransfer(None, SimpleNamespace(sink_phaseref=PhaseRef("liquid", 0)), None, stream, stream.vol_flow,
                                     stream.mass_j_flow.copy(), "outlet")
    original_volume = transfer.vol_flow
    buffer.aux[buffer.OUTLET].append(transfer)
    buffer.contributions[buffer.OUTLET] = -transfer.species_flow
    limit(unit, state, buffer)
    np.testing.assert_allclose(-buffer.contributions[buffer.OUTLET], stream.mass_j_flow)
    np.testing.assert_allclose(transfer.species_flow, stream.mass_j_flow)
    np.testing.assert_allclose(stream.mass_frac, [.25] * 4)
    assert transfer.vol_flow == pytest.approx(original_volume * available)
    assert stream.vol_flow == pytest.approx(transfer.vol_flow)
    energy = {"outlet": 0.}
    unit.add_outlet_energy_terms(energy, buffer.aux[buffer.OUTLET], 0., state)
    h = stream.getEnthalpy(stream.temp, temp_ref=unit.temp_ref, total_h=True, basis="mass")
    assert energy["outlet"] == pytest.approx(-stream.mass_flow * h)


def test_transfer_heat_is_scaled_with_material(monkeypatch):
    unit, state, buffer = system([[1.], [0.]])
    buffer.contributions[buffer.CROSSPHASE] = [-2., 2.]
    limit(unit, state, buffer)
    unit._timers = {}
    # Isolate the energy assembly from thermodynamic property evaluation.
    for name in ("inlet", "intraphase", "outlet", "utility", "mixing", "shaftwork"):
        monkeypatch.setattr(unit, "add_" + name + "_energy_terms", lambda *a: None)
    mechanism = SimpleNamespace(get_heat_generation=lambda **kw: 100.)
    buffer.aux[buffer.CROSSPHASE][0]["connection"] = SimpleNamespace(mechanism=mechanism)
    thermo = Path(__file__).parents[1] / "examples/bioreactors/batch_ecoli_dfba/inputs/thermo.json"
    unit.Phases = LiquidPhase(thermo, mass=1., mass_frac=np.full(4, .25), verbose=False)
    monkeypatch.setattr(unit.Phases, "getCp", lambda **kw: 10.)
    result = unit.energy_balances(0., state, buffer)
    assert result[StateKey("global_temp")] == pytest.approx(5.)
    buffer.reset()
    assert buffer.aux[buffer.CROSSPHASE] == []


def test_native_batch_reactor_through_solve_unit():
    from PharmaPy.DataClasses import IntraPhaseProcess, TransferResult
    from PharmaPy.IntegratorBackends import FixedStepBackend
    from PharmaPy.Mechanisms import Mechanism
    from PharmaPy.Reactors_Refactored import BatchReactor

    class Conversion(Mechanism):
        def get_overrides(self, name=None):
            return {} if name is None else None

        def get_solver_state_rates(self, process, phase, **kwargs):
            rate = .5 * phase.mass_j[0]
            return TransferResult({StateKey("mass_j", process.phaseref):
                                   np.array([-rate, rate, 0., 0.])}, {}, 0.)

    thermo = Path(__file__).parents[1] / "examples/bioreactors/batch_ecoli_dfba/inputs/thermo.json"
    unit = BatchReactor(isothermal=True, integrator=FixedStepBackend())
    unit.Phases = LiquidPhase(thermo, mass=1., mass_frac=np.full(4, .25), verbose=False)
    unit.intraphase_processes = [IntraPhaseProcess(PhaseRef("liquid", 0), Conversion())]
    times, states = unit.solve_unit(time_grid=np.linspace(0., 1., 11), verbose=False)
    expected = .25 * .95 ** np.arange(11)
    np.testing.assert_allclose(states[:, 0], expected)
    np.testing.assert_allclose(states[:, 1], .5 - expected)
    np.testing.assert_allclose(states.sum(axis=1), 1.)
    np.testing.assert_allclose(unit.result.mass_j_liquid0, states)


def test_native_paired_transfer_through_solve_unit():
    from PharmaPy.DataClasses import PhaseConnection, TransferResult
    from PharmaPy.IntegratorBackends import FixedStepBackend
    from PharmaPy.Mechanisms import CrossPhaseTransferMechanism

    class Exchange(CrossPhaseTransferMechanism):
        def __init__(self):
            super().__init__()
            self.aux = {}

        def get_overrides(self, name=None):
            return {} if name is None else None

        def get_solver_state_rates(self, source_phase, connection, **kwargs):
            rate = np.array([2. * source_phase.mass_j[0], 0., 0., 0.])
            self.aux["connection"] = connection
            return TransferResult({StateKey("mass_j", connection.source_phaseref): -rate,
                                   StateKey("mass_j", connection.sink_phaseref): rate},
                                  self.aux, rate.sum())

    thermo = Path(__file__).parents[1] / "examples/bioreactors/batch_ecoli_dfba/inputs/thermo.json"
    unit = MultiPhaseVessel(isothermal=True, integrator=FixedStepBackend())
    unit.Phases = [LiquidPhase(thermo, mass=1., mass_frac=np.full(4, .25), verbose=False)
                   for _ in range(2)]
    unit.phase_connections = [PhaseConnection(PhaseRef("liquid", 0), PhaseRef("liquid", 1),
                                              None, mechanism=Exchange())]
    times, states = unit.solve_unit(time_grid=np.arange(3.), verbose=False)
    np.testing.assert_allclose(states.sum(axis=1), 2.)
    np.testing.assert_allclose(states[1:, 0], 0.)
    np.testing.assert_allclose(states[1:, 4], .5)
    assert np.all(states >= 0.)
    buffer = unit._material_contributions
    assert buffer.aux[buffer.CROSSPHASE][0]["transfer_scale"] == 1.
