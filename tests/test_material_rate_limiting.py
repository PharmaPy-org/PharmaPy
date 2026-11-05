"""Conservation and stream consistency when native inventory limiting activates."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from PharmaPy.DataClasses import (IntraPhaseProcess, MaterialContributionBuffer,
                                  PhaseConnection, PhaseRef, ResolvedPhaseTransfer,
                                  StateCollection, StateKey, StateVariable, TransferResult)
from PharmaPy.IntegratorBackends import ScipyBackend
from PharmaPy.Kinetics import RxnKinetics
from PharmaPy.Mechanisms import CrossPhaseTransferMechanism, ReactionMechanism
from PharmaPy.MultiPhaseVessel import MultiPhaseVessel
from PharmaPy.Phases_Refactored import LiquidPhase
from PharmaPy.Streams_Refactored import LiquidStream


# Real fixture construction and independent qualification helpers.
class ReactiveExchange(CrossPhaseTransferMechanism):
    """Consume donor reactants and release products into a receiving phase.

    PharmaPy has a native intraphase reaction model, but no corresponding
    cross-phase reaction adapter. This analytical extension delegates all
    kinetics and enthalpy calculations to that real native model. Its only
    additional rule assigns negative species rates to the source and positive
    rates to the sink; no rate or heat value is configured on the adapter.
    The solve_unit test below qualifies the full native cross-phase handoff.
    """

    def __init__(self, reaction: ReactionMechanism) -> None:
        """Bind the actual native reaction model.

        Parameters
        ----------
        reaction : ReactionMechanism
            Real kinetics, thermophysical phase, and reaction enthalpies.
        """
        super().__init__()
        self.reaction = reaction

    def get_overrides(self, name=None):
        """Preserve the native no-property-override contract.

        Parameters
        ----------
        name : str or None
            Requested property; None enumerates all overrides.

        Returns
        -------
        dict or None
            Empty enumeration, otherwise no named override.
        """
        return {} if name is None else None

    def get_solver_state_rates(self, source_phase: LiquidPhase, connection: PhaseConnection,
                               completed_state: dict, time: float, **kwargs) -> TransferResult:
        """Route actual native reaction rates to the two participating phases.

        Parameters
        ----------
        source_phase : LiquidPhase
            Reactant phase with component inventories [kg].
        connection : PhaseConnection
            Real source and sink phase references, sharing species order.
        completed_state : dict
            Native named state, including species masses [kg].
        time : float
            Native integration time [s].
        **kwargs : dict
            Other native transfer arguments, unused by this reaction rule.

        Returns
        -------
        TransferResult
            Paired component rates [kg/s], actual reaction auxiliary payload,
            and transferred mass rate [kg/s].

        Notes
        -----
        Splitting signs routes every signed source term exactly once; it does
        not clip a physical result. The synthetic reaction conserves mass.
        """
        self.reaction.positivity_horizon = self.positivity_horizon  # [s]
        process = IntraPhaseProcess(connection.source_phaseref, self.reaction)
        result = self.reaction.get_solver_state_rates(process, source_phase, time, completed_state)
        rates = result.state_rates[StateKey('mass_j', connection.source_phaseref)]  # [kg/s]
        consumed = np.minimum(rates, 0.)  # [kg/s], every negative term goes to the source
        produced = np.maximum(rates, 0.)  # [kg/s], every positive term goes to the sink
        aux = dict(result.aux, connection=connection)
        return TransferResult({StateKey('mass_j', connection.source_phaseref): consumed,
                               StateKey('mass_j', connection.sink_phaseref): produced},
                              aux, -consumed.sum())

    def get_heat_generation(self, aux: dict, completed_state: dict, time: float) -> float:
        """Use the actual native reaction enthalpy and returned reaction rates.

        Parameters
        ----------
        aux : dict
            Actual native reaction rates [kmol/(m3 s)] and phase metadata.
        completed_state : dict
            Native named state with temperature [K].
        time : float
            Native integration time [s].

        Returns
        -------
        float
            Real native reaction power [W], positive into the vessel.
        """
        return self.reaction.get_heat_generation(aux, completed_state, time)


def reaction_heat(rate_constant: float) -> tuple[ReactiveExchange, dict, LiquidPhase]:
    """Build a real reaction heat source for the material/heat scaling contract.

    Parameters
    ----------
    rate_constant : float
        First-order glucose conversion coefficient [1/s].

    Returns
    -------
    tuple
        Native-based ReactiveExchange, its actual rate-evaluation auxiliary payload,
        and the reacting LiquidPhase [kg, m3, K].

    Notes
    -----
    The synthetic, atom-balanced glucose -> 3 acetate reaction releases a
    declared 1000 J/mol. Initially 0.1 mol glucose and k=1 or 2 /s therefore
    give 100 or 200 W, independently of the material-scaling implementation.
    The low-level buffer fixture declares its transfer writes separately; this
    source exercises the real generic mechanism heat hook and thermal-property
    handoff, rather than a configured callback. Full native solve_unit paths
    are qualified below.
    """
    thermo = Path(__file__).parents[1] / "tests/fixtures/native_four_species.json"
    initial_moles = .1  # [mol], synthetic transparent 100 J initial reaction inventory
    substrate_mass = initial_moles * 180.156 / 1000.  # [kg], glucose MW from native_four_species.json
    phase = LiquidPhase(thermo, mass=1., mass_frac=[substrate_mass, 0., 0., 1.-substrate_mass], verbose=False)
    heat_capacity = 10.  # [J/(kg K)], synthetic constant for all four species
    phase.cp_liq = (heat_capacity * np.asarray(phase.mw) / 1000.)[:, None]  # [J/(mol K)]
    kinetics = RxnKinetics(str(thermo), k_params=[rate_constant], ea_params=[0.],
                          stoich_matrix=[[-1., 3.]], partic_species=['glucose', 'acetate'],
                          delta_hrxn=[-1000.])
    mechanism = ReactionMechanism(kinetics, phase)
    mechanism.positivity_horizon = .1  # [s], k*dt <= .2 keeps this rate evaluation away from depletion
    process = IntraPhaseProcess(PhaseRef('liquid', 0), mechanism)
    rates = mechanism.get_solver_state_rates(process, phase, 0., {})
    return ReactiveExchange(mechanism), rates.aux, phase


def system(inventories):
    """Build a real native material-buffer contract from explicit inventories.

    Parameters
    ----------
    inventories : sequence of array-like
        Per-phase species masses [kg], one vector for each liquid phase.

    Returns
    -------
    tuple
        Native vessel, named species states [kg], and material-rate buffer [kg/s].
    """
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
    """Apply the real native inventory-rate limiter without replacing collaborators.

    Parameters
    ----------
    unit : MultiPhaseVessel
        Real native vessel used by the qualification.
    state : dict
        StateKey-indexed per-phase component mass vectors [kg].
    buffer : MaterialContributionBuffer
        Species-rate contributions [kg/s] and their real auxiliary payloads.
    dt : float
        Forward inventory-limiting horizon [s].

    Returns
    -------
    object
        The native limiter's return value; the supplied material buffer is updated in place [kg/s].
    """
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


@pytest.mark.parametrize("inventory", [-2.e-12, -1.e-8, 0.])
@pytest.mark.parametrize("production", [0., 1.e-13])
def test_nonconsuming_trial_state_is_not_rejected_or_clipped(inventory, production):
    """Allow stationary/recovering trial states without certifying their accuracy.

    Parameters
    ----------
    inventory : float
        Trial species inventory [kg]. Negative cases straddle the existing
        depletion threshold and remain far below ordinary solver tolerances.
    production : float
        Nonnegative species rate [kg/s]. The positive case does not restore
        the negative inventory within the one-second limiting horizon.
    """
    unit, state, buffer = system([[inventory, 1.]])  # [kg]
    buffer.contributions[buffer.INTRAPHASE] = [production, -production]  # [kg/s]
    original_state = {key: value.copy() for key, value in state.items()}  # [kg]
    original_rates = buffer.contributions.copy()  # [kg/s]
    assert not limit(unit, state, buffer)
    for key in state:
        np.testing.assert_array_equal(state[key], original_state[key])
    np.testing.assert_array_equal(buffer.contributions, original_rates)


@pytest.mark.parametrize("inventory", [-2.e-12, 0.])
def test_empty_trial_inventory_still_rejects_consumption(inventory):
    """Reject actual depletion even when the solver trial inventory is negative.

    Parameters
    ----------
    inventory : float
        Empty or slightly negative trial inventory [kg]. The imposed
        consumption is much larger than the existing depletion tolerance.
    """
    unit, state, buffer = system([[inventory, 1.]])  # [kg]
    buffer.contributions[buffer.INTRAPHASE] = [-1.e-4, 1.e-4]  # [kg/s]
    original_state = {key: value.copy() for key, value in state.items()}  # [kg]
    original_rates = buffer.contributions.copy()  # [kg/s]
    with pytest.raises(RuntimeError, match="Inventory depletion"):
        limit(unit, state, buffer)
    for key in state:
        np.testing.assert_array_equal(state[key], original_state[key])
    np.testing.assert_array_equal(buffer.contributions, original_rates)


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


def test_negative_spectator_does_not_block_conservative_transfer_limiting():
    """Preserve a paired transfer beside a negative, nonconsuming trial species."""
    unit, state, buffer = system([[.5, -2.e-12], [0., 0.]])  # [kg]
    buffer.contributions[buffer.CROSSPHASE] = [-2., 0., 2., 0.]  # [kg/s]
    original_state = {key: value.copy() for key, value in state.items()}  # [kg]
    assert limit(unit, state, buffer)
    # [kg/s] The half-kilogram donor supports 0.5 kg/s for one second.
    np.testing.assert_allclose(buffer.contributions[buffer.CROSSPHASE],
                               [-.5, 0., .5, 0.])
    for key in state:
        np.testing.assert_array_equal(state[key], original_state[key])


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
    for offset, rate_constant in ((0, 1.), (2, 2.)):  # [1/s], gives 100 and 200 W
        mechanism, aux, _ = reaction_heat(rate_constant)
        buffer.aux[buffer.CROSSPHASE].append({
            **aux,
            "connection": PhaseConnection(PhaseRef("liquid", offset), PhaseRef("liquid", offset+1), mechanism.reaction.kinetics, mechanism=mechanism),
            "material_writes": [(slice(offset, offset + 1), np.array([-2.])),
                                (slice(offset + 1, offset + 2), np.array([2.]))],
        })
    limit(unit, state, buffer)
    np.testing.assert_allclose(buffer.contributions[buffer.CROSSPHASE], [-.5, .5, -2., 2.])
    energy = {"crossphase": 0.}
    unit.add_crossphase_energy_terms(energy, buffer.aux[buffer.CROSSPHASE], 0., state)
    assert energy["crossphase"] == pytest.approx(225.)


@pytest.mark.parametrize("available", [0., .25])
@pytest.mark.parametrize("temperature_difference", [0., 10.])
def test_outlet_species_volume_composition_and_energy_stay_consistent(available, temperature_difference):
    """Check limited outflow and its relative sensible-heat contribution.

    Parameters
    ----------
    available : float
        Inventory of each species [kg] over the one-second limiting horizon.
    temperature_difference : float
        Outlet minus vessel temperature [K]. Equal temperatures remove no
        sensible heat from the temperature balance, even with nonzero flow.
    """
    unit, state, buffer = system([[available] * 4])
    thermo = Path(__file__).parents[1] / "tests/fixtures/native_four_species.json"
    vessel = LiquidPhase(thermo, mass=1., temp=310., mass_frac=np.full(4, .25), verbose=False)  # [kg], [K], [-]
    stream = LiquidStream(thermo, mass_flow=4., temp=310. + temperature_difference,
                          mass_frac=np.full(4, .25), verbose=False)  # [kg/s], [K], [-]
    heat_capacity = 2000.  # [J/(kg K)] Constant, identical fixture value for each species.
    for phase in (vessel, stream):
        phase.cp_liq = (heat_capacity * np.asarray(phase.mw) / 1000.)[:, None]  # [J/(mol K)]
    transfer = ResolvedPhaseTransfer(None, SimpleNamespace(sink_phaseref=PhaseRef("liquid", 0)), vessel, stream, stream.vol_flow,
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
    expected_heat = -stream.mass_flow * heat_capacity * temperature_difference  # [W]
    assert energy["outlet"] == pytest.approx(expected_heat)


def test_transfer_heat_is_scaled_with_material():
    """Verify paired heat scaling using real constant heat-capacity properties."""
    unit, state, buffer = system([[1.], [0.]])
    buffer.contributions[buffer.CROSSPHASE] = [-2., 2.]
    limit(unit, state, buffer)
    unit._timers = {}
    mechanism, aux, phase = reaction_heat(1.)
    buffer.aux[buffer.CROSSPHASE][0].update(aux)
    buffer.aux[buffer.CROSSPHASE][0]["connection"] = PhaseConnection(
        PhaseRef("liquid", 0), PhaseRef("liquid", 1), mechanism.reaction.kinetics, mechanism=mechanism)
    unit.Phases = phase
    result = unit.energy_balances(0., state, buffer)
    assert result[StateKey("global_temp")] == pytest.approx(5.)
    buffer.reset()
    assert buffer.aux[buffer.CROSSPHASE] == []


def test_native_reactive_transfer_conserves_mass_and_reaction_heat():
    """Qualify the cross-phase fixture with the pre-integration SciPy backend.

    The synthetic first-order coefficient is 0.1 /s, well away from depletion
    over the 0.1 s run. Initial glucose is 0.1 mol; its 1000 J/mol heat release
    warms a 2 kg holdup with constant Cp=10 J/(kg K). The independent exact
    conversion is 1-exp(-k*t), including both phase inventories and temperature.
    """
    rate_constant = .1  # [1/s], resolves this fixture away from inventory limiting
    duration = .1  # [s], one hundredth of the reaction timescale
    mechanism, _, source = reaction_heat(rate_constant)
    receiver = LiquidPhase(Path(__file__).parent / 'fixtures/native_four_species.json',
                           mass=1., mass_frac=[0., 0., 0., 1.], verbose=False)
    heat_capacity = 10.  # [J/(kg K)], same constant synthetic property in both phases
    receiver.cp_liq = (heat_capacity * np.asarray(receiver.mw) / 1000.)[:, None]  # [J/(mol K)]
    comparison_rtol = 1e-7  # [-], the established NumPy comparison default used by these tests
    relative_tolerance = 1e-10  # [-], three orders tighter than the comparison budget
    absolute_tolerance = 1e-12  # [kg, K], resolves the roughly 1.8e-4 kg transfer and 0.05 K rise
    backend = ScipyBackend({'rtol': relative_tolerance, 'atol': absolute_tolerance})
    unit = MultiPhaseVessel(isothermal=False, adiabatic=True, integrator=backend)
    unit.Phases = [source, receiver]
    unit.phase_connections = [PhaseConnection(PhaseRef('liquid', 0), PhaseRef('liquid', 1),
                                              mechanism.reaction.kinetics, mechanism=mechanism)]
    unit.solve_unit(time_grid=[0., duration], verbose=False)
    initial_glucose = .1 * 180.156 / 1000.  # [kg], 0.1 mol and fixture glucose MW [g/mol]
    converted = -np.expm1(-rate_constant * duration)  # [-], independent first-order solution
    np.testing.assert_allclose(unit.result.mass_j_liquid0[-1, 0],
                               (1.-converted) * initial_glucose, rtol=comparison_rtol)
    np.testing.assert_allclose(unit.result.mass_j_liquid1[-1, 1],
                               converted * initial_glucose, rtol=comparison_rtol)
    total_mass = unit.result.mass_j_liquid0.sum(axis=1) + unit.result.mass_j_liquid1.sum(axis=1)  # [kg]
    np.testing.assert_allclose(total_mass, 2.)
    expected_rise = .1 * 1000. * converted / (2. * heat_capacity)  # [K], released reaction energy / thermal holdup
    np.testing.assert_allclose(unit.result.global_temp[-1] - unit.result.global_temp[0],
                               expected_rise, rtol=comparison_rtol)
