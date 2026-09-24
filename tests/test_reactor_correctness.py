"""Core reactor contracts for #34, #54, #169 and #245.

Real thermodynamics and elementary A + B -> C kinetics use the shipped PFR
property data. Solver-boundary checks run native CVode and inspect initial
result rows, physical derivatives, and complete steady-to-dynamic reuse.

Refs:
https://github.com/PharmaPy-org/PharmaPy/issues/34
https://github.com/PharmaPy-org/PharmaPy/issues/54
https://github.com/PharmaPy-org/PharmaPy/issues/169
https://github.com/PharmaPy-org/PharmaPy/issues/245
"""

from pathlib import Path
import json

import numpy as np
import pytest

from PharmaPy import Reactors
from PharmaPy.Commons import unpack_states
from PharmaPy.Kinetics import RxnKinetics
from PharmaPy.Phases import LiquidPhase
from PharmaPy.ProcessControl import DynamicInput
from PharmaPy.Streams import LiquidStream
from PharmaPy.Utilities import CoolingWater


THERMO_PATH = str(Path(__file__).parent / "integration/data/pfr_test_pure_comp.json")
# Synthetic, asymmetric A + B -> C mixture with solvent inert. These values
# exercise ordering and a nonzero reverse rate; they are not a calibrated case.
CONCENTRATIONS = np.array([0.15, 0.10, 0.02, 5.0])  # [mol/L]
TEMPERATURE = 320.0  # [K], off the heat reference to exercise sensible heat
HEAT_REFERENCE_TEMPERATURE = 298.15  # [K], standard reference for the fixture
REACTION_HEAT = -1.0e4  # [J/mol of reaction as written], synthetic exotherm
RATE_CONSTANT = 1.0e-3  # [L/mol/s], slow elementary bimolecular reaction
EQUILIBRIUM_CONSTANT = 2.0  # [L**2/mol**2], synthetic 2 A + B -> C equilibrium
EQUILIBRIUM_RATE_CONSTANT = 1.0e-3  # [L**2/mol**2/s], synthetic third-order rate
GAS_CONSTANT = 8.314  # [J/mol/K], precision used by the model's van 't Hoff law
UTILITY_RAMP = 1.0  # [K/s], synthetic ramp distinguishes current from initial input
REACTOR_VOLUME = 0.002  # [m**3], shipped PFR case volume
RESIDENCE_TIME = 1800.0  # [s], shipped PFR case residence time
TUBE_DIAMETER = 0.0254  # [m], shipped PFR case diameter
UTILITY_TEMPERATURE = 300.0  # [K], shipped PFR utility inlet
UTILITY_MASS_FLOW = 0.1  # [kg/s], shipped PFR utility flow
NUM_CELLS = 2  # [-], unequal to both the three reacting and four total species
RUNTIME = 1.0  # [s], short initialization/reuse check, not convergence to steady state
# Algebra comparisons allow roundoff; solver comparisons use CVode's default
# relative tolerance, since both runs have identical initial-value problems.
ALGEBRA_RTOL = 1.0e-12  # [-]
SOLVER_RTOL = 1.0e-6  # [-]


def _configured_reactor(reactor, equilibrium=False):
    """Attach real collaborators for the synthetic reaction case.

    Parameters
    ----------
    reactor : _BaseReactor
        Reactor instance to configure.
    equilibrium : bool, optional
        Use synthetic 2 A + B -> C equilibrium kinetics instead of the
        irreversible A + B -> C case.

    Returns
    -------
    _BaseReactor
        Configured reactor with concentrations [mol/L], volume [m**3], and
        temperature [K] specified by the module fixture constants.
    """
    reactor.Phases = LiquidPhase(
        THERMO_PATH, temp=TEMPERATURE, vol=REACTOR_VOLUME,
        mole_conc=CONCENTRATIONS)
    reactor.Kinetics = RxnKinetics(
        THERMO_PATH, k_params=[EQUILIBRIUM_RATE_CONSTANT if equilibrium else RATE_CONSTANT],
        ea_params=[0.0],
        rxn_list=["2 A + B --> C" if equilibrium else "A + B --> C"], delta_hrxn=REACTION_HEAT,
        temp_ref=HEAT_REFERENCE_TEMPERATURE,
        tref_hrxn=HEAT_REFERENCE_TEMPERATURE,
        keq_params=[EQUILIBRIUM_CONSTANT] if equilibrium else None)
    # Zero activation energy isolates equilibrium's temperature dependence.
    reactor.Utility = CoolingWater(
        temp_in=UTILITY_TEMPERATURE, mass_flow=UTILITY_MASS_FLOW)
    if not isinstance(reactor, Reactors.BatchReactor):
        reactor.Inlet = LiquidStream(
            THERMO_PATH, temp=TEMPERATURE, mole_conc=CONCENTRATIONS,
            vol_flow=REACTOR_VOLUME / RESIDENCE_TIME)
    return reactor


@pytest.mark.unit
def test_cstr_equilibrium_rate_through_unit_model():
    """#34: real heat and keyword handoff determine the species-rate values."""
    reactor = _configured_reactor(Reactors.CSTR(isothermal=True), equilibrium=True)
    reactor.set_names()
    # Integrate the shipped Cp polynomials independently with numpy.polynomial.
    # For 2 A + B -> C, delta Cp = Cp_C - 2 Cp_A - Cp_B, on the raw basis.
    with open(THERMO_PATH) as stream:
        properties = json.load(stream)
    delta_cp = (np.array(properties["C"]["cp_liq"])
                - 2 * np.array(properties["A"]["cp_liq"])
                - np.array(properties["B"]["cp_liq"]))  # coefficients [J/mol/K**(i+1)]
    integrated_cp = np.polynomial.Polynomial(delta_cp).integ()
    heat = (REACTION_HEAT + integrated_cp(TEMPERATURE)
            - integrated_cp(HEAT_REFERENCE_TEMPERATURE))  # [J/mol of raw reaction]
    # The model's van 't Hoff approximation uses the evaluation-temperature
    # raw heat in exp[-delta H(T)/R * (1/T - 1/Tref)]. Zero Ea makes k(T) = kref.
    keq = EQUILIBRIUM_CONSTANT * np.exp(
        -heat / GAS_CONSTANT
        * (1 / TEMPERATURE - 1 / HEAT_REFERENCE_TEMPERATURE))  # [L**2/mol**2]
    rate = EQUILIBRIUM_RATE_CONSTANT * (
        CONCENTRATIONS[0]**2 * CONCENTRATIONS[1]
        - CONCENTRATIONS[2] / keq)  # [mol/L/s], normalized kinetic extent
    normalized_stoich = np.array([-1.0, -0.5, 0.5, 0.0])  # [-], raw row / 2
    # Change inlet A so the public balance also exercises its flow contribution.
    inlet_conc = CONCENTRATIONS.copy()  # [mol/L]
    inlet_conc[0] *= 2  # Synthetic feed has twice the holdup A concentration.
    reactor.Inlet = LiquidStream(
        THERMO_PATH, temp=TEMPERATURE, mole_conc=inlet_conc,
        vol_flow=REACTOR_VOLUME / RESIDENCE_TIME)
    expected = (rate * normalized_stoich + (inlet_conc - CONCENTRATIONS)
                / RESIDENCE_TIME)  # [mol/L/s]
    actual = reactor.unit_model(0.0, CONCENTRATIONS.copy())  # [mol/L/s]
    np.testing.assert_allclose(actual, expected, rtol=ALGEBRA_RTOL, atol=0)
    assert actual[-1] == 0  # Inert solvent has no reaction or flow source.


@pytest.mark.unit
@pytest.mark.parametrize("reactor_cls,controlled", [
    pytest.param(cls, controlled, marks=([] if cls is Reactors.PlugFlowReactor and controlled
                                        else [pytest.mark.assimulo, pytest.mark.integration]))
    for cls in [Reactors.BatchReactor, Reactors.CSTR, Reactors.SemibatchReactor,
                Reactors.PlugFlowReactor]
    for controlled in [False, True]
])
@pytest.mark.parametrize("ht_mode", ["bath", "jacket"])
def test_tank_initial_state_metadata(reactor_cls, ht_mode, controlled):
    """Check named state packing against native solver initial rows and heat.

    Parameters
    ----------
    reactor_cls : type
        Batch, continuous, semibatch, or plug-flow reactor.
    ht_mode : str
        Bath or jacket heat-transfer mode.
    controlled : bool
        Prescribe reactor temperature; unsupported PFR controls stay in core.
    """
    controls = {"temp": lambda time: TEMPERATURE} if controlled else None
    kwargs = {}
    if reactor_cls is Reactors.SemibatchReactor:
        kwargs["vol_tank"] = REACTOR_VOLUME  # [m**3]
    elif reactor_cls is Reactors.PlugFlowReactor:
        kwargs = {"diam_in": TUBE_DIAMETER, "num_discr": NUM_CELLS}
    if reactor_cls is Reactors.PlugFlowReactor and controlled:
        with pytest.raises(NotImplementedError, match='PlugFlowReactor controls'):
            reactor_cls(isothermal=False, ht_mode=ht_mode, controls=controls, **kwargs)
        return
    pytest.importorskip('assimulo')
    reactor = _configured_reactor(reactor_cls(
        isothermal=False, ht_mode=ht_mode, controls=controls, **kwargs))
    _, states = reactor.solve_unit(time_grid=[0.0, RUNTIME], verbose=False)
    initial = states[0]  # [mol/L], optional volume [m**3] and temperatures [K]
    is_pfr = reactor_cls is Reactors.PlugFlowReactor
    cell_initial = initial.reshape(NUM_CELLS, -1)[0] if is_pfr else initial
    assert sum(reactor.dim_states) == len(cell_initial)
    assert set(reactor.states_di) == set(reactor.states_uo)
    assert reactor.name_states == reactor.states_uo
    unpacked = unpack_states(cell_initial, reactor.dim_states, reactor.name_states)
    assert list(unpacked) == reactor.states_uo
    assert [np.size(value) for value in unpacked.values()] == reactor.dim_states
    assert [reactor.states_di[name]["dim"] for name in unpacked] == reactor.dim_states
    if reactor_cls is Reactors.SemibatchReactor:
        assert unpacked["vol"] == REACTOR_VOLUME
    expected_conc = (CONCENTRATIONS[:3] if reactor_cls is Reactors.BatchReactor
                     else CONCENTRATIONS)  # [mol/L]
    np.testing.assert_array_equal(unpacked["mole_conc"], expected_conc)
    # Uncontrolled PFR retains its integrated temperature.
    has_temperature = not controlled or is_pfr
    assert ("temp" in unpacked) == has_temperature
    assert ("temp_ht" in unpacked) == (ht_mode == "jacket" and has_temperature and not is_pfr)
    if has_temperature:
        assert unpacked["temp"] == TEMPERATURE
    if "temp_ht" in unpacked:
        assert unpacked["temp_ht"] == UTILITY_TEMPERATURE
    np.testing.assert_array_equal(
        np.concatenate([np.atleast_1d(value) for value in unpacked.values()]),
        cell_initial)
    if controlled:
        return  # Full controlled tank balances are covered in the #232 regressions.
    derivative = reactor.unit_model(0.0, initial)  # concentrations [mol/L/s], [K/s], [m**3/s]
    assert derivative.shape == initial.shape
    assert np.all(np.isfinite(derivative))
    if is_pfr:
        return  # PFR has its own tube area model and no jacket temperature state.
    # A real dynamic utility distinguishes current time from a cached inlet.
    utility_input = DynamicInput()
    utility_input.add_variable("temp_in", lambda time: UTILITY_TEMPERATURE + UTILITY_RAMP * time)
    reactor.Utility.DynamicInlet = utility_input
    derivative = reactor.unit_model(RUNTIME, initial)  # [state units/s]
    utility_temp = (UTILITY_TEMPERATURE + UTILITY_RAMP * RUNTIME
                    if ht_mode == "bath" else UTILITY_TEMPERATURE)  # [K]
    # Cylinder lateral wetted area = perimeter * (liquid volume / base area).
    area = (np.pi * reactor.diam * REACTOR_VOLUME / reactor.area_base
            + reactor.area_base)  # [m**2]
    expected_heat = reactor.u_ht * area * (TEMPERATURE - utility_temp)  # [W]
    with open(THERMO_PATH) as stream:
        properties = json.load(stream)
    cp = np.array([np.polynomial.Polynomial(item['cp_liq'])(TEMPERATURE)
                   for item in properties.values()])  # [J/mol/K]
    enthalpy = np.array([
        np.polynomial.Polynomial(item['cp_liq']).integ()(TEMPERATURE)
        - np.polynomial.Polynomial(item['cp_liq']).integ()(HEAT_REFERENCE_TEMPERATURE)
        for item in properties.values()])  # [J/mol]
    reaction_heat = REACTION_HEAT + enthalpy[2] - enthalpy[0] - enthalpy[1]  # [J/mol]
    source = -reaction_heat * RATE_CONSTANT * CONCENTRATIONS[0] * CONCENTRATIONS[1] * REACTOR_VOLUME * 1000
    # [W], exact m**3-to-L conversion; inlet and tank temperatures match
    capacitance = REACTOR_VOLUME * 1000 * np.dot(CONCENTRATIONS, cp)  # [J/K]
    temperature_rate = unpack_states(derivative, reactor.dim_states, reactor.name_states)['temp']
    # [K/s], actual full RHS handoff, with zero inlet sensible-heat contribution
    removed_heat = source - capacitance * temperature_rate  # [W]
    np.testing.assert_allclose(removed_heat, expected_heat, rtol=ALGEBRA_RTOL, atol=0)


REACTOR_CONSTRUCTORS = [
    (Reactors.BatchReactor, {}), (Reactors.CSTR, {}),
    (Reactors.SemibatchReactor, {"vol_tank": REACTOR_VOLUME}),
    (Reactors.PlugFlowReactor, {"diam_in": TUBE_DIAMETER, "num_discr": NUM_CELLS}),
]


@pytest.mark.unit
@pytest.mark.parametrize("reactor_cls, kwargs", REACTOR_CONSTRUCTORS)
@pytest.mark.parametrize("ht_mode", ["jacket", "bath", "coil", "jackt"])
def test_heat_transfer_mode_constructor(reactor_cls, kwargs, ht_mode):
    """#169: accept supported modes and reject unsupported modes immediately."""
    if ht_mode == "coil":
        with pytest.raises(NotImplementedError, match="coil.*#168"):
            reactor_cls(ht_mode=ht_mode, **kwargs)
    elif ht_mode == "jackt":
        with pytest.raises(ValueError, match="jackt.*jacket.*bath"):
            reactor_cls(ht_mode=ht_mode, **kwargs)
    else:
        reactor = reactor_cls(ht_mode=ht_mode, **kwargs)
        assert reactor.ht_mode == ht_mode


@pytest.mark.unit
def test_coil_defensive_heat_transfer_check():
    """Keep the call-time guard for instances mutated after construction."""
    reactor = Reactors.BatchReactor(isothermal=False)
    reactor.ht_mode = "coil"
    with pytest.raises(NotImplementedError, match="coil.*not supported"):
        reactor.heat_transfer(TEMPERATURE, UTILITY_TEMPERATURE, REACTOR_VOLUME)


@pytest.mark.assimulo
@pytest.mark.integration
@pytest.mark.parametrize("isothermal", [False, True])
@pytest.mark.parametrize("adiabatic", [False, True])
def test_pfr_steady_bookkeeping_preserves_dynamic_initial_state(isothermal, adiabatic):
    """Restore temporary steady modes after a real solver argument rejection.

    Parameters
    ----------
    isothermal : bool
        Persistent dynamic temperature mode.
    adiabatic : bool
        Temporary steady heat-transfer mode to restore after failure.
    """
    pytest.importorskip('assimulo')
    reused = _configured_reactor(Reactors.PlugFlowReactor(
        diam_in=TUBE_DIAMETER, num_discr=NUM_CELLS, isothermal=isothermal,
        adiabatic=not adiabatic))
    fresh = _configured_reactor(Reactors.PlugFlowReactor(
        diam_in=TUBE_DIAMETER, num_discr=NUM_CELLS, isothermal=isothermal,
        adiabatic=not adiabatic))
    phase = reused.Liquid_1
    original_states = reused.states_uo
    for _ in range(2):
        # CVode rejects a nonnumeric endpoint after solve_steady installs its
        # temporary modes and metadata, exercising its actual finally block.
        with pytest.raises(TypeError, match='must be real number, not str'):
            reused.solve_steady('invalid volume', adiabatic=adiabatic)
        assert reused.states_uo is original_states
        assert reused.states_uo == fresh.states_uo
        assert reused.isothermal == fresh.isothermal
        assert reused.adiabatic == fresh.adiabatic
    assert reused.num_species == len(CONCENTRATIONS)
    assert reused.num_species_steady == len(reused.Kinetics.partic_species)
    np.testing.assert_array_equal(reused.c_inert, CONCENTRATIONS[3:])
    expected_material = RATE_CONSTANT * CONCENTRATIONS[0] * CONCENTRATIONS[1] \
        / reused.Inlet.vol_flow * np.array([-1.0, -1.0, 1.0])  # [mol/L/m**3]
    np.testing.assert_allclose(reused.material_steady(CONCENTRATIONS[:3], TEMPERATURE),
                               expected_material, rtol=ALGEBRA_RTOL, atol=0)
    grid = np.array([0.0, RUNTIME])  # [s], common dynamic reporting points
    _, reused_states = reused.solve_unit(time_grid=grid, verbose=False)
    _, fresh_states = fresh.solve_unit(time_grid=grid, verbose=False)
    np.testing.assert_array_equal(reused_states[0], fresh_states[0])
    cells = reused_states[0].reshape(NUM_CELLS, -1)  # [mol/L], optional [K]
    np.testing.assert_array_equal(cells[:, :4], np.tile(CONCENTRATIONS, (NUM_CELLS, 1)))
    assert reused.Liquid_1 is phase
    assert reused.name_species == fresh.name_species
    assert reused.states_in_dict == fresh.states_in_dict
    assert reused.states_out_dict == fresh.states_out_dict
    assert reused.states_di == fresh.states_di
    assert reused.name_states == fresh.name_states


@pytest.mark.assimulo
@pytest.mark.integration
@pytest.mark.parametrize("isothermal", [False, True])
@pytest.mark.parametrize("adiabatic", [False, True])
def test_pfr_complete_steady_to_dynamic_reuse(isothermal, adiabatic):
    """Check steady initial values, result slicing, and subsequent dynamics.

    Parameters
    ----------
    isothermal : bool
        Persistent dynamic temperature mode.
    adiabatic : bool
        Temporary steady heat-transfer mode.
    """
    pytest.importorskip("assimulo")
    reused = _configured_reactor(Reactors.PlugFlowReactor(
        diam_in=TUBE_DIAMETER, num_discr=NUM_CELLS, isothermal=isothermal,
        adiabatic=not adiabatic))
    fresh = _configured_reactor(Reactors.PlugFlowReactor(
        diam_in=TUBE_DIAMETER, num_discr=NUM_CELLS, isothermal=isothermal,
        adiabatic=not adiabatic))
    phase = reused.Liquid_1
    original_states = reused.states_uo
    # Repeat the real solve to check restoration after successful integration.
    for _ in range(2):
        _, steady = reused.solve_steady(REACTOR_VOLUME, adiabatic=adiabatic)  # [mol/L], optional [K]
        assert reused.states_uo is original_states
        assert reused.states_uo == fresh.states_uo
        assert reused.isothermal == fresh.isothermal
        assert reused.adiabatic == fresh.adiabatic
        has_temperature = adiabatic or not isothermal
        assert steady.shape[1] == 3 + has_temperature
        np.testing.assert_array_equal(steady[0, :3], CONCENTRATIONS[:3])
        if has_temperature:
            assert steady[0, -1] == pytest.approx(TEMPERATURE, rel=ALGEBRA_RTOL)
    assert reused.num_species == len(CONCENTRATIONS)
    np.testing.assert_array_equal(reused.concProfSteady, steady[:, :3])
    np.testing.assert_allclose(reused.tempProfSteady,
                               TEMPERATURE if isothermal and not adiabatic else steady[:, -1],
                               rtol=ALGEBRA_RTOL, atol=0)
    assert reused.concProfSteady[-1, 0] < CONCENTRATIONS[0]
    assert reused.Liquid_1 is phase
    np.testing.assert_array_equal(phase.mole_conc, CONCENTRATIONS)
    grid = np.array([0.0, RUNTIME])  # [s], matching requested output times
    reused_time, reused_states = reused.solve_unit(time_grid=grid, verbose=False)
    fresh_time, fresh_states = fresh.solve_unit(time_grid=grid, verbose=False)
    np.testing.assert_array_equal(reused_time, fresh_time)
    np.testing.assert_array_equal(reused_states[0], fresh_states[0])
    np.testing.assert_allclose(reused_states, fresh_states, rtol=SOLVER_RTOL, atol=0)
    assert reused.Outlet.name_species == fresh.Outlet.name_species
    np.testing.assert_allclose(reused.Outlet.mole_conc, fresh.Outlet.mole_conc,
                               rtol=SOLVER_RTOL, atol=0)
    assert reused.Outlet.mole_conc[-1] == pytest.approx(CONCENTRATIONS[-1], rel=SOLVER_RTOL)


@pytest.mark.assimulo
@pytest.mark.integration
@pytest.mark.parametrize("reactor_cls", [Reactors.BatchReactor, Reactors.CSTR])
def test_bath_solve_and_heat_profile(reactor_cls):
    """Bath RHS and result retrieval use the time-dependent utility inlet."""
    pytest.importorskip("assimulo")
    reactor = _configured_reactor(reactor_cls(isothermal=False, ht_mode="bath"))
    utility_input = DynamicInput()
    utility_input.add_variable("temp_in", lambda time: UTILITY_TEMPERATURE + UTILITY_RAMP * time)
    reactor.Utility.DynamicInlet = utility_input
    grid = np.array([0.0, RUNTIME])  # [s], sample both ends of the utility ramp
    time, states = reactor.solve_unit(time_grid=grid, verbose=False)
    assert "temp_ht" not in reactor.name_states
    assert len(states) == len(time) == len(grid)
    utility_temp = UTILITY_TEMPERATURE + UTILITY_RAMP * np.asarray(time)  # [K]
    area = (np.pi * reactor.diam * REACTOR_VOLUME / reactor.area_base
            + reactor.area_base)  # [m**2], lateral wetted cylinder plus base
    expected_heat = -reactor.u_ht * area * (states[:, -1] - utility_temp)  # [W]
    np.testing.assert_allclose(reactor.result.q_ht, expected_heat,
                               rtol=ALGEBRA_RTOL, atol=0)
    np.testing.assert_array_equal(reactor.result.temp, states[:, -1])
    assert reactor.Outlet.temp == states[-1, -1]
