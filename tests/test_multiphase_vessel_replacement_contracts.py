"""Regression tests for legacy unit-operation and phase contracts on the MultiPhaseVessel stack.

Verifies that the refactored ``MultiPhaseVessel``, ``Reactors_Refactored``,
``Crystallizers_Refactored``, ``Mechanisms``, ``Phases_Refactored``,
``Streams_Refactored``, ``MixedPhases_Refactored``, and ``Connections``
components satisfy the physical, numerical, and API contracts established by
the legacy reactor, crystallizer, phase, stream, utility, and flowsheet test
suites and open issue audit:

- Heat-transfer mode validation (#169), a positive vessel-side ``h_conv``
  before a utility is assigned, and utility heat-transfer rate / dynamic
  ``CoolingWater`` controls (#279, #409).
- Reaction enthalpy stoichiometry normalization (#22) and reversible reaction
  energy balance shape consistency (#270).
- Transient CSTR sensible enthalpy balance with differing feed/holdup
  compositions (#272), inert solvent mass conservation (#440), outlet
  property-database ownership (#442), and caller phase-sequence immutability
  (#450).
- Semibatch crystallizer slurry-feed energy balance (#265), idempotent phase
  reconfiguration and symbolic event state lookup (#274), static ``x_grid``
  preservation (#281), scale-independent moment/FVM balances (#286),
  arbitrary moment orders and legacy stream moment conversion (#288, #330),
  concentration-buffer immutability (#312), mass-only solid volume (#419), and
  subclass classification by ``phase_family`` (#427).
- Flowsheet connection operating-mode validation and routing (#436, #438,
  #445, #447) across batch, semibatch, and continuous refactored vessels.
"""

import json
import os

import numpy as np
import pytest

from PharmaPy.Connections import Connection
from PharmaPy.Containers import DynamicCollector
from PharmaPy.Crystallizers_Refactored import (
    BatchCrystallizer,
    ContinuousCrystallizer,
    SemiBatchCrystallizer,
)
from PharmaPy.DataClasses import OperatingKey, PhaseRef, StateEvent, StateKey
from PharmaPy.Errors import PharmaPyValueError
from PharmaPy.IntegratorBackends import ScipyBackend
from PharmaPy.Kinetics import CrystKinetics, RxnKinetics
from PharmaPy.Mechanisms import MomentsPopulationBalance, OneDFVMMechanism
from PharmaPy.MixedPhases_Refactored import MixedPhase, MixedStream
from PharmaPy.Phases import SolidPhase as LegacySolidPhase
from PharmaPy.Phases_Refactored import (
    LiquidPhase,
    SolidPhase,
    VaporPhase,
    classify_phases,
    getPropsPhaseMix,
)
from PharmaPy.ProcessControl import DynamicInput
from PharmaPy.ProcessControl_Refactored import Controller
from PharmaPy.Reactors_Refactored import (
    BatchReactor,
    ContinuousReactor,
    SemiBatchReactor,
)
from PharmaPy.SimExec import SimulationExec
from PharmaPy.Streams import SolidStream as LegacySolidStream
from PharmaPy.Streams_Refactored import LiquidStream, SolidStream, VaporStream
from PharmaPy.Utilities import CoolingWater

pytestmark = pytest.mark.unit

DATA_PATH = os.path.join(
    os.path.dirname(__file__), "Flowsheet", "data", "compound_database.json"
)

# 5-species composition vectors (A, B, C, D, solvent); each sums to 1.0 [-].
LIQUID_MASS_FRAC = np.array([0.20, 0.10, 0.05, 0.0, 0.65])  # [-]
INLET_MASS_FRAC = np.array([0.05, 0.30, 0.15, 0.0, 0.50])  # [-]
SOLID_C_MASS_FRAC = np.array([0.0, 0.0, 1.0, 0.0, 0.0])  # [-]
VAPOR_MOLE_FRAC = np.array([0.25, 0.25, 0.0, 0.0, 0.50])  # [-]

# Representative physical state values and vessel geometry.
INITIAL_TEMP = 310.0  # [K]
UTILITY_TEMP_IN = 290.0  # [K]
UTILITY_TEMP_STEP = 285.0  # [K]
CUSTOM_UTILITY_CP = 1850.0  # [J/kg/K], synthetic non-water heat-transfer fluid
CUSTOM_UTILITY_RHO = 860.0  # [kg/m**3], synthetic non-water fluid density
UTILITY_MASS_FLOW = 0.25  # [kg/s]
UTILITY_MASS_FLOW_STEP = 0.50  # [kg/s]
VESSEL_H_CONV = 800.0  # [W/m**2/K]
UTILITY_H_CONV = 1200.0  # [W/m**2/K]
# Series combination of the two film coefficients, 1 / (1/800 + 1/1200).
SERIES_H_CONV = 480.0  # [W/m**2/K]
VESSEL_DIAM = 0.50  # [m]
VESSEL_AREA_BASE = 0.05  # [m**2]
LIQUID_MASS = 5.0  # [kg]
SOLID_MASS = 0.8  # [kg]
INLET_MASS_FLOW = 0.05  # [kg/s]
INLET_VOL_FLOW = 5.0e-5  # [m**3/s]
RUNTIME = 20.0  # [s]
SWITCH_TIME = 10.0  # [s]

# Strict relative tolerance for algebraic identities; CONSERVATION_RTOL
# accounts for multi-step floating-point round-off in integrated balances;
# EVENT_RTOL accounts for SciPy solve_ivp event root-finding interpolation
# error at default LSODA tolerances.
ALGEBRA_RTOL = 1e-12  # [-]
CONSERVATION_RTOL = 1e-9  # [-]
ODE_RTOL = 1e-6  # [-]
ODE_ATOL = 1e-9  # [-]
EVENT_RTOL = 1e-4  # [-]
LITERS_PER_M3 = 1000.0  # [L/m**3], exact SI volume conversion factor
MICRON_TO_M = 1e-6  # [m/um], exact SI length conversion factor


def _scipy_lsoda():
    """Return a deterministic SciPy LSODA integrator.

    Returns
    -------
    ScipyBackend
        Configured SciPy LSODA integrator backend.
    """
    return ScipyBackend(method="LSODA")


@pytest.mark.parametrize(
    "vessel_cls",
    [
        BatchReactor,
        SemiBatchReactor,
        ContinuousReactor,
        BatchCrystallizer,
        SemiBatchCrystallizer,
        ContinuousCrystallizer,
    ],
)
@pytest.mark.parametrize("ht_mode", ["jacket", "bath", "coil", "jackt"])
def test_heat_transfer_mode_validation_in_constructor_and_rate(vessel_cls, ht_mode):
    """#169: accept 'jacket' and 'bath', reject 'coil' with NotImplementedError and unknown modes with ValueError."""
    if ht_mode == "coil":
        with pytest.raises(NotImplementedError, match="coil.*#168"):
            vessel_cls(ht_mode=ht_mode, integrator=_scipy_lsoda())
        return
    if ht_mode == "jackt":
        with pytest.raises(ValueError, match="jackt.*jacket.*bath"):
            vessel_cls(ht_mode=ht_mode, integrator=_scipy_lsoda())
        return

    vessel = vessel_cls(
        ht_mode=ht_mode,
        h_conv=VESSEL_H_CONV,
        diam=VESSEL_DIAM,
        area_base=VESSEL_AREA_BASE,
        integrator=_scipy_lsoda(),
    )
    assert vessel.ht_mode == ht_mode

    liquid = LiquidPhase(
        DATA_PATH,
        temp=INITIAL_TEMP,
        mass=LIQUID_MASS,
        mass_frac=LIQUID_MASS_FRAC,
    )
    vessel.Phases = [liquid]
    vessel.Utility = CoolingWater(
        mass_flow=UTILITY_MASS_FLOW,
        temp_in=UTILITY_TEMP_IN,
        h_conv=UTILITY_H_CONV,
    )
    vessel.compile_structure()

    expected_u_ht = 1.0 / (1.0 / VESSEL_H_CONV + 1.0 / UTILITY_H_CONV)  # [W/m**2/K]
    expected_area = 4.0 * liquid.vol / VESSEL_DIAM + VESSEL_AREA_BASE  # [m**2]
    expected_q = expected_u_ht * expected_area * (INITIAL_TEMP - UTILITY_TEMP_IN)  # [W]
    assert vessel.get_heat_transfer_rate(INITIAL_TEMP, UTILITY_TEMP_IN) == pytest.approx(
        expected_q, rel=ALGEBRA_RTOL
    )

    vessel.ht_mode = "coil"
    with pytest.raises(NotImplementedError, match="coil.*not supported"):
        vessel.get_heat_transfer_rate(INITIAL_TEMP, UTILITY_TEMP_IN)

    vessel.ht_mode = "jackt"
    with pytest.raises(ValueError, match="jackt.*jacket.*bath"):
        vessel.get_heat_transfer_rate(INITIAL_TEMP, UTILITY_TEMP_IN)


@pytest.mark.parametrize(
    "vessel_cls",
    [
        BatchReactor,
        SemiBatchReactor,
        ContinuousReactor,
        BatchCrystallizer,
        SemiBatchCrystallizer,
        ContinuousCrystallizer,
    ],
)
def test_utility_assignment_requires_positive_vessel_h_conv(vessel_cls):
    """The default h_conv=0 cannot take a utility and says so by name.

    The overall coefficient is the series combination of the vessel-side and
    utility-side film coefficients, which has no value at h_conv=0, so the
    assignment is refused with PharmaPyValueError instead of a bare
    ZeroDivisionError and the vessel keeps no utility. With a positive
    coefficient the same assignment gives the series value the legacy tanks
    compute.
    """
    utility = CoolingWater(
        mass_flow=UTILITY_MASS_FLOW,
        temp_in=UTILITY_TEMP_IN,
        h_conv=UTILITY_H_CONV,
    )

    vessel = vessel_cls(integrator=_scipy_lsoda())
    with pytest.raises(PharmaPyValueError, match="h_conv"):
        vessel.Utility = utility
    assert vessel.Utility is None

    vessel = vessel_cls(h_conv=VESSEL_H_CONV, integrator=_scipy_lsoda())
    vessel.Utility = utility
    assert vessel.Utility is utility
    assert vessel.u_ht == pytest.approx(SERIES_H_CONV, rel=ALGEBRA_RTOL)


@pytest.mark.parametrize("utility_first", [True, False])
def test_utility_heat_transfer_honors_custom_cp_and_dynamic_cooling_water_controls(
    utility_first,
):
    """#279, #409: MultiPhaseVessel evaluates dynamic CoolingWater controls and uses Utility.cp."""
    reactor = BatchReactor(
        isothermal=False,
        adiabatic=False,
        ht_mode="jacket",
        h_conv=VESSEL_H_CONV,
        diam=VESSEL_DIAM,
        area_base=VESSEL_AREA_BASE,
        integrator=_scipy_lsoda(),
    )
    utility = CoolingWater(
        mass_flow=UTILITY_MASS_FLOW,
        temp_in=UTILITY_TEMP_IN,
        h_conv=UTILITY_H_CONV,
    )
    utility.cp = CUSTOM_UTILITY_CP  # [J/kg/K]
    utility.rho = CUSTOM_UTILITY_RHO  # [kg/m**3]
    utility.vol_flow = UTILITY_MASS_FLOW / CUSTOM_UTILITY_RHO  # [m**3/s]
    dyn_util = DynamicInput()
    dyn_util.add_variable(
        "temp_in",
        lambda t: UTILITY_TEMP_IN if float(t) < SWITCH_TIME else UTILITY_TEMP_STEP,
    )
    dyn_util.add_variable(
        "mass_flow",
        lambda t: UTILITY_MASS_FLOW if float(t) < SWITCH_TIME else UTILITY_MASS_FLOW_STEP,
    )
    utility.DynamicInlet = dyn_util

    liquid = LiquidPhase(
        DATA_PATH,
        temp=INITIAL_TEMP,
        mass=LIQUID_MASS,
        mass_frac=LIQUID_MASS_FRAC,
    )
    if utility_first:
        reactor.Utility = utility
        reactor.Phases = [liquid]
    else:
        reactor.Phases = [liquid]
        reactor.Utility = utility

    reactor.RxnKinetics = RxnKinetics(
        path=DATA_PATH,
        rxn_list=["A --> B"],
        k_params=0.0,  # [1/s], zero reaction isolates utility heat transfer
        ea_params=0.0,  # [J/mol]
    )

    expected_u_ht = 1.0 / (1.0 / VESSEL_H_CONV + 1.0 / UTILITY_H_CONV)  # [W/m**2/K]
    area_0 = 4.0 * liquid.vol / VESSEL_DIAM + VESSEL_AREA_BASE  # [m**2]

    # Directly verify unit_model temperature derivative dT/dt before and after SWITCH_TIME
    # against independent hand derivation -U*A*(T0 - T_in(t)) / (m_liq * Cp_liq).
    reactor.compile_structure()
    total_cp_0 = reactor.Phases.get_total_heat_capacity()  # [J/K]
    y0 = reactor.create_solver_init_states()
    dy_before = reactor.solver_state_collection.unpack(
        reactor.unit_model(1.0, y0)
    )
    dy_after = reactor.solver_state_collection.unpack(
        reactor.unit_model(15.0, y0)
    )
    expected_dt_before = (
        -expected_u_ht * area_0 * (INITIAL_TEMP - UTILITY_TEMP_IN) / total_cp_0
    )  # [K/s]
    expected_dt_after = (
        -expected_u_ht * area_0 * (INITIAL_TEMP - UTILITY_TEMP_STEP) / total_cp_0
    )  # [K/s]
    assert float(dy_before[StateKey("global_temp")]) == pytest.approx(
        expected_dt_before, rel=CONSERVATION_RTOL
    )
    assert float(dy_after[StateKey("global_temp")]) == pytest.approx(
        expected_dt_after, rel=CONSERVATION_RTOL
    )

    time_grid = np.array([0.0, 5.0, 15.0, RUNTIME])  # [s]
    reactor.solve_unit(time_grid=time_grid)

    # Verify q_ht at t=0 (before switch) and t=20 s (after switch) against independent U*A*(T - T_in(t)).
    q_0_expected = expected_u_ht * area_0 * (
        reactor.result.global_temp[0] - UTILITY_TEMP_IN
    )  # [W]
    q_end_expected = expected_u_ht * (
        4.0 * reactor.Phases.Liquids[0].vol / VESSEL_DIAM + VESSEL_AREA_BASE
    ) * (reactor.result.global_temp[-1] - UTILITY_TEMP_STEP)  # [W]

    assert reactor.result.q_ht[0] == pytest.approx(q_0_expected, rel=CONSERVATION_RTOL)
    assert reactor.result.q_ht[-1] == pytest.approx(
        q_end_expected, rel=CONSERVATION_RTOL
    )

    expected_temp_out_end = UTILITY_TEMP_STEP + q_end_expected / (
        UTILITY_MASS_FLOW_STEP * CUSTOM_UTILITY_CP
    )  # [K]
    assert reactor.Utility.temp_out == pytest.approx(
        expected_temp_out_end, rel=CONSERVATION_RTOL
    )
    final_util_inputs = reactor.Utility.get_inputs(RUNTIME)
    assert float(final_util_inputs["mass_flow"]) == pytest.approx(
        UTILITY_MASS_FLOW_STEP, rel=ALGEBRA_RTOL
    )
    assert float(final_util_inputs["vol_flow"]) == pytest.approx(
        UTILITY_MASS_FLOW_STEP / CUSTOM_UTILITY_RHO, rel=ALGEBRA_RTOL
    )


def test_reaction_enthalpy_normalization_and_reversible_shape_contract():
    """#22, #270: non-unit stoichiometry normalizes delta_hrxn and reversible kinetics run in non-isothermal vessels."""
    liquid = LiquidPhase(
        DATA_PATH,
        temp=INITIAL_TEMP,
        mass=LIQUID_MASS,
        mass_frac=np.array([0.30, 0.10, 0.0, 0.0, 0.60]),  # [-]
    )
    # 2 B --> A (2 * 50 g/mol = 100 g/mol, mass-conserving) with non-unit
    # reactant coefficient 2 and active equilibrium thermodynamics.
    kinetics = RxnKinetics(
        path=DATA_PATH,
        rxn_list=["2 B --> A"],
        k_params=1.0e-2,  # [L/(mol*s)]
        ea_params=0.0,  # [J/mol]
        keq_params=[2.5, 0.0],  # [-], temperature-independent Keq = 2.5
        delta_hrxn=-4.0e4,  # [J/mol of reaction as written]
        tref_hrxn=INITIAL_TEMP,  # [K], eliminates sensible delta_Cp correction at INITIAL_TEMP
    )
    reactor = BatchReactor(
        isothermal=False,
        adiabatic=True,
        temp_ref=INITIAL_TEMP,
        integrator=_scipy_lsoda(),
    )
    reactor.Phases = [liquid]
    reactor.RxnKinetics = kinetics
    reactor.compile_structure()

    # Check instantaneous energy balance rate dT/dt at t = 0 against independent hand derivation:
    # Q_gen = -(delta_hrxn / stoich_norm) * r_rxn * 1000 * V_liq [W], stoich_norm = 2.0 [-].
    # Note: RxnKinetics orders partic_species in database order: ['A', 'B'].
    conc_init = liquid.mole_conc[:2]  # [mol/L]
    rxn_rates, _ = kinetics.get_rxn_rates(
        conc_init,
        INITIAL_TEMP,
        return_both=True,
        delta_hrxn=np.array([-4.0e4]),
    )
    expected_q_gen = (
        -(-4.0e4 / 2.0) * float(rxn_rates[0]) * LITERS_PER_M3 * liquid.vol
    )  # [W]
    expected_dtemp_dt = expected_q_gen / reactor.Phases.get_total_heat_capacity()  # [K/s]

    y0 = reactor.create_solver_init_states()
    dy0 = reactor.unit_model(0.0, y0)
    unpacked_rates = reactor.solver_state_collection.unpack(dy0)
    assert float(unpacked_rates[StateKey("global_temp")]) == pytest.approx(
        expected_dtemp_dt, rel=CONSERVATION_RTOL
    )

    # Also integrate over a finite horizon to verify smooth non-isothermal trajectory and mass conservation.
    reactor.solve_unit(runtime=5.0)
    assert reactor.result.global_temp[-1] > INITIAL_TEMP
    assert reactor.Phases.mass == pytest.approx(LIQUID_MASS, rel=CONSERVATION_RTOL)


def test_cstr_zero_reaction_differing_feed_composition_has_zero_temperature_drift(
    tmp_path,
):
    """#272, #414, #440, #442: adiabatic CSTR fed at vessel temperature has dT/dt = 0, conserves inert solvent, and keeps its database."""
    holdup = LiquidPhase(
        DATA_PATH,
        temp=INITIAL_TEMP,
        mass=LIQUID_MASS,
        mass_frac=LIQUID_MASS_FRAC,
    )
    feed = LiquidStream(
        DATA_PATH,
        temp=INITIAL_TEMP,
        mass_flow=INLET_MASS_FLOW,
        mass_frac=INLET_MASS_FRAC,
    )
    cstr = ContinuousReactor(
        isothermal=False,
        adiabatic=True,
        temp_ref=298.15,
        integrator=_scipy_lsoda(),
    )
    cstr.Phases = [holdup]
    cstr.Inlet = feed
    cstr.RxnKinetics = RxnKinetics(
        path=DATA_PATH,
        rxn_list=["A --> B"],
        k_params=0.0,  # [1/s]
        ea_params=0.0,  # [J/mol]
    )

    cstr.solve_unit(time_grid=np.linspace(0.0, RUNTIME, 5))
    np.testing.assert_allclose(
        cstr.result.global_temp,
        np.full(5, INITIAL_TEMP),
        rtol=CONSERVATION_RTOL,
        atol=CONSERVATION_RTOL,
    )
    # Outlet stream retains the reactor's property database path (#442).
    outlet_stream = cstr.outlet_connections[0].stream
    assert outlet_stream.path_thermo == DATA_PATH

    # Also verify vapor inlet/outlet energy balance when an absent species (C, zero flow)
    # lacks cp_vapor coefficients (#414).
    with open(DATA_PATH, encoding="utf-8") as fh:
        db = json.load(fh)
    for rec in db.values():
        rec["t_crit"] = 200.0  # [K], supercritical at INITIAL_TEMP (310 K)
        rec["cp_vapor"] = [30.0, 0.0, 0.0, 0.0]  # [J/mol/K]
    db["C"]["cp_vapor"] = []  # absent species C lacks cp_vapor
    vap_db_path = str(tmp_path / "vapor_partial.json")
    with open(vap_db_path, "w", encoding="utf-8") as fh:
        json.dump(db, fh)

    vap_holdup = VaporPhase(
        vap_db_path,
        temp=INITIAL_TEMP,
        mass=0.5,  # [kg]
        mole_frac=VAPOR_MOLE_FRAC,  # C is 0.0
    )
    vap_feed = VaporStream(
        vap_db_path,
        temp=INITIAL_TEMP,
        mass_flow=0.01,  # [kg/s]
        mole_frac=np.array([0.10, 0.40, 0.0, 0.0, 0.50]),  # [-], C is 0.0
    )
    vap_cstr = ContinuousReactor(
        isothermal=False,
        adiabatic=True,
        temp_ref=298.15,
        integrator=_scipy_lsoda(),
    )
    vap_cstr.Phases = [vap_holdup]
    vap_cstr.Inlet = vap_feed
    vap_cstr.solve_unit(time_grid=np.linspace(0.0, 5.0, 4))
    np.testing.assert_allclose(
        vap_cstr.result.global_temp,
        np.full(4, INITIAL_TEMP),
        rtol=CONSERVATION_RTOL,
        atol=CONSERVATION_RTOL,
    )


def test_batch_reactor_conserves_inert_solvent_and_preserves_caller_phase_containers():
    """#440, #450: reaction conserves inert solvent mass and Phases setter does not mutate caller list or fail on tuple."""
    liquid = LiquidPhase(
        DATA_PATH,
        temp=INITIAL_TEMP,
        mass=LIQUID_MASS,
        mass_frac=LIQUID_MASS_FRAC,
    )
    caller_list = [liquid]
    reactor = BatchReactor(isothermal=True, integrator=_scipy_lsoda())
    reactor.Phases = caller_list
    assert len(caller_list) == 1

    # Also verify tuple assignment works without TypeError (#450).
    reactor.Phases = (liquid,)
    reactor.RxnKinetics = RxnKinetics(
        path=DATA_PATH,
        rxn_list=["A --> B"],
        k_params=0.05,  # [1/s]
        ea_params=0.0,  # [J/mol]
    )

    initial_solvent_mass = float(liquid.mass_j[4])  # [kg]
    reactor.solve_unit(time_grid=np.linspace(0.0, RUNTIME, 6))
    final_solvent_mass = float(reactor.Phases.Liquids[0].mass_j[4])  # [kg]
    assert final_solvent_mass == pytest.approx(
        initial_solvent_mass, rel=CONSERVATION_RTOL
    )
    assert float(reactor.result.mass_j_liquid0[-1, 4]) == pytest.approx(
        initial_solvent_mass, rel=CONSERVATION_RTOL
    )


def test_semibatch_crystallizer_slurry_feed_at_vessel_temperature_has_zero_temperature_drift():
    """#265: adiabatic SemiBatchCrystallizer with inactive kinetics fed slurry at its own temperature holds T constant."""
    grid = np.array([10.0, 25.0, 50.0, 100.0])  # [um]
    seed_distrib = np.array([1.0e8, 2.0e8, 1.0e8, 0.5e8])  # [#/(um*m**3 slurry)]

    liquid = LiquidPhase(
        DATA_PATH,
        temp=INITIAL_TEMP,
        mass=LIQUID_MASS,
        mass_frac=LIQUID_MASS_FRAC,
        name_solv="solvent",
    )
    solid = SolidPhase(
        DATA_PATH,
        temp=INITIAL_TEMP,
        mass=0.0,
        mass_frac=SOLID_C_MASS_FRAC,
    )
    solid.mechanisms = OneDFVMMechanism(
        owning_phase=solid,
        target_components="C",
        solvent_name="solvent",
        x_grid=grid,
        distrib_init=seed_distrib.copy(),
        kv=0.5,
    )

    feed_liq = LiquidStream(
        DATA_PATH,
        temp=INITIAL_TEMP,
        vol_flow=INLET_VOL_FLOW,
        mass_frac=LIQUID_MASS_FRAC,
        name_solv="solvent",
    )
    feed_sol = SolidStream(
        DATA_PATH,
        temp=INITIAL_TEMP,
        vol_flow=0.0,
        mass_frac=SOLID_C_MASS_FRAC,
    )
    feed_sol.mechanisms = OneDFVMMechanism(
        owning_phase=feed_sol,
        target_components="C",
        solvent_name="solvent",
        x_grid=grid,
        distrib_init=seed_distrib.copy(),
        kv=0.5,
    )
    slurry_inlet = MixedStream([feed_liq, feed_sol])

    cryst = SemiBatchCrystallizer(
        isothermal=False,
        adiabatic=True,
        integrator=_scipy_lsoda(),
    )
    cryst.Phases = [liquid, solid]
    cryst.CrystKinetics = CrystKinetics(
        coeff_solub=[2000.0],  # [kg/m**3], high solubility keeps growth = 0
        growth=(0.0, 0.0, 1.0),
        nucl_prim=(0.0, 0.0, 1.0),
    )
    cryst.Inlet = slurry_inlet

    cryst.solve_unit(time_grid=np.linspace(0.0, 10.0, 5))
    np.testing.assert_allclose(
        cryst.result.global_temp,
        np.full(5, INITIAL_TEMP),
        rtol=CONSERVATION_RTOL,
        atol=CONSERVATION_RTOL,
    )
    # #281: x_grid remains a static 1-D size grid matching the configured nodes.
    np.testing.assert_array_equal(cryst.Phases.Solids[0].x_grid, grid)


def test_crystallizer_scale_deprecation_and_arbitrary_moment_orders():
    """#286, #288, #330: scale= is deprecated and ignored; MomentsPopulationBalance supports >4 moment orders and legacy round-trip."""
    moments_6 = np.array([1.0e10, 5.0e11, 3.0e13, 2.0e15, 1.5e17, 1.2e19])  # [um**k/m**3 slurry]
    liquid = LiquidPhase(
        DATA_PATH,
        temp=INITIAL_TEMP,
        mass=LIQUID_MASS,
        mass_frac=LIQUID_MASS_FRAC,
        name_solv="solvent",
    )
    solid_unscaled = SolidPhase(
        DATA_PATH,
        temp=INITIAL_TEMP,
        mass=0.0,
        mass_frac=SOLID_C_MASS_FRAC,
    )
    mech_unscaled = MomentsPopulationBalance(
        owning_phase=solid_unscaled,
        target_components="C",
        solvent_name="solvent",
        moments_init=moments_6.copy(),
        kv=0.5,
    )
    solid_unscaled.mechanisms = mech_unscaled

    solid_scaled = SolidPhase(
        DATA_PATH,
        temp=INITIAL_TEMP,
        mass=0.0,
        mass_frac=SOLID_C_MASS_FRAC,
    )
    with pytest.warns(DeprecationWarning, match="scale= is deprecated and ignored"):
        mech_scaled = MomentsPopulationBalance(
            owning_phase=solid_scaled,
            target_components="C",
            solvent_name="solvent",
            moments_init=moments_6.copy(),
            kv=0.5,
            scale=1e-9,
        )
    solid_scaled.mechanisms = mech_scaled

    assert mech_unscaled.num_mom == 6
    assert mech_scaled.num_mom == 6

    cryst_a = BatchCrystallizer(isothermal=True, integrator=_scipy_lsoda())
    cryst_a.Phases = [
        LiquidPhase(
            DATA_PATH,
            temp=INITIAL_TEMP,
            mass=LIQUID_MASS,
            mass_frac=LIQUID_MASS_FRAC,
            name_solv="solvent",
        ),
        solid_unscaled,
    ]
    cryst_a.CrystKinetics = CrystKinetics(
        coeff_solub=[10.0],  # [kg/m**3]
        growth=(0.5, 0.0, 1.0),
        nucl_prim=(100.0, 0.0, 1.0),
    )

    cryst_b = BatchCrystallizer(isothermal=True, integrator=_scipy_lsoda())
    cryst_b.Phases = [liquid, solid_scaled]
    cryst_b.CrystKinetics = CrystKinetics(
        coeff_solub=[10.0],  # [kg/m**3]
        growth=(0.5, 0.0, 1.0),
        nucl_prim=(100.0, 0.0, 1.0),
    )

    cryst_a.solve_unit(time_grid=np.linspace(0.0, 5.0, 4))
    cryst_b.solve_unit(time_grid=np.linspace(0.0, 5.0, 4))

    assert cryst_a.result.mu_n_solid0.shape == (4, 6)
    np.testing.assert_allclose(
        cryst_a.result.mu_n_solid0,
        cryst_b.result.mu_n_solid0,
        rtol=CONSERVATION_RTOL,
    )
    assert cryst_a.Phases.Solids[0].mass == pytest.approx(
        cryst_b.Phases.Solids[0].mass, rel=CONSERVATION_RTOL
    )

    # Verify legacy conversion round-trip preserves all 6 moment orders (#288, #330).
    vol_slurry = mech_unscaled._slurry_volume(mech_unscaled.mu_n)  # [m**3]
    legacy_state = mech_unscaled.to_legacy_state()
    assert legacy_state["num_mom"] == 6
    assert legacy_state["moments"].shape == (6,)
    expected_legacy_moments = (
        mech_unscaled.mu_n * vol_slurry * (MICRON_TO_M ** np.arange(6))
    )  # [m**k]
    np.testing.assert_allclose(
        legacy_state["moments"], expected_legacy_moments, rtol=ALGEBRA_RTOL
    )


def test_mass_only_solid_volume_and_phase_family_subclass_classification():
    """#419, #427: mass-only SolidPhase/SolidStream define vol, and custom subclasses classify by phase_family."""

    class CustomBrine(LiquidPhase):
        """User subclass whose name omits the substring 'Liquid' (#427)."""

    class CustomCrystal(SolidPhase):
        """User subclass whose name omits the substring 'Solid' (#427)."""

    class CustomHeadspace(VaporPhase):
        """User subclass whose name omits the substring 'Vapor' (#427)."""

    class LiquidCrystal(SolidPhase):
        """Solid subclass whose class name contains 'Liquid' (#427)."""

    class SolidifiedSolvent(LiquidPhase):
        """Liquid subclass whose class name contains 'Solid' (#427)."""

    brine = CustomBrine(
        DATA_PATH,
        temp=INITIAL_TEMP,
        mass=LIQUID_MASS,
        mass_frac=LIQUID_MASS_FRAC,
    )
    crystal = CustomCrystal(
        DATA_PATH,
        temp=INITIAL_TEMP,
        mass=SOLID_MASS,
        mass_frac=SOLID_C_MASS_FRAC,
    )
    headspace = CustomHeadspace(
        DATA_PATH,
        temp=INITIAL_TEMP,
        mass=0.05,  # [kg]
        mole_frac=VAPOR_MOLE_FRAC,
    )
    liq_crystal = LiquidCrystal(
        DATA_PATH,
        temp=INITIAL_TEMP,
        mass=SOLID_MASS,
        mass_frac=SOLID_C_MASS_FRAC,
    )
    solid_solvent = SolidifiedSolvent(
        DATA_PATH,
        temp=INITIAL_TEMP,
        mass=LIQUID_MASS,
        mass_frac=LIQUID_MASS_FRAC,
    )

    # #419: mass-only SolidPhase and SolidStream have well-defined vol = mass / density.
    expected_solid_vol = SOLID_MASS / crystal.getDensity()  # [m**3]
    assert crystal.vol == pytest.approx(expected_solid_vol, rel=ALGEBRA_RTOL)
    crystal_stream = crystal.to_stream()
    assert crystal_stream.vol_flow == pytest.approx(
        expected_solid_vol, rel=ALGEBRA_RTOL
    )

    # #427: MixedPhase and deprecated classify_phases / getPropsPhaseMix classify by phase_family.
    mixed = MixedPhase([brine, crystal, headspace])
    assert list(mixed.Liquids) == [brine]
    assert list(mixed.Solids) == [crystal]
    assert list(mixed.Vapors) == [headspace]
    assert mixed.vol == pytest.approx(
        brine.vol + crystal.vol + headspace.vol, rel=ALGEBRA_RTOL
    )

    class _Container:
        """Minimal holder object with a Phases attribute for classify_phases."""

        def __init__(self, phases):
            """Initialize the holder with a sequence of phase objects.

            Parameters
            ----------
            phases : Sequence[object]
                Phase objects to store on ``self.Phases``.
            """
            self.Phases = phases

    holder = _Container([brine, crystal, headspace])
    with pytest.warns(DeprecationWarning, match="classify_phases"):
        classify_phases(holder)
    assert holder.Liquid_1 is brine
    assert holder.Solid_1 is crystal
    assert holder.Vapor_1 is headspace

    # Conflicting class-name substrings still classify by explicit phase_family in both orders.
    holder_conflict = _Container([liq_crystal, solid_solvent])
    with pytest.warns(DeprecationWarning, match="classify_phases"):
        classify_phases(holder_conflict)
    assert holder_conflict.Solid_1 is liq_crystal
    assert holder_conflict.Liquid_1 is solid_solvent

    total_vol = solid_solvent.vol + liq_crystal.vol  # [m**3]
    expected_vfrac_liq_first = np.array(
        [solid_solvent.vol / total_vol, liq_crystal.vol / total_vol]
    )  # [-]
    expected_vfrac_sol_first = np.array(
        [liq_crystal.vol / total_vol, solid_solvent.vol / total_vol]
    )  # [-]

    with pytest.warns(DeprecationWarning, match="getPropsPhaseMix"):
        _, _, _, vfrac_1, mfrac_1 = getPropsPhaseMix([solid_solvent, liq_crystal])
    with pytest.warns(DeprecationWarning, match="getPropsPhaseMix"):
        _, _, _, vfrac_2, mfrac_2 = getPropsPhaseMix([liq_crystal, solid_solvent])

    np.testing.assert_allclose(vfrac_1, expected_vfrac_liq_first, rtol=ALGEBRA_RTOL)
    np.testing.assert_allclose(vfrac_2, expected_vfrac_sol_first, rtol=ALGEBRA_RTOL)
    assert np.sum(mfrac_1) == pytest.approx(1.0, rel=ALGEBRA_RTOL)
    assert np.sum(mfrac_2) == pytest.approx(1.0, rel=ALGEBRA_RTOL)


def test_symbolic_state_event_and_idempotent_phase_reconfiguration():
    """#274, #312: StateEvent queries StateKey cleanly and repeated Phases assignment does not corrupt state layout or caller arrays."""
    caller_conc = np.array([0.8, 0.2, 0.0, 0.0, 0.0])  # [mol/L], before solvent closure
    caller_snapshot = caller_conc.copy()

    liquid_1 = LiquidPhase(
        DATA_PATH,
        temp=INITIAL_TEMP,
        vol=0.005,  # [m**3]
        mole_conc=caller_conc,
        name_solv="solvent",
    )
    # Caller's concentration array was not mutated in place (#312).
    np.testing.assert_array_equal(caller_conc, caller_snapshot)

    target_mass_a = 0.5 * float(liquid_1.mass_j[0])  # [kg], terminal event at 50 % conversion of A
    event = StateEvent(
        name="half_conversion_A",
        function=lambda t, state, unit: float(
            state[StateKey("mass_j", PhaseRef("liquid", 0))][0] - target_mass_a
        ),
        direction=-1,
        terminal=True,
    )
    reactor = BatchReactor(
        isothermal=True,
        state_events=[event],
        integrator=_scipy_lsoda(),
    )
    # Repeated phase assignment (#274) remains idempotent.
    reactor.Phases = [liquid_1]
    liquid_2 = LiquidPhase(
        DATA_PATH,
        temp=INITIAL_TEMP,
        vol=0.005,  # [m**3]
        mole_conc=caller_conc,
        name_solv="solvent",
    )
    reactor.Phases = [liquid_2]
    reactor.RxnKinetics = RxnKinetics(
        path=DATA_PATH,
        rxn_list=["A --> B"],
        k_params=0.10,  # [1/s]
        ea_params=0.0,  # [J/mol]
    )

    t_expected = np.log(2.0) / 0.10  # [s], analytical half-life ln(2)/k
    reactor.solve_unit(runtime=20.0)
    assert reactor.result.time[-1] == pytest.approx(t_expected, rel=EVENT_RTOL)
    assert float(reactor.Phases.Liquids[0].mass_j[0]) == pytest.approx(
        target_mass_a, rel=EVENT_RTOL
    )


def test_flowsheet_connection_operating_mode_contracts_with_refactored_vessels():
    """#436, #438, #447: Connection and SimulationExec enforce batch/semibatch/continuous compatibility on refactored vessels."""
    # 1. Direct BatchReactor -> ContinuousReactor connection is rejected when solved (#436).
    sim_bad = SimulationExec(DATA_PATH, flowsheet="R01 --> R02")
    sim_bad.R01 = BatchReactor(isothermal=True, integrator=_scipy_lsoda())
    sim_bad.R01.Phases = [
        LiquidPhase(
            DATA_PATH,
            temp=INITIAL_TEMP,
            mass=LIQUID_MASS,
            mass_frac=LIQUID_MASS_FRAC,
        )
    ]
    sim_bad.R01.RxnKinetics = RxnKinetics(
        path=DATA_PATH, rxn_list=["A --> B"], k_params=0.05, ea_params=0.0
    )
    sim_bad.R02 = ContinuousReactor(isothermal=True, integrator=_scipy_lsoda())
    sim_bad.R02.Phases = [
        LiquidPhase(
            DATA_PATH,
            temp=INITIAL_TEMP,
            mass=LIQUID_MASS,
            mass_frac=LIQUID_MASS_FRAC,
        )
    ]
    sim_bad.R02.RxnKinetics = RxnKinetics(
        path=DATA_PATH, rxn_list=["A --> B"], k_params=0.05, ea_params=0.0
    )
    with pytest.raises(ValueError, match="Batch.*Continuous|BatchToFlowConnector"):
        sim_bad.SolveFlowsheet(
            kwargs_run={"R01": {"runtime": 5.0}, "R02": {"runtime": 5.0}},
            verbose=False,
        )

    # 2. Direct ContinuousReactor -> BatchReactor charge is rejected with TypeError (#436).
    cstr = ContinuousReactor(
        Phases=[
            LiquidPhase(
                DATA_PATH,
                temp=INITIAL_TEMP,
                mass=LIQUID_MASS,
                mass_frac=LIQUID_MASS_FRAC,
                name_solv="D",
            )
        ],
        isothermal=True,
        integrator=_scipy_lsoda(),
    )
    assert len(cstr.outlet_connections) == 1
    cstr.Inlet = LiquidStream(
        DATA_PATH,
        temp=INITIAL_TEMP,
        mass_flow=INLET_MASS_FLOW,
        mass_frac=LIQUID_MASS_FRAC,
        name_solv="D",
    )
    cstr.RxnKinetics = RxnKinetics(
        path=DATA_PATH, rxn_list=["A --> B"], k_params=0.05, ea_params=0.0
    )
    cstr.solve_unit(runtime=5.0)

    batch_dest = BatchReactor(isothermal=True, integrator=_scipy_lsoda())
    with pytest.raises(TypeError, match="batch unit.*DynamicCollector"):
        batch_dest.Phases = cstr.Outlet
    with pytest.raises(ValueError, match="is continuous.*DynamicCollector"):
        Connection(cstr, batch_dest).transfer_data()

    # 3. ContinuousReactor -> SemiBatchReactor routes stream to SemiBatchReactor's inlet (#438)
    #    without overwriting its initial holdup!
    semibatch = SemiBatchReactor(isothermal=True, integrator=_scipy_lsoda())
    initial_holdup = LiquidPhase(
        DATA_PATH,
        temp=INITIAL_TEMP,
        mass=LIQUID_MASS,
        mass_frac=INLET_MASS_FRAC,
        name_solv="D",
    )
    semibatch.Phases = [initial_holdup]
    semibatch.RxnKinetics = RxnKinetics(
        path=DATA_PATH, rxn_list=["A --> B"], k_params=0.02, ea_params=0.0
    )

    Connection(cstr, semibatch).transfer_data()
    # Initial holdup mass is still LIQUID_MASS before solving semibatch, and after solving
    # semibatch its mass increases by the integrated inflow from cstr.
    assert semibatch.Phases.mass == pytest.approx(LIQUID_MASS, rel=ALGEBRA_RTOL)
    semibatch.solve_unit(runtime=5.0)
    assert semibatch.Phases.mass > LIQUID_MASS

    # 4. SemiBatchReactor -> BatchReactor transfers final holdup (#438, #447).
    batch_after_semi = BatchReactor(isothermal=True, integrator=_scipy_lsoda())
    assert batch_after_semi.Phases is None
    batch_after_semi.Phases = [
        LiquidPhase(
            DATA_PATH,
            temp=INITIAL_TEMP,
            mass=1.0,  # [kg], placeholder charge to be replaced by upstream holdup
            mass_frac=LIQUID_MASS_FRAC,
            name_solv="D",
        )
    ]
    batch_after_semi.RxnKinetics = RxnKinetics(
        path=DATA_PATH, rxn_list=["A --> B"], k_params=0.02, ea_params=0.0
    )
    Connection(semibatch, batch_after_semi).transfer_data()
    assert batch_after_semi.Phases.mass == pytest.approx(
        semibatch.Phases.mass, rel=CONSERVATION_RTOL
    )
