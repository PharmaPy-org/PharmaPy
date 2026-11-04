"""Native JSON thermal paths against analytical balances and independent ODEs.

Fixtures are deliberately synthetic: 1 kg/L, Cp=1 kJ/(kg K), and a mass-
conserving substrate-to-product reaction isolate the energy balance from
organism-specific assumptions. No optional solver or substituted collaborator
is used. Thermal coefficients construct analytical cases, not empirical fits.
"""

from functools import partial

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from PharmaPy.Bioreactors import build_bioreactor
from PharmaPy.Bioreactors.observations import predict_measurements, read_observations
from PharmaPy.Bioreactors.input_format import split_forward_inputs, normalize_forward_inputs, EVENT_FIELDS
from PharmaPy.ThermoModule import MissingPropertyError
from PharmaPy.DataClasses import StateKey
from PharmaPy.Commons import temp_ufun
from PharmaPy.IntegratorBackends import ScipyBackend
from PharmaPy.Kinetics import RxnKinetics
from PharmaPy.Phases_Refactored import LiquidPhase
from PharmaPy.ProcessControl_Refactored import SimpleTemperatureController, ContinuousVesselController
from PharmaPy.Reactors_Refactored import BatchReactor, SemiBatchReactor, ContinuousReactor
from PharmaPy.Streams_Refactored import LiquidStream
from PharmaPy.Utilities import CoolingWater


EXAMPLES = Path(__file__).parents[2] / "examples/bioreactors"
CP = 1000.0  # [J/(kg K)]; chosen constant gives a transparent 1 kJ/K initial holdup
INITIAL_TEMPERATURE = 300.0  # [K]; synthetic reference, not an organism operating recommendation
RATE_CONSTANT = 0.01  # [1/s]; chosen reaction timescale 100 s
REACTION_ENTHALPY = -200000.0  # [J/kg substrate]; synthetic exothermic conversion gives 2 K full-conversion rise
RTOL = 1e-9  # [-]; tighter than the 1e-6 relative analytical acceptance budget
ATOL = 1e-11  # [native kg, K, m3]; resolves 1e-2 kg substrate and temperature to the stated budgets


# Real fixture construction and independent qualification helpers.
@pytest.fixture
def thermal_case(tmp_path):
    """Construct a constant-Cp liquid and first-order S-to-P JSON model.

    Parameters
    ----------
    tmp_path : Path
        Pytest's isolated fixture directory.

    Returns
    -------
    tuple of dict and Path
        Existing combined-input layout and synthetic thermodynamic fixture.
    """
    folder = EXAMPLES / "Doyle_Exercise_4_15/inputs"
    document = json.loads((folder / "mechanism.json").read_text())
    thermo = json.loads((folder / "thermo.json").read_text())
    for properties in thermo.values():
        properties["cp_liq"] = [CP * properties["mw"] / 1000.0]  # [J/(mol K)]; exact kg-to-g conversion
    path = tmp_path / "thermo.json"
    path.write_text(json.dumps(thermo))
    document["operation"].update(
        mode="batch", temperature_k=INITIAL_TEMPERATURE,
        runtime={"unit": "s", "value": 100.0},  # [s]; one reaction timescale
        thermal={"mode": "adiabatic", "heat_generation_w": {
            "op": "multiply", "args": [REACTION_ENTHALPY, {"source_rate": "substrate"}]}})
    document["recipes"] = {"continuous": []}
    document["state"]["species"]["amounts"].update(
        biomass=0.0, substrate=10.0, product=0.0)  # [g]; 0.01 kg substrate, full-conversion heat 2000 J
    document["model"]["kinetics"] = {
        "rule_graph": [
            {"identifier": "uptake", "expression": {"op": "multiply", "args": [
                -RATE_CONSTANT, {"concentration": "substrate"}]}},
            {"identifier": "production", "expression": {"op": "negate", "args": [{"ref": "uptake"}]}}],
        "species_mass_rates": {"substrate": "uptake", "product": "production"},
        "rate_unit": "kg/(m3 s)", "constants": {}}
    # Doyle's formal substrate MW=1000 g/mol makes mmol/L numerically kg/m3.
    assert thermo["substrate"]["mw"] == 1000.0  # [g/mol]; fixture's formal concentration conversion
    document["rate_provider"] = None
    document["numerical"].update(backend="scipy", step={"unit": "s", "value": 5.0},
                                  relative_tolerance=RTOL, absolute_tolerance=ATOL)
    return document, path


def _independent_thermal_rhs(time, state):
    """Evaluate an independent Arrhenius S-to-P adiabatic reference.

    Parameters
    ----------
    time : float
        Time [s]; autonomous problem.
    state : ndarray
        Substrate mass [kg] and temperature [K].

    Returns
    -------
    list of float
        Substrate derivative [kg/s] and temperature derivative [K/s].
    """
    temperature_sensitivity = 3000.0  # [K]; synthetic Ea/R, selected to resolve thermal feedback
    rate = RATE_CONSTANT * np.exp(temperature_sensitivity * (1 / INITIAL_TEMPERATURE - 1 / state[1]))  # [1/s]
    consumed = rate * state[0]  # [kg/s]
    return [-consumed, -REACTION_ENTHALPY * consumed / CP]


def _native_reaction_reference(document, path):
    """Build the same synthetic S-to-P process using native tools directly.

    Parameters
    ----------
    document : dict
        Synthetic process declaration; thermal modes and initial temperature
        mirror the selected test scenario.
    path : Path
        Synthetic native thermodynamic database.

    Returns
    -------
    MultiPhaseVessel
        Native reactor with RxnKinetics, an optional native utility or
        prescribed-temperature controller, and a real water feed when required.
    """
    operation = document["operation"]
    mode = operation["mode"]
    thermal_mode = operation["thermal"]["mode"]
    reactor = {"batch": BatchReactor, "fed-batch": SemiBatchReactor, "continuous": ContinuousReactor}[mode]
    options = {"isothermal": thermal_mode == "isothermal", "adiabatic": thermal_mode == "adiabatic"}
    if thermal_mode == "heat-transfer":
        options.update(h_conv=100.0, diam=0.1, area_base=0.0)  # [W/(m2 K)], [m], [m2]; U*A=2 W/K
    if thermal_mode == "prescribed-temperature":
        controller = ContinuousVesselController if mode == "continuous" else SimpleTemperatureController
        options["controller"] = controller(partial(temp_ufun, temp_zero=300.0, rate=0.02))  # [K], [K/s]; 2 K rise in 100 s
    unit = reactor(integrator=ScipyBackend({"rtol": RTOL, "atol": ATOL}, inventory_boundaries=True), **options)
    properties = json.loads(path.read_text())
    masses = {"substrate": 0.01, "product": 0.0, "biomass": 0.0, "water": 0.99}  # [kg]; 1 kg charge
    unit.Phases = LiquidPhase(path, mass=1.0, mass_frac=[masses[name] for name in properties],
                             temp=INITIAL_TEMPERATURE, verbose=False)
    # Formal substrate/product molecular weights are 1 kg/mol in this fixture.
    # Thus the J/kg substrate heat equals J/mol reaction numerically.
    unit.RxnKinetics = RxnKinetics(path, k_params=[RATE_CONSTANT], ea_params=[0.0],
        stoich_matrix=[[-1.0, 1.0]], partic_species=["substrate", "product"],
        delta_hrxn=[REACTION_ENTHALPY], tref_hrxn=INITIAL_TEMPERATURE)
    if thermal_mode == "heat-transfer":
        unit.Utility = CoolingWater(temp_in=290.0, mass_flow=0.1, h_conv=100.0)  # [K], [kg/s], [W/(m2 K)]
    if mode != "batch":
        water_fraction = [float(name == "water") for name in properties]  # [-]
        unit.Inlet = LiquidStream(path, mass_flow=0.001, mass_frac=water_fraction,
                                  temp=280.0, verbose=False)  # [kg/s], [K]; 3.6 L/h at fixture density
    if mode == "continuous":
        unit.controller.target_volume = 0.001  # [m3]; 1 L initial charge
    return unit


# Public-path regressions, independent balances, and failure qualification.
@pytest.mark.parametrize("enthalpy", [REACTION_ENTHALPY, -REACTION_ENTHALPY, 0.0])
def test_adiabatic_reaction_conserves_energy_and_mass(thermal_case, enthalpy):
    document, path = thermal_case
    document["operation"]["thermal"]["heat_generation_w"]["args"][0] = enthalpy
    original = deepcopy(document)
    assembly = build_bioreactor(document, thermo_path=path)
    result = assembly.solve()[0]
    names = assembly.phase.name_species
    substrate = result.mass_j_liquid0[:, names.index("substrate")]  # [kg]
    temperature = np.asarray(result.global_temp).reshape(-1)  # [K]
    initial_substrate = 0.01  # [kg]; 10 g in the fixture
    expected_substrate = initial_substrate * np.exp(-RATE_CONSTANT * result.time)  # [kg]
    np.testing.assert_allclose(substrate, expected_substrate, rtol=1e-6, atol=ATOL)
    energy = CP * (temperature - INITIAL_TEMPERATURE) + enthalpy * (initial_substrate - substrate)  # [J]; 1 kg total
    np.testing.assert_allclose(energy, 0.0, atol=1e-7)  # [J]; cancellation budget relative to 2000 J full-conversion heat
    np.testing.assert_allclose(result.mass_j_liquid0.sum(axis=1), 1.0, rtol=0.0, atol=ATOL)  # [kg]
    assert document == original


def test_explicit_isothermal_preserves_legacy_history(thermal_case):
    document, path = thermal_case
    document["operation"].pop("thermal")
    legacy = build_bioreactor(document, thermo_path=path).solve()[0]
    document["operation"]["thermal"] = {"mode": "isothermal"}
    explicit = build_bioreactor(document, thermo_path=path).solve()[0]
    np.testing.assert_array_equal(explicit.mass_j_liquid0, legacy.mass_j_liquid0)
    np.testing.assert_array_equal(explicit.global_temp, legacy.global_temp)


def test_bath_cooling_matches_exponential_and_native_heat_output(thermal_case):
    document, path = thermal_case
    bath_temperature = 290.0  # [K]; 10 K below synthetic initial liquid
    document["operation"]["thermal"] = {
        "mode": "heat-transfer", "heat_generation_w": 0.0,
        "heat_transfer": {
            "vessel_coefficient_w_m2_k": 100.0,  # [W/(m2 K)]; equal film coefficients give overall U=50
            "utility_coefficient_w_m2_k": 100.0,  # [W/(m2 K)]
            "diameter_m": 0.1, "base_area_m2": 0.0,  # [m], [m2]; lateral wetted area=4V/d=0.04 m2
            "utility_temperature_k": bath_temperature,
            "utility_mass_flow_kg_s": 0.1}}  # [kg/s]; native constant-inlet utility
    assembly = build_bioreactor(document, thermo_path=path)
    result = assembly.solve()[0]
    conductance = 2.0  # [W/K]; U*A=50*0.04
    expected = bath_temperature + 10.0 * np.exp(-conductance * result.time / CP)  # [K]
    np.testing.assert_allclose(np.asarray(result.global_temp).reshape(-1), expected, rtol=0.0, atol=1e-7)
    np.testing.assert_allclose(np.asarray(result.q_ht).reshape(-1), conductance * (expected - bath_temperature),
                               rtol=1e-6, atol=1e-7)  # [W]; positive heat leaves vessel


def test_live_temperature_couples_kinetics_and_converges(thermal_case):
    document, path = thermal_case
    original_rate = document["model"]["kinetics"]["rule_graph"][0]["expression"]
    temperature_sensitivity = 3000.0  # [K]; same explicitly stated synthetic Ea/R as independent reference
    arrhenius = {"op": "exp", "args": [{"op": "multiply", "args": [temperature_sensitivity,
        {"op": "subtract", "args": [1 / INITIAL_TEMPERATURE,
            {"op": "divide", "args": [1.0, {"state": "temperature_k"}]}]}]}]}
    document["model"]["kinetics"]["rule_graph"][0]["expression"] = {
        "op": "multiply", "args": [original_rate, arrhenius]}
    histories = []
    for tolerance in (1e-5, 1e-7, RTOL):  # [-]; successive two-decade solver refinement
        document["numerical"]["relative_tolerance"] = tolerance
        histories.append(build_bioreactor(document, thermo_path=path).solve()[0])
    reference = solve_ivp(_independent_thermal_rhs, (0.0, 100.0), [0.01, INITIAL_TEMPERATURE],
                          method="DOP853", rtol=1e-12, atol=1e-13, t_eval=histories[-1].time)
    assert reference.success
    errors = [np.max(np.abs(np.asarray(result.global_temp).reshape(-1) - reference.y[1])) for result in histories]  # [K]
    assert errors[-1] < errors[0]
    assert errors[-1] < 1e-6  # [K]; independent thermal trajectory acceptance budget
    np.testing.assert_allclose(histories[-1].mass_j_liquid0[:, 1], reference.y[0], rtol=1e-6, atol=ATOL)


@pytest.mark.parametrize("mode", ["fed-batch", "continuous"])
def test_continuous_cold_feed_matches_independent_mixing_balance(thermal_case, mode):
    document, path = thermal_case
    document["operation"].update(mode=mode)
    document["operation"]["thermal"]["heat_generation_w"] = 0.0
    feed_temperature = 280.0  # [K]; cold nutrient-free liquid
    document["recipes"]["continuous"] = [{"time": 0.0, "event_type": "flow", "volume_l_h": 3.6,
        "composition": "nutrient-free", "temperature_k": feed_temperature}]  # [L/h]; mass flow=0.001 kg/s
    assembly = build_bioreactor(document, thermo_path=path)
    result = assembly.solve()[0]
    if mode == "fed-batch":
        expected = feed_temperature + 20.0 / (1.0 + 0.001 * result.time)  # [K]; mass-weighted mixing, equal Cp
    else:
        expected = feed_temperature + 20.0 * np.exp(-0.001 * result.time)  # [K]; constant-mass CSTR
    np.testing.assert_allclose(np.asarray(result.global_temp).reshape(-1), expected, atol=1e-6, rtol=0.0)


def test_bolus_mixing_and_temperature_observation_event_sides(thermal_case):
    document, path = thermal_case
    document["operation"].update(mode="fed-batch")
    document["operation"]["thermal"]["heat_generation_w"] = 0.0
    document["recipes"]["continuous"] = [{"time": 50.0, "event_type": "feed", "volume_l": 0.5,
        "concentrations_mmol_l": {}, "composition": "nutrient-free",
        "temperature_k": 270.0}]  # [s], [L], [K]; adds 0.5 kg cold water
    assembly = build_bioreactor(document, thermo_path=path)
    histories = assembly.solve(report_times=[50.0])
    declaration = {"schema_version": 1, "time_unit": "s", "fields": {
        "before": {"species": None, "quantity": "temperature", "unit": "K", "event_side": "pre", "time_key": "t", "value_key": "v"},
        "after": {"species": None, "quantity": "temperature", "unit": "degC", "event_side": "post", "time_key": "t", "value_key": "v"}}}
    observations = read_observations(declaration, {name: {"t": [50.0], "v": [0.0]} for name in declaration["fields"]})
    prediction = predict_measurements(assembly, histories, declaration, observations)
    np.testing.assert_allclose(prediction["values"]["before"], [300.0], rtol=0.0, atol=1e-8)  # [K]
    np.testing.assert_allclose(prediction["values"]["after"], [16.85], rtol=0.0, atol=1e-8)  # [degC]; (300+0.5*270)/1.5=290 K


@pytest.mark.parametrize("thermal,match", [
    ({"mode": "unknown"}, "thermal.mode"),
    ({"mode": "adiabatic"}, "requires exactly"),
    ({"mode": "adiabatic", "heat_generation_w": {"source_rate": "missing"}}, "known source_rate"),
    ({"mode": "isothermal", "heat_generation_w": 0.0}, "only mode"),
    ({"mode": "heat-transfer", "heat_generation_w": 0.0, "heat_transfer": None}, "heat_transfer requires"),
])
def test_invalid_thermal_configuration_is_actionable(thermal_case, thermal, match):
    document, path = thermal_case
    document["operation"]["thermal"] = thermal
    with pytest.raises(ValueError, match=match):
        build_bioreactor(document, thermo_path=path)


def test_nonisothermal_requires_real_heat_capacity_data(thermal_case):
    document, path = thermal_case
    data = json.loads(path.read_text())
    for properties in data.values():
        properties["cp_liq"] = []
    path.write_text(json.dumps(data))
    with pytest.raises(MissingPropertyError, match="cp_liq"):
        build_bioreactor(document, thermo_path=path)
    document["operation"].pop("thermal")
    assert build_bioreactor(document, thermo_path=path).unit.isothermal


@pytest.mark.parametrize("folder", ["generic_batch", "ecoli_ye_fed_batch/reconciliation"])
def test_optimized_biology_uses_accepted_sources_for_heat_and_live_temperature(tmp_path, folder):
    """Exercise real LP/MFA collaborators through native material/heat handoffs."""
    inputs = EXAMPLES / folder / "inputs"
    document = json.loads((inputs / "mechanism.json").read_text())
    thermo_path = inputs / "thermo.json"
    if not thermo_path.exists():
        thermo_path = EXAMPLES / "ecoli_ye_fed_batch/inputs/thermo.json"
    properties = json.loads(thermo_path.read_text())
    for entry in properties.values():
        entry["cp_liq"] = [CP * entry["mw"] / 1000.0]  # [J/(mol K)]
    path = tmp_path / "thermal-properties.json"
    path.write_text(json.dumps(properties))
    species = "nutrient" if folder == "generic_batch" else "glucose"
    document["operation"]["thermal"] = {"mode": "adiabatic", "heat_generation_w": {
        "op": "multiply", "args": [REACTION_ENTHALPY, {"source_rate": species}]}}
    document["operation"]["mode"] = "batch"
    document["operation"]["temperature_k"] = INITIAL_TEMPERATURE
    document["operation"]["runtime"] = {"unit": "s", "value": 1.0}  # [s]; short handoff probe
    document["recipes"][document["operation"]["recipe"]["name"]] = []
    if folder == "generic_batch":
        # Use the existing public provider boundary; this example has a
        # constant uptake declaration rather than an installed kinetic graph.
        pathways = document["model"]["pathways"]
        pathways["uptake_constraints"] = [{"identifier": "nutrient_uptake",
            "type": "provider", "species": "nutrient", "output": "capacity"}]
        pathways["rule_graph"] = [{"identifier": "capacity", "expression": 1.0}]  # [mol/(kgDW h)]
        document["rate_provider"] = {"type": "rule-graph",
            "output_units": {"capacity": "mol/(kgDW h)"}}
        first_rule = "capacity"
    else:
        first_rule = document["model"]["kinetics"]["kinetic_outputs"]["growth"]
    graph = document["model"]["pathways" if folder == "generic_batch" else "kinetics"]["rule_graph"]
    selected_rule = next(rule for rule in graph if rule["identifier"] == first_rule)
    # Replacing an existing rate expression with live T/300 tests the actual
    # provider snapshot; the quantity retains the original rule's declared unit.
    selected_rule["expression"] = {"op": "divide", "args": [{"state": "temperature_k"}, INITIAL_TEMPERATURE]}
    document["numerical"].update(backend="scipy", step={"unit": "s", "value": 1.0},
                                  relative_tolerance=RTOL, absolute_tolerance=ATOL)
    assembly = build_bioreactor(document, thermo_path=path)
    unit = assembly.unit
    unit.compile_integrator(verbose=False)
    states = unit.create_solver_init_states()
    derivatives = unit.unit_model(0.0, states)
    slices = unit.solver_state_collection.slices
    phase_key = next(key for key in slices if key.name == "mass_j")
    source = derivatives[slices[phase_key]][assembly.phase.name_species.index(species)]  # [kg/s]
    assert source < 0.0
    expected = REACTION_ENTHALPY * source / (assembly.phase.mass * CP)  # [K/s]
    # [K/s] The nonzero handoff is an algebraic identity; a state-unit
    # absolute tolerance must not mask a wrong small metabolic heat rate.
    np.testing.assert_allclose(derivatives[slices[StateKey("global_temp")]], [expected], rtol=1e-10, atol=0.0)
    result = assembly.mechanism.last_provider_result
    if folder == "generic_batch":
        result = result.diagnostics["uptake_provider"]
    assert result.rates[first_rule] == pytest.approx(1.0, abs=ATOL)
    states[slices[StateKey("global_temp")]] = 330.0  # [K]; asymmetric live-temperature perturbation
    unit.unit_model(0.0, states)
    result = assembly.mechanism.last_provider_result
    if folder == "generic_batch":
        result = result.diagnostics["uptake_provider"]
    assert result.rates[first_rule] == pytest.approx(1.1, abs=ATOL)


def test_temperature_dependent_cp_bolus_uses_integral_not_endpoint_cp(thermal_case):
    document, path = thermal_case
    properties = json.loads(path.read_text())
    # Cp(T)=T J/(kg K), shared by all species, is a positive synthetic
    # polynomial. Its primitive T^2/2 gives an independent mixing solution.
    for entry in properties.values():
        entry["cp_liq"] = [0.0, entry["mw"] / 1000.0]  # [J/(mol K)], [J/(mol K2)]
    path.write_text(json.dumps(properties))
    document["operation"].update(mode="fed-batch")
    document["operation"]["thermal"]["heat_generation_w"] = 0.0
    document["recipes"]["continuous"] = [{"time": 50.0, "event_type": "feed", "volume_l": 0.5,
        "concentrations_mmol_l": {}, "composition": "nutrient-free", "temperature_k": 270.0}]
    histories = build_bioreactor(document, thermo_path=path).solve()
    expected = np.sqrt((300.0**2 + 0.5 * 270.0**2) / 1.5)  # [K]; conserved sum m*T^2/2
    np.testing.assert_allclose(histories[1].global_temp, expected, rtol=0.0, atol=1e-8)


def test_fixed_step_thermal_solution_converges_under_step_refinement(thermal_case):
    document, path = thermal_case
    errors = []
    for step in (5.0, 2.5, 1.25):  # [s]; repeated halving for explicit first-order propagation
        document["numerical"] = {"backend": "fixed-step", "step": {"unit": "s", "value": step},
            "relative_tolerance": None, "absolute_tolerance": None, "maximum_step": None,
            "jacobian_relative_step": None, "jacobian_state_scale": None}
        result = build_bioreactor(document, thermo_path=path).solve()[0]
        expected_final = INITIAL_TEMPERATURE + 2.0 * (1 - np.exp(-1.0))  # [K]; one reaction timescale, 2 K ultimate rise
        errors.append(abs(float(np.asarray(result.global_temp).reshape(-1)[-1]) - expected_final))  # [K]
    assert errors[2] < errors[1] < errors[0]
    np.testing.assert_allclose(np.array(errors[:-1]) / errors[1:], 2.0, rtol=0.05)  # [-]; observed first-order convergence, 5% finite-step allowance


@pytest.mark.parametrize("temperature", [True, 0.0, -1.0, float("nan"), float("inf")])
def test_invalid_feed_temperatures_fail_at_construction(thermal_case, temperature):
    document, path = thermal_case
    document["recipes"]["continuous"] = [{"time": 0.0, "event_type": "flow", "volume_l_h": 1.0,
                                           "temperature_k": temperature}]
    with pytest.raises(ValueError, match="temperature_k must be finite and positive"):
        build_bioreactor(document, thermo_path=path)


def test_null_feed_temperature_retains_existing_blank_slot_semantics(thermal_case):
    document, path = thermal_case
    document["operation"]["mode"] = "fed-batch"
    document["operation"]["thermal"]["heat_generation_w"] = 0.0
    document["recipes"]["continuous"] = [{"time": 0.0, "event_type": "flow", "volume_l_h": 3.6,
        "composition": "nutrient-free", "temperature_k": None}]  # [s], [L/h]; null optional fields are omitted by existing normalization
    assembly = build_bioreactor(document, thermo_path=path)
    assert "temperature_k" not in assembly.events[0]
    np.testing.assert_allclose(assembly.solve()[0].global_temp, INITIAL_TEMPERATURE, rtol=0.0, atol=ATOL)


def test_species_reordering_preserves_thermal_and_material_values(thermal_case):
    document, path = thermal_case
    expected = build_bioreactor(document, thermo_path=path)
    reference = expected.solve()[0]
    properties = json.loads(path.read_text())
    path.write_text(json.dumps(dict(reversed(list(properties.items())))))
    reordered = build_bioreactor(document, thermo_path=path)
    result = reordered.solve()[0]
    assert reordered.phase.name_species == list(reversed(expected.phase.name_species))
    for name in expected.phase.name_species:
        np.testing.assert_allclose(result.mass_j_liquid0[:, reordered.phase.name_species.index(name)],
            reference.mass_j_liquid0[:, expected.phase.name_species.index(name)], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(result.global_temp, reference.global_temp, rtol=0.0, atol=ATOL)


def test_temperature_predictions_retain_field_time_order_and_trials(thermal_case):
    document, path = thermal_case
    document["operation"]["thermal"]["heat_generation_w"]["args"][0] = {"parameter": "coefficients.enthalpy"}
    document["parameters"]["coefficients"]["enthalpy"] = REACTION_ENTHALPY  # [J/kg substrate]
    original = deepcopy(document)
    declaration = {"schema_version": 1, "time_unit": "s", "fields": {
        "temperature": {"species": None, "quantity": "temperature", "unit": "K", "event_side": "post", "time_key": "t", "value_key": "v"},
        "substrate": {"species": "substrate", "quantity": "concentration", "unit": "g/L", "event_side": "post", "time_key": "t", "value_key": "v"}}}
    times = np.array([100.0, 0.0, 50.0, 50.0])  # [s]; unsorted and repeated, with unequal row/column dimensions
    for enthalpy in (REACTION_ENTHALPY, 0.0, REACTION_ENTHALPY):  # [J/kg]; changed trial then repeat
        trial = deepcopy(document)
        trial["parameters"]["coefficients"]["enthalpy"] = enthalpy  # [J/kg substrate]
        assembly = build_bioreactor(trial, thermo_path=path)
        histories = assembly.solve(report_times=np.unique(times))
        observations = read_observations(declaration, {
            name: {"t": times, "v": np.zeros_like(times)} for name in declaration["fields"]})
        prediction = predict_measurements(assembly, histories, declaration, observations)
        actual = np.column_stack(list(prediction["values"].values()))  # [K, g/L] columns in declared order
        substrate = 10.0 * np.exp(-RATE_CONSTANT * times)  # [g/L]; 10 g initially in 1 L
        temperature = INITIAL_TEMPERATURE - enthalpy * (0.01 - substrate / 1000.0) / CP  # [K]
        assert actual.shape == (4, 2)
        # Separate absolute budgets prevent a 300 K offset from masking
        # temperature error or mixing concentration and temperature scales.
        np.testing.assert_allclose(actual[:, 0], temperature, rtol=0.0, atol=1e-6)  # [K]
        np.testing.assert_allclose(actual[:, 1], substrate, rtol=0.0, atol=1e-6)  # [g/L]
    assert document == original


@pytest.mark.parametrize("mode", ["batch", "fed-batch", "continuous"])
@pytest.mark.parametrize("thermal_mode", ["isothermal", "adiabatic", "heat-transfer", "prescribed-temperature"])
def test_json_adapter_matches_direct_native_reaction_process(thermal_case, mode, thermal_mode):
    document, path = thermal_case
    document["operation"]["mode"] = mode
    heat = document["operation"]["thermal"]["heat_generation_w"]
    if thermal_mode == "isothermal":
        document["operation"]["thermal"] = {"mode": "isothermal"}
    elif thermal_mode == "prescribed-temperature":
        document["operation"]["thermal"] = {"mode": thermal_mode,
            "temperature_profile": {"temp_zero": 300.0, "rate": 0.02}}  # [K], [K/s]
    elif thermal_mode == "heat-transfer":
        document["operation"]["thermal"] = {"mode": thermal_mode, "heat_generation_w": heat,
            "heat_transfer": {"h_conv": 100.0, "diam": 0.1, "area_base": 0.0,
                              "utility": {"temp_in": 290.0, "mass_flow": 0.1, "h_conv": 100.0}}}
    if mode != "batch":
        document["recipes"]["continuous"] = [{"time": 0.0, "event_type": "flow", "volume_l_h": 3.6,
            "composition": "nutrient-free", "temperature_k": 280.0}]
    native = _native_reaction_reference(document, path)
    assembly = build_bioreactor(document, thermo_path=path)
    native.solve_unit(time_grid=assembly.time_grid, verbose=False)
    result = assembly.solve()[0]
    np.testing.assert_allclose(result.mass_j_liquid0, native.result.mass_j_liquid0, rtol=1e-7, atol=1e-10)
    np.testing.assert_allclose(result.global_temp, native.result.global_temp, rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(result.vessel_vol, native.result.vessel_vol, rtol=1e-7, atol=1e-12)
    if thermal_mode == "heat-transfer":
        np.testing.assert_allclose(result.q_ht, native.result.q_ht, rtol=1e-7, atol=1e-6)
    if thermal_mode == "prescribed-temperature":
        assert type(assembly.unit.controller) is type(native.controller)
        assert not assembly.unit.has_energy_balance
        np.testing.assert_allclose(result.global_temp, 300.0 + 0.02 * result.time, rtol=0.0, atol=ATOL)


def test_prescribed_temperature_does_not_require_unused_thermal_properties(thermal_case):
    document, path = thermal_case
    document["operation"].update(mode="fed-batch", thermal={"mode": "prescribed-temperature",
        "temperature_profile": {"temp_zero": 300.0, "rate": 0.02}})
    properties = json.loads(path.read_text())
    for entry in properties.values():
        entry["cp_liq"] = []
    path.write_text(json.dumps(properties))
    document["recipes"]["continuous"] = [{"time": 50.0, "event_type": "feed", "volume_l": 0.5,
        "concentrations_mmol_l": {}, "composition": "nutrient-free", "temperature_k": 270.0}]
    assembly = build_bioreactor(document, thermo_path=path)
    for result in assembly.solve():
        np.testing.assert_allclose(result.global_temp, 300.0 + 0.02 * result.time, rtol=0.0, atol=ATOL)
    assembly.reset()
    for result in assembly.solve():
        np.testing.assert_allclose(result.global_temp, 300.0 + 0.02 * result.time, rtol=0.0, atol=ATOL)


@pytest.mark.parametrize("profile,match", [
    ({"temp_zero": 301.0, "rate": 0.02}, "must match"),
    ({"temp_zero": 300.0, "rate": -4.0}, "finite and positive"),
    ({"temp_zero": 300.0, "rate": float("nan")}, "finite real"),
    ({"temp_zero": 300.0}, "requires native"),
])
def test_invalid_prescribed_temperature_configuration_is_rejected(thermal_case, profile, match):
    document, path = thermal_case
    document["operation"]["thermal"] = {"mode": "prescribed-temperature", "temperature_profile": profile}
    with pytest.raises(ValueError, match=match):
        build_bioreactor(document, thermo_path=path)


def test_kirchhoff_heat_and_variable_mixture_capacity_match_native_and_energy_invariant(thermal_case):
    document, path = thermal_case
    properties = json.loads(path.read_text())
    properties["product"]["cp_liq"] = [2000.0]  # [J/(mol K)]; formal MW=1 kg/mol, Cp_product-Cp_substrate=1000 J/(kg K)
    path.write_text(json.dumps(properties))
    document["operation"]["thermal"]["heat_generation_w"] = {"op": "multiply", "args": [
        {"source_rate": "substrate"}, {"op": "add", "args": [REACTION_ENTHALPY,
            {"op": "multiply", "args": [1000.0, {"op": "subtract", "args": [
                {"state": "temperature_k"}, INITIAL_TEMPERATURE]}]}]}]}
    assembly = build_bioreactor(document, thermo_path=path)
    native = _native_reaction_reference(document, path)
    native.solve_unit(time_grid=assembly.time_grid, verbose=False)
    result = assembly.solve()[0]
    extent = 0.01 * (1.0 - np.exp(-RATE_CONSTANT * result.time))  # [kg converted]; independent first-order solution
    capacity = CP + 1000.0 * extent  # [J/K]; 1 kg mixture, heavier-Cp product replaces substrate
    expected = INITIAL_TEMPERATURE - REACTION_ENTHALPY * extent / capacity  # [K]; conserved chemical+sensible enthalpy
    np.testing.assert_allclose(result.global_temp, expected, rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(result.global_temp, native.result.global_temp, rtol=0.0, atol=1e-6)


def test_native_utility_volume_and_mass_flow_configurations_are_equivalent(thermal_case):
    document, path = thermal_case
    document["operation"]["thermal"] = {"mode": "heat-transfer", "heat_generation_w": 0.0,
        "heat_transfer": {"h_conv": 100.0, "diam": 0.1, "ht_mode": "bath",
                          "utility": {"temp_in": 290.0, "mass_flow": 0.1, "h_conv": 100.0}}}
    mass_flow = build_bioreactor(document, thermo_path=path)
    reference = mass_flow.solve()[0]
    utility = document["operation"]["thermal"]["heat_transfer"]["utility"]
    utility.pop("mass_flow")
    utility["vol_flow"] = 1e-4  # [m3/s]; 0.1 kg/s divided by native water density 1000 kg/m3
    volume_flow = build_bioreactor(document, thermo_path=path)
    actual = volume_flow.solve()[0]
    np.testing.assert_allclose(actual.global_temp, reference.global_temp, rtol=0.0, atol=ATOL)
    np.testing.assert_allclose(actual.q_ht, reference.q_ht, rtol=RTOL, atol=ATOL)
    assert volume_flow.unit.ht_mode == "bath"
    assert volume_flow.unit.area_base == 0.0  # [m2]; unchanged native constructor default


@pytest.mark.parametrize("change,match", [
    ({"ht_mode": "coil"}, "not implemented"),
    ({"h_conv": -1.0}, "finite and positive"),
    ({"diam": True}, "finite real"),
    ({"utility": {"temp_in": 290.0, "mass_flow": 0.1, "vol_flow": 1e-4, "h_conv": 100.0}}, "exactly one"),
])
def test_invalid_native_thermal_settings_fail_before_solving(thermal_case, change, match):
    document, path = thermal_case
    transfer = {"h_conv": 100.0, "diam": 0.1,
                "utility": {"temp_in": 290.0, "mass_flow": 0.1, "h_conv": 100.0}}
    transfer.update(change)
    document["operation"]["thermal"] = {"mode": "heat-transfer", "heat_generation_w": 0.0,
                                          "heat_transfer": transfer}
    with pytest.raises(ValueError, match=match):
        build_bioreactor(document, thermo_path=path)


def test_native_bolus_mixer_runs_without_assimulo(thermal_case):
    """Block the optional backend only inside the isolated child process."""
    document, path = thermal_case
    document["operation"].update(mode="fed-batch")
    document["operation"]["thermal"]["heat_generation_w"] = 0.0
    document["recipes"]["continuous"] = [{"time": 50.0, "event_type": "feed", "volume_l": 0.5,
        "concentrations_mmol_l": {}, "composition": "nutrient-free", "temperature_k": 270.0}]
    script = '''
import sys
class BlockAssimulo:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "assimulo" or fullname.startswith("assimulo."):
            raise ModuleNotFoundError("backend deliberately absent", name="assimulo")
sys.meta_path.insert(0, BlockAssimulo())
import json
from PharmaPy.Bioreactors import build_bioreactor
assembly = build_bioreactor(json.loads(sys.argv[1]), thermo_path=sys.argv[2])
result = assembly.solve()
assert not any(name == "assimulo" or name.startswith("assimulo.") for name in sys.modules)
print(json.dumps(float(result[-1].global_temp[-1])))
'''
    result = subprocess.run([sys.executable, "-c", script, json.dumps(document), str(path)],
                            cwd=EXAMPLES.parents[1], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == pytest.approx(290.0, abs=1e-8)  # [K]; independent equal-Cp mixing solution


def test_stock_addition_with_unequal_species_cp_and_sampling_conserves_sensible_heat(thermal_case):
    """Isolate physical mixing: zero reaction heat is an explicit test assumption."""
    document, path = thermal_case
    properties = json.loads(path.read_text())
    properties["substrate"]["cp_liq"] = [2000.0]  # [J/(mol K)]; formal MW=1 kg/mol, twice the carrier's mass-specific Cp
    path.write_text(json.dumps(properties))
    document["operation"]["mode"] = "fed-batch"
    document["operation"]["thermal"]["heat_generation_w"] = 0.0
    document["recipes"]["continuous"] = [
        {"time": 0.0, "event_type": "stock_to_target", "target_species": "substrate",
         "target_concentration_mmol_l": 20.0, "concentrations_mmol_l": {"substrate": 100.0},
         "composition": "supplied", "temperature_k": 280.0},
        {"time": 50.0, "event_type": "sample", "volume_l": 0.125}]
    # [L] Stock volume=(20*1-10)/(100-20)=0.125; it contains
    # 0.0125 kg substrate and 0.1125 kg water. Initial heat capacity is
    # 0.01*2000+0.99*1000=1010 J/K; added capacity is 137.5 J/K.
    expected_temperature = (1010.0 * 300.0 + 137.5 * 280.0) / 1147.5  # [K]
    assembly = build_bioreactor(document, thermo_path=path)
    histories = assembly.solve()
    for result in histories:
        np.testing.assert_allclose(result.global_temp, expected_temperature, rtol=0.0, atol=1e-8)
    substrate = assembly.phase.name_species.index("substrate")
    assert histories[0].mass_j_liquid0[0, substrate] == pytest.approx(0.0225, abs=ATOL)  # [kg]
    assert histories[0].vessel_vol[0] == pytest.approx(0.001125, abs=ATOL)  # [m3]
    # [kg] Sampling removes one ninth of the well-mixed mass, preserving T.
    np.testing.assert_allclose(histories[1].mass_j_liquid0[0], histories[0].mass_j_liquid0[-1] * 8.0 / 9.0,
                               rtol=RTOL, atol=ATOL)


def test_nonpositive_live_heat_capacity_is_rejected_by_native_public_rhs(thermal_case):
    document, path = thermal_case
    properties = json.loads(path.read_text())
    for entry in properties.values():
        # Cp(T)=301-T J/(kg K): positive at 300 K, negative at 302 K.
        # This deliberately invalid extrapolation tests the thermal boundary.
        entry["cp_liq"] = [301.0 * entry["mw"] / 1000.0, -entry["mw"] / 1000.0]  # [J/(mol K)], [J/(mol K2)]
    path.write_text(json.dumps(properties))
    assembly = build_bioreactor(document, thermo_path=path)
    assembly.unit.compile_integrator(verbose=False)
    states = assembly.unit.create_solver_init_states()
    states[assembly.unit.solver_state_collection.slices[StateKey("global_temp")]] = 302.0  # [K]
    with pytest.raises(ValueError, match="heat capacity must be finite and positive"):
        assembly.unit.unit_model(0.0, states)


def test_prescribed_temperature_drives_biological_rates_in_seconds_with_hour_runtime(thermal_case):
    document, path = thermal_case
    slope = 0.02  # [K/s]; synthetic linear temperature response
    document["operation"]["thermal"] = {"mode": "prescribed-temperature",
        "temperature_profile": {"temp_zero": 300.2, "rate": slope, "t_zero": 10.0}}
    document["operation"]["runtime"] = {"unit": "h", "value": 100.0 / 3600.0}  # [h]; exact unit conversion of the 100 s scenario
    document["numerical"]["step"] = {"unit": "h", "value": 5.0 / 3600.0}  # [h]; 5 s reports
    graph = document["model"]["kinetics"]["rule_graph"]
    graph[0]["expression"] = {"op": "multiply", "args": [graph[0]["expression"],
        {"op": "divide", "args": [{"state": "temperature_k"}, INITIAL_TEMPERATURE]}]}
    assembly = build_bioreactor(document, thermo_path=path)
    result = assembly.solve()[0]
    substrate = assembly.phase.name_species.index("substrate")
    # Integral of k0*T(t)/T0 is k0*(t+slope*t^2/(2*T0)).
    expected = 0.01 * np.exp(-RATE_CONSTANT * (result.time + slope * result.time**2 / (2.0 * INITIAL_TEMPERATURE)))  # [kg]
    np.testing.assert_allclose(result.mass_j_liquid0[:, substrate], expected, rtol=1e-6, atol=ATOL)
    np.testing.assert_allclose(result.global_temp, INITIAL_TEMPERATURE + slope * result.time, rtol=0.0, atol=ATOL)


@pytest.mark.parametrize("folder", ["generic_batch", "ecoli_ye_fed_batch/reconciliation"])
def test_legacy_and_combined_thermal_feeds_reenter_without_losing_temperature(tmp_path, folder):
    inputs = EXAMPLES / folder / "inputs"
    document = json.loads((inputs / "mechanism.json").read_text())
    thermo_path = inputs / "thermo.json"
    if not thermo_path.exists():
        thermo_path = EXAMPLES / "ecoli_ye_fed_batch/inputs/thermo.json"
    properties = json.loads(thermo_path.read_text())
    for entry in properties.values():
        entry["cp_liq"] = [CP * entry["mw"] / 1000.0]  # [J/(mol K)]
    path = tmp_path / "thermo.json"
    path.write_text(json.dumps(properties))
    document["operation"].update(mode="fed-batch", temperature_k=300.0,
        runtime={"unit": "s", "value": 1.0}, thermal={"mode": "adiabatic", "heat_generation_w": 0.0})
    document["numerical"].update(backend="scipy", step={"unit": "s", "value": 1.0},
        relative_tolerance=RTOL, absolute_tolerance=ATOL)
    # Legacy templates require inactive event slots explicitly null.
    event = dict.fromkeys(EVENT_FIELDS)
    event.update(time=0.0, event_type="feed", volume_l=0.5, concentrations_mmol_l={},
                 temperature_k=270.0, composition="nutrient-free")  # [s], [L], [K]; equal-Cp 1 kg + 0.5 kg mixing
    document["recipes"][document["operation"]["recipe"]["name"]] = [event]
    case, definition = split_forward_inputs(document)
    normalized_case, normalized_definition = normalize_forward_inputs(case, definition)
    assemblies = [build_bioreactor(document, thermo_path=path),
                  build_bioreactor(case, definition, path),
                  build_bioreactor(normalized_case, normalized_definition, path)]
    reference = assemblies[0].solve()[0]
    assert reference.global_temp[0] == pytest.approx(290.0, abs=1e-8)  # [K]
    for assembly in assemblies[1:]:
        result = assembly.solve()[0]
        np.testing.assert_array_equal(result.global_temp, reference.global_temp)
        np.testing.assert_array_equal(result.mass_j_liquid0, reference.mass_j_liquid0)
