"""Inlet phase/temperature pairing through real extraction thermodynamics.

Synthetic constant-property liquids give an independent linear energy balance.
Core cases exercise initialization and the public DAE residual without a solver;
the Assimulo cases compare ``solve_unit`` with independent energy and material
matrix-exponential solutions under default algebraic-state suppression.
These fixtures establish a routing contract, not calibrated LLE predictions.
"""

import json

import numpy as np
import pytest
from scipy.linalg import expm

from PharmaPy.DynamicExtraction import DynamicExtractor, get_alg_map
from PharmaPy.Phases import LiquidPhase
from PharmaPy.Streams import LiquidStream


# Equal molecular weights make mass and molar density ordering agree. Unequal
# densities and heat capacities distinguish the two phase roles and enthalpies.
SPECIES = {
    "a": {"mw": 100.0, "rho_liq": 600.0, "cp_liq": [80.0], "p_vap": [8.0, 1500.0, -40.0]},
    "b": {"mw": 100.0, "rho_liq": 1200.0, "cp_liq": [120.0], "p_vap": [8.0, 1500.0, -40.0]},
    "c": {"mw": 100.0, "rho_liq": 900.0, "cp_liq": [200.0], "p_vap": [8.0, 1500.0, -40.0]},
}  # mw [g/mol], rho_liq [kg/m**3], cp_liq [J/mol/K]; synthetic constants
# Antoine A [-], B [K], C [K] are constructor data, unused by this liquid-only
# constant-K model; the synthetic correlation is finite at 290--330 K.
LIGHT_COMPOSITION = np.array([0.4, 0.2, 0.4])  # [-], light equilibrium phase
HEAVY_COMPOSITION = np.array([0.2, 0.4, 0.4])  # [-], heavy equilibrium phase
LIGHT_CP = 136.0  # [J/mol/K], 0.4*80 + 0.2*120 + 0.4*200
HEAVY_CP = 144.0  # [J/mol/K], 0.2*80 + 0.4*120 + 0.4*200
PHASE_HOLDUP = 10.0  # [mol], equal phase inventories in the batch seed
LIGHT_FLOW = 2.0  # [mol/s], asymmetric fixed inlet flows
HEAVY_FLOW = 3.0  # [mol/s]
INITIAL_TEMPERATURE = 300.0  # [K], between the unequal inlet temperatures
ENTHALPY_REFERENCE = 298.15  # [K], LiquidPhase.getEnthalpy reference
ROUND_OFF_RTOL = 1e-10  # [-], allows rounding in the batch/root initialization
BALANCE_ATOL = 1e-8  # [J/s] or [J], roundoff allowance for kJ-scale balances


def distribution_coefficients(x_light, x_heavy, temp):
    """Return the prescribed equilibrium ratio of the synthetic phases.

    Parameters
    ----------
    x_light, x_heavy : ndarray
        Light/heavy mole fractions [-], component vector or stage matrix.
    temp : float or ndarray
        Temperature [K]; coefficients are constant in this test case.

    Returns
    -------
    ndarray
        Component coefficients [-], ``[0.2, 0.4, 0.4]/[0.4, 0.2, 0.4]``.
    """
    return np.array([0.5, 2.0, 1.0])  # [-]


@pytest.fixture
def extractor_factory(tmp_path):
    """Construct initialized two-stage extractors with either inlet ordering.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary directory for the synthetic thermodynamic database.

    Returns
    -------
    callable
        Factory taking a feed-heavy flag and feed/solvent temperatures [K],
        returning a real extractor and its initialized state dictionary.
    """
    path = tmp_path / "extractor_temperatures.json"
    path.write_text(json.dumps(SPECIES))

    def build(feed_is_heavy, feed_temperature, solvent_temperature):
        """Build real phases and let initialization determine their roles.

        Parameters
        ----------
        feed_is_heavy : bool
            Select whether feed carries the denser equilibrium composition.
        feed_temperature, solvent_temperature : float
            Inlet temperatures [K].

        Returns
        -------
        tuple
            Extractor and initialized mole fractions [-], energies [J], and
            temperatures [K].
        """
        extractor = DynamicExtractor(
            num_stages=2, k_fun=distribution_coefficients, gamma_model="ideal")
        extractor.Phases = LiquidPhase(
            str(path), mole_frac=(LIGHT_COMPOSITION + HEAVY_COMPOSITION) / 2,
            moles=2 * PHASE_HOLDUP, temp=INITIAL_TEMPERATURE, verbose=False)
        light_name = "solvent" if feed_is_heavy else "feed"
        heavy_name = "feed" if feed_is_heavy else "solvent"
        temperatures = {
            "feed": feed_temperature, "solvent": solvent_temperature,
        }  # [K]
        extractor.Inlet = {
            light_name: LiquidStream(
                str(path), mole_frac=LIGHT_COMPOSITION, mole_flow=LIGHT_FLOW,
                temp=temperatures[light_name], verbose=False),
            heavy_name: LiquidStream(
                str(path), mole_frac=HEAVY_COMPOSITION, mole_flow=HEAVY_FLOW,
                temp=temperatures[heavy_name], verbose=False),
        }
        initial = extractor.initialize_model()
        assert extractor.target_states["light_phase"] == light_name
        assert extractor.target_states["heavy_phase"] == heavy_name
        assert (extractor.Inlet[light_name].getDensity()
                < extractor.Inlet[heavy_name].getDensity())
        np.testing.assert_allclose(
            initial["x_i"], np.tile(LIGHT_COMPOSITION, (2, 1)),
            rtol=ROUND_OFF_RTOL)
        np.testing.assert_allclose(
            initial["y_i"], np.tile(HEAVY_COMPOSITION, (2, 1)),
            rtol=ROUND_OFF_RTOL)
        np.testing.assert_allclose(
            [extractor.fixed_vals["H_R"], extractor.fixed_vals["H_E"]],
            PHASE_HOLDUP, rtol=ROUND_OFF_RTOL)
        return extractor, initial

    return build


@pytest.mark.unit
@pytest.mark.parametrize(
    "feed_is_heavy,feed_temperature,solvent_temperature,expected_power",
    [
        (True, 330.0, 290.0, [1600.0, 5920.0]),
        (False, 330.0, 290.0, [12480.0, -11360.0]),
        (True, 315.0, 315.0, [8400.0, -560.0]),
        (False, 315.0, 315.0, [8400.0, -560.0]),
    ],
    ids=["heavy_feed", "light_feed", "equal_heavy_feed", "equal_light_feed"],
)
def test_residual_pairs_inlet_enthalpies_with_phase_roles(
        extractor_factory, feed_is_heavy, feed_temperature,
        solvent_temperature, expected_power):
    """Check both stage energy rates against independent heat-flow arithmetic.

    Parameters
    ----------
    extractor_factory : callable
        Factory for a real initialized extractor.
    feed_is_heavy : bool
        Select the feed density ordering.
    feed_temperature, solvent_temperature : float
        Inlet temperatures [K].
    expected_power : list of float
        Stage energy rates [J/s]. With stages at 300 and 310 K, light and heavy
        heat-flow coefficients are 2*136=272 and 3*144=432 J/s/K. For example,
        heavy feed gives 272*(-10)+432*10=1600 J/s in stage one and
        272*(-10)+432*20=5920 J/s in stage two. Equal inlet temperatures also
        test the case where the original routing defect is unobservable.
    """
    extractor, initial = extractor_factory(
        feed_is_heavy, feed_temperature, solvent_temperature)
    initial["temp"] = np.array([300.0, 310.0])  # [K], nonuniform stage profile
    stage_heat_capacity = PHASE_HOLDUP * (LIGHT_CP + HEAVY_CP)  # [J/K]
    initial["u_int"] = stage_heat_capacity * (
        initial["temp"] - ENTHALPY_REFERENCE)  # [J]
    states = np.column_stack([
        initial[name] for name in extractor.name_states
    ]).ravel()  # x_i/y_i [-], u_int [J], temp [K] per stage

    residual = extractor.unit_model(0.0, states).reshape(2, -1)
    # Residual columns: x_i [1/s or -], y_i [-], u_int [J/s], temp [J].
    np.testing.assert_allclose(
        residual[:, -2], expected_power,
        rtol=ROUND_OFF_RTOL, atol=BALANCE_ATOL)
    np.testing.assert_allclose(residual[:, -1], 0.0, rtol=0, atol=BALANCE_ATOL)


@pytest.mark.unit
@pytest.mark.parametrize(
    "state_order,expected_flags",
    [
        (["composition", "temperature", "energy"],
         [1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1]),
        (["energy", "temperature", "composition"],
         [1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1]),
    ],
)
def test_algebraic_map_preserves_stage_and_field_order(state_order, expected_flags):
    """Use unequal field sizes and a content-preserving field permutation.

    Parameters
    ----------
    state_order : list of str
        Metadata insertion order used by the flattened state vector.
    expected_flags : list of int
        Three repeated stage blocks of differential (1) and algebraic (0)
        flags [-], enumerated independently for each field order.
    """
    metadata = {
        "composition": {"dim": 2, "type": "diff"},
        "temperature": {"dim": 1, "type": "alg"},
        "energy": {"dim": 1, "type": "diff"},
    }  # dimensions and flags [-]; two composition entries distinguish widths
    ordered = {name: metadata[name] for name in state_order}
    np.testing.assert_array_equal(get_alg_map(ordered, nstages=3), expected_flags)
    np.testing.assert_array_equal(get_alg_map(ordered), expected_flags[:4])


@pytest.mark.unit
@pytest.mark.parametrize("feed_is_heavy", [True, False])
def test_solver_flags_match_actual_residual_derivatives(extractor_factory, feed_is_heavy):
    """Check IDA's flags against derivative dependence of the real residual.

    Parameters
    ----------
    extractor_factory : callable
        Factory for a real initialized extractor.
    feed_is_heavy : bool
        Select the feed density ordering.

    Notes
    -----
    Per stage, only the first two light-phase mole fractions and energy have
    derivatives. The dependent third mole fraction, all heavy-phase entries,
    and temperature are algebraic. A unit derivative probe must therefore
    subtract one only in the three differential residual rows.
    """
    extractor, initial = extractor_factory(feed_is_heavy, 330.0, 290.0)
    # Inlet temperatures [K] deliberately differ to retain nonzero energy rates.
    states = np.column_stack([
        initial[name] for name in extractor.name_states
    ]).ravel()  # x_i/y_i [-], u_int [J], temp [K] per stage
    expected_flags = np.array([
        1, 1, 0, 0, 0, 0, 1, 0,
        1, 1, 0, 0, 0, 0, 1, 0,
    ])  # [-], stage-major order for two stages and three components
    np.testing.assert_array_equal(extractor.alg_map, expected_flags)
    zero_derivative = np.zeros_like(states)  # x_i/y_i [1/s], u_int [J/s], temp [K/s]
    probe_derivative = np.ones_like(states)  # unit probes in the same state-rate units
    baseline = extractor.unit_model(0.0, states, zero_derivative)
    probed = extractor.unit_model(0.0, states, probe_derivative)
    # Residual differences [1/s] or [J/s]; algebraic rows stay exactly unchanged.
    np.testing.assert_allclose(
        baseline - probed, expected_flags, rtol=0, atol=BALANCE_ATOL)


@pytest.mark.assimulo
@pytest.mark.integration
@pytest.mark.parametrize("feed_is_heavy", [True, False])
@pytest.mark.parametrize("solver_tolerance", [None, 1e-9], ids=["defaults", "refined"])
def test_solve_matches_independent_constant_property_energy_balance(
        extractor_factory, feed_is_heavy, solver_tolerance):
    """Compare IDA temperatures with the linear two-stage energy solution.

    Parameters
    ----------
    extractor_factory : callable
        Factory for a real initialized extractor.
    feed_is_heavy : bool
        Select the feed density ordering.
    solver_tolerance : float or None
        IDA relative tolerance [-] and absolute tolerance [state units], or
        None to use the solver defaults. The refined case tightens accuracy.

    Notes
    -----
    Both inlets already have their respective equilibrium compositions, so
    component inventories stay fixed. Each stage has heat capacity 2800 J/K.
    Its inlet/outlet heat-flow coefficients are 272 and 432 J/s/K. The exact
    solution is T(t)=T_ss+exp(-A*t/C)*(T(0)-T_ss), where A has diagonal 704
    J/s/K and countercurrent off-diagonal entries -432 and -272 J/s/K.

    Both cases retain the default suppression of algebraic states in error
    control. Differential energy and independent compositions must still be
    controlled; the algebraic temperature follows the energy constraint.
    """
    pytest.importorskip("assimulo")
    feed_temperature = 330.0  # [K], warm feed
    solvent_temperature = 290.0  # [K], cool solvent
    extractor, _ = extractor_factory(
        feed_is_heavy, feed_temperature, solvent_temperature)
    runtime = 2.0  # [s], about half the C/(272+432) thermal time scale
    solver_options = (None if solver_tolerance is None else
                      {"rtol": solver_tolerance, "atol": solver_tolerance})
    # rtol [-], atol [state units]; no override of algebraic error suppression.
    time, states = extractor.solve_unit(
        runtime, sundials_opts=solver_options,
        verbose=False)  # time [s]; state columns: x_i/y_i [-], u_int [J], temp [K]
    light_temperature, heavy_temperature = (
        (solvent_temperature, feed_temperature) if feed_is_heavy
        else (feed_temperature, solvent_temperature)
    )  # [K]
    heat_flow_matrix = np.array([[704.0, -432.0], [-272.0, 704.0]])  # [J/s/K]
    stage_heat_capacity = 2800.0  # [J/K], 10*(136+144)
    inlet_power = np.array([
        272.0 * light_temperature, 432.0 * heavy_temperature,
    ])  # [J/s], constant forcing in the absolute-temperature equation
    steady_temperature = np.linalg.solve(heat_flow_matrix, inlet_power)  # [K]
    expected_temperature = np.array([
        steady_temperature + expm(-heat_flow_matrix * instant / stage_heat_capacity)
        @ (np.full(2, INITIAL_TEMPERATURE) - steady_temperature)
        for instant in time
    ])  # [K]
    stage_states = states.reshape(len(time), 2, -1)
    # x_i/y_i [-], u_int [J], temp [K]; no iteration count or output-grid pin.
    # Allow 1 mK with default IDA tolerances, and 0.01 mK after refinement;
    # both resolve the approximately 10 K transient without pinning its grid.
    temperature_atol = 1e-3 if solver_tolerance is None else 1e-5  # [K]
    composition_atol = 1e-9  # [-], roundoff allowance for invariant inventories
    np.testing.assert_allclose(
        stage_states[:, :, -1], expected_temperature,
        rtol=0, atol=temperature_atol)
    np.testing.assert_allclose(
        stage_states[:, :, :3], np.broadcast_to(LIGHT_COMPOSITION, (len(time), 2, 3)),
        rtol=0, atol=composition_atol)
    np.testing.assert_allclose(
        stage_states[:, :, 3:6], np.broadcast_to(HEAVY_COMPOSITION, (len(time), 2, 3)),
        rtol=0, atol=composition_atol)


@pytest.mark.assimulo
@pytest.mark.integration
@pytest.mark.parametrize("feed_is_heavy", [True, False])
def test_solve_integrates_independent_compositions(extractor_factory, feed_is_heavy):
    """Check changing material states against independent linear solutions.

    Parameters
    ----------
    extractor_factory : callable
        Factory for a real initialized extractor.
    feed_is_heavy : bool
        Select the feed density ordering.

    Notes
    -----
    For independent component i, y_i=K_i*x_i. With phase inventories 10 mol,
    light flow 2 mol/s and heavy flow 3 mol/s, the two-stage material equation
    has effective inventory 10*(1+K_i) mol, diagonal 2+3*K_i mol/s, upper
    diagonal -3*K_i mol/s, and lower diagonal -2 mol/s. The two independent
    species have K=0.5 and 2, yielding the explicit matrices below.

    The last component follows phase normalization under the existing
    constant-flow approximation (#195), not a separate equilibrium equation.
    This test preserves that closure and does not certify the deferred MESH
    formulation or stage-dependent equilibrium callbacks (#123).
    """
    pytest.importorskip("assimulo")
    extractor, _ = extractor_factory(feed_is_heavy, 300.0, 300.0)
    # Equal inlet temperatures [K] isolate a composition perturbation.
    light_inlet = np.array([0.45, 0.15, 0.4])  # [-], enrich a and deplete b
    heavy_inlet = np.array([0.25, 0.35, 0.4])  # [-], same perturbation direction
    extractor.Inlet[extractor.target_states["light_phase"]].updatePhase(
        mole_frac=light_inlet, mole_flow=LIGHT_FLOW)
    extractor.Inlet[extractor.target_states["heavy_phase"]].updatePhase(
        mole_frac=heavy_inlet, mole_flow=HEAVY_FLOW)
    runtime = 2.0  # [s], resolves the initial composition transient
    solver_tolerance = 1e-10  # rtol [-], atol [state units]; controls accumulated error
    time, states = extractor.solve_unit(
        runtime, sundials_opts={"rtol": solver_tolerance, "atol": solver_tolerance},
        verbose=False)  # time [s], states: x_i/y_i [-], u_int [J], temp [K]
    coefficients = [
        (15.0, np.array([[3.5, -1.5], [-2.0, 3.5]])),
        (30.0, np.array([[8.0, -6.0], [-2.0, 8.0]])),
    ]  # effective inventories [mol], material-flow matrices [mol/s]
    expected_components = []  # [-], one time/stage mole-fraction array per species
    for component, (inventory, flow_matrix) in enumerate(coefficients):
        # inventory [mol], flow_matrix [mol/s]
        forcing = np.array([
            LIGHT_FLOW * light_inlet[component],
            HEAVY_FLOW * heavy_inlet[component],
        ])  # [mol/s], independent inlet component flows
        steady = np.linalg.solve(flow_matrix, forcing)  # [-]
        expected_components.append(np.array([
            steady + expm(-flow_matrix * instant / inventory)
            @ (np.full(2, LIGHT_COMPOSITION[component]) - steady)
            for instant in time
        ]))
    expected_light = np.stack(expected_components, axis=-1)  # [-], time/stage/species
    stage_states = states.reshape(len(time), 2, -1)
    # x_i/y_i [-], u_int [J], temp [K]
    composition_atol = 1e-8  # [-], global accuracy target for both liquid compositions
    np.testing.assert_allclose(
        stage_states[:, :, :2], expected_light, rtol=0, atol=composition_atol)
    np.testing.assert_allclose(
        stage_states[:, :, 3:5], expected_light * np.array([0.5, 2.0]),
        rtol=0, atol=composition_atol)
    np.testing.assert_allclose(
        stage_states[:, :, :3].sum(axis=-1), 1.0, rtol=0, atol=composition_atol)
    np.testing.assert_allclose(
        stage_states[:, :, 3:6].sum(axis=-1), 1.0, rtol=0, atol=composition_atol)
