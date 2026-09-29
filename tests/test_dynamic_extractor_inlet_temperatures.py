"""Inlet phase/temperature pairing through real extraction thermodynamics.

Synthetic constant-property liquids give an independent linear energy balance.
Core cases exercise initialization and the public DAE residual without a solver;
the Assimulo cases compare ``solve_unit`` with its matrix-exponential solution.
These fixtures establish a routing contract, not calibrated LLE predictions.
"""

import json

import numpy as np
import pytest
from scipy.linalg import expm

from PharmaPy.DynamicExtraction import DynamicExtractor
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


@pytest.mark.assimulo
@pytest.mark.integration
@pytest.mark.parametrize("feed_is_heavy", [True, False])
def test_solve_matches_independent_constant_property_energy_balance(
        extractor_factory, feed_is_heavy):
    """Compare IDA temperatures with the linear two-stage energy solution.

    Parameters
    ----------
    extractor_factory : callable
        Factory for a real initialized extractor.
    feed_is_heavy : bool
        Select the feed density ordering.

    Notes
    -----
    Both inlets already have their respective equilibrium compositions, so
    component inventories stay fixed. Each stage has heat capacity 2800 J/K.
    Its inlet/outlet heat-flow coefficients are 272 and 432 J/s/K. The exact
    solution is T(t)=T_ss+exp(-A*t/C)*(T(0)-T_ss), where A has diagonal 704
    J/s/K and countercurrent off-diagonal entries -432 and -272 J/s/K.

    Include algebraic states in error control to verify temperature accuracy.
    The pre-existing multistage alg_map layout is recorded under the audit
    tracker https://github.com/PharmaPy-org/PharmaPy/issues/67; its default
    suppression can omit the energy states from error control. This test
    certifies the explicit all-state error-control setting, not that default.
    """
    pytest.importorskip("assimulo")
    feed_temperature = 330.0  # [K], warm feed
    solvent_temperature = 290.0  # [K], cool solvent
    extractor, _ = extractor_factory(
        feed_is_heavy, feed_temperature, solvent_temperature)
    runtime = 2.0  # [s], about half the C/(272+432) thermal time scale
    solver_rtol = 1e-9  # [-], tighter than the trajectory assertion tolerance
    solver_atol = 1e-9  # state units, absolute IDA tolerance
    time, states = extractor.solve_unit(
        runtime, sundials_opts={"rtol": solver_rtol, "atol": solver_atol,
                                "suppress_alg": False},
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
    temperature_atol = 1e-5  # [K], exceeds IDA's 1e-9 relative state tolerance
    np.testing.assert_allclose(
        stage_states[:, :, -1], expected_temperature,
        rtol=0, atol=temperature_atol)
    np.testing.assert_allclose(
        stage_states[:, :, :3], np.broadcast_to(LIGHT_COMPOSITION, (len(time), 2, 3)),
        rtol=0, atol=solver_atol)
    np.testing.assert_allclose(
        stage_states[:, :, 3:6], np.broadcast_to(HEAVY_COMPOSITION, (len(time), 2, 3)),
        rtol=0, atol=solver_atol)
