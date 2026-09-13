"""Drying gas-balance regressions using real phase and cake collaborators."""

import numpy as np
import pytest


pytestmark = pytest.mark.unit


def test_unit_model_uses_relative_permeability_for_gas_velocity(
        drying_unit_factory):
    """Darcy velocity exposes the clipped relative-permeability result."""
    dryer = drying_unit_factory(number_nodes=3)
    probe_state = dryer.initialize_states(
        deltaP=5.0e4).ravel()  # [-] and [K]
    state_width = 3 + dryer.Liquid_1.num_species + dryer.num_volatiles  # [-]
    states_by_node = probe_state.reshape(dryer.num_nodes, state_width)
    states_by_node[:, 0] = np.array([1.0, dryer.s_inf, 1.2])  # [-]

    dryer.unit_model(0.0, probe_state)

    temp_gas = states_by_node[:, -2]  # [K]
    y_gas = states_by_node[:, 1:1 + dryer.Liquid_1.num_species]  # [-]
    viscosity = dryer.Vapor_1.getViscosity(
        temp=temp_gas, mass_frac=y_gas
    )  # [Pa*s]
    expected_velocity = np.array([
        0.0,
        dryer.k_perm * dryer.dPg_dz / viscosity[1],
        0.0,
    ])  # [m/s]
    np.testing.assert_allclose(dryer.gas_velocity, expected_velocity)


def test_material_balance_uses_single_gas_holdup_factor_for_transfer(
        drying_unit_factory):
    """Gas transfer divides by the real model's gas holdup exactly once."""
    dryer = drying_unit_factory(number_nodes=3)
    dryer.idx_volatiles = np.array([0, 1])  # component indices [-]
    dryer.porosity = 0.5  # [-]
    dryer.rho_liq = np.array([800.0, 900.0, 1000.0])  # [kg/m**3]
    dryer.dz = np.ones(3)  # [m]

    satur = np.array([0.75, 0.25, 0.50])  # [-]
    y_gas = np.zeros((3, 3))  # [-], excludes saturation correction
    dry_rate = np.array([
        [0.2, 0.0, 0.0],
        [0.4, 0.0, 0.0],
        [0.6, 0.0, 0.0],
    ])  # [kg/m**3/s]

    _, dygas_dt, _ = dryer.material_balance(
        time=0.0,
        satur=satur,
        temp_gas=np.array([300.0, 305.0, 310.0]),  # [K]
        temp_sol=np.array([299.0, 304.0, 309.0]),  # [K]
        y_gas=y_gas,
        x_liq=np.array([
            [0.25, 0.75],
            [0.40, 0.60],
            [0.55, 0.45],
        ]),  # [-]
        u_gas=np.zeros(3),  # [m/s]
        dens_gas=np.array([2.0, 4.0, 5.0]),  # [kg/m**3]
        dry_rate=dry_rate,
        inputs={"mass_frac": np.zeros(3)},  # [-]
    )

    expected = np.array([
        [0.8, 0.0, 0.0],
        [0.26666666666666666, 0.0, 0.0],
        [0.48, 0.0, 0.0],
    ])  # [1/s]
    np.testing.assert_allclose(dygas_dt, expected)


@pytest.mark.assimulo
def test_solve_unit_single_node_initial_state_includes_both_temperatures(
        drying_unit_factory):
    """A real single-node solve carries gas and condensed temperatures."""
    pytest.importorskip("assimulo")
    gas_temperature = 300.0  # [K]
    condensed_temperature = 302.0  # [K]
    dryer = drying_unit_factory(
        number_nodes=1,
        gas_temperature=gas_temperature,
        condensed_temperature=condensed_temperature,
    )

    _, states = dryer.solve_unit(
        deltaP=5.0e4,
        runtime=1.0e-8,  # [s]
        verbose=False,
    )

    expected_width = 3 + dryer.Liquid_1.num_species + dryer.num_volatiles  # [-]
    assert np.shape(states)[1] == expected_width
    assert states[0, -2] == pytest.approx(gas_temperature)
    assert states[0, -1] == pytest.approx(condensed_temperature)
