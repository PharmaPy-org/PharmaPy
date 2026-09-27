"""Drying energy-basis regressions with real thermophysical collaborators."""

import numpy as np
import pytest


pytestmark = pytest.mark.unit


def test_energy_balance_consumes_component_mass_drying_rates(
        drying_unit_factory):
    """Combine converted mass rates with real mass-basis latent heats."""
    dryer = drying_unit_factory(number_nodes=3)
    dryer.idx_volatiles = np.array([0, 1])  # component indices [-]
    dryer.idx_supercrit = np.array([2])  # component indices [-]
    dryer.porosity = 0.5  # [-]
    dryer.rho_sol = dryer.Solid_1.getDensity()  # [kg/m**3]
    dryer.cp_sol = dryer.Solid_1.getCp()  # [J/kg/K]
    dryer.rho_liq = np.array([800.0, 850.0, 900.0])  # [kg/m**3]
    dryer.h_T_j = 0.0  # [W/m**2/K]
    dryer.a_V = 1.0  # [m**2/m**3]
    dryer.dz = np.ones(3)  # [m]

    molar_dry_rate = np.array([
        [2.0, 4.0, 0.0],
        [3.0, 5.0, 0.0],
        [1.0, 2.0, 0.0],
    ])  # [mol/m**3/s]
    mass_dry_rate = dryer._drying_rate_mass_basis(
        molar_dry_rate
    )  # [kg/m**3/s]
    temp_gas = np.full(3, 295.0)  # [K], wet-bulb reference in energy_balance
    temp_cond = np.array([299.0, 304.0, 309.0])  # [K]
    saturation = np.full(3, 0.5)  # [-]
    y_gas = np.array([
        [0.10, 0.10, 0.80],
        [0.20, 0.10, 0.70],
        [0.30, 0.10, 0.60],
    ])  # [-]
    x_liq = np.array([
        [0.25, 0.75],
        [0.40, 0.60],
        [0.55, 0.45],
    ])  # [-]

    _, condensed_rate = dryer.energy_balance(
        time=0.0,
        temp_gas=temp_gas,
        temp_sol=temp_cond,
        satur=saturation,
        y_gas=y_gas,
        x_liq=x_liq,
        u_gas=np.zeros(3),  # [m/s]
        rho_gas=np.ones(3),  # [kg/m**3]
        dry_rate=mass_dry_rate,
        inputs={"temp": 295.0},  # [K]
    )

    latent_heat = dryer.Vapor_1.getHeatVaporization(
        temp_cond, basis="mass"
    )[:, dryer.idx_volatiles]  # [J/kg]
    liquid_composition = np.column_stack(
        (x_liq, np.zeros(3))
    )  # [-]
    liquid_cp = dryer.Liquid_1.getCp(
        temp=temp_cond,
        mass_frac=liquid_composition,
        basis="mass",
    )  # [J/kg/K]
    heat_capacity = (
        dryer.rho_sol * (1.0 - dryer.porosity) * dryer.cp_sol
        + dryer.porosity * saturation * liquid_cp * dryer.rho_liq
    )  # [J/m**3/K]
    latent_power = (
        mass_dry_rate[:, dryer.idx_volatiles] * latent_heat
    ).sum(axis=1)  # [J/m**3/s]
    expected_rate = -latent_power / heat_capacity  # [K/s]

    np.testing.assert_allclose(condensed_rate, expected_rate)
    assert np.all(np.isfinite(condensed_rate))
