"""Regression coverage for the real Drying condensed latent-heat term."""

import numpy as np
import pytest


pytestmark = pytest.mark.unit


def test_condensed_energy_uses_single_latent_heat_factor(
        drying_unit_factory):
    """Mass drying rates times latent heat already give full latent power."""
    dryer = drying_unit_factory(number_nodes=4)
    dryer.idx_volatiles = np.array([0, 1])  # component indices [-]
    dryer.idx_supercrit = np.array([2])  # component indices [-]
    dryer.porosity = 0.5  # [-]
    dryer.rho_sol = dryer.Solid_1.getDensity()  # [kg/m**3]
    dryer.cp_sol = dryer.Solid_1.getCp()  # [J/kg/K]
    dryer.rho_liq = np.full(4, 800.0)  # [kg/m**3]
    dryer.h_T_j = 0.0  # [W/m**2/K]
    dryer.a_V = 1.0  # [m**2/m**3]
    dryer.dz = np.ones(4)  # [m]

    temp_gas = np.full(4, 295.0)  # [K], wet-bulb reference
    temp_cond = np.array([299.0, 304.0, 309.0, 314.0])  # [K]
    saturation = np.full(4, 0.5)  # [-]
    y_gas = np.array([
        [0.10, 0.10, 0.80],
        [0.20, 0.10, 0.70],
        [0.30, 0.10, 0.60],
        [0.40, 0.10, 0.50],
    ])  # [-]
    x_liq = np.array([
        [0.25, 0.75],
        [0.40, 0.60],
        [0.55, 0.45],
        [0.70, 0.30],
    ])  # [-]
    dry_rate = np.array([
        [0.036, 0.184, 0.0],
        [0.054, 0.230, 0.0],
        [0.018, 0.092, 0.0],
        [0.027, 0.046, 0.0],
    ])  # [kg/m**3/s]
    balance_arguments = {
        "time": 0.0,  # [s]
        "temp_gas": temp_gas,
        "temp_sol": temp_cond,
        "satur": saturation,
        "y_gas": y_gas,
        "x_liq": x_liq,
        "u_gas": np.zeros(4),  # [m/s]
        "rho_gas": np.ones(4),  # [kg/m**3]
        "dry_rate": dry_rate,
        "inputs": {"temp": 295.0},  # [K]
    }

    _, condensed_rate = dryer.energy_balance(**balance_arguments)
    _, drying_term, _, _ = dryer.energy_balance(
        **balance_arguments, return_terms=True
    )

    latent_heat = dryer.Vapor_1.getHeatVaporization(
        temp_cond, basis="mass"
    )[:, dryer.idx_volatiles]  # [J/kg]
    liquid_cp = dryer.Liquid_1.getCp(
        temp=temp_cond,
        mass_frac=np.column_stack((x_liq, np.zeros(4))),
        basis="mass",
    )  # [J/kg/K]
    heat_capacity = (
        dryer.rho_sol * (1.0 - dryer.porosity) * dryer.cp_sol
        + dryer.porosity * saturation * liquid_cp * dryer.rho_liq
    )  # [J/m**3/K]
    expected_drying_term = (
        dry_rate[:, dryer.idx_volatiles] * latent_heat
    ).sum(axis=1) / heat_capacity  # [K/s]

    np.testing.assert_allclose(condensed_rate, -expected_drying_term)
    np.testing.assert_allclose(drying_term, expected_drying_term)
