"""Solver-free regressions for drying-rate unit contracts.

The fixtures are intentionally small, made-up drying states. They isolate the
molar-to-mass drying-rate conversion and the downstream material-balance terms
without running an end-to-end drying simulation.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from PharmaPy.Drying_Model import Drying

pytestmark = pytest.mark.unit


def test_drying_rate_mass_basis_converts_molar_rates_with_component_mw():
    dryer = Drying(number_nodes=2, supercrit_names=["nitrogen"])
    dryer.Liquid_1 = SimpleNamespace(mw=np.array([18.0, 28.0, 46.0]))  # [g/mol]
    dry_rate = np.array([
        [2.0, 0.0, 4.0],
        [3.0, 0.0, 5.0],
    ])  # [mol/m**3/s]

    dry_rate_mass = dryer._drying_rate_mass_basis(dry_rate)  # [kg/m**3/s]

    expected = np.array([
        [0.036, 0.0, 0.184],
        [0.054, 0.0, 0.230],
    ])  # [kg/m**3/s]
    np.testing.assert_allclose(dry_rate_mass, expected)


def test_material_balance_uses_mass_drying_rate_for_saturation():
    dryer = Drying(number_nodes=2, supercrit_names=["nitrogen"])
    dryer.idx_volatiles = np.array([0, 2])  # [-]
    dryer.porosity = 0.4  # [-]
    dryer.rho_liq = np.array([800.0, 900.0])  # [kg/m**3]
    dryer.dz = np.ones(2)  # [m]
    dryer.Liquid_1 = SimpleNamespace(mw=np.array([18.0, 28.0, 46.0]))  # [g/mol]

    satur = np.array([0.6, 0.8])  # [-]
    temp_gas = np.array([300.0, 305.0])  # [K]
    temp_sol = np.array([299.0, 304.0])  # [K]
    y_gas = np.array([
        [0.02, 0.96, 0.02],
        [0.03, 0.94, 0.03],
    ])  # [-]
    x_liq = np.array([
        [0.25, 0.75],
        [0.40, 0.60],
    ])  # [-]
    u_gas = np.zeros(2)  # [m/s]
    dens_gas = np.ones(2)  # [kg/m**3]
    dry_rate = np.array([
        [0.036, 0.0, 0.184],
        [0.054, 0.0, 0.230],
    ])  # [kg/m**3/s]
    inputs = {"mass_frac": np.array([0.01, 0.98, 0.01])}  # [-]

    dsat_dt, _, _ = dryer.material_balance(
        time=0.0,
        satur=satur,
        temp_gas=temp_gas,
        temp_sol=temp_sol,
        y_gas=y_gas,
        x_liq=x_liq,
        u_gas=u_gas,
        dens_gas=dens_gas,
        dry_rate=dry_rate,
        inputs=inputs,
    )  # [1/s]

    mass_rate = np.array([0.22, 0.284])  # [kg/m**3/s]
    expected_dsat_dt = -mass_rate / dryer.rho_liq / dryer.porosity  # [1/s]

    np.testing.assert_allclose(dsat_dt, expected_dsat_dt)


def test_unit_model_converts_drying_rate_before_balance_equations(monkeypatch):
    dryer = Drying(number_nodes=2, supercrit_names=["nitrogen"])
    dryer.num_volatiles = 2  # [-]
    dryer.idx_volatiles = np.array([0, 2])  # [-]
    dryer.idx_supercrit = np.array([1])  # [-]
    dryer.porosity = 0.5  # [-]
    dryer.rho_sol = 1500.0  # [kg/m**3]
    dryer.s_inf = 0.1  # [-]
    dryer.k_perm = 1.0  # [m**2]
    dryer.dPg_dz = 2.0  # [Pa/m]
    dryer.pres_gas = np.array([101325.0, 101000.0])  # [Pa]
    dryer.CakePhase = SimpleNamespace(alpha=1.2)  # [m/kg]
    dryer.Liquid_1 = SimpleNamespace(
        num_species=3,  # [-]
        mw=np.array([18.0, 28.0, 46.0]),  # [g/mol]
        rho_liq=np.array([800.0, 850.0, 900.0]),  # [kg/m**3]
    )
    dryer.Vapor_1 = SimpleNamespace(
        mw=np.array([18.0, 28.0, 46.0]),  # [g/mol]
        getViscosity=lambda temp, mass_frac: np.ones(2),  # [Pa*s]
    )
    dryer.get_inputs = lambda time: {
        "Inlet": {"mass_frac": np.array([0.05, 0.90, 0.05])}
    }  # [-]

    molar_dry_rate = np.array([
        [2.0, 0.0, 4.0],
        [3.0, 0.0, 5.0],
    ])  # [mol/m**3/s]
    expected_mass_rate = np.array([
        [0.036, 0.0, 0.184],
        [0.054, 0.0, 0.230],
    ])  # [kg/m**3/s]
    monkeypatch.setattr(
        dryer,
        "get_drying_rate",
        lambda x_liq, temp_sol, y_gas, pres_gas: molar_dry_rate,
    )

    captured = {}  # [-]

    def material_balance(*args, **kwargs):
        captured["material_dry_rate"] = args[8].copy()  # [kg/m**3/s]
        return [np.zeros(2), np.zeros((2, 3)), np.zeros((2, 2))]  # [1/s]

    def energy_balance(*args, **kwargs):
        captured["energy_dry_rate"] = args[8].copy()  # [kg/m**3/s]
        return [np.zeros(2), np.zeros(2)]  # [K/s]

    monkeypatch.setattr(dryer, "material_balance", material_balance)
    monkeypatch.setattr(dryer, "energy_balance", energy_balance)

    states = np.array([
        [0.6, 0.10, 0.80, 0.10, 0.55, 0.45, 300.0, 299.0],
        [0.7, 0.20, 0.70, 0.10, 0.60, 0.40, 301.0, 300.0],
    ])  # S [-], w_gas [-], w_liq [-], Tg [K], Ts [K]

    model_eqns = dryer.unit_model(time=0.0, states=states.ravel())  # [1/s] and [K/s]

    np.testing.assert_allclose(captured["material_dry_rate"], expected_mass_rate)
    np.testing.assert_allclose(captured["energy_dry_rate"], expected_mass_rate)
    np.testing.assert_allclose(dryer.dry_rate, expected_mass_rate)
    np.testing.assert_allclose(model_eqns, np.zeros(states.size))


def test_material_balance_uses_mass_drying_rate_for_gas_species():
    """Verify mass-basis transfer and normalized gas accumulation (#230).

    Notes
    -----
    The old saturation-only correction was an artifact of the defect: it
    omitted total gas-inventory changes and made normalized fractions grow.
    The constant-pressure pore gas is open, with a vent carrying the bulk
    composition. That vent cancels between component and total balances,
    giving H*dy_i/dt = r_i - y_i*sum(r_j) at zero convection, where H is gas
    mass per bed volume. The explicit transfer values below retain the
    molar-to-mass and single-holdup (#81) regressions.
    """
    dryer = Drying(number_nodes=2, supercrit_names=["nitrogen"])
    dryer.idx_volatiles = np.array([0, 2])  # [-]
    dryer.porosity = 0.5  # [-]
    dryer.rho_liq = np.array([1000.0, 1000.0])  # [kg/m**3]
    dryer.dz = np.ones(2)  # [m]

    satur = np.array([0.5, 0.5])  # [-]
    temp_gas = np.array([300.0, 300.0])  # [K]
    temp_sol = np.array([299.0, 299.0])  # [K]
    y_gas = np.array([
        [0.10, 0.80, 0.10],
        [0.20, 0.70, 0.10],
    ])  # [-]
    x_liq = np.array([
        [0.50, 0.50],
        [0.50, 0.50],
    ])  # [-]
    u_gas = np.zeros(2)  # [m/s]
    dens_gas = np.array([1.2, 1.5])  # [kg/m**3]
    dry_rate = np.array([
        [0.036, 0.0, 0.184],
        [0.054, 0.0, 0.230],
    ])  # [kg/m**3/s]
    inputs = {"mass_frac": np.array([0.05, 0.90, 0.05])}  # [-]

    _, dygas_dt, _ = dryer.material_balance(
        time=0.0,
        satur=satur,
        temp_gas=temp_gas,
        temp_sol=temp_sol,
        y_gas=y_gas,
        x_liq=x_liq,
        u_gas=u_gas,
        dens_gas=dens_gas,
        dry_rate=dry_rate,
        inputs=inputs,
    )  # [1/s]

    expected_transfer = np.array([
        [0.12, 0.0, 0.6133333333333334],
        [0.144, 0.0, 0.6133333333333333],
    ])  # [1/s]
    expected_dygas_dt = expected_transfer - y_gas * np.array([
        [0.22 / 0.3], [0.284 / 0.375],
    ])  # [1/s], total mass sources divided by gas holdups [kg/m**3]
    np.testing.assert_allclose(dygas_dt, expected_dygas_dt)
    # Roundoff from three component rates of order one per second.
    np.testing.assert_allclose(dygas_dt.sum(axis=1), 0., atol=1e-15)
