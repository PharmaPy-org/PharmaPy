"""Focused Drying gas-balance regressions for issue #81.

These tests use compact synthetic RHS/state-assembly fixtures rather than a
full ``Drying.solve_unit`` transient. The end-to-end transient remains deferred
until the open Drying correctness issues are resolved, so the assertions here
pin the local unit and holdup contracts affected by #81.
"""

from types import SimpleNamespace

import numpy as np
import pytest

import PharmaPy.Drying_Model as drying_model


pytestmark = pytest.mark.unit


def test_unit_model_uses_relative_permeability_for_gas_velocity():
    """Darcy velocity uses the clipped relative permeability factor [-]."""
    dryer = drying_model.Drying(number_nodes=3, supercrit_names=["nitrogen"])
    dryer.idx_volatiles = np.array([0, 2])  # component indices [-]
    dryer.num_volatiles = 2  # [-]
    dryer.s_inf = 0.1  # [-]
    dryer.porosity = 0.5  # [-]
    dryer.rho_sol = 5.0  # [kg/m**3]
    dryer.dPg_dz = 11.0  # [Pa/m]
    dryer.k_perm = 0.2  # [m**2]
    dryer.pres_gas = np.array([101325.0, 101300.0, 101275.0])  # [Pa]
    dryer.CakePhase = SimpleNamespace(alpha=2.0)  # [m/kg]
    dryer.Liquid_1 = SimpleNamespace(
        num_species=3,
        mw=np.array([18.0, 28.0, 44.0]),  # [g/mol]
        rho_liq=np.array([800.0, 850.0, 900.0]),  # [kg/m**3]
    )
    dryer.Vapor_1 = SimpleNamespace(
        mw=np.array([18.0, 28.0, 44.0]),  # [g/mol]
        getViscosity=lambda temp, mass_frac: np.array([2.0, 4.0, 5.0]),  # [Pa*s]
    )
    dryer.get_drying_rate = lambda *args: np.zeros((3, 3))  # [mol/m**3/s]
    dryer.get_inputs = lambda time: {
        "Inlet": {
            "mass_frac": np.array([0.01, 0.98, 0.01]),  # [-]
            "temp": 300.0,  # [K]
        }
    }

    captured = {}

    def material_balance(
        time,
        satur,
        temp_gas,
        temp_sol,
        y_gas,
        x_liq,
        u_gas,
        dens_gas,
        dry_rate,
        inputs,
    ):
        captured["u_gas"] = u_gas.copy()
        return [
            np.zeros(3),  # saturation derivative [1/s]
            np.zeros((3, 3)),  # gas mass-fraction derivative [1/s]
            np.zeros((3, 2)),  # liquid mass-fraction derivative [1/s]
        ]

    def energy_balance(
        time,
        temp_gas,
        temp_sol,
        satur,
        y_gas,
        x_liq,
        u_gas,
        rho_gas,
        dry_rate,
        inputs,
    ):
        return [
            np.zeros(3),  # gas-temperature derivative [K/s]
            np.zeros(3),  # condensed-temperature derivative [K/s]
        ]

    dryer.material_balance = material_balance
    dryer.energy_balance = energy_balance

    states = np.array(
        [
            [1.0, 0.02, 0.96, 0.02, 0.25, 0.75, 300.0, 299.0],
            [0.1, 0.02, 0.96, 0.02, 0.25, 0.75, 301.0, 298.0],
            [1.2, 0.02, 0.96, 0.02, 0.25, 0.75, 302.0, 297.0],
        ]
    )  # columns: saturation [-], y_gas [-], x_liq [-], temp_gas/temp_cond [K]

    dryer.unit_model(0.0, states.ravel())

    # k_ra is dimensionless. At S=1 and above it is zero; at S=s_inf it is one.
    # k_perm*dP/dz/viscosity: [m**2] * [Pa/m] / [Pa*s] = [m/s].
    np.testing.assert_allclose(captured["u_gas"], np.array([0.0, 0.55, 0.0]))


def test_material_balance_uses_single_gas_holdup_factor_for_transfer():
    """Gas transfer divides by the gas holdup exactly once."""
    dryer = drying_model.Drying(number_nodes=3, supercrit_names=["nitrogen"])
    dryer.idx_volatiles = np.array([0, 2])  # component indices [-]
    dryer.porosity = 0.5  # [-]
    dryer.rho_liq = np.array([800.0, 900.0, 1000.0])  # [kg/m**3]
    dryer.dz = np.ones(3)  # [m]
    dryer.Liquid_1 = SimpleNamespace(mw=np.array([18.0, 28.0, 46.0]))  # [g/mol]

    satur = np.array([0.75, 0.25, 0.50])  # [-]
    temp_gas = np.array([300.0, 305.0, 310.0])  # [K]
    temp_sol = np.array([299.0, 304.0, 309.0])  # [K]
    y_gas = np.zeros((3, 3))  # [-], keeps the saturation correction out
    x_liq = np.array([
        [0.25, 0.75],
        [0.40, 0.60],
        [0.55, 0.45],
    ])  # [-]
    u_gas = np.zeros(3)  # [m/s], isolates the transfer term
    dens_gas = np.array([2.0, 4.0, 5.0])  # [kg/m**3]
    dry_rate = np.array(
        [
            [0.2, 0.0, 0.0],
            [0.4, 0.0, 0.0],
            [0.6, 0.0, 0.0],
        ]
    )  # [kg/m**3/s]
    inputs = {"mass_frac": np.zeros(3)}  # [-]

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
    )

    expected = np.array(
        [
            [0.8, 0.0, 0.0],
            [0.26666666666666666, 0.0, 0.0],
            [0.48, 0.0, 0.0],
        ]
    )  # [1/s]
    np.testing.assert_allclose(dygas_dt, expected)


def test_solve_unit_single_node_initial_state_includes_condensed_temperature(
    monkeypatch,
):
    """Pin single-node state assembly before the CVode solve boundary.

    The test uses real ``solve_unit`` initialization, including the
    ``get_sat_inf`` saturation calculation. It replaces ``unit_model`` only at
    the final handoff so the initial state can be inspected without running a
    full transient; the broader end-to-end Drying solve remains deferred until
    the open Drying correctness issues are resolved.
    """
    dryer = drying_model.Drying(number_nodes=1, supercrit_names=["nitrogen"])
    dryer.names_states_in = ["temp", "mass_frac"]
    dryer.idx_supercrit = np.array([1])  # component indices [-]
    dryer.cake_height = 1.0  # [m]
    dryer.Liquid_1 = SimpleNamespace(
        num_species=3,
        mass_frac=np.array([0.20, 0.10, 0.70]),  # [-]
        mw=np.array([18.0, 28.0, 46.0]),  # [g/mol]
        getDensity=lambda temp, mass_frac, basis: np.array([900.0]),  # [kg/m**3]
        getSurfTension=lambda temp, mass_frac: np.array([0.072]),  # [N/m]
    )
    solid = SimpleNamespace(
        temp=302.0,  # [K]
        x_distrib=np.array([50.0, 100.0]),  # [um]
        distrib=np.array([1.0, 1.0]),  # [#/m**3/um]
        # Synthetic distribution moments [m**0, m, m**2, m**3, m**4].
        moments=np.array([1.0, 1.0, 2.0, 4.0, 0.0]),
        getDensity=lambda: 1200.0,  # [kg/m**3]
        getCp=lambda: 700.0,  # [J/kg/K]
        getMoments=lambda mom_num: np.array([1.0, 1.0, 2.0, 4.0, 0.0]),
    )
    dryer.Solid_1 = solid
    dryer.Vapor_1 = SimpleNamespace(
        mass_frac=np.array([0.01, 0.98, 0.01]),  # [-]
        temp=300.0,  # [K]
    )
    dryer.CakePhase = SimpleNamespace(
        Liquid_1=SimpleNamespace(mass_frac=np.array([0.20, 0.10, 0.70])),  # [-]
        Solid_1=solid,
        saturation=np.array([0.55]),  # [-]
        z_external=np.array([0.0, 1.0]),  # [m]
        alpha=2.0,  # [m/kg]
        porosity=0.45,  # [-]
    )

    class CapturedInitialState(Exception):
        """Stop the solve after checking the assembled initial state."""

    def assert_initial_state_width(time, states, sw=None):
        """Assert that the single-node drying state contains both temperatures.

        Parameters
        ----------
        time : float
            Initial solve time [s].
        states : ndarray
            Flattened initial drying state [-] and [K].
        sw : list, optional
            Assimulo event switches [-].

        Raises
        ------
        CapturedInitialState
            Always raised after checking the assembled state.
        """
        expected_width = 3 + dryer.Liquid_1.num_species + 2  # state columns [-]
        assert states.size == dryer.num_nodes * expected_width
        node = states.reshape(dryer.num_nodes, expected_width)
        assert node[0, -2] == 300.0  # temp_gas [K]
        assert node[0, -1] == 302.0  # temp_cond [K]
        raise CapturedInitialState

    dryer.unit_model = assert_initial_state_width
    monkeypatch.setattr(
        drying_model,
        "Explicit_Problem",
        lambda *args, **kwargs: SimpleNamespace(),
    )

    with pytest.raises(CapturedInitialState):
        dryer.solve_unit(deltaP=10.0)
