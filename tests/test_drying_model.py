"""Solver-free and real-backend regressions for drying-rate unit contracts."""

import numpy as np
import pytest


pytestmark = pytest.mark.unit

SPECIES_MOLAR_MASS_KG = np.array([18.0, 46.0, 28.0]) / 1000.0  # [kg/mol]


def test_drying_rate_mass_basis_converts_component_rates(
        drying_unit_factory):
    """Convert real-phase molar rates with each component molar mass."""
    dryer = drying_unit_factory(number_nodes=2)
    dry_rate = np.array([
        [2.0, 4.0, 0.0],
        [3.0, 5.0, 0.0],
    ])  # [mol/m**3/s]

    dry_rate_mass = dryer._drying_rate_mass_basis(dry_rate)  # [kg/m**3/s]

    expected = np.array([
        [0.036, 0.184, 0.0],
        [0.054, 0.230, 0.0],
    ])  # [kg/m**3/s], molar masses are 18, 46, and 28 g/mol
    np.testing.assert_allclose(dry_rate_mass, expected)


def test_material_balance_uses_mass_drying_rate_for_saturation(
        drying_unit_factory):
    """Use mass-basis volatile rates in the real material balance."""
    dryer = drying_unit_factory(number_nodes=2)
    dryer.idx_volatiles = np.array([0, 1])  # component indices [-]
    dryer.porosity = 0.4  # [-]
    dryer.rho_liq = np.array([800.0, 900.0])  # [kg/m**3]
    dryer.dz = np.ones(2)  # [m]

    satur = np.array([0.6, 0.8])  # [-]
    temp_gas = np.array([300.0, 305.0])  # [K]
    temp_sol = np.array([299.0, 304.0])  # [K]
    y_gas = np.array([
        [0.02, 0.02, 0.96],
        [0.03, 0.03, 0.94],
    ])  # [-]
    x_liq = np.array([
        [0.25, 0.75],
        [0.40, 0.60],
    ])  # [-]
    dry_rate = np.array([
        [0.036, 0.184, 0.0],
        [0.054, 0.230, 0.0],
    ])  # [kg/m**3/s]

    dsat_dt, _, _ = dryer.material_balance(
        time=0.0,
        satur=satur,
        temp_gas=temp_gas,
        temp_sol=temp_sol,
        y_gas=y_gas,
        x_liq=x_liq,
        u_gas=np.zeros(2),  # [m/s]
        dens_gas=np.ones(2),  # [kg/m**3]
        dry_rate=dry_rate,
        inputs={"mass_frac": np.array([0.01, 0.01, 0.98])},  # [-]
    )

    mass_rate = np.array([0.22, 0.284])  # [kg/m**3/s]
    expected_dsat_dt = -mass_rate / dryer.rho_liq / dryer.porosity  # [1/s]
    np.testing.assert_allclose(dsat_dt, expected_dsat_dt)


def test_unit_model_hands_mass_rates_to_real_balance_path(
        drying_unit_factory):
    """The real RHS stores mass rates converted from its molar correlation."""
    dryer = drying_unit_factory(number_nodes=2)
    initial_state = dryer.initialize_states(
        deltaP=5.0e4).ravel()  # [-] and [K]

    state_width = 3 + dryer.Liquid_1.num_species + dryer.num_volatiles  # [-]
    states_by_node = initial_state.reshape(dryer.num_nodes, state_width)
    y_gas = states_by_node[:, 1:1 + dryer.Liquid_1.num_species]  # [-]
    x_liq = states_by_node[
        :, 1 + dryer.Liquid_1.num_species:
        1 + dryer.Liquid_1.num_species + dryer.num_volatiles
    ].copy()  # [-]
    # [-], mirrors the #42 defect that zeroes the first volatile column;
    # remove this line and re-derive the expectation once #42 is fixed.
    x_liq[:, -2] = 0.0
    temp_cond = states_by_node[:, -1]  # [K]
    molar_rate = dryer.get_drying_rate(
        x_liq, temp_cond, y_gas, dryer.pres_gas
    )  # [mol/m**3/s]
    expected_mass_rate = (
        molar_rate * SPECIES_MOLAR_MASS_KG)  # [kg/m**3/s]

    model_equations = dryer.unit_model(0.0, initial_state.copy())

    assert model_equations.shape == initial_state.shape
    assert np.all(np.isfinite(model_equations))
    assert np.any(expected_mass_rate > 0)
    np.testing.assert_allclose(dryer.dry_rate, expected_mass_rate)


def test_material_balance_uses_mass_drying_rate_for_gas_species(
        drying_unit_factory):
    """Verify mass-basis transfer and normalized gas accumulation (#230).

    Parameters
    ----------
    drying_unit_factory : callable
        Construct a drying unit with real phases in water/ethanol/carrier order.

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
    dryer = drying_unit_factory(number_nodes=2)
    dryer.idx_volatiles = np.array([0, 1])  # component indices [-]
    dryer.porosity = 0.5  # [-]
    dryer.rho_liq = np.array([1000.0, 1000.0])  # [kg/m**3]
    dryer.dz = np.ones(2)  # [m]

    satur = np.array([0.5, 0.5])  # [-]
    y_gas = np.array([
        [0.10, 0.10, 0.80],
        [0.20, 0.10, 0.70],
    ])  # [-]
    x_liq = np.full((2, 2), 0.5)  # [-]
    dry_rate = np.array([
        [0.036, 0.184, 0.0],
        [0.054, 0.230, 0.0],
    ])  # [kg/m**3/s]
    dens_gas = np.array([1.2, 1.5])  # [kg/m**3]

    _, dygas_dt, _ = dryer.material_balance(
        time=0.0,
        satur=satur,
        temp_gas=np.full(2, 300.0),  # [K]
        temp_sol=np.full(2, 299.0),  # [K]
        y_gas=y_gas,
        x_liq=x_liq,
        u_gas=np.zeros(2),  # [m/s]
        dens_gas=dens_gas,
        dry_rate=dry_rate,
        inputs={"mass_frac": np.array([0.01, 0.01, 0.98])},  # [-]
    )

    expected_transfer = np.array([
        [0.12, 0.6133333333333334, 0.0],
        [0.144, 0.6133333333333333, 0.0],
    ])  # [1/s]
    expected_dygas_dt = expected_transfer - y_gas * np.array([
        [0.22 / 0.3], [0.284 / 0.375],
    ])  # [1/s], total mass sources divided by gas holdups [kg/m**3]
    np.testing.assert_allclose(dygas_dt, expected_dygas_dt)
    # Roundoff from three component rates of order one per second.
    np.testing.assert_allclose(dygas_dt.sum(axis=1), 0., atol=1e-15)
