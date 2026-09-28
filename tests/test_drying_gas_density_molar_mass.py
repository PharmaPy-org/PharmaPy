"""Drying gas molar-mass regressions with real vapor collaborators.

``y_gas`` uses a mass basis, so mixture molar mass is
``sum(w_i) / sum(w_i / MW_i)`` [g/mol]. The cases cover density and energy
consumers, pure-species limits, basis equivalence, and scale drift.
"""

import numpy as np
import pytest


pytestmark = pytest.mark.unit

GAS_CONSTANT = 8.314  # [J/mol/K]
MW_GAS = np.array([18.0, 46.0, 28.0])  # [g/mol], water, ethanol, nitrogen
Y_GAS = np.array([
    [0.60, 0.10, 0.30],
    [0.05, 0.15, 0.80],
    [0.25, 0.25, 0.50],
])  # [-]
TEMP_GAS = np.array([313.15, 333.15, 353.15])  # [K]
TEMP_COND = np.array([310.15, 330.15, 350.15])  # [K]
PRES_GAS = np.array([101325.0, 100000.0, 98000.0])  # [Pa]
SATURATION = np.array([0.45, 0.30, 0.20])  # [-]
X_LIQ = np.array([
    [0.30, 0.70],
    [0.40, 0.60],
    [0.50, 0.50],
])  # [-], water and ethanol


def _mixture_molar_mass(mass_fractions=Y_GAS):
    """Calculate mixture molar mass from mass fractions.

    Parameters
    ----------
    mass_fractions : ndarray, optional
        Node-wise gas mass fractions [-].

    Returns
    -------
    ndarray
        Node-wise mixture molar mass [g/mol].
    """
    mass_fractions = np.asarray(mass_fractions)  # [-]
    return mass_fractions.sum(axis=-1) / (
        mass_fractions / MW_GAS
    ).sum(axis=-1)


def _expected_gas_density():
    """Calculate ideal-gas density from the reciprocal-sum molar mass.

    Returns
    -------
    ndarray
        Node-wise gas density [kg/m**3].
    """
    return (
        PRES_GAS * _mixture_molar_mass()
        / (GAS_CONSTANT * TEMP_GAS * 1000.0)
    )


def test_unit_model_gas_density_uses_mixture_molar_mass(
        drying_unit_factory):
    """The real RHS exposes ideal-gas density with reciprocal-sum MW."""
    dryer = drying_unit_factory(number_nodes=3)
    probe_state = dryer.initialize_states(
        deltaP=5.0e4).ravel()  # [-] and [K]
    state_width = 3 + dryer.Liquid_1.num_species + dryer.num_volatiles  # [-]
    states_by_node = probe_state.reshape(dryer.num_nodes, state_width)
    states_by_node[:, 0] = SATURATION
    states_by_node[:, 1:4] = Y_GAS
    states_by_node[:, 4:6] = X_LIQ
    states_by_node[:, -2] = TEMP_GAS
    states_by_node[:, -1] = TEMP_COND
    dryer.pres_gas = PRES_GAS

    derivatives = dryer.unit_model(0.0, probe_state)

    assert np.all(np.isfinite(derivatives))
    np.testing.assert_allclose(dryer.rho_gas, _expected_gas_density(), rtol=1e-12)
    # Hand-calculated wet-node anchor from 18, 46, and 28 g/mol species.
    assert dryer.rho_gas[0] == pytest.approx(0.841996, rel=1e-6)


def test_energy_specific_gas_constant_uses_mixture_molar_mass(
        drying_unit_factory):
    """Use ``cv = cp - R/MW`` with the real vapor mixture molar mass."""
    dryer = drying_unit_factory(number_nodes=3)
    dryer.idx_volatiles = np.array([0, 1])  # species indices [-]
    dryer.idx_supercrit = np.array([2])  # species indices [-]
    dryer.porosity = 0.5  # [-]
    dryer.rho_sol = dryer.Solid_1.getDensity()  # [kg/m**3]
    dryer.cp_sol = dryer.Solid_1.getCp()  # [J/kg/K]
    dryer.rho_liq = np.array([940.0, 900.0, 880.0])  # [kg/m**3]
    dryer.h_T_j = 12.0  # [W/m**2/K]
    dryer.a_V = 2500.0  # [m**2/m**3]
    dryer.dz = np.full(3, 0.01)  # [m]
    rho_gas = _expected_gas_density()  # [kg/m**3]

    _, _, heat_cond, _ = dryer.energy_balance(
        time=0.0,
        temp_gas=TEMP_GAS,
        temp_sol=TEMP_COND,
        satur=SATURATION,
        y_gas=Y_GAS,
        x_liq=X_LIQ,
        u_gas=np.zeros(3),  # [m/s]
        rho_gas=rho_gas,
        dry_rate=np.zeros((3, 3)),  # [kg/m**3/s]
        inputs={"temp": 353.15},  # [K]
        return_terms=True,
    )

    cp_gas = dryer.Vapor_1.getCp(
        temp=TEMP_GAS, mass_frac=Y_GAS, basis="mass"
    )  # [J/kg/K]
    specific_gas_constant = (
        GAS_CONSTANT / _mixture_molar_mass() * 1000.0
    )  # [J/kg/K]
    cv_gas = cp_gas - specific_gas_constant  # [J/kg/K]
    epsilon_gas = dryer.porosity * (1.0 - SATURATION)  # [-]
    expected = (
        dryer.h_T_j * dryer.a_V * (TEMP_GAS - TEMP_COND)
        / (cv_gas * epsilon_gas * rho_gas)
    )  # [K/s]
    np.testing.assert_allclose(heat_cond, expected, rtol=1e-12)


@pytest.mark.parametrize("species_index", range(MW_GAS.size))
def test_gas_mixture_molar_mass_has_single_species_limit(
        drying_unit_factory, species_index):
    """A pure real gas collaborator has its species molar mass [g/mol]."""
    dryer = drying_unit_factory(number_nodes=1)
    mass_fractions = np.zeros(MW_GAS.size)  # [-]
    mass_fractions[species_index] = 1.0  # [-]

    mixture_molar_mass = dryer._gas_mixture_molar_mass(
        mass_fractions
    )  # [g/mol]

    assert mixture_molar_mass == pytest.approx(MW_GAS[species_index])


def test_gas_mixture_molar_mass_matches_real_mole_fraction_conversion(
        drying_unit_factory):
    """Mass- and mole-basis mixture molar masses are equivalent [g/mol]."""
    dryer = drying_unit_factory(number_nodes=3)

    mass_basis_molar_mass = dryer._gas_mixture_molar_mass(Y_GAS)  # [g/mol]
    gas_mole_fractions = dryer.Vapor_1.frac_to_frac(
        mass_frac=Y_GAS
    )  # [-]
    mole_basis_molar_mass = np.dot(gas_mole_fractions, MW_GAS)  # [g/mol]

    np.testing.assert_allclose(
        mass_basis_molar_mass, mole_basis_molar_mass, rtol=1e-12
    )


def test_gas_mixture_molar_mass_is_invariant_to_composition_sum_drift(
        drying_unit_factory):
    """Common mass-fraction scale drift does not change molar mass."""
    dryer = drying_unit_factory(number_nodes=3)
    composition_scales = np.array([0.99, 1.01, 1.05])  # [-]
    drifted_mass_fractions = Y_GAS * composition_scales[:, None]  # [-]

    drifted_molar_mass = dryer._gas_mixture_molar_mass(
        drifted_mass_fractions
    )  # [g/mol]
    gas_mole_fractions = dryer.Vapor_1.frac_to_frac(
        mass_frac=drifted_mass_fractions
    )  # [-]
    mole_basis_molar_mass = np.dot(gas_mole_fractions, MW_GAS)  # [g/mol]

    np.testing.assert_allclose(
        drifted_molar_mass, mole_basis_molar_mass, rtol=1e-12
    )
    np.testing.assert_allclose(
        drifted_molar_mass, _mixture_molar_mass(), rtol=1e-12
    )
