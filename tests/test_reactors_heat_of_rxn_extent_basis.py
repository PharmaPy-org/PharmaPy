"""Reaction heat-source extent-basis tests with production collaborators."""

import numpy as np
import pytest

from PharmaPy.Kinetics import RxnKinetics
from PharmaPy.Phases import LiquidPhase
from PharmaPy.Reactors import (
    BatchReactor,
    CSTR,
    PlugFlowReactor,
    SemibatchReactor,
)
from PharmaPy.Streams import LiquidStream
from PharmaPy.Utilities import CoolingWater


pytestmark = pytest.mark.unit

STOICH_AS_WRITTEN = np.array([[-2.0, 1.0, 0.0, 0.0]])  # [-]
STOICH_HALF_SCALED = np.array([[-1.0, 0.5, 0.0, 0.0]])  # [-]
DELTA_HRXN_AS_WRITTEN = -1.0e5  # [J/mol of reaction as written]
DELTA_HRXN_HALF_SCALED = DELTA_HRXN_AS_WRITTEN / 2.0  # [J/mol]
REACTION_ORDERS = [[1.0]]  # [-]
K_PARAMS = np.array([1.0e-3])  # [1/s]
EA_PARAMS = np.array([0.0])  # [J/mol]

MULTI_STOICH_AS_WRITTEN = np.array([
    [-2.0, 1.0, 0.0, 0.0],
    [0.0, -3.0, 1.0, 0.0],
])  # [-]
MULTI_STOICH_NORMALIZATION = np.array([2.0, 3.0])  # [-]
MULTI_STOICH_NORMALIZED = (
    MULTI_STOICH_AS_WRITTEN / MULTI_STOICH_NORMALIZATION[:, None]
)  # [-]
MULTI_DELTA_HRXN_AS_WRITTEN = np.array([-1.0e5, -6.0e4])  # [J/mol]
MULTI_DELTA_HRXN_NORMALIZED = (
    MULTI_DELTA_HRXN_AS_WRITTEN / MULTI_STOICH_NORMALIZATION
)  # [J/mol]
MULTI_REACTION_ORDERS = [[1.0], [1.0]]  # [-]
MULTI_K_PARAMS = np.array([1.0e-3, 2.0e-3])  # [1/s]
MULTI_EA_PARAMS = np.array([0.0, 0.0])  # [J/mol]

# Reversible 2A <-> B and 3B <-> C use elementary mass action under #263's
# equilibrium contract. These synthetic factors retain forward rates of
# 1e-3 and 2e-3 mol/L/s at 1 mol/L, on their respective reaction-order bases.
REVERSIBLE_REACTION_ORDERS = [[2.0], [3.0]]  # [-]
REVERSIBLE_K_PARAMS = np.array([1.0e-3, 2.0e-3])  # [L/mol/s; L**2/mol**2/s]

TREF_HRXN = 298.15  # [K]
TEMP_EVAL = 320.0  # [K]
VOL_EVAL = 2.0  # [m**3]
MOLE_CONC = np.array([1.5, 0.4, 0.0, 5.0])  # [mol/L]
MULTI_MOLE_CONC = np.array([1.5, 0.4, 0.3, 5.0])  # [mol/L]
INLET_VOL_FLOW = 1.0  # [m**3/s]


def _thermo_path(data_path):
    """Return the real four-species thermodynamic database path.

    Parameters
    ----------
    data_path : dict
        Paths to repository test-data directories.

    Returns
    -------
    str
        Path to the PFR thermodynamic JSON file.
    """
    return str(data_path["integration"] / "pfr_test_pure_comp.json")


def _build_reactor(
        data_path, reactor, stoich_matrix, delta_hrxn, *,
        k_params=K_PARAMS, ea_params=EA_PARAMS,
        reaction_orders=REACTION_ORDERS, mole_conc=MOLE_CONC,
        reversible=False):
    """Attach production phase, stream, kinetics, utility, and metadata.

    Parameters
    ----------
    data_path : dict
        Paths to repository test-data directories.
    reactor : BatchReactor, CSTR, SemibatchReactor, or PlugFlowReactor
        Production reactor to configure.
    stoich_matrix : ndarray
        Raw reaction stoichiometry with shape ``(n_reactions, 4)`` [-].
    delta_hrxn : float or ndarray
        Reference reaction enthalpies [J/mol of reaction as written].
    k_params : ndarray, optional
        Pre-exponential factors [(mol/L)**(1 - total forward order)/s].
    ea_params : ndarray, optional
        Activation energies [J/mol].
    reaction_orders : list of list of float, optional
        Forward reaction orders for each reactant [-].
    mole_conc : ndarray, optional
        Four-species liquid concentrations [mol/L].
    reversible : bool, optional
        Whether to configure finite equilibrium constants [-].

    Returns
    -------
    BatchReactor, CSTR, SemibatchReactor, or PlugFlowReactor
        Fully configured production reactor.
    """
    thermo_path = _thermo_path(data_path)
    liquid = LiquidPhase(
        thermo_path,
        temp=TEMP_EVAL,
        vol=VOL_EVAL,  # [m**3]
        mole_conc=mole_conc,
        verbose=False,
    )
    equilibrium_constants = (
        np.full(len(np.atleast_1d(k_params)), 8.0) if reversible else None
    )  # [mol/L]**(product order - reactant order)
    kinetics = RxnKinetics(
        path=thermo_path,
        k_params=k_params,
        ea_params=ea_params,
        stoich_matrix=stoich_matrix,
        partic_species=liquid.name_species,
        keq_params=equilibrium_constants,
        params_f=reaction_orders,
        delta_hrxn=delta_hrxn,
        tref_hrxn=TREF_HRXN,
    )
    reactor.Phases = liquid
    reactor.Kinetics = kinetics
    reactor.set_names()

    if isinstance(reactor, BatchReactor):
        reactor.conc_inert = liquid.mole_conc[~reactor.mask_species]  # [mol/L]
    else:
        inlet = LiquidStream(
            thermo_path,
            temp=TEMP_EVAL,
            vol_flow=INLET_VOL_FLOW,  # [m**3/s]
            mole_conc=mole_conc,
            verbose=False,
        )
        reactor.Inlet = inlet

    if isinstance(reactor, PlugFlowReactor):
        reactor.c_inert = liquid.mole_conc[~reactor.mask_species]  # [mol/L]
    elif not reactor.isothermal:
        reactor.Utility = CoolingWater(
            vol_flow=2.0e-5,  # [m**3/s]
            temp_in=TEMP_EVAL,  # [K]
        )
        if not isinstance(reactor, SemibatchReactor):
            design_volume = liquid.vol / reactor.vol_offset  # [m**3]
            reactor.diam = (4.0 / np.pi * design_volume) ** (1.0 / 3.0)  # [m]
            reactor.area_base = np.pi / 4.0 * reactor.diam**2  # [m**2]

    return reactor


def _raw_basis_source(reactor, mole_conc, temp):
    """Derive reaction heat from real raw-basis enthalpies and rates.

    Parameters
    ----------
    reactor : BatchReactor, CSTR, SemibatchReactor, or PlugFlowReactor
        Configured production reactor.
    mole_conc : ndarray
        Species concentrations with shape ``(4,)`` or ``(n_states, 4)``
        [mol/L].
    temp : float or ndarray
        Evaluation temperature [K].

    Returns
    -------
    ndarray
        Volumetric reaction heat source [W/m**3].
    """
    raw_heat = reactor.Liquid_1.getHeatOfRxn(
        reactor.Kinetics.stoich_matrix,
        temp,
        reactor.mask_species,
        reactor.Kinetics.delta_hrxn,
        reactor.Kinetics.tref_hrxn,
    )  # [J/mol of reaction as written]
    concentrations = np.asarray(mole_conc)
    participating = concentrations[..., reactor.mask_species]  # [mol/L]
    extent_rates = reactor.Kinetics.get_rxn_rates(
        participating,
        temp,
        overall_rates=False,
        delta_hrxn=raw_heat,
    )  # [mol/L/s]
    normalized_heat = (
        raw_heat / reactor.Kinetics.stoich_normalization
    )  # [J/mol of normalized reaction extent]
    return -np.sum(normalized_heat * extent_rates, axis=-1) * 1000.0  # [W/m**3]


def _batch(
        data_path, stoich_matrix, delta_hrxn, *, isothermal=True,
        k_params=K_PARAMS, ea_params=EA_PARAMS,
        reaction_orders=REACTION_ORDERS, mole_conc=MOLE_CONC,
        reversible=False):
    """Create a configured production batch reactor for a fixture.

    Parameters
    ----------
    data_path : dict
        Paths to repository test-data directories.
    stoich_matrix : ndarray
        Raw stoichiometric matrix [-].
    delta_hrxn : float or ndarray
        Raw-basis reaction enthalpies [J/mol].
    isothermal : bool, optional
        Whether temperature is prescribed [-].
    k_params : ndarray, optional
        Pre-exponential factors [(mol/L)**(1 - total forward order)/s].
    ea_params : ndarray, optional
        Activation energies [J/mol].
    reaction_orders : list of list of float, optional
        Forward reaction orders [-].
    mole_conc : ndarray, optional
        Species concentrations [mol/L].
    reversible : bool, optional
        Whether reactions include real reverse terms [-].

    Returns
    -------
    BatchReactor
        Configured production batch reactor.
    """
    return _build_reactor(
        data_path,
        BatchReactor(isothermal=isothermal),
        stoich_matrix,
        delta_hrxn,
        k_params=k_params,
        ea_params=ea_params,
        reaction_orders=reaction_orders,
        mole_conc=mole_conc,
        reversible=reversible,
    )


def _batch_heat_generation(reactor, mole_conc=MOLE_CONC):
    """Return positive exothermic batch heat generation [W].

    Parameters
    ----------
    reactor : BatchReactor
        Configured isothermal production reactor.
    mole_conc : ndarray, optional
        Species concentrations [mol/L].

    Returns
    -------
    float
        Positive exothermic reaction power [W].
    """
    profile = reactor.energy_balances(
        0.0, mole_conc, VOL_EVAL, TEMP_EVAL, TEMP_EVAL, {}, heat_prof=True
    )  # [W]
    # #263 aligns the stored reaction heat with the positive exothermic source.
    return float(profile[0, 0])  # [W]


def test_species_rates_are_invariant_to_stoichiometric_writing(data_path):
    """Equivalent raw reaction writings produce identical species rates."""
    as_written = _batch(
        data_path, STOICH_AS_WRITTEN, DELTA_HRXN_AS_WRITTEN
    )
    half_scaled = _batch(
        data_path, STOICH_HALF_SCALED, DELTA_HRXN_HALF_SCALED
    )

    rates_as_written = as_written.Kinetics.get_rxn_rates(
        MOLE_CONC, TEMP_EVAL
    )  # [mol/L/s]
    rates_half_scaled = half_scaled.Kinetics.get_rxn_rates(
        MOLE_CONC, TEMP_EVAL
    )  # [mol/L/s]
    expected_extent_rate = K_PARAMS[0] * MOLE_CONC[0]  # [mol/L/s]
    expected_species_rates = (
        np.array([-1.0, 0.5, 0.0, 0.0]) * expected_extent_rate
    )  # [mol/L/s]

    np.testing.assert_allclose(rates_as_written, expected_species_rates)
    np.testing.assert_allclose(rates_half_scaled, expected_species_rates)


def test_heat_of_reaction_matches_normalized_extent_basis(data_path):
    """Equivalent reaction writings release the same power."""
    as_written = _batch(
        data_path, STOICH_AS_WRITTEN, DELTA_HRXN_AS_WRITTEN
    )
    half_scaled = _batch(
        data_path, STOICH_HALF_SCALED, DELTA_HRXN_HALF_SCALED
    )
    expected_power = float(
        _raw_basis_source(half_scaled, MOLE_CONC, TEMP_EVAL) * VOL_EVAL
    )  # [W]

    assert _batch_heat_generation(half_scaled) == pytest.approx(expected_power)
    assert _batch_heat_generation(as_written) == pytest.approx(expected_power)


def test_multi_reaction_heat_keeps_reaction_axis_alignment(data_path):
    """Distinct normalization factors remain aligned by reaction."""
    common = {
        "k_params": MULTI_K_PARAMS,
        "ea_params": MULTI_EA_PARAMS,
        "reaction_orders": MULTI_REACTION_ORDERS,
        "mole_conc": MULTI_MOLE_CONC,
    }
    as_written = _batch(
        data_path,
        MULTI_STOICH_AS_WRITTEN,
        MULTI_DELTA_HRXN_AS_WRITTEN,
        **common,
    )
    normalized = _batch(
        data_path,
        MULTI_STOICH_NORMALIZED,
        MULTI_DELTA_HRXN_NORMALIZED,
        **common,
    )
    expected_power = float(
        _raw_basis_source(normalized, MULTI_MOLE_CONC, TEMP_EVAL) * VOL_EVAL
    )  # [W]

    np.testing.assert_allclose(
        as_written.Kinetics.stoich_normalization,
        MULTI_STOICH_NORMALIZATION,
    )
    assert _batch_heat_generation(
        normalized, MULTI_MOLE_CONC
    ) == pytest.approx(expected_power)
    assert _batch_heat_generation(
        as_written, MULTI_MOLE_CONC
    ) == pytest.approx(expected_power)


def test_batch_reported_heat_matches_temperature_source(data_path):
    """The reported heat and temperature derivative share one source."""
    reactor = _batch(
        data_path,
        STOICH_AS_WRITTEN,
        DELTA_HRXN_AS_WRITTEN,
        isothermal=False,
    )
    profile = reactor.energy_balances(
        0.0, MOLE_CONC, VOL_EVAL, TEMP_EVAL, TEMP_EVAL, {}, heat_prof=True
    )  # [W]
    temperature_rates = reactor.energy_balances(
        0.0, MOLE_CONC, VOL_EVAL, TEMP_EVAL, TEMP_EVAL, {}, heat_prof=False
    )  # [K/s]
    _, species_cp = reactor.Liquid_1.getCpPure(TEMP_EVAL)  # [J/mol/K]
    capacitance = VOL_EVAL * np.dot(
        MOLE_CONC * 1000.0, species_cp
    )  # [J/K]

    assert float(np.ravel(temperature_rates)[0]) == pytest.approx(
        profile[0, 0] / capacitance
    )


def test_batch_uses_normalized_heat_with_raw_equilibrium_handoff(data_path):
    """Batch reverse rates use raw enthalpy while heat uses normalized basis."""
    reactor = _batch(
        data_path,
        MULTI_STOICH_AS_WRITTEN,
        MULTI_DELTA_HRXN_AS_WRITTEN,
        k_params=REVERSIBLE_K_PARAMS,
        ea_params=MULTI_EA_PARAMS,
        reaction_orders=REVERSIBLE_REACTION_ORDERS,
        mole_conc=MULTI_MOLE_CONC,
        reversible=True,
    )
    expected_power = float(
        _raw_basis_source(reactor, MULTI_MOLE_CONC, TEMP_EVAL) * VOL_EVAL
    )  # [W]

    assert _batch_heat_generation(
        reactor, MULTI_MOLE_CONC
    ) == pytest.approx(expected_power)


@pytest.mark.parametrize("reactor_type", [CSTR, SemibatchReactor])
def test_tank_reactors_use_normalized_heat_with_raw_equilibrium_handoff(
        data_path, reactor_type):
    """Flow-through tank reactors preserve both reaction-energy bases."""
    if reactor_type is SemibatchReactor:
        reactor = reactor_type(vol_tank=3.0, isothermal=True)  # [m**3]
    else:
        reactor = reactor_type(isothermal=True)
    reactor = _build_reactor(
        data_path,
        reactor,
        MULTI_STOICH_AS_WRITTEN,
        MULTI_DELTA_HRXN_AS_WRITTEN,
        k_params=REVERSIBLE_K_PARAMS,
        ea_params=MULTI_EA_PARAMS,
        reaction_orders=REVERSIBLE_REACTION_ORDERS,
        mole_conc=MULTI_MOLE_CONC,
        reversible=True,
    )
    inputs = {"Inlet": {
        "vol_flow": reactor.Inlet.vol_flow,  # [m**3/s]
        "mole_conc": reactor.Inlet.mole_conc[np.newaxis, :],  # [mol/L]
        "temp": reactor.Inlet.temp,  # [K]
    }}
    profile = reactor.energy_balances(
        0.0,
        MULTI_MOLE_CONC[np.newaxis, :],
        VOL_EVAL,
        TEMP_EVAL,
        TEMP_EVAL,
        inputs,
        heat_prof=True,
    )  # [W]
    expected_source = float(
        _raw_basis_source(reactor, MULTI_MOLE_CONC, TEMP_EVAL) * VOL_EVAL
    )  # [W]

    assert profile[0, 0] == pytest.approx(expected_source)


def test_pfr_steady_energy_uses_normalized_heat_basis(data_path):
    """The production steady PFR uses normalized reaction heat."""
    reactor = _build_reactor(
        data_path,
        PlugFlowReactor(
            diam_in=1.0,  # [m]
            num_discr=3,  # [-]
            isothermal=False,
            adiabatic=True,
        ),
        MULTI_STOICH_AS_WRITTEN,
        MULTI_DELTA_HRXN_AS_WRITTEN,
        k_params=REVERSIBLE_K_PARAMS,
        ea_params=MULTI_EA_PARAMS,
        reaction_orders=REVERSIBLE_REACTION_ORDERS,
        mole_conc=MULTI_MOLE_CONC,
        reversible=True,
    )

    actual_rate = reactor.energy_steady(
        MULTI_MOLE_CONC, TEMP_EVAL
    )  # [K/m**3]
    _, species_cp = reactor.Liquid_1.getCpPure(TEMP_EVAL)  # [J/mol/K]
    heat_capacity_flow = reactor.Inlet.vol_flow * np.dot(
        species_cp, MULTI_MOLE_CONC
    ) * 1000.0  # [W/K]
    expected_rate = (
        _raw_basis_source(reactor, MULTI_MOLE_CONC, TEMP_EVAL)
        / heat_capacity_flow
    )  # [K/m**3]

    assert float(np.ravel(actual_rate)[0]) == pytest.approx(expected_rate)


def test_pfr_dynamic_energy_uses_normalized_heat_basis(data_path):
    """The production dynamic PFR uses normalized reaction heat."""
    reactor = _build_reactor(
        data_path,
        PlugFlowReactor(
            diam_in=1.0,  # [m]
            num_discr=3,  # [-]
            isothermal=False,
            adiabatic=True,
        ),
        MULTI_STOICH_AS_WRITTEN,
        MULTI_DELTA_HRXN_AS_WRITTEN,
        k_params=REVERSIBLE_K_PARAMS,
        ea_params=MULTI_EA_PARAMS,
        reaction_orders=REVERSIBLE_REACTION_ORDERS,
        mole_conc=MULTI_MOLE_CONC,
        reversible=True,
    )
    reactor.vol_discr = np.array([0.0, 1.0, 2.0])  # [m**3]
    volume_steps = np.diff(reactor.vol_discr)  # [m**3]
    temperatures = np.full(3, TEMP_EVAL)  # [K]
    concentrations = np.tile(MULTI_MOLE_CONC, (3, 1))  # [mol/L]
    raw_heat = reactor.Liquid_1.getHeatOfRxn(
        reactor.Kinetics.stoich_matrix,
        temperatures,
        reactor.mask_species,
        reactor.Kinetics.delta_hrxn,
        reactor.Kinetics.tref_hrxn,
    )  # [J/mol]
    extent_rates = reactor.Kinetics.get_rxn_rates(
        concentrations,
        temperatures,
        overall_rates=False,
        delta_hrxn=raw_heat,
    )  # [mol/L/s]

    actual_rates = reactor.energy_balances(
        0.0,
        concentrations,
        volume_steps,
        temperatures,
        reactor.Inlet.vol_flow,
        extent_rates,
        heat_profile=False,
    )  # [K/s]
    _, species_cp = reactor.Liquid_1.getCpPure(temperatures)  # [J/mol/K]
    volumetric_capacitance = (
        species_cp * concentrations
    ).sum(axis=1) * 1000.0  # [J/m**3/K]
    expected_sources = _raw_basis_source(
        reactor, concentrations, temperatures
    )  # [W/m**3]
    expected_rates = expected_sources[1:] / volumetric_capacitance[1:]  # [K/s]

    np.testing.assert_allclose(actual_rates, expected_rates)
