"""Regression coverage for non-isothermal batch-reactor sensitivities.

The core tests exercise Jacobian assembly without Assimulo. The optional
integration test confirms the same callbacks through ``solve_unit`` when the
solver backend is installed.
"""

import importlib.util
from unittest.mock import Mock

import numpy as np
import pytest

from PharmaPy.Kinetics import RxnKinetics
from PharmaPy.Phases import LiquidPhase
from PharmaPy.Reactors import BatchReactor
from PharmaPy.Utilities import CoolingWater


HAS_ASSIMULO = importlib.util.find_spec("assimulo") is not None
_FIVE_POINT_REL_STEP = np.finfo(float).eps ** (1 / 5)  # [-], O(h**4) optimum


def _build_nonisothermal_reactor(data_path):
    """Build a jacketed batch reactor with unequal species concentrations.

    Parameters
    ----------
    data_path : dict
        Repository test-data paths [-].

    Returns
    -------
    reactor : PharmaPy.Reactors.BatchReactor
        Configured non-isothermal batch reactor.
    states : numpy.ndarray
        Species concentrations followed by reactor and jacket temperatures
        [mol/L for the first three entries; K for the last two].
    params : numpy.ndarray
        Concatenated kinetic parameters [parameter-dependent units].

    Notes
    -----
    The explicit floating-point stoichiometric matrix avoids the fractional-
    order truncation tracked by issue #44. Once #44 is fixed, this fixture can
    use integer stoichiometry while retaining the expected fractional order.
    """
    thermo_path = str(
        data_path["integration"] / "pfr_test_pure_comp.json")
    mole_conc = np.array([0.15, 0.20, 0.01, 0.64])  # [mol/L]
    reactor_temp = 305.0  # [K]
    liquid_volume = 0.002  # [m**3]
    phase = LiquidPhase(
        thermo_path,
        mole_conc=mole_conc,
        temp=reactor_temp,
        vol=liquid_volume,
        name_solv="solv",
        verbose=False,
    )

    rate_constant = 40 / 60  # [1/s], converted exactly from 40 1/min
    activation_energy = 2.0e3  # [J/mol]
    reaction_enthalpy = -5.0e3  # [J/mol of reaction as written]
    stoich_matrix = np.array([[-1.0, -1.0, 1.0]])  # [-]
    reaction_orders = np.array(
        [[0.5, 1.0]])  # [-], fractional A order exercises the zero boundary
    kinetics = RxnKinetics(
        thermo_path,
        stoich_matrix=stoich_matrix,
        k_params=[rate_constant],
        ea_params=[activation_energy],
        partic_species=["A", "B", "C"],
        params_f=reaction_orders,
        delta_hrxn=[reaction_enthalpy],
    )

    utility_temp = 300.0  # [K]
    utility_mass_flow = 0.1  # [kg/s]
    utility = CoolingWater(
        temp_in=utility_temp, mass_flow=utility_mass_flow)

    reactor = BatchReactor(isothermal=False, ht_mode="jacket")
    reactor.Phases = phase
    reactor.Kinetics = kinetics
    reactor.Utility = utility
    reactor.set_names()
    reactor.conc_inert = phase.mole_conc[~reactor.mask_species]  # [mol/L]

    vessel_volume = liquid_volume / reactor.vol_offset  # [m**3]
    reactor.diam = (4 / np.pi * vessel_volume) ** (1 / 3)  # [m]
    reactor.area_base = np.pi / 4 * reactor.diam**2  # [m**2]

    states = np.array(
        [*mole_conc[reactor.mask_species], reactor_temp, utility_temp]
    )  # [mol/L for species; K for temperatures]
    params = kinetics.concat_params().copy()  # [parameter-dependent units]

    return reactor, states, params


def _five_point_jacobian(func, values, args=()):
    """Approximate a vector-valued Jacobian with a five-point stencil.

    Parameters
    ----------
    func : callable
        Function mapping ``values`` to model balance rates.
    values : array-like
        Differentiated coordinates [coordinate-dependent units].
    args : tuple, optional
        Additional function arguments [argument-dependent units].

    Returns
    -------
    numpy.ndarray
        Fourth-order derivative estimate [output-rate units / coordinate
        units].

    Notes
    -----
    The relative step ``machine_epsilon**(1/5)`` balances fourth-order
    truncation error against floating-point roundoff. This differs from the
    production two-point stencil and therefore provides an independent
    numerical expectation.
    """
    coordinates = np.asarray(
        values, dtype=float)  # [units of each differentiated coordinate]
    steps = _FIVE_POINT_REL_STEP * np.maximum(
        1.0, np.abs(coordinates))  # [units of each differentiated coordinate]
    base_rates = np.atleast_1d(func(coordinates, *args))  # [mixed rate units]
    jacobian = np.empty(
        (base_rates.size, coordinates.size))  # [mixed rate/coordinate units]

    for index, step in enumerate(steps):  # step [coordinate-dependent units]
        perturbation = np.zeros_like(coordinates)  # [coordinate-dependent units]
        perturbation[index] = step
        rate_plus_two = np.atleast_1d(
            func(coordinates + 2 * perturbation, *args))  # [mixed rate units]
        rate_plus_one = np.atleast_1d(
            func(coordinates + perturbation, *args))  # [mixed rate units]
        rate_minus_one = np.atleast_1d(
            func(coordinates - perturbation, *args))  # [mixed rate units]
        rate_minus_two = np.atleast_1d(
            func(coordinates - 2 * perturbation, *args))  # [mixed rate units]
        jacobian[:, index] = (
            -rate_plus_two + 8 * rate_plus_one
            - 8 * rate_minus_one + rate_minus_two
        ) / (12 * step)

    return jacobian


def _unclipped_power_law(concentrations, reaction_orders):
    """Evaluate a normalized power law without clipping negative inputs.

    Parameters
    ----------
    concentrations : array-like
        Participating-species concentrations [mol/L].
    reaction_orders : numpy.ndarray
        Reaction orders with shape ``(n_reactions, n_species)`` [-].

    Returns
    -------
    numpy.ndarray
        Per-reaction concentration terms [mol/L].

    Notes
    -----
    A 1 mol/L reference concentration preserves the rate-term unit while
    making fractional powers dimensionless. The deliberate absence of a
    concentration floor makes a negative perturbation observable in the
    zero-boundary regression.
    """
    reference_concentration = 1.0  # [mol/L], exact normalization basis
    concentration_ratios = (
        np.asarray(concentrations) / reference_concentration
    )  # [-]
    rate_terms = reference_concentration * np.prod(
        concentration_ratios**reaction_orders, axis=-1
    )  # [mol/L]

    return rate_terms


def _balances_at_states(states, reactor, time, params):
    """Evaluate reactor balances at candidate states and fixed parameters.

    Parameters
    ----------
    states : array-like
        Species concentrations and thermal states [mol/L; K].
    reactor : PharmaPy.Reactors.BatchReactor
        Configured reactor model.
    time : float
        Integration time [s].
    params : array-like
        Fixed kinetic parameters [parameter-dependent units].

    Returns
    -------
    numpy.ndarray
        Species and thermal balance rates [mol/L/s; K/s].
    """
    reactor.Kinetics.set_params(params)
    return np.asarray(reactor.unit_model(time, states, params=params))


def _balances_at_params(params, reactor, time, states):
    """Evaluate reactor balances at fixed states and candidate parameters.

    Parameters
    ----------
    params : array-like
        Candidate kinetic parameters [parameter-dependent units].
    reactor : PharmaPy.Reactors.BatchReactor
        Configured reactor model.
    time : float
        Integration time [s].
    states : array-like
        Fixed species concentrations and thermal states [mol/L; K].

    Returns
    -------
    numpy.ndarray
        Species and thermal balance rates [mol/L/s; K/s].
    """
    reactor.Kinetics.set_params(params)
    return np.asarray(reactor.unit_model(time, states, params=params))


def _assert_jacobians_agree(actual, expected):
    """Assert agreement within the independent stencil's relative step.

    Parameters
    ----------
    actual : numpy.ndarray
        Production Jacobian [mixed row-rate / column-coordinate units].
    expected : numpy.ndarray
        Independent Jacobian [mixed row-rate / column-coordinate units].

    Returns
    -------
    None
        The assertion succeeds when every scaled error is below the
        five-point perturbation fraction.
    """
    scale = np.maximum(
        1.0, np.maximum(np.abs(actual), np.abs(expected))
    )  # [units of each Jacobian entry]
    scaled_error = np.abs(actual - expected) / scale  # [-]
    assert np.max(scaled_error) < _FIVE_POINT_REL_STEP


@pytest.mark.unit
def test_nonisothermal_state_jacobian_covers_species_and_thermal_blocks(
        data_path):
    """Match all state derivatives and pin the physical jacket couplings."""
    reactor, states, params = _build_nonisothermal_reactor(data_path)
    time = 0.0  # [s]

    expected_orders = np.array([[0.5, 1.0, 0.0]])  # [-]
    np.testing.assert_array_equal(reactor.Kinetics.params_f, expected_orders)

    actual = reactor.get_jacobians(
        time, states, None, None, params, wrt_states=True
    )  # [mixed rate/state units]
    expected = _five_point_jacobian(
        _balances_at_states, states, args=(reactor, time, params)
    )  # [mixed rate/state units]
    reactor.unit_model(time, states, params=params)

    assert actual.shape == (states.size, states.size)
    assert np.all(np.isfinite(actual))
    _assert_jacobians_agree(actual, expected)

    reactor_temp_index = reactor.Kinetics.num_species
    jacket_temp_index = reactor_temp_index + 1
    species_state_block = actual[
        :reactor_temp_index, :reactor_temp_index
    ]  # [1/s]
    assert np.all(np.any(species_state_block[:, :2] != 0, axis=0))
    np.testing.assert_array_equal(species_state_block[:, 2], 0)
    assert np.any(actual[:reactor_temp_index, reactor_temp_index] != 0)
    assert np.any(actual[reactor_temp_index, :reactor_temp_index] != 0)
    assert actual[reactor_temp_index, jacket_temp_index] > 0
    assert actual[jacket_temp_index, reactor_temp_index] > 0
    assert actual[jacket_temp_index, jacket_temp_index] < 0


@pytest.mark.unit
def test_nonisothermal_parameter_sensitivity_includes_energy_row(data_path):
    """Match direct parameter derivatives and retain zero jacket coupling."""
    reactor, states, params = _build_nonisothermal_reactor(data_path)
    time = 0.0  # [s]
    sensitivities = np.zeros(
        (states.size, params.size)
    )  # [state units / parameter units]

    actual = reactor.get_jacobians(
        time, states, None, sensitivities, params, wrt_states=False
    )  # [state units / parameter units / s]
    original_params = reactor.Kinetics.concat_params().copy(
    )  # [parameter-dependent units]
    try:
        expected = _five_point_jacobian(
            _balances_at_params, params, args=(reactor, time, states)
        )  # [state units / parameter units / s]
    finally:
        reactor.Kinetics.set_params(original_params)
        reactor.unit_model(time, states, params=original_params)

    assert actual.shape == (states.size, params.size)
    assert np.all(np.isfinite(actual))
    _assert_jacobians_agree(actual, expected)

    reactor_temp_index = reactor.Kinetics.num_species
    jacket_temp_index = reactor_temp_index + 1
    assert np.any(actual[reactor_temp_index] != 0)
    np.testing.assert_array_equal(actual[jacket_temp_index], 0)


@pytest.mark.unit
def test_nonisothermal_sensitivity_rhs_couples_state_jacobian(data_path):
    """Assert ``df/dy @ S + df/dtheta`` with asymmetric sensitivities."""
    reactor, states, params = _build_nonisothermal_reactor(data_path)
    time = 0.0  # [s]
    sensitivities = np.arange(
        1.0, states.size * params.size + 1.0
    ).reshape(states.size, params.size)  # [state units / parameter units]

    actual = reactor.get_jacobians(
        time, states, None, sensitivities, params, wrt_states=False
    )  # [state units / parameter units / s]

    original_params = reactor.Kinetics.concat_params().copy(
    )  # [parameter-dependent units]
    try:
        expected_states = _five_point_jacobian(
            _balances_at_states, states, args=(reactor, time, params)
        )  # [mixed rate/state units]
        expected_params = _five_point_jacobian(
            _balances_at_params, params, args=(reactor, time, states)
        )  # [state units / parameter units / s]
    finally:
        reactor.Kinetics.set_params(original_params)
        reactor.unit_model(time, states, params=original_params)

    expected = (
        expected_states.dot(sensitivities) + expected_params
    )  # [state units / parameter units / s]
    _assert_jacobians_agree(actual, expected)


@pytest.mark.unit
def test_nonisothermal_state_jacobian_stays_in_concentration_domain(
        data_path):
    """Exercise the one-sided boundary through the shipped kinetic model."""
    reactor, states, params = _build_nonisothermal_reactor(data_path)
    fractional_order = 1.5  # [-]
    params[-2] = fractional_order
    reactor.Kinetics.set_params(params)
    production_model = reactor.Kinetics.kinetic_model
    recording_model = Mock(wraps=production_model)
    reactor.Kinetics.kinetic_model = recording_model
    states[0] = 0.0  # [mol/L], depleted reactant boundary
    time = 0.0  # [s]

    jacobian = reactor.get_jacobians(
        time, states, None, None, params, wrt_states=True
    )  # [mixed rate/state units]

    evaluated_concentrations = [
        np.asarray(call.args[0]) for call in recording_model.call_args_list
    ]  # [mol/L]
    minimum_valid_concentration = 0.0  # [mol/L]
    assert evaluated_concentrations
    assert all(
        np.all(concentrations >= minimum_valid_concentration)
        for concentrations in evaluated_concentrations
    )
    assert np.all(np.isfinite(jacobian))


@pytest.mark.unit
def test_nonisothermal_state_jacobian_is_finite_at_zero_concentration(
        data_path):
    """Keep a valid unclipped custom power law finite at the boundary."""
    reactor, states, params = _build_nonisothermal_reactor(data_path)
    fractional_order = 1.5  # [-], defined power law with zero derivative at zero
    params[-2] = fractional_order
    reactor.Kinetics.set_params(params)
    reactor.Kinetics.kinetic_model = _unclipped_power_law
    states[0] = 0.0  # [mol/L], depleted reactant boundary
    time = 0.0  # [s]

    jacobian = reactor.get_jacobians(
        time, states, None, None, params, wrt_states=True
    )  # [mixed rate/state units]

    assert jacobian.shape == (states.size, states.size)
    assert np.all(np.isfinite(jacobian))


@pytest.mark.assimulo
@pytest.mark.integration
@pytest.mark.skipif(
    not HAS_ASSIMULO,
    reason="assimulo is not installed; solver-backed integration test skipped",
)
def test_nonisothermal_jacket_sensitivities_run_through_solve_unit(data_path):
    """Exercise the repaired Jacobians through the public solver path."""
    reactor, _, params = _build_nonisothermal_reactor(data_path)
    time_grid = np.array([0.0, 0.05, 0.10])  # [s]

    time, states, sensitivities = reactor.solve_unit(
        time_grid=time_grid, eval_sens=True, verbose=False)
    # `time` [s]; `states` [mol/L and K]; `sensitivities` [state/parameter].

    assert np.asarray(time).shape == time_grid.shape
    expected_state_count = reactor.Kinetics.num_species + 2
    assert np.asarray(states).shape == (time_grid.size, expected_state_count)
    assert len(sensitivities) == params.size
    assert all(
        np.asarray(sensitivity).shape == (time_grid.size, states.shape[1])
        for sensitivity in sensitivities
    )
    assert all(np.all(np.isfinite(sensitivity)) for sensitivity in sensitivities)
