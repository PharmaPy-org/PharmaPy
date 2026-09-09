"""B010 evaporator contracts with synthetic thermodynamics and real phases.

Core tests exercise initialization, residuals, and synthetic result profiles.
Solver regressions are marked assimulo and use the optional IDA backend.
The synthetic UNIQUAC interaction is a contract fixture, not measured data.
"""

from copy import deepcopy
import json

import numpy as np
import pytest

from conftest import THERMO_TWO_SPECIES
from PharmaPy.Commons import unpack_states
from PharmaPy.Evaporators import AdiabaticFlash, ContinuousEvaporator, Evaporator
from PharmaPy.Phases import LiquidPhase
from PharmaPy.ProcessControl import DynamicInput
from PharmaPy.Streams import LiquidStream, VaporStream
from PharmaPy.Utilities import CoolingWater

PRESSURE = 1000.0  # [Pa], keeps both synthetic species subcritical at boiling
TEMPERATURE = 350.0  # [K], initial liquid and UNIQUAC reference temperature
FRACTIONS = np.array([0.4, 0.6])  # [-], asymmetric binary exposes ordering
VOLUME = 1.0  # [m**3], drum diameter is therefore different from 0.438 m
LIQUID_VOLUME = 0.2  # [m**3], partly filled drum
FEED = 2.0  # [mol/s], positive feed for the level-control tests
UTILITY_TEMP = 450.0  # [K], heating utility above both bubble points
RTOL = 1e-10  # [-], roundoff allowance for property and balance arithmetic
ATOL = 1e-9  # [mol/s or J/s], cancellation allowance in residual comparisons
GAS_CONSTANT = 8.314  # [J/mol/K], retained value of PharmaPy.Evaporators.gas_ct
DURATION = 1.0  # [s], short trajectory with negligible depletion


@pytest.fixture
def thermo_path(tmp_path):
    """Write a synthetic binary with equal UNIQUAC sizes and unlike tau=1/2.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Test directory inside the working tree (pytest --basetemp).

    Returns
    -------
    str
        Thermodynamic JSON path; unlike interaction [J/mol] is R*T*ln(2).
    """
    data = deepcopy(THERMO_TWO_SPECIES)
    for species in data.values():
        species.update(ri=1.0, qi=1.0, qip=1.0)  # [-], equal molecular sizes
    interaction = GAS_CONSTANT * TEMPERATURE * np.log(2)  # [J/mol], tau=1/2 at T
    data['interaction'] = {'amk': [[0, interaction], [interaction, 0]]}
    path = tmp_path / 'thermo.json'
    path.write_text(json.dumps(data))
    return str(path)


def make_unit(thermo_path, unit_class=Evaporator, **kwargs):
    """Construct a partly filled drum with finite utility resistance.

    Parameters
    ----------
    thermo_path : str
        Synthetic thermodynamic JSON path.
    unit_class : type
        Batch or continuous evaporator class.
    **kwargs : dict
        Constructor overrides with units defined by the evaporator API.

    Returns
    -------
    Evaporator or ContinuousEvaporator
        One cubic metre drum with 0.2 m**3 liquid and a heating utility.
    """
    if unit_class is ContinuousEvaporator:
        kwargs.setdefault('frac_liq', LIQUID_VOLUME / VOLUME)  # [-], initial level setpoint
    unit = unit_class(VOLUME, pressure=PRESSURE, **kwargs)
    unit.Phases = LiquidPhase(thermo_path, temp=TEMPERATURE, pres=PRESSURE,
                              vol=LIQUID_VOLUME, mole_frac=FRACTIONS)
    unit.Utility = CoolingWater(mass_flow=1.0, temp_in=UTILITY_TEMP)  # [kg/s], [K]
    if unit_class is ContinuousEvaporator:
        unit.Inlet = LiquidStream(thermo_path, temp=TEMPERATURE, pres=PRESSURE,
                                   mole_flow=FEED, mole_frac=FRACTIONS)
    return unit


@pytest.mark.parametrize('stream_class', [LiquidStream, VaporStream])
@pytest.mark.unit
def test_static_dynamic_inlet_can_be_cleared(thermo_path, stream_class):
    stream = stream_class(thermo_path, mole_frac=FRACTIONS, mole_flow=FEED)
    control = DynamicInput()
    stream.DynamicInlet = control
    assert control.parent_instance is stream
    stream.DynamicInlet = None
    assert stream.DynamicInlet is None
    assert stream.evaluate_inputs(0)['mole_flow'] == pytest.approx(FEED, rel=RTOL)

@pytest.mark.unit
def test_adiabatic_flash_accepts_liquid_stream(thermo_path):
    inlet = LiquidStream(thermo_path, mole_frac=FRACTIONS, mole_flow=FEED)
    flash = AdiabaticFlash(PRESSURE)
    flash.Inlet = inlet
    np.testing.assert_allclose(flash.VaporOut.mole_frac, FRACTIONS, rtol=RTOL)
    assert flash.in_flow == pytest.approx(FEED, rel=RTOL)


@pytest.mark.parametrize('unit_class', [Evaporator, ContinuousEvaporator])
@pytest.mark.unit
def test_seed_heat_matches_overall_residual_heat(thermo_path, unit_class):
    unit = make_unit(thermo_path, unit_class)
    states, derivatives = unit.init_unit()  # units and order in states_di
    values = unpack_states(states, unit.dim_states, unit.name_states)
    # Cylinder side area = circumference * liquid height, plus heated base.
    radius = unit.diam_tank / 2  # [m]
    height = LIQUID_VOLUME / (np.pi * radius**2)  # [m]
    area = 2 * np.pi * radius * height + np.pi * radius**2  # [m**2]
    overall = unit.h_conv * unit.Utility.h_conv / (unit.h_conv + unit.Utility.h_conv)  # [W/m**2/K]
    expected_heat = overall * area * (UTILITY_TEMP - values['temp'])  # [J/s]
    expected_energy = (
        values['mol_liq'] * unit.Liquid_1.getEnthalpy(values['temp'], mole_frac=values['x_liq'], basis='mole')
        + values['mol_vap'] * unit.Vapor_1.getEnthalpy(values['temp'], mole_frac=values['y_vap'], basis='mole')
        - PRESSURE * VOLUME)  # [J], U = sum(N*h) - P*V
    assert values['u_int'] == pytest.approx(expected_energy, rel=RTOL)
    inputs = unit.get_inputs(0)['Inlet']
    if unit_class is Evaporator:
        expected_rate = expected_heat  # [J/s], batch has no feed
        heat = unit.energy_balances(0, LIQUID_VOLUME, 0, **values, u_inputs=inputs)
    else:
        flow_liq = max(np.finfo(float).eps,
                       unit.k_liq * (LIQUID_VOLUME - unit.vol_liq_set) + FEED)  # [mol/s]
        flow_energy = (FEED * unit.Inlet.getEnthalpy(basis='mole')
                       - flow_liq * unit.Liquid_1.getEnthalpy(values['temp'], basis='mole'))  # [J/s]
        expected_rate = expected_heat + flow_energy  # [J/s]
        _, loss = unit.energy_balances(0, 0, 0, LIQUID_VOLUME,
                                      **values, u_inputs=inputs, heat_prof=True)
        heat = -loss  # [J/s], heat entering drum
    assert unit.diam_tank != pytest.approx(0.438)  # [m], historical wrong diameter
    assert derivatives[-2] == pytest.approx(expected_rate, rel=RTOL, abs=ATOL)
    assert heat == pytest.approx(expected_heat, rel=RTOL)


@pytest.mark.parametrize('unit_class', [Evaporator, ContinuousEvaporator])
@pytest.mark.unit
def test_bubble_seed_uses_configured_activity_model(thermo_path, unit_class):
    ideal = make_unit(thermo_path, unit_class)
    nonideal = make_unit(thermo_path, unit_class, activity_model='UNIQUAC')
    ideal_states, _ = ideal.init_unit()
    states, _ = nonideal.init_unit()
    assert states[-1] < ideal_states[-1]
    # Verify equilibrium directly, independently of getBubblePoint.
    gamma = nonideal.Liquid_1.UNIQUAC(mole_frac=FRACTIONS, temp=states[-1])  # [-]
    partial_pressures = FRACTIONS * gamma * nonideal.Liquid_1.AntoineEquation(states[-1])  # [Pa]
    assert partial_pressures.sum() == pytest.approx(PRESSURE, rel=RTOL)
    values = unpack_states(states, nonideal.dim_states, nonideal.name_states)
    np.testing.assert_allclose(values['y_vap'], partial_pressures / PRESSURE, rtol=RTOL)


@pytest.mark.parametrize('unit_class', [Evaporator, ContinuousEvaporator])
@pytest.mark.unit
def test_headspace_seed_uses_bubble_temperature(thermo_path, unit_class):
    unit = make_unit(thermo_path, unit_class)
    states, derivatives = unit.init_unit()
    values = unpack_states(states, unit.dim_states, unit.name_states)
    expected_moles = PRESSURE * (VOLUME - LIQUID_VOLUME) / (GAS_CONSTANT * values['temp'])  # [mol], ideal gas law
    assert values['mol_vap'] == pytest.approx(expected_moles, rel=RTOL)
    if unit_class is ContinuousEvaporator:
        assert unit.Vapor_1.temp == pytest.approx(values['temp'], rel=RTOL)
        assert unit.Vapor_1.pres == pytest.approx(PRESSURE, rel=RTOL)
        assert unit.Vapor_1.vol == pytest.approx(VOLUME - LIQUID_VOLUME, rel=RTOL)


@pytest.mark.parametrize('unit_class', [Evaporator, ContinuousEvaporator])
@pytest.mark.unit
def test_rhs_stores_vapor_composition_without_corrupting_flow(thermo_path, unit_class):
    unit = make_unit(thermo_path, unit_class)
    states, derivatives = unit.init_unit()
    values = unpack_states(states, unit.dim_states, unit.name_states)
    # Reorder fractions without changing their sum, exposing a stale composition.
    values['y_vap'][:] = values['y_vap'][::-1]
    unit.is_supercritic = np.zeros(len(FRACTIONS), dtype=bool)
    vapor_flow = unit.Vapor_1.mole_flow  # [mol/s], scalar stream alias
    unit.unit_model(0, states, derivatives, None)
    assert np.ndim(unit.Vapor_1.mole_flow) == 0
    assert unit.Vapor_1.mole_flow == pytest.approx(vapor_flow, rel=RTOL)
    np.testing.assert_allclose(unit.Vapor_1.mole_frac, values['y_vap'], rtol=RTOL)


@pytest.mark.parametrize('reflux', [0.0, 0.3])  # [-], no reflux and partial condensate return
@pytest.mark.parametrize('feed_factor', [2.0, 1.0, 0.5])  # [-], above, at, below net vapor removal
@pytest.mark.unit
def test_reflux_level_control_conserves_material(thermo_path, reflux, feed_factor):
    unit = make_unit(thermo_path, ContinuousEvaporator, reflux_ratio=reflux,
                     frac_liq=LIQUID_VOLUME / VOLUME)
    states, _ = unit.init_unit()
    values = unpack_states(states, unit.dim_states, unit.name_states)
    values['pres'] = 2 * PRESSURE  # [Pa], positive pressure drop to drive vapor
    vapor_density = values['pres'] / (GAS_CONSTANT * values['temp'])  # [mol/m**3]
    mass_density = vapor_density * np.dot(unit.Vapor_1.mw, values['y_vap']) / 1000  # [kg/m**3]
    velocity = np.sqrt(2 * PRESSURE / mass_density)  # [m/s], Bernoulli outlet
    vapor_flow = vapor_density * unit.area_out * velocity * unit.cv_gas * unit.k_vap  # [mol/s]
    net_vapor = (1 - reflux) * vapor_flow  # [mol/s], condensate return retained
    input_flow = feed_factor * net_vapor  # [mol/s]
    inputs = {'mole_flow': input_flow, 'mole_frac': FRACTIONS}
    rates, _, liquid_flow, actual_vapor, _ = unit.material_balances(
        0, **values, u_inputs=inputs)
    # The level-feedback term vanishes because this fixture is at the setpoint.
    expected_liquid = max(0, input_flow - net_vapor)  # [mol/s]
    assert liquid_flow == pytest.approx(expected_liquid, abs=ATOL)
    assert actual_vapor == pytest.approx(vapor_flow, rel=RTOL)
    expected_accumulation = min(0, input_flow - net_vapor)  # [mol/s], depletion if feed is insufficient
    assert rates.sum() == pytest.approx(expected_accumulation, abs=ATOL)
    if reflux == 0:
        old_liquid = max(0, unit.k_liq * (LIQUID_VOLUME - unit.vol_liq_set)
                         + input_flow - vapor_flow)  # [mol/s], pre-B010 equation
        old_rates = input_flow * FRACTIONS - old_liquid * FRACTIONS - vapor_flow * values['y_vap']  # [mol/s]
        np.testing.assert_allclose(rates, old_rates, rtol=RTOL, atol=ATOL)

@pytest.mark.unit
def test_real_steady_state_has_small_residual(thermo_path):
    unit = make_unit(thermo_path, ContinuousEvaporator)
    seed, _ = unit.init_unit()
    # Inverse initial magnitudes scale the mixed mol, fraction, Pa, J and K
    # unknowns to order one. All seed entries are nonzero in this fixture.
    inverse_scales = 1 / abs(seed)  # [inverse packed-state units]
    states = unit.solve_unit(DURATION, steady_state=True,
                             fsolve_opts={'diag': inverse_scales, 'xtol': RTOL})
    residual = unit.unit_model(0, states, None, None)
    # Each equation must close within one micro-unit in its native basis:
    # mol/s, mol, mole fraction, m**3, Pa, J/s, and J, respectively.
    residual_atol = 1e-6  # [native residual units], numerical closure requirement
    assert residual.shape == states.shape == (3 * len(FRACTIONS) + 5,)
    assert np.linalg.norm(residual, ord=np.inf) < residual_atol
    values = unpack_states(states, unit.dim_states, unit.name_states)
    assert values['x_liq'].sum() == pytest.approx(1, rel=RTOL)
    assert values['y_vap'].sum() == pytest.approx(1, rel=RTOL)

@pytest.mark.unit
def test_nonconverging_steady_state_raises(thermo_path):
    unit = make_unit(thermo_path, ContinuousEvaporator)
    # Two evaluations cannot resolve this coupled eleven-state system.
    with pytest.raises(RuntimeError, match=r'ier=2.*maxfev'):
        unit.solve_unit(DURATION, steady_state=True, fsolve_opts={'maxfev': 2})

@pytest.mark.unit
def test_absolute_result_times_accumulate_once(thermo_path):
    unit = make_unit(thermo_path)
    states, _ = unit.init_unit()
    for segment in range(3):
        start = unit.elapsed_time  # [s]
        times = np.array([start, start + DURATION])  # [s], absolute solver times
        unit.retrieve_results(times, np.tile(states, (2, 1)))
        assert unit.elapsed_time == pytest.approx((segment + 1) * DURATION, rel=RTOL)
    np.testing.assert_allclose(unit.result.time, np.arange(4) * DURATION, rtol=RTOL)

@pytest.mark.unit
@pytest.mark.parametrize('dynamic', [False, True])
def test_nitrogen_layout_and_postprocessing_preserve_phase_species(thermo_path, dynamic):
    unit = make_unit(thermo_path, include_nitrogen=True)
    inlet = LiquidStream(thermo_path, temp=TEMPERATURE, pres=PRESSURE,
                         mole_flow=FEED, mole_frac=FRACTIONS)
    if dynamic:
        inlet.DynamicInlet = DynamicInput()
    unit.Inlet = inlet
    original_phase = unit.__original_phase__
    states, _ = unit.init_unit()
    assert sum(unit.dim_states) == len(states)
    expected_names = ['light', 'heavy', 'nitrogen']
    for field in ('mol_i', 'x_liq', 'y_vap'):
        assert unit.states_di[field]['index'] == expected_names
    assert unit.Phases[0] is unit.Liquid_1
    assert unit.Phases[1] is unit.Vapor_1
    values = unpack_states(states, unit.dim_states, unit.name_states)
    assert values['x_liq'][-1] == 0
    assert values['y_vap'][-1] > 0
    if dynamic:
        assert inlet.DynamicInlet.parent_instance is unit.Inlet
    else:
        assert unit.Inlet.DynamicInlet is None
    assert unit.get_inputs(0)['Inlet']['mole_frac'].shape == (3,)
    unit.retrieve_results(np.array([0, DURATION]), np.tile(states, (2, 1)))
    assert unit.Liquid_1.name_species == ['light', 'heavy']
    assert unit.Inlet is inlet
    if dynamic:
        assert inlet.DynamicInlet.parent_instance is inlet
    assert unit.__original_phase__ is original_phase
    assert unit.result.y_vap.shape == (2, 3)
    np.testing.assert_allclose(unit.result.y_vap[-1], values['y_vap'], rtol=RTOL)
    np.testing.assert_allclose(unit.Outlet.mole_frac, values['x_liq'][:-1], rtol=RTOL)
    assert np.isfinite(unit.heat_duty).all()
    # Another initialization retains the original feed and adds only one N2.
    next_states, _ = unit.init_unit()
    assert len(next_states) == len(states)
    assert unit.Liquid_1.name_species == expected_names
    assert unit.Inlet is inlet
    assert unit.Inlet.mole_frac.shape == (2,)
    assert unit.get_inputs(DURATION)['Inlet']['mole_frac'].shape == (3,)


@pytest.mark.integration
@pytest.mark.assimulo
@pytest.mark.parametrize('unit_class', [Evaporator, ContinuousEvaporator])
def test_real_ida_evaporator_run(thermo_path, unit_class):
    pytest.importorskip('assimulo')
    unit = make_unit(thermo_path, unit_class)
    time, states = unit.solve_unit(DURATION, verbose=False)
    assert time[-1] == pytest.approx(DURATION, rel=RTOL)
    assert states.shape[1] == sum(unit.dim_states)
    assert np.isfinite(states).all()
    values = unpack_states(states[-1], unit.dim_states, unit.name_states)
    reconstructed = (values['mol_liq'] * values['x_liq']
                     + values['mol_vap'] * values['y_vap'])  # [mol], component inventory closure
    np.testing.assert_allclose(values['mol_i'], reconstructed, rtol=RTOL)
    assert values['x_liq'].sum() == pytest.approx(1, rel=RTOL)
    assert values['y_vap'].sum() == pytest.approx(1, rel=RTOL)
    assert unit.result.mol_liq[-1] > 0
    assert np.isfinite(unit.heat_duty).all()


@pytest.mark.unit
@pytest.mark.parametrize('unit_class', [Evaporator, ContinuousEvaporator])
def test_segment_duties_use_local_states_and_accumulate(thermo_path, unit_class):
    unit = make_unit(thermo_path, unit_class)
    seed, _ = unit.init_unit()
    expected_total = np.zeros(2)  # [J], heating/removal and condensation energies
    for segment in range(2):
        values = unpack_states(seed.copy(), unit.dim_states, unit.name_states)
        values['x_liq'][:] = FRACTIONS if segment == 0 else FRACTIONS[::-1]
        values['y_vap'][:] = FRACTIONS[::-1] if segment == 0 else FRACTIONS
        values['temp'] += segment * 20  # [K], separated temperatures expose stale rows
        values['pres'] = (segment + 2) * PRESSURE  # [Pa], distinct nonzero vapor flows
        row = np.concatenate([np.atleast_1d(values[name]) for name in unit.name_states])
        times = np.array([segment, segment + 1]) * DURATION  # [s]
        profiles = np.tile(row, (2, 1))  # units in states_di; constant segment states
        rho_liq = unit.Liquid_1.getDensity(temp=values['temp'], mole_frac=values['x_liq'], basis='mole')  # [mol/L]
        liquid_volume = values['mol_liq'] / (1000 * rho_liq)  # [m**3]
        area = unit.area_base + np.pi * unit.diam_tank * liquid_volume / unit.area_base  # [m**2], base + wetted wall
        heat_into_drum = unit.u_ht * area * (UTILITY_TEMP - values['temp'])  # [J/s]
        vapor_density = values['pres'] / (GAS_CONSTANT * values['temp'])  # [mol/m**3]
        gas_density = vapor_density * np.dot(unit.Vapor_1.mw, values['y_vap']) / 1000  # [kg/m**3]
        velocity = np.sqrt(2 * (values['pres'] - PRESSURE) / gas_density)  # [m/s]
        vapor_flow = vapor_density * unit.area_out * velocity * unit.cv_gas * unit.k_vap  # [mol/s]
        condensate_frac = values['x_liq'] if unit_class is Evaporator else values['y_vap']  # [-], continuous condenser has vapor composition
        bubble_temp = unit.Liquid_1.getBubblePoint(pres=values['pres'], mole_frac=condensate_frac)  # [K]
        liquid_h = unit.Liquid_1.getEnthalpy(temp=bubble_temp, mole_frac=condensate_frac, basis='mole')  # [J/mol]
        vapor_h = unit.Vapor_1.getEnthalpy(temp=values['temp'], mole_frac=values['y_vap'], basis='mole')  # [J/mol]
        condensation = vapor_flow * (liquid_h - vapor_h)  # [J/s]
        expected_power = np.array([heat_into_drum if unit_class is Evaporator else -heat_into_drum,
                                   condensation])  # [J/s], established class sign conventions
        unit.retrieve_results(times, profiles)
        np.testing.assert_allclose(unit.heat_profile, np.tile(expected_power, (2, 1)), rtol=RTOL)
        segment_energy = expected_power * DURATION  # [J], exact rectangular integral
        expected_total += segment_energy if unit_class is Evaporator else abs(segment_energy)
        np.testing.assert_allclose(unit.heat_duty, expected_total, rtol=RTOL)


@pytest.mark.integration
@pytest.mark.assimulo
@pytest.mark.parametrize('nitrogen', [False, True])
def test_batch_continuation_matches_uninterrupted_run(thermo_path, nitrogen, monkeypatch):
    pytest.importorskip('assimulo')
    segment_duration = 50.0  # [s], long enough to expose re-flash energy discontinuity
    solver_rtol = 1e-8  # [-], requested relative integration accuracy
    # Trace nitrogen falls below the solver's default absolute tolerance.
    # Resolve it to 1e-10 mol (and the corresponding native units for the
    # other states), without relaxing the comparison tolerance below.
    options = {'rtol': solver_rtol, 'atol': 1e-10}  # [-], [native state units]
    whole = make_unit(thermo_path, include_nitrogen=nitrogen)
    split = make_unit(thermo_path, include_nitrogen=nitrogen)
    # Request exact communication points from the real IDA solver; the unit
    # API otherwise returns only adaptive steps, not the shared endpoint.
    from assimulo.solvers import IDA
    class EndpointIDA(IDA):
        """Real IDA solver returning the comparison endpoint explicitly."""

        def simulate(self, final_time):
            """Run IDA with the reference midpoint included in its returned profile.

            Parameters
            ----------
            final_time : float
                Absolute final time [s].

            Returns
            -------
            tuple
                IDA times [s], states, and derivatives in evaporator state units.
            """
            assert final_time == 2 * segment_duration
            return super().simulate(final_time,
                            ncp_list=[segment_duration, final_time])
    with monkeypatch.context() as patch:
        patch.setattr('PharmaPy.Evaporators.IDA', EndpointIDA)
        whole_time, whole_states = whole.solve_unit(2 * segment_duration, verbose=False,
                                                   sundials_opts=options)
    first_time, first_states = split.solve_unit(segment_duration, verbose=False,
                                               sundials_opts=options)
    terminal = first_states[-1].copy()  # units and order in states_di
    second_time, second_states = split.solve_unit(segment_duration, verbose=False,
                                                 sundials_opts=options)
    # Continuation must copy all terminal states without re-flashing.
    np.testing.assert_allclose(second_states[0], terminal, rtol=solver_rtol)
    # Independent adaptive meshes have accumulated local errors; allow ten
    # requested relative tolerances for comparison of the two trajectories.
    comparison_rtol = 10 * solver_rtol  # [-], accumulated integration error budget
    shared = np.flatnonzero(np.asarray(whole_time) == segment_duration)
    assert len(shared) == 1
    np.testing.assert_allclose(terminal, whole_states[shared[0]], rtol=comparison_rtol,
                               atol=solver_rtol)
    np.testing.assert_allclose(second_states[-1], whole_states[-1], rtol=comparison_rtol,
                               atol=solver_rtol)
    assert first_time[-1] == pytest.approx(segment_duration, rel=RTOL)
    assert second_time[0] == pytest.approx(segment_duration, rel=RTOL)
    assert second_time[-1] == whole_time[-1] == pytest.approx(2 * segment_duration, rel=RTOL)
    assert split.elapsed_time == pytest.approx(2 * segment_duration, rel=RTOL)


@pytest.mark.integration
@pytest.mark.assimulo
def test_nitrogen_initial_residual_is_square_and_ida_integrates(thermo_path):
    pytest.importorskip('assimulo')
    unit = make_unit(thermo_path, include_nitrogen=True)
    # Capture IDA's consistent initial state through the public simulation.
    time, states = unit.solve_unit(DURATION, verbose=False)
    assert states.shape[1] == sum(unit.dim_states) == 14
    # Reinstall the augmented phases before evaluating the initial residual.
    unit.init_unit()
    zeros = np.zeros_like(states[0])  # state derivative units in init_unit
    residual = unit.unit_model(time[0], states[0], zeros, None)
    assert residual.shape == states[0].shape
    derivatives = np.zeros_like(zeros)
    derivatives[:3] = residual[:3]  # [mol/s], balance-determined differential rates
    derivatives[-2] = residual[-2]  # [J/s]
    consistent_residual = unit.unit_model(time[0], states[0], derivatives, None)
    closure_atol = 1e-6  # [native residual units], IDA algebraic initialization accuracy
    np.testing.assert_allclose(consistent_residual, 0, atol=closure_atol, rtol=0)
    assert time[-1] == pytest.approx(DURATION, rel=RTOL)
    assert np.isfinite(states).all()

@pytest.mark.unit
@pytest.mark.parametrize('reflux', [-0.1, 1.1, np.nan, np.inf, -np.inf])  # [-], outside fraction domain
def test_reflux_ratio_rejects_invalid_fractions(reflux):
    with pytest.raises(ValueError, match=r'reflux_ratio.*\[0, 1\]'):
        ContinuousEvaporator(VOLUME, reflux_ratio=reflux)

@pytest.mark.unit
@pytest.mark.parametrize('reflux', [0.0, 1.0])  # [-], closed-interval endpoints
def test_reflux_ratio_accepts_endpoints(reflux):
    unit = ContinuousEvaporator(VOLUME, reflux_ratio=reflux)
    assert unit.reflux_ratio == reflux

@pytest.mark.unit
def test_utility_dynamic_inlet_can_be_cleared():
    utility = CoolingWater(mass_flow=1.0, temp_in=UTILITY_TEMP)  # [kg/s], [K]
    control = DynamicInput()
    utility.DynamicInlet = control
    assert control.parent_instance is utility
    utility.DynamicInlet = None
    assert utility.DynamicInlet is None
    inputs = utility.get_inputs(0)
    assert inputs['temp_in'] == pytest.approx(UTILITY_TEMP, rel=RTOL)
    assert inputs['mass_flow'] == pytest.approx(utility.mass_flow, rel=RTOL)

@pytest.mark.unit
@pytest.mark.parametrize('excluded_index', [0, 1])
def test_excluded_species_closure_retains_species_order(thermo_path, excluded_index):
    unit = make_unit(thermo_path)
    states, _ = unit.init_unit()
    values = unpack_states(states, unit.dim_states, unit.name_states)
    unit.is_supercritic = np.arange(len(FRACTIONS)) == excluded_index
    unit.Liquid_1.henry_constant = np.full(len(FRACTIONS), np.nan)  # [Pa], missing data must not poison excluded rows
    unit.Liquid_1.t_crit[excluded_index] = values['temp'] / 2  # [K], force exclusion of either non-nitrogen species
    result = unit.material_balances(0, **values, u_inputs=unit.get_inputs(0)['Inlet'],
                                    dmoli_dt=np.zeros(len(FRACTIONS)))
    assert len(result[1]) == 2 * len(FRACTIONS) + 3
    closures = result[1][len(FRACTIONS):2 * len(FRACTIONS)]  # [-], one row per species
    assert closures[excluded_index] == pytest.approx(values['x_liq'][excluded_index], rel=RTOL)
    included = 1 - excluded_index
    expected_vle = (values['y_vap'][included] - values['x_liq'][included]
                    * unit.Liquid_1.AntoineEquation(values['temp'])[included] / values['pres'])  # [-], ideal Raoult closure
    assert closures[included] == pytest.approx(expected_vle, abs=RTOL)



@pytest.mark.unit
def test_replacing_batch_charge_invalidates_continuation(thermo_path):
    unit = make_unit(thermo_path)
    seed, _ = unit.init_unit()
    unit.retrieve_results(np.array([0, DURATION]), np.tile(seed, (2, 1)))
    replacement_fractions = np.array([0.7, 0.3])  # [-], distinct replacement charge
    replacement = LiquidPhase(thermo_path, temp=TEMPERATURE, pres=PRESSURE,
                               vol=LIQUID_VOLUME / 2, mole_frac=replacement_fractions)
    replacement_moles = replacement.moles  # [mol], independently constructed new charge
    unit.Phases = replacement
    states, _ = unit.init_unit()
    values = unpack_states(states, unit.dim_states, unit.name_states)
    assert values['mol_liq'] == pytest.approx(replacement_moles, rel=RTOL)
    np.testing.assert_allclose(values['x_liq'], replacement_fractions, rtol=RTOL)
    assert unit.elapsed_time == 0
    assert unit._terminal_states is None
    assert unit.profiles_runs == []
    np.testing.assert_array_equal(unit.heat_duty, np.zeros(2))


@pytest.mark.integration
@pytest.mark.assimulo
@pytest.mark.parametrize('unit_class, nitrogen', [
    (Evaporator, False), (Evaporator, True), (ContinuousEvaporator, False)])
def test_reset_then_solve_matches_fresh_unit(thermo_path, unit_class, nitrogen):
    pytest.importorskip('assimulo')
    options = {'include_nitrogen': nitrogen} if unit_class is Evaporator else {}
    unit = make_unit(thermo_path, unit_class, **options)
    fresh = make_unit(thermo_path, unit_class, **options)
    unit.solve_unit(DURATION, verbose=False)
    unit.reset()
    if unit_class is Evaporator:
        assert unit.elapsed_time == 0
        assert unit._terminal_states is None
    else:
        np.testing.assert_array_equal(unit._signed_heat_duty, np.zeros(2))
    assert unit.profiles_runs == []
    np.testing.assert_array_equal(unit.heat_duty, np.zeros(2))
    np.testing.assert_allclose(unit.Liquid_1.mole_frac, FRACTIONS, rtol=RTOL)
    assert unit.Liquid_1.vol == pytest.approx(LIQUID_VOLUME, rel=RTOL)
    times, states = unit.solve_unit(DURATION, verbose=False)
    fresh_times, fresh_states = fresh.solve_unit(DURATION, verbose=False)
    np.testing.assert_allclose(times, fresh_times, rtol=RTOL)
    np.testing.assert_allclose(states, fresh_states, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(unit.heat_duty, fresh.heat_duty, rtol=RTOL)


@pytest.mark.unit
@pytest.mark.parametrize('adiabatic', [False, True])
@pytest.mark.parametrize('reflux', [0.0, 0.3])  # [-], bypass and active reflux condenser
def test_continuous_reports_each_physical_duty_once(thermo_path, adiabatic, reflux):
    unit = make_unit(thermo_path, ContinuousEvaporator, adiabatic=adiabatic, reflux_ratio=reflux)
    states, _ = unit.init_unit()
    values = unpack_states(states, unit.dim_states, unit.name_states)
    values['pres'] = 2 * PRESSURE  # [Pa], ensure positive gross vapor discharge
    row = np.concatenate([np.atleast_1d(values[name]) for name in unit.name_states])
    times = np.array([0, DURATION])  # [s], constant synthetic segment
    molar_density = values['pres'] / (GAS_CONSTANT * values['temp'])  # [mol/m**3]
    mass_density = molar_density * np.dot(unit.Vapor_1.mw, values['y_vap']) / 1000  # [kg/m**3]
    velocity = np.sqrt(2 * PRESSURE / mass_density)  # [m/s]
    vapor_flow = molar_density * unit.area_out * velocity * unit.cv_gas * unit.k_vap  # [mol/s]
    condensate_temperature = unit.Liquid_1.getBubblePoint(pres=values['pres'], mole_frac=values['y_vap'])  # [K]
    condensate_enthalpy = unit.Liquid_1.getEnthalpy(temp=condensate_temperature, mole_frac=values['y_vap'], basis='mole')  # [J/mol]
    vapor_enthalpy = unit.Vapor_1.getEnthalpy(temp=values['temp'], mole_frac=values['y_vap'], basis='mole')  # [J/mol]
    condenser_power = vapor_flow * (condensate_enthalpy - vapor_enthalpy)  # [J/s], negative for cooling vapor
    area = unit.area_base + np.pi * unit.diam_tank * LIQUID_VOLUME / unit.area_base  # [m**2]
    jacket_power = 0 if adiabatic else unit.u_ht * area * (values['temp'] - UTILITY_TEMP)  # [J/s], positive removal
    unit.retrieve_results(times, np.tile(row, (2, 1)))
    expected_power = np.array([jacket_power, condenser_power])  # [J/s], separate physical duties
    np.testing.assert_allclose(unit.heat_profile, np.tile(expected_power, (2, 1)), rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(unit.heat_duty, abs(expected_power * DURATION), rtol=RTOL, atol=ATOL)


@pytest.mark.unit
def test_continuous_signed_duty_is_independent_of_segment_partition(thermo_path):
    whole = make_unit(thermo_path, ContinuousEvaporator)
    split = make_unit(thermo_path, ContinuousEvaporator)
    states, _ = whole.init_unit()
    temperature = states[-1]  # [K], fixed drum temperature
    slope = 1.0  # [K/s], symmetric utility ramp crossing drum temperature at 1 s

    def utility_temperature(time):
        """Return a linear utility-temperature ramp through the drum temperature.

        Parameters
        ----------
        time : float or ndarray
            Time [s].

        Returns
        -------
        float or ndarray
            Utility inlet temperature [K].
        """
        return temperature + slope * (time - DURATION)

    for unit in (whole, split):
        control = DynamicInput()
        control.add_variable('temp_in', utility_temperature)
        unit.Utility.DynamicInlet = control
    times = np.arange(3) * DURATION  # [s], equally spaced about the zero crossing
    profiles = np.tile(states, (3, 1))
    whole.retrieve_results(times, profiles)
    split.retrieve_results(times[:2], profiles[:2])
    assert split.heat_duty[0] > 0
    split.retrieve_results(times[1:], profiles[1:])
    # Equal triangular signed areas cancel exactly, independently of partition.
    assert whole.heat_duty[0] == pytest.approx(0, abs=ATOL)
    assert split.heat_duty[0] == pytest.approx(0, abs=ATOL)
    np.testing.assert_allclose(split.heat_duty, whole.heat_duty, rtol=RTOL, atol=ATOL)


@pytest.mark.unit
def test_batch_integer_time_array_preserves_fractional_heat_rates(thermo_path):
    unit = make_unit(thermo_path)
    states, _ = unit.init_unit()
    radius = unit.diam_tank / 2  # [m]
    area = np.pi * radius**2 + 2 * LIQUID_VOLUME / radius  # [m**2], base plus cylindrical wall
    expected_heat = unit.u_ht * area * (UTILITY_TEMP - states[-1])  # [J/s]
    states[-3] = 2 * PRESSURE  # [Pa], positive discharge exposes truncated vapor flow
    values = unpack_states(states, unit.dim_states, unit.name_states)
    molar_density = values['pres'] / (GAS_CONSTANT * values['temp'])  # [mol/m**3]
    mass_density = molar_density * np.dot(unit.Vapor_1.mw, values['y_vap']) / 1000  # [kg/m**3]
    velocity = np.sqrt(2 * PRESSURE / mass_density)  # [m/s]
    vapor_flow = molar_density * unit.area_out * velocity * unit.cv_gas * unit.k_vap  # [mol/s]
    bubble_temp = unit.Liquid_1.getBubblePoint(pres=values['pres'], mole_frac=values['x_liq'])  # [K]
    liquid_h = unit.Liquid_1.getEnthalpy(temp=bubble_temp, mole_frac=values['x_liq'], basis='mole')  # [J/mol]
    vapor_h = unit.Vapor_1.getEnthalpy(temp=values['temp'], mole_frac=values['y_vap'], basis='mole')  # [J/mol]
    expected_condensation = vapor_flow * (liquid_h - vapor_h)  # [J/s], existing batch reporting basis
    times = np.array([0, 1])  # [s], deliberately integer dtype
    unit.retrieve_results(times, np.tile(states, (2, 1)))
    expected = np.array([expected_heat, expected_condensation])  # [J/s]
    np.testing.assert_allclose(unit.heat_profile, np.tile(expected, (2, 1)), rtol=RTOL)


@pytest.mark.unit
def test_batch_continuation_updates_vapor_inventory_without_flow_alias(thermo_path):
    unit = make_unit(thermo_path)
    states, _ = unit.init_unit()
    vapor_flow = unit.Vapor_1.mole_flow  # [mol/s], unrelated placeholder flow
    unit.retrieve_results(np.array([0, DURATION]), np.tile(states, (2, 1)))
    resumed, _ = unit.init_unit()
    values = unpack_states(resumed, unit.dim_states, unit.name_states)
    assert unit.Vapor_1.moles == pytest.approx(values['mol_vap'], rel=RTOL)
    assert unit.Vapor_1.vol == pytest.approx(VOLUME - LIQUID_VOLUME, rel=RTOL)
    assert unit.Vapor_1.mole_flow == pytest.approx(vapor_flow, rel=RTOL)



@pytest.mark.unit
@pytest.mark.parametrize('unit_class', [Evaporator, ContinuousEvaporator])
def test_reset_requires_attached_charge(unit_class):
    unit = unit_class(VOLUME)
    with pytest.raises(RuntimeError, match='Phases.*before.*reset'):
        unit.reset()


@pytest.mark.unit
@pytest.mark.parametrize('restart', ['replace', 'reset'])
def test_continuous_new_charge_clears_run_and_matches_fresh_duties(thermo_path, restart):
    unit = make_unit(thermo_path, ContinuousEvaporator)
    seed, _ = unit.init_unit()
    times = np.array([0, DURATION])  # [s], constant synthetic segment
    unit.retrieve_results(times, np.tile(seed, (2, 1)))
    assert np.any(unit._signed_heat_duty != 0)
    fresh = make_unit(thermo_path, ContinuousEvaporator)
    if restart == 'replace':
        fractions = np.array([0.7, 0.3])  # [-], distinguish replacement charge
        charge = LiquidPhase(thermo_path, temp=TEMPERATURE, pres=PRESSURE,
                             vol=LIQUID_VOLUME / 2, mole_frac=fractions)
        unit.Phases = deepcopy(charge)
        fresh.Phases = charge
    else:
        unit.reset()
    np.testing.assert_array_equal(unit._signed_heat_duty, np.zeros(2))
    np.testing.assert_array_equal(unit.heat_duty, np.zeros(2))
    assert unit.profiles_runs == []
    assert unit.outputs is None
    assert unit.tau is None
    for name in ('result', 'Outlet', 'heat_profile', 'liqFlowProf', 'vapFlowProf'):
        assert not hasattr(unit, name)
    seed, _ = unit.init_unit()
    fresh_seed, _ = fresh.init_unit()
    np.testing.assert_allclose(seed, fresh_seed, rtol=RTOL, atol=ATOL)
    unit.retrieve_results(times, np.tile(seed, (2, 1)))
    fresh.retrieve_results(times, np.tile(fresh_seed, (2, 1)))
    np.testing.assert_allclose(unit.heat_duty, fresh.heat_duty, rtol=RTOL, atol=ATOL)
    assert len(unit.profiles_runs) == 1


@pytest.mark.unit
@pytest.mark.parametrize('activity_model', ['ideal', 'UNIQUAC'])
@pytest.mark.parametrize('site', ['batch_reporting', 'continuous_energy', 'continuous_reporting'])
def test_condensate_temperature_uses_configured_activity_model(
        thermo_path, monkeypatch, activity_model, site):
    unit_class = Evaporator if site == 'batch_reporting' else ContinuousEvaporator
    options = {} if unit_class is Evaporator else {'reflux_ratio': 0.3}  # [-], active reflux
    unit = make_unit(thermo_path, unit_class, activity_model=activity_model, **options)
    seed, _ = unit.init_unit()
    seed[-3] = 2 * PRESSURE  # [Pa], positive vapor discharge
    values = unpack_states(seed, unit.dim_states, unit.name_states)
    composition = values['x_liq' if unit_class is Evaporator else 'y_vap']  # [-]
    bubble_point = unit.Liquid_1.getBubblePoint
    expected = bubble_point(pres=values['pres'], mole_frac=composition,
                            thermo_method=activity_model)  # [K]
    ideal = bubble_point(pres=values['pres'], mole_frac=composition)  # [K]
    if activity_model == 'UNIQUAC':
        assert not np.isclose(expected, ideal, rtol=RTOL)
    temperatures = []  # [K], real bubble-point outputs consumed by the model

    def record_bubble_point(*args, **kwargs):
        """Record the real condensate temperature without replacing thermodynamics.

        Parameters
        ----------
        *args, **kwargs
            LiquidPhase.getBubblePoint inputs, including pressure [Pa].

        Returns
        -------
        float
            Bubble-point temperature [K].
        """
        temperature = bubble_point(*args, **kwargs)  # [K]
        temperatures.append(temperature)
        return temperature

    monkeypatch.setattr(unit.Liquid_1, 'getBubblePoint', record_bubble_point)
    if site == 'continuous_energy':
        unit.unit_model(0, seed, states_dot=None, sw=None, enrgy_bce=True)
    else:
        unit.get_heat_duty(np.array([0, DURATION]), np.tile(seed, (2, 1)))
    assert temperatures
    np.testing.assert_allclose(temperatures, expected, rtol=RTOL)


@pytest.mark.integration
@pytest.mark.assimulo
@pytest.mark.parametrize('source', ['controller', 'upstream'])
def test_nitrogen_semibatch_evaluates_original_composition_source(thermo_path, source):
    pytest.importorskip('assimulo')
    unit = make_unit(thermo_path, include_nitrogen=True)
    inlet = LiquidStream(thermo_path, temp=TEMPERATURE, pres=PRESSURE,
                         mole_flow=FEED, mole_frac=FRACTIONS)
    controller = DynamicInput()
    final_fractions = FRACTIONS[::-1].copy()  # [-], asymmetric composition ramp
    controller_owners = []

    def composition(time):
        """Interpolate the original binary feed over the simulation duration.

        Parameters
        ----------
        time : float or ndarray
            Absolute input time [s].

        Returns
        -------
        ndarray
            Original-species mole fractions [-], shape (2,) or (num_times, 2).
        """
        controller_owners.append(controller.parent_instance)
        progress = np.asarray(time)[..., None] / DURATION  # [-]
        return FRACTIONS + progress * (final_fractions - FRACTIONS)

    if source == 'controller':
        controller.add_variable('mole_frac', composition)
        inlet.DynamicInlet = controller
    else:
        inlet.time_upstream = np.array([0, DURATION / 2, DURATION])  # [s], linear ramp samples
        inlet.y_inlet = {'mole_frac': composition(inlet.time_upstream)}  # [-]
        inlet.y_upstream = inlet.y_inlet
    unit.Inlet = inlet
    times, states = unit.solve_unit(DURATION, verbose=False)
    assert times[-1] == pytest.approx(DURATION, rel=RTOL)
    assert np.isfinite(states).all()
    assert unit.Inlet is inlet
    if source == 'controller':
        assert controller.parent_instance is inlet
        assert controller_owners and all(owner is inlet for owner in controller_owners)
    inputs = unit.get_inputs(np.array([0, DURATION]))['Inlet']
    expected = np.column_stack((np.vstack((FRACTIONS, final_fractions)), np.zeros(2)))  # [-]
    np.testing.assert_allclose(inputs['mole_frac'], expected, rtol=RTOL)
    if source == 'controller':
        np.testing.assert_allclose(controller.evaluate_inputs(DURATION)['mole_frac'],
                                   final_fractions, rtol=RTOL)
    else:
        assert unit.Inlet.y_inlet is inlet.y_upstream
        np.testing.assert_allclose(inlet.y_inlet['mole_frac'][-1], final_fractions, rtol=RTOL)


@pytest.mark.integration
@pytest.mark.assimulo
def test_nitrogen_continuation_reads_updated_public_inlet(thermo_path):
    pytest.importorskip('assimulo')
    unit = make_unit(thermo_path, include_nitrogen=True)
    inlet = LiquidStream(thermo_path, temp=TEMPERATURE, pres=PRESSURE,
                         mole_flow=FEED, mole_frac=FRACTIONS)
    unit.Inlet = inlet
    # Resolve the small nitrogen inventory on both otherwise identical runs.
    options = {'rtol': 1e-8, 'atol': 1e-10}  # [-], [native packed-state units]
    unit.solve_unit(DURATION, verbose=False, sundials_opts=options)
    reference = deepcopy(unit)
    changed_flow = 2 * FEED  # [mol/s], double feed after the first segment
    inlet.updatePhase(mole_flow=changed_flow)
    reference.Inlet = deepcopy(inlet)
    times, states = unit.solve_unit(DURATION, verbose=False, sundials_opts=options)
    reference_times, reference_states = reference.solve_unit(
        DURATION, verbose=False, sundials_opts=options)
    assert unit.Inlet is inlet
    assert times[0] == pytest.approx(DURATION, rel=RTOL)
    assert times[-1] == pytest.approx(2 * DURATION, rel=RTOL)
    np.testing.assert_allclose(states[-1], reference_states[-1], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(times, reference_times, rtol=RTOL)
    np.testing.assert_allclose(states, reference_states, rtol=RTOL, atol=ATOL)
    assert unit.get_inputs(times[-1])['Inlet']['mole_flow'] == pytest.approx(changed_flow, rel=RTOL)


@pytest.mark.unit
@pytest.mark.parametrize('source', ['controller', 'upstream'])
def test_nitrogen_feed_mapping_preserves_current_values_and_axes(thermo_path, source):
    unit = make_unit(thermo_path, include_nitrogen=True)
    inlet = LiquidStream(thermo_path, temp=TEMPERATURE, pres=PRESSURE,
                         mole_flow=FEED, mole_frac=FRACTIONS)
    times = np.arange(4) * DURATION  # [s], four samples differ from three packed species
    fractions = np.array([[0.7, 0.3], [0.6, 0.4],
                          [0.5, 0.5], [0.4, 0.6]])  # [-], linear binary feed profile
    if source == 'controller':
        controller = DynamicInput()

        def composition(time):
            """Evaluate the binary feed's linear composition ramp.

            Parameters
            ----------
            time : float or ndarray
                Absolute input time [s].

            Returns
            -------
            ndarray
                Binary mole fractions [-], shape (2,) or (num_times, 2).
            """
            progress = np.asarray(time)[..., None] / times[-1]  # [-]
            return fractions[0] + progress * (fractions[-1] - fractions[0])

        controller.add_variable('mole_frac', composition)
        inlet.DynamicInlet = controller
    else:
        inlet.time_upstream = times
        inlet.y_inlet = {'mole_frac': fractions}
        inlet.y_upstream = inlet.y_inlet
    unit.Inlet = inlet
    states, derivatives = unit.init_unit()
    expected = np.column_stack((fractions, np.zeros(len(times))))  # [-], no feed nitrogen
    np.testing.assert_allclose(unit.get_inputs(times)['Inlet']['mole_frac'], expected, rtol=RTOL)
    np.testing.assert_allclose(unit.get_inputs(0)['Inlet']['mole_frac'], expected[0], rtol=RTOL)
    np.testing.assert_allclose(derivatives[:3], FEED * expected[0], rtol=RTOL)
    assert unit.Inlet is inlet
    # A live flow update is visible immediately, including to continuation.
    changed_flow = 2 * FEED  # [mol/s], double the original flow
    inlet.updatePhase(mole_flow=changed_flow)
    assert unit.get_inputs(DURATION)['Inlet']['mole_flow'] == pytest.approx(changed_flow, rel=RTOL)
    if source == 'controller':
        assert controller.parent_instance is inlet
    else:
        assert inlet.y_inlet['mole_frac'] is fractions
        assert fractions.shape == (4, 2)
