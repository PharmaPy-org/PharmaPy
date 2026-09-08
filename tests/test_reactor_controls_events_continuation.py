"""Contracts for #232–#234 using real liquid/kinetic collaborators.

Core cases drive the RHS and retrieval directly; only solver construction is
intercepted for callback registration. Marked cases use the real CVode backend.
Synthetic ramps and profiles test balances, not calibrated process predictions.
"""

from types import SimpleNamespace
import json

import numpy as np
import pytest
from scipy.interpolate import interp1d

from PharmaPy import Reactors
from PharmaPy.ProcessControl import DynamicInput
from PharmaPy.Commons import (eval_state_events, handle_events, TerminateSimulation,
                             unpack_states, check_steady_state)
from test_reactor_correctness import (
    _configured_reactor, CONCENTRATIONS, TEMPERATURE, REACTOR_VOLUME,
    THERMO_PATH, RATE_CONSTANT, REACTION_HEAT, HEAT_REFERENCE_TEMPERATURE,
    ALGEBRA_RTOL, RESIDENCE_TIME, UTILITY_TEMPERATURE,
)

TANKS = [Reactors.CSTR, Reactors.SemibatchReactor]
ALL_TANKS = [Reactors.BatchReactor, *TANKS]
TIMES = np.array([0., 1., 2.])  # [s], equally spaced synthetic profile
PROFILE_CONVERSION_RATE = 1e-3  # [mol/L/s], synthetic two-second A-to-C trajectory
RAMP = 0.5  # [K/s], linear control with an exact finite-difference derivative


def configured(cls, **kwargs):
    """Build a tank with the shared synthetic mixture.

    Parameters
    ----------
    cls : type
        Tank reactor class.
    **kwargs : dict
        Constructor settings; control temperatures are [K].

    Returns
    -------
    _BaseReactor
        Tank ready for RHS and profile evaluation.
    """
    if cls is Reactors.SemibatchReactor:
        kwargs['vol_tank'] = REACTOR_VOLUME * 2  # [m**3], twice the charge
    reactor = _configured_reactor(cls(**kwargs))
    reactor.set_names()
    reactor.conc_inert = CONCENTRATIONS[~reactor.mask_species]  # [mol/L]
    tank_vol = REACTOR_VOLUME / reactor.vol_offset  # [m**3], existing geometry
    reactor.diam = (4 * tank_vol / np.pi)**(1 / 3)  # [m]
    reactor.area_base = np.pi * reactor.diam**2 / 4  # [m**2]
    return reactor


def profile(reactor, time=TIMES):
    """Pack an asymmetric synthetic profile in reactor state order.

    Parameters
    ----------
    reactor : _BaseReactor
        Configured tank.
    time : numpy.ndarray
        Absolute profile times [s].

    Returns
    -------
    numpy.ndarray
        Packed concentrations [mol/L], volume [m**3] and temperatures [K].
    """
    conc = np.tile(CONCENTRATIONS, (len(time), 1))  # [mol/L]
    # A -> C conversion changes the terminal composition without empty columns.
    extent = time * PROFILE_CONVERSION_RATE  # [mol/L], synthetic trajectory only
    conc[:, 0] -= extent
    conc[:, 2] += extent
    if isinstance(reactor, Reactors.BatchReactor):
        conc = conc[:, reactor.mask_species]
    data = {'mole_conc': conc,
            'vol': REACTOR_VOLUME + time * REACTOR_VOLUME / RESIDENCE_TIME,
            'temp': TEMPERATURE + RAMP * time,
            'temp_ht': np.full(len(time), TEMPERATURE)}  # [mol/L], [m**3], [K], [K]
    return np.column_stack([data[name] for name in reactor.name_states])


def independent_terms(reactor, time, states, temperature=None):
    """Evaluate fixture energy terms from database Cp polynomials.

    Parameters
    ----------
    reactor : _BaseReactor
        Tank with zero activation energy A + B -> C kinetics.
    time : numpy.ndarray
        Profile times [s].
    states : numpy.ndarray
        Packed profile, beginning with concentrations [mol/L].
    temperature : numpy.ndarray or None, optional
        Prescribed temperatures [K]; defaults to the fixture linear ramp.

    Returns
    -------
    tuple of numpy.ndarray
        Reaction heat [W], inlet sensible heat [W], thermal capacitance [J/K].
    """
    with open(THERMO_PATH) as stream:
        properties = json.load(stream)
    temp = TEMPERATURE + RAMP * time if temperature is None else temperature  # [K]
    conc = np.tile(CONCENTRATIONS, (len(time), 1))  # [mol/L]
    count = len(reactor.Kinetics.partic_species) if isinstance(reactor, Reactors.BatchReactor) else len(CONCENTRATIONS)
    conc[:, :count] = states[:, :count]
    cp = np.column_stack([
        np.polynomial.Polynomial(properties[name]['cp_liq'])(temp)
        for name in properties])  # [J/mol/K]
    h = np.column_stack([
        np.polynomial.Polynomial(properties[name]['cp_liq']).integ()(temp)
        - np.polynomial.Polynomial(properties[name]['cp_liq']).integ()(HEAT_REFERENCE_TEMPERATURE)
        for name in properties])  # [J/mol]
    h_ref = np.array([
        np.polynomial.Polynomial(properties[name]['cp_liq']).integ()(TEMPERATURE)
        - np.polynomial.Polynomial(properties[name]['cp_liq']).integ()(HEAT_REFERENCE_TEMPERATURE)
        for name in properties])  # [J/mol]
    delta_h = REACTION_HEAT + h[:, 2] - h[:, 0] - h[:, 1]  # [J/mol reaction]
    vol = np.full(len(time), REACTOR_VOLUME)  # [m**3]
    flow_heat = np.zeros(len(time))  # [W]
    if not isinstance(reactor, Reactors.BatchReactor):
        inlet_flow = reactor.Inlet.vol_flow  # [m**3/s]
        if isinstance(reactor, Reactors.SemibatchReactor):
            vol = states[:, 4]  # [m**3]
            outlet_conc = CONCENTRATIONS  # [mol/L], feed displacement term
        else:
            outlet_conc = conc  # [mol/L]
        flow_heat = inlet_flow * 1000 * (
            np.sum(CONCENTRATIONS * h_ref) - np.sum(outlet_conc * h, axis=1))  # [W]
    reaction_heat = -delta_h * RATE_CONSTANT * conc[:, 0] * conc[:, 1] * vol * 1000  # [W]
    capacity = vol * 1000 * np.sum(conc * cp, axis=1)  # [J/K]
    return reaction_heat, flow_heat, capacity


@pytest.mark.unit
@pytest.mark.parametrize('cls', ALL_TANKS)
@pytest.mark.parametrize('record', [False, True])
def test_prescribed_temperature_rhs_and_energy_duty(cls, record):
    control = lambda time: TEMPERATURE + RAMP * time
    if record:
        control = {'fun': lambda time, slope, offset: offset + slope * time,
                   'args': (RAMP,), 'kwargs': {'offset': TEMPERATURE}}
    reactor = configured(cls, isothermal=False, controls={'temp': control})
    states = profile(reactor)  # [mol/L], optional [m**3]
    actual = reactor.unit_model(TIMES[0], states[0])  # [mol/L/s], optional [m**3/s]
    assert actual.shape == states[0].shape
    assert 'temp' not in reactor.name_states
    reactor.retrieve_results(TIMES, states)
    np.testing.assert_allclose(reactor.result.temp, TEMPERATURE + RAMP * TIMES, rtol=ALGEBRA_RTOL)
    reaction, flow, capacity = independent_terms(reactor, TIMES, states)  # [W], [W], [J/K]
    expected = capacity * RAMP - reaction - flow  # [W], heat added to liquid
    np.testing.assert_allclose(reactor.result.q_rxn, reaction, rtol=ALGEBRA_RTOL)
    np.testing.assert_allclose(reactor.result.q_ht, expected, rtol=ALGEBRA_RTOL)
    assert reactor.heat_duty[0] == pytest.approx(sum(np.diff(TIMES) * (expected[1:] + expected[:-1]) / 2), rel=ALGEBRA_RTOL)


@pytest.mark.unit
@pytest.mark.parametrize('record', [
    {'fun': lambda time: time, 'typo': 1},
    {'fun': lambda time: time, 'args': (), 'kwargs': {}, 'typo': 1}])
def test_control_record_keys_validated(record):
    with pytest.raises(ValueError, match='fun.*args.*kwargs'):
        Reactors.BatchReactor(controls={'temp': record})


@pytest.mark.unit
@pytest.mark.parametrize('nodes, components, expected', [
    (3, 2, [-2., -4., -6.]), (2, 3, [-2., -5.])])
def test_distributed_event_component_selection(nodes, components, expected):
    states = np.arange(1., nodes * components + 1)  # [mol/L], unique per node/component
    event = {'state_name': 'mole_conc', 'state_idx': 1, 'value': 0.,
             'num_conditions': nodes}  # concentration threshold [mol/L]
    actual = eval_state_events(0., states, [True], [components], ['mole_conc'],
                               [event], discretized_model=True)  # [mol/L]
    np.testing.assert_array_equal(actual, expected)
    event['node_idx'] = nodes - 1
    event['num_conditions'] = 1
    actual = eval_state_events(0., states, [True], [components], ['mole_conc'],
                               [event], discretized_model=True)  # [mol/L]
    np.testing.assert_array_equal(actual, expected[-1:])


@pytest.mark.unit
def test_vector_condition_owner_and_direction(capsys):
    events = [{'num_conditions': 3, 'direction': -1, 'event_name': 'distributed'},
              {'direction': 1, 'event_name': 'last'}]
    handle_events(None, ([0, 0, 1, -1],), events)
    with pytest.raises(TerminateSimulation):
        handle_events(None, ([0, 0, -1, 0],), events)
    assert "'distributed'" in capsys.readouterr().out
    with pytest.raises(TerminateSimulation):
        handle_events(None, ([0, 0, 0, 1],), events)
    assert "'last'" in capsys.readouterr().out


@pytest.mark.unit
@pytest.mark.parametrize('cls', TANKS)
def test_retrieval_commits_terminal_state_time_and_integrated_duty(cls):
    reactor = configured(cls, isothermal=False, ht_mode='bath')
    states = profile(reactor)  # [mol/L], optional [m**3], [K]
    reactor.retrieve_results(TIMES, states)
    np.testing.assert_allclose(reactor.Liquid_1.mole_conc, states[-1, :4], rtol=ALGEBRA_RTOL)
    assert reactor.Liquid_1.temp == pytest.approx(TEMPERATURE + RAMP * TIMES[-1])
    assert reactor.elapsed_time == TIMES[-1]
    assert reactor.heat_duty[0] == pytest.approx(sum(np.diff(TIMES) * (reactor.result.q_ht[1:] + reactor.result.q_ht[:-1]) / 2), rel=ALGEBRA_RTOL)
    assert reactor.heat_duty[1] == 0
    assert reactor.duty_type == [0, 0]
    segmented = configured(cls, isothermal=False, ht_mode='bath')
    segmented.retrieve_results(TIMES[:2], states[:2])
    segmented.retrieve_results(TIMES[1:], states[1:])
    np.testing.assert_array_equal(segmented.result.time, TIMES)
    np.testing.assert_allclose(segmented.result.q_ht, reactor.result.q_ht, rtol=ALGEBRA_RTOL)
    np.testing.assert_allclose(segmented.heat_duty, reactor.heat_duty, rtol=ALGEBRA_RTOL)
    np.testing.assert_allclose(segmented.Liquid_1.mole_conc, reactor.Liquid_1.mole_conc, rtol=ALGEBRA_RTOL)


@pytest.mark.unit
@pytest.mark.parametrize('cls', TANKS)
@pytest.mark.parametrize('any_event', [True, False])
def test_solve_registers_shared_event_callbacks(monkeypatch, cls, any_event):
    event = {'state_name': 'mole_conc', 'state_idx': 0, 'value': CONCENTRATIONS[0] / 2}
    reactor = configured(cls, state_events=[event])
    captured = {}
    class StopBeforeSolver(Exception):
        """Stop after the production problem is completely configured."""
    def capture_problem(*args, **kwargs):
        """Capture the optional problem constructor.

        Parameters
        ----------
        *args : tuple
            RHS and initial states [mol/L], optionally volume [m**3].
        **kwargs : dict
            Start time [s] and event switches.

        Returns
        -------
        SimpleNamespace
            Attribute shell for callback registration at the optional boundary.
        """
        captured['problem'] = SimpleNamespace()
        captured['kwargs'] = kwargs
        return captured['problem']
    def stop_solver(problem):
        """Stop before integration.

        Parameters
        ----------
        problem : object
            Configured solver problem.

        Raises
        ------
        StopBeforeSolver
            Always, so the core test does not load Assimulo.
        """
        raise StopBeforeSolver
    monkeypatch.setattr(Reactors, 'Explicit_Problem', capture_problem)
    monkeypatch.setattr(Reactors, 'CVode', stop_solver)
    with pytest.raises(StopBeforeSolver):
        reactor.solve_unit(runtime=TIMES[-1], any_event=any_event, verbose=False)
    problem = captured['problem']
    assert problem.state_events == reactor._eval_state_events
    np.testing.assert_allclose(
        problem.state_events(TIMES[0], profile(reactor)[0], [True]),
        [event['value'] - CONCENTRATIONS[0]], rtol=ALGEBRA_RTOL)
    assert problem.handle_event.func is handle_events
    assert problem.handle_event.keywords['any_event'] is any_event
    assert problem.handle_event.keywords['state_event_list'] is reactor.state_event_list
    assert captured['kwargs']['sw0'] == [True]
    assert reactor.derivatives.shape == profile(reactor)[0].shape


@pytest.mark.integration
@pytest.mark.assimulo
@pytest.mark.parametrize('cls', TANKS)
def test_real_solver_state_event_termination(cls):
    pytest.importorskip('assimulo')
    # A time root tests event wiring independently of kinetics and tolerance.
    stop_time = TIMES[1]  # [s], interior of requested run
    event = {'callable': lambda time, states, sdot: stop_time - time,
             'direction': -1, 'event_name': 'stop'}
    reactor = configured(cls, state_events=[event])
    time, states = reactor.solve_unit(runtime=TIMES[-1], verbose=False)
    assert time[-1] == pytest.approx(stop_time, rel=ALGEBRA_RTOL)
    assert reactor.elapsed_time == time[-1]


@pytest.mark.unit
@pytest.mark.parametrize('cls', ALL_TANKS)
def test_isothermal_heat_sign_matches_balanced_nonisothermal(cls):
    isothermal = configured(cls)
    states = profile(isothermal)  # [mol/L], optional [m**3]
    isothermal.retrieve_results(TIMES, states)
    assert np.all(isothermal.result.q_rxn > 0)
    assert np.all(isothermal.result.q_ht < 0)
    dynamic = configured(cls, isothermal=False, ht_mode='jacket')
    full_states = profile(dynamic)  # [mol/L], optional [m**3], [K], [K]
    full_states[:, -2] = TEMPERATURE  # [K], same liquid thermal state
    vol = isothermal.result.vol  # [m**3]
    area = 4 / dynamic.diam * vol + dynamic.area_base  # [m**2], cylindrical wetted surface
    full_states[:, -1] = TEMPERATURE + isothermal.result.q_ht / (dynamic.u_ht * area)  # [K], utility balances source
    dynamic.retrieve_results(TIMES, full_states)
    np.testing.assert_allclose(dynamic.result.q_rxn, isothermal.result.q_rxn, rtol=ALGEBRA_RTOL)
    # Recovering a small utility temperature difference permits cancellation
    # error at the scale of the 320 K thermal state.
    heat_atol = np.max(dynamic.u_ht * area) * np.spacing(TEMPERATURE) * 2  # [W], two rounded temperatures
    np.testing.assert_allclose(dynamic.result.q_ht, isothermal.result.q_ht,
                               rtol=ALGEBRA_RTOL, atol=heat_atol)
    np.testing.assert_allclose(dynamic.heat_duty, isothermal.heat_duty,
                               rtol=ALGEBRA_RTOL, atol=heat_atol * TIMES[-1])


@pytest.mark.unit
@pytest.mark.parametrize('cls', ALL_TANKS)
def test_constant_callable_and_control_precedence(cls):
    reactor = configured(cls, isothermal=True, controls={'temp': lambda time: TEMPERATURE})
    reactor.retrieve_results(TIMES, profile(reactor))
    np.testing.assert_array_equal(reactor.result.temp, np.full(len(TIMES), TEMPERATURE))
    assert np.all(reactor.result.q_ht < 0)


@pytest.mark.unit
@pytest.mark.parametrize('cls', ALL_TANKS)
@pytest.mark.parametrize('ht_mode', ['bath', 'jacket'])
def test_missing_utility_fails_before_optional_solver(cls, ht_mode):
    reactor = configured(cls, isothermal=False, ht_mode=ht_mode)
    reactor._Utility = None
    with pytest.warns(UserWarning, match='Utility'), pytest.raises(ValueError, match='Utility.*bath'):
        reactor.solve_unit(runtime=TIMES[-1], verbose=False)


@pytest.mark.unit
@pytest.mark.parametrize('cls', TANKS)
def test_bath_temperature_control_cannot_override_utility(cls):
    plain = configured(cls, isothermal=False, ht_mode='bath')
    controlled = configured(cls, isothermal=False, ht_mode='bath',
                            controls={'temp_ht': lambda time: TEMPERATURE})
    states = profile(plain)  # [mol/L], optional [m**3], [K]
    np.testing.assert_allclose(controlled.unit_model(TIMES[0], states[0]),
                               plain.unit_model(TIMES[0], states[0]), rtol=ALGEBRA_RTOL)
    controlled.retrieve_results(TIMES, states)
    plain.retrieve_results(TIMES, states)
    np.testing.assert_allclose(controlled.result.q_ht, plain.result.q_ht, rtol=ALGEBRA_RTOL)


@pytest.mark.integration
@pytest.mark.assimulo
@pytest.mark.parametrize('cls', TANKS)
@pytest.mark.parametrize('ht_mode', ['bath', 'jacket'])
def test_solver_continuation_matches_uninterrupted_dynamic_inputs(cls, ht_mode):
    pytest.importorskip('assimulo')
    uninterrupted = configured(cls, isothermal=False, ht_mode=ht_mode)
    segmented = configured(cls, isothermal=False, ht_mode=ht_mode)
    # Dense common output grid controls trapezoidal quadrature; tight solver
    # tolerances isolate continuation errors from ordinary integration error.
    # Assimulo 3.4.3's noncontinuous reporting omitted requested penultimate
    # samples in this CSTR case; continuous reporting enforces the common grid.
    grid = np.linspace(TIMES[0], TIMES[-1], 21)  # [s], ten intervals per segment
    options = {'rtol': 1e-10, 'atol': 1e-12, 'report_continuously': True}  # [-], [state units], accuracy for comparison
    compare_rtol = 1e-7  # [-], allows accumulated local error through restart
    for reactor in (uninterrupted, segmented):
        inlet = DynamicInput()
        inlet.add_variable('temp', lambda time: TEMPERATURE + RAMP * time)
        reactor.Inlet.DynamicInlet = inlet
        utility = DynamicInput()
        utility.add_variable('temp_in', lambda time: UTILITY_TEMPERATURE + RAMP * time)
        reactor.Utility.DynamicInlet = utility
    uninterrupted.solve_unit(time_grid=grid, sundials_opts=options, verbose=False)
    midpoint = len(grid) // 2
    segmented.solve_unit(time_grid=grid[:midpoint + 1], sundials_opts=options, verbose=False)
    segmented.solve_unit(time_grid=grid[midpoint:] - grid[midpoint], sundials_opts=options, verbose=False)
    np.testing.assert_allclose(segmented.result.time, uninterrupted.result.time, rtol=ALGEBRA_RTOL)
    for name in ('mole_conc', 'temp', 'vol', 'q_ht'):
        np.testing.assert_allclose(getattr(segmented.result, name), getattr(uninterrupted.result, name), rtol=compare_rtol)
    np.testing.assert_allclose(segmented.heat_duty, uninterrupted.heat_duty, rtol=compare_rtol)
    np.testing.assert_allclose(segmented.Liquid_1.mole_conc, uninterrupted.Liquid_1.mole_conc, rtol=compare_rtol)
    assert segmented.Outlet.temp == pytest.approx(uninterrupted.Outlet.temp, rel=compare_rtol)
    assert segmented.elapsed_time == uninterrupted.elapsed_time


@pytest.mark.unit
@pytest.mark.parametrize('cls', ALL_TANKS)
def test_initial_jacket_uses_nominal_utility_charge(monkeypatch, cls):
    reactor = configured(cls, isothermal=False)
    utility = DynamicInput()
    utility.add_variable('temp_in', lambda time: TEMPERATURE + RAMP * time)
    reactor.Utility.DynamicInlet = utility
    captured = {}
    class ProblemCaptured(Exception):
        """Stop at the optional problem constructor, before integration."""
    def capture_problem(rhs, initial, **kwargs):
        """Capture a packed initial state then stop construction.

        Parameters
        ----------
        rhs : callable
            Production reactor RHS.
        initial : numpy.ndarray
            Concentrations [mol/L], optional volume [m**3], temperatures [K].
        **kwargs : dict
            Problem settings including start time [s].
        """
        captured['initial'] = initial.copy()
        raise ProblemCaptured
    monkeypatch.setattr(Reactors, 'Explicit_Problem', capture_problem)
    with pytest.raises(ProblemCaptured):
        reactor.solve_unit(runtime=TIMES[-1], verbose=False)
    assert captured['initial'][-1] == UTILITY_TEMPERATURE
    assert reactor.Utility.get_inputs(TIMES[0])['temp_in'] == TEMPERATURE


@pytest.mark.unit
def test_distributed_selection_and_definition_order():
    states = np.arange(1., 7.)  # [mol/L], three nodes and two components
    components = {'state_name': 'mole_conc', 'state_idx': [1, 0],
                  'node_idx': [2, 0], 'value': 0., 'num_conditions': 4}  # [mol/L]
    node = {'state_name': 'mole_conc', 'node_idx': 1,
            'value': 0., 'num_conditions': 2}  # [mol/L]
    for definitions, expected in [([components, node], [-6, -5, -2, -1, -3, -4]),
                                  ([node, components], [-3, -4, -6, -5, -2, -1])]:
        actual = eval_state_events(TIMES[0], states, [True, True], [2], ['mole_conc'],
                                   definitions, discretized_model=True)  # [mol/L]
        np.testing.assert_array_equal(actual, expected)
    scalar = {'state_name': 'temp', 'node_idx': [2, 0], 'value': 0., 'num_conditions': 2}  # [K]
    actual = eval_state_events(TIMES[0], states[:3], [True], [1], ['temp'],
                               [scalar], discretized_model=True)  # [K]
    np.testing.assert_array_equal(actual, [-3., -1.])


@pytest.mark.unit
def test_event_condition_count_validation_and_disabled_roots():
    definition = {'state_name': 'mole_conc', 'value': 0.}  # [mol/L]
    with pytest.raises(ValueError, match='num_conditions'):
        eval_state_events(TIMES[0], CONCENTRATIONS, [True], [4], ['mole_conc'], [definition])
    with pytest.raises(ValueError, match='num_conditions'):
        handle_events(None, ([0, 0],), [definition])
    definition['num_conditions'] = len(CONCENTRATIONS)
    roots = eval_state_events(TIMES[0], CONCENTRATIONS, [False], [4], ['mole_conc'], [definition])  # [-]
    np.testing.assert_array_equal(roots, np.ones(len(CONCENTRATIONS)))


@pytest.mark.integration
@pytest.mark.assimulo
@pytest.mark.parametrize('cls', TANKS)
@pytest.mark.parametrize('direction', [-1, 1])
def test_real_solver_vector_events_and_direction(cls, direction):
    pytest.importorskip('assimulo')
    event = {'callable': lambda time, states, sdot: np.array([TIMES[-1] * 2, TIMES[1]]) - time,
             'num_conditions': 2, 'direction': direction, 'event_name': 'vector'}
    reactor = configured(cls, state_events=[event])
    time, _ = reactor.solve_unit(runtime=TIMES[-1], verbose=False)
    expected = TIMES[1] if direction == -1 else TIMES[-1]  # [s]
    assert time[-1] == pytest.approx(expected, rel=ALGEBRA_RTOL)


@pytest.mark.unit
@pytest.mark.parametrize('time', [np.array([0., 0.]),
                                 np.array([0., np.nan]), np.array([1., 0.])])
def test_prescribed_heat_rejects_invalid_time_samples(time):
    reactor = configured(Reactors.CSTR, controls={'temp': lambda time: TEMPERATURE})
    with pytest.raises(ValueError, match='finite increasing time samples'):
        reactor.retrieve_results(time, profile(reactor, time))


@pytest.mark.integration
@pytest.mark.assimulo
@pytest.mark.parametrize('cls', TANKS)
def test_reset_starts_a_fresh_duration_after_completed_run(cls):
    pytest.importorskip('assimulo')
    reactor = configured(cls, reset_states=True)
    first_time, first_states = reactor.solve_unit(runtime=TIMES[-1], verbose=False)
    second_time, second_states = reactor.solve_unit(runtime=TIMES[-1], verbose=False)
    assert second_time[0] == TIMES[0]
    assert second_time[-1] == first_time[-1]
    np.testing.assert_allclose(second_states[-1], first_states[-1], rtol=ALGEBRA_RTOL)
    assert len(reactor.profiles_runs) == 1


@pytest.mark.unit
@pytest.mark.parametrize('field, value, message', [
    ('fun', None, 'fun.*callable'), ('args', None, 'sequence args'),
    ('kwargs', None, 'dictionary kwargs')])
def test_control_record_value_types(field, value, message):
    record = {'fun': lambda time: TEMPERATURE, 'args': (), 'kwargs': {}}
    record[field] = value
    with pytest.raises(TypeError, match=message):
        Reactors.CSTR(controls={'temp': record})


@pytest.mark.unit
@pytest.mark.parametrize('cls', ALL_TANKS)
@pytest.mark.parametrize('kind', ['quadratic', 'sinusoidal'])
@pytest.mark.parametrize('count', [1, 2, 5])
def test_nonlinear_prescribed_duty_uses_callable(cls, kind, count):
    # A 40 s sine profile exercises curvature without appreciable conversion.
    duration = 40.0  # [s], two radians of the synthetic sine
    amplitude = 5.0  # [K], synthetic oscillation amplitude
    timescale = 20.0  # [s], inverse angular frequency
    curvature = 0.5  # [K/s**2], quadratic coefficient from the review case
    if kind == 'quadratic':
        control = lambda time: TEMPERATURE + curvature * time**2
        slope = lambda time: 2 * curvature * time
        duration = TIMES[-1]  # [s], bounds the temperature rise to 2 K
    else:
        control = lambda time: TEMPERATURE + amplitude * np.sin(time / timescale)
        slope = lambda time: amplitude / timescale * np.cos(time / timescale)
    # A lone point must also have a defined derivative when no solve preceded it.
    time = np.linspace(0., duration, count)  # [s]
    reactor = configured(cls, isothermal=False, controls={'temp': {'fun': control, 'args': (), 'kwargs': {}}})
    states = profile(reactor, time)  # [mol/L], optional [m**3]
    reactor.retrieve_results(time, states)
    reaction, flow, capacity = independent_terms(reactor, time, states, control(time))  # [W], [W], [J/K]
    expected = capacity * slope(time) - reaction - flow  # [W]
    # Forward three-point truncation <= h**2 max|T'''|/3. The chosen production
    # step is ~0.001 of this run span. Bound roundoff separately for 320 K data.
    step_bound = max(duration, 1.) / 1024  # [s], binary subdivision specified by the API
    third_derivative = 0. if kind == 'quadratic' else amplitude / timescale**3  # [K/s**3]
    slope_atol = step_bound**2 * third_derivative / 3 + 1e-9  # [K/s], subtraction roundoff allowance
    np.testing.assert_allclose(reactor.result.q_ht, expected,
                               atol=np.max(capacity) * slope_atol, rtol=ALGEBRA_RTOL)


@pytest.mark.unit
@pytest.mark.parametrize('cls', TANKS)
def test_prescribed_synthetic_segments_have_equal_duties(cls):
    control = lambda time: TEMPERATURE + RAMP * time**2
    whole = configured(cls, controls={'temp': control})
    split = configured(cls, controls={'temp': control})
    states = profile(whole)  # [mol/L], optional [m**3]
    whole.retrieve_results(TIMES, states)
    split.retrieve_results(TIMES[:2], states[:2])
    split.retrieve_results(TIMES[1:], states[1:])
    np.testing.assert_array_equal(split.result.time, whole.result.time)
    np.testing.assert_allclose(split.result.q_ht, whole.result.q_ht, rtol=ALGEBRA_RTOL, atol=0)
    np.testing.assert_allclose(split.heat_duty, whole.heat_duty, rtol=ALGEBRA_RTOL, atol=0)


@pytest.mark.unit
@pytest.mark.parametrize('cls', TANKS)
def test_retrieval_preserves_segment_inlet_history(cls):
    reactor = configured(cls)
    time = np.array([0., 0.5, 1., 1.5, 2.])  # [s], common boundary appears once
    states = profile(reactor, time)  # [mol/L], optional [m**3]
    initial_flow = reactor.Inlet.vol_flow  # [m**3/s]
    reactor.retrieve_results(time[:3], states[:3])
    first_heat = reactor.result.q_ht.copy()  # [W]
    reactor.Inlet.updatePhase(vol_flow=2 * initial_flow)
    reactor.Inlet.temp = TEMPERATURE + RAMP  # [K], distinguish segment sensible heat
    reactor.retrieve_results(time[2:], states[2:])
    np.testing.assert_array_equal(reactor.result.time, time)
    np.testing.assert_allclose(reactor.outputs['vol_flow'],
                               initial_flow * np.array([1., 1., 1., 2., 2.]), rtol=ALGEBRA_RTOL)
    np.testing.assert_array_equal(reactor.result.q_ht[:3], first_heat)


@pytest.mark.unit
def test_semibatch_metadata_follows_named_packing(monkeypatch):
    reactor = configured(Reactors.SemibatchReactor, isothermal=False)
    # Content-preserving permutation keeps material states before temperatures
    # but places scalar volume before the four-component concentration state.
    monkeypatch.setattr(Reactors, 'order_state_names',
                        lambda names: ['vol', 'mole_conc', 'temp', 'temp_ht'])
    reactor.set_names()
    packed = np.r_[REACTOR_VOLUME, CONCENTRATIONS, TEMPERATURE, UTILITY_TEMPERATURE]  # [m**3], [mol/L], [K], [K]
    actual = unpack_states(packed, reactor.dim_states, reactor.name_states)
    assert list(actual) == reactor.name_states
    np.testing.assert_array_equal(actual['mole_conc'], CONCENTRATIONS)
    assert actual['vol'] == REACTOR_VOLUME
    assert actual['temp'] == TEMPERATURE
    assert actual['temp_ht'] == UTILITY_TEMPERATURE
    reactor.derivatives = np.zeros_like(packed)  # [state units/s], event fixture
    reactor.state_event_list = [{'state_name': 'mole_conc', 'state_idx': 1, 'value': 0.}]  # [mol/L]
    np.testing.assert_array_equal(reactor._eval_state_events(0., packed, [True]), [-CONCENTRATIONS[1]])


@pytest.mark.assimulo
@pytest.mark.integration
def test_batch_cumulative_duty_after_two_solver_segments():
    pytest.importorskip('assimulo')
    reactor = configured(Reactors.BatchReactor)
    reactor.solve_unit(runtime=TIMES[1], verbose=False)
    first_duty = reactor.heat_duty.copy()  # [J]
    reactor.solve_unit(runtime=TIMES[1], verbose=False)
    time = reactor.result.time  # [s]
    heat = reactor.result.q_ht  # [W]
    expected = sum(np.diff(time) * (heat[1:] + heat[:-1]) / 2)  # [J]
    assert reactor.heat_duty[0] == pytest.approx(expected, rel=ALGEBRA_RTOL)
    assert reactor.heat_duty[0] < first_duty[0] < 0
    assert reactor.elapsed_time == TIMES[-1]


@pytest.mark.unit
def test_minimal_control_record_defaults_without_mutating_caller():
    record = {'fun': lambda time: TEMPERATURE + RAMP * time}
    reactor = configured(Reactors.CSTR, controls={'temp': record})
    assert set(record) == {'fun'}
    assert reactor.controls['temp']['args'] == ()
    assert reactor.controls['temp']['kwargs'] == {}
    reactor.retrieve_results(TIMES, profile(reactor))
    np.testing.assert_array_equal(reactor.result.temp, TEMPERATURE + RAMP * TIMES)


@pytest.mark.unit
@pytest.mark.parametrize('name', [None, 'concentration limit'])
def test_event_count_errors_identify_definition_and_counts(name):
    definition = {'state_name': 'mole_conc', 'num_conditions': 2, 'value': 0.}  # [mol/L]
    if name is not None:
        definition['event_name'] = name
    for action in (
            lambda: eval_state_events(0., CONCENTRATIONS, [True], [4], ['mole_conc'], [definition]),
            lambda: handle_events(None, ([0, 0, 0, 0],), [definition])):
        with pytest.raises(ValueError) as error:
            action()
        message = str(error.value)
        assert (name or '#0 (mole_conc)') in message
        assert 'expected 2' in message
        assert 'got 4' in message


@pytest.mark.assimulo
@pytest.mark.integration
@pytest.mark.parametrize('cls', TANKS)
def test_solver_retains_inlet_history_after_flow_change(cls):
    pytest.importorskip('assimulo')
    reactor = configured(cls)
    offsets = np.array([0., 0.5, 1.])  # [s], includes the common segment boundary
    options = {'report_continuously': True}
    initial_flow = reactor.Inlet.vol_flow  # [m**3/s]
    reactor.solve_unit(time_grid=offsets, sundials_opts=options, verbose=False)
    initial_heat = reactor.result.q_flow.copy()  # [W]
    reactor.Inlet.updatePhase(vol_flow=2 * initial_flow)
    reactor.Inlet.temp = TEMPERATURE + RAMP  # [K], second segment's distinct feed
    reactor.solve_unit(time_grid=offsets, sundials_opts=options, verbose=False)
    np.testing.assert_array_equal(reactor.result.time, [0., 0.5, 1., 1.5, 2.])
    np.testing.assert_allclose(reactor.outputs['vol_flow'],
                               initial_flow * np.array([1., 1., 1., 2., 2.]), rtol=ALGEBRA_RTOL)
    np.testing.assert_array_equal(reactor.result.q_flow[:3], initial_heat)
    np.testing.assert_array_equal(reactor.result.inlet_temp,
                                  TEMPERATURE + RAMP * np.array([0., 0., 0., 1., 1.]))
    np.testing.assert_array_equal(reactor.result.inlet_mole_conc,
                                  np.tile(CONCENTRATIONS, (5, 1)))


@pytest.mark.assimulo
@pytest.mark.integration
def test_cstr_steady_state_callable_accepts_any_event_false():
    pytest.importorskip('assimulo')
    initial_rate = RATE_CONSTANT * CONCENTRATIONS[0] * CONCENTRATIONS[1]  # [mol/L/s]
    event = {'callable': check_steady_state, 'event_name': 'steady_state',
             'kwargs': {'tau': 0., 'time_stop': TIMES[1], 'threshold': 2 * initial_rate}}  # [s], [s], [mol/L/s]
    reactor = configured(Reactors.CSTR, state_events=[event])
    # check_steady_state returns a zero plateau, rather than a signed smooth
    # root. Bound the solver step so this gate is sampled before the run end.
    maximum_step = (TIMES[-1] - TIMES[1]) / 4  # [s], four opportunities after the gate
    time, states = reactor.solve_unit(
        runtime=TIMES[-1], any_event=False, verbose=False,
        sundials_opts={'maxh': maximum_step})
    assert TIMES[1] < time[-1] < TIMES[-1]
    rates = reactor.unit_model(time[-1], states[-1])  # [mol/L/s]
    assert np.linalg.norm(rates) < event['kwargs']['threshold']


@pytest.mark.assimulo
@pytest.mark.integration
@pytest.mark.parametrize('cls', TANKS)
@pytest.mark.parametrize('kind', ['quadratic', 'sinusoidal'])
def test_prescribed_solver_segments_share_rates_and_duties(cls, kind):
    pytest.importorskip('assimulo')
    amplitude = 5.0  # [K], synthetic sinusoid from the review
    timescale = 20.0  # [s], inverse angular frequency
    if kind == 'quadratic':
        control = lambda time: TEMPERATURE + RAMP * time**2
        duration = TIMES[-1]  # [s], quadratic temperature rise limited to 2 K
        third_derivative = 0.0  # [K/s**3]
    else:
        control = lambda time: TEMPERATURE + amplitude * np.sin(time / timescale)
        duration = 40.0  # [s], covers two radians and a derivative sign change
        third_derivative = amplitude / timescale**3  # [K/s**3], supremum bound
    whole = configured(cls, controls={'temp': control})
    split = configured(cls, controls={'temp': control})
    intervals = 32  # [-], sixteen intervals per segment, dyadic common times
    grid = np.linspace(0., duration, intervals + 1)  # [s]
    options = {'rtol': 1e-10, 'atol': 1e-12, 'report_continuously': True}  # [-], [state units], reporting choice
    whole.solve_unit(time_grid=grid, sundials_opts=options, verbose=False)
    middle = intervals // 2
    split.solve_unit(time_grid=grid[:middle + 1], sundials_opts=options, verbose=False)
    split.solve_unit(time_grid=grid[middle:] - grid[middle], sundials_opts=options, verbose=False)
    np.testing.assert_array_equal(whole.result.time, grid)
    np.testing.assert_array_equal(split.result.time, grid)
    reaction, flow, capacity = independent_terms(whole, grid, whole.statesProf, control(grid))  # [W], [W], [J/K]
    whole_step = duration / 1024  # [s], documented control differentiation rule
    split_step = whole_step / 2  # [s], half-duration runs
    # Sum the two forward-stencil error bounds; the central bound is smaller.
    # The absolute floor covers subtraction at 320 K; relative allowance is
    # 1000 times local rtol to cover accumulated solver/restart error.
    rate_atol = np.max(capacity) * (whole_step**2 + split_step**2) * third_derivative / 3 + 1e-6  # [W]
    comparison_rtol = 1e-7  # [-], accumulated numerical integration allowance
    analytic_slope = (2 * RAMP * grid if kind == 'quadratic'
                      else amplitude / timescale * np.cos(grid / timescale))  # [K/s]
    expected_rate = capacity * analytic_slope - reaction - flow  # [W]
    np.testing.assert_allclose(whole.result.q_ht, expected_rate,
                               rtol=ALGEBRA_RTOL, atol=rate_atol)
    np.testing.assert_allclose(split.result.q_ht, whole.result.q_ht,
                               rtol=comparison_rtol, atol=rate_atol)
    # On this identical grid, trapezoidal weights are positive and sum to the
    # duration. Integrating the rate-error bound also bounds the duty difference;
    # discretization error in either absolute duty is common to both runs.
    heat = np.abs(whole.result.q_ht)  # [W]
    absolute_integral = sum(np.diff(grid) * (heat[1:] + heat[:-1]) / 2)  # [J]
    duty_atol = duration * rate_atol + comparison_rtol * absolute_integral  # [J]
    np.testing.assert_allclose(split.heat_duty, whole.heat_duty, rtol=0, atol=duty_atol)


@pytest.mark.unit
@pytest.mark.parametrize('cls', ALL_TANKS)
@pytest.mark.parametrize('duration', [0., 32.])
def test_single_reported_point_uses_requested_run_duration(monkeypatch, cls, duration):
    evaluations = []  # Each entry is a scalar or vector of times [s].
    def control(time):
        """Record evaluations of a smooth control defined only after run start.

        Parameters
        ----------
        time : float or numpy.ndarray
            Absolute evaluation times [s].

        Returns
        -------
        float or numpy.ndarray
            Synthetic quadratic temperature [K].
        """
        evaluations.append(np.asarray(time).copy())  # [s]
        assert np.all(np.asarray(time) >= 0)
        return TEMPERATURE + RAMP * time**2
    reactor = configured(cls, controls={'temp': control})
    def problem(rhs, initial, t0, **kwargs):
        """Capture the optional problem's physical initial state.

        Parameters
        ----------
        rhs : callable
            Reactor right-hand side.
        initial : numpy.ndarray
            Packed concentrations [mol/L], optional volume [m**3].
        t0 : float
            Absolute start time [s].
        **kwargs : dict
            Backend switches and parameters.

        Returns
        -------
        SimpleNamespace
            Initial state and time for an immediate-stop solver result.
        """
        return SimpleNamespace(initial=initial, time=t0)
    def solver(problem):
        """Represent a solver stopping immediately with one reported point.

        Parameters
        ----------
        problem : SimpleNamespace
            Initial concentrations [mol/L], volume [m**3] and time [s].

        Returns
        -------
        SimpleNamespace
            Optional solver boundary returning only the initial profile row.
        """
        return SimpleNamespace(simulate=lambda final_time, ncp_list: (
            np.array([problem.time]), np.atleast_2d(problem.initial)))
    monkeypatch.setattr(Reactors, 'Explicit_Problem', problem)
    monkeypatch.setattr(Reactors, 'CVode', solver)
    reactor.solve_unit(runtime=duration, verbose=False)
    # The documented forward stencil samples t+h and t+2h, even though the
    # returned profile has zero span. The zero-duration case uses the 1 ms floor.
    expected_step = max(duration, 1.) / 1024  # [s], documented differentiation rule
    assert max(float(np.max(value)) for value in evaluations) == 2 * expected_step
    assert reactor.result.time.size == 1
    assert np.isfinite(reactor.result.q_ht[0])

    reactor.reset()
    assert not ({'_run_start', '_run_duration'} & reactor.__dict__.keys())


@pytest.mark.unit
def test_control_record_missing_function_names_control():
    with pytest.raises(KeyError, match='temp'):
        Reactors.CSTR(controls={'temp': {}})


@pytest.mark.unit
@pytest.mark.parametrize('cls', ALL_TANKS)
# The decimal interval exercises endpoint roundoff in short-run stencils.
@pytest.mark.parametrize('start, duration', [(4., 2.), (4., 1 / 2048), (.0001, .0004)])
def test_bounded_quadratic_control_retrieval(cls, start, duration):
    """Recover quadratic slopes without leaving a nonzero run interval.

    Parameters
    ----------
    cls : type
        Tank reactor class.
    start : float
        Nonzero absolute start [s], including a decimal sub-millisecond origin.
    duration : float
        Run duration [s]; short cases fall below the default minimum step.
    """
    end = start + duration  # [s]
    curvature = 0.5  # [K/s**2], gives the exact slope (t - start) K/s
    time = start + duration * np.array([0., 1/8, 1/2, 7/8, 1.])  # [s], both edges and interior
    evaluations = []  # Evaluation times [s].

    def control(time):
        """Evaluate a quadratic on its bounded run domain.

        Parameters
        ----------
        time : float or numpy.ndarray
            Evaluation times [s], restricted to [start, end].

        Returns
        -------
        float or numpy.ndarray
            Prescribed temperature [K].
        """
        evaluations.append(np.asarray(time).copy())
        assert np.all((time >= start) & (time <= end))
        return TEMPERATURE + curvature * (time - start)**2

    reactor = configured(cls, controls={'temp': control})
    states = profile(reactor, time)  # [mol/L], optional [m**3]
    reactor.retrieve_results(time, states)
    reaction, flow, capacity = independent_terms(
        reactor, time, states, control(time))  # [W], [W], [J/K]
    actual_slope = (reactor.result.q_ht + reaction + flow) / capacity  # [K/s]
    expected_slope = 2 * curvature * (time - start)  # [K/s]
    # Second-order stencils are exact for a quadratic; allow only cancellation
    # of temperatures near 320 K, amplified by the short interval's 1/h.
    slope_atol = 1e-8  # [K/s], conservative float64 subtraction allowance
    np.testing.assert_allclose(actual_slope, expected_slope, rtol=0, atol=slope_atol)
    assert len(evaluations) > len(time)


@pytest.mark.assimulo
@pytest.mark.integration
def test_batch_solver_accepts_bounded_interpolated_temperature():
    """Solve the review's two-second linear ramp with no extrapolation."""
    pytest.importorskip('assimulo')
    temperatures = np.array([300., 301., 302.])  # [K], one kelvin per second
    control = interp1d(TIMES, temperatures)
    reactor = configured(Reactors.BatchReactor, controls={'temp': control})
    time, states = reactor.solve_unit(time_grid=TIMES, verbose=False)
    time = np.asarray(time)  # [s], normalize the backend's list return
    states = np.asarray(states)  # [mol/L]
    # Default solver reporting may omit the requested interior sample.
    # Both run endpoints must be retrieved safely, with the prescribed slope.
    np.testing.assert_array_equal(time[[0, -1]], TIMES[[0, -1]])
    slope = 1.0  # [K/s], prescribed ramp
    expected_temperatures = temperatures[0] + slope * time  # [K]
    np.testing.assert_allclose(reactor.result.temp, expected_temperatures, rtol=ALGEBRA_RTOL)
    reaction, flow, capacity = independent_terms(
        reactor, time, states, expected_temperatures)  # [W], [W], [J/K]
    np.testing.assert_allclose(reactor.result.q_ht, capacity * slope - reaction - flow,
                               rtol=ALGEBRA_RTOL)
