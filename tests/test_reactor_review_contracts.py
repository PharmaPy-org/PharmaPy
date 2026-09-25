"""PR 263 regressions for control derivatives and Batch lifecycle contracts.

Real phase and kinetic fixtures come from the reactor contract suite. Native
CVode cases are confined to the Assimulo lane. Synthetic control frequencies
and charges expose numerical behavior rather than calibrated process data.
"""

import numpy as np
import pytest

from PharmaPy import Reactors
from test_reactor_controls_events_continuation import (
    ALL_TANKS, configured, independent_terms,
)
from test_reactor_correctness import TEMPERATURE, TUBE_DIAMETER, NUM_CELLS


@pytest.mark.unit
def test_control_derivative_uses_sampling_timescale():
    period = 0.1  # [s], much shorter than the requested hour / 1024
    frequency = 2 * np.pi / period  # [rad/s]
    control = lambda time: TEMPERATURE + np.sin(frequency * time)
    reactor = configured(Reactors.BatchReactor, controls={'temp': control})
    reactor._run_start = 0.0  # [s]
    reactor._run_duration = 3600.0  # [s], requested horizon must not set resolution
    time = np.linspace(1800.0, 1800.1, 101)  # [s], resolved reporting grid
    slope = reactor._prescribed_heat(
        time, control(time), np.ones_like(time), np.zeros_like(time),
        minimum_step=1e-7)  # [K/s], unit capacitance and zero source
    expected = frequency * np.cos(frequency * time)  # [K/s], analytic derivative
    # A 1e-5 relative bound is well above roundoff at absolute time 1800 s,
    # and rejects the O(1) aliasing from the former hour-dependent step.
    np.testing.assert_allclose(slope, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.unit
@pytest.mark.parametrize('isothermal', [False, True])
def test_pfr_rejects_unsupported_controls(isothermal):
    with pytest.raises(NotImplementedError, match='PlugFlowReactor.*controls'):
        Reactors.PlugFlowReactor(
            TUBE_DIAMETER, NUM_CELLS, isothermal=isothermal,
            controls={'temp': lambda time: TEMPERATURE})
    # Empty controls leave both established modes available.
    Reactors.PlugFlowReactor(TUBE_DIAMETER, NUM_CELLS,
                            isothermal=isothermal, controls={})


@pytest.mark.assimulo
@pytest.mark.integration
@pytest.mark.parametrize('cls', ALL_TANKS)
def test_public_control_steps_and_early_stop_reconstruct_heat(cls):
    pytest.importorskip('assimulo')
    stop = 0.1  # [s], completed interval much shorter than requested horizon
    frequency = 20 * np.pi  # [rad/s], one sinusoidal cycle before the event

    def control(time):
        """Evaluate the bounded synthetic temperature program.

        Parameters
        ----------
        time : float or ndarray
            Absolute time [s].

        Returns
        -------
        float or ndarray
            Temperature [K], with a one-kelvin sinusoidal amplitude.
        """
        return TEMPERATURE + np.sin(frequency * time)

    event = {'callable': lambda time, states, sdot: stop - time,
             'direction': -1, 'event_name': 'short run'}
    reactor = configured(cls, controls={'temp': control}, state_events=[event])
    time, states = reactor.solve_unit(
        runtime=3600.0, verbose=False,
        sundials_opts={'maxh': stop / 100, 'rtol': 1e-9, 'atol': 1e-11},
        control_step_fraction=1 / 1024, control_minimum_step=1e-7)
    time = np.asarray(time)  # [s]
    # maxh [s] resolves the imposed oscillation; tolerances resolve the state.
    reaction, flow, capacity = independent_terms(
        reactor, time, states, control(time))  # [W], [W], [J/K]
    expected_slope = frequency * np.cos(frequency * time)  # [K/s]
    recovered_slope = (reactor.result.q_ht + reaction + flow) / capacity  # [K/s]
    assert time[-1] == pytest.approx(stop, abs=1e-10)
    np.testing.assert_allclose(recovered_slope, expected_slope,
                               rtol=1e-5, atol=1e-5)


@pytest.mark.assimulo
@pytest.mark.integration
def test_batch_repeated_grid_preserves_jacket_and_absolute_time():
    pytest.importorskip('assimulo')
    grid = np.linspace(0.0, 2.0, 101)  # [s], two-second continuation offsets
    options = {'rtol': 1e-9, 'atol': 1e-11, 'report_continuously': True}  # [-], state-unit absolute tolerances
    whole = configured(Reactors.BatchReactor, isothermal=False)
    split = configured(Reactors.BatchReactor, isothermal=False)
    whole.solve_unit(time_grid=np.linspace(0.0, 4.0, 201),
                     sundials_opts=options, verbose=False)
    split.solve_unit(time_grid=grid, sundials_opts=options, verbose=False)
    terminal_jacket = split.result.temp_ht[-1]  # [K], retained coolant charge
    time, states = split.solve_unit(time_grid=grid, sundials_opts=options,
                                    verbose=False)
    np.testing.assert_allclose(np.asarray(time)[[0, -1]], [2.0, 4.0], rtol=0, atol=1e-12)
    assert states[0, -1] == pytest.approx(terminal_jacket, rel=1e-12)
    np.testing.assert_allclose(split.result.temp, whole.result.temp, rtol=1e-7)
    np.testing.assert_allclose(split.result.temp_ht, whole.result.temp_ht, rtol=1e-7)
    np.testing.assert_allclose(split.heat_duty, whole.heat_duty, rtol=1e-7)


@pytest.mark.assimulo
@pytest.mark.integration
def test_batch_reset_states_repeats_original_charge_and_duty():
    pytest.importorskip('assimulo')
    reactor = configured(Reactors.BatchReactor, reset_states=True)
    first_time, first = reactor.solve_unit(runtime=2.0, verbose=False)  # [s]
    first_duty = reactor.heat_duty.copy()  # [J]
    second_time, second = reactor.solve_unit(runtime=2.0, verbose=False)  # [s]
    np.testing.assert_allclose(second_time, first_time, rtol=1e-12)
    np.testing.assert_allclose(second, first, rtol=1e-12)
    np.testing.assert_allclose(reactor.heat_duty, first_duty, rtol=1e-12)
    assert len(reactor.profiles_runs) == 1


@pytest.mark.unit
@pytest.mark.parametrize('cls', ALL_TANKS)
@pytest.mark.parametrize('bad', [np.array([320.0]), [320.0, 321.0], np.nan])
def test_tank_control_rejects_invalid_scalar_returns(cls, bad):
    reactor = configured(cls, controls={'temp': lambda time: bad})
    with pytest.raises(ValueError, match='temp.*finite scalar'):
        reactor.solve_unit(runtime=1.0, verbose=False)
    assert getattr(reactor, 'Outlet', None) is None
    assert not reactor.profiles_runs


@pytest.mark.unit
def test_control_derivative_stays_inside_completed_interval():
    stop = 0.1  # [s], early event before a requested hour
    seen = []  # [s], control evaluations

    def control(time):
        """Evaluate a control defined on the completed interval only.

        Parameters
        ----------
        time : float
            Time [s], restricted to [0, stop].

        Returns
        -------
        float
            Quadratic temperature [K].
        """
        assert 0 <= time <= stop
        seen.append(time)
        return TEMPERATURE + time**2

    reactor = configured(Reactors.BatchReactor, controls={'temp': control})
    reactor._run_start = 0.0  # [s]
    reactor._run_duration = 3600.0  # [s], requested but not completed
    time = np.linspace(0, stop, 5)  # [s]
    actual = reactor._prescribed_heat(time, TEMPERATURE + time**2,
                                      np.ones(5), np.zeros(5))  # [K/s]
    np.testing.assert_allclose(actual, 2*time, atol=1e-9, rtol=0)
    assert seen and min(seen) >= 0 and max(seen) <= stop


@pytest.mark.assimulo
@pytest.mark.integration
def test_pfr_component_and_node_event_through_native_solver():
    """Localize a tracer step in the third of three finite volumes.

    The independent oracle is the three equal-tank step response. Four species
    and three nodes expose swapped axes; only species C changes at the inlet.
    Refs https://github.com/PharmaPy-org/PharmaPy/issues/233.
    """
    pytest.importorskip('assimulo')
    from math import factorial
    from test_reactor_correctness import (
        _configured_reactor, CONCENTRATIONS, RESIDENCE_TIME,
    )

    nodes = 3  # [-], unequal to the four species
    stop = RESIDENCE_TIME / nodes  # [s], one cell residence time
    concentration_step = CONCENTRATIONS[2]  # [mol/L], doubles the inlet tracer
    threshold = CONCENTRATIONS[2] + concentration_step * (
        1 - np.exp(-1) * (1 + 1 + 1/2))  # [mol/L], Erlang-3 CDF at t/tau=1
    event = {'state_name': 'mole_conc', 'state_idx': 2, 'node_idx': 2,
             'value': threshold, 'num_conditions': 1, 'direction': -1,
             'event_name': 'third-cell tracer'}
    reactor = _configured_reactor(Reactors.PlugFlowReactor(
        TUBE_DIAMETER, nodes, isothermal=True, state_events=[event]))
    # A zero prefactor makes this an inert transport case; Kinetics' epsilon
    # offset contributes only machine-scale concentration drift.
    reactor.Kinetics.set_params({'k_params': [0.0], 'ea_params': [0.0]})
    inlet_concentration = CONCENTRATIONS.copy()  # [mol/L]
    inlet_concentration[2] += concentration_step  # [mol/L]
    from PharmaPy.Streams import LiquidStream
    reactor.Inlet = LiquidStream(
        reactor.Inlet.path_data, temp=TEMPERATURE,
        vol_flow=reactor.Inlet.vol_flow, mole_conc=inlet_concentration)
    time, states = reactor.solve_unit(
        runtime=2 * stop, verbose=False,
        sundials_opts={'rtol': 1e-10, 'atol': 1e-12, 'maxh': stop/100})
    # Root and state bounds are wider than integration tolerances but resolve
    # the distinct responses of all three cells at one residence time.
    assert time[-1] == pytest.approx(stop, rel=1e-6)
    terminal = states[-1].reshape(nodes, len(CONCENTRATIONS))  # [mol/L]
    expected = np.tile(CONCENTRATIONS, (nodes, 1))  # [mol/L]
    for node in range(nodes):
        response = 1 - np.exp(-1) * sum(1/factorial(order)
                                       for order in range(node + 1))  # [-]
        expected[node, 2] += concentration_step * response  # [mol/L]
    np.testing.assert_allclose(terminal, expected, rtol=1e-6, atol=1e-10)
    assert terminal[0, 2] > terminal[1, 2] > terminal[2, 2]
