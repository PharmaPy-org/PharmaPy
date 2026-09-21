"""review regressions for real container/connection boundaries.

Core tests use synthetic profiles, real phases, and the holdup RHS. Constant
and linear profiles give independent interpolation and endpoint expectations.


Related issue scope:
https://github.com/PharmaPy-org/PharmaPy/issues/220
https://github.com/PharmaPy-org/PharmaPy/issues/231
https://github.com/PharmaPy-org/PharmaPy/issues/243
"""

from pathlib import Path

import numpy as np
import pytest

from PharmaPy.Connections import Connection, interpolate_inputs
from PharmaPy.Containers import ContinuousHoldup, Mixer
from PharmaPy.Drying_Model import Drying
from PharmaPy.MixedPhases import Cake
from PharmaPy.Phases import LiquidPhase, SolidPhase
from PharmaPy.Results import DynamicResult
from PharmaPy.SimExec import SimulationExec
from PharmaPy.SolidLiquidSep import Filter
from PharmaPy.Streams import LiquidStream
from test_container_liquid_continuation import (
    COLD_FRAC, COLD_TEMP, FEED_FLOW, HOLDUP_MASS, HOT_FRAC, HOT_TEMP,
    RTOL, expected_mixture, integrate_core, make_holdup, profile_source,
    thermo_path,
)


@pytest.mark.integration
def test_batch_filter_connection_initializes_lazy_drying_names(thermo_path):
    grid = np.array([0., 100., 200., 300.])  # [um], test_mixer_container_balances.py four-bin crystal fixture
    population = np.array([0., 1e6, 1.25e5, 0.])  # [#/um], third moment 2e-4 m**3
    solid = SolidPhase(thermo_path, mass_frac=[1, 0, 0, 0, 0],
                        x_distrib=grid, distrib=population)
    cake = Cake()
    cake.Phases = [LiquidPhase(thermo_path, mass=HOLDUP_MASS,
                               mass_frac=COLD_FRAC, temp=COLD_TEMP), solid]
    diameter = 0.1  # [m], arbitrary common bench-scale filter/dryer diameter
    source = Filter(station_diam=diameter)
    source.Outlet = cake
    source.outputs = {'mass': HOLDUP_MASS}  # [kg], synthetic completed filter result
    source.result = DynamicResult({}, time=np.array([0., 1.]))  # [s], one-second batch
    dryer = Drying(number_nodes=2, supercrit_names=[], diam_unit=diameter)
    assert not hasattr(dryer, 'names_states_in')
    Connection(source, dryer).transfer_data()
    assert dryer.names_states_in == ['temp', 'mass_frac']
    assert dryer.CakePhase is not cake
    np.testing.assert_allclose(dryer.Liquid_1.mass_frac, COLD_FRAC, rtol=RTOL)
    np.testing.assert_allclose(dryer.Solid_1.distrib, population, rtol=RTOL)
    assert dryer.Liquid_1.mass == pytest.approx(cake.Liquid_1.mass, rel=RTOL)


def constant_feed(times, hot=False):
    """Construct constant synthetic liquid profiles.

    Parameters
    ----------
    times : numpy.ndarray
        Profile times [s].
    hot : bool, optional
        Choose the hotter, solute-rich feed instead of the cold feed.

    Returns
    -------
    dict
        Mass flow [kg/s], temperature [K], and mass fractions [-] profiles.
    """
    return {'mass_flow': np.full(len(times), FEED_FLOW),  # [kg/s]
            'temp': np.full(len(times), HOT_TEMP if hot else COLD_TEMP),  # [K]
            'mass_frac': np.tile(HOT_FRAC if hot else COLD_FRAC, (len(times), 1))}  # [-]


@pytest.mark.integration
@pytest.mark.parametrize('reverse', [False, True])
def test_mixer_rejects_disjoint_connected_windows(thermo_path, reverse):
    early = np.array([0., 1., 2., 3.])  # [s], first disjoint support
    late = np.array([10., 11., 12., 13.])  # [s], second disjoint support
    windows = [late, early] if reverse else [early, late]
    mixer = Mixer()
    for times in windows:
        Connection(profile_source(thermo_path, constant_feed(times), times), mixer).transfer_data()
    with pytest.raises(ValueError, match=r'Inlet 1.*window.*grid') as error:
        mixer.solve_unit()
    for endpoint in windows[1][[0, -1]]:
        assert str(endpoint) in str(error.value)


@pytest.mark.integration
def test_mixer_rejects_overlap_that_does_not_cover_grid_start(thermo_path):
    chosen = np.array([0., 1., 2., 3.])  # [s]
    later = np.array([1., 2., 3., 4.])  # [s], overlaps but misses the grid start
    mixer = Mixer()
    for times in (chosen, later):
        Connection(profile_source(thermo_path, constant_feed(times), times), mixer).transfer_data()
    with pytest.raises(ValueError, match=r'Inlet 1.*window.*grid'):
        mixer.solve_unit()


@pytest.mark.integration
def test_mixer_interpolates_then_holds_partially_overlapping_feed(thermo_path):
    chosen = np.array([28., 29., 31., 35.])  # [s], deliberately omits upstream endpoint
    shorter = np.array([0., 10., 20., 30.])  # [s], overlaps chosen grid at its start
    first = constant_feed(chosen)
    second = constant_feed(shorter, hot=True)
    second['mass_flow'] = np.array([1., 2., 3., 4.])  # [kg/s], 1 + time/10
    aligned_second = constant_feed(chosen, hot=True)
    aligned_second['mass_flow'] = np.array([3.8, 3.9, 4., 4.])  # [kg/s], linear then held
    mixer = Mixer()
    for feed, times in ((first, chosen), (second, shorter)):
        Connection(profile_source(thermo_path, feed, times), mixer).transfer_data()
    states = mixer.solve_unit()
    expected = expected_mixture(thermo_path, (first, aligned_second))
    np.testing.assert_allclose(mixer.result.time, chosen, rtol=RTOL)
    for actual, target in zip(states, expected):
        np.testing.assert_allclose(actual, target, rtol=RTOL)


@pytest.mark.unit
@pytest.mark.parametrize('vector', [False, True])
def test_interpolate_inputs_holds_true_endpoint(vector):
    times = np.array([28., 29., 31., 35.])  # [s]
    upstream = np.array([0., 10., 20., 30.])  # [s]
    values = np.array([1., 2., 3., 4.])  # [kg/s], linear synthetic ramp
    expected = np.array([3.8, 3.9, 4., 4.])  # [kg/s], endpoint held at 4 kg/s
    if vector:
        values = np.column_stack((values, -values))  # [kg/s], asymmetric channels
        expected = np.column_stack((expected, -expected))  # [kg/s]
    np.testing.assert_allclose(interpolate_inputs(times, upstream, values), expected, rtol=RTOL)


@pytest.mark.integration
def test_holdup_publishes_flow_held_at_upstream_endpoint(thermo_path):
    upstream = np.array([0., 10., 20., 30.])  # [s]
    feed = constant_feed(upstream, hot=True)
    feed['mass_flow'] = np.array([1., 2., 3., 4.])  # [kg/s], 1 + time/10
    unit = make_holdup(thermo_path)
    Connection(profile_source(thermo_path, feed, upstream), unit).transfer_data()
    unit.elapsed_time = 28.0  # [s], last reported in-grid value differs from endpoint
    duration = 7.0  # [s], integration ends at 35 s
    integrate_core(unit, duration)
    np.testing.assert_allclose(unit.outputs['mass_flow'], [3.8, 4.0], rtol=RTOL)
    assert unit.Outlet.mass_flow == pytest.approx(4.0, rel=RTOL)


@pytest.mark.unit
@pytest.mark.parametrize('vector', [False, True])
@pytest.mark.parametrize('scalar', [False, True])
def test_single_point_interpolation_holds_value(vector, scalar):
    upstream = np.zeros(1)  # [s], static-source convention
    values = np.array([[2., 5.]]) if vector else np.array([2.])  # [kg/s], distinct channels
    times = 7.0 if scalar else np.array([-1., 0., 7.])  # [s], before/at/after sample
    expected = values[0] if scalar else np.repeat(values, len(times), axis=0)  # [kg/s]
    np.testing.assert_allclose(interpolate_inputs(times, upstream, values), expected, rtol=RTOL)


@pytest.mark.integration
@pytest.mark.parametrize('destination_kind', ['mixer', 'holdup'])
def test_static_mixer_connects_to_liquid_units(thermo_path, destination_kind):
    source = Mixer()
    source.Inlets = [LiquidStream(thermo_path, mass_frac=COLD_FRAC, temp=COLD_TEMP,
                                  mass_flow=FEED_FLOW) for _ in range(2)]
    source.solve_unit()
    assert isinstance(source.result.time, np.ndarray)
    assert source.result.time.dtype.kind == 'f'
    destination = Mixer() if destination_kind == 'mixer' else make_holdup(thermo_path)
    Connection(source, destination).transfer_data()
    if destination_kind == 'mixer':
        destination.solve_unit()
    else:
        states = np.r_[COLD_FRAC, COLD_TEMP]  # [-], [K], equilibrium with feed
        derivative = destination.unit_model(1.0, states)  # [1/s], [K/s]
        np.testing.assert_array_equal(derivative, np.zeros_like(states))
        times = np.array([1., 2., 3.])  # [s], constant synthetic tank trajectory
        destination.retrieve_results(times, np.tile(states, (len(times), 1)))
        np.testing.assert_allclose(destination.outputs['mass_flow'], 2 * FEED_FLOW, rtol=RTOL)
    np.testing.assert_allclose(destination.Outlet.mass_frac, COLD_FRAC, rtol=RTOL)
    assert destination.Outlet.temp == pytest.approx(COLD_TEMP, rel=RTOL)
    assert destination.Outlet.mass_flow == pytest.approx(2 * FEED_FLOW, rel=RTOL)


@pytest.mark.unit
def test_holdup_single_species_retrieval_keeps_terminal_vector():
    path = str(Path(__file__).resolve().parents[1] / 'data/evaporator/props_nitrogen.json')
    temperature = 78.0  # [K], nitrogen database vaporization reference, liquid fixture
    unit = ContinuousHoldup()
    unit.Phases = LiquidPhase(path, mass=HOLDUP_MASS, mass_frac=[1.], temp=temperature)
    unit.Inlet = LiquidStream(path, mass_flow=FEED_FLOW, mass_frac=[1.], temp=temperature)
    times = np.array([0., 1., 2.])  # [s], constant single-species trajectory
    states = np.column_stack((np.ones(len(times)), np.full(len(times), temperature)))  # [-], [K]
    unit.retrieve_results(times, states)
    assert unit.Liquid_1.mass_frac.shape == (1,)
    assert unit.Outlet.mass_frac.shape == (1,)
    np.testing.assert_allclose(unit.Outlet.mass_frac, [1.], rtol=RTOL)
    assert unit.Outlet.temp == pytest.approx(temperature, rel=RTOL)
    assert unit.Outlet.mass_flow == pytest.approx(FEED_FLOW, rel=RTOL)


@pytest.mark.integration
@pytest.mark.parametrize('static_first', [False, True])
def test_single_sample_mixer_feed_preserves_other_inlet_grid(thermo_path, static_first):
    times = np.array([7., 9., 12.])  # [s], later than static source's timestamp
    static = Mixer()
    static.Inlets = [LiquidStream(thermo_path, mass_frac=COLD_FRAC, temp=COLD_TEMP,
                                  mass_flow=FEED_FLOW) for _ in range(2)]
    static.solve_unit()
    hot_feed = constant_feed(times, hot=True)
    hot_feed['mass_flow'] *= 2  # [kg/s], equal total cold and hot flow
    profiled = profile_source(thermo_path, hot_feed, times)
    destination = Mixer()
    for source in ((static, profiled) if static_first else (profiled, static)):
        Connection(source, destination).transfer_data()
    states = destination.solve_unit()
    cold_feed = constant_feed(times)
    cold_feed['mass_flow'] *= 2  # [kg/s], the two static cold streams combined
    expected = expected_mixture(thermo_path, (cold_feed, hot_feed))
    np.testing.assert_allclose(destination.result.time, times, rtol=RTOL)
    for actual, target in zip(states, expected):
        np.testing.assert_allclose(actual, target, rtol=RTOL)
    assert destination.Outlet.mass_flow == pytest.approx(4 * FEED_FLOW, rel=RTOL)
    np.testing.assert_allclose(destination.Outlet.mass_frac,
                               (COLD_FRAC + HOT_FRAC) / 2, rtol=RTOL)


@pytest.mark.integration
def test_simulation_exec_solves_two_static_mixers(thermo_path):
    simulation = SimulationExec(thermo_path, {'M01': ['M02'], 'M02': []})
    simulation.M01 = Mixer()
    simulation.M01.Inlets = [
        LiquidStream(thermo_path, mass_frac=COLD_FRAC, temp=COLD_TEMP,
                     mass_flow=FEED_FLOW) for _ in range(2)]
    simulation.M02 = Mixer()
    simulation.M02.Inlets = LiquidStream(
        thermo_path, mass_frac=HOT_FRAC, temp=HOT_TEMP, mass_flow=FEED_FLOW)
    simulation.SolveFlowsheet(verbose=False)
    outlet = simulation.M02.Outlet
    assert outlet.mass_flow == pytest.approx(3 * FEED_FLOW, rel=RTOL)
    np.testing.assert_allclose(outlet.mass_frac,
                               (2 * COLD_FRAC + HOT_FRAC) / 3, rtol=RTOL)
    for name in ('M01', 'M02'):
        np.testing.assert_allclose(getattr(simulation, name).result.time, [0.], rtol=RTOL)
        assert simulation.time_processing[name] == pytest.approx(0.0, abs=0.0)


@pytest.mark.integration
@pytest.mark.parametrize('boundary', ['start', 'end'])
@pytest.mark.parametrize('roundoff', [False, True])
def test_mixer_accepts_equal_or_roundoff_shifted_window_endpoints(
        thermo_path, boundary, roundoff):
    chosen = np.array([7., 9., 12.])  # [s]
    support = chosen.copy() if boundary == 'start' else np.array([2., 4., 7.])  # [s]
    if roundoff:
        # One representable clock step, independently smaller than the numerical
        # allowance; move start later or end earlier to exercise both bounds.
        clock_step = np.spacing(chosen[0])  # [s]
        support += clock_step if boundary == 'start' else -clock_step
    mixer = Mixer()
    for times in (chosen, support):
        Connection(profile_source(thermo_path, constant_feed(times), times), mixer).transfer_data()
    mass_flow, fractions, temperature = mixer.solve_unit()
    np.testing.assert_allclose(mixer.result.time, chosen, rtol=RTOL)
    np.testing.assert_allclose(mass_flow, 2 * FEED_FLOW, rtol=RTOL)
    np.testing.assert_allclose(fractions, np.tile(COLD_FRAC, (len(chosen), 1)), rtol=RTOL)
    np.testing.assert_allclose(temperature, COLD_TEMP, rtol=RTOL)


@pytest.mark.unit
@pytest.mark.parametrize('scalar_query', [False, True])
def test_interpolate_inputs_accepts_scalar_upstream_time(scalar_query):
    upstream = 2.0  # [s], scalar representation of one upstream sample
    values = np.array([[2., 5.]])  # [kg/s], distinct synthetic channels
    times = 7.0 if scalar_query else np.array([1., 2., 7.])  # [s]
    expected = values[0] if scalar_query else np.repeat(values, len(times), axis=0)  # [kg/s]
    np.testing.assert_allclose(interpolate_inputs(times, upstream, values), expected, rtol=RTOL)


@pytest.mark.unit
@pytest.mark.parametrize('time', [np.array(1.5), np.float32(1.5)])  # [s]
def test_interpolate_inputs_accepts_zero_dimensional_time(time):
    upstream = np.array([0., 1., 2.])  # [s], three points for local Newton interpolation
    values = np.array([1., 3., 5.])  # [kg/s], ramp 1 + 2*time
    expected = 4.0  # [kg/s], independent ramp value at 1.5 s
    assert interpolate_inputs(time, upstream, values) == pytest.approx(expected, rel=RTOL)


@pytest.mark.unit
@pytest.mark.parametrize('scalar_query', [False, True])
def test_single_sample_interpolation_rejects_multiple_value_rows(scalar_query):
    upstream = np.zeros(1)  # [s], one sample must have exactly one value row
    values = np.array([[2., 5.], [3., 7.]])  # [kg/s], inconsistent two-row profile
    times = 7.0 if scalar_query else np.array([1., 2., 7.])  # [s]
    with pytest.raises(ValueError, match=r'y_inlet.*exactly one.*row.*t_inlet'):
        interpolate_inputs(times, upstream, values)


@pytest.mark.integration
def test_raw_dynamic_feed_uses_connected_mixer_grid(thermo_path):
    from PharmaPy.ProcessControl import DynamicInput
    times = np.array([0., 1., 2.])  # [s], a connected source supplies the horizon
    mixer = Mixer()
    source = profile_source(thermo_path, constant_feed(times), times)
    Connection(source, mixer).transfer_data()
    stream = LiquidStream(thermo_path, mass_flow=FEED_FLOW,
                          mass_frac=COLD_FRAC, temp=COLD_TEMP)
    stream.DynamicInlet = DynamicInput()
    stream.DynamicInlet.add_variable('mass_flow', lambda time: FEED_FLOW * (1 + time))
    mixer.Inlets = stream
    flow, fractions, temperature = mixer.solve_unit()  # [kg/s], [-], [K]
    np.testing.assert_array_equal(mixer.result.time, times)
    np.testing.assert_allclose(flow, FEED_FLOW * (2 + times), rtol=RTOL)
    np.testing.assert_allclose(fractions, np.tile(COLD_FRAC, (len(times), 1)), rtol=RTOL)
    np.testing.assert_allclose(temperature, COLD_TEMP, rtol=RTOL)
