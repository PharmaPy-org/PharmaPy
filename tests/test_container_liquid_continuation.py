"""B009b liquid Mixer and ContinuousHoldup contracts with real streams.

Shipped five-species thermodynamics give asymmetric mixing fixtures. Core tests
use synthetic profiles and the real holdup RHS; the Assimulo test additionally
exercises continuation through the public solver. Existing B009 tests stay intact.
"""

import json

import numpy as np
import pytest
from numpy.polynomial import Polynomial
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

import PharmaPy.Containers as containers
from PharmaPy.Connections import Connection
from PharmaPy.Containers import ContinuousHoldup, Mixer
from PharmaPy.Phases import LiquidPhase
from PharmaPy.Streams import LiquidStream


COLD_FRAC = np.array([0.1, 0.1, 0.1, 0.1, 0.6])  # [-], solvent-rich feed
HOT_FRAC = np.array([0.2, 0.1, 0.2, 0.1, 0.4])  # [-], solute-rich feed
COLD_TEMP = 300.0  # [K], lower mixing bracket and initial tank temperature
HOT_TEMP = 350.0  # [K], upper mixing bracket and incoming tank temperature
FEED_FLOW = 1.0  # [kg/s], sets a ten-second residence time with HOLDUP_MASS
HOLDUP_MASS = 10.0  # [kg], constant inventory for continuation tests
PROFILE_TIME = np.array([7.0, 9.0, 12.0])  # [s], nonzero start catches clock resets
RTOL = 1e-10  # [-], permits float64 property/root error in algebraic balances
ODE_RTOL = 1e-5  # [-], trajectory comparison allowance, above local solver tolerances
CORE_RTOL = 1e-10  # [-], tight reference integration accuracy
CORE_ATOL = 1e-11  # [-] for fractions, [K] for temperature, absolute error floor


@pytest.fixture
def thermo_path(data_path):
    """Return the shipped five-species database path.

    Parameters
    ----------
    data_path : dict
        Repository data directories.

    Returns
    -------
    str
        Thermodynamic database filename.
    """
    return str(data_path['flowsheet'] / 'compound_database.json')


def heat_capacity_polynomials(thermo_path):
    """Read independent mass-basis heat-capacity polynomials.

    Parameters
    ----------
    thermo_path : str
        Shipped database filename, in declared species order.

    Returns
    -------
    list of Polynomial
        Species Cp [J/kg/K], evaluated at temperature [K].
    """
    with open(thermo_path) as database:
        species = json.load(database)
    # Database coefficients are molar; 1000 g/kg is the exact mass conversion.
    return [Polynomial(item['cp_liq']) * 1000 / item['mw']
            for item in species.values()]


def expected_mixture(thermo_path, feeds):
    """Solve component conservation and independently integrated Cp balances.

    Parameters
    ----------
    thermo_path : str
        Shipped database filename.
    feeds : sequence of dict
        Per-feed profiles: mass_flow [kg/s], temp [K], and mass_frac [-],
        with shapes (num_times,), (num_times,), and (num_times, num_species).

    Returns
    -------
    tuple of numpy.ndarray
        Total flow [kg/s], composition [-], and temperature [K] profiles.
    """
    enthalpies = [cp.integ() for cp in heat_capacity_polynomials(thermo_path)]
    # [J/kg], arbitrary common species integration constants cancel by conservation
    flows = sum(feed['mass_flow'] for feed in feeds)  # [kg/s]
    fractions = sum(feed['mass_flow'][:, None] * feed['mass_frac']
                    for feed in feeds) / flows[:, None]  # [-]
    temperatures = []  # [K]
    for row in range(len(flows)):
        energy = sum(feed['mass_flow'][row] * sum(
            fraction * enthalpy(feed['temp'][row])
            for fraction, enthalpy in zip(feed['mass_frac'][row], enthalpies))
            for feed in feeds)  # [J/s]
        mixture_enthalpy = sum(fraction * enthalpy for fraction, enthalpy
                               in zip(fractions[row], enthalpies))  # [J/kg]
        residual = flows[row] * mixture_enthalpy - energy  # [J/s]
        temperatures.append(brentq(residual, COLD_TEMP, HOT_TEMP))
    return flows, fractions, np.asarray(temperatures)


def profile_source(thermo_path, feed, times=PROFILE_TIME):
    """Publish a synthetic liquid profile using a real Mixer and LiquidStream.

    Parameters
    ----------
    thermo_path : str
        Shipped database filename.
    feed : dict
        Profiles on times [s]: mass_flow [kg/s], mass_frac [-], temp [K].
    times : numpy.ndarray, optional
        Absolute profile times [s]; defaults to PROFILE_TIME.

    Returns
    -------
    Mixer
        Upstream unit with a real stream and the standard result contract.
    """
    source = Mixer()
    source.Inlets = LiquidStream(thermo_path, mass_flow=FEED_FLOW,
                                 mass_frac=COLD_FRAC, temp=COLD_TEMP)
    source.Liquid_1 = source.Inlets[0]
    source.names_states_out = source.names_states_in
    source.retrieve_results(times, tuple(feed[key] for key in
                            ('mass_flow', 'mass_frac', 'temp')))
    return source


@pytest.fixture
def feeds():
    """Return unequal feeds over three times and five species.

    Returns
    -------
    tuple of dict
        Synthetic mass flows [kg/s], temperatures [K], and fractions [-].
        Reordered fractions and crossing flows expose time/species confusion.
    """
    return (
        {'mass_flow': np.array([1.0, 2.0, 3.0]),  # [kg/s], increasing feed
         'temp': np.array([300.0, 310.0, 320.0]),  # [K], warming feed
         'mass_frac': np.array([COLD_FRAC, HOT_FRAC, COLD_FRAC[::-1]])},  # [-]
        {'mass_flow': np.array([2.0, 1.0, 4.0]),  # [kg/s], crossing feed
         'temp': np.array([350.0, 340.0, 330.0]),  # [K], cooling feed
         'mass_frac': np.array([HOT_FRAC, COLD_FRAC[::-1], HOT_FRAC[::-1]])},  # [-]
    )


@pytest.mark.parametrize('profiled', [False, True])
@pytest.mark.unit
def test_mixer_inputs_keep_time_and_species_axes(thermo_path, feeds, profiled):
    mixer = Mixer()
    for feed in feeds:
        mixer.Inlets = LiquidStream(thermo_path, mass_flow=feed['mass_flow'][0],
                                    mass_frac=feed['mass_frac'][0],
                                    temp=feed['temp'][0])
    mixer.states_in_dict = {'Inlet': {'mass_frac': len(COLD_FRAC),
                                     'mass_flow': 1, 'temp': 1}}
    times = PROFILE_TIME if profiled else np.array([0.0])  # [s]
    inputs = mixer.get_inputs_new(times)
    for index, feed in enumerate(feeds):
        np.testing.assert_allclose(inputs['mass_frac'][index],
                                   np.tile(feed['mass_frac'][0], (len(times), 1)),
                                   rtol=RTOL)
        assert inputs['mass_flow'][index].shape == times.shape
        assert inputs['temp'][index].shape == times.shape


@pytest.mark.unit
def test_direct_liquid_mixer_solve_commits_balanced_stream(thermo_path, feeds):
    mixer = Mixer(temp_refer=HOT_TEMP)
    static_feeds = [{key: value[:1] for key, value in feed.items()} for feed in feeds]
    for feed in static_feeds:
        mixer.Inlets = LiquidStream(thermo_path, mass_flow=feed['mass_flow'][0],
                                    mass_frac=feed['mass_frac'][0], temp=feed['temp'][0])
    states = mixer.solve_unit()
    expected = expected_mixture(thermo_path, static_feeds)
    for actual, target in zip(states, expected):
        np.testing.assert_allclose(actual, target, rtol=RTOL)
    assert mixer.Outlet.temp == pytest.approx(expected[-1][-1], rel=RTOL)
    np.testing.assert_allclose(mixer.result.time, [0.0], rtol=RTOL)
    assert mixer.Outlet.mass_flow == pytest.approx(expected[0][-1], rel=RTOL)
    assert mixer.Outlet.vol_flow == pytest.approx(
        mixer.Outlet.mass_flow / mixer.Outlet.getDensity(), rel=RTOL)
    assert mixer.Outlet.mole_flow == pytest.approx(
        mixer.Outlet.mass_flow / mixer.Outlet.mw_av * 1000, rel=RTOL)  # exact g/kg


@pytest.mark.unit
def test_mixer_retrieval_commits_temperature_before_reconciling_flow(thermo_path, feeds):
    feed = feeds[1]
    source = profile_source(thermo_path, feed)
    assert source.Outlet.temp == pytest.approx(feed['temp'][-1], rel=RTOL)
    np.testing.assert_allclose(source.Outlet.mass_frac, feed['mass_frac'][-1], rtol=RTOL)
    assert source.Outlet.mass_flow == pytest.approx(feed['mass_flow'][-1], rel=RTOL)
    assert source.Outlet.vol_flow == pytest.approx(
        feed['mass_flow'][-1] / source.Outlet.getDensity(), rel=RTOL)


@pytest.mark.integration
def test_connected_liquid_profiles_select_flow_before_conversion(thermo_path, feeds):
    mixer = Mixer(temp_refer=HOT_TEMP)
    for feed in feeds:
        source = profile_source(thermo_path, feed)
        Connection(source, mixer).transfer_data()
        # Assert the converted handoff, before input defaults can mask lost states.
        for key, target in feed.items():
            np.testing.assert_allclose(mixer.Inlets[-1].y_inlet[key], target, rtol=RTOL)
    states = mixer.solve_unit()
    expected = expected_mixture(thermo_path, feeds)
    for actual, target in zip(states, expected):
        np.testing.assert_allclose(actual, target, rtol=RTOL)
    np.testing.assert_allclose(mixer.result.time, PROFILE_TIME, rtol=RTOL)
    np.testing.assert_allclose(mixer.Outlet.mass_frac, expected[1][-1], rtol=RTOL)
    assert mixer.Outlet.temp == pytest.approx(expected[2][-1], rel=RTOL)
    assert mixer.Outlet.mass_flow == pytest.approx(expected[0][-1], rel=RTOL)


def make_holdup(thermo_path):
    """Create a constant-inventory tank with a warmer, unequal-composition feed.

    Parameters
    ----------
    thermo_path : str
        Shipped database filename.

    Returns
    -------
    ContinuousHoldup
        Inventory HOLDUP_MASS [kg], feed FEED_FLOW [kg/s], fractions [-], and
        temperatures [K] defined by the module's two synthetic liquid charges.
    """
    unit = ContinuousHoldup()
    unit.Phases = LiquidPhase(thermo_path, mass=HOLDUP_MASS,
                              mass_frac=COLD_FRAC, temp=COLD_TEMP)
    unit.Inlet = LiquidStream(thermo_path, mass_flow=FEED_FLOW,
                              mass_frac=HOT_FRAC, temp=HOT_TEMP)
    return unit


@pytest.mark.integration
@pytest.mark.parametrize('contract', ['outputs', 'phase', 'outlet', 'clock', 'connection'])
def test_holdup_retrieval_commits_each_public_contract(thermo_path, feeds, contract):
    unit = make_holdup(thermo_path)
    Connection(profile_source(thermo_path, feeds[1]), unit).transfer_data()
    states = np.column_stack((feeds[0]['mass_frac'], feeds[0]['temp']))  # [-], [K]
    unit.retrieve_results(PROFILE_TIME, states)
    if contract == 'outputs':
        np.testing.assert_allclose(unit.outputs['mass_frac'], states[:, :-1], rtol=RTOL)
        np.testing.assert_allclose(unit.outputs['temp'], states[:, -1], rtol=RTOL)
        np.testing.assert_allclose(unit.outputs['mass_flow'], feeds[1]['mass_flow'], rtol=RTOL)
        assert unit.output is unit.outputs  # retain the legacy singular alias
    elif contract == 'phase':
        np.testing.assert_allclose(unit.Liquid_1.mass_frac, states[-1, :-1], rtol=RTOL)
        assert unit.Liquid_1.temp == pytest.approx(states[-1, -1], rel=RTOL)
        assert unit.Liquid_1.mass == pytest.approx(HOLDUP_MASS, rel=RTOL)
    elif contract == 'outlet':
        assert isinstance(unit.Outlet, LiquidStream)
        assert unit.Outlet is not unit.Liquid_1
        np.testing.assert_allclose(unit.Outlet.mass_frac, states[-1, :-1], rtol=RTOL)
        assert unit.Outlet.temp == pytest.approx(states[-1, -1], rel=RTOL)
        assert unit.Outlet.mass_flow == pytest.approx(feeds[1]['mass_flow'][-1], rel=RTOL)
    elif contract == 'clock':
        assert unit.elapsed_time == pytest.approx(PROFILE_TIME[-1], rel=RTOL)
    else:
        destination = Mixer()
        Connection(unit, destination).transfer_data()
        transferred = destination.Inlets[0]
        np.testing.assert_allclose(transferred.time_upstream, PROFILE_TIME, rtol=RTOL)
        np.testing.assert_allclose(transferred.y_inlet['mass_frac'], states[:, :-1], rtol=RTOL)
        np.testing.assert_allclose(transferred.y_inlet['temp'], states[:, -1], rtol=RTOL)
        np.testing.assert_allclose(transferred.y_inlet['mass_flow'], feeds[1]['mass_flow'], rtol=RTOL)


@pytest.mark.unit
def test_holdup_unit_model_after_synthetic_retrieval(thermo_path, feeds):
    unit = make_holdup(thermo_path)
    states = np.column_stack((feeds[0]['mass_frac'], feeds[0]['temp']))  # [-], [K]
    unit.retrieve_results(PROFILE_TIME, states)
    terminal = np.r_[unit.Liquid_1.mass_frac, unit.Liquid_1.temp]  # [-], [K]
    derivative = unit.unit_model(unit.elapsed_time, terminal)  # [1/s], [K/s]
    expected_fractions = FEED_FLOW / HOLDUP_MASS * (HOT_FRAC - states[-1, :-1])  # [1/s]
    capacities = heat_capacity_polynomials(thermo_path)  # [J/kg/K]
    enthalpies = [cp.integ() for cp in capacities]  # [J/kg]
    reference = 298.15  # [K], established LiquidPhase.getEnthalpy default reference
    inlet_enthalpy = sum(fraction * (enthalpy(HOT_TEMP) - enthalpy(reference))
                         for fraction, enthalpy in zip(HOT_FRAC, enthalpies))  # [J/kg]
    tank_enthalpy = sum(fraction * (enthalpy(states[-1, -1]) - enthalpy(reference))
                        for fraction, enthalpy in zip(states[-1, :-1], enthalpies))  # [J/kg]
    tank_cp = sum(fraction * cp(states[-1, -1])
                  for fraction, cp in zip(states[-1, :-1], capacities))  # [J/kg/K]
    expected_temperature = FEED_FLOW / HOLDUP_MASS * (inlet_enthalpy - tank_enthalpy) / tank_cp
    # [K/s], existing constant-inventory tank energy equation
    np.testing.assert_allclose(derivative[:-1], expected_fractions, rtol=RTOL)
    assert derivative[-1] == pytest.approx(expected_temperature, rel=RTOL)


def integrate_core(unit, duration):
    """Integrate the actual holdup RHS and publish it through real retrieval.

    Parameters
    ----------
    unit : ContinuousHoldup
        Tank whose committed state initializes this segment.
    duration : float
        Segment duration [s].

    Returns
    -------
    tuple of numpy.ndarray
        Absolute times [s] and state rows (mass fractions [-], temperature [K]).
    """
    initial = np.r_[unit.Liquid_1.mass_frac, unit.Liquid_1.temp]  # [-], [K]
    times = np.array([unit.elapsed_time, unit.elapsed_time + duration])  # [s]
    solution = solve_ivp(unit.unit_model, times, initial, t_eval=times,
                         rtol=CORE_RTOL, atol=CORE_ATOL)
    assert solution.success, solution.message
    unit.retrieve_results(solution.t, solution.y.T)
    return solution.t, solution.y.T


@pytest.mark.parametrize('backend', [
    pytest.param('core', marks=pytest.mark.unit),
    pytest.param('assimulo', marks=[pytest.mark.assimulo, pytest.mark.integration]),
])
def test_holdup_segmented_dynamics_match_uninterrupted(thermo_path, backend, monkeypatch):
    if backend == 'assimulo':
        pytest.importorskip('assimulo')
        from assimulo.solvers import CVode

        def configured_solver(problem):
            """Set explicit tolerances on the real optional solver boundary.

            Parameters
            ----------
            problem : assimulo.problem.Explicit_Problem
                Production holdup problem: fractions [-], temperature [K],
                and absolute time [s].

            Returns
            -------
            assimulo.solvers.CVode
                Real integrator with the core reference's local error limits.

            Notes
            -----
            ContinuousHoldup exposes no solver options. Its backend defaults
            permit segment-dependent integration error larger than the test's
            comparison tolerance, so configure accuracy at construction only.
            """
            solver = CVode(problem)
            solver.rtol = CORE_RTOL  # [-]
            solver.atol = CORE_ATOL  # [-] for fractions, [K] for temperature
            return solver

        monkeypatch.setattr(containers, 'CVode', configured_solver)
    segmented = make_holdup(thermo_path)
    whole = make_holdup(thermo_path)
    for unit in (segmented, whole):
        unit.elapsed_time = PROFILE_TIME[0]  # [s]
    first_duration, second_duration = np.diff(PROFILE_TIME)  # [s]
    if backend == 'core':
        first_time, first = integrate_core(segmented, first_duration)
        second_time, second = integrate_core(segmented, second_duration)
        whole_time, uninterrupted = integrate_core(whole, first_duration + second_duration)
    else:
        first_time, first = segmented.solve_unit(first_duration, verbose=False)
        second_time, second = segmented.solve_unit(second_duration, verbose=False)
        whole_time, uninterrupted = whole.solve_unit(first_duration + second_duration, verbose=False)
    np.testing.assert_allclose(second[0], first[-1], rtol=RTOL)
    assert second_time[0] == pytest.approx(first_time[-1], rel=RTOL)
    assert second_time[-1] == pytest.approx(PROFILE_TIME[-1], rel=RTOL)
    assert whole_time[-1] == pytest.approx(PROFILE_TIME[-1], rel=RTOL)
    np.testing.assert_allclose(second[-1], uninterrupted[-1], rtol=ODE_RTOL)
    elapsed = PROFILE_TIME[-1] - PROFILE_TIME[0]  # [s]
    expected_fraction = HOT_FRAC + (COLD_FRAC - HOT_FRAC) * np.exp(
        -FEED_FLOW / HOLDUP_MASS * elapsed)  # [-], exact constant-feed material balance
    np.testing.assert_allclose(second[-1, :-1], expected_fraction, rtol=ODE_RTOL)
    assert segmented.elapsed_time == pytest.approx(PROFILE_TIME[-1], rel=RTOL)
