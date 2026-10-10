"""Result retrieval and plotting without function-of-state metadata (#428).

``DynamicResult`` documents ``di_fstates`` as optional, and several units
(the liquid ``Mixer``, ``BatchToFlowConnector`` and others) publish results
without it. ``get_states_result`` and ``plot_function`` must then treat it
as empty for plain and indexed states, while results that carry it behave
as before. Single-sample results whose states lack the leading time axis
(the batch liquid Mixer publishes 0-d amounts and a per-species vector) get
that axis in ``get_states_result``, so indexed picks and plots work, while
states that already have it are returned as published. ``plot_distrib``
works on the solids Mixer, which has no ``fstates_di``, with or without
caller-supplied axes (issue #431), and with supplied axes it returns the
caller's own axes object for both ``times`` and ``x_vals``.

Fixtures use the shipped five-species database: a continuous liquid Mixer of
1 kg/s (A, B, C, D, solvent = 0.4, 0.05, 0.15, 0.1, 0.3) and 3 kg/s of pure
solvent, the issue #428 batch Mixer of 1 kg (0.1, 0.1, 0.1, 0.1, 0.6) and
2 kg (0.2, 0, 0, 0, 0.8), a connector discharging 2 kg of liquid over 10 s,
and the continuous solids Mixer of
tests/test_dynamic_collector_slurry_sources.py. Expected values are mass
balances derived by hand; the solids Mixer's slurry number density is the
fixture crystal number rate divided by the outlet volume flow derived with
that module's ``_feed`` (ideal mixing of the database liquid densities).
Run with MPLBACKEND=Agg; figures are closed after each test. No ODE backend
is needed.
"""

import json

import matplotlib.pyplot as plt
import numpy as np
import pytest

from PharmaPy.Containers import Mixer
from PharmaPy.Phases import LiquidPhase
from PharmaPy.Plotting import get_states_result, plot_distrib, plot_function
from PharmaPy.Results import DynamicResult
from PharmaPy.Streams import BatchToFlowConnector, LiquidStream
from test_dynamic_collector_slurry_sources import (GRID, NUMBER_RATE, _feed,
                                                   _solids_mixer)

pytestmark = pytest.mark.unit

FEED_COMPOSITION = np.array([0.4, 0.05, 0.15, 0.1, 0.3])  # [-]
SOLVENT = np.array([0., 0., 0., 0., 1.])  # [-], pure solvent
FEED_FLOW = 1.0  # [kg/s]
SOLVENT_FLOW = 3.0  # [kg/s], unequal to FEED_FLOW
HOT = 320.0  # [K], feed
COLD = 300.0  # [K], solvent
HOLDUP_MASS = 2.0  # [kg], connector holdup
CYCLE_TIME = 10.0  # [s]
FLOW_MULT = 0.5  # [-], non-unit discharge multiplier
BATCH_MASSES = (1.0, 2.0)  # [kg], issue #428 batch charges
BATCH_COMPOSITIONS = (np.array([0.1, 0.1, 0.1, 0.1, 0.6]),
                      np.array([0.2, 0.0, 0.0, 0.0, 0.8]))  # [-]
RTOL = 1e-12  # [-], float64 roundoff of hand mass balances
# [-], independently recomputed ideal-mixing densities and trapezoidal
# moments; float64 roundoff only, as in the collector module.
DENSITY_RTOL = 1e-10


@pytest.fixture
def path(data_path):
    """Return the shipped five-species property database.

    Parameters
    ----------
    data_path : dict
        Repository fixture directories.

    Returns
    -------
    str
        Database path.
    """
    return str(data_path['flowsheet'] / 'compound_database.json')


@pytest.fixture(autouse=True)
def close_figures():
    """Close every figure a test creates."""
    yield
    plt.close('all')


def _mixer(path):
    """Solve the continuous liquid Mixer fixture.

    Parameters
    ----------
    path : str
        Database path.

    Returns
    -------
    Mixer
        Solved Mixer whose result has no ``di_fstates``.
    """
    mixer = Mixer()
    mixer.Inlets = [LiquidStream(path, temp=HOT, mass_frac=FEED_COMPOSITION,
                                 mass_flow=FEED_FLOW),
                    LiquidStream(path, temp=COLD, mass_frac=SOLVENT,
                                 mass_flow=SOLVENT_FLOW)]
    mixer.solve_unit()
    return mixer


def _mixed_fractions():
    """Return the hand mass balance of the Mixer outlet fractions.

    Returns
    -------
    numpy.ndarray
        Mass fractions [-], shape (5,), in database order.
    """
    return ((FEED_FLOW * FEED_COMPOSITION + SOLVENT_FLOW * SOLVENT)
            / (FEED_FLOW + SOLVENT_FLOW))


def test_liquid_mixer_result_without_fstates(path):
    """Retrieve plain and indexed Mixer states."""
    mixer = _mixer(path)
    assert mixer.result.di_fstates is None

    time, data = get_states_result(mixer.result, 'mass_flow', 'temp',
                                   ('mass_frac', ['solvent', 'A']))

    assert time is mixer.result.time
    np.testing.assert_array_equal(time, [0.])
    # States that already have their time axis are returned as published.
    assert data['mass_flow'] is mixer.result.mass_flow
    np.testing.assert_allclose(data['mass_flow'], [FEED_FLOW + SOLVENT_FLOW],
                               rtol=RTOL)
    assert data['temp'] is mixer.result.temp
    assert COLD < data['temp'][0] < HOT
    fractions = _mixed_fractions()  # [-]
    # Picks keep their order: solvent (index 4) before A (index 0).
    np.testing.assert_allclose(data['mass_frac'], [[fractions[4], fractions[0]]],
                               rtol=RTOL)


def test_connector_result_without_fstates(path):
    """Retrieve plain and indexed BatchToFlowConnector states."""
    connector = BatchToFlowConnector(cycle_time=CYCLE_TIME, flow_mult=FLOW_MULT)
    connector.Phases = LiquidPhase(path, temp=COLD, mass=HOLDUP_MASS,
                                   mass_frac=FEED_COMPOSITION)
    connector.solve_unit()
    assert connector.result.di_fstates is None

    time, data = get_states_result(connector.result, 'mass_flow',
                                   ('mass_frac', ['D', 'B']))

    np.testing.assert_array_equal(time, [0.])
    np.testing.assert_allclose(data['mass_flow'],
                               [HOLDUP_MASS / CYCLE_TIME * FLOW_MULT],
                               rtol=RTOL)
    np.testing.assert_allclose(data['mass_frac'],
                               [[FEED_COMPOSITION[3], FEED_COMPOSITION[1]]],
                               rtol=RTOL)


def test_result_with_fstates_is_unchanged():
    """Retrieve states and indexed functions of state as before."""
    time = np.array([0., 1., 2.])  # [s]
    temp = np.array([300., 305., 310.])  # [K]
    rates = np.array([[1., 2.], [3., 4.], [5., 6.]])  # [mol/s]
    result = DynamicResult(
        {'temp': {'units': 'K', 'dim': 1}},
        {'rates': {'units': 'mol/s', 'dim': 2, 'index': ['r1', 'r2']}},
        time=time, temp=temp, rates=rates)

    out_time, data = get_states_result(result, 'temp', ('rates', ['r2']))

    assert out_time is time
    assert data['temp'] is temp
    np.testing.assert_array_equal(data['rates'], rates[:, [1]])


def test_plot_function_without_fstates(path):
    """Plot Mixer states with units and legends from states_di alone."""
    mixer = _mixer(path)
    assert not hasattr(mixer, 'fstates_di')

    fig, axes = plot_function(mixer, ['mass_flow', ('mass_frac', ['solvent', 'A'])],
                              nrows=2)

    fractions = _mixed_fractions()  # [-]
    flow_line, = axes[0].lines
    np.testing.assert_allclose(flow_line.get_ydata(), [FEED_FLOW + SOLVENT_FLOW],
                               rtol=RTOL)
    assert axes[0].get_ylabel().startswith('mass_flow (')
    plotted = [line.get_ydata()[0] for line in axes[1].lines]
    np.testing.assert_allclose(plotted, [fractions[4], fractions[0]], rtol=RTOL)
    assert [text.get_text() for text in axes[1].get_legend().get_texts()] == [
        'solvent', 'A']
    assert axes[1].get_ylabel() == 'mass_frac'  # dimensionless: no units


def _batch_mixer(path):
    """Solve the issue #428 batch liquid Mixer.

    Parameters
    ----------
    path : str
        Database path.

    Returns
    -------
    Mixer
        Solved batch Mixer; its result has one time sample, 0-d ``mass``
        and ``temp``, and ``mass_frac`` of shape (5,).
    """
    mixer = Mixer()
    mixer.Inlets = [
        LiquidPhase(path, mass=mass, temp=temp, mass_frac=fractions)
        for mass, temp, fractions in zip(BATCH_MASSES, (COLD, HOT),
                                         BATCH_COMPOSITIONS)]
    mixer.solve_unit()
    return mixer


def _batch_fractions():
    """Return the hand mass balance of the batch Mixer fractions.

    Returns
    -------
    numpy.ndarray
        Mass fractions [-], shape (5,).
    """
    return (sum(mass * fractions for mass, fractions
                in zip(BATCH_MASSES, BATCH_COMPOSITIONS)) / sum(BATCH_MASSES))


def test_batch_mixer_single_sample_states(path):
    """Add the missing time axis to single-sample batch Mixer states."""
    mixer = _batch_mixer(path)
    assert np.ndim(mixer.result.temp) == 0
    assert np.shape(mixer.result.mass_frac) == (5,)

    time, data = get_states_result(mixer.result, 'mass', 'temp',
                                   ('mass_frac', ['solvent', 'A']))

    assert time is mixer.result.time
    assert np.shape(data['mass']) == (1,)
    np.testing.assert_allclose(data['mass'], [sum(BATCH_MASSES)], rtol=RTOL)
    np.testing.assert_array_equal(data['temp'], [mixer.result.temp])
    assert COLD < data['temp'][0] < HOT
    fractions = _batch_fractions()  # [-]
    np.testing.assert_allclose(data['mass_frac'],
                               [[fractions[4], fractions[0]]], rtol=RTOL)


def test_plot_function_on_batch_mixer(path):
    """Plot single-sample batch Mixer states, picks in order."""
    mixer = _batch_mixer(path)

    fig, axes = plot_function(mixer, ['mass', 'temp',
                                      ('mass_frac', ['solvent', 'A'])],
                              nrows=3)

    mass_line, = axes[0].lines
    np.testing.assert_allclose(mass_line.get_ydata(), [sum(BATCH_MASSES)],
                               rtol=RTOL)
    np.testing.assert_array_equal(axes[1].lines[0].get_ydata(),
                                  [mixer.result.temp])
    fractions = _batch_fractions()  # [-]
    plotted = [line.get_ydata()[0] for line in axes[2].lines]
    np.testing.assert_allclose(plotted, [fractions[4], fractions[0]],
                               rtol=RTOL)
    assert [text.get_text() for text in axes[2].get_legend().get_texts()] == [
        'solvent', 'A']


def _expected_number_density(path):
    """Return the solids Mixer outlet number density from the fixtures.

    Parameters
    ----------
    path : str
        Database path.

    Returns
    -------
    numpy.ndarray
        Crystal number rate [#/um/s] over the mixed outlet volume flow
        [m**3/s], i.e. [#/m**3/um], shape (len(GRID),).
    """
    with open(path) as handle:
        data = json.load(handle)
    database = {'species': list(data),
                'rho_liq': np.array([data[name]['rho_liq']
                                     for name in data])}  # [kg/m**3]
    return NUMBER_RATE / _feed(database, mixed=True)['vol_flow']


def _solved_solids_mixer(path):
    """Solve the continuous solids Mixer of liquid and slurry.

    Parameters
    ----------
    path : str
        Database path.

    Returns
    -------
    Mixer
        Solved solids Mixer without ``fstates_di``.
    """
    mixer = _solids_mixer(path)
    mixer.solve_unit()
    assert not hasattr(mixer, 'fstates_di')
    return mixer


@pytest.mark.parametrize('supplied', [False, True], ids=['new', 'axes'])
def test_plot_distrib_profile_along_the_grid(path, supplied):
    """Plot the published distribution with or without caller axes."""
    mixer = _solved_solids_mixer(path)
    if supplied:
        fig, axes = plt.subplots()
        returned = plot_distrib(mixer, ['distrib'], 'x_cryst', axes=axes,
                                times=[0.])
        assert returned is axes
    else:
        fig, axes = plot_distrib(mixer, ['distrib'], 'x_cryst', times=[0.])

    line, = axes.lines
    np.testing.assert_array_equal(line.get_xdata(), GRID)
    np.testing.assert_allclose(line.get_ydata(),
                               _expected_number_density(path),
                               rtol=DENSITY_RTOL)
    assert axes.get_ylabel().startswith('distrib (')
    assert [text.get_text() for text in fig.texts] == ['x_cryst']


@pytest.mark.parametrize('supplied', [False, True], ids=['new', 'axes'])
def test_plot_distrib_time_profile_at_a_size(path, supplied):
    """Plot the distribution value at one grid size against time."""
    mixer = _solved_solids_mixer(path)
    size = GRID[1]  # [um], a grid node, so no interpolation is involved

    if supplied:
        _, axes = plt.subplots()
        returned = plot_distrib(mixer, ['distrib'], 'x_cryst', axes=axes,
                                x_vals=[size])
        assert returned is axes
    else:
        _, returned = plot_distrib(mixer, ['distrib'], 'x_cryst',
                                   x_vals=[size])
        axes, = returned  # created figures keep the one-element list

    line, = axes.lines
    np.testing.assert_array_equal(line.get_xdata(), [0.])
    np.testing.assert_allclose(line.get_ydata(),
                               [_expected_number_density(path)[1]],
                               rtol=DENSITY_RTOL)
    assert axes.get_ylabel().startswith('distrib (')


def test_multi_sample_vector_state_is_returned_as_published():
    """Never reshape a multi-sample result, even a length-matching vector."""
    time = np.array([0., 1., 2.])  # [s]
    # [mol/s], one value per time; its length equals its index by design,
    # so only the single-sample guard prevents a spurious reshape.
    rates = np.array([1., 2., 3.])
    result = DynamicResult(
        {'rates': {'units': 'mol/s', 'dim': 3, 'index': ['r1', 'r2', 'r3']}},
        time=time, rates=rates)

    _, data = get_states_result(result, 'rates')

    assert data['rates'] is rates


def test_single_sample_state_with_time_axis_is_returned_as_published():
    """Leave a single-sample state that already has its time axis alone."""
    time = np.array([0.])  # [s]
    # [mol/s], one sample of a scalar state; its index has three entries,
    # so its length does not match and no time axis is added.
    rate = np.array([4.])
    result = DynamicResult(
        {'rate': {'units': 'mol/s', 'dim': 1, 'index': ['r1', 'r2', 'r3']}},
        time=time, rate=rate)

    _, data = get_states_result(result, 'rate')

    assert data['rate'] is rate
