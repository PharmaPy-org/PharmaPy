"""Result retrieval and plotting without function-of-state metadata (#428).

``DynamicResult`` documents ``di_fstates`` as optional, and several units
(the liquid ``Mixer``, ``BatchToFlowConnector`` and others) publish results
without it. ``get_states_result`` and ``plot_function`` must then treat it
as empty for plain and indexed states, while results that carry it behave
as before.

Fixtures use the shipped five-species database: a continuous liquid Mixer of
1 kg/s (A, B, C, D, solvent = 0.4, 0.05, 0.15, 0.1, 0.3) and 3 kg/s of pure
solvent, and a connector discharging 2 kg of liquid over 10 s. Expected
values are mass balances derived by hand. Plots use the Agg backend and are
closed after each test; no ODE backend is needed.
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

from PharmaPy.Containers import Mixer
from PharmaPy.Phases import LiquidPhase
from PharmaPy.Plotting import get_states_result, plot_function
from PharmaPy.Results import DynamicResult
from PharmaPy.Streams import BatchToFlowConnector, LiquidStream

matplotlib.use('Agg')

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
RTOL = 1e-12  # [-], float64 roundoff of hand mass balances


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
