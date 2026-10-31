"""Merging of ``DynamicExtractor.Inlet`` assignments (issue #449).

Assigning ``Inlet`` merges the given streams into the held ones: a key that
is assigned again replaces the stream held under it, and keys that are not
named are kept. The cases assign real ``LiquidStream`` objects by hand and
through a real ``Connection`` from a static continuous ``Mixer``, so no ODE
backend is needed; the shipped five-species database
(tests/Flowsheet/data/compound_database.json) supplies the properties. The
streams differ only in molar flow, the issue reproduction's 3.0 and
6.0 mol/s, so the held flow identifies the held stream. A connected feed
replaced on a repeated ``SolveFlowsheet`` is covered by
tests/test_flowsheet_resolve_replaces_transfers.py (issue #441).

Related issues: https://github.com/PharmaPy-org/PharmaPy/issues/449,
https://github.com/PharmaPy-org/PharmaPy/issues/441
"""

import numpy as np
import pytest

from PharmaPy.Connections import Connection
from PharmaPy.Containers import Mixer
from PharmaPy.DynamicExtraction import DynamicExtractor
from PharmaPy.Streams import LiquidStream

pytestmark = pytest.mark.unit

COMPOSITION = np.array([0.1, 0.1, 0.1, 0.1, 0.6])  # [-], issue #449
FIRST_FLOW = 3.0  # [mol/s], issue #449 first feed
SECOND_FLOW = 6.0  # [mol/s], issue #449 re-assigned feed
SOLVENT_FLOW = 4.0  # [mol/s], a third flow so the solvent is distinct
TEMP = 300.0  # [K]
NUM_STAGES = 2  # [-], issue #449 extractor
RTOL = 1e-12  # [-], float64 roundoff of the mass-to-mole flow conversion


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


def _stream(path, mole_flow):
    """Build a liquid stream of COMPOSITION at TEMP.

    Parameters
    ----------
    path : str
        Database path.
    mole_flow : float
        Molar flow [mol/s].

    Returns
    -------
    LiquidStream
        Raw stream.
    """
    return LiquidStream(path, temp=TEMP, mole_flow=mole_flow,
                        mole_frac=COMPOSITION)


def test_reassigned_feed_replaces_the_held_feed(path):
    """Hold the second feed, not the first, after assigning 'feed' again."""
    first, second = _stream(path, FIRST_FLOW), _stream(path, SECOND_FLOW)
    extractor = DynamicExtractor(num_stages=NUM_STAGES)
    extractor.Inlet = {'feed': first}

    extractor.Inlet = {'feed': second}

    assert list(extractor.Inlet) == ['feed']
    assert extractor.Inlet['feed'] is second
    assert extractor.Inlet['feed'].mole_flow == pytest.approx(SECOND_FLOW,
                                                              rel=RTOL)


@pytest.mark.parametrize('first_key, later_key', [('feed', 'solvent'),
                                                  ('solvent', 'feed')])
def test_assigning_another_key_keeps_the_held_one(path, first_key, later_key):
    """Keep the stream under a key the later assignment does not name."""
    held = _stream(path, FIRST_FLOW)
    later = _stream(path, SOLVENT_FLOW)
    extractor = DynamicExtractor(num_stages=NUM_STAGES)
    extractor.Inlet = {first_key: held}

    extractor.Inlet = {later_key: later}

    assert set(extractor.Inlet) == {'feed', 'solvent'}
    assert extractor.Inlet[first_key] is held
    assert extractor.Inlet[later_key] is later


def test_reassigning_both_keys_replaces_both(path):
    """Replace both held streams when both keys are assigned again."""
    extractor = DynamicExtractor(num_stages=NUM_STAGES)
    extractor.Inlet = {'feed': _stream(path, FIRST_FLOW),
                       'solvent': _stream(path, SOLVENT_FLOW)}
    feed, solvent = _stream(path, SECOND_FLOW), _stream(path, FIRST_FLOW)

    extractor.Inlet = {'feed': feed, 'solvent': solvent}

    assert extractor.Inlet['feed'] is feed
    assert extractor.Inlet['solvent'] is solvent


def _solved_source(path, mole_flow):
    """Solve a static continuous Mixer of one stream.

    Parameters
    ----------
    path : str
        Database path.
    mole_flow : float
        Molar flow [mol/s] of its single inlet.

    Returns
    -------
    Mixer
        Solved Mixer whose outlet carries ``mole_flow``.
    """
    mixer = Mixer()
    mixer.Inlets = [_stream(path, mole_flow)]
    mixer.solve_unit()
    return mixer


def test_connected_feed_replaces_a_held_feed(path):
    """Let a connection's feed replace a raw or earlier connected feed.

    Each ``Connection`` is built without ``replaced_matter``, so the
    replacement relies on the setter alone.
    """
    raw_feed = _stream(path, FIRST_FLOW)
    solvent = _stream(path, SOLVENT_FLOW)
    extractor = DynamicExtractor(num_stages=NUM_STAGES)
    extractor.Inlet = {'feed': raw_feed, 'solvent': solvent}

    for mole_flow in (SECOND_FLOW, FIRST_FLOW):  # [mol/s]
        connection = Connection(_solved_source(path, mole_flow), extractor)
        connection.transfer_data()

        assert extractor.Inlet['feed'] is connection.transferred_matter
        assert extractor.Inlet['feed'].mole_flow == pytest.approx(
            mole_flow, rel=RTOL)
        assert extractor.Inlet['solvent'] is solvent
