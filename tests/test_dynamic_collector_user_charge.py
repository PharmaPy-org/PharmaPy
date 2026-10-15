"""A DynamicCollector rejects a user initial charge (issue #459).

A ``DynamicCollector`` starts from a small provisional seed built from its
``Inlet`` (issue #342) and collects the ``Inlet``: ``solve_unit`` builds
the phases it starts from, so a charge assigned
through ``Phases`` was silently ignored and disappeared from the holdup
and the raw materials. The public ``Phases`` setter now raises
``NotImplementedError`` naming the unit and pointing to issue #342, which
tracks starting from an initial holdup, and leaves the collector
unchanged. The collector's own phases are installed internally, so raw-
inlet and connected collectors are unaffected; their solves are covered by
the existing collector tests (tests/test_dynamic_collector_reuse.py,
tests/test_simexec_raw_materials.py, tests/test_batch_to_flow_connector.py).

Fixture: the issue's 2 kg liquid charge and 1 kg/s feed on the shipped
five-species database (tests/Flowsheet/data/compound_database.json). No
ODE backend is needed: nothing is solved.
"""

import copy
import re

import pytest

from PharmaPy.Containers import DynamicCollector
from PharmaPy.Phases import LiquidPhase
from PharmaPy.Streams import LiquidStream

pytestmark = pytest.mark.unit

FEED_FRACTIONS = [0.1, 0.1, 0.1, 0.1, 0.6]  # [-], mass fractions
CHARGE_FRACTIONS = [0.0, 0.2, 0.0, 0.0, 0.8]  # [-], mass fractions
TEMP = 300.0  # [K]
FEED_FLOW = 1.0  # [kg/s], issue #459 feed
CHARGE_MASS = 2.0  # [kg], issue #459 charge
MESSAGE = ('DynamicCollector does not take an initial charge through '
           'Phases: a collector starts from a small provisional seed that '
           'it builds from its Inlet and collects the Inlet, so an '
           'assigned charge would be ignored. Remove the Phases assignment; '
           'starting from a user initial holdup is not implemented yet '
           '(issue #342).')


def _state(collector):
    """Record the collector's attributes for an unchanged-state check.

    Parameters
    ----------
    collector : DynamicCollector
        Collector to inspect.

    Returns
    -------
    dict
        Attribute name to value: shallow copies of lists and dicts, so
        that in-place changes compare unequal, and the objects themselves
        otherwise, compared by identity.
    """
    return {name: (copy.copy(value) if isinstance(value, (list, dict))
                   else value)
            for name, value in vars(collector).items()}


def _same_state(after, before):
    """Return whether two recorded collector states are the same.

    Parameters
    ----------
    after, before : dict
        States recorded by ``_state``.

    Returns
    -------
    bool
        True if both hold the same attributes, equal containers and
        identical other objects.
    """
    return after.keys() == before.keys() and all(
        after[name] == before[name] if isinstance(before[name], (list, dict))
        else after[name] is before[name] for name in before)


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


@pytest.mark.parametrize('container', [None, tuple, list],
                         ids=['phase', 'tuple', 'list'])
def test_user_charge_is_rejected_unchanged(path, container):
    """Reject the issue's charge and keep the collector as it was."""
    collector = DynamicCollector()
    collector.Inlet = LiquidStream(path, temp=TEMP, mass_flow=FEED_FLOW,
                                   mass_frac=FEED_FRACTIONS)
    charge = LiquidPhase(path, temp=TEMP, mass=CHARGE_MASS,
                         mass_frac=CHARGE_FRACTIONS)
    before = _state(collector)
    with pytest.raises(NotImplementedError,
                       match='^' + re.escape(MESSAGE) + '$'):
        collector.Phases = (charge if container is None
                            else container([charge]))

    assert _same_state(_state(collector), before)
    assert collector.Phases is None
