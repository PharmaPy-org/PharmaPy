"""Accepted forms of the ``SimulationExec`` flowsheet argument (issue #448).

``SimulationExec(pure_path, flowsheet)`` accepts a dict adjacency, mapping
each unit name to its ordered successors, or a linear ``'A --> B'`` string.
Any other value, such as a list of unit names or None, raises
``PharmaPyTypeError`` naming the received type and both accepted forms;
earlier releases failed with an unactionable ``UnboundLocalError``. The two
accepted forms keep their graphs and execution orders. Only the shipped
five-species database is read; no unit is built or solved.
"""

import re

import pytest

from PharmaPy.Errors import PharmaPyTypeError
from PharmaPy.SimExec import SimulationExec

pytestmark = pytest.mark.unit

ACCEPTED_FORMS = ("flowsheet must be a dict mapping each unit name to its "
                  "successors, such as {'R01': ['R02'], 'R02': []}, or a "
                  "linear string such as 'R01 --> R02'; got ")


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


@pytest.mark.parametrize('flowsheet, type_name', [
    (['R01', 'R02'], 'list'), (None, 'NoneType'),
    ((('R01', 'R02'),), 'tuple')], ids=['list', 'none', 'tuple'])
def test_unsupported_flowsheet_is_rejected(path, flowsheet, type_name):
    """Name the received type and the two accepted forms."""
    message = '^' + re.escape(ACCEPTED_FORMS + type_name + '.') + '$'
    with pytest.raises(PharmaPyTypeError, match=message):
        SimulationExec(path, flowsheet)

    # Callers that caught the builtin exception still catch it.
    assert issubclass(PharmaPyTypeError, TypeError)


def test_dict_flowsheet_is_unchanged(path):
    """Keep a dict adjacency as the graph, in declaration order."""
    graph = {'R02': ['R03'], 'R01': ['R02'], 'R03': []}
    sim = SimulationExec(path, graph)

    assert sim.graph is graph
    assert list(sim.execution_names) == ['R01', 'R02', 'R03']


def test_string_flowsheet_is_unchanged(path):
    """Convert a linear string into a chain adjacency."""
    sim = SimulationExec(path, 'R01 --> R02 --> R03')

    assert sim.graph == {'R01': ['R02'], 'R02': ['R03'], 'R03': []}
    assert list(sim.execution_names) == ['R01', 'R02', 'R03']
