"""ContinuousHoldup.Phases rejects objects that are not phases (#461).

``ContinuousHoldup.Phases`` accepts a ``LiquidPhase`` (any
``PharmaPy.Phases`` object) or a list or tuple of phases. Anything else
used to fall through both branches: a ``LiquidStream`` was silently
ignored and the previous charge kept, and a dict or a string failed with an
unrelated ``AttributeError``. Each now raises ``PharmaPyTypeError`` naming
the received type and the accepted ones, and the held charge is unchanged.

Fixture: the issue's 5 kg liquid charge and 1 kg/s stream on the shipped
five-species database (tests/Flowsheet/data/compound_database.json).
Nothing is solved, so no ODE backend is needed.
"""

import re

import numpy as np
import pytest

from PharmaPy.Containers import ContinuousHoldup
from PharmaPy.Errors import PharmaPyTypeError
from PharmaPy.Phases import LiquidPhase
from PharmaPy.Streams import LiquidStream

pytestmark = pytest.mark.unit

CHARGE_FRACTIONS = [0.0, 0.2, 0.0, 0.0, 0.8]  # [-], mass fractions
STREAM_FRACTIONS = [0.1, 0.1, 0.1, 0.1, 0.6]  # [-], mass fractions
TEMP = 300.0  # [K]
CHARGE_MASS = 5.0  # [kg], issue #461 charge
STREAM_FLOW = 1.0  # [kg/s], issue #461 stream
MESSAGE = ('ContinuousHoldup.Phases takes a LiquidPhase (a PharmaPy.Phases '
           'object) or a list or tuple of phases; got {}.')


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


@pytest.mark.parametrize('kind', ['stream', 'dict', 'str'])
def test_object_that_is_not_a_phase_is_rejected(path, kind):
    """Reject the assignment and keep the first charge and its snapshot."""
    first = LiquidPhase(path, temp=TEMP, mass=CHARGE_MASS,
                        mass_frac=CHARGE_FRACTIONS)
    holdup = ContinuousHoldup()
    holdup.Phases = first
    snapshot = holdup.__original_phase__
    wrong = {'stream': LiquidStream(path, temp=TEMP, mass_flow=STREAM_FLOW,
                                    mass_frac=STREAM_FRACTIONS),
             'dict': {'liquid': first},
             'str': 'not a phase'}[kind]
    message = MESSAGE.format(type(wrong).__name__)
    with pytest.raises(PharmaPyTypeError,
                       match='^' + re.escape(message) + '$'):
        holdup.Phases = wrong

    assert holdup.Phases == [first]
    assert holdup.Liquid_1 is first
    assert holdup.__original_phase__ is snapshot
    assert holdup.mass == CHARGE_MASS  # [kg]
    np.testing.assert_array_equal(snapshot.mass_frac, first.mass_frac)
    assert issubclass(PharmaPyTypeError, TypeError)
