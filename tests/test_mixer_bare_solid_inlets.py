"""A Mixer rejects bare solid inlets (issue #458).

The ``Mixer`` mixes crystals only inside a mixed-phase inlet (``Slurry``,
``SlurryStream`` or ``Cake``). A bare ``SolidPhase`` or ``SolidStream`` was
mixed as a liquid, so the crystals came out as dissolved solute without an
error. The ``Inlets`` setter, and the in-place replacement a ``Connection``
uses on a re-solve, now raise ``NotImplementedError`` naming each bare
solid by position and class and pointing to the mixed-phase inlets, before
the Mixer changes. Liquid, slurry and cake inlets are covered by the
existing Mixer tests (tests/test_mixer_container_balances.py,
tests/test_mixer_solids_publication.py).

Fixture: the shipped five-species database
(tests/Flowsheet/data/compound_database.json) and the issue's charge: a
1 L liquid of mass fractions (0.1, 0.1, 0.1, 0.1, 0.6) at 300 K and
crystals of A on a 100, 200, 300 um grid with populations (1, 2, 1)e4
#/um, as inventories [kg] or as flows [kg/s]. No ODE backend is needed.
"""

import copy
import re

import numpy as np
import pytest

from PharmaPy.Containers import Mixer
from PharmaPy.Phases import LiquidPhase, SolidPhase
from PharmaPy.Streams import LiquidStream, SolidStream

pytestmark = pytest.mark.unit

COMPOSITION = [0.1, 0.1, 0.1, 0.1, 0.6]  # [-], mass fractions
SOLID = [1., 0., 0., 0., 0.]  # [-], crystals of A
TEMP = 300.0  # [K]
LIQUID_VOLUME = 1e-3  # [m**3]
LIQUID_FLOW = 1e-3  # [m**3/s]
GRID = np.array([100., 200., 300.])  # [um]
POPULATION = np.array([1e4, 2e4, 1e4])  # [#/um], or [#/um/s] as a flow
MESSAGE = ('Mixer cannot mix bare solid inlets ({solids} of the assigned '
           'inlets): it mixes crystals only with their liquid, inside a '
           'Slurry or a Cake (a SlurryStream for flows). Wrap the crystals '
           'and their liquid in one of them; a bare solid would otherwise '
           'be mixed as a dissolved liquid.')


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


def _inlets(path, flows):
    """Build the issue's liquid and crystals.

    Parameters
    ----------
    path : str
        Database path.
    flows : bool
        Build streams [kg/s] instead of inventories [kg].

    Returns
    -------
    tuple
        Liquid and bare solid, both at TEMP [K].
    """
    if flows:
        return (LiquidStream(path, temp=TEMP, vol_flow=LIQUID_FLOW,
                             mass_frac=COMPOSITION),
                SolidStream(path, temp=TEMP, x_distrib=GRID.copy(),
                            distrib=POPULATION.copy(), mass_frac=SOLID))
    return (LiquidPhase(path, temp=TEMP, vol=LIQUID_VOLUME,
                        mass_frac=COMPOSITION),
            SolidPhase(path, temp=TEMP, x_distrib=GRID.copy(),
                       distrib=POPULATION.copy(), mass_frac=SOLID))


def _state(mixer):
    """Copy the Mixer's attributes one level deep.

    Parameters
    ----------
    mixer : Mixer
        Mixer to inspect.

    Returns
    -------
    dict
        Attribute name to a shallow copy of its value, so that lists
        changed in place compare unequal.
    """
    return {name: copy.copy(value) for name, value in vars(mixer).items()}


@pytest.mark.parametrize('flows', [False, True], ids=['phases', 'streams'])
def test_bare_solid_is_rejected_unchanged(path, flows):
    """Reject a liquid assigned together with bare crystals."""
    liquid, solid = _inlets(path, flows)
    mixer = Mixer()
    before = _state(mixer)
    solids = f'{type(solid).__name__} at position 1'
    with pytest.raises(NotImplementedError,
                       match='^' + re.escape(MESSAGE.format(solids=solids))
                       + '$'):
        mixer.Inlets = [liquid, solid]

    assert _state(mixer) == before
    assert mixer.Inlets == []


def test_bare_solid_is_rejected_beside_held_inlets(path):
    """Keep the held liquid when bare crystals are added later."""
    liquid, solid = _inlets(path, flows=False)
    mixer = Mixer()
    mixer.Inlets = liquid
    before = _state(mixer)
    message = MESSAGE.format(solids='SolidPhase at position 0')
    with pytest.raises(NotImplementedError,
                       match='^' + re.escape(message) + '$'):
        mixer.Inlets = solid

    assert _state(mixer) == before
    assert mixer.Inlets == [liquid]


def test_bare_solid_cannot_replace_a_held_inlet(path):
    """Reject bare crystals as the replacement of a transferred copy.

    ``Connection`` calls ``_replace_inlet`` when it transfers again on the
    same flowsheet edge; no PharmaPy unit publishes a bare solid outlet, so
    the method is exercised directly.
    """
    liquid, solid = _inlets(path, flows=False)
    mixer = Mixer()
    mixer.Inlets = liquid
    before = _state(mixer)
    message = MESSAGE.format(solids='SolidPhase at position 0')
    with pytest.raises(NotImplementedError,
                       match='^' + re.escape(message) + '$'):
        mixer._replace_inlet(liquid, solid)

    assert _state(mixer) == before
    assert mixer.Inlets == [liquid]
