"""Raw materials of a ContinuousHoldup's liquid charge (issue #456).

``ContinuousHoldup.Phases`` keeps a deep copy of the liquid inventory as
the ``__original_phase__`` initial-holdup snapshot, so
``SimulationExec.GetRawMaterials`` reports the user's charge once, as the
unit's ``Initial_holdup``, beside its raw feed. The solve updates the
charged ``LiquidPhase`` in place, so the snapshot is taken at assignment
and the reported charge keeps the charged composition, also after a
second, continued solve. A feed connected from a ``Mixer`` stays a raw
material of the Mixer only. Earlier releases kept no snapshot and
silently omitted the charge.

Fixture: the shipped five-species database
(tests/Flowsheet/data/compound_database.json) and the charges of
tests/test_container_liquid_continuation.py: a 10 kg solvent-rich liquid
at 300 K, fed 1 kg/s of a solute-rich liquid at 350 K. Expected amounts
are the charge mass times its mass fractions and, on the mole basis,
divided by the database molar masses. The holdup is integrated with
CVode, so every case is marked ``assimulo``.
"""

import json

import numpy as np
import pytest

pytestmark = [pytest.mark.assimulo, pytest.mark.integration]
pytest.importorskip('assimulo')

from PharmaPy.Containers import ContinuousHoldup, Mixer
from PharmaPy.Phases import LiquidPhase
from PharmaPy.SimExec import SimulationExec
from PharmaPy.Streams import LiquidStream

COLD_FRAC = np.array([0.1, 0.1, 0.1, 0.1, 0.6])  # [-], charged liquid
HOT_FRAC = np.array([0.2, 0.1, 0.2, 0.1, 0.4])  # [-], feed liquid
COLD_TEMP = 300.0  # [K], charge temperature
HOT_TEMP = 350.0  # [K], feed temperature
FEED_FLOW = 1.0  # [kg/s]
HOLDUP_MASS = 10.0  # [kg], constant inventory
RUNTIME = 10.0  # [s], one residence time per solve
G_TO_KG = 1e-3  # [kg/g], exact SI conversion
# [-], float64 roundoff of the reported conversions: the holdup is read
# from the stored charge, not from the integration.
RTOL = 1e-12
SPECIES = ('A', 'B', 'C', 'D', 'solvent')  # database order
RUN = {'H01': {'runtime': RUNTIME, 'verbose': False}}


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


def _feed(path):
    """Build the hot feed stream.

    Parameters
    ----------
    path : str
        Database path.

    Returns
    -------
    LiquidStream
        FEED_FLOW [kg/s] of HOT_FRAC [-] at HOT_TEMP [K].
    """
    return LiquidStream(path, mass_flow=FEED_FLOW, mass_frac=HOT_FRAC,
                        temp=HOT_TEMP)


def _holdup(path):
    """Build the holdup charged with the cold liquid.

    Parameters
    ----------
    path : str
        Database path.

    Returns
    -------
    ContinuousHoldup
        Holdup of HOLDUP_MASS [kg] of COLD_FRAC [-] at COLD_TEMP [K].
    """
    unit = ContinuousHoldup()
    unit.Phases = LiquidPhase(path, mass=HOLDUP_MASS, mass_frac=COLD_FRAC,
                              temp=COLD_TEMP)
    return unit


def _assert_charge(sim, path, rows):
    """Assert the raw rows and the charge's amounts on both bases.

    Parameters
    ----------
    sim : SimulationExec
        Solved flowsheet.
    path : str
        Database path.
    rows : list of tuple
        Expected ``(unit, source)`` pairs, in table order; the
        ``('H01', 'Initial_holdup')`` row is checked against the charge.
    """
    with open(path) as handle:
        weights = np.array([value['mw'] for value in json.load(handle)
                            .values()])  # [g/mol]
    species_mass = HOLDUP_MASS * COLD_FRAC  # [kg]
    expected = {'mass': species_mass,
                'mole': species_mass / (weights * G_TO_KG)}  # [kg], [mol]
    for basis, amount in (('mass', 'mass'), ('mole', 'moles')):
        raw = sim.GetRawMaterials(basis=basis)
        assert [key[:2] for key in raw.index] == rows
        position = rows.index(('H01', 'Initial_holdup'))
        assert raw.index[position][2].split('.')[0] == 'LiquidPhase'
        total = expected[basis].sum()  # [kg] or [mol]
        assert raw[amount].iloc[position] == pytest.approx(total, rel=RTOL)
        reported = raw[['%s_%s' % (amount, name) for name in SPECIES]]
        np.testing.assert_allclose(reported.to_numpy(dtype=float)[position],
                                   expected[basis], rtol=RTOL,
                                   atol=RTOL * total)


def test_charge_is_reported_once_with_its_original_composition(path):
    """Report the charge once, as charged, after two solves."""
    sim = SimulationExec(path, {'H01': []})
    sim.H01 = _holdup(path)
    sim.H01.Inlet = _feed(path)
    for _ in range(2):
        sim.SolveFlowsheet(kwargs_run=RUN, verbose=False)

    # The solve moved the held liquid toward the feed composition.
    assert sim.H01.Liquid_1.mass_frac[0] > COLD_FRAC[0]
    _assert_charge(sim, path, [('H01', 'Inlet_0'),
                               ('H01', 'Initial_holdup')])


def test_connected_feed_stays_with_its_source(path):
    """Report a Mixer's feed under the Mixer and the charge under H01."""
    sim = SimulationExec(path, {'M01': ['H01'], 'H01': []})
    sim.M01 = Mixer()
    sim.M01.Inlets = _feed(path)
    sim.H01 = _holdup(path)
    sim.SolveFlowsheet(kwargs_run=RUN, verbose=False)

    assert sim.H01.Inlet.transferred_from_uo
    _assert_charge(sim, path, [('M01', 'Inlet_0'),
                               ('H01', 'Initial_holdup')])
