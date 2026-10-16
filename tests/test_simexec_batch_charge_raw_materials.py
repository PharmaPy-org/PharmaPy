"""Raw materials of user-charged batch extractors and connectors (#456).

``BatchExtractor`` and ``BatchToFlowConnector`` are charged through
``Phases``. Their setters keep the charge as the ``__original_phase__``
initial-holdup snapshot, so ``SimulationExec.GetRawMaterials`` reports a
user charge once, as the unit's ``Initial_holdup``, and skips a charge
transferred from an upstream batch ``Mixer``, which is counted under the
Mixer only, also after a re-solve. Earlier releases kept no snapshot and
silently omitted the user charge.

Fixture: the shipped five-species database
(tests/Flowsheet/data/compound_database.json) and the charge of the issue
reproduction, 10 mol of an equimolar A/B liquid at 298.15 K and 1 atm. The
extractor uses the issue's constant synthetic partition coefficients, so
it splits deterministically. Expected amounts come from the charge
specification: species moles are the amount times the mole fractions, and
species masses multiply them by the database molar masses. No ODE backend
is needed.
"""

import json

import numpy as np
import pytest

from PharmaPy.Containers import Mixer
from PharmaPy.Extractors import BatchExtractor
from PharmaPy.Phases import LiquidPhase
from PharmaPy.SimExec import SimulationExec
from PharmaPy.Streams import BatchToFlowConnector

pytestmark = pytest.mark.unit

FRACTIONS = np.array([0.5, 0.5, 0., 0., 0.])  # [-], mole fractions
CHARGE_MOLES = 10.0  # [mol], issue #456 charge
TEMP = 298.15  # [K]
PRESSURE = 101325.0  # [Pa], standard atmosphere
# [-], constant liquid-liquid partition coefficients of the issue: A to
# one liquid, B to the other; synthetic values.
PARTITION = np.array([0.1, 10., 1., 1., 1.])
CYCLE_TIME = 100.0  # [s], issue #456 connector discharge time
G_TO_KG = 1e-3  # [kg/g], exact SI conversion
# [-], float64 roundoff of the reported amounts and conversions. Absent
# species are compared in absolute terms against this bound times the
# charge.
RTOL = 1e-12
SPECIES = ('A', 'B', 'C', 'D', 'solvent')  # database order


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


def _charge(path):
    """Build the issue's liquid charge.

    Parameters
    ----------
    path : str
        Database path.

    Returns
    -------
    LiquidPhase
        CHARGE_MOLES [mol] of FRACTIONS [-] at TEMP [K] and PRESSURE [Pa].
    """
    return LiquidPhase(path, temp=TEMP, pres=PRESSURE, moles=CHARGE_MOLES,
                       mole_frac=FRACTIONS)


def _expected(path):
    """Return the species amounts of the charge on both bases.

    Parameters
    ----------
    path : str
        Database path.

    Returns
    -------
    dict
        ``{'mole': mol, 'mass': kg}`` species amounts, database order.
    """
    with open(path) as handle:
        weights = np.array([value['mw'] for value in json.load(handle)
                            .values()])  # [g/mol]
    moles = CHARGE_MOLES * FRACTIONS  # [mol]
    return {'mole': moles, 'mass': moles * weights * G_TO_KG}  # [mol], [kg]


UNITS = {
    'batch-extractor': lambda: BatchExtractor(k_fun=lambda x_1, x_2, temp:
                                              PARTITION),
    'batch-to-flow-connector': lambda: BatchToFlowConnector(
        cycle_time=CYCLE_TIME),
    }


def _assert_single_charge(sim, path, unit, source, phase_class):
    """Assert that the charge is reported once, on both bases.

    Parameters
    ----------
    sim : SimulationExec
        Solved flowsheet.
    path : str
        Database path.
    unit, source : str
        Expected unit name and raw source (``'Initial_holdup'`` or
        ``'Inlet_0'``) of the only raw-material row.
    phase_class : str
        Expected class name of the reported phase.
    """
    expected = _expected(path)
    for basis, amount in (('mass', 'mass'), ('mole', 'moles')):
        raw = sim.GetRawMaterials(basis=basis)
        assert [key[:2] for key in raw.index] == [(unit, source)]
        assert raw.index[0][2].split('.')[0] == phase_class
        species = raw[['%s_%s' % (amount, name) for name in SPECIES]]
        total = expected[basis].sum()  # [kg] or [mol]
        assert raw[amount].iloc[0] == pytest.approx(total, rel=RTOL)
        np.testing.assert_allclose(species.to_numpy(dtype=float)[0],
                                   expected[basis], rtol=RTOL,
                                   atol=RTOL * total)


@pytest.mark.parametrize('case', list(UNITS))
def test_user_charge_is_a_raw_material(path, case):
    """Report the user's charge as the unit's initial holdup."""
    sim = SimulationExec(path, {'U01': []})
    sim.U01 = UNITS[case]()
    sim.U01.Phases = _charge(path)
    sim.SolveFlowsheet(verbose=False)

    _assert_single_charge(sim, path, 'U01', 'Initial_holdup', 'LiquidPhase')


@pytest.mark.parametrize('case', list(UNITS))
def test_transferred_charge_is_counted_under_its_source(path, case):
    """Count a Mixer-transferred charge under the Mixer only, also re-solved.

    The re-solve replaces the transferred charge on the same edge, so the
    unit's snapshot is again a transferred copy.
    """
    sim = SimulationExec(path, {'M01': ['U01'], 'U01': []})
    sim.M01 = Mixer()
    sim.M01.Inlets = _charge(path)
    sim.U01 = UNITS[case]()
    for _ in range(2):
        sim.SolveFlowsheet(verbose=False)
        assert sim.U01.__original_phase__.transferred_from_uo

        _assert_single_charge(sim, path, 'M01', 'Inlet_0', 'LiquidPhase')
