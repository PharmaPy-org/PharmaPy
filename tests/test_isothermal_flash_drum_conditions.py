"""IsothermalFlash drum conditions follow the current inlet (issue #452).

With ``temp_drum`` or ``pres_drum`` left at None, the flash separates its
feed at the feed's own temperature [K] or pressure [Pa]. These settings
must not be overwritten by the first inlet: a re-assigned inlet, an inlet
changed in place, or a flowsheet re-solved after a feed change is flashed
at its current conditions and gives the same products and heat duty as a
fresh flash. A drum value set by the user keeps precedence.

Fixture: the shipped five-species database
(tests/Flowsheet/data/compound_database.json). An equimolar feed of
1.5 mol/s (or 1.5 mol) splits into a liquid and a vapor at 355-360 K and
1 atm, as in tests/test_flash_solution_validation.py; 150 kPa suppresses
part of the vaporization at 360 K. The flowsheet case uses the asymmetric
Mixer feeds of tests/test_connection_unit_contract.py. Every expectation
is a fresh flash of the same feed. No ODE backend is needed.
"""

import numpy as np
import pytest

from PharmaPy.Containers import Mixer
from PharmaPy.Evaporators import IsothermalFlash
from PharmaPy.Phases import LiquidPhase
from PharmaPy.SimExec import SimulationExec
from PharmaPy.Streams import LiquidStream

pytestmark = pytest.mark.unit

FRACTIONS = np.full(5, 0.2)  # [-], equimolar shipped mixture
FEED_FLOW = 1.5  # [mol/s] for streams, [mol] for phases
TEMP_FIRST = 360.0  # [K], first feed temperature
TEMP_SECOND = 355.0  # [K], later feed temperature, still two-phase
PRESSURE = 101325.0  # [Pa], standard atmosphere
PRESSURE_HIGH = 150000.0  # [Pa], later feed pressure, still two-phase
FEED_1 = np.array([0.3, 0.1, 0.3, 0.1, 0.2])  # [-], mole fractions
FEED_2 = np.array([0.1, 0.3, 0.1, 0.3, 0.2])  # [-], mole fractions
FLOW_1 = 1.0  # [mol/s]
FLOW_2 = 0.5  # [mol/s]
# [-], relative agreement of two identical solves from the same seed:
# the flash residual allowance sqrt(float64 epsilon), with margin, as in
# tests/test_connection_unit_contract.py.
SOLVE_RTOL = 1e-7


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


def _stream(path, temp, pres=PRESSURE):
    """Build the equimolar feed stream.

    Parameters
    ----------
    path : str
        Database path.
    temp : float
        Feed temperature [K].
    pres : float, optional
        Feed pressure [Pa].

    Returns
    -------
    LiquidStream
        Feed of ``FEED_FLOW`` [mol/s].
    """
    return LiquidStream(path, temp=temp, pres=pres, mole_flow=FEED_FLOW,
                        mole_frac=FRACTIONS)


def _assert_same_flash(unit, reference):
    """Assert that two solved flashes have the same conditions and products.

    Parameters
    ----------
    unit, reference : IsothermalFlash
        Solved flashes; ``reference`` is a fresh flash of the same feed.
    """
    assert unit.temp == reference.temp  # [K]
    assert unit.pres == reference.pres  # [Pa]
    for name in ('LiquidOut', 'VaporOut'):
        product = getattr(unit, name)
        expected = getattr(reference, name)
        assert (product.temp, product.pres) == (expected.temp, expected.pres)
        assert product.moles == pytest.approx(expected.moles, rel=SOLVE_RTOL)
        np.testing.assert_allclose(product.mole_frac, expected.mole_frac,
                                   rtol=SOLVE_RTOL)


@pytest.mark.parametrize('change', ['temperature', 'pressure'])
def test_reassigned_inlet_is_flashed_at_its_own_conditions(path, change):
    """Flash a re-assigned inlet like a fresh flash of that inlet."""
    if change == 'temperature':
        drum = {}  # both drum conditions follow the inlet
        second = _stream(path, TEMP_SECOND)
        fresh_feed = _stream(path, TEMP_SECOND)
    else:
        drum = {'temp_drum': TEMP_FIRST}  # [K]; the pressure follows
        second = _stream(path, TEMP_FIRST, PRESSURE_HIGH)
        fresh_feed = _stream(path, TEMP_FIRST, PRESSURE_HIGH)

    reused = IsothermalFlash(**drum)
    reused.Inlet = _stream(path, TEMP_FIRST)
    _, first_heat = reused.solve_unit()  # [W]
    first_vapor = reused.VaporOut.mole_flow  # [mol/s]
    reused.Inlet = second
    # The effective conditions are derived on assignment, before solving.
    assert (reused.temp, reused.pres) == (second.temp, second.pres)
    _, heat = reused.solve_unit()  # [W]

    fresh = IsothermalFlash(**drum)
    fresh.Inlet = fresh_feed
    _, fresh_heat = fresh.solve_unit()  # [W]

    _assert_same_flash(reused, fresh)
    assert (reused.temp, reused.pres) == (second.temp, second.pres)
    assert heat == pytest.approx(fresh_heat, rel=SOLVE_RTOL)
    # The change matters: the first feed vaporizes more.
    assert reused.VaporOut.mole_flow < first_vapor * (1 - SOLVE_RTOL)
    assert heat != pytest.approx(first_heat, rel=SOLVE_RTOL)
    # The user's settings are kept as given.
    assert reused.pres_drum is None
    assert reused.temp_drum == (None if change == 'temperature'
                                else TEMP_FIRST)


def test_inlet_changed_in_place_is_flashed_at_its_current_temperature(path):
    """Read the inlet temperature again when solving."""
    inlet = LiquidPhase(path, temp=TEMP_FIRST, pres=PRESSURE,
                        moles=FEED_FLOW, mole_frac=FRACTIONS)
    reused = IsothermalFlash()
    reused.Inlet = inlet
    reused.solve_unit()
    inlet.updatePhase(temp=TEMP_SECOND)  # [K]
    reused.solve_unit()

    fresh = IsothermalFlash()
    fresh.Inlet = LiquidPhase(path, temp=TEMP_SECOND, pres=PRESSURE,
                              moles=FEED_FLOW, mole_frac=FRACTIONS)
    fresh.solve_unit()

    _assert_same_flash(reused, fresh)
    assert reused.temp == TEMP_SECOND


def test_user_drum_values_keep_precedence(path):
    """Keep a set drum temperature and pressure for every inlet."""
    flash = IsothermalFlash(temp_drum=TEMP_FIRST, pres_drum=PRESSURE)
    flash.Inlet = _stream(path, TEMP_SECOND, PRESSURE_HIGH)
    assert (flash.temp, flash.pres) == (TEMP_FIRST, PRESSURE)

    flash.Inlet = _stream(path, TEMP_SECOND)
    flash.solve_unit()
    assert (flash.temp, flash.pres) == (TEMP_FIRST, PRESSURE)
    assert (flash.VaporOut.temp, flash.VaporOut.pres) == (TEMP_FIRST,
                                                          PRESSURE)
    assert (flash.temp_drum, flash.pres_drum) == (TEMP_FIRST, PRESSURE)


def _mixer(path, temp):
    """Build the continuous Mixer of the two asymmetric feeds.

    Parameters
    ----------
    path : str
        Database path.
    temp : float
        Temperature of both feeds [K].

    Returns
    -------
    Mixer
        Unsolved continuous Mixer.
    """
    mixer = Mixer()
    mixer.Inlets = [
        LiquidStream(path, temp=temp, pres=PRESSURE, mole_flow=FLOW_1,
                     mole_frac=FEED_1),
        LiquidStream(path, temp=temp, pres=PRESSURE, mole_flow=FLOW_2,
                     mole_frac=FEED_2)]
    return mixer


def test_resolved_flowsheet_matches_a_fresh_flowsheet(path):
    """Flash the new Mixer outlet after a flowsheet re-solve.

    ``Mixer.Inlets`` appends, so the feed change is a new Mixer at the
    lower temperature; the flash object is the same.
    """
    sim = SimulationExec(path, {'M01': ['U01'], 'U01': []})
    sim.M01 = _mixer(path, TEMP_FIRST)
    sim.U01 = IsothermalFlash()
    sim.SolveFlowsheet(verbose=False)
    first_vapor = sim.U01.VaporOut.mole_flow  # [mol/s]

    sim.M01 = _mixer(path, TEMP_SECOND)
    sim.SolveFlowsheet(verbose=False)

    fresh = SimulationExec(path, {'M01': ['U01'], 'U01': []})
    fresh.M01 = _mixer(path, TEMP_SECOND)
    fresh.U01 = IsothermalFlash()
    fresh.SolveFlowsheet(verbose=False)

    assert sim.U01.Inlet.temp == pytest.approx(TEMP_SECOND)
    _assert_same_flash(sim.U01, fresh.U01)
    assert sim.U01.VaporOut.mole_flow < first_vapor * (1 - SOLVE_RTOL)
