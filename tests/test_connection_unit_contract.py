"""Units that declare the flowsheet connection contract (issue #445).

``IsothermalFlash``, ``AdiabaticFlash`` and ``ContinuousExtractor`` are
steady-state stream units. They now declare ``oper_mode``, ``is_continuous``
and ``names_states_in``, so a continuous ``Mixer`` can feed them through
``SimulationExec.SolveFlowsheet``; their two outlets must then conserve the
mixed feed. ``Connection`` rejects units that lack the attributes it reads
with a ``PharmaPySpecificationError`` naming the unit, its class and the
missing attributes, before it touches either unit: the flashes cannot be
sources (they publish no ``Outlet`` or time profile), the steady-state
``DistillationColumn`` declares no operating mode, a ``ThreePhaseSettler``
declares its input names only once it is charged, and the
``MetaModelingClass`` template generator declares the lowercase mode
``'batch'``, which no connection route handles.

Fixture: the shipped five-species database
(tests/Flowsheet/data/compound_database.json). Two feeds at 360 K and
1 atm, 1.0 and 0.5 mol/s of asymmetric compositions, mix to 1.5 mol/s of
mole fractions (0.233, 0.167, 0.233, 0.167, 0.2). Like the equimolar
mixture of tests/test_flash_solution_validation.py, it splits into a
liquid and a vapor at 360 K and atmospheric pressure; the tests assert
that both outlets carry matter. The extractor uses constant partition
coefficients, as the ``BatchExtractor`` case of
tests/test_simexec_instantaneous_raw_duration.py, so its split is
deterministic. Expected outlet totals are the sums of the feeds, computed
from the feed specifications. No ODE backend is needed.
"""

import json
import warnings

import numpy as np
import pytest

from PharmaPy.Containers import Mixer
from PharmaPy.Distillation import DistillationColumn
from PharmaPy.Errors import PharmaPySpecificationError
from PharmaPy.Evaporators import AdiabaticFlash, IsothermalFlash
from PharmaPy.Extractors import BatchExtractor, ContinuousExtractor
from PharmaPy.MetaModeler import MetaModelingClass
from PharmaPy.Phases import LiquidPhase
from PharmaPy.SimExec import SimulationExec
from PharmaPy.Streams import LiquidStream
from PharmaPy.ThreePhaseSettler import ThreePhaseSettler

pytestmark = pytest.mark.unit

FEED_1 = np.array([0.3, 0.1, 0.3, 0.1, 0.2])  # [-], mole fractions
FEED_2 = np.array([0.1, 0.3, 0.1, 0.3, 0.2])  # [-], mole fractions
FLOW_1 = 1.0  # [mol/s]
FLOW_2 = 0.5  # [mol/s]
TEMP = 360.0  # [K], two-phase for the equimolar mixture at 1 atm
PRESSURE = 101325.0  # [Pa], standard atmosphere
# [-], constant liquid-liquid partition coefficients of species A to D and
# the solvent; synthetic values that split A and B and leave the others.
PARTITION = np.array([2., .5, 1., 1., 1.])
G_TO_KG = 1e-3  # [kg/g], exact SI conversion
# [-], relative balance tolerance: the flash residual allowance
# sqrt(float64 epsilon) of PharmaPy.Evaporators, with margin, as in
# tests/test_flash_solution_validation.py; the extractor's Newton
# iteration on the phase fraction converges far below it.
BALANCE_RTOL = 1e-7
BATCH_MOLES = 1.0  # [mol], batch charge of the unsupported-mode case
# [m], settler geometry; synthetic, the settler is constructed but never
# solved in these tests.
IMPELLER_DIAMETER = 0.05
VESSEL_DIAMETER = 0.1


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


def _feed(path, fractions, flow):
    """Build a raw liquid feed at the fixture temperature and pressure.

    Parameters
    ----------
    path : str
        Database path.
    fractions : numpy.ndarray
        Mole fractions [-], shape (5,).
    flow : float
        Molar flow [mol/s].

    Returns
    -------
    LiquidStream
        Raw feed stream.
    """
    return LiquidStream(path, temp=TEMP, pres=PRESSURE, mole_flow=flow,
                        mole_frac=fractions)


def _mixer(path):
    """Build the continuous Mixer of the two fixture feeds.

    Parameters
    ----------
    path : str
        Database path.

    Returns
    -------
    Mixer
        Unsolved continuous Mixer.
    """
    mixer = Mixer()
    mixer.Inlets = [_feed(path, FEED_1, FLOW_1), _feed(path, FEED_2, FLOW_2)]
    return mixer


def _outlets(unit):
    """Return the two outlet streams of a flash or extractor.

    Parameters
    ----------
    unit : IsothermalFlash, AdiabaticFlash or ContinuousExtractor
        Solved unit.

    Returns
    -------
    tuple of LiquidStream or VaporStream
        Liquid and vapor outlets of a flash, or the two liquid outlets of
        an extractor.
    """
    if isinstance(unit, ContinuousExtractor):
        return unit.Liquid_2, unit.Liquid_3
    return unit.LiquidOut, unit.VaporOut


UNITS = {
    'isothermal-flash': lambda: IsothermalFlash(pres_drum=PRESSURE),
    'adiabatic-flash': lambda: AdiabaticFlash(pres_drum=PRESSURE),
    'continuous-extractor': lambda: ContinuousExtractor(
        k_fun=lambda x_1, x_2, temp: PARTITION),
    }


@pytest.mark.parametrize('case', list(UNITS))
def test_mixer_feeds_a_stream_unit_through_the_flowsheet(path, case):
    """Conserve the mixed feed in both outlets of the connected unit."""
    sim = SimulationExec(path, {'M01': ['U01'], 'U01': []})
    sim.M01 = _mixer(path)
    sim.U01 = UNITS[case]()
    with warnings.catch_warnings():
        # No missing-Phases warning: the unit needs only an Inlet.
        warnings.simplefilter('error', UserWarning)
        sim.SolveFlowsheet(verbose=False)

    inlet = sim.U01.Inlet
    assert inlet is not sim.M01.Outlet and inlet.transferred_from_uo

    first, second = _outlets(sim.U01)
    species_in = FLOW_1 * FEED_1 + FLOW_2 * FEED_2  # [mol/s]
    total_in = FLOW_1 + FLOW_2  # [mol/s]
    species_out = (first.mole_flow * first.mole_frac
                   + second.mole_flow * second.mole_frac)  # [mol/s]
    np.testing.assert_allclose(species_out, species_in, rtol=0,
                               atol=BALANCE_RTOL * total_in)
    assert first.mole_flow + second.mole_flow == pytest.approx(
        total_in, rel=BALANCE_RTOL)
    # Both outlets carry matter, so the balance is not a trivial split.
    assert 0 < first.mole_flow < total_in
    assert 0 < second.mole_flow < total_in

    with open(path) as handle:
        weights = np.array([value['mw'] for value in json.load(handle)
                            .values()])  # [g/mol], database order
    mass_in = species_in @ weights * G_TO_KG  # [kg/s]
    assert first.mass_flow + second.mass_flow == pytest.approx(
        mass_in, rel=BALANCE_RTOL)


def test_isothermal_flash_separates_at_the_mixed_temperature(path):
    """Flash at the Mixer outlet temperature when no drum value is set."""
    sim = SimulationExec(path, {'M01': ['U01'], 'U01': []})
    sim.M01 = _mixer(path)
    sim.U01 = IsothermalFlash(pres_drum=PRESSURE)
    sim.SolveFlowsheet(verbose=False)

    assert sim.U01.temp == sim.M01.Outlet.temp  # [K]
    assert sim.U01.VaporOut.temp == sim.M01.Outlet.temp  # [K]


@pytest.mark.parametrize('case', ['isothermal-flash', 'adiabatic-flash',
                                  'continuous-extractor'])
def test_unit_without_outlet_cannot_be_a_source(path, case):
    """Reject a unit that publishes no outlet before touching the Mixer."""
    sim = SimulationExec(path, {'U01': ['M01'], 'M01': []})
    sim.U01 = UNITS[case]()
    mixed = (FLOW_1 * FEED_1 + FLOW_2 * FEED_2) / (FLOW_1 + FLOW_2)  # [-]
    sim.U01.Inlet = _feed(path, mixed, FLOW_1 + FLOW_2)  # two-phase feed
    sim.M01 = Mixer()
    class_name = type(sim.U01).__name__
    message = (rf"^Cannot connect these units: source 'U01' \({class_name}\) "
               r"lacks \['Outlet', 'outputs', 'result\.time', "
               r"'names_states_out'\], which a flowsheet source must "
               r"publish\. ")
    with pytest.raises(PharmaPySpecificationError, match=message):
        sim.SolveFlowsheet(verbose=False)

    assert sim.M01.Inlets == []


def test_unit_without_operating_mode_cannot_be_a_destination(path):
    """Reject a DistillationColumn before the transfer mutates any unit."""
    sim = SimulationExec(path, {'M01': ['D01'], 'D01': []})
    sim.M01 = _mixer(path)
    # Sharp A/B split; the column is never solved.
    sim.D01 = DistillationColumn(PRESSURE, q_feed=1.0, LK='A', HK='B',
                                 perc_LK=95.0, perc_HK=5.0)  # [Pa], [-], [%]
    message = (r"^Cannot connect these units: destination 'D01' "
               r"\(DistillationColumn\) lacks \['oper_mode'\], which a "
               r"flowsheet destination must declare\. ")
    with pytest.raises(PharmaPySpecificationError, match=message):
        sim.SolveFlowsheet(verbose=False)

    assert sim.D01.Inlet is None
    # FeedConnection did not run: the Mixer outlet has no upstream profile.
    assert sim.M01.Outlet.y_upstream is None
    # The specific error still satisfies callers that caught AttributeError.
    assert issubclass(PharmaPySpecificationError, AttributeError)


def test_mixer_feeding_a_flash_has_no_fed_duration(path):
    """Report NaN for a Mixer whose consumer is a steady-state flash.

    The flash processes a flow [mol/s] without a time profile, so the
    duration over which the Mixer's raw feeds are consumed is undefined.
    """
    sim = SimulationExec(path, {'M01': ['U01'], 'U01': []})
    sim.M01 = _mixer(path)
    sim.U01 = IsothermalFlash(pres_drum=PRESSURE)
    sim.SolveFlowsheet(verbose=False)

    message = (r"^Raw inlets of instantaneous unit 'M01' have no defined fed "
               r"duration: downstream unit 'U01' has no timed result: its "
               r"solve failed, or it is a steady-state unit such as a flash; "
               r"their totals are reported as NaN\.$")
    with pytest.warns(RuntimeWarning, match=message):
        raw = sim.GetRawMaterials(basis='mole')

    assert [key[:2] for key in raw.index] == [('M01', 'Inlet_0'),
                                              ('M01', 'Inlet_1')]
    assert np.isnan(raw['moles'].to_numpy()).all()


def test_partial_solve_skips_an_unsolved_flash(path):
    """Solve only the Mixer of a flowsheet that ends in a flash."""
    sim = SimulationExec(path, {'M01': ['U01'], 'U01': []})
    sim.M01 = _mixer(path)
    sim.U01 = IsothermalFlash(pres_drum=PRESSURE)
    sim.SolveFlowsheet(pick_units=['M01'], verbose=False)

    assert sim.M01.Outlet.mole_flow == pytest.approx(FLOW_1 + FLOW_2,
                                                     rel=BALANCE_RTOL)
    assert sim.U01.Inlet is None
    assert list(sim.time_processing) == ['M01']


def test_other_units_declare_their_continuity():
    """Declare ``is_continuous`` consistently with each unit's mode.

    ``FeedConnection`` reads a source's ``is_continuous`` to hand over a
    stream profile or a final holdup; these classes did not declare it.
    """
    settler = ThreePhaseSettler(d_impeller=IMPELLER_DIAMETER,
                                d_vessel=VESSEL_DIAMETER, target_compound='A')
    extractor = BatchExtractor(k_fun=lambda x_1, x_2, temp: PARTITION)

    assert (settler.oper_mode, settler.is_continuous) == ('Continuous', True)
    assert (extractor.oper_mode, extractor.is_continuous) == ('Batch', False)


def test_destination_with_an_unsupported_mode_is_rejected(path, tmp_path):
    """Reject a lowercase 'batch' mode, for which nothing would be installed.

    ``MetaModelingClass`` writes unit-operation templates and declares the
    mode of the template it writes, by default ``'batch'``. The check
    rejects it before the transfer instead of letting ``PassPhases`` match
    no route and silently install nothing.
    """
    sim = SimulationExec(path, {'M01': ['D01'], 'D01': []})
    sim.M01 = Mixer()
    sim.M01.Inlets = LiquidPhase(path, temp=TEMP, pres=PRESSURE,
                                 moles=BATCH_MOLES, mole_frac=FEED_1)
    sim.D01 = MetaModelingClass(str(tmp_path / 'template.py'), 'Template')
    attributes = dict(vars(sim.D01))
    message = (r"^'D01' \(MetaModelingClass\) declares oper_mode 'batch'; a "
               r"flowsheet destination other than a Mixer must declare one "
               r"of \['Batch', 'Semibatch', 'Continuous'\] so that "
               r"Connection can install the transferred matter\.$")
    with pytest.raises(PharmaPySpecificationError, match=message):
        sim.SolveFlowsheet(verbose=False)

    assert vars(sim.D01) == attributes
    assert sim.M01.Outlet.y_upstream is None
    assert not (tmp_path / 'template.py').exists()


def test_destination_without_input_names_is_rejected(path):
    """Reject a stream for a settler that declares no input names yet.

    A ``ThreePhaseSettler`` declares ``names_states_in`` when its
    ``Phases`` are assigned, so an uncharged settler cannot receive a
    converted stream.
    """
    sim = SimulationExec(path, {'M01': ['D01'], 'D01': []})
    sim.M01 = _mixer(path)
    sim.D01 = ThreePhaseSettler(d_impeller=IMPELLER_DIAMETER,
                                d_vessel=VESSEL_DIAMETER, target_compound='A')
    attributes = dict(vars(sim.D01))
    message = (r"^Cannot connect these units: destination 'D01' "
               r"\(ThreePhaseSettler\) lacks \['names_states_in'\], which a "
               r"flowsheet destination must declare\. ")
    with pytest.raises(PharmaPySpecificationError, match=message):
        sim.SolveFlowsheet(verbose=False)

    assert vars(sim.D01) == attributes
    assert sim.M01.Outlet.y_upstream is None
