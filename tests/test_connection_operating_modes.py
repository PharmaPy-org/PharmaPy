"""Connections keep the amount basis of the source outlet (issue #436).

A continuous source (``is_continuous``) hands over a stream with a flow
rate [kg/s]; any other source hands over its final holdup, an amount [kg].
``Connection`` rejects, with ``PharmaPyValueError`` and before either unit
is changed, a holdup for a continuous unit or a ``DynamicCollector`` (the
remedy is a ``BatchToFlowConnector``) and a stream for a batch unit (the
remedy is a ``DynamicCollector``). A ``Mixer`` destination keeps its own
basis rules, and the remedy routes still work.

Unit cases solve static batch or continuous liquid ``Mixer`` sources, which
need no ODE backend, on the shipped five-species database
(tests/Flowsheet/data/compound_database.json); the rejected destination is
never solved. Cases marked ``assimulo`` reproduce the issue with solved
reactor and crystallizer sources: the synthetic ``configured`` tanks of
tests/test_reactor_controls_events_continuation.py on the four-species
database of tests/test_reactor_correctness.py, and an ``MSMPR`` seeded with
three crystal size classes (the triage fixture of issue #447).

Unchanged routes covered elsewhere: continuous Mixer -> CSTR and
-> DynamicCollector (tests/test_simexec_instantaneous_raw_duration.py),
collector -> BatchToFlowConnector -> collector
(tests/test_batch_to_flow_connector.py), batch Mixer -> BatchExtractor
(test_untimed_destination_survives_repeated_solves in
tests/test_simexec_instantaneous_raw_duration.py), continuous Mixer ->
stream units (tests/test_connection_unit_contract.py), and the batch R01 -> CR01 -> F01 and continuous PFR -> collector -> batch
flowsheets (tests/Flowsheet/flowsheet_tests.py).
"""

import json

import numpy as np
import pytest

from PharmaPy.Containers import DynamicCollector, Mixer
from PharmaPy.Crystallizers import BatchCryst, MSMPR
from PharmaPy.Errors import PharmaPyValueError
from PharmaPy.Evaporators import Evaporator, IsothermalFlash
from PharmaPy.Extractors import ContinuousExtractor
from PharmaPy.Phases import LiquidPhase
from PharmaPy.Reactors import BatchReactor, CSTR, PlugFlowReactor
from PharmaPy.SimExec import SimulationExec
from PharmaPy.Streams import BatchToFlowConnector, LiquidStream

COMPOSITION = np.array([0.1, 0.1, 0.1, 0.1, 0.6])  # [-], mass fractions
FLOW = 1.0  # [kg/s], continuous feed
MASS = 2.0  # [kg], batch charge
TEMP = 300.0  # [K]
CYCLE_TIME = 10.0  # [s], connector discharge time
PRESSURE = 101325.0  # [Pa], standard atmosphere
VOL_DRUM = 1e-3  # [m**3], evaporator drum, never solved
TUBE_DIAMETER = 0.0254  # [m], PFR, never solved
TUBE_CELLS = 2  # [-], PFR grid, never solved
G_TO_KG = 1e-3  # [kg/g], exact SI conversion
# [-], constant liquid-liquid partition coefficients of species A to D and
# the solvent, as in tests/test_connection_unit_contract.py
PARTITION = np.array([2., .5, 1., 1., 1.])
MOLES = 10.0  # [mol], batch charge discharged to the extractor
# [-], mole fractions; A between half and twice B, so the constant
# partition coefficients split the feed into two liquids (phase fraction
# 0.5 [-] from the Rachford-Rice equation).
EXTRACTION_FEED = np.array([0.2, 0.2, 0.1, 0.1, 0.4])
RTOL = 1e-12  # [-], float64 roundoff of algebraic sums and conversions

HOLDUP_MESSAGE = (
    r"^Source '{source}' \({source_class}\) is not continuous: its outlet "
    r"is a holdup, an amount \[kg\] without a flow rate, but destination "
    r"'{destination}' \({destination_class}\) takes a feed stream \[kg/s\] "
    r"as its Inlet\. Insert a BatchToFlowConnector between them to "
    r"discharge the holdup as a stream over a cycle time; it accepts a "
    r"single liquid holdup only, solids are not supported yet "
    r"\(issue #423\)\.$")
STREAM_MESSAGE = (
    r"^Source '{source}' \({source_class}\) is continuous: its outlet is a "
    r"stream, a flow rate \[kg/s\], but destination '{destination}' "
    r"\({destination_class}\) is a batch unit charged with a holdup, an "
    r"amount \[kg\], as its Phases\. Collect the stream for a duration in a "
    r"DynamicCollector and connect the collector, whose holdup can charge "
    r"the batch unit\.$")


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


def _mixer(path, batch):
    """Build a one-inlet liquid Mixer.

    Parameters
    ----------
    path : str
        Database path.
    batch : bool
        Charge a holdup of ``MASS`` [kg] instead of a stream of ``FLOW``
        [kg/s].

    Returns
    -------
    Mixer
        Unsolved batch or continuous Mixer.
    """
    mixer = Mixer()
    if batch:
        mixer.Inlets = LiquidPhase(path, temp=TEMP, mass=MASS,
                                   mass_frac=COMPOSITION)
    else:
        mixer.Inlets = LiquidStream(path, temp=TEMP, mass_flow=FLOW,
                                    mass_frac=COMPOSITION)
    return mixer


def _message(template, source, destination, sim):
    """Fill a message pattern with unit names and classes.

    Parameters
    ----------
    template : str
        ``HOLDUP_MESSAGE`` or ``STREAM_MESSAGE``.
    source, destination : str
        Flowsheet unit names.
    sim : SimulationExec
        Executor holding the units.

    Returns
    -------
    str
        Regular expression for the full error message.
    """
    return template.format(
        source=source, destination=destination,
        source_class=type(getattr(sim, source)).__name__,
        destination_class=type(getattr(sim, destination)).__name__)


CONTINUOUS_DESTINATIONS = {
    'CSTR': CSTR,
    'PlugFlowReactor': lambda: PlugFlowReactor(TUBE_DIAMETER, TUBE_CELLS),
    'MSMPR': lambda: MSMPR('A'),
    'IsothermalFlash': lambda: IsothermalFlash(pres_drum=PRESSURE),
    'DynamicCollector': DynamicCollector,
    }

BATCH_DESTINATIONS = {
    'BatchToFlowConnector': lambda: BatchToFlowConnector(CYCLE_TIME),
    'BatchReactor': BatchReactor,
    'BatchCryst': lambda: BatchCryst('A'),
    'Evaporator': lambda: Evaporator(VOL_DRUM),
    }


@pytest.mark.unit
@pytest.mark.parametrize('case', list(CONTINUOUS_DESTINATIONS))
def test_holdup_is_rejected_by_a_continuous_destination(path, case):
    """Reject a batch Mixer feeding a continuous unit or a collector."""
    sim = SimulationExec(path, {'M01': ['U01'], 'U01': []})
    sim.M01 = _mixer(path, batch=True)
    sim.U01 = CONTINUOUS_DESTINATIONS[case]()

    with pytest.raises(PharmaPyValueError,
                       match=_message(HOLDUP_MESSAGE, 'M01', 'U01', sim)):
        sim.SolveFlowsheet(verbose=False)

    assert sim.U01.Inlet is None
    # FeedConnection did not run: the Mixer outlet has no upstream profile.
    assert sim.M01.Outlet.y_upstream is None
    assert list(sim.uos_instances) == ['M01']


@pytest.mark.unit
@pytest.mark.parametrize('case', list(BATCH_DESTINATIONS))
def test_stream_is_rejected_by_a_batch_destination(path, case):
    """Reject a continuous Mixer charging a batch unit or a connector."""
    sim = SimulationExec(path, {'M01': ['U01'], 'U01': []})
    sim.M01 = _mixer(path, batch=False)
    sim.U01 = BATCH_DESTINATIONS[case]()

    with pytest.raises(PharmaPyValueError,
                       match=_message(STREAM_MESSAGE, 'M01', 'U01', sim)):
        sim.SolveFlowsheet(verbose=False)

    assert sim.U01.Phases is None
    assert sim.M01.Outlet.y_upstream is None
    assert list(sim.uos_instances) == ['M01']


@pytest.mark.unit
def test_rejection_is_a_value_error(path):
    """Keep the basis error catchable as a plain ``ValueError``."""
    sim = SimulationExec(path, {'M01': ['R01'], 'R01': []})
    sim.M01 = _mixer(path, batch=True)
    sim.R01 = CSTR()

    with pytest.raises(ValueError):
        sim.SolveFlowsheet(verbose=False)


@pytest.mark.unit
def test_connector_discharges_a_holdup_into_a_continuous_unit(path):
    """Feed an extractor from a batch Mixer through a connector."""
    sim = SimulationExec(path, {'M01': ['B01'], 'B01': ['F01'], 'F01': []})
    sim.M01 = Mixer()
    sim.M01.Inlets = LiquidPhase(path, temp=TEMP, moles=MOLES,
                                 mole_frac=EXTRACTION_FEED)
    sim.B01 = BatchToFlowConnector(CYCLE_TIME)
    sim.F01 = ContinuousExtractor(k_fun=lambda x_1, x_2, temp: PARTITION)
    sim.SolveFlowsheet(verbose=False)

    with open(path) as handle:
        weights = np.array([value['mw'] for value in json.load(handle)
                            .values()])  # [g/mol], database order
    mass = MOLES * EXTRACTION_FEED @ weights * G_TO_KG  # [kg]
    feed = sim.F01.Inlet
    assert feed.transferred_from_uo
    assert feed.mass_flow == pytest.approx(mass / CYCLE_TIME, rel=RTOL)
    assert feed.mole_flow == pytest.approx(MOLES / CYCLE_TIME, rel=RTOL)
    np.testing.assert_allclose(feed.mole_frac, EXTRACTION_FEED, rtol=RTOL)
    raffinate, extract = sim.F01.Liquid_2, sim.F01.Liquid_3
    assert raffinate.mole_flow + extract.mole_flow == pytest.approx(
        MOLES / CYCLE_TIME, rel=RTOL)


@pytest.mark.unit
@pytest.mark.parametrize('batch', [True, False], ids=['batch', 'continuous'])
def test_mixer_destination_keeps_the_source_basis(path, batch):
    """Mix a batch holdup or a stream in a downstream Mixer as before."""
    sim = SimulationExec(path, {'M01': ['M02'], 'M02': []})
    sim.M01 = _mixer(path, batch=batch)
    sim.M02 = Mixer()
    sim.SolveFlowsheet(verbose=False)

    if batch:
        assert sim.M02.oper_mode == 'Batch'
        assert sim.M02.Outlet.mass == pytest.approx(MASS, rel=RTOL)
    else:
        assert sim.M02.oper_mode == 'Continuous'
        assert sim.M02.Outlet.mass_flow == pytest.approx(FLOW, rel=RTOL)


# Solver-backed reproductions of the issue.
REACTOR_RUNTIME = 1.0  # [s], issue #438 reactor runs
CRYST_RUNTIME = 100.0  # [s], issue #447 triage crystallizer runs
CRYST_TEMP = 310.0  # [K], issue #447 triage charge temperature
CRYST_GRID = np.array([10., 20., 40.])  # [um], three size classes
CRYST_SEED = np.array([1e7, 2e7, 1e7])  # [#/um], asymmetric seed
CRYST_SOLID = [1, 0, 0, 0, 0]  # [-], crystals of species A
# [-], volumetric shape factor (crystal volume kv * x**3) of the issue #447
# triage seed, as in tests/test_semibatch_cryst_operating_mode.py; a
# synthetic cube-like value, not a calibrated crystal shape.
SHAPE_FACTOR = 1.0
# [kg/m**3], constant solubility of A, the issue #447 triage kinetics
CRYST_SOLUBILITY = [2000.]
UTILITY_FLOW = 1e-5  # [m**3/s], cooling water
UTILITY_TEMP = 300.0  # [K], cooling water inlet
MSMPR_VOLUME = 9e-4  # [m**3], liquid charge
MSMPR_TANK = 1e-3  # [m**3]
CRYST_FEED = 1e-5  # [m**3/s], liquid feed of the MSMPR


def _seeded_phases(path, vol):
    """Build the triage liquid charge and crystal seed.

    Parameters
    ----------
    path : str
        Database path.
    vol : float
        Liquid volume [m**3].

    Returns
    -------
    tuple
        ``LiquidPhase`` and ``SolidPhase`` at ``CRYST_TEMP`` [K].
    """
    from PharmaPy.Phases import SolidPhase
    return (LiquidPhase(path, temp=CRYST_TEMP, vol=vol,
                        mass_frac=COMPOSITION),
            SolidPhase(path, temp=CRYST_TEMP, x_distrib=CRYST_GRID.copy(),
                       distrib=CRYST_SEED.copy(), kv=SHAPE_FACTOR,
                       mass_frac=CRYST_SOLID))


def _crystallizer(cls, path, **kwargs):
    """Attach the triage kinetics and cooling water to a crystallizer.

    Parameters
    ----------
    cls : type
        Crystallizer class.
    path : str
        Database path.
    **kwargs : dict
        Constructor settings, for example ``vol_tank`` [m**3].

    Returns
    -------
    object
        Crystallizer of species A on the ``1D-FVM`` grid, without phases.
    """
    from PharmaPy.Kinetics import CrystKinetics
    from PharmaPy.Utilities import CoolingWater
    unit = cls('A', method='1D-FVM', **kwargs)
    unit.Kinetics = CrystKinetics(coeff_solub=CRYST_SOLUBILITY)
    unit.Utility = CoolingWater(vol_flow=UTILITY_FLOW, temp_in=UTILITY_TEMP)
    return unit


def _msmpr(path):
    """Build the triage MSMPR with a liquid feed.

    Parameters
    ----------
    path : str
        Database path.

    Returns
    -------
    MSMPR
        Charged MSMPR fed ``CRYST_FEED`` [m**3/s].
    """
    unit = _crystallizer(MSMPR, path, vol_tank=MSMPR_TANK)
    unit.Phases = _seeded_phases(path, MSMPR_VOLUME)
    unit.Inlet = LiquidStream(path, temp=CRYST_TEMP, mass_frac=COMPOSITION,
                              vol_flow=CRYST_FEED)
    return unit


@pytest.mark.assimulo
@pytest.mark.integration
@pytest.mark.parametrize('case', ['CSTR', 'DynamicCollector'])
def test_batch_reactor_holdup_is_rejected(data_path, case):
    """Reject a solved BatchReactor feeding a CSTR or a collector."""
    pytest.importorskip('assimulo')
    from test_reactor_controls_events_continuation import configured
    thermo = str(data_path['integration'] / 'pfr_test_pure_comp.json')
    sim = SimulationExec(thermo, {'R01': ['U01'], 'U01': []})
    sim.R01 = configured(BatchReactor)
    sim.U01 = configured(CSTR) if case == 'CSTR' else DynamicCollector()
    own_inlet = sim.U01.Inlet

    with pytest.raises(PharmaPyValueError,
                       match=_message(HOLDUP_MESSAGE, 'R01', 'U01', sim)):
        sim.SolveFlowsheet(kwargs_run={'R01': {'runtime': REACTOR_RUNTIME,
                                               'verbose': False}},
                           verbose=False)

    assert sim.U01.Inlet is own_inlet
    assert sim.R01.Outlet.y_upstream is None


@pytest.mark.assimulo
@pytest.mark.integration
def test_cstr_stream_is_rejected_by_a_batch_reactor(data_path):
    """Reject a solved CSTR charging a BatchReactor."""
    pytest.importorskip('assimulo')
    from test_reactor_controls_events_continuation import configured
    thermo = str(data_path['integration'] / 'pfr_test_pure_comp.json')
    sim = SimulationExec(thermo, {'R01': ['R02'], 'R02': []})
    sim.R01 = configured(CSTR)
    sim.R02 = configured(BatchReactor)
    own_charge = sim.R02.Phases

    with pytest.raises(PharmaPyValueError,
                       match=_message(STREAM_MESSAGE, 'R01', 'R02', sim)):
        sim.SolveFlowsheet(kwargs_run={'R01': {'runtime': REACTOR_RUNTIME,
                                               'verbose': False}},
                           verbose=False)

    assert sim.R02.Phases is own_charge


@pytest.mark.assimulo
@pytest.mark.integration
def test_msmpr_slurry_is_rejected_by_a_batch_crystallizer(path):
    """Reject an MSMPR slurry stream that BatchCryst took as its holdup."""
    pytest.importorskip('assimulo')
    sim = SimulationExec(path, {'S01': ['D01'], 'D01': []})
    sim.S01 = _msmpr(path)
    sim.D01 = _crystallizer(BatchCryst, path)

    with pytest.raises(PharmaPyValueError,
                       match=_message(STREAM_MESSAGE, 'S01', 'D01', sim)):
        sim.SolveFlowsheet(kwargs_run={'S01': {'runtime': CRYST_RUNTIME,
                                               'verbose': False}},
                           verbose=False)

    assert sim.D01.Phases is None


@pytest.mark.assimulo
@pytest.mark.integration
def test_semibatch_reactor_charges_a_batch_reactor(data_path):
    """Charge a BatchReactor with a SemibatchReactor's final holdup."""
    pytest.importorskip('assimulo')
    from PharmaPy.Reactors import SemibatchReactor
    from test_reactor_controls_events_continuation import configured
    thermo = str(data_path['integration'] / 'pfr_test_pure_comp.json')
    sim = SimulationExec(thermo, {'R01': ['R02'], 'R02': []})
    sim.R01 = configured(SemibatchReactor)
    sim.R02 = configured(BatchReactor)
    run = {'runtime': REACTOR_RUNTIME, 'verbose': False}
    sim.SolveFlowsheet(kwargs_run={'R01': run, 'R02': run}, verbose=False)

    charge = sim.R02.__original_phase__  # copy taken when it was charged
    assert charge.transferred_from_uo
    # The semibatch volume grew by its feed; the batch charge holds it all.
    assert charge.vol == pytest.approx(sim.R01.result.vol[-1], rel=RTOL)
    np.testing.assert_allclose(charge.mole_conc,
                               sim.R01.result.mole_conc[-1], rtol=RTOL)
    assert charge.vol > sim.R01.result.vol[0]
