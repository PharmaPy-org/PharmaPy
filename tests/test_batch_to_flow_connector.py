"""Single-liquid discharge contract of ``BatchToFlowConnector`` (issue #423).

The connector turns one liquid batch holdup (a ``LiquidPhase``, bare or as
the only element of a list or tuple) into a ``LiquidStream`` whose mass flow
[kg/s] is the holdup mass [kg] over ``cycle_time`` [s] times ``flow_mult``
[-]. Every other input is rejected at ``Phases`` assignment, before any
state changes, so a fresh connector stays empty and a loaded one keeps its
previous holdup: ``NotImplementedError`` for valid but unsupported holdups
(slurries, cakes, solid or vapor phases, several phases, and subclasses),
``TypeError`` for streams and non-phase objects, ``ValueError`` for an empty
sequence. Earlier releases, for example, failed later with
``UnboundLocalError`` for slurries, failed with ``TypeError: 'NoneType'
object is not iterable`` when a ``Cake`` or a non-phase object was assigned
to a fresh connector, silently discharged the previous holdup when a
``Cake``, a bare solid or vapor phase, a liquid stream or a non-phase object
was assigned to a loaded one, and silently truncated a phase list or tuple
to its first liquid, dropping solids, vapor or a second liquid.

Issue #426: the connector is an instantaneous transfer. Its ``result`` and
``outputs`` hold one sample at time 0 s with a leading time axis of length
one, so a ``SimulationExec`` flowsheet records zero processing time for it
and hands the downstream unit a constant feed; earlier releases published
``time = None`` and every flowsheet containing the connector failed.

Fixtures follow the issue reproduction with the shipped five-species
database: liquid of mass fractions 0.1/0.1/0.1/0.1/0.6 and pure-A crystals
of 1e7/2e7/1e7 #/um on a 10/20/40 um grid at 310 K. Unit tests need no ODE
backend; the ``assimulo`` tests integrate a ``BatchCryst`` feeding the
connector, and a liquid ``DynamicCollector`` -> connector ->
``DynamicCollector`` flowsheet.
"""

import numpy as np
import pytest

from PharmaPy.Containers import DynamicCollector
from PharmaPy.Crystallizers import BatchCryst
from PharmaPy.Kinetics import CrystKinetics
from PharmaPy.MixedPhases import Cake, Slurry, SlurryStream
from PharmaPy.Phases import LiquidPhase, SolidPhase, VaporPhase
from PharmaPy.SimExec import SimulationExec
from PharmaPy.Streams import (BatchToFlowConnector, LiquidStream, SolidStream,
                              VaporStream)

LIQUID_COMPOSITION = np.array([0.1, 0.1, 0.1, 0.1, 0.6])  # [-], issue #423
SOLID_COMPOSITION = [1.0, 0.0, 0.0, 0.0, 0.0]  # [-], pure A crystals
GRID = np.array([10., 20., 40.])  # [um], issue #423 grid
SEED = np.array([1e7, 2e7, 1e7])  # [#/um], issue #423 total population
KV = 0.5  # [-], issue #423 shape factor
TEMP = 310.0  # [K], issue #423 holdup temperature
LIQUID_VOL = 1e-3  # [m**3], issue #423 liquid holdup
SLURRY_VOL = 1.1e-3  # [m**3], issue #423 slurry volume
LIQUID_MASS = 2.0  # [kg], liquid-route holdup
VAPOR_MASS = 0.5  # [kg], issue #423 comment vapor case
VAPOR_COMPOSITION = [0., 0., 0., 0., 1.]  # [-], solvent vapor
VAPOR_FLOW = 0.05  # [kg/s], any positive vapor stream flow; it is rejected
# Liquid flowsheet of issue #426: a collector fills for UPSTREAM_RUNTIME, the
# connector discharges its holdup, a second collector fills for
# DOWNSTREAM_RUNTIME.
FEED_FLOW = 0.1  # [kg/s], issue #426 raw liquid feed
UPSTREAM_RUNTIME = 20.0  # [s], issue #426 upstream collection
# [s], issue #426 downstream collection; equal to CYCLE_TIME, the run that
# transfers the holdup times FLOW_MULT (the connector's feed never stops).
DOWNSTREAM_RUNTIME = 10.0
# DynamicCollector.solve_unit seeds a liquid collector with mass_flow / 10
# [kg], the feed of 0.1 s (Containers.py, liquid-mixer branch), a
# provisional seed tracked by issue #342. GetRawMaterials does not count it,
# so the upstream holdup is 2.01 kg while its raw feed is 2.0 kg. Drop this
# seed term from the expectations once #342 replaces the seed.
COLLECTOR_SEED_TIME = 0.1  # [s]
# [-], the default CVode rtol: collectors run with default solver options,
# and a constant feed makes the mass balance linear in time.
FLOWSHEET_RTOL = 1e-6
PRESSURE = 2e5  # [Pa], non-default so a reset to 101325 Pa would show
CYCLE_TIME = 10.0  # [s], issue #423 discharge time
FLOW_MULT = 0.5  # [-], non-unit so an ignored multiplier would show
SOLUBILITY = [2000.]  # [kg/m**3], undersaturated, as in issue #423
CRYST_RUNTIME = 1.0  # [s], issue #423 BatchCryst run
RTOL = 1e-12  # [-], float64 roundoff of one division and product


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


def _phases(path):
    """Build the issue liquid and solid phases.

    Parameters
    ----------
    path : str
        Database path.

    Returns
    -------
    tuple
        LiquidPhase of LIQUID_VOL [m**3] and SolidPhase of SEED [#/um].
    """
    return (LiquidPhase(path, temp=TEMP, vol=LIQUID_VOL,
                        mass_frac=LIQUID_COMPOSITION),
            SolidPhase(path, temp=TEMP, x_distrib=GRID.copy(),
                       distrib=SEED.copy(), kv=KV, mass_frac=SOLID_COMPOSITION))


def _slurry(path):
    """Build the issue batch Slurry.

    Parameters
    ----------
    path : str
        Database path.

    Returns
    -------
    Slurry
        Slurry of SLURRY_VOL [m**3].
    """
    slurry = Slurry(vol=SLURRY_VOL, x_distrib=GRID.copy(), distrib=SEED.copy())
    slurry.Phases = _phases(path)
    return slurry


def _slurry_stream(path):
    """Build a slurry stream carrying the SEED values as a rate [#/um/s].

    Parameters
    ----------
    path : str
        Database path.

    Returns
    -------
    SlurryStream
        Stream with LIQUID_MASS / CYCLE_TIME [kg/s] of liquid.
    """
    stream = SlurryStream()
    stream.Phases = [
        LiquidStream(path, temp=TEMP, mass_frac=LIQUID_COMPOSITION,
                     mass_flow=LIQUID_MASS / CYCLE_TIME),
        SolidStream(path, temp=TEMP, x_distrib=GRID.copy(), distrib=SEED.copy(),
                    kv=KV, mass_frac=SOLID_COMPOSITION)]
    return stream


def _cake(path):
    """Build a Cake of the issue liquid and solid phases.

    Parameters
    ----------
    path : str
        Database path.

    Returns
    -------
    Cake
        Cake on the default axial grid.
    """
    cake = Cake()
    cake.Phases = list(_phases(path))
    return cake


class Solvent(LiquidPhase):
    """LiquidPhase subclass whose class name lacks 'Liquid'."""


class CustomSlurry(Slurry):
    """User subclass of Slurry; it must not bypass the rejection."""


class CustomCake(Cake):
    """User subclass of Cake; it must not bypass the rejection."""


def _custom_slurry(path):
    """Build the issue Slurry as a user subclass.

    Parameters
    ----------
    path : str
        Database path.

    Returns
    -------
    CustomSlurry
        Slurry of SLURRY_VOL [m**3].
    """
    slurry = CustomSlurry(vol=SLURRY_VOL, x_distrib=GRID.copy(),
                          distrib=SEED.copy())
    slurry.Phases = _phases(path)
    return slurry


def _custom_cake(path):
    """Build the issue Cake as a user subclass.

    Parameters
    ----------
    path : str
        Database path.

    Returns
    -------
    CustomCake
        Cake on the default axial grid.
    """
    cake = CustomCake()
    cake.Phases = list(_phases(path))
    return cake


def _vapor(path):
    """Build a vapor phase of VAPOR_MASS [kg].

    Parameters
    ----------
    path : str
        Database path.

    Returns
    -------
    VaporPhase
        Solvent vapor at TEMP [K].
    """
    return VaporPhase(path, temp=TEMP, mass=VAPOR_MASS,
                      mole_frac=VAPOR_COMPOSITION)


UNSUPPORTED = (NotImplementedError,
               r"^BatchToFlowConnector discharges exactly one liquid batch "
               r"holdup \(a LiquidPhase\); got \[{names}\]\. Discharging "
               r"solids, slurries, cakes, vapor or several phases is not "
               r"implemented: a liquid-only outlet would drop them from the "
               r"mass balance\.$")
NOT_HOLDUP = (TypeError,
              r"^BatchToFlowConnector\.Phases takes a batch holdup; got "
              r"\[{names}\], which are streams or not phases\. Assign the "
              r"single batch LiquidPhase to discharge\.$")
EMPTY = (ValueError,
         r"^BatchToFlowConnector\.Phases needs one liquid batch holdup "
         r"\(a LiquidPhase\); got an empty sequence\.$")
REJECTED_INPUTS = {
    'slurry': (_slurry, UNSUPPORTED, "'Slurry'"),
    'slurry-subclass': (_custom_slurry, UNSUPPORTED, "'CustomSlurry'"),
    'cake': (_cake, UNSUPPORTED, "'Cake'"),
    'cake-subclass': (_custom_cake, UNSUPPORTED, "'CustomCake'"),
    'liquid-solid-tuple': (_phases, UNSUPPORTED,
                           "'LiquidPhase', 'SolidPhase'"),
    'liquid-vapor-tuple': (lambda path: (_liquid(path), _vapor(path)),
                           UNSUPPORTED, "'LiquidPhase', 'VaporPhase'"),
    'two-liquids-tuple': (lambda path: (_liquid(path), _liquid(path)),
                          UNSUPPORTED, "'LiquidPhase', 'LiquidPhase'"),
    'bare-vapor': (_vapor, UNSUPPORTED, "'VaporPhase'"),
    'bare-solid': (lambda path: _phases(path)[1], UNSUPPORTED, "'SolidPhase'"),
    'bare-solid-stream': (lambda path: _slurry_stream(path).Solid_1,
                          NOT_HOLDUP, "'SolidStream'"),
    'slurry-stream': (_slurry_stream, NOT_HOLDUP, "'SlurryStream'"),
    'liquid-solid-stream-list': (
        lambda path: [_liquid(path), _slurry_stream(path).Solid_1],
        NOT_HOLDUP, "'SolidStream'"),
    'bare-liquid-stream': (
        lambda path: LiquidStream(path, temp=TEMP, mass_frac=LIQUID_COMPOSITION,
                                  mass_flow=LIQUID_MASS / CYCLE_TIME),
        NOT_HOLDUP, "'LiquidStream'"),
    'bare-vapor-stream': (
        lambda path: VaporStream(path, temp=TEMP, mass_flow=VAPOR_FLOW,
                                 mole_frac=VAPOR_COMPOSITION),
        NOT_HOLDUP, "'VaporStream'"),
    'non-phase': (lambda path: {'mass': LIQUID_MASS}, NOT_HOLDUP, "'dict'"),
    'empty-list': (lambda path: [], EMPTY, None),
    }  # builder, (exception type, message pattern), rejected class names


def _liquid(path, phase_class=LiquidPhase):
    """Build the liquid-route holdup.

    Parameters
    ----------
    path : str
        Database path.
    phase_class : type, optional
        ``LiquidPhase`` or a subclass of it.

    Returns
    -------
    LiquidPhase
        LIQUID_MASS [kg] at TEMP [K] and PRESSURE [Pa].
    """
    return phase_class(path, temp=TEMP, pres=PRESSURE, mass=LIQUID_MASS,
                       mass_frac=LIQUID_COMPOSITION)


@pytest.mark.unit
@pytest.mark.parametrize('container, phase_class', [
    (None, LiquidPhase), (list, LiquidPhase), (tuple, LiquidPhase),
    (None, Solvent)], ids=['phase', 'list', 'tuple', 'subclass'])
def test_liquid_holdup_discharges_over_the_cycle(path, container,
                                                 phase_class):
    """Discharge the single liquid holdup mass over the cycle time."""
    liquid = _liquid(path, phase_class)
    connector = BatchToFlowConnector(cycle_time=CYCLE_TIME, flow_mult=FLOW_MULT)
    connector.Phases = liquid if container is None else container([liquid])

    connector.solve_unit()

    expected_flow = LIQUID_MASS / CYCLE_TIME * FLOW_MULT  # [kg/s]
    outlet = connector.Outlet
    assert connector.Liquid_1 is liquid
    assert liquid.name == 'Liquid_1'
    assert connector.has_solids is False  # documented compatibility value
    assert isinstance(outlet, LiquidStream)
    assert outlet.mass_flow == pytest.approx(expected_flow, rel=RTOL)
    assert outlet.temp == TEMP
    assert outlet.pres == PRESSURE
    np.testing.assert_allclose(outlet.mass_frac, LIQUID_COMPOSITION, rtol=RTOL)

    # Issue #426: one instantaneous sample with a leading time axis.
    outputs = connector.outputs
    names = ['temp', 'pres', 'mass_frac', 'mass_flow']
    assert list(connector.names_states_out) == names
    assert list(connector.states_di) == names
    assert {name: connector.states_di[name]['units'] for name in names} == {
        'temp': 'K', 'pres': 'Pa', 'mass_frac': '', 'mass_flow': 'kg/s'}
    assert list(outputs) == names + ['time']
    assert {name: np.shape(outputs[name]) for name in outputs} == {
        'temp': (1,), 'pres': (1,), 'mass_frac': (1, len(LIQUID_COMPOSITION)),
        'mass_flow': (1,), 'time': (1,)}
    np.testing.assert_array_equal(outputs['time'], np.zeros(1))
    np.testing.assert_array_equal(outputs['temp'], [TEMP])
    np.testing.assert_array_equal(outputs['pres'], [PRESSURE])
    np.testing.assert_allclose(outputs['mass_frac'], [LIQUID_COMPOSITION],
                               rtol=RTOL)
    np.testing.assert_allclose(outputs['mass_flow'], [expected_flow],
                               rtol=RTOL)
    for name in outputs:
        assert getattr(connector.result, name) is outputs[name]


@pytest.mark.unit
@pytest.mark.parametrize('loaded', [False, True], ids=['fresh', 'loaded'])
@pytest.mark.parametrize('case', list(REJECTED_INPUTS))
def test_unsupported_input_is_rejected_unchanged(path, case, loaded):
    """Reject anything but one liquid holdup and keep the previous state."""
    build, (error_type, pattern), names = REJECTED_INPUTS[case]
    connector = BatchToFlowConnector(cycle_time=CYCLE_TIME)
    liquid = _liquid(path)
    if loaded:
        connector.Phases = liquid
    previous_phases = connector.Phases
    rejected = build(path)

    with pytest.raises(error_type,
                       match=pattern.format(names=names)) as error:
        connector.Phases = rejected
        connector.solve_unit()  # unfixed code fails here or drops phases

    assert error.traceback[-1].name == 'Phases'
    assert connector.Phases is previous_phases
    if loaded:
        assert connector.Liquid_1 is liquid
        assert connector.has_solids is False
    else:
        assert not hasattr(connector, 'Liquid_1')
        assert not hasattr(connector, 'has_solids')
    assert not hasattr(connector, 'Outlet')


@pytest.mark.assimulo
@pytest.mark.integration
def test_batch_crystallizer_flowsheet_rejects_slurry_discharge(path):
    """Stop a BatchCryst to BatchToFlowConnector flowsheet at the handoff."""
    pytest.importorskip('assimulo')
    cryst = BatchCryst('A', method='1D-FVM', controls={
        'temp': lambda time: TEMP + np.zeros_like(time)})  # [K], isothermal
    cryst.Phases = _phases(path)
    cryst.Kinetics = CrystKinetics(coeff_solub=SOLUBILITY)
    sim = SimulationExec(path, {'CR01': ['B01'], 'B01': []})
    sim.CR01 = cryst
    sim.B01 = BatchToFlowConnector(cycle_time=CYCLE_TIME)

    error_type, pattern = UNSUPPORTED
    with pytest.raises(error_type, match=pattern.format(names="'Slurry'")):
        sim.SolveFlowsheet(kwargs_run={'CR01': {'runtime': CRYST_RUNTIME,
                                                'verbose': False}},
                           verbose=False)

    assert isinstance(cryst.Outlet, Slurry)
    assert sim.B01.Phases is None
    assert not hasattr(sim.B01, 'Outlet')


@pytest.mark.assimulo
@pytest.mark.integration
def test_liquid_flowsheet_discharges_through_connector(path):
    """Solve collector -> connector -> collector and report it (issue #426)."""
    pytest.importorskip('assimulo')
    upstream = DynamicCollector()
    upstream.Inlet = LiquidStream(path, temp=TEMP, mass_flow=FEED_FLOW,
                                  mass_frac=LIQUID_COMPOSITION)
    sim = SimulationExec(path, {'C1': ['B1'], 'B1': ['C2'], 'C2': []})
    sim.C1 = upstream
    sim.B1 = BatchToFlowConnector(cycle_time=CYCLE_TIME, flow_mult=FLOW_MULT)
    sim.C2 = DynamicCollector()

    sim.SolveFlowsheet(kwargs_run={
        'C1': {'runtime': UPSTREAM_RUNTIME, 'verbose': False},
        'C2': {'runtime': DOWNSTREAM_RUNTIME, 'verbose': False}},
        verbose=False)

    holdup = FEED_FLOW * (UPSTREAM_RUNTIME + COLLECTOR_SEED_TIME)  # [kg]
    discharge = holdup / CYCLE_TIME * FLOW_MULT  # [kg/s]
    assert sim.time_processing == pytest.approx(
        {'C1': UPSTREAM_RUNTIME, 'B1': 0.0, 'C2': DOWNSTREAM_RUNTIME},
        rel=RTOL)
    assert sim.B1.Liquid_1.mass == pytest.approx(holdup, rel=FLOWSHEET_RTOL)
    downstream = sim.C2
    assert downstream.get_inputs_new(0.)['Inlet']['mass_flow'] == pytest.approx(
        discharge, rel=FLOWSHEET_RTOL)
    collected = discharge * (DOWNSTREAM_RUNTIME + COLLECTOR_SEED_TIME)  # [kg]
    assert downstream.result.mass[-1] == pytest.approx(collected,
                                                       rel=FLOWSHEET_RTOL)
    np.testing.assert_allclose(downstream.result.mass_frac[-1],
                               LIQUID_COMPOSITION, rtol=FLOWSHEET_RTOL)

    table = sim.result.GetStreamTable()
    source = table.index.droplevel(2)
    connector_rows = table[(source == ('B1', 'Outlet'))]  # keeps row order
    assert len(connector_rows) == 1
    assert connector_rows['mass_flow'].iloc[0] == pytest.approx(
        discharge, rel=FLOWSHEET_RTOL)
    # Raw feed excludes the collector's provisional seed (issue #342).
    raw = sim.GetRawMaterials(totals=False)
    assert [key[:2] for key in raw.index] == [('C1', 'Inlet_0')]
    assert raw['mass'].iloc[0] == pytest.approx(FEED_FLOW * UPSTREAM_RUNTIME,
                                                rel=FLOWSHEET_RTOL)
