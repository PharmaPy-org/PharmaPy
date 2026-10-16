"""Rejection of flowsheet fan-in to non-Mixer units (issue #444).

Only a ``Mixer`` combines the streams ``Connection.PassPhases`` hands it
into one balanced mixture; any other destination would replace its
``Inlet`` or ``Phases`` or (``BatchCryst``, ``Drying``) stack the phases
unmixed, so a second predecessor's material
would be silently dropped or left unmixed.
``SimulationExec.SolveFlowsheet`` must therefore raise
``PharmaPyValueError`` naming the destination, its class and its
predecessors before any unit is solved or any stream is transferred,
irrespective of ``pick_units`` and whether the destination has an
adjacency entry. A ``Mixer`` fed by several units still mixes them. The
remedy depends on the destination: predecessors of one amount basis go
through a ``Mixer``, while a semibatch unit fed by a holdup and a stream
takes one connected input and the other assigned directly, because a
``Mixer`` cannot mix an inventory [kg] with a flow rate [kg/s], and a
``DynamicExtractor`` takes its feed connected and its solvent assigned.
A collector collects streams only, so a holdup is first discharged through
a ``BatchToFlowConnector``. A predecessor whose basis is unknown (not
assigned, or a ``Mixer`` fed only through the flowsheet) leaves the
``Mixer`` remedy.

Unit cases solve real batch liquid ``Mixer`` units and a
``BatchToFlowConnector``, which need no ODE backend, on the shipped
five-species database (tests/Flowsheet/data/compound_database.json).
Predecessor ``M01`` holds MASS_1 = 2.0 kg of FEED_A at 300 K and ``M02``
MASS_2 = 3.0 kg of FEED_B at 320 K; the mixed amounts expected downstream
are their sums. The ``assimulo`` case is the issue reproduction: two
continuous Mixers of 1.5 kg/s and 0.25 kg/s feeding one
``DynamicCollector``, which solves with CVode when the guard is missing.

Related issues: https://github.com/PharmaPy-org/PharmaPy/issues/444,
https://github.com/PharmaPy-org/PharmaPy/issues/437
"""

import re

import numpy as np
import pytest

from PharmaPy.Containers import Mixer
from PharmaPy.Errors import PharmaPyValueError
from PharmaPy.Evaporators import Evaporator
from PharmaPy.Phases import LiquidPhase
from PharmaPy.Reactors import SemibatchReactor
from PharmaPy.SimExec import SimulationExec
from PharmaPy.Streams import BatchToFlowConnector, LiquidStream

FEED_A = np.array([0.1, 0.1, 0.1, 0.1, 0.6])  # [-], issue #444 feed
FEED_B = np.array([0.0, 0.2, 0.0, 0.0, 0.8])  # [-], solvent-rich feed
MASS_1 = 2.0  # [kg], batch charge of M01
MASS_2 = 3.0  # [kg], unequal batch charge of M02
TEMP_1 = 300.0  # [K]
TEMP_2 = 320.0  # [K]
CYCLE_TIME = 10.0  # [s], arbitrary BatchToFlowConnector discharge time
FLOW_1 = 1.5  # [kg/s], issue #444 Mixer M01 feed
FLOW_2 = 0.25  # [kg/s], issue #444 Mixer M02 feed
RUNTIME = 10.0  # [s], issue #444 collector run
TANK_VOLUME = 1e-3  # [m**3], semibatch reactor vessel; never solved
EXTRACTOR_STAGES = 2  # [-], dynamic extractor stages; never solved
PARTITION = np.ones(5)  # [-], constant distribution coefficients; unused
# [-], float64 roundoff of the Mixer's algebraic sums of a few kilograms
RTOL = 1e-12


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


def _batch_source(path, mass, fraction, temp):
    """Build a batch liquid Mixer charged with one liquid phase.

    Parameters
    ----------
    path : str
        Database path.
    mass : float
        Charged liquid mass [kg].
    fraction : numpy.ndarray
        Mass fractions [-], shape (5,).
    temp : float
        Liquid temperature [K].

    Returns
    -------
    Mixer
        Unsolved batch Mixer.
    """
    mixer = Mixer()
    mixer.Inlets = LiquidPhase(path, temp=temp, mass=mass, mass_frac=fraction)
    return mixer


def _assign_sources(sim, path):
    """Assign the M01 and M02 batch sources to an executor.

    Parameters
    ----------
    sim : SimulationExec
        Executor whose graph contains ``M01`` and ``M02``.
    path : str
        Database path.
    """
    sim.M01 = _batch_source(path, MASS_1, FEED_A, TEMP_1)
    sim.M02 = _batch_source(path, MASS_2, FEED_B, TEMP_2)


@pytest.mark.unit
@pytest.mark.parametrize('graph', [
    {'M01': ['B01'], 'M02': ['B01'], 'B01': []},
    {'M01': ['B01'], 'M02': ['B01']},
    {'M01': np.array(['B01']), 'M02': np.array(['B01']), 'B01': np.array([])}],
    ids=['declared', 'successor_only', 'array'])
def test_non_mixer_fan_in_is_rejected_before_any_unit_is_solved(path, graph):
    """Name the connector, its class and both sources; solve nothing."""
    sim = SimulationExec(path, graph)
    _assign_sources(sim, path)
    sim.B01 = BatchToFlowConnector(cycle_time=CYCLE_TIME)

    with pytest.raises(PharmaPyValueError,
                       match=r"'B01' \(BatchToFlowConnector\) receives "
                             r"\['M01', 'M02'\]\. Only a Mixer combines "
                             r"several inlets.*Connect the predecessors to "
                             r"a Mixer and the Mixer to the unit\."):
        sim.SolveFlowsheet(verbose=False)

    assert sim.B01.Phases is None
    assert not hasattr(sim.B01, 'Outlet')
    assert not hasattr(sim.M01, 'result')
    assert not hasattr(sim.M02, 'result')
    assert sim.uos_instances == {}
    assert not hasattr(sim, 'connections')


@pytest.mark.unit
def test_non_mixer_fan_in_is_rejected_irrespective_of_pick_units(path):
    """Reject already solved, unpicked sources that would still transfer."""
    sim = SimulationExec(path, {'M01': ['B01'], 'M02': ['B01'], 'B01': []})
    _assign_sources(sim, path)
    sim.M01.solve_unit()  # already solved units transfer when not picked
    sim.M02.solve_unit()
    sim.B01 = BatchToFlowConnector(cycle_time=CYCLE_TIME)

    with pytest.raises(PharmaPyValueError,
                       match=r"'B01' \(BatchToFlowConnector\) receives "
                             r"\['M01', 'M02'\]"):
        sim.SolveFlowsheet(pick_units=['B01'], verbose=False)

    assert sim.B01.Phases is None
    assert not hasattr(sim.B01, 'Outlet')
    assert not hasattr(sim, 'connections')


@pytest.mark.unit
def test_every_non_mixer_fan_in_is_named(path):
    """List every offending destination in order of first appearance."""
    graph = {'M03': ['B02'], 'M01': ['B01'], 'M02': ['B01'], 'M04': ['B02'],
             'B01': [], 'B02': []}
    sim = SimulationExec(path, graph)
    _assign_sources(sim, path)
    sim.M03 = _batch_source(path, MASS_1, FEED_B, TEMP_1)
    sim.M04 = _batch_source(path, MASS_2, FEED_A, TEMP_2)
    sim.B01 = BatchToFlowConnector(cycle_time=CYCLE_TIME)
    sim.B02 = BatchToFlowConnector(cycle_time=CYCLE_TIME)

    with pytest.raises(PharmaPyValueError,
                       match=r"several units: 'B02' \(BatchToFlowConnector\) "
                             r"receives \['M03', 'M04'\]; 'B01' "
                             r"\(BatchToFlowConnector\) receives "
                             r"\['M01', 'M02'\]\. "):
        sim.SolveFlowsheet(verbose=False)

    assert sim.B01.Phases is None
    assert sim.B02.Phases is None


@pytest.mark.unit
def test_mixer_fan_in_still_mixes_every_predecessor(path):
    """Keep Mixer fan-in: the Mixer holds the sum of both charges."""
    sim = SimulationExec(path, {'M01': ['M03'], 'M02': ['M03'], 'M03': []})
    _assign_sources(sim, path)
    sim.M03 = Mixer()

    sim.SolveFlowsheet(verbose=False)

    total_mass = MASS_1 + MASS_2  # [kg], 2 + 3 kg
    species_mass = MASS_1 * FEED_A + MASS_2 * FEED_B  # [kg]
    assert [inlet.mass for inlet in sim.M03.Inlets] == pytest.approx(
        [MASS_1, MASS_2], rel=RTOL)
    assert sim.M03.Outlet.mass == pytest.approx(total_mass, rel=RTOL)
    np.testing.assert_allclose(sim.M03.Outlet.mass * sim.M03.Outlet.mass_frac,
                               species_mass, rtol=RTOL)


@pytest.mark.assimulo
@pytest.mark.integration
def test_issue_reproduction_two_mixers_to_collector_is_rejected(path):
    """Reject the issue #444 graph instead of collecting only M02's flow."""
    pytest.importorskip('assimulo')
    from PharmaPy.Containers import DynamicCollector

    sources = []
    for flow in (FLOW_1, FLOW_2):
        mixer = Mixer()
        mixer.Inlets = [LiquidStream(path, temp=TEMP_1, mass_flow=flow,
                                     mass_frac=FEED_A)]
        sources.append(mixer)
    sim = SimulationExec(path, {'M01': ['C01'], 'M02': ['C01'], 'C01': []})
    sim.M01, sim.M02 = sources
    sim.C01 = DynamicCollector()

    with pytest.raises(PharmaPyValueError,
                       match=r"'C01' \(DynamicCollector\) receives "
                             r"\['M01', 'M02'\]"):
        sim.SolveFlowsheet(kwargs_run={'C01': {'runtime': RUNTIME,
                                               'verbose': False}},
                           verbose=False)

    assert sim.C01.Inlet is None
    assert not hasattr(sim.C01, 'result')
    assert not any(hasattr(mixer, 'result') for mixer in sources)
    assert sim.uos_instances == {}


def _stream_source(path, flow, fraction, temp):
    """Build a continuous liquid Mixer of one raw stream.

    Parameters
    ----------
    path : str
        Database path.
    flow : float
        Mass flow [kg/s].
    fraction : numpy.ndarray
        Mass fractions [-], shape (5,).
    temp : float
        Stream temperature [K].

    Returns
    -------
    Mixer
        Unsolved continuous Mixer.
    """
    mixer = Mixer()
    mixer.Inlets = LiquidStream(path, temp=temp, mass_flow=flow,
                                mass_frac=fraction)
    return mixer


@pytest.mark.unit
def test_semibatch_fed_a_holdup_and_a_stream_assigns_one_directly(path):
    """Advise a direct assignment, since a Mixer cannot mix both bases."""
    sim = SimulationExec(path, {'M01': ['R01'], 'S01': ['R01'], 'R01': []})
    sim.M01 = _batch_source(path, MASS_1, FEED_A, TEMP_1)
    sim.S01 = _stream_source(path, FLOW_1, FEED_B, TEMP_2)
    sim.R01 = SemibatchReactor(vol_tank=TANK_VOLUME)
    message = (
        "Flowsheet units that are not a Mixer are fed by several units: "
        "'R01' (SemibatchReactor) receives ['M01', 'S01']. Only a Mixer "
        "combines several inlets into one balanced mixture; any other unit "
        "would keep only one transferred stream, or stack them unmixed. "
        "'R01' (SemibatchReactor) takes one connected input: a stream "
        "[kg/s] as its feed (Inlet) or a holdup [kg] as its initial charge "
        "(Phases), and a Mixer cannot mix a holdup with a stream. Connect "
        "one predecessor and assign the other input directly on the unit: "
        "the charge through Phases or the feed through Inlet.")
    with pytest.raises(PharmaPyValueError,
                       match='^' + re.escape(message) + '$'):
        sim.SolveFlowsheet(verbose=False)

    assert sim.R01.Inlet is None
    assert not hasattr(sim, 'connections')


@pytest.mark.unit
def test_mixable_and_direct_remedies_are_both_given(path):
    """Name the Mixer remedy's units beside a direct-assignment remedy."""
    graph = {'M01': ['R01'], 'S01': ['R01'], 'M02': ['B01'], 'M03': ['B01'],
             'R01': [], 'B01': []}
    sim = SimulationExec(path, graph)
    sim.M01 = _batch_source(path, MASS_1, FEED_A, TEMP_1)
    sim.S01 = _stream_source(path, FLOW_1, FEED_B, TEMP_2)
    sim.M02 = _batch_source(path, MASS_2, FEED_B, TEMP_2)
    sim.M03 = _batch_source(path, MASS_1, FEED_B, TEMP_1)
    sim.R01 = SemibatchReactor(vol_tank=TANK_VOLUME)
    sim.B01 = BatchToFlowConnector(cycle_time=CYCLE_TIME)
    message = (r"unmixed\. Connect the predecessors of \['B01'\] to a Mixer "
               r"and the Mixer to the unit\. 'R01' \(SemibatchReactor\) "
               r"takes one connected input: ")
    with pytest.raises(PharmaPyValueError, match=message):
        sim.SolveFlowsheet(verbose=False)


@pytest.mark.unit
def test_dynamic_extractor_fed_two_streams_assigns_the_solvent(path):
    """Advise assigning the solvent, which a Mixer must not mix in."""
    from PharmaPy.DynamicExtraction import DynamicExtractor

    sim = SimulationExec(path, {'S01': ['E01'], 'S02': ['E01'], 'E01': []})
    sim.S01 = _stream_source(path, FLOW_1, FEED_A, TEMP_1)
    sim.S02 = _stream_source(path, FLOW_2, FEED_B, TEMP_1)
    sim.E01 = DynamicExtractor(num_stages=EXTRACTOR_STAGES,
                               k_fun=lambda *args, **kwargs: PARTITION,
                               gamma_model='ideal')
    message = (r"Connect the feed source and assign the solvent directly: "
               r"E01\.Inlet = \{'solvent': stream\}\.$")
    with pytest.raises(PharmaPyValueError, match=message):
        sim.SolveFlowsheet(verbose=False)


MIXER_ADVICE = (r"unmixed\. Connect the predecessors to a Mixer and the "
                r"Mixer to the unit\.$")
DIRECT_ADVICE = (r"unmixed\. '{name}' \({cls}\) takes one connected input: "
                 r"a stream \[kg/s\] as its feed \(Inlet\) or a holdup "
                 r"\[kg\] as its initial charge \(Phases\)")


@pytest.mark.unit
def test_flowsheet_fed_mixer_has_an_unknown_basis(path):
    """Give the Mixer remedy when a predecessor's basis is not known yet.

    A Mixer fed only through the flowsheet declares no basis before it
    solves; here it would carry a stream, so the Mixer remedy is right.
    """
    sim = SimulationExec(path, {'S01': ['X01'], 'X01': ['R01'],
                                'S02': ['R01'], 'R01': []})
    sim.S01 = _stream_source(path, FLOW_1, FEED_A, TEMP_1)
    sim.X01 = Mixer()
    sim.S02 = _stream_source(path, FLOW_2, FEED_B, TEMP_2)
    sim.R01 = SemibatchReactor(vol_tank=TANK_VOLUME)
    with pytest.raises(PharmaPyValueError,
                       match=r"'R01' \(SemibatchReactor\) receives "
                             r"\['X01', 'S02'\]\..*" + MIXER_ADVICE):
        sim.SolveFlowsheet(verbose=False)


@pytest.mark.unit
def test_unassigned_predecessor_still_reports_the_fan_in(path):
    """Raise the fan-in error, not an AttributeError, for a missing source."""
    sim = SimulationExec(path, {'M01': ['R01'], 'M02': ['R01'], 'R01': []})
    sim.M01 = _batch_source(path, MASS_1, FEED_A, TEMP_1)
    sim.R01 = SemibatchReactor(vol_tank=TANK_VOLUME)
    with pytest.raises(PharmaPyValueError,
                       match=r"'R01' \(SemibatchReactor\) receives "
                             r"\['M01', 'M02'\]\..*" + MIXER_ADVICE):
        sim.SolveFlowsheet(verbose=False)


@pytest.mark.unit
def test_evaporator_fed_a_holdup_and_a_stream_assigns_one_directly(path):
    """Advise a direct assignment whatever the unit's current mode.

    An Evaporator without an Inlet is a batch unit; it becomes semibatch
    once its feed is assigned, so the direct remedy applies.
    """
    sim = SimulationExec(path, {'M01': ['E01'], 'S01': ['E01'], 'E01': []})
    sim.M01 = _batch_source(path, MASS_1, FEED_A, TEMP_1)
    sim.S01 = _stream_source(path, FLOW_1, FEED_B, TEMP_2)
    sim.E01 = Evaporator(vol_drum=TANK_VOLUME)
    assert sim.E01.oper_mode == 'Batch'
    with pytest.raises(PharmaPyValueError,
                       match=DIRECT_ADVICE.format(name='E01',
                                                  cls='Evaporator')):
        sim.SolveFlowsheet(verbose=False)


@pytest.mark.unit
@pytest.mark.parametrize('destination', ['semibatch', 'collector'])
def test_same_basis_streams_get_the_mixer_remedy(path, destination):
    """Combine two streams in a Mixer for a semibatch unit or collector."""
    from PharmaPy.Containers import DynamicCollector

    sim = SimulationExec(path, {'S01': ['D01'], 'S02': ['D01'], 'D01': []})
    sim.S01 = _stream_source(path, FLOW_1, FEED_A, TEMP_1)
    sim.S02 = _stream_source(path, FLOW_2, FEED_B, TEMP_2)
    sim.D01 = (SemibatchReactor(vol_tank=TANK_VOLUME)
               if destination == 'semibatch' else DynamicCollector())
    with pytest.raises(PharmaPyValueError, match=MIXER_ADVICE):
        sim.SolveFlowsheet(verbose=False)


@pytest.mark.unit
def test_collector_fed_a_holdup_discharges_it_first(path):
    """Route a holdup into a collector through a BatchToFlowConnector."""
    from PharmaPy.Containers import DynamicCollector

    sim = SimulationExec(path, {'M01': ['C01'], 'S01': ['C01'], 'C01': []})
    sim.M01 = _batch_source(path, MASS_1, FEED_A, TEMP_1)
    sim.S01 = _stream_source(path, FLOW_1, FEED_B, TEMP_2)
    sim.C01 = DynamicCollector()
    message = (r"unmixed\. 'C01' \(DynamicCollector\) collects streams "
               r"\[kg/s\] only: discharge each holdup \[kg\] through a "
               r"BatchToFlowConnector, then combine the streams in a Mixer "
               r"and connect the Mixer to the collector\.$")
    with pytest.raises(PharmaPyValueError, match=message):
        sim.SolveFlowsheet(verbose=False)
