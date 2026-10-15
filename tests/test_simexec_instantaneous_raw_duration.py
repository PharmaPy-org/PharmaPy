"""Raw-material duration of instantaneous continuous units (issue #430).

A static continuous ``Mixer`` is an explicitly instantaneous unit
(``is_instantaneous``) whose result spans 0 s. Its static raw inlets are
charged, conservatively, only when it has exactly one successor in the
flowsheet graph and that successor verifiably uses the transferred outlet:
for the successor's fed duration, recursively through chains of
instantaneous units, so a consumer solved for 0 s gives a total of 0.
Otherwise the totals are NaN with a ``RuntimeWarning`` naming the unit and
the reason: a terminal unit, fan-out to several successors (each receives a
full copy, issue #437), or a successor that no longer holds the transfer,
such as a charged ``SemibatchReactor`` (issue #438). Units without the
flag, such as a ``DynamicCollector`` solved for 0 s, keep their own (zero)
fed duration for static and controlled feeds. Batch inlets are unchanged.
A raw inlet with a ``DynamicInput`` that controls no field feeds the static
stream and follows the same contract; one whose controls were added after
the solve, which the instantaneous Mixer never consumed, raises
``ValueError``.

Provenance is recorded per flowsheet edge across ``SolveFlowsheet`` calls
and checked at accounting time. Covered cases: repeated solves that leave
several outlet copies in a ``Mixer`` (issue #441); a source re-solved after
its transfer; a consumer whose solve after a new transfer failed; a
consumer solved before any transfer; partial solves (``pick_units``) of
unrelated units or of the consumer alone; consumers that keep several solve
segments (a continued ``CSTR``, and a continued ``ContinuousEvaporator``
that restarts its clock per segment, issue #278) or none (a ``CSTR`` after
``reset()``), which give NaN; and reset consumers (``reset()`` or
``reset_states=True``), charged for their own fresh run. Discarded transfer
copies are pruned from the record and freed, and a destination with an
untimed result (``BatchExtractor``) does not break re-solving.

Fixtures use the shipped five-species database and the issue reproduction:
1.0 kg/s (A, B, C, D, solvent = 0.1, 0.1, 0.1, 0.1, 0.6) at 300 K and
0.5 kg/s (0, 0.2, 0, 0, 0.8) at 320 K. Expected totals are flow times the
consumer run time; molar totals use the molecular weights read from the
JSON database. Reactor cases use the four-species database and the
synthetic ``configured`` CSTR or ``SemibatchReactor`` of
tests/test_reactor_controls_events_continuation.py, fed by a Mixer of
1.0 and 0.5 kg/s. The evaporator case uses ``make_unit`` and the UNIQUAC
binary of tests/test_evaporator_consistency.py, a ContinuousEvaporator fed
through a one-inlet Mixer. Unit tests need no ODE backend; tests marked
``assimulo`` integrate the consumers with CVode.
"""

import gc
import json
import warnings
import weakref

import numpy as np
import pytest

from PharmaPy.Containers import DynamicCollector, Mixer
from PharmaPy.Phases import LiquidPhase
from PharmaPy.ProcessControl import DynamicInput
from PharmaPy.SimExec import SimulationExec
from PharmaPy.Streams import LiquidStream

COMPOSITION_A = np.array([0.1, 0.1, 0.1, 0.1, 0.6])  # [-], issue #430
COMPOSITION_B = np.array([0.0, 0.2, 0.0, 0.0, 0.8])  # [-], issue #430
COMPOSITION_C = np.array([0.3, 0.0, 0.2, 0.0, 0.5])  # [-], third feed
FLOW_A = 1.0  # [kg/s], issue #430
FLOW_B = 0.5  # [kg/s], issue #430
FLOW_C = 0.25  # [kg/s], third feed of the chained case
TEMP_A = 300.0  # [K]
TEMP_B = 320.0  # [K]
RUNTIME = 10.0  # [s], issue #430 consumer run
SHORT_RUNTIME = 4.0  # [s], shorter second consumer
KG_TO_G = 1000.0  # [g/kg], exact SI conversion
# [-], float64 roundoff of a product of a flow and a time; the solver does
# not enter these totals, only the consumer's recorded time span.
RTOL = 1e-12
UNDEFINED_WARNING = (r"^Raw inlets of instantaneous unit '{unit}' have no "
                     r"defined fed duration: {reason}; their totals are "
                     r"reported as NaN\.$")
TERMINAL_REASON = (r"no downstream unit consumes its outlet "
                   r"\(terminal unit\)")
TERMINAL_WARNING = UNDEFINED_WARNING.replace('{reason}', TERMINAL_REASON)
SEVERAL_SOLVES = (r"downstream unit '{unit}' keeps 2 solve segments "
                  r"\(profiles_runs\), so the share consumed from this "
                  r"transfer is not defined \(issues #278, #443\)")
# [s], run time of each reactor or evaporator solve in the reviewer
# reproductions (issue #438, continued, reset and emptied consumers)
REACTOR_RUNTIME = 1.0
REPEATED_SOLVES = 3  # [-], more than two so pruning is not coincidental


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


@pytest.fixture
def molecular_weights(path):
    """Read molecular weights from the JSON database.

    Parameters
    ----------
    path : str
        Database path.

    Returns
    -------
    numpy.ndarray
        Molecular weights [g/mol], shape (5,), in database order.
    """
    with open(path) as handle:
        data = json.load(handle)
    return np.array([data[name]['mw'] for name in data])  # [g/mol]


def _stream(path, composition, flow, temp=TEMP_A):
    """Build a raw liquid stream.

    Parameters
    ----------
    path : str
        Database path.
    composition : numpy.ndarray
        Mass fractions [-], shape (5,).
    flow : float
        Mass flow [kg/s].
    temp : float, optional
        Temperature [K].

    Returns
    -------
    LiquidStream
        Raw stream.
    """
    return LiquidStream(path, temp=temp, mass_flow=flow, mass_frac=composition)


def _mixer(path, dynamic=False):
    """Build the issue #430 continuous Mixer of feeds A and B.

    Parameters
    ----------
    path : str
        Database path.
    dynamic : bool, optional
        Attach a ``DynamicInput`` without controls to feed B.

    Returns
    -------
    Mixer
        Unsolved continuous Mixer.
    """
    feed_b = _stream(path, COMPOSITION_B, FLOW_B, TEMP_B)
    if dynamic:
        feed_b.DynamicInlet = DynamicInput()
    mixer = Mixer()
    mixer.Inlets = [_stream(path, COMPOSITION_A, FLOW_A, TEMP_A), feed_b]
    return mixer


def _raw_amounts(raw, basis):
    """Return the raw table's (unit, inlet) keys and totals.

    Parameters
    ----------
    raw : pandas.DataFrame
        ``GetRawMaterials`` table with totals.
    basis : {'mass', 'mole'}
        Reporting basis.

    Returns
    -------
    tuple
        List of (unit, inlet) keys and totals [kg] or [mol] in row order.
    """
    column = 'mass' if basis == 'mass' else 'moles'
    return [key[:2] for key in raw.index], raw[column].to_numpy()


def _expected(flow, composition, duration, basis, molecular_weights):
    """Return a raw total on the requested basis.

    Parameters
    ----------
    flow : float
        Mass flow [kg/s].
    composition : numpy.ndarray
        Mass fractions [-].
    duration : float
        Fed duration [s].
    basis : {'mass', 'mole'}
        Reporting basis.
    molecular_weights : numpy.ndarray
        Molecular weights [g/mol].

    Returns
    -------
    float
        Total [kg] or [mol].
    """
    mass = flow * duration  # [kg]
    if basis == 'mass':
        return mass
    return mass * KG_TO_G * np.dot(composition, 1 / molecular_weights)  # [mol]


@pytest.mark.unit
@pytest.mark.parametrize('dynamic', [False, True], ids=['static', 'dynamic'])
@pytest.mark.parametrize('basis', ['mass', 'mole'])
def test_terminal_continuous_mixer_reports_nan(path, basis, dynamic):
    """Report NaN totals, not zero, for a Mixer no unit consumes."""
    sim = SimulationExec(path, {'M01': []})
    sim.M01 = _mixer(path, dynamic)
    sim.SolveFlowsheet(verbose=False)

    with pytest.warns(RuntimeWarning,
                      match=TERMINAL_WARNING.format(unit='M01')):
        raw = sim.GetRawMaterials(basis=basis)

    keys, totals = _raw_amounts(raw, basis)
    assert keys == [('M01', 'Inlet_0'), ('M01', 'Inlet_1')]
    assert np.isnan(totals).all()
    assert np.isnan(raw.to_numpy(dtype=float)).all()  # species columns too


@pytest.mark.unit
def test_controls_added_after_the_solve_are_rejected(path):
    """Refuse to account controls the instantaneous Mixer never evaluated."""
    sim = SimulationExec(path, {'M01': []})
    sim.M01 = _mixer(path, dynamic=True)
    sim.SolveFlowsheet(verbose=False)
    feed_b = sim.M01.Inlets[1]
    feed_b.DynamicInlet.add_variable('mass_flow', lambda time: 2 * FLOW_B)

    # Feed A (static) is reported first, with the terminal-unit warning.
    with pytest.warns(RuntimeWarning,
                      match=TERMINAL_WARNING.format(unit='M01')), \
            pytest.raises(ValueError,
                          match=r"of instantaneous Mixer has a DynamicInlet "
                                r"controlling \['mass_flow'\], which the "
                                r"unit evaluated at a single time"):
        sim.GetRawMaterials()


@pytest.mark.unit
def test_batch_mixer_totals_are_unchanged(path):
    """Keep batch inventories as raw totals; no duration is involved."""
    masses = (2.0, 3.0)  # [kg], unequal batch charges
    sim = SimulationExec(path, {'M01': []})
    sim.M01 = Mixer()
    sim.M01.Inlets = [LiquidPhase(path, temp=TEMP_A, mass=masses[0],
                                  mass_frac=COMPOSITION_A),
                      LiquidPhase(path, temp=TEMP_B, mass=masses[1],
                                  mass_frac=COMPOSITION_B)]
    sim.SolveFlowsheet(verbose=False)

    _, totals = _raw_amounts(sim.GetRawMaterials(), 'mass')

    np.testing.assert_allclose(totals, masses, rtol=RTOL)


@pytest.mark.assimulo
@pytest.mark.integration
@pytest.mark.parametrize('dynamic', [False, True], ids=['static', 'dynamic'])
@pytest.mark.parametrize('basis', ['mass', 'mole'])
def test_mixer_feed_is_charged_for_the_consumer_run(path, molecular_weights,
                                                    basis, dynamic):
    """Charge the issue #430 Mixer feeds for the collector's 10 s run."""
    pytest.importorskip('assimulo')
    sim = SimulationExec(path, {'M01': ['C01'], 'C01': []})
    sim.M01 = _mixer(path, dynamic)
    sim.C01 = DynamicCollector()
    sim.SolveFlowsheet(kwargs_run={'C01': {'runtime': RUNTIME,
                                           'verbose': False}}, verbose=False)

    keys, totals = _raw_amounts(sim.GetRawMaterials(basis=basis), basis)

    assert sim.time_processing == pytest.approx({'M01': 0.0, 'C01': RUNTIME},
                                                rel=RTOL)
    assert keys == [('M01', 'Inlet_0'), ('M01', 'Inlet_1')]
    expected = [_expected(FLOW_A, COMPOSITION_A, RUNTIME, basis,
                          molecular_weights),
                _expected(FLOW_B, COMPOSITION_B, RUNTIME, basis,
                          molecular_weights)]  # 10 kg and 5 kg on mass basis
    np.testing.assert_allclose(totals, expected, rtol=RTOL)


@pytest.mark.assimulo
@pytest.mark.integration
def test_chained_mixers_follow_to_the_consumer(path):
    """Charge both Mixers of a chain for the collector at its end."""
    pytest.importorskip('assimulo')
    sim = SimulationExec(path, {'M01': ['M02'], 'M02': ['C01'], 'C01': []})
    sim.M01 = _mixer(path)
    sim.M02 = Mixer()
    sim.M02.Inlets = [_stream(path, COMPOSITION_C, FLOW_C)]
    sim.C01 = DynamicCollector()
    sim.SolveFlowsheet(kwargs_run={'C01': {'runtime': RUNTIME,
                                           'verbose': False}}, verbose=False)

    keys, totals = _raw_amounts(sim.GetRawMaterials(), 'mass')

    assert keys == [('M01', 'Inlet_0'), ('M01', 'Inlet_1'), ('M02', 'Inlet_0')]
    np.testing.assert_allclose(
        totals, [FLOW_A * RUNTIME, FLOW_B * RUNTIME, FLOW_C * RUNTIME],
        rtol=RTOL)


@pytest.mark.assimulo
@pytest.mark.integration
def test_fan_out_reports_nan(path):
    """Report NaN, not a duration, when two units each copy the outlet.

    Provisional: each successor receives a full copy of the outlet (issue
    #437), so no total is defensible. Once #437 splits fan-out, expect a
    per-branch charge instead.
    """
    pytest.importorskip('assimulo')
    sim = SimulationExec(path, {'M01': ['C01', 'C02'], 'C01': [], 'C02': []})
    sim.M01 = _mixer(path)
    sim.C01 = DynamicCollector()
    sim.C02 = DynamicCollector()
    sim.SolveFlowsheet(kwargs_run={
        'C01': {'runtime': SHORT_RUNTIME, 'verbose': False},
        'C02': {'runtime': RUNTIME, 'verbose': False}}, verbose=False)

    reason = (r"its outlet feeds 2 units \['C01', 'C02'\], each receiving a "
              r"full copy of it \(fan-out, issue #437\)")
    with pytest.warns(RuntimeWarning, match=UNDEFINED_WARNING.format(
            unit='M01', reason=reason)):
        _, totals = _raw_amounts(sim.GetRawMaterials(), 'mass')

    assert np.isnan(totals).all()


@pytest.mark.assimulo
@pytest.mark.integration
def test_successor_ignoring_the_transfer_reports_nan(data_path):
    """Report NaN when a charged semibatch reactor keeps its own inlet.

    Provisional: the fixture relies on issue #438 (a charged semibatch
    destination ignores the transferred stream). Once #438 is fixed,
    replace it with another successor that does not consume the outlet.
    """
    pytest.importorskip('assimulo')
    from PharmaPy.Reactors import SemibatchReactor
    from test_reactor_controls_events_continuation import configured
    thermo = str(data_path['integration'] / 'pfr_test_pure_comp.json')
    sim = SimulationExec(thermo, {'M01': ['R01'], 'R01': []})
    sim.M01 = Mixer()
    sim.M01.Inlets = [
        LiquidStream(thermo, mass_flow=FLOW_A, mass_frac=[.5, .2, .3, 0.],
                     temp=TEMP_A),
        LiquidStream(thermo, mass_flow=FLOW_B, mass_frac=[.2, .5, .3, 0.],
                     temp=TEMP_A)]  # [kg/s], [-], [K]; four-species database
    sim.R01 = configured(SemibatchReactor)
    own_inlet = sim.R01.Inlet
    sim.SolveFlowsheet(kwargs_run={'R01': {'runtime': REACTOR_RUNTIME,
                                           'verbose': False}}, verbose=False)
    assert sim.R01.Inlet is own_inlet  # the transfer was not installed

    reason = (r"downstream unit 'R01' no longer holds the transferred outlet "
              r"\(it ignored the transfer, see issue #438, or another "
              r"transfer replaced it, see issue #444\)")
    with pytest.warns(RuntimeWarning, match=UNDEFINED_WARNING.format(
            unit='M01', reason=reason)):
        raw = sim.GetRawMaterials(include_holdups=False)

    keys, totals = _raw_amounts(raw, 'mass')
    assert keys == [('M01', 'Inlet_0'), ('M01', 'Inlet_1'), ('R01', 'Inlet_0')]
    assert np.isnan(totals[:2]).all()
    # The reactor's own raw inlet is charged for its own run.
    assert totals[2] == pytest.approx(own_inlet.mass_flow * REACTOR_RUNTIME,
                                      rel=RTOL)


@pytest.mark.assimulo
@pytest.mark.integration
def test_zero_runtime_consumer_charges_zero(path):
    """Charge 0 kg, not NaN, for a consumer solved for 0 s."""
    pytest.importorskip('assimulo')
    sim = SimulationExec(path, {'M01': ['C01'], 'C01': []})
    sim.M01 = _mixer(path)
    sim.C01 = DynamicCollector()
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        sim.SolveFlowsheet(kwargs_run={'C01': {'runtime': 0.,
                                               'verbose': False}},
                           verbose=False)
        _, totals = _raw_amounts(sim.GetRawMaterials(), 'mass')

    assert sim.time_processing == pytest.approx({'M01': 0.0, 'C01': 0.0},
                                                abs=0.0)
    np.testing.assert_array_equal(totals, [0., 0.])


@pytest.mark.assimulo
@pytest.mark.integration
@pytest.mark.parametrize('controlled', [False, True],
                         ids=['static', 'controlled'])
def test_zero_runtime_collector_feeds_stay_zero(path, controlled):
    """Keep a 0 s collector's own raw feeds at 0, as before issue #430."""
    pytest.importorskip('assimulo')
    feed = _stream(path, COMPOSITION_A, FLOW_A)
    if controlled:
        feed.DynamicInlet = DynamicInput()
        # [kg/s], a time-varying flow; over 0 s it delivers nothing.
        feed.DynamicInlet.add_variable('mass_flow',
                                       lambda time: FLOW_A * (1 + time))
    sim = SimulationExec(path, {'C01': []})
    sim.C01 = DynamicCollector()
    sim.C01.Inlet = feed
    sim.SolveFlowsheet(kwargs_run={'C01': {'runtime': 0., 'verbose': False}},
                       verbose=False)

    _, totals = _raw_amounts(sim.GetRawMaterials(), 'mass')

    np.testing.assert_array_equal(totals, [0.])


@pytest.mark.unit
def test_unsolved_successor_reports_nan(path):
    """Report NaN when the consumer was left out of a partial solve."""
    sim = SimulationExec(path, {'M01': ['C01'], 'C01': []})
    sim.M01 = _mixer(path)
    sim.C01 = DynamicCollector()
    sim.SolveFlowsheet(pick_units=['M01'], verbose=False)

    reason = r"downstream unit 'C01' was not solved"
    with pytest.warns(RuntimeWarning, match=UNDEFINED_WARNING.format(
            unit='M01', reason=reason)):
        _, totals = _raw_amounts(sim.GetRawMaterials(), 'mass')

    assert np.isnan(totals).all()


@pytest.mark.unit
def test_unit_outside_the_flowsheet_reports_nan(path):
    """Report NaN for a solved Mixer the executor never ran."""
    mixer = _mixer(path)
    mixer.solve_unit()
    sim = SimulationExec(path, {'M01': []})

    reason = r"it is not part of the solved flowsheet"
    with pytest.warns(RuntimeWarning, match=UNDEFINED_WARNING.format(
            unit='Mixer', reason=reason)):
        records = sim.get_raw_inlets(mixer)

    totals = [record['mass'] for inlet in records.values()
              for record in inlet.values()]  # [kg]
    assert len(totals) == 2 and np.isnan(totals).all()


@pytest.mark.unit
def test_profiled_mixer_is_charged_its_own_span(path):
    """Charge a Mixer whose connected profile spans time for that span."""
    grid = np.array([0., RUNTIME])  # [s], two-sample connected profile
    connected = _stream(path, COMPOSITION_A, FLOW_A)
    profile = {'mass_flow': np.full(len(grid), FLOW_A),  # [kg/s]
               'mass_frac': np.tile(COMPOSITION_A, (len(grid), 1)),  # [-]
               'temp': np.full(len(grid), TEMP_A)}  # [K]
    connected.y_upstream = profile
    connected.y_inlet = profile
    connected.time_upstream = grid
    sim = SimulationExec(path, {'M01': []})
    sim.M01 = Mixer()
    sim.M01.Inlets = [connected, _stream(path, COMPOSITION_B, FLOW_B, TEMP_B)]
    sim.SolveFlowsheet(verbose=False)
    assert sim.time_processing == pytest.approx({'M01': RUNTIME}, rel=RTOL)

    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        keys, totals = _raw_amounts(sim.GetRawMaterials(), 'mass')

    assert keys == [('M01', 'Inlet_1')]  # the connected inlet is not raw
    np.testing.assert_allclose(totals, [FLOW_B * RUNTIME], rtol=RTOL)


def _chain(path):
    """Build Mixer M01 feeding Mixer M02 feeding collector C01.

    Parameters
    ----------
    path : str
        Database path.

    Returns
    -------
    SimulationExec
        Unsolved flowsheet; M02 has a raw feed of FLOW_C [kg/s].
    """
    sim = SimulationExec(path, {'M01': ['M02'], 'M02': ['C01'], 'C01': []})
    sim.M01 = _mixer(path)
    sim.M02 = Mixer()
    sim.M02.Inlets = [_stream(path, COMPOSITION_C, FLOW_C)]
    sim.C01 = DynamicCollector()
    return sim


RUN_C01 = {'C01': {'runtime': RUNTIME, 'verbose': False}}


@pytest.mark.assimulo
@pytest.mark.integration
def test_repeated_solve_with_copied_outlet_reports_nan(path):
    """Report NaN when a re-solve leaves two outlet copies in a Mixer.

    Provisional: a connected Mixer appends another transferred copy on
    every solve (issue #441); once fixed, a repeated solve should charge
    the latest run.
    """
    pytest.importorskip('assimulo')
    sim = _chain(path)
    sim.SolveFlowsheet(kwargs_run=RUN_C01, verbose=False)
    sim.SolveFlowsheet(kwargs_run=RUN_C01, verbose=False)
    assert len(sim.M02.Inlets) == 3  # raw feed plus two copies of M01's outlet

    reason = (r"downstream unit 'M02' holds 2 copies of its outlet "
              r"\(issue #441\)")
    with pytest.warns(RuntimeWarning, match=UNDEFINED_WARNING.format(
            unit='M01', reason=reason)):
        keys, totals = _raw_amounts(sim.GetRawMaterials(), 'mass')

    assert keys == [('M01', 'Inlet_0'), ('M01', 'Inlet_1'), ('M02', 'Inlet_0')]
    assert np.isnan(totals[:2]).all()
    np.testing.assert_allclose(totals[2], FLOW_C * RUNTIME, rtol=RTOL)


@pytest.mark.assimulo
@pytest.mark.integration
def test_source_resolved_after_transfer_reports_nan(path):
    """Report NaN when the Mixer was re-solved after feeding the collector."""
    pytest.importorskip('assimulo')
    sim = SimulationExec(path, {'M01': ['C01'], 'C01': []})
    sim.M01 = _mixer(path)
    sim.C01 = DynamicCollector()
    sim.SolveFlowsheet(kwargs_run=RUN_C01, verbose=False)
    sim.M01.Inlets = _stream(path, COMPOSITION_C, FLOW_C)  # a new feed
    sim.M01.solve_unit()

    reason = (r"it was re-solved after its outlet was transferred to "
              r"downstream unit 'C01'")
    with pytest.warns(RuntimeWarning, match=UNDEFINED_WARNING.format(
            unit='M01', reason=reason)):
        _, totals = _raw_amounts(sim.GetRawMaterials(), 'mass')

    assert len(totals) == 3 and np.isnan(totals).all()


@pytest.mark.assimulo
@pytest.mark.integration
def test_partial_solve_of_an_unrelated_unit_keeps_totals(path):
    """Keep the verified totals after re-solving an unrelated unit."""
    pytest.importorskip('assimulo')
    sim = SimulationExec(path, {'M01': ['C01'], 'C01': [], 'B01': []})
    sim.M01 = _mixer(path)
    sim.C01 = DynamicCollector()
    sim.B01 = Mixer()
    sim.B01.Inlets = [LiquidPhase(path, mass=2.0, temp=TEMP_A,
                                  mass_frac=COMPOSITION_A)]  # [kg]
    sim.SolveFlowsheet(kwargs_run=RUN_C01, verbose=False)
    sim.SolveFlowsheet(pick_units=['B01'], verbose=False)

    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        raw = sim.GetRawMaterials()

    keys, totals = _raw_amounts(raw, 'mass')
    assert keys[:2] == [('M01', 'Inlet_0'), ('M01', 'Inlet_1')]
    np.testing.assert_allclose(totals[:2],
                               [FLOW_A * RUNTIME, FLOW_B * RUNTIME], rtol=RTOL)


@pytest.mark.assimulo
@pytest.mark.integration
def test_partial_solve_of_the_consumer_charges_its_new_run(path):
    """Charge the consumer's latest run after re-solving only it."""
    pytest.importorskip('assimulo')
    sim = SimulationExec(path, {'M01': ['C01'], 'C01': []})
    sim.M01 = _mixer(path)
    sim.C01 = DynamicCollector()
    sim.SolveFlowsheet(kwargs_run=RUN_C01, verbose=False)
    short_run = {'C01': {'runtime': SHORT_RUNTIME, 'verbose': False}}
    sim.SolveFlowsheet(kwargs_run=short_run, pick_units=['C01'], verbose=False)

    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        _, totals = _raw_amounts(sim.GetRawMaterials(), 'mass')

    span = sim.C01.result.time[-1] - sim.C01.result.time[0]  # [s]
    assert span == pytest.approx(SHORT_RUNTIME, rel=RTOL)
    np.testing.assert_allclose(
        totals, [FLOW_A * SHORT_RUNTIME, FLOW_B * SHORT_RUNTIME], rtol=RTOL)


@pytest.mark.assimulo
@pytest.mark.integration
def test_consumer_not_solved_after_the_transfer_reports_nan(path):
    """Report NaN when the consumer's solve after a new transfer failed."""
    pytest.importorskip('assimulo')
    sim = SimulationExec(path, {'M01': ['C01'], 'C01': []})
    sim.M01 = _mixer(path)
    sim.C01 = DynamicCollector()
    sim.SolveFlowsheet(kwargs_run=RUN_C01, verbose=False)
    previous_result = sim.C01.result
    # Without runtime the collector's solve fails after the new transfer.
    with pytest.raises(ValueError, match='requires .runtime.'):
        sim.SolveFlowsheet(verbose=False)
    assert sim.C01.result is previous_result

    reason = r"downstream unit 'C01' was not solved after the transfer"
    with pytest.warns(RuntimeWarning, match=UNDEFINED_WARNING.format(
            unit='M01', reason=reason)):
        _, totals = _raw_amounts(sim.GetRawMaterials(), 'mass')

    assert np.isnan(totals).all()


@pytest.mark.assimulo
@pytest.mark.integration
def test_consumer_solved_before_any_transfer_reports_nan(path):
    """Report NaN when the consumer ran on its own inlet before a transfer."""
    pytest.importorskip('assimulo')
    sim = SimulationExec(path, {'M01': ['C01'], 'C01': []})
    sim.M01 = _mixer(path)
    sim.C01 = DynamicCollector()
    sim.C01.Inlet = _stream(path, COMPOSITION_C, FLOW_C)
    sim.SolveFlowsheet(kwargs_run=RUN_C01, pick_units=['C01'], verbose=False)
    sim.SolveFlowsheet(pick_units=['M01'], verbose=False)

    reason = r"no transfer to downstream unit 'C01' is recorded"
    with pytest.warns(RuntimeWarning, match=UNDEFINED_WARNING.format(
            unit='M01', reason=reason)):
        raw = sim.GetRawMaterials()

    keys, totals = _raw_amounts(raw, 'mass')
    # Units report in the order they were first solved: C01, then M01.
    assert keys == [('C01', 'Inlet_0'), ('M01', 'Inlet_0'), ('M01', 'Inlet_1')]
    np.testing.assert_allclose(totals[0], FLOW_C * RUNTIME, rtol=RTOL)
    assert np.isnan(totals[1:]).all()


@pytest.mark.assimulo
@pytest.mark.integration
def test_repeated_solves_keep_only_held_transfers(path):
    """Prune discarded transfer copies so they can be freed."""
    pytest.importorskip('assimulo')
    sim = SimulationExec(path, {'M01': ['C01'], 'C01': []})
    sim.M01 = _mixer(path)
    sim.C01 = DynamicCollector()
    sim.SolveFlowsheet(kwargs_run=RUN_C01, verbose=False)
    first = weakref.ref(sim.C01.Inlet)
    for _ in range(REPEATED_SOLVES):
        sim.SolveFlowsheet(kwargs_run=RUN_C01, verbose=False)

    history = sim._transfer_history[('M01', 'C01')]
    assert len(history) == 1 and history[0] is sim.C01.Inlet
    gc.collect()
    assert first() is None


@pytest.mark.assimulo
@pytest.mark.integration
def test_mixer_history_keeps_every_held_copy(path):
    """Keep the copies a re-solved Mixer still holds (issue #441)."""
    pytest.importorskip('assimulo')
    sim = _chain(path)
    for _ in range(REPEATED_SOLVES):
        sim.SolveFlowsheet(kwargs_run=RUN_C01, verbose=False)

    history = sim._transfer_history[('M01', 'M02')]
    held = sim.M02.Inlets[1:]  # every copy of M01's outlet
    assert len(history) == len(held) == REPEATED_SOLVES
    assert all(any(copy is item for item in held) for copy in history)


def _cstr_flowsheet(data_path, **reactor_kwargs):
    """Build Mixer M01 feeding the shared synthetic CSTR R01.

    Parameters
    ----------
    data_path : dict
        Repository fixture directories.
    **reactor_kwargs
        Extra ``CSTR`` constructor options, such as ``reset_states``.

    Returns
    -------
    SimulationExec
        Unsolved flowsheet on the four-species database, feeds of FLOW_A
        and FLOW_B [kg/s] at TEMP_A [K].
    """
    from PharmaPy.Reactors import CSTR
    from test_reactor_controls_events_continuation import configured
    thermo = str(data_path['integration'] / 'pfr_test_pure_comp.json')
    sim = SimulationExec(thermo, {'M01': ['R01'], 'R01': []})
    sim.M01 = Mixer()
    sim.M01.Inlets = [
        LiquidStream(thermo, mass_flow=FLOW_A, mass_frac=[.5, .2, .3, 0.],
                     temp=TEMP_A),
        LiquidStream(thermo, mass_flow=FLOW_B, mass_frac=[.2, .5, .3, 0.],
                     temp=TEMP_A)]  # [kg/s], [-], [K]; four-species database
    sim.R01 = configured(CSTR, **reactor_kwargs)
    return sim


RESET_CASES = {
    'reset-states-longer': (True, False, 3.0),
    'reset-states-equal': (True, False, 1.0),
    'explicit-reset': (False, True, 2.0),
    }  # reset_states, call reset() between runs, second runtime [s]


@pytest.mark.assimulo
@pytest.mark.integration
@pytest.mark.parametrize('case', list(RESET_CASES))
def test_reset_consumer_is_charged_its_fresh_run(data_path, case):
    """Charge a reset CSTR for its new run, not as a continuation."""
    pytest.importorskip('assimulo')
    reset_states, explicit_reset, second_runtime = RESET_CASES[case]
    sim = _cstr_flowsheet(data_path, reset_states=reset_states)
    sim.SolveFlowsheet(kwargs_run={'R01': {'runtime': REACTOR_RUNTIME,
                                           'verbose': False}}, verbose=False)
    if explicit_reset:
        sim.R01.reset()
    sim.SolveFlowsheet(kwargs_run={'R01': {'runtime': second_runtime,
                                           'verbose': False}}, verbose=False)
    span = sim.R01.result.time[-1] - sim.R01.result.time[0]  # [s]
    assert span == pytest.approx(second_runtime, rel=RTOL)  # a fresh run

    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        raw = sim.GetRawMaterials(include_holdups=False)

    _, totals = _raw_amounts(raw, 'mass')
    np.testing.assert_allclose(
        totals, [FLOW_A * second_runtime, FLOW_B * second_runtime], rtol=RTOL)


@pytest.mark.unit
def test_untimed_destination_survives_repeated_solves(path):
    """Re-solve a batch Mixer feeding a BatchExtractor with an untimed result."""
    from PharmaPy.Extractors import BatchExtractor
    moles = 10.0  # [mol], batch charge
    fractions = [.5, .5, 0., 0., 0.]  # [-], mole fractions of A and B
    sim = SimulationExec(path, {'M01': ['E01'], 'E01': []})
    sim.M01 = Mixer()
    sim.M01.Inlets = LiquidPhase(path, moles=moles, mole_frac=fractions)
    # Constant partition coefficients [-], one per species.
    sim.E01 = BatchExtractor(
        k_fun=lambda x, y, t: np.array([2., .5, 1., 1., 1.]))

    for _ in range(REPEATED_SOLVES):
        sim.SolveFlowsheet(verbose=False)

    assert not hasattr(sim.E01.result, 'time')
    _, totals = _raw_amounts(sim.GetRawMaterials(basis='mole'), 'mole')
    np.testing.assert_allclose(totals, [moles], rtol=RTOL)


@pytest.mark.assimulo
@pytest.mark.integration
def test_continued_consumer_reports_nan(data_path):
    """Report NaN for a CSTR whose result spans the earlier solve too."""
    pytest.importorskip('assimulo')
    sim = _cstr_flowsheet(data_path)
    run = {'R01': {'runtime': REACTOR_RUNTIME, 'verbose': False}}
    sim.SolveFlowsheet(kwargs_run=run, verbose=False)
    sim.SolveFlowsheet(kwargs_run=run, verbose=False)
    assert len(sim.R01.profiles_runs) == 2  # the result spans both solves

    with pytest.warns(RuntimeWarning, match=UNDEFINED_WARNING.format(
            unit='M01', reason=SEVERAL_SOLVES.format(unit='R01'))):
        raw = sim.GetRawMaterials(include_holdups=False)

    keys, totals = _raw_amounts(raw, 'mass')
    assert keys == [('M01', 'Inlet_0'), ('M01', 'Inlet_1')]
    assert np.isnan(totals).all()


@pytest.mark.assimulo
@pytest.mark.integration
def test_continued_evaporator_reports_nan(tmp_path):
    """Report NaN for a ContinuousEvaporator continued by a second solve.

    The evaporator keeps its first segment and restarts its clock at 0 s
    for each segment (issue #278), so no share of its result can be
    attributed to the latest transfer.
    """
    pytest.importorskip('assimulo')
    from PharmaPy.Evaporators import ContinuousEvaporator
    from test_evaporator_consistency import make_unit, write_uniquac_binary
    thermo = write_uniquac_binary(tmp_path)
    unit = make_unit(thermo, ContinuousEvaporator)
    feed = unit.Inlet
    sim = SimulationExec(thermo, {'M01': ['E01'], 'E01': []})
    sim.M01 = Mixer()
    sim.M01.Inlets = [feed]
    sim.E01 = unit
    run = {'E01': {'runtime': REACTOR_RUNTIME, 'verbose': False}}
    sim.SolveFlowsheet(kwargs_run=run, verbose=False)
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        first = sim.GetRawMaterials(include_holdups=False, basis='mole')
    np.testing.assert_allclose(first['moles'],
                               [feed.mole_flow * REACTOR_RUNTIME], rtol=RTOL)

    sim.SolveFlowsheet(kwargs_run=run, verbose=False)
    assert len(sim.E01.profiles_runs) == 2

    with pytest.warns(RuntimeWarning, match=UNDEFINED_WARNING.format(
            unit='M01', reason=SEVERAL_SOLVES.format(unit='E01'))):
        raw = sim.GetRawMaterials(include_holdups=False, basis='mole')

    assert np.isnan(raw['moles'].to_numpy()).all()


@pytest.mark.assimulo
@pytest.mark.integration
def test_restarted_collector_clock_is_charged_its_run(path):
    """Charge a collector run whose clock was restarted, one solve.

    ``DynamicCollector.solve_unit`` integrates from its documented
    ``elapsed_time`` [s]; setting it back to 0 restarts the run. The
    collector keeps no ``profiles_runs`` and publishes only its latest
    solve, so its own span is the duration.
    """
    pytest.importorskip('assimulo')
    sim = SimulationExec(path, {'M01': ['C01'], 'C01': []})
    sim.M01 = _mixer(path)
    sim.C01 = DynamicCollector()
    sim.SolveFlowsheet(kwargs_run=RUN_C01, verbose=False)
    sim.C01.elapsed_time = 0.0  # [s], restart the collector clock
    sim.SolveFlowsheet(kwargs_run=RUN_C01, verbose=False)
    assert sim.C01.result.time[0] == 0.0

    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        _, totals = _raw_amounts(sim.GetRawMaterials(), 'mass')

    np.testing.assert_allclose(totals, [FLOW_A * RUNTIME, FLOW_B * RUNTIME],
                               rtol=RTOL)


@pytest.mark.assimulo
@pytest.mark.integration
def test_consumer_without_solve_segments_reports_nan(data_path):
    """Report NaN for a CSTR reset after its solve, keeping no segment."""
    pytest.importorskip('assimulo')
    sim = _cstr_flowsheet(data_path)
    sim.SolveFlowsheet(kwargs_run={'R01': {'runtime': REACTOR_RUNTIME,
                                           'verbose': False}}, verbose=False)
    sim.R01.reset()
    assert sim.R01.profiles_runs == []

    reason = (r"downstream unit 'R01' keeps no record of the solve behind "
              r"its result")
    with pytest.warns(RuntimeWarning, match=UNDEFINED_WARNING.format(
            unit='M01', reason=reason)):
        raw = sim.GetRawMaterials(include_holdups=False)

    _, totals = _raw_amounts(raw, 'mass')
    assert len(totals) == 2 and np.isnan(totals).all()
