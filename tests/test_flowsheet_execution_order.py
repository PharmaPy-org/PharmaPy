"""Deterministic flowsheet execution order and rendered topology (issue #309).

``topological_bfs`` ties are broken by a first-in, first-out queue seeded in
graph declaration order, so execution order, ``Mixer`` inlet order and the
``Mixer`` evaluation grid must not depend on ``PYTHONHASHSEED``. Hash-seed
cases run real ``SimulationExec`` flowsheets in isolated child processes with
fixed seeds; the Assimulo case solves two real ``ContinuousHoldup`` units with
CVode there. ``SimulationResult`` representations are checked for fan-in,
fan-out, disconnected chains, an isolated unit and a true chain. Recycles,
including ones only visible by counting successor-only units, must be rejected
by ``SimulationExec``; the diagram helper's edge-list fallback for such orders
is therefore exercised directly.

Fixtures use the shipped five-species database. Published source profiles use
a real ``Mixer`` and ``LiquidStream`` with unequal time grids [s] and flows
[kg/s] so that the selected grid and inlet order identify their source.

Related issues: https://github.com/PharmaPy-org/PharmaPy/issues/309,
https://github.com/PharmaPy-org/PharmaPy/issues/220
"""

import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import numpy as np
import pytest

from PharmaPy.Connections import topological_bfs
from PharmaPy.Containers import ContinuousHoldup, Mixer
from PharmaPy.Phases import LiquidPhase
from PharmaPy.Errors import PharmaPyNonImplementedError
from PharmaPy.Results import SimulationResult, _flowsheet_diagram_lines
from PharmaPy.SimExec import SimulationExec
from PharmaPy.Streams import LiquidStream


REPO_ROOT = Path(__file__).resolve().parents[1]
THERMO_PATH = str(REPO_ROOT / 'tests' / 'Flowsheet' / 'data'
                  / 'compound_database.json')

# Seeds 0-3 gave four different orders of the disconnected-chain graph with
# the former set-based traversal, and seeds 0 and 1 swapped the fan-in sources.
HASH_SEEDS = (0, 1, 2, 3)
CHILD_TIMEOUT = 300  # [s], generous bound for one import and flowsheet solve

PHASE_FRAC = [0.1, 0.1, 0.1, 0.1, 0.6]  # [-], five-species solvent-rich liquid
HOLDUP_MASS = 1.0  # [kg], arbitrary; 1 s and 0.5 s residence at HOLDUP_FLOWS
HOLDUP_TEMP = 350.0  # [K], arbitrary initial holdup temperature

# MANY_SOURCES and FAN_OUT declare units or successors out of alphabetical
# order, so sorting by name cannot reproduce the expected orders.
FAN_IN = {'R01': ['MIX'], 'R02': ['MIX'], 'MIX': ['HOLD'], 'HOLD': []}
MANY_SOURCES = {'S5': ['M'], 'S3': ['M'], 'S9': ['M'], 'S1': ['M'],
                'S7': ['M'], 'M': []}
FAN_OUT = {'A': ['C', 'B'], 'B': ['D'], 'C': ['D'], 'D': []}
DISCONNECTED = {'A': ['B'], 'B': [], 'C': ['D'], 'D': []}
# S2 is ready before P, so M receives S2 before P although P is declared first.
RELAY = {'S1': ['P'], 'P': ['M'], 'S2': ['M'], 'M': []}

# Expected FIFO orders written from the documented rule: sources in key order,
# then each unit when its last predecessor has been visited.
EXPECTED_ORDERS = {
    'fan_in': ['R01', 'R02', 'MIX', 'HOLD'],
    'many_sources': ['S5', 'S3', 'S9', 'S1', 'S7', 'M'],
    'fan_out': ['A', 'C', 'B', 'D'],
    'disconnected': ['A', 'C', 'B', 'D'],
    'relay': ['S1', 'S2', 'P', 'M'],
}
GRAPHS = {'fan_in': FAN_IN, 'many_sources': MANY_SOURCES, 'fan_out': FAN_OUT,
          'disconnected': DISCONNECTED, 'relay': RELAY}

SOURCE_TIMES = {
    'R01': [0.0, 5.0, 10.0],  # [s], first declared source grid
    'R02': [0.0, 4.0, 8.0, 12.0],  # [s], longer and finer second source grid
}
SOURCE_FLOWS = {'R01': 1.0, 'R02': 2.0}  # [kg/s], identify each inlet
SOURCE_TEMPS = {'R01': 300.0, 'R02': 350.0}  # [K]
# [-], float64 roundoff allowance for summing interpolated constant flows
FLOW_RTOL = 1e-12

HOLDUP_RUNTIMES = {'H01': 10.0, 'H02': 12.0}  # [s], give distinct CVode grids
HOLDUP_FLOWS = {'H01': 1.0, 'H02': 2.0}  # [kg/s]
HOLDUP_FEED_TEMPS = {'H01': 350.0, 'H02': 330.0}  # [K]


def _publish_source(thermo_path, times, mass_flow, temp):
    """Publish a constant liquid profile with a real Mixer and LiquidStream.

    Parameters
    ----------
    thermo_path : str
        Shipped five-species database filename.
    times : sequence of float
        Absolute profile times [s], shape (num_times,).
    mass_flow : float
        Constant outlet mass flow [kg/s].
    temp : float
        Constant outlet temperature [K].

    Returns
    -------
    Mixer
        Source holding the published profile in the Mixer result contract,
        so ``SolveFlowsheet`` transfers it without solving it again.
    """
    source = Mixer()
    source.Inlets = LiquidStream(thermo_path, mass_flow=mass_flow,
                                 mass_frac=PHASE_FRAC, temp=temp)
    source.Liquid_1 = source.Inlets[0]
    source.names_states_out = source.names_states_in
    num_times = len(times)
    source.retrieve_results(
        np.asarray(times, dtype=float),
        (np.full(num_times, mass_flow),  # [kg/s]
         np.tile(PHASE_FRAC, (num_times, 1)),  # [-]
         np.full(num_times, temp)))  # [K]
    return source


def _solve_fan_in(thermo_path):
    """Mix two published sources through a real flowsheet fan-in.

    Parameters
    ----------
    thermo_path : str
        Shipped five-species database filename.

    Returns
    -------
    SimulationExec
        Flowsheet ``{'R01': ['MIX'], 'R02': ['MIX'], 'MIX': []}`` after
        solving only the continuous ``Mixer``.
    """
    sim = SimulationExec(thermo_path,
                         flowsheet={'R01': ['MIX'], 'R02': ['MIX'], 'MIX': []})
    for name, times in SOURCE_TIMES.items():
        setattr(sim, name, _publish_source(
            thermo_path, times, SOURCE_FLOWS[name], SOURCE_TEMPS[name]))
    sim.MIX = Mixer()
    sim.SolveFlowsheet(pick_units=['MIX'], verbose=False)
    return sim


def _structure_flowsheet(flowsheet):
    """Build a flowsheet of real holdups for structure and summary checks.

    Parameters
    ----------
    flowsheet : dict or str
        Adjacency mapping or ``'A --> B'`` string accepted by SimulationExec.

    Returns
    -------
    SimulationExec
        Executor with an unsolved ``ContinuousHoldup`` per unit name.
    """
    sim = SimulationExec(THERMO_PATH, flowsheet=flowsheet)
    for name in sim.execution_names:
        unit = ContinuousHoldup()
        unit.Phases = LiquidPhase(THERMO_PATH, mass=HOLDUP_MASS,
                                  temp=HOLDUP_TEMP, mass_frac=PHASE_FRAC)
        setattr(sim, name, unit)
    return sim


def _diagram_lines(text):
    """Extract the flowsheet-structure lines of a SimulationResult repr.

    Parameters
    ----------
    text : str
        Output of ``repr(SimulationResult(sim))``.

    Returns
    -------
    list of str
        Lines after ``'Flowsheet structure:'`` up to the next blank line.
    """
    lines = text.splitlines()
    start = lines.index('Flowsheet structure:') + 1
    end = lines.index('', start)
    return lines[start:end]


def _run_child(script, seed, tmp_path, *args):
    """Run a script in a fresh interpreter with a fixed string-hash seed.

    Parameters
    ----------
    script : str
        Python source that prints one JSON document on its last line.
    seed : int
        ``PYTHONHASHSEED`` for the child interpreter.
    tmp_path : pathlib.Path
        Writable directory for the child's Matplotlib configuration.
    *args : str
        Command-line arguments passed to the script.

    Returns
    -------
    dict
        Parsed JSON document printed by the child.
    """
    environment = os.environ.copy()
    environment['PYTHONHASHSEED'] = str(seed)
    environment['MPLBACKEND'] = 'Agg'
    environment['MPLCONFIGDIR'] = str(tmp_path)
    completed = subprocess.run(
        [sys.executable, '-c', script, *args], cwd=REPO_ROOT,
        env=environment, capture_output=True, text=True,
        timeout=CHILD_TIMEOUT, check=False)
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout.strip().splitlines()[-1])
    assert Path(result['pharmapy_file']).resolve().is_relative_to(REPO_ROOT)
    return result


_STRUCTURE_CHILD = textwrap.dedent('''
    import json
    import sys

    import PharmaPy
    from PharmaPy.Results import SimulationResult
    import tests.test_flowsheet_execution_order as cases

    report = {'pharmapy_file': PharmaPy.__file__, 'graphs': {}}
    for label, graph in cases.GRAPHS.items():
        sim = cases._structure_flowsheet(graph)
        report['graphs'][label] = {
            'order': sim.execution_names,
            'residuals': sim.in_degree,
            'diagram': cases._diagram_lines(repr(SimulationResult(sim))),
        }
    sim = cases._solve_fan_in(cases.THERMO_PATH)
    report['mixer_grid'] = sim.MIX.result.time.tolist()
    report['inlet_grids'] = [inlet.time_upstream.tolist()
                             for inlet in sim.MIX.Inlets]
    print(json.dumps(report))
''')

_CVODE_CHILD = textwrap.dedent('''
    import json

    import numpy as np
    import PharmaPy
    from PharmaPy.Containers import ContinuousHoldup, Mixer
    from PharmaPy.Phases import LiquidPhase
    from PharmaPy.SimExec import SimulationExec
    from PharmaPy.Streams import LiquidStream
    import tests.test_flowsheet_execution_order as cases

    HOT_FRAC = [0.2, 0.1, 0.2, 0.1, 0.4]  # [-], solute-rich feed
    sim = SimulationExec(cases.THERMO_PATH,
                         flowsheet={'H01': ['MIX'], 'H02': ['MIX'], 'MIX': []})
    for name, flow in cases.HOLDUP_FLOWS.items():
        unit = ContinuousHoldup()
        unit.Phases = LiquidPhase(cases.THERMO_PATH, mass=cases.HOLDUP_MASS,
                                  temp=cases.HOLDUP_TEMP,
                                  mass_frac=cases.PHASE_FRAC)
        unit.Inlet = LiquidStream(cases.THERMO_PATH, mass_flow=flow,
                                  mass_frac=HOT_FRAC,
                                  temp=cases.HOLDUP_FEED_TEMPS[name])
        setattr(sim, name, unit)
    sim.MIX = Mixer()
    sim.SolveFlowsheet(
        kwargs_run={name: {'runtime': runtime, 'verbose': False}
                    for name, runtime in cases.HOLDUP_RUNTIMES.items()},
        verbose=False)
    print(json.dumps({
        'pharmapy_file': PharmaPy.__file__,
        'order': sim.execution_names,
        'source_grids': {name: getattr(sim, name).result.time.tolist()
                         for name in cases.HOLDUP_RUNTIMES},
        'inlet_grids': [inlet.time_upstream.tolist()
                        for inlet in sim.MIX.Inlets],
        'mixer_grid': sim.MIX.result.time.tolist(),
        'mixer_temp': sim.MIX.result.temp.tolist(),
    }))
''')


@pytest.mark.unit
@pytest.mark.parametrize('label', sorted(EXPECTED_ORDERS))
def test_topological_bfs_breaks_ties_in_declaration_fifo_order(label):
    in_degree, path = topological_bfs(GRAPHS[label])

    assert path == EXPECTED_ORDERS[label]
    assert set(in_degree) == set(EXPECTED_ORDERS[label])
    assert all(count == 0 for count in in_degree.values())


@pytest.mark.unit
def test_topological_bfs_keeps_residual_counters_downstream_of_cycle():
    """Residuals stay counters after a cycle, not the raw in-degree of 2."""
    in_degree, path = topological_bfs({'A': ['B'], 'B': ['C'], 'C': ['B']})

    assert path == ['A']
    assert in_degree == {'A': 0, 'B': 1, 'C': 1}


@pytest.mark.unit
def test_topological_bfs_rejects_unordered_successors():
    with pytest.raises(TypeError,
                       match="unit 'A' must be an ordered sequence"):
        topological_bfs({'A': {'B', 'C'}, 'B': [], 'C': []})


@pytest.mark.unit
def test_execution_order_and_diagram_are_independent_of_hash_seed(tmp_path):
    expected_diagrams = {
        'fan_in': ['R01 --> MIX', 'R02 --> MIX', 'MIX --> HOLD'],
        'many_sources': ['S5 --> M', 'S3 --> M', 'S9 --> M', 'S1 --> M',
                         'S7 --> M'],
        'fan_out': ['A --> C', 'A --> B', 'B --> D', 'C --> D'],
        'disconnected': ['A --> B', 'C --> D'],
        'relay': ['S1 --> P', 'P --> M', 'S2 --> M'],
    }
    for seed in HASH_SEEDS:
        report = _run_child(_STRUCTURE_CHILD, seed, tmp_path)

        for label, order in EXPECTED_ORDERS.items():
            observed = report['graphs'][label]
            assert observed['order'] == order, (seed, label)
            assert observed['residuals'] == dict.fromkeys(order, 0)
            assert observed['diagram'] == expected_diagrams[label]
        # The first declared source supplies the Mixer grid in every process.
        assert report['inlet_grids'] == [SOURCE_TIMES['R01'],
                                         SOURCE_TIMES['R02']], seed
        assert report['mixer_grid'] == SOURCE_TIMES['R01'], seed


@pytest.mark.unit
def test_fan_in_mixer_receives_sources_in_declaration_order():
    sim = _solve_fan_in(THERMO_PATH)

    inlet_grids = [inlet.time_upstream for inlet in sim.MIX.Inlets]  # [s]
    assert len(inlet_grids) == len(SOURCE_TIMES)
    np.testing.assert_array_equal(inlet_grids[0], SOURCE_TIMES['R01'])
    np.testing.assert_array_equal(inlet_grids[1], SOURCE_TIMES['R02'])
    np.testing.assert_array_equal(sim.MIX.result.time, SOURCE_TIMES['R01'])
    total_flow = SOURCE_FLOWS['R01'] + SOURCE_FLOWS['R02']  # [kg/s]
    np.testing.assert_allclose(sim.MIX.result.mass_flow, total_flow,
                               rtol=FLOW_RTOL)


@pytest.mark.unit
def test_relay_mixer_receives_ready_source_before_declared_relay():
    """M is fed by S2 before P because S2 becomes ready first (FIFO)."""
    relay_times = SOURCE_TIMES['R01']  # [s], reaches M through relay P
    direct_times = SOURCE_TIMES['R02']  # [s], feeds M directly
    sim = SimulationExec(THERMO_PATH, flowsheet=RELAY)
    sim.S1 = _publish_source(THERMO_PATH, relay_times, SOURCE_FLOWS['R01'],
                             SOURCE_TEMPS['R01'])
    sim.S2 = _publish_source(THERMO_PATH, direct_times, SOURCE_FLOWS['R02'],
                             SOURCE_TEMPS['R02'])
    sim.P = Mixer()
    sim.M = Mixer()
    sim.SolveFlowsheet(pick_units=['P', 'M'], verbose=False)

    assert sim.execution_names == EXPECTED_ORDERS['relay']
    assert len(sim.M.Inlets) == 2
    np.testing.assert_array_equal(sim.M.Inlets[0].time_upstream, direct_times)
    np.testing.assert_array_equal(sim.M.Inlets[1].time_upstream, relay_times)
    np.testing.assert_array_equal(sim.M.result.time, direct_times)


@pytest.mark.unit
@pytest.mark.parametrize('flowsheet, expected', [
    ({'R01': ['HOLD01'], 'HOLD01': ['CR01'], 'CR01': ['F01'], 'F01': []},
     ['R01 --> HOLD01 --> CR01 --> F01']),
    ('R01 --> HOLD01 --> CR01 --> F01', ['R01 --> HOLD01 --> CR01 --> F01']),
    # A true chain declared out of order still renders along the path.
    ({'F01': [], 'CR01': ['F01'], 'R01': ['HOLD01'], 'HOLD01': ['CR01']},
     ['R01 --> HOLD01 --> CR01 --> F01']),
    ({'R01': []}, ['R01']),
    (FAN_IN, ['R01 --> MIX', 'R02 --> MIX', 'MIX --> HOLD']),
    # Reordered declaration of the same fan-in reorders the edge lines.
    ({'R02': ['MIX'], 'MIX': ['HOLD'], 'R01': ['MIX'], 'HOLD': []},
     ['R02 --> MIX', 'MIX --> HOLD', 'R01 --> MIX']),
    (FAN_OUT, ['A --> C', 'A --> B', 'B --> D', 'C --> D']),
    # Successors in other ordered sequences render the same edges.
    ({'A': ('C', 'B'), 'B': ('D',), 'C': ('D',), 'D': ()},
     ['A --> C', 'A --> B', 'B --> D', 'C --> D']),
    ({'A': np.array(['C', 'B']), 'B': np.array(['D']),
      'C': np.array(['D']), 'D': np.array([], dtype=str)},
     ['A --> C', 'A --> B', 'B --> D', 'C --> D']),
    ({'A': ['B'], 'B': ['E'], 'C': ['D'], 'D': [], 'E': []},
     ['A --> B', 'B --> E', 'C --> D']),
    ({'A': ['B'], 'ISO': [], 'B': []}, ['A --> B', 'ISO']),
    ({'X': [], 'Y': []}, ['X', 'Y']),
], ids=['chain', 'chain_string', 'chain_reordered', 'single_unit', 'fan_in',
        'fan_in_reordered', 'fan_out', 'fan_out_tuple', 'fan_out_ndarray',
        'disconnected_chains', 'isolated_unit', 'isolated_units'])
def test_simulation_result_repr_renders_actual_edges(flowsheet, expected):
    sim = _structure_flowsheet(flowsheet)

    text = repr(SimulationResult(sim))

    assert _diagram_lines(text) == expected
    # The equation table still follows the diagram for every unit.
    for name in sim.execution_names:
        assert any(line.startswith(name + ' ') for line in text.splitlines())


@pytest.mark.unit
@pytest.mark.parametrize('flowsheet, unscheduled', [
    ({'A': ['B'], 'C': ['C'], 'E': []}, 'C'),
    ({'A': ['X'], 'B': ['Y'], 'E': [], 'C': ['D'], 'D': ['C']}, 'C, D'),
    ({'A': ['B'], 'B': ['C'], 'C': ['B', 'Z']}, 'B, C, Z'),
], ids=['self_loop', 'two_unit_recycle', 'downstream_successor_only'])
def test_simulation_exec_rejects_recycles_counting_every_unit(flowsheet,
                                                              unscheduled):
    """Reject recycles even when scheduled units match the number of keys."""
    with pytest.raises(PharmaPyNonImplementedError,
                       match=f'cannot be scheduled: {unscheduled}$'):
        SimulationExec(THERMO_PATH, flowsheet=flowsheet)


@pytest.mark.unit
@pytest.mark.parametrize('flowsheet, order', [
    ({'A': ['B']}, ['A', 'B']),
    ({'R01': ['MIX'], 'R02': ['MIX']}, ['R01', 'R02', 'MIX']),
], ids=['successor_only_chain', 'successor_only_mixer'])
def test_simulation_exec_accepts_successor_only_units(flowsheet, order):
    sim = SimulationExec(THERMO_PATH, flowsheet=flowsheet)

    assert sim.execution_names == order


@pytest.mark.unit
def test_successor_only_mixer_sink_solves_and_transfers_nothing():
    """Solve a Mixer that appears only as a successor (no adjacency entry)."""
    sim = SimulationExec(THERMO_PATH, flowsheet={'R01': ['MIX'], 'R02': ['MIX']})
    for name, times in SOURCE_TIMES.items():
        setattr(sim, name, _publish_source(
            THERMO_PATH, times, SOURCE_FLOWS[name], SOURCE_TEMPS[name]))
    sim.MIX = Mixer()

    sim.SolveFlowsheet(pick_units=['MIX'], verbose=False)

    assert len(sim.connections) == len(SOURCE_TIMES)
    np.testing.assert_array_equal(sim.MIX.result.time, SOURCE_TIMES['R01'])
    total_flow = SOURCE_FLOWS['R01'] + SOURCE_FLOWS['R02']  # [kg/s]
    np.testing.assert_allclose(sim.MIX.result.mass_flow, total_flow,
                               rtol=FLOW_RTOL)


@pytest.mark.unit
@pytest.mark.parametrize('graph, execution_names, expected', [
    # Execution order misses a unit (recycle C) and joins A, E and B.
    ({'A': ['B'], 'C': ['C'], 'E': []}, ['A', 'E', 'B'],
     ['A --> B', 'C --> C', 'E']),
    ({'A': ['X'], 'B': ['Y'], 'E': [], 'C': ['D'], 'D': ['C']},
     ['A', 'B', 'E', 'X', 'Y'],
     ['A --> X', 'B --> Y', 'E', 'C --> D', 'D --> C']),
    # Covers every unit with one edge fewer than units and unit degrees,
    # but B --> C is not an edge.
    ({'A': ['B'], 'C': ['C']}, ['A', 'B', 'C'], ['A --> B', 'C --> C']),
], ids=['self_loop', 'two_unit_recycle', 'unconnected_consecutive_pair'])
def test_diagram_helper_lists_edges_unless_order_is_the_graph_path(
        graph, execution_names, expected):
    """Call the private helper directly with orders SimulationExec rejects.

    ``SimulationExec`` now raises for recycles, so ``repr`` cannot reach an
    execution order that misses units or joins unconnected ones. The helper
    must still fall back to listing the actual edges in that case.
    """
    assert _flowsheet_diagram_lines(graph, execution_names) == expected


@pytest.mark.assimulo
@pytest.mark.integration
def test_cvode_fan_in_mixer_grid_is_independent_of_hash_seed(tmp_path):
    pytest.importorskip('assimulo')
    reports = [_run_child(_CVODE_CHILD, seed, tmp_path)
               for seed in HASH_SEEDS[:2]]

    for report in reports:
        source_grids = report['source_grids']  # [s]
        assert report['order'] == ['H01', 'H02', 'MIX']
        assert len(source_grids['H01']) != len(source_grids['H02'])
        assert report['inlet_grids'] == [source_grids['H01'],
                                         source_grids['H02']]
        assert report['mixer_grid'] == source_grids['H01']
    # Identical inputs and operation order must reproduce bit-for-bit.
    np.testing.assert_array_equal(reports[0]['mixer_temp'],
                                  reports[1]['mixer_temp'])
