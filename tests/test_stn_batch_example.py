"""Exercise the documented STN export with two real CVode batch solves."""

import json
from pathlib import Path
import subprocess
import sys

import pytest


pytestmark = [pytest.mark.assimulo, pytest.mark.integration]
pytest.importorskip("assimulo")


def test_solved_batch_example_exports_reference_stn(tmp_path):
    """Check fixed batch masses, material arcs and rounding in the CLI output."""
    checkout = Path(__file__).resolve().parents[1]
    output = tmp_path / "batch.json"
    result = subprocess.run(
        [sys.executable, "-m", "doc.online_docs.examples.stn_batch_example",
         str(output)],
        cwd=checkout, check=False, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    stn = json.loads(output.read_text(encoding="utf-8"))

    assert stn["I"] == ["R01", "R02"]
    assert stn["J"] == ["reactor-1", "reactor-2"]
    assert stn["K"] == ["feed", "intermediate", "product"]
    assert stn["delta"] == pytest.approx(20.0)  # [s], example grid width
    # 25 s needs two 20 s periods and 43 s needs three, so both fit in five.
    assert stn["lastT"] == 5  # [-]
    task_units = [["R01", "reactor-1"], ["R02", "reactor-2"]]
    assert [entry["index"] for entry in stn["tau_p"]] == task_units
    assert [entry["value"] for entry in stn["tau_p"]] == pytest.approx(
        [25.0, 43.0])  # [s], stage run durations requested by the example
    assert stn["tau"] == [
        {"index": task_units[0], "value": 2},
        {"index": task_units[1], "value": 3},
    ]  # [-], minimal durations on the 20 s grid
    assert stn["I_i_j_prod"] == [
        {"index": index, "value": 1} for index in task_units]

    inputs = [["R01", "feed"], ["R02", "intermediate"]]
    outputs = [["R01", "intermediate"], ["R02", "product"]]
    for direction, indices in (("minus", inputs), ("plus", outputs)):
        assert stn[f"I_i_k_{direction}"] == [
            {"index": index, "value": 1} for index in indices]
        ratios = stn[f"rho_{direction}"]
        assert [entry["index"] for entry in ratios] == indices
        assert [entry["value"] for entry in ratios] == pytest.approx([1.0, 1.0])
        # [-], one entire liquid batch on each side of the mass-balanced task.

    # [kg], independently from 2 L * (0.1*105.52 + 0.2*99.1741 + 5*72.107) g/L
    # with 1000 g/kg. The shipped molar masses satisfy MW(C) = MW(A) + MW(B).
    batch_kg = 0.78184364
    for capacity in ("beta_min", "beta_max"):
        assert [entry["index"] for entry in stn[capacity]] == task_units
        assert [entry["value"] for entry in stn[capacity]] == pytest.approx(
            [batch_kg, batch_kg])
    assert [entry["index"] for entry in stn["gamma"]] == stn["K"]
    assert [entry["value"] for entry in stn["gamma"]] == pytest.approx(
        [batch_kg, batch_kg, batch_kg])
    assert stn["S0"] == [{"index": "feed", "value": pytest.approx(batch_kg)}]
    assert stn["demand"] == [
        {"index": ["product", 5], "value": pytest.approx(batch_kg)}]
    assert stn["cost"] == []


@pytest.mark.parametrize(('run_duration', 'expected_periods'), [
    (2.1, 7),  # [s] and [periods], exactly the represented seventh endpoint
    (2.100001, 8),  # [s] and [periods], one microsecond beyond that endpoint
])
def test_decimal_grid_endpoint_and_unequal_reference_arcs(run_duration,
                                                          expected_periods):
    """Use grid endpoints and normalize separated masses on the input basis."""
    from doc.online_docs.examples.stn_batch_example import solve_reference_sequence
    from PharmaPy.Scheduling import BatchTask, export_batch_stn

    # [s], the second step just completes the example flowsheet. Only the
    # first is exported for this scheduling case.
    second_duration = 1.0
    simulation, feed_kg = solve_reference_sequence(run_duration, second_duration)  # [kg]
    period_width = 0.3  # [s], 7 * 0.3 equals the represented duration of 2.1 s
    # [kg], independently from 0.1 mol/L * 2 L * 105.52 g/mol * 1e-3 kg/g.
    feed_a = 0.021104
    feed_rest = feed_kg - feed_a  # [kg], remaining constituents in the premix
    # Assume a separate handling step splits the solved liquid into a 50%
    # main draw, 25% side draw and 25% drain outside the scheduling inventory.
    # Its different input and tracked output totals discriminate the rho basis.
    outlet_kg = simulation.R01.Outlet.mass  # [kg]
    side_draw = outlet_kg / 4  # [kg], 25% side draw
    main_draw = outlet_kg / 2  # [kg], 50% main draw
    stn = export_batch_stn(
        simulation, {'R01': BatchTask('reactor-1', {'rest': feed_rest, 'A': feed_a},
                                      {'main': main_draw, 'side': side_draw})},
        period_seconds=period_width, horizon_periods=expected_periods,
        storage_capacities_kg={'A': feed_a, 'rest': feed_rest,
                               'side': side_draw, 'main': main_draw},
        initial_inventory_kg={'rest': feed_rest, 'A': feed_a}, demand_kg={},
    )
    assert stn['tau'] == [{'index': ['R01', 'reactor-1'], 'value': expected_periods}]
    # The independent mixture calculation in the CLI test gives this total.
    expected_feed = 0.78184364  # [kg]
    expected_a_fraction = feed_a / expected_feed  # [kg/kg reference input]
    assert {tuple(row['index']): row['value'] for row in stn['rho_minus']} == {
        ('R01', 'rest'): pytest.approx(1 - expected_a_fraction),
        ('R01', 'A'): pytest.approx(expected_a_fraction),
    }
    assert {tuple(row['index']): row['value'] for row in stn['rho_plus']} == {
        ('R01', 'main'): pytest.approx(0.5),  # [kg/kg], half the input batch
        ('R01', 'side'): pytest.approx(0.25),  # [kg/kg], quarter of the input batch
    }
    assert stn['beta_min'][0]['value'] == pytest.approx(expected_feed)
    assert stn['beta_max'][0]['value'] == pytest.approx(expected_feed)


def test_reference_duration_covers_retained_continuation():
    """Use the complete result profile when the declared batch is continued."""
    from doc.online_docs.examples.stn_batch_example import solve_reference_sequence
    from PharmaPy.Scheduling import BatchTask, export_batch_stn

    initial_duration = 25.0  # [s], first stage duration of the documented case
    continuation_duration = 43.0  # [s], appended to the same reactor's result
    grid_width = 20.0  # [s], four periods suffice for the 68 s whole batch
    horizon = 4  # [-], last period of the whole-batch scheduling example
    simulation, feed_kg = solve_reference_sequence(
        initial_duration, continuation_duration)  # [kg], original reference charge
    simulation.SolveFlowsheet(
        kwargs_run={'R01': {'runtime': continuation_duration, 'verbose': False}},
        pick_units=['R01'], verbose=False,
    )
    outlet_kg = simulation.R01.Outlet.mass  # [kg], after the complete batch
    stn = export_batch_stn(
        simulation,
        {'R01': BatchTask('reactor-1', {'feed': feed_kg}, {'product': outlet_kg})},
        period_seconds=grid_width, horizon_periods=horizon,
        storage_capacities_kg={'feed': feed_kg, 'product': outlet_kg},
        initial_inventory_kg={'feed': feed_kg}, demand_kg={},
    )
    expected_duration = 68.0  # [s], 25 s plus 43 s in the same reactor
    assert stn['tau_p'] == [{'index': ['R01', 'reactor-1'],
                             'value': pytest.approx(expected_duration)}]
    assert stn['tau'] == [{'index': ['R01', 'reactor-1'], 'value': horizon}]


@pytest.fixture(scope='module')
def connected_batch():
    """Return the solved documentation case and mass data for boundary checks.

    Returns
    -------
    tuple
        Solved flowsheet, task declarations, and scheduling scenario with
        reference masses and storage [kg] and time grid [s, periods].
    """
    from doc.online_docs.examples.stn_batch_example import solve_reference_sequence
    from PharmaPy.Scheduling import BatchTask

    first_duration, second_duration = 25.0, 43.0  # [s], example reference durations
    simulation, feed_kg = solve_reference_sequence(first_duration, second_duration)  # [kg]
    intermediate_kg = simulation.R01.Outlet.mass  # [kg]
    product_kg = simulation.R02.Outlet.mass  # [kg]
    # Define one shared equipment choice for the scheduling abstraction even
    # though the solve used two distinct reactor instances.
    tasks = {
        'R02': BatchTask('reactor', {'middle': intermediate_kg},
                         {'product': product_kg}),
        'R01': BatchTask('reactor', {'feed': feed_kg},
                         {'middle': intermediate_kg}),
    }
    scenario = dict(
        period_seconds=20.0,  # [s], grid from the reference example
        horizon_periods=5,  # [-], two plus three periods for the sequence
        storage_capacities_kg={'feed': feed_kg, 'middle': intermediate_kg,
                               'product': product_kg},
        initial_inventory_kg={'feed': feed_kg},
        demand_kg={('product', 5): product_kg},
    )  # [kg] for storage, inventory and demand
    return simulation, tasks, scenario


def test_share_resource_and_keep_execution_order(connected_batch):
    """One equipment identifier serves both tasks in execution order."""
    from PharmaPy.Scheduling import export_batch_stn

    simulation, tasks, scenario = connected_batch
    stn = export_batch_stn(simulation, tasks, **scenario)
    assert stn['I'] == ['R01', 'R02']
    assert stn['J'] == ['reactor']
    assert stn['I_i_j_prod'] == [
        {'index': ['R01', 'reactor'], 'value': 1},
        {'index': ['R02', 'reactor'], 'value': 1},
    ]


def test_require_a_shared_state_on_direct_selected_edge(connected_batch):
    """Reject a downstream declaration that omits the selected handoff."""
    from PharmaPy.Scheduling import BatchTask, export_batch_stn

    simulation, tasks, scenario = connected_batch
    task = tasks['R02']
    disconnected = dict(tasks, R02=BatchTask(task.resource,
                                               {'feed': task.inputs_kg['middle']},
                                               task.outputs_kg))
    with pytest.raises(ValueError, match='R01.*R02.*share.*material state'):
        export_batch_stn(simulation, disconnected, **scenario)


@pytest.mark.parametrize(('field', 'value', 'message'), [
    ('horizon_periods', 2, 'R02.*exceeds.*horizon'),  # [-], R02 needs three periods
    ('initial_inventory_kg', {'feed': 1.0}, 'feed.*exceeds.*capacity'),  # [kg], above 0.782
    ('demand_kg', {('product', 6): 0.1}, 'period 6.*outside'),  # [period], [kg]
    ('demand_kg', {('product', 1.5): 0.1}, 'period 1.5.*integer'),  # invalid [period], [kg]
    ('demand_kg', {('unknown', 5): 0.1}, 'unknown state'),  # [period], [kg]
])
def test_invalid_connected_case_inputs_are_rejected(connected_batch, field,
                                                       value, message):
    """Reject invalid capacities, horizon and demand after a real solve."""
    from PharmaPy.Scheduling import export_batch_stn

    simulation, tasks, scenario = connected_batch
    with pytest.raises(ValueError, match=message):
        export_batch_stn(simulation, tasks, **{**scenario, field: value})


def test_reject_unaccounted_mass_gain(connected_batch):
    """Reject an output that exceeds the full declared input inventory."""
    from PharmaPy.Scheduling import BatchTask, export_batch_stn

    simulation, tasks, scenario = connected_batch
    first = tasks['R01']
    # Use the valid task result with its feed reduced by half to represent an
    # omitted feed contribution [kg]. The exporter must identify the task.
    reduced_charge = first.inputs_kg['feed'] / 2  # [kg]
    created = dict(tasks, R01=BatchTask(first.resource,
                                        {'feed': reduced_charge}, first.outputs_kg))
    with pytest.raises(ValueError, match='R01.*outputs more mass'):
        export_batch_stn(simulation, created, **scenario)
