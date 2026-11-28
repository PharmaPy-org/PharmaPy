"""Validate the STN boundary with the solver-independent batch mixer path."""

import pytest

from PharmaPy.Containers import Mixer
from PharmaPy.Phases import LiquidPhase
from PharmaPy.Scheduling import BatchTask, export_batch_stn
from PharmaPy.SimExec import SimulationExec

pytestmark = pytest.mark.unit


@pytest.fixture
def batch_case(data_path):
    """Build an unsolved batch mixer and matching reference scheduling data.

    Parameters
    ----------
    data_path : dict
        Repository fixture directories.

    Returns
    -------
    tuple
        Simulation, task mapping with masses [kg], and scenario inputs
        with period width [s], horizon [-], capacities and demands [kg].
    """
    properties = str(data_path['flowsheet'] / 'compound_database.json')
    simulation = SimulationExec(properties, {'Mix': []})
    simulation.Mix = Mixer()
    # [kg], round reference charge for a static mixer. It retains the full
    # mass in the outlet and has no processing time on its simulation grid.
    reference_mass = 1.0
    simulation.Mix.Inlets = LiquidPhase(
        properties, mass=reference_mass, mass_frac=[1., 0., 0., 0., 0.])  # [-]
    tasks = {
        'Mix': BatchTask('Mixer', {'Feed': reference_mass},
                         {'Blend': reference_mass}),
    }  # [kg] per reference batch on each side
    scenario = dict(
        period_seconds=10.0,  # [s], arbitrary grid for checking rejection
        horizon_periods=1,  # [-], minimal positive horizon
        storage_capacities_kg={'Feed': reference_mass, 'Blend': reference_mass},
        initial_inventory_kg={'Feed': reference_mass},
        demand_kg={('Blend', 1): reference_mass},
    )  # [kg] for storage, inventory and demand
    return simulation, tasks, scenario


def test_rejects_unsolved_and_instantaneous_tasks(batch_case):
    """A registered unit and a real zero-time mixer lack schedulable spans."""
    simulation, tasks, scenario = batch_case
    with pytest.raises(ValueError, match='Mix.*no solved processing time'):
        export_batch_stn(simulation, tasks, **scenario)

    simulation.SolveFlowsheet(verbose=False)
    assert simulation.time_processing['Mix'] == pytest.approx(0.0)  # [s]
    with pytest.raises(ValueError, match='Duration for .*Mix.*positive'):
        export_batch_stn(simulation, tasks, **scenario)


def test_rejects_unrecognized_flowsheet_task(batch_case):
    simulation, tasks, scenario = batch_case
    with pytest.raises(ValueError, match='not flowsheet units.*Unknown'):
        export_batch_stn(simulation, {'Unknown': tasks['Mix']}, **scenario)


@pytest.mark.parametrize(('field', 'value', 'message'), [
    ('period_seconds', float('inf'), 'period_seconds.*finite'),  # [s]
    ('horizon_periods', 1.5, 'horizon_periods.*positive integer'),  # [-]
    ('storage_capacities_kg', {'Feed': -1.0}, 'Capacity for .*nonnegative'),  # [kg]
])
def test_rejects_invalid_scenario_quantities(batch_case, field, value, message):
    """Reject an infinite grid, fractional horizon, or negative storage."""
    simulation, tasks, scenario = batch_case
    scenario[field] = value
    with pytest.raises(ValueError, match=message):
        export_batch_stn(simulation, tasks, **scenario)
