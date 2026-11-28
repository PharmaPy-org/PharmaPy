"""Export reference batches from a solved PharmaPy flowsheet as STN data."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import ceil, fsum, isfinite
from numbers import Integral, Real
from typing import TYPE_CHECKING, Any, Union

if TYPE_CHECKING:
    from PharmaPy.SimExec import SimulationExec


@dataclass(frozen=True)
class BatchTask:
    """Map a simulated unit's reference batch to scheduling states.

    Parameters
    ----------
    resource : str
        Identifier of the equipment occupied by the task. Tasks that use the
        same identifier share one unit of equipment.
    inputs_kg, outputs_kg : mapping of str to float
        Mass consumed and produced per reference batch [kg], keyed by
        declared material state. Include only positive amounts. A state may
        represent a complete stream or inventory of known composition.

    Notes
    -----
    These amounts are a case-level declaration. The exporter does not infer
    state identity from phase names, match species, or estimate amounts for
    a different recipe or batch size.
    """

    resource: str
    inputs_kg: Mapping[str, float]
    outputs_kg: Mapping[str, float]


def _check_name(name: str, kind: str) -> None:
    """Require a nonblank string identifier for the named STN entity.

    Parameters
    ----------
    name : str
        Identifier [-] supplied for the scheduling entity.
    kind : str
        Entity description for errors.

    Raises
    ------
    ValueError
        If the identifier is not a nonblank string.
    """
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f'{kind} must have a nonblank string name.')


def _quantity(value: float, field: str, *, positive: bool = False) -> float:
    """Convert a validated scheduling quantity to a JSON-compatible float.

    Parameters
    ----------
    value : float
        Quantity [kg] for a material mass or [s] for a time.
    field : str
        Quantity name, including units, for errors.
    positive : bool, optional
        Require a strictly positive value. Otherwise zero is accepted.

    Returns
    -------
    float
        Finite value in the input units.

    Raises
    ------
    ValueError
        If the quantity is not real, finite, or within the required range.
    """
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f'{field} must be a real number.')
    try:
        value = float(value)  # [kg] or [s], same basis as the input
    except OverflowError as error:
        raise ValueError(f'{field} must be finite.') from error
    if not isfinite(value) or value < 0 or (positive and value == 0):
        constraint = 'positive' if positive else 'nonnegative'
        raise ValueError(f'{field} must be finite and {constraint}.')
    return value


def _record(index: Union[str, tuple[str, str], tuple[str, int]],
            value: float) -> dict[str, Any]:
    """Encode an indexed STN parameter using the scheduling JSON format.

    Parameters
    ----------
    index : str or tuple
        State name for a single index, or a pair of names or a state and
        discrete period [-] for a compound index.
    value : float
        Parameter in the units defined in :func:`export_batch_stn`.

    Returns
    -------
    dict
        JSON-compatible ``index`` and ``value`` fields in the input units.
    """
    return {'index': list(index) if isinstance(index, tuple) else index,
            'value': value}


def export_batch_stn(
        simulation: SimulationExec, tasks: Mapping[str, BatchTask], *,
        period_seconds: float, horizon_periods: int,
        storage_capacities_kg: Mapping[str, float],
        initial_inventory_kg: Mapping[str, float],
        demand_kg: Mapping[tuple[str, int], float]) -> dict[str, Any]:
    """Build a fixed-batch State-Task Network from selected solved units.

    Parameters
    ----------
    simulation : SimulationExec
        Solved flowsheet. Task duration comes from ``time_processing`` [s]
        recorded by the most recent ``SolveFlowsheet`` call.
    tasks : mapping of str to BatchTask
        Reference batches keyed by solved unit name. One task and one
        compatible resource per selected unit. Input and output masses [kg]
        are supplied by the case. For edges between selected units, a
        predecessor's output and its successor's input must share a state.
    period_seconds : float
        Positive duration [s] of a discrete scheduling period.
    horizon_periods : int
        Positive final period index [-]. The time grid starts at zero.
    storage_capacities_kg : mapping of str to float
        Material states and their finite storage limits [kg].
    initial_inventory_kg : mapping of str to float
        Initial charge in each state [kg]. Omitted states start empty.
    demand_kg : mapping of (str, int) to float
        Withdrawals [kg] by material state and period index in the horizon.

    Returns
    -------
    dict
        STN instance ready for JSON serialization. ``I``, ``J`` and ``K``
        list tasks, resources and material states. ``delta`` and ``tau_p``
        are seconds, ``lastT`` and ``tau`` are period indices or counts [-],
        ``rho_minus`` and ``rho_plus`` are kg per kg of reference batch,
        and ``beta_min``, ``beta_max``, ``gamma``, ``S0`` and ``demand``
        are kg. Compatibility and arc indicators are dimensionless. ``cost``
        is an empty cost table, so the scheduling model charges zero per start.

    Raises
    ------
    ValueError
        If the reference case lacks tasks, a selected unit lacks a positive
        duration or declared material states, an edge lacks a common state,
        a task exceeds the planning horizon, or a quantity is invalid.

    Notes
    -----
    The batch-size coordinate is the total input mass. Both bounds are set to
    the simulated reference size because scaling duration and yield away from
    that case is unverified. ``tau = ceil(tau_p / delta)`` reserves a whole
    number of periods and never intentionally shortens the simulated task.
    The STN consumes material at task start and releases it at completion.
    This is an approximation for a model with time-distributed transfers. It
    leaves composition, operating conditions, equipment sizing, cleaning and
    economics to the caller's documented case assumptions.
    """
    if not tasks:
        raise ValueError('Select at least one solved unit for STN export.')
    if (isinstance(horizon_periods, bool)
            or not isinstance(horizon_periods, Integral)
            or horizon_periods <= 0):
        raise ValueError('horizon_periods must be a positive integer.')
    horizon_periods = int(horizon_periods)  # [-], final grid index
    period_seconds = _quantity(period_seconds, 'period_seconds [s]',
                               positive=True)  # [s]
    durations = getattr(simulation, 'time_processing', {})  # [s], by unit
    task_names = [name for name in simulation.execution_names if name in tasks]
    unknown = [name for name in tasks if name not in simulation.execution_names]
    if unknown:
        raise ValueError(f'STN tasks are not flowsheet units: {unknown!r}.')
    for name, task in tasks.items():
        if not isinstance(task, BatchTask):
            raise ValueError(f'Task {name!r} must be declared with a BatchTask.')
        if (not isinstance(task.inputs_kg, Mapping)
                or not isinstance(task.outputs_kg, Mapping)):
            raise ValueError(f'Task {name!r} input and output masses must be '
                             'mappings from states to kilograms.')

    capacities = {}  # [kg], by material state
    for state, capacity in storage_capacities_kg.items():
        _check_name(state, 'Material state')
        capacities[state] = _quantity(capacity, f'Capacity for {state!r} [kg]')

    data = {
        'I': task_names,
        'J': [],
        'K': list(capacities),
        'delta': period_seconds,  # [s]
        'lastT': horizon_periods,  # [-]
        'I_i_k_minus': [], 'I_i_k_plus': [],  # [-]
        'rho_minus': [], 'rho_plus': [],  # [kg/kg reference batch]
        'I_i_j_prod': [],  # [-]
        'tau_p': [], 'tau': [],  # [s], [periods]
        'beta_min': [], 'beta_max': [],  # [kg of reference batch]
        'cost': [],  # [currency/task start], unspecified costs default to zero
        'gamma': [_record(state, capacity)
                  for state, capacity in capacities.items()],  # [kg]
        'S0': [], 'demand': [],  # [kg]
    }
    resources = {}
    for name in task_names:
        _check_name(name, 'Task')
        task = tasks[name]
        _check_name(task.resource, 'Resource')
        resources[task.resource] = None
        if name not in durations:
            raise ValueError(f'Task {name!r} has no solved processing time [s].')
        duration = _quantity(durations[name], f'Duration for {name!r} [s]',
                             positive=True)  # [s]
        duration_in_periods = duration / period_seconds  # [-]
        if not isfinite(duration_in_periods) or duration_in_periods > horizon_periods:
            raise ValueError(f'Task {name!r} exceeds the scheduling horizon.')
        # A positive duration must occupy at least one period, even if the
        # ratio underflows because the chosen period is much larger.
        periods = max(1, ceil(duration_in_periods))  # [periods]
        inputs = _material_amounts(name, task.inputs_kg, capacities)  # [kg]
        outputs = _material_amounts(name, task.outputs_kg, capacities)  # [kg]
        try:
            batch_size = fsum(inputs.values())  # [kg], total reference input
        except OverflowError as error:
            raise ValueError(f'Task {name!r} exceeds the representable '
                             'reference batch mass [kg].') from error
        for direction, amounts in (('minus', inputs), ('plus', outputs)):
            for state, amount in amounts.items():
                index = (name, state)
                ratio = amount / batch_size  # [kg/kg reference batch]
                if not isfinite(ratio) or ratio <= 0:
                    raise ValueError(f'Task {name!r}, state {state!r} has an '
                                     'unrepresentable mass coefficient.')
                data[f'I_i_k_{direction}'].append(_record(index, 1))
                data[f'rho_{direction}'].append(_record(index, ratio))

        index = (name, task.resource)
        data['I_i_j_prod'].append(_record(index, 1))
        data['tau_p'].append(_record(index, duration))
        data['tau'].append(_record(index, periods))
        data['beta_min'].append(_record(index, batch_size))
        data['beta_max'].append(_record(index, batch_size))
        for successor in simulation.graph.get(name, ()):
            if successor in tasks and outputs.keys().isdisjoint(tasks[successor].inputs_kg):
                raise ValueError(f'Tasks {name!r} and {successor!r} must share '
                                 'a material state for their flowsheet edge.')
    data['J'] = list(resources)

    for state, amount in initial_inventory_kg.items():
        if state not in capacities:
            raise ValueError(f'Initial inventory refers to unknown state {state!r}.')
        amount = _quantity(amount, f'Initial inventory for {state!r} [kg]')  # [kg]
        if amount > capacities[state]:
            raise ValueError(f'Initial inventory for {state!r} exceeds its capacity.')
        data['S0'].append(_record(state, amount))
    for index, amount in demand_kg.items():
        if not isinstance(index, tuple) or len(index) != 2:
            raise ValueError('Demand keys must be (state, period) pairs.')
        state, period = index
        if state not in capacities:
            raise ValueError(f'Demand refers to unknown state {state!r}.')
        if (isinstance(period, bool) or not isinstance(period, Integral)
                or not 0 <= period <= horizon_periods):
            raise ValueError(f'Demand period {period!r} is outside the time grid '
                             'or is not an integer.')
        amount = _quantity(amount, f'Demand for {state!r} [kg]')  # [kg]
        data['demand'].append(_record((state, int(period)), amount))
    return data


def _material_amounts(task: str, amounts: Mapping[str, float],
                      capacities: Mapping[str, float]) -> dict[str, float]:
    """Validate one direction of a reference batch's material mapping.

    Parameters
    ----------
    task : str
        Scheduling task name [-].
    amounts : mapping of str to float
        Reference masses [kg] consumed or produced per state.
    capacities : mapping of str to float
        Declared material state names and storage capacities [kg].

    Returns
    -------
    dict of str to float
        Validated material names and masses [kg] in declared order.

    Raises
    ------
    ValueError
        If no materials were supplied, a state is undeclared, or a mass is
        outside the supported positive finite range.
    """
    if not amounts:
        raise ValueError(f'Task {task!r} must declare input and output mass.')
    validated = {}  # [kg], reference batch amounts by material state
    for state, amount in amounts.items():
        if state not in capacities:
            raise ValueError(f'Task {task!r} uses unknown state {state!r}.')
        validated[state] = _quantity(amount, f'{task!r}/{state!r} mass [kg]',
                                     positive=True)
    return validated
