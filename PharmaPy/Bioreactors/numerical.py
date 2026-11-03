"""Translate explicit inventory accuracy requirements using native state metadata."""

import numpy as np

from PharmaPy.IntegratorBackends import check_inventory
from .BioreactorUnitConverter import BioreactorUnitConverter


# Acceptance requirements stay separate from the integrator's local error controls.
def inventory_limits(unit, declaration: dict) -> dict:
    """Convert named absolute acceptance budgets into native state coordinates.

    Parameters
    ----------
    unit : MultiPhaseVessel
        Compiled native unit with named, indexed and unit-bearing solver states.
    declaration : dict
        State name to ``{unit, value}`` (uniform limit), or ``{unit, values}``
        with every named component supplied. Values are nonnegative absolute
        negative-error budgets in the declared unit, not solver tolerances.

    Returns
    -------
    dict
        StateKey to budgets [native state units]. Unnamed states retain the
        existing native compatibility budget. No relative scale is inferred.

    Raises
    ------
    ValueError
        A declaration is empty, ambiguous, malformed, dimensionally unsupported,
        or targets an unconstrained state or an incomplete component mapping.
    """
    if not isinstance(declaration, dict) or not declaration:
        raise ValueError("inventory_acceptance must be a nonempty state mapping")
    collection = unit.solver_state_collection
    limits = {}
    for name, quantity in declaration.items():
        matches = [key for key in collection.slices if collection.states[key].name == name]
        if len(matches) != 1:
            raise ValueError(f"inventory_acceptance state {name!r} is unknown or ambiguous")
        key, = matches
        state = collection.states[key]
        if not isinstance(quantity, dict) or set(quantity) not in ({'unit', 'value'}, {'unit', 'values'}):
            raise ValueError(f"inventory_acceptance {name} requires unit and value or values")
        if not isinstance(quantity['unit'], str) or not quantity['unit']:
            raise ValueError('inventory_acceptance requires an explicit unit string')
        if 'values' in quantity:
            components = quantity['values']
            if (state.index is None or not isinstance(components, dict)
                    or set(components) != set(state.index)):
                raise ValueError(f"inventory_acceptance {name} must name every native component")
            values = [components[label] for label in state.index]  # [declared unit]
        else:
            values = quantity['value']  # [declared unit]
            if np.ndim(values) != 0:
                raise ValueError("inventory_acceptance value must be scalar; use named values for components")
        try:
            values = np.asarray(values, dtype=float)  # [declared unit]
        except (TypeError, ValueError) as error:
            raise ValueError('inventory_acceptance values must be numeric') from error
        if not np.all(np.isfinite(values)) or np.any(values < 0.):
            raise ValueError("inventory_acceptance values must be finite and nonnegative")
        if quantity['unit'] == state.units:
            limits[key] = values.copy()  # [state.units]; exact native basis, including specialized units
        else:
            limits[key] = BioreactorUnitConverter.convert(values, quantity['unit'], state.units)  # [state.units]
    initial = np.asarray(unit.create_solver_init_states(), dtype=float)  # [native state units]
    check_inventory(unit, [unit.elapsed_time], initial[None, :], limits)
    return limits
