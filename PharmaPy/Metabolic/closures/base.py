"""Immutable contracts for metabolic closure backends.

The contracts connect an extracellular environment to a metabolic solver and
retain bounds, objectives, feasibility residuals, active constraints, policy
identity, cache status, and optional flux variability.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


# Declare shared metabolic inputs and immutable solution outputs.
@dataclass(frozen=True)
class MetabolicEnvironment:
    """Describe bound overrides and context for one closure evaluation.

    Attributes
    ----------
    bound_overrides : Mapping[str, tuple of float]
        ``(lower, upper)`` flux bounds by reaction, in network coordinates.
    context : Mapping[str, float], optional
        Operating conditions recorded with errors.
    """

    bound_overrides: Mapping[str, Tuple[float, float]]
    context: Mapping[str, float] = None

    def __post_init__(self):
        """Freeze the override and context mappings."""
        object.__setattr__(
            self,
            "bound_overrides",
            MappingProxyType(dict(self.bound_overrides)),
        )
        object.__setattr__(self, "context", MappingProxyType(dict(self.context or {})))


@dataclass(frozen=True)
class FluxSolution:
    """Store one auditable metabolic flux solution.

    Fluxes, bounds and residuals are in the network's declared flux
    coordinates; objectives are dimensionless scores in maximization
    convention.

    Attributes
    ----------
    reaction_ids : tuple of str
        Reaction order of ``fluxes`` and bounds.
    fluxes : ndarray
        Accepted fluxes.
    primary_objective, secondary_objective : float
        Objective values; the secondary score is None without a tie-break.
    growth_rate : float or None
        Specific growth implied by the fluxes, when defined.
    lower_bounds, upper_bounds : ndarray
        Bounds used for this solve.
    active_lower, active_upper : tuple of str
        Reactions at a bound within the solver tolerance.
    mass_balance_residual_inf, bound_violation_inf : float
        Infinity norms of ``S v`` and of bound violations.
    primal_status : str
        ``'optimal'`` or ``'locally_optimal'``.
    solver_status, solver_message : object, str
        Solver output.
    closure_policy_id : str
        Identity of the closure and tie-break policy.
    cache_hit : bool
        Whether the solution came from a cache.
    fva_intervals : Mapping, optional
        Flux variability intervals.
    inventory_violation_inf : float, optional
        Largest scaled inventory violation [-].
    tie_break : Mapping, optional
        Tie-break diagnostics.
    """

    reaction_ids: Tuple[str, ...]
    fluxes: FloatArray
    primary_objective: float
    secondary_objective: Optional[float]
    growth_rate: Optional[float]
    lower_bounds: FloatArray
    upper_bounds: FloatArray
    active_lower: Tuple[str, ...]
    active_upper: Tuple[str, ...]
    mass_balance_residual_inf: Optional[float]
    bound_violation_inf: float
    primal_status: str
    solver_status: object
    solver_message: str
    closure_policy_id: str
    cache_hit: bool
    fva_intervals: Optional[Mapping[str, Tuple[float, float]]] = None
    inventory_violation_inf: Optional[float] = None
    tie_break: Optional[Mapping] = None

    def diagnostics(self):
        """Common numerical evidence; residuals do not certify physical closure.

        Returns
        -------
        dict
            Policy, statuses, objectives, tie-break, residual norms and active
            bounds.
        """
        return dict(policy=self.closure_policy_id, primal_status=self.primal_status,
                    solver_status=self.solver_status, solver_message=self.solver_message,
                    primary_objective=self.primary_objective,
                    secondary_objective=self.secondary_objective,
                    tie_break=None if self.tie_break is None else dict(self.tie_break),
                    mass_balance_residual_inf=self.mass_balance_residual_inf,
                    bound_violation_inf=self.bound_violation_inf,
                    inventory_violation_inf=self.inventory_violation_inf,
                    active_lower=list(self.active_lower), active_upper=list(self.active_upper))


# Keep numerical failure distinct from a physically infeasible model.
class MetabolicNumericalError(RuntimeError):
    """Report an unresolved solve, not evidence of physical infeasibility."""


class MetabolicInfeasibleError(RuntimeError):
    """Report an infeasible metabolic problem with its defining bounds."""

    def __init__(self, message, *, network_id, lower_bounds, upper_bounds, status,
                 environment_context=None):
        """Record the infeasible problem's bounds and context.

        Parameters
        ----------
        message : str
            Error message.
        network_id : str
            Network or problem identity.
        lower_bounds, upper_bounds : array_like
            Bounds that define the infeasible problem, in network coordinates.
        status : object
            Solver status.
        environment_context : Mapping, optional
            Operating conditions at the failure.
        """
        super().__init__(message)
        self.network_id = network_id
        self.lower_bounds = np.asarray(lower_bounds, dtype=float).copy()
        self.upper_bounds = np.asarray(upper_bounds, dtype=float).copy()
        self.status = status
        self.environment_context = MappingProxyType(dict(environment_context or {}))
