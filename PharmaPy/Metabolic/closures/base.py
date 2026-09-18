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


@dataclass(frozen=True)
class MetabolicEnvironment:
    """Describe bound overrides and context for one closure evaluation."""

    bound_overrides: Mapping[str, Tuple[float, float]]
    context: Mapping[str, float] = None

    def __post_init__(self):
        object.__setattr__(
            self,
            "bound_overrides",
            MappingProxyType(dict(self.bound_overrides)),
        )
        object.__setattr__(self, "context", MappingProxyType(dict(self.context or {})))


@dataclass(frozen=True)
class FluxSolution:
    """Store one auditable metabolic flux solution."""

    reaction_ids: Tuple[str, ...]
    fluxes: FloatArray
    primary_objective: float
    secondary_objective: Optional[float]
    growth_rate: Optional[float]
    lower_bounds: FloatArray
    upper_bounds: FloatArray
    active_lower: Tuple[str, ...]
    active_upper: Tuple[str, ...]
    mass_balance_residual_inf: float
    bound_violation_inf: float
    primal_status: str
    solver_status: object
    solver_message: str
    closure_policy_id: str
    cache_hit: bool
    fva_intervals: Optional[Mapping[str, Tuple[float, float]]] = None


class MetabolicNumericalError(RuntimeError):
    """Report an unresolved solve, not evidence of physical infeasibility."""


class MetabolicInfeasibleError(RuntimeError):
    """Report an infeasible metabolic problem with its defining bounds."""

    def __init__(self, message, *, network_id, lower_bounds, upper_bounds, status,
                 environment_context=None):
        super().__init__(message)
        self.network_id = network_id
        self.lower_bounds = np.asarray(lower_bounds, dtype=float).copy()
        self.upper_bounds = np.asarray(upper_bounds, dtype=float).copy()
        self.status = status
        self.environment_context = MappingProxyType(dict(environment_context or {}))
