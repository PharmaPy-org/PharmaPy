"""Rate-reconciled metabolic closure with optional inventory feasibility.

The closure regularizes intracellular flux and reconciles predicted exchange
rates subject to ``S v = 0`` and declared bounds. Results retain feasibility,
residual, active-bound, and flux-variability information. The base problem is
convex; growth-dependent inventory constraints can make it nonlinear, so a
successful constrained solve is not a certificate of global optimality.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from types import MappingProxyType
from typing import Tuple

import numpy as np
from scipy.linalg import qr
from scipy.optimize import Bounds, LinearConstraint, NonlinearConstraint, linprog, minimize

from .base import (FluxSolution, MetabolicEnvironment, MetabolicInfeasibleError,
                   MetabolicNumericalError)
from ..network import MetabolicNetworkDefinition


@dataclass(frozen=True)
class ReconciliationTarget:
    """One observed or predicted flux with an explicit residual scale."""

    reaction_id: str
    value: float
    scale: float

    def __post_init__(self):
        if not self.reaction_id:
            raise ValueError("reconciliation targets require a reaction ID.")
        if not np.isfinite([self.value, self.scale]).all() or self.scale <= 0.0:
            raise ValueError("target values must be finite and scales positive.")


@dataclass(frozen=True)
class ReconciliationDiagnostics:
    """Auditable target residuals and objective-level flux intervals."""

    target_residuals: Mapping[str, float]
    objective_value: float
    fva_intervals: Mapping[str, Tuple[float, float]]

    def __post_init__(self):
        object.__setattr__(
            self,
            "target_residuals",
            MappingProxyType(dict(self.target_residuals)),
        )
        object.__setattr__(
            self,
            "fva_intervals",
            MappingProxyType(dict(self.fva_intervals)),
        )


@dataclass(frozen=True)
class _QuadraticTerms:
    """Store the precomputed terms of the reconciliation objective."""

    diagonal: np.ndarray
    linear: np.ndarray
    target_positions: np.ndarray
    target_values: np.ndarray
    target_scales: np.ndarray
    internal_positions: np.ndarray
    internal_scale: float


@dataclass(frozen=True)
class _QPResult:
    """Store the subset of optimizer output used by the closure."""

    x: np.ndarray
    fun: float
    success: bool
    status: int
    message: str
    stage: str = 'SLSQP'


class RateReconciledMFAClosure:
    """Solve rate reconciliation, optionally constrained by end inventories."""

    def __init__(self, network: MetabolicNetworkDefinition,
                 internal_reaction_ids: Sequence[str], *, tolerance=1e-8):
        self.network = network
        self.internal_reaction_ids = tuple(internal_reaction_ids)
        self.tolerance = float(tolerance)
        if not np.isfinite(self.tolerance) or self.tolerance <= 0.0:
            raise ValueError("tolerance must be finite and positive.")
        self.closure_id = "scipy-slsqp:rate-reconciled-mfa"
        self.last_diagnostics = None

    def _solve_problem(self, lower, upper, targets, internal_scale, initial=None,
                       availability=None):
        terms = _make_terms(self.network, self.internal_reaction_ids, targets, internal_scale)
        matrix = _independent_matrix(self.network.stoichiometric_matrix)
        inequality, capacity = (None, None) if availability is None else (
            availability.linear_relaxation(lower, upper))
        if availability is not None and availability.growth_index is None:
            # Constant exposure permits an exact feasibility LP, including
            # negative other increments that require compensating production.
            exposure, _ = availability._exposure(np.zeros(len(lower)))
            inequality = -exposure * availability.matrix / availability.scale[:, None]
            capacity = (availability.amounts + availability.other) / availability.scale
        feasible = linprog(
            np.zeros(len(lower)), A_eq=matrix, b_eq=np.zeros(len(matrix)),
            A_ub=inequality, b_ub=capacity,
            bounds=list(zip(lower, upper)), method="highs",
        )
        if not feasible.success:
            return _QPResult(feasible.x if feasible.x is not None else np.zeros(len(lower)),
                             np.inf, False, int(feasible.status), str(feasible.message), 'HiGHS'), terms
        start = feasible.x if initial is None else np.clip(initial, lower, upper)
        if np.max(np.abs(matrix @ start), initial=0.) > self.tolerance:
            start = feasible.x
        if availability is not None and np.min(availability.residual(start)) < -self.tolerance:
            start = feasible.x
        # Normalize the quadratic curvature for inventory-constrained solves;
        # this changes coordinates, not the objective or feasible flux set.
        variable_scale = np.ones(len(lower))
        if availability is not None:
            positive = terms.diagonal > 0.
            variable_scale[positive] = 1. / np.sqrt(terms.diagonal[positive])
        equality = matrix * variable_scale
        if availability is not None:
            equality /= np.maximum(np.max(abs(equality), axis=1), np.finfo(float).tiny)[:, None]
        constraints = [LinearConstraint(equality, 0.0, 0.0)] if len(equality) else []
        if availability is not None:
            constraints.append(NonlinearConstraint(
                lambda z: availability.residual(z * variable_scale), 0.0, np.inf,
                jac=lambda z: availability.jacobian(z * variable_scale) * variable_scale))
        def optimize(seed):
            result = minimize(
                lambda z: _reconciliation_objective(z * variable_scale, terms),
                seed / variable_scale,
                jac=lambda z: _reconciliation_gradient(z * variable_scale, terms) * variable_scale,
                bounds=Bounds(lower / variable_scale, upper / variable_scale), constraints=constraints,
                method="SLSQP", options={"ftol": 1e-10, "maxiter": 1000},
            )
            return _QPResult(result.x * variable_scale, result.fun, result.success,
                             result.status, result.message)

        result = optimize(start)
        if (result.success and np.isfinite(result.fun)
                and self._acceptable(result.x, lower, upper, availability)):
            return result, terms
        from ._recovery import boundary_candidate
        candidate = boundary_candidate(self.network.stoichiometric_matrix,
                                       lower, upper, terms, availability, self.tolerance)
        if candidate is not None:
            flux, value, gap, defect = candidate
            if self._acceptable(flux, lower, upper, availability):
                return _QPResult(flux, value, True, 0,
                    f'original-problem objective gap={gap}; reconstruction allowance={defect}',
                    'bounded reconciliation'), terms
        # An unqualified boundary candidate says nothing about interior optima.
        # Allow one full-bound solve from the LP seed, without fixing growth.
        retry = optimize(feasible.x)
        if (retry.success and np.isfinite(retry.fun)
                and self._acceptable(retry.x, lower, upper, availability)):
            return retry, terms
        return _QPResult(result.x, result.fun, False, result.status,
                         f'{result.message}; boundary recovery not qualified'), terms

    def _acceptable(self, flux, lower, upper, availability):
        """Check the full original constraints, including dependent balance rows."""
        if not np.isfinite(flux).all():
            return False
        tolerance = self.tolerance * max(1., float(np.max(abs(flux))))
        if (np.max(abs(self.network.stoichiometric_matrix @ flux)) > tolerance
                or np.max(lower - flux) > tolerance or np.max(flux - upper) > tolerance):
            return False
        if availability is None:
            return True
        residual = availability.residual(flux)
        return np.isfinite(residual).all() and np.min(residual) >= -self.tolerance

    def solve(self, environment: MetabolicEnvironment, previous_solution=None, *,
              targets: Sequence[ReconciliationTarget], internal_scale: float,
              fva_reactions: Sequence[str] = (), fva_fraction=0.05,
              availability=None):
        self.last_diagnostics = None
        if not np.isfinite(internal_scale) or internal_scale <= 0.0:
            raise ValueError("internal_scale must be finite and positive.")
        if not targets:
            raise ValueError("rate reconciliation requires at least one target.")
        lower, upper = self.network.bounds(environment.bound_overrides)
        initial = None if previous_solution is None else np.asarray(previous_solution.fluxes)
        result, terms = self._solve_problem(lower, upper, tuple(targets), internal_scale,
                                          initial, availability)
        residual = self.network.stoichiometric_matrix @ result.x
        violation = max(float(np.max(lower - result.x)), float(np.max(result.x - upper)), 0.0)
        scale = max(1.0, float(np.max(np.abs(result.x))))
        if result.stage == 'HiGHS' and result.status == 2:
            raise MetabolicInfeasibleError(
                f"rate-reconciled MFA failed: {result.message}",
                network_id=self.network.network_id, lower_bounds=lower,
                upper_bounds=upper, status=result.status,
                environment_context=environment.context,
            )
        if (not result.success or not np.isfinite(result.fun)
                or not self._acceptable(result.x, lower, upper, availability)):
            raise MetabolicNumericalError(
                f'rate-reconciled MFA unresolved for {self.network.network_id}: '
                f'{result.stage}: {result.status}: {result.message}; '
                f'balance residual={np.max(abs(residual))}, bound violation={violation}')
        fluxes = np.asarray(result.x, dtype=float)
        intervals = self.flux_variability(
            environment, targets, internal_scale, fluxes, float(result.fun),
            fva_reactions, fva_fraction, availability=availability
        ) if fva_reactions else {}
        target_residuals = {
            target.reaction_id: float((fluxes[position] - value) / target_scale)
            for target, position, value, target_scale in zip(
                targets, terms.target_positions, terms.target_values, terms.target_scales
            )
        }
        self.last_diagnostics = ReconciliationDiagnostics(
            target_residuals, float(result.fun), intervals
        )
        active_tolerance = self.tolerance * scale
        fluxes.setflags(write=False)
        lower.setflags(write=False)
        upper.setflags(write=False)
        return FluxSolution(
            reaction_ids=self.network.reaction_ids, fluxes=fluxes,
            primary_objective=-float(result.fun), secondary_objective=None,
            growth_rate=None, lower_bounds=lower, upper_bounds=upper,
            active_lower=tuple(
                name
                for name, value, bound in zip(
                    self.network.reaction_ids, fluxes, lower
                )
                if abs(value - bound) <= active_tolerance
            ),
            active_upper=tuple(
                name
                for name, value, bound in zip(
                    self.network.reaction_ids, fluxes, upper
                )
                if abs(value - bound) <= active_tolerance
            ),
            mass_balance_residual_inf=float(np.max(np.abs(residual))),
            bound_violation_inf=violation,
            primal_status="optimal" if availability is None else "locally_optimal",
            solver_status=int(result.status), solver_message=str(result.message),
            closure_policy_id=self.closure_id, cache_hit=False,
            fva_intervals=MappingProxyType(intervals) if intervals else None,
        )

    def flux_variability(self, environment, targets, internal_scale, nominal_fluxes,
                         optimum, reaction_ids, fraction, availability=None):
        if not np.isfinite(fraction) or fraction < 0.0:
            raise ValueError("FVA objective fraction must be finite and nonnegative.")
        lower, upper = self.network.bounds(environment.bound_overrides)
        terms = _make_terms(self.network, self.internal_reaction_ids, targets, internal_scale)
        ceiling = optimum + max(self.tolerance, abs(optimum) * fraction)
        objective_limit = partial(_objective_slack, terms=terms, ceiling=ceiling)
        constraints = [
            LinearConstraint(_independent_matrix(self.network.stoichiometric_matrix), 0.0, 0.0),
            {"type": "ineq", "fun": objective_limit},
        ]
        if availability is not None:
            constraints.append(NonlinearConstraint(
                availability.residual, 0.0, np.inf, jac=availability.jacobian))
        index = {name: position for position, name in enumerate(self.network.reaction_ids)}
        intervals = {}
        for name in reaction_ids:
            if name not in index:
                raise ValueError(f"FVA references unknown reaction {name!r}.")
            position = index[name]
            low = minimize(_selected_flux, nominal_fluxes, args=(position, 1.0),
                           bounds=Bounds(lower, upper), constraints=constraints,
                           method="SLSQP", options={"ftol": 1e-10, "maxiter": 2000})
            high = minimize(_selected_flux, nominal_fluxes, args=(position, -1.0),
                            bounds=Bounds(lower, upper), constraints=constraints,
                            method="SLSQP", options={"ftol": 1e-10, "maxiter": 2000})
            if not low.success or not high.success:
                raise RuntimeError(f"reconciled-MFA FVA failed for {name!r}.")
            if availability is not None and any(
                not np.isfinite(availability.residual(x)).all()
                or np.min(availability.residual(x)) < -self.tolerance
                for x in (low.x, high.x)
            ):
                raise RuntimeError(f"inventory-constrained FVA failed for {name!r}.")
            intervals[name] = (float(low.x[position]), float(high.x[position]))
        return intervals


def _index_targets(reaction_ids, targets):
    index = {name: position for position, name in enumerate(reaction_ids)}
    if len({target.reaction_id for target in targets}) != len(targets):
        raise ValueError("reconciliation target reaction IDs must be unique.")
    unknown = sorted(target.reaction_id for target in targets if target.reaction_id not in index)
    if unknown:
        raise ValueError(f"reconciliation targets reference unknown reactions {unknown}.")
    positions = np.asarray([index[target.reaction_id] for target in targets], dtype=int)
    values = np.asarray([target.value for target in targets], dtype=float)
    scales = np.asarray([target.scale for target in targets], dtype=float)
    return positions, values, scales


def _selected_flux(flux, position, direction):
    return float(direction * flux[position])


def _reconciliation_objective(flux, terms):
    internal = np.sum((flux[terms.internal_positions] / terms.internal_scale) ** 2)
    reconciled = np.sum(
        ((flux[terms.target_positions] - terms.target_values) / terms.target_scales) ** 2
    )
    return float(internal + reconciled)


def _reconciliation_gradient(flux, terms):
    return 2.0 * (terms.diagonal * flux + terms.linear)


def _objective_slack(flux, terms, ceiling):
    return float(ceiling - _reconciliation_objective(flux, terms))


def _independent_matrix(matrix):
    rank = int(np.linalg.matrix_rank(matrix))
    if rank == len(matrix):
        return matrix
    _, _, pivots = qr(matrix.T, mode="economic", pivoting=True)
    return matrix[np.sort(pivots[:rank])]


def _make_terms(network, internal_reaction_ids, targets, internal_scale):
    index = {name: position for position, name in enumerate(network.reaction_ids)}
    unknown = set(internal_reaction_ids) - set(index)
    if unknown:
        raise ValueError(
            "internal regularization references unknown reactions "
            f"{sorted(unknown)}."
        )
    diagonal = np.zeros(len(network.reaction_ids), dtype=float)
    linear = np.zeros(len(network.reaction_ids), dtype=float)
    internal_positions = np.asarray([index[name] for name in internal_reaction_ids], dtype=int)
    diagonal[internal_positions] += 1.0 / internal_scale ** 2
    positions, values, scales = _index_targets(network.reaction_ids, targets)
    for position, value, scale in zip(positions, values, scales):
        diagonal[position] += 1.0 / scale ** 2
        linear[position] -= value / scale ** 2
    return _QuadraticTerms(
        diagonal, linear, positions, values, scales, internal_positions, internal_scale
    )
