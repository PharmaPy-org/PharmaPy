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
from scipy.optimize import Bounds, LinearConstraint, NonlinearConstraint, linprog, minimize, nnls

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
                       availability=None, _reduce_intervals=True, _presolve=True):
        terms = _make_terms(self.network, self.internal_reaction_ids, targets, internal_scale)
        matrix = self.network.stoichiometric_matrix
        # Compatible targets have a zero-error optimum. Recover it directly
        # when balances uniquely determine the remaining fluxes.
        if not len(terms.internal_positions):
            candidate = np.zeros(len(lower))
            candidate[terms.target_positions] = terms.target_values
            unknown = np.ones(len(lower), dtype=bool)
            unknown[terms.target_positions] = False
            rank = 0
            if unknown.any():
                row_scale = np.sum(abs(matrix[:, unknown]), axis=1)
                row_scale[row_scale == 0.] = 1.
                reduced = matrix[:, unknown] / row_scale[:, None]
                rank = np.linalg.matrix_rank(reduced)
                if rank == np.count_nonzero(unknown):
                    _, _, pivots = qr(reduced.T, mode='economic', pivoting=True)
                    rows = pivots[:rank]
                    try:
                        candidate[unknown] = np.linalg.solve(
                            reduced[rows], -(matrix @ candidate)[rows] / row_scale[rows])
                    except np.linalg.LinAlgError:
                        rank = 0
            roundoff = 64 * np.finfo(float).eps
            near_bound = unknown & (
                np.maximum(lower - candidate, candidate - upper)
                <= roundoff * np.maximum(1., abs(candidate)))
            candidate[near_bound] = np.clip(candidate[near_bound], lower[near_bound], upper[near_bound])
            if (rank == np.count_nonzero(unknown)
                    and np.all(abs(matrix @ candidate) <= roundoff * np.maximum(1., abs(matrix) @ abs(candidate)))
                    and np.all(candidate >= lower) and np.all(candidate <= upper)
                    and self._acceptable(candidate, lower, upper, availability)):
                return _QPResult(candidate, 0., True, 0, 'unique compatible targets'), terms
        original_bounds = lower, upper
        if availability is not None and _presolve:
            inequality, capacity = availability.linear_relaxation(lower, upper)
            lower, upper = lower.copy(), upper.copy()
            if inequality is not None:
                for row, limit in zip(inequality, capacity):
                    positions = np.flatnonzero(row)
                    if len(positions) == 1:
                        i = positions[0]
                        if row[i] > 0.:
                            upper[i] = min(upper[i], limit / row[i])
                        else:
                            lower[i] = max(lower[i], limit / row[i])
            if np.any(lower > upper):
                return _QPResult(lower, np.inf, False, 2,
                                 'inventory limits conflict with flux bounds', 'HiGHS'), terms
        variable_scale = np.ones(len(lower))
        positive = terms.diagonal > 0.
        variable_scale[positive] = 1. / np.sqrt(terms.diagonal[positive])
        variable_scale[~positive] = np.max(variable_scale[positive], initial=0.) or 1.
        narrow = (_reduce_intervals & (upper > lower) &
                  (upper - lower <= 32 * np.finfo(float).eps * variable_scale))
        free = (lower != upper) & ~narrow
        offset = np.where(free, 0., lower)

        def certified(flux):
            from ._recovery import _objective_gap
            a, b = (None, None) if availability is None else availability.linear_relaxation(lower, upper)
            if a is None:
                a, b = np.empty((0, len(lower))), np.empty(0)
            try:
                gap, error, value = _objective_gap(matrix, lower, upper, terms, a, b, flux)
            except (ValueError, RuntimeError, np.linalg.LinAlgError):
                return False
            return gap >= -error and gap + error <= self.tolerance * max(1., abs(value))

        def expand(z):
            flux = offset.copy()
            flux[free] = z * variable_scale[free]
            return flux

        if not np.any(free):
            balance_scale = abs(matrix) @ abs(offset)
            balance_scale[balance_scale == 0.] = 1.
            acceptable = (self._acceptable(offset, lower, upper, availability)
                          and np.max(abs(matrix @ offset) / balance_scale) <= self.tolerance)
            acceptable = acceptable and (not narrow.any() or certified(offset))
            if not acceptable and narrow.any():
                return self._solve_problem(*original_bounds, targets, internal_scale,
                                           initial, availability, False, _presolve)
            return _QPResult(offset, _reconciliation_objective(offset, terms),
                             acceptable, 0 if acceptable else 2,
                             'all fluxes fixed', 'SLSQP' if narrow.any() else 'HiGHS'), terms
        equality = matrix[:, free] * variable_scale[free]
        rhs = -matrix @ offset
        row_scale = np.maximum(np.max(abs(equality), axis=1), abs(rhs))
        row_scale[row_scale == 0.] = 1.
        equality = equality / row_scale[:, None]
        rhs = rhs / row_scale
        inequality, capacity = (None, None) if availability is None else (
            availability.linear_relaxation(lower, upper))
        if inequality is not None:
            capacity = capacity - inequality @ offset
            inequality = inequality[:, free] * variable_scale[free]
            norms = np.maximum(np.max(abs(inequality), axis=1), abs(capacity))
            norms[norms == 0.] = 1.
            inequality, capacity = inequality / norms[:, None], capacity / norms
        low, high = lower[free] / variable_scale[free], upper[free] / variable_scale[free]
        # Keep every row for feasibility; only the optimizer needs a full-rank subset.
        feasible = linprog(
            np.zeros(sum(free)), A_eq=equality, b_eq=rhs,
            A_ub=inequality, b_ub=capacity,
            bounds=list(zip(low, high)), method="highs",
        )
        if not feasible.success:
            if narrow.any():
                return self._solve_problem(*original_bounds, targets, internal_scale,
                                           initial, availability, False, _presolve)
            return _QPResult(offset, np.inf, False, int(feasible.status),
                             str(feasible.message), 'HiGHS'), terms
        singular = np.linalg.svd(equality, compute_uv=False)
        rank = int(sum(singular > np.max(singular, initial=0.) * max(equality.shape)
                       * np.finfo(float).eps))
        orthogonal, _, pivots = qr(equality.T, mode="full", pivoting=True)
        rows = np.sort(pivots[:rank])
        reduced, reduced_rhs = equality[rows], rhs[rows]
        tangent = orthogonal[:, rank:]
        if rank and singular[rank - 1] < singular[0] * np.sqrt(np.finfo(float).eps):
            from ._recovery import _affine_basis
            origin, basis = _affine_basis(matrix[:, free], -matrix @ offset)
            origin = np.asarray(origin, dtype=float) / variable_scale[free]
            basis = np.asarray(basis, dtype=float).reshape(sum(free), -1)
            orthogonal, _ = qr(basis / variable_scale[free, None], mode="full")
            tangent = orthogonal[:, :basis.shape[1]]
            reduced = orthogonal[:, basis.shape[1]:].T
            reduced_rhs = reduced @ origin
        constraints = [LinearConstraint(reduced, reduced_rhs, reduced_rhs)] if rank else []
        if availability is not None:
            constraints.append(NonlinearConstraint(
                lambda z: availability.residual(expand(z)), 0.0, np.inf,
                jac=lambda z: availability.jacobian(expand(z))[:, free] * variable_scale[free]))
        start = feasible.x
        if initial is not None:
            proposed = np.clip(initial[free] / variable_scale[free], low, high)
            if (np.max(abs(equality @ proposed - rhs), initial=0.) <= self.tolerance
                    and self._acceptable(expand(proposed), lower, upper, availability)):
                start = proposed

        def feasible_flux(flux):
            z = flux[free] / variable_scale[free]
            return (self._acceptable(flux, lower, upper, availability)
                    and np.max(abs(equality @ z - rhs), initial=0.) <= self.tolerance
                    and np.max(low - z, initial=0.) <= self.tolerance
                    and np.max(z - high, initial=0.) <= self.tolerance)

        def qualified(flux):
            if not feasible_flux(flux):
                return False
            z = flux[free] / variable_scale[free]
            gradient = _reconciliation_gradient(flux, terms)[free] * variable_scale[free]
            active = np.eye(len(z))[z - low <= self.tolerance]
            active = np.vstack((active, -np.eye(len(z))[high - z <= self.tolerance]))
            if availability is not None:
                residual = availability.residual(flux)
                jacobian = availability.jacobian(flux)[:, free] * variable_scale[free]
                active = np.vstack((active, jacobian[residual <= self.tolerance]))
            projected = tangent.T @ gradient
            if active.size and projected.size:
                try:
                    directions = tangent.T @ active.T
                    norms = np.linalg.norm(directions, axis=0)
                    # Constant directions can leave roundoff after projection.
                    # Compare against the original row before normalizing it.
                    keep = norms > (32 * np.finfo(float).eps * max(equality.shape)
                                    * np.linalg.norm(active, axis=1))
                    if keep.any():
                        _, defect = nnls(directions[:, keep] / norms[keep], projected,
                                         maxiter=10 * (len(active) + 1))
                    else:
                        defect = np.linalg.norm(projected)
                except RuntimeError:
                    return False
            else:
                defect = np.linalg.norm(projected)
            if defect > 1e-5 * max(1., np.linalg.norm(gradient)):
                return False
            if narrow.any():
                # Collapsing a numerically unresolved interval only proposes a
                # candidate. Certify it against the uncollapsed problem.
                return certified(flux)
            return True

        def optimize(seed, eliminate=False):
            transform, origin = np.eye(len(seed)), np.zeros(len(seed))
            local_constraints, bounds = constraints, Bounds(low, high)
            if eliminate:
                transform = tangent
                origin = np.linalg.lstsq(reduced, reduced_rhs, rcond=None)[0]
                if not transform.shape[1]:
                    flux = expand(origin)
                    return _QPResult(flux, _reconciliation_objective(flux, terms),
                                     qualified(flux), 0, 'balance-determined fluxes')
                norms = np.linalg.norm(transform, axis=1)
                keep = norms > 32 * np.finfo(float).eps * max(equality.shape)
                local_constraints = [LinearConstraint(
                    transform[keep] / norms[keep, None],
                    (low - origin)[keep] / norms[keep],
                    (high - origin)[keep] / norms[keep])]
                if availability is not None:
                    local_constraints.append(NonlinearConstraint(
                        lambda w: availability.residual(expand(origin + transform @ w)), 0., np.inf,
                        jac=lambda w: (availability.jacobian(expand(origin + transform @ w))[:, free]
                                       * variable_scale[free]) @ transform))
                bounds = Bounds(-np.inf, np.inf)
            result = minimize(
                lambda w: _reconciliation_objective(expand(origin + transform @ w), terms),
                transform.T @ (seed - origin),
                jac=lambda w: transform.T @ (
                    _reconciliation_gradient(expand(origin + transform @ w), terms)[free]
                    * variable_scale[free]),
                bounds=bounds, constraints=local_constraints,
                method="SLSQP", options={"ftol": 1e-15, "maxiter": 1000},
            )
            z = origin + transform @ result.x
            # Linear-bound solvers can reconstruct just outside an active bound.
            # Restore declared bounds only within tolerance, then audit all balances.
            if max(np.max(low - z), np.max(z - high)) <= self.tolerance:
                z = np.clip(z, low, high)
            flux = expand(z)
            return _QPResult(flux, _reconciliation_objective(flux, terms), result.success,
                             result.status, result.message)

        result = optimize(start)
        if (result.success and np.isfinite(result.fun)
                and qualified(result.x)):
            return result, terms
        # Try full-bound floating solves before exact boundary recovery.
        retry = optimize(feasible.x)
        if (retry.success and np.isfinite(retry.fun)
                and qualified(retry.x)):
            return retry, terms
        # Solve in the balance null space if constrained coordinates stagnate.
        seed = feasible.x
        for _ in range(2):
            retry = optimize(seed, eliminate=True)
            if retry.success and np.isfinite(retry.fun) and qualified(retry.x):
                return retry, terms
            if not retry.success or not feasible_flux(retry.x):
                break
            seed = retry.x[free] / variable_scale[free]
        from ._recovery import boundary_candidate
        candidate = boundary_candidate(self.network.stoichiometric_matrix,
                                       lower, upper, terms, availability, self.tolerance)
        if candidate is not None:
            flux, value, gap, defect = candidate
            if feasible_flux(flux):
                return _QPResult(flux, value, True, 0,
                    f'original-problem objective gap={gap}; reconstruction allowance={defect}',
                    'bounded reconciliation'), terms
        if narrow.any():
            return self._solve_problem(*original_bounds, targets, internal_scale,
                                       initial, availability, False, _presolve)
        if _presolve and availability is not None:
            return self._solve_problem(*original_bounds, targets, internal_scale,
                                       initial, availability, False, False)
        return _QPResult(result.x, result.fun, False, result.status,
                         f'{result.message}; stationarity/feasibility or boundary recovery not qualified'), terms

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

    def _tie_break(self, flux, terms, lower, upper, availability):
        """Select minimum capacity-normalized flux on the primary optimal face."""
        free = (terms.diagonal == 0.) & (lower < upper)
        if not free.any():
            return flux, None, None
        scales = np.maximum(abs(self.network.lower_bounds), abs(self.network.upper_bounds))
        matrix = self.network.stoichiometric_matrix[:, free] * scales[free]
        norms = np.max(abs(matrix), axis=1)
        matrix = matrix / np.where(norms > 0., norms, 1.)[:, None]
        if np.linalg.matrix_rank(matrix) == free.sum():
            return flux, None, None
        fixed = ~free
        low, high = lower.copy(), upper.copy()
        low[fixed] = high[fixed] = flux[fixed]
        # Common conditioning changes neither the norm's minimizer nor its units.
        magnitude = np.max(abs(flux[free]) / scales[free]) or 1.
        targets = [ReconciliationTarget(name, 0., scale * magnitude)
                   for name, scale in zip(np.asarray(self.network.reaction_ids)[free], scales[free])]
        selector = RateReconciledMFAClosure(self.network, (), tolerance=self.tolerance)
        result, _ = selector._solve_problem(low, high, targets, 1., availability=availability)
        if (not result.success or not self._acceptable(result.x, lower, upper, availability)
                or not np.array_equal(result.x[fixed], flux[fixed])):
            raise MetabolicNumericalError(f'automatic reconciliation tie-break failed: {result.message}')
        nonlinear = (getattr(availability, 'growth_index', None) is not None
                     and hasattr(availability, 'step_day'))
        value = float(np.sum((result.x[free] / scales[free]) ** 2))
        diagnostic = MappingProxyType(dict(
            policy='minimum-capacity-normalized-squared-flux',
            guarantee='local-selection' if nonlinear else 'unique-on-primary-face',
            changed_reactions=tuple(name for name, change, reference in zip(
                self.network.reaction_ids, abs(result.x - flux), np.maximum(abs(flux), 1e-30))
                if change > self.tolerance * reference),
            primary_objective_change=float(_reconciliation_objective(result.x, terms)
                                           - _reconciliation_objective(flux, terms))))
        return result.x, -value, diagnostic

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
        fluxes, secondary, tie_break = self._tie_break(result.x, terms, lower, upper, availability)
        fluxes = np.asarray(fluxes, dtype=float)
        residual = self.network.stoichiometric_matrix @ fluxes
        violation = max(float(np.max(lower - fluxes)), float(np.max(fluxes - upper)), 0.0)
        scale = max(1.0, float(np.max(np.abs(fluxes))))
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
            primary_objective=-float(result.fun), secondary_objective=secondary,
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
            inventory_violation_inf=(None if availability is None else
                                     float(max(0., -np.min(availability.residual(fluxes))))),
            tie_break=tie_break,
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
