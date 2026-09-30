"""Solve arbitrary input-declared extracellular pathway models.

The declarations define a dimension-independent pathway linear program. State
and pathway identities, exchange coefficients, objectives, bounds, and kinetic
parameters are supplied by the owning simulation input.

"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linprog

from .closures.base import FluxSolution, MetabolicInfeasibleError, MetabolicNumericalError

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class PathwayModelDefinition:
    """Describe a dimension-independent extracellular pathway optimization."""

    state_ids: tuple
    pathway_ids: tuple
    exchange_matrix: FloatArray
    growth_coefficients: FloatArray
    objective_coefficients: FloatArray
    parameters: Mapping[str, float] = field(default_factory=dict)
    uptake_constraints: tuple = ()
    pathway_bound_rules: tuple = ()
    depletion_tolerance: float = 1.0e-8
    problem_name: str = "configured pathway optimization"

    def __post_init__(self):
        states = tuple(str(item) for item in self.state_ids)
        pathways = tuple(str(item) for item in self.pathway_ids)
        if (
            not states
            or len(states) != len(set(states))
            or any(not item for item in states)
        ):
            raise ValueError("state_ids must be unique nonempty identifiers.")
        if (
            not pathways
            or len(pathways) != len(set(pathways))
            or any(not item for item in pathways)
        ):
            raise ValueError("pathway_ids must be unique nonempty identifiers.")
        matrix = _readonly_matrix(
            self.exchange_matrix, "exchange_matrix", (len(states), len(pathways))
        )
        growth = _readonly_vector(
            self.growth_coefficients, "growth_coefficients", len(pathways)
        )
        objective = _readonly_vector(
            self.objective_coefficients, "objective_coefficients", len(pathways)
        )
        parameters = {str(key): float(value) for key, value in self.parameters.items()}
        if any(not key for key in parameters) or not np.isfinite(
            list(parameters.values())
        ).all():
            raise ValueError("parameters must contain named finite values.")
        tolerance = float(self.depletion_tolerance)
        if not np.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("depletion_tolerance must be finite and nonnegative.")
        if not isinstance(self.problem_name, str) or not self.problem_name:
            raise ValueError("problem_name must be a nonempty string.")
        object.__setattr__(self, "state_ids", states)
        object.__setattr__(self, "pathway_ids", pathways)
        object.__setattr__(self, "exchange_matrix", matrix)
        object.__setattr__(self, "growth_coefficients", growth)
        object.__setattr__(self, "objective_coefficients", objective)
        object.__setattr__(self, "parameters", MappingProxyType(parameters))
        object.__setattr__(self, "uptake_constraints", tuple(
            MappingProxyType(dict(item)) for item in self.uptake_constraints
        ))
        object.__setattr__(self, "pathway_bound_rules", tuple(
            MappingProxyType(dict(item)) for item in self.pathway_bound_rules
        ))
        self._validate_rules()

    def _validate_rules(self):
        """Validate all rule references and registered mathematical forms."""
        state_set = set(self.state_ids)
        pathway_set = set(self.pathway_ids)
        parameter_set = set(self.parameters)
        uptake_types = {"constant", "monod", "transfer-cap", "provider"}
        identifiers = tuple(rule.get("identifier") for rule in self.uptake_constraints)
        if any(not isinstance(item, str) or not item for item in identifiers) or len(
            identifiers
        ) != len(set(identifiers)):
            raise ValueError("uptake constraint identifiers must be unique nonempty strings.")
        for rule in self.uptake_constraints:
            required = {"identifier", "type", "species"}
            if not required <= set(rule) or rule["type"] not in uptake_types:
                raise ValueError(f"invalid uptake constraint declaration: {dict(rule)!r}")
            if rule["species"] not in state_set:
                raise ValueError(
                    "uptake constraint references unknown state "
                    f"{rule['species']!r}."
                )
            if rule['type'] == 'provider':
                if set(rule) != required | {'output'} or not isinstance(rule['output'], str) or not rule['output']:
                    raise ValueError('provider uptake requires a named output')
                continue
            parameter_fields = {
                "constant": ("maximum_parameter",),
                "monod": ("maximum_parameter", "half_saturation_parameter"),
                "transfer-cap": (
                    "maximum_parameter", "mass_transfer_parameter", "saturation_parameter",
                ),
            }[rule["type"]]
            allowed = required | set(parameter_fields)
            if set(rule) != allowed:
                raise ValueError(
                    f"uptake constraint fields mismatch for {rule['type']!r}."
                )
            missing = [
                field
                for field in parameter_fields
                if rule.get(field) not in parameter_set
            ]
            if missing:
                raise ValueError(f"uptake constraint has unknown parameter references: {missing}")
        for rule in self.pathway_bound_rules:
            if (
                set(rule) != {"type", "species", "pathways"}
                or rule.get("type") != "disable-at-depletion"
            ):
                raise ValueError(f"unknown pathway bound rule {rule.get('type')!r}.")
            if rule.get("species") not in state_set:
                raise ValueError("pathway bound rule references an unknown state.")
            selected = tuple(rule.get("pathways", ()))
            if not selected or not set(selected) <= pathway_set:
                raise ValueError("pathway bound rule references unknown pathways.")

    @classmethod
    def from_mapping(cls, payload):
        """Construct a definition from a JSON-compatible model declaration."""
        required = {
            "state_ids", "pathway_ids", "exchange_matrix", "growth_coefficients",
            "objective_coefficients", "parameters", "uptake_constraints",
            "pathway_bound_rules", "depletion_tolerance", "problem_name",
        }
        if set(payload) != required:
            raise ValueError(
                "pathway model fields mismatch; "
                f"missing={sorted(required - set(payload))}, "
                f"unknown={sorted(set(payload) - required)}"
            )
        return cls(**payload)


@dataclass(frozen=True)
class PathwayOptimizationResult(FluxSolution):
    """Represent one optimized pathway allocation and its audited bounds."""

    extracellular_rates: FloatArray = None
    constraint_limits: Mapping[str, float] = field(default_factory=dict)

    @property
    def status(self):
        return self.solver_status

    @property
    def message(self):
        return self.solver_message


class ConfiguredPathwayModel:
    """Solve an arbitrary input-declared pathway LP without biological identities."""

    def __init__(self, definition: PathwayModelDefinition, *, exchange_scale=1.):
        self.definition = definition
        self.state_names = definition.state_ids
        self.pathway_ids = definition.pathway_ids
        self.source_matrix = definition.exchange_matrix
        self.growth_coefficients = definition.growth_coefficients
        self.depletion_tolerance = definition.depletion_tolerance
        if not np.isfinite(exchange_scale) or exchange_scale <= 0.:
            raise ValueError('exchange_scale must be finite and positive')
        self.exchange_scale = float(exchange_scale)

    def _limit(self, rule, concentrations, biomass):
        """Evaluate one registered uptake-capacity equation."""
        parameters = self.definition.parameters
        concentration = float(concentrations[rule["species"]])
        # Inventory constraints account for stocks and actual reactor supply;
        # only concentration-dependent kinetics vanish at zero concentration.
        concentration = max(concentration, 0.0)
        maximum = parameters[rule["maximum_parameter"]]
        if rule["type"] == "constant":
            return maximum
        if rule["type"] == "monod":
            half = parameters[rule["half_saturation_parameter"]]
            if maximum <= 0.0 or half <= 0.0:
                raise ValueError("Monod parameters must be positive.")
            if concentration <= self.depletion_tolerance:
                return 0.0
            return maximum * concentration / (half + concentration)
        if biomass <= 0.0:
            raise ValueError("transfer-cap uptake requires positive biomass concentration.")
        mass_transfer = parameters[rule["mass_transfer_parameter"]]
        saturation = parameters[rule["saturation_parameter"]]
        if maximum <= 0.0 or mass_transfer < 0.0 or saturation < 0.0:
            raise ValueError("transfer-cap parameters have invalid signs.")
        return min(maximum, mass_transfer * saturation / biomass / self.exchange_scale)

    def solve_fluxes(self, concentrations, biomass=None, *, availability=None, uptake_limits=None):
        """Maximize the declared objective under configured uptake and pathway bounds."""
        if isinstance(concentrations, Mapping):
            if set(concentrations) != set(self.state_names):
                raise ValueError("concentration identifiers must exactly match state_ids.")
            state = {name: float(concentrations[name]) for name in self.state_names}
        else:
            values = _readonly_vector(
                concentrations, "concentrations", len(self.state_names)
            )
            state = dict(zip(self.state_names, values))
        if not np.isfinite(list(state.values())).all():
            raise ValueError("concentrations must be finite.")
        biomass_value = float(biomass) if biomass is not None else np.nan
        if not np.isfinite(biomass_value) or biomass_value <= 0.0:
            raise ValueError("biomass must be finite and positive.")

        rows = []
        limits = []
        named_limits = {}
        state_index = {name: index for index, name in enumerate(self.state_names)}
        expected = {r['output'] for r in self.definition.uptake_constraints if r['type'] == 'provider'}
        supplied = dict(uptake_limits or {})
        if set(supplied) != expected or any(not np.isfinite(v) or v < 0. for v in supplied.values()):
            raise ValueError('provider uptake limits must match declared outputs and be finite nonnegative capacities')
        for rule in self.definition.uptake_constraints:
            limit = (supplied[rule['output']] if rule['type'] == 'provider'
                     else self._limit(rule, state, biomass_value))
            rows.append(-self.source_matrix[state_index[rule["species"]], :])
            limits.append(limit)
            named_limits[rule["identifier"]] = float(limit)

        if availability is None:
            # Without a reactor supply map, exhausted pools cannot supply flux.
            for name, concentration in state.items():
                if concentration <= 0.:
                    rows.append(-self.source_matrix[state_index[name]])
                    limits.append(0.)
        else:
            matrix, capacity = availability.linear_relaxation(None, None)
            rows.extend(matrix)
            limits.extend(capacity)

        lower = np.zeros(len(self.pathway_ids), dtype=float)
        upper = np.full(len(self.pathway_ids), np.inf, dtype=float)
        pathway_index = {name: index for index, name in enumerate(self.pathway_ids)}
        for rule in self.definition.pathway_bound_rules:
            if state[rule["species"]] <= self.depletion_tolerance:
                for pathway in rule["pathways"]:
                    upper[pathway_index[pathway]] = 0.0
        bounds = tuple(
            (float(lo), None if np.isinf(hi) else float(hi))
            for lo, hi in zip(lower, upper)
        )
        a_ub = np.asarray(rows, dtype=float) if rows else None
        b_ub = np.asarray(limits, dtype=float) if limits else None
        # Solve more accurately than the final 1e-8 feasibility audit, including
        # each secondary solve that fixes an optimal pathway allocation.
        solver_options = {"primal_feasibility_tolerance": 1e-9,
                          "dual_feasibility_tolerance": 1e-9}
        result = linprog(
            c=-self.definition.objective_coefficients,
            A_ub=a_ub, b_ub=b_ub, bounds=bounds, method="highs", options=solver_options,
        )
        if not result.success:
            if result.status == 2:
                raise MetabolicInfeasibleError(
                    result.message, network_id=self.definition.problem_name,
                    lower_bounds=lower, upper_bounds=upper, status=result.status,
                    environment_context=state)
            raise MetabolicNumericalError(
                f"Pathway LP failed for {self.definition.problem_name}: "
                f"status={result.status}, message={result.message}"
            )

        optimum = float(self.definition.objective_coefficients @ result.x)
        deterministic_bounds = list(bounds)
        deterministic_result = result
        for index in sorted(range(len(self.pathway_ids)), key=self.pathway_ids.__getitem__):
            pathway = self.pathway_ids[index]
            objective = np.zeros(len(self.pathway_ids), dtype=float)
            objective[index] = 1.0
            deterministic_result = linprog(
                c=objective, A_ub=a_ub, b_ub=b_ub,
                A_eq=self.definition.objective_coefficients[np.newaxis, :],
                b_eq=np.array([optimum]), bounds=deterministic_bounds, method="highs",
                options=solver_options,
            )
            if not deterministic_result.success:
                raise MetabolicNumericalError(f"Pathway LP tie-break failed for {pathway!r}.")
            fixed = float(deterministic_result.x[index])
            if abs(fixed) <= 1.0e-10:
                fixed = 0.0
            deterministic_bounds[index] = (fixed, fixed)

        fluxes = _readonly_vector(
            deterministic_result.x, "optimized fluxes", len(self.pathway_ids)
        )
        if a_ub is not None and np.any(a_ub @ fluxes - b_ub > 1.0e-8):
            raise MetabolicNumericalError(
                "HiGHS returned a flux vector outside configured constraints."
            )
        rates = _readonly_vector(
            self.source_matrix @ fluxes, "extracellular rates", len(self.state_names)
        )
        lower.setflags(write=False)
        upper.setflags(write=False)
        return PathwayOptimizationResult(
            reaction_ids=self.pathway_ids, fluxes=fluxes,
            primary_objective=float(self.definition.objective_coefficients @ fluxes),
            secondary_objective=None, growth_rate=float(self.growth_coefficients @ fluxes),
            lower_bounds=lower, upper_bounds=upper,
            active_lower=tuple(name for name, value in zip(self.pathway_ids, fluxes) if abs(value) <= 1e-8),
            active_upper=tuple(name for name, value, bound in zip(self.pathway_ids, fluxes, upper)
                               if abs(value - bound) <= 1e-8),
            mass_balance_residual_inf=None,
            bound_violation_inf=float(max(0., np.max(lower - fluxes), np.max(fluxes - upper),
                                          np.max(a_ub @ fluxes - b_ub) if a_ub is not None else 0.)),
            primal_status='optimal', solver_status=int(result.status),
            solver_message=str(result.message), closure_policy_id='highs:pathway-lp:lexicographic-id',
            cache_hit=False, extracellular_rates=rates, constraint_limits=MappingProxyType(named_limits),
            inventory_violation_inf=(None if availability is None else
                                     float(max(0., -np.min(availability.residual(fluxes))))),
        )

    def audit_closed_uptake(self):
        """Screen for growth or secretion without net uptake; not elemental closure."""
        count = len(self.pathway_ids)
        matrix = np.vstack((-self.source_matrix, np.ones(count)))
        capacity = np.r_[np.zeros(len(self.state_names)), 1.]
        maxima = {}
        for name, objective in [('growth', self.growth_coefficients),
                                *zip(self.state_names, self.source_matrix)]:
            result = linprog(-objective, A_ub=matrix, b_ub=capacity,
                             bounds=(0., None), method='highs')
            if not result.success:
                return dict(status='UNRESOLVED', reason=result.message)
            maxima[name] = max(0., float(-result.fun))
        return dict(status='FAIL' if max(maxima.values()) > 1e-7 else 'PASS',
                    maximum_exports=maxima, scope='Normalized total pathway activity <= 1; no net uptake.')


def _readonly_vector(values, name, length=None):
    """Return one finite immutable vector with an optional required length."""
    result = np.asarray(values, dtype=float)
    if result.ndim != 1 or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite one-dimensional sequence.")
    if length is not None and len(result) != length:
        raise ValueError(f"{name} must contain {length} values; got {len(result)}.")
    result = result.copy()
    result.setflags(write=False)
    return result


def _readonly_matrix(values, name, shape=None):
    """Return one finite immutable matrix with an optional required shape."""
    result = np.asarray(values, dtype=float)
    if result.ndim != 2 or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite two-dimensional array.")
    if shape is not None and result.shape != shape:
        raise ValueError(f"{name} must have shape {shape}; got {result.shape}.")
    result = result.copy()
    result.setflags(write=False)
    return result
