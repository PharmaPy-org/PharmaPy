"""General model-based experiment design for native bioreactor simulations.

The declarations define an immutable design problem. Evaluation computes
finite-difference sensitivities and covariance-weighted Fisher information,
ranks feasible candidates deterministically, and independently replays the
selected design.
"""

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from ..Metabolic.closures.base import MetabolicNumericalError


@dataclass(frozen=True)
class DesignParameter:
    """Declare one uncertain parameter and its finite-difference scale."""

    name: str
    nominal: float
    lower_bound: float
    upper_bound: float
    relative_step: float = 1e-4

    def __post_init__(self):
        values = (
            self.nominal,
            self.lower_bound,
            self.upper_bound,
            self.relative_step,
        )
        if not self.name or not np.isfinite(values).all():
            raise ValueError("design parameters require a name and finite values")
        if (
            self.lower_bound >= self.upper_bound
            or not self.lower_bound <= self.nominal <= self.upper_bound
            or self.relative_step <= 0.0
        ):
            raise ValueError("design parameter bounds, nominal value, or step is invalid")


@dataclass(frozen=True)
class DesignCandidate:
    """Declare one identified set of experimental controls and its cost."""

    identifier: str
    values: Mapping[str, float]
    cost: float = 0.0

    def __post_init__(self):
        values = {str(name): float(value) for name, value in self.values.items()}
        if (not self.identifier or not values or
                not np.isfinite(list(values.values())).all() or
                not np.isfinite(self.cost) or self.cost < 0.0):
            raise ValueError(
                "design candidates require identity, finite values, and "
                "nonnegative cost"
            )
        object.__setattr__(self, "values", MappingProxyType(values))


@dataclass(frozen=True)
class DesignMeasurement:
    """Declare one measured output with timing, uncertainty context, and cost."""

    name: str
    unit: str
    sample_time: float
    availability_time: float
    cost: float = 0.0

    def __post_init__(self):
        values = (self.sample_time, self.availability_time, self.cost)
        if not self.name or not self.unit or not np.isfinite(values).all():
            raise ValueError("design measurements require name, unit, and finite values")
        if self.sample_time < 0.0 or self.availability_time < self.sample_time or self.cost < 0.0:
            raise ValueError("measurement timing must be causal and cost nonnegative")


@dataclass(frozen=True)
class BioreactorDesignProblem:
    """Declare candidates and optional independent baseline information in parameter order."""

    candidates: Sequence[DesignCandidate]
    parameters: Sequence[DesignParameter]
    measurements: Sequence[DesignMeasurement]
    covariance: Sequence[Sequence[float]]
    variable_bounds: Mapping[str, Sequence[float]]
    baseline_id: str
    criterion: str = "D"
    maximum_cost: float = np.inf
    baseline_information: object = None

    def __post_init__(self):
        candidates = tuple(self.candidates)
        parameters = tuple(self.parameters)
        measurements = tuple(self.measurements)
        covariance = np.asarray(self.covariance, dtype=float)
        bounds = {str(name): tuple(map(float, limits))
                  for name, limits in self.variable_bounds.items()}
        if not candidates or len({item.identifier for item in candidates}) != len(candidates):
            raise ValueError("candidate identifiers must be nonempty and unique")
        if not parameters or len({item.name for item in parameters}) != len(parameters):
            raise ValueError("design parameter names must be nonempty and unique")
        if not measurements or len({item.name for item in measurements}) != len(measurements):
            raise ValueError("measurement names must be nonempty and unique")
        expected = (len(measurements), len(measurements))
        if (covariance.shape != expected or not np.isfinite(covariance).all() or
                not np.allclose(covariance, covariance.T) or
                np.min(np.linalg.eigvalsh(covariance)) <= 0.0):
            raise ValueError(
                "measurement covariance must be finite, symmetric, and "
                "positive definite"
            )
        if not bounds or any(len(limits) != 2 or limits[0] > limits[1]
                             for limits in bounds.values()):
            raise ValueError("each design variable requires ordered bounds")
        expected_variables = set(bounds)
        for candidate in candidates:
            if set(candidate.values) != expected_variables:
                raise ValueError("every candidate must declare exactly the design variables")
            if any(not bounds[name][0] <= value <= bounds[name][1]
                   for name, value in candidate.values.items()):
                raise ValueError("candidate value lies outside its declared bounds")
        if self.baseline_id not in {item.identifier for item in candidates}:
            raise ValueError("baseline_id must identify a declared candidate")
        if self.criterion not in {"D", "A", "E"}:
            raise ValueError("criterion must be D, A, or E")
        if np.isnan(self.maximum_cost) or self.maximum_cost < 0.0:
            raise ValueError("maximum_cost must be nonnegative")
        covariance.setflags(write=False)
        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(self, "parameters", parameters)
        object.__setattr__(self, "measurements", measurements)
        object.__setattr__(self, "covariance", covariance)
        object.__setattr__(self, "variable_bounds", MappingProxyType(bounds))
        if self.baseline_information is not None:
            prior = np.array(self.baseline_information, dtype=float, copy=True)
            if (prior.shape != (len(parameters), len(parameters))
                    or not np.isfinite(prior).all() or not np.allclose(prior, prior.T)
                    or np.min(np.linalg.eigvalsh(prior)) < 0.0):
                raise ValueError("baseline information must be symmetric positive semidefinite in parameter order")
            prior.setflags(write=False)
            object.__setattr__(self, "baseline_information", prior)

    @property
    def measurement_names(self):
        """Return measured output identifiers in covariance-matrix order."""
        return tuple(item.name for item in self.measurements)


@dataclass(frozen=True)
class DesignEvaluation:
    """Record incremental information; rank, condition, and score include any baseline."""

    identifier: str
    values: Mapping[str, float]
    feasible: bool
    reason: str
    cost: float
    predictions: Mapping[str, float]
    sensitivities: np.ndarray
    information_matrix: np.ndarray
    rank: int
    condition: float
    score: float

    def __post_init__(self):
        sensitivity = np.asarray(self.sensitivities, dtype=float).copy()
        information = np.asarray(self.information_matrix, dtype=float).copy()
        sensitivity.setflags(write=False)
        information.setflags(write=False)
        object.__setattr__(self, "values", MappingProxyType(dict(self.values)))
        object.__setattr__(self, "predictions", MappingProxyType(dict(self.predictions)))
        object.__setattr__(self, "sensitivities", sensitivity)
        object.__setattr__(self, "information_matrix", information)


@dataclass(frozen=True)
class BioreactorDesignResult:
    """Return evaluated candidates, ranking, and independent replay evidence."""

    evaluations: tuple
    ranking: tuple
    selected_id: str
    baseline_id: str
    baseline_score: float
    selected_score: float
    score_improvement: float
    replayed: bool


def _outputs(simulator, candidate, parameters, measurements):
    measurement_names = tuple(item.name for item in measurements)
    result = simulator(dict(candidate.values), parameters, measurements)
    if not isinstance(result, Mapping) or set(result) != set(measurement_names):
        raise ValueError("simulator outputs must exactly match measurement names")
    values = np.asarray([result[name] for name in measurement_names], dtype=float)
    if values.shape != (len(measurement_names),) or not np.isfinite(values).all():
        raise ValueError("simulator outputs must be finite scalars")
    return values


def _criterion(information, criterion, rank):
    dimension = len(information)
    if rank < dimension:
        return -np.inf
    eigenvalues = np.linalg.eigvalsh(information)
    if criterion == "D":
        return float(np.sum(np.log(eigenvalues)))
    if criterion == "E":
        return float(eigenvalues[0])
    return float(-np.trace(np.linalg.inv(information)))


def _evaluate(problem, candidate, simulator, feasibility):
    empty_sensitivity = np.empty((len(problem.measurement_names), len(problem.parameters)))
    empty_information = np.empty((len(problem.parameters), len(problem.parameters)))
    total_cost = candidate.cost + sum(item.cost for item in problem.measurements)
    if total_cost > problem.maximum_cost:
        return DesignEvaluation(candidate.identifier, candidate.values, False,
                                "candidate exceeds maximum cost", total_cost,
                                {}, empty_sensitivity, empty_information, 0, np.inf, -np.inf)
    if feasibility is not None:
        allowed, reason = feasibility(dict(candidate.values))
        if not allowed:
            return DesignEvaluation(candidate.identifier, candidate.values, False,
                                    str(reason), total_cost, {}, empty_sensitivity,
                                    empty_information, 0, np.inf, -np.inf)
    nominal = {item.name: item.nominal for item in problem.parameters}
    try:
        predictions = _outputs(simulator, candidate, nominal, problem.measurements)
        sensitivity = np.empty((len(predictions), len(problem.parameters)))
        for column, parameter in enumerate(problem.parameters):
            step = parameter.relative_step * max(abs(parameter.nominal), 1.0)
            low = max(parameter.lower_bound, parameter.nominal - step)
            high = min(parameter.upper_bound, parameter.nominal + step)
            if high <= low:
                raise ValueError(f"no valid perturbation for parameter {parameter.name!r}")
            low_parameters = dict(nominal)
            low_parameters[parameter.name] = low
            high_parameters = dict(nominal)
            high_parameters[parameter.name] = high
            sensitivity[:, column] = (
                _outputs(simulator, candidate, high_parameters, problem.measurements)
                - _outputs(simulator, candidate, low_parameters, problem.measurements)
            ) / (high - low)
        information = sensitivity.T @ np.linalg.inv(problem.covariance) @ sensitivity
        total = information if problem.baseline_information is None else information + problem.baseline_information
        rank = int(np.linalg.matrix_rank(total))
        condition = float(np.linalg.cond(total)) if rank == len(total) else np.inf
        score = _criterion(total, problem.criterion, rank)
        if not np.isfinite(score):
            return DesignEvaluation(candidate.identifier, candidate.values, False,
                                    "information matrix is rank deficient", total_cost,
                                    dict(zip(problem.measurement_names, predictions)),
                                    sensitivity, information, rank, condition, score)
        return DesignEvaluation(candidate.identifier, candidate.values, True, "",
                                total_cost,
                                MappingProxyType(dict(zip(
                                    problem.measurement_names, predictions
                                ))),
                                sensitivity, information, rank, condition, score)
    except MetabolicNumericalError:
        raise
    except Exception as error:
        return DesignEvaluation(candidate.identifier, candidate.values, False,
                                f"{type(error).__name__}: {error}", total_cost,
                                {}, empty_sensitivity, empty_information, 0, np.inf, -np.inf)


def evaluate_design_candidate(problem, candidate, simulator, *, feasibility=None):
    """Evaluate one declared candidate without ranking or replaying it."""
    if not isinstance(problem, BioreactorDesignProblem):
        raise TypeError("problem must be a BioreactorDesignProblem")
    if not isinstance(simulator, Callable):
        raise TypeError("simulator must be callable")
    return _evaluate(problem, candidate, simulator, feasibility)


def evaluate_bioreactor_design(problem, simulator, *, feasibility=None):
    """Rank declared candidates and independently replay the accepted design."""
    if not isinstance(problem, BioreactorDesignProblem):
        raise TypeError("problem must be a BioreactorDesignProblem")
    if not isinstance(simulator, Callable):
        raise TypeError("simulator must be callable")
    evaluations = tuple(_evaluate(problem, item, simulator, feasibility)
                        for item in problem.candidates)
    feasible = [item for item in evaluations if item.feasible]
    if not feasible:
        raise RuntimeError("no feasible and identifiable design candidates")
    ranked = sorted(feasible, key=lambda item: (-item.score, item.cost, item.identifier))
    selected = ranked[0]
    baseline = next(item for item in evaluations if item.identifier == problem.baseline_id)
    if not baseline.feasible:
        raise RuntimeError("declared baseline design is not feasible and identifiable")
    candidate = next(item for item in problem.candidates
                     if item.identifier == selected.identifier)
    replay = _evaluate(problem, candidate, simulator, feasibility)
    replayed = bool(replay.feasible and np.isclose(replay.score, selected.score,
                                                   rtol=1e-10, atol=1e-12))
    return BioreactorDesignResult(
        evaluations=evaluations,
        ranking=tuple(item.identifier for item in ranked),
        selected_id=selected.identifier,
        baseline_id=baseline.identifier,
        baseline_score=baseline.score,
        selected_score=replay.score,
        score_improvement=replay.score - baseline.score,
        replayed=replayed,
    )


def replace_declared_values(mapping, replacements):
    """Return a deep copy with values replaced at declared dotted paths."""
    result = deepcopy(mapping)
    for path, value in replacements.items():
        tokens = str(path).split(".")
        if not tokens or any(not token for token in tokens):
            raise ValueError("replacement paths must contain nonempty tokens")
        target = result
        for token in tokens[:-1]:
            key = int(token) if isinstance(target, list) else token
            try:
                target = target[key]
            except (KeyError, IndexError, TypeError, ValueError) as error:
                raise ValueError(f"replacement path {path!r} does not exist") from error
        final = int(tokens[-1]) if isinstance(target, list) else tokens[-1]
        try:
            target[final]
        except (KeyError, IndexError, TypeError, ValueError) as error:
            raise ValueError(f"replacement path {path!r} does not exist") from error
        target[final] = value
    return result
