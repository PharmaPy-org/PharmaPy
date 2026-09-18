"""Native PharmaPy parameter-estimation adapter for biological models.

Declared observations and named parameters are mapped to PharmaPy's existing
``ParameterEstimation`` residual, weighting, and sensitivity contract. The
result includes fit, uncertainty, and identifiability diagnostics.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
from scipy.optimize import least_squares

from PharmaPy.ParamEstim import ParameterEstimation


@dataclass(frozen=True)
class ObservationSeries:
    """Represent one named measured quantity with explicit timing and units."""

    name: str
    unit: str
    sample_times: Sequence[float]
    values: Sequence[float]
    availability_times: Sequence[float] = None
    valid: Sequence[bool] = None
    provenance: str = "unspecified"

    def __post_init__(self):
        times = np.asarray(self.sample_times, dtype=float)
        values = np.asarray(self.values, dtype=float)
        available = times if self.availability_times is None else np.asarray(
            self.availability_times, dtype=float
        )
        valid = np.isfinite(values) if self.valid is None else np.asarray(
            self.valid, dtype=bool
        )
        if not self.name or not self.unit or not self.provenance:
            raise ValueError("observations require name, unit, and provenance")
        if any(array.ndim != 1 for array in (times, values, available, valid)):
            raise ValueError("observation fields must be one-dimensional")
        if len({len(times), len(values), len(available), len(valid)}) != 1:
            raise ValueError("observation fields must have equal lengths")
        if len(times) == 0 or not np.isfinite(times).all() or np.any(times < 0.0):
            raise ValueError("sample times must be finite, nonnegative, and nonempty")
        if not np.isfinite(available).all() or np.any(available < times):
            raise ValueError("availability times must be finite and not precede sampling")
        if np.any(valid & ~np.isfinite(values)):
            raise ValueError("valid observations must have finite values")
        if not np.any(valid):
            raise ValueError("each observation series requires a valid value")
        for array in (times, values, available, valid):
            array.setflags(write=False)
        object.__setattr__(self, "sample_times", times)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "availability_times", available)
        object.__setattr__(self, "valid", valid)


@dataclass(frozen=True)
class BioreactorEstimationProblem:
    """Declare named estimands and the observation model used for inference."""

    parameter_names: Sequence[str]
    initial_values: Sequence[float]
    lower_bounds: Sequence[float]
    upper_bounds: Sequence[float]
    observations: Sequence[ObservationSeries]
    covariance: Sequence[Sequence[float]]

    def __post_init__(self):
        names = tuple(str(name) for name in self.parameter_names)
        initial = np.asarray(self.initial_values, dtype=float)
        lower = np.asarray(self.lower_bounds, dtype=float)
        upper = np.asarray(self.upper_bounds, dtype=float)
        observations = tuple(self.observations)
        covariance = np.asarray(self.covariance, dtype=float)
        if not names or any(not name for name in names) or len(set(names)) != len(names):
            raise ValueError("parameter names must be nonempty and unique")
        if any(array.shape != (len(names),) for array in (initial, lower, upper)):
            raise ValueError("parameter values and bounds must match parameter names")
        if not np.isfinite(np.r_[initial, lower, upper]).all():
            raise ValueError("parameter values and bounds must be finite")
        if np.any(lower >= upper) or np.any(initial < lower) or np.any(initial > upper):
            raise ValueError("parameter bounds must be ordered and contain the initial values")
        if not observations or len({item.name for item in observations}) != len(observations):
            raise ValueError("observation series names must be nonempty and unique")
        expected = (len(observations), len(observations))
        if covariance.shape != expected or not np.isfinite(covariance).all():
            raise ValueError(f"covariance must be a finite matrix with shape {expected}")
        if not np.allclose(covariance, covariance.T):
            raise ValueError("covariance must be symmetric")
        if np.min(np.linalg.eigvalsh(covariance)) <= 0.0:
            raise ValueError("covariance must be positive definite")
        for array in (initial, lower, upper, covariance):
            array.setflags(write=False)
        object.__setattr__(self, "parameter_names", names)
        object.__setattr__(self, "initial_values", initial)
        object.__setattr__(self, "lower_bounds", lower)
        object.__setattr__(self, "upper_bounds", upper)
        object.__setattr__(self, "observations", observations)
        object.__setattr__(self, "covariance", covariance)


@dataclass(frozen=True)
class BioreactorEstimationResult:
    """Report estimates, fit quality, uncertainty, and identifiability."""

    estimates: Mapping[str, float]
    parameter_bounds: Mapping[str, tuple]
    objective: float
    residual_norm: float
    covariance: np.ndarray
    standard_errors: Mapping[str, float]
    jacobian_rank: int
    jacobian_condition: float
    identifiable: bool
    success: bool
    message: str
    evaluations: int

    def __post_init__(self):
        covariance = np.asarray(self.covariance, dtype=float).copy()
        covariance.setflags(write=False)
        object.__setattr__(self, "estimates", MappingProxyType(dict(self.estimates)))
        object.__setattr__(self, "parameter_bounds", MappingProxyType(dict(self.parameter_bounds)))
        object.__setattr__(self, "standard_errors", MappingProxyType(dict(self.standard_errors)))
        object.__setattr__(self, "covariance", covariance)


def estimate_bioreactor_parameters(problem, model, *, finite_difference_step=None):
    """Estimate declared parameters through PharmaPy's weighted residual model."""
    if not isinstance(problem, BioreactorEstimationProblem):
        raise TypeError("problem must be a BioreactorEstimationProblem")
    if not isinstance(model, Callable):
        raise TypeError("model must be callable")
    sample_times = [item.sample_times[item.valid] for item in problem.observations]
    measured = [item.values[item.valid] for item in problem.observations]
    output_names = tuple(item.name for item in problem.observations)

    def native_model(parameter_values, times):
        parameters = dict(zip(problem.parameter_names, map(float, parameter_values)))
        predicted = model(parameters, np.asarray(times, dtype=float))
        if isinstance(predicted, Mapping):
            if set(predicted) != set(output_names):
                raise ValueError("model outputs must exactly match observation names")
            array = np.column_stack([predicted[name] for name in output_names])
        else:
            array = np.asarray(predicted, dtype=float)
            if array.ndim == 1:
                array = array[:, None]
        if array.shape != (len(times), len(output_names)):
            raise ValueError("model output shape does not match requested times and observations")
        if not np.isfinite(array).all():
            raise ValueError("model returned nonfinite predictions")
        return array

    estimator = ParameterEstimation(
        native_model,
        problem.initial_values,
        x_data=[sample_times],
        y_data=[measured],
        weight_matrix=problem.covariance,
        name_params=list(problem.parameter_names),
        name_states=list(output_names),
    )
    estimator.optimize_flag = True
    optimization = least_squares(
        lambda values: estimator.get_objective(values, out_array=True),
        problem.initial_values,
        bounds=(problem.lower_bounds, problem.upper_bounds),
        diff_step=finite_difference_step,
    )
    residual = optimization.fun
    jacobian = np.asarray(optimization.jac, dtype=float)
    rank = int(np.linalg.matrix_rank(jacobian))
    singular_values = np.linalg.svd(jacobian, compute_uv=False)
    condition = (float(singular_values[0] / singular_values[-1])
                 if singular_values.size and singular_values[-1] > 0.0 else np.inf)
    degrees_freedom = len(residual) - len(optimization.x)
    identifiable = rank == len(optimization.x) and degrees_freedom > 0
    if identifiable:
        residual_variance = float(residual @ residual / degrees_freedom)
        covariance = residual_variance * np.linalg.inv(jacobian.T @ jacobian)
        standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    else:
        covariance = np.full((len(optimization.x), len(optimization.x)), np.nan)
        standard_errors = np.full(len(optimization.x), np.nan)
    return BioreactorEstimationResult(
        estimates=dict(zip(problem.parameter_names, map(float, optimization.x))),
        parameter_bounds={
            name: (float(lower), float(upper))
            for name, lower, upper in zip(
                problem.parameter_names, problem.lower_bounds, problem.upper_bounds
            )
        },
        objective=float(optimization.cost),
        residual_norm=float(np.linalg.norm(residual)),
        covariance=covariance,
        standard_errors=dict(zip(problem.parameter_names, map(float, standard_errors))),
        jacobian_rank=rank,
        jacobian_condition=condition,
        identifiable=identifiable,
        success=bool(optimization.success),
        message=str(optimization.message),
        evaluations=int(optimization.nfev),
    )
