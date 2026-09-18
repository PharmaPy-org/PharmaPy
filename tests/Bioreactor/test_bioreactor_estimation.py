import inspect

import numpy as np
import pytest

from PharmaPy.Bioreactors import (
    BioreactorEstimationProblem,
    ObservationSeries,
    estimate_bioreactor_parameters,
)
from PharmaPy.Bioreactors import estimation as estimation_module
from PharmaPy.ParamEstim import ParameterEstimation


def _problem(observations, *, initial=(1.0,), lower=(0.0,), upper=(5.0,),
             covariance=None, names=("rate",)):
    if covariance is None:
        covariance = np.eye(len(observations))
    return BioreactorEstimationProblem(
        parameter_names=names,
        initial_values=initial,
        lower_bounds=lower,
        upper_bounds=upper,
        observations=observations,
        covariance=covariance,
    )


def test_adapter_uses_native_parameter_estimation_contract():
    source = inspect.getsource(estimation_module.estimate_bioreactor_parameters)
    assert "ParameterEstimation(" in source
    assert estimation_module.ParameterEstimation is ParameterEstimation


def test_asynchronous_correlated_observations_recover_known_truth():
    observations = (
        ObservationSeries(
            "rate", "kg/s", [1.0, 2.0, 4.0], [3.0, 6.0, 12.0],
            availability_times=[1.2, 2.2, 4.2], provenance="synthetic truth",
        ),
        ObservationSeries(
            "product", "kg", [1.5, 3.0], [9.0, 18.0],
            availability_times=[2.0, 4.0], provenance="synthetic truth",
        ),
    )
    problem = _problem(observations, covariance=[[1.0, 0.2], [0.2, 2.0]])

    def model(parameters, times):
        theta = parameters["rate"]
        return {"rate": theta * times, "product": 2.0 * theta * times}

    result = estimate_bioreactor_parameters(problem, model)
    assert result.success and result.identifiable
    assert result.jacobian_rank == 1
    assert result.estimates["rate"] == pytest.approx(3.0, abs=1e-8)
    assert result.residual_norm < 1e-8


def test_invalid_or_unavailable_observations_fail_before_estimation():
    with pytest.raises(ValueError, match="not precede sampling"):
        ObservationSeries(
            "rate", "1/s", [2.0], [1.0], availability_times=[1.0],
            provenance="invalid fixture",
        )
    with pytest.raises(ValueError, match="valid observations"):
        ObservationSeries(
            "rate", "1/s", [1.0], [np.nan], valid=[True],
            provenance="invalid fixture",
        )


def test_invalid_covariance_and_parameter_bounds_fail_explicitly():
    observation = ObservationSeries(
        "rate", "1/s", [1.0, 2.0], [1.0, 2.0], provenance="fixture"
    )
    with pytest.raises(ValueError, match="positive definite"):
        _problem((observation,), covariance=[[0.0]])
    with pytest.raises(ValueError, match="contain the initial"):
        _problem((observation,), initial=(6.0,))


def test_bounds_are_enforced_and_nonidentifiability_is_reported():
    observation = ObservationSeries(
        "rate", "1/s", [1.0, 2.0], [10.0, 20.0], provenance="fixture"
    )
    bounded = estimate_bioreactor_parameters(
        _problem((observation,), upper=(2.0,)),
        lambda parameters, times: {"rate": parameters["rate"] * times},
    )
    assert bounded.estimates["rate"] == pytest.approx(2.0)
    flat = estimate_bioreactor_parameters(
        _problem((observation,)),
        lambda parameters, times: {"rate": np.zeros_like(times)},
    )
    assert not flat.identifiable
    assert flat.jacobian_rank == 0
    assert np.isnan(flat.covariance).all()


def test_incompatible_or_nonfinite_model_outputs_fail():
    observation = ObservationSeries(
        "rate", "1/s", [1.0, 2.0], [1.0, 2.0], provenance="fixture"
    )
    problem = _problem((observation,))
    with pytest.raises(ValueError, match="exactly match"):
        estimate_bioreactor_parameters(
            problem, lambda parameters, times: {"wrong": times}
        )
    with pytest.raises(ValueError, match="nonfinite"):
        estimate_bioreactor_parameters(
            problem, lambda parameters, times: {"rate": times * np.nan}
        )
