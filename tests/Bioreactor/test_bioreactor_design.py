import inspect

import numpy as np
import pytest

from PharmaPy.Bioreactors import (
    BioreactorDesignProblem, DesignCandidate, DesignMeasurement,
    DesignParameter, evaluate_bioreactor_design, replace_declared_values,
)
from PharmaPy.Bioreactors import design as design_module


def _problem(candidates, *, parameters=None, covariance=None, baseline="base",
             maximum_cost=np.inf):
    if parameters is None:
        parameters = (DesignParameter("theta", 3.0, 1.0, 5.0),)
    measurements = (
        DesignMeasurement("first", "kg", 1.0, 1.0, 0.25),
        DesignMeasurement("second", "kg", 1.0, 2.0, 0.25),
    )
    if covariance is None:
        covariance = [[1.0, 0.0], [0.0, 4.0]]
    return BioreactorDesignProblem(
        candidates=candidates, parameters=parameters, measurements=measurements,
        covariance=covariance, variable_bounds={"input": [0.0, 3.0]},
        baseline_id=baseline, criterion="D", maximum_cost=maximum_cost,
    )


def _linear_simulator(values, parameters, measurements):
    product = values["input"] * parameters["theta"]
    return {"first": product, "second": 2.0 * product}


def test_hand_calculated_sensitivity_information_and_replay():
    problem = _problem((
        DesignCandidate("base", {"input": 1.0}),
        DesignCandidate("strong", {"input": 2.0}),
    ))
    result = evaluate_bioreactor_design(problem, _linear_simulator)
    strong = next(item for item in result.evaluations if item.identifier == "strong")
    assert strong.sensitivities[:, 0] == pytest.approx([2.0, 4.0])
    assert strong.information_matrix == pytest.approx(np.array([[8.0]]))
    assert strong.score == pytest.approx(np.log(8.0))
    assert result.selected_id == "strong"
    assert result.score_improvement == pytest.approx(np.log(4.0))
    assert result.replayed


def test_correlated_covariance_is_used_in_information_matrix():
    covariance = np.array([[2.0, 0.5], [0.5, 1.0]])
    problem = _problem((DesignCandidate("base", {"input": 1.0}),),
                       covariance=covariance)
    result = evaluate_bioreactor_design(problem, _linear_simulator)
    sensitivity = np.array([[1.0], [2.0]])
    expected = sensitivity.T @ np.linalg.inv(covariance) @ sensitivity
    assert result.evaluations[0].information_matrix == pytest.approx(expected)


def test_rank_deficient_design_is_not_promoted():
    parameters = (
        DesignParameter("theta", 1.0, 0.0, 2.0),
        DesignParameter("phi", 1.0, 0.0, 2.0),
    )
    problem = _problem((DesignCandidate("base", {"input": 1.0}),),
                       parameters=parameters)
    simulator = lambda values, pars, measurements: {
        "first": values["input"] * (pars["theta"] + pars["phi"]),
        "second": 2.0 * values["input"] * (pars["theta"] + pars["phi"]),
    }
    with pytest.raises(RuntimeError, match="no feasible and identifiable"):
        evaluate_bioreactor_design(problem, simulator)


def test_failed_constrained_and_over_budget_candidates_are_excluded():
    candidates = (
        DesignCandidate("base", {"input": 1.0}, cost=0.5),
        DesignCandidate("blocked", {"input": 2.0}, cost=0.5),
        DesignCandidate("expensive", {"input": 3.0}, cost=2.0),
    )
    problem = _problem(candidates, maximum_cost=1.0)
    result = evaluate_bioreactor_design(
        problem, _linear_simulator,
        feasibility=lambda values: (values["input"] != 2.0, "declared constraint"),
    )
    status = {item.identifier: item for item in result.evaluations}
    assert result.selected_id == "base"
    assert not status["blocked"].feasible
    assert status["blocked"].reason == "declared constraint"
    assert not status["expensive"].feasible
    assert "maximum cost" in status["expensive"].reason


def test_simulator_failure_is_recorded_and_candidate_is_not_promoted():
    candidates = (
        DesignCandidate("base", {"input": 1.0}),
        DesignCandidate("failed", {"input": 2.0}),
    )
    def simulator(values, parameters, measurements):
        if values["input"] == 2.0:
            raise RuntimeError("declared process is infeasible")
        return _linear_simulator(values, parameters, measurements)
    result = evaluate_bioreactor_design(_problem(candidates), simulator)
    failed = next(item for item in result.evaluations if item.identifier == "failed")
    assert result.selected_id == "base"
    assert not failed.feasible
    assert "declared process is infeasible" in failed.reason


def test_ranking_is_deterministic_for_tied_candidates():
    candidates = (
        DesignCandidate("zeta", {"input": 1.0}),
        DesignCandidate("base", {"input": 1.0}),
        DesignCandidate("alpha", {"input": 1.0}),
    )
    result = evaluate_bioreactor_design(_problem(candidates), _linear_simulator)
    assert result.ranking == ("alpha", "base", "zeta")


def test_declared_path_replacement_is_copy_safe_and_strict():
    source = {"outer": {"values": [1.0, 2.0]}}
    changed = replace_declared_values(source, {"outer.values.1": 4.0})
    assert changed["outer"]["values"] == [1.0, 4.0]
    assert source["outer"]["values"] == [1.0, 2.0]
    with pytest.raises(ValueError, match="does not exist"):
        replace_declared_values(source, {"outer.missing": 3.0})


def test_invalid_measurement_timing_and_covariance_fail_early():
    with pytest.raises(ValueError, match="causal"):
        DesignMeasurement("first", "kg", 2.0, 1.0)
    candidates = (DesignCandidate("base", {"input": 1.0}),)
    with pytest.raises(ValueError, match="positive definite"):
        _problem(candidates, covariance=[[1.0, 1.0], [1.0, 1.0]])


def test_general_engine_and_example_runner_have_separate_responsibilities():
    source = inspect.getsource(design_module)
    for identity in ("CHO", "Reddy", "pH", "VRC01"):
        assert identity.lower() not in source.lower()


@pytest.fixture
def study_settings():
    return dict(parameters=[dict(name="theta", initial_value=1.5, lower_bound=0.1,
                                 upper_bound=5., relative_step=1e-4)],
        observations=[dict(name="y", unit="kg")], covariance=[[0.01]],
        sample_times_s=[1., 2., 3., 4.], withheld_times_s=[5., 6.],
        candidates=[dict(identifier="base", values={"u": 1.}, cost=0.),
                    dict(identifier="strong", values={"u": 3.}, cost=0.)],
        baseline_id="base", variable_bounds={"u": [0., 4.]},
        criterion="D", maximum_cost=4., assay_cost=1., fit_relative_step=1e-4,
        prediction_controls={"u": 2.}, acceptance=dict(minimum_sd_reduction=0.01),
        synthetic_data=dict(true_parameters={"theta": 2.}, baseline_seed=1701,
                            confirmation_seeds=[2701, 2702, 2703, 2704, 2705]))


@pytest.mark.parametrize("profile", [None, "capability", "predictive-benefit"])
@pytest.mark.parametrize("withheld", ["informative", "flat", "nonfinite"])
def test_design_acceptance_profiles_preserve_empirical_evidence(study_settings, profile, withheld):
    from PharmaPy.Bioreactors.experiments import run_design_study
    if profile is not None:
        study_settings["acceptance"]["profile"] = profile
    def model(controls, parameters, times):
        values = controls["u"] * parameters["theta"] * np.asarray(times)
        if min(times) >= 5 and withheld != "informative":
            values = np.full(len(times), 0. if withheld == "flat" else np.nan)
        return {"y": values}
    report = run_design_study(study_settings, model, model)
    assert report["acceptance_profile"] == (profile or "predictive-benefit")
    assert report["capability_status"] == ("FAIL" if withheld == "nonfinite" else "PASS")
    assert report["empirical_benefit_status"] == {
        "informative": "OBSERVED", "flat": "NOT_DEMONSTRATED", "nonfinite": "INVALID"}[withheld]
    passed = withheld == "informative" or (withheld == "flat" and profile == "capability")
    assert report["status"] == ("PASS" if passed else "FAIL")
    assert report["checks"]["realized_prediction_benefit"] == (withheld == "informative")
    assert all(report["checks"][name] for name in report["required_checks"]) == passed


@pytest.mark.parametrize("profile", ["capability", "predictive-benefit"])
def test_design_profiles_cannot_bypass_numerical_failure_or_refinement(study_settings, profile):
    from PharmaPy.Bioreactors.experiments import run_design_study
    from PharmaPy.Metabolic.closures.base import MetabolicNumericalError
    study_settings["acceptance"]["profile"] = profile
    def failed(*args):
        raise MetabolicNumericalError("unqualified numerical result")
    with pytest.raises(MetabolicNumericalError, match="unqualified"):
        run_design_study(study_settings, failed, failed)
    def model(controls, parameters, times):
        return {"y": controls["u"] * parameters["theta"] * np.asarray(times)}
    def refined(controls, parameters, times):
        return model({"u": 1.}, parameters, times)
    report = run_design_study(study_settings, model, refined)
    assert not report["checks"]["ranking_and_benefit_stability"]
    assert report["status"] == report["capability_status"] == "FAIL"


def test_unknown_acceptance_profile_fails_before_simulation(study_settings):
    from PharmaPy.Bioreactors.experiments import run_design_study
    study_settings["acceptance"]["profile"] = "skip-checks"
    def unexpected(*args):
        pytest.fail("invalid profile reached the model")
    with pytest.raises(ValueError, match="acceptance profile"):
        run_design_study(study_settings, unexpected, unexpected)
