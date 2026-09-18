"""Calibrate and compare input-declared experiments using native reactor models.

Experiments have independent errors, a shared observation plan, and named
parameters. Information matrices use absolute measurement covariance; fitted
residual-scaled covariance is reported separately, never inverted as a prior.
"""

from dataclasses import replace

import numpy as np
from scipy.linalg import block_diag

from .construction import build_bioreactor
from .design import (BioreactorDesignProblem, DesignCandidate, DesignMeasurement,
                     DesignParameter, evaluate_bioreactor_design,
                     evaluate_design_candidate, replace_declared_values)
from .estimation import (BioreactorEstimationProblem, ObservationSeries,
                         estimate_bioreactor_parameters)


def extract_observations(histories, times, declarations, species_names, density):
    """Read each observation from its own segment and contemporaneous inventory."""
    if not np.isfinite(density) or density <= 0:
        raise ValueError("observation density must be finite and positive")
    outputs = {item["name"]: [] for item in declarations}
    for time in times:
        matches = [(segment, int(index)) for segment in histories
                   for index in np.flatnonzero(np.isclose(segment.time, time, rtol=0, atol=1e-8))]
        if not matches:
            raise ValueError(f"no recorded observation at {time} seconds")
        for item in declarations:
            side = item["event_side"]
            if side not in {"pre", "post"}:
                raise ValueError("event_side must be pre or post")
            segment, index = matches[0 if side == "pre" else -1]
            if item["quantity"] == "species_concentration":
                j = list(species_names).index(item["species"])
                value = np.asarray(getattr(segment, item["result_attribute"]))[index, j]
            elif item["quantity"] == "state":
                value = np.asarray(getattr(segment, item["result_attribute"]))[index]
            else:
                raise ValueError("unknown observation quantity")
            if item["divide_by_volume"]:
                volume = np.asarray(getattr(segment, item["mass_attribute"]))[index].sum() / density
                if not np.isfinite(volume) or volume <= 0:
                    raise ValueError("observation volume must be finite and positive")
                value /= volume
            value = float(value) * item["multiplier"]
            if not np.isfinite(value):
                raise ValueError("nonfinite observation")
            outputs[item["name"]].append(value)
    return {name: np.asarray(values) for name, values in outputs.items()}


def native_experiment_model(case, mechanism, thermo_path, settings, *, refinement=1.0):
    """Create fresh native reactors; keep fixed meshes and explicit event-side observations."""
    units = {"s": 1., "min": 60., "h": 3600., "day": 86400.}
    parameter_paths = {item["name"]: item["mechanism_path"] for item in settings["parameters"]}

    def model(controls, parameters, times):
        times = np.asarray(times, dtype=float)
        if times.ndim != 1 or not times.size or not np.isfinite(times).all() or np.any(times < 0):
            raise ValueError("measurement times must be finite nonnegative seconds")
        replacements = dict(settings["case_overrides"])
        mechanism_replacements = {parameter_paths[name]: value for name, value in parameters.items()}
        for name, value in controls.items():
            declaration = settings["variable_paths"][name]
            destination = replacements if declaration["document"] == "case" else mechanism_replacements
            if declaration["document"] not in {"case", "mechanism"}:
                raise ValueError("control document must be case or mechanism")
            destination[declaration["path"]] = value
        operation = replace_declared_values(case, replacements)
        definition = replace_declared_values(mechanism, mechanism_replacements)
        runtime = operation["operation"]["runtime"]
        if times.max() > runtime["value"] * units[runtime["unit"]] or times.max() <= 0:
            raise ValueError("requested times exceed runtime or contain no positive time")
        numerics = operation["numerics"]
        if numerics["backend"] == "fixed-step":
            numerics["step"]["value"] *= refinement
        else:
            numerics["relative_tolerance"] *= refinement
            numerics["absolute_tolerance"] *= refinement
        step = numerics["step"]["value"] * units[numerics["step"]["unit"]]
        if numerics["backend"] == "fixed-step" and not np.allclose(times / step, np.rint(times / step), rtol=0, atol=1e-8):
            raise ValueError("fixed-step observations must lie on the declared mesh")
        events = definition["recipes"][operation["recipe"]["name"]]
        event_times = [event["time"] * units[numerics["step"]["unit"]] for event in events]
        end = float(times.max())
        if any(np.isclose(end, t, rtol=0, atol=1e-8) for t in event_times):
            if any(item["event_side"] == "post" for item in settings["observations"]):
                end += step
        if end > runtime["value"] * units[runtime["unit"]]:
            raise ValueError("post-event endpoint observation needs a recorded following segment")
        operation["operation"]["runtime"] = {"unit": "s", "value": end}
        assembly = build_bioreactor(operation, definition, thermo_path)
        if numerics["backend"] == "fixed-step":
            if any(not np.any(np.isclose(assembly.time_grid, t, rtol=0, atol=1e-8)) for t in times):
                raise ValueError("fixed-step observations must lie on the declared mesh")
        else:
            assembly.time_grid = np.unique(np.r_[assembly.time_grid, times])
        histories = assembly.solve(verbose=False)
        return extract_observations(histories, times, settings["observations"],
                                    assembly.phase.name_species, settings["observation_density_kg_m3"])
    return model


def _parameter_sd(information):
    """Convert information into local parameter standard deviations."""
    return np.sqrt(np.diag(np.linalg.inv(information)))


def _sd_reduction(selected_sd, control_sd):
    """Measure the geometric-mean relative reduction in parameter uncertainty."""
    return float(1. - np.exp(np.mean(np.log(selected_sd / control_sd))))


def _required_design_checks(profile):
    """Keep capability gates mandatory and optionally require observed error benefit."""
    required = ("replayed", "matched_cost", "predicted_uncertainty_benefit",
                "ranking_and_benefit_stability", "finite_confirmation")
    if profile == "capability":
        return required
    if profile == "predictive-benefit":
        return required + ("realized_prediction_benefit",)
    raise ValueError("acceptance profile must be capability or predictive-benefit")


def run_design_study(settings, model, refined_model):
    """Fit baseline data, rank experiments, and compare selected/control joint fits."""
    profile = settings["acceptance"].get("profile", "predictive-benefit")
    required_checks = _required_design_checks(profile)
    declarations = settings["parameters"]
    names = [item["name"] for item in declarations]
    observation_names = [item["name"] for item in settings["observations"]]
    covariance = np.asarray(settings["covariance"], dtype=float)
    times = np.asarray(settings["sample_times_s"], dtype=float)
    withheld = np.asarray(settings["withheld_times_s"], dtype=float)
    if (times.ndim != 1 or withheld.ndim != 1 or not times.size or not withheld.size
            or not np.isfinite(np.r_[times, withheld]).all()
            or np.any(np.r_[times, withheld] < 0) or np.any(np.diff(times) <= 0)):
        raise ValueError("sample times must increase and held-out times must be nonempty finite seconds")
    if (covariance.shape != (len(observation_names), len(observation_names))
            or not np.isfinite(covariance).all() or not np.allclose(covariance, covariance.T)
            or np.min(np.linalg.eigvalsh(covariance)) <= 0):
        raise ValueError("measurement covariance must be symmetric positive definite")
    if np.intersect1d(times, withheld).size:
        raise ValueError("withheld times must not overlap calibration/design times")
    sigma = np.sqrt(np.diag(covariance))
    candidates = tuple(DesignCandidate(**item) for item in settings["candidates"])
    baseline = next(item for item in candidates if item.identifier == settings["baseline_id"])
    measurements = tuple(
        DesignMeasurement(name=f"{item['name']}@{t}", unit=item["unit"],
                          sample_time=t, availability_time=t,
                          cost=settings["assay_cost"])
        for item in settings["observations"] for t in times
    )
    # Flatten in output-major order; independent times, correlated simultaneous assays.
    design_covariance = np.kron(covariance, np.eye(len(times)))

    def design_simulator(simulator):
        def evaluate(controls, parameters, requested):
            requested_times = np.unique([item.sample_time for item in requested])
            values = simulator(controls, parameters, requested_times)
            return {f"{name}@{t}": float(values[name][np.searchsorted(requested_times, t)])
                    for name in observation_names for t in requested_times}
        return evaluate

    def fit(experiments, initial_parameters):
        series = tuple(ObservationSeries(f"{i}:{name}", item["unit"], times,
                       data[name], provenance="synthetic independent reactor experiment")
                       for i, (controls, data) in enumerate(experiments)
                       for name, item in zip(observation_names, settings["observations"]))
        problem = BioreactorEstimationProblem(names, [initial_parameters[name] for name in names],
            [item["lower_bound"] for item in declarations],
            [item["upper_bound"] for item in declarations], series,
            block_diag(*[covariance for _ in experiments]))
        def predict(parameters, requested_times):
            predictions = {}
            evaluated = {}
            for i, (controls, _) in enumerate(experiments):
                key = tuple(sorted(controls.items()))
                # Reuse deterministic predictions only within this parameter trial.
                if key not in evaluated:
                    evaluated[key] = model(controls, parameters, requested_times)
                predictions.update({f"{i}:{name}": value
                                    for name, value in evaluated[key].items()})
            return predictions
        result = estimate_bioreactor_parameters(problem, predict,
                                                finite_difference_step=settings["fit_relative_step"])
        if not result.success or not result.identifiable:
            raise RuntimeError("calibration failed or is not locally identifiable")
        return result

    truth = settings["synthetic_data"]["true_parameters"]
    seeds = settings["synthetic_data"]["confirmation_seeds"]
    if len(seeds) < 2 or len(set(seeds)) != len(seeds) or settings["synthetic_data"]["baseline_seed"] in seeds:
        raise ValueError("confirmation requires distinct seeds independent of baseline")
    if set(truth) != set(names):
        raise ValueError("synthetic truth must match declared parameters")
    baseline_exact = model(dict(baseline.values), truth, times)
    def noisy(exact, seed):
        errors = np.random.default_rng(seed).multivariate_normal(np.zeros(len(observation_names)), covariance, len(times))
        return {name: exact[name] + errors[:, j] for j, name in enumerate(observation_names)}
    data = noisy(baseline_exact, settings["synthetic_data"]["baseline_seed"])
    calibrated = fit([(dict(baseline.values), data)], {item["name"]: item["initial_value"] for item in declarations})
    parameters = tuple(DesignParameter(item["name"], calibrated.estimates[item["name"]],
                       item["lower_bound"], item["upper_bound"], item["relative_step"]) for item in declarations)
    problem = BioreactorDesignProblem(candidates, parameters, measurements, design_covariance,
              settings["variable_bounds"], baseline.identifier, settings["criterion"], settings["maximum_cost"])
    def baseline_information(problem, simulator):
        evaluation = evaluate_design_candidate(
            replace(problem, candidates=(baseline,), baseline_information=None),
            baseline, design_simulator(simulator))
        if not evaluation.feasible:
            raise RuntimeError("no feasible and identifiable design candidates")
        return evaluation.information_matrix

    prior = baseline_information(problem, model)
    problem = replace(problem, baseline_information=prior)
    result = evaluate_bioreactor_design(problem, design_simulator(model))
    selected = next(item for item in candidates if item.identifier == result.selected_id)
    evaluations = []
    for item in result.evaluations:
        total = prior + item.information_matrix
        evaluations.append({"identifier": item.identifier, "values": dict(item.values),
            "feasible": item.feasible, "reason": item.reason, "cost": item.cost,
            "predictions": dict(item.predictions), "sensitivities": item.sensitivities.tolist(),
            "information_matrix": item.information_matrix.tolist(), "total_information": total.tolist(),
            "parameter_sd": _parameter_sd(total).tolist() if item.feasible else None,
            "criterion_score": item.score})
    selected_sd = np.asarray(next(item["parameter_sd"] for item in evaluations if item["identifier"] == selected.identifier))
    control_sd = np.asarray(next(item["parameter_sd"] for item in evaluations if item["identifier"] == baseline.identifier))
    sd_reduction = _sd_reduction(selected_sd, control_sd)
    checks = {"replayed": result.replayed,
              "matched_cost": selected.cost == baseline.cost,
              "predicted_uncertainty_benefit": sd_reduction >= settings["acceptance"]["minimum_sd_reduction"]}
    stability = []
    for factor, simulator in ((0.5, model), (2., model), (1., refined_model)):
        changed = replace(problem, parameters=tuple(replace(p, relative_step=p.relative_step * factor) for p in parameters))
        changed = replace(changed, baseline_information=baseline_information(changed, simulator))
        replay = evaluate_bioreactor_design(changed, design_simulator(simulator))
        chosen = next(x for x in replay.evaluations if x.identifier == replay.selected_id)
        control = next(x for x in replay.evaluations if x.identifier == baseline.identifier)
        selected_sd = _parameter_sd(changed.baseline_information + chosen.information_matrix)
        control_sd = _parameter_sd(changed.baseline_information + control.information_matrix)
        benefit = _sd_reduction(selected_sd, control_sd)
        stability.append({"perturbation_factor": factor, "refined_solver": simulator is refined_model,
                          "selected_id": replay.selected_id, "sd_reduction": benefit})
    checks["ranking_and_benefit_stability"] = all(x["selected_id"] == selected.identifier and
        x["sd_reduction"] >= settings["acceptance"]["minimum_sd_reduction"] for x in stability)
    selected_exact = model(dict(selected.values), truth, times)
    target_controls = settings["prediction_controls"]
    target = model(target_controls, truth, withheld)
    def prediction_error(parameters):
        predicted = model(target_controls, parameters, withheld)
        return float(np.mean([np.mean(((predicted[name] - target[name]) / sigma[j]) ** 2)
                              for j, name in enumerate(observation_names)]))
    repeats = []
    for seed in settings["synthetic_data"]["confirmation_seeds"]:
        selected_data = noisy(selected_exact, seed)
        control_data = noisy(baseline_exact, seed)
        selected_fit = fit([(dict(baseline.values), data), (dict(selected.values), selected_data)], calibrated.estimates)
        control_fit = fit([(dict(baseline.values), data), (dict(baseline.values), control_data)], calibrated.estimates)
        repeats.append({"seed": seed, "selected_estimates": dict(selected_fit.estimates),
                        "control_estimates": dict(control_fit.estimates),
                        "selected_prediction_mse": prediction_error(selected_fit.estimates),
                        "control_prediction_mse": prediction_error(control_fit.estimates),
                        "selected_observations": {k: v.tolist() for k, v in selected_data.items()},
                        "control_observations": {k: v.tolist() for k, v in control_data.items()}})
    mean_selected = float(np.mean([x["selected_prediction_mse"] for x in repeats]))
    mean_control = float(np.mean([x["control_prediction_mse"] for x in repeats]))
    baseline_mse = prediction_error(calibrated.estimates)
    paired = np.asarray([x["control_prediction_mse"] - x["selected_prediction_mse"] for x in repeats])
    paired_mean = float(paired.mean())
    paired_error = float(paired.std(ddof=1) / np.sqrt(len(paired)))
    checks["finite_confirmation"] = bool(np.isfinite([baseline_mse, mean_selected, mean_control,
        paired_mean, paired_error,
        *[x[key] for x in repeats for key in ("selected_prediction_mse", "control_prediction_mse")]]).all())
    checks["realized_prediction_benefit"] = mean_selected < mean_control
    capability_passed = all(checks[name] for name in _required_design_checks("capability"))
    empirical_status = ("OBSERVED" if checks["realized_prediction_benefit"] else "NOT_DEMONSTRATED")
    if not checks["finite_confirmation"]:
        empirical_status = "INVALID"
    return {"status": "PASS" if all(checks[name] for name in required_checks) else "FAIL",
        "acceptance_profile": profile, "required_checks": list(required_checks),
        "capability_status": "PASS" if capability_passed else "FAIL",
        "empirical_benefit_status": empirical_status, "checks": checks,
        "criterion": problem.criterion, "baseline_id": baseline.identifier,
        "selected_id": selected.identifier, "ranking": list(result.ranking),
        "baseline_score": result.baseline_score, "selected_score": result.selected_score,
        "score_improvement": result.score_improvement, "replayed": result.replayed,
        "candidate_evaluations": evaluations, "baseline_information": prior.tolist(),
        "calibration": {"estimates": dict(calibrated.estimates), "residual_scaled_covariance": calibrated.covariance.tolist(),
                        "observations": {k: v.tolist() for k, v in data.items()}},
        "sample_times_s": times.tolist(), "withheld_times_s": withheld.tolist(),
        "predicted_sd_reduction_vs_matched_control": sd_reduction,
        "stability": stability, "confirmation": repeats,
        "baseline_prediction_mse": baseline_mse,
        "mean_selected_prediction_mse": mean_selected, "mean_control_prediction_mse": mean_control,
        "paired_prediction_benefit_mean": paired_mean,
        "paired_prediction_benefit_standard_error": paired_error,
        "parameter_order": names, "configuration": settings,
        "baseline_absolute_parameter_sd": _parameter_sd(prior).tolist(),
        "evidence_limit": "Synthetic same-model study; local information and small paired-seed comparison, not independent validation or guaranteed experimental benefit."}
