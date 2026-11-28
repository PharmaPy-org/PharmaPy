"""Fit the selected December caffeine case through the shared WP1 interfaces.

This first pass fits the growth prefactor and order to the two identified
ReactIR runs. Secondary nucleation follows the case starting values. The
untouched third run supplies a limited validation trace. See the caffeine
README for the interpretation and a command that writes and replays the
calibration record.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from functools import partial
from hashlib import sha256
import json
from pathlib import Path
from typing import Mapping

import numpy as np
from scipy.optimize import Bounds

from PharmaPy import PATCalibration, SlurryMeasurements
from PharmaPy.ExperimentDesign import CandidateExperiment, ExperimentDesignReport, rank_experiments
from PharmaPy.ParamEstim import Experiment, Measurement, ParameterEstimation
from PharmaPy.Records import CalibrationRecord, DataSource, ModelRevision, Transformation
from PharmaPy.StatsModule import IdentifiabilityReport
from PharmaPy.jac_module import DEFAULT_OUTPUT_RTOL, numerical_jac_central

from . import batch_cooling, calibrate_band, dataset


CASE = Path(__file__).with_name("caffeine")
# Run roles and final temperature from the case-selection note, sections 2 and 5.
FIT_RUNS = ("high_seed_slow_cool", "low_seed_fast_cool")
VALIDATION_RUN = "low_seed_slow_cool"
FINAL_TEMPERATURE = 293.15  # [K], 20 C target of the final cooling ramp
PARAMETER_UNITS = {"k_g": "um/s", "g": "1"}
# The growth prefactor is nonnegative and growth does not increase as
# supersaturation approaches zero, so the growth order is nonnegative.
GROWTH_BOUNDS = Bounds(np.zeros(len(PARAMETER_UNITS)), np.full(len(PARAMETER_UNITS), np.inf))
# Reduce the standard CVode relative tolerance by two orders so model error
# is small when differentiating a trajectory. Use a correspondingly smaller
# absolute tolerance for the liquid volume (order 1e-4 m**3). Other states
# are orders of magnitude larger in their own solver units.
CVODE_RTOL = DEFAULT_OUTPUT_RTOL / 100  # [-]
CVODE_ATOL = CVODE_RTOL * 1e-4  # [component-dependent solver units]
CVODE_OPTIONS = {"rtol": CVODE_RTOL, "atol": CVODE_ATOL}
# Balance second-order truncation and trajectory error, as derived in
# ParameterEstimation.check_derivatives.
CENTRAL_RELATIVE_STEP = CVODE_RTOL ** (1.0 / 3.0)  # [-]
# The original simulation verifies this relative inventory tolerance on both
# temperature controls in tests/test_pat_batch_cooling_models.py.
BALANCE_RTOL = 1e-6  # [-]
# Compare the discrete designs at the same budget, two observations per trial.
# Each pair divides the logged spectrum times into three successive windows.
PLAN_NAMES = ("early", "middle", "late")
SPECTRA_PER_PLAN = 2
# Same-source deterministic replay compares fitted parameters at the default
# SciPy relative parameter stopping tolerance. Both estimates are nonzero.
REPLAY_RTOL = 1e-8  # [-]


@dataclass(frozen=True)
class CaffeineRun:
    """One measured cooling and IR trace in the selected case.

    Attributes
    ----------
    cooling : batch_cooling.CoolingCase
        Logged temperature and charge used by the simulation.
    kinetics : dict
        Source growth and secondary nucleation parameters in their model units.
    calibration : dict
        Band baseline [-], slope [-] per [kg/kg], slope per kelvin [1/K] per
        [kg/kg], and reference temperature [K].
    observation : Measurement
        Band height [-] on model times [s], one row per measured spectrum
        during the cooling window. Uncertainty is the pre-seeding plateau
        scatter [-] used for relative weighting.
    seeding_time, window_end : float
        Seeding time on the run clock [s] and end of the fitted cooling
        window measured from seeding [s].
    """

    cooling: batch_cooling.CoolingCase
    kinetics: dict
    calibration: dict
    observation: Measurement
    seeding_time: float
    window_end: float


def load_runs(case: Path, treatment: str) -> dict[str, CaffeineRun]:
    """Load the fitting and validation runs in their declared order.

    Parameters
    ----------
    case : pathlib.Path
        Caffeine dataset with run description, band calibration, solubility
        and reduced IR and temperature files.
    treatment : str
        Either ``"common_slope"`` or ``"common_baseline"`` from the case
        selection. The same treatment supplies both the band model and the
        solubility curve.

    Returns
    -------
    dict of str to CaffeineRun
        Two fit runs followed by the held-out run. Spectra are retained on
        their original grid, after seeding through the first logged sample
        at or below the final cooling target.

    Raises
    ------
    ValueError
        If the calibration treatment, cooling window, or repeated spectrum
        timestamps do not permit this analysis.
    """
    if treatment not in calibrate_band.TREATMENTS:
        raise ValueError(f"Choose a band treatment from {calibrate_band.TREATMENTS}.")
    description = dataset.load_description(case)
    calibration = json.loads((case / "band_calibration.json").read_text(encoding="utf-8"))
    settings = description["band_calibration"]
    metadata = {run["id"]: run for run in description["runs"]}
    runs = {}
    for name in (*FIT_RUNS, VALIDATION_RUN):
        cooling, kinetics = batch_cooling.load_case(case, name, treatment)
        seeding_time = metadata[name]["seeding_time_s"]  # [s] on the run clock
        log = dataset.read_reduced(case, name, settings["temperature_table"])
        log_times = log["time"] - seeding_time  # [s] from seeding
        cold = (log_times > 0) & (log[settings["temperature_column"]] <= FINAL_TEMPERATURE)
        if not cold.any():
            raise ValueError(f"{name}: the logged cooling path never reaches the final target.")
        window_end = float(log_times[cold][0])  # [s] from seeding
        run_clock, band_height, _ = calibrate_band.spectra_with_temperature(case, name, settings)
        time = run_clock - seeding_time  # [s] from seeding
        in_window = (time > 0) & (time <= window_end)
        selected_times = time[in_window]  # [s] from seeding
        if selected_times.size == 0 or (np.diff(selected_times) <= 0).any():
            raise ValueError(f"{name}: the cooling window needs strictly increasing spectrum times.")
        run_calibration = calibration["runs"][name]
        # Root mean square of the two dissolved-plateau standard deviations.
        # This defines relative instrument weights before fitting; the report
        # estimates their common residual variance multiplier from the fit.
        plateaus = run_calibration["plateaus"]
        band_scatter = float(np.sqrt(np.mean([
            plateaus[window]["scatter"] ** 2 for window in calibrate_band.DISSOLVED
        ])))  # [-] absorbance
        if not np.isfinite(band_scatter) or band_scatter <= 0:
            raise ValueError(f"{name}: the dissolved plateaus need positive finite band scatter.")
        observation = Measurement(
            0, selected_times, band_height[in_window], uncertainty=band_scatter,
            units="1", basis="1700 cm^-1 absorbance above the baseline",
        )
        coefficients = {
            key: run_calibration[treatment][key]
            for key in ("baseline", "slope", "slope_per_kelvin")
        }  # [-], [-] per [kg/kg], [1/K] per [kg/kg]
        coefficients["reference_temperature"] = calibration["reference_temperature_K"]  # [K]
        runs[name] = CaffeineRun(
            cooling, kinetics, coefficients, observation, seeding_time, window_end
        )
    return runs


def predict_band(
    params: np.ndarray, times: np.ndarray, name: str, *, runs: Mapping[str, CaffeineRun]
) -> np.ndarray:
    """Simulate one run and return band heights in the supplied time order.

    Parameters
    ----------
    params : numpy.ndarray
        Fitted growth prefactor [um/s] and order [-], shape (2,).
    times : numpy.ndarray
        Requested observation times [s] after seeding, shape (n_times,).
    name : str
        Run identifier used by the calibration record.
    runs : mapping of str to CaffeineRun
        Model inputs and calibrated signal coefficients per run.

    Returns
    -------
    numpy.ndarray
        Predicted 1700 cm^-1 IR band height [-], shape (n_times,).

    Raises
    ------
    RuntimeError
        If the solver result fails the caffeine or solvent inventory check.
    """
    run = runs[name]
    growth_prefactor, growth_order = params  # [um/s], [-]
    kinetics = dict(
        run.kinetics,
        growth=(growth_prefactor, run.kinetics["growth"][1], growth_order),
    )  # ([um/s], [J/mol], [-]) for growth
    cooling = run.cooling
    slurry = batch_cooling.simulate(
        cooling, kinetics, times, "legacy", solver_options=CVODE_OPTIONS
    )
    solute_charge = cooling.solute_mass + cooling.seed_mass  # [kg] caffeine
    solute_total = SlurryMeasurements.total_solute(
        slurry, cooling.solute_density, cooling.shape_factor
    )  # [kg] caffeine
    if not (
        np.allclose(solute_total, solute_charge, rtol=BALANCE_RTOL, atol=0)
        and np.allclose(slurry.solvent_mass, cooling.solvent_mass, rtol=BALANCE_RTOL, atol=0)
    ):
        raise RuntimeError(f"{name}: the model exceeded the relative inventory tolerance.")
    ratio = SlurryMeasurements.dissolved_ratio(slurry)  # [kg caffeine/kg water]
    temperature = cooling.temperature(times)  # [K]
    return PATCalibration.band_height_from_ratio(ratio, temperature, **run.calibration)


def capture_record(
    case: Path, revision: str, treatment: str, runs: Mapping[str, CaffeineRun],
    experiments: Mapping[str, Experiment], estimator: ParameterEstimation,
) -> CalibrationRecord:
    """Bind the accepted fit to the observations, source hashes and model.

    Parameters
    ----------
    case : pathlib.Path
        Directory of the selected caffeine dataset.
    revision : str
        Revision of the source tree running this study.
    treatment : str
        Name of the band and solubility treatment applied.
    runs : mapping of str to CaffeineRun
        Fitting and validation run inputs, in source order.
    experiments : mapping of str to Experiment
        Ordered fit observations and callback inputs.
    estimator : ParameterEstimation
        Accepted fit of the recorded experiments.

    Returns
    -------
    CalibrationRecord
        Replayable model, provenance, predictions and fit statistics.
    """
    sources = []
    for name in runs:
        role = "calibration" if name in experiments else "validation"
        for kind in ("ir", "temperature"):
            filename = f"reduced/{name}_{kind}.csv"
            sources.append(DataSource(
                filename, "experimental", "Reduced data shareable in PharmaPy repository",
                uri=f"examples/pat/caffeine/{filename}",
                sha256=sha256((case / filename).read_bytes()).hexdigest(), role=role,
                description="Nagy group (Purdue), October–November 2026",
            ))
    model_inputs = {
        filename: sha256((case / filename).read_bytes()).hexdigest()
        for filename in ("runs.json", "compounds.json", "band_calibration.json", "solubility.json",
                         "reduced_ir.layout.json", "reduced_temperature.layout.json")
    }
    return CalibrationRecord.from_estimation(
        estimator, experiments, case="caffeine_batch_cooling",
        parameter_units=PARAMETER_UNITS,
        model=ModelRevision(
            "examples.pat.caffeine_m6:predict_band", revision=revision,
            description="Seeded moments BatchCryst with measured temperature control",
        ),
        sources=sources,
        transformations=[Transformation(
            "Use band heights as reduced, crop to cooling, subtract the FBRM seeding time, "
            "and weight by the dissolved-plateau scatter.",
            fields=("ir_band",), parameters={
                "treatment": treatment, "source_sha256": model_inputs,
                "seeding_time_s": {name: run.seeding_time for name, run in runs.items()},
                "window_end_s": {name: run.window_end for name, run in runs.items()},
                "fixed_activation_energy_J_mol": runs[FIT_RUNS[0]].kinetics["growth"][1],
                "fixed_secondary_nucleation": {
                    "values": runs[FIT_RUNS[0]].kinetics["secondary_nucleation"],
                    "units": ("1/m^3/s", "J/mol", "1", "1"),
                },
            },
        )],
        model_solver={"name": "Assimulo CVode", "report_continuously": True,
                      **CVODE_OPTIONS, "inventory_rtol": BALANCE_RTOL},
        covariance_include_mse=True,
    )


def compare_sampling(
    fit: IdentifiabilityReport, runs: Mapping[str, CaffeineRun]
) -> tuple[ExperimentDesignReport, dict[str, np.ndarray]]:
    """Compare three equal-cost sampling schedules on the same demonstrated recipe.

    Parameters
    ----------
    fit : IdentifiabilityReport
        Accepted growth prefactor [um/s] and order [-], with a common residual
        variance scale.
    runs : mapping of str to CaffeineRun
        Logged runs, including the recipe used to propose additional samples.

    Returns
    -------
    report : ExperimentDesignReport
        Estimated local information from repeating the high-seed slow
        cooling recipe under each fixed schedule.
    schedules : dict of str to numpy.ndarray
        Sample times [s] after seeding in each plan, shape (2,) per plan.

    Raises
    ------
    ValueError
        If the available observation times cannot populate the schedules.
    """
    name = FIT_RUNS[0]
    observation = runs[name].observation
    times = observation.x  # [s] after seeding
    if times.size < len(PLAN_NAMES) * SPECTRA_PER_PLAN:
        raise ValueError(f"{name}: too few recorded spectra for the sampling comparison.")
    # Use actual acquisition times nearest the ends of each successive
    # third of the recorded spectrum indices. Two samples per plan.
    windows = np.array_split(times, len(PLAN_NAMES))
    schedules = {
        plan: window[[0, -1]] for plan, window in zip(PLAN_NAMES, windows)
    }  # [s] after seeding
    design_times = np.concatenate(tuple(schedules.values()))  # [s] after seeding
    parameters = fit.estimates  # [um/s], [-]
    deltas = CENTRAL_RELATIVE_STEP * np.abs(parameters)  # [um/s], [-]
    derivatives = numerical_jac_central(
        partial(predict_band, runs=runs), parameters, dx=deltas,
        args=(design_times, name), num_nonnegative=len(PARAMETER_UNITS),
    )  # [absorbance / parameter unit]
    covariance = np.eye(SPECTRA_PER_PLAN) * observation.uncertainty[0]**2  # [absorbance**2]

    def supply_sensitivities(point: Mapping[str, float], rows: np.ndarray) -> dict:
        """Map derivatives on selected rows to the accepted parameter coordinates.

        Parameters
        ----------
        point : mapping of str to float
            Named growth prefactor [um/s] and order [-].
        rows : numpy.ndarray
            Raw band sensitivities [absorbance / parameter unit], shape (2, 2).

        Returns
        -------
        dict of str to numpy.ndarray
            Sensitivities for each parameter, ordered by sample time.

        Raises
        ------
        ValueError
            If the requested point differs from the evaluated coordinates.
        """
        if any(point[key] != value for key, value in zip(PARAMETER_UNITS, parameters)):
            raise ValueError("The design must use the accepted fit coordinates.")
        return dict(zip(PARAMETER_UNITS, rows.T))

    candidates = [
        CandidateExperiment(
            plan, partial(supply_sensitivities, rows=rows),
            covariance, relative_noise=True,
        )
        for plan, rows in zip(PLAN_NAMES, np.split(derivatives, len(PLAN_NAMES)))
    ]
    return rank_experiments(fit, candidates, parameter_units=PARAMETER_UNITS), schedules


def fit_caffeine(
    runs: Mapping[str, CaffeineRun],
) -> tuple[ParameterEstimation, dict[str, Experiment]]:
    """Fit the measured IR responses through the shared estimation API.

    Parameters
    ----------
    runs : mapping of str to CaffeineRun
        Prepared case with time [s] and band height [-] on each run's grid.

    Returns
    -------
    estimator : ParameterEstimation
        Fit with growth prefactor [um/s] and order [-] in accepted parameter
        order.
    experiments : dict of str to Experiment
        Only the two source-declared fit observations and the run identifiers
        passed to the model. The held-out run remains on its source grid.

    Raises
    ------
    RuntimeError
        If the fit does not report successful termination.
    """
    experiments = {
        name: Experiment({"ir_band": runs[name].observation}, args=(name,))
        for name in FIT_RUNS
    }
    growth = runs[FIT_RUNS[0]].kinetics["growth"]  # [um/s], [J/mol], [-]
    seed = (growth[0], growth[2])  # [um/s], [-]
    estimator = ParameterEstimation(
        partial(predict_band, runs=runs), seed, experiments=experiments,
        name_params=list(PARAMETER_UNITS),
    )
    _, _, status = estimator.estimate_parameters(
        method="scipy", bounds=GROWTH_BOUNDS, x_scale="jac", verbose=False
    )
    if not status["success"]:
        raise RuntimeError(f"Calibration did not converge: {status['message']}")
    return estimator, experiments


def replay_caffeine(
    case: Path, record: CalibrationRecord, revision: str
) -> ParameterEstimation:
    """Rebuild a fit using the recorded observations and verified case inputs.

    Parameters
    ----------
    case : pathlib.Path
        Directory of the reduced case data.
    record : CalibrationRecord
        JSON round-trip of a record written by this workflow.
    revision : str
        Revision of the source tree running this replay.

    Returns
    -------
    ParameterEstimation
        Replayed fit with growth prefactor [um/s] and order [-].

    Raises
    ------
    ValueError
        If the model or case input files differ from the record.
    PharmaPy.Records.RerunMismatchError
        If the accepted parameters fail the record's repeatability check.
    """
    if record.model.revision != revision:
        raise ValueError("Pass the source revision recorded by the caffeine fit.")
    inputs = record.transformations[0].parameters
    digests = dict(inputs["source_sha256"])
    digests.update((source.name, source.sha256) for source in record.sources)
    for filename, digest in digests.items():
        relative = Path(filename)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"The calibration record contains an invalid source path: {filename}")
        if sha256((case / relative).read_bytes()).hexdigest() != digest:
            raise ValueError(f"{filename}: the bytes differ from the calibration record.")
    runs = load_runs(case, inputs["treatment"])
    return record.rerun(partial(predict_band, runs=runs), rtol=REPLAY_RTOL, atol=0)


def main() -> None:
    """Run the case fit or replay, and print the quantitative evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--treatment", choices=calibrate_band.TREATMENTS, default="common_slope")
    parser.add_argument("--revision", required=True, help="Source Git revision of this checkout")
    artifacts = parser.add_mutually_exclusive_group(required=True)
    artifacts.add_argument("--record", type=Path, help="Path to write the calibration JSON")
    artifacts.add_argument("--replay", type=Path, help="Path to an existing calibration JSON")
    options = parser.parse_args()
    if options.replay is not None:
        record = CalibrationRecord.load(options.replay)
        replay_caffeine(CASE, record, options.revision)
        print(f"Reproduced the fit in {options.replay}.")
        return

    runs = load_runs(CASE, options.treatment)
    estimator, experiments = fit_caffeine(runs)
    parameters = estimator.params_convg  # [um/s], [-]
    record = capture_record(CASE, options.revision, options.treatment, runs, experiments, estimator)
    record.save(options.record)
    print(f"Wrote calibration record to {options.record}.")

    derivatives = estimator.check_derivatives(
        rel_step=CENTRAL_RELATIVE_STEP, bounds=GROWTH_BOUNDS
    )
    fit = estimator.get_identifiability(
        include_mse=True, output_rtol=CVODE_RTOL, jacobian_rtol=np.maximum(
            derivatives.relative_error, np.finfo(float).eps
        )
    )
    print("Fit termination:", fit.convergence["message"])
    for name, estimate, error, units, disagreement in zip(
        fit.parameter_names, parameters, fit.standard_errors, PARAMETER_UNITS.values(),
        derivatives.relative_error
    ):
        print(f"{name}: {estimate:.6g} [{units}], conditional standard error {error:.3g}, "
              f"relative derivative disagreement {disagreement:.3g}")
    print(f"Practically identifiable under the declared noise model: {fit.identifiable}")
    print(fit.noise_assumption)
    print("Parameter correlation:", fit.correlation[0, 1])
    for name, run in runs.items():
        measured = run.observation  # [-] measured IR band height
        residual = predict_band(parameters, measured.x, name, runs=runs) - measured.values  # [-]
        rmse = np.sqrt(np.mean(residual**2))  # [-]
        role = "fit" if name in FIT_RUNS else "held-out"
        print(f"{name} ({role}): {measured.x.size} observations, IR RMSE {rmse:.4g}")

    if fit.identifiable and not fit.active_bounds:
        report, schedules = compare_sampling(fit, runs)
        for candidate in report.candidates:
            print(f"{candidate.name}: {candidate.status}, time [s] {schedules[candidate.name]}, "
                  f"log determinant gain {candidate.log_information_gain}")


if __name__ == "__main__":
    main()
