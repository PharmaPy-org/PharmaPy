"""Fit the selected December caffeine case through the shared WP1 interfaces.

This first pass fits the growth prefactor and order to the two identified
ReactIR runs. Secondary nucleation follows the case starting values. Only
the third run's post-seeding observations are held out of growth fitting.
See the caffeine README for the fit record, replay and interpretation.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from functools import partial
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
from subprocess import run as run_process
import sys
from typing import Mapping

import numpy as np
from scipy.optimize import Bounds

from PharmaPy import PATCalibration, SlurryMeasurements
from PharmaPy.ExperimentDesign import CandidateExperiment, ExperimentDesignReport, rank_experiments
from PharmaPy.ParamEstim import DerivativeCheck, Experiment, Measurement, ParameterEstimation
from PharmaPy.Records import CalibrationRecord, DataSource, ModelRevision, Transformation
from PharmaPy.StatsModule import IdentifiabilityReport
from PharmaPy.jac_module import DEFAULT_OUTPUT_RTOL, numerical_jac_central

from . import batch_cooling, calibrate_band, dataset


CASE = Path(__file__).with_name("caffeine")
CASE_ID = "caffeine_batch_cooling"
MODEL_NAME = "examples.pat.caffeine_m6:predict_band"
# Run roles and final temperature from the case-selection note, sections 2 and 5.
FIT_RUNS = ("high_seed_slow_cool", "low_seed_fast_cool")
VALIDATION_RUN = "low_seed_slow_cool"
FINAL_TEMPERATURE = 293.15  # [K], 20 C target of the final cooling ramp
PARAMETER_UNITS = {"k_g": "um/s", "g": "1"}
# Nonnegative numerical bounds. A zero order would mean constant growth in
# the supersaturated regime, so active-bound fits require interpretation.
GROWTH_BOUNDS = Bounds(np.zeros(len(PARAMETER_UNITS)), np.full(len(PARAMETER_UNITS), np.inf))
# Start two decimal orders below PharmaPy's reference CVode tolerance, then
# compare the observable at a second refinement of the same size.
REFINEMENT_FACTOR = 100  # [-]
CVODE_RTOL = DEFAULT_OUTPUT_RTOL / REFINEMENT_FACTOR  # [-]
# Set the common absolute state tolerance from the order of the charged
# liquid volume, roughly 0.35 kg of water at 1,000 kg/m**3.
CVODE_ATOL = CVODE_RTOL * 1e-4  # [component-dependent solver units]
CVODE_OPTIONS = {"rtol": CVODE_RTOL, "atol": CVODE_ATOL}
# Start the central stencil with the second-order step suggestion in
# ParameterEstimation.check_derivatives and measure observable error separately.
CENTRAL_RELATIVE_STEP = CVODE_RTOL ** (1.0 / 3.0)  # [-]
# The original simulation verifies this relative inventory tolerance on both
# temperature controls in tests/test_pat_batch_cooling_models.py.
BALANCE_RTOL = 1e-6  # [-]
# Compare the discrete designs at the same budget, two observations per trial.
# Each pair divides the logged spectrum times into three successive windows.
PLAN_NAMES = ("early", "middle", "late")
SPECTRA_PER_PLAN = 2
# Same-source deterministic replay compares fitted parameters at the default
# SciPy relative parameter stopping tolerance. Recorded zero requires exact zero.
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
    simulation = description["simulation"]
    temperature_keys = ("temperature_table", "temperature_column")
    if any(settings[key] != simulation[key] for key in temperature_keys):
        raise ValueError("The band equation and cooling model need the same logged temperature.")
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
    params: np.ndarray, times: np.ndarray, name: str, *, runs: Mapping[str, CaffeineRun],
    solver_options: Mapping[str, float] = CVODE_OPTIONS,
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
    solver_options : mapping of str to float, optional
        CVode options. Defaults to the case-specific relative tolerance [-]
        and common absolute tolerance [component-dependent solver units].

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
        cooling, kinetics, times, "legacy", solver_options=solver_options
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


def case_inputs(case: Path, description: Mapping) -> tuple[dict[str, str], dict[str, str]]:
    """Hash the files that supply the measured data and model inputs.

    Parameters
    ----------
    case : pathlib.Path
        Directory of the reduced case data.
    description : mapping
        Loaded run description with band-calibration and model tables.

    Returns
    -------
    measurements : dict of str to str
        SHA-256 hex digests by relative path of the reduced data files.
    model : dict of str to str
        SHA-256 hex digests of the model, calibration and layout inputs.
    """
    measurement = description["band_calibration"]
    kinds = (measurement["table"], measurement["temperature_table"])
    measurement_files = (
        f"reduced/{name}_{kind}.csv"
        for name in (*FIT_RUNS, VALIDATION_RUN) for kind in kinds
    )
    model_files = (
        "runs.json", description["properties"]["file"], "band_calibration.json",
        "solubility.json", *(f"reduced_{kind}.layout.json" for kind in kinds),
    )
    return (
        {name: sha256((case / name).read_bytes()).hexdigest() for name in measurement_files},
        {name: sha256((case / name).read_bytes()).hexdigest() for name in model_files},
    )


def capture_record(
    case: Path, revision: str, treatment: str, runs: Mapping[str, CaffeineRun],
    experiments: Mapping[str, Experiment], estimator: ParameterEstimation,
    diagnostics: Mapping, environment: Mapping,
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
    diagnostics : mapping
        Reported numerical comparison and residual diagnostics with units
        in their field names or the documented scale.
    environment : mapping
        Interpreter and numerical-package version strings and lock digest.

    Returns
    -------
    CalibrationRecord
        Replayable model, provenance, predictions and fit statistics.
    """
    description = dataset.load_description(case)
    measurements, model_inputs = case_inputs(case, description)
    sources = [
        DataSource(
            filename, "experimental", "Reduced data shareable in PharmaPy repository",
            uri=f"examples/pat/caffeine/{filename}", sha256=digest,
            role=("band and solubility calibration; growth-fit holdout"
                  if Path(filename).name.startswith(VALIDATION_RUN)
                  else "band and solubility calibration; growth fitting"),
            description=description["description"],
        )
        for filename, digest in measurements.items()
    ]
    return CalibrationRecord.from_estimation(
        estimator, experiments, case=CASE_ID,
        parameter_units=PARAMETER_UNITS,
        model=ModelRevision(
            MODEL_NAME, revision=revision,
            description="Seeded moments BatchCryst with measured temperature control",
        ),
        sources=sources,
        transformations=[Transformation(
            "Use reduced band heights, crop to cooling, subtract the run-description seeding time, "
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
                      **CVODE_OPTIONS, "inventory_rtol": BALANCE_RTOL,
                      "diagnostics": dict(diagnostics), "environment": dict(environment)},
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
        If the available observation times cannot populate the schedules
        or the fit lacks positive, resolved, interior growth coordinates.
    """
    parameters = fit.estimates  # [um/s], [-]
    if (fit.parameter_names != tuple(PARAMETER_UNITS) or not fit.identifiable
            or fit.active_bounds or not np.isfinite(parameters).all() or (parameters <= 0).any()):
        raise ValueError("The comparison requires positive, resolved growth estimates at an interior point.")
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
        method="scipy", bounds=(GROWTH_BOUNDS.lb, GROWTH_BOUNDS.ub),
        x_scale="jac", verbose=False,
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
    if record.case != CASE_ID or record.model.name != MODEL_NAME:
        raise ValueError("The record belongs to a different case or model.")
    if record.model.revision != revision:
        raise ValueError("Pass the source revision recorded by the caffeine fit.")
    if len(record.transformations) != 1:
        raise ValueError("The caffeine record requires one input provenance entry.")
    inputs = record.transformations[0].parameters
    recorded = {source.name: source.sha256 for source in record.sources}
    recorded.update(inputs["source_sha256"])
    measurements, model_inputs = case_inputs(case, dataset.load_description(case))
    current = measurements | model_inputs
    if current.keys() != recorded.keys():
        raise ValueError("The record does not describe the complete caffeine input set.")
    for filename, digest in current.items():
        if recorded[filename] != digest:
            raise ValueError(f"{filename}: the bytes differ from the calibration record.")
    model_settings = None if record.solver is None else record.solver.model_solver
    required = {"name": "Assimulo CVode", "report_continuously": True, **CVODE_OPTIONS}
    if model_settings is None or any(model_settings.get(key) != value for key, value in required.items()):
        raise ValueError("The record's model solver settings differ from the caffeine workflow.")
    runs = load_runs(case, inputs["treatment"])
    return record.rerun(partial(predict_band, runs=runs), rtol=REPLAY_RTOL, atol=0)


def observable_refinement(
    derivatives: DerivativeCheck, runs: Mapping[str, CaffeineRun]
) -> dict[str, float]:
    """Audit CVode refinement on both fit grids at the derivative stencils.

    Parameters
    ----------
    derivatives : DerivativeCheck
        Accepted growth prefactor [um/s] and order [-], with the step in
        each parameter's units and the reference difference formula.
    runs : mapping of str to CaffeineRun
        The measured sample grids [s] and inputs for the two fitted runs.

    Returns
    -------
    dict of str to float
        Maximum relative difference [-] in the modeled band between the
        working and refined solve, at the nominal point and all reference
        difference points for each run. This is a sampled error scale.
    """
    # Coefficients of the central and one-sided second-order stencils.
    offsets = {"central": (-1, 1), "forward": (1, 2), "backward": (-1, -2)}
    points = [derivatives.params]  # [um/s], [-]
    for index, (step, formula) in enumerate(zip(derivatives.steps, derivatives.formulas)):
        for offset in offsets[formula]:
            point = derivatives.params.copy()  # [um/s], [-]
            point[index] += offset * step
            points.append(point)

    tighter = {key: value / REFINEMENT_FACTOR for key, value in CVODE_OPTIONS.items()}
    maximum = {}
    for name in FIT_RUNS:
        errors = []
        times = runs[name].observation.x  # [s] after seeding
        for point in points:
            base = predict_band(point, times, name, runs=runs)  # [-] absorbance
            fine = predict_band(point, times, name, runs=runs, solver_options=tighter)  # [-]
            errors.append(float(np.max(np.abs(base - fine) / np.abs(fine))))  # [-]
        maximum[name] = max(errors)
    return maximum


def measurement_diagnostics(
    observation: Measurement, prediction: np.ndarray,
) -> dict[str, float | int | None]:
    """Compare model residuals with instrument scatter and sample ordering.

    Parameters
    ----------
    observation : Measurement
        Measured IR band height and plateau scatter [-], on times [s].
    prediction : numpy.ndarray
        Predicted IR band height [-], in the observation order.

    Returns
    -------
    dict
        Number of observations, mean and RMSE [-] absorbance residual,
        plateau scatter [-] absorbance, mean squared residual
        relative to plateau variance [-], and lag-one Pearson correlation
        [-] between adjacent retained rows. None indicates undefined
        correlation. Adjacent samples may have unequal time intervals.
    """
    observed = np.isfinite(observation.values)
    residual = prediction[observed] - observation.values[observed]  # [-] absorbance
    weighted = residual / observation.uncertainty[observed]  # [-] standardized residual
    correlation = None  # [-]
    if residual.size > 2 and np.ptp(residual[:-1]) > 0 and np.ptp(residual[1:]) > 0:
        correlation = float(np.corrcoef(residual[:-1], residual[1:])[0, 1])  # [-]
    return {
        "count": int(residual.size),
        "bias_absorbance": float(np.mean(residual)),
        "plateau_scatter_absorbance": float(observation.uncertainty[observed][0]),
        "rmse_absorbance": float(np.sqrt(np.mean(residual**2))),
        "variance_vs_plateau": float(np.mean(weighted**2)),
        "adjacent_residual_correlation": correlation,
    }


def verify_source(revision: str) -> Path:
    """Verify the running source and model inputs match the requested Git commit.

    Parameters
    ----------
    revision : str
        Full commit hash selected for the calibration or replay.

    Returns
    -------
    pathlib.Path
        Root of the verified checkout.

    Raises
    ------
    ValueError
        If the running code is outside the checkout, HEAD differs from the
        requested commit, or tracked case, package or lock bytes have changed.
    subprocess.CalledProcessError
        If Git cannot read the source tree.
    """
    root = Path(__file__).resolve().parents[2]
    if Path(PATCalibration.__file__).resolve().parent != root / "PharmaPy":
        raise ValueError("Import PharmaPy from the same checkout as the caffeine example.")
    head = run_process(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True,
    ).stdout.strip()
    if revision != head:
        raise ValueError(f"Pass the current full Git revision ({head}).")
    status = run_process(
        ["git", "status", "--porcelain", "--untracked-files=no", "--",
         "PharmaPy", "examples/pat", "pyproject.toml", "pixi.lock"],
        cwd=root, check=True, capture_output=True, text=True,
    ).stdout
    if status:
        raise ValueError("Commit or discard tracked model, data and lock edits before running.")
    return root


def environment_identity(root: Path) -> dict[str, str | None]:
    """Record the numerical environment used for a case solve.

    Parameters
    ----------
    root : pathlib.Path
        Verified checkout with the solver environment lock.

    Returns
    -------
    dict of str to str or None
        Python and imported dependency versions, SUNDIALS installation version
        when Conda metadata identifies it, and SHA-256 of the Pixi lock.
    """
    native = sorted((Path(sys.prefix) / "conda-meta").glob("sundials-*.json"))
    sundials = json.loads(native[0].read_text(encoding="utf-8"))["version"] if len(native) == 1 else None
    return {
        "python": sys.version.split()[0],
        "assimulo": version("assimulo"),
        "numpy": np.__version__,
        "scipy": version("scipy"),
        "sundials": sundials,
        "pixi_lock_sha256": sha256((root / "pixi.lock").read_bytes()).hexdigest(),
    }


def main() -> None:
    """Run the case fit or replay, and print the quantitative evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--treatment", choices=calibrate_band.TREATMENTS, default="common_slope")
    parser.add_argument("--revision", required=True, help="Source Git revision of this checkout")
    artifacts = parser.add_mutually_exclusive_group(required=True)
    artifacts.add_argument("--record", type=Path, help="Path to write the calibration JSON")
    artifacts.add_argument("--replay", type=Path, help="Path to an existing calibration JSON")
    options = parser.parse_args()
    root = verify_source(options.revision)
    if options.replay is not None:
        record = CalibrationRecord.load(options.replay)
        replay_caffeine(CASE, record, options.revision)
        print(f"Reproduced the fit in {options.replay}.")
        return

    runs = load_runs(CASE, options.treatment)
    estimator, experiments = fit_caffeine(runs)
    parameters = estimator.params_convg  # [um/s], [-]
    print("Fit completed. Comparing derivatives and observable refinement.")
    derivatives = estimator.check_derivatives(
        rel_step=CENTRAL_RELATIVE_STEP, bounds=GROWTH_BOUNDS
    )
    refinement = observable_refinement(derivatives, runs)  # [-] relative changes
    output_scale = max(max(refinement.values()), np.finfo(float).eps)  # [-]
    fit = estimator.get_identifiability(
        include_mse=True, output_rtol=output_scale, jacobian_rtol=np.maximum(
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
    print(f"Numerical directions resolved under the sampled error scale: {fit.identifiable}")
    print("Maximum relative band changes on the sampled stencil:", refinement)
    print(fit.noise_assumption)
    print("Parameter correlation:", fit.correlation[0, 1])
    print("The precision diagnostics are conditional on the working independent-error model.")
    residuals = {}
    for name, run in runs.items():
        observation = run.observation  # [-] measured IR band height
        prediction = predict_band(parameters, observation.x, name, runs=runs)  # [-]
        metrics = measurement_diagnostics(observation, prediction)
        residuals[name] = metrics
        role = "growth fit" if name in FIT_RUNS else "excluded from growth fit"
        ratio = np.sqrt(metrics["variance_vs_plateau"])  # [-]
        print(f"{name} ({role}): {metrics['count']} observations, "
              f"IR RMSE {metrics['rmse_absorbance']:.4g} ({ratio:.2f}× plateau scatter), "
              f"bias {metrics['bias_absorbance']:.4g}, "
              f"adjacent residual correlation {metrics['adjacent_residual_correlation']}")

    diagnostics = {
        "stencil_relative_output_changes": refinement,
        "reference_stencil_relative_step": derivatives.rel_step,
        "relative_derivative_disagreement": dict(zip(PARAMETER_UNITS, derivatives.relative_error)),
        "sampled_output_relative_error_scale": output_scale,
        "residuals": residuals,
        "residual_variance_multiplier": fit.residual_mean_square,
        "numerical_directions_resolved": fit.identifiable,
        "active_bounds": fit.active_bounds,
        "precision_assumption": "Local linearization with independent residuals.",
    }
    record = capture_record(CASE, options.revision, options.treatment, runs, experiments, estimator,
                            diagnostics, environment_identity(root))
    record.save(options.record)
    print(f"Wrote calibration record to {options.record}.")

    if fit.identifiable and not fit.active_bounds:
        report, schedules = compare_sampling(fit, runs)
        by_name = {entry.name: entry for entry in report.candidates}
        print("Hypothetical equal-count subsampling scores under independent errors.")
        for name, times in schedules.items():
            print(f"{name}: time [s] {times}, log determinant gain {by_name[name].log_information_gain}")
    else:
        print("Subsampling scores require resolved growth directions at an interior fitted point.")
    print("A physical scheduling recommendation requires a verified residual model or independent repeats.")


if __name__ == "__main__":
    main()
