"""Fit and replay the existing Doyle synthetic example with native PharmaPy APIs.

Assets are ordinary native input JSON, the generator's CSV/manifest, a native
CalibrationRecord and an example-specific numerical verification report.
"""

import argparse
from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path
import platform
import subprocess
from typing import Optional

import numpy as np
import scipy
import PharmaPy
from PharmaPy.Bioreactors import simulate_measurements
from PharmaPy.ParamEstim import Experiment, Measurement, ParameterEstimation
from PharmaPy.Records import CalibrationRecord, DataSource, ModelRevision, Transformation

if __package__:
    from .generate_synthetic import generate, write_rows
else:
    from generate_synthetic import generate, write_rows


HERE = Path(__file__).resolve().parent
HOUR_S = 3600.  # [s/h], exact conversion
PARAMETER_NAMES = ('corrections.theta',)
CONVERGED_REASONS = ('Small gradient', 'Small step')
BASELINE_REVISION = 'fa47a388745080a7aaf275cdebee2289ffb0b345'


# Read and verify input assets before invoking the existing model interfaces.
def digest(path: Path) -> str:
    """Return the SHA-256 content checksum of a file.

    Parameters
    ----------
    path : Path
        Existing file.

    Returns
    -------
    str
        Hexadecimal checksum; content integrity, not authentication.
    """
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def selected_data(folder: Path, settings: dict) -> tuple[list, dict]:
    """Read one alternative realization without pooling or transforming it.

    Parameters
    ----------
    folder : Path
        Existing generator output directory.
    settings : dict
        Noise label and realization [-].

    Returns
    -------
    tuple
        Selected CSV rows (time [h], concentration/sigma [g/L]) and manifest.

    Raises
    ------
    ValueError
        Assets changed, selection is absent, or CSV values are invalid.
    """
    folder = Path(folder)
    manifest = json.loads((folder/'manifest.json').read_text())
    label = settings['noise_label']
    if label not in manifest['design']['noise_fractions']:
        raise ValueError('noise_label is not in the declared synthetic design')
    realization = settings['realization']
    count = (1 if manifest['design']['noise_fractions'][label] == 0.
             else manifest['design']['realizations'])
    if not isinstance(realization, int) or isinstance(realization, bool) or not 0 <= realization < count:
        raise ValueError('realization is outside the selected noise dataset')
    path = folder/f'{label}.csv'
    if digest(path) != manifest['sha256'][path.name]:
        raise ValueError('observation checksum does not match the generator manifest')
    with path.open(newline='') as stream:
        rows = [row for row in csv.DictReader(stream)
                if int(row['realization']) == realization]
    for row in rows:
        for key in ('time_h', 'value_g_l', 'sigma_g_l'):
            row[key] = float(row[key])  # [h], [g/L], [g/L], respectively
        if (not np.isfinite([row['time_h'], row['value_g_l'], row['sigma_g_l']]).all()
                or row['sigma_g_l'] < 0.):
            raise ValueError('observations must be finite with nonnegative uncertainties')
    if not rows:
        raise ValueError('selected realization contains no observations')
    return rows, manifest


def prepare_experiments(folder: Path, settings: dict) -> tuple[dict, dict, list]:
    """Connect saved native inputs and CSVs to shared Experiment/Measurement.

    Parameters
    ----------
    folder : Path
        Workflow assets, including data/ and inputs/.
    settings : dict
        Explicit field/parameter order, noise selection and numerical scales.

    Returns
    -------
    tuple
        All named Experiment objects, their fit/validation membership, and
        selected source rows [h, g/L, g/L]. Only fit members reach the estimator.

    Raises
    ------
    ValueError
        Parameter selection, source identities, units, splits or scales disagree.
    """
    folder = Path(folder)
    if tuple(settings['parameter_names']) != PARAMETER_NAMES or settings['parameter_units'] != ['-']:
        raise ValueError('this qualification fits only dimensionless corrections.theta')
    for key in ('dx_finitediff', 'zero_noise_scale_g_l',
                'replay_parameter_atol', 'replay_prediction_atol_g_l'):
        value = settings[key]  # [-] or [g/L], as named/documented in calibration.json
        if not np.isscalar(value) or not np.isfinite(value) or value <= 0.:
            raise ValueError(f'{key} must be positive and finite')
    declaration = settings['observation_declaration']
    if (declaration['time_unit'] != 'h'
            or set(declaration['fields']) != {'substrate', 'biomass'}
            or any(field['unit'] != 'g/L' or field['species'] != name
                   for name, field in declaration['fields'].items())):
        raise ValueError('Doyle workflow observations require substrate/biomass in g/L and h')
    rows, manifest = selected_data(folder/'data', settings)
    experiments, splits = {}, {}
    for entry in manifest['design']['experiments']:
        name = entry['name']
        if entry['split'] not in {'fit', 'validation'} or name in experiments:
            raise ValueError('experiment identities and fit/validation splits must be explicit')
        measurements = {}
        for field in declaration['fields']:
            samples = [row for row in rows if row['experiment'] == name and row['variable'] == field]
            if not samples or any(row['split'] != entry['split'] for row in samples):
                raise ValueError('CSV experiment identity/split disagrees with design')
            times = np.array([row['time_h'] for row in samples]) * HOUR_S  # [s]
            values = np.array([row['value_g_l'] for row in samples])  # [g/L]
            std = np.array([row['sigma_g_l'] for row in samples])  # [g/L]
            if manifest['design']['noise_fractions'][settings['noise_label']] == 0.:
                scale = settings['zero_noise_scale_g_l']  # [g/L], numerical scale, not assay error
                if not np.isfinite(scale) or scale <= 0.:
                    raise ValueError('zero_noise_scale_g_l must be positive and finite')
                std = np.full(values.shape, scale)  # [g/L]
            measurements[field] = Measurement(field, times, values,
                uncertainty=std, units='g/L', basis='mass concentration', x_units='s')
        model = json.loads((folder/'inputs'/f'{name}.json').read_text())
        experiments[name] = Experiment(measurements, kwargs=dict(case=model,
            parameter_names=tuple(settings['parameter_names']), declaration=declaration,
            thermo_path=str((folder/'inputs/thermo.json').resolve())))
        splits[name] = entry['split']
    if not {'fit', 'validation'} <= set(splits.values()):
        raise ValueError('at least one complete fit and validation experiment is required')
    if len(rows) != sum(len(measurement.values) for experiment in experiments.values()
                        for measurement in experiment.measurements.values()):
        raise ValueError('source CSV contains unknown experiments or fields')
    return experiments, splits, rows


def fit_experiments(experiments: dict, splits: dict, settings: dict) -> ParameterEstimation:
    """Fit only predeclared training experiments through the shared estimator.

    Parameters
    ----------
    experiments : dict
        Shared Experiment instances; observations [g/L], times [s].
    splits : dict
        Membership by experiment, 'fit' or 'validation'.
    settings : dict
        Parameter seed/step [-] and native LM options.

    Returns
    -------
    ParameterEstimation
        Fitted estimator, including its native termination status and accepted
        predictions. Budget exhaustion is retained, not relabelled convergence.
    """
    training = {name: experiment for name, experiment in experiments.items()
                if splits[name] == 'fit'}
    estimator = ParameterEstimation(simulate_measurements,
        np.asarray(settings['parameter_seed']), training,
        output_names=tuple(settings['observation_declaration']['fields']),
        name_params=settings['parameter_names'], dx_finitediff=settings['dx_finitediff'])
    estimator.optimize_fn(optim_options=deepcopy(settings['optim_options']), verbose=False)
    return estimator


def assess_fit(estimator: ParameterEstimation, settings: dict) -> dict:
    """Qualify the single Doyle correction at the declared replay resolution.

    Parameters
    ----------
    estimator : ParameterEstimation
        Fitted training-only estimator; theta [-], weighted residuals [-].
    settings : dict
        ``replay_parameter_atol`` [-] sets the accepted parameter resolution.

    Returns
    -------
    dict
        Solver termination, feasible local Gauss-Newton displacement [-],
        derivative-refinement agreement and numerical acceptance. This does
        not establish global optimality, biological adequacy or CI coverage.

    Raises
    ------
    ValueError
        Parameter count or declared resolution is unsupported.

    Notes
    -----
    Two secant resolutions (one and half the replay budget) test whether a
    feasible improvement exceeds that budget. Bounds are the actual provider
    domain. A constrained optimum may have a nonzero raw gradient. A copy
    preserves accepted estimator state; held-out observations are never used.
    """
    point = np.asarray(estimator.params_convg)  # [-]
    resolution = float(settings['replay_parameter_atol'])  # [-]
    if (settings['parameter_names'] != ['corrections.theta'] or point.shape != (1,)
            or not np.isfinite(resolution) or resolution <= 0.):
        raise ValueError('Doyle acceptance requires corrections.theta alone and positive finite resolution')
    domain = json.loads((HERE/'inputs/hybrid.json').read_text())['rate_provider']['validity_domain']['theta']
    lower, upper = domain  # [-], published provider validity limits
    trial = deepcopy(estimator)
    residual = np.asarray(trial.get_objective(point, True))  # [-]
    displacements = []  # [-]
    information = []  # [-], theta dimensionless and residuals standardized
    for step in (resolution, resolution / 2.):
        left = max(lower, float(point[0]) - step)  # [-]
        right = min(upper, float(point[0]) + step)  # [-]
        slope = (trial.get_objective(np.array([right]), True)
                 - trial.get_objective(np.array([left]), True)) / (right-left)  # [-]
        curvature = float(slope @ slope)  # [-], local Gauss-Newton information
        information.append(curvature)
        correction = -float(slope @ residual) / curvature if curvature > 0. else float('inf')  # [-]
        displacements.append(float(np.clip(point[0]+correction, lower, upper)-point[0]))
    finite = bool(np.isfinite(residual).all() and np.isfinite(information).all()
                  and min(information) > 0. and np.isfinite(displacements).all())
    refined = finite and abs(displacements[0]-displacements[1]) <= resolution
    stationary = finite and max(abs(value) for value in displacements) <= resolution
    solver_success = estimator.info_opt['stop_criterion'] in CONVERGED_REASONS
    return dict(accepted=bool(solver_success and refined and stationary),
        solver_terminated=solver_success, parameter_resolution=resolution,
        feasible_displacements=displacements, derivative_refinement_agrees=bool(refined),
        locally_stationary=bool(stationary), local_information=information,
        scope='Single-parameter local numerical acceptance; model adequacy and uncertainty coverage are not established')


def prediction_rows(experiments: dict, splits: dict, parameters) -> list:
    """Evaluate observations at accepted parameters without fitting held-out data.

    Parameters
    ----------
    experiments : dict
        Configured experiments, with possibly distinct per-field grids [s].
    splits : dict
        Experiment membership.
    parameters : array_like
        Log correction vector [-], shape (1,).

    Returns
    -------
    list of dict
        Named experiment/field rows: time [s], observed, fitted, neutral and
        residual [g/L], residual scale [g/L], standardized residual [-].
    """
    rows = []
    for name, experiment in experiments.items():
        fields = tuple(experiment.kwargs['declaration']['fields'])
        for field, measurement in experiment.measurements.items():
            column = fields.index(field)
            fitted = simulate_measurements(parameters, measurement.x, **experiment.kwargs)[:, column]  # [g/L]
            neutral = simulate_measurements([0.], measurement.x, **experiment.kwargs)[:, column]  # [g/L]
            for index, time_s in enumerate(measurement.x):
                residual = fitted[index] - measurement.values[index]  # [g/L]
                rows.append(dict(experiment=name, split=splits[name], field=field,
                    time_s=float(time_s), observed_g_l=float(measurement.values[index]),
                    predicted_g_l=float(fitted[index]), neutral_g_l=float(neutral[index]),
                    residual_g_l=float(residual), residual_scale_g_l=float(measurement.uncertainty[index]),
                    standardized_residual=float(residual/measurement.uncertainty[index])))
    return rows


def source_evidence() -> dict:
    """Capture actual code revision, dirty state, versions and source checksums.

    Returns
    -------
    dict
        Model-code identifiers and checksums. Git absence is recorded explicitly;
        the caller never fabricates a clean revision or an upstream acceptance.
    """
    root = Path(PharmaPy.__file__).resolve().parent
    try:
        revision = subprocess.run(['git', '-C', str(HERE), 'rev-parse', 'HEAD'],
                                  capture_output=True, text=True, check=False)
        status = subprocess.run(['git', '-C', str(HERE), 'status', '--porcelain'],
                                capture_output=True, text=True, check=False)
    except FileNotFoundError:
        revision, status = None, None  # Git is absent; source hashes remain available.

    return dict(structural_baseline=BASELINE_REVISION,
        revision=(revision.stdout.strip() if revision is not None and revision.returncode == 0 else None),
        dirty=(bool(status.stdout) if status is not None and status.returncode == 0 else None),
        python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
        package_source_sha256={str(path.relative_to(root)): digest(path)
                               for path in sorted(root.rglob('*.py'))},
        example_source_sha256={path.name: digest(path) for path in
            (Path(__file__), HERE/'generate_synthetic.py', HERE/'run_benchmarks.py')})


# Orchestrate existing inputs, generation, estimation and evidence outputs.
def run_workflow(output: Path, *, settings: Optional[dict] = None,
                 design: Optional[dict] = None) -> dict:
    """Generate, fit, validate and save this one reproducible example.

    Parameters
    ----------
    output : Path
        New or empty destination. All generated artifacts stay outside source.
    settings : dict, optional
        Existing calibration.json configuration, with documented units.
    design : dict, optional
        Existing generator design override, chiefly for short qualifications.

    Returns
    -------
    dict
        Example verification report with fit status, parameters [-], absolute
        parameter covariance [-], errors [g/L], input hashes and replay settings.
        A native training-only CalibrationRecord is saved alongside this report.

    Raises
    ------
    FileExistsError
        Destination is nonempty; prior scientific evidence is never overwritten.
    ValueError, RuntimeError
        Input validation, parameter domain, reference or native integration fails.
    """
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('workflow output must be new or empty')
    settings = deepcopy(settings if settings is not None else
                        json.loads((HERE/'inputs/calibration.json').read_text()))
    output.mkdir(parents=True, exist_ok=True)
    (output/'inputs').mkdir()
    manifest = generate(output/'data', truth_theta=settings['truth_theta'], design=design)
    hybrid = json.loads((HERE/'inputs/hybrid.json').read_text())
    for entry in manifest['design']['experiments']:
        model = deepcopy(hybrid)
        model['recipes']['continuous'][0]['volume_l_h'] = entry['dilution_h_inv']  # [L/h], 1 L working volume
        model['operation']['runtime']['value'] = manifest['design']['sample_times_h'][-1]  # [h], observed window
        (output/'inputs'/f"{entry['name']}.json").write_text(json.dumps(model, indent=2, allow_nan=False)+'\n')
    (output/'inputs/thermo.json').write_bytes((HERE/'inputs/thermo.json').read_bytes())
    (output/'inputs/calibration.json').write_text(json.dumps(settings, indent=2, allow_nan=False)+'\n')
    experiments, splits, _ = prepare_experiments(output, settings)
    estimator = fit_experiments(experiments, splits, settings)
    fitted = estimator.params_convg  # [-]
    rows = prediction_rows(experiments, splits, fitted)
    write_rows(output/'predictions.csv', list(rows[0]), rows)
    layout = estimator.get_residual_layout()
    layout['weighted_residual'] = estimator.info_opt['fun']  # [-]
    layout.to_csv(output/'fit-residual-layout.csv', index=False)
    for name, experiment in experiments.items():
        model = deepcopy(experiment.kwargs['case'])
        model['parameters']['corrections']['theta'] = float(fitted[0])  # [-]
        (output/'inputs'/f'{name}-fitted.json').write_text(json.dumps(model, indent=2, allow_nan=False)+'\n')
    metrics = {}
    for split in ('fit', 'validation'):
        selected = [row for row in rows if row['split'] == split]
        residuals = np.array([row['residual_g_l'] for row in selected])  # [g/L]
        metrics[split] = dict(count=len(selected), rmse_g_l=float(np.sqrt(np.mean(residuals**2))),
            weighted_sse=float(sum(row['standardized_residual']**2 for row in selected)),
            neutral_weighted_sse=float(sum(((row['neutral_g_l']-row['observed_g_l'])/row['residual_scale_g_l'])**2 for row in selected)))
    acceptance = assess_fit(estimator, settings)
    evidence = source_evidence()
    training = {name: experiments[name] for name in estimator.experim_names}
    source = DataSource('Doyle synthetic observations', 'synthetic', 'public',
        uri=f"data/{settings['noise_label']}.csv",
        sha256=digest(output/'data'/f"{settings['noise_label']}.csv"),
        description='Generated concentrations; only the declared realization and training experiments are fitted.')
    record = CalibrationRecord.from_estimation(estimator, training,
        case='Doyle synthetic kinetics correction',
        parameter_units=settings['parameter_units'], covariance_include_mse=False,
        model=ModelRevision('PharmaPy.Bioreactors.simulate_measurements',
            revision=evidence['revision'],
            description=f"Source checksums and dirty={evidence['dirty']} are retained in fit-report.json."),
        sources=[source], transformations=[Transformation(
            'Select one realization and training experiments; convert hours to seconds; retain concentration units.',
            source=source.name, parameters=dict(realization=settings['realization'],
                experiments=list(training), seconds_per_hour=HOUR_S,  # [s/h]
                zero_noise_numerical_scale_g_l=settings['zero_noise_scale_g_l']))],  # [g/L], numerical
        model_solver={name: deepcopy(experiment.kwargs['case']['operation'])
                      for name, experiment in training.items()},
        description='Synthetic single-parameter demonstration; covariance is linearized, not empirical confidence.')
    record.save(output/'calibration-record.json')
    report = dict(description='Doyle numerical verification accompanying the native CalibrationRecord',
        converged=acceptance['accepted'], fit_acceptance=acceptance,
        stop_criterion=estimator.info_opt['stop_criterion'],
        optimizer_counters={key: int(estimator.info_opt[key]) for key in ('num_iter', 'num_fun_eval')},
        fitted_parameters=dict(zip(settings['parameter_names'], fitted.tolist())),
        parameter_units=settings['parameter_units'],
        covariance=estimator.get_covariance(include_mse=False).tolist(),
        covariance_include_mse=False,
        covariance_meaning=('linearized, known synthetic absolute uncertainties; not empirical confidence'
            if manifest['design']['noise_fractions'][settings['noise_label']] else
            'inverse information with numerical scales; not statistical uncertainty'),
        fitted_experiments=list(estimator.experim_names), heldout_experiments=[name for name in splits if splits[name] == 'validation'],
        field_order=list(settings['observation_declaration']['fields']), metrics=metrics,
        model=evidence,
        sha256={str(path.relative_to(output)): digest(path)
                for path in sorted(output.rglob('*')) if path.is_file()})
    (output/'fit-report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    return report


def replay(output: Path, *, refit: bool = False) -> dict:
    """Reload native assets, reproduce predictions, and optionally rerun the fit.

    Parameters
    ----------
    output : Path
        Saved workflow directory; may be relocated before replay.
    refit : bool, optional
        Refit training data from the saved seed/options and compare parameters.

    Returns
    -------
    dict
        Prediction discrepancy [g/L], parameter discrepancy [-] if refitted,
        and the original convergence status. Source assets remain unchanged.

    Raises
    ------
    ValueError
        Asset/source checksums, stored identities, predictions or refit disagree.

    Notes
    -----
    Refitting reuses the native record's estimation and optimizer settings.
    After asset verification, only the thermo file path is rebound to the
    relocated directory; measurements and model configuration are preserved.
    """
    output = Path(output)
    report = json.loads((output/'fit-report.json').read_text())
    for name, checksum in report['sha256'].items():
        path = Path(name)
        if (path.is_absolute() or '..' in path.parts or not (output/path).is_file()
                or digest(output/path) != checksum):
            raise ValueError(f'workflow asset checksum mismatch: {name}')
    current = source_evidence()
    for key in ('package_source_sha256', 'example_source_sha256'):
        if current[key] != report['model'][key]:
            raise ValueError('model source changed since fitting; replay requires the recorded code')
    settings = json.loads((output/'inputs/calibration.json').read_text())
    record = CalibrationRecord.load(output/'calibration-record.json')
    experiments, splits, _ = prepare_experiments(output, settings)
    fitted = np.array([report['fitted_parameters'][name] for name in settings['parameter_names']])  # [-]
    if not np.array_equal(fitted, record.fitted_values):
        raise ValueError('verification report and native record parameter values disagree')
    actual = prediction_rows(experiments, splits, fitted)
    with (output/'predictions.csv').open(newline='') as stream:
        stored = list(csv.DictReader(stream))
    if len(actual) != len(stored):
        raise ValueError('replay prediction row count differs')
    discrepancy = 0.  # [g/L]
    for expected, observed in zip(stored, actual):
        if any(str(expected[key]) != str(observed[key]) for key in ('experiment', 'split', 'field')) or float(expected['time_s']) != observed['time_s']:
            raise ValueError('replay prediction identity/order differs')
        stored_value = float(expected['predicted_g_l'])  # [g/L]
        if not np.isfinite(stored_value):
            raise ValueError('stored predictions must be finite')
        discrepancy = max(discrepancy, abs(stored_value-observed['predicted_g_l']))  # [g/L]
    if discrepancy > settings['replay_prediction_atol_g_l']:
        raise ValueError('replayed predictions exceed the declared concentration budget')
    result = dict(converged=report['converged'], prediction_max_abs_g_l=discrepancy)
    if refit:
        inputs = record.estimation_inputs()
        # Rebind the one file-backed collaborator after its saved checksum was verified.
        for name, experiment in inputs['x_data'].items():
            kwargs = dict(experiment.kwargs)
            kwargs['thermo_path'] = str((output/'inputs/thermo.json').resolve())
            inputs['x_data'][name] = Experiment(experiment.measurements,
                args=experiment.args, kwargs=kwargs, name=experiment.name,
                author=experiment.author, date=experiment.date)
        estimator = ParameterEstimation(simulate_measurements, **inputs)
        estimator.optimize_fn(**record.optimizer_inputs(), verbose=False)
        record.check_rerun(estimator, rtol=0., atol=settings['replay_parameter_atol'])
        difference = np.abs(estimator.params_convg-fitted)  # [-]
        if (not assess_fit(estimator, settings)['accepted']
                or np.any(difference > settings['replay_parameter_atol'])):
            raise ValueError('refit does not reproduce a converged recorded parameter')
        result['parameter_max_abs'] = float(np.max(difference))  # [-]
    return result


def main() -> None:
    """Run or replay the native workflow from either supported launch directory.

    Notes
    -----
    ``--settings`` and ``--design`` select ordinary JSON inputs for a new run.
    Replay always uses saved inputs. Nonconvergence is reported with exit code 2;
    input/solver errors propagate and never become successful fits.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=HERE/'outputs/calibration')
    parser.add_argument('--settings', type=Path)
    parser.add_argument('--design', type=Path)
    parser.add_argument('--replay', action='store_true')
    parser.add_argument('--refit', action='store_true')
    arguments = parser.parse_args()
    if arguments.refit and not arguments.replay:
        parser.error('--refit requires --replay')
    if arguments.replay and (arguments.settings is not None or arguments.design is not None):
        parser.error('replay uses saved settings/design; do not supply replacements')
    settings = None if arguments.settings is None else json.loads(arguments.settings.read_text())
    design = None if arguments.design is None else json.loads(arguments.design.read_text())
    result = (replay(arguments.output, refit=arguments.refit) if arguments.replay
              else run_workflow(arguments.output, settings=settings, design=design))
    print(json.dumps(result, indent=2, allow_nan=False))
    if not result['converged']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
