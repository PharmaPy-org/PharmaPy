"""Real Doyle generation/fit/replay, independent optima and strict holdout.

Short 2 h fixtures qualify orchestration without changing the full 100 h user
configuration. No mocks or optional solver are used. Fresh-process replay runs
from a relocated directory; original inputs and source observations are guarded.
"""

from copy import deepcopy
import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest
from scipy.optimize import minimize_scalar
from PharmaPy.Records import CalibrationRecord

from examples.bioreactors.Doyle_Exercise_4_15 import calibrate, generate_synthetic
from .test_bioreactor_estimation_handoff import (
    HOUR_S, SAMPLE_H as HANDOFF_SAMPLE_H, TRUTH_THETA, FD_STEP,
    CONCENTRATION_ATOL, independent_prediction,
)


pytestmark = pytest.mark.integration
SAMPLE_H = HANDOFF_SAMPLE_H[:4]  # [h], 2 h orchestration fixture; 8 h physics covered by stress tests
FIT_ATOL = 5. * FD_STEP  # [-], established stress qualification budget
ROOT = Path(__file__).parents[2]
EXAMPLE = ROOT / 'examples/bioreactors/Doyle_Exercise_4_15'


# Real fixture construction and independent qualification helpers.
def _calibration_row_key(row):
    """Order observations by experiment, variable, realization, and time.

    Parameters
    ----------
    row : dict
        One saved observation record with experiment/variable identifiers, realization index, and time_h [h].

    Returns
    -------
    tuple
        Observation identity and sampling time [h].
    """
    return (row['experiment'], row['variable'], int(row['realization']), float(row['time_h']))


@pytest.fixture
def settings():
    """Load the example's actual fitting and replay declarations.

    Returns
    -------
    dict
        Native parameter/solver settings, units recorded in calibration.json.
    """
    return json.loads((EXAMPLE/'inputs/calibration.json').read_text())


@pytest.fixture
def short_design():
    """Retain design conditions/noise while shortening the qualification window.

    Returns
    -------
    dict
        Existing design, times [h] through 2 h and two alternative realizations.
    """
    design = json.loads((EXAMPLE/'inputs/synthetic_design.json').read_text())
    design['sample_times_h'] = SAMPLE_H.tolist()  # [h]
    design['realizations'] = 2  # [-], bounded mechanical qualification, not coverage
    return design


@pytest.fixture(scope="module")
def workflow_base(tmp_path_factory):
    """Run a real noisy fit in a disposable artifact directory.

    Parameters
    ----------
    tmp_path_factory : pytest.TempPathFactory
        Creates an isolated module-level reference directory.

    Returns
    -------
    tuple
        Artifact directory and fit report (theta [-], predictions [g/L]).
    """
    settings = json.loads((EXAMPLE/'inputs/calibration.json').read_text())
    design = json.loads((EXAMPLE/'inputs/synthetic_design.json').read_text())
    design['sample_times_h'] = SAMPLE_H.tolist()  # [h], qualification window
    design['realizations'] = 2  # [-], alternative realizations remain unpooled
    output = tmp_path_factory.mktemp('doyle-workflow')/'reference'
    report = calibrate.run_workflow(output, settings=settings, design=design)
    return output, report


@pytest.fixture
def workflow(workflow_base, tmp_path):
    """Copy real generated artifacts so each mutation has independent inputs.

    Parameters
    ----------
    workflow_base : tuple
        Once-executed real workflow and report (theta [-], predictions [g/L]).
    tmp_path : Path
        Per-test destination; no state is shared between mutation tests.

    Returns
    -------
    tuple
        Private artifact directory and copied report.
    """
    source, report = workflow_base
    output = tmp_path/'workflow'
    shutil.copytree(source, output)
    return output, deepcopy(report)


def reference_loss(theta, records):
    """Compute weighted loss using independently written balance equations.

    Parameters
    ----------
    theta : float
        Trial correction [-].
    records : list of tuple
        D [1/h], times [s], field index, observations/sigma [g/L].

    Returns
    -------
    float
        Half squared standardized residual norm [-].
    """
    objective = 0.  # [-]
    for dilution, times, column, values, std in records:
        residual = (independent_prediction(theta, times, dilution)[:, column]-values)/std  # [-]
        objective += float(residual@residual)/2.  # [-]
    return objective


def rewrite_observations(output, rows):
    """Rewrite a manufactured CSV and update its generator content checksum.

    Parameters
    ----------
    output : Path
        Existing workflow artifact folder.
    rows : list of dict
        Explicitly transformed source rows, concentration/sigma [g/L], time [h].
    """
    path = output/'data/noise_1pct.csv'
    generate_synthetic.write_rows(path, list(reversed(list(rows[0]))), rows)
    manifest_path = output/'data/manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['sha256'][path.name] = calibrate.digest(path)
    manifest_path.write_text(json.dumps(manifest))


def execute_calibration_notebook(output: Path) -> None:
    """Execute the six real workbook cells with a short qualification design.

    Parameters
    ----------
    output : Path
        New destination for workflow assets and figure.

    Notes
    -----
    Uses the repository's existing solver-independent notebook execution pattern.
    Cells execute in an isolated process; no imports, functions or environment
    are replaced. Only declared user inputs are shortened to the 2 h fixture.
    """
    notebook = json.loads((EXAMPLE/'calibration.ipynb').read_text())
    namespace = {'__name__': '__notebook__'}
    cells = [cell for cell in notebook['cells'] if cell['cell_type'] == 'code']
    assert len(cells) == 6
    try:
        for index, cell in enumerate(cells):
            exec(compile(''.join(cell['source']), f'calibration.ipynb:step{index+1}', 'exec'), namespace)
            if index == 0:
                namespace['OUTPUT_DIR'] = output
            elif index == 1:
                namespace['design']['sample_times_h'] = SAMPLE_H.tolist()  # [h]
                namespace['design']['realizations'] = 2  # [-], mechanical qualification
        assert namespace['report']['converged']
        assert (output/'calibration.png').stat().st_size > 0
        assert namespace['completed'].returncode == 0
        assert json.loads(namespace['completed'].stdout)['converged']
    finally:
        import matplotlib.pyplot as plt
        plt.close('all')


# Public-path regressions, independent balances, and failure qualification.
def test_native_record_roundtrip_preserves_training_data_and_estimator_inputs(workflow):
    """Check native persistence separately from the verification report.

    Parameters
    ----------
    workflow : tuple
        Saved directory and report from the real training-only fit; observations
        [g/L], times [s] and the fitted correction parameter [-].
    """
    output, report = workflow
    record = CalibrationRecord.load(output/'calibration-record.json')
    settings = json.loads((output/'inputs/calibration.json').read_text())
    experiments, splits, _ = calibrate.prepare_experiments(output, settings)
    training = [name for name in experiments if splits[name] == 'fit']
    assert len(training) == 2
    assert list(record.experiments) == training == report['fitted_experiments']
    assert set(record.experiments).isdisjoint(report['heldout_experiments'])
    assert [parameter.name for parameter in record.parameters] == settings['parameter_names']
    assert [parameter.units for parameter in record.parameters] == settings['parameter_units']
    np.testing.assert_array_equal(record.fitted_values,
        [report['fitted_parameters'][name] for name in settings['parameter_names']])
    np.testing.assert_array_equal(record.covariance, report['covariance'])
    assert record.covariance_include_mse is False
    for name, saved in record.experiments.items():
        expected = experiments[name]
        assert list(saved.measurements) == list(expected.measurements)
        for field, actual in saved.measurements.items():
            original = expected.measurements[field]
            assert actual.units == original.units == 'g/L'
            assert actual.basis == original.basis == 'mass concentration'
            assert actual.x_units == original.x_units == 's'
            np.testing.assert_array_equal(actual.x, original.x)
            np.testing.assert_array_equal(actual.values, original.values)
            np.testing.assert_array_equal(actual.uncertainty, original.uncertainty)
    inputs = record.estimation_inputs()
    assert list(inputs['x_data']) == training
    assert inputs['output_names'] == list(settings['observation_declaration']['fields'])
    assert inputs['dx_finitediff'] == pytest.approx(settings['dx_finitediff'])
    assert record.sources[0].origin == 'synthetic'
    assert record.sources[0].sha256 == report['sha256'][record.sources[0].uri]
    assert CalibrationRecord.from_json(record.to_json()).to_dict() == record.to_dict()


def test_small_step_stagnation_is_not_accepted(tmp_path, settings, short_design):
    """Overwhelming LM damping must not qualify an unchanged, poor seed."""
    settings['parameter_seed'] = [.4]  # [-], valid seed away from the optimum
    settings['optim_options']['mu'] = 1e20  # [-], deliberate stagnation fixture
    report = calibrate.run_workflow(tmp_path/'stalled', settings=settings,
                                    design=short_design)
    assert report['stop_criterion'] == 'Small step'
    assert not report['converged']
    assert not report['fit_acceptance']['locally_stationary']
    assert report['metrics']['fit']['weighted_sse'] > report['metrics']['fit']['neutral_weighted_sse']


def test_acceptance_rejects_a_different_parameter_contract(workflow_base, settings):
    experiments, splits, _ = calibrate.prepare_experiments(workflow_base[0], settings)
    estimator = calibrate.fit_experiments(experiments, splits, settings)
    settings['parameter_names'] = ['rates.mu_max']
    with pytest.raises(ValueError, match='corrections.theta alone'):
        calibrate.assess_fit(estimator, settings)


def test_noisy_fit_matches_independent_optimum_and_records_exact_handoff(workflow, settings):
    output, report = workflow
    experiments, splits, _ = calibrate.prepare_experiments(output, settings)
    records = []
    for name, experiment in experiments.items():
        if splits[name] == 'fit':
            dilution = experiment.kwargs['case']['recipes']['continuous'][0]['volume_l_h']  # [1/h], 1 L
            for field, measurement in experiment.measurements.items():
                with (output/'data/noise_1pct.csv').open() as stream:
                    source = [row for row in csv.DictReader(stream)
                              if row['experiment'] == name and row['variable'] == field
                              and int(row['realization']) == settings['realization']]
                times = np.array([float(row['time_h']) for row in source])*HOUR_S  # [s]
                values = np.array([float(row['value_g_l']) for row in source])  # [g/L]
                std = np.array([float(row['sigma_g_l']) for row in source])  # [g/L]
                np.testing.assert_array_equal(measurement.x, times)
                np.testing.assert_array_equal(measurement.values, values)
                np.testing.assert_array_equal(measurement.uncertainty, std)
                records.append((dilution, times, ('substrate', 'biomass').index(field), values, std))
    limit = np.log(2.)  # [-], declared model domain
    reference = minimize_scalar(reference_loss, args=(records,), method='bounded',
        bounds=(-limit, limit), options={'xatol': 1e-8})  # [-], 50000 times below FIT_ATOL
    assert reference.success and report['converged']
    fitted = report['fitted_parameters']['corrections.theta']  # [-]
    np.testing.assert_allclose(fitted, reference.x, rtol=0., atol=FIT_ATOL)
    np.testing.assert_allclose(report['metrics']['fit']['weighted_sse']/2.,
                               reference_loss(fitted, records), rtol=1e-6, atol=1e-6)
    assert report['fitted_experiments'] == ['fit_low_dilution', 'fit_high_dilution']
    assert report['heldout_experiments'] == ['heldout_dilution']
    assert report['metrics']['fit']['count'] == 2*2*len(SAMPLE_H)
    assert report['metrics']['validation']['count'] == 2*len(SAMPLE_H)
    assert report['model']['structural_baseline'] == calibrate.BASELINE_REVISION
    covariance = np.asarray(report['covariance'])  # [-]
    assert covariance.shape == (1, 1) and covariance[0, 0] > 0.
    assert report['covariance_include_mse'] is False
    with (output/'predictions.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    heldout = [row for row in rows if row['split'] == 'validation' and row['field'] == 'substrate']
    expected = independent_prediction(fitted, SAMPLE_H*HOUR_S, .12)[:, 0]  # [g/L], declared holdout D
    np.testing.assert_allclose([float(row['predicted_g_l']) for row in heldout],
                               expected, rtol=0., atol=CONCENTRATION_ATOL)
    with (output/'fit-residual-layout.csv').open() as stream:
        layout = list(csv.DictReader(stream))
    assert {row['experiment'] for row in layout} == set(report['fitted_experiments'])
    np.testing.assert_allclose(sum(float(row['weighted_residual'])**2 for row in layout),
                               report['metrics']['fit']['weighted_sse'], rtol=1e-12, atol=1e-12)


def test_heldout_values_cannot_influence_fit_and_inputs_are_unchanged(workflow, settings):
    output, report = workflow
    path = output/'data/noise_1pct.csv'
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        if row['split'] == 'validation':
            row['value_g_l'] = float(row['value_g_l']) + .1  # [g/L], 100 times the declared absolute noise floor
    rewrite_observations(output, rows)
    experiments, splits, _ = calibrate.prepare_experiments(output, settings)
    original_inputs = {name: deepcopy(dict(exp.kwargs)) for name, exp in experiments.items()}
    original_values = {name: {field: measurement.values.copy() for field, measurement in exp.measurements.items()}
                       for name, exp in experiments.items()}
    fitted = calibrate.fit_experiments(experiments, splits, settings)
    np.testing.assert_array_equal(fitted.params_convg,
                                  [report['fitted_parameters']['corrections.theta']])
    for name, experiment in experiments.items():
        assert experiment.kwargs == original_inputs[name]
        for field, measurement in experiment.measurements.items():
            np.testing.assert_array_equal(measurement.values, original_values[name][field])


def test_field_and_csv_column_reordering_preserves_values_and_fit(workflow, settings):
    output, report = workflow
    with (output/'data/noise_1pct.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    # Move entire experiments/fields, preserving the chronological samples within each.
    rows.sort(key=_calibration_row_key)
    rewrite_observations(output, rows)
    settings['observation_declaration']['fields'] = dict(reversed(list(settings['observation_declaration']['fields'].items())))
    experiments, splits, _ = calibrate.prepare_experiments(output, settings)
    estimator = calibrate.fit_experiments(experiments, splits, settings)
    np.testing.assert_allclose(estimator.params_convg,
        [report['fitted_parameters']['corrections.theta']], rtol=0., atol=FIT_ATOL)
    assert estimator.output_names == ('biomass', 'substrate')
    np.testing.assert_allclose(estimator.info_opt['fun']@estimator.info_opt['fun'],
                               report['metrics']['fit']['weighted_sse'], rtol=1e-6, atol=1e-6)


def test_relocated_assets_replay_and_refit_in_a_fresh_process(workflow, tmp_path):
    output, report = workflow
    relocated = tmp_path/'relocated'
    shutil.move(str(output), relocated)
    before = {str(path.relative_to(relocated)): calibrate.digest(path)
              for path in relocated.rglob('*') if path.is_file()}
    result = subprocess.run([sys.executable, str(EXAMPLE/'calibrate.py'), '--output',
        str(relocated), '--replay', '--refit'], cwd=tmp_path,
        text=True, capture_output=True, timeout=120)
    assert result.returncode == 0, result.stdout+result.stderr
    replayed = json.loads(result.stdout)
    assert replayed['converged']
    assert replayed['prediction_max_abs_g_l'] <= CONCENTRATION_ATOL
    assert replayed['parameter_max_abs'] <= FIT_ATOL
    assert before == {str(path.relative_to(relocated)): calibrate.digest(path)
                      for path in relocated.rglob('*') if path.is_file()}


@pytest.mark.parametrize('theta', [0., TRUTH_THETA])
def test_zero_noise_uses_numerical_scales_and_recovers_truth(tmp_path, settings, short_design, theta):
    settings['noise_label'] = 'noise_0pct'
    settings['truth_theta'] = theta  # [-], neutral and nonneutral controls
    output = tmp_path/'noiseless'
    report = calibrate.run_workflow(output, settings=settings, design=short_design)
    assert report['converged']
    np.testing.assert_allclose(report['fitted_parameters']['corrections.theta'],
                               theta, rtol=0., atol=1e-5)  # [-], prior noiseless qualification budget
    assert report['metrics']['fit']['rmse_g_l'] < CONCENTRATION_ATOL
    assert 'not statistical' in report['covariance_meaning']
    _, manifest = calibrate.selected_data(output/'data', settings)
    assert manifest['truth_theta'] == theta


def test_budget_exhaustion_is_saved_as_nonconvergence(tmp_path, settings, short_design):
    settings['optim_options'].update(max_fun_eval=1, eps_1=0., eps_2=0.)
    output = tmp_path/'budget'
    settings_path, design_path = tmp_path/'settings.json', tmp_path/'design.json'
    settings_path.write_text(json.dumps(settings))
    design_path.write_text(json.dumps(short_design))
    completed = subprocess.run([sys.executable, str(EXAMPLE/'calibrate.py'),
        '--output', str(output), '--settings', str(settings_path), '--design', str(design_path)],
        cwd=ROOT, text=True, capture_output=True, timeout=120)
    assert completed.returncode == 2, completed.stdout+completed.stderr
    report = json.loads(completed.stdout)
    assert report['converged'] is False
    assert report['stop_criterion'] == 'Maximum iterations exceeded'
    assert json.loads((output/'fit-report.json').read_text())['converged'] is False
    assert calibrate.replay(output)['converged'] is False


@pytest.mark.parametrize('asset', ['inputs/thermo.json', 'inputs/fit_low_dilution.json',
                                 'data/noise_1pct.csv', 'predictions.csv', 'calibration-record.json'])
def test_changed_native_inputs_observations_and_predictions_are_rejected(workflow, asset):
    output, _ = workflow
    path = output/asset
    path.write_bytes(path.read_bytes()+b'\n')
    with pytest.raises(ValueError, match='asset checksum mismatch'):
        calibrate.replay(output)


def test_wrong_parameters_and_nonfinite_cached_predictions_fail_replay(workflow):
    output, report = workflow
    saved = deepcopy(report)
    report['fitted_parameters']['corrections.theta'] = 0.  # [-], deliberately wrong neutral prediction
    (output/'fit-report.json').write_text(json.dumps(report))
    with pytest.raises(ValueError, match='native record parameter values disagree'):
        calibrate.replay(output)
    with (output/'predictions.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    rows[0]['predicted_g_l'] = 'nan'
    generate_synthetic.write_rows(output/'predictions.csv', list(rows[0]), rows)
    saved['sha256']['predictions.csv'] = calibrate.digest(output/'predictions.csv')
    (output/'fit-report.json').write_text(json.dumps(saved))
    with pytest.raises(ValueError, match='stored predictions must be finite'):
        calibrate.replay(output)


def test_existing_outputs_and_invalid_selection_are_not_silently_replaced(workflow, settings):
    output, _ = workflow
    checksum = calibrate.digest(output/'fit-report.json')
    with pytest.raises(FileExistsError, match='new or empty'):
        calibrate.run_workflow(output)
    assert calibrate.digest(output/'fit-report.json') == checksum
    settings['realization'] = 2  # [-], one past the short fixture's two realizations
    with pytest.raises(ValueError, match='realization is outside'):
        calibrate.prepare_experiments(output, settings)
    settings['realization'] = 0
    settings['noise_label'] = 'typo'
    with pytest.raises(ValueError, match='noise_label'):
        calibrate.prepare_experiments(output, settings)


def test_exported_fitted_input_runs_through_the_ordinary_forward_api(workflow, settings):
    from PharmaPy.Bioreactors import build_bioreactor, predict_measurements

    output, report = workflow
    model = json.loads((output/'inputs/heldout_dilution-fitted.json').read_text())
    theta = report['fitted_parameters']['corrections.theta']  # [-]
    assert model['parameters']['corrections']['theta'] == theta
    assembly = build_bioreactor(model, thermo_path=output/'inputs/thermo.json')
    observations = {'time_s': {name: SAMPLE_H*HOUR_S
                    for name in settings['observation_declaration']['fields']}}  # [s]
    histories = assembly.solve(report_times=SAMPLE_H*HOUR_S)
    predicted = predict_measurements(assembly, histories,
        settings['observation_declaration'], observations)
    expected = independent_prediction(theta, SAMPLE_H*HOUR_S, .12)  # [g/L]
    for column, name in enumerate(('substrate', 'biomass')):
        np.testing.assert_allclose(predicted['values'][name], expected[:, column],
                                   rtol=0., atol=CONCENTRATION_ATOL)


@pytest.mark.parametrize('working_directory', [ROOT, EXAMPLE], ids=['root', 'example'])
def test_companion_notebook_executes_from_both_launch_directories(tmp_path, working_directory):
    result = subprocess.run([sys.executable, '-c',
        'from pathlib import Path; import sys; '
        'sys.path[:0] = [sys.argv[1], str(Path(sys.argv[1])/"tests")]; '
        'from Bioreactor.test_bioreactor_calibration_workflow import execute_calibration_notebook; '
        'execute_calibration_notebook(Path(sys.argv[2]))',
        str(ROOT), str(tmp_path/'notebook')], cwd=working_directory,
        capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout+result.stderr


@pytest.mark.parametrize('key,value', [('dx_finitediff', 0.),
    ('replay_parameter_atol', float('nan')), ('replay_prediction_atol_g_l', -1.)])
def test_invalid_numerical_settings_are_rejected_before_fitting(workflow, settings, key, value):
    output, _ = workflow
    settings[key] = value  # [theta], [theta], [g/L], respectively; invalid declarations
    with pytest.raises(ValueError, match=f'{key} must be positive and finite'):
        calibrate.prepare_experiments(output, settings)


@pytest.mark.parametrize('name', ['../escape', 'nested/name', 'nested\\name'])
def test_generation_names_cannot_escape_the_output_directory(tmp_path, short_design, name):
    short_design['experiments'][0]['name'] = name
    with pytest.raises(ValueError, match='plain file identifiers'):
        generate_synthetic.generate(tmp_path/'data', design=short_design)
    assert not (tmp_path/'data').exists()


def test_zero_absolute_noise_floor_remains_a_valid_generator_input(tmp_path, short_design):
    short_design['absolute_sd_g_l'] = 0.  # [g/L], purely relative noise is a valid design
    manifest = generate_synthetic.generate(tmp_path/'data', design=short_design)
    assert manifest['design']['absolute_sd_g_l'] == 0.
    with (tmp_path/'data/noise_0pct.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    assert rows and all(float(row['sigma_g_l']) == 0. for row in rows)


def test_provenance_reports_absent_git_without_inventing_a_revision(tmp_path):
    # A genuinely Git-free child environment exercises the optional executable
    # boundary; the test process does not replace its imports or environment.
    import os

    environment = dict(os.environ, PATH=str(tmp_path))
    result = subprocess.run([sys.executable, '-c',
        'from examples.bioreactors.Doyle_Exercise_4_15.calibrate import source_evidence; '
        'import json; print(json.dumps(source_evidence()))'], cwd=ROOT,
        env=environment, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout+result.stderr
    evidence = json.loads(result.stdout)
    assert evidence['revision'] is None and evidence['dirty'] is None
    assert evidence['package_source_sha256'] and evidence['example_source_sha256']


def test_zero_noise_does_not_hide_invalid_source_uncertainty(workflow, settings):
    output, _ = workflow
    path = output/'data/noise_0pct.csv'
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    rows[0]['sigma_g_l'] = -.001  # [g/L], deliberately invalid standard deviation
    generate_synthetic.write_rows(path, list(rows[0]), rows)
    manifest_path = output/'data/manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['sha256'][path.name] = calibrate.digest(path)
    manifest_path.write_text(json.dumps(manifest))
    settings['noise_label'] = 'noise_0pct'
    with pytest.raises(ValueError, match='nonnegative uncertainties'):
        calibrate.prepare_experiments(output, settings)


def test_negative_observation_is_retained_without_clipping(workflow, settings):
    output, _ = workflow
    with (output/'data/noise_1pct.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    row = next(row for row in rows if row['experiment'] == 'fit_low_dilution'
               and row['variable'] == 'substrate' and int(row['realization']) == 0)
    negative = -float(row['value_g_l'])  # [g/L], possible Gaussian observation, not a negative model inventory
    row['value_g_l'] = negative
    rewrite_observations(output, rows)
    experiments, _, _ = calibrate.prepare_experiments(output, settings)
    assert experiments['fit_low_dilution'].measurements['substrate'].values[0] == negative
