"""Calibration record of the Ziegler multi-temperature reactor fit (#351).

Scope: the workshop's multi-temperature Ziegler fit (data of unrecorded
origin, issue #478), the stand-in case adopted by #351 while the December
crystallization (#358) and bioreactor (#361) cases are not on master. The
workflow of ``doc/online_docs/examples/param_estimation_solved.ipynb`` is
reproduced with CVode-integrated reactors: stage 1 fits ln k_i at each
temperature (cells 20-26), whose Arrhenius regression seeds stage 2, the
simultaneous fit of the five experiments '673K' ... '773K' with
reformulated kinetics (cells 28-32). Stage 2 is built from ``Experiment``
and ``Measurement`` objects instead of the notebook's legacy arrays, fitted
through ``SimulationExec`` with per-experiment phase modifiers, captured
with ``CalibrationRecord.from_estimation`` and saved. The reloaded record
must reproduce the observations (experiment and measurement order, x and
values; these data have no missing observations, whose round trip the
analytic case of ``test_records_calibration.py`` covers) and the fitted
values and covariance exactly, rerun through both documented paths
(``SimulationExec.SetParamEstimation`` with a newly built reactor and
``CalibrationRecord.rerun`` with the reactor's ``paramest_wrapper``), and
agree with the unchanged 2023 workshop table
``tests/fixtures/param_estimation_2023.json``.

Data: the headerless CSV files, the initial concentrations and the
pure-component database in ``workshop_data/Case_study_1``. Their original
source, and whether any license covers them, are not recorded upstream
(issue #478), so the record declares origin ``'unknown'`` and states only
verifiable facts; the SHA-256 digests pin the bytes read. The notebook
labels the CSV columns time [s] and concentrations [mol/L]; the initial
concentrations carry no units and are interpreted as [mol/L] because they
are passed as ``LiquidPhase`` ``mole_conc``. The
zero-inventory and missing-Utility warnings of this teaching example remain
visible (issue #172). Assimulo lane; about 10 s, shared by a module-scoped
fixture.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import linregress

from PharmaPy.Kinetics import RxnKinetics, gas_ct
from PharmaPy.ParamEstim import Experiment, Measurement
from PharmaPy.Phases import LiquidPhase
from PharmaPy.Records import (
    CalibrationRecord, DataSource, ModelRevision, Transformation)

# The reactor solves and the CVode-settings check need the optional
# Assimulo backend; skip at collection without it.
assimulo = pytest.importorskip('assimulo')

from assimulo.problem import Explicit_Problem
from assimulo.solvers import CVode

from PharmaPy.Reactors import BatchReactor
from PharmaPy.SimExec import SimulationExec

pytestmark = [pytest.mark.assimulo, pytest.mark.integration, pytest.mark.slow]

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'workshop_data' / 'Case_study_1'
PURE = str(DATA / 'pure_components_ziegler.json')
CONC_INIT = DATA / 'ziegler_conc_init.json'
GOLDEN = ROOT / 'tests' / 'fixtures' / 'param_estimation_2023.json'
REACTIONS = ['A --> B', 'A --> C', 'B --> C', 'B --> D']
# Model output columns of BatchReactor.paramest_wrapper: molar
# concentrations [mol/L] of the participating species in pure-component
# database order (RxnKinetics.partic_species).
OUTPUT_NAMES = ('A', 'B', 'C', 'D')
# Measurement name -> (output column, CSV column); the notebook measures
# columns 1 and 2 (measured_ind=(1, 2)), i.e. species B and C.
MEASUREMENTS = {'C_B': ('B', 1), 'C_C': ('C', 2)}
TEMPERATURES = (673.0, 698.0, 723.0, 748.0, 773.0)  # [K], notebook cell 20
# Notebook cells 3 and 20: stage-1 seeds k_i = 0.1 1/s and E_a,i = 0 J/mol,
# with ln k_i fitted and the activation-energy parameters fixed.
STAGE1_RATE = 0.1  # [1/s]
STAGE1_ACTIVATION = 0.0  # [J/mol]
STAGE1_INITIAL = [1.0, 0.0, 0.0, 0.0, 0.0]  # [mol/L], A, B, C, D, solvent
NUM_REACTIONS = len(REACTIONS)  # [-]
# Notebook cell 30: T_ref is the mean of the five temperatures, 723 K.
TEMP_REF = float(np.mean(TEMPERATURES))  # [K]
# Notebook cell 32: LM step and gradient tolerances eps_1, eps_2 [-].
OPTIM_OPTIONS = {'eps_1': 1e-11, 'eps_2': 1e-11}  # [-]
# Notebook cell 30 builds the stage-2 reactor with the 698 K phase; every
# experiment then overwrites temp and mole_conc through modify_phase.
BUILD_EXPERIMENT = '698K'
# Golden-table tolerances of tests/test_workshop_param_estimation.py:
# FIT_RTOL is CVode's default relative tolerance and PRINT_ATOL half the
# last printed digit of the six-decimal 2023 table.
FIT_RTOL = 1e-6  # [-]
PRINT_ATOL = 0.5e-6  # [-] for ln(A / (1/s)); [K] for Ea/R
# A rerun replays bit-identical data, seeds and options through the same
# deterministic CVode sensitivity solves and LM iterations on the same
# stack, so the fit is reproduced to the last bits; four ulps of relative
# slack absorb any operation reordering (as for the analytic case in
# test_records_calibration.py). The parameters are O(1)-O(10) and nonzero,
# so no absolute slack is needed.
REPLAY_RTOL = 4 * np.finfo(float).eps  # [-]
REPLAY_ATOL = 0.0  # [-], the parameters are dimensionless
# Effective CVode settings of BatchReactor.solve_unit inside
# paramest_wrapper (return_sens=True by default): Assimulo's defaults,
# verified against a fresh CVode instance below, and the sensitivity
# settings solve_unit applies when eval_sens is True.
MODEL_SOLVER = {
    'integrator': 'Assimulo CVode', 'rtol': 1e-6, 'atol': 1e-6,
    'discr': 'BDF', 'iter': 'Newton', 'linear_solver': 'DENSE',
    'sensmethod': 'SIMULTANEOUS', 'report_continuously': True,
    'sundials_opts': None}  # rtol [-]; atol [state units, mol/L]
# Verifiable facts only: where the files are and when they were added. No
# license is claimed, because none is recorded for the data (#478).
SHARING = (
    "Shipped in the PharmaPy repository (workshop_data/Case_study_1, added "
    "in commit 235d76b 'Adding ESCAPE workshop data'); the original source, "
    "and whether the repository LICENSE.md or any other conditions cover "
    "the data, are not recorded upstream (issue #478)")
PURE_SOURCE = Path(PURE)
# One source per CSV file, plus ziegler_conc_init.json and
# pure_components_ziegler.json.
NUM_SOURCES = len(TEMPERATURES) + 2  # [-]


def read_csv(temperature):
    """Read one headerless Ziegler CSV file as the notebook does.

    Parameters
    ----------
    temperature : float
        Experiment temperature [K].

    Returns
    -------
    path : pathlib.Path
        CSV file.
    data : numpy.ndarray
        Shape (n, 3): time [s], then C_B and C_C [mol/L].
    """
    path = DATA / f'ziegler_data_{temperature:.0f}K.csv'
    return path, np.genfromtxt(path, delimiter=',')


def build_reactor(kinetics, initial):
    """Build the notebook's batch reactor around a kinetics object.

    Parameters
    ----------
    kinetics : RxnKinetics
        Reaction kinetics.
    initial : dict
        ``LiquidPhase`` keywords ``temp`` [K] and ``mole_conc`` [mol/L].

    Returns
    -------
    SimulationExec
        One-unit flowsheet ``R01``.
    """
    simulation = SimulationExec(PURE, flowsheet='R01')
    simulation.R01 = BatchReactor()
    simulation.R01.Kinetics = kinetics
    simulation.R01.Phases = LiquidPhase(PURE, **initial, name_solv='solvent')
    return simulation


def stage1_seeds():
    """Fit ln k_i per temperature and regress Arrhenius seeds (cells 20-30).

    Returns
    -------
    numpy.ndarray
        Pre-exponential factors A_i [1/s], shape (4,).
    numpy.ndarray
        Activation energies E_a,i [J/mol], shape (4,).
    """
    log_rates = []  # [-], ln(k_i(T) / (1/s)) per temperature
    for temperature in TEMPERATURES:
        _, data = read_csv(temperature)
        kinetics = RxnKinetics(PURE, rxn_list=REACTIONS,
                               k_params=[STAGE1_RATE] * NUM_REACTIONS,
                               ea_params=[STAGE1_ACTIVATION] * NUM_REACTIONS,
                               reformulate_kin=True)
        simulation = build_reactor(kinetics, {'temp': temperature,
                                              'mole_conc': STAGE1_INITIAL})
        simulation.SetParamEstimation(
            x_data=data[:, 0], y_data=data[:, 1:], measured_ind=(1, 2),
            optimize_flags=[True] * NUM_REACTIONS + [False] * NUM_REACTIONS)
        fitted, _, _ = simulation.EstimateParams(verbose=False)
        log_rates.append(fitted)
    inverse_temperature = 1 / np.array(TEMPERATURES)  # [1/K]
    fits = [linregress(inverse_temperature, column)
            for column in np.array(log_rates).T]
    log_factors = np.array([fit.intercept for fit in fits])  # [-]
    ea_over_r = -np.array([fit.slope for fit in fits])  # [K]
    return np.exp(log_factors), gas_ct * ea_over_r


def initial_conditions():
    """Return the per-experiment phase modifiers of notebook cell 28.

    Returns
    -------
    dict
        Experiment name to ``{'temp': T [K], 'mole_conc': [mol/L] of A, B,
        C, D and solvent}``.
    """
    concentrations = json.loads(CONC_INIT.read_text())
    return {f'{temperature:.0f}K': {
        'temp': temperature,
        'mole_conc': concentrations[f'{temperature:.0f}K']}
        for temperature in TEMPERATURES}


def make_experiments():
    """Build one ``Experiment`` per temperature from the CSV files.

    Returns
    -------
    dict
        Experiment name ('673K', ...) to ``Experiment`` with measurements
        ``C_B`` and ``C_C`` [mol/L] on the CSV time column [s]. The phase
        modifiers are bound by ``SetParamEstimation``.
    """
    experiments = {}
    for temperature in TEMPERATURES:
        _, data = read_csv(temperature)
        experiments[f'{temperature:.0f}K'] = Experiment({
            name: Measurement(field, data[:, 0], data[:, column],
                              units='mol/L', basis='molar concentration',
                              x_units='s')
            for name, (field, column) in MEASUREMENTS.items()})
    return experiments


def provenance():
    """Describe the data sources and transformations of the case.

    Returns
    -------
    sources : list of DataSource
        One source per CSV file, one for the initial concentrations and one
        for the pure-component database, each with the SHA-256 digest of
        the bytes read.
    transformations : list of Transformation
        How each source enters the record.
    """
    sources, transformations = [], []
    for temperature in TEMPERATURES:
        path, _ = read_csv(temperature)
        sources.append(DataSource(
            path.name, 'unknown', SHARING,
            description=(
                f"Ziegler case of the ESCAPE workshop at {temperature:.0f} K: "
                "headerless CSV whose columns the workshop notebook labels "
                "time [s] and the molar concentrations of B and C [mol/L]"),
            uri=path.relative_to(ROOT).as_posix(),
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            role='calibration'))
        transformations.append(Transformation(
            "Read with numpy.genfromtxt(delimiter=','); column 0 is the time "
            "[s] of both measurements, columns 1 and 2 are C_B and C_C "
            "[mol/L] of experiment "
            f"'{temperature:.0f}K'; values used as stored, without unit "
            "conversion, smoothing or outlier removal",
            fields=tuple(MEASUREMENTS), source=path.name))
    sources.append(DataSource(
        CONC_INIT.name, 'unknown', SHARING,
        description=("Initial concentrations of A, B, C, D and solvent of "
                     "each Ziegler experiment; the file states no units. "
                     "They are passed as LiquidPhase mole_conc and are "
                     "therefore interpreted as [mol/L] (issue #478)"),
        uri=CONC_INIT.relative_to(ROOT).as_posix(),
        sha256=hashlib.sha256(CONC_INIT.read_bytes()).hexdigest(),
        role='initial conditions'))
    transformations.append(Transformation(
        "Entry '<T>K' passed unchanged as modify_phase 'mole_conc' "
        "(interpreted as [mol/L]) of experiment '<T>K', with 'temp' = T [K] "
        "from the experiment name",
        source=CONC_INIT.name))
    sources.append(DataSource(
        PURE_SOURCE.name, 'unknown', SHARING,
        description=("Pure-component property database of A, B, C, D and "
                     "solvent; it fixes the species and therefore the order "
                     "of the model output columns A, B, C, D"),
        uri=PURE_SOURCE.relative_to(ROOT).as_posix(),
        sha256=hashlib.sha256(PURE_SOURCE.read_bytes()).hexdigest(),
        role='model input'))
    transformations.append(Transformation(
        "Used unchanged as the path of LiquidPhase and RxnKinetics",
        source=PURE_SOURCE.name))
    return sources, transformations


MODEL = ModelRevision(
    'BatchReactor.paramest_wrapper',
    description=(
        "Model code: the PharmaPy checkout under test; no revision is "
        "recorded here; users should supply their commit as revision. "
        "Isothermal BatchReactor (default isothermal=True) with RxnKinetics "
        "of A --> B, A --> C, B --> C, B --> D (elementary first-order "
        "rates) and reformulated Arrhenius kinetics k_i(T) = exp(phi_1,i + "
        "exp(phi_2,i) * (1/T_ref - 1/T)), T_ref = 723 K (mean of the five "
        "temperatures): phi_1,i = ln(k_i(T_ref) / (1/s)) and phi_2,i = "
        "ln((E_a,i/R) / K). Parameter order phi_1,1..4, phi_2,1..4."))


@pytest.fixture(scope='module')
def ziegler():
    """Fit stage 1 and stage 2 once and capture the stage-2 record.

    Returns
    -------
    dict
        ``simulation`` (fitted ``SimulationExec``), ``experiments`` (the
        caller's ``Experiment`` objects), ``record`` and ``fit`` (the
        parameters, covariance and info returned by ``EstimateParams``).
    """
    factors, activation = stage1_seeds()  # [1/s], [J/mol]
    kinetics = RxnKinetics(PURE, rxn_list=REACTIONS, k_params=factors,
                           ea_params=activation, reformulate_kin=True,
                           temp_ref=TEMP_REF)
    modifiers = initial_conditions()
    simulation = build_reactor(kinetics, modifiers[BUILD_EXPERIMENT])
    experiments = make_experiments()
    simulation.SetParamEstimation(x_data=experiments,
                                  phase_modifiers=modifiers,
                                  output_names=OUTPUT_NAMES)
    fit = simulation.EstimateParams(optim_options=dict(OPTIM_OPTIONS),
                                    verbose=False)
    sources, transformations = provenance()
    names = simulation.ParamInst.name_params_total
    record = CalibrationRecord.from_estimation(
        simulation.ParamInst, experiments, case='ziegler-batch-reactor',
        parameter_units={name: '-' for name in names}, model=MODEL,
        sources=sources, transformations=transformations,
        model_solver=dict(MODEL_SOLVER, assimulo=assimulo.__version__),
        description=("Simultaneous fit of the five Ziegler experiments of "
                     "the parameter-estimation workshop, seeded by the "
                     "per-temperature Arrhenius regression"))
    return {'simulation': simulation, 'experiments': experiments,
            'record': record, 'fit': fit}


def assert_bits_equal(actual, expected):
    """Assert two float arrays are equal bit for bit, NaN included.

    Parameters
    ----------
    actual, expected : array_like
        Float arrays of equal shape.
    """
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    assert actual.shape == expected.shape
    np.testing.assert_array_equal(actual.view(np.int64),
                                  expected.view(np.int64))


def test_model_solver_entry_matches_cvode_defaults():
    solver = CVode(Explicit_Problem(lambda time, state: -state,
                                    np.array([1.0])))
    for key in ('rtol', 'discr', 'iter', 'linear_solver'):
        assert solver.options[key] == MODEL_SOLVER[key]
    np.testing.assert_array_equal(solver.options['atol'],
                                  [MODEL_SOLVER['atol']])


def test_capture_records_fit_inputs_and_provenance(ziegler):
    record, estimator = ziegler['record'], ziegler['simulation'].ParamInst
    names = list(estimator.name_params_total)
    assert [item.name for item in record.parameters] == names
    assert len(names) == 2 * NUM_REACTIONS
    assert all(item.units == '-' and item.optimized
               for item in record.parameters)
    assert_bits_equal([item.seed for item in record.parameters],
                      estimator.param_seed)
    assert_bits_equal(record.fitted_values, estimator.params_convg)
    assert_bits_equal(record.fitted_values, ziegler['fit'][0])
    # optimize_fn returns the include_mse=True covariance that the record
    # stores by default.
    assert record.covariance_include_mse is True
    assert_bits_equal(record.covariance, ziegler['fit'][1])

    assert record.solver.method == 'LM'
    assert record.solver.options == OPTIM_OPTIONS
    assert record.solver.derivatives == {
        'jacobian': 'model_sensitivities', 'dx_finitediff': 'default'}
    assert record.solver.model_solver['sundials_opts'] is None
    assert record.output_names == OUTPUT_NAMES
    # The model description states an isothermal reactor and the stated
    # output columns; check both on the fitted unit.
    reactor = ziegler['simulation'].R01
    assert reactor.isothermal is True
    assert tuple(reactor.Kinetics.partic_species) == OUTPUT_NAMES

    # The phase modifiers that SetParamEstimation bound are recorded per
    # experiment, in the declared experiment order.
    modifiers = initial_conditions()
    assert list(record.experiments) == list(modifiers)
    for name, experiment in record.experiments.items():
        assert experiment.args == ()
        assert dict(experiment.kwargs) == {'modify_phase': modifiers[name],
                                           'modify_controls': {},
                                           'run_args': {}}
    assert len(record.sources) == NUM_SOURCES
    assert [item.uri for item in record.sources] == [
        read_csv(temperature)[0].relative_to(ROOT).as_posix()
        for temperature in TEMPERATURES] + [
        'workshop_data/Case_study_1/ziegler_conc_init.json',
        'workshop_data/Case_study_1/pure_components_ziegler.json']
    # One transformation per source, in source order.
    assert [item.source for item in record.transformations] == [
        item.name for item in record.sources]
    for source in record.sources:
        assert source.origin == 'unknown'
        assert source.sha256 == hashlib.sha256(
            (ROOT / source.uri).read_bytes()).hexdigest()
        # Verifiable facts only: the open issue, and no claimed license.
        assert 'issue #478' in source.sharing
        assert 'distributed under' not in source.sharing
        assert 'licensed' not in source.sharing.lower()
    # Model code identity: the record names no revision; the description
    # says the checkout under test is the model code.
    assert record.model.revision is None
    assert 'PharmaPy checkout under test' in record.model.description
    assert [(field.experiment, field.name) for field in record.fields] == [
        (name, measurement) for name in modifiers
        for measurement in MEASUREMENTS]


def test_saved_record_reloads_observations_and_fit_unchanged(ziegler,
                                                             tmp_path):
    record = ziegler['record']
    path = tmp_path / 'ziegler.json'
    record.save(path)
    loaded = CalibrationRecord.load(path)
    assert loaded == record
    assert path.read_text(encoding='utf-8') == record.to_json()

    assert list(loaded.experiments) == [f'{temperature:.0f}K'
                                        for temperature in TEMPERATURES]
    for temperature in TEMPERATURES:
        name = f'{temperature:.0f}K'
        _, data = read_csv(temperature)  # independent of the record
        measurements = loaded.experiments[name].measurements
        assert list(measurements) == list(MEASUREMENTS)
        for measurement_name, (field, column) in MEASUREMENTS.items():
            measurement = measurements[measurement_name]
            assert measurement.field == field
            assert (measurement.units, measurement.basis,
                    measurement.x_units) == ('mol/L', 'molar concentration',
                                             's')
            assert_bits_equal(measurement.x, data[:, 0])
            assert_bits_equal(measurement.values, data[:, column])
            assert measurement.uncertainty is None
        # These data have no missing observations; the null/NaN round trip
        # is covered by the analytic case (test_records_calibration.py).
        assert not np.isnan(data).any()

    assert_bits_equal(loaded.fitted_values, record.fitted_values)
    assert_bits_equal(loaded.covariance, record.covariance)
    assert_bits_equal([item.seed for item in loaded.parameters],
                      [item.seed for item in record.parameters])
    assert loaded.model == record.model
    assert loaded.model.revision is None
    assert loaded.model.description == MODEL.description
    for reloaded, captured in zip(loaded.fields, record.fields):
        assert reloaded.identity == captured.identity
        assert_bits_equal(reloaded.x, captured.x)
        assert_bits_equal(reloaded.values, captured.values)


def test_reloaded_record_reruns_through_simulation_exec(ziegler, tmp_path):
    path = tmp_path / 'ziegler.json'
    ziegler['record'].save(path)
    loaded = CalibrationRecord.load(path)
    inputs = loaded.estimation_inputs()
    seeds = inputs['param_seed']  # [-], phi_1,1..4 then phi_2,1..4

    # Case construction is the user's: rebuild the reactor with kinetics
    # whose A [1/s] and E_a [J/mol] invert the reformulation at the
    # recorded seeds. SetParamEstimation then sets the exact seed vector.
    ea_over_r = np.exp(seeds[NUM_REACTIONS:])  # [K]
    kinetics = RxnKinetics(
        PURE, rxn_list=REACTIONS,
        k_params=np.exp(seeds[:NUM_REACTIONS] + ea_over_r / TEMP_REF),
        ea_params=gas_ct * ea_over_r, reformulate_kin=True,
        temp_ref=TEMP_REF)
    replay = build_reactor(kinetics, initial_conditions()[BUILD_EXPERIMENT])
    replay.SetParamEstimation(**inputs)
    assert_bits_equal(replay.R01.Kinetics.concat_params(), seeds)
    assert_bits_equal(replay.ParamInst.param_seed, seeds)

    replay.EstimateParams(verbose=False, **loaded.optimizer_inputs())
    difference = loaded.check_rerun(replay.ParamInst, rtol=REPLAY_RTOL,
                                    atol=REPLAY_ATOL)
    assert difference.shape == (2 * NUM_REACTIONS,)


def test_reloaded_record_reruns_directly(ziegler, tmp_path):
    path = tmp_path / 'ziegler.json'
    ziegler['record'].save(path)
    loaded = CalibrationRecord.load(path)
    # Any kinetics with the case's structure: paramest_wrapper sets the
    # parameter vector it receives before every solve.
    reactor = build_reactor(
        RxnKinetics(PURE, rxn_list=REACTIONS,
                    k_params=[STAGE1_RATE] * NUM_REACTIONS,
                    ea_params=[STAGE1_ACTIVATION] * NUM_REACTIONS,
                    reformulate_kin=True, temp_ref=TEMP_REF),
        initial_conditions()[BUILD_EXPERIMENT]).R01
    estimator = loaded.rerun(reactor.paramest_wrapper, rtol=REPLAY_RTOL,
                             atol=REPLAY_ATOL)
    assert_bits_equal(estimator.param_seed,
                      [item.seed for item in loaded.parameters])


def test_fit_matches_the_2023_workshop_table(ziegler):
    record = ziegler['record']
    golden = json.loads(GOLDEN.read_text())
    assert golden['reactions'] == REACTIONS
    phi_1 = record.fitted_values[:NUM_REACTIONS]  # [-]
    phi_2 = record.fitted_values[NUM_REACTIONS:]  # [-]
    # Notebook cells 36 and 41: E_a = exp(phi_2) R, A = exp(E_a/(R T_ref)
    # + phi_1); the table lists ln(A / (1/s)) [-] and E_a/R [K].
    activation = np.exp(phi_2) * gas_ct  # [J/mol]
    factor = np.exp(activation / gas_ct / TEMP_REF + phi_1)  # [1/s]
    table = np.column_stack((np.log(factor), activation / gas_ct))
    np.testing.assert_allclose(table, golden['fit'], rtol=FIT_RTOL,
                               atol=PRINT_ATOL)
