"""Calibration provenance records: serialization, capture and rerun (#351).

Scope: ``PharmaPy.Records.CalibrationRecord`` and its component records
(``DataSource``, ``Transformation``, ``ParameterRecord``, ``ModelRevision``,
``SolverSettings``, ``FieldIdentity``/``FieldRecord``); exact JSON
round trips of ``Experiment``/``Measurement`` observations, callback
arguments and fitted results; capture of a fitted ``ParameterEstimation``
with ``from_estimation`` and its consistency checks; the documented
reload-and-rerun path; atomic saving; and validation messages.

Fixture: the two-experiment first-order decay A -> B with an adiabatic
temperature rise of ``doc/online_docs/examples/experiment_measurement.rst``
(hot/cold runs, c_A [mol/L] and T [K], one missing observation and
per-sample uncertainties). Its data are a synthetic verification fixture,
recorded with origin ``'synthetic'``; they are not evidence of an empirical
calibration. Expected values are written out by hand or come from the
closed-form model, never from the record code. Core lane; PharmaPy's
Levenberg-Marquardt solver fits the one-parameter model in milliseconds.
"""

import copy
import json
import os
from pathlib import Path, PureWindowsPath
import pickle
import stat
import subprocess
import sys
import textwrap
import warnings

import numpy as np
import pandas as pd
import pytest

from PharmaPy.ParamEstim import Experiment, Measurement, ParameterEstimation
from PharmaPy.Records import (
    RECORD_SCHEMA, RECORD_SCHEMA_VERSION, CalibrationRecord, DataSource,
    FieldIdentity, FieldRecord, ModelRevision, ParameterRecord,
    RerunMismatchError, SolverSettings, Transformation, _effective_options,
    fields_to_array, fields_to_mapping)

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[1]
SEED_RATE = 0.3  # [1/s], the documented seed of k
# A non-default LM option (the default max_fun_eval is 100) that must be
# recorded verbatim; it exceeds the nine evaluations this one-parameter fit
# needs, so it does not change the fit.
LM_MAX_FUN_EVAL = 60  # [-]
# The rerun replays bit-identical inputs through the same deterministic LM
# iterations, so the fitted rate is reproduced exactly; four ulps of
# relative slack absorb any platform-dependent operation ordering.
REPLAY_RTOL = 4 * np.finfo(float).eps  # [-]
REPLAY_ATOL = 0.0  # [1/s]
# Production and closed-form predictions evaluate the same IEEE expressions
# in a different association order; a few ulps of O(300 K) values bound the
# difference.
ROUNDOFF_RTOL = 1e-12  # [-]
ROUNDOFF_ATOL = 1e-12  # [mol/L or K]
# A deliberate offset of the recorded rate, far above REPLAY_RTOL * k.
RATE_OFFSET = 1e-3  # [1/s]
DIGEST = 'ab' * 32  # [-], 64 lowercase hexadecimal digits of SHA-256

# Hand-written copy of the documented data:
# experiment -> (initial c_A [mol/L], rise [K*L/mol], initial T [K]).
CALLBACK = {'hot': (2.0, 20.0, 300.0), 'cold': (1.0, 10.0, 290.0)}
HOT_CA_TIME = [0.0, 1.0, 2.0, 4.0]  # [s]
HOT_CA = [2.010, 1.200, 0.750, 0.265]  # [mol/L]
HOT_T_TIME = [2.0]  # [s]
HOT_T = [325.6]  # [K]
COLD_CA_TIME = [0.5, 1.5, 3.0]  # [s]
COLD_CA = [0.785, np.nan, 0.220]  # [mol/L], NaN = missing
COLD_CA_STD = [0.01, 0.01, 0.02]  # [mol/L]
HOT_CA_STD = 0.02  # [mol/L]
HOT_T_STD = 0.5  # [K]
# Model grids: the multiset union of each experiment's measurement grids.
GRID = {'hot': [0.0, 1.0, 2.0, 4.0], 'cold': [0.5, 1.5, 3.0]}  # [s]

# Arbitrary values of the manually built record. They exercise the codec
# and the record entries only; they are not fitted or measured results.
MANUAL_FITTED_RATE = 0.5011  # [1/s]
MANUAL_RATE_BOUNDS = (0.0, 5.0)  # [1/s]
MANUAL_RISE = 20.0  # [K*L/mol], fixed parameter
MANUAL_COVARIANCE = [[1.9e-5]]  # [1/s**2]
MANUAL_OBJECTIVE = 1.25  # [-]
MANUAL_IPOPT_OPTIONS = {'tol': 1e-10, 'max_iter': 50,
                        'scaling': (1.0, np.float64(2.0))}  # [-]
MANUAL_MODEL_SOLVER = {'rtol': 1e-8, 'atol': [1e-10, 1e-9]}  # [-], [mol/L]
MANUAL_FACTOR = {'value': 0.4, 'units': 'mol/L'}  # conversion factor
COMPOSITION_TIME = [0.0, 2.0, 2.0, 5.0]  # [s], with a replicate time
COMPOSITION = [[0.9, 0.1, 0.0], [0.7, np.nan, 0.1], [0.6, 0.25, 0.15],
               [0.5, 0.3, 0.2]]  # [-], mole fractions A, B, solvent
DENSITY_TIME = [1.0, 3.0]  # [min]
DENSITY = [np.nan, 1012.5]  # [kg/m**3]
PURITY_TIME = 7.5  # [s]
PURITY = 0.995  # [-]


def decay(params, time_s, initial_mol_l, rise_k_l_mol=0.0, temp0_k=300.0):
    """Return c_A and T of a first-order A -> B adiabatic batch reaction.

    Parameters
    ----------
    params : numpy.ndarray
        Rate constant k, shape (1,) [1/s].
    time_s : numpy.ndarray
        Model grid, shape (n_times,) [s].
    initial_mol_l : float
        Initial concentration of A [mol/L].
    rise_k_l_mol : float, optional
        Adiabatic temperature rise per converted concentration [K*L/mol].
    temp0_k : float, optional
        Initial temperature [K].

    Returns
    -------
    numpy.ndarray
        Shape (n_times, 2); column 0 is c_A [mol/L], column 1 is T [K].
    """
    converted = initial_mol_l * (1 - np.exp(-params[0] * time_s))  # [mol/L]
    return np.column_stack((initial_mol_l - converted,
                            temp0_k + rise_k_l_mol * converted))


def decay_rise_first(params, time_s, initial_mol_l, temp0_k=300.0):
    """Return c_A and T with the temperature rise as the first parameter.

    Parameters
    ----------
    params : numpy.ndarray
        Shape (2,): rise [K*L/mol], then rate constant k [1/s].
    time_s : numpy.ndarray
        Model grid, shape (n_times,) [s].
    initial_mol_l : float
        Initial concentration of A [mol/L].
    temp0_k : float, optional
        Initial temperature [K].

    Returns
    -------
    numpy.ndarray
        Shape (n_times, 2); column 0 is c_A [mol/L], column 1 is T [K].
    """
    return decay(params[1:], time_s, initial_mol_l, params[0], temp0_k)


RISE_PARAMETER = 20.0  # [K*L/mol], the hot-run rise as a fixed parameter


def experiments_without_rise():
    """Build the documented experiments without the rise keyword.

    Returns
    -------
    dict
        Experiment name to ``Experiment``; the temperature rise is then a
        parameter of ``decay_rise_first`` instead of a callback keyword.
    """
    return {name: Experiment(experiment.measurements, args=experiment.args,
                             kwargs={key: value for key, value
                                     in experiment.kwargs.items()
                                     if key != 'rise_k_l_mol'})
            for name, experiment in make_experiments().items()}


def closed_form(rate, experiment, time):
    """Evaluate c_A and T of the decay model point by point.

    Parameters
    ----------
    rate : float
        Rate constant k [1/s].
    experiment : str
        ``'hot'`` or ``'cold'``; selects ``CALLBACK``.
    time : float
        Time [s].

    Returns
    -------
    tuple of float
        c_A [mol/L] and T [K].
    """
    initial, rise, temp0 = CALLBACK[experiment]  # [mol/L], [K*L/mol], [K]
    remaining = initial * np.exp(-rate * time)  # [mol/L]
    return remaining, temp0 + rise * (initial - remaining)


# Observed entries of the guide data, written out by hand: (experiment,
# column (0 = c_A, 1 = T), time [s], observation [mol/L or K], standard
# deviation [mol/L or K]). The missing cold c_A sample at 1.5 s is absent.
OBSERVED_ENTRIES = [
    ('hot', 0, 0.0, 2.010, 0.02), ('hot', 0, 1.0, 1.200, 0.02),
    ('hot', 0, 2.0, 0.750, 0.02), ('hot', 0, 4.0, 0.265, 0.02),
    ('hot', 1, 2.0, 325.6, 0.5),
    ('cold', 0, 0.5, 0.785, 0.01), ('cold', 0, 3.0, 0.220, 0.02)]
NUM_PARAMETERS = 1  # [-], k only


def standardized_residuals(rate):
    """Return the closed-form standardized residuals of the guide data.

    Parameters
    ----------
    rate : float
        Rate constant k [1/s].

    Returns
    -------
    numpy.ndarray
        ``(model - data) / sigma`` [-] of the observed entries, in
        ``OBSERVED_ENTRIES`` order.
    """
    return np.array([
        (closed_form(rate, experiment, time)[column] - value) / sigma
        for experiment, column, time, value, sigma in OBSERVED_ENTRIES])


def make_experiments(order=('hot', 'cold'), hot_order=('c_A', 'T'),
                     uncertain=True):
    """Build the documented experiments in a chosen order.

    Parameters
    ----------
    order : tuple of str, optional
        Experiment order.
    hot_order : tuple of str, optional
        Measurement order of the hot experiment.
    uncertain : bool, optional
        Whether the measurements declare their uncertainties.

    Returns
    -------
    dict
        Experiment name to ``Experiment``.
    """
    def std(value):
        """Return an uncertainty if declared.

        Parameters
        ----------
        value : float or list
            Standard deviation [mol/L or K].

        Returns
        -------
        float, list or None
            ``value`` when ``uncertain``, otherwise None.
        """
        return value if uncertain else None

    hot_measurements = {
        'c_A': Measurement(0, HOT_CA_TIME, HOT_CA,
                           uncertainty=std(HOT_CA_STD), units='mol/L',
                           basis='molar concentration'),
        'T': Measurement(1, HOT_T_TIME, HOT_T, uncertainty=std(HOT_T_STD),
                         units='K')}
    built = {
        'hot': Experiment({name: hot_measurements[name]
                           for name in hot_order},
                          args=(CALLBACK['hot'][0],),
                          kwargs={'rise_k_l_mol': CALLBACK['hot'][1]}),
        'cold': Experiment(
            {'c_A': Measurement(0, COLD_CA_TIME, COLD_CA,
                                uncertainty=std(COLD_CA_STD), units='mol/L',
                                basis='molar concentration')},
            args=(CALLBACK['cold'][0],),
            kwargs={'rise_k_l_mol': CALLBACK['cold'][1],
                    'temp0_k': CALLBACK['cold'][2]})}
    return {name: built[name] for name in order}


SOURCES = (
    DataSource('guide', 'synthetic',
               'synthetic values written for the user guide; no external '
               'data',
               description='Synthetic decay data of the user guide',
               uri='doc/online_docs/examples/experiment_measurement.rst',
               sha256=DIGEST, role='calibration'),
)


def capture(estimator, experiments, **options):
    """Capture a fitted estimator with the default provenance.

    Parameters
    ----------
    estimator : ParameterEstimation
        Fitted estimator.
    experiments : dict
        Its experiments.
    **options
        Further ``from_estimation`` keywords.

    Returns
    -------
    CalibrationRecord
        Captured record.
    """
    arguments = {'case': 'decay-guide', 'parameter_units': {'k': '1/s'},
                 'model': ModelRevision('decay', revision='guide-example'),
                 'sources': SOURCES}
    arguments.update(options)
    return CalibrationRecord.from_estimation(estimator, experiments,
                                             **arguments)


def fitted_record(experiments=None, **options):
    """Fit the documented case and capture it in a record.

    Parameters
    ----------
    experiments : dict, optional
        Experiments to fit; the documented order by default.
    **options
        Further ``from_estimation`` keywords.

    Returns
    -------
    tuple
        ``(record, estimator, experiments)``.
    """
    experiments = make_experiments() if experiments is None else experiments
    estimator = ParameterEstimation(decay, [SEED_RATE], experiments,
                                    name_params=['k'])
    estimator.optimize_fn(verbose=False)
    return capture(estimator, experiments, **options), estimator, experiments


# Callback arguments of every supported type, with asymmetric shapes and a
# big-endian array. Arbitrary codec values in model-defined units.
EXOTIC_ARGS = (
    2.0, 3, True, None, 'label', [1, 2.5, [False]], (1.0, ('nested',)),
    np.float32(0.1), np.int64(-7), np.bool_(False),
    np.arange(6, dtype=np.int32).reshape(2, 3),
    np.array([[0.1, 0.2, 0.3]]), (), np.array([1.5, -2.0], dtype='>f8'),
    np.float16(0.5), np.array([True, False]),
)
EXOTIC_KWARGS = {'zeta': {'b': (1, 2), 'a': [np.uint8(3)]},
                 'alpha': np.array([], dtype=np.float64).reshape(0, 2)}


def manual_record(order=('cold', 'hot'), hot_order=('T', 'c_A')):
    """Build a record directly, with every optional entry populated.

    Parameters
    ----------
    order : tuple of str, optional
        Experiment order.
    hot_order : tuple of str, optional
        Measurement order of the hot experiment.

    Returns
    -------
    CalibrationRecord
        Record with exotic callback arguments, two parameters (one fixed),
        a transformation, solver settings, stored fields and a covariance.
    """
    experiments = make_experiments(order, hot_order)
    experiments['hot'] = Experiment(
        experiments['hot'].measurements, args=EXOTIC_ARGS,
        kwargs=EXOTIC_KWARGS)
    composition = FieldRecord(
        FieldIdentity('x_liq', 'state', '-', basis='mole fraction'),
        COMPOSITION_TIME, COMPOSITION, columns=('A', 'B', 'solvent'),
        experiment='hot')
    density = FieldRecord(
        FieldIdentity('rho', 'algebraic', 'kg/m**3', basis='mass density'),
        DENSITY_TIME, DENSITY, x_units='min')
    purity = FieldRecord(
        FieldIdentity('purity', 'quality_attribute', '-',
                      description='Final purity'),
        PURITY_TIME, PURITY, experiment='cold')
    return CalibrationRecord(
        'manual', experiments,
        [ParameterRecord('k', '1/s', SEED_RATE,
                         lower=MANUAL_RATE_BOUNDS[0],
                         upper=MANUAL_RATE_BOUNDS[1],
                         fitted=MANUAL_FITTED_RATE,
                         description='rate constant'),
         ParameterRecord('rise', 'K*L/mol', MANUAL_RISE, optimized=False)],
        SOURCES + (DataSource('lab', 'experimental', 'internal only',
                              citation='Notebook 12', revision='r2'),),
        ModelRevision('decay', revision='abc123', pharmapy_version=None,
                      description='closed form'),
        description='every optional entry',
        output_names=('c_A', 'T'),
        transformations=[
            Transformation('absorbance to c_A [mol/L] with factor [mol/L]',
                           fields=('c_A',), source='lab',
                           parameters={'factor': MANUAL_FACTOR}),
            Transformation('dropped an outlier at 1.5 s')],
        solver=SolverSettings('IPOPT', options=MANUAL_IPOPT_OPTIONS,
                              model_solver=MANUAL_MODEL_SOLVER,
                              outcome={'objective': MANUAL_OBJECTIVE}),
        fields=[composition, density, purity],
        covariance=MANUAL_COVARIANCE, covariance_include_mse=False)


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


def assert_same_value(actual, expected):
    """Assert equal value and exact type, recursing into containers.

    Parameters
    ----------
    actual, expected : object
        Values supported by the record codec.
    """
    assert type(actual) is type(expected)
    if isinstance(expected, np.ndarray):
        assert actual.dtype == expected.dtype  # includes byte order
        assert actual.dtype.str == expected.dtype.str
        assert actual.shape == expected.shape
        np.testing.assert_array_equal(actual, expected)
    elif isinstance(expected, np.generic):
        assert actual.dtype == expected.dtype
        assert actual == expected
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for item, reference in zip(actual, expected):
            assert_same_value(item, reference)
    elif isinstance(expected, dict):
        assert list(actual) == list(expected)
        for key in expected:
            assert_same_value(actual[key], expected[key])
    else:
        assert actual == expected


def record_arguments(**changes):
    """Return valid ``CalibrationRecord`` arguments with some replaced.

    Parameters
    ----------
    **changes
        Keyword arguments replacing the defaults.

    Returns
    -------
    dict
        Keyword arguments for ``CalibrationRecord``.
    """
    arguments = {
        'case': 'decay', 'experiments': make_experiments(),
        'parameters': [ParameterRecord('k', '1/s', SEED_RATE)],
        'sources': SOURCES, 'model': ModelRevision('decay')}
    arguments.update(changes)
    return arguments


# ------------------------------------------------------------------ round trip


def test_file_round_trip_restores_every_entry(tmp_path):
    record = manual_record()
    path = tmp_path / 'record.json'
    record.save(path)
    loaded = CalibrationRecord.load(path)

    assert loaded == record
    assert loaded.case == 'manual'
    assert loaded.description == 'every optional entry'
    assert loaded.output_names == ('c_A', 'T')

    experiments = loaded.experiments
    assert list(experiments) == ['cold', 'hot']
    hot, cold = experiments['hot'], experiments['cold']
    assert hot.measurement_names == ('T', 'c_A')
    assert_bits_equal(hot.measurements['c_A'].x, HOT_CA_TIME)
    assert_bits_equal(hot.measurements['c_A'].values, HOT_CA)
    assert_bits_equal(hot.measurements['c_A'].uncertainty, [HOT_CA_STD] * 4)
    assert hot.measurements['c_A'].basis == 'molar concentration'
    assert_bits_equal(hot.measurements['T'].values, HOT_T)
    assert hot.measurements['T'].field == 1
    assert hot.measurements['T'].basis is None
    assert (hot.measurements['T'].units, hot.x_units) == ('K', 's')
    measured = cold.measurements['c_A']
    assert_bits_equal(measured.values, COLD_CA)
    assert np.isnan(measured.values).tolist() == [False, True, False]
    assert_bits_equal(measured.uncertainty, COLD_CA_STD)
    assert_same_value(hot.args, EXOTIC_ARGS)
    assert_same_value(dict(hot.kwargs), EXOTIC_KWARGS)
    assert list(hot.kwargs) == ['zeta', 'alpha']
    assert_same_value(cold.args, (1.0,))
    assert_same_value(dict(cold.kwargs),
                      {'rise_k_l_mol': 10.0, 'temp0_k': 290.0})

    k, rise = loaded.parameters
    assert (k.name, k.units, k.seed, k.lower, k.upper, k.optimized,
            k.fitted) == ('k', '1/s', SEED_RATE, *MANUAL_RATE_BOUNDS, True,
                          MANUAL_FITTED_RATE)
    assert (rise.name, rise.optimized, rise.fitted) == ('rise', False, None)
    np.testing.assert_array_equal(loaded.fitted_values,
                                  [MANUAL_FITTED_RATE])
    np.testing.assert_array_equal(loaded.covariance, MANUAL_COVARIANCE)
    assert loaded.covariance_include_mse is False

    assert [source.name for source in loaded.sources] == ['guide', 'lab']
    assert loaded.sources[0].sha256 == DIGEST
    assert loaded.sources[1].origin == 'experimental'
    first, second = loaded.transformations
    assert (first.fields, first.source) == (('c_A',), 'lab')
    assert first.parameters == {'factor': MANUAL_FACTOR}
    assert (second.fields, second.parameters) == ((), {})
    assert loaded.model.pharmapy_version is None
    assert loaded.model.revision == 'abc123'

    assert loaded.solver.method == 'IPOPT'
    assert_same_value(loaded.solver.options, MANUAL_IPOPT_OPTIONS)
    assert loaded.solver.model_solver == MANUAL_MODEL_SOLVER
    assert loaded.solver.outcome == {'objective': MANUAL_OBJECTIVE}

    composition, density, purity = loaded.fields
    assert composition.identity == FieldIdentity(
        'x_liq', 'state', '-', basis='mole fraction')
    assert composition.columns == ('A', 'B', 'solvent')
    assert_bits_equal(composition.x, COMPOSITION_TIME)
    assert_bits_equal(composition.values, COMPOSITION)
    assert (density.x_units, density.experiment) == ('min', None)
    assert_bits_equal(density.values, DENSITY)
    assert (purity.identity.kind, purity.experiment) == (
        'quality_attribute', 'cold')
    assert_bits_equal(purity.values, [PURITY])


def test_saved_json_is_standard_and_writes_missing_values_as_null(tmp_path):
    path = tmp_path / 'record.json'
    manual_record().save(path)
    raw = path.read_bytes()

    assert not raw.startswith(b'\xef\xbb\xbf')  # plain UTF-8, no BOM
    assert b'\r\n' not in raw
    text = raw.decode('utf-8')
    assert 'NaN' not in text and 'Infinity' not in text
    data = json.loads(text, parse_constant=pytest.fail)
    assert (data['schema'], data['schema_version']) == (
        'pharmapy.calibration-record', 1)
    assert (RECORD_SCHEMA, RECORD_SCHEMA_VERSION) == (
        'pharmapy.calibration-record', 1)
    cold = data['experiments'][0]
    assert cold['name'] == 'cold'
    assert cold['measurements'][0]['values'] == [0.785, None, 0.22]
    assert data['fields'][1]['values'] == [None, 1012.5]
    assert data['covariance'] == {'parameters': ['k'], 'include_mse': False,
                                  'values': MANUAL_COVARIANCE}
    big_endian = data['experiments'][1]['args'][13]
    assert big_endian['dtype'] == '>f8'


def test_load_tolerates_a_byte_order_mark(tmp_path):
    record = manual_record()
    path = tmp_path / 'bom.json'
    path.write_bytes(b'\xef\xbb\xbf' + record.to_json().encode('utf-8'))
    assert CalibrationRecord.load(path) == record


@pytest.mark.skipif(not hasattr(os, 'geteuid') or os.geteuid() == 0,
                    reason='needs POSIX permissions that bind the user')
def test_failed_save_leaves_the_existing_file_unchanged(tmp_path):
    folder = tmp_path / 'records'
    folder.mkdir()
    path = folder / 'record.json'
    manual_record().save(path)
    before = path.read_bytes()
    changed = manual_record(order=('hot', 'cold'))
    os.chmod(folder, stat.S_IRUSR | stat.S_IXUSR)  # no new files allowed
    try:
        with pytest.raises(PermissionError):
            changed.save(path)
    finally:
        os.chmod(folder, stat.S_IRWXU)
    assert path.read_bytes() == before
    assert [item.name for item in folder.iterdir()] == ['record.json']

    target = tmp_path / 'directory'
    target.mkdir()
    with pytest.raises(OSError):
        changed.save(target)  # a directory cannot be replaced by a file
    assert sorted(item.name for item in tmp_path.iterdir()) == [
        'directory', 'records']  # no temporary file is left behind


def test_save_overwrites_atomically_and_keeps_permissions(tmp_path):
    path = tmp_path / 'record.json'
    path.write_bytes(b'previous contents')
    os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
    record = manual_record()
    record.save(path)
    assert CalibrationRecord.load(path) == record
    # POSIX keeps all permission bits; Windows os.chmod sets only the
    # read-only flag, so the 0o600 mode is not representable there.
    if os.name == 'posix':
        assert stat.S_IMODE(os.stat(path).st_mode) == (stat.S_IRUSR
                                                       | stat.S_IWUSR)
    assert list(tmp_path.iterdir()) == [path]


def test_floats_round_trip_bit_for_bit():
    awkward = [0.1 + 0.2, 1 / 3, 5e-324, -0.0, 1e308,
               np.nextafter(1.0, 2.0)]  # [mol/L]
    times = np.arange(len(awkward), dtype=float)  # [s]
    experiments = {'run': Experiment(
        {'c_A': Measurement(0, times, awkward, units='mol/L')},
        args=(np.array(awkward),))}
    record = CalibrationRecord(
        'floats', experiments,
        [ParameterRecord('k', '1/s', 1 / 7, fitted=2 / 3)], SOURCES,
        ModelRevision('decay'))
    loaded = CalibrationRecord.from_json(record.to_json())

    run = loaded.experiments['run']
    assert_bits_equal(run.measurements['c_A'].values, awkward)
    assert_bits_equal(run.args[0], awkward)
    assert loaded.parameters[0].seed == 1 / 7
    assert loaded.fitted_values[0] == 2 / 3


def test_declared_orders_are_preserved_not_sorted():
    forward = manual_record(order=('hot', 'cold'), hot_order=('c_A', 'T'))
    backward = manual_record(order=('cold', 'hot'), hot_order=('T', 'c_A'))

    assert forward != backward  # same content, different declared order
    for record, order, hot_order in ((forward, ['hot', 'cold'],
                                      ('c_A', 'T')),
                                     (backward, ['cold', 'hot'],
                                      ('T', 'c_A'))):
        loaded = CalibrationRecord.from_json(record.to_json())
        assert list(loaded.experiments) == order
        assert loaded.experiments['hot'].measurement_names == hot_order
        assert [item['name'] for item in loaded.to_dict()['experiments']] \
            == order
        assert [field.name for field in loaded.fields] == [
            'x_liq', 'rho', 'purity']
        assert list(loaded.experiments['hot'].kwargs) == ['zeta', 'alpha']


def test_state_and_prediction_fields_may_share_a_name():
    fields = [
        FieldRecord(FieldIdentity('c_A', 'measurement_prediction', 'mol/L',
                                  basis='molar concentration'),
                    [0.0, 1.0], [2.0, 1.2], experiment='hot'),
        FieldRecord(FieldIdentity('c_A', 'state', 'mol/L'), [0.0, 1.0],
                    [2.0, 1.21], experiment='hot'),
        FieldRecord(FieldIdentity('c_A', 'state', 'mol/L'), [0.5],
                    [0.78], experiment='cold')]  # [s], [mol/L]
    record = CalibrationRecord(**record_arguments(fields=fields))
    loaded = CalibrationRecord.from_json(record.to_json())
    assert [(field.experiment, field.identity.kind) for field in
            loaded.fields] == [('hot', 'measurement_prediction'),
                               ('hot', 'state'), ('cold', 'state')]
    assert_bits_equal(loaded.fields[1].values, [2.0, 1.21])
    with pytest.raises(ValueError, match=r"\(experiment, kind, name\) names "
                                         "must be unique"):
        CalibrationRecord(**record_arguments(fields=fields + fields[1:2]))


def test_records_are_immutable_and_isolated_from_callers():
    record = manual_record()
    with pytest.raises(AttributeError, match='immutable'):
        record.case = 'other'
    with pytest.raises(ValueError, match='read-only'):
        record.fields[0].values[0, 0] = 1.0

    returned = record.experiments['hot']
    returned.args[10][0, 0] = 99  # caller mutates a returned array
    assert record.experiments['hot'].args[10][0, 0] == 0
    options = record.solver.options
    options['tol'] = 1.0
    assert record.solver.options['tol'] == MANUAL_IPOPT_OPTIONS['tol']

    array_arg = np.array([1.0, 2.0])  # [mol/L]
    experiments = {'run': Experiment(
        {'c_A': Measurement(0, [0.0, 1.0], [1.0, 0.5], units='mol/L')},
        args=(array_arg,))}
    built = CalibrationRecord('isolation', experiments,
                              [ParameterRecord('k', '1/s', 0.1)], SOURCES,
                              ModelRevision('decay'))
    array_arg[0] = -1.0  # caller mutates its own argument afterwards
    np.testing.assert_array_equal(built.experiments['run'].args[0],
                                  [1.0, 2.0])


# ------------------------------------------------- capture, reload and rerun


def test_from_estimation_captures_fit_and_closed_form_predictions():
    record, estimator, _ = fitted_record()
    rate = record.fitted_values[0]  # [1/s]

    assert [item.name for item in record.parameters] == ['k']
    parameter = record.parameters[0]
    assert (parameter.units, parameter.seed, parameter.optimized) == (
        '1/s', SEED_RATE, True)
    assert parameter.fitted == estimator.params_convg[0]
    np.testing.assert_array_equal(record.covariance, estimator.covar_params)
    assert record.covariance_include_mse is True
    assert record.solver.method == 'LM'
    assert record.solver.options == {}  # injected LM keys are omitted
    outcome = record.solver.outcome
    assert set(outcome) == {'stop_criterion', 'num_iter', 'num_fun_eval',
                            'objective'}
    residuals = standardized_residuals(rate)  # [-], closed form
    assert outcome['objective'] == pytest.approx(
        np.sum(residuals ** 2) / 2, rel=ROUNDOFF_RTOL)
    assert record.solver.derivatives == {'jacobian': 'finite_difference',
                                         'dx_finitediff': 'default'}

    predictions = [field for field in record.fields
                   if field.identity.kind == 'measurement_prediction']
    assert [(field.experiment, field.name) for field in predictions] == [
        ('hot', 'c_A'), ('hot', 'T'), ('cold', 'c_A'), ('cold', 'T')]
    units = {'c_A': ('mol/L', 'molar concentration'), 'T': ('K', None)}
    for field in predictions:
        assert (field.identity.units, field.identity.basis) == units[
            field.name]
        assert field.x_units == 's'
        assert_bits_equal(field.x, GRID[field.experiment])
        column = 0 if field.name == 'c_A' else 1
        # Every grid row, observed or not, is the model output at k.
        expected = [closed_form(rate, field.experiment, time)[column]
                    for time in GRID[field.experiment]]  # [mol/L or K]
        assert not np.isnan(field.values).any()
        np.testing.assert_allclose(field.values, expected,
                                   rtol=ROUNDOFF_RTOL, atol=ROUNDOFF_ATOL)


def test_from_estimation_leaves_estimator_and_experiments_unchanged():
    experiments = make_experiments()
    estimator = ParameterEstimation(decay, [SEED_RATE], experiments,
                                    name_params=['k'])
    options = {'max_fun_eval': LM_MAX_FUN_EVAL}
    estimator.optimize_fn(optim_options=options, verbose=False)
    # Default form, then the absolute form: covar_params holds the latter.
    estimator.get_covariance(include_mse=False)
    snapshot = {name: np.array(getattr(estimator, name), copy=True)
                for name in ('params_convg', 'covar_params', 'correl_params',
                             'param_seed', 'params_iter', 'objfun_iter')}
    y_runs = [np.array(item, copy=True) for item in estimator.y_runs]
    info_fun = np.array(estimator.info_opt['fun'], copy=True)
    options_after_fit = dict(options)

    record = capture(estimator, experiments, covariance_include_mse=True)

    for name, value in snapshot.items():
        np.testing.assert_array_equal(getattr(estimator, name), value)
    for actual, value in zip(estimator.y_runs, y_runs):
        np.testing.assert_array_equal(actual, value)
    np.testing.assert_array_equal(estimator.info_opt['fun'], info_fun)
    assert options == options_after_fit
    assert record.solver.options == {'max_fun_eval': LM_MAX_FUN_EVAL}
    # The recorded covariance is the requested include_mse=True form,
    # s2 times the absolute form held by covar_params; it is smaller here
    # because the reduced chi-square s2 of the documented fit is below one.
    assert record.covariance_include_mse is True
    assert record.covariance[0, 0] < snapshot['covar_params'][0, 0]
    assert list(experiments) == ['hot', 'cold']
    assert_bits_equal(experiments['cold'].measurements['c_A'].values,
                      COLD_CA)
    assert dict(experiments['cold'].kwargs) == {'rise_k_l_mol': 10.0,
                                                'temp0_k': 290.0}


def test_covariance_form_is_recorded_as_requested():
    experiments = make_experiments()
    estimator = ParameterEstimation(decay, [SEED_RATE], experiments,
                                    name_params=['k'])
    estimator.optimize_fn(verbose=False)
    default = estimator.covar_params.copy()  # [1/s**2]
    absolute = capture(estimator, experiments,
                       covariance_include_mse=False)
    scaled = capture(estimator, experiments)
    np.testing.assert_array_equal(scaled.covariance, default)
    # scaled = s2 * absolute with the residual mean square
    # s2 = r @ r / (n - p) of the closed-form standardized residuals.
    residuals = standardized_residuals(
        estimator.params_convg[0])  # [-]
    mean_square = np.sum(residuals ** 2) / (
        len(OBSERVED_ENTRIES) - NUM_PARAMETERS)  # [-]
    assert mean_square < 1  # so the scaled form is the smaller one
    assert scaled.covariance[0, 0] == pytest.approx(
        mean_square * absolute.covariance[0, 0], rel=ROUNDOFF_RTOL)
    assert absolute.covariance_include_mse is False
    np.testing.assert_array_equal(estimator.covar_params, default)
    loaded = CalibrationRecord.from_json(absolute.to_json())
    assert loaded.covariance_include_mse is False


def test_reload_and_rerun_reproduces_recorded_fit(tmp_path):
    record, original, _ = fitted_record()
    path = tmp_path / 'decay.json'
    record.save(path)
    loaded = CalibrationRecord.load(path)

    inputs = loaded.estimation_inputs()
    assert list(inputs) == ['param_seed', 'x_data', 'optimize_flags',
                            'name_params', 'output_names', 'weight_matrix']
    assert inputs['param_seed'].tolist() == [SEED_RATE]
    assert (inputs['optimize_flags'], inputs['name_params']) == ([True],
                                                                 ['k'])
    rebuilt = ParameterEstimation(decay, **inputs)
    pd.testing.assert_frame_equal(rebuilt.get_residual_layout(),
                                  original.get_residual_layout())
    recorded = loaded.fitted_values  # [1/s]
    assert rebuilt.get_objective(recorded) == pytest.approx(
        original.get_objective(original.params_convg),
        rel=REPLAY_RTOL, abs=0)

    optimizer_inputs = loaded.optimizer_inputs()
    assert optimizer_inputs == {'method': 'LM', 'optim_options': {},
                                'bounds': None}
    rebuilt.optimize_fn(verbose=False, **optimizer_inputs)
    # optimize_fn wrote its own entries into the dict it received (#476);
    # the record hands out a fresh dict every time.
    assert set(optimizer_inputs['optim_options']) == {'full_output',
                                                      'verbose'}
    assert loaded.optimizer_inputs()['optim_options'] == {}
    difference = loaded.check_rerun(rebuilt, rtol=REPLAY_RTOL,
                                    atol=REPLAY_ATOL)
    assert difference.shape == (1,)
    np.testing.assert_allclose(rebuilt.params_convg, original.params_convg,
                               rtol=REPLAY_RTOL, atol=REPLAY_ATOL)

    rerun = loaded.rerun(decay, rtol=REPLAY_RTOL, atol=REPLAY_ATOL)
    np.testing.assert_allclose(rerun.params_convg, recorded,
                               rtol=REPLAY_RTOL, atol=REPLAY_ATOL)


def test_fixed_parameter_before_optimized_one_round_trips_and_reruns():
    experiments = experiments_without_rise()
    rise = RISE_PARAMETER  # [K*L/mol]
    estimator = ParameterEstimation(
        decay_rise_first, [rise, SEED_RATE], experiments,
        name_params=['rise', 'k'], optimize_flags=[False, True])
    estimator.optimize_fn(verbose=False)
    user_field = FieldRecord(FieldIdentity('purity', 'quality_attribute',
                                           '-'), PURITY_TIME, PURITY,
                             experiment='cold')
    transformation = Transformation('converted absorbance to c_A [mol/L]',
                                    fields=('c_A',), source='guide')
    record = capture(estimator, experiments,
                     parameter_units={'rise': 'K*L/mol', 'k': '1/s'},
                     bounds=[MANUAL_RATE_BOUNDS], fields=[user_field],
                     transformations=[transformation],
                     model_solver=MANUAL_MODEL_SOLVER)
    loaded = CalibrationRecord.from_json(record.to_json())

    by_name = {item.name: item for item in loaded.parameters}
    assert list(by_name) == ['rise', 'k']
    assert (by_name['rise'].seed, by_name['rise'].optimized,
            by_name['rise'].fitted, by_name['rise'].lower,
            by_name['rise'].upper) == (rise, False, None, None, None)
    assert (by_name['k'].seed, by_name['k'].lower, by_name['k'].upper) == (
        SEED_RATE, *MANUAL_RATE_BOUNDS)
    assert by_name['k'].fitted == estimator.params_convg[0]
    assert [(field.identity.kind, field.experiment, field.name)
            for field in loaded.fields] == [
        ('quality_attribute', 'cold', 'purity'),
        ('measurement_prediction', 'hot', 'c_A'),
        ('measurement_prediction', 'hot', 'T'),
        ('measurement_prediction', 'cold', 'c_A'),
        ('measurement_prediction', 'cold', 'T')]
    assert loaded.transformations == (transformation,)
    assert loaded.solver.model_solver == MANUAL_MODEL_SOLVER
    assert loaded.optimizer_inputs()['bounds'] == [MANUAL_RATE_BOUNDS]
    assert loaded.estimation_inputs()['optimize_flags'] == [False, True]

    rerun = loaded.rerun(decay_rise_first, rtol=REPLAY_RTOL,
                         atol=REPLAY_ATOL)
    np.testing.assert_allclose(rerun.params_convg, estimator.params_convg,
                               rtol=REPLAY_RTOL, atol=REPLAY_ATOL)


def test_check_rerun_reports_a_different_recorded_fit():
    record, _, _ = fitted_record()
    data = record.to_dict()
    data['parameters'][0]['fitted'] += RATE_OFFSET
    shifted = CalibrationRecord.from_dict(data)

    with pytest.raises(RerunMismatchError, match=r"k: rerun 0\.50"):
        shifted.rerun(decay, rtol=REPLAY_RTOL, atol=REPLAY_ATOL)
    # The unshifted record passes the same check.
    record.rerun(decay, rtol=REPLAY_RTOL, atol=REPLAY_ATOL)


def test_non_finite_rerun_values_are_mismatches():
    record, _, _ = fitted_record()
    inputs = record.estimation_inputs()
    inputs['param_seed'] = np.array([np.nan])  # LM stays at a NaN seed
    estimator = ParameterEstimation(decay, **inputs)
    estimator.optimize_fn(verbose=False)
    assert np.isnan(estimator.params_convg[0])
    with pytest.raises(RerunMismatchError, match="k: rerun nan"):
        record.check_rerun(estimator, rtol=1.0, atol=1.0)


@pytest.mark.parametrize('tolerances, error, message', [
    ({'rtol': np.nan, 'atol': 0.0}, ValueError, "rtol must be finite"),
    ({'rtol': 0.0, 'atol': np.inf}, ValueError, "atol must be finite"),
    ({'rtol': -1e-9, 'atol': 0.0}, ValueError,
     "rtol must be non-negative; got -1e-09"),
    ({'rtol': 0.0, 'atol': -1.0}, ValueError,
     "atol must be non-negative; got -1.0"),
    ({'rtol': None, 'atol': 0.0}, TypeError,
     "rtol must be a real number; got NoneType"),
])
def test_rerun_tolerances_are_validated(tolerances, error, message):
    record, estimator, _ = fitted_record()
    with pytest.raises(error, match=message):
        record.check_rerun(estimator, **tolerances)
    evaluations = []

    def counting_decay(*args, **kwargs):
        """Count model evaluations of ``decay``.

        Parameters
        ----------
        *args, **kwargs
            Arguments of ``decay``.

        Returns
        -------
        numpy.ndarray
            ``decay(*args, **kwargs)``.
        """
        evaluations.append(1)
        return decay(*args, **kwargs)

    with pytest.raises(error, match=message):
        record.rerun(counting_decay, **tolerances)
    assert evaluations == []  # rejected before any model evaluation


def test_check_rerun_rejects_estimators_of_another_case():
    record, _, _ = fitted_record()
    reordered = ParameterEstimation(decay, [SEED_RATE],
                                    make_experiments(order=('cold', 'hot')),
                                    name_params=['k'])
    renamed = ParameterEstimation(decay, [SEED_RATE], make_experiments(),
                                  name_params=['rate'])
    unfitted = ParameterEstimation(decay, [SEED_RATE], make_experiments(),
                                   name_params=['k'])
    for estimator in (reordered, renamed):
        estimator.optimize_fn(verbose=False)
        with pytest.raises(ValueError, match='not built from this record'):
            record.check_rerun(estimator, rtol=REPLAY_RTOL,
                               atol=REPLAY_ATOL)
    # Same names, other optimize flags: the rise is fitted as well.
    experiments = experiments_without_rise()
    fixed_rise = ParameterEstimation(
        decay_rise_first, [RISE_PARAMETER, SEED_RATE], experiments,
        name_params=['rise', 'k'], optimize_flags=[False, True])
    fixed_rise.optimize_fn(verbose=False)
    fixed_record = capture(fixed_rise, experiments,
                           parameter_units=['K*L/mol', '1/s'])
    both = ParameterEstimation(
        decay_rise_first, [RISE_PARAMETER, SEED_RATE], experiments,
        name_params=['rise', 'k'], optimize_flags=[True, True])
    both.optimize_fn(verbose=False)
    with pytest.raises(ValueError, match='not built from this record'):
        fixed_record.check_rerun(both, rtol=REPLAY_RTOL, atol=REPLAY_ATOL)
    with pytest.raises(ValueError, match='has not been fitted'):
        record.check_rerun(unfitted, rtol=REPLAY_RTOL, atol=REPLAY_ATOL)


def test_weight_matrix_lower_triangle_is_canonicalized_and_reruns():
    experiments = make_experiments(uncertain=False)
    # Error covariance labelled in the reverse of the measurement order
    # (T then c_A) with an unread NaN upper triangle: T variance [K**2],
    # covariance [K*mol/L], c_A variance [(mol/L)**2].
    correlation = 0.3  # [-], arbitrary correlation of the two errors
    covariance_t_ca = correlation * HOT_T_STD * HOT_CA_STD  # [K*mol/L]
    weights = pd.DataFrame(
        [[HOT_T_STD ** 2, np.nan], [covariance_t_ca, HOT_CA_STD ** 2]],
        index=['T', 'c_A'], columns=['T', 'c_A'])
    estimator = ParameterEstimation(decay, [SEED_RATE], experiments,
                                    name_params=['k'], weight_matrix=weights)
    estimator.optimize_fn(verbose=False)
    with pytest.raises(ValueError, match='uses a weight_matrix'):
        capture(estimator, experiments)
    different = weights.copy()
    different.loc['c_A', 'c_A'] = 2 * HOT_CA_STD ** 2  # [(mol/L)**2]
    with pytest.raises(ValueError, match="weight_matrix differs from the "
                                         "estimator's"):
        capture(estimator, experiments, weight_matrix=different)
    record = capture(estimator, experiments, weight_matrix=weights)

    loaded = CalibrationRecord.from_json(record.to_json())
    expected = pd.DataFrame(
        [[HOT_CA_STD ** 2, covariance_t_ca],
         [covariance_t_ca, HOT_T_STD ** 2]],
        index=['c_A', 'T'], columns=['c_A', 'T'])  # measurement order
    pd.testing.assert_frame_equal(loaded.weight_matrix, expected)
    rerun = loaded.rerun(decay, rtol=REPLAY_RTOL, atol=REPLAY_ATOL)
    np.testing.assert_allclose(rerun.params_convg, estimator.params_convg,
                               rtol=REPLAY_RTOL, atol=REPLAY_ATOL)


def test_weight_matrix_is_rejected_with_declared_uncertainties():
    record, estimator, experiments = fitted_record()
    weights = pd.DataFrame(np.eye(2), index=['c_A', 'T'],
                           columns=['c_A', 'T'])  # [-]
    with pytest.raises(ValueError, match="weight_matrix must be None: the "
                                         "estimator's measurements declare"):
        capture(estimator, experiments, weight_matrix=weights)


def test_from_estimation_rejects_experiments_that_were_not_fitted():
    record, estimator, experiments = fitted_record()
    altered_values = make_experiments()
    altered_values['cold'] = Experiment(
        {'c_A': Measurement(0, COLD_CA_TIME, [0.785, 0.5, 0.220],
                            uncertainty=COLD_CA_STD, units='mol/L',
                            basis='molar concentration')},
        args=(1.0,), kwargs={'rise_k_l_mol': 10.0, 'temp0_k': 290.0})
    altered_std = make_experiments()
    altered_std['hot'] = Experiment(
        {'c_A': Measurement(0, HOT_CA_TIME, HOT_CA, uncertainty=0.03,
                            units='mol/L', basis='molar concentration'),
         'T': altered_std['hot'].measurements['T']},
        args=(2.0,), kwargs={'rise_k_l_mol': 20.0})
    altered_grid = make_experiments()
    altered_grid['hot'] = Experiment(
        {'c_A': Measurement(0, [0.0, 1.0, 2.0, 5.0], HOT_CA,
                            uncertainty=HOT_CA_STD, units='mol/L',
                            basis='molar concentration'),
         'T': altered_grid['hot'].measurements['T']},
        args=(2.0,), kwargs={'rise_k_l_mol': 20.0})
    altered_args = make_experiments()
    altered_args['cold'] = Experiment(altered_args['cold'].measurements,
                                      args=(1.5,),
                                      kwargs=altered_args['cold'].kwargs)
    altered_kwargs = make_experiments()
    altered_kwargs['cold'] = Experiment(altered_kwargs['cold'].measurements,
                                        args=(1.0,),
                                        kwargs={'rise_k_l_mol': 10.0,
                                                'temp0_k': 291.0})
    extra_kwargs = make_experiments()
    extra_kwargs['cold'] = Experiment(extra_kwargs['cold'].measurements,
                                      args=(1.0,),
                                      kwargs={'rise_k_l_mol': 10.0})
    reordered = make_experiments(hot_order=('T', 'c_A'))
    # Equal values of another type or order are different callback inputs.
    int_args = make_experiments()
    int_args['cold'] = Experiment(int_args['cold'].measurements, args=(1,),
                                  kwargs=int_args['cold'].kwargs)
    int_kwargs = make_experiments()
    int_kwargs['cold'] = Experiment(int_kwargs['cold'].measurements,
                                    args=(1.0,),
                                    kwargs={'rise_k_l_mol': 10,
                                            'temp0_k': 290.0})
    reordered_kwargs = make_experiments()
    reordered_kwargs['cold'] = Experiment(
        reordered_kwargs['cold'].measurements, args=(1.0,),
        kwargs={'temp0_k': 290.0, 'rise_k_l_mol': 10.0})
    cases = [
        (altered_values, "Experiment 'cold': the estimator's observations "
                         "y_data differ"),
        (altered_std, "Experiment 'hot': the estimator's measurement "
                      "standard deviations differ"),
        (altered_grid, "Experiment 'hot': the estimator's model grid "
                       "x_model differ"),
        (altered_args, r"Experiment 'cold': the estimator's positional "
                       r"callback arguments \(1\.0,\) differ from the "
                       r"experiment's args \(1\.5,\)"),
        (int_args, r"positional callback arguments \(1\.0,\) differ from "
                   r"the experiment's args \(1,\)"),
        (int_kwargs, "callback keyword 'rise_k_l_mol' differs"),
        (reordered_kwargs, r"keywords come in the order \['rise_k_l_mol', "
                           r"'temp0_k'\], the experiment's in "
                           r"\['temp0_k', 'rise_k_l_mol'\]"),
        (altered_kwargs, "Experiment 'cold': the estimator's callback "
                         "keyword 'temp0_k' differs"),
        (extra_kwargs, r"Experiment 'cold': the estimator passes callback "
                       r"keywords \['temp0_k'\]"),
        (reordered, "measurement columns"),
        (make_experiments(uncertain=False), "uncertainties are declared"),
    ]
    for candidate, message in cases:
        with pytest.raises(ValueError, match=message):
            capture(estimator, candidate)


def test_from_estimation_requires_a_fitted_matching_estimator():
    experiments = make_experiments()
    estimator = ParameterEstimation(decay, [SEED_RATE], experiments,
                                    name_params=['k'])
    with pytest.raises(ValueError, match='has not been fitted'):
        capture(estimator, experiments)
    estimator.optimize_fn(verbose=False)
    reordered = make_experiments(order=('cold', 'hot'))
    with pytest.raises(ValueError, match="differ from \\['cold', 'hot'\\]"):
        capture(estimator, reordered)
    with pytest.raises(ValueError, match="parameter_units keys"):
        capture(estimator, experiments, parameter_units={'E': 'J/mol'})
    with pytest.raises(ValueError, match="one \\(lower, upper\\) pair"):
        capture(estimator, experiments, bounds=[(0.0, 1.0)] * 2)
    with pytest.raises(ValueError, match="bounds of 'k' must be a "):
        capture(estimator, experiments, bounds=[(0.0, 1.0, 2.0)])
    # A later evaluation at another point leaves y_runs at that point.
    estimator.get_objective(estimator.params_convg * 2)
    with pytest.raises(ValueError, match='not evaluated at its accepted'):
        capture(estimator, experiments)


def test_lm_bounds_are_recorded_as_declared_ranges():
    unbounded, _, _ = fitted_record(bounds=[(None, None)])
    assert (unbounded.parameters[0].lower,
            unbounded.parameters[0].upper) == (None, None)
    assert unbounded.optimizer_inputs()['bounds'] is None
    lower_only, _, _ = fitted_record(bounds=[(0.0, None)])
    assert lower_only.optimizer_inputs()['bounds'] == [(0.0, None)]  # [1/s]


# ------------------------------------------------------------ validation


@pytest.mark.parametrize('build, message', [
    (lambda: FieldIdentity('c', 'stat', 'mol/L'),
     r"FieldIdentity kind must be one of \('state', 'algebraic', "
     r"'quality_attribute', 'measurement_prediction'\); got 'stat'"),
    (lambda: DataSource('lab', 'measured', 'internal'),
     r"DataSource 'lab' origin must be one of \('experimental', "
     r"'literature', 'synthetic', 'simulation', 'unknown'\); got "
     r"'measured'"),
    (lambda: DataSource('lab', 'experimental', 'internal',
                        sha256='ab' * 31 + 'a'),
     'sha256 must be 64 lowercase hexadecimal digits'),
    (lambda: DataSource('lab', 'experimental', 'internal',
                        sha256='AB' * 32),
     'sha256 must be 64 lowercase hexadecimal digits'),
    (lambda: DataSource('lab', 'experimental', 'internal',
                        sha256='zz' * 32),
     'sha256 must be 64 lowercase hexadecimal digits'),
    (lambda: DataSource('lab', 'experimental', ''),
     "DataSource 'lab' sharing must be a non-empty str"),
    (lambda: DataSource('lab', 'unknown', 'internal'),
     r"^DataSource 'lab' has origin 'unknown' and must state what is known "
     r"about the data in its description$"),
    (lambda: DataSource('lab', 'unknown', 'internal', description=''),
     "DataSource 'lab' description must be a non-empty str"),
    (lambda: ParameterRecord('k', '1/s', 1.0, lower=2.0, upper=1.0),
     "lower bound 2.0 exceeds its upper bound 1.0"),
    (lambda: ParameterRecord('k', '1/s', np.nan),
     "Parameter 'k' seed must be finite; got nan"),
    (lambda: ParameterRecord('k', '1/s', 1.0, fitted=np.inf),
     "Parameter 'k' fitted must be finite; got inf"),
    (lambda: ParameterRecord('k', '1/s', 10 ** 400),
     "Parameter 'k' seed = 1000.* is too large for a float"),
    (lambda: ParameterRecord('k', '1/s', 1.0, optimized=False, fitted=1.0),
     "is not optimized and keeps its seed"),
    (lambda: SolverSettings('BFGS'),
     r"SolverSettings method must be one of \('LM', 'IPOPT'\)"),
])
def test_component_validation_messages(build, message):
    with pytest.raises(ValueError, match=message):
        build()


def test_unknown_origin_states_what_is_known(tmp_path):
    # Data whose provider did not record their origin (e.g. issue #478)
    # are recorded as such, with the known facts in sharing/description.
    unknown = DataSource('workshop', 'unknown',
                         'shipped with the repository; original license '
                         'not recorded',
                         description='origin not recorded by the provider')
    record, _, _ = fitted_record(sources=(unknown,))
    path = tmp_path / 'unknown.json'
    record.save(path)
    assert '"origin": "unknown"' in path.read_text(encoding='utf-8')
    loaded = CalibrationRecord.load(path)
    assert loaded.sources == (unknown,)
    assert loaded.sources[0].origin == 'unknown'
    # Loading enforces the description too.
    data = json.loads(path.read_text(encoding='utf-8'))
    data['sources'][0]['description'] = None
    with pytest.raises(ValueError, match="has origin 'unknown' and must "
                                         "state what is known"):
        CalibrationRecord.from_dict(data)


def test_seed_outside_its_bounds_is_an_infeasible_start():
    # IPOPT accepts infeasible starting points and moves them into bounds.
    outside = ParameterRecord('k', '1/s', 6.0, lower=0.0, upper=5.0)
    assert (outside.seed, outside.lower, outside.upper) == (6.0, 0.0, 5.0)


def test_callback_values_compare_by_type_order_and_value():
    from PharmaPy.Records import _same_value
    assert _same_value({'a': (1, 2.0)}, {'a': (1, 2.0)})
    assert not _same_value(True, 1)
    assert not _same_value(1, 1.0)
    assert not _same_value((1, 2), [1, 2])
    assert not _same_value({'a': 1, 'b': 2}, {'b': 2, 'a': 1})
    assert not _same_value(0.0, -0.0)
    assert not _same_value(np.array([1.0], dtype='<f8'),
                           np.array([1.0], dtype='>f8'))
    marker = object()  # unrecordable values are equal only if identical
    assert _same_value(marker, marker)
    assert not _same_value(marker, object())


def test_boolean_seed_is_not_a_number():
    with pytest.raises(TypeError, match="seed must be a real number; got "
                                        "bool"):
        ParameterRecord('k', '1/s', True)


@pytest.mark.parametrize('args, kwargs, error, message', [
    ((object(),), {}, TypeError,
     r"experiments\['run'\]\.args\[0\] has unsupported type object"),
    ((len,), {}, TypeError,
     r"experiments\['run'\]\.args\[0\] has unsupported type "
     r"builtin_function_or_method"),
    ((), {'phase': [1.0, {'model': object()}]}, TypeError,
     r"experiments\['run'\]\.kwargs\['phase'\]\[1\]\['model'\] has "
     "unsupported type object"),
    ((), {'temp0_k': np.nan}, ValueError,
     r"experiments\['run'\]\.kwargs\['temp0_k'\] = nan cannot be recorded"),
    ((np.array([1.0, np.inf]),), {}, ValueError,
     r"experiments\['run'\]\.args\[0\] holds NaN or infinite entries"),
    ((np.array([1j]),), {}, TypeError,
     r"args\[0\] has unsupported NumPy dtype complex128"),
    ((np.complex128(1j),), {}, TypeError,
     r"args\[0\] has unsupported NumPy dtype complex128"),
    ((np.array(['a']),), {}, TypeError, "unsupported NumPy dtype <U1"),
    ((np.array([1.0], dtype=np.longdouble),), {}, TypeError,
     r"args\[0\] has NumPy dtype float128, wider than 64 bits"),
    ((np.longdouble(1.0),), {}, TypeError,
     r"args\[0\] has NumPy dtype float128, wider than 64 bits"),
    ((np.zeros(1, dtype=np.dtype(float, metadata={'units': 'K'})),), {},
     TypeError, r"args\[0\] has a NumPy dtype carrying metadata"),
    ((np.zeros(1, dtype=[('temp', 'f8')]),), {}, TypeError,
     r"args\[0\] has structured NumPy dtype"),
    ((np.zeros(1, dtype=[('temp', 'f8')])[0],), {}, TypeError,
     r"args\[0\] has structured NumPy dtype"),
    ((np.matrix([[1.0]]),), {}, TypeError,
     r"args\[0\] is a matrix, an ndarray subclass"),
    ((np.ma.masked_array([1.0], mask=[False]),), {}, TypeError,
     r"args\[0\] is a MaskedArray, an ndarray subclass"),
    (({1: 'a'},), {}, TypeError, r"args\[0\] keys must be str; got int 1"),
    (({'__type__': 'tuple'},), {}, ValueError,
     "uses the reserved key '__type__'"),
    ((frozenset(),), {}, TypeError, "unsupported type frozenset"),
])
def test_unsupported_callback_arguments_name_their_location(args, kwargs,
                                                            error, message):
    if 'float128' in message and np.dtype(np.longdouble).itemsize <= 8:
        pytest.skip('longdouble is binary64 on this platform')
    experiments = {'run': Experiment(
        {'c_A': Measurement(0, [0.0, 1.0], [1.0, 0.5], units='mol/L')},
        args=args, kwargs=kwargs)}
    with pytest.raises(error, match=message):
        CalibrationRecord('bad', experiments,
                          [ParameterRecord('k', '1/s', 0.1)], SOURCES,
                          ModelRevision('decay'))


@pytest.mark.parametrize('measurement, message', [
    (Measurement(0, [0.0, 1.0], [1.0, 0.5]),
     r"measurements\['c_A'\] must declare units"),
    (Measurement(0, [0.0, 1.0], [1.0, 0.5], units='mol/L', x_units=None),
     r"measurements\['c_A'\] must declare x_units"),
    (Measurement(0, [0.0, 1.0], [1.0, 0.5], units='mol/L', basis=''),
     r"experiments\['run'\]\.measurements\['c_A'\] declares an empty "
     "basis"),
    (Measurement(0, [0.0, 1.0], [1.0, np.nan], units='mol/L',
                 uncertainty=[0.1, np.inf]),
     r"measurements\['c_A'\]\.uncertainty\[1\] = inf cannot be recorded"),
])
def test_measurements_must_be_fully_documented(measurement, message):
    with pytest.raises(ValueError, match=message):
        CalibrationRecord('bad', {'run': Experiment({'c_A': measurement})},
                          [ParameterRecord('k', '1/s', 0.1)], SOURCES,
                          ModelRevision('decay'))


def test_experiment_sets_are_validated_as_parameter_estimation_does():
    def run(measurements, **kwargs):
        """Wrap measurements in a one-argument experiment.

        Parameters
        ----------
        measurements : dict
            Named measurements.
        **kwargs
            Callback keywords.

        Returns
        -------
        Experiment
            Experiment with initial c_A 1 mol/L.
        """
        return Experiment(measurements, args=(1.0,), kwargs=kwargs)

    times, values = [0.0, 1.0], [1.0, 0.5]  # [s], [mol/L]
    cases = [
        ({'a': run({'c_A': Measurement(0, times, values, units='mol/L')}),
          'b': run({'c_A': Measurement(0, times, values, units='g/L')})},
         "Inconsistent units for measurement name 'c_A'"),
        ({'a': run({'c_A': Measurement(0, times, values, units='mol/L',
                                       basis='molar concentration')}),
          'b': run({'c_B': Measurement(0, times, values, units='mol/L',
                                       basis='mass concentration')})},
         "Inconsistent basis for model output column 0"),
        ({'a': run({'c_A': Measurement(0, times, values, units='mol/L',
                                       uncertainty=0.1),
                    'T': Measurement(1, times, values, units='K')})},
         "Either every Measurement declares an uncertainty or none does"),
        ({'a': run({'c_A': Measurement(0, times, values, units='mol/L')}),
          'b': run({'c_A': Measurement(0, times, values, units='mol/L',
                                       x_units='min')})},
         "Inconsistent x_units for all experiments"),
        ({'a': run({'c_A': Measurement('c_A', times, values,
                                       units='mol/L')})},
         "names model output 'c_A', but output_names was not given"),
    ]
    for experiments, message in cases:
        with pytest.raises(ValueError, match=message):
            CalibrationRecord(**record_arguments(experiments=experiments))
    with pytest.raises(ValueError, match="weight_matrix must be None when "
                                         "measurements declare"):
        CalibrationRecord(**record_arguments(weight_matrix=pd.DataFrame(
            np.eye(2), index=['c_A', 'T'], columns=['c_A', 'T'])))


@pytest.mark.parametrize('changes, error, message', [
    ({'fields': [FieldRecord(FieldIdentity('c_A', 'state', 'mol/L'),
                             [0.0], [1.0], experiment='warm')]},
     ValueError, "names experiment 'warm', which is not among"),
    ({'transformations': [Transformation('scaled', source='notebook')]},
     ValueError, "names source 'notebook', which is not among"),
    ({'transformations': [Transformation('scaled', fields=('c_B',))]},
     ValueError, r"names fields \['c_B'\] that are neither measurements"),
    ({'sources': []}, ValueError, "sources must hold at least one"),
    ({'sources': ['guide']}, TypeError,
     "sources entries must be DataSource; got str"),
    ({'sources': SOURCES * 2}, ValueError, "DataSource names must be unique"),
    ({'parameters': []}, ValueError, "parameters must hold at least one"),
    ({'parameters': [{'name': 'k'}]}, TypeError,
     "parameters entries must be ParameterRecord; got dict"),
    ({'parameters': [ParameterRecord('k', '1/s', 0.1),
                     ParameterRecord('k', '1/s', 0.2)]},
     ValueError, "Parameter names must be unique"),
    ({'parameters': [ParameterRecord('k', '1/s', 0.1, fitted=0.2),
                     ParameterRecord('E', 'J/mol', 1.0)]},
     ValueError, r"every optimized parameter or for none; missing for "
                 r"\['E'\]"),
    ({'covariance': [[1.0]], 'covariance_include_mse': True}, ValueError,
     "covariance requires fitted values"),
    ({'parameters': [ParameterRecord('k', '1/s', 0.1, fitted=0.2)],
      'covariance': [[1.0]]}, TypeError,
     "covariance_include_mse must be True or False"),
    ({'covariance_include_mse': True}, ValueError,
     "covariance_include_mse requires a covariance"),
    ({'parameters': [ParameterRecord('k', '1/s', 0.1, fitted=0.2)],
      'covariance': [[1.0, 0.0]], 'covariance_include_mse': True},
     ValueError, r"square with one row and column per optimized parameter "
                 r"\(1\); got shape \(1, 2\)"),
    ({'parameters': [ParameterRecord('k', '1/s', 0.1, fitted=0.2)],
      'covariance': [[np.nan]], 'covariance_include_mse': True},
     ValueError, "covariance must be finite"),
    ({'parameters': [ParameterRecord('k', '1/s', 0.1, fitted=0.2)],
      'covariance': [[1j]], 'covariance_include_mse': True},
     TypeError, "covariance must hold real numbers; got complex"),
    ({'parameters': [ParameterRecord('k', '1/s', 0.1, fitted=0.2)],
      'covariance': [['a']], 'covariance_include_mse': True},
     TypeError, "covariance must hold real numbers"),
    ({'experiments': {}}, ValueError, "at least one Experiment"),
    ({'output_names': 'c_A'}, TypeError, "output_names must be a sequence"),
])
def test_record_cross_references_are_validated(changes, error, message):
    with pytest.raises(error, match=message):
        CalibrationRecord(**record_arguments(**changes))


@pytest.mark.parametrize('weights, error, message', [
    (pd.DataFrame([[1.0]], index=['c_A'], columns=['c_A']), ValueError,
     r"labels must be exactly the measurement names; missing=\['T'\]"),
    (np.eye(2), TypeError, "weight_matrix must be a pandas.DataFrame"),
    (pd.DataFrame([[1.0, 0.0], [0.0, 1j]], index=['c_A', 'T'],
                  columns=['c_A', 'T']), TypeError, "got complex"),
    (pd.DataFrame([[1.0, 0.0], [0.0, 'x']], index=['c_A', 'T'],
                  columns=['c_A', 'T']), TypeError, "real numbers"),
    (pd.DataFrame([[1.0, 0.0], [np.nan, 1.0]], index=['c_A', 'T'],
                  columns=['c_A', 'T']), ValueError,
     "finite in its lower triangle"),
    (pd.DataFrame([[1.0, 0.0], [2.0, 1.0]], index=['c_A', 'T'],
                  columns=['c_A', 'T']), ValueError, "positive-definite"),
])
def test_weight_matrix_values_are_validated(weights, error, message):
    experiments = make_experiments(uncertain=False)
    with pytest.raises(error, match=message):
        CalibrationRecord(**record_arguments(experiments=experiments,
                                             weight_matrix=weights))


def test_repeated_fields_with_and_without_experiment_are_reported():
    fields = [FieldRecord(FieldIdentity('c_A', 'state', 'mol/L'), [0.0],
                          [1.0], experiment=experiment)
              for experiment in (None, 'hot', None, 'hot')]
    with pytest.raises(ValueError, match=r"repeated \[\(None, 'state', "
                                         r"'c_A'\), \('hot', 'state', "
                                         r"'c_A'\)\]"):
        CalibrationRecord(**record_arguments(fields=fields))


@pytest.mark.parametrize('mutate, error, message', [
    (lambda data: data.update(schema_version=2), ValueError,
     "Unsupported calibration record schema_version 2; this PharmaPy reads "
     "version 1"),
    (lambda data: data.update(schema_version=True), ValueError,
     "Unsupported calibration record schema_version True"),
    (lambda data: data.update(schema='pharmapy.other'), ValueError,
     "Not a calibration record"),
    (lambda data: data.update(extra=1), ValueError,
     r"unknown keys \['extra'\]"),
    (lambda data: data.update(transformations=None), TypeError,
     "CalibrationRecord 'transformations' must be a JSON array; got "
     "NoneType"),
    (lambda data: data.update(fields=None), TypeError,
     "CalibrationRecord 'fields' must be a JSON array; got NoneType"),
    (lambda data: data.update(weight_matrix={
        'index': 'c_A', 'columns': ['c_A', 'T'], 'values': [[1.0, 0.0],
                                                          [0.0, 1.0]]}),
     TypeError, "weight_matrix 'index' must be a JSON array of measurement "
                "names; got str"),
    (lambda data: data.update(weight_matrix={
        'index': ['c_A', 'T'], 'columns': None, 'values': [[1.0, 0.0],
                                                         [0.0, 1.0]]}),
     TypeError, "weight_matrix 'columns' must be a JSON array"),
    (lambda data: data.update(output_names='c_A'), TypeError,
     "CalibrationRecord 'output_names' must be a JSON array or null; got "
     "str"),
    (lambda data: data['parameters'][0].update(unit='1/s'), ValueError,
     r"ParameterRecord has missing keys \[\] and unknown keys \['unit'\]"),
    (lambda data: data['parameters'][0].update(seed=10 ** 400), ValueError,
     "seed = 1000.* is too large for a float"),
    (lambda data: data['experiments'].append(dict(data['experiments'][0])),
     ValueError, "Experiment name 'hot' is repeated"),
    (lambda data: data['covariance'].update(parameters=['E']), ValueError,
     r"covariance parameters \['E'\] must be the optimized parameters"),
    (lambda data: data['experiments'][0]['args'].append(
        {'__type__': 'set', 'items': []}), ValueError,
     "unknown '__type__' tag 'set'"),
    (lambda data: data['experiments'][0]['measurements'][0].update(
        values=[1.0, 'x', 0.5, 0.2]), TypeError,
     "entries must be numbers or null; got str 'x'"),
    (lambda data: data['experiments'][0]['measurements'][0].update(
        values=[1.0, 10 ** 400, 0.5, 0.2]), ValueError,
     "values entry 1000.* is too large for a float"),
])
def test_malformed_dictionaries_are_rejected(mutate, error, message):
    record, _, _ = fitted_record()
    data = record.to_dict()
    mutate(data)
    with pytest.raises(error, match=message):
        CalibrationRecord.from_dict(data)


@pytest.mark.parametrize('tagged, error, message', [
    ({'__type__': 'ndarray', 'dtype': 8, 'shape': [1], 'data': [1.0]},
     TypeError, "dtype must be a str"),
    ({'__type__': 'ndarray', 'dtype': 'not-a-dtype', 'shape': [1],
      'data': [1.0]}, ValueError, "unknown dtype 'not-a-dtype'"),
    ({'__type__': 'ndarray', 'dtype': '<c16', 'shape': [1], 'data': [1.0]},
     ValueError, "unsupported NumPy dtype complex128"),
    ({'__type__': 'ndarray', 'dtype': '|b1', 'shape': [2],
      'data': [1, 0]}, TypeError, "must be one of the JSON booleans"),
    ({'__type__': 'ndarray', 'dtype': '<i8', 'shape': [1], 'data': [1.0]},
     TypeError, "must be one of the JSON integers"),
    ({'__type__': 'ndarray', 'dtype': '<i8', 'shape': [1], 'data': [True]},
     TypeError, "must be one of the JSON integers"),
    ({'__type__': 'ndarray', 'dtype': '<f8', 'shape': [1], 'data': [True]},
     TypeError, "must be one of the JSON numbers"),
    ({'__type__': 'ndarray', 'dtype': '|u1', 'shape': [1], 'data': [300]},
     ValueError, "out of range for dtype uint8"),
    ({'__type__': 'ndarray', 'dtype': '<i8', 'shape': [1],
      'data': [2 ** 70]}, ValueError, "out of range for dtype int64"),
    ({'__type__': 'ndarray', 'dtype': '<i2', 'shape': [1],
      'data': [-40000]}, ValueError, "out of range for dtype int16"),
    ({'__type__': 'ndarray', 'dtype': '<f4', 'shape': [1], 'data': [1e300]},
     ValueError, "out of range for dtype float32"),
    ({'__type__': 'ndarray', 'dtype': '<f8', 'shape': [1],
      'data': [10 ** 400]}, ValueError, "out of range for dtype float64"),
    ({'__type__': 'ndarray', 'dtype': '<f4', 'shape': [1], 'data': [0.1]},
     ValueError, "dtype float32 cannot represent exactly"),
    ({'__type__': 'ndarray', 'dtype': '<f8', 'shape': [2], 'data': [1.0]},
     ValueError, r"1 entries, which do not fill shape \(2,\)"),
    ({'__type__': 'ndarray', 'dtype': '<f8', 'shape': [2 ** 64],
      'data': [1.0]}, ValueError,
     r"1 entries, which do not fill shape \(18446744073709551616,\)"),
    ({'__type__': 'numpy_scalar', 'dtype': '<i4', 'value': 1.5},
     TypeError, "must be one of the JSON integers"),
])
def test_tagged_values_are_decoded_strictly(tagged, error, message):
    record, _, _ = fitted_record()
    data = record.to_dict()
    data['experiments'][0]['args'].append(tagged)
    # Ranges are checked before NumPy converts, so NumPy 1.x emits no
    # wrap-around DeprecationWarning or overflow RuntimeWarning.
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        with pytest.raises(error, match=message):
            CalibrationRecord.from_dict(data)


def test_bytes_options_left_by_cyipopt_are_decoded():
    """Decode the in-place option rewrite of cyipopt (issue #476).

    ``_effective_options`` is tested directly because only a real cyipopt
    solve produces bytes entries; the cyipopt lane (test_records_ipopt.py)
    exercises it through ``from_estimation``.
    """
    rewritten = {b'tol': 1e-8, b'print_level': 0,
                 b'mu_strategy': b'adaptive'}  # [-]
    assert _effective_options(rewritten, 'IPOPT') == {
        'tol': 1e-8, 'mu_strategy': 'adaptive'}
    assert _effective_options({'verbose': False,
                               'max_fun_eval': LM_MAX_FUN_EVAL,
                               'full_output': True}, 'LM') == {
        'max_fun_eval': LM_MAX_FUN_EVAL}
    with pytest.raises(ValueError, match=r"optim_options key b'\\xff' is "
                                         "not UTF-8"):
        _effective_options({b'\xff': 1}, 'IPOPT')
    with pytest.raises(ValueError, match="value of 'mu_strategy'"):
        _effective_options({b'mu_strategy': b'\xff'}, 'IPOPT')
    with pytest.raises(ValueError, match="holds 'tol' both as str and as "
                                         "bytes"):
        _effective_options({'tol': 1e-8, b'tol': 1e-9}, 'IPOPT')


@pytest.mark.parametrize('text, message', [
    ('{"schema": NaN}', 'standard JSON; found NaN'),
    ('{"schema": -Infinity}', 'standard JSON; found -Infinity'),
    ('{"case": "a", "case": "b"}', r"repeated \['case'\]"),
])
def test_non_standard_json_is_rejected(text, message):
    with pytest.raises(ValueError, match=message):
        CalibrationRecord.from_json(text)


def provides_imported_pharmapy() -> bool:
    """Tell whether the installed distribution is this source tree.

    Returns
    -------
    bool
        True if ``pharmapy-sim`` is installed and either one of its recorded
        files is this tree's ``PharmaPy/__init__.py`` or its PEP 610
        ``direct_url.json`` declares an editable install of this tree.
    """
    from importlib import metadata
    try:
        distribution = metadata.distribution('pharmapy-sim')
    except metadata.PackageNotFoundError:
        return False
    package_init = (ROOT / 'PharmaPy' / '__init__.py').resolve()
    if any(Path(distribution.locate_file(file)).resolve() == package_init
           for file in distribution.files or ()):
        return True
    direct_url = json.loads(distribution.read_text('direct_url.json')
                            or 'null')
    return (isinstance(direct_url, dict)
            and direct_url.get('dir_info', {}).get('editable') is True
            and direct_url.get('url') == ROOT.resolve().as_uri())


def test_model_revision_detects_the_installed_version_by_default():
    import PharmaPy
    assert Path(PharmaPy.__file__).resolve().parent == ROOT / 'PharmaPy'
    tomllib = pytest.importorskip('tomllib')
    with open(ROOT / 'pyproject.toml', 'rb') as file:
        declared = tomllib.load(file)['project']['version']
    # Only a distribution that provides this tree describes the code run.
    expected = declared if provides_imported_pharmapy() else None
    assert ModelRevision('decay').pharmapy_version == expected
    assert ModelRevision('decay', pharmapy_version='0.9').pharmapy_version \
        == '0.9'
    assert ModelRevision.from_dict({'name': 'decay'}).pharmapy_version is None


FOREIGN_VERSION = '9.9.9'  # version label of the synthetic distribution


def run_with_foreign_distribution(tmp_path, editable_directory):
    """Report the detected version beside a foreign editable distribution.

    A child process puts a synthetic ``pharmapy_sim-9.9.9.dist-info``
    first on ``sys.path``, so ``importlib.metadata`` finds it before any
    installed copy, while ``PharmaPy`` is imported from this tree. The test
    process itself is not changed.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Scratch directory holding the distribution metadata.
    editable_directory : pathlib.Path
        Directory named by the distribution's PEP 610 editable
        ``direct_url.json``.

    Returns
    -------
    tuple of str
        ``metadata.version('pharmapy-sim')``,
        ``ModelRevision('m').pharmapy_version`` (``'None'`` if None) and the
        imported package directory, as printed by the child.
    """
    site = tmp_path / 'site'
    info = site / f'pharmapy_sim-{FOREIGN_VERSION}.dist-info'
    info.mkdir(parents=True)
    (info / 'METADATA').write_text(
        f'Metadata-Version: 2.1\nName: pharmapy-sim\n'
        f'Version: {FOREIGN_VERSION}\n')
    (info / 'direct_url.json').write_text(json.dumps(
        {'url': editable_directory.resolve().as_uri(),
         'dir_info': {'editable': True}}))
    script = textwrap.dedent(f"""
        import sys
        sys.path.insert(0, {str(site)!r})
        sys.path.insert(1, {str(ROOT)!r})
        from importlib import metadata
        import PharmaPy
        from PharmaPy.Records import ModelRevision
        print(metadata.version('pharmapy-sim'))
        print(ModelRevision('m').pharmapy_version)
        print(PharmaPy.__path__[0])
    """)
    environment = os.environ.copy()
    environment['MPLCONFIGDIR'] = str(tmp_path)
    environment['MPLBACKEND'] = 'Agg'
    result = subprocess.run([sys.executable, '-c', script], cwd=tmp_path,
                            env=environment, capture_output=True, text=True,
                            check=False)
    assert result.returncode == 0, result.stderr
    return tuple(result.stdout.split())


@pytest.mark.parametrize('editable, expected', [
    ('parent', 'None'),  # an outer project containing this tree
    ('unrelated', 'None'),
    ('tree', FOREIGN_VERSION),  # exactly this project directory
])
def test_installed_version_requires_the_distribution_of_this_tree(
        tmp_path, editable, expected):
    directory = {'parent': ROOT.parent, 'unrelated': tmp_path,
                 'tree': ROOT}[editable]
    version, recorded, package = run_with_foreign_distribution(tmp_path,
                                                               directory)
    assert Path(package).resolve() == (ROOT / 'PharmaPy').resolve()
    assert version == FOREIGN_VERSION
    assert recorded == expected


@pytest.mark.parametrize('url, windows, expected', [
    ('file:///home/user/project', False, '/home/user/project'),
    ('file:///home/user/my%20project%23', False, '/home/user/my project#'),
    ('file://localhost/home/user/project', False, '/home/user/project'),
    ('file://server/share/project', False, None),  # no UNC paths on POSIX
    ('file:///C:/Users/user/my%20project', True, 'C:/Users/user/my project'),
    ('file:///c%7C/Users/user', True, 'c:/Users/user'),  # legacy C|
    ('file:///C|/Users/user', True, 'C:/Users/user'),
    ('file://localhost/D:/project', True, 'D:/project'),
    ('file://server/share/my%20project', True, '//server/share/my project'),
    ('https://example.org/project', True, None),
    ('https://example.org/project', False, None),
])
def test_file_urls_become_local_paths(url, windows, expected):
    from PharmaPy.Records import _file_url_path
    path = _file_url_path(url, windows=windows)
    if expected is None:
        assert path is None
    elif windows:
        # Windows path semantics on every platform: drive, UNC root and
        # case-insensitive comparison.
        assert PureWindowsPath(path) == PureWindowsPath(expected)
        assert PureWindowsPath(path).drive == PureWindowsPath(
            expected).drive
    else:
        assert path == expected


def test_file_url_paths_follow_the_running_platform():
    from PharmaPy.Records import _file_url_path
    url = ROOT.resolve().as_uri()
    assert Path(_file_url_path(url)).resolve() == ROOT.resolve()
    assert _file_url_path(url) == _file_url_path(
        url, windows=os.name == 'nt')


def components_of(record):
    """List a record and every component record it holds.

    Parameters
    ----------
    record : CalibrationRecord
        Record.

    Returns
    -------
    list
        The record, its model, sources, transformations, parameters,
        solver settings (if any), fields and their identities.
    """
    items = [record, record.model, *record.sources, *record.transformations,
             *record.parameters, *record.fields,
             *(field.identity for field in record.fields)]
    return items + ([] if record.solver is None else [record.solver])


@pytest.mark.parametrize('duplicate', [
    copy.copy, copy.deepcopy,
    lambda item: pickle.loads(pickle.dumps(item)),
], ids=['copy', 'deepcopy', 'pickle'])
def test_records_copy_and_pickle_exactly(duplicate):
    manual = manual_record()
    captured = fitted_record()[0]
    items = components_of(manual) + components_of(captured)
    # Every record class is covered.
    assert {type(item) for item in items} == {
        CalibrationRecord, ModelRevision, DataSource, Transformation,
        ParameterRecord, SolverSettings, FieldRecord, FieldIdentity}
    for item in items:
        duplicated = duplicate(item)
        assert type(duplicated) is type(item)
        assert duplicated is not item
        assert duplicated == item
        assert json.dumps(duplicated.to_dict()) == json.dumps(item.to_dict())
    # Stored PharmaPy versions are kept, never detected again.
    for version in (None, '0.9'):
        model = ModelRevision('decay', pharmapy_version=version)
        assert duplicate(model).pharmapy_version == version
    assert duplicate(captured).fitted_values.tolist() == \
        captured.fitted_values.tolist()


def test_from_estimation_rejects_values_that_are_not_experiments():
    _, estimator, experiments = fitted_record()
    mixed = {'hot': {'c_A': experiments['hot'].measurements['c_A']},
             'cold': experiments['cold']}
    with pytest.raises(TypeError, match=r"^experiments\['hot'\] must be an "
                                        r"Experiment; got dict$"):
        capture(estimator, mixed)


def test_unfitted_records_have_no_optimizer_inputs():
    record = CalibrationRecord(**record_arguments())
    assert record.fitted_values is None
    with pytest.raises(ValueError, match="has no solver settings"):
        record.optimizer_inputs()
    estimator = ParameterEstimation(decay, **record.estimation_inputs())
    with pytest.raises(ValueError, match="has no fitted values"):
        record.check_rerun(estimator, rtol=REPLAY_RTOL, atol=REPLAY_ATOL)


# ------------------------------------------- metadata, derivatives, adapters


def test_prediction_metadata_merges_labels_across_experiments():
    experiments = make_experiments()
    # Only the cold run declares the c_A basis.
    experiments['hot'] = Experiment(
        {'c_A': Measurement(0, HOT_CA_TIME, HOT_CA, uncertainty=HOT_CA_STD,
                            units='mol/L'),
         'T': experiments['hot'].measurements['T']},
        args=experiments['hot'].args, kwargs=experiments['hot'].kwargs)
    estimator = ParameterEstimation(decay, [SEED_RATE], experiments,
                                    name_params=['k'])
    estimator.optimize_fn(verbose=False)
    record = capture(estimator, experiments)
    hot_prediction = next(field for field in record.fields
                          if (field.experiment, field.name)
                          == ('hot', 'c_A'))
    assert hot_prediction.identity.basis == 'molar concentration'
    assert CalibrationRecord.from_json(record.to_json()) == record


def prediction(name='c_A', experiment='hot', units='mol/L',
               basis='molar concentration', x_units='s'):
    """Build a user-supplied measurement prediction of the hot run.

    Parameters
    ----------
    name, experiment, units, basis, x_units : str or None
        Identity and labels of the prediction.

    Returns
    -------
    FieldRecord
        Prediction on the hot model grid, values arbitrary [units].
    """
    return FieldRecord(
        FieldIdentity(name, 'measurement_prediction', units, basis=basis),
        GRID['hot'], [2.0, 1.2, 0.73, 0.27], x_units=x_units,
        experiment=experiment)


@pytest.mark.parametrize('field, message', [
    (prediction(experiment=None), "Measurement prediction 'c_A' must name "
                                  "its experiment"),
    (prediction(name='c_B'), r"names no measurement of the record "
                             r"\['c_A', 'T'\]"),
    (prediction(units='g/L'), r"has units and basis \('g/L', 'molar "
                              r"concentration'\); the measurement declares "
                              r"\('mol/L', 'molar concentration'\)"),
    (prediction(basis=None), r"has units and basis \('mol/L', None\)"),
    (prediction(x_units='min'), "has x_units 'min'; the experiment declares "
                                "'s'"),
])
def test_user_measurement_predictions_are_validated(field, message):
    with pytest.raises(ValueError, match=message):
        CalibrationRecord(**record_arguments(fields=[field]))


def test_user_measurement_prediction_round_trips():
    record = CalibrationRecord(**record_arguments(fields=[prediction()]))
    assert CalibrationRecord.from_json(record.to_json()) == record


def test_adapters_filter_a_captured_record_by_experiment_and_kind():
    record, _, _ = fitted_record()
    with pytest.raises(ValueError, match=r"repeated \['c_A', 'T'\]. Select "
                                         "one experiment and kind"):
        fields_to_array(record.fields)
    x, array = fields_to_array(record.fields, experiment='cold',
                               kind='measurement_prediction')
    assert_bits_equal(x, GRID['cold'])
    rate = record.fitted_values[0]  # [1/s]
    expected = [closed_form(rate, 'cold', time)
                for time in GRID['cold']]  # [mol/L], [K]
    np.testing.assert_allclose(array, expected, rtol=ROUNDOFF_RTOL,
                               atol=ROUNDOFF_ATOL)
    x, mapping = fields_to_mapping(record.fields, experiment='hot')
    assert_bits_equal(x, GRID['hot'])
    assert list(mapping) == ['c_A', 'T']
    with pytest.raises(ValueError, match="passes the experiment and kind "
                                         "filters"):
        fields_to_mapping(record.fields, experiment='warm')
    with pytest.raises(ValueError, match="kind must be one of"):
        fields_to_mapping(record.fields, kind='prediction')


def decay_with_sensitivities(params, time_s, initial_mol_l,
                             rise_k_l_mol=0.0, temp0_k=300.0):
    """Return ``decay`` and its analytic rate sensitivities.

    Parameters
    ----------
    params, time_s, initial_mol_l, rise_k_l_mol, temp0_k
        As for ``decay``.

    Returns
    -------
    tuple of numpy.ndarray
        States, shape (n_times, 2) [mol/L, K], and sensitivities, shape
        (2 * n_times, 1), state-major: dc_A/dk [mol/L*s], then dT/dk [K*s].
    """
    states = decay(params, time_s, initial_mol_l, rise_k_l_mol, temp0_k)
    # d(converted)/dk = t * c_A0 * exp(-k t) [mol/L*s]
    converted_rate = time_s * initial_mol_l * np.exp(-params[0] * time_s)
    return states, np.concatenate(
        (-converted_rate, rise_k_l_mol * converted_rate))[:, np.newaxis]


def rate_jacobian(params, time_s, initial_mol_l, rise_k_l_mol=0.0,
                  temp0_k=300.0):
    """Return the analytic rate sensitivities of ``decay`` as a jac_fun.

    Parameters
    ----------
    params, time_s, initial_mol_l, rise_k_l_mol, temp0_k
        As for ``decay``.

    Returns
    -------
    numpy.ndarray
        Shape (2 * n_times, 1): dc_A/dk [mol/L*s], then dT/dk [K*s].
    """
    return decay_with_sensitivities(params, time_s, initial_mol_l,
                                    rise_k_l_mol, temp0_k)[1]


def relative_step(params):
    """Return a user finite-difference step, 1e-4 relative to |k|.

    Parameters
    ----------
    params : numpy.ndarray
        Full parameter vector [1/s].

    Returns
    -------
    numpy.ndarray
        Step per parameter [1/s].
    """
    relative = 1e-4  # [-], arbitrary user choice for this test
    return relative * np.abs(params)


# Numeric dx_finitediff [1/s]: 1/300 of the seed rate 0.3 1/s and about
# twice the default forward step (1e-3 relative, i.e. 5e-4 1/s near the
# optimum k = 0.501 1/s). Rerunning with the default step instead would
# produce other finite-difference Jacobians and LM iterates, which the
# four-ulp replay tolerance detects.
FD_STEP = 1e-3  # [1/s]
# Asymmetric per-parameter step for (rise [K*L/mol], k [1/s]): arbitrary
# user values, far apart so a swapped or scalar replay is detected.
FD_STEP_ARRAY = np.array([1e-4, 1e-5])  # [K*L/mol], [1/s]


@pytest.mark.parametrize('func, options, expected', [
    (decay, {'dx_finitediff': FD_STEP},
     {'jacobian': 'finite_difference', 'dx_finitediff': FD_STEP}),
    (decay, {'jac_fun': rate_jacobian},
     {'jacobian': 'jac_fun', 'dx_finitediff': 'default'}),
    (decay, {'dx_finitediff': relative_step},
     {'jacobian': 'finite_difference', 'dx_finitediff': 'callable'}),
    (decay_with_sensitivities, {},
     {'jacobian': 'model_sensitivities', 'dx_finitediff': 'default'}),
])
def test_derivative_settings_are_recorded_and_replayed(func, options,
                                                       expected):
    experiments = make_experiments()
    estimator = ParameterEstimation(func, [SEED_RATE], experiments,
                                    name_params=['k'], **options)
    estimator.optimize_fn(verbose=False)
    record = capture(estimator, experiments)
    loaded = CalibrationRecord.from_json(record.to_json())
    assert loaded.solver.derivatives == expected
    inputs = loaded.estimation_inputs()
    if expected['dx_finitediff'] == FD_STEP:
        assert inputs['dx_finitediff'] == FD_STEP
    else:
        assert 'dx_finitediff' not in inputs
    callables = {key: value for key, value in options.items()
                 if callable(value)}
    if callables:
        key = next(iter(callables))
        with pytest.raises(ValueError, match=f"pass it as rerun\\(\\.\\.\\., "
                                             f"{key}=\\.\\.\\.\\)|pass the "
                                             f"original function as rerun"):
            loaded.rerun(func, rtol=REPLAY_RTOL, atol=REPLAY_ATOL)
    rerun = loaded.rerun(func, rtol=REPLAY_RTOL, atol=REPLAY_ATOL,
                         **callables)
    np.testing.assert_allclose(rerun.params_convg, estimator.params_convg,
                               rtol=REPLAY_RTOL, atol=REPLAY_ATOL)


def test_array_step_is_recorded_and_replayed(tmp_path):
    experiments = experiments_without_rise()
    estimator = ParameterEstimation(
        decay_rise_first, [RISE_PARAMETER, SEED_RATE], experiments,
        name_params=['rise', 'k'], dx_finitediff=FD_STEP_ARRAY.copy())
    estimator.optimize_fn(verbose=False)
    record = capture(estimator, experiments,
                     parameter_units=['K*L/mol', '1/s'])
    path = tmp_path / 'array-step.json'
    record.save(path)
    for candidate in (record, CalibrationRecord.load(path)):
        step = candidate.solver.derivatives['dx_finitediff']
        assert type(step) is np.ndarray and step.dtype == np.float64
        np.testing.assert_array_equal(step, [1e-4, 1e-5])
        replayed = candidate.estimation_inputs()['dx_finitediff']
        np.testing.assert_array_equal(replayed, [1e-4, 1e-5])
        rerun = candidate.rerun(decay_rise_first, rtol=REPLAY_RTOL,
                                atol=REPLAY_ATOL)
        np.testing.assert_allclose(rerun.params_convg,
                                   estimator.params_convg,
                                   rtol=REPLAY_RTOL, atol=REPLAY_ATOL)


@pytest.mark.parametrize('descriptor, error, message', [
    ({'jacobian': 'analytic', 'dx_finitediff': 'default'}, ValueError,
     "jacobian must be one of"),
    ({'jacobian': 'jac_fun', 'dx_finitediff': 'auto'}, ValueError,
     r"dx_finitediff must be one of \('default', 'callable'\)"),
    ({'jacobian': 'jac_fun', 'dx_finitediff': True}, TypeError,
     "got bool"),
    ({'jacobian': 'jac_fun', 'dx_finitediff': np.inf}, ValueError,
     "cannot be recorded"),
    ({'jacobian': 'jac_fun'}, ValueError, r"missing keys \['dx_finitediff'"),
])
def test_derivative_descriptors_are_validated(descriptor, error, message):
    with pytest.raises(error, match=message):
        SolverSettings('LM', derivatives=descriptor)


def test_simulated_estimators_are_not_captured():
    record, estimator, experiments = fitted_record()
    estimator.optimize_fn(simulate=True, verbose=False)
    with pytest.raises(ValueError, match="simulate=True"):
        capture(estimator, experiments)


def test_new_files_get_the_permissions_open_would_give(tmp_path):
    reference = tmp_path / 'reference.json'
    reference.write_text('{}')  # created by open() under the same umask
    path = tmp_path / 'record.json'
    manual_record().save(path)
    assert stat.S_IMODE(os.stat(path).st_mode) == stat.S_IMODE(
        os.stat(reference).st_mode)


def test_save_writes_through_a_symbolic_link(tmp_path):
    target = tmp_path / 'stored.json'
    target.write_text('old')
    link = tmp_path / 'link.json'
    try:
        link.symlink_to(target)
    except (OSError, NotImplementedError):
        pytest.skip('symbolic links are not supported here')
    record = manual_record()
    record.save(link)
    assert link.is_symlink()
    assert CalibrationRecord.load(target) == record
    assert sorted(item.name for item in tmp_path.iterdir()) == [
        'link.json', 'stored.json']


def keyword_record(**kwargs):
    """Build a minimal record whose experiment has the given keywords.

    Parameters
    ----------
    **kwargs
        Callback keywords of the single experiment.

    Returns
    -------
    CalibrationRecord
        Unfitted record.
    """
    experiments = {'run': Experiment(
        {'c_A': Measurement(0, [0.0, 1.0], [1.0, 0.5], units='mol/L')},
        args=(1.0,), kwargs=kwargs)}
    return CalibrationRecord(**record_arguments(experiments=experiments))


@pytest.mark.parametrize('first, second', [
    ({'flag': True}, {'flag': 1}),
    ({'value': 1}, {'value': 1.0}),
    ({'value': 0.0}, {'value': -0.0}),
    ({'a': 1.0, 'b': 2.0}, {'b': 2.0, 'a': 1.0}),
])
def test_records_compare_by_exact_json(first, second):
    one, other = keyword_record(**first), keyword_record(**second)
    assert first == second  # equal as Python dicts
    assert one != other
    assert CalibrationRecord.from_json(one.to_json()) == one
    assert CalibrationRecord.from_json(other.to_json()) == other
