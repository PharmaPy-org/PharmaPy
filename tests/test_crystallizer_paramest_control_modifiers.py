"""Crystallizer parameter-estimation callbacks apply control modifiers (#492).

``_BaseCryst.paramest_wrapper(..., modify_controls=...)``, shared by
``BatchCryst``, ``MSMPR`` and ``SemibatchCryst``, follows the reactor
contract of #271 through ``ProcessControl.apply_control_modifier``: each
call evaluates a copy of the configured controls in which ``fun`` and
``args`` are replaced and ``kwargs`` is merged key by key; None or ``{}``
evaluates the configured controls; unknown control names or record entries
raise ``ValueError`` before any state changes; the configured controls are
restored afterwards, also when the evaluation raises; and no caller
dictionary is modified. The crystallizer constructors, like the
tank-reactor constructors that share ``ProcessControl.normalize_controls``,
store new control records, so they neither share nor modify the caller's
dictionaries.

Fixture: the seeded three-bin 1D-FVM ``BatchCryst`` of the #225 heat-duty
tests (real compound database, phases, CVode and cooling water), with
growth-only kinetics and a synthetic solubility rising linearly with
temperature, so a prescribed temperature profile changes the crystal size
distribution. Temperature profiles are compared with values of the control
function written out by hand. Expected solver outputs come from reference
crystallizers configured directly, through the constructor, with the control
an experiment should use and integrated by ``solve_unit``; they never pass
through the modifier path under test, and a separation guard confirms that
the configured and modified experiments differ. Validation, caller
isolation and restoration on error run in the core lane; trajectory checks
need the optional Assimulo stack and take about thirty short solves.
"""

import copy

import numpy as np
import pytest

from PharmaPy import Reactors
from PharmaPy.Crystallizers import BatchCryst, MSMPR, SemibatchCryst
from PharmaPy.Kinetics import CrystKinetics
from PharmaPy.ParamEstim import Experiment, Measurement
from PharmaPy.Phases import LiquidPhase, SolidPhase
from PharmaPy.ProcessControl import analyze_controls
from PharmaPy.SimExec import SimulationExec
from PharmaPy.Utilities import CoolingWater

CRYSTALLIZER_CLASSES = [BatchCryst, MSMPR, SemibatchCryst]
# Units whose constructors normalize controls with normalize_controls.
CONTROLLED_UNIT_CLASSES = CRYSTALLIZER_CLASSES + [
    Reactors.BatchReactor, Reactors.CSTR, Reactors.SemibatchReactor]
# Default start of the control function, as in the issue #492 reproduction;
# the configured start differs so that discarded configured kwargs show.
DEFAULT_START = 300.0  # [K]
CONFIGURED_START = 310.0  # [K], issue #492 configured start
REPLACED_START = 305.0  # [K], positional start supplied by an args modifier
HEATING_SLOPE = 1.0  # [K/s], issue #492 modifier: 310 K to 315 K over 5 s
HOLD_TIME = 3.0  # [s], ramp-then-hold profile stops rising here
END_TIME = 5.0  # [s], issue #492 horizon
SAMPLE_COUNT = 6  # [-], 1 s sampling
# Crystallizer callbacks return the solver rows, which include the charge
# time, so experiment grids start at 0 s (issue #328).
TIMES = np.linspace(0.0, END_TIME, SAMPLE_COUNT)  # [s]
# Control values at TIMES, written out instead of recomputed.
CONFIGURED_PROFILE = np.full(SAMPLE_COUNT, 310.0)  # [K]
HEATED_PROFILE = np.array([310.0, 311.0, 312.0, 313.0, 314.0, 315.0])  # [K]

# Issue #225 heat-duty fixture: 0.9 L of liquid with mass fractions of A to
# E (about 93 kg/m**3 of A with the compound database) and an asymmetric
# seed on unequal size bins.
LIQUID_VOLUME = 9e-4  # [m**3]
LIQUID_FRACTIONS = [0.1, 0.1, 0.1, 0.1, 0.6]  # [-], A to E
SEED_GRID = np.array([10.0, 20.0, 40.0])  # [um]
SEED_POPULATION = np.array([1e7, 2e7, 1e7])  # [#/um]
SHAPE_FACTOR = 0.5  # [-], crystal volume shape factor
SOLID_FRACTIONS = [1, 0, 0, 0, 0]  # [-], pure A crystals
UTILITY_FLOW = 1e-5  # [m**3/s], #225 cooling-water flow
UTILITY_INLET_TEMPERATURE = 290.0  # [K], #225 cooling-water inlet
# Synthetic solubility c_sat = 80 kg/m**3 + 1 kg/m**3/K * (T - 310 K): with
# about 93 kg/m**3 of A, the relative supersaturation falls from 0.17 at
# 310 K to 0.10 at 315 K, so heating slows growth but keeps it positive.
SOLUBILITY_AT_START = 80.0  # [kg/m**3], at CONFIGURED_START
SOLUBILITY_SLOPE = 1.0  # [kg/m**3/K]
# Ascending polynomial coefficients [kg/m**3], [kg/m**3/K].
SOLUBILITY_COEFFICIENTS = [
    SOLUBILITY_AT_START - SOLUBILITY_SLOPE * CONFIGURED_START,
    SOLUBILITY_SLOPE]
# Growth 1 um/s at unit relative supersaturation, no activation energy,
# linear driving force: [um/s], [J/mol], [-].
GROWTH = (1.0, 0.0, 1.0)
# The reference and the experiment integrate identical inputs with the same
# deterministic CVode settings, so only floating-point roundoff may differ.
PROFILE_RTOL = 1e-12  # [-]
# Heating shifts the size distribution by about 2% of its peak at 5 s; 0.1%
# guards against comparing identical experiments.
MIN_RELATIVE_SEPARATION = 1e-3  # [-]
POPULATION_COLUMNS = [0, 1, 2]  # [#/um], the three size bins


def ramp_temperature(time, start=DEFAULT_START, slope=0.0):
    """Return a linear temperature ramp.

    Parameters
    ----------
    time : float or numpy.ndarray
        Time [s].
    start : float, optional
        Temperature at zero time [K].
    slope : float, optional
        Heating rate [K/s].

    Returns
    -------
    float or numpy.ndarray
        Temperature [K], shaped like ``time``.
    """
    return start + slope * np.asarray(time)  # [K]


def ramp_then_hold(time, start=DEFAULT_START, slope=0.0):
    """Return a linear ramp that holds its value after ``HOLD_TIME``.

    Parameters
    ----------
    time : float or numpy.ndarray
        Time [s].
    start : float, optional
        Temperature at zero time [K].
    slope : float, optional
        Heating rate before ``HOLD_TIME`` [K/s].

    Returns
    -------
    float or numpy.ndarray
        Temperature [K], shaped like ``time``.
    """
    return start + slope * np.minimum(time, HOLD_TIME)  # [K]


def _crystallizer(data_path, control):
    """Build the configured fixture crystallizer.

    Parameters
    ----------
    data_path : dict
        Repository thermodynamic data paths.
    control : dict or callable
        Configured ``temp`` control: a record or a callable returning
        temperature [K] from time [s].

    Returns
    -------
    BatchCryst
        Seeded 1D-FVM crystallizer with prescribed temperature, growth-only
        kinetics and the temperature-dependent solubility above.
    """
    path = str(data_path['flowsheet'] / 'compound_database.json')
    unit = BatchCryst('A', controls={'temp': control})
    liquid = LiquidPhase(path, temp=CONFIGURED_START, vol=LIQUID_VOLUME,
                         mass_frac=LIQUID_FRACTIONS)
    solid = SolidPhase(path, temp=CONFIGURED_START, x_distrib=SEED_GRID,
                       distrib=SEED_POPULATION, kv=SHAPE_FACTOR,
                       mass_frac=SOLID_FRACTIONS)
    unit.Phases = (liquid, solid)
    unit.Kinetics = CrystKinetics(coeff_solub=SOLUBILITY_COEFFICIENTS,
                                  growth=GROWTH)
    unit.Utility = CoolingWater(vol_flow=UTILITY_FLOW,
                                temp_in=UTILITY_INLET_TEMPERATURE)
    return unit


def _configured_record():
    """Return the issue #492 configured control record.

    Returns
    -------
    dict
        ``ramp_temperature`` record with the configured start [K] and a
        zero slope [K/s].
    """
    return {'fun': ramp_temperature,
            'kwargs': {'start': CONFIGURED_START, 'slope': 0.0}}


def _reference(data_path, slope):
    """Integrate a crystallizer configured directly with a ramp slope.

    Parameters
    ----------
    data_path : dict
        Repository thermodynamic data paths.
    slope : float
        Configured heating rate [K/s] from ``CONFIGURED_START``.

    Returns
    -------
    numpy.ndarray
        Solver rows at ``TIMES``: populations [#/um], concentrations
        [kg/m**3] and liquid volume [m**3], shape ``(SAMPLE_COUNT, 9)``.
    """
    unit = _crystallizer(data_path, {
        'fun': ramp_temperature,
        'kwargs': {'start': CONFIGURED_START, 'slope': slope}})
    _, states = unit.solve_unit(time_grid=TIMES, verbose=False)
    return states


def _evaluate(unit, modify_controls, **kwargs):
    """Evaluate the crystallizer callback at the current parameters.

    Parameters
    ----------
    unit : BatchCryst
        Configured crystallizer.
    modify_controls : dict or None
        Control modifier passed to ``paramest_wrapper``.
    **kwargs
        Further ``paramest_wrapper`` keywords.

    Returns
    -------
    numpy.ndarray
        Solver rows at ``TIMES`` (see ``_reference``).
    """
    return unit.paramest_wrapper(unit.Kinetics.concat_params(), TIMES,
                                 modify_controls=modify_controls, **kwargs)


def _assert_separated(first, second):
    """Assert that two population profiles differ beyond roundoff.

    Parameters
    ----------
    first, second : numpy.ndarray
        Solver rows (see ``_reference``) of two experiments.
    """
    populations = first[:, POPULATION_COLUMNS]  # [#/um]
    difference = np.abs(populations - second[:, POPULATION_COLUMNS])  # [#/um]
    assert difference.max() > MIN_RELATIVE_SEPARATION * populations.max()


@pytest.mark.unit
def test_analyze_controls_returns_new_records():
    record = {'fun': ramp_temperature}
    caller = {'temp': record, 'temp_ht': ramp_then_hold}
    before = copy.deepcopy(caller)
    controls = analyze_controls(caller)
    assert caller == before
    assert record == {'fun': ramp_temperature}
    assert controls['temp'] is not record
    assert controls == {
        'temp': {'fun': ramp_temperature, 'args': (), 'kwargs': {}},
        'temp_ht': {'fun': ramp_then_hold, 'args': (), 'kwargs': {}}}


def _bare_unit(unit_cls, controls):
    """Construct a unit without phases or kinetics.

    Parameters
    ----------
    unit_cls : type
        A crystallizer or tank-reactor class.
    controls : dict or None
        Configured controls passed to the constructor.

    Returns
    -------
    object
        The constructed unit. Crystallizers target A; the semibatch
        reactor, which requires a tank volume, receives ``LIQUID_VOLUME``
        [m**3]. These units are never solved.
    """
    if unit_cls in CRYSTALLIZER_CLASSES:
        return unit_cls('A', controls=controls)
    if unit_cls is Reactors.SemibatchReactor:
        return unit_cls(vol_tank=LIQUID_VOLUME, controls=controls)
    return unit_cls(controls=controls)


@pytest.mark.unit
@pytest.mark.parametrize('unit_cls', CONTROLLED_UNIT_CLASSES)
def test_constructor_does_not_share_caller_controls(unit_cls):
    caller = {'temp': _configured_record()}
    before = copy.deepcopy(caller)
    unit = _bare_unit(unit_cls, caller)
    assert caller == before  # no args default added to the caller's record
    assert unit.controls == {'temp': {
        'fun': ramp_temperature, 'args': (),
        'kwargs': {'start': CONFIGURED_START, 'slope': 0.0}}}
    assert unit.controls is not caller
    assert unit.controls['temp'] is not caller['temp']
    # Later changes to the caller's kwargs do not reach the unit.
    caller['temp']['kwargs']['start'] = REPLACED_START  # [K]
    assert unit.controls['temp']['kwargs'] == {'start': CONFIGURED_START,
                                               'slope': 0.0}


@pytest.mark.unit
@pytest.mark.parametrize('unit_cls', CRYSTALLIZER_CLASSES)
def test_constructor_rejects_unknown_record_entries(unit_cls):
    record = {'fun': ramp_temperature, 'kwarg': {'start': CONFIGURED_START}}
    with pytest.raises(ValueError, match=(
            r"^Control 'temp' must be callable or have exactly fun, args, "
            r"kwargs keys")):
        _bare_unit(unit_cls, {'temp': record})


# Rejected modifiers, the exception and the start of its message.
INVALID_MODIFIERS = {
    'unknown control': (
        {'vol': {'kwargs': {'slope': HEATING_SLOPE}}}, ValueError,
        r"^modify_controls names unknown controls \['vol'\]; the "
        r"crystallizer's configured controls are \['temp'\]"),
    'unknown entry': (
        {'temp': {'kwarg': {'slope': HEATING_SLOPE}}}, ValueError,
        r"^modify_controls\['temp'\] has unknown entries \['kwarg'\]; "
        r"allowed entries are \['fun', 'args', 'kwargs'\]"),
    'entry not a dictionary': (
        {'temp': HEATING_SLOPE}, TypeError,
        r"^modify_controls\['temp'\] must be a dictionary"),
    'kwargs not a dictionary': (
        {'temp': {'kwargs': HEATING_SLOPE}}, TypeError,
        r"^modify_controls\['temp'\]\['kwargs'\] must be a dictionary"),
    'modifier not a dictionary': (
        [('temp', {'kwargs': {'slope': HEATING_SLOPE}})], TypeError,
        r"^modify_controls must be a dictionary"),
    'fun not callable': (
        {'temp': {'fun': HEATING_SLOPE}}, TypeError,
        r"^Object passed to the 'fun' field must be a callable"),
    'args not a sequence': (
        {'temp': {'args': HEATING_SLOPE}}, TypeError,
        r"^Control 'temp' requires sequence args and dictionary kwargs"),
}
# State that the wrapper's reset and parameter update would overwrite.
ELAPSED_SENTINEL = 7.0  # [s], reset restores 0 s
LIQUID_TEMPERATURE_SENTINEL = 320.0  # [K], reset restores CONFIGURED_START


@pytest.mark.unit
@pytest.mark.parametrize('case', list(INVALID_MODIFIERS))
def test_invalid_modifiers_are_rejected_before_any_state_change(
        data_path, case):
    modifier, error, message = INVALID_MODIFIERS[case]
    unit = _crystallizer(data_path, _configured_record())
    configured = unit.controls
    configured_before = copy.deepcopy(configured)
    modifier_before = copy.deepcopy(modifier)
    unit.elapsed_time = ELAPSED_SENTINEL  # [s]
    unit.Liquid_1.temp = LIQUID_TEMPERATURE_SENTINEL  # [K]
    params = unit.Kinetics.concat_params()  # [kinetic parameter units]
    with pytest.raises(error, match=message):
        unit.paramest_wrapper(2 * params, TIMES, modify_controls=modifier)
    assert unit.controls is configured
    assert unit.controls == configured_before
    assert modifier == modifier_before
    assert unit.elapsed_time == ELAPSED_SENTINEL
    assert unit.Liquid_1.temp == LIQUID_TEMPERATURE_SENTINEL
    np.testing.assert_array_equal(unit.Kinetics.concat_params(), params)
    assert not hasattr(unit, 'result')  # nothing was integrated


@pytest.mark.unit
@pytest.mark.parametrize('unit_cls', CRYSTALLIZER_CLASSES)
@pytest.mark.parametrize('configured', [True, False])
def test_unknown_control_is_rejected_by_every_crystallizer(unit_cls,
                                                           configured):
    controls = {'temp': _configured_record()} if configured else None
    unit = _bare_unit(unit_cls, controls)
    names = r"\['temp'\]" if configured else r"\[\]"
    # The unit has no kinetics; the modifier is rejected before parameters
    # are used, so any parameter vector serves.
    params = np.ones(1)  # [kinetic parameter units]
    with pytest.raises(ValueError, match=(
            r"^modify_controls names unknown controls \['vol'\]; the "
            r"crystallizer's configured controls are " + names)):
        unit.paramest_wrapper(
            params, TIMES,
            modify_controls={'vol': {'kwargs': {'slope': HEATING_SLOPE}}})


@pytest.mark.unit
def test_configured_controls_are_restored_when_evaluation_raises(data_path):
    unit = _crystallizer(data_path, _configured_record())
    configured = unit.controls
    configured_before = copy.deepcopy(configured)
    # solve_unit rejects the unknown keyword after the experiment's controls
    # were installed, without needing the solver stack.
    with pytest.raises(TypeError, match="unexpected keyword argument "
                                        "'unsupported_option'"):
        _evaluate(unit, {'temp': {'kwargs': {'slope': HEATING_SLOPE}}},
                  run_args={'unsupported_option': True})
    assert unit.controls is configured
    assert unit.controls == configured_before


@pytest.mark.assimulo
@pytest.mark.integration
def test_issue_492_modifier_merges_and_does_not_carry_over(data_path):
    pytest.importorskip('assimulo')
    expected = {'configured': _reference(data_path, 0.0),
                'heated': _reference(data_path, HEATING_SLOPE)}
    _assert_separated(expected['configured'], expected['heated'])
    caller = {'temp': _configured_record()}
    caller_before = copy.deepcopy(caller)
    unit = _crystallizer(data_path, caller['temp'])
    configured = unit.controls
    configured_before = copy.deepcopy(configured)
    modifier = {'temp': {'kwargs': {'slope': HEATING_SLOPE}}}
    modifier_before = copy.deepcopy(modifier)

    # The configured 310 K start is kept and the slope modified.
    heated = _evaluate(unit, modifier)
    np.testing.assert_allclose(unit.result.temp, HEATED_PROFILE,
                               rtol=PROFILE_RTOL, atol=0.0)  # [K]
    np.testing.assert_allclose(heated, expected['heated'],
                               rtol=PROFILE_RTOL, atol=0.0)
    assert unit.controls is configured
    assert unit.controls == configured_before
    assert modifier == modifier_before
    assert caller == caller_before

    # Later experiments without a modifier use the configured 310 K.
    for no_modifier in ({}, None):
        constant = _evaluate(unit, no_modifier)
        np.testing.assert_allclose(unit.result.temp, CONFIGURED_PROFILE,
                                   rtol=PROFILE_RTOL, atol=0.0)  # [K]
        np.testing.assert_allclose(constant, expected['configured'],
                                   rtol=PROFILE_RTOL, atol=0.0)
        assert unit.controls is configured
        assert unit.controls == configured_before
    assert caller == caller_before


# (configured control record, modifier, temperature [K] at TIMES that the
# merged record must prescribe, written out by hand).
RECORD_MERGES = {
    'kwargs merged key by key': (
        _configured_record(),
        {'kwargs': {'slope': HEATING_SLOPE}},
        HEATED_PROFILE),
    'args replaced, kwargs kept': (
        {'fun': ramp_temperature, 'args': (CONFIGURED_START,),
         'kwargs': {'slope': HEATING_SLOPE}},
        {'args': (REPLACED_START,)},
        np.array([305.0, 306.0, 307.0, 308.0, 309.0, 310.0])),
    'fun replaced, kwargs kept': (
        {'fun': ramp_temperature,
         'kwargs': {'start': CONFIGURED_START, 'slope': HEATING_SLOPE}},
        {'fun': ramp_then_hold},
        np.array([310.0, 311.0, 312.0, 313.0, 313.0, 313.0])),
}


@pytest.mark.assimulo
@pytest.mark.integration
@pytest.mark.parametrize('case', list(RECORD_MERGES))
def test_modifier_entries_merge_into_configured_record(data_path, case):
    pytest.importorskip('assimulo')
    record, modifier_entry, temperature = RECORD_MERGES[case]
    unit = _crystallizer(data_path, record)
    configured_before = copy.deepcopy(unit.controls)
    modifier = {'temp': modifier_entry}
    modifier_before = copy.deepcopy(modifier)
    _evaluate(unit, modifier)
    np.testing.assert_allclose(unit.result.temp, temperature,
                               rtol=PROFILE_RTOL, atol=0.0)  # [K]
    assert unit.controls == configured_before
    assert modifier == modifier_before


def _estimation(data_path, order, form, expected):
    """Set up a two-experiment crystallizer estimation.

    Parameters
    ----------
    data_path : dict
        Repository thermodynamic data paths.
    order : tuple of str
        Experiment order, a permutation of ``('configured', 'heated')``.
    form : str
        ``'legacy'`` for named time arrays with ``control_modifiers`` (the
        configured experiment's entry is None), ``'experiment'`` for
        ``Experiment`` objects of which only the heated one carries
        ``modify_controls`` in its own kwargs.
    expected : dict
        Reference solver rows (see ``_reference``) per experiment name.

    Returns
    -------
    tuple
        The estimator and the fitted crystallizer.
    """
    modifiers = {'configured': None,
                 'heated': {'temp': {'kwargs': {'slope': HEATING_SLOPE}}}}
    path = str(data_path['flowsheet'] / 'compound_database.json')
    simulation = SimulationExec(path, {'CR01': []})
    simulation.CR01 = _crystallizer(data_path, _configured_record())
    if form == 'legacy':
        simulation.SetParamEstimation(
            {name: TIMES for name in order},
            {name: expected[name][:, POPULATION_COLUMNS] for name in order},
            measured_ind=POPULATION_COLUMNS,
            control_modifiers={name: modifiers[name] for name in order})
    else:
        experiments = {}
        for name in order:
            kwargs = ({} if modifiers[name] is None
                      else {'modify_controls': modifiers[name]})
            experiments[name] = Experiment(
                {f'bin_{column}': Measurement(column, TIMES,
                                              expected[name][:, column])
                 for column in POPULATION_COLUMNS},
                kwargs=kwargs)
        simulation.SetParamEstimation(experiments)
    return simulation.ParamInst, simulation.CR01


@pytest.mark.assimulo
@pytest.mark.integration
@pytest.mark.parametrize('form', ['legacy', 'experiment'])
@pytest.mark.parametrize('order', [('configured', 'heated'),
                                   ('heated', 'configured')])
def test_simulation_exec_experiments_use_their_own_controls(data_path, order,
                                                            form):
    pytest.importorskip('assimulo')
    expected = {'configured': _reference(data_path, 0.0),
                'heated': _reference(data_path, HEATING_SLOPE)}
    _assert_separated(expected['configured'], expected['heated'])
    estimator, unit = _estimation(data_path, order, form, expected)
    seed = estimator.param_seed[estimator.map_variable]
    configured = unit.controls
    peak = expected['configured'][:, POPULATION_COLUMNS].max()  # [#/um]
    # Two successive objective evaluations: neither the previous experiment
    # nor the previous evaluation may leave its control behind.
    for _ in range(2):
        residual = estimator.get_objective(seed, out_array=True)  # [#/um]
        np.testing.assert_allclose(residual, 0.0, rtol=0.0,
                                   atol=PROFILE_RTOL * peak)
        for index, name in enumerate(order):
            np.testing.assert_allclose(
                estimator.y_runs[index], expected[name][:, POPULATION_COLUMNS],
                rtol=PROFILE_RTOL, atol=0.0)
    assert unit.controls is configured
    assert unit.controls['temp']['kwargs'] == {'start': CONFIGURED_START,
                                               'slope': 0.0}
