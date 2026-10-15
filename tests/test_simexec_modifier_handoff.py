"""Experiment-aligned callback keywords built by ``SetParamEstimation`` (#346).

Scope: how ``SimulationExec.SetParamEstimation`` turns ``phase_modifiers``,
``control_modifiers`` and ``wrapper_kwargs`` into one ``paramest_wrapper``
keyword dictionary per experiment, for named, positional and single legacy
experiments and for ``Experiment`` objects; validation of modifier keys,
lengths and entry types; state names handed to the estimator; and
caller-data isolation. Only the public handoff
to ``simulation.ParamInst`` is checked; nothing is integrated, so these
tests need no ODE solver and run in the core lane. Physical application of
control modifiers inside reactor callbacks belongs to issue #271; the
solver-backed trajectories are in ``test_simexec_experiment_alignment.py``.

Fixture: a real ``SimulationExec`` around an isothermal batch reactor for a
first-order A -> B reaction with zero activation energy. Observations are
its analytic concentrations; modifier payloads are physically meaningful
initial charges, temperatures or sentinel control records whose values are
only compared, never applied here.
"""

import copy
from pathlib import Path
import re

import numpy as np
import pytest

from PharmaPy.Kinetics import RxnKinetics
from PharmaPy.ParamEstim import Experiment, Measurement
from PharmaPy.Phases import LiquidPhase
from PharmaPy.Reactors import BatchReactor
from PharmaPy.SimExec import SimulationExec
from PharmaPy.Utilities import CoolingWater

pytestmark = pytest.mark.unit

THERMO_PATH = str(
    Path(__file__).parent / 'integration/data/pfr_test_pure_comp.json')
TEMPERATURE = 300.0  # [K], isothermal; zero activation energy below
VOLUME = 0.002  # [m**3], concentrations are independent of volume
RATE = 0.2  # [1/s], synthetic five-second reaction time constant
UTILITY_FLOW = 0.1  # [kg/s], unused by the isothermal balance
# [mol/L]: A, B, an absent species and the solvent of the base charge.
BASE_CONCENTRATIONS = [0.1, 0.2, 0.0, 5.0]
# Unequal sampling schedules [s] of up to three experiments.
TIMES_S = {'dilute': np.array([0.0, 1.0, 3.0]),
           'concentrated': np.array([0.0, 2.0, 4.0, 6.0]),
           'late': np.array([0.0, 1.0, 2.0])}
INITIAL_A_MOL_L = {'dilute': 0.05, 'concentrated': 0.3,
                   'late': 0.15}  # [mol/L], asymmetric charges of A
SOLVER_OPTIONS = {'rtol': 1e-9, 'atol': 1e-11}  # [-], [mol/L]
# Expected run_args, written out independently of SOLVER_OPTIONS so that a
# handoff of the wrong object or values is detected.
EXPECTED_RUN_ARGS = {'sundials_opts': {'rtol': 1e-9,  # [-]
                                       'atol': 1e-11}}  # [mol/L]
# Sentinel solver options stored in an Experiment's own kwargs; chosen to
# differ from SOLVER_OPTIONS so the experiment's run_args are recognizable.
OWN_SOLVER_OPTIONS = {'rtol': 1e-8, 'atol': 1e-10}  # [-], [mol/L]
# paramest_wrapper(params, t_vals, modify_phase, modify_controls, reord_sens,
# run_args) accepts four positional callback arguments after the parameters
# and time grid; one more cannot be bound.
TOO_MANY_WRAPPER_ARGS = 5  # [-]
# Control sentinels: argument tuples of a temperature ramp control
# temp(t) = temp_init + ramp * t, distinct per experiment so misrouting is
# visible. They are only compared here; issue #271 owns applying them.
CONTROL_INITIAL_TEMP_K = {'dilute': 300.0, 'concentrated': 310.0,
                          'late': 320.0}  # [K]
CONTROL_RAMP_K_S = -0.01  # [K/s], slow cooling
JACKET_TEMP_K = 295.0  # [K], constant heat-transfer fluid temperature
MODIFIER_ARGUMENTS = ('phase_modifiers', 'control_modifiers')
WRAPPER_KEY = {'phase_modifiers': 'modify_phase',
               'control_modifiers': 'modify_controls'}


def _simulation():
    """Build a simulation around a real isothermal batch reactor.

    Returns
    -------
    SimulationExec
        Flowsheet whose only unit, ``R01``, is the reactor.
    """
    reactor = BatchReactor(isothermal=True, mask_params=[True, False],
                           return_sens=False)
    reactor.Kinetics = RxnKinetics(
        THERMO_PATH, k_params=[RATE], ea_params=[0.0],
        rxn_list=['A --> B'])  # [1/s], [J/mol]
    reactor.Phases = LiquidPhase(THERMO_PATH, temp=TEMPERATURE, vol=VOLUME,
                                 mole_conc=BASE_CONCENTRATIONS)
    reactor.Utility = CoolingWater(temp_in=TEMPERATURE,
                                   mass_flow=UTILITY_FLOW)
    simulation = SimulationExec(THERMO_PATH, {'R01': []})
    simulation.R01 = reactor
    return simulation


def _observations(name):
    """Return analytic A and B concentrations of one experiment.

    Parameters
    ----------
    name : str
        Key of ``TIMES_S`` and ``INITIAL_A_MOL_L``.

    Returns
    -------
    numpy.ndarray
        Shape ``(n_times, 2)``, columns A and B [mol/L].
    """
    reactant = INITIAL_A_MOL_L[name] * np.exp(
        -RATE * TIMES_S[name])  # [mol/L]
    return np.column_stack(
        (reactant,
         BASE_CONCENTRATIONS[1] + INITIAL_A_MOL_L[name] - reactant))


def _phase_modifier(name):
    """Return the initial-charge modifier of one experiment.

    Parameters
    ----------
    name : str
        Key of ``INITIAL_A_MOL_L``.

    Returns
    -------
    dict
        ``{'mole_conc': [...]}`` [mol/L], A, B, absent species, solvent.
    """
    return {'mole_conc': [INITIAL_A_MOL_L[name], BASE_CONCENTRATIONS[1],
                          0.0, BASE_CONCENTRATIONS[3]]}


def _modifier(argument, name):
    """Return a distinguishable modifier entry for one experiment.

    Parameters
    ----------
    argument : str
        ``'phase_modifiers'`` or ``'control_modifiers'``.
    name : str
        Experiment name.

    Returns
    -------
    dict
        Initial charge [mol/L] for phase modifiers; for control modifiers
        the experiment's temperature ramp record (see ``_control_modifier``).
    """
    if argument == 'phase_modifiers':
        return _phase_modifier(name)
    return _control_modifier(name)


def _control_modifier(name):
    """Return the temperature-ramp control record of one experiment.

    Parameters
    ----------
    name : str
        Key of ``CONTROL_INITIAL_TEMP_K``.

    Returns
    -------
    dict
        ``{'temp': {'args': (temp_init, ramp)}}`` with the initial
        temperature [K] and ramp rate [K/s].
    """
    return {'temp': {'args': (CONTROL_INITIAL_TEMP_K[name],
                              CONTROL_RAMP_K_S)}}


def _named(names):
    """Return named legacy times and observations.

    Parameters
    ----------
    names : sequence of str
        Experiment names in experiment order.

    Returns
    -------
    tuple of dict
        Times [s] and observations [mol/L] keyed by name.
    """
    return ({name: TIMES_S[name] for name in names},
            {name: _observations(name) for name in names})


def _positional(names):
    """Return positional legacy times and observations.

    Parameters
    ----------
    names : sequence of str
        Experiments in experiment order.

    Returns
    -------
    tuple of list
        Times [s] and observations [mol/L] in ``names`` order.
    """
    return ([TIMES_S[name] for name in names],
            [_observations(name) for name in names])


def _experiment(name, kwargs=None, args=()):
    """Return an ``Experiment`` measuring A and B of one run.

    Parameters
    ----------
    name : str
        Key of ``TIMES_S``.
    kwargs : dict, optional
        Callback keywords stored in the experiment.
    args : tuple, optional
        Positional callback arguments, bound after the parameters and time
        grid of ``paramest_wrapper``.

    Returns
    -------
    Experiment
        Measurements ``c_A`` (field 0) and ``c_B`` (field 1) [mol/L].
    """
    data = _observations(name)  # [mol/L]
    return Experiment(
        {'c_A': Measurement(0, TIMES_S[name], data[:, 0], units='mol/L'),
         'c_B': Measurement(1, TIMES_S[name], data[:, 1], units='mol/L')},
        args=args, kwargs=kwargs)


def _set(simulation, x_data, y_data=None, **options):
    """Call ``SetParamEstimation`` with the reactor's estimation options.

    Parameters
    ----------
    simulation : SimulationExec
        Simulation to configure.
    x_data, y_data
        Experiments, as for ``SetParamEstimation``.
    **options
        Further ``SetParamEstimation`` keywords.

    Returns
    -------
    ParameterEstimation
        ``simulation.ParamInst``.
    """
    if y_data is not None:
        options['measured_ind'] = [0, 1]
    simulation.SetParamEstimation(x_data, y_data,
                                  optimize_flags=[True, False], **options)
    return simulation.ParamInst


def _snapshot(value):
    """Return an exact, comparable copy of nested caller data.

    Parameters
    ----------
    value : object
        Arrays, mappings, sequences, ``Experiment`` or ``Measurement``
        objects, and scalars.

    Returns
    -------
    object
        Nested tuples in which arrays are represented bitwise (dtype, shape,
        bytes), so NaN observations compare equal to themselves, and
        container types and key order are recorded.
    """
    if isinstance(value, np.ndarray):
        return ('ndarray', value.dtype.str, value.shape, value.tobytes())
    if isinstance(value, Experiment):
        return ('Experiment', _snapshot(dict(value.measurements)),
                _snapshot(value.args), _snapshot(dict(value.kwargs)))
    if isinstance(value, Measurement):
        return ('Measurement', value.field, _snapshot(value.x),
                _snapshot(value.values), _snapshot(value.uncertainty),
                value.units, value.basis, value.x_units)
    if isinstance(value, dict):
        return ('dict', tuple((key, _snapshot(item))
                              for key, item in value.items()))
    if isinstance(value, (list, tuple)):
        return (type(value).__name__,
                tuple(_snapshot(item) for item in value))
    return ('value', value)


def _wrapper(phase=None, control=None, run_args=None):
    """Return the expected callback keywords of one experiment.

    Parameters
    ----------
    phase, control : dict, optional
        Expected modifiers; None stands for the empty default.
    run_args : dict, optional
        Expected solver options; None stands for the empty default.

    Returns
    -------
    dict
        ``modify_phase``, ``modify_controls`` and ``run_args``.
    """
    return {'modify_phase': {} if phase is None else phase,
            'modify_controls': {} if control is None else control,
            'run_args': {} if run_args is None else run_args}


@pytest.mark.parametrize('argument', MODIFIER_ARGUMENTS)
@pytest.mark.parametrize('keys, missing, unexpected', [
    (('dilute', 'concentrated', 'typo_experiment'), [], ['typo_experiment']),
    (('dilute',), ['concentrated'], []),
    (('concentrated', 'Dilute'), ['dilute'], ['Dilute']),
])
def test_named_modifiers_must_have_exactly_the_experiment_keys(
        argument, keys, missing, unexpected):
    simulation = _simulation()
    times, observations = _named(('dilute', 'concentrated'))
    modifiers = {key: ({'temp': 310.0} if key not in times
                       else _modifier(argument, key))
                 for key in keys}  # [K] sentinel payload for unknown keys
    before = _snapshot(modifiers)
    expected = (f"{argument} experiment keys must match x_data; "
                f"missing={missing!r}, unexpected={unexpected!r}")
    with pytest.raises(ValueError, match=f'^{re.escape(expected)}$'):
        _set(simulation, times, observations, **{argument: modifiers})
    assert not hasattr(simulation, 'ParamInst')
    assert _snapshot(modifiers) == before


@pytest.mark.parametrize('argument', MODIFIER_ARGUMENTS)
def test_named_modifiers_must_be_a_mapping(argument):
    times, observations = _named(('dilute', 'concentrated'))
    modifiers = [_modifier(argument, name) for name in times]
    with pytest.raises(TypeError, match=(
            rf"^{argument} must be a dictionary keyed by the x_data "
            r"experiment names \['dilute', 'concentrated'\]; got list$")):
        _set(_simulation(), times, observations, **{argument: modifiers})


@pytest.mark.parametrize('argument', MODIFIER_ARGUMENTS)
@pytest.mark.parametrize('form, offending', [
    ('named', ['concentrated']), ('positional', [1]), ('single', [0])])
def test_modifier_entries_must_be_dictionaries(argument, form, offending):
    names = ('dilute', 'concentrated')
    if form == 'named':
        times, observations = _named(names)
        # A bare temperature [K] lacks the field name it would update.
        modifiers = {'dilute': _modifier(argument, 'dilute'),
                     'concentrated': 310.0}
    elif form == 'positional':
        times, observations = _positional(names)
        # A list of (key, value) pairs is not a modifier dictionary.
        modifiers = [_modifier(argument, 'dilute'),
                     list(_modifier(argument, 'concentrated').items())]
    else:
        times, observations = TIMES_S['dilute'], _observations('dilute')
        modifiers = [310.0]  # [K], bare temperature without a field name
    expected = (f"Each {argument} entry must be a dictionary of modifier "
                f"fields or None; offending experiments: {offending!r}")
    with pytest.raises(TypeError, match=f'^{re.escape(expected)}$'):
        _set(_simulation(), times, observations, **{argument: modifiers})


@pytest.mark.parametrize('argument', MODIFIER_ARGUMENTS)
@pytest.mark.parametrize('form', ['named', 'positional', 'single'])
def test_none_modifier_entries_reach_the_wrapper_as_none(argument, form):
    names = ('dilute', 'concentrated')
    if form == 'named':
        times, observations = _named(names)
        modifiers = {'concentrated': None,
                     'dilute': _modifier(argument, 'dilute')}
        expected = [_modifier(argument, 'dilute'), None]
    elif form == 'positional':
        times, observations = _positional(names)
        modifiers = (None, _modifier(argument, 'concentrated'))
        expected = [None, _modifier(argument, 'concentrated')]
    else:
        times, observations = TIMES_S['dilute'], _observations('dilute')
        modifiers = [None]
        expected = [None]
    estimator = _set(_simulation(), times, observations,
                     **{argument: modifiers})
    # The unit wrappers treat None as no modification, as in master.
    assert [keywords[WRAPPER_KEY[argument]]
            for keywords in estimator.kwargs_fun] == expected


@pytest.mark.parametrize('argument', MODIFIER_ARGUMENTS)
@pytest.mark.parametrize('count', [2, 3])
def test_dict_modifier_cannot_be_aligned_with_unnamed_experiments(
        argument, count):
    names = ('dilute', 'concentrated', 'late')[:count]
    simulation = _simulation()
    times, observations = _positional(names)
    modifiers = {name: _modifier(argument, name) for name in names}
    with pytest.raises(ValueError, match=(
            rf"^{argument} is a dictionary, but its keys cannot be aligned "
            rf"with the {count} unnamed experiments in x_data; pass "
            rf"{argument} as a list with one dictionary per experiment")):
        _set(simulation, times, observations, **{argument: modifiers})
    assert not hasattr(simulation, 'ParamInst')


@pytest.mark.parametrize('argument', MODIFIER_ARGUMENTS)
@pytest.mark.parametrize('length', [1, 3])
def test_positional_modifiers_need_one_entry_per_experiment(argument,
                                                            length):
    times, observations = _positional(('dilute', 'concentrated'))
    modifiers = [_modifier(argument, 'dilute')] * length
    with pytest.raises(ValueError, match=(
            rf"^{argument} must contain one dictionary per experiment in "
            rf"x_data order; expected 2, got {length}$")):
        _set(_simulation(), times, observations, **{argument: modifiers})


@pytest.mark.parametrize('argument', MODIFIER_ARGUMENTS)
def test_modifier_of_unsupported_type_is_rejected(argument):
    with pytest.raises(TypeError, match=(
            rf"^{argument} must be a dictionary, or a list or tuple with "
            r"one dictionary per experiment in x_data order; got str$")):
        _set(_simulation(), TIMES_S['dilute'], _observations('dilute'),
             **{argument: 'temp'})


def test_wrapper_kwargs_must_be_a_mapping():
    with pytest.raises(TypeError, match=(
            r"^wrapper_kwargs must be a mapping of solve_unit keyword "
            r"arguments; got list$")):
        _set(_simulation(), TIMES_S['dilute'], _observations('dilute'),
             wrapper_kwargs=[('sundials_opts', SOLVER_OPTIONS)])


@pytest.mark.parametrize('names', [('concentrated',),
                                   ('dilute', 'concentrated', 'late')])
def test_reordered_named_modifiers_align_by_experiment_name(names):
    times, observations = _named(names)
    observations = dict(reversed(list(observations.items())))
    phase = {name: _modifier('phase_modifiers', name)
             for name in reversed(names)}
    control = {name: _modifier('control_modifiers', name)
               for name in names[1:] + names[:1]}
    solver = {'sundials_opts': SOLVER_OPTIONS}
    estimator = _set(_simulation(), times, observations,
                     phase_modifiers=phase, control_modifiers=control,
                     wrapper_kwargs=solver)
    expected = [_wrapper(_phase_modifier(name),
                         _control_modifier(name),
                         EXPECTED_RUN_ARGS)
                for name in names]
    assert estimator.experim_names == list(names)
    assert estimator.kwargs_fun == expected
    for name, keywords in zip(names, estimator.kwargs_fun):
        # Caller modifiers are handed over, not copied ...
        assert keywords['modify_phase'] is phase[name]
        assert keywords['modify_controls'] is control[name]
        # ... while each experiment owns a shallow copy of wrapper_kwargs.
        assert keywords['run_args'] is not solver
    assert len({id(keywords['run_args'])
                for keywords in estimator.kwargs_fun}) == len(names)


@pytest.mark.parametrize('form', ['named', 'positional', 'array', 'list'])
def test_none_defaults_pass_empty_modifiers_to_every_experiment(form):
    names = ('dilute', 'concentrated', 'late')
    if form == 'named':
        times, observations = _named(names)
    elif form == 'positional':
        times, observations = _positional(names)
    elif form == 'array':
        names = ('dilute',)
        times, observations = TIMES_S['dilute'], _observations('dilute')
    else:
        names = ('dilute',)
        times, observations = _positional(names)
    estimator = _set(_simulation(), times, observations)
    assert estimator.kwargs_fun == [_wrapper() for _ in names]


@pytest.mark.parametrize('argument', MODIFIER_ARGUMENTS)
@pytest.mark.parametrize('x_form', ['array', 'list'])
@pytest.mark.parametrize('modifier_form', ['dict', 'list', 'tuple'])
def test_single_unnamed_experiment_takes_its_modifier_fields(
        argument, x_form, modifier_form):
    times = (TIMES_S['dilute'] if x_form == 'array'
             else [TIMES_S['dilute']])
    observations = (_observations('dilute') if x_form == 'array'
                    else [_observations('dilute')])
    # Multiple fields in one dictionary are that experiment's fields.
    entry = (dict(_phase_modifier('dilute'), temp=310.0)  # [K]
             if argument == 'phase_modifiers'
             else dict(_modifier(argument, 'dilute'),
                       temp_ht={'args': (JACKET_TEMP_K,)}))
    modifiers = {'dict': entry, 'list': [entry],
                 'tuple': (entry,)}[modifier_form]
    estimator = _set(_simulation(), times, observations,
                     **{argument: modifiers})
    assert len(estimator.kwargs_fun) == 1
    assert estimator.kwargs_fun[0][WRAPPER_KEY[argument]] is entry
    other = WRAPPER_KEY[({'phase_modifiers', 'control_modifiers'}
                         - {argument}).pop()]
    assert estimator.kwargs_fun[0][other] == {}


@pytest.mark.parametrize('count', [2, 3])
def test_positional_experiments_receive_their_own_wrapper_keywords(count):
    names = ('dilute', 'concentrated', 'late')[:count]
    times, observations = _positional(names)
    phase = [_modifier('phase_modifiers', name) for name in names]
    control = tuple(_modifier('control_modifiers', name) for name in names)
    solver = {'sundials_opts': SOLVER_OPTIONS}
    estimator = _set(_simulation(), times, observations,
                     phase_modifiers=phase, control_modifiers=control,
                     wrapper_kwargs=solver)
    # Three experiments once matched the three flat wrapper fields by count.
    assert estimator.kwargs_fun == [
        _wrapper(_phase_modifier(name),
                 _control_modifier(name),
                 EXPECTED_RUN_ARGS)
        for name in names]
    for index, keywords in enumerate(estimator.kwargs_fun):
        assert keywords['modify_phase'] is phase[index]
        assert keywords['modify_controls'] is control[index]
        assert keywords['run_args'] is not solver
        assert keywords['run_args']['sundials_opts'] is solver[
            'sundials_opts']  # shallow copy shares nested options
    assert len({id(keywords['run_args'])
                for keywords in estimator.kwargs_fun}) == count


def test_single_staggered_experiment_in_a_list_is_one_experiment():
    times_b = np.array([1.0, 4.0])  # [s], B sampled on its own grid
    reactant = INITIAL_A_MOL_L['dilute'] * np.exp(
        -RATE * TIMES_S['dilute'])  # [mol/L]
    product = (BASE_CONCENTRATIONS[1] + INITIAL_A_MOL_L['dilute']
               * (1 - np.exp(-RATE * times_b)))  # [mol/L]
    entry = _phase_modifier('dilute')
    estimator = _set(_simulation(), [[TIMES_S['dilute'], times_b]],
                     [[reactant, product]], phase_modifiers=entry)
    np.testing.assert_array_equal(estimator.x_model[0],
                                  [0.0, 1.0, 3.0, 4.0])  # [s], union grid
    assert estimator.kwargs_fun == [_wrapper(entry)]


@pytest.mark.parametrize('form', ['mapping', 'sequence', 'single'])
def test_experiment_objects_receive_aligned_wrapper_keywords(form):
    names = ('dilute', 'concentrated', 'late')
    solver = {'sundials_opts': SOLVER_OPTIONS}
    if form == 'mapping':
        experiments = {name: _experiment(name) for name in names}
        phase = {name: _phase_modifier(name) for name in reversed(names)}
        control = {name: _modifier('control_modifiers', name)
                   for name in names[1:] + names[:1]}
    elif form == 'sequence':
        experiments = [_experiment(name) for name in names]
        phase = [_phase_modifier(name) for name in names]
        control = [_modifier('control_modifiers', name) for name in names]
    else:
        names = ('late',)
        experiments = _experiment('late')
        phase = _phase_modifier('late')
        control = [_modifier('control_modifiers', 'late')]
    before = _snapshot(experiments)
    estimator = _set(_simulation(), experiments, phase_modifiers=phase,
                     control_modifiers=control, wrapper_kwargs=solver)
    assert estimator.kwargs_fun == [
        _wrapper(_phase_modifier(name),
                 _control_modifier(name),
                 EXPECTED_RUN_ARGS)
        for name in names]
    for index, name in enumerate(names):
        np.testing.assert_array_equal(estimator.x_model[index],
                                      TIMES_S[name])  # [s]
        # Observations of A then B [mol/L], placed by experiment.
        np.testing.assert_array_equal(estimator.y_data[index],
                                      _observations(name))
    assert _snapshot(experiments) == before


def test_experiment_keywords_are_kept_when_arguments_are_absent():
    own = {'modify_phase': _phase_modifier('concentrated'),
           'run_args': {'sundials_opts': OWN_SOLVER_OPTIONS},
           'reord_sens': True}
    experiments = {'concentrated': _experiment('concentrated', own),
                   'dilute': _experiment('dilute')}
    control = {'dilute': _modifier('control_modifiers', 'dilute'),
               'concentrated': {}}
    estimator = _set(_simulation(), experiments, control_modifiers=control)
    assert estimator.kwargs_fun == [
        dict(_wrapper(own['modify_phase'], {}, own['run_args']),
             reord_sens=True),
        _wrapper(None, _modifier('control_modifiers', 'dilute'))]
    assert dict(experiments['concentrated'].kwargs) == own
    assert dict(experiments['dilute'].kwargs) == {}


@pytest.mark.parametrize('key, argument, value', [
    ('modify_phase', 'phase_modifiers', {'temp': 310.0}),  # [K]
    ('modify_controls', 'control_modifiers', {}),
    ('run_args', 'wrapper_kwargs', {'verbose': False}),
])
@pytest.mark.parametrize('form', ['mapping', 'sequence'])
def test_experiment_keywords_conflicting_with_arguments_are_rejected(
        key, argument, value, form):
    simulation = _simulation()
    own = {key: value}
    if form == 'mapping':
        experiments = {'dilute': _experiment('dilute'),
                       'concentrated': _experiment('concentrated', own)}
        name = 'concentrated'
        supplied = ({'dilute': {}, 'concentrated': {}}
                    if argument != 'wrapper_kwargs' else {})
    else:
        experiments = [_experiment('dilute'),
                       _experiment('concentrated', own)]
        name = 'exp_2'
        supplied = [{}, {}] if argument != 'wrapper_kwargs' else {}
    with pytest.raises(ValueError, match=(
            rf"^Experiment '{name}' kwargs define '{key}', which "
            rf"SetParamEstimation also supplies through {argument}; set it "
            r"in one place only$")):
        _set(simulation, experiments, **{argument: supplied})
    assert not hasattr(simulation, 'ParamInst')
    assert dict(experiments[-1 if form == 'sequence' else name].kwargs) == own


@pytest.mark.parametrize('argument', MODIFIER_ARGUMENTS)
def test_experiment_mapping_modifiers_are_validated_by_name(argument):
    experiments = {name: _experiment(name)
                   for name in ('dilute', 'concentrated')}
    modifiers = {'dilute': {}, 'concentrated': {},
                 'typo_experiment': {'temp': 310.0}}  # [K] sentinel
    with pytest.raises(ValueError, match=(
            rf"^{argument} experiment keys must match x_data; missing=\[\], "
            r"unexpected=\['typo_experiment'\]$")):
        _set(_simulation(), experiments, **{argument: modifiers})


def test_spectral_fit_rejects_experiment_objects():
    simulation = _simulation()
    with pytest.raises(TypeError, match=(
            r"^fit_spectra=True fits spectra with MultipleCurveResolution, "
            r"which does not accept Experiment objects")):
        simulation.SetParamEstimation(
            {'dilute': _experiment('dilute')}, fit_spectra=True,
            optimize_flags=[True, False])
    assert not hasattr(simulation, 'ParamInst')


@pytest.mark.parametrize('form', ['named', 'positional', 'experiments'])
def test_set_param_estimation_leaves_caller_data_unchanged(form):
    names = ('dilute', 'concentrated', 'late')
    if form == 'named':
        times, observations = _named(names)
        phase = {name: _phase_modifier(name) for name in names}
        control = {name: _modifier('control_modifiers', name)
                   for name in names}
    else:
        times, observations = _positional(names)
        phase = [_phase_modifier(name) for name in names]
        control = [_modifier('control_modifiers', name) for name in names]
    if form == 'experiments':
        times = {name: _experiment(name) for name in names}
        observations = None
        phase = dict(zip(names, phase))
        control = dict(zip(names, control))
    solver = {'sundials_opts': copy.deepcopy(SOLVER_OPTIONS),
              'verbose': False}
    caller = (times, observations, phase, control, solver)
    before = _snapshot(copy.deepcopy(caller))
    _set(_simulation(), times, observations, phase_modifiers=phase,
         control_modifiers=control, wrapper_kwargs=solver)
    assert _snapshot(caller) == before


@pytest.mark.parametrize('supplied', [None, ['A [mol/L]', 'B [mol/L]']])
def test_experiment_input_keeps_the_estimator_state_names(supplied):
    experiments = {'concentrated': _experiment('concentrated'),
                   'dilute': _experiment('dilute')}
    options = {} if supplied is None else {'name_states': supplied}
    estimator = _set(_simulation(), experiments, **options)
    expected = ['c_A', 'c_B'] if supplied is None else supplied
    assert estimator.name_states == expected


@pytest.mark.parametrize('supplied', [None, ['A [mol/L]', 'B [mol/L]']])
def test_legacy_input_names_states_after_the_unit(supplied):
    simulation = _simulation()
    times, observations = _named(('dilute', 'concentrated'))
    options = {} if supplied is None else {'name_states': supplied}
    estimator = _set(simulation, times, observations, **options)
    # Legacy behavior: the unit's state names replace any supplied names.
    assert estimator.name_states == simulation.R01.states_uo
    assert estimator.name_states == ['mole_conc']


@pytest.mark.parametrize('form', ['mapping', 'sequence'])
def test_positional_experiment_args_bind_wrapper_parameters(form):
    names = ('concentrated', 'dilute')
    own_args = {'concentrated': (_phase_modifier('concentrated'),),
                'dilute': (_phase_modifier('dilute'),
                           _control_modifier('dilute'))}
    experiments = {name: _experiment(name, args=own_args[name])
                   for name in names}
    if form == 'sequence':
        experiments = list(experiments.values())
    solver = {'sundials_opts': SOLVER_OPTIONS}
    before = _snapshot(experiments)
    estimator = _set(_simulation(), experiments, wrapper_kwargs=solver)
    # The experiment's own args reach the callback unchanged ...
    for index, name in enumerate(names):
        assert estimator.args_fun[index] == own_args[name]
        assert estimator.args_fun[index][0] is own_args[name][0]
    # ... and wrapper keywords bound by them are not injected again.
    assert estimator.kwargs_fun == [
        {'modify_controls': {}, 'run_args': solver},
        {'run_args': solver}]
    assert _snapshot(experiments) == before


@pytest.mark.parametrize('argument, key, args', [
    ('phase_modifiers', 'modify_phase', (_phase_modifier('dilute'),)),
    ('control_modifiers', 'modify_controls',
     (_phase_modifier('dilute'), _control_modifier('dilute'))),
])
def test_positional_args_conflicting_with_arguments_are_rejected(
        argument, key, args):
    simulation = _simulation()
    experiments = {'concentrated': _experiment('concentrated'),
                   'dilute': _experiment('dilute', args=args)}
    supplied = {'concentrated': {}, 'dilute': {}}
    with pytest.raises(ValueError, match=(
            rf"^Experiment 'dilute' args bind '{key}' positionally, which "
            rf"SetParamEstimation also supplies through {argument}; set it "
            r"in one place only$")):
        _set(simulation, experiments, **{argument: supplied})
    assert not hasattr(simulation, 'ParamInst')


@pytest.mark.parametrize('args, kwargs, reason', [
    ((None,) * TOO_MANY_WRAPPER_ARGS, None, 'too many positional arguments'),
    ((_phase_modifier('dilute'),), {'modify_phase': {}},
     "multiple values for argument 'modify_phase'"),
    ((), {'modify_temperature': {}},
     "got an unexpected keyword argument 'modify_temperature'"),
])
def test_experiment_arguments_must_bind_to_the_wrapper(args, kwargs, reason):
    simulation = _simulation()
    experiments = [_experiment('concentrated'),
                   _experiment('dilute', kwargs=kwargs, args=args)]
    with pytest.raises(TypeError, match=(
            r"^Experiment 'exp_2' args .* cannot be bound to the unit's "
            r"paramest_wrapper\(params, t_vals, .*\) after its parameters "
            rf"and time grid: {re.escape(reason)}$")):
        _set(simulation, experiments)
    assert not hasattr(simulation, 'ParamInst')
