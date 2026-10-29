"""Reactor parameter-estimation callbacks apply control modifiers (#271).

``_BaseReactor.paramest_wrapper(..., modify_controls=...)`` evaluates each
call with a copy of the reactor's configured controls updated by the
modifier: ``fun`` and ``args`` replace the configured entries, ``kwargs`` is
merged key by key, unknown control names or record entries raise
``ValueError``, and the configured controls and the caller's modifier are
restored or left unchanged afterwards, also when the evaluation raises.
The tank reactors' ``solve_unit`` keeps its unsupported ``params_control``
slot so that positional arguments bind as before, and rejects any value
other than None with ``ValueError``.

Fixture: the synthetic irreversible A + B -> C case of
``test_reactor_correctness.py`` (real thermodynamics, kinetics, phases and
CVode), with a nonzero activation energy so that a prescribed temperature
profile changes the concentrations. Expected profiles come from an
independent SciPy LSODA integration of the same elementary rate law,
k(T) = k_ref exp(-Ea/R (1/T - 1/T_ref)) c_A c_B, plus the CSTR or semibatch
inflow terms. Validation errors are raised before any solve, and the
restore-on-error check fails at the ``solve_unit`` call itself; both run in
the core lane. Trajectory checks require the optional Assimulo stack and
take a few dozen short reactor solves.
"""

import copy
from functools import partial
import inspect

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from PharmaPy import Reactors
from PharmaPy.Kinetics import RxnKinetics
from PharmaPy.ParamEstim import Experiment, Measurement
from PharmaPy.SimExec import SimulationExec
from test_reactor_correctness import (
    CONCENTRATIONS, GAS_CONSTANT, HEAT_REFERENCE_TEMPERATURE,
    RATE_CONSTANT, REACTION_HEAT, REACTOR_VOLUME, RESIDENCE_TIME,
    TEMPERATURE, THERMO_PATH, _configured_reactor)

# Synthetic activation energy of the issue #271 case: k(350 K)/k(320 K)
# = exp(Ea/R (1/320 - 1/350)) = 5.0, so the 30 K ramp changes the final
# concentrations by about 0.024 mol/L, far above the tolerances below.
ACTIVATION_ENERGY = 5.0e4  # [J/mol]
RAMP_SLOPE = 0.05  # [K/s], issue #271 ramp: 320 K to 350 K over 600 s
END_TIME = 600.0  # [s], issue #271 horizon
SAMPLE_COUNT = 13  # [-], 50 s sampling of the issue #271 grid
TIMES = np.linspace(0.0, END_TIME, SAMPLE_COUNT)  # [s]
# 320 K + 0.05 K/s * 600 s, written out instead of recomputed.
RAMP_END_TEMPERATURE = 350.0  # [K]
# Configured start temperature of the multi-argument controls; it differs
# from the control function's default (TEMPERATURE) so that a modifier that
# discarded configured kwargs would change the profile.
CONFIGURED_START = 330.0  # [K]
REPLACED_START = 300.0  # [K], positional start supplied by an args modifier
HOLD_TIME = 300.0  # [s], ramp-then-hold profile stops rising here
# Both integrators target local errors of rtol 1e-10 and atol 1e-12 mol/L.
# A 1000-fold budget for global error accumulation over the few hundred
# steps of each integration bounds their difference.
SOLVER_OPTIONS = {'rtol': 1e-10, 'atol': 1e-12}  # [-], [mol/L]
PROFILE_RTOL = 1e-7  # [-]
PROFILE_ATOL = 1e-9  # [mol/L]
# Issue #271 reports a 0.024 mol/L change of the terminal concentrations
# between the 320 K and ramped batch runs; 0.01 mol/L confirms the two
# experiments differ by far more than the profile tolerances.
MIN_PROFILE_SEPARATION = 1e-2  # [mol/L]
NUM_REPORTED = 3  # [-], A, B and C columns returned by the wrapper
MEASURED_COLUMNS = [0, 1, 2]  # A, B, C [mol/L]
# Semibatch: the tank holds the 0.002 m**3 charge plus 600 s of feed at
# 0.002/1800 m**3/s (0.00267 m**3); twice the charge avoids an overflow.
SEMIBATCH_TANK_VOLUME = 2 * REACTOR_VOLUME  # [m**3]
REACTOR_CLASSES = [Reactors.BatchReactor, Reactors.CSTR,
                   Reactors.SemibatchReactor]


def ramp_temperature(time, start=TEMPERATURE, slope=0.0):
    """Return a linear temperature ramp.

    Parameters
    ----------
    time : float
        Time [s].
    start : float, optional
        Temperature at zero time [K].
    slope : float, optional
        Heating rate [K/s].

    Returns
    -------
    float
        Temperature [K].
    """
    return start + slope * time  # [K]


def ramp_then_hold(time, start=TEMPERATURE, slope=0.0):
    """Return a linear ramp that holds its value after ``HOLD_TIME``.

    Parameters
    ----------
    time : float
        Time [s].
    start : float, optional
        Temperature at zero time [K].
    slope : float, optional
        Heating rate before ``HOLD_TIME`` [K/s].

    Returns
    -------
    float
        Temperature [K].
    """
    return start + slope * min(time, HOLD_TIME)  # [K]


def _reactor(reactor_cls, control, return_sens=False):
    """Build a configured reactor with a prescribed temperature.

    Parameters
    ----------
    reactor_cls : type
        ``BatchReactor``, ``CSTR`` or ``SemibatchReactor``.
    control : dict or None
        Configured ``temp`` control record (temperature [K] from time
        [s]), or None for no controls.
    return_sens : bool, optional
        Whether ``paramest_wrapper`` also returns CVodes sensitivities
        (supported by ``BatchReactor`` only). The default is False.

    Returns
    -------
    _BaseReactor
        Isothermal reactor with the fixture charge [mol/L] and volume
        [m**3], inflow for CSTR and semibatch reactors, and A + B -> C
        kinetics with ``ACTIVATION_ENERGY``.
    """
    kwargs = {}
    if reactor_cls is Reactors.SemibatchReactor:
        kwargs['vol_tank'] = SEMIBATCH_TANK_VOLUME  # [m**3]
    controls = None if control is None else {'temp': control}
    reactor = _configured_reactor(reactor_cls(
        isothermal=True, return_sens=return_sens, controls=controls,
        **kwargs))
    reactor.Kinetics = RxnKinetics(
        THERMO_PATH, k_params=[RATE_CONSTANT],
        ea_params=[ACTIVATION_ENERGY], rxn_list=['A + B --> C'],
        delta_hrxn=REACTION_HEAT, temp_ref=HEAT_REFERENCE_TEMPERATURE,
        tref_hrxn=HEAT_REFERENCE_TEMPERATURE)
    return reactor


def _independent_profile(reactor_cls, temperature, times=TIMES):
    """Integrate the fixture balances with SciPy for a temperature profile.

    Parameters
    ----------
    reactor_cls : type
        Reactor class whose balances are integrated: batch (no flow),
        CSTR (constant volume, outflow equal to inflow) or semibatch
        (accumulating volume, no outflow).
    temperature : callable
        Temperature [K] as a function of time [s].
    times : numpy.ndarray, optional
        Sample times [s], increasing from 0 s.

    Returns
    -------
    numpy.ndarray
        Concentrations of A, B and C [mol/L], shape
        ``(len(times), NUM_REPORTED)``.
    """
    inflow = REACTOR_VOLUME / RESIDENCE_TIME  # [m**3/s]
    stoichiometry = np.array([-1.0, -1.0, 1.0, 0.0])  # [-], A, B, C, solvent

    def rhs(time, states):
        """Return concentration [mol/L/s] and volume [m**3/s] rates.

        Parameters
        ----------
        time : float
            Time [s].
        states : numpy.ndarray
            Concentrations of A, B, C and solvent [mol/L], then the liquid
            volume [m**3], shape ``(5,)``.

        Returns
        -------
        numpy.ndarray
            Time derivatives of ``states``, shape ``(5,)``.
        """
        conc = states[:4]  # [mol/L], A, B, C, solvent
        # Arrhenius law about the kinetics reference temperature.
        rate_constant = RATE_CONSTANT * np.exp(  # [L/mol/s]
            -ACTIVATION_ENERGY / GAS_CONSTANT
            * (1 / temperature(time) - 1 / HEAT_REFERENCE_TEMPERATURE))
        rate = rate_constant * conc[0] * conc[1]  # [mol/L/s]
        dconc = stoichiometry * rate  # [mol/L/s]
        if reactor_cls is Reactors.BatchReactor:
            return np.append(dconc, 0.0)
        volume = (REACTOR_VOLUME if reactor_cls is Reactors.CSTR
                  else states[4])  # [m**3]
        dconc = dconc + inflow / volume * (CONCENTRATIONS - conc)  # [mol/L/s]
        dvolume = 0.0 if reactor_cls is Reactors.CSTR else inflow  # [m**3/s]
        return np.append(dconc, dvolume)

    initial = np.append(CONCENTRATIONS, REACTOR_VOLUME)  # [mol/L], [m**3]
    solution = solve_ivp(rhs, (times[0], times[-1]), initial,
                         method='LSODA', t_eval=times,
                         rtol=SOLVER_OPTIONS['rtol'],
                         atol=SOLVER_OPTIONS['atol'])
    assert solution.success
    return solution.y[:NUM_REPORTED].T


def _evaluate(reactor, modify_controls, params=None):
    """Evaluate the reactor callback.

    Parameters
    ----------
    reactor : _BaseReactor
        Configured reactor.
    modify_controls : dict or None
        Control modifier passed to ``paramest_wrapper``.
    params : numpy.ndarray or None, optional
        Rate constant [L/mol/s] and activation energy [J/mol]; None uses
        the kinetics' current parameters.

    Returns
    -------
    numpy.ndarray or tuple
        Concentrations of A, B and C [mol/L] at ``TIMES``, shape
        ``(SAMPLE_COUNT, NUM_REPORTED)``; with ``return_sens`` also their
        state-major sensitivities, shape ``(NUM_REPORTED * SAMPLE_COUNT,
        2)``, in [mol**2*s/L**2] per rate constant and [mol**2/(L*J)] per
        activation energy.
    """
    if params is None:
        params = reactor.Kinetics.concat_params()
    return reactor.paramest_wrapper(
        params, TIMES,
        modify_controls=modify_controls,
        run_args={'sundials_opts': dict(SOLVER_OPTIONS)})


def _isothermal(time):
    """Return the configured isothermal temperature profile.

    Parameters
    ----------
    time : float or numpy.ndarray
        Time [s].

    Returns
    -------
    float
        ``TEMPERATURE`` [K] at every time.
    """
    return TEMPERATURE


def _issue_ramp(time):
    """Return the issue #271 temperature ramp.

    Parameters
    ----------
    time : float or numpy.ndarray
        Time [s].

    Returns
    -------
    float or numpy.ndarray
        ``TEMPERATURE + RAMP_SLOPE * time`` [K], shaped like ``time``.
    """
    return TEMPERATURE + RAMP_SLOPE * time


@pytest.mark.assimulo
@pytest.mark.integration
@pytest.mark.parametrize('reactor_cls', REACTOR_CLASSES)
def test_wrapper_applies_ramp_modifier_and_restores_controls(reactor_cls):
    pytest.importorskip('assimulo')
    record = {'fun': ramp_temperature, 'args': (), 'kwargs': {'slope': 0.0}}
    reactor = _reactor(reactor_cls, record)
    configured = reactor.controls
    configured_before = copy.deepcopy(configured)
    modifier = {'temp': {'kwargs': {'slope': RAMP_SLOPE}}}
    modifier_before = copy.deepcopy(modifier)

    ramped = _evaluate(reactor, modifier)  # [mol/L]
    assert reactor.result.temp[-1] == pytest.approx(RAMP_END_TEMPERATURE,
                                                    rel=PROFILE_RTOL)
    np.testing.assert_allclose(reactor.result.temp, _issue_ramp(TIMES),
                               rtol=PROFILE_RTOL, atol=0.0)  # [K]
    np.testing.assert_allclose(
        ramped, _independent_profile(reactor_cls, _issue_ramp),
        rtol=PROFILE_RTOL, atol=PROFILE_ATOL)
    assert reactor.controls is configured
    assert reactor.controls == configured_before
    assert reactor.controls['temp']['kwargs'] == {'slope': 0.0}
    assert modifier == modifier_before

    # The next evaluation without a modifier uses the configured 320 K.
    isothermal = _evaluate(reactor, None)  # [mol/L]
    np.testing.assert_array_equal(reactor.result.temp,
                                  np.full(SAMPLE_COUNT, TEMPERATURE))
    np.testing.assert_allclose(
        isothermal, _independent_profile(reactor_cls, _isothermal),
        rtol=PROFILE_RTOL, atol=PROFILE_ATOL)
    # Guard against comparing two identical experiments.
    assert np.max(np.abs(ramped - isothermal)) > MIN_PROFILE_SEPARATION


# Central differences of the wrapper output in the rate constant k: a
# relative step of 1e-3 bounds the O(h**2) truncation by about 1e-6 of
# |dc/dk| (here |dc/dk| <= 29 mol**2*s/L**2); 1e-5 relative keeps a factor
# of ten. Each solve agrees with SciPy to about 1e-10 mol/L (rtol 1e-10),
# so the difference quotient carries noise of about 1e-10 mol/L / (2 h) =
# 5e-5 mol**2*s/L**2 for h = 1e-6 L/mol/s; 1e-3 mol**2*s/L**2 keeps a
# factor of twenty.
FD_RELATIVE_STEP = 1e-3  # [-]
FD_SENS_RTOL = 1e-5  # [-]
FD_SENS_ATOL = 1e-3  # [mol**2*s/L**2]
# The ramp raises the largest |dc/dk| from about 21 to 29 mol**2*s/L**2;
# requiring 1 mol**2*s/L**2 guards against comparing identical experiments.
MIN_SENS_SEPARATION = 1.0  # [mol**2*s/L**2]


@pytest.mark.assimulo
@pytest.mark.integration
def test_wrapper_applies_ramp_modifier_with_sensitivities():
    pytest.importorskip('assimulo')
    record = {'fun': ramp_temperature, 'args': (), 'kwargs': {'slope': 0.0}}
    reactor = _reactor(Reactors.BatchReactor, record, return_sens=True)
    configured = reactor.controls
    configured_before = copy.deepcopy(configured)
    modifier = {'temp': {'kwargs': {'slope': RAMP_SLOPE}}}
    modifier_before = copy.deepcopy(modifier)
    params = reactor.Kinetics.concat_params().copy()  # [L/mol/s], [J/mol]

    ramped, sens = _evaluate(reactor, modifier, params)  # [mol/L], see _evaluate
    assert reactor.result.temp[-1] == pytest.approx(RAMP_END_TEMPERATURE,
                                                    rel=PROFILE_RTOL)
    np.testing.assert_allclose(reactor.result.temp, _issue_ramp(TIMES),
                               rtol=PROFILE_RTOL, atol=0.0)  # [K]
    np.testing.assert_allclose(
        ramped, _independent_profile(Reactors.BatchReactor, _issue_ramp),
        rtol=PROFILE_RTOL, atol=PROFILE_ATOL)
    assert reactor.controls is configured
    assert reactor.controls == configured_before
    assert modifier == modifier_before

    step = FD_RELATIVE_STEP * params[0]  # [L/mol/s]
    plus, minus = params.copy(), params.copy()
    plus[0] += step
    minus[0] -= step
    finite_difference = ((_evaluate(reactor, modifier, plus)[0]
                          - _evaluate(reactor, modifier, minus)[0])
                         / (2 * step))  # [mol**2*s/L**2]
    assert sens.shape == (NUM_REPORTED * SAMPLE_COUNT, len(params))
    np.testing.assert_allclose(sens[:, 0], finite_difference.T.ravel(),
                               rtol=FD_SENS_RTOL, atol=FD_SENS_ATOL)
    assert reactor.controls is configured
    assert reactor.controls == configured_before

    # Without the modifier the configured 320 K returns, with its own
    # (smaller) rate sensitivity.
    isothermal, isothermal_sens = _evaluate(reactor, None, params)
    np.testing.assert_array_equal(reactor.result.temp,
                                  np.full(SAMPLE_COUNT, TEMPERATURE))
    np.testing.assert_allclose(
        isothermal, _independent_profile(Reactors.BatchReactor, _isothermal),
        rtol=PROFILE_RTOL, atol=PROFILE_ATOL)
    assert (np.max(np.abs(sens[:, 0] - isothermal_sens[:, 0]))
            > MIN_SENS_SEPARATION)


@pytest.mark.assimulo
@pytest.mark.integration
def test_rejected_modifier_leaves_solved_reactor_untouched():
    pytest.importorskip('assimulo')
    record = {'fun': ramp_temperature, 'args': (), 'kwargs': {'slope': 0.0}}
    reactor = _reactor(Reactors.BatchReactor, record)
    reactor.solve_unit(time_grid=TIMES, verbose=False,
                       sundials_opts=dict(SOLVER_OPTIONS))
    runs = len(reactor.profiles_runs)
    elapsed = reactor.elapsed_time  # [s]
    liquid = (reactor.Liquid_1.mole_conc.copy(), reactor.Liquid_1.temp,
              reactor.Liquid_1.vol)  # [mol/L], [K], [m**3]
    result = reactor.result
    # The solve advanced the reactor away from its reset state.
    assert runs == 1
    assert elapsed == END_TIME
    assert not np.allclose(liquid[0], CONCENTRATIONS)

    with pytest.raises(ValueError, match=r"^modify_controls names unknown"):
        _evaluate(reactor, {'vol': {'kwargs': {'slope': RAMP_SLOPE}}})
    assert len(reactor.profiles_runs) == runs
    assert reactor.elapsed_time == elapsed
    np.testing.assert_array_equal(reactor.Liquid_1.mole_conc, liquid[0])
    assert reactor.Liquid_1.temp == liquid[1]
    assert reactor.Liquid_1.vol == liquid[2]
    assert reactor.result is result


# (configured control record, modifier, the control call the merged record
# must make, giving temperature [K] of time [s], and that call's terminal
# temperature [K] at 600 s, written out by hand).
RECORD_MERGES = {
    'kwargs merged key by key': (
        {'fun': ramp_temperature,
         'kwargs': {'start': CONFIGURED_START, 'slope': 0.0}},
        {'kwargs': {'slope': RAMP_SLOPE}},
        partial(ramp_temperature, start=CONFIGURED_START, slope=RAMP_SLOPE),
        360.0),
    'args replaced, kwargs kept': (
        {'fun': ramp_temperature, 'args': (CONFIGURED_START,),
         'kwargs': {'slope': RAMP_SLOPE}},
        {'args': (REPLACED_START,)},
        partial(ramp_temperature, start=REPLACED_START, slope=RAMP_SLOPE),
        330.0),
    'fun replaced, kwargs kept': (
        {'fun': ramp_temperature,
         'kwargs': {'start': CONFIGURED_START, 'slope': RAMP_SLOPE}},
        {'fun': ramp_then_hold},
        partial(ramp_then_hold, start=CONFIGURED_START, slope=RAMP_SLOPE),
        345.0),
}


@pytest.mark.assimulo
@pytest.mark.integration
@pytest.mark.parametrize('case', list(RECORD_MERGES))
def test_modifier_entries_merge_into_configured_record(case):
    pytest.importorskip('assimulo')
    record, modifier_entry, temperature, end_temperature = RECORD_MERGES[case]
    reactor = _reactor(Reactors.BatchReactor, record)
    configured_before = copy.deepcopy(reactor.controls)
    modifier = {'temp': modifier_entry}
    modifier_before = copy.deepcopy(modifier)

    profile = _evaluate(reactor, modifier)  # [mol/L]
    assert reactor.result.temp[-1] == pytest.approx(end_temperature,
                                                    rel=PROFILE_RTOL)
    np.testing.assert_allclose(
        reactor.result.temp, [temperature(time) for time in TIMES],
        rtol=PROFILE_RTOL, atol=0.0)  # [K]
    np.testing.assert_allclose(
        profile, _independent_profile(Reactors.BatchReactor, temperature),
        rtol=PROFILE_RTOL, atol=PROFILE_ATOL)
    assert reactor.controls == configured_before
    assert modifier == modifier_before


@pytest.mark.unit
def test_configured_controls_are_restored_when_evaluation_raises():
    record = {'fun': ramp_temperature, 'args': (), 'kwargs': {'slope': 0.0}}
    reactor = _reactor(Reactors.BatchReactor, record)
    configured = reactor.controls
    configured_before = copy.deepcopy(configured)
    # solve_unit rejects the unknown keyword after the experiment's controls
    # were installed, without needing the solver stack.
    with pytest.raises(TypeError, match="unexpected keyword argument "
                                        "'unsupported_option'"):
        reactor.paramest_wrapper(
            reactor.Kinetics.concat_params(), TIMES,
            modify_controls={'temp': {'kwargs': {'slope': RAMP_SLOPE}}},
            run_args={'unsupported_option': True})
    assert reactor.controls is configured
    assert reactor.controls == configured_before


def _estimation(order, form):
    """Set up a two-experiment estimation through ``SimulationExec``.

    Parameters
    ----------
    order : tuple of str
        Experiment order, a permutation of ``('iso', 'ramp')``.
    form : str
        ``'legacy'`` for named time arrays with ``control_modifiers`` (the
        isothermal experiment's entry is None), ``'experiment'`` for
        ``Experiment`` objects of which only the ramp carries
        ``modify_controls`` in its own kwargs.

    Returns
    -------
    tuple
        The estimator, the fitted reactor, and the expected concentrations
        of A, B and C [mol/L] at ``TIMES`` per experiment name.
    """
    expected = {
        'iso': _independent_profile(Reactors.BatchReactor, _isothermal),
        'ramp': _independent_profile(Reactors.BatchReactor, _issue_ramp)}
    modifiers = {'iso': None,
                 'ramp': {'temp': {'kwargs': {'slope': RAMP_SLOPE}}}}
    simulation = SimulationExec(THERMO_PATH, {'R01': []})
    simulation.R01 = _reactor(
        Reactors.BatchReactor,
        {'fun': ramp_temperature, 'args': (), 'kwargs': {'slope': 0.0}})
    wrapper_kwargs = {'sundials_opts': dict(SOLVER_OPTIONS)}
    if form == 'legacy':
        simulation.SetParamEstimation(
            {name: TIMES for name in order},
            {name: expected[name] for name in order},
            measured_ind=MEASURED_COLUMNS,
            control_modifiers={name: modifiers[name] for name in order},
            wrapper_kwargs=wrapper_kwargs)
    else:
        experiments = {}
        for name in order:
            kwargs = ({} if modifiers[name] is None
                      else {'modify_controls': modifiers[name]})
            experiments[name] = Experiment(
                {species: Measurement(column, TIMES,
                                      expected[name][:, column])
                 for column, species in enumerate(('A', 'B', 'C'))},
                kwargs=kwargs)
        simulation.SetParamEstimation(experiments,
                                      wrapper_kwargs=wrapper_kwargs)
    return simulation.ParamInst, simulation.R01, expected


@pytest.mark.assimulo
@pytest.mark.integration
@pytest.mark.parametrize('form', ['legacy', 'experiment'])
@pytest.mark.parametrize('order', [('iso', 'ramp'), ('ramp', 'iso')])
def test_simulation_exec_experiments_use_their_own_controls(order, form):
    pytest.importorskip('assimulo')
    estimator, reactor, expected = _estimation(order, form)
    seed = estimator.param_seed[estimator.map_variable]
    configured = reactor.controls
    # Two successive objective evaluations: neither the previous experiment
    # nor the previous evaluation may leave its control behind.
    for _ in range(2):
        residual = estimator.get_objective(seed, out_array=True)
        np.testing.assert_allclose(residual, 0.0, rtol=0.0,
                                   atol=PROFILE_ATOL + PROFILE_RTOL
                                   * np.max(CONCENTRATIONS[:NUM_REPORTED]))
        for index, name in enumerate(order):
            np.testing.assert_allclose(
                estimator.y_runs[index], expected[name],
                rtol=PROFILE_RTOL, atol=PROFILE_ATOL)
    assert reactor.controls is configured
    assert reactor.controls['temp']['kwargs'] == {'slope': 0.0}


# Rejected modifiers, the exception and the start of its message.
INVALID_MODIFIERS = {
    'unknown control': (
        {'vol': {'kwargs': {'slope': RAMP_SLOPE}}}, ValueError,
        r"^modify_controls names unknown controls \['vol'\]; the reactor's "
        r"configured controls are \['temp'\]"),
    'unknown entry': (
        {'temp': {'kwarg': {'slope': RAMP_SLOPE}}}, ValueError,
        r"^modify_controls\['temp'\] has unknown entries \['kwarg'\]; "
        r"allowed entries are \['fun', 'args', 'kwargs'\]"),
    'entry not a dictionary': (
        {'temp': RAMP_SLOPE}, TypeError,
        r"^modify_controls\['temp'\] must be a dictionary"),
    'kwargs not a dictionary': (
        {'temp': {'kwargs': RAMP_SLOPE}}, TypeError,
        r"^modify_controls\['temp'\]\['kwargs'\] must be a dictionary"),
    'modifier not a dictionary': (
        [('temp', {'kwargs': {'slope': RAMP_SLOPE}})], TypeError,
        r"^modify_controls must be a dictionary"),
    'fun not callable': (
        {'temp': {'fun': RAMP_SLOPE}}, TypeError,
        r"^Object passed to the 'fun' field must be a callable"),
    'args not a sequence': (
        {'temp': {'args': RAMP_SLOPE}}, TypeError,
        r"^Control 'temp' requires sequence args and dictionary kwargs"),
}


@pytest.mark.unit
@pytest.mark.parametrize('case', list(INVALID_MODIFIERS))
def test_invalid_modifiers_are_rejected_before_solving(case):
    modifier, error, message = INVALID_MODIFIERS[case]
    record = {'fun': ramp_temperature, 'args': (), 'kwargs': {'slope': 0.0}}
    reactor = _reactor(Reactors.BatchReactor, record)
    configured = reactor.controls
    configured_before = copy.deepcopy(configured)
    modifier_before = copy.deepcopy(modifier)
    with pytest.raises(error, match=message):
        _evaluate(reactor, modifier)
    assert reactor.controls is configured
    assert reactor.controls == configured_before
    assert modifier == modifier_before
    assert not hasattr(reactor, 'result')  # nothing was integrated


@pytest.mark.unit
def test_modifier_for_reactor_without_controls_is_rejected():
    reactor = _reactor(Reactors.BatchReactor, None)
    with pytest.raises(ValueError, match=(
            r"^modify_controls names unknown controls \['temp'\]; the "
            r"reactor's configured controls are \[\]")):
        _evaluate(reactor, {'temp': {'kwargs': {'slope': RAMP_SLOPE}}})
    assert reactor.controls == {}


# Leading solve_unit parameters of the tank reactors in their established
# positional order; params_control keeps its slot so later arguments bind
# as in earlier releases.
SOLVE_UNIT_POSITIONAL_ORDER = ['self', 'runtime', 'time_grid', 'eval_sens',
                               'params_control', 'verbose', 'sundials_opts']


@pytest.mark.unit
@pytest.mark.parametrize('reactor_cls', REACTOR_CLASSES)
@pytest.mark.parametrize('positional', [False, True])
def test_solve_unit_rejects_params_control(reactor_cls, positional):
    record = {'fun': ramp_temperature, 'args': (), 'kwargs': {'slope': 0.0}}
    reactor = _reactor(reactor_cls, record)
    legacy_value = {'temp': {'kwargs': {'slope': RAMP_SLOPE}}}
    message = (r"^params_control is not supported by solve_unit and never "
               r"had an effect; configure the reactor's controls")
    with pytest.raises(ValueError, match=message):
        if positional:
            reactor.solve_unit(None, TIMES, False, legacy_value)
        else:
            reactor.solve_unit(time_grid=TIMES, params_control=legacy_value)
    assert not hasattr(reactor, 'result')  # rejected before integrating
    assert reactor.controls['temp']['kwargs'] == {'slope': 0.0}


@pytest.mark.unit
@pytest.mark.parametrize('reactor_cls', REACTOR_CLASSES)
def test_solve_unit_positional_arguments_bind_as_before(reactor_cls):
    signature = inspect.signature(reactor_cls.solve_unit)
    names = list(signature.parameters)
    assert names[:len(SOLVE_UNIT_POSITIONAL_ORDER)] == SOLVE_UNIT_POSITIONAL_ORDER
    options = dict(SOLVER_OPTIONS)
    bound = signature.bind(None, END_TIME, TIMES, False, None, False, options)
    assert bound.arguments['runtime'] == END_TIME
    assert bound.arguments['time_grid'] is TIMES
    assert bound.arguments['params_control'] is None
    assert bound.arguments['verbose'] is False
    assert bound.arguments['sundials_opts'] is options


@pytest.mark.assimulo
@pytest.mark.integration
@pytest.mark.parametrize('reactor_cls', REACTOR_CLASSES)
def test_positional_solve_unit_call_matches_keyword_call(reactor_cls):
    pytest.importorskip('assimulo')
    record = {'fun': ramp_temperature, 'args': (),
              'kwargs': {'slope': RAMP_SLOPE}}
    positional = _reactor(reactor_cls, record)
    time, states = positional.solve_unit(None, TIMES, False, None, False,
                                         dict(SOLVER_OPTIONS))
    keyword = _reactor(reactor_cls, record)
    expected_time, expected_states = keyword.solve_unit(
        time_grid=TIMES, verbose=False, sundials_opts=dict(SOLVER_OPTIONS))
    np.testing.assert_array_equal(time, expected_time)
    np.testing.assert_array_equal(states, expected_states)
    np.testing.assert_allclose(
        states[:, :NUM_REPORTED],
        _independent_profile(reactor_cls, _issue_ramp),
        rtol=PROFILE_RTOL, atol=PROFILE_ATOL)
