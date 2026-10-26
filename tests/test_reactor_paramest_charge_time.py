"""Reactor estimation experiments sampled only at the charge time (#418).

A reactor experiment whose requested times all equal the charge time, such
as ``[0.0]`` or ``[0.0, 0.0]``, has zero integration duration. The shared
``_BaseReactor.paramest_wrapper`` of ``BatchReactor``, ``CSTR`` and
``SemibatchReactor`` now returns the charged state for every requested time
(replicates repeated) without integrating, with exactly zero sensitivities
in the layout of the integrated path when ``return_sens`` is True, instead
of failing inside CVode with ``IndexError``. The evaluation still applies
and restores the experiment's control modifier, and leaves ``result`` None.

Fixture: the isothermal first-order A -> B batch reactor of
``test_reactor_paramest_requested_rows.py`` (c_A0 = 0.1 mol/L,
c_B0 = 0.2 mol/L, k = 0.15 or 0.2 1/s), plus a feed of A only for the CSTR
and semibatch variants, which cannot change the charged state. At the
charge time the analytic solution c_A = c_A0 exp(-k t),
c_B = c_B0 + c_A0 (1 - exp(-k t)) gives the charge itself and
dc/dk = -t c_A0 exp(-k t) = 0. A few dozen short reactor solves; requires
the optional Assimulo stack.
"""

import copy

import numpy as np
import pytest

pytestmark = [pytest.mark.assimulo, pytest.mark.integration]
pytest.importorskip('assimulo')

from PharmaPy.Kinetics import RxnKinetics
from PharmaPy.ParamEstim import Experiment, Measurement
from PharmaPy.Phases import LiquidPhase
from PharmaPy.Reactors import CSTR, BatchReactor, SemibatchReactor
from PharmaPy.SimExec import SimulationExec
from PharmaPy.Streams import LiquidStream
from PharmaPy.Utilities import CoolingWater
from test_reactor_paramest_requested_rows import (
    BASE_A, FD_NOISE_ATOL, FD_RELATIVE_STEP, PRODUCT_INITIAL, PROFILE_ATOL,
    PROFILE_RTOL, SENSITIVITY_ATOL, SOLVENT, SOLVER_OPTIONS, TEMPERATURE,
    THERMO_PATH, TRIAL_RATE, TRUE_RATE, UTILITY_FLOW, VOLUME, _analytic,
    _reactor)

REACTOR_CLASSES = [BatchReactor, CSTR, SemibatchReactor]
CHARGE_GRIDS = {'single': [0.0], 'replicates': [0.0, 0.0]}  # [s]
# Charged concentrations of A and B [mol/L], written out from the fixture.
CHARGE = np.array([0.1, 0.2])  # [mol/L]
NUM_STATES = 2  # [-], integrated A and B concentrations (isothermal)
NUM_PARAMS = 2  # [-], rate constant [1/s] and activation energy [J/mol]
# Synthetic feed of A only, away from the charge, so that any inflow
# changes the state: at the charge dc_A/dt = -k c_A0 + (FEED_A - c_A0) / tau
# = +0.06 mol/L/s and dc_B/dt = k c_A0 - c_B0 / tau = -0.02 mol/L/s.
FEED_A = 0.5  # [mol/L]
RESIDENCE_TIME = 5.0  # [s], CSTR and semibatch inflow V / Q
REACTED_TIME = 2.0  # [s], horizon of a preceding integrated evaluation
# Over REACTED_TIME every reported concentration moves by at least 8e-3
# mol/L (observed minimum: c_B in the semibatch reactor; c_A0 (1 - e^-0.4)
# = 0.033 mol/L in the batch reactor), so a stale state exceeds 5e-3 mol/L.
MIN_SEPARATION = 5e-3  # [mol/L]
SEMIBATCH_TANK_VOLUME = 2 * VOLUME  # [m**3], room for the feed
MODIFIED_A = 0.05  # [mol/L], charge of A set by a phase modifier
MODIFIED_CHARGE = np.array([0.05, 0.2])  # [mol/L], A and B written out
LEVEL_SHIFT = 10.0  # [K], modified constant control level above TEMPERATURE
# Deliberate observation errors of the initial-only experiment [mol/L],
# unequal so that the residuals identify each replicate.
INITIAL_ERRORS = {'single': [2e-3], 'replicates': [2e-3, -1e-3]}
POSITIVE_TIMES = [1.0, 2.0]  # [s]
LATER_TIME = 1.0  # [s], appended to integrate past the charge time
# Per-entry error bound of the integrated residuals (requested-rows module).
RESIDUAL_ATOL = PROFILE_RTOL * 0.5 + PROFILE_ATOL  # [mol/L]
# Synthetic exotherm so that an integrated isothermal run removes heat and
# leaves a nonzero heat duty; concentrations do not depend on it.
REACTION_HEAT = -1.0e4  # [J/mol]
# Charged states of the CSTR (A, B, C, solvent) [mol/L], written out; the
# semibatch reactor appends its charged volume VOLUME [m**3].
CSTR_CHARGED_STATES = np.array([0.1, 0.2, 0.0, 5.0])  # [mol/L]
# Smallest positive grids: one sample 0.1 ms after the charge, alone and
# after a charge-time sample. c_A0 (1 - exp(-k t)) = 2.0e-6 mol/L at
# k = 0.2 1/s, far above PROFILE_ATOL (1e-9 mol/L) and far below the
# tolerance mutation (1e-3 s) that would treat the sample as the charge.
NEAR_CHARGE_GRIDS = {'positive only': [1e-4],
                     'after charge': [0.0, 1e-4]}  # [s]


def _tank(reactor_cls, return_sens=False, controls=None, isothermal=True,
          reaction_heat=None):
    """Build the first-order fixture as a batch, CSTR or semibatch reactor.

    Parameters
    ----------
    reactor_cls : type
        ``BatchReactor``, ``CSTR`` or ``SemibatchReactor``.
    return_sens : bool, optional
        Whether ``paramest_wrapper`` returns sensitivities.
    controls : dict or None, optional
        Configured controls (temperature [K] from time [s]).
    isothermal : bool, optional
        If False, integrate the reactor and jacket temperatures [K] too.
    reaction_heat : float or None, optional
        Heat of reaction [J/mol] at ``TEMPERATURE``; None keeps the
        kinetics default.

    Returns
    -------
    _BaseReactor
        Reactor with rate constant ``TRUE_RATE`` [1/s], charged with A, B
        and solvent [mol/L] at ``TEMPERATURE`` [K]; the CSTR and semibatch
        reactors are fed A at ``FEED_A`` [mol/L].
    """
    kwargs = {}
    if reactor_cls is SemibatchReactor:
        kwargs['vol_tank'] = SEMIBATCH_TANK_VOLUME  # [m**3]
    reactor = reactor_cls(isothermal=isothermal, mask_params=[True, False],
                          return_sens=return_sens, controls=controls,
                          **kwargs)
    # Heat of reaction [J/mol] at reference temperature [K].
    heat = ({} if reaction_heat is None else
            {'delta_hrxn': reaction_heat, 'tref_hrxn': TEMPERATURE})
    reactor.Kinetics = RxnKinetics(
        THERMO_PATH, k_params=[TRUE_RATE], ea_params=[0.0],
        rxn_list=['A --> B'], **heat)  # [1/s], [J/mol]
    reactor.Phases = LiquidPhase(
        THERMO_PATH, temp=TEMPERATURE, vol=VOLUME,
        mole_conc=[BASE_A, PRODUCT_INITIAL, 0.0, SOLVENT])
    reactor.Utility = CoolingWater(temp_in=TEMPERATURE,
                                   mass_flow=UTILITY_FLOW)
    if reactor_cls is not BatchReactor:
        reactor.Inlet = LiquidStream(
            THERMO_PATH, temp=TEMPERATURE,
            mole_conc=[FEED_A, 0.0, 0.0, SOLVENT],
            vol_flow=VOLUME / RESIDENCE_TIME)  # [m**3/s]
    return reactor


def _evaluate(reactor, times, **kwargs):
    """Call ``paramest_wrapper`` at the reactor's current parameters.

    Parameters
    ----------
    reactor : _BaseReactor
        Configured reactor.
    times : list of float
        Requested times [s].
    **kwargs
        Further ``paramest_wrapper`` keywords.

    Returns
    -------
    numpy.ndarray or tuple
        The wrapper's concentrations [mol/L] and, with ``return_sens``,
        sensitivities [mol/L per parameter unit].
    """
    return reactor.paramest_wrapper(
        reactor.Kinetics.concat_params(), np.array(times),
        run_args={'sundials_opts': SOLVER_OPTIONS}, **kwargs)


@pytest.mark.parametrize('grid', list(CHARGE_GRIDS))
@pytest.mark.parametrize('reactor_cls', REACTOR_CLASSES)
def test_charge_time_grid_returns_charged_rows(reactor_cls, grid):
    times = CHARGE_GRIDS[grid]  # [s]
    reactor = _tank(reactor_cls)
    # A positive-time evaluation first leaves a reacted liquid and a result.
    reacted = _evaluate(reactor, [REACTED_TIME])  # [mol/L]
    assert np.all(np.abs(reacted[0] - CHARGE) > MIN_SEPARATION)
    assert reactor.result is not None

    profile = _evaluate(reactor, times)  # [mol/L]
    np.testing.assert_array_equal(profile, np.tile(CHARGE, (len(times), 1)))
    # Nothing was integrated: no stored profile, clock at the charge time.
    assert reactor.result is None
    assert reactor.elapsed_time == 0.0
    np.testing.assert_array_equal(reactor.Liquid_1.mole_conc[:2], CHARGE)

    # A phase modifier sets the charge that the rows report.
    modifier = {'mole_conc': [MODIFIED_A, PRODUCT_INITIAL, 0.0, SOLVENT]}
    modified = _evaluate(reactor, times, modify_phase=modifier)  # [mol/L]
    np.testing.assert_array_equal(
        modified, np.tile(MODIFIED_CHARGE, (len(times), 1)))


@pytest.mark.parametrize('grid', list(CHARGE_GRIDS))
def test_batch_charge_time_sensitivities_are_zero(grid):
    times = CHARGE_GRIDS[grid]  # [s]
    reactor = _tank(BatchReactor, return_sens=True)
    # An integrated evaluation first stores a result and sensitivities.
    _evaluate(reactor, [REACTED_TIME])
    assert reactor.sensit is not None
    assert reactor.result is not None

    profile, sens = _evaluate(reactor, times)  # [mol/L], [mol/L per param]
    assert reactor.sensit is None
    assert reactor.result is None
    np.testing.assert_array_equal(profile, np.tile(CHARGE, (len(times), 1)))
    # Reordered layout: state-major rows (A, then B, over the requested
    # times), one column per parameter.
    np.testing.assert_array_equal(
        sens, np.zeros((NUM_STATES * len(times), NUM_PARAMS)))
    stacked = _evaluate(reactor, times,
                        reord_sens=False)[1]  # [mol/L per param unit]
    np.testing.assert_array_equal(
        stacked, np.zeros((NUM_PARAMS, len(times), NUM_STATES)))


@pytest.mark.parametrize('reactor_cls', REACTOR_CLASSES)
def test_charge_time_evaluation_replaces_solve_artifacts(reactor_cls):
    """No artifact of an earlier integrated evaluation survives.

    Profiles become None; heat duty, terminal state and outlet describe
    the charge (a CSTR has no outlet stream without a flow history).
    """
    reactor = _tank(reactor_cls, reaction_heat=REACTION_HEAT)
    _evaluate(reactor, [0.0, REACTED_TIME])
    # Sentinels: the integrated evaluation stored every artifact.
    assert reactor.result is not None
    assert reactor.outputs is not None
    assert reactor.heat_duty[0] != 0.0
    assert reactor.Outlet is not None
    if reactor_cls is not BatchReactor:
        assert reactor.statesProf is not None
        assert np.all(np.abs(reactor.states[:2] - CHARGE) > MIN_SEPARATION)

    _evaluate(reactor, [0.0])
    assert reactor.result is None
    assert reactor.sensit is None
    assert reactor.outputs is None
    np.testing.assert_array_equal(reactor.heat_duty, [0.0, 0.0])  # [J]
    assert reactor.duty_type == [0, 0]
    assert reactor.elapsed_time == 0.0  # [s]
    if reactor_cls is CSTR:
        assert reactor.Outlet is None
    else:
        # The charged holdup is the batch and semibatch outlet.
        assert reactor.Outlet is reactor.Liquid_1
        np.testing.assert_array_equal(reactor.Outlet.mole_conc[:2], CHARGE)
    if reactor_cls is not BatchReactor:
        assert reactor.statesProf is None
        # Concentrations [mol/L], then the semibatch volume [m**3].
        expected = (CSTR_CHARGED_STATES if reactor_cls is CSTR
                    else np.append(CSTR_CHARGED_STATES, VOLUME))
        np.testing.assert_array_equal(reactor.states, expected)


@pytest.mark.parametrize('grid', list(NEAR_CHARGE_GRIDS))
def test_positive_time_near_charge_is_integrated(grid):
    """The exact charge-time rule integrates any later sample."""
    times = NEAR_CHARGE_GRIDS[grid]  # [s]
    reactor = _tank(BatchReactor)
    profile = _evaluate(reactor, times)  # [mol/L]
    expected = _analytic(BASE_A, TRUE_RATE, times)[0]  # [mol/L]
    np.testing.assert_allclose(profile, expected, rtol=PROFILE_RTOL,
                               atol=PROFILE_ATOL)
    # The last sample differs from the charge by c_A0 (1 - exp(-k t)).
    reacted_a = BASE_A * (1 - np.exp(-TRUE_RATE * times[-1]))  # [mol/L]
    assert reacted_a > 100 * PROFILE_ATOL
    np.testing.assert_allclose(np.abs(profile[-1] - CHARGE), reacted_a,
                               rtol=0.0, atol=2 * PROFILE_ATOL)
    assert reactor.result is not None


@pytest.mark.parametrize('isothermal', [True, False])
@pytest.mark.parametrize('grid', list(CHARGE_GRIDS))
def test_batch_charge_rows_match_integrated_initial_rows(grid, isothermal):
    """Charge-only rows equal the integrated path's charge-time rows.

    A grid with one later time integrates; its leading charge-time rows
    have the integrated path's layout, including the temperature states
    of a non-isothermal reactor, and must equal the charge-only rows.
    """
    times = CHARGE_GRIDS[grid]  # [s]
    reactor = _tank(BatchReactor, return_sens=True, isothermal=isothermal)
    params = reactor.Kinetics.concat_params()  # [1/s], [J/mol]
    run_args = {'sundials_opts': SOLVER_OPTIONS}
    # Concentrations [mol/L] and stacked sensitivities [state units per
    # parameter unit: mol/L*s and mol**2/(L*J); K*s and K*mol/J].
    integrated = reactor.paramest_wrapper(
        params, np.array(times + [LATER_TIME]), reord_sens=False,
        run_args=run_args)
    charge_only = reactor.paramest_wrapper(
        params, np.array(times), reord_sens=False, run_args=run_args)
    rows = len(times)
    np.testing.assert_array_equal(charge_only[0], integrated[0][:rows])
    assert charge_only[1].shape == (NUM_PARAMS, rows,
                                    integrated[1].shape[2])
    np.testing.assert_array_equal(charge_only[1], integrated[1][:, :rows])
    np.testing.assert_array_equal(charge_only[1], 0.0)


@pytest.mark.parametrize('grid', list(CHARGE_GRIDS))
@pytest.mark.parametrize('reactor_cls', [CSTR, SemibatchReactor])
def test_unsupported_sensitivities_are_rejected_at_charge_time(reactor_cls,
                                                               grid):
    reactor = _tank(reactor_cls, return_sens=True)
    # The same error as the integrated path of these reactors.
    with pytest.raises(NotImplementedError,
                       match='sensitivity evaluation is not supported'):
        _evaluate(reactor, CHARGE_GRIDS[grid])


def constant_temperature(time, level=TEMPERATURE):
    """Return a constant temperature.

    Parameters
    ----------
    time : float
        Time [s].
    level : float, optional
        Temperature [K].

    Returns
    -------
    float
        ``level`` [K].
    """
    return level


@pytest.mark.parametrize('reactor_cls', REACTOR_CLASSES)
def test_charge_time_evaluation_applies_and_restores_control_modifier(
        reactor_cls):
    record = {'fun': constant_temperature, 'args': (),
              'kwargs': {'level': TEMPERATURE}}
    reactor = _tank(reactor_cls, controls={'temp': record})
    configured = reactor.controls
    configured_before = copy.deepcopy(configured)

    # The experiment's modified control is validated at the charge time:
    # a NaN level is rejected, and the configured controls are restored.
    invalid = {'temp': {'kwargs': {'level': np.nan}}}
    with pytest.raises(ValueError, match="Tank control 'temp' must return"):
        _evaluate(reactor, [0.0], modify_controls=invalid)
    assert reactor.controls is configured
    assert reactor.controls == configured_before

    # Unknown controls are rejected before the reactor is reset.
    unknown = r"^modify_controls names unknown controls \['vol'\]"
    with pytest.raises(ValueError, match=unknown):
        _evaluate(reactor, [0.0], modify_controls={'vol': {}})

    modifier = {'temp': {'kwargs': {'level': TEMPERATURE + LEVEL_SHIFT}}}
    modifier_before = copy.deepcopy(modifier)
    profile = _evaluate(reactor, [0.0, 0.0],
                        modify_controls=modifier)  # [mol/L]
    np.testing.assert_array_equal(profile, np.tile(CHARGE, (2, 1)))
    assert reactor.controls is configured
    assert reactor.controls == configured_before
    assert modifier == modifier_before


def _experiments(grid):
    """Build an initial-only and a positive-time experiment.

    Parameters
    ----------
    grid : str
        Key of ``CHARGE_GRIDS`` for the initial-only experiment.

    Returns
    -------
    tuple
        ``Experiment`` objects keyed by name, and the expected residuals
        model minus data [mol/L] and rate sensitivities [mol/L*s] at
        ``TRIAL_RATE``, packed by experiment and then sample.
    """
    times = CHARGE_GRIDS[grid]  # [s]
    errors = np.array(INITIAL_ERRORS[grid])  # [mol/L]
    initial_obs = CHARGE[0] + errors  # [mol/L]
    positive_obs = _analytic(BASE_A, TRUE_RATE,
                             POSITIVE_TIMES)[0][:, 0]  # [mol/L]
    experiments = {
        'initial_only': Experiment(
            {'c_A': Measurement(0, times, list(initial_obs))}),
        'positive': Experiment(
            {'c_A': Measurement(0, POSITIVE_TIMES, list(positive_obs))}),
    }
    # Model concentrations [mol/L] and rate sensitivities [mol/L*s].
    model, sens = _analytic(BASE_A, TRIAL_RATE, POSITIVE_TIMES)
    # Charge rows hold c_A0 exactly, so their residuals are c_A0 - data.
    residuals = np.concatenate((CHARGE[0] - initial_obs,
                                model[:, 0] - positive_obs))  # [mol/L]
    sensitivities = np.concatenate((np.zeros(len(times)),
                                    sens[:, 0]))  # [mol/L*s]
    return experiments, residuals, sensitivities


@pytest.mark.parametrize('return_sens', [False, True])
@pytest.mark.parametrize('grid', list(CHARGE_GRIDS))
def test_initial_only_experiment_through_simulation_exec(grid, return_sens):
    # Expected residuals [mol/L] and rate sensitivities [mol/L*s].
    experiments, residuals, sensitivities = _experiments(grid)
    simulation = SimulationExec(THERMO_PATH, {'R01': []})
    simulation.R01 = _reactor(TRIAL_RATE, return_sens)
    simulation.SetParamEstimation(
        x_data=experiments, optimize_flags=[True, False],
        wrapper_kwargs={'sundials_opts': SOLVER_OPTIONS})
    estimator = simulation.ParamInst

    residual = estimator.get_objective([TRIAL_RATE],
                                       out_array=True)  # [mol/L]
    np.testing.assert_allclose(residual, residuals, rtol=0.0,
                               atol=RESIDUAL_ATOL)
    # The charge rows carry no integration error.
    np.testing.assert_array_equal(residual[:len(CHARGE_GRIDS[grid])],
                                  residuals[:len(CHARGE_GRIDS[grid])])
    assert [len(run) for run in estimator.y_runs] == [
        len(CHARGE_GRIDS[grid]), len(POSITIVE_TIMES)]

    # Objective 1/2 r.r with identity weights; the error bound follows
    # from |r_i| errors of at most RESIDUAL_ATOL.
    objective = estimator.get_objective([TRIAL_RATE])  # [mol**2/L**2]
    expected = 0.5 * residuals @ residuals  # [mol**2/L**2]
    bound = (np.sum(np.abs(residuals)) * RESIDUAL_ATOL
             + 0.5 * residuals.size * RESIDUAL_ATOL**2)  # [mol**2/L**2]
    assert objective == pytest.approx(expected, rel=0.0, abs=bound)

    jacobian = estimator.get_gradient([TRIAL_RATE],
                                      out_array=True)[0]  # [mol/L*s]
    rows = len(CHARGE_GRIDS[grid])
    # Charge-time rows do not depend on k: exactly zero for CVodes
    # sensitivities and for finite differences of identical charge rows.
    np.testing.assert_array_equal(jacobian[:rows], 0.0)
    if return_sens:
        np.testing.assert_allclose(jacobian, sensitivities, rtol=0.0,
                                   atol=SENSITIVITY_ATOL)
    else:
        latest_s = POSITIVE_TIMES[-1]  # [s]
        np.testing.assert_allclose(
            jacobian, sensitivities,
            rtol=FD_RELATIVE_STEP * TRIAL_RATE * latest_s, atol=FD_NOISE_ATOL)
