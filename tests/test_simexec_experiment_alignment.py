"""Experiment alignment through real SimulationExec, batch kinetics and CVode.

Issue #238 (named experiments) and issue #346 (positional experiments and
``Experiment`` objects, with per-experiment wrapper keywords built by
``SetParamEstimation``). A synthetic isothermal first-order A -> B reaction
with zero activation energy has the analytic solution
c_A = c_A0 exp(-k t), c_B = c_B0 + c_A0 (1 - exp(-k t)), with rate
sensitivities dc_A/dk = -t c_A0 exp(-k t) = -dc_B/dk. Different initial
concentrations (phase modifiers) and unequal schedules expose experiment
mismatches. Each test integrates the reactor a few dozen times at most;
requires the optional Assimulo stack.
"""

import copy
from pathlib import Path

import numpy as np
import pytest

pytestmark = [pytest.mark.assimulo, pytest.mark.integration]
pytest.importorskip('assimulo')

from PharmaPy.Kinetics import RxnKinetics
from PharmaPy.ParamEstim import Experiment, Measurement
from PharmaPy.Phases import LiquidPhase
from PharmaPy.Reactors import BatchReactor
from PharmaPy.SimExec import SimulationExec
from PharmaPy.Utilities import CoolingWater


def _batch_simulation(rate, return_sens=False):
    """Build a simulation around an isothermal first-order batch reactor.

    Parameters
    ----------
    rate : float
        First-order A -> B rate constant [1/s], also the estimation seed.
    return_sens : bool, optional
        Whether the reactor wrapper returns CVodes rate sensitivities
        instead of leaving them to finite differences. The default is False.

    Returns
    -------
    SimulationExec
        Flowsheet whose only unit, ``R01``, is the reactor.
    """
    path = str(Path(__file__).parent / 'integration/data/pfr_test_pure_comp.json')
    reactor = BatchReactor(isothermal=True, mask_params=[True, False],
                           return_sens=return_sens)
    reactor.Kinetics = RxnKinetics(
        path, k_params=[rate], ea_params=[0.], rxn_list=['A --> B'])
    # Zero activation energy [J/mol] removes temperature dependence.
    reactor.Phases = LiquidPhase(
        path, temp=300., vol=0.002, mole_conc=[0.1, 0.2, 0., 5.])
    # [K], [m**3], [mol/L]; dilute reacting solutes in an inert solvent.
    reactor.Utility = CoolingWater(temp_in=300., mass_flow=.1)
    # [K], [kg/s], satisfies the collaborator contract; temperature is fixed.
    simulation = SimulationExec(path, {'R01': []})
    simulation.R01 = reactor
    return simulation


@pytest.mark.parametrize('single', [False, True])
def test_named_experiments_reach_the_matching_reactor_initial_state(single):
    rate = 0.2  # [1/s], synthetic first-order rate, reaction time scale 5 s
    simulation = _batch_simulation(rate)
    times = {'dilute': np.array([0., 1., 3.]),
             'concentrated': np.array([0., 2., 4., 6.])}  # [s]
    initial = {'dilute': .1, 'concentrated': .3}  # [mol/L], asymmetric charges
    if single:
        times.pop('concentrated')
        initial.pop('concentrated')
    observations = {}
    modifiers = {}
    for name in reversed(times):
        reactant = initial[name] * np.exp(-rate * times[name])  # [mol/L]
        observations[name] = np.column_stack(
            (reactant, .2 + initial[name] - reactant))  # [mol/L], A then B
        modifiers[name] = {'mole_conc': [initial[name], .2, 0., 5.]}  # [mol/L]
    # Tight integration tolerances make a 2e-7 mol/L comparison conservative.
    simulation.SetParamEstimation(
        times, observations, measured_ind=[0, 1],
        optimize_flags=[True, False], phase_modifiers=modifiers,
        wrapper_kwargs={'sundials_opts': {'rtol': 1e-9, 'atol': 1e-11}})
    residual = simulation.ParamInst.get_objective([rate], out_array=True)
    # [mol/L] under identity numerical weighting; all observed entries checked.
    np.testing.assert_allclose(residual, 0., rtol=0., atol=2e-7)
    for index, name in enumerate(times):
        np.testing.assert_allclose(simulation.ParamInst.y_runs[index],
                                   observations[name], rtol=0., atol=2e-7)
    assert list(observations) == list(reversed(times))


TRUE_RATE = 0.2  # [1/s], generates the noise-free observations
TRIAL_RATE = 0.15  # [1/s], evaluation point and fit seed, O(1e-2) residuals
PRODUCT_INITIAL = 0.2  # [mol/L], c_B0 of every experiment (base charge)
SOLVENT = 5.0  # [mol/L], inert solvent of the base charge
# CVode tolerances of the committed reactor regressions (sundials_opts).
SOLVER_RTOL = 1e-9  # [-]
SOLVER_ATOL = 1e-11  # [mol/L]
# 100-fold integration-error budget of test_paramestim_accepted_reactor.py.
PROFILE_RTOL = 1e-7  # [-]
PROFILE_ATOL = 1e-9  # [mol/L]
# c_B0 + c_A0 of the most concentrated run bounds every state [mol/L], so
# PROFILE_RTOL * MAX_CONCENTRATION + PROFILE_ATOL bounds each profile error.
MAX_CONCENTRATION = 0.5  # [mol/L]
# ParameterEstimation's default forward-difference step is k * sqrt(1e-6)
# (PharmaPy.jac_module.dx_jac_p with rel_tol 1e-6).
FD_RELATIVE_STEP = 1e-3  # [-]
# Two CVode solves differ by at most twice the profile error budget; over the
# step 1e-3 * 0.15 1/s that is 7e-4 mol/L*s, a worst case. The solves share
# step sequences, so the observed difference error is below 1e-6 mol/L*s;
# 1e-5 mol/L*s keeps a decade of margin (test_simexec_observation_masks.py).
FD_NOISE_ATOL = 1e-5  # [mol/L*s]
# Assimulo's CVode includes the sensitivities in its error test by default
# (suppress_sens=False), so their relative error is within the profile budget
# PROFILE_RTOL of |dc/dk| = t * c_A0 * exp(-k t) <= 1.5 mol/L*s here, i.e.
# 1.5e-7 mol/L*s; 1e-6 mol/L*s leaves a factor of six.
SENSITIVITY_ATOL = 1e-6  # [mol/L*s]
# Noise-free data are fitted exactly up to integration error: profile errors
# e (at most e_max = PROFILE_RTOL * MAX_CONCENTRATION + PROFILE_ATOL, 5.1e-8
# mol/L) shift the least-squares rate by |J^T e| / |J|**2 <= sqrt(n) e_max /
# |J| over the n observed entries, with |J| the analytic rate sensitivity
# norm at k = 0.2 1/s: n = 9, |J| = 0.914 mol/L*s (Experiment objects, 1.7e-7
# 1/s); n = 14, |J| = 1.229 (two positional runs, 1.6e-7); n = 24, |J| =
# 1.328 (three, 1.9e-7). Even the worst pairing sqrt(24) e_max / 0.914 gives
# 2.7e-7 1/s; LM's default termination adds about 1e-8 relative. 1e-6 1/s
# keeps a factor of 3.7 over that pairing.
FIT_ATOL = 1e-6  # [1/s]

# Experiment specifications: initial A [mol/L] and measurements in declared
# order, each (name, model field, times [s], NaN-masked sample indices).
# 'concentrated' measures B at one point only; 'dilute' never measures B and
# has one missing A observation at t = 4 s.
EXPERIMENT_SPECS = {
    'concentrated': (0.3, [('c_B', 1, [2.5], []),
                           ('c_A', 0, [0.0, 1.0, 3.0, 5.0], [])]),
    'dilute': (0.05, [('c_A', 0, [0.0, 2.0, 4.0, 6.0, 8.0], [2])]),
}


def _analytic(field, initial, rate, time_s):
    """Return an analytic concentration and its rate sensitivity.

    Parameters
    ----------
    field : int
        Model output column: 0 for A, 1 for B.
    initial : float
        Initial concentration of A, c_A0 [mol/L].
    rate : float
        Rate constant k [1/s].
    time_s : numpy.ndarray
        Times [s], shape ``(n,)``.

    Returns
    -------
    tuple of numpy.ndarray
        Concentration [mol/L] and its derivative with respect to k
        [mol/L*s], each of shape ``(n,)``.
    """
    reactant = initial * np.exp(-rate * time_s)  # [mol/L]
    sensitivity = -time_s * reactant  # [mol/L*s], dc_A/dk
    if field == 0:
        return reactant, sensitivity
    return PRODUCT_INITIAL + initial - reactant, -sensitivity


def _experiments(order, measurement_order):
    """Build noise-free ``Experiment`` objects and their phase modifiers.

    Parameters
    ----------
    order : sequence of str
        Experiment names in mapping order.
    measurement_order : {1, -1}
        Declared measurement order (1) or its reverse (-1).

    Returns
    -------
    tuple of dict
        Experiments and ``phase_modifiers`` (initial charge [mol/L], in the
        reverse experiment order), both keyed by name.
    """
    experiments = {}
    for name in order:
        initial, specs = EXPERIMENT_SPECS[name]
        measurements = {}
        for label, field, times, missing in specs[::measurement_order]:
            time_s = np.array(times)  # [s]
            values = _analytic(field, initial, TRUE_RATE, time_s)[0]
            values[missing] = np.nan  # [mol/L], missing observations
            measurements[label] = Measurement(field, time_s, values,
                                              units='mol/L')
        experiments[name] = Experiment(measurements)
    modifiers = {name: {'mole_conc': [EXPERIMENT_SPECS[name][0],
                                      PRODUCT_INITIAL, 0.0, SOLVENT]}
                 for name in reversed(order)}  # [mol/L]
    return experiments, modifiers


def _expected_packing(order, measurement_order, rate):
    """Pack analytic residuals and sensitivities by the documented layout.

    The layout is experiment order, then measurement columns in
    first-appearance order, then sorted model-grid samples (the union of
    the experiment's measurement grids); unobserved entries are zero.

    Parameters
    ----------
    order : sequence of str
        Experiment names in experiment order.
    measurement_order : {1, -1}
        Declared measurement order (1) or its reverse (-1).
    rate : float
        Evaluation rate constant [1/s].

    Returns
    -------
    tuple of numpy.ndarray
        Residuals model minus data [mol/L], sensitivities [mol/L*s], and the
        observed mask, all of shape ``(n_entries,)``, plus the bound
        ``|model|`` [mol/L] used for the integration-error budget.
    """
    columns = []
    for name in order:
        for label, *_ in EXPERIMENT_SPECS[name][1][::measurement_order]:
            if label not in columns:
                columns.append(label)
    residuals, sensitivities, observed, magnitude = [], [], [], []
    for name in order:
        initial, specs = EXPERIMENT_SPECS[name]
        by_label = {label: (field, np.array(times), missing)
                    for label, field, times, missing in specs}
        grid = np.unique(np.concatenate(
            [times for _, times, _ in by_label.values()]))  # [s]
        for label in columns:
            residual = np.zeros(grid.size)  # [mol/L]
            sensitivity = np.zeros(grid.size)  # [mol/L*s]
            mask = np.zeros(grid.size, dtype=bool)
            model = np.zeros(grid.size)  # [mol/L]
            if label in by_label:
                field, times, missing = by_label[label]
                kept = np.delete(times, missing)  # [s], observed samples
                rows = np.searchsorted(grid, kept)
                data = _analytic(field, initial, TRUE_RATE, kept)[0]
                value, derivative = _analytic(field, initial, rate, kept)
                residual[rows] = value - data
                sensitivity[rows] = derivative
                mask[rows] = True
                model[rows] = value
            residuals.append(residual)
            sensitivities.append(sensitivity)
            observed.append(mask)
            magnitude.append(model)
    return (np.concatenate(residuals), np.concatenate(sensitivities),
            np.concatenate(observed), np.concatenate(magnitude))


def _assert_unchanged(caller, before):
    """Assert that nested caller data equal their deep-copied snapshot.

    Parameters
    ----------
    caller : object
        Arrays, ``Experiment`` and ``Measurement`` objects, mappings,
        sequences and scalars passed to ``SetParamEstimation``.
    before : object
        ``copy.deepcopy`` of ``caller`` taken before the calls.

    Raises
    ------
    AssertionError
        If a value, container type, key order or array dtype changed. NaN
        observations compare equal to NaN.
    """
    assert type(caller) is type(before)
    if isinstance(caller, np.ndarray):
        assert caller.dtype == before.dtype
        np.testing.assert_array_equal(caller, before)
    elif isinstance(caller, Experiment):
        _assert_unchanged(dict(caller.measurements),
                          dict(before.measurements))
        _assert_unchanged(caller.args, before.args)
        _assert_unchanged(dict(caller.kwargs), dict(before.kwargs))
    elif isinstance(caller, Measurement):
        assert (caller.field, caller.units, caller.basis, caller.x_units) == (
            before.field, before.units, before.basis, before.x_units)
        for attribute in ('x', 'values', 'uncertainty'):
            _assert_unchanged(getattr(caller, attribute),
                              getattr(before, attribute))
    elif isinstance(caller, dict):
        assert list(caller) == list(before)
        for key in caller:
            _assert_unchanged(caller[key], before[key])
    elif isinstance(caller, (list, tuple)):
        assert len(caller) == len(before)
        for item, item_before in zip(caller, before):
            _assert_unchanged(item, item_before)
    else:
        assert caller == before


@pytest.mark.parametrize('return_sens', [False, True])
@pytest.mark.parametrize('order, measurement_order', [
    (('concentrated', 'dilute'), 1), (('dilute', 'concentrated'), -1)])
def test_experiment_objects_fit_through_simulation_exec(
        return_sens, order, measurement_order):
    simulation = _batch_simulation(TRIAL_RATE, return_sens=return_sens)
    experiments, modifiers = _experiments(order, measurement_order)
    solver = {'sundials_opts': {'rtol': SOLVER_RTOL, 'atol': SOLVER_ATOL}}
    caller = (experiments, modifiers, solver)
    before = copy.deepcopy(caller)
    simulation.SetParamEstimation(experiments, phase_modifiers=modifiers,
                                  optimize_flags=[True, False],
                                  wrapper_kwargs=solver)
    estimator = simulation.ParamInst
    _assert_unchanged(caller, before)

    residuals, sensitivities, observed, magnitude = _expected_packing(
        order, measurement_order, TRIAL_RATE)
    budget = PROFILE_RTOL * np.abs(magnitude) + PROFILE_ATOL  # [mol/L]
    # [mol/L], identity weights; unobserved entries are exactly zero.
    residual = estimator.get_objective([TRIAL_RATE], out_array=True)
    np.testing.assert_array_equal(residual[~observed], 0.0)
    np.testing.assert_array_less(np.abs(residual - residuals), budget)
    objective = estimator.get_objective([TRIAL_RATE])  # [(mol/L)**2]
    assert objective == pytest.approx(
        0.5 * residuals @ residuals, rel=0.0,
        abs=np.abs(residuals) @ budget + 0.5 * budget @ budget)

    jacobian = estimator.get_gradient([TRIAL_RATE], out_array=True)
    assert jacobian.shape == (1, residuals.size)
    np.testing.assert_array_equal(jacobian[0, ~observed], 0.0)
    if return_sens:
        np.testing.assert_allclose(jacobian[0], sensitivities, rtol=0.0,
                                   atol=SENSITIVITY_ATOL)
    else:
        # Forward-difference truncation is (h/2)|y''| = (h t / 2)|y'| with
        # h = FD_RELATIVE_STEP * k; h * t_max bounds it with a factor 2.
        latest_s = max(max(times) for _, specs in EXPERIMENT_SPECS.values()
                       for _, _, times, _ in specs)  # [s]
        truncation_rtol = FD_RELATIVE_STEP * TRIAL_RATE * latest_s  # [-]
        np.testing.assert_allclose(jacobian[0], sensitivities,
                                   rtol=truncation_rtol, atol=FD_NOISE_ATOL)

    fitted, _, info = simulation.EstimateParams(verbose=False)  # [1/s]
    np.testing.assert_allclose(fitted, [TRUE_RATE], rtol=0.0, atol=FIT_ATOL)
    np.testing.assert_array_equal(info['jac'][0, ~observed], 0.0)
    for index, name in enumerate(order):
        assert estimator.kwargs_fun[index]['modify_phase'] == modifiers[name]
    _assert_unchanged(caller, before)


@pytest.mark.parametrize('count', [2, 3])
def test_positional_experiments_get_their_own_wrapper_keywords(count):
    simulation = _batch_simulation(TRIAL_RATE)
    times = [np.array([0., 1., 3.]), np.array([0., 2., 4., 6.]),
             np.array([0., .5, 1., 2., 7.])][:count]  # [s], unequal
    initial = [.05, .3, .15][:count]  # [mol/L], asymmetric charges of A
    observations = [np.column_stack(
        [_analytic(field, charge, TRUE_RATE, time)[0] for field in (0, 1)])
        for charge, time in zip(initial, times)]  # [mol/L], A then B
    phase = [{'mole_conc': [charge, PRODUCT_INITIAL, 0., SOLVENT]}
             for charge in initial]  # [mol/L]
    control = [{} for _ in initial]  # isothermal reactor, no controls
    solver = {'sundials_opts': {'rtol': SOLVER_RTOL, 'atol': SOLVER_ATOL}}
    caller = (times, observations, phase, control, solver)
    before = copy.deepcopy(caller)
    simulation.SetParamEstimation(
        times, observations, measured_ind=[0, 1],
        optimize_flags=[True, False], phase_modifiers=phase,
        control_modifiers=control, wrapper_kwargs=solver)
    estimator = simulation.ParamInst
    # Three experiments once matched the three flat wrapper fields by count.
    assert estimator.kwargs_fun == [
        {'modify_phase': {'mole_conc': [charge, .2, 0., 5.]},
         'modify_controls': {},
         'run_args': {'sundials_opts': {'rtol': 1e-9, 'atol': 1e-11}}}
        for charge in initial]
    assert all(keywords['run_args'] is not solver
               for keywords in estimator.kwargs_fun)
    _assert_unchanged(caller, before)

    residual = estimator.get_objective([TRUE_RATE], out_array=True)
    np.testing.assert_allclose(
        residual, 0., rtol=0.,
        atol=PROFILE_RTOL * MAX_CONCENTRATION + PROFILE_ATOL)  # [mol/L]
    for index, expected in enumerate(observations):
        np.testing.assert_allclose(
            estimator.y_runs[index], expected, rtol=PROFILE_RTOL,
            atol=PROFILE_ATOL)  # [mol/L], experiment's own initial charge

    fitted, _, _ = simulation.EstimateParams(verbose=False)  # [1/s]
    np.testing.assert_allclose(fitted, [TRUE_RATE], rtol=0.0, atol=FIT_ATOL)
    _assert_unchanged(caller, before)


def test_experiment_args_bind_the_phase_modifier_positionally():
    # Experiment(args=(phase_modifier,)) binds paramest_wrapper's
    # modify_phase; SetParamEstimation must not inject it again by keyword.
    order = ('concentrated', 'dilute')
    simulation = _batch_simulation(TRIAL_RATE)
    keyword_experiments, modifiers = _experiments(order, 1)
    experiments = {name: Experiment(dict(experiment.measurements),
                                    args=(modifiers[name],))
                   for name, experiment in keyword_experiments.items()}
    solver = {'sundials_opts': {'rtol': SOLVER_RTOL, 'atol': SOLVER_ATOL}}
    simulation.SetParamEstimation(experiments, optimize_flags=[True, False],
                                  wrapper_kwargs=solver)
    estimator = simulation.ParamInst
    assert [args[0] for args in estimator.args_fun] == [
        modifiers[name] for name in order]

    residuals, _, observed, magnitude = _expected_packing(order, 1,
                                                          TRIAL_RATE)
    budget = PROFILE_RTOL * np.abs(magnitude) + PROFILE_ATOL  # [mol/L]
    residual = estimator.get_objective([TRIAL_RATE], out_array=True)
    np.testing.assert_array_equal(residual[~observed], 0.0)
    np.testing.assert_array_less(np.abs(residual - residuals), budget)
    # Each run starts from its own charge of A [mol/L], not the base 0.1.
    # Columns are c_B then c_A (first appearance); both grids start at 0 s.
    for index, name in enumerate(order):
        np.testing.assert_allclose(
            estimator.y_runs[index][0, 1], EXPERIMENT_SPECS[name][0],
            rtol=0.0, atol=PROFILE_ATOL)
