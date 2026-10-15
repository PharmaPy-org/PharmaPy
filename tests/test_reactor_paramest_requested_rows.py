"""Reactor parameter-estimation callbacks return the requested sample rows.

``_BaseReactor.paramest_wrapper`` integrates with the requested times as
CVode output points. CVode also reports the initial state and merges
repeated initial times, so the wrapper selects one row per requested time,
in request order with replicates repeated (issue #346), matching times
exactly so multiscale grids keep distinct early samples, and reports
requested points that CVode's default output omits. A real
``BatchReactor`` integrates an isothermal first-order A -> B reaction with
zero activation energy, whose analytic solution
c_A = c_A0 exp(-k t), c_B = c_B0 + c_A0 (1 - exp(-k t)) and rate
sensitivities dc_A/dk = -t c_A0 exp(-k t) = -dc_B/dk give every expected
value. A few dozen reactor solves; requires the optional Assimulo stack.
"""

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


THERMO_PATH = str(
    Path(__file__).parent / 'integration/data/pfr_test_pure_comp.json')
TEMPERATURE = 300.0  # [K], isothermal; zero activation energy below
VOLUME = 0.002  # [m**3], concentrations are independent of volume
UTILITY_FLOW = 0.1  # [kg/s], unused by the isothermal balance
BASE_A = 0.1  # [mol/L], reactor charge of A
PRODUCT_INITIAL = 0.2  # [mol/L], c_B0
SOLVENT = 5.0  # [mol/L], inert solvent
TRUE_RATE = 0.2  # [1/s], generates the observations
TRIAL_RATE = 0.15  # [1/s], evaluation point and fit seed
# CVode tolerances and the 100-fold profile budget of the committed reactor
# regressions (test_paramestim_accepted_reactor.py).
SOLVER_OPTIONS = {'rtol': 1e-9, 'atol': 1e-11}  # [-], [mol/L]
PROFILE_RTOL = 1e-7  # [-]
PROFILE_ATOL = 1e-9  # [mol/L]
# Assimulo's CVode includes sensitivities in its error test by default
# (suppress_sens=False): PROFILE_RTOL of |dc/dk| <= 1.5 mol/L*s here is
# 1.5e-7 mol/L*s; 1e-6 mol/L*s leaves a factor of six.
SENSITIVITY_ATOL = 1e-6  # [mol/L*s]
# ParameterEstimation's default forward step is k * sqrt(1e-6); the
# truncation (h t / 2)|dc/dk| is bounded by FD_RELATIVE_STEP * k * t_max
# relative, and differences of two CVode solves stay below 1e-6 mol/L*s
# (test_simexec_observation_masks.py), bounded here by 1e-5 mol/L*s.
FD_RELATIVE_STEP = 1e-3  # [-]
FD_NOISE_ATOL = 1e-5  # [mol/L*s]
# Profile errors e_max = PROFILE_RTOL * 0.5 + PROFILE_ATOL (5.1e-8 mol/L)
# over n = 6 observed entries with rate-sensitivity norm |J| = 0.574
# mol/L*s at k = 0.2 1/s shift the fit by sqrt(n) e_max / |J| = 2.2e-7 1/s;
# the t = 0 replicates have zero rate sensitivity, so their deliberate
# errors do not move it. 1e-6 1/s keeps a factor of four.
FIT_ATOL = 1e-6  # [1/s]
# Requested grids and, by hand, the rows of the raw solver output that hold
# them: CVode prepends t = 0 and merges repeated initial times.
GRIDS = {
    'endpoint': ([2.0], [1]),
    'interior': ([1.0, 2.0, 3.0], [1, 2, 3]),
    'initial replicates': ([0.0, 0.0, 1.0, 2.0], [0, 0, 1, 2]),
}


def _reactor(rate, return_sens):
    """Build a real isothermal first-order batch reactor.

    Parameters
    ----------
    rate : float
        Rate constant k [1/s].
    return_sens : bool
        Whether ``paramest_wrapper`` returns CVodes sensitivities.

    Returns
    -------
    BatchReactor
        Reactor charged with A, B and solvent [mol/L].
    """
    reactor = BatchReactor(isothermal=True, mask_params=[True, False],
                           return_sens=return_sens)
    reactor.Kinetics = RxnKinetics(
        THERMO_PATH, k_params=[rate], ea_params=[0.0],
        rxn_list=['A --> B'])  # [1/s], [J/mol]
    reactor.Phases = LiquidPhase(
        THERMO_PATH, temp=TEMPERATURE, vol=VOLUME,
        mole_conc=[BASE_A, PRODUCT_INITIAL, 0.0, SOLVENT])
    reactor.Utility = CoolingWater(temp_in=TEMPERATURE,
                                   mass_flow=UTILITY_FLOW)
    return reactor


def _analytic(initial, rate, time_s):
    """Return analytic concentrations and rate sensitivities.

    Parameters
    ----------
    initial : float
        Initial concentration of A [mol/L].
    rate : float
        Rate constant k [1/s].
    time_s : array_like
        Times [s], shape ``(n,)``.

    Returns
    -------
    tuple of numpy.ndarray
        Concentrations of A and B [mol/L], shape ``(n, 2)``, and their
        derivatives with respect to k [mol/L*s], shape ``(n, 2)``.
    """
    time_s = np.asarray(time_s, dtype=float)  # [s]
    reactant = initial * np.exp(-rate * time_s)  # [mol/L]
    sensitivity = -time_s * reactant  # [mol/L*s]
    return (np.column_stack((reactant,
                             PRODUCT_INITIAL + initial - reactant)),
            np.column_stack((sensitivity, -sensitivity)))


@pytest.mark.parametrize('grid', list(GRIDS))
@pytest.mark.parametrize('return_sens', [False, True])
def test_wrapper_returns_one_row_per_requested_time(grid, return_sens):
    times, raw_rows = GRIDS[grid]
    reactor = _reactor(TRUE_RATE, return_sens)
    params = reactor.Kinetics.concat_params()  # [1/s], [J/mol]
    run_args = {'sundials_opts': SOLVER_OPTIONS}
    result = reactor.paramest_wrapper(params, np.array(times),
                                      run_args=run_args)
    profile = result[0] if return_sens else result  # [mol/L]
    expected, expected_sens = _analytic(BASE_A, TRUE_RATE, times)
    assert profile.shape == (len(times), 2)
    np.testing.assert_allclose(profile, expected, rtol=PROFILE_RTOL,
                               atol=PROFILE_ATOL)
    if not return_sens:
        return

    # Reordered layout: rows state-major (A then B, each over the requested
    # times), columns parameters (k, then activation energy).
    sensitivity = result[1]
    assert sensitivity.shape == (2 * len(times), 2)
    np.testing.assert_allclose(sensitivity[:, 0], expected_sens.T.ravel(),
                               rtol=0.0, atol=SENSITIVITY_ATOL)

    # Unreordered layout: every parameter block holds the requested rows of
    # the raw solver output, selected by the hand-written row map.
    stacked = reactor.paramest_wrapper(params, np.array(times),
                                       reord_sens=False, run_args=run_args)[1]
    # Same fresh batch the wrapper integrates: reset charge and clock.
    reactor.reset()
    reactor.Kinetics.set_params(params)
    reactor.elapsed_time = 0.0  # [s]
    _, _, raw = reactor.solve_unit(time_grid=np.array(times), verbose=False,
                                   eval_sens=True, **run_args)
    assert stacked.shape == (2, len(times), 2)
    for block, raw_block in zip(stacked, raw):
        np.testing.assert_array_equal(block, raw_block[raw_rows])


def test_missing_requested_time_is_reported():
    reactor = _reactor(TRUE_RATE, return_sens=False)
    # CVode integrates to the last requested time, so an earlier request
    # after it in a decreasing grid is never reported.
    with pytest.raises(ValueError, match=(
            r"^Requested time 2\.0 s is absent from the reactor solution")):
        reactor.paramest_wrapper(reactor.Kinetics.concat_params(),
                                 np.array([0.0, 2.0, 1.0]),
                                 run_args={'sundials_opts': SOLVER_OPTIONS})


# Multiscale grids, each with a rate making k * t_max moderate (36 and 50),
# so the decay is well resolved, and the raw solver rows of each request.
# A tolerance relative to the largest time (4 ulps of 3600 s is 1.8e-12 s;
# of 1e8 s, 6e-8 s) would merge the early samples with neighbours.
MULTISCALE_GRIDS = {
    'picoseconds in an hour': ([0.0, 1e-12, 2e-12, 3600.0], 0.01,
                               [0, 1, 2, 3]),
    'ten nanoseconds in 1e8 s': ([1e-8, 1e8], 5e-7, [1, 2]),
    'initial replicates in 1e8 s': ([0.0, 0.0, 1e-8, 1e8], 5e-7,
                                    [0, 0, 1, 2]),
}
# Samples with k t < EARLY_KT are in the initial transient, where CVode's
# steps are tiny and its relative error is of order SOLVER rtol; they are
# checked relative to PROFILE_RTOL. Late samples are checked absolutely.
EARLY_KT = 1.0  # [-]
# Assimulo CVode defaults (rtol = atol = 1e-6), times the same 100-fold
# integration-error budget as PROFILE_RTOL and PROFILE_ATOL.
DEFAULT_PROFILE_RTOL = 1e-4  # [-]
DEFAULT_PROFILE_ATOL = 1e-4  # [mol/L]


@pytest.mark.parametrize('grid', list(MULTISCALE_GRIDS))
@pytest.mark.parametrize('return_sens', [False, True])
def test_multiscale_grids_keep_distinct_samples(grid, return_sens):
    times, rate, raw_rows = MULTISCALE_GRIDS[grid]
    times = np.array(times)  # [s]
    reactor = _reactor(rate, return_sens)
    params = reactor.Kinetics.concat_params()  # [1/s], [J/mol]
    run_args = {'sundials_opts': SOLVER_OPTIONS}
    result = reactor.paramest_wrapper(params, times, reord_sens=False,
                                      run_args=run_args)
    profile = result[0] if return_sens else result  # [mol/L]
    expected, expected_sens = _analytic(BASE_A, rate, times)
    early = rate * times < EARLY_KT
    assert profile.shape == (len(times), 2)
    np.testing.assert_allclose(profile[early], expected[early],
                               rtol=PROFILE_RTOL, atol=0.0)
    np.testing.assert_allclose(profile[~early], expected[~early], rtol=0.0,
                               atol=PROFILE_ATOL)

    # Replicates of the initial time all hold the charge itself [mol/L].
    initial_rows = profile[times == 0.0]
    np.testing.assert_array_equal(
        initial_rows, np.tile([BASE_A, PRODUCT_INITIAL],
                              (len(initial_rows), 1)))

    # Every returned row is the raw solver row of its requested time.
    reactor.reset()
    reactor.Kinetics.set_params(params)
    reactor.elapsed_time = 0.0  # [s]
    raw = reactor.solve_unit(time_grid=times, verbose=False,
                             eval_sens=return_sens, **run_args)
    np.testing.assert_array_equal(profile, raw[1][raw_rows][:, :2])
    if return_sens:
        # Early k-sensitivities (-t c_A0 at t = 1e-12 s or 1e-8 s) differ
        # from those of neighbouring samples by a factor of two or more.
        np.testing.assert_allclose(result[1][0][early],
                                   expected_sens[early], rtol=PROFILE_RTOL,
                                   atol=0.0)
        for block, raw_block in zip(result[1], raw[2]):
            np.testing.assert_array_equal(block, raw_block[raw_rows])


def test_points_omitted_by_default_output_are_reported():
    times = np.linspace(0.0, 10.0, 51)  # [s], 0.2 s spacing
    reactor = _reactor(TRUE_RATE, return_sens=False)
    params = reactor.Kinetics.concat_params()  # [1/s], [J/mol]
    # With CVode's default tolerances and non-continuous output, the points
    # 9.6 s and 9.8 s inside its last internal step are not reported.
    with pytest.raises(ValueError, match=(
            r"^Requested time 9\.600000000000001 s is absent from the reactor "
            r"solution, which reports 49 times from 0\.0 s to 10\.0 s\. "
            r"CVode's default \(non-continuous\) output can omit requested "
            r"points inside its last internal step before the end time; pass "
            r"run_args=\{'sundials_opts': \{'report_continuously': True\}\}")):
        reactor.paramest_wrapper(params, times)

    profile = reactor.paramest_wrapper(
        params, times,
        run_args={'sundials_opts': {'report_continuously': True}})
    np.testing.assert_allclose(
        profile, _analytic(BASE_A, TRUE_RATE, times)[0],
        rtol=DEFAULT_PROFILE_RTOL, atol=DEFAULT_PROFILE_ATOL)


def _experiments():
    """Build an endpoint-only and an initial-replicate experiment.

    Returns
    -------
    tuple of dict
        ``Experiment`` objects and phase modifiers (initial charges
        [mol/L]), keyed by experiment name, plus the hand-packed expected
        layout: per experiment ``(initial A [mol/L], model grid [s],
        observed c_A, observed c_B)`` with observations [mol/L] or None
        for unobserved entries.
    """
    endpoint_a = 0.3  # [mol/L]
    replicate_a = 0.05  # [mol/L]
    endpoint_state = _analytic(endpoint_a, TRUE_RATE, [2.0])[0][0]  # [mol/L]
    replicate_state = _analytic(replicate_a, TRUE_RATE, [1.0, 2.0])[0]
    # Replicate initial observations straddle c_A0 by +-2e-3 mol/L; the
    # initial state does not depend on k, so they do not bias the fit.
    initial_error = 2e-3  # [mol/L]
    replicate_a_obs = [replicate_a + initial_error,
                       replicate_a - initial_error,
                       replicate_state[0, 0], replicate_state[1, 0]]
    experiments = {
        'endpoint': Experiment(
            {'c_A': Measurement(0, [2.0], [endpoint_state[0]]),
             'c_B': Measurement(1, [2.0], [endpoint_state[1]])}),
        'replicate': Experiment(
            {'c_A': Measurement(0, [0.0, 0.0, 1.0, 2.0], replicate_a_obs)}),
    }
    modifiers = {name: {'mole_conc': [initial, PRODUCT_INITIAL, 0.0,
                                      SOLVENT]}
                 for name, initial in (('endpoint', endpoint_a),
                                       ('replicate', replicate_a))}
    layout = [(endpoint_a, [2.0], [endpoint_state[0]], [endpoint_state[1]]),
              (replicate_a, [0.0, 0.0, 1.0, 2.0], replicate_a_obs,
               [None] * 4)]
    return experiments, modifiers, layout


def _expected_residuals(layout, rate):
    """Pack analytic residuals and rate sensitivities by hand.

    Parameters
    ----------
    layout : list of tuple
        ``_experiments`` layout.
    rate : float
        Evaluation rate constant [1/s].

    Returns
    -------
    tuple of numpy.ndarray
        Residuals model minus data [mol/L] and sensitivities [mol/L*s],
        packed by experiment, then column (c_A, c_B), then grid sample;
        unobserved entries are zero.
    """
    residuals, sensitivities = [], []
    for initial, grid, *columns in layout:
        model, sens = _analytic(initial, rate, grid)
        for column, observations in enumerate(columns):
            for row, value in enumerate(observations):
                observed = value is not None
                residuals.append(model[row, column] - value if observed
                                 else 0.0)
                sensitivities.append(sens[row, column] if observed else 0.0)
    return np.array(residuals), np.array(sensitivities)


@pytest.mark.parametrize('return_sens', [False, True])
def test_unsampled_initial_time_and_replicates_through_simulation_exec(
        return_sens):
    experiments, modifiers, layout = _experiments()
    simulation = SimulationExec(THERMO_PATH, {'R01': []})
    simulation.R01 = _reactor(TRIAL_RATE, return_sens)
    simulation.SetParamEstimation(
        experiments, phase_modifiers=modifiers, optimize_flags=[True, False],
        wrapper_kwargs={'sundials_opts': SOLVER_OPTIONS})
    estimator = simulation.ParamInst

    residuals, sensitivities = _expected_residuals(layout, TRIAL_RATE)
    residual = estimator.get_objective([TRIAL_RATE], out_array=True)
    np.testing.assert_allclose(residual, residuals, rtol=0.0,
                               atol=PROFILE_RTOL * 0.5 + PROFILE_ATOL)
    assert [len(run) for run in estimator.y_runs] == [1, 4]

    jacobian = estimator.get_gradient([TRIAL_RATE], out_array=True)[0]
    if return_sens:
        np.testing.assert_allclose(jacobian, sensitivities, rtol=0.0,
                                   atol=SENSITIVITY_ATOL)
    else:
        latest_s = 2.0  # [s], last sample of both experiments
        np.testing.assert_allclose(
            jacobian, sensitivities,
            rtol=FD_RELATIVE_STEP * TRIAL_RATE * latest_s, atol=FD_NOISE_ATOL)

    fitted, _, _ = simulation.EstimateParams(verbose=False)  # [1/s]
    np.testing.assert_allclose(fitted, [TRUE_RATE], rtol=0.0, atol=FIT_ATOL)
