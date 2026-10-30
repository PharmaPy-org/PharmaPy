"""Crystallizer estimation callbacks return the requested sample rows (#328).

``_BaseCryst.paramest_wrapper``, shared by ``BatchCryst``, ``MSMPR`` and
``SemibatchCryst``, integrates with the requested times as CVode output
points. CVode also reports the charged state at 0 s and merges replicates of
the initial time, so on its default paths (moments: states and
sensitivities; finite volumes: states) the wrapper returns one row per
requested time, in request order with replicates repeated, for the states
and for every sensitivity block. Rows are matched by exact time through the
helper the reactors use, so a missing time, for example after a terminating
state event, raises ``ValueError`` instead of returning misaligned rows. The
stored ``result`` and heat duty still cover the whole integration from the
charge, and a custom moments ``param_wrapper`` still receives every solver
row. A grid of charge times only is rejected for every method and wrapper.

Fixture: the seeded, growth-only ``BatchCryst`` of the issue #328
reproduction (real compound database, CVode/CVODES), charged at 310 K with
93.3 kg/m**3 of A against a constant solubility of 2 kg/m**3; ``MSMPR`` and
``SemibatchCryst`` variants add a constant feed of the charged slurry
(0.02 m**3/s, a 100 s residence time for the 2 m**3 charge). Only the
growth rate constant and growth order are estimated, so the sensitivity
blocks have unequal dimensions (2 parameters, 10 states). Expected rows come
from a reference ``solve_unit`` of an identically built unit on the same
grid (reset first for the fed units, as the wrapper does), selected by row
indices written out by hand from the reported times,
which the tests also assert; the charged state comes from the unsolved
phases. Growth without nucleation keeps the zeroth moment constant, and the
charged state does not depend on the kinetic parameters, so its
sensitivities are zero. CVode reports a replicate later than the charge,
or a terminating event time, twice; the repeated time makes the stored heat
duty NaN with a RuntimeWarning (issue #314), which these tests do not
assert. The replicate mapping of the shared row helper is checked without a
solver in ``test_commons_helpers.py``. About forty sub-second moments solves
and six finite-volume solves of one to two seconds each; requires the
optional Assimulo stack.
"""

from pathlib import Path

import numpy as np
import pytest

pytestmark = [pytest.mark.assimulo, pytest.mark.integration]
pytest.importorskip('assimulo')

from PharmaPy.Crystallizers import MSMPR, BatchCryst, SemibatchCryst
from PharmaPy.Errors import PharmaPyNonImplementedError
from PharmaPy.Kinetics import CrystKinetics
from PharmaPy.ParamEstim import Experiment, Measurement, ParameterEstimation
from PharmaPy.MixedPhases import SlurryStream
from PharmaPy.Phases import LiquidPhase, SolidPhase
from PharmaPy.Streams import LiquidStream, SolidStream


DATABASE = str(Path(__file__).parent / 'Flowsheet' / 'data'
               / 'compound_database.json')
# Synthetic diagnostic case of the issue #328 reproduction, not a fitted
# process: prescribed constant temperature, five database species with the
# target A first, and a Gaussian seed population.
TEMPERATURE = 310.0  # [K]
VOLUME = 2.0  # [m**3], liquid charge
LIQUID_MASS_FRACTIONS = [0.1, 0.1, 0.1, 0.1, 0.6]  # [-]
SOLID_MASS_FRACTIONS = [1.0, 0.0, 0.0, 0.0, 0.0]  # [-], crystals of A
SHAPE_FACTOR = 0.5  # [-], volumetric shape factor kv
SATURATION = 2.0  # [kg/m**3], constant solubility of A
# Growth rate constant [um/s], activation energy [J/mol] and order [-];
# nucleation and dissolution parameters stay at their zero defaults.
GROWTH = (1.0, 0.0, 1.0)
SEED_PEAK = 1e7  # [#/um], number density at the seed mode
SEED_MODE = 100.0  # [um]
SEED_WIDTH = 40.0  # [um], Gaussian 1/e half width
NUM_MOMENTS = 4  # [-], moments mu_0 to mu_3
# Size grid of the issue fixture for the moments method (only the seed
# moments use it), and a coarser grid that keeps a finite-volume solve near
# one second.
MOMENTS_SIZE_GRID = np.geomspace(1.0, 500.0, 40)  # [um]
FVM_SIZE_GRID = np.geomspace(1.0, 500.0, 20)  # [um]
# CrystKinetics.concat_params order: primary nucleation (3), secondary
# nucleation (4), growth (3), dissolution (3).
NUM_KINETIC_PARAMS = 13  # [-]
GROWTH_RATE_COLUMN = 7
GROWTH_ORDER_COLUMN = 9
ACTIVE_COLUMNS = [GROWTH_RATE_COLUMN, GROWTH_ORDER_COLUMN]
NUM_PARAMS = 2  # [-]
# Moments solver states: mu_0..mu_3 [um**n], five mass concentrations
# [kg/m**3] and the liquid volume [m**3].
NUM_MOMENT_STATES = 10  # [-]
TARGET_COLUMN = 4  # mass concentration of A [kg/m**3], after the moments
NUM_SPECIES = 5  # [-]
MICROMETERS_PER_METER = 1e6  # [um/m], exact
# Assimulo CVode defaults, which paramest_wrapper uses without run_args
# (recorded in BatchCryst.sundials_opt after a solve).
CVODE_RTOL = 1e-6  # [-]
CVODE_ATOL = 1e-6  # [state units]
# Requested times [s], the times CVode reports for them [s] (the charge at
# 0 s is prepended, initial replicates merge, later replicates repeat), and,
# by hand, the reported row of every requested time. Reported rows of a
# replicated time are identical, so the replicate grids check only the
# returned values; which reported row each replicate maps to is checked on a
# hand-built profile in test_commons_helpers.py.
MOMENT_GRIDS = {
    'from charge': ([0.0, 10.0, 20.0, 30.0], [0.0, 10.0, 20.0, 30.0],
                    [0, 1, 2, 3]),
    'after charge': ([10.0, 20.0, 30.0], [0.0, 10.0, 20.0, 30.0],
                     [1, 2, 3]),
    'endpoint': ([30.0], [0.0, 30.0], [1]),
    'replicates': ([0.0, 0.0, 10.0, 10.0, 30.0], [0.0, 10.0, 10.0, 30.0],
                   [0, 0, 1, 2, 3]),
}
FVM_GRID = ([10.0, 10.0, 30.0], [0.0, 10.0, 10.0, 30.0], [1, 2, 3])
# Samples after the charge and the times CVode reports for them [s].
LATER_TIMES = [10.0, 20.0, 30.0]  # [s]
LATER_REPORTED = [0.0, 10.0, 20.0, 30.0]  # [s]
# Time-only terminating event between the samples at 10 s and 20 s.
EVENT_TIME = 15.0  # [s]
# Constant feed of the charged liquid and seed slurry for MSMPR and
# SemibatchCryst: a residence time VOLUME / FEED_FLOW = 100 s, so the feed
# visibly changes the vessel over the 30 s horizon.
FEED_FLOW = 0.02  # [m**3/s]
# MSMPR moments states omit the liquid volume of the batch layouts.
NUM_MSMPR_MOMENT_STATES = NUM_MOMENT_STATES - 1  # [-]


def _seed(size_grid):
    """Return the Gaussian seed number density on a size grid.

    Parameters
    ----------
    size_grid : numpy.ndarray
        Crystal sizes [um], shape ``(num_sizes,)``.

    Returns
    -------
    numpy.ndarray
        Number density [#/um], shape ``(num_sizes,)``.
    """
    return SEED_PEAK * np.exp(-((size_grid - SEED_MODE) / SEED_WIDTH)**2)


def _crystallizer(method='moments', state_events=None, param_wrapper=None,
                  unit_type=BatchCryst):
    """Build the seeded growth-only crystallizer.

    Parameters
    ----------
    method : str, optional
        ``'moments'`` (default) or ``'1D-FVM'``.
    state_events : list of dict, optional
        State-event definitions passed to the constructor.
    param_wrapper : callable, optional
        Custom estimation callback passed to the constructor.
    unit_type : type, optional
        ``BatchCryst`` (default), ``MSMPR`` or ``SemibatchCryst``; the
        continuous units receive a ``FEED_FLOW`` feed of the charged liquid
        composition and seed population per slurry volume.

    Returns
    -------
    BatchCryst, MSMPR or SemibatchCryst
        Crystallizer whose ``mask_params`` select the growth rate constant
        [um/s] and growth order [-].
    """
    size_grid = MOMENTS_SIZE_GRID if method == 'moments' else FVM_SIZE_GRID

    def temperature(time):
        """Return the prescribed constant temperature.

        Parameters
        ----------
        time : float or numpy.ndarray
            Time [s].

        Returns
        -------
        float or numpy.ndarray
            Temperature [K], shaped like ``time``.
        """
        return TEMPERATURE + np.zeros_like(time)

    mask = np.zeros(NUM_KINETIC_PARAMS, dtype=bool)
    mask[ACTIVE_COLUMNS] = True
    unit = unit_type('A', method=method, mask_params=list(mask),
                     controls={'temp': temperature},
                     state_events=state_events, param_wrapper=param_wrapper)
    liquid = LiquidPhase(DATABASE, temp=TEMPERATURE, vol=VOLUME,
                         mass_frac=LIQUID_MASS_FRACTIONS)
    solid = SolidPhase(DATABASE, temp=TEMPERATURE, x_distrib=size_grid,
                       distrib=_seed(size_grid), kv=SHAPE_FACTOR,
                       mass_frac=SOLID_MASS_FRACTIONS, num_mom=NUM_MOMENTS)
    unit.Phases = (liquid, solid)
    unit.Kinetics = CrystKinetics(coeff_solub=[SATURATION], growth=GROWTH)
    if unit_type is not BatchCryst:
        feed_density = unit.Slurry.distrib  # [#/m**3/um], charged slurry
        inlet = SlurryStream(vol_flow=FEED_FLOW, x_distrib=size_grid,
                             distrib=feed_density)
        inlet.Phases = (
            LiquidStream(DATABASE, temp=TEMPERATURE,
                         mass_frac=LIQUID_MASS_FRACTIONS, vol_flow=FEED_FLOW),
            SolidStream(DATABASE, temp=TEMPERATURE,
                        mass_frac=SOLID_MASS_FRACTIONS, x_distrib=size_grid,
                        distrib=feed_density, kv=SHAPE_FACTOR))
        unit.Inlet = inlet
    return unit


def _reset_reference(unit_type, method, requested, eval_sens):
    """Solve a fresh fed crystallizer prepared as ``paramest_wrapper`` does.

    The wrapper resets the unit before solving. For the fed units the reset
    rebuilds the slurry and residence time with roundoff-level changes
    (observed 6e-10 relative in the MSMPR residence time), which a direct
    solve of a never-reset unit does not share, so the reference is reset
    first: the same preparation, without the wrapper's row selection.

    Parameters
    ----------
    unit_type : type
        ``MSMPR`` or ``SemibatchCryst``.
    method : str
        ``'moments'`` or ``'1D-FVM'``.
    requested : numpy.ndarray
        Output times [s], shape ``(num_times,)``.
    eval_sens : bool
        Whether to integrate parameter sensitivities.

    Returns
    -------
    tuple
        The ``solve_unit`` return: reported times [s], raw states in
        ``solve_unit`` units and, if ``eval_sens``, sensitivities per
        parameter [state unit / parameter unit].
    """
    reference = _crystallizer(method, unit_type=unit_type)
    params = reference.Kinetics.concat_params()  # [native parameter units]
    reference.reset()
    reference.Kinetics.set_params(params)
    reference.elapsed_time = 0.0  # [s]
    return reference.solve_unit(time_grid=requested, eval_sens=eval_sens,
                                verbose=False)


def _report_rows(result, sens, reord_sens=True):
    """Return the reported times and sensitivity shapes a wrapper receives.

    Parameters
    ----------
    result : DynamicResult
        Complete crystallizer result; ``time`` in [s].
    sens : dict of numpy.ndarray
        Raw sensitivities per state [state unit / parameter unit].
    reord_sens : bool, optional
        Layout flag forwarded by ``paramest_wrapper``; unused.

    Returns
    -------
    tuple
        Reported times [s] and a mapping of state name to sensitivity shape.
    """
    return (np.array(result.time),
            {name: block.shape for name, block in sens.items()})


def _charged_state():
    """Return the moments solver state of the unsolved charge.

    Returns
    -------
    numpy.ndarray
        Moments mu_0..mu_3 [um**n], mass concentrations [kg/m**3] and
        liquid volume [m**3], shape ``(NUM_MOMENT_STATES,)``.
    """
    unit = _crystallizer('moments')
    orders = np.arange(NUM_MOMENTS)  # [-]
    moments = unit.Solid_1.moments * MICROMETERS_PER_METER**orders  # [um**n]
    return np.concatenate((moments, unit.Liquid_1.mass_conc, [VOLUME]))


def _flags(params):
    """Return the optimize flags of the estimated growth parameters.

    Parameters
    ----------
    params : numpy.ndarray
        Full kinetic parameter vector, shape ``(NUM_KINETIC_PARAMS,)``.

    Returns
    -------
    numpy.ndarray
        Boolean flags, True at ``ACTIVE_COLUMNS``.
    """
    flags = np.zeros(params.size, dtype=bool)
    flags[ACTIVE_COLUMNS] = True
    return flags


@pytest.mark.parametrize('grid', list(MOMENT_GRIDS))
def test_moments_states_and_sensitivities_use_requested_rows(grid):
    times, reported_times, rows = MOMENT_GRIDS[grid]
    requested = np.array(times)  # [s]
    num_times = len(times)
    unit = _crystallizer()
    params = unit.Kinetics.concat_params()  # [native parameter units]
    states, sens = unit.paramest_wrapper(params, requested)
    _, stacked = unit.paramest_wrapper(params, requested, reord_sens=False)

    reference = _crystallizer()
    raw_time, raw_states, raw_sens = reference.solve_unit(
        time_grid=requested, eval_sens=True, verbose=False)
    np.testing.assert_array_equal(raw_time, reported_times)

    assert states.shape == (num_times, NUM_MOMENT_STATES)
    np.testing.assert_array_equal(states, raw_states[rows])

    # Reordered layout, written element by element: rows state-major (each
    # state over the requested times), columns the active parameters.
    expected = np.empty((NUM_MOMENT_STATES * num_times, NUM_PARAMS))
    for state in range(NUM_MOMENT_STATES):
        for sample, row in enumerate(rows):
            for param in range(NUM_PARAMS):
                expected[state * num_times + sample, param] = \
                    raw_sens[param][row, state]
    assert sens.shape == expected.shape
    np.testing.assert_array_equal(sens, expected)

    # Unreordered layout: parameter, requested time, state.
    assert stacked.shape == (NUM_PARAMS, num_times, NUM_MOMENT_STATES)
    for param in range(NUM_PARAMS):
        np.testing.assert_array_equal(stacked[param], raw_sens[param][rows])

    # Physical checks independent of the reference solve: requested charge
    # times return the charge with zero sensitivities, later times a grown
    # population, and growth without nucleation keeps mu_0.
    charge = _charged_state()
    for sample, time in enumerate(times):
        if time == 0.0:
            np.testing.assert_array_equal(states[sample], charge)
            np.testing.assert_array_equal(stacked[:, sample], 0.0)
        else:
            assert states[sample, 1] > charge[1]  # mu_1 [um]
    np.testing.assert_allclose(states[:, 0], charge[0], rtol=CVODE_RTOL)


def test_finite_volume_states_use_requested_rows():
    times, reported_times, rows = FVM_GRID
    requested = np.array(times)  # [s]
    unit = _crystallizer('1D-FVM')
    states = unit.paramest_wrapper(unit.Kinetics.concat_params(), requested)

    reference = _crystallizer('1D-FVM')
    raw_time, raw_states = reference.solve_unit(time_grid=requested,
                                                verbose=False)
    np.testing.assert_array_equal(raw_time, reported_times)
    # Distribution [#/um] on every size bin, concentrations, volume.
    num_states = FVM_SIZE_GRID.size + NUM_SPECIES + 1
    assert states.shape == (len(times), num_states)
    np.testing.assert_array_equal(states, raw_states[rows])
    # The returned rows are integrated states, not the charged volume
    # (2 m**3), which crystal growth depletes.
    assert np.all(states[:, -1] < VOLUME)


@pytest.mark.parametrize('times', [[30.0], LATER_TIMES])
def test_self_generated_data_give_zero_objective(times):
    """Data generated by the model at its own parameters fit exactly.

    The data are the A concentrations of a reference solve at the requested
    times, rows selected by hand (the charge is reported first). Model and
    data come from the same deterministic sensitivity solve, so the
    expected objective is zero; the bound allows every residual one CVode
    tolerance unit, rtol |c| + atol, for rounding differences.
    """
    requested = np.array(times)  # [s]
    reference = _crystallizer()
    params = reference.Kinetics.concat_params()  # [native parameter units]
    _, raw_states, _ = reference.solve_unit(time_grid=requested,
                                            eval_sens=True, verbose=False)
    data = raw_states[1:, TARGET_COLUMN]  # [kg/m**3]
    assert data.shape == requested.shape

    flags = _flags(params)
    estimator = ParameterEstimation(
        _crystallizer().paramest_wrapper, params,
        {'run': Experiment({'c_A': Measurement(TARGET_COLUMN, requested,
                                               data)})},
        optimize_flags=flags)
    objective = estimator.get_objective(params[flags])  # [(kg/m**3)**2]
    residual_bound = CVODE_RTOL * np.max(np.abs(data)) + CVODE_ATOL  # [kg/m**3]
    assert 0.0 <= objective <= 0.5 * len(times) * residual_bound**2


def test_result_and_heat_duty_keep_the_charge_row():
    requested = np.array(LATER_TIMES)  # [s]
    unit = _crystallizer()
    states, _ = unit.paramest_wrapper(unit.Kinetics.concat_params(),
                                      requested)
    np.testing.assert_array_equal(unit.result.time, LATER_REPORTED)
    # Returned concentrations are the stored result rows after the charge.
    np.testing.assert_array_equal(
        states[:, TARGET_COLUMN:TARGET_COLUMN + NUM_SPECIES],
        unit.result.mass_conc[1:])

    reference = _crystallizer()
    reference.solve_unit(time_grid=np.array(LATER_REPORTED),
                         eval_sens=True, verbose=False)
    assert reference.heat_duty[1] != 0.0  # [J], first interval included
    np.testing.assert_array_equal(unit.heat_duty, reference.heat_duty)


def test_custom_wrapper_receives_every_solver_row():
    """A custom moments ``param_wrapper`` still selects its own rows."""
    unit = _crystallizer(param_wrapper=_report_rows)
    times, shapes = unit.paramest_wrapper(unit.Kinetics.concat_params(),
                                          np.array(LATER_TIMES))
    np.testing.assert_array_equal(times, LATER_REPORTED)
    assert len(shapes) == NUM_MOMENT_STATES
    assert set(shapes.values()) == {(len(LATER_REPORTED), NUM_PARAMS)}


def test_decreasing_grid_reports_absent_time():
    unit = _crystallizer()
    # CVode integrates to the last requested time, 10 s, so the earlier
    # request of 20 s is never reported.
    with pytest.raises(ValueError, match=(
            r"^Requested time 20\.0 s is absent from the crystallizer "
            r"solution, which reports 2 times from 0\.0 s to 10\.0 s")):
        unit.paramest_wrapper(unit.Kinetics.concat_params(),
                              np.array([0.0, 20.0, 10.0]))


def test_terminating_event_before_samples_reports_absent_time():
    """A state event ending the run before later samples is reported.

    Finite volumes without sensitivities: state events with the moments
    sensitivities of ``paramest_wrapper`` fail in CVode setup (issue #302),
    and the time-only callable avoids the state-name mapping of issue
    #274. The repeated event time of the event output makes the stored heat
    duty NaN with a RuntimeWarning (issue #314); it is not asserted here.
    """
    events = [{'callable': lambda time, states, derivatives:
               EVENT_TIME - time}]  # [s]
    unit = _crystallizer('1D-FVM', state_events=events)
    with pytest.raises(ValueError, match=(
            r"^Requested time 20\.0 s is absent from the crystallizer "
            r"solution, which reports \d+ times from 0\.0 s to 15\.0 s")):
        unit.paramest_wrapper(unit.Kinetics.concat_params(),
                              np.array(LATER_TIMES))


# Method, custom wrapper and charge-only grid [s]: the guard precedes the
# method and wrapper branches, so it covers every combination.
CHARGE_ONLY_CASES = {
    'moments': ('moments', None, [0.0]),
    'moments replicates': ('moments', None, [0.0, 0.0]),
    'finite volumes': ('1D-FVM', None, [0.0]),
    'custom wrapper': ('moments', _report_rows, [0.0, 0.0]),
}


@pytest.mark.parametrize('case', list(CHARGE_ONLY_CASES))
def test_charge_time_only_grid_is_rejected_before_reset(case):
    method, param_wrapper, times = CHARGE_ONLY_CASES[case]
    unit = _crystallizer(method, param_wrapper=param_wrapper)
    params = unit.Kinetics.concat_params()  # [native parameter units]
    unit.paramest_wrapper(params, np.array([10.0]))
    solved_conc = unit.Liquid_1.mass_conc.copy()  # [kg/m**3], after 10 s
    solved_result = unit.result
    with pytest.raises(PharmaPyNonImplementedError, match=(
            r"^Every requested time is the charge time \(0 s\)")):
        unit.paramest_wrapper(params, np.array(times))
    # Not reset: the phases keep the solved state, not the charge.
    assert unit.result is solved_result
    np.testing.assert_array_equal(unit.Liquid_1.mass_conc, solved_conc)
    assert solved_conc[0] < _charged_state()[TARGET_COLUMN]


@pytest.mark.parametrize('unit_type,num_states',
                         [(MSMPR, NUM_MSMPR_MOMENT_STATES),
                          (SemibatchCryst, NUM_MOMENT_STATES)])
def test_fed_crystallizer_moments_use_requested_rows(unit_type, num_states):
    requested = np.array(LATER_TIMES)  # [s]
    rows = [1, 2, 3]  # reported rows of LATER_TIMES, after the charge
    unit = _crystallizer(unit_type=unit_type)
    params = unit.Kinetics.concat_params()  # [native parameter units]
    states, sens = unit.paramest_wrapper(params, requested)
    np.testing.assert_array_equal(unit.result.time, LATER_REPORTED)

    raw_time, raw_states, raw_sens = _reset_reference(
        unit_type, 'moments', requested, eval_sens=True)
    np.testing.assert_array_equal(raw_time, LATER_REPORTED)
    num_times = len(LATER_TIMES)
    assert states.shape == (num_times, num_states)
    np.testing.assert_array_equal(states, raw_states[rows])
    expected = np.empty((num_states * num_times, NUM_PARAMS))
    for state in range(num_states):
        for sample, row in enumerate(rows):
            for param in range(NUM_PARAMS):
                expected[state * num_times + sample, param] = \
                    raw_sens[param][row, state]
    assert sens.shape == expected.shape
    np.testing.assert_array_equal(sens, expected)
    # Feed and growth change the A concentration [kg/m**3] at every sample,
    # so no returned row is the charge.
    assert np.all(states[:, TARGET_COLUMN] != raw_states[0, TARGET_COLUMN])


def test_fed_crystallizer_finite_volumes_use_requested_rows():
    requested = np.array(LATER_TIMES)  # [s]
    rows = [1, 2, 3]  # reported rows of LATER_TIMES, after the charge
    unit = _crystallizer('1D-FVM', unit_type=SemibatchCryst)
    states = unit.paramest_wrapper(unit.Kinetics.concat_params(), requested)
    np.testing.assert_array_equal(unit.result.time, LATER_REPORTED)

    raw_time, raw_states = _reset_reference(SemibatchCryst, '1D-FVM',
                                            requested, eval_sens=False)
    np.testing.assert_array_equal(raw_time, LATER_REPORTED)
    # Distribution [#/um] on every size bin, concentrations, volume.
    num_states = FVM_SIZE_GRID.size + NUM_SPECIES + 1
    assert states.shape == (len(LATER_TIMES), num_states)
    np.testing.assert_array_equal(states, raw_states[rows])
    # The feed raises the liquid volume [m**3] above the charge.
    assert np.all(states[:, -1] > VOLUME)
