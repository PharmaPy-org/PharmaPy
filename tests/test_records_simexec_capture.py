"""Capturing a ``SimulationExec`` fit of ``Experiment`` objects (#351).

Scope: ``CalibrationRecord.from_estimation`` accepts an estimator built by
``SimulationExec.SetParamEstimation`` from ``Experiment`` objects, which
rebuilds each experiment with the wrapper keywords ``modify_phase``,
``modify_controls`` and ``run_args`` added. The record stores those
effective keywords, so it reruns both directly and through
``SimulationExec`` without modifier arguments; experiments that were not
the ones fitted and unrecordable modifier values are rejected.

Boundary contract object: ``ClosedFormDecayUnit`` stands in for a reactor.
Real reactors integrate with CVode from the optional Assimulo stack, which
the core lane must not require. The stand-in has only what
``SetParamEstimation`` reads from a unit (``params``, ``param_seed`` and a
bound ``paramest_wrapper`` whose signature binds experiment arguments); it
evaluates the closed-form first-order decay of the user guide, taking the
initial temperature [K] from its ``modify_phase`` keyword. It verifies the
handoffs SetParamEstimation -> estimator -> from_estimation and back
(record -> ``rerun`` or ``SetParamEstimation``), not reactor physics.
The real collaborator, a CVode ``BatchReactor`` fitted to the Ziegler
workshop data, is captured and rerun through both paths in the Assimulo
lane by ``tests/test_records_ziegler_case.py``. Core lane; milliseconds.
"""

from pathlib import Path

import numpy as np
import pytest

from PharmaPy.ParamEstim import Experiment, Measurement
from PharmaPy.Records import CalibrationRecord, DataSource, ModelRevision
from PharmaPy.SimExec import SimulationExec

pytestmark = pytest.mark.unit

PURE_PATH = str(Path(__file__).parent
                / 'integration/data/pfr_test_pure_comp.json')
SEED_RATE = 0.3  # [1/s], LM seed of k
# Production and closed-form predictions evaluate the same IEEE expressions
# in a different association order; a few ulps of O(300 K) values bound the
# difference.
ROUNDOFF_RTOL = 1e-12  # [-]
ROUNDOFF_ATOL = 1e-12  # [mol/L or K]
# The user guide's synthetic data (see test_records_calibration.py):
# experiment -> (initial c_A [mol/L], rise [K*L/mol], initial T [K]).
CALLBACK = {'hot': (2.0, 20.0, 300.0), 'cold': (1.0, 10.0, 290.0)}
GRID = {'hot': [0.0, 1.0, 2.0, 4.0], 'cold': [0.5, 1.5, 3.0]}  # [s]
SOURCES = (DataSource('guide', 'synthetic',
                      'synthetic values written for the user guide; no '
                      'external data'),)
# Both reruns replay bit-identical inputs through the same deterministic LM
# iterations; four ulps of relative slack absorb operation ordering.
REPLAY_RTOL = 4 * np.finfo(float).eps  # [-]
REPLAY_ATOL = 0.0  # [1/s]
WARMER_HOT_START = 305.0  # [K], hot initial temperature of the second fit


class ClosedFormDecayUnit:
    """Closed-form stand-in for a reactor in ``SetParamEstimation``.

    See the module docstring for why it is needed and which lane exercises
    the real collaborator.
    """

    def __init__(self) -> None:
        """Store the parameter seed read by ``SetParamEstimation``."""
        self.params = np.array([SEED_RATE])  # [1/s]
        self.param_seed = self.params  # [1/s]

    def paramest_wrapper(self, params, t_vals, initial_mol_l,
                         rise_k_l_mol=0.0, modify_phase=None,
                         modify_controls=None, run_args=None):
        """Evaluate c_A and T of first-order A -> B for estimation.

        Parameters
        ----------
        params : numpy.ndarray
            Rate constant k, shape (1,) [1/s].
        t_vals : numpy.ndarray
            Model grid, shape (n_times,) [s].
        initial_mol_l : float
            Initial concentration of A [mol/L].
        rise_k_l_mol : float, optional
            Adiabatic temperature rise per converted concentration
            [K*L/mol].
        modify_phase : dict
            Phase modifier with the initial temperature ``'temp0_k'`` [K].
        modify_controls, run_args : dict, optional
            Wrapper keywords added by ``SetParamEstimation``; unused.

        Returns
        -------
        numpy.ndarray
            Shape (n_times, 2); c_A [mol/L], then T [K].
        """
        converted = initial_mol_l * (1 - np.exp(-params[0] * t_vals))
        # [mol/L] above; [K] below.
        temperature = modify_phase['temp0_k'] + rise_k_l_mol * converted
        return np.column_stack((initial_mol_l - converted, temperature))


def make_experiments(cold_initial=CALLBACK['cold'][0]):
    """Build the guide's experiments without the temperature keyword.

    Parameters
    ----------
    cold_initial : float, optional
        Initial c_A of the cold run [mol/L].

    Returns
    -------
    dict
        Experiment name to ``Experiment``.
    """
    return {
        'hot': Experiment(
            {'c_A': Measurement(0, [0.0, 1.0, 2.0, 4.0],
                                [2.010, 1.200, 0.750, 0.265],
                                uncertainty=0.02, units='mol/L'),
             'T': Measurement(1, 2.0, 325.6, uncertainty=0.5, units='K')},
            args=(CALLBACK['hot'][0],),
            kwargs={'rise_k_l_mol': CALLBACK['hot'][1]}),
        'cold': Experiment(
            {'c_A': Measurement(0, [0.5, 1.5, 3.0], [0.785, np.nan, 0.220],
                                uncertainty=[0.01, 0.01, 0.02],
                                units='mol/L')},
            args=(cold_initial,),
            kwargs={'rise_k_l_mol': CALLBACK['cold'][1]})}


def initial_temperatures(hot=CALLBACK['hot'][2]):
    """Return per-experiment phase modifiers with the initial temperature.

    Parameters
    ----------
    hot : float, optional
        Initial temperature of the hot run [K].

    Returns
    -------
    dict
        Experiment name to ``{'temp0_k': T0}`` [K].
    """
    return {'hot': {'temp0_k': hot},
            'cold': {'temp0_k': CALLBACK['cold'][2]}}


def fitted_simulation(experiments, **modifiers):
    """Fit the experiments through ``SimulationExec``.

    Parameters
    ----------
    experiments : dict
        Experiment name to ``Experiment``.
    **modifiers
        ``SetParamEstimation`` modifier arguments, e.g. ``phase_modifiers``.

    Returns
    -------
    SimulationExec
        Simulation whose ``ParamInst`` has been fitted with LM.
    """
    simulation = SimulationExec(PURE_PATH, {'R01': []})
    simulation.R01 = ClosedFormDecayUnit()
    simulation.SetParamEstimation(experiments, name_params=['k'],
                                  **modifiers)
    simulation.EstimateParams(verbose=False)
    return simulation


def capture(simulation, experiments):
    """Capture a fitted simulation.

    Parameters
    ----------
    simulation : SimulationExec
        Fitted simulation.
    experiments : dict
        The experiments given to ``SetParamEstimation``.

    Returns
    -------
    CalibrationRecord
        Captured record.
    """
    return CalibrationRecord.from_estimation(
        simulation.ParamInst, experiments, case='decay-simexec',
        parameter_units=['1/s'], model=ModelRevision('R01.paramest_wrapper'),
        sources=SOURCES)


def test_simexec_capture_records_effective_wrapper_keywords():
    experiments = make_experiments()
    simulation = fitted_simulation(experiments,
                                   phase_modifiers=initial_temperatures())
    estimator = simulation.ParamInst
    record = capture(simulation, experiments)

    rate = record.fitted_values[0]  # [1/s]
    assert rate == estimator.params_convg[0]
    for name in ('hot', 'cold'):
        recorded = record.experiments[name]
        # The caller's keywords first, then the wrapper keywords used.
        assert list(recorded.kwargs) == ['rise_k_l_mol', 'modify_phase',
                                         'modify_controls', 'run_args']
        assert dict(recorded.kwargs) == {
            'rise_k_l_mol': CALLBACK[name][1],
            'modify_phase': {'temp0_k': CALLBACK[name][2]},
            'modify_controls': {}, 'run_args': {}}
        assert recorded.args == (CALLBACK[name][0],)
    # The caller's experiments are unchanged.
    assert dict(experiments['cold'].kwargs) == {
        'rise_k_l_mol': CALLBACK['cold'][1]}
    text = record.to_json()
    assert '"temp0_k": 290.0' in text and '"temp0_k": 300.0' in text

    assert [(field.experiment, field.identity.kind, field.name)
            for field in record.fields] == [
        ('hot', 'measurement_prediction', 'c_A'),
        ('hot', 'measurement_prediction', 'T'),
        ('cold', 'measurement_prediction', 'c_A'),
        ('cold', 'measurement_prediction', 'T')]
    for field in record.fields:
        assert field.x_units == 's'
        np.testing.assert_array_equal(field.x, GRID[field.experiment])
        initial, rise, temp0 = CALLBACK[field.experiment]
        remaining = initial * np.exp(
            -rate * np.array(GRID[field.experiment]))  # [mol/L]
        expected = (remaining if field.name == 'c_A'
                    else temp0 + rise * (initial - remaining))  # [K]
        np.testing.assert_allclose(field.values, expected,
                                   rtol=ROUNDOFF_RTOL, atol=ROUNDOFF_ATOL)


def test_simexec_record_reruns_directly_and_through_simexec():
    experiments = make_experiments()
    simulation = fitted_simulation(experiments,
                                   phase_modifiers=initial_temperatures())
    loaded = CalibrationRecord.from_json(
        capture(simulation, experiments).to_json())

    # (a) Directly: ParameterEstimation passes the recorded wrapper
    # keywords to the unit's callback.
    direct = loaded.rerun(ClosedFormDecayUnit().paramest_wrapper,
                          rtol=REPLAY_RTOL, atol=REPLAY_ATOL)
    np.testing.assert_allclose(direct.params_convg,
                               simulation.ParamInst.params_convg,
                               rtol=REPLAY_RTOL, atol=REPLAY_ATOL)

    # (b) Through SimulationExec without modifier arguments: the recorded
    # experiments already carry them, and SetParamEstimation keeps keywords
    # an experiment defines when the matching argument is omitted.
    replay = SimulationExec(PURE_PATH, {'R01': []})
    replay.R01 = ClosedFormDecayUnit()
    inputs = loaded.estimation_inputs()
    replay.SetParamEstimation(inputs['x_data'],
                              name_params=inputs['name_params'],
                              optimize_flags=inputs['optimize_flags'])
    replay.EstimateParams(verbose=False, **loaded.optimizer_inputs())
    loaded.check_rerun(replay.ParamInst, rtol=REPLAY_RTOL, atol=REPLAY_ATOL)


def test_different_modifiers_give_different_records():
    experiments = make_experiments()
    reference = capture(
        fitted_simulation(experiments,
                          phase_modifiers=initial_temperatures()),
        experiments)
    warmer = capture(
        fitted_simulation(experiments, phase_modifiers=initial_temperatures(
            hot=WARMER_HOT_START)),
        experiments)
    assert reference != warmer
    assert (warmer.experiments['hot'].kwargs['modify_phase']
            == {'temp0_k': WARMER_HOT_START})
    # A warmer start makes the hot temperature observation imply a
    # different conversion at 2 s, hence a different rate.
    assert warmer.fitted_values[0] != reference.fitted_values[0]


def test_unrecordable_wrapper_values_are_named():
    experiments = make_experiments()
    simulation = fitted_simulation(
        experiments, phase_modifiers=initial_temperatures(),
        control_modifiers={'hot': {'temp': len}, 'cold': {}})
    with pytest.raises(TypeError, match="Experiment 'hot': SimulationExec "
                                        "wrapper keyword 'modify_controls' "
                                        "holds a value that a record "
                                        "cannot hold"):
        capture(simulation, experiments)


def test_simexec_capture_rejects_experiments_that_were_not_fitted():
    simulation = fitted_simulation(make_experiments(),
                                   phase_modifiers=initial_temperatures())
    with pytest.raises(ValueError, match=r"Experiment 'cold': the "
                                         r"estimator's positional callback "
                                         r"arguments \(1\.0,\) differ"):
        capture(simulation, make_experiments(cold_initial=1.5))
