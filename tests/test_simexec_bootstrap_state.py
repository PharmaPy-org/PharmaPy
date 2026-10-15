"""Bootstrap refits through ``SimulationExec`` keep the accepted fit (#262).

A real ``BatchReactor`` (CVode, finite-difference sensitivities) is fitted
through ``SimulationExec.EstimateParams``; ``StatisticsClass`` then refits
residual-bootstrap datasets. The estimator's observations and fitted outputs
must be unchanged afterwards, the shared reactor must be back at the
accepted rate without a user re-evaluation (the #252 refresh boundary),
and re-evaluating must reproduce the pre-bootstrap objective. Three
bootstrap fits of a few reactor solves each; requires the optional
Assimulo stack.
"""

import copy
from pathlib import Path

import numpy as np
import pytest

pytestmark = [pytest.mark.assimulo, pytest.mark.integration]
pytest.importorskip('assimulo')

from PharmaPy.Kinetics import RxnKinetics
from PharmaPy.Phases import LiquidPhase
from PharmaPy.Reactors import BatchReactor
from PharmaPy.SimExec import SimulationExec
from PharmaPy.StatsModule import StatisticsClass
from PharmaPy.Utilities import CoolingWater


THERMO_PATH = str(
    Path(__file__).parent / 'integration/data/pfr_test_pure_comp.json')
TEMPERATURE = 298.15  # [K], isothermal at the kinetics reference temperature
VOLUME = 0.002  # [m**3], concentrations are independent of volume
# [mol/L]: dilute reactant A (0.1), no product B or third species, and the
# solvent near 5 mol/L, the shipped test system of
# test_paramestim_accepted_reactor.py; only A is measured and fitted.
INITIAL_CONCENTRATIONS = np.array([0.1, 0.0, 0.0, 5.0])
# Zero activation energy [J/mol] with temp_ref = TEMPERATURE makes the rate
# constant exactly the fitted k_params value, isolating first-order decay.
ACTIVATION_ENERGY = 0.0  # [J/mol]
TRUE_RATE = 0.2  # [1/s], synthetic five-second reaction time constant
SEED_RATE = 0.1  # [1/s], half the true rate so the fit moves
UTILITY_FLOW = 0.1  # [kg/s], unused by the isothermal balance
TIME_S = np.array([0.0, 1.0, 2.0, 4.0, 8.0])  # [s]
# Asymmetric deterministic errors make the bootstrap datasets differ.
ERRORS_MOL_L = np.array([0.0, 0.0015, -0.001, 0.0005, -0.002])  # [mol/L]
# CVode tolerances of test_paramestim_accepted_reactor.py: integration
# error near 1e-10 mol/L, five orders below the 1e-3 mol/L synthetic errors
# that distinguish the bootstrap datasets, so refits move by resampling,
# not solver noise. The bit-identity checks rely on CVode being
# deterministic for identical inputs, not on these values.
SOLVER_RTOL = 1.0e-9  # [-]
SOLVER_ATOL = 1.0e-11  # [mol/L]
NUM_SAMPLES = 3  # [-], bootstrap refits
BOOTSTRAP_SEED = 262  # [-], arbitrary fixed seed; any value works
# Estimator outputs that a bootstrap refit would rebind.
FITTED_ATTRIBUTES = ('y_data', 'params_convg', 'info_opt', 'covar_params',
                     'resid_runs', 'y_runs', 'y_model', 'residuals',
                     'weighted_residuals', 'params_residuals', 'sens',
                     'params_iter', 'objfun_iter', 'optim_options')


def _assert_same(expected, actual, path):
    """Assert recursive, bit-identical equality of fitted outputs.

    Parameters
    ----------
    expected, actual : object
        Arrays, containers or scalars; arrays compare exactly.
    path : str
        Location reported on failure.
    """
    if isinstance(expected, dict):
        assert list(expected) == list(actual), path
        for key in expected:
            _assert_same(expected[key], actual[key], f'{path}[{key!r}]')
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected), path
        for index, (item, other) in enumerate(zip(expected, actual)):
            _assert_same(item, other, f'{path}[{index}]')
    elif isinstance(expected, np.ndarray):
        np.testing.assert_array_equal(actual, expected, err_msg=path)
    else:
        assert actual == expected, path


def test_simexec_bootstrap_keeps_accepted_fit():
    reactor = BatchReactor(isothermal=True, return_sens=False,
                           mask_params=[True, False])
    reactor.Phases = LiquidPhase(THERMO_PATH, temp=TEMPERATURE, vol=VOLUME,
                                 mole_conc=INITIAL_CONCENTRATIONS)
    reactor.Kinetics = RxnKinetics(
        THERMO_PATH, k_params=[SEED_RATE], ea_params=[ACTIVATION_ENERGY],
        rxn_list=['A --> B'], temp_ref=TEMPERATURE, reformulate_kin=False)
    reactor.Utility = CoolingWater(temp_in=TEMPERATURE,
                                   mass_flow=UTILITY_FLOW)
    simulation = SimulationExec(THERMO_PATH, flowsheet={'R01': []})
    simulation.R01 = reactor
    observations = (INITIAL_CONCENTRATIONS[0] * np.exp(-TRUE_RATE * TIME_S)
                    + ERRORS_MOL_L)  # [mol/L]
    simulation.SetParamEstimation(
        TIME_S, observations, measured_ind=[0], optimize_flags=[True, False],
        name_params=['rate'], wrapper_kwargs={
            'sundials_opts': {'rtol': SOLVER_RTOL, 'atol': SOLVER_ATOL}})
    accepted, _, _ = simulation.EstimateParams(verbose=False)  # [1/s]
    estimator = simulation.ParamInst
    statistics = StatisticsClass(estimator)
    objective = estimator.get_objective(accepted)  # [(mol/L)**2]
    before = {name: copy.deepcopy(getattr(estimator, name))
              for name in FITTED_ATTRIBUTES}

    previous_state = np.random.get_state()
    try:
        np.random.seed(BOOTSTRAP_SEED)
        boot_params = statistics.bootstrap_params(NUM_SAMPLES)  # [1/s]
    finally:
        np.random.set_state(previous_state)

    assert boot_params.shape == (NUM_SAMPLES, 1)
    assert np.all(np.isfinite(boot_params))
    # The datasets differ from the observations, so the refits move.
    assert np.all(boot_params[:, 0] != accepted[0])
    for name in FITTED_ATTRIBUTES:
        _assert_same(before[name], getattr(estimator, name), name)
    # bootstrap_params itself returned the shared reactor to the accepted
    # rate (#252 boundary); no user re-evaluation is needed.
    assert reactor.Kinetics.concat_params()[0] == accepted[0]
    # The deterministic CVode solve reproduces the pre-bootstrap objective.
    assert estimator.get_objective(accepted) == objective
