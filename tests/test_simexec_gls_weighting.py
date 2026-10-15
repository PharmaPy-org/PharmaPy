"""Correlated GLS weights through ``SimulationExec.SetParamEstimation``.

Issue #237 public-caller regression: a correlated measurement-error
covariance passed to ``SetParamEstimation`` must reach the estimator and
weight residuals by its inverse in measured-state order. A real
``BatchReactor`` integrates an isothermal first-order A -> B reaction with
CVode; its analytic solution supplies the independent residuals. One
reactor solve per test; requires the optional Assimulo stack.
"""

from pathlib import Path

import numpy as np
import pytest
from scipy.linalg import ldl

pytestmark = [pytest.mark.assimulo, pytest.mark.integration]
pytest.importorskip('assimulo')

from PharmaPy.Kinetics import RxnKinetics
from PharmaPy.Phases import LiquidPhase
from PharmaPy.Reactors import BatchReactor
from PharmaPy.SimExec import SimulationExec
from PharmaPy.Utilities import CoolingWater


THERMO_PATH = str(
    Path(__file__).parent / 'integration/data/pfr_test_pure_comp.json')
TEMPERATURE = 300.0  # [K], isothermal; zero activation energy below
VOLUME = 0.002  # [m**3], concentrations are independent of volume
# [mol/L]: A, B, an absent species, and the solvent.
INITIAL_CONCENTRATIONS = np.array([0.1, 0.2, 0.0, 5.0])
TRUE_RATE = 0.2  # [1/s], synthetic five-second reaction time constant
TRIAL_RATE = 0.15  # [1/s], away from TRUE_RATE so residuals are O(1e-2)
UTILITY_FLOW = 0.1  # [kg/s], unused by the isothermal balance
TIME_S = np.array([0.0, 1.0, 2.5, 4.0, 7.0])  # [s], unequal spacing
# Fixed asymmetric measurement errors, columns (A, B).
MEASUREMENT_ERRORS_MOL_L = 1e-3 * np.array([  # [mol/L]
    [1.0, -0.5],
    [-1.5, 2.0],
    [0.5, -1.0],
    [2.0, 0.5],
    [-1.0, 1.5],
])
# Power-of-two variance scale 2**-14 [(mol/L)**2] keeps the covariance and
# precision literals exact inverses (det of the integer cores is 1); the
# standard deviations are about 0.017 and 0.008 mol/L.
VARIANCE_SCALE = 2.0**-14  # [(mol/L)**2]
COVARIANCE = np.array(
    [[5.0, -2.0], [-2.0, 1.0]]) * VARIANCE_SCALE  # [(mol/L)**2]
PRECISION = np.array([[1.0, 2.0], [2.0, 5.0]]) / VARIANCE_SCALE  # [(L/mol)**2]
# Tight CVode tolerances; the profile budget below is 100-fold the
# integration error, as in test_paramestim_accepted_reactor.py.
SOLVER_RTOL = 1.0e-9  # [-]
SOLVER_ATOL = 1.0e-11  # [mol/L]
PROFILE_RTOL = 1.0e-7  # [-]
PROFILE_ATOL = 1.0e-9  # [mol/L]


def _analytic_profiles(rate):
    """Return analytic A and B concentrations of the batch reaction.

    Parameters
    ----------
    rate : float
        First-order A -> B rate constant [1/s].

    Returns
    -------
    numpy.ndarray
        Shape ``(len(TIME_S), 2)``, columns A and B [mol/L].
    """
    reactant = INITIAL_CONCENTRATIONS[0] * np.exp(-rate * TIME_S)  # [mol/L]
    product = (INITIAL_CONCENTRATIONS[1] + INITIAL_CONCENTRATIONS[0]
               - reactant)  # [mol/L]
    return np.column_stack((reactant, product))


def _batch_simulation():
    """Build a simulation around an isothermal first-order batch reactor.

    Returns
    -------
    SimulationExec
        Flowsheet whose only unit, ``R01``, is the reactor seeded at
        ``TRUE_RATE``.
    """
    reactor = BatchReactor(isothermal=True, mask_params=[True, False],
                           return_sens=False)
    reactor.Kinetics = RxnKinetics(
        THERMO_PATH, k_params=[TRUE_RATE], ea_params=[0.0],
        rxn_list=['A --> B'])  # [1/s], [J/mol]
    reactor.Phases = LiquidPhase(THERMO_PATH, temp=TEMPERATURE, vol=VOLUME,
                                 mole_conc=INITIAL_CONCENTRATIONS)
    reactor.Utility = CoolingWater(temp_in=TEMPERATURE,
                                   mass_flow=UTILITY_FLOW)
    simulation = SimulationExec(THERMO_PATH, {'R01': []})
    simulation.R01 = reactor
    return simulation


def test_set_param_estimation_weights_residuals_by_correlated_precision():
    assert not np.array_equal(ldl(PRECISION)[2], [0, 1])
    observations = (_analytic_profiles(TRUE_RATE)
                    + MEASUREMENT_ERRORS_MOL_L)  # [mol/L]
    simulation = _batch_simulation()
    simulation.SetParamEstimation(
        TIME_S, observations, measured_ind=[0, 1],
        optimize_flags=[True, False], weight_matrix=COVARIANCE,
        wrapper_kwargs={'sundials_opts': {'rtol': SOLVER_RTOL,
                                          'atol': SOLVER_ATOL}})
    estimator = simulation.ParamInst

    # Handoff: the stored root reproduces the precision in (A, B) order.
    # cond(COVARIANCE) is about 34, so roundoff stays near 1e-14 relative.
    root_rtol = 1e-11  # [-]
    np.testing.assert_allclose(estimator.sigma_inv @ estimator.sigma_inv.T,
                               PRECISION, rtol=root_rtol, atol=0.0)

    objective = estimator.get_objective([TRIAL_RATE])  # [-]

    trial_profiles = _analytic_profiles(TRIAL_RATE)  # [mol/L]
    residuals = trial_profiles - observations  # [mol/L]
    # [-]: 1/2 sum_k r_k^T P r_k from the precision literal.
    expected_objective = 0.5 * np.einsum('ki,ij,kj->', residuals, PRECISION,
                                         residuals)
    # Integration error |delta_k| <= PROFILE_RTOL |c_k| + PROFILE_ATOL
    # perturbs the quadratic form by at most
    # sum_k |r_k|^T |P| delta_k + 1/2 delta_k^T |P| delta_k.
    profile_error = (PROFILE_RTOL * np.abs(trial_profiles)
                     + PROFILE_ATOL)  # [mol/L]
    abs_precision = np.abs(PRECISION)  # [(L/mol)**2]
    objective_atol = (
        np.einsum('ki,ij,kj->', np.abs(residuals), abs_precision,
                  profile_error)
        + 0.5 * np.einsum('ki,ij,kj->', profile_error, abs_precision,
                          profile_error))  # [-]
    assert objective == pytest.approx(expected_objective, rel=0.0,
                                      abs=objective_atol)
    np.testing.assert_allclose(estimator.y_runs[0], trial_profiles,
                               rtol=PROFILE_RTOL, atol=PROFILE_ATOL)
