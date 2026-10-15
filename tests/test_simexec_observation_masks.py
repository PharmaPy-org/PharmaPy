"""Staggered observation grids through ``SimulationExec`` and CVode.

Issue #236 public-caller regression: ``SetParamEstimation`` with per-state
sampling times builds masked estimators whose finite-difference reactor
sensitivities and correlated-error weighting must cover observed entries
only. A real ``BatchReactor`` integrates an isothermal first-order A -> B
reaction whose analytic solution supplies the expected values. Two reactor
solves per sensitivity evaluation; requires the optional Assimulo stack.
"""

from pathlib import Path

import numpy as np
import pytest

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
# Per-state sampling times [s]: A and B share t = 1 and 4 only, so the union
# grid (0, 1, 2.5, 4, 7) has A-only, B-only and fully observed rows.
TIMES_A_S = np.array([0.0, 1.0, 2.5, 4.0])
TIMES_B_S = np.array([1.0, 4.0, 7.0])
ERRORS_A_MOL_L = 1e-3 * np.array([1.0, -1.5, 0.5, 2.0])  # [mol/L]
ERRORS_B_MOL_L = 1e-3 * np.array([-1.0, 1.5, 0.5])  # [mol/L]
# Power-of-two scale 2**-14 [(mol/L)**2]; correlated (A, B) covariance with
# standard deviations about 0.017 and 0.008 mol/L.
VARIANCE_SCALE = 2.0**-14  # [(mol/L)**2]
COVARIANCE = np.array(
    [[5.0, -2.0], [-2.0, 1.0]]) * VARIANCE_SCALE  # [(mol/L)**2]
# CVode tolerances and the 100-fold profile budget of
# test_paramestim_accepted_reactor.py.
SOLVER_RTOL = 1.0e-9  # [-]
SOLVER_ATOL = 1.0e-11  # [mol/L]
PROFILE_RTOL = 1.0e-7  # [-]
PROFILE_ATOL = 1.0e-9  # [mol/L]
# ParameterEstimation's default finite-difference step is
# |k| * sqrt(1e-6) (PharmaPy.jac_module.dx_jac_p with rel_tol 1e-6).
FD_RELATIVE_STEP = 1e-3  # [-]


def _profiles(rate, time_s):
    """Return analytic A and B concentrations.

    Parameters
    ----------
    rate : float
        First-order A -> B rate constant [1/s].
    time_s : numpy.ndarray
        Times, shape ``(n_times,)`` [s].

    Returns
    -------
    numpy.ndarray
        Shape ``(n_times, 2)``, columns A and B [mol/L].
    """
    reactant = INITIAL_CONCENTRATIONS[0] * np.exp(-rate * time_s)  # [mol/L]
    return np.column_stack(
        (reactant,
         INITIAL_CONCENTRATIONS[1] + INITIAL_CONCENTRATIONS[0] - reactant))


def _estimator(weight_matrix):
    """Set up a staggered-grid estimation on a real batch reactor.

    Parameters
    ----------
    weight_matrix : numpy.ndarray or None
        Measurement covariance of (A, B) [(mol/L)**2].

    Returns
    -------
    ParameterEstimation
        ``SimulationExec.ParamInst`` with one experiment named ``run``.
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
    observations = [_profiles(TRUE_RATE, TIMES_A_S)[:, 0] + ERRORS_A_MOL_L,
                    _profiles(TRUE_RATE, TIMES_B_S)[:, 1] + ERRORS_B_MOL_L]
    simulation.SetParamEstimation(
        {'run': [TIMES_A_S, TIMES_B_S]}, {'run': observations},
        measured_ind=[0, 1], optimize_flags=[True, False],
        weight_matrix=weight_matrix,
        wrapper_kwargs={'sundials_opts': {'rtol': SOLVER_RTOL,
                                          'atol': SOLVER_ATOL}})
    return simulation.ParamInst


def test_reactor_sensitivities_vanish_at_unobserved_entries():
    estimator = _estimator(None)
    union_s = np.union1d(TIMES_A_S, TIMES_B_S)  # [s]
    np.testing.assert_array_equal(estimator.x_model[0], union_s)
    observed = np.column_stack((np.isin(union_s, TIMES_A_S),
                                np.isin(union_s, TIMES_B_S)))
    np.testing.assert_array_equal(estimator.x_masks[0], observed)

    # LM evaluates residuals before each Jacobian; do the same.
    estimator.get_objective([TRIAL_RATE])
    # [mol/L*s] with identity weights
    jacobian = estimator.get_gradient([TRIAL_RATE], out_array=True)

    observed_flat = observed.T.ravel()  # state-major, matching residuals
    np.testing.assert_array_equal(jacobian[:, ~observed_flat], 0.0)
    # d(A)/dk = -t A and d(B)/dk = +t A for the analytic solution [mol/L*s].
    reactant = _profiles(TRIAL_RATE, union_s)[:, 0]  # [mol/L]
    expected = np.concatenate(
        (-union_s * reactant, union_s * reactant))  # [mol/L*s]
    # Forward-difference truncation is (dx/2)|y''/y'| = dx t / 2 relative
    # with dx = FD_RELATIVE_STEP * k; twice that bounds it. CVode error
    # 1e-10 mol/L over dx (1.5e-4 1/s) adds below 1e-6 mol/L*s.
    truncation_rtol = (FD_RELATIVE_STEP * TRIAL_RATE * union_s.max())  # [-]
    fd_noise_atol = 1e-5  # [mol/L*s]
    np.testing.assert_allclose(jacobian[0, observed_flat],
                               expected[observed_flat],
                               rtol=truncation_rtol, atol=fd_noise_atol)


def test_reactor_objective_uses_marginal_precision_of_observed_states():
    estimator = _estimator(COVARIANCE)
    union_s = np.union1d(TIMES_A_S, TIMES_B_S)  # [s]

    objective = estimator.get_objective([TRIAL_RATE])  # [-]

    trial = _profiles(TRIAL_RATE, union_s)  # [mol/L]
    expected = 0.0  # [-]
    budget = 0.0  # [-], integration-error bound on the objective
    for row, time in enumerate(union_s):
        states = [state for state, times in enumerate((TIMES_A_S, TIMES_B_S))
                  if time in times]
        data = np.array([
            (_profiles(TRUE_RATE, np.array([time]))[0, state]
             + (ERRORS_A_MOL_L, ERRORS_B_MOL_L)[state][
                 np.flatnonzero((TIMES_A_S, TIMES_B_S)[state] == time)[0]])
            for state in states])  # [mol/L]
        residual = trial[row, states] - data  # [mol/L]
        precision = np.linalg.inv(
            COVARIANCE[np.ix_(states, states)])  # [(L/mol)**2]
        expected += 0.5 * residual @ precision @ residual
        error = (PROFILE_RTOL * np.abs(trial[row, states])
                 + PROFILE_ATOL)  # [mol/L]
        budget += (np.abs(residual) @ np.abs(precision) @ error
                   + 0.5 * error @ np.abs(precision) @ error)
    # Pre-#236 weighting used inv(COVARIANCE)[o, o] on the single-state
    # rows, five times the marginal precision of A here.
    assert objective == pytest.approx(expected, rel=0.0, abs=budget)
