"""Accepted-point reporting through real SimExec, BatchReactor and CVode.

The synthetic first-order A -> B case uses shipped properties and fixed zero
activation energy. The zero-trial case exposes the last finite-difference
perturbation left in the reactor; the full solve checks normal fit reporting.
"""

from pathlib import Path

import numpy as np
import pytest

pytestmark = [pytest.mark.assimulo, pytest.mark.integration]
pytest.importorskip("assimulo")

from PharmaPy.Kinetics import RxnKinetics
from PharmaPy.Phases import LiquidPhase
from PharmaPy.Reactors import BatchReactor
from PharmaPy.SimExec import SimulationExec
from PharmaPy.Utilities import CoolingWater


THERMO_PATH = str(Path(__file__).parent / "integration/data/pfr_test_pure_comp.json")
TEMPERATURE = 298.15  # [K], isothermal case at the kinetics reference temperature
VOLUME = 0.002  # [m**3], shipped reactor volume; concentration decay is independent
INITIAL_CONCENTRATIONS = np.array([0.1, 0.0, 0.0, 5.0])  # [mol/L], A and solvent
TRUE_RATE = 0.2  # [1/s], synthetic five-second reaction time constant
SEED_RATE = 0.1  # [1/s], half the true rate to require fitting
ACTIVATION_ENERGY = 0.0  # [J/mol], isolates isothermal first-order decay
UTILITY_FLOW = 0.1  # [kg/s], unused in isothermal balance; completes unit setup
# Tight solver tolerances keep integration error below the reporting comparison.
SOLVER_RTOL = 1.0e-9  # [-]
SOLVER_ATOL = 1.0e-11  # [mol/L] for concentrations; [K] for fixed temperature
PROFILE_RTOL = 1.0e-7  # [-], 100-fold integration-error budget
PROFILE_ATOL = 1.0e-9  # [mol/L], includes nearly zero product concentration
PARAMETER_ATOL = 64 * np.finfo(float).eps  # [1/s], only arithmetic roundoff


@pytest.mark.parametrize("trial_budget", [0, 100])
def test_simexec_restores_reactor_and_reports_accepted_rate(trial_budget):
    reactor = BatchReactor(isothermal=True, return_sens=False,
                           mask_params=[True, False])
    reactor.Phases = LiquidPhase(THERMO_PATH, temp=TEMPERATURE, vol=VOLUME,
                                mole_conc=INITIAL_CONCENTRATIONS)
    reactor.Kinetics = RxnKinetics(
        THERMO_PATH, k_params=[SEED_RATE], ea_params=[ACTIVATION_ENERGY],
        rxn_list=["A --> B"], temp_ref=TEMPERATURE, reformulate_kin=False)
    reactor.Utility = CoolingWater(temp_in=TEMPERATURE, mass_flow=UTILITY_FLOW)
    simulation = SimulationExec(THERMO_PATH, flowsheet={"R01": []})
    simulation.R01 = reactor
    time_s = np.array([0.0, 1.0, 2.0, 4.0, 8.0])  # [s], spans the time constant
    # Small deterministic measurement perturbations retain nonzero variance.
    observations = (INITIAL_CONCENTRATIONS[0] * np.exp(-TRUE_RATE * time_s)
                    + np.array([0.0, 0.001, -0.001, 0.001, -0.001]))  # [mol/L]
    simulation.SetParamEstimation(
        time_s, observations, measured_ind=[0], optimize_flags=[True, False],
        name_params=["rate"], wrapper_kwargs={
            "sundials_opts": {"rtol": SOLVER_RTOL, "atol": SOLVER_ATOL}})
    accepted, _, info = simulation.EstimateParams(
        optim_options={"max_fun_eval": trial_budget}, verbose=False)
    estimator = simulation.ParamInst
    # The mutable reactor must finish at the accepted rate, not the most
    # recent finite-difference perturbation, even with no optimization trials.
    np.testing.assert_allclose(reactor.Kinetics.concat_params()[0], accepted[0],
                               rtol=0.0, atol=PARAMETER_ATOL)
    # Fixed activation energy [J/mol] is handed back unchanged. Kinetics may
    # normalize a zero seed internally; compare its stored input exactly.
    np.testing.assert_array_equal(reactor.Kinetics.concat_params()[1:],
                                  estimator.param_seed[1:])
    expected_a = INITIAL_CONCENTRATIONS[0] * np.exp(-accepted[0] * time_s)  # [mol/L]
    np.testing.assert_allclose(estimator.y_model[0][:, 0], expected_a,
                               rtol=PROFILE_RTOL, atol=PROFILE_ATOL)
    np.testing.assert_allclose(estimator.resid_runs[0][:, 0],
                               expected_a - observations,
                               rtol=PROFILE_RTOL, atol=PROFILE_ATOL)
    np.testing.assert_allclose(info["fun"], expected_a - observations,
                               rtol=PROFILE_RTOL, atol=PROFILE_ATOL)
    if trial_budget == 0:
        np.testing.assert_allclose(accepted, [SEED_RATE],
                                   rtol=0.0, atol=PARAMETER_ATOL)
    else:
        assert info["num_iter"] > 0
