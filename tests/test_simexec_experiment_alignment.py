"""Issue #238 through real SimulationExec, batch kinetics and CVode callbacks.

A synthetic isothermal first-order A -> B reaction has an analytic solution;
different initial concentrations and schedules expose experiment mismatches.
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


@pytest.mark.parametrize('single', [False, True])
def test_named_experiments_reach_the_matching_reactor_initial_state(single):
    path = str(Path(__file__).parent / 'integration/data/pfr_test_pure_comp.json')
    rate = 0.2  # [1/s], synthetic first-order rate, reaction time scale 5 s
    reactor = BatchReactor(isothermal=True, mask_params=[True, False],
                           return_sens=False)
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
