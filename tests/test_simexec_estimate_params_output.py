"""Console output and timing of ``SimulationExec.EstimateParams`` (#481).

Scope: ``EstimateParams`` prints its ``Optimization time`` line only when
``verbose`` is truthy and always stores the wall-clock duration [s] as
``optimization_time``; the fitted parameters do not depend on
``verbose``. Fixture: a real isothermal batch reactor for the first-order
reaction A -> B with zero activation energy, fitted from a slow seed to
noise-free analytic concentrations c_A = c_A0 exp(-k t),
c_B = c_B0 + c_A0 (1 - exp(-k t)). Requires the optional Assimulo stack;
each case runs one short Levenberg-Marquardt fit of a few dozen CVode
solves.
"""

from pathlib import Path
import re

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
UTILITY_FLOW = 0.1  # [kg/s], unused by the isothermal balance
INITIAL = [0.1, 0.2, 0.0, 5.0]  # [mol/L], A, B, absent species, solvent
TRUE_RATE = 0.2  # [1/s], generates the observations
SEED_RATE = 0.15  # [1/s], LM seed
TIMES_S = np.array([1.0, 2.0, 4.0])  # [s]
SOLVER_OPTIONS = {'rtol': 1e-9, 'atol': 1e-11}  # [-], [mol/L]
# Fit tolerance of the committed reactor regressions
# (test_reactor_paramest_requested_rows.py) at these CVode tolerances.
FIT_ATOL = 1e-6  # [1/s]
TIMING_LINE = re.compile(r'^Optimization time: (\d\.\d{2}e[+-]\d{2}) s\.$')


def _fit(verbose):
    """Fit the reactor's rate constant through ``EstimateParams``.

    Parameters
    ----------
    verbose : int or bool
        Passed to ``EstimateParams``.

    Returns
    -------
    tuple
        The simulation and the fitted variable parameters [1/s].
    """
    reactor = BatchReactor(isothermal=True, mask_params=[True, False],
                           return_sens=False)
    reactor.Kinetics = RxnKinetics(
        THERMO_PATH, k_params=[SEED_RATE], ea_params=[0.0],
        rxn_list=['A --> B'])  # [1/s], [J/mol]
    reactor.Phases = LiquidPhase(THERMO_PATH, temp=TEMPERATURE, vol=VOLUME,
                                 mole_conc=INITIAL)
    reactor.Utility = CoolingWater(temp_in=TEMPERATURE,
                                   mass_flow=UTILITY_FLOW)
    simulation = SimulationExec(THERMO_PATH, {'R01': []})
    simulation.R01 = reactor
    reactant = INITIAL[0] * np.exp(-TRUE_RATE * TIMES_S)  # [mol/L]
    observed = np.column_stack(
        (reactant, INITIAL[1] + INITIAL[0] - reactant))  # [mol/L]
    simulation.SetParamEstimation(
        TIMES_S, observed, measured_ind=[0, 1], optimize_flags=[True, False],
        wrapper_kwargs={'sundials_opts': SOLVER_OPTIONS})
    assert simulation.optimization_time is None
    fitted, _, _ = simulation.EstimateParams(verbose=verbose)
    return simulation, fitted


@pytest.mark.parametrize('verbose', [False, 0])
def test_quiet_estimation_prints_nothing(capsys, verbose):
    simulation, fitted = _fit(verbose)
    assert capsys.readouterr().out == ''
    assert simulation.optimization_time > 0  # [s]
    assert fitted == pytest.approx([TRUE_RATE], abs=FIT_ATOL)


@pytest.mark.parametrize('verbose', [True, 1])
def test_verbose_estimation_prints_the_stored_time(capsys, verbose):
    simulation, fitted = _fit(verbose)
    lines = capsys.readouterr().out.splitlines()
    timing = [line for line in lines if line.startswith('Optimization time')]
    assert len(timing) == 1
    assert timing[0] == lines[-1]
    match = TIMING_LINE.match(timing[0])
    assert match is not None
    assert match.group(1) == '{:.2e}'.format(simulation.optimization_time)
    assert simulation.optimization_time > 0  # [s]
    assert fitted == pytest.approx([TRUE_RATE], abs=FIT_ATOL)
