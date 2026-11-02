"""Unit selection of ``SimulationExec.SetParamEstimation`` (issue #241).

Scope: ``pick_unit`` resolves the fitted unit over every flowsheet graph
node, including a unit that appears only as a successor, before any
kinetics or parameter metadata is read. The selected unit supplies the
estimator callback, the signature against which ``Experiment`` arguments
are bound (the per-experiment wrapper handoff of #346), the parameter seed
and the parameter names. Unknown names and an omitted ``pick_unit`` on a
multi-unit flowsheet are rejected; the single-unit default is kept.

Fixtures: two real isothermal batch reactors run the first-order reaction
A -> B with zero activation energy and distinguishable rate constants and
parameter masks; a real ``Filter`` provides a unit whose
``paramest_wrapper`` accepts one positional argument fewer than a
reactor's. Both graph forms are used: ``{'R01': [], 'R02': []}`` and the
successor-only form ``{'R01': ['R02']}``. The unit tests only build the
estimator, so they need no ODE solver and run in the core lane. The
Assimulo test evaluates the estimator's objective with CVode and compares
the predictions with the analytic solution
c_A = c_A0 exp(-k t), c_B = c_B0 + c_A0 (1 - exp(-k t)) of the selected
reactor; it integrates each reactor once per case.
"""

from pathlib import Path
import re

import numpy as np
import pytest

from PharmaPy.Errors import PharmaPyValueError
from PharmaPy.Kinetics import RxnKinetics
from PharmaPy.ParamEstim import Experiment, Measurement
from PharmaPy.Phases import LiquidPhase
from PharmaPy.Reactors import BatchReactor
from PharmaPy.SimExec import SimulationExec
from PharmaPy.SolidLiquidSep import Filter
from PharmaPy.Utilities import CoolingWater
from test_separation_balance_fixes import separation_phases

DATA = Path(__file__).parent
THERMO_PATH = str(DATA / 'integration/data/pfr_test_pure_comp.json')
TEMPERATURE = 300.0  # [K], isothermal; zero activation energy below
VOLUME = 0.002  # [m**3], concentrations are independent of volume
UTILITY_FLOW = 0.1  # [kg/s], unused by the isothermal balance
PRODUCT_INITIAL = 0.2  # [mol/L], c_B0
SOLVENT = 5.0  # [mol/L], inert solvent
# Synthetic first-order rate constants [1/s], distinct so that the selected
# reactor's seed and predictions identify it.
RATES = {'R01': 0.2, 'R02': 0.5}
# Parameter masks over (k_1, E_a1): R01 fits k only, R02 both, so the
# estimator's parameter names identify the selected reactor too.
MASKS = {'R01': [True, False], 'R02': [True, True]}
NAMES = {'R01': ['k_1'], 'R02': ['k_1', 'E_{a, 1}']}
GRAPHS = {'independent': {'R01': [], 'R02': []},
          'successor-only': {'R01': ['R02']}}
TIMES_S = np.array([0.0, 1.0, 3.0])  # [s], unequal spacing
INITIAL_A = 0.05  # [mol/L], charge of A set by the phase modifier
# RxnKinetics stores k with a machine-epsilon offset (Kinetics.set_params);
# 1e-12 relative bounds that roundoff.
SEED_RTOL = 1e-12  # [-]
# 100-fold integration-error budget of the committed reactor regressions
# (test_reactor_paramest_requested_rows.py), at rtol 1e-9, atol 1e-11.
SOLVER_OPTIONS = {'rtol': 1e-9, 'atol': 1e-11}  # [-], [mol/L]
PROFILE_RTOL = 1e-7  # [-]
PROFILE_ATOL = 1e-9  # [mol/L]
# A Filter's paramest_wrapper(params, time_vals, modify_phase,
# modify_controls, run_args) binds three positional arguments after the
# parameters and time grid; a reactor's binds four (reord_sens before
# run_args).
REACTOR_POSITIONAL_ARGS = 4  # [-]


def _reactor(name):
    """Build a real isothermal first-order batch reactor.

    Parameters
    ----------
    name : str
        Key of ``RATES`` and ``MASKS``.

    Returns
    -------
    BatchReactor
        Reactor charged with A, B, an absent species and solvent [mol/L].
    """
    reactor = BatchReactor(isothermal=True, mask_params=MASKS[name],
                           return_sens=False)
    reactor.Kinetics = RxnKinetics(
        THERMO_PATH, k_params=[RATES[name]], ea_params=[0.0],
        rxn_list=['A --> B'])  # [1/s], [J/mol]
    reactor.Phases = LiquidPhase(
        THERMO_PATH, temp=TEMPERATURE, vol=VOLUME,
        mole_conc=[0.1, PRODUCT_INITIAL, 0.0, SOLVENT])  # [mol/L]
    reactor.Utility = CoolingWater(temp_in=TEMPERATURE,
                                   mass_flow=UTILITY_FLOW)
    return reactor


def _two_reactors(graph):
    """Build a two-reactor flowsheet.

    Parameters
    ----------
    graph : str
        Key of ``GRAPHS``.

    Returns
    -------
    SimulationExec
        Flowsheet with reactors ``R01`` and ``R02``.
    """
    simulation = SimulationExec(THERMO_PATH, GRAPHS[graph])
    simulation.R01 = _reactor('R01')
    simulation.R02 = _reactor('R02')
    return simulation


def _phase_modifier():
    """Return the initial charge bound positionally by the experiment.

    Returns
    -------
    dict
        ``{'mole_conc': [...]}`` [mol/L]: A, B, absent species, solvent.
    """
    return {'mole_conc': [INITIAL_A, PRODUCT_INITIAL, 0.0, SOLVENT]}


def _analytic(rate):
    """Return analytic A and B concentrations after the modifier charge.

    Parameters
    ----------
    rate : float
        Rate constant k [1/s].

    Returns
    -------
    numpy.ndarray
        Shape ``(len(TIMES_S), 2)``, columns A and B [mol/L].
    """
    reactant = INITIAL_A * np.exp(-rate * TIMES_S)  # [mol/L]
    return np.column_stack((reactant,
                            PRODUCT_INITIAL + INITIAL_A - reactant))


def _experiment(args):
    """Return an experiment measuring A and B with positional arguments.

    Parameters
    ----------
    args : tuple
        Positional callback arguments, bound after the parameters and time
        grid of the selected unit's ``paramest_wrapper``.

    Returns
    -------
    Experiment
        Measurements ``c_A`` (field 0) and ``c_B`` (field 1) [mol/L],
        generated by R01's analytic solution; only their shape matters to
        the selection.
    """
    data = _analytic(RATES['R01'])  # [mol/L]
    return Experiment(
        {'c_A': Measurement(0, TIMES_S, data[:, 0], units='mol/L'),
         'c_B': Measurement(1, TIMES_S, data[:, 1], units='mol/L')},
        args=args)


def _other(name):
    """Return the name of the reactor that was not selected.

    Parameters
    ----------
    name : str
        ``'R01'`` or ``'R02'``.

    Returns
    -------
    str
        The other reactor's name.
    """
    return 'R02' if name == 'R01' else 'R01'


@pytest.mark.unit
@pytest.mark.parametrize('graph', list(GRAPHS))
@pytest.mark.parametrize('picked', ['R01', 'R02'])
def test_pick_unit_binds_the_selected_unit(graph, picked):
    simulation = _two_reactors(graph)
    modifier = _phase_modifier()
    simulation.SetParamEstimation(_experiment((modifier,)),
                                  pick_unit=picked,
                                  optimize_flags=MASKS[picked],
                                  wrapper_kwargs={'sundials_opts':
                                                  SOLVER_OPTIONS})
    estimator = simulation.ParamInst
    selected = getattr(simulation, picked)
    assert estimator.function.__self__ is selected
    assert estimator.function.__self__ is not getattr(simulation,
                                                      _other(picked))
    assert estimator.function.__func__ is BatchReactor.paramest_wrapper
    assert estimator.param_seed[0] == pytest.approx(RATES[picked],
                                                    rel=SEED_RTOL)
    assert estimator.name_params == NAMES[picked]
    # The experiment's positional modifier is bound against the selected
    # wrapper's signature: it stays positional and is not injected again.
    assert estimator.args_fun == [(modifier,)]
    assert estimator.args_fun[0][0] is modifier
    assert estimator.kwargs_fun == [
        {'modify_controls': {},
         'run_args': {'sundials_opts': {'rtol': 1e-9,  # [-]
                                        'atol': 1e-11}}}]  # [mol/L]


@pytest.mark.unit
@pytest.mark.parametrize('graph', list(GRAPHS))
@pytest.mark.parametrize('pick_unit', ['R03', 'r01', ''])
def test_unknown_pick_unit_is_rejected_with_the_valid_names(graph, pick_unit):
    simulation = _two_reactors(graph)
    with pytest.raises(PharmaPyValueError, match=(
            rf"^pick_unit {re.escape(repr(pick_unit))} is not a unit of the "
            r"flowsheet; valid names: \['R01', 'R02'\]$")):
        simulation.SetParamEstimation(_experiment((_phase_modifier(),)),
                                      pick_unit=pick_unit)
    assert not hasattr(simulation, 'ParamInst')


@pytest.mark.unit
@pytest.mark.parametrize('graph', list(GRAPHS))
def test_missing_pick_unit_is_rejected_on_multi_unit_graphs(graph):
    simulation = _two_reactors(graph)
    with pytest.raises(RuntimeError, match=(
            r"^Two or more unit operations detected\. Select one using the "
            r"'pick_unit' argument; valid names: \['R01', 'R02'\]$")):
        simulation.SetParamEstimation(_experiment((_phase_modifier(),)))
    assert not hasattr(simulation, 'ParamInst')


@pytest.mark.unit
@pytest.mark.parametrize('pick_unit', [None, 'R02'])
def test_single_unit_default_and_explicit_name(pick_unit):
    simulation = SimulationExec(THERMO_PATH, {'R02': []})
    simulation.R02 = _reactor('R02')
    simulation.SetParamEstimation(_experiment((_phase_modifier(),)),
                                  pick_unit=pick_unit)
    assert simulation.ParamInst.function.__self__ is simulation.R02
    assert simulation.ParamInst.param_seed[0] == pytest.approx(
        RATES['R02'], rel=SEED_RTOL)


@pytest.mark.unit
def test_single_unit_rejects_another_name():
    simulation = SimulationExec(THERMO_PATH, {'R02': []})
    simulation.R02 = _reactor('R02')
    simulation.R01 = _reactor('R01')  # attached, but not a graph node
    with pytest.raises(PharmaPyValueError, match=(
            r"^pick_unit 'R01' is not a unit of the flowsheet; valid "
            r"names: \['R02'\]$")):
        simulation.SetParamEstimation(_experiment((_phase_modifier(),)),
                                      pick_unit='R01')
    assert not hasattr(simulation, 'ParamInst')


@pytest.mark.unit
@pytest.mark.parametrize('graph', [{'R01': [], 'F01': []}, {'R01': ['F01']}],
                         ids=['independent', 'successor-only'])
def test_experiment_args_bind_against_the_selected_signature(
        separation_phases, graph):
    path = str(DATA / 'Flowsheet/data/compound_database.json')
    reactor = BatchReactor(isothermal=True, mask_params=[True, False],
                           return_sens=False)
    reactor.Kinetics = RxnKinetics(path, k_params=[RATES['R01']],
                                   ea_params=[0.0],
                                   rxn_list=['A --> B'])  # [1/s], [J/mol]
    reactor.Phases = LiquidPhase(path, temp=TEMPERATURE, vol=VOLUME,
                                 mole_conc=[0.1, PRODUCT_INITIAL, 0.0, 0.0,
                                            SOLVENT])  # [mol/L]
    reactor.Utility = CoolingWater(temp_in=TEMPERATURE,
                                   mass_flow=UTILITY_FLOW)
    unit = Filter(station_diam=0.1, alpha=1e11)  # [m], [m/kg]
    unit.Phases = list(separation_phases)
    simulation = SimulationExec(path, graph)
    simulation.R01 = reactor
    simulation.F01 = unit
    # modify_phase, modify_controls, reord_sens, run_args of a reactor.
    args = ({}, {}, False, {})
    assert len(args) == REACTOR_POSITIONAL_ARGS
    simulation.SetParamEstimation(_experiment(args), pick_unit='R01')
    assert simulation.ParamInst.function.__self__ is reactor
    assert simulation.ParamInst.args_fun == [args]
    del simulation.ParamInst
    with pytest.raises(TypeError, match=(
            r"^Experiment 'exp_1' args .* cannot be bound to the unit's "
            r"paramest_wrapper\(params: .*, time_vals: .*, run_args: dict "
            r"= \{\}\) -> numpy\.ndarray after its parameters and time "
            r"grid: too many positional arguments$")):
        simulation.SetParamEstimation(_experiment(args), pick_unit='F01')
    assert not hasattr(simulation, 'ParamInst')


@pytest.mark.assimulo
@pytest.mark.integration
@pytest.mark.parametrize('graph', list(GRAPHS))
@pytest.mark.parametrize('picked', ['R01', 'R02'])
def test_selected_reactor_callback_predicts_its_own_solution(graph, picked):
    pytest.importorskip('assimulo')
    simulation = _two_reactors(graph)
    simulation.SetParamEstimation(_experiment((_phase_modifier(),)),
                                  pick_unit=picked,
                                  optimize_flags=MASKS[picked],
                                  wrapper_kwargs={'sundials_opts':
                                                  SOLVER_OPTIONS})
    estimator = simulation.ParamInst
    estimator.get_objective(estimator.param_seed[estimator.map_variable])
    predicted = estimator.y_runs[0]  # [mol/L], A then B
    expected = _analytic(RATES[picked])  # [mol/L]
    np.testing.assert_allclose(predicted, expected, rtol=PROFILE_RTOL,
                               atol=PROFILE_ATOL)
    # The other reactor's solution lies outside the same error budget.
    other = _analytic(RATES[_other(picked)])  # [mol/L]
    assert not np.allclose(predicted, other, rtol=PROFILE_RTOL,
                           atol=PROFILE_ATOL)
    # Only the selected reactor was integrated.
    assert hasattr(getattr(simulation, picked), 'result')
    assert getattr(simulation, picked).elapsed_time == pytest.approx(
        TIMES_S[-1], rel=SEED_RTOL)
    assert not hasattr(getattr(simulation, _other(picked)), 'result')
