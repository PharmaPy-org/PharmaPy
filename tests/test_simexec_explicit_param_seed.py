"""Explicit ``param_seed`` handoff of ``SetParamEstimation`` (issue #264).

Scope: an explicit ``param_seed`` takes precedence over a unit's default
seed. For a unit without kinetics (a real ``Filter``) it reaches the
estimator unchanged, in the Filter's declared basis (physical resistances,
or their natural logarithms with ``log_params=True``) and order (alpha,
then medium resistance), as a copy that leaves the caller's array and the
Filter's configured resistances untouched; invalid seeds are rejected.
The default-seed path, including ``Filter.param_seed`` validation, and the
kinetics-backed path (``Kinetics.set_params`` then ``concat_params``) are
kept.

Fixtures: the dilute litre-scale slurry of
``test_separation_balance_fixes.separation_phases``; a Filter configured
with alpha = 1e11 m/kg and medium resistance 1e9 1/m, and a supplied seed
of 2e11 m/kg and 3e9 1/m, deliberately different in both entries; an
isothermal first-order A -> B batch reactor with zero activation energy
for the kinetics path. The unit tests only build the estimator and run in
the core lane. The Assimulo test integrates the Filter once and compares
its filtrate masses with the constant-pressure Darcy integral
a m**2 + b m = c t, solved for m, at the supplied resistances.
"""

from pathlib import Path

import numpy as np
import pytest

from PharmaPy.Kinetics import RxnKinetics
from PharmaPy.Phases import LiquidPhase
from PharmaPy.Reactors import BatchReactor
from PharmaPy.SimExec import SimulationExec
from PharmaPy.SolidLiquidSep import Filter
from PharmaPy.Utilities import CoolingWater
from test_separation_balance_fixes import RTOL, separation_phases

STATION_DIAMETER = 0.1  # [m]
CONFIGURED = (1e11, 1e9)  # [m/kg], [1/m], Filter constructor resistances
SUPPLIED = (2e11, 3e9)  # [m/kg], [1/m], explicit seed, unlike CONFIGURED
GRID = np.array([0.0, 0.01, 0.02])  # [s], before cake completion
OBSERVED = np.array([0.0, 0.001, 0.002])  # [kg], synthetic filtrate masses
# CVode tolerances of the Filter regressions in test_filter_solver_options.
SOLVER_OPTIONS = {'rtol': 1e-9, 'atol': 1e-11}  # [-], [kg]
# 100-fold integration-error budget on SOLVER_OPTIONS, the convention of
# the reactor regressions (PROFILE_RTOL and PROFILE_ATOL of
# test_reactor_paramest_requested_rows.py). For this fixture CVode's
# filtrate masses differ from the Darcy integral by at most 1.2e-8
# relative and 4e-11 kg, well inside the budget.
MASS_RTOL = 100 * SOLVER_OPTIONS['rtol']  # [-]
MASS_ATOL = 100 * SOLVER_OPTIONS['atol']  # [kg]
FILTER_NAMES = ['alpha', 'Rm']
REACTOR_PATH = str(
    Path(__file__).parent / 'integration/data/pfr_test_pure_comp.json')
# Synthetic pressure-dependent resistance slope [m/kg/Pa] for alpha, or
# [1/m/Pa] for the medium, as in test_filter_solver_options.py; 1e6 gives
# the configured 1e11 m/kg at one bar (1e5 Pa). Only its callability
# matters here: callable resistances are rejected before any evaluation.
PRESSURE_SLOPE = 1e6
# Kinetics-path case: an isothermal first-order A -> B batch reaction in
# an inert solvent, the reactor fixture of test_simexec_modifier_handoff.py.
# Nothing is integrated; only the seed handoff is checked.
DEFAULT_RATE = 0.2  # [1/s], first-order rate constant at construction
SUPPLIED_RATE = 0.35  # [1/s], explicit reactor seed, unlike DEFAULT_RATE
ACTIVATION_ENERGY = 0.0  # [J/mol], zero: rate independent of temperature
REACTOR_TEMPERATURE = 300.0  # [K], isothermal liquid and utility inlet
REACTOR_VOLUME = 0.002  # [m**3], 2 L charge; not used by the seed checks
# [mol/L]: A, B, an absent species and the solvent; dilute reacting
# solutes in an inert solvent.
REACTOR_CHARGE = [0.1, 0.2, 0.0, 5.0]
# [kg/s], cooling-water flow required by the BatchReactor collaborator
# contract; unused by the isothermal balance.
UTILITY_FLOW = 0.1


def _filter_simulation(phases, log_params, resist_medium=CONFIGURED[1]):
    """Build a single-unit flowsheet around a real Filter.

    Parameters
    ----------
    phases : tuple
        Liquid and solid phases of the slurry.
    log_params : bool
        Whether the Filter estimates natural logarithms of its resistances.
    resist_medium : float, optional
        Configured medium resistance [1/m].

    Returns
    -------
    SimulationExec
        Flowsheet whose only unit, ``F01``, is the Filter.
    """
    unit = Filter(station_diam=STATION_DIAMETER, alpha=CONFIGURED[0],
                  resist_medium=resist_medium, log_params=log_params)
    unit.Phases = list(phases)
    simulation = SimulationExec(unit.Liquid_1.path_data, {'F01': []})
    simulation.F01 = unit
    return simulation


def _supplied_seed(log_params):
    """Return the supplied seed in the Filter's estimation basis.

    Parameters
    ----------
    log_params : bool
        Whether the Filter estimates natural logarithms.

    Returns
    -------
    numpy.ndarray
        Shape (2,): alpha [m/kg] and medium resistance [1/m], or their
        natural logarithms [-].
    """
    physical = np.array(SUPPLIED)  # [m/kg, 1/m]
    return np.log(physical) if log_params else physical


@pytest.mark.unit
@pytest.mark.parametrize('log_params', [False, True])
@pytest.mark.parametrize('container', ['array', 'list'])
def test_explicit_filter_seed_reaches_the_estimator_unchanged(
        separation_phases, log_params, container):
    simulation = _filter_simulation(separation_phases, log_params)
    expected = _supplied_seed(log_params)  # [m/kg, 1/m] or [-]
    seed = expected.copy() if container == 'array' else expected.tolist()
    simulation.SetParamEstimation(GRID, OBSERVED, param_seed=seed)
    estimator = simulation.ParamInst
    # Exact values, order and basis: no conversion between caller and
    # estimator.
    assert estimator.param_seed.dtype == np.float64
    np.testing.assert_array_equal(estimator.param_seed, expected)
    assert estimator.name_params == FILTER_NAMES
    # The caller's seed is neither modified nor shared.
    np.testing.assert_array_equal(seed, expected)
    if container == 'array':
        assert not np.shares_memory(estimator.param_seed, seed)
        estimator.param_seed[0] = 0.0
        np.testing.assert_array_equal(seed, expected)
    # The Filter's configured resistances and default seed are unchanged.
    unit = simulation.F01
    assert (unit.alpha, unit.r_medium) == CONFIGURED
    default = np.array(CONFIGURED)  # [m/kg, 1/m]
    np.testing.assert_array_equal(
        unit.param_seed, np.log(default) if log_params else default)


@pytest.mark.unit
def test_explicit_seed_takes_precedence_over_an_invalid_default(
        separation_phases):
    # Zero medium resistance is a valid constructor value, but
    # Filter.param_seed rejects it as an estimation seed.
    simulation = _filter_simulation(separation_phases, False,
                                    resist_medium=0.0)
    with pytest.raises(ValueError, match='medium resistance'):
        simulation.F01.param_seed
    simulation.SetParamEstimation(GRID, OBSERVED,
                                  param_seed=np.array(SUPPLIED))
    np.testing.assert_array_equal(simulation.ParamInst.param_seed, SUPPLIED)


@pytest.mark.unit
@pytest.mark.parametrize('seed', [
    [1e11],  # one entry short
    [1e11, 1e9, 1.0],  # one entry too many
    [[1e11, 1e9]],  # two-dimensional
    [np.nan, 1e9],
    [1e11, np.inf],
    {'alpha': 2e11, 'Rm': 3e9},  # mapping, not an ordered array
    ['2e11', '3e9'],  # strings are not coerced
    [2e11 + 1e5j, 3e9],  # complex
    [[2e11], [3e9, 1.0]],  # ragged
    [True, True],  # booleans are not resistances
], ids=['short', 'long', '2-D', 'nan', 'inf', 'dict', 'strings', 'complex',
        'ragged', 'bool'])
def test_invalid_explicit_filter_seed_is_rejected(separation_phases, seed):
    simulation = _filter_simulation(separation_phases, False)
    with pytest.raises(ValueError, match=(
            r"^param_seed for a Filter must be a finite one-dimensional "
            r"array of real numbers with one entry per parameter "
            r"\['alpha', 'Rm'\], in the unit's estimation basis; got ")):
        simulation.SetParamEstimation(GRID, OBSERVED, param_seed=seed)
    assert not hasattr(simulation, 'ParamInst')


CALLABLE_MESSAGE = (r"^Callable resistances are not estimable; configure "
                    r"numeric alpha \[m/kg\] and medium resistance \[1/m\] "
                    r"before estimation\.$")
DOMAIN_MESSAGE = (r"^Estimation seeds require finite alpha \[m/kg\] \(strictly "
                  r"positive\) and medium resistance \[1/m\] \(strictly "
                  r"positive\); positive medium resistance is required for "
                  r"the zero-filtrate startup in either parameterization; "
                  r"received ")


def _pressure_resistance(pressure):
    """Return a synthetic pressure-dependent resistance.

    Parameters
    ----------
    pressure : float
        Pressure drop [Pa].

    Returns
    -------
    float
        Resistance [m/kg] or [1/m], ``PRESSURE_SLOPE * pressure``.
    """
    return PRESSURE_SLOPE * pressure


@pytest.mark.unit
@pytest.mark.parametrize('log_params, physical_seed, configured, message', [
    pytest.param(log_params, SUPPLIED, configured, CALLABLE_MESSAGE,
                 id=f'callable {configured}, log_params={log_params}')
    for configured in ('alpha', 'r_medium') for log_params in (False, True)
] + [
    # Nonpositive resistances have no real logarithm; the log-basis domain
    # is covered by test_explicit_log_seed_must_give_positive_finite_resistances.
    pytest.param(False, (2e11, 0.0), None, DOMAIN_MESSAGE, id='zero medium'),
    pytest.param(False, (-2e11, 3e9), None, DOMAIN_MESSAGE,
                 id='negative alpha'),
])
def test_explicit_filter_seed_follows_the_filter_rules(
        separation_phases, log_params, physical_seed, configured, message):
    simulation = _filter_simulation(separation_phases, log_params)
    unit = simulation.F01
    if configured is not None:
        setattr(unit, configured, _pressure_resistance)
    seed = np.array(physical_seed)  # [m/kg, 1/m]
    if log_params:
        seed = np.log(seed)  # [-]
    before = seed.copy()
    with pytest.raises(ValueError, match=message):
        simulation.SetParamEstimation(GRID, OBSERVED, param_seed=seed)
    assert not hasattr(simulation, 'ParamInst')
    np.testing.assert_array_equal(seed, before)
    expected = {'alpha': CONFIGURED[0], 'r_medium': CONFIGURED[1]}
    if configured is not None:
        expected[configured] = _pressure_resistance
    assert (unit.alpha, unit.r_medium) == (expected['alpha'],
                                           expected['r_medium'])


@pytest.mark.unit
@pytest.mark.parametrize('logarithms', [[1000.0, 21.0], [-1000.0, 21.0],
                                        [26.0, -1000.0]],
                         ids=['alpha overflows', 'alpha underflows',
                              'medium underflows'])
def test_explicit_log_seed_must_give_positive_finite_resistances(
        separation_phases, logarithms):
    # [-]: exp(1000) overflows float64 and exp(-1000) underflows to zero,
    # as in test_filter_review_contracts.py; 21 and 26 are ordinary
    # logarithms of about 1.3e9 1/m and 2e11 m/kg.
    simulation = _filter_simulation(separation_phases, True)
    with pytest.raises(ValueError, match=DOMAIN_MESSAGE):
        simulation.SetParamEstimation(GRID, OBSERVED,
                                      param_seed=np.array(logarithms))
    assert not hasattr(simulation, 'ParamInst')


@pytest.mark.unit
@pytest.mark.parametrize('seed', [
    [2e11],
    [2e11, 3e9, 1.0],
    [[2e11, 3e9]],
    {'alpha': 2e11, 'Rm': 3e9},
    ['2e11', '3e9'],
    np.array(['2e11', '3e9']),
    [2e11 + 1e5j, 3e9],
    np.array([2e11 + 0j, 3e9 + 0j]),  # complex dtype, zero imaginary parts
    [True, True],
    [[2e11], [3e9, 1.0]],
], ids=['short', 'long', '2-D', 'dict', 'string list', 'string array',
        'complex list', 'complex array', 'bool', 'ragged'])
def test_filter_validator_requires_two_real_resistances(separation_phases,
                                                        seed):
    unit = _filter_simulation(separation_phases, False).F01
    with pytest.raises(ValueError, match=(
            r"^Filter estimation seeds hold alpha and medium resistance as "
            r"two real numbers, shape \(2,\); received ")):
        unit.validate_param_seed(seed)


@pytest.mark.unit
@pytest.mark.parametrize('log_params', [False, True])
def test_filter_validator_accepts_a_valid_seed_unchanged(separation_phases,
                                                         log_params):
    unit = _filter_simulation(separation_phases, log_params).F01
    seed = _supplied_seed(log_params)  # [m/kg, 1/m] or [-]
    before = seed.copy()
    assert unit.validate_param_seed(seed) is None
    np.testing.assert_array_equal(seed, before)
    assert (unit.alpha, unit.r_medium) == CONFIGURED


@pytest.mark.unit
@pytest.mark.parametrize('log_params', [False, True])
def test_default_filter_seed_is_kept(separation_phases, log_params):
    simulation = _filter_simulation(separation_phases, log_params)
    simulation.SetParamEstimation(GRID, OBSERVED)
    default = np.array([1e11, 1e9])  # [m/kg, 1/m], CONFIGURED written out
    np.testing.assert_array_equal(
        simulation.ParamInst.param_seed,
        np.log(default) if log_params else default)


@pytest.mark.unit
def test_default_filter_seed_validation_is_kept(separation_phases):
    simulation = _filter_simulation(separation_phases, False)
    simulation.F01.alpha = _pressure_resistance
    with pytest.raises(ValueError,
                       match='^Callable resistances are not estimable'):
        simulation.SetParamEstimation(GRID, OBSERVED)
    assert not hasattr(simulation, 'ParamInst')


@pytest.mark.unit
def test_explicit_seed_is_set_on_reactor_kinetics():
    reactor = BatchReactor(isothermal=True, mask_params=[True, False],
                           return_sens=False)
    reactor.Kinetics = RxnKinetics(
        REACTOR_PATH, k_params=[DEFAULT_RATE],
        ea_params=[ACTIVATION_ENERGY], rxn_list=['A --> B'])
    reactor.Phases = LiquidPhase(REACTOR_PATH, temp=REACTOR_TEMPERATURE,
                                 vol=REACTOR_VOLUME,
                                 mole_conc=REACTOR_CHARGE)
    reactor.Utility = CoolingWater(temp_in=REACTOR_TEMPERATURE,
                                   mass_flow=UTILITY_FLOW)
    simulation = SimulationExec(REACTOR_PATH, {'R01': []})
    simulation.R01 = reactor
    seed = np.array([SUPPLIED_RATE, ACTIVATION_ENERGY])  # [1/s], [J/mol]
    simulation.SetParamEstimation(GRID, np.zeros((len(GRID), 2)),
                                  measured_ind=[0, 1],
                                  optimize_flags=[True, False],
                                  param_seed=seed)
    # [1/s], [J/mol]: SUPPLIED_RATE and zero activation energy, written out
    # independently of the seed constants.
    expected = np.array([0.35, 0.0])
    np.testing.assert_array_equal(reactor.Kinetics.concat_params(), expected)
    np.testing.assert_array_equal(simulation.ParamInst.param_seed, expected)
    np.testing.assert_array_equal(seed, expected)


@pytest.mark.assimulo
@pytest.mark.integration
@pytest.mark.parametrize('log_params', [False, True])
def test_explicit_filter_seed_drives_the_filter_callback(
        separation_phases, log_params):
    pytest.importorskip('assimulo')
    simulation = _filter_simulation(separation_phases, log_params)
    simulation.SetParamEstimation(
        GRID, OBSERVED, param_seed=_supplied_seed(log_params),
        wrapper_kwargs={'sundials_opts': SOLVER_OPTIONS})
    estimator = simulation.ParamInst
    estimator.get_objective(estimator.param_seed)
    unit = simulation.F01
    # The callback resolved the supplied seed into physical resistances
    # once; exp(log(x)) differs from x by roundoff only.
    np.testing.assert_allclose(unit.params, SUPPLIED, rtol=RTOL)
    predicted = estimator.y_runs[0][:, 0]  # [kg]
    density, viscosity, _, _ = unit.physical_props  # [kg/m**3], [Pa*s], [N/m], [kg/m**3]
    forcing = unit.deltaP / viscosity  # [1/s]

    def darcy_mass(alpha, resistance):
        """Return filtrate masses of the constant-pressure Darcy integral.

        Parameters
        ----------
        alpha : float
            Specific cake resistance [m/kg].
        resistance : float
            Medium resistance [1/m].

        Returns
        -------
        numpy.ndarray
            Filtrate mass [kg] at ``GRID``.
        """
        quadratic = alpha * unit.c_solids / (
            2 * (unit.area_filt * density)**2)  # [1/kg**2]
        linear = resistance / (unit.area_filt * density)  # [1/kg]
        return 2 * forcing * GRID / (
            linear + np.sqrt(linear**2 + 4 * quadratic * forcing * GRID))

    np.testing.assert_allclose(predicted, darcy_mass(*SUPPLIED),
                               rtol=MASS_RTOL, atol=MASS_ATOL)
    assert predicted[-1] > 0  # [kg], filtration was exercised
    assert predicted[-1] < unit.mass_crit  # [kg], before cake completion
    # The configured resistances would give a different filtrate history.
    assert not np.allclose(predicted, darcy_mass(*CONFIGURED),
                           rtol=MASS_RTOL, atol=MASS_ATOL)
