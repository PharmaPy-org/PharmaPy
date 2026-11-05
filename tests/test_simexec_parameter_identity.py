"""Real reactor/crystallizer mask handoff, fitted coordinates and records.

The reactor is analytic first-order A -> B, with either the reference rate
or the activation energy fixed. The seeded crystallizer reuses the #352
growth-only case; its known generating parameters check estimator handoff,
not empirical calibration. All fits use real CVode/CVODES in the Assimulo lane.
"""

import numpy as np
import pytest

pytestmark = [pytest.mark.assimulo, pytest.mark.integration]
pytest.importorskip('assimulo')

import test_paramestim_accepted_reactor as reactor_case
import test_calibration_validation_crystallizer as growth_case
import test_crystallizer_paramest_requested_rows as cryst
from PharmaPy.Kinetics import RxnKinetics
from PharmaPy.ParamEstim import Experiment, Measurement
from PharmaPy.Phases import LiquidPhase
from PharmaPy.Reactors import BatchReactor
from PharmaPy.Records import CalibrationRecord, DataSource, ModelRevision
from PharmaPy.SimExec import SimulationExec
from PharmaPy.Utilities import CoolingWater

# Legacy RxnKinetics uses this rounded gas constant; retain its numerical
# convention in the independently rearranged Arrhenius reference below.
GAS_CONSTANT = 8.314  # [J/mol/K], RxnKinetics module constant
HOT_TEMPERATURE = 310.0  # [K], above T_ref so Ea is observable with fixed k_ref
TRUE_ENERGY = 2000.0  # [J/mol], synthetic modest temperature dependence
TIMES = np.array([0.0, 1.0, 2.0, 4.0, 8.0])  # [s], spans reaction time constant
# Recovering Ea divides log(k/k_ref) by (T-T_ref)/(R*T*T_ref), amplifying
# profile error. 1e-5 bounds that conditioning at CVode rtol=1e-9 here.
RECOVERY_RTOL = 1e-5  # [-]
SOURCES = (DataSource('analytic', 'synthetic', 'test-generated'),)


def reactor_fit(free_index: int, default_mask: bool = False) -> tuple:
    """Build a one-coordinate Arrhenius fit with an analytic reference.

    Parameters
    ----------
    free_index : int
        Index of the free coordinate: k_ref [1/s] (0) or Ea [J/mol] (1).
    default_mask : bool, optional
        Use the reactor's all-True default mask for explicit-override tests.

    Returns
    -------
    tuple
        SimulationExec, experiments, full seed ([1/s], [J/mol]), and
        optimize flags in k_ref, Ea order.
    """
    flags = [free_index == index for index in range(2)]
    seed = np.array([reactor_case.TRUE_RATE, TRUE_ENERGY])  # [1/s], [J/mol]
    seed[free_index] /= 2  # [-], finite offset from the generating parameter
    reactor = BatchReactor(isothermal=True, return_sens=False,
                           mask_params=None if default_mask else flags)
    reactor.Phases = LiquidPhase(
        reactor_case.THERMO_PATH, temp=HOT_TEMPERATURE, vol=reactor_case.VOLUME,
        mole_conc=reactor_case.INITIAL_CONCENTRATIONS)
    reactor.Kinetics = RxnKinetics(
        reactor_case.THERMO_PATH, k_params=[seed[0]], ea_params=[seed[1]],
        rxn_list=['A --> B'], temp_ref=reactor_case.TEMPERATURE,
        reformulate_kin=False)
    reactor.Utility = CoolingWater(temp_in=HOT_TEMPERATURE,
                                   mass_flow=reactor_case.UTILITY_FLOW)
    # Arrhenius referenced at T_ref, rearranged independently of get_rxn_rates.
    rate = reactor_case.TRUE_RATE * np.exp(
        TRUE_ENERGY * (HOT_TEMPERATURE - reactor_case.TEMPERATURE)
        / (GAS_CONSTANT * HOT_TEMPERATURE * reactor_case.TEMPERATURE))  # [1/s]
    concentrations = (reactor_case.INITIAL_CONCENTRATIONS[0]
                      * np.exp(-rate * TIMES))  # [mol/L]
    experiments = {'run': Experiment({
        'c_A': Measurement(0, TIMES, concentrations, units='mol/L', x_units='s')})}
    simulation = SimulationExec(reactor_case.THERMO_PATH, {'R01': []})
    simulation.R01 = reactor
    # RxnKinetics normalizes its physical inputs through an internal floating
    # representation. The public seed is that stored vector, not the literal.
    return simulation, experiments, reactor.Kinetics.concat_params().copy(), flags


@pytest.mark.parametrize('free_index', [0, 1])
@pytest.mark.parametrize('selection', ['implicit', 'none', 'matching', 'override',
                                       'short-names', 'full-names'])
def test_reactor_fit_record_preserves_fixed_and_free_coordinates(
        free_index, selection):
    simulation, experiments, seed, flags = reactor_fit(
        free_index, default_mask=selection == 'override')
    options = {} if selection == 'implicit' else {'optimize_flags': flags}
    expected_names = ['k_1', 'E_{a, 1}']
    if selection == 'none':
        options['optimize_flags'] = None
    elif selection == 'short-names':
        options['name_params'] = ['custom_kinetic']
        expected_names = ['theta_1', 'theta_2']
        expected_names[free_index] = 'custom_kinetic'
    elif selection == 'full-names':
        options['name_params'] = ['reference_rate', 'activation_energy']
        expected_names = ['reference_rate', 'activation_energy']
    simulation.SetParamEstimation(experiments, wrapper_kwargs={
        'sundials_opts': {'rtol': reactor_case.SOLVER_RTOL,
                           'atol': reactor_case.SOLVER_ATOL,
                           'report_continuously': True}}, **options)
    estimator = simulation.ParamInst
    assert estimator.name_params_total == expected_names
    assert estimator.name_params == [expected_names[free_index]]
    np.testing.assert_array_equal(estimator.map_variable, flags)
    # Ea's dimensional gradient is small [mol/L per J/mol]. Scale trial steps
    # by their Jacobian and use gtol=1e-14 so the dimensional stopping rule
    # does not accept a visibly inaccurate energy. These are fixture options.
    simulation.EstimateParams(method='scipy',
                              optim_options={'x_scale': 'jac', 'gtol': 1e-14},
                              verbose=False)
    expected = np.array([reactor_case.TRUE_RATE, TRUE_ENERGY])  # [1/s], [J/mol]
    np.testing.assert_allclose(estimator.reconstruct_params(estimator.params_convg),
                               expected, rtol=RECOVERY_RTOL, atol=0)
    np.testing.assert_array_equal(simulation.R01.Kinetics.concat_params()[~np.array(flags)],
                                  seed[~np.array(flags)])
    record = CalibrationRecord.from_estimation(
        estimator, experiments, case='reactor-identity',
        parameter_units=['1/s', 'J/mol'], model=ModelRevision('Arrhenius decay'),
        sources=SOURCES)
    loaded = CalibrationRecord.from_json(record.to_json())
    assert [item.name for item in loaded.parameters] == expected_names
    assert [item.optimized for item in loaded.parameters] == flags
    np.testing.assert_array_equal([item.seed for item in loaded.parameters], seed)
    for index, item in enumerate(loaded.parameters):
        if flags[index]:
            assert item.fitted == pytest.approx(expected[index], rel=RECOVERY_RTOL)
        else:
            assert item.fitted is None


@pytest.mark.parametrize('explicit', [False, True])
def test_crystallizer_masked_fit_records_all_thirteen_coordinates(explicit):
    truth_unit = cryst._crystallizer()
    outputs, _ = truth_unit.paramest_wrapper(
        growth_case.full_vector(growth_case.TRUE_PARAMS), growth_case.SAMPLE_TIMES)
    experiments = growth_case.make_experiments({
        'c_A': outputs[:, cryst.TARGET_COLUMN],
        'mu_3': outputs[:, growth_case.MOMENT_3_COLUMN]})
    unit = cryst._crystallizer()
    seed = growth_case.full_vector(growth_case.SEED)  # [growth_case.PARAMETER_UNITS]
    simulation = SimulationExec(cryst.DATABASE, {'C01': []})
    simulation.C01 = unit
    options = {'optimize_flags': list(unit.mask_params)} if explicit else {}
    simulation.SetParamEstimation(experiments, param_seed=seed, **options)
    estimator = simulation.ParamInst
    expected_names = ['k_{bp}', 'E_{bp}', 'b', 'k_{bs}', 'E_{bs}', 's_1', 's_2',
                      'k_{g}', 'E_{g}', 'g', 'k_{d}', 'E_{d}', 'd']
    assert estimator.name_params_total == expected_names
    assert estimator.name_params == ['k_{g}', 'g']
    simulation.EstimateParams(verbose=False)
    np.testing.assert_allclose(estimator.params_convg, growth_case.TRUE_PARAMS,
                               rtol=RECOVERY_RTOL, atol=0)
    fixed_indices = [0, 1, 2, 3, 4, 5, 6, 8, 10, 11, 12]
    np.testing.assert_array_equal(unit.Kinetics.concat_params()[fixed_indices],
                                  seed[fixed_indices])
    record = CalibrationRecord.from_estimation(
        estimator, experiments, case='growth-identity',
        parameter_units=growth_case.PARAMETER_UNITS,
        model=ModelRevision('seeded growth'), sources=SOURCES)
    loaded = CalibrationRecord.from_json(record.to_json())
    assert [item.name for item in loaded.parameters] == expected_names
    assert [index for index, item in enumerate(loaded.parameters)
            if item.optimized] == [7, 9]
    np.testing.assert_array_equal([item.seed for item in loaded.parameters], seed)
    for index in fixed_indices:
        assert loaded.parameters[index].fitted is None
    np.testing.assert_allclose([loaded.parameters[index].fitted for index in [7, 9]],
                               [1.0, 1.0], rtol=RECOVERY_RTOL, atol=0)


@pytest.mark.parametrize('kind', ['reactor', 'crystallizer'])
def test_explicit_flags_conflicting_with_partial_unit_mask_raise(kind):
    if kind == 'reactor':
        simulation, experiments, _, _ = reactor_fit(0)
        flags = [False, True]
    else:
        simulation = SimulationExec(cryst.DATABASE, {'C01': []})
        simulation.C01 = cryst._crystallizer()
        # Same count, different coordinates: comparing only counts is unsafe.
        flags = [index in (7, 8) for index in range(13)]
        experiments = growth_case.make_experiments({
            'c_A': np.ones_like(growth_case.SAMPLE_TIMES),
            'mu_3': np.ones_like(growth_case.SAMPLE_TIMES)})
    with pytest.raises(ValueError, match='optimize_flags conflicts with.*mask_params'):
        simulation.SetParamEstimation(experiments, optimize_flags=flags)
    assert not hasattr(simulation, 'ParamInst')
