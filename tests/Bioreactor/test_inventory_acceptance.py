"""Absolute inventory acceptance, unit conversions and real solver handoffs.

Analytical first-order conversion isolates output accuracy from physical
feasibility. Native phases and metadata are used without substituted callbacks.
"""

import copy
import numpy as np
import pytest

from PharmaPy.Bioreactors import build_bioreactor
from PharmaPy.Bioreactors.numerical import inventory_limits
from PharmaPy.Bioreactors.verification import run_convergence
from PharmaPy.Bioreactors.input_format import split_forward_inputs
from PharmaPy.IntegratorBackends import check_inventory, ScipyBackend
from PharmaPy.DataClasses import StateKey, PhaseRef
from tests.Bioreactor.test_bioreactor_adaptive_inventory import configured_case, vessel, EXAMPLE, RATE


# Helpers build the same physical case with explicit accuracy requirements.
def assembly_with_limit(unit='kg', value=1e-12, backend='scipy'):
    """Build an analytical batch and compile real state metadata.

    Parameters
    ----------
    unit : str
        Acceptance amount unit.
    value : float
        Absolute negative-error budget [unit]; default matches legacy kg limit.
    backend : str
        Native solver selection.

    Returns
    -------
    NativeBioreactorAssembly
        Compiled model with explicit limits, separate from solver tolerances.
    """
    case = configured_case(backend)
    case['numerical']['inventory_acceptance'] = {'mass_j': {'unit': unit, 'value': value}}
    assembly = build_bioreactor(case, thermo_path=EXAMPLE/'thermo.json')
    assembly.unit.compile_integrator(verbose=False)
    return assembly


@pytest.mark.unit
@pytest.mark.parametrize('unit,value', [('kg', 1e-12), ('g', 1e-9), ('mg', 1e-6)])
def test_equivalent_units_and_raw_negative_values(unit, value):
    """One nanogram has the same acceptance meaning in every supported mass unit."""
    assembly = assembly_with_limit(unit, value)
    limits = inventory_limits(assembly.unit, assembly.inventory_acceptance)
    key, = limits
    assert limits[key] == pytest.approx(1e-12, rel=1e-14, abs=0.)  # [kg]; exact metric prefixes
    states = np.asarray(assembly.unit.create_solver_init_states())[None, :]  # [native state units]
    position = assembly.unit.solver_state_collection.slices[key].start
    states[0, position] = -5e-13  # [kg]; half of the explicit 1 ng budget
    original = states.copy()
    check_inventory(assembly.unit, [2.], states, limits)  # [s]
    np.testing.assert_array_equal(states, original)
    states[0, position] = -2e-12  # [kg]; twice the same independent acceptance budget
    with pytest.raises(RuntimeError, match=r'component .*time 2 s, value -2e-12 kg, limit 1e-12 kg'):
        check_inventory(assembly.unit, [2.], states, limits)


@pytest.mark.unit
def test_named_component_limits_follow_native_order():
    assembly = assembly_with_limit()
    names = list(assembly.phase.name_species)
    values = dict(zip(reversed(names), [1., 2., 3., 4.]))  # [ng]; asymmetric named budgets expose permutations
    declaration = {'mass_j': {'unit': 'ng', 'values': values}}
    limits = inventory_limits(assembly.unit, declaration)
    key, = limits
    np.testing.assert_allclose(limits[key], [values[name]*1e-12 for name in names], rtol=1e-14, atol=0.)  # [kg]


@pytest.mark.unit
@pytest.mark.parametrize('quantity,message', [
    ({'unit': 'kg', 'value': -1.}, 'finite and nonnegative'),
    ({'unit': 'kg', 'value': float('nan')}, 'finite and nonnegative'),
    ({'unit': 'kg', 'value': float('inf')}, 'finite and nonnegative'),
    ({'unit': 'L', 'value': 1.}, 'unsupported conversion from L to kg'),
    ({'unit': None, 'value': 1.}, 'explicit unit string'),
    ({'unit': 'kg', 'value': None}, 'finite and nonnegative'),
    ({'unit': 'kg', 'values': {'missing': 1.}}, 'every native component'),
    ({'unit': 'kg', 'value': [1., 2.]}, 'must be scalar'),
])
def test_invalid_acceptance_is_rejected(quantity, message):
    assembly = assembly_with_limit()
    with pytest.raises(ValueError, match=message):
        inventory_limits(assembly.unit, {'mass_j': quantity})


@pytest.mark.integration
@pytest.mark.parametrize('scale', [1e-3, 1., 1e3])
def test_volume_scaling_preserves_conversion_and_declared_accuracy(scale):
    """Scale all extensive inputs/budgets together; intensive kinetics stay fixed."""
    case = configured_case('scipy')
    original = copy.deepcopy(case)
    case['operation']['initial_mass_kg'] *= scale  # [kg]
    case['operation']['initial_volume']['value'] *= scale  # [declared volume unit]
    for name in case['state']['species']['amounts']:
        case['state']['species']['amounts'][name] *= scale  # [g]
    case['numerical']['absolute_tolerance'] *= scale  # [native state units]; independently maintain integration resolution
    case['numerical']['inventory_acceptance'] = {'mass_j': {'unit': 'kg', 'value': 1e-12*scale}}  # [kg]; fixed accuracy per scaled initial inventory
    assembly = build_bioreactor(case, thermo_path=EXAMPLE/'thermo.json')
    history, = assembly.solve()
    index = assembly.phase.name_species.index('biomass')
    expected = .001*scale*np.exp(-RATE*history.time)  # [kg]; analytical first-order decay
    np.testing.assert_allclose(history.mass_j_liquid0[:, index], expected, rtol=1e-6, atol=1e-10*scale)
    assert assembly.unit.integrator.inventory_limits is not None
    assert case['numerical']['relative_tolerance'] == original['numerical']['relative_tolerance']


@pytest.mark.assimulo
@pytest.mark.integration
def test_cvode_uses_the_same_explicit_acceptance():
    pytest.importorskip('assimulo')
    assembly = assembly_with_limit(backend='assimulo')
    history, = assembly.solve()
    index = assembly.phase.name_species.index('biomass')
    expected = .001*np.exp(-RATE*history.time)  # [kg]; same analytical oracle
    np.testing.assert_allclose(history.mass_j_liquid0[:, index], expected, rtol=1e-6, atol=1e-10)
    assert assembly.unit.integrator.inventory_limits is not None


@pytest.mark.unit
def test_zero_limit_and_unknown_state_are_explicit():
    assembly = assembly_with_limit(value=0.)  # [kg]; exact nonnegativity explicitly required
    limits = inventory_limits(assembly.unit, assembly.inventory_acceptance)
    key, = limits
    states = np.asarray(assembly.unit.create_solver_init_states())[None, :]  # [native state units]
    states[0, assembly.unit.solver_state_collection.slices[key].start] = -1e-20  # [kg]; nonzero violation even though below legacy budget
    with pytest.raises(RuntimeError, match='limit 0 kg'):
        check_inventory(assembly.unit, [0.], states, limits)
    with pytest.raises(ValueError, match='unknown or ambiguous'):
        inventory_limits(assembly.unit, {'unknown': {'unit': 'kg', 'value': 0.}})


@pytest.mark.integration
def test_convergence_retains_failures_and_independent_mass_balance():
    case = configured_case('scipy')
    original = copy.deepcopy(case)
    settings = [
        {'relative_tolerance': 1e-5, 'absolute_tolerance': 1e-10},
        {'relative_tolerance': 1e-9, 'absolute_tolerance': 1e-14},
        {'method': 'misspelled'},
    ]  # [-], [native state units]; four-decade comparison with fixed physical case
    times = np.linspace(0., 2., 21)  # [s]; same reporting grid for every trial
    records = run_convergence(case, EXAMPLE/'thermo.json', settings, report_times=times,
                              reference_index=1, conserved_mass_kg=case['operation']['initial_mass_kg'])
    assert [record['status'] for record in records] == ['succeeded', 'succeeded', 'failed']
    assert records[2]['error_type'] == 'ValueError'
    assert 'scipy method' in records[2]['error']
    assert records[2]['comparison'] is None
    biomass = records[0]['species'].index('biomass')
    expected = .001*np.exp(-RATE*times)  # [kg]; independent first-order solution
    coarse_error = np.max(abs(records[0]['mass_kg'][:, biomass]-expected))  # [kg]
    fine_error = np.max(abs(records[1]['mass_kg'][:, biomass]-expected))  # [kg]
    assert fine_error < coarse_error
    assert records[1]['conservation_error_kg'] < 1e-12  # [kg]; closed, equal-mass conversion fixture
    assert records[0]['comparison']['mass_kg'] > 0.
    assert case == original


@pytest.mark.integration
def test_failed_reference_is_not_replaced_and_acceptance_cannot_be_relaxed():
    case = configured_case('scipy')
    records = run_convergence(case, EXAMPLE/'thermo.json', [{}, {'method': 'bad'}],
                              report_times=[0., 2.], reference_index=1)  # [s]
    assert records[0]['status'] == 'succeeded'
    assert all(record['comparison'] is None for record in records)
    with pytest.raises(ValueError, match='acceptance must remain fixed'):
        run_convergence(case, EXAMPLE/'thermo.json', [{'inventory_acceptance': {}}],
                        report_times=[0., 2.], reference_index=0)


@pytest.mark.integration
@pytest.mark.parametrize('fraction', [1e-6, 1e-3, 1.])
def test_trace_concentration_at_fixed_volume(fraction):
    """Change concentration independently of vessel size over six decades."""
    case = configured_case('scipy')
    case['state']['species']['amounts']['biomass'] *= fraction  # [g]; carrier fills the unchanged 1 kg liquid inventory
    case['numerical']['absolute_tolerance'] = 1e-15  # [native state units]; resolves smallest 1e-9 kg initial inventory
    case['numerical']['inventory_acceptance'] = {'mass_j': {'unit': 'ng', 'value': 1.}}  # [ng]; same physical requirement across concentrations
    assembly = build_bioreactor(case, thermo_path=EXAMPLE/'thermo.json')
    history, = assembly.solve()
    column = assembly.phase.name_species.index('biomass')
    expected = .001*fraction*np.exp(-RATE*history.time)  # [kg]
    np.testing.assert_allclose(history.mass_j_liquid0[:, column], expected, atol=1e-12, rtol=1e-6)




@pytest.mark.unit
@pytest.mark.parametrize('separate', [False, True])
def test_empty_acceptance_is_not_silently_removed(separate):
    case = configured_case('scipy')
    definition = None
    if separate:
        case, definition = split_forward_inputs(case)
    case['numerics' if separate else 'numerical']['inventory_acceptance'] = {}
    with pytest.raises(ValueError, match='nonempty state mapping'):
        build_bioreactor(case, definition, thermo_path=EXAMPLE/'thermo.json')


@pytest.mark.unit
def test_checker_rejects_nonfinite_states_and_misaligned_times():
    assembly = assembly_with_limit()
    states = np.asarray(assembly.unit.create_solver_init_states())[None, :]  # [native state units]
    with pytest.raises(ValueError, match='aligned times'):
        check_inventory(assembly.unit, [0., 1.], states)  # [s]; deliberately mismatched rows
    states[0, 0] = float('nan')  # [kg]; invalid reported mass cannot be accepted
    with pytest.raises(RuntimeError, match='value nan'):
        check_inventory(assembly.unit, [0.], states)
