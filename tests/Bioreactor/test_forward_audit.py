"""Exact optimization invariances and read-only native forward diagnostics.

Small deterministic references do not depend on experimental trajectories.
"""

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from PharmaPy.Metabolic.pathways import ConfiguredPathwayModel, PathwayModelDefinition
from PharmaPy.Bioreactors import build_bioreactor, read_observations
from PharmaPy.Bioreactors.diagnostics import (inventory_report, refinement_report,
                                             validity_report, observation_report)

pytestmark = pytest.mark.integration


# Real fixture construction and independent qualification helpers.
def pathway_model(coefficient=1., names=('first',), scale=1., minimum_norm=False):
    """Construct a unit-capacity model with analytically known growth.

    Parameters
    ----------
    coefficient : float
        Positive primary objective multiplier [-].
    names : tuple[str]
        One or two pathway names.
    scale : float
        Flux coordinate conversion [-]; physical flux equals coordinate/scale.
    minimum_norm : bool
        Select the existing optional norm policy.

    Returns
    -------
    ConfiguredPathwayModel
        Uptake capacity 1 [normalized source/time]; optimum growth 1 [1/time].
    """
    count = len(names)
    exchange = np.array([[-1.] * count, [0.] if count == 1 else [0., 1.]]) / scale  # [source/flux]
    definition = PathwayModelDefinition(
        state_ids=('substrate', 'product'), pathway_ids=names,
        exchange_matrix=exchange, growth_coefficients=np.ones(count)/scale,
        objective_coefficients=np.full(count, coefficient/scale),
        parameters={'capacity': 1.}, uptake_constraints=({'identifier': 'uptake',
            'type': 'constant', 'species': 'substrate', 'maximum_parameter': 'capacity'},),
        secondary_optimization=({'policy': 'scaled-minimum-norm',
            'reference_scales': dict.fromkeys(names, scale)} if minimum_norm else None))
    return ConfiguredPathwayModel(definition)


def native_case(*, rate=0., step=.05, fixed=False, flow=False, pulse=False, domain=False):
    """Build an actual native two-pool conversion, with optional material operations.

    Parameters
    ----------
    rate : float
        First-order conversion constant [1/h], nutrient -> water with unit mass yield.
    step : float
        Reporting/integration step [h].
    fixed : bool
        Use the existing fixed-step backend rather than adaptive SciPy.
    flow, pulse : bool
        Add the analytical flow/sample or pulse schedule, respectively.
    domain : bool
        Declare an illustrative upper substrate domain of 15 mmol/L, not an
        organism assumption. The pulse crosses it and subsequent decay returns.

    Returns
    -------
    NativeBioreactorAssembly
        One litre, 1 g nutrient, 0.5 h duration with real PharmaPy objects.
    """
    folder = Path(__file__).parents[2]/'examples/bioreactors/generic_batch/inputs'
    config = json.loads((folder/'mechanism.json').read_text())
    config['mechanism'] = 'configured-kinetics'
    config['rate_provider'] = None
    config['state']['biomass_kg'] = None
    config['parameters']['rates'] = {'conversion': rate}
    config['model'] = {'kinetics': {'rule_graph': [
        {'identifier': 'substrate', 'expression': {'concentration': 'nutrient'}},
        {'identifier': 'gain', 'expression': {'op': 'multiply', 'args': [
            {'parameter': 'rates.conversion'}, .1, {'ref': 'substrate'}]}},
        {'identifier': 'loss', 'expression': {'op': 'negate', 'args': [{'ref': 'gain'}]}}],
        # 100 g/mol * 1e-3 mol/mmol = 0.1 g/mmol; hence gain in g/(L h).
        'species_mass_rates': {'nutrient': 'loss', 'water': 'gain'}, 'rate_unit': 'g/(L h)'},
        'pathways': {}, 'reconciliation': {}}
    config['flux_time_unit'] = None
    config['operation']['runtime']['value'] = .5  # [h]
    config['operation']['volume_policy'] = 'working-volume'
    config['operation']['mode'] = 'fed-batch' if flow or pulse else 'batch'
    config['numerical']['step']['value'] = step
    if fixed:
        config['numerical'] = dict.fromkeys(config['numerical'])
        config['numerical'].update(backend='fixed-step', step={'unit': 'h', 'value': step})
    events = []
    if flow:
        events = [{'event_type': 'flow', 'time': 0., 'volume_l_h': .1,
                   'concentrations_g_l': {'nutrient': 1.}, 'density_kg_l': 1., 'composition': 'supplied'},
                  {'event_type': 'sample', 'time': .25, 'volume_l': .05}]
    if pulse:
        events = [{'event_type': 'feed', 'time': .25, 'volume_l': .1,
                   'concentrations_g_l': {'nutrient': 20.}, 'density_kg_l': 1., 'volume_basis': 'solution'}]
    config['recipes']['batch'] = events
    if domain:
        config['rate_provider'] = {'type': 'rule-graph',
            'output_units': {'gain': 'g/(L h)', 'loss': 'g/(L h)'},
            'validity_domain': {'substrate': [0., 15.]}, 'extrapolation': 'allow'}
    return build_bioreactor(config, thermo_path=folder/'thermo.json')


# Public-path regressions, independent balances, and failure qualification.
@pytest.mark.parametrize('coefficient', [1e-12, 1e-10, 1., 1e12])
@pytest.mark.parametrize('scale', [1e-3, 1., 1e3])
@pytest.mark.parametrize('minimum_norm', [False, True])
def test_pathway_primary_optimum_survives_secondary_scaling(coefficient, scale, minimum_norm):
    model = pathway_model(coefficient, scale=scale, minimum_norm=minimum_norm)
    solved = model.solve_fluxes({'substrate': 1., 'product': 1.}, biomass=1.)
    # Unique physical optimum: maximize c*v/scale subject to 0 <= v/scale <= 1.
    np.testing.assert_allclose(solved.fluxes/scale, [1.], rtol=1e-8, atol=1e-10)
    assert solved.primary_objective/coefficient == pytest.approx(1., abs=1e-8)


@pytest.mark.parametrize('coefficient', [0., -1.])
def test_zero_and_negative_objectives_keep_their_meaning(coefficient):
    solved = pathway_model(coefficient).solve_fluxes({'substrate': 1., 'product': 1.}, biomass=1.)
    np.testing.assert_allclose(solved.fluxes, 0., atol=1e-10)


def test_source_intervals_are_naming_invariant_and_do_not_change_selection():
    results = []
    for names in [('first', 'second'), ('second', 'first')]:
        model = pathway_model(names=names)
        original = model.solve_fluxes({'substrate': 1., 'product': 1.}, biomass=1.)
        audited = model.solve_fluxes({'substrate': 1., 'product': 1.}, biomass=1., source_ranges=True)
        np.testing.assert_array_equal(original.fluxes, audited.fluxes)
        assert audited.source_intervals['growth'] == pytest.approx((1., 1.))
        assert audited.source_intervals['exchange:substrate'] == pytest.approx((-1., -1.))
        assert audited.source_intervals['exchange:product'] == pytest.approx((0., 1.))
        results.append(audited.extracellular_rates[1])
    assert results == [1., 0.]  # Existing selection is preserved, not silently replaced.


def test_source_ranges_preserve_minimum_norm_policy_and_are_opt_in():
    model = pathway_model(names=('one', 'two'), minimum_norm=True)
    plain = model.solve_fluxes({'substrate': 1., 'product': 1.}, biomass=1.)
    checked = model.solve_fluxes({'substrate': 1., 'product': 1.}, biomass=1., source_ranges=True)
    np.testing.assert_allclose(plain.fluxes, [.5, .5], atol=1e-8)
    np.testing.assert_array_equal(plain.fluxes, checked.fluxes)
    assert 'source_intervals' not in plain.diagnostics()
    assert checked.diagnostics()['source_intervals']['exchange:product'] == pytest.approx((0., 1.))


@pytest.mark.parametrize('multiplier', [1e-12, 1., 1e12])
def test_unequal_flux_coordinates_keep_all_physical_objective_terms(multiplier):
    # Independent capacities uniquely imply fluxes 0.5/1e10 and 1, with growth 1.5.
    conversion = 1e10  # [-]; coordinate change, not a biological coefficient.
    definition = PathwayModelDefinition(
        ('A', 'B'), ('one', 'two'), np.diag([-conversion, -1.]),
        np.array([conversion, 1.]), multiplier*np.array([conversion, 1.]),
        parameters={'capA': .5, 'capB': 1.},
        uptake_constraints=tuple({'identifier': name, 'type': 'constant', 'species': name,
            'maximum_parameter': 'cap'+name} for name in ('A', 'B')))
    result = ConfiguredPathwayModel(definition).solve_fluxes([1., 1.], biomass=1., source_ranges=True)
    np.testing.assert_allclose(result.extracellular_rates, [-.5, -1.], atol=1e-8)
    assert result.primary_objective/multiplier == pytest.approx(1.5, abs=1e-8)
    assert result.source_intervals['growth'] == pytest.approx((1.5, 1.5), abs=1e-8)


def test_native_inventory_ledger_accounts_for_feed_and_sample_without_mutation():
    assembly = native_case(flow=True)
    histories = assembly.solve()
    saved = [h.mass_j_liquid0.copy() for h in histories]
    names = list(assembly.phase.name_species)
    feed = np.array([.0001 if name == 'nutrient' else .0999 for name in names])/3600.  # [kg/s]
    ledger = {kind: [np.zeros_like(h.mass_j_liquid0) for h in histories]
              for kind in ('inlet', 'outlet', 'transfer', 'biological')}
    ledger['inlet'] = [np.tile(feed, (len(h.time), 1)) for h in histories]
    initial = np.array([.001 if name == 'nutrient' else .999 for name in names])  # [kg]
    jump = -(initial + feed*900.) * (.05/1.025)  # [kg]; exact well-mixed sample at 0.25 h
    budget = np.full(len(names), 1e-10)  # [kg]; much tighter than physical inventories, above ODE roundoff
    error = np.zeros(len(names))  # [kg]; constant rates integrate exactly
    result = inventory_report(assembly, histories, ledger=ledger, jumps_kg=[jump],
        budget_kg=budget, quadrature_error_kg=error,
        invariants={'mass': dict.fromkeys(names, 1.)}, invariant_budget={'mass': 1e-10})
    assert result['status'] == 'passed'
    assert result['invariants']['mass']['status'] == 'passed'
    wrong_jump = [np.zeros_like(jump)]  # [kg]; intentionally omits the real sample.
    incorrect = inventory_report(assembly, histories, ledger=ledger, jumps_kg=wrong_jump,
        budget_kg=budget, quadrature_error_kg=error,
        invariants={'mass': dict.fromkeys(names, 1.)}, invariant_budget={'mass': 1e-10})
    assert incorrect['status'] == 'failed'
    assert incorrect['invariants']['mass']['status'] == 'failed'
    bad = deepcopy(ledger)
    bad['inlet'][0][:, names.index('nutrient')] *= 2.
    assert inventory_report(assembly, histories, ledger=bad, jumps_kg=[jump],
                            budget_kg=budget, quadrature_error_kg=error)['status'] == 'failed'
    assert inventory_report(assembly, histories)['status'] == 'not_assessable'
    assert inventory_report(assembly, histories, ledger=ledger)['status'] == 'not_assessable'
    assert inventory_report(assembly, histories, ledger=ledger, jumps_kg=[jump],
        budget_kg=budget, quadrature_error_kg=np.ones(len(names)))['status'] == 'unresolved_quadrature'
    for before, history in zip(saved, histories):
        np.testing.assert_array_equal(before, history.mass_j_liquid0)


def test_refinement_compares_real_fixed_step_native_solutions():
    runs = [native_case(rate=1., step=step, fixed=True).solve() for step in (.1, .05, .025)]
    # First-order discretization of n'=-n: refinement shrinks the actual error.
    report = refinement_report(*runs, scales={'mass_j_liquid0': [.001, 1.]}, tolerance=.1)
    assert report['passed']
    assert report['fields']['mass_j_liquid0']['decreasing']
    assert not refinement_report(*runs, scales={'mass_j_liquid0': [.001, 1.]}, tolerance=1e-8)['passed']
    expected = .001*np.exp(-.5)  # [kg], independent exact concentration conversion at constant volume
    errors = [abs(run[-1].mass_j_liquid0[-1, 0]-expected) for run in runs]  # [kg]
    assert errors[2] < errors[1] < errors[0]
    incompatible = deepcopy(runs[2])
    for history in incompatible:
        history.mass_j_liquid0 = history.mass_j_liquid0[:, :1]
    with pytest.raises(ValueError, match='column order and count'):
        refinement_report(runs[0], runs[1], incompatible,
                          scales={'mass_j_liquid0': [.001, 1.]}, tolerance=.1)


def test_validity_retains_intermediate_violation_and_does_not_mutate_provider():
    assembly = native_case(rate=4., pulse=True, domain=True)
    histories = assembly.solve()
    provider = assembly.mechanism.last_provider_result
    assert provider.inside_validity_domain
    report = validity_report(assembly, histories)
    assert report['status'] == 'outside'
    assert any(item['time_s'] == 900. for item in report['violations'])
    assert assembly.mechanism.last_provider_result is provider
    plain = native_case()
    assert validity_report(plain, plain.solve())['status'] == 'unspecified'


def test_observation_report_matches_times_and_keeps_unobserved_tail_unscored():
    assembly = native_case(rate=1.)
    declaration = {'schema_version': 1, 'time_unit': 'h', 'fields': {'substrate': {
        'species': 'nutrient', 'quantity': 'concentration', 'unit': 'g/L', 'event_side': 'pre',
        'time_key': 't', 'value_key': 'y'}}}
    observations = read_observations(declaration, {'substrate': {'t': [0., .25], 'y': [1., np.exp(-.25)]}})
    histories = assembly.solve(report_times=[0., 900.])
    report = observation_report(assembly, histories, declaration, observations,
                                windows={'late': (1200., 1800.)})
    field = report['fields']['substrate']
    assert field['metrics']['all']['rmse'] < 1e-8
    assert field['metrics']['late']['status'] == 'no_observations'
    assert field['unobserved_tail']
    assert report['statistical_acceptance'] == 'not_assessed'
    with pytest.raises(ValueError, match="'all' is reserved"):
        observation_report(assembly, histories, declaration, observations, windows={'all': (0., 1.)})


@pytest.mark.parametrize('side, expected', [('pre', 1.), ('post', 3./1.1)])
def test_observation_report_preserves_event_side(side, expected):
    assembly = native_case(pulse=True)
    declaration = {'schema_version': 1, 'time_unit': 'h', 'fields': {'substrate': {
        'species': 'nutrient', 'quantity': 'concentration', 'unit': 'g/L', 'event_side': side,
        'time_key': 't', 'value_key': 'y'}}}
    # 1 g in 1 L before; add 2 g in 0.1 L, then 3 g in 1.1 L after.
    observations = read_observations(declaration, {'substrate': {'t': [.25], 'y': [expected]}})
    histories = assembly.solve(report_times=[900.])
    field = observation_report(assembly, histories, declaration, observations)['fields']['substrate']
    assert field['event_side'] == side
    assert field['metrics']['all']['rmse'] < 1e-8
    assert field['unobserved_head'] and field['unobserved_tail']
