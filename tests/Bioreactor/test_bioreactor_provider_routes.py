"""Provider substitution preserves native balances in all three mechanisms."""

import json
from pathlib import Path

import numpy as np
import pytest

from PharmaPy.Bioreactors import build_bioreactor

ROOT = Path(__file__).parents[2] / 'examples/bioreactors'


def inputs(kind):
    folder = (Path(__file__).parent/'fixtures/reconciliation/analytic_balances/inputs/consistent'
              if kind == 'reconciled' else ROOT/'generic_batch/inputs')
    case = json.loads((folder/'case.json').read_text())
    model = json.loads((folder/'mechanism.json').read_text())
    if kind == 'direct':
        model['mechanism'] = 'configured-kinetics'
        model['state']['biomass_kg'] = None
        model['flux_time_unit'] = None
        model['model'] = {'pathways': {}, 'reconciliation': {}, 'kinetics': {
            'rule_graph': [{'identifier': 'consumption', 'expression': {
                'op': 'multiply', 'args': [-.001, {'concentration': 'nutrient'}]}}],
            'species_mass_rates': {'nutrient': 'consumption'}, 'rate_unit': 'g/(L h)'}}
    return case, model, folder/'thermo.json'


def replacement(kind, model):
    """Declare equivalent rule-graph rates through the public provider route.

    Parameters
    ----------
    kind : str
        Direct kinetics, pathway capacity or reconciled-culture route.
    model : dict
        Input declaration modified in place; units remain explicit in outputs.

    Returns
    -------
    dict
        Rule-graph provider declaration, including an explicit concentration
        validity rule [mmol/L].
    """
    if kind == 'pathway':
        model['model']['pathways']['uptake_constraints'] = [
            {'identifier': 'nutrient_uptake', 'type': 'provider', 'species': 'nutrient', 'output': 'capacity'}]
        # [mol/(kgDW h)] Constant capacity matches the baseline uptake bound.
        name, unit, intercept, slope = 'capacity', 'mol/(kgDW h)', 2., 0.
    elif kind == 'direct':
        # [g/(L h)], [g/(mmol h)] Same linear consumption as the direct fixture.
        name, unit, intercept, slope = 'consumption', 'g/(L h)', 0., -.001
    else:
        base = model['rate_provider']
        name = model['model']['kinetics']['kinetic_outputs']['exchange_fluxes']['uptake']
        unit = base['output_units'][name]
        intercept, slope = 2., 0.
    section = model['model']['pathways' if kind == 'pathway' else 'kinetics']
    rules = section.setdefault('rule_graph', [])
    expression = {'op': 'add', 'args': [intercept,
        {'op': 'multiply', 'args': [slope, {'concentration': 'nutrient'}]}]}
    rule = next((item for item in rules if item['identifier'] == name), None)
    if rule is None:
        rules.append({'identifier': name, 'expression': expression})
    else:
        rule['expression'] = expression
    rules.append({'identifier': 'domain_nutrient',
                  'expression': {'concentration': 'nutrient'}})
    units = dict(base['output_units']) if kind == 'reconciled' else {name: unit}
    provider = {'type': 'rule-graph', 'identifier': 'replacement-rules',
                'output_units': units, 'validity_domain': {'domain_nutrient': [0., 100.]},
                'extrapolation': 'error'}
    model['rate_provider'] = provider
    return provider


@pytest.mark.parametrize('kind', ['direct', 'pathway', 'reconciled'])
def test_all_routes_preserve_equivalent_provider_trajectories(kind):
    case, model, thermo = inputs(kind)
    baseline = build_bioreactor(case, model, thermo)
    before = baseline.solve(verbose=False)
    replacement(kind, model)
    swapped = build_bioreactor(case, model, thermo)
    after = swapped.solve(verbose=False)
    assert type(swapped.unit) is type(baseline.unit)
    assert swapped.events == baseline.events
    for a, b in zip(before, after):
        np.testing.assert_allclose(a.mass_j_liquid0, b.mass_j_liquid0, rtol=1e-12, atol=1e-14)
    assert swapped.mechanism.last_provider_result.inside_validity_domain


@pytest.mark.parametrize('kind', ['direct', 'pathway', 'reconciled'])
@pytest.mark.parametrize('defect', ['domain', 'units'])
def test_provider_failures_are_not_hidden(kind, defect):
    case, model, thermo = inputs(kind)
    provider = replacement(kind, model)
    if defect == 'domain':
        provider['validity_domain']['domain_nutrient'] = [200., 300.]
    else:
        for name in provider['output_units']:
            provider['output_units'][name] = 'incorrect-unit'
    with pytest.raises(ValueError, match='domain|units'):
        build_bioreactor(case, model, thermo).solve(verbose=False)


def test_negative_uptake_is_rejected_not_clipped():
    case, model, thermo = inputs('pathway')
    provider = replacement('pathway', model)
    next(rule for rule in model['model']['pathways']['rule_graph']
         if rule['identifier'] == 'capacity')['expression'] = -1.
    with pytest.raises(ValueError, match='nonnegative'):
        build_bioreactor(case, model, thermo).solve(verbose=False)


def test_provider_capacity_converts_to_declared_pathway_basis():
    case, model, thermo = inputs('pathway')
    replacement('pathway', model)
    expected = build_bioreactor(case, model, thermo).solve(verbose=False)
    pathways = model['model']['pathways']
    pathways['flux_basis'] = {
        'exchange': {'amount_unit': 'umol', 'normalization': 'gDW', 'time_unit': 'h'},
        'growth': {'amount_unit': '1', 'normalization': 'none', 'time_unit': 'h'}}
    pathways['exchange_matrix'] = (np.asarray(pathways['exchange_matrix'])*1000.).tolist()
    observed = build_bioreactor(case, model, thermo).solve(verbose=False)
    for a,b in zip(expected,observed):
        np.testing.assert_allclose(a.mass_j_liquid0,b.mass_j_liquid0,rtol=1e-12,atol=1e-14)
