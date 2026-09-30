"""Provider substitution preserves native balances in all three mechanisms."""

import json
from pathlib import Path

import numpy as np
import pytest

from PharmaPy.Bioreactors import build_bioreactor

ROOT = Path(__file__).parents[2] / 'examples/bioreactors'


def inputs(kind):
    folder = (ROOT/'generic_fed_batch/reconciliation/inputs/consistent'
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
    if kind == 'pathway':
        model['model']['pathways']['uptake_constraints'] = [
            {'identifier': 'nutrient_uptake', 'type': 'provider', 'species': 'nutrient', 'output': 'capacity'}]
        name, unit, intercept, slope = 'capacity', 'mol/(kgDW h)', 2., 0.
    elif kind == 'direct':
        name, unit, intercept, slope = 'consumption', 'g/(L h)', 0., -.001
    else:
        base = model['rate_provider']
        name = model['model']['kinetics']['kinetic_outputs']['exchange_fluxes']['uptake']
        unit = base['output_units'][name]
        intercept, slope = 2., 0.
    provider = {'type': 'affine-surrogate', 'identifier': 'test-affine',
        'inputs': {'s': {'source': 'concentration', 'name': 'nutrient', 'unit': 'mmol/L'}},
        'outputs': {name: {'unit': unit, 'intercept': intercept, 'coefficients': {'s': slope}}},
        'validity_domain': {'s': [0., 100.]}, 'extrapolation': 'error'}
    model['rate_provider'] = ({'type': 'hybrid', 'providers': [base, provider],
        'default_provider': base.get('identifier', 'rule-graph'),
        'output_sources': {name: 'test-affine'}} if kind == 'reconciled' else provider)
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
@pytest.mark.parametrize('defect', ['domain', 'units', 'input-units'])
def test_provider_failures_are_not_hidden(kind, defect):
    case, model, thermo = inputs(kind)
    provider = replacement(kind, model)
    if defect == 'domain':
        provider['validity_domain']['s'] = [200., 300.]
    elif defect == 'input-units':
        provider['inputs']['s']['unit'] = 'g/L'
    else:
        for output in provider['outputs'].values():
            output['unit'] = 'incorrect-unit'
    with pytest.raises(ValueError, match='domain|units'):
        build_bioreactor(case, model, thermo).solve(verbose=False)


def test_direct_surrogate_does_not_require_unused_kinetic_equations():
    case, model, thermo = inputs('direct')
    replacement('direct', model)
    model['model']['kinetics']['rule_graph'] = []
    assembly = build_bioreactor(case, model, thermo)
    result = assembly.solve(verbose=False)
    assert result[-1].mass_j_liquid0[-1,0] < result[0].mass_j_liquid0[0,0]


def test_negative_uptake_is_rejected_not_clipped():
    case, model, thermo = inputs('pathway')
    provider = replacement('pathway', model)
    provider['outputs']['capacity']['intercept'] = -1.
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
