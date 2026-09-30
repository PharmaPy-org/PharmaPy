"""Physical unit equivalence of the opt-in reconciliation residual metric."""

from copy import deepcopy

import numpy as np
import pytest

from PharmaPy.Bioreactors import build_bioreactor
from PharmaPy.Bioreactors.inventory import LinearInventoryAvailability
from test_bioreactor_qualification import network_load as load, network_observed as observed


def configured(variant='conflicting'):
    case, config, thermo = load(variant)
    kinetics = config['model']['kinetics']
    scales = {}
    for name in kinetics['kinetic_outputs']['exchange_fluxes']:
        group = 'product' if name == 'p' else 'growth' if name == 'g' else 'exchange'
        scales[name] = dict(value=1., basis=dict(kinetics['flux_basis'][group]))
    config['model']['reconciliation'].update(reconciliation_policy='unit-scaled',
        reconciliation_scaling={'target_scales': scales})
    return case, config, thermo


def convert_coordinates(config, variant):
    """Independent coordinate changes; keep the physical reference metric fixed."""
    config = deepcopy(config)
    basis = config['model']['kinetics']['flux_basis']
    factors = {item['identifier']: 1. for item in config['network']['reactions']}
    if variant == 'product-umol':
        basis['product']['amount_unit'] = 'umol'
        factors['p'] = 1000.
    elif variant == 'product-mg':
        basis['product']['amount_unit'] = 'mg'
        factors['p'] = 100.  # 100 g/mol: one mmol is 100 mg.
    elif variant == 'growth-umol':
        basis['growth']['amount_unit'] = 'umol'
        basis['growth']['amount_per_million_cells'] *= 1000.
        factors['g'] = 1000.
    elif variant == 'exchange-umol':
        basis['exchange']['amount_unit'] = 'umol'
        for name in ('ua', 'ub', 'q', 'x', 'y'):
            factors[name] = 1000.
    else:
        for declaration in basis.values():
            if variant == 'hours':
                declaration['time_unit'] = 'h'
            elif variant == 'cells':
                declaration['normalization'] = 'cell'
            else:
                raise ValueError(variant)
        factors = dict.fromkeys(factors, 1./24. if variant == 'hours' else 1e-6)
    for name in config['model']['kinetics']['kinetic_outputs']['exchange_fluxes']:
        group = 'product' if name == 'p' else 'growth' if name == 'g' else 'exchange'
        b = basis[group]
        config['rate_provider']['output_units'][name] = f"{b['amount_unit']}/({b['normalization']} {b['time_unit']})"
        config['parameters']['rates'][name] *= factors[name]
    for reaction in config['network']['reactions']:
        factor = factors[reaction['identifier']]
        reaction['stoichiometry'] = {name: value/factor for name, value in reaction['stoichiometry'].items()}
        reaction['lower'] *= factor
        reaction['upper'] *= factor
    config['network']['objective'] = {name: value/factors[name] for name, value in config['network']['objective'].items()}
    return config, factors


def targets(config):
    return {name: config['parameters']['rates'][name]
            for name in config['model']['kinetics']['kinetic_outputs']['exchange_fluxes']}


VARIANTS = ['product-umol', 'product-mg', 'growth-umol', 'exchange-umol', 'hours', 'cells']


@pytest.mark.parametrize('variant', VARIANTS)
@pytest.mark.parametrize('limited', [False, True])
def test_primary_projection_preserves_physical_units_and_inventory_limits(variant, limited):
    case, config, thermo = configured()
    changed, factors = convert_coordinates(config, variant)
    solutions = []
    for definition, conversion in [(config, dict.fromkeys(factors, 1.)), (changed, factors)]:
        mechanism = build_bioreactor(case, definition, thermo).mechanism
        ids = mechanism.network.reaction_ids
        matrix = np.array([[-1./conversion[name] if name == 'ua' else 0. for name in ids]])
        availability = LinearInventoryAvailability.boundary([0.], matrix, [.6]) if limited else None
        solution = mechanism._solve(targets(definition), availability)
        physical = solution.fluxes / np.array([conversion[name] for name in ids])
        solutions.append(physical)
        if limited:
            assert physical[ids.index('ua')] <= .6 + 1e-8
        for name, scale in mechanism.target_scales.items():
            assert scale == pytest.approx(conversion[name])
    np.testing.assert_allclose(solutions[0], solutions[1], atol=1e-7, rtol=1e-7)


@pytest.mark.parametrize('variant', VARIANTS)
def test_native_growth_product_and_feed_trajectories_preserve_units(variant):
    case, config, thermo = configured('growth')
    changed, _ = convert_coordinates(config, variant)
    assemblies = [build_bioreactor(case, definition, thermo) for definition in (config, changed)]
    histories = [assembly.solve() for assembly in assemblies]
    assert len(histories[0]) == len(histories[1])
    for left, right in zip(*histories):
        np.testing.assert_array_equal(left.time, right.time)
        np.testing.assert_allclose(observed(assemblies[0], left), observed(assemblies[1], right),
                                   atol=1e-7, rtol=1e-6)


def test_unit_scaled_reference_preserves_legacy_optimum_and_copy():
    case, config, thermo = configured()
    selected = build_bioreactor(case, config, thermo).mechanism
    legacy = deepcopy(config)
    legacy['model']['reconciliation'].update(reconciliation_policy='unweighted', reconciliation_scaling=None)
    old = build_bioreactor(case, legacy, thermo).mechanism
    expected = old._solve(targets(config))
    for mechanism in (selected, deepcopy(selected)):
        np.testing.assert_allclose(mechanism._solve(targets(config)).fluxes, expected.fluxes, atol=1e-9)
        zero = mechanism._solve(dict.fromkeys(targets(config), 0.))
        np.testing.assert_allclose(zero.fluxes, 0., atol=1e-9)


@pytest.mark.parametrize('bad', ['missing', 'extra', 'null-basis', 'zero', 'negative', 'legacy-floor', 'no-basis'])
def test_unit_scaled_policy_rejects_ambiguous_or_invalid_reference_metrics(bad):
    case, config, thermo = configured()
    policy = config['model']['reconciliation']['reconciliation_scaling']
    if bad == 'missing':
        policy['target_scales'].pop('p')
    elif bad == 'extra':
        policy['target_scales']['unknown'] = policy['target_scales']['p']
    elif bad == 'null-basis':
        policy['target_scales']['p']['basis'] = None
    elif bad in ('zero', 'negative'):
        policy['target_scales']['p']['value'] = 0. if bad == 'zero' else -1.
    elif bad == 'legacy-floor':
        policy['target_floor'] = dict(value=1., basis=None)
    else:
        config['model']['kinetics']['flux_basis'] = {}
    with pytest.raises(ValueError):
        build_bioreactor(case, config, thermo)


def test_specific_growth_reference_scale_converts_hours_without_inventing_cell_mass():
    case, config, thermo = configured('growth')
    specific = dict(amount_unit='1', normalization='none', time_unit='day')
    config['model']['kinetics']['flux_basis']['growth'] = specific
    config['model']['reconciliation']['reconciliation_scaling']['target_scales']['g'] = dict(value=.5, basis=dict(specific))
    config['parameters']['rates']['g'] /= 2.
    config['rate_provider']['output_units']['g'] = '1/day'
    for reaction in config['network']['reactions']:
        if reaction['identifier'] == 'g':
            reaction['stoichiometry'] = {name: value*2. for name, value in reaction['stoichiometry'].items()}
            reaction['lower'] /= 2.
            reaction['upper'] /= 2.
    changed, factors = convert_coordinates(config, 'hours')
    changed['rate_provider']['output_units']['g'] = '1/h'
    solutions = []
    for definition, multiplier in ((config, 1.), (changed, 24.)):
        mechanism = build_bioreactor(case, definition, thermo).mechanism
        solution = mechanism._solve(targets(definition))
        solutions.append(solution.fluxes * multiplier)
        assert mechanism.target_scales['g'] * multiplier == pytest.approx(.5)
    np.testing.assert_allclose(solutions[0], solutions[1], atol=1e-8)


def test_reference_weights_are_preserved_and_untargeted_rates_are_excluded():
    case, config, thermo = configured()
    policy = config['model']['reconciliation']
    policy['untargeted_exchanges'] = ['q']
    policy['reconciliation_scaling']['target_scales'].pop('q')
    policy['reconciliation_scaling']['target_scales']['p']['value'] = .2
    converted, factors = convert_coordinates(config, 'product-mg')
    baseline = build_bioreactor(case, config, thermo).mechanism
    other = build_bioreactor(case, converted, thermo).mechanism
    assert other.target_scales['p'] == pytest.approx(20.)
    first = baseline._solve(targets(config))
    changed_targets = dict(targets(converted), q=123456.)
    second = other._solve(changed_targets)
    np.testing.assert_allclose(first.fluxes, second.fluxes / np.array([factors[n] for n in second.reaction_ids]),
                               atol=1e-8)


@pytest.mark.parametrize('variant', ['growth-umol', 'hours'])
def test_primary_metric_preserves_growth_dependent_inventory_limits(variant):
    from PharmaPy.Bioreactors.inventory import InventoryAvailability
    case, config, thermo = configured('growth')
    changed, factors = convert_coordinates(config, variant)
    answers = []
    for definition, conversion in ((config, dict.fromkeys(factors, 1.)), (changed, factors)):
        mechanism = build_bioreactor(case, definition, thermo).mechanism
        ids = mechanism.network.reaction_ids
        matrix = np.array([[-1./conversion[name] if name == 'ua' else 0. for name in ids]])
        availability = InventoryAvailability([.1], matrix, [0.], viable=1., step_day=.25,
            cell_flux_scale=1., growth=.15, death=.05, growth_index=ids.index('g'),
            growth_conversion=mechanism.unit_converter.growth_scale)
        solution = mechanism._solve(targets(definition), availability)
        assert abs(availability.residual(solution.fluxes)[0]) < 1e-7
        answers.append(solution.fluxes / np.array([conversion[name] for name in ids]))
    np.testing.assert_allclose(answers[0], answers[1], atol=1e-7, rtol=1e-6)
