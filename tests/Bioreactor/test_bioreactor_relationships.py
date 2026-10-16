"""Independent equations and cross-family checks for declarative relationships."""

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from PharmaPy.Bioreactors.relationships import expand_relationships
from PharmaPy.Bioreactors.culture import evaluate_rule_graph, _validate_graph
from PharmaPy.Bioreactors.input_format import normalize_forward_inputs
from PharmaPy.Metabolic.pathways import PathwayModelDefinition, ConfiguredPathwayModel
from PharmaPy.Metabolic.network import MetabolicNetworkDefinition, ReactionDefinition
from PharmaPy.Metabolic.closures.base import MetabolicEnvironment
from PharmaPy.Metabolic.closures.reconciled import RateReconciledMFAClosure, ReconciliationTarget


def relation(name, **inputs):
    return {'relationship': name, 'inputs': inputs}


def evaluate(expression, parameters=None):
    model = {'rule_graph': [{'identifier': 'answer', 'expression': expression}]}
    original = deepcopy(model)
    expanded = expand_relationships(model, parameters or {})
    assert model == original
    assert expand_relationships(expanded, parameters or {}) == expanded
    definition = SimpleNamespace(**expanded, parameters=parameters or {}, constants={})
    _validate_graph(definition, ['answer'])
    return evaluate_rule_graph(definition, SimpleNamespace(), {})['answer']


@pytest.mark.parametrize('substrate', [0., .001, 1., 1000.])
def test_saturating_uptake_against_equation(substrate):
    expression = relation('saturating-uptake', maximum={'parameter': 'v'},
                          substrate=substrate, half_saturation={'parameter': 'k'})
    assert evaluate(expression, {'v': 3., 'k': 2.}) == pytest.approx(3*substrate/(2+substrate))


@pytest.mark.parametrize('count', [1, 2, 7, 31])
def test_arbitrary_substrate_yields_and_inhibitors(count):
    terms = [{'rate': float(i+1), 'yield': 1./(i+1)} for i in range(count)]
    assert evaluate(relation('yield-sum', terms=terms)) == pytest.approx(count)
    inputs = [{'concentration': float(i+1), 'constant': float(i+1)} for i in range(count)]
    assert evaluate(relation('additive-inhibition', rate=4., inhibitors=inputs)) == pytest.approx(4/(1+count))


@pytest.mark.parametrize('expression, expected', [
    (relation('capacity-limited-rate', rate=3., capacity=2.), 1.2),
    (relation('net-production', production=3., consumption=2., **{'yield': .5}), -.5),
    (relation('maintenance-uptake', growth=.4, maintenance=.1, **{'yield': .5}), .9),
    (relation('linear-transfer', coefficient=2., equilibrium=1., concentration=3.), -4.),
])
def test_independent_relationship_answers(expression, expected):
    assert evaluate(expression) == pytest.approx(expected)


def test_composition_and_explicit_equations():
    uptake = relation('saturating-uptake', maximum=4., substrate=2., half_saturation=2.)
    inhibited = relation('additive-inhibition', rate=uptake,
                          inhibitors=[{'concentration': 3., 'constant': 1.}])
    assert evaluate({'op': 'multiply', 'args': [inhibited, 2.]}) == pytest.approx(1.)


@pytest.mark.parametrize('bad', [0., -1., float('nan'), float('inf'), True, {'parameter': 'missing'}])
def test_invalid_affinity_rejected(bad):
    with pytest.raises(ValueError):
        evaluate(relation('saturating-uptake', maximum=2., substrate=1., half_saturation=bad))


@pytest.mark.parametrize('expression', [
    relation('unknown', rate=1.),
    relation('yield-sum', terms=[]),
    relation('yield-sum', terms=[{'rate': 1., 'yield': -.1}]),
    relation('additive-inhibition', rate=1., inhibitors=[{'concentration': 1., 'constant': 0.}]),
    relation('saturating-uptake', maximum=2., substrate=1.),
    relation('saturating-uptake', maximum=-2., substrate=1., half_saturation=1.),
    {'relationship': 'yield-sum', 'inputs': {'terms': []}, 'fallback': 1.},
])
def test_incomplete_or_unsupported_relationships_fail(expression):
    with pytest.raises(ValueError):
        evaluate(expression)


def test_conflicting_ownership_and_forward_dependencies_fail():
    with pytest.raises(ValueError, match='unique'):
        expand_relationships({'rule_graph': [dict(identifier='a', expression=1.)]*2}, {})
    with pytest.raises(ValueError, match='forward references'):
        evaluate(relation('capacity-limited-rate', rate={'ref': 'unknown'}, capacity=1.))


@pytest.mark.parametrize('factor', [1e-3, 1., 1e3])
def test_equivalent_concentration_units(factor):
    assert evaluate(relation('saturating-uptake', maximum=4., substrate=2.*factor,
                             half_saturation=3.*factor)) == pytest.approx(1.6)


@pytest.mark.parametrize('mode', ['direct', 'reconciled'])
@pytest.mark.parametrize('run', ['', 'F11'])
def test_examples_compile_without_mutation(mode, run):
    root = Path(__file__).parents[2] / 'examples/bioreactors/ecoli_ye_fed_batch'
    folder = root / ('inputs' if mode == 'direct' else 'reconciliation/inputs') / run
    case = json.loads((folder/'case.json').read_text())
    mechanism = json.loads((folder/'mechanism.json').read_text())
    before = deepcopy(mechanism)
    _, normalized = normalize_forward_inputs(case, mechanism)
    assert mechanism == before
    assert '"relationship"' not in json.dumps(normalized)
    assert '"relationship"' in json.dumps(mechanism)


@pytest.mark.parametrize('count', [1, 2, 8, 32])
def test_pathway_dimension_independence_known_optimum(count):
    definition = PathwayModelDefinition(
        state_ids=('arbitrary_pool',), pathway_ids=tuple(f'route_{i}' for i in range(count)),
        exchange_matrix=-np.ones((1, count)), growth_coefficients=np.arange(1., count+1),
        objective_coefficients=np.arange(1., count+1),
        parameters={'cap': 2., 'affinity': 1.}, uptake_constraints=({
            'identifier': 'supply', 'type': 'monod', 'species': 'arbitrary_pool',
            'maximum_parameter': 'cap', 'half_saturation_parameter': 'affinity'},))
    solution = ConfiguredPathwayModel(definition).solve_fluxes({'arbitrary_pool': 1.}, biomass=1.)
    assert solution.growth_rate == pytest.approx(count)
    assert solution.fluxes[-1] == pytest.approx(1.)
    assert solution.bound_violation_inf <= 1e-8


@pytest.mark.parametrize('count', [3, 9, 33])
@pytest.mark.parametrize('conflicting', [False, True])
def test_reconciliation_dimension_independence_known_projection(count, conflicting):
    pools = [f'pool_{i}' for i in range(count-1)]
    ids = [f'reaction_{i}' for i in range(count)]
    reactions = []
    for i, name in enumerate(ids):
        stoichiometry = {}
        if i:
            stoichiometry[pools[i-1]] = -1.
        if i < count-1:
            stoichiometry[pools[i]] = 1.
        reactions.append(ReactionDefinition(name, stoichiometry, 0., 10.))
    network = MetabolicNetworkDefinition('configured-chain', '1', pools, reactions,
                                         {ids[-1]: 1.}, (ids[0], ids[-1]))
    # Produce one target through the new input layer; the oracle is the mean of
    # two equally weighted targets along the independently known common flux.
    first = evaluate(relation('saturating-uptake', maximum=2., substrate=1., half_saturation=1.))
    last = 3. if conflicting else 1.
    solution = RateReconciledMFAClosure(network, ()).solve(
        MetabolicEnvironment({}), targets=[ReconciliationTarget(ids[0], first, 1.),
                                         ReconciliationTarget(ids[-1], last, 1.)], internal_scale=1.)
    np.testing.assert_allclose(solution.fluxes, (first+last)/2, atol=1e-7, rtol=0.)
    assert solution.mass_balance_residual_inf < 1e-8
