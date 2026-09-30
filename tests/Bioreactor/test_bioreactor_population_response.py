"""Optional accepted-flux responses preserve the native population balances."""

from copy import deepcopy

import numpy as np
import pytest

from PharmaPy.Bioreactors import build_bioreactor
from test_bioreactor_qualification import network_load as load
from test_bioreactor_rate_providers import _definition, _snapshot, _rule_provider
from PharmaPy.Bioreactors.rate_providers import build_rate_provider


def configured(backend='scipy', mode='batch', expression=0.):
    case, config, thermo = load('growth')
    case['numerics']['backend'] = backend
    case['numerics']['step']['value'] = .01
    if backend=='fixed-step':
        for name in case['numerics']:
            if name not in {'backend','step'}:case['numerics'][name]=None
    case['operation']['runtime']['value'] = .1
    case['operation']['mode'] = mode
    config['recipes']['benchmark'] = (config['recipes']['benchmark'][:1]
                                      if mode=='fed-batch' else [])
    config['model']['kinetics']['kinetic_outputs']['death_increment'] = {
        'unit': '1/day', 'expression': expression}
    return case, config, thermo


def response():
    return {'op':'multiply','args':[.2,{'op':'maximum','args':[0.,
        {'op':'subtract','args':[2.,{'accepted_flux':'ua'}]}]}]}


@pytest.mark.parametrize('backend',['scipy','fixed-step'])
@pytest.mark.parametrize('mode',['batch','fed-batch'])
def test_accepted_flux_survival_has_known_population_balance(backend,mode):
    case, config, thermo = configured(backend,mode,response())
    assembly = build_bioreactor(case,config,thermo)
    history = assembly.solve()[-1]
    # Accepted ua=1.5, growth=.3/2=.15, basal death=.05, increment=.1.
    np.testing.assert_allclose(history.viable_cells_million_liquid0,1.,atol=2e-7)
    expected = (.1*(1.-np.exp(-.15*.01))/.01 if backend=='fixed-step' else .015)
    assert history.dead_cells_million_liquid0[-1]==pytest.approx(expected,abs=2e-7)
    rates = assembly.mechanism.last_population_rates
    assert rates['growth']==pytest.approx(.15,abs=2e-7)
    assert rates['death']==pytest.approx(.15,abs=2e-7)
    assert rates['growth_source']=='accepted_flux'
    report=assembly.diagnostics([history])
    assert report['population_rates']==rates
    assert report['kinetics_validity']['inside_validity_domain']
    assert report['conservation']['post_reconciliation_death']


@pytest.mark.parametrize('backend',['scipy','fixed-step'])
def test_absent_and_zero_response_preserve_results(backend):
    case, config, thermo = configured(backend)
    old = deepcopy(config)
    del old['model']['kinetics']['kinetic_outputs']['death_increment']
    results = []
    for declaration in [old,config]:
        h = build_bioreactor(case,declaration,thermo).solve()[-1]
        results.append(np.column_stack((h.mass_j_liquid0,h.viable_cells_million_liquid0,
                                       h.dead_cells_million_liquid0,h.product_g_liquid0)))
    np.testing.assert_array_equal(*results)


def test_response_uses_reconciled_uptake_not_target():
    case, config, thermo = configured(expression=response())
    next(r for r in config['network']['reactions'] if r['identifier']=='ua')['upper']=.25
    a=build_bioreactor(case,config,thermo);a.solve()
    flux=dict(zip(a.mechanism.last_solution.reaction_ids,a.mechanism.last_solution.fluxes))
    assert flux['ua']<=.250001
    assert a.mechanism.last_population_rates['death_increment']==pytest.approx(.2*(2.-flux['ua']))
    assert a.mechanism.last_population_rates['death_increment']>.34


@pytest.mark.parametrize('expression',[-1.,float('nan'),float('inf')])
def test_invalid_death_response_is_rejected(expression):
    case, config, thermo = configured(expression=expression)
    with pytest.raises(ValueError,match='death_increment'):
        build_bioreactor(case,config,thermo).solve()


def test_unknown_accepted_flux_and_preclosure_feedback_are_rejected():
    case, config, thermo = configured(expression={'accepted_flux':'missing'})
    with pytest.raises(ValueError,match='undeclared exchange'):
        build_bioreactor(case,config,thermo)
    config['model']['kinetics']['rule_graph'][0]['expression']={'accepted_flux':'ua'}
    with pytest.raises(ValueError,match='after reconciliation'):
        build_bioreactor(case,config,thermo)


@pytest.mark.parametrize('variant',['exchange-umol','product-umol','product-mg','growth-umol','hours','cells'])
def test_population_response_is_invariant_to_flux_units(variant):
    from test_bioreactor_objective_units import convert_coordinates, configured as scaled
    case, config, thermo = scaled('growth')
    case['operation']['runtime']['value']=.05
    config['recipes']['benchmark']=[]
    expression={'op':'add','args':[response(),{'accepted_flux':'g'},{'accepted_flux':'p'}]}
    config['model']['kinetics']['kinetic_outputs']['death_increment']={'unit':'1/day','expression':expression}
    changed,unused=convert_coordinates(config,variant)
    results=[]
    for definition in [config,changed]:
        a=build_bioreactor(case,definition,thermo);h=a.solve()[-1]
        results.append([h.viable_cells_million_liquid0[-1],h.dead_cells_million_liquid0[-1],
                        a.mechanism.last_population_rates['death_increment']])
    np.testing.assert_allclose(*results,rtol=2e-6,atol=2e-8)


@pytest.mark.parametrize('policy',['error','allow'])
def test_rule_graph_declares_extrapolation_without_clipping(policy):
    declaration=_rule_provider()
    declaration.update(validity_domain={'growth':[0.,.1]},extrapolation=policy)
    p=build_rate_provider(declaration,_definition(),('growth','death','uptake'))
    if policy=='error':
        with pytest.raises(ValueError,match='outside validity'):
            p.evaluate(_snapshot(),{})
    else:
        result=p.evaluate(_snapshot(),{})
        assert not result.inside_validity_domain
        assert result.rates['growth']==pytest.approx(.8*2./3.)
        assert result.diagnostics['outside_rules']==('growth',)


@pytest.mark.parametrize('domain',[{'unknown':[0.,1.]},{'growth':[2.,1.]},
                                    {'growth':[0.,float('nan')]},{'growth':[1.]}])
def test_invalid_rule_validity_domains(domain):
    declaration=dict(_rule_provider(),validity_domain=domain,extrapolation='error')
    with pytest.raises(ValueError):
        build_rate_provider(declaration,_definition(),('growth','death','uptake'))


@pytest.mark.parametrize('change',['unit','basis','ref'])
def test_response_declaration_is_explicit(change):
    case, config, thermo=configured()
    k=config['model']['kinetics']
    if change=='unit':k['kinetic_outputs']['death_increment']['unit']='1/h'
    elif change=='basis':k['flux_basis']=None
    else:k['kinetic_outputs']['death_increment']['expression']={'ref':'not_assigned'}
    with pytest.raises(ValueError,match='death_increment'):
        build_bioreactor(case,config,thermo)


def test_post_reconciliation_conditional_remains_lazy():
    guarded={'op':'greater-select','threshold':0.,'args':[
        {'accepted_flux':'ua'},0.,{'op':'divide','args':[1.,0.]}]}
    case, config, thermo=configured(expression=guarded)
    a=build_bioreactor(case,config,thermo);a.solve()
    assert a.mechanism.last_population_rates['death_increment']==0.
    a.mechanism.reset()
    assert a.mechanism.last_population_rates is None


def test_fixed_step_rechecks_required_production_after_extra_death():
    from PharmaPy.Metabolic.closures.base import MetabolicNumericalError
    case, config, thermo = configured('fixed-step',expression=100.)
    case['numerics']['step']['value']=1.
    case['operation']['runtime']['value']=1.
    config['state']['species']['amounts']['byproduct']=.1
    k=config['model']['kinetics']
    k['rule_graph'].append({'identifier':'degradation','expression':8.})
    k['degradation_mappings']=[{'source':'byproduct','rate_output':'degradation','products':{}}]
    config['rate_provider']['output_units']['degradation']='1/day'
    # Basal exposure produces enough byproduct to offset a .8 mmol loss;
    # the added mortality cuts that source. Never silently accept depletion.
    with pytest.raises(MetabolicNumericalError,match='post-reconciliation death'):
        build_bioreactor(case,config,thermo).solve()



