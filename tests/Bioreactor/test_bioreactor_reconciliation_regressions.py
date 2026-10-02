"""Independent regressions for depletion geometry and optimality qualification."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from PharmaPy.Bioreactors import build_bioreactor
from PharmaPy.Bioreactors.inventory import InventoryAvailability, LinearInventoryAvailability
import PharmaPy.Metabolic.closures.reconciled as reconciled
from reconciliation_reference import diagnose_failure

STATES = json.loads((Path(__file__).parent / "fixtures/reconciliation/closure_states.json").read_text())
from reconciliation_reference import Reference, load
from test_bioreactor_qualification import network_load as load_small


@pytest.mark.parametrize('scale',[1.,1e-12,1e12])
def test_sparse_exact_recovery_retains_affine_identities(scale):
    from fractions import Fraction
    from PharmaPy.Metabolic.closures._recovery import _affine_basis
    matrix=scale*np.array([[1.,0.,-2.,0.],[0.,3.,0.,-1.],[2.,0.,-4.,0.]])
    rhs=scale*np.array([.5,.25,1.])
    origin,basis=_affine_basis(matrix,rhs)
    for row,b in zip(matrix,rhs):
        exact=[Fraction(float(x)) for x in row]
        assert sum(a*x for a,x in zip(exact,origin))==Fraction(float(b))
        for j in range(len(basis[0])):
            assert sum(a*x[j] for a,x in zip(exact,basis))==0
    with pytest.raises(ValueError,match='inconsistent'):
        _affine_basis(np.array([[1.,0.],[0.,0.]]),np.array([0.,1.]))


@pytest.mark.parametrize('require_recovery',[False,True])
def test_floating_fallback_precedes_but_retains_exact_recovery(monkeypatch,require_recovery):
    from PharmaPy.Metabolic.closures import _recovery
    from PharmaPy.Metabolic.closures.base import MetabolicEnvironment
    from PharmaPy.Metabolic.network import MetabolicNetworkDefinition, ReactionDefinition
    network=MetabolicNetworkDefinition('fallback','1',['pool'],[
        ReactionDefinition('in',{'pool':1.},0.,2.),
        ReactionDefinition('out',{'pool':-1.},0.,2.)],{'in':1.,'out':1.})
    closure=reconciled.RateReconciledMFAClosure(network,[])
    minimize=reconciled.minimize;attempts=[];recovered=[]
    def fail_primary(fun,x0,*args,**kwargs):
        attempts.append(1)
        if require_recovery or len(attempts)==1:
            return SimpleNamespace(x=np.array(x0),success=False,status=8,message='injected stagnation')
        return minimize(fun,x0,*args,**kwargs)
    def exact_candidate(*args,**kwargs):
        assert require_recovery, 'A qualified floating retry should avoid exact recovery'
        assert len(attempts)>=3
        recovered.append(1)
        return np.ones(2),0.5,0.,0.
    monkeypatch.setattr(reconciled,'minimize',fail_primary)
    monkeypatch.setattr(_recovery,'boundary_candidate',exact_candidate)
    solution=closure.solve(MetabolicEnvironment({}),internal_scale=1.,
        targets=[reconciled.ReconciliationTarget(s,v,1.)
                 for s,v in (('in',.5),('out',1.5))])
    np.testing.assert_allclose(solution.fluxes,1.,atol=1e-9,rtol=0.)
    assert bool(recovered)==require_recovery


@pytest.mark.parametrize('reverse',[False,True])
@pytest.mark.parametrize('name,time,segment',[
    ('coupled04',.31,2),('coupled10',.21,2),
    ('coupled20',.12,1),('coupled20_scaled',.12,1)])
def test_reconciled_flux_precision_leaves_trajectory_error_budget(name,time,segment,reverse):
    case,config,thermo=load(name)
    reference=Reference(config)
    state=np.array(STATES[name])
    state[:reference.n+1]=np.where(state[:reference.n+1]<=1e-12,0.,state[:reference.n+1])
    inlets={}
    for event in config['recipes']['benchmark']:
        if event['time']['value']<=time and event['event_type']=='flow':
            inlets[event['inlet']]=event['volume_flow']['value']*np.array([
                event['concentrations']['values'].get(s,0.) for s in reference.species])
    supply=sum(inlets.values(),start=np.zeros(reference.n+1))
    targets=reference.targets(state)
    caps=np.where(state[:reference.n]==0.,supply[:reference.n]/state[reference.n+1],np.inf)
    expected,proof=reference.solve(state,caps)
    if reverse:
        config['network']['reactions'].reverse()
        config['network']['internal_metabolites'].reverse()
    assembly=build_bioreactor(case,config,thermo)
    ids=list(assembly.mechanism.network.reaction_ids)
    mapping=np.zeros((reference.n+1,len(ids)))
    for i,reaction in enumerate(reference.exchanges[:reference.n]):
        mapping[i,ids.index(reaction)]=-state[reference.n+1]
    mapping[-1,ids.index('w')]=state[reference.n+1]
    availability=LinearInventoryAvailability.boundary(state[:reference.n+1],mapping,supply)
    solution=assembly.mechanism._solve(dict(zip(reference.exchanges,targets)),availability)
    flux=np.array([solution.fluxes[ids.index(r)] for r in reference.exchanges])
    # Reserve 90% of the existing exchange-error budget for other numerical errors.
    assert max(abs(flux-expected[reference.selected])/np.maximum(abs(targets),1e-6))<1e-7
    assert abs(-solution.primary_objective-proof['objective'])<1e-8


def test_inventory_presolve_retains_unreduced_recovery():
    folder=Path(__file__).parents[2]/'tests/Bioreactor/fixtures/fed_batch_cho/inputs'
    case,config=[json.loads((folder/name).read_text()) for name in ('case.json','mechanism.json')]
    assembly=build_bioreactor(case,config,folder/'thermo.json')
    data=STATES['cho_closure']
    availability=object.__new__(InventoryAvailability)
    for key,value in data['availability'].items():
        setattr(availability,key,np.array(value) if isinstance(value,list) else value)
    lower,upper=np.array(data['lower']),np.array(data['upper'])
    targets=tuple(reconciled.ReconciliationTarget(**t) for t in data['targets'])
    closure=assembly.mechanism.closure
    result,_=closure._solve_problem(lower,upper,targets,data['internal_scale'],
        np.array(data['initial']),availability)
    assert result.success,result.message
    assert result.fun==pytest.approx(data['expected_objective'],abs=1e-8)
    assert closure._acceptable(result.x,lower,upper,availability)


@pytest.mark.parametrize('reverse',[False,True])
@pytest.mark.parametrize('remaining',[0.,1e-24])
def test_depleted_branch_rejects_false_optimizer_success(monkeypatch,reverse,remaining):
    case,config,thermo=load_small('switching')
    if reverse:
        config['network']['reactions'].reverse()
        config['network']['internal_metabolites'].reverse()
    assembly=build_bioreactor(case,config,thermo)
    ids=assembly.mechanism.network.reaction_ids
    matrix=np.zeros((3,len(ids)))
    matrix[0,ids.index('ua')]=-1.;matrix[1,ids.index('ub')]=-1.;matrix[2,ids.index('q')]=1.
    availability=InventoryAvailability([.3,remaining,.1],matrix,np.zeros(3),
        viable=1.,step_day=.01,cell_flux_scale=1.,growth=0.,death=0.,growth_index=ids.index('g'))
    minimize=reconciled.minimize
    calls=[]
    def false_success(fun,x0,*args,**kwargs):
        calls.append(1)
        if len(calls)==1:
            return SimpleNamespace(x=np.zeros_like(x0),success=True,status=0,message='injected false success')
        return minimize(fun,x0,*args,**kwargs)
    nnls=reconciled.nnls
    def nonempty_nnls(matrix,*args,**kwargs):
        assert matrix.shape[1]>0, 'Empty cones must use the projected gradient norm'
        return nnls(matrix,*args,**kwargs)
    monkeypatch.setattr(reconciled,'nnls',nonempty_nnls)
    monkeypatch.setattr(reconciled,'minimize',false_success)
    result=assembly.mechanism._solve({s:config['parameters']['rates'][s] for s in ['ua','ub','p','q','g']},availability)
    # Analytic minimum of ((x-1.5)^2 + .5^2 + (x-1)^2 + 1)/1.5^2.
    assert result.fluxes[ids.index('p')]==pytest.approx(1.25,abs=1e-7)
    assert -result.primary_objective==pytest.approx(11./18.,abs=1e-9)


@pytest.mark.parametrize('name,backend,variant',[
    ('coupled10','scipy','diagnostic'),('coupled20','scipy','diagnostic'),
    ('coupled20','fixed-step','half'),('coupled20_scaled','scipy','base')])
def test_frozen_failed_closure_matches_independent_optimum(name,backend,variant):
    report=STATES['/'.join((name,backend,variant))]
    data=report['failed_closure'];case,config,thermo=load(name)
    assembly=build_bioreactor(case,config,thermo)
    saved=data['availability']
    availability=object.__new__(InventoryAvailability if saved['type']=='InventoryAvailability' else LinearInventoryAvailability)
    for key,value in saved['attributes'].items():
        setattr(availability,key,np.array(value) if isinstance(value,list) else value)
    lower,upper=assembly.mechanism.network.bounds(data['bounds'])
    targets=tuple(reconciled.ReconciliationTarget(**t) for t in data['targets'])
    solution,_=assembly.mechanism.closure._solve_problem(lower,upper,targets,data['internal_scale'],availability=availability)
    proof=diagnose_failure(report)
    assert proof['status']=='CERTIFIED_FEASIBLE_OPTIMUM'
    assert solution.success,solution.message
    assert abs(solution.fun-proof['objective'])<1e-8*max(1.,proof['objective'])
    assert np.min(availability.residual(solution.x))>=-1e-10


def test_finite_step_reference_does_not_use_infeasible_upper_bound():
    _,config,_=load('coupled04');reference=Reference(config)
    state=np.array(STATES['finite_step'])
    event=config['recipes']['benchmark'][0]
    supply=event['volume_flow']['value']*np.array([event['concentrations']['values'].get(s,0.) for s in reference.species])
    flux,proof=reference.fixed_step(state,supply,.002,gap_tolerance=1e-10)
    used=.001*state[reference.n+1]*(1.+np.exp((flux[reference.growth_index]-.03)*.002))*flux[reference.selected[:reference.n]]
    remaining=state[:reference.n]+.002*supply[:reference.n]
    assert np.max(used-remaining)<1e-18
    assert proof['objective']==pytest.approx(.5798456208637075,abs=1e-10)
    assert 0.<=proof['objective_gap_bound']<=1e-10


def test_finite_reference_polishing_resolves_rates_with_original_certificate():
    reference=Reference(load('coupled20')[1])
    state=np.array(STATES['polishing'])
    supply=np.zeros(reference.n+1);step=.002/64
    coarse,proof=reference.fixed_step(state,supply,step,gap_tolerance=1e-11,polish_tolerance=1e-10)
    fine,refined=reference.fixed_step(state,supply,step,gap_tolerance=5e-12,polish_tolerance=1e-13)
    target=reference.targets(state)
    assert max(abs(coarse[reference.selected]-fine[reference.selected])/np.maximum(abs(target),1e-6))<2e-7
    assert 0.<=refined['objective_gap_bound']<=5e-12*max(1.,refined['objective'])
    assert refined['polish_evaluations']>0
    exposure=.5*state[reference.n+1]*step*(1.+np.exp((fine[reference.growth_index]-.03)*step))
    used=exposure*fine[reference.selected[:reference.n]]
    remaining=state[:reference.n].copy()
    remaining[-1]-=step*reference.parameters['rates']['degradation']*state[reference.n-1]
    assert np.all(used-remaining<=8*np.finfo(float).eps*(abs(used)+abs(remaining)))


def test_unresolved_interval_reduction_does_not_invent_infeasibility():
    from PharmaPy.Metabolic.network import MetabolicNetworkDefinition, ReactionDefinition
    from PharmaPy.Metabolic.closures.base import MetabolicEnvironment
    network=MetabolicNetworkDefinition('narrow','1',['pool'],[
        ReactionDefinition('variable',{'pool':1.},0.,1e-16),
        ReactionDefinition('fixed',{'pool':-1.},1e-16,1e-16)],{'variable':1.})
    closure=reconciled.RateReconciledMFAClosure(network,[])
    result=closure.solve(MetabolicEnvironment({}),targets=[reconciled.ReconciliationTarget('variable',0.,1.)],internal_scale=1.)
    lower,upper=network.bounds({})
    original,_=closure._solve_problem(lower,upper,[reconciled.ReconciliationTarget('variable',0.,1.)],1.,_reduce_intervals=False)
    assert original.success
    np.testing.assert_array_equal(result.fluxes,original.x)


@pytest.mark.parametrize('name',['coupled04','coupled10','coupled20','coupled20_scaled'])
def test_reference_upper_bound_at_depleted_nutrients(name):
    reference=Reference(load(name)[1]);state=reference.initial();state[:2]=0.
    flux,proof=reference.fixed_step(state,np.zeros(reference.n+1),.002)
    assert np.max(flux[reference.selected[:2]])<=0.
    assert 0.<=proof['objective_gap_bound']<=1e-9*max(1.,proof['objective'])


def _optimal_face_ranges(network, fixed, outputs):
    """Independent LP ranges with penalized coordinates fixed at their optimum."""
    from scipy.optimize import linprog
    bounds = list(zip(network.lower_bounds, network.upper_bounds))
    for name, value in fixed.items():
        bounds[network.reaction_ids.index(name)] = (value, value)
    intervals = []
    for row in outputs:
        ends = []
        for sign in (1., -1.):
            result = linprog(sign * row, A_eq=network.stoichiometric_matrix,
                             b_eq=np.zeros(len(network.internal_metabolite_ids)),
                             bounds=bounds, method='highs',
                             options={'primal_feasibility_tolerance': 1e-9})
            assert result.success, result.message
            ends.append(row @ result.x)
        intervals.append(ends)
    return np.asarray(intervals)


@pytest.mark.parametrize('reverse', [False, True])
@pytest.mark.parametrize('target_output', [False, True])
@pytest.mark.parametrize('separate_outputs', [False, True])
def test_optimal_flux_ambiguity_matters_only_when_reactor_sources_differ(
        reverse, target_output, separate_outputs):
    from PharmaPy.Metabolic.network import MetabolicNetworkDefinition, ReactionDefinition
    from PharmaPy.Metabolic.closures.base import MetabolicEnvironment
    reactions = [ReactionDefinition('in', {'pool': 1.}, 0., 2.),
                 ReactionDefinition('out1', {'pool': -1.}, 0., 2.),
                 ReactionDefinition('out2', {'pool': -1.}, 0., 2.)]
    network = MetabolicNetworkDefinition('ambiguity', '1', ['pool'],
        reactions[::-1] if reverse else reactions, {'in': 1.})
    ids = network.reaction_ids
    closure = reconciled.RateReconciledMFAClosure(network, ())
    fixed = {'in': 1., **({'out1': .25} if target_output else {})}
    targets = [reconciled.ReconciliationTarget(name, value, 1.) for name, value in fixed.items()]
    source = np.array([[float(name == output) for name in ids] for output in ('out1', 'out2')])
    if not separate_outputs:
        source = source.sum(axis=0, keepdims=True)
    ranges = _optimal_face_ranges(network, fixed, source)
    ambiguous = separate_outputs and not target_output
    if ambiguous:
        np.testing.assert_allclose(ranges, [[0., 1.], [0., 1.]], atol=1e-9)
    else:
        np.testing.assert_allclose(ranges[:, 0], ranges[:, 1], atol=1e-9)
    rates = []
    for split in (0., 1.):
        warm = {'in': 1., 'out1': split, 'out2': 1. - split}
        solution = closure.solve(MetabolicEnvironment({}),
            SimpleNamespace(fluxes=np.array([warm[name] for name in ids])),
            targets=targets, internal_scale=1.)
        assert abs(solution.primary_objective) < 1e-12
        np.testing.assert_allclose(network.stoichiometric_matrix @ solution.fluxes, 0., atol=1e-9)
        rates.append(source @ solution.fluxes)
    if ambiguous:
        # Primary ambiguity remains, but automatic selection is independent of warm starts.
        np.testing.assert_allclose(rates, .5, atol=1e-8)
        assert solution.tie_break['guarantee'] == 'unique-on-primary-face'
        assert solution.tie_break['primary_objective_change'] == 0.
    else:
        np.testing.assert_allclose(rates[0], rates[1], atol=1e-8)


@pytest.mark.parametrize('factor', [.001, 1., 1000.])
@pytest.mark.parametrize('transform_metric', [False, True])
@pytest.mark.parametrize('capacity', [.25, 2.])
def test_mixed_coordinate_units_require_transformed_residual_scales(factor, transform_metric, capacity):
    from PharmaPy.Metabolic.network import MetabolicNetworkDefinition, ReactionDefinition
    from PharmaPy.Metabolic.closures.base import MetabolicEnvironment
    network = MetabolicNetworkDefinition('mixed-units', '1', ['pool'], [
        ReactionDefinition('uptake', {'pool': 1.}, 0., 2.),
        ReactionDefinition('product', {'pool': -1. / factor}, 0., capacity * factor)],
        {'product': 1. / factor})
    closure = reconciled.RateReconciledMFAClosure(network, ())
    solution = closure.solve(MetabolicEnvironment({}), targets=[
        reconciled.ReconciliationTarget('uptake', 1., 1.),
        reconciled.ReconciliationTarget('product', 0., factor if transform_metric else 1.)],
        internal_scale=1.)
    physical = solution.fluxes / np.array([1., factor])
    expected = min(capacity, .5 if transform_metric else 1. / (1. + factor**2))
    np.testing.assert_allclose(physical, expected, rtol=1e-7, atol=1e-9)
    objective = (physical[0] - 1.)**2 + (physical[1] * (1. if transform_metric else factor))**2
    assert -solution.primary_objective == pytest.approx(objective, abs=1e-9)


def test_native_unweighted_product_unit_change_alters_conflicting_projection():
    from copy import deepcopy
    from PharmaPy.Metabolic.closures.base import MetabolicEnvironment
    case, config, thermo = load_small('conflicting')
    baseline = build_bioreactor(case, config, thermo).mechanism
    targets = {name: config['parameters']['rates'][name]
               for name in config['model']['kinetics']['kinetic_outputs']['exchange_fluxes']}
    expected = baseline._solve(targets).fluxes
    changed = deepcopy(config)
    changed['model']['kinetics']['flux_basis']['product']['amount_unit'] = 'umol'
    changed['rate_provider']['output_units']['p'] = 'umol/(10^6 cell day)'
    changed['parameters']['rates']['p'] *= 1000.
    for reaction in changed['network']['reactions']:
        if reaction['identifier'] == 'p':
            reaction['stoichiometry'] = {name: value / 1000.
                                         for name, value in reaction['stoichiometry'].items()}
            reaction['lower'] *= 1000.
            reaction['upper'] *= 1000.
    changed['network']['objective']['p'] /= 1000.
    mechanism = build_bioreactor(case, changed, thermo).mechanism
    converted = dict(targets, p=targets['p'] * 1000.)
    coordinate_scale = np.array([1000. if name == 'p' else 1. for name in mechanism.network.reaction_ids])
    actual = mechanism._solve(converted).fluxes / coordinate_scale
    assert np.max(abs(actual - expected)) > .01
    # The converter correctly preserves physical product conversion. The metric differs.
    assert mechanism.unit_converter.product_scale * 1000. == pytest.approx(baseline.unit_converter.product_scale)
    restored = mechanism.closure.solve(MetabolicEnvironment({}), targets=[
        reconciled.ReconciliationTarget(name, value, 1000. if name == 'p' else 1.)
        for name, value in converted.items()], internal_scale=1.)
    np.testing.assert_allclose(restored.fluxes / coordinate_scale, expected, atol=1e-7, rtol=1e-7)




@pytest.mark.parametrize('factor', [1e-6, 1., 1e6])
@pytest.mark.parametrize('reverse', [False, True])
def test_automatic_tie_break_preserves_units_capacity_metric_and_primary_optimum(factor, reverse):
    from PharmaPy.Metabolic.network import MetabolicNetworkDefinition, ReactionDefinition
    from PharmaPy.Metabolic.closures.base import MetabolicEnvironment
    declarations = [ReactionDefinition('in', {'pool': 1.}, 0., 2.),
                    ReactionDefinition('a', {'pool': -1. / factor}, 0., factor),
                    ReactionDefinition('b', {'pool': -1.}, 0., 2.)]
    network = MetabolicNetworkDefinition('selection', '1', ['pool'],
        declarations[::-1] if reverse else declarations, {'in': 1.})
    closure = reconciled.RateReconciledMFAClosure(network, ())
    for split in (0., 1.):
        warm = {'in': 1., 'a': split * factor, 'b': 1. - split}
        solution = closure.solve(MetabolicEnvironment({}),
            SimpleNamespace(fluxes=np.array([warm[name] for name in network.reaction_ids])),
            targets=[reconciled.ReconciliationTarget('in', 1., 1.)], internal_scale=1.)
        flux = dict(zip(solution.reaction_ids, solution.fluxes))
        # Minimize a^2 + (b/2)^2 subject to a+b=1: a=1/5, b=4/5.
        np.testing.assert_allclose([flux['in'], flux['a']/factor, flux['b']], [1., .2, .8], atol=1e-8)
        assert solution.primary_objective == 0.
        assert solution.secondary_objective == pytest.approx(-.2)
        assert solution.tie_break['primary_objective_change'] == 0.


@pytest.mark.parametrize('kind', ['boundary', 'fixed', 'growth-dependent'])
def test_automatic_tie_break_preserves_inventory_constraints(kind):
    from scipy.optimize import brentq
    from PharmaPy.Metabolic.network import MetabolicNetworkDefinition, ReactionDefinition
    from PharmaPy.Metabolic.closures.base import MetabolicEnvironment
    network = MetabolicNetworkDefinition('inventory-selection', '1', ['pool'], [
        ReactionDefinition('in', {'pool': 1.}, 0., 2.),
        ReactionDefinition('a', {'pool': -1.}, 0., 2.),
        ReactionDefinition('b', {'pool': -1.}, 0., 2.)], {'in': 1.})
    matrix = np.array([[0., -1., 0.]])
    if kind == 'boundary':
        availability = LinearInventoryAvailability.boundary([0.], matrix, [.1])
    else:
        availability = InventoryAvailability([.1], matrix, [0.], viable=1., step_day=1.,
            cell_flux_scale=1., growth=0., death=0.,
            growth_index=1 if kind == 'growth-dependent' else None)
    expected = (brentq(lambda a: a * (1. + np.exp(a))/2. - .1, 0., 1.)
                if kind == 'growth-dependent' else .1)
    closure = reconciled.RateReconciledMFAClosure(network, ())
    for seed in ([1., 0., 1.], [1., .09, .91]):
        solution = closure.solve(MetabolicEnvironment({}), SimpleNamespace(fluxes=np.array(seed)),
            targets=[reconciled.ReconciliationTarget('in', 1., 1.)], internal_scale=1.,
            availability=availability)
        np.testing.assert_allclose(solution.fluxes, [1., expected, 1.-expected], atol=1e-8)
        assert min(availability.residual(solution.fluxes)) >= -1e-8
        assert solution.tie_break['primary_objective_change'] == 0.
        assert solution.tie_break['guarantee'] == ('local-selection' if kind == 'growth-dependent'
                                                  else 'unique-on-primary-face')


@pytest.mark.parametrize('target', [-1., 0., 1.])
def test_automatic_tie_break_handles_reversible_and_zero_fluxes(target):
    from PharmaPy.Metabolic.network import MetabolicNetworkDefinition, ReactionDefinition
    from PharmaPy.Metabolic.closures.base import MetabolicEnvironment
    network = MetabolicNetworkDefinition('reversible-selection', '1', ['pool'], [
        ReactionDefinition('in', {'pool': 1.}, -2., 2.),
        ReactionDefinition('a', {'pool': -1.}, -2., 2.),
        ReactionDefinition('b', {'pool': -1.}, -2., 2.)], {'in': 1.})
    closure = reconciled.RateReconciledMFAClosure(network, ())
    for split in (-.5, .5):
        solution = closure.solve(MetabolicEnvironment({}),
            SimpleNamespace(fluxes=np.array([target, split, target-split])),
            targets=[reconciled.ReconciliationTarget('in', target, 1.)], internal_scale=1.)
        np.testing.assert_allclose(solution.fluxes, [target, target/2., target/2.], atol=1e-8)
        assert solution.tie_break['primary_objective_change'] == 0.


def test_native_diagnostics_distinguish_internal_selection_from_reactor_changes():
    case, config, thermo = load_small('nonunique')
    assembly = build_bioreactor(case, config, thermo)
    targets = {name: config['parameters']['rates'][name]
               for name in config['model']['kinetics']['kinetic_outputs']['exchange_fluxes']}
    assembly.mechanism.last_solution = assembly.mechanism._solve(targets)
    report = assembly.diagnostics([])
    assert report['optimization']['tie_break']['guarantee'] == 'unique-on-primary-face'
    assert report['tie_break_reactor_reactions'] == []
    assert report['optimization']['tie_break']['primary_objective_change'] == 0.
