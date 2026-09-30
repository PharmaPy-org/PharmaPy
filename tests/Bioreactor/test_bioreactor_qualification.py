"""Independent population, network, constraint, operation and coupled-model checks."""

import json
import os
import signal
import time
from copy import deepcopy
from decimal import Decimal, localcontext
from itertools import combinations
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.optimize import brentq, minimize_scalar

from PharmaPy.Bioreactors import build_bioreactor
from PharmaPy.Bioreactors.culture import CultureSnapshot
from PharmaPy.Metabolic.closures.base import MetabolicInfeasibleError, MetabolicNumericalError
from reconciliation_reference import (
    compare, CASES as COUPLED_CASES, Reference, load as coupled_load,
    network_matrix, quadratic_program,
)


# Population qualification

POPULATION_BASE = Path(__file__).parents[2] / 'examples/bioreactors/generic_fed_batch/reconciliation/level2'
POPULATION_SCALES = np.array([10., 1., .2, .1, 1., .001])


def population_load(variant):
    p = POPULATION_BASE/'inputs'/variant
    return json.loads((p/'case.json').read_text()), json.loads((p/'mechanism.json').read_text()), p/'thermo.json'


def population_observed(a, h):
    j = list(a.phase.name_species).index('nutrient')
    return np.column_stack((h.mass_j_liquid0[:, j]*10000., h.viable_cells_million_liquid0,
                            h.dead_cells_million_liquid0, h.product_g_liquid0,
                            np.asarray(h.vessel_vol).reshape(-1)*1000.,
                            h.ivcd_million_cell_day_per_ml_liquid0))


def population_reference(histories, variant):
    """Independent ODE; constant-rate cases additionally checked analytically below."""
    y = np.array([10., 1., 0., 0., 1., 0.])
    answer = []
    for h in histories:
        start, end = h.time[[0, -1]]/86400.
        flow = .1 if start < .5 else .2 if start < 1. else 0.
        if np.isclose(start, .75):
            y[:4] *= 1.-.1/y[4]
            y[4] -= .1
        if np.isclose(start, 1.):
            y[0] += .3
            y[4] += .1
        death = 0. if variant == 'growth' else .1

        def rhs(t, state):
            amount, viable, dead, product, volume, exposure = state
            mu = (.44*(amount/volume)/(1.+amount/volume) if variant == 'monod'
                  else .25 if variant == 'constrained_growth' else .4)
            return [2.*flow-2.4*mu*viable, (mu-death)*viable, death*viable,
                    .04*mu*viable, flow, viable/(1000.*volume)]

        solution = solve_ivp(rhs, (start, end), y, t_eval=h.time/86400.,
                             method='DOP853', rtol=2e-13, atol=2e-15, max_step=.0025)
        assert solution.success
        rows = solution.y.T
        if variant != 'monod':
            mu = .25 if variant == 'constrained_growth' else .4
            dt = h.time/86400.-start
            total = y[1]*np.expm1((mu-death)*dt)/(mu-death)
            analytic = np.column_stack((y[0]+2.*flow*dt-2.4*mu*total,
                                        y[1]*np.exp((mu-death)*dt), y[2]+death*total,
                                        y[3]+.04*mu*total, y[4]+flow*dt))
            np.testing.assert_allclose(rows[:, :5], analytic, atol=2e-12, rtol=2e-12)
        answer.append(rows)
        y = rows[-1].copy()
    return answer


@pytest.mark.parametrize('variant', ['growth', 'growth_death', 'constrained_growth', 'monod'])
def test_population_adaptive_trajectories(variant):
    a = build_bioreactor(*population_load(variant))
    histories = a.solve()
    expected = population_reference(histories, variant)
    for h, exact in zip(histories, expected):
        actual = population_observed(a, h)
        assert np.isfinite(actual).all() and actual.min() >= -1e-10
        assert np.max(abs(actual-exact)/POPULATION_SCALES) < 1e-6
        total = actual[:, 0]+10.*actual[:, 3]+2.*(actual[:, 1]+actual[:, 2])
        exact_total = exact[:, 0]+10.*exact[:, 3]+2.*(exact[:, 1]+exact[:, 2])
        np.testing.assert_allclose(total, exact_total, atol=1e-7, rtol=0.)
    for previous, following in zip(histories[:-1], histories[1:]):
        if np.isclose(following.time[0]/86400., .75):
            before, after = population_observed(a, previous)[-1], population_observed(a, following)[0]
            np.testing.assert_allclose(after[:4]/after[4], before[:4]/before[4], rtol=1e-12, atol=1e-12)
            assert after[5] == pytest.approx(before[5], abs=1e-14)


@pytest.mark.parametrize('variant, expected', [('growth', [.96, .8, .16, .8]),
                                              ('constrained_growth', [.6, .5, .1, .5])])
def test_population_growth_flux_projection(variant, expected):
    a = build_bioreactor(*population_load(variant))
    solution = a.mechanism._solve({'uptake':.96, 'growth_sink':.8, 'secretion':.16})
    np.testing.assert_allclose(solution.fluxes, expected, atol=1e-9, rtol=0.)
    assert a.mechanism.unit_converter.growth_scale == .5


@pytest.mark.parametrize('variant', ['growth', 'growth_death', 'constrained_growth', 'monod'])
def test_population_fixed_step_convergence(variant):
    errors = []
    for step in (.02, .01, .005):
        case, config, thermo = population_load(variant)
        case['numerics'] = {k: (v if k == 'step' else 'fixed-step' if k == 'backend' else None)
                            for k, v in case['numerics'].items()}
        case['numerics']['step']['value'] = step
        a = build_bioreactor(case, config, thermo)
        histories = a.solve()
        exact = population_reference(histories, variant)
        errors.append(np.max(np.concatenate([abs(population_observed(a,h)-r)/POPULATION_SCALES for h,r in zip(histories,exact)])))
    assert errors[1] < .65*errors[0] and errors[2] < .65*errors[1]


@pytest.mark.parametrize('boundary', ['zero_cells', 'zero_nutrient'])
def test_population_zero_inventory_boundary(boundary):
    case, config, thermo = population_load('growth_death')
    case['operation']['runtime'] = dict(unit='day', value=.01)
    config['recipes']['benchmark'] = []
    if boundary == 'zero_cells':
        config['state']['viable_cells_million'] = 0.
    else:
        config['state']['species']['amounts']['nutrient'] = 0.
    a = build_bioreactor(case, config, thermo)
    result, = a.solve()
    actual = population_observed(a, result)
    np.testing.assert_allclose(actual[:, 0], actual[0, 0], atol=1e-9)
    np.testing.assert_allclose(actual[:, 3], 0., atol=1e-10)
    expected = 0. if boundary == 'zero_cells' else np.exp(-.1*.01)
    assert actual[-1, 1] == pytest.approx(expected, abs=1e-8)


def test_population_equivalent_molecular_rate_basis():
    case, config, thermo = population_load('constrained_growth')
    original = build_bioreactor(case, config, thermo).solve()
    config = deepcopy(config)
    config['parameters']['coefficients']['biomass_amount_per_million_cells'] *= 1000.
    for name, declaration in config['model']['kinetics']['flux_basis'].items():
        declaration['amount_unit'] = 'umol'
        if name == 'growth':
            declaration['amount_per_million_cells'] *= 1000.
    for name in ('uptake', 'growth_flux', 'product'):
        config['rate_provider']['output_units'][name] = 'umol/(10^6 cell day)'
    for reaction in config['network']['reactions']:
        reaction['lower'] *= 1000.
        reaction['upper'] *= 1000.
    a = build_bioreactor(case, config, thermo)
    converted = a.solve()
    for x, y in zip(original, converted):
        assert np.max(abs(population_observed(a, x)-population_observed(a, y))/POPULATION_SCALES) < 1e-6


def test_population_equivalent_specific_growth_basis():
    case, config, thermo = population_load('constrained_growth')
    original = build_bioreactor(case, config, thermo).solve()
    kinetics = config['model']['kinetics']
    kinetics['flux_basis']['growth'] = dict(amount_unit='1', normalization='none', time_unit='day')
    kinetics['kinetic_outputs']['exchange_fluxes']['growth_sink'] = 'growth'
    for reaction in config['network']['reactions']:
        if reaction['identifier'] == 'growth_sink':
            reaction['stoichiometry']['B'] *= 2.
            reaction['upper'] /= 2.
    a = build_bioreactor(case, config, thermo)
    converted = a.solve()
    for x, y in zip(original, converted):
        assert np.max(abs(population_observed(a, x)-population_observed(a, y))/POPULATION_SCALES) < 1e-6


# Network qualification

NETWORK_BASE = Path(__file__).parents[2] / 'examples/bioreactors/generic_fed_batch/reconciliation/level3'
NETWORK_FIELDS = ['nutrient_a_mmol', 'nutrient_b_mmol', 'byproduct_mmol', 'product_g',
          'viable_cells_million', 'dead_cells_million', 'volume_l', 'ivcd_million_cell_day_per_ml']
NETWORK_SCALES = np.array([10., 10., 2., .1, 1., .1, 1., .001])
NETWORK_VARIANTS = ['compatible', 'conflicting', 'switching', 'growth', 'nonunique']


@pytest.mark.parametrize('policy', ['unweighted', 'relative-regularized'])
def test_reconciliation_copy_preserves_policy(policy):
    case, config, thermo = network_load('compatible')
    config['model']['reconciliation']['reconciliation_policy'] = policy
    if policy == 'relative-regularized':
        config['model']['reconciliation'].update(reconciliation_relative_bound=1.,
                                                reconciliation_bound_increment=.5)
    original = build_bioreactor(case, config, thermo).mechanism
    original.closure.tolerance = 5e-8
    copied = deepcopy(original)
    assert copied.closure is not original.closure
    assert copied.closure.internal_reaction_ids == original.closure.internal_reaction_ids
    assert copied.closure.tolerance == original.closure.tolerance
    targets = {name:config['parameters']['rates'][name] for name in ('ua', 'ub', 'p', 'q', 'g')}
    np.testing.assert_allclose(copied._solve(targets).fluxes, original._solve(targets).fluxes,
                               atol=1e-9, rtol=1e-9)


def network_load(variant):
    folder = NETWORK_BASE/'inputs'/variant
    return (json.loads((folder/'case.json').read_text()),
            json.loads((folder/'mechanism.json').read_text()), folder/'thermo.json')


def network_projection(variant, limits=(np.inf, np.inf)):
    """Enumerate all possible active sets in an independently reduced 2-D QP."""
    b = .2 if variant == 'growth' else 0.
    mapping = np.array([[1., 1.], [0., 1.], [1.-b, 0.], [0., 2.-b], [b, b]])
    target = np.array([4., 1., 1., 4., 0.] if variant == 'conflicting'
                      else [1.5, .5, 1.-b, (2.-b)*.5, 1.5*b])
    constraints = np.vstack((-np.eye(2), mapping, [1., 1.], [0., 1.]))
    bounds = np.array([0., 0., 10., 10., 10., 10., 10., *limits])
    finite = np.isfinite(bounds)
    constraints, bounds = constraints[finite], bounds[finite]
    hessian, linear = mapping.T@mapping, mapping.T@target
    candidates = []
    for size in range(3):
        for subset in combinations(range(len(bounds)), size):
            rows = constraints[list(subset)]
            system = np.block([[hessian, rows.T], [rows, np.zeros((size, size))]])
            if np.linalg.matrix_rank(system) < len(system):
                continue
            solution = np.linalg.solve(system, np.r_[linear, bounds[list(subset)]])
            z = solution[:2]
            if np.max(constraints@z-bounds) <= 1e-9 and np.all(solution[2:] >= -1e-9):
                candidates.append(z)
    assert candidates
    z = min(candidates, key=lambda z: np.sum((mapping@z-target)**2))
    return z, mapping@z, float(np.sum((mapping@z-target)**2))


def network_observed(assembly, history, names=('nutrient_a', 'nutrient_b', 'byproduct')):
    indexes = [list(assembly.phase.name_species).index(n) for n in names]
    return np.column_stack((history.mass_j_liquid0[:, indexes]*10000.,
                            history.product_g_liquid0, history.viable_cells_million_liquid0,
                            history.dead_cells_million_liquid0,
                            np.asarray(history.vessel_vol).reshape(-1)*1000.,
                            history.ivcd_million_cell_day_per_ml_liquid0))


def network_reference(histories, variant, continuous=False):
    """Exact finite-step balances for zero growth; independent ODE for growth."""
    state = np.array([.6, .1, 0., 0., 1., 0., 1., 0.] if variant == 'switching'
                     else [10., 10., 0., 0., 1., 0., 1., 0.])
    answer = []
    for history in histories:
        times = history.time/86400.
        if np.isclose(times[0], .5):
            state[:2] += [.6, .2]
            state[6] += .1
        if np.isclose(times[0], .75):
            state[:6] *= 1.-.1/state[6]
            state[6] -= .1
        if variant == 'growth':
            _, fluxes, _ = network_projection(variant)
            ua, ub, p, q, g = fluxes
            def rhs(t, y):
                return [-ua*y[4], -ub*y[4], q*y[4], .1*p*y[4],
                        (g/2.-.05)*y[4], .05*y[4], .1, y[4]/(1000.*y[6])]
            result = solve_ivp(rhs, (times[0], times[-1]), state, t_eval=times,
                               method='DOP853', rtol=2e-13, atol=2e-15, max_step=.0025)
            assert result.success
            rows = result.y.T
        else:
            rows = [state.copy()]
            for dt in np.diff(times):
                remaining = dt
                while remaining > 1e-13:
                    limits = (np.where(state[:2] > 1e-10, np.inf, 0.) if continuous
                              else np.maximum(state[:2], 0.)/(state[4]*remaining))
                    _, fluxes, _ = network_projection(variant, limits)
                    ua, ub, p, q, _ = fluxes
                    consumption = state[4]*np.array([ua, ub])
                    duration = remaining
                    if continuous:
                        positive = consumption > 1e-10
                        duration = min(duration, np.min(state[:2][positive]/consumption[positive], initial=np.inf))
                    state[:4] += duration*state[4]*np.array([-ua, -ub, q, .1*p])
                    state[7] += (state[4]/100.*np.log1p(.1*duration/state[6]) if continuous
                                 else duration*state[4]/(1000.*state[6]))
                    state[6] += .1*duration
                    remaining -= duration
                rows.append(state.copy())
            rows = np.asarray(rows)
        answer.append(rows)
        state = rows[-1].copy()
    return answer


@pytest.mark.parametrize('variant', NETWORK_VARIANTS)
def test_network_trajectories(variant):
    assembly = build_bioreactor(*network_load(variant))
    histories = assembly.solve()
    for history, expected in zip(histories, network_reference(histories, variant)):
        actual = network_observed(assembly, history)
        assert np.isfinite(actual).all() and actual.min() >= -1e-9
        assert np.max(abs(actual-expected)/NETWORK_SCALES) < 1e-6
        weights = np.array([1., 1., 1., 10., 2., 2., 0., 0.])
        np.testing.assert_allclose(actual@weights, expected@weights, atol=1e-7, rtol=0.)
    for before, after in zip(histories, histories[1:]):
        if np.isclose(after.time[0]/86400., .75):
            x, y = network_observed(assembly, before)[-1], network_observed(assembly, after)[0]
            np.testing.assert_allclose(x[:6]/x[6], y[:6]/y[6], atol=1e-12)
            assert x[7] == pytest.approx(y[7], abs=1e-14)


@pytest.mark.parametrize('variant', NETWORK_VARIANTS)
def test_network_optimum(variant):
    case, config, thermo = network_load(variant)
    a = build_bioreactor(case, config, thermo)
    targets = {n:config['parameters']['rates'][n] for n in ('ua', 'ub', 'p', 'q', 'g')}
    result = a.mechanism._solve(targets)
    flux = dict(zip(result.reaction_ids, result.fluxes))
    z, expected, objective = network_projection(variant)
    actual = np.array([flux[n] for n in targets])
    np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=0.)
    assert np.sum((actual-np.array(list(targets.values())))**2) == pytest.approx(objective, abs=1e-9)
    assert result.mass_balance_residual_inf < 1e-9 and result.bound_violation_inf < 1e-9
    assert flux['x']+flux.get('x_copy', 0.) == pytest.approx(z[0], abs=2e-6)
    assert flux['y'] == pytest.approx(z[1], abs=2e-6)
    if variant == 'nonunique':
        # Every split of total x between identical columns preserves balances and objective.
        reactions = config['network']['reactions']
        matrix = np.array([[r['stoichiometry'].get(s, 0.) for r in reactions]
                           for s in config['network']['internal_metabolites']])
        ids = [r['identifier'] for r in reactions]
        for fraction in (0., .5, 1.):
            candidate = np.array([flux[n] for n in ids])
            candidate[ids.index('x')], candidate[ids.index('x_copy')] = fraction*z[0], (1.-fraction)*z[0]
            np.testing.assert_allclose(matrix@candidate, 0., atol=2e-6)
            assert candidate.min() >= -1e-9 and candidate.max() <= 10.


@pytest.mark.parametrize('variant', ['conflicting', 'switching', 'growth', 'nonunique'])
def test_network_rename_and_permute(variant, tmp_path):
    case, config, thermo = network_load(variant)
    a = build_bioreactor(case, config, thermo)
    original = a.solve()
    names = ['nutrient_a', 'nutrient_b', 'byproduct']
    identifiers = names+config['network']['internal_metabolites']+[r['identifier'] for r in config['network']['reactions']]
    rename = {name:'renamed_'+str(i) for i,name in enumerate(identifiers)}
    def transform(value):
        if isinstance(value, dict):
            return {rename.get(k,k):transform(v) for k,v in reversed(list(value.items()))}
        if isinstance(value, list):
            return [transform(v) for v in value]
        return rename.get(value, value) if isinstance(value, str) else value
    config, thermo = transform(config), transform(json.loads(thermo.read_text()))
    for key in ('reactions', 'internal_metabolites', 'exchange_reactions'):
        config['network'][key].reverse()
    config['model']['kinetics']['extracellular_species'].reverse()
    # Parameter paths are strings in the general rule language, not species identities.
    for rule in config['model']['kinetics']['rule_graph']:
        path = rule['expression']['parameter'].split('.')
        rule['expression']['parameter'] = '.'.join(rename.get(part, part) for part in path)
    thermo_path = tmp_path/'thermo.json'
    thermo_path.write_text(json.dumps(thermo))
    b = build_bioreactor(case, config, thermo_path)
    changed = b.solve()
    assert len(original) == len(changed)
    for x, y in zip(original, changed):
        assert np.max(abs(network_observed(a,x)-network_observed(b,y,tuple(rename[n] for n in names)))/NETWORK_SCALES) < 1e-6


def test_network_switching_and_replenishment():
    a = build_bioreactor(*network_load('switching'))
    h = a.solve()
    rows = np.concatenate([network_observed(a,x) for x in h])
    time = np.concatenate([x.time/86400. for x in h])
    # B depletes first; the A-only branch continues, then stops until the bolus.
    for day, empty in ((.25, [False, True]), (.49, [True, True]), (.6, [False, False])):
        row = rows[np.argmin(abs(time-day))]
        assert list(row[:2] < 1e-8) == empty
    assert rows[np.argmin(abs(time-.49)), 3] > rows[np.argmin(abs(time-.25)), 3]
    for column, start, expected in ((1, 0., .2), (0, 0., .44), (0, .5, .9), (1, .5, .9)):
        depleted = (time >= start) & (rows[:, column] < 1e-8)
        if start:
            depleted &= time > start
        assert time[depleted][0] == pytest.approx(expected, abs=1e-10)


@pytest.mark.parametrize('variant', ['switching', 'growth'])
def test_network_continuous_reference_convergence(variant):
    errors = []
    for step in (.01, .005, .0025):
        case, config, thermo = network_load(variant)
        case['numerics'] = {k:(v if k == 'step' else 'fixed-step' if k == 'backend' else None)
                            for k,v in case['numerics'].items()}
        case['numerics']['step']['value'] = step
        a = build_bioreactor(case, config, thermo)
        histories = a.solve()
        errors.append(max(np.max(abs(network_observed(a,h)-r)/NETWORK_SCALES)
                          for h,r in zip(histories, network_reference(histories, variant, continuous=True))))
    assert errors[1] < .65*errors[0] and errors[2] < .65*errors[1]


@pytest.mark.parametrize('change', ['tolerance', 'maximum_step'])
def test_network_adaptive_refinement(change):
    case, config, thermo = network_load('growth')
    if change == 'tolerance':
        case['numerics']['relative_tolerance'] /= 10.
        case['numerics']['absolute_tolerance'] /= 10.
    else:
        case['numerics']['maximum_step']['value'] /= 2.
    a = build_bioreactor(case, config, thermo)
    histories = a.solve()
    assert max(np.max(abs(network_observed(a,h)-r)/NETWORK_SCALES)
               for h,r in zip(histories, network_reference(histories, 'growth'))) < 1e-6


# Constraint qualification

CONSTRAINT_BASE = Path(__file__).parents[2]/'examples/bioreactors/generic_fed_batch/reconciliation/level4'
CONSTRAINT_CASES = ['relative', 'relaxed', 'untargeted', 'normalization', 'near_zero',
         'redundant', 'equivalent_units']
CONSTRAINT_IDS = ['ua', 'ub', 'p', 'q', 'g']
CONSTRAINT_MAPPING = np.array([[1., 1.], [0., 1.], [1., 0.], [0., 2.], [0., 0.]])


def constraint_load(name):
    folder = CONSTRAINT_BASE/'inputs'/name
    return (json.loads((folder/'case.json').read_text()),
            json.loads((folder/'mechanism.json').read_text()), folder/'thermo.json')


def constraint_reference(config, solve=np.linalg.solve):
    """Enumerate unconstrained, edge and vertex minima, independently of SLSQP."""
    policy = config['model']['reconciliation']
    targets = np.array([config['parameters']['rates'][n] for n in CONSTRAINT_IDS])
    selected = np.array([n not in policy['untargeted_exchanges'] for n in CONSTRAINT_IDS])
    # All campaign fixtures declare the same physical mmol floor, independently
    # converted here without calling the production unit converter.
    amount_unit = config['model']['kinetics']['flux_basis']['exchange']['amount_unit']
    floor = 1e-6 * {'mmol': 1., 'umol': 1000.}[amount_unit]
    scales = np.maximum(abs(targets[selected]), floor)
    total = max(abs(sum(t for n,t,keep in zip(CONSTRAINT_IDS,targets,selected)
                        if keep and n not in policy['normalization_exclusions'])), floor)
    rows, values = CONSTRAINT_MAPPING[selected]/scales[:,None], targets[selected]/scales
    hessian, linear = rows.T@rows+np.eye(2)/total**2, rows.T@values
    coordinate = 1./np.sqrt(np.diag(hessian))
    hessian = coordinate[:,None]*hessian*coordinate
    linear = coordinate*linear
    reaction_rows = dict(zip(CONSTRAINT_IDS, CONSTRAINT_MAPPING))
    reaction_rows.update(x=np.array([1.,0.]), y=np.array([0.,1.]))
    # These fixtures declare envelopes 0.5 or 1.0 with increment 0.5.
    envelopes = [.5] if policy['reconciliation_relative_bound'] == .5 else [.5,1.]
    for envelope in envelopes:
        constraints, bounds = [], []
        for reaction in config['network']['reactions']:
            name = reaction['identifier']
            low, high = reaction['lower'], reaction['upper']
            if name in CONSTRAINT_IDS and selected[CONSTRAINT_IDS.index(name)]:
                t = targets[CONSTRAINT_IDS.index(name)]
                low, high = max(low,(1.-envelope)*t), min(high,(1.+envelope)*t)
            constraints.extend([reaction_rows[name], -reaction_rows[name]])
            bounds.extend([high, -low])
        a, b = np.array(constraints)*coordinate, np.array(bounds)
        norm = np.linalg.norm(a,axis=1)
        if np.any(b[norm == 0.] < 0.):
            continue
        keep = norm > 0.
        a, b = a[keep]/norm[keep,None], b[keep]/norm[keep]
        candidates = [solve(hessian,linear)]
        for row,bound in zip(a,b):
            system = np.block([[hessian,row[:,None]],[row[None,:],np.zeros((1,1))]])
            candidates.append(solve(system,np.r_[linear,bound])[:2])
        for i,j in combinations(range(len(b)),2):
            matrix = a[[i,j]]
            if abs(np.linalg.det(matrix)) > 1e-12:
                candidates.append(solve(matrix,b[[i,j]]))
        feasible = [z*coordinate for z in candidates if np.max(a@z-b) <= 1e-9]
        if feasible:
            def objective(z):
                return float(np.sum((rows@z-values)**2)+np.sum(z**2)/total**2)
            z = min(feasible,key=objective)
            return z, objective(z), envelope
    raise ValueError('Independent reference is infeasible')


@pytest.mark.parametrize('name', CONSTRAINT_CASES)
def test_constraint_policy_optimum(name):
    case, config, thermo = constraint_load(name)
    mechanism = build_bioreactor(case,config,thermo).mechanism
    targets = {n:config['parameters']['rates'][n] for n in CONSTRAINT_IDS}
    result = mechanism._solve(targets)
    z, objective, envelope = constraint_reference(config)
    flux = dict(zip(result.reaction_ids,result.fluxes))
    unit_scale = 1000. if name == 'equivalent_units' else 1.
    np.testing.assert_allclose(np.array([flux['x'],flux['y']])/unit_scale,
                               z/unit_scale,atol=2e-6,rtol=0.)
    np.testing.assert_allclose(np.array([flux[n] for n in CONSTRAINT_IDS])/unit_scale,
                               CONSTRAINT_MAPPING@z/unit_scale,atol=2e-6,rtol=0.)
    assert -result.primary_objective == pytest.approx(objective,abs=1e-8,rel=1e-8)
    assert mechanism.last_reconciliation_relative_bound == envelope
    assert result.mass_balance_residual_inf/unit_scale < 1e-8
    assert result.bound_violation_inf/unit_scale < 1e-8


@pytest.mark.parametrize('name', ['relative','relaxed','untargeted'])
def test_constraint_forward_balances(name):
    case, config, thermo = constraint_load(name)
    a = build_bioreactor(case,config,thermo)
    histories = a.solve()
    z, _, _ = constraint_reference(config)
    ua,ub,p,q,g = CONSTRAINT_MAPPING@z
    state = np.array([10.,10.,0.,0.,1.,0.,1.,0.])
    scales = np.array([10.,10.,2.,.1,1.,.1,1.,.001])
    species = [list(a.phase.name_species).index(n) for n in ('nutrient_a','nutrient_b','byproduct')]
    for h in histories:
        times = h.time/86400.
        if np.isclose(times[0],.5):
            state[:2] += [.6,.2]
            state[6] += .1
        if np.isclose(times[0],.75):
            state[:6] *= 1.-.1/state[6]
            state[6] -= .1
        expected = [state.copy()]
        for dt in np.diff(times):
            state[:4] += dt*state[4]*np.array([-ua,-ub,q,.1*p])
            state[7] += dt*state[4]/(1000.*state[6])
            state[6] += .1*dt
            expected.append(state.copy())
        actual = np.column_stack((h.mass_j_liquid0[:,species]*10000.,h.product_g_liquid0,
                                  h.viable_cells_million_liquid0,h.dead_cells_million_liquid0,
                                  np.asarray(h.vessel_vol).reshape(-1)*1000.,
                                  h.ivcd_million_cell_day_per_ml_liquid0))
        assert np.isfinite(actual).all() and actual.min() >= -1e-9
        assert np.max(abs(actual-expected)/scales) < 1e-6


def test_constraint_infeasible_rejected():
    case,config,thermo = constraint_load('infeasible')
    with pytest.raises(ValueError,match='reference is infeasible'):
        constraint_reference(config)
    with pytest.raises(MetabolicInfeasibleError):
        build_bioreactor(case,config,thermo).solve()


def test_constraint_recovery_retains_problem_or_fails(monkeypatch):
    import PharmaPy.Metabolic.closures.reconciled as module
    case,config,thermo = constraint_load('relative')
    mechanism = build_bioreactor(case,config,thermo).mechanism
    z, objective, _ = constraint_reference(config)
    def failed(objective,seed,**kwargs):
        return SimpleNamespace(x=seed,fun=objective(seed),success=False,status=9,
                               message='Injected primary optimizer failure')
    monkeypatch.setattr(module,'minimize',failed)
    try:
        result = mechanism._solve({n:config['parameters']['rates'][n] for n in CONSTRAINT_IDS})
    except MetabolicNumericalError:
        return
    flux = dict(zip(result.reaction_ids,result.fluxes))
    np.testing.assert_allclose([flux['x'],flux['y']],z,atol=2e-6,rtol=0.)
    assert -result.primary_objective == pytest.approx(objective,abs=1e-8,rel=1e-8)
    assert result.mass_balance_residual_inf < 1e-8 and result.bound_violation_inf < 1e-8


def constraint_decimal_solver(precision):
    def solve(matrix, vector):
        with localcontext() as ctx:
            ctx.prec = precision
            a = [[Decimal(str(v)) for v in row]+[Decimal(str(b))] for row,b in zip(matrix,vector)]
            n = len(a)
            for i in range(n):
                pivot = max(range(i,n),key=lambda k:abs(a[k][i]))
                a[i],a[pivot] = a[pivot],a[i]
                value = a[i][i]
                a[i] = [v/value for v in a[i]]
                for k in range(n):
                    if k != i:
                        value = a[k][i]
                        a[k] = [v-value*w for v,w in zip(a[k],a[i])]
            return np.array([float(row[-1]) for row in a])
    return solve


@pytest.mark.parametrize('name', CONSTRAINT_CASES)
def test_constraint_reference_precision(name):
    _, config, _ = constraint_load(name)
    normal = constraint_reference(config)
    for precision in (40,80):
        refined = constraint_reference(config,solve=constraint_decimal_solver(precision))
        scale = 1000. if name == 'equivalent_units' else 1.
        np.testing.assert_allclose(normal[0]/scale,refined[0]/scale,atol=4e-7,rtol=0.)
        assert normal[1] == pytest.approx(refined[1],abs=2e-9,rel=2e-9)
        assert normal[2] == refined[2]


def constraint_floor_case(factor):
    case, config, thermo = constraint_load('relative')
    config['parameters']['rates'].update(ua=1.5e-7,ub=5e-8,p=1e-7,q=2e-7,g=0.)
    if factor != 1.:
        for name in CONSTRAINT_IDS:
            config['parameters']['rates'][name] *= factor
        for reaction in config['network']['reactions']:
            reaction['lower'] *= factor
            reaction['upper'] *= factor
        for name,basis in config['model']['kinetics']['flux_basis'].items():
            basis['amount_unit'] = 'umol'
            if name == 'growth':basis['amount_per_million_cells'] *= factor
        for name in CONSTRAINT_IDS:config['rate_provider']['output_units'][name]='umol/(10^6 cell day)'
    return case,config,thermo


def test_constraint_floor_unit_equivalence():
    physical = []
    for factor in (1.,1000.):
        case,config,thermo = constraint_floor_case(factor)
        a = build_bioreactor(case,config,thermo)
        result = a.mechanism._solve({n:config['parameters']['rates'][n] for n in CONSTRAINT_IDS})
        flux = dict(zip(result.reaction_ids,result.fluxes))
        physical.append(np.array([flux['x'],flux['y']])/factor)
    # Scale to the small physical target, not to one mmol: expose floor dependence.
    assert np.max(abs(physical[0]-physical[1]))/1e-7 < 1e-6


def constraint_nonlinear_reference(points=257):
    """Reduce the nonlinear QP to total uptake s=x+y and solve each y exactly."""
    mapping = np.array([[1.,1.],[0.,1.],[.8,0.],[0.,1.8],[.2,.2]])
    target = np.array([1.5,.5,.8,.9,.3])
    rows = mapping/target[:,None]
    h = rows.T@rows+np.eye(2)/target.sum()**2
    linear = rows.T@np.ones(5)
    exposure = lambda s:.05*(1.+np.exp((.1*s-.05)*.1))
    top = brentq(lambda s:s*exposure(s)-.12,.75,2.25,xtol=1e-14)
    direction = np.array([-1.,1.])
    def candidate(s):
        low = max(.25,s-1.5)
        high = min(.75,s-.5,.04/exposure(s))
        if low > high+1e-12:return np.array([np.nan,np.nan]),np.inf
        origin = np.array([s,0.])
        y = np.clip(direction@(linear-h@origin)/(direction@h@direction),low,high)
        z = np.array([s-y,y])
        objective = np.sum((rows@z-1.)**2)+np.sum(z*z)/target.sum()**2
        return z,float(objective)
    grid = np.linspace(.75,top,points)
    choices = [candidate(s) for s in grid]
    for i in range(1,len(grid)-1):
        if choices[i][1] <= min(choices[i-1][1],choices[i+1][1]):
            opt = minimize_scalar(lambda s:candidate(s)[1],bounds=(grid[i-1],grid[i+1]),
                                  method='bounded',options={'xatol':1e-14})
            choices.append(candidate(opt.x))
    z,obj = min(choices,key=lambda c:c[1])
    return z,obj,exposure(z.sum())


@pytest.mark.parametrize('seed', [None,(.5,.25),(.6,.3),(1.,.5)])
def test_constraint_nonlinear_multiple_starts(seed):
    case,config,thermo = constraint_load('nonlinear')
    a = build_bioreactor(case,config,thermo)
    if seed is not None:
        x,y = seed
        flux = dict(ua=x+y,ub=y,x=x,y=y,p=.8*x,q=1.8*y,g=.2*(x+y))
        a.mechanism.previous_solution = SimpleNamespace(fluxes=np.array([flux[n] for n in a.mechanism.network.reaction_ids]))
    h, = a.solve()
    z,obj,exposure = constraint_nonlinear_reference()
    result = a.mechanism.last_solution
    actual = dict(zip(result.reaction_ids,result.fluxes))
    np.testing.assert_allclose([actual['x'],actual['y']],z,atol=2e-6,rtol=0.)
    assert -result.primary_objective == pytest.approx(obj,abs=1e-8,rel=1e-8)
    indexes = [list(a.phase.name_species).index(n) for n in ('nutrient_a','nutrient_b','byproduct')]
    expected = [.12-z.sum()*exposure,.04-z[1]*exposure,1.8*z[1]*exposure]
    np.testing.assert_allclose(h.mass_j_liquid0[-1,indexes]*10000.,expected,atol=1e-7,rtol=0.)
    assert h.viable_cells_million_liquid0[-1] == pytest.approx(np.exp((.1*z.sum()-.05)*.1),abs=1e-7)
    assert h.product_g_liquid0[-1] == pytest.approx(.08*z[0]*exposure,abs=1e-8)


def test_constraint_nonlinear_reference_refinement():
    coarse = constraint_nonlinear_reference(257)
    fine = constraint_nonlinear_reference(2049)
    np.testing.assert_allclose(coarse[0],fine[0],atol=4e-7,rtol=0.)
    assert coarse[1] == pytest.approx(fine[1],abs=2e-9,rel=2e-9)


def test_constraint_forward_floor_unit_equivalence():
    physical = []
    for factor in (1.,1000.):
        a = build_bioreactor(*constraint_floor_case(factor))
        h = a.solve()[0]
        dt = (h.time[1]-h.time[0])/86400.
        j = list(a.phase.name_species).index('byproduct')
        x = (h.product_g_liquid0[1]-h.product_g_liquid0[0])/(.1*dt)
        y = (h.mass_j_liquid0[1,j]-h.mass_j_liquid0[0,j])*10000./(2.*dt)
        physical.append(np.array([x,y]))
    assert np.max(abs(physical[0]-physical[1]))/1e-7 < 1e-6


def constraint_small_closure(matrix, lower, upper, internal=()):
    from PharmaPy.Metabolic.network import MetabolicNetworkDefinition, ReactionDefinition
    from PharmaPy.Metabolic.closures.reconciled import RateReconciledMFAClosure
    matrix = np.asarray(matrix, dtype=float)
    metabolites = tuple(f'm{i}' for i in range(len(matrix)))
    reactions = tuple(ReactionDefinition(
        f'r{j}', dict(zip(metabolites, matrix[:, j])), lo, hi)
        for j, (lo, hi) in enumerate(zip(lower, upper)))
    network = MetabolicNetworkDefinition('edge', '1', metabolites, reactions, {'r0': 1.})
    return RateReconciledMFAClosure(network, internal)


def constraint_solve_small(closure, value, scale=1.):
    from PharmaPy.Metabolic.closures.base import MetabolicEnvironment
    from PharmaPy.Metabolic.closures.reconciled import ReconciliationTarget
    return closure.solve(MetabolicEnvironment({}, {}),
                         targets=(ReconciliationTarget('r0', value, scale),), internal_scale=1.)


@pytest.mark.parametrize('factor', [1e-10, 1., 1e10])
def test_fixed_nonzero_flux_and_dependent_rows(factor):
    closure = constraint_small_closure([[1., -1.], [2., -2.]], [0., 2.*factor], [4.*factor, 2.*factor])
    result = constraint_solve_small(closure, factor, factor)
    np.testing.assert_allclose(result.fluxes/factor, [2., 2.], atol=1e-8)


@pytest.mark.parametrize('consistent', [True, False])
def test_all_fixed_fluxes(consistent):
    values = [2., 2. if consistent else 3.]
    closure = constraint_small_closure([[1., -1.]], values, values)
    if consistent:
        np.testing.assert_array_equal(constraint_solve_small(closure, 1.).fluxes, values)
    else:
        with pytest.raises(MetabolicInfeasibleError):
            constraint_solve_small(closure, 1.)


def test_inconsistent_eliminated_row():
    closure = constraint_small_closure([[0., 1.]], [0., 2.], [4., 2.])
    with pytest.raises(MetabolicInfeasibleError):
        constraint_solve_small(closure, 1.)


@pytest.mark.parametrize('value', [-2., 0., 2.])
def test_signed_zero_targets_and_unpenalized_directions(value):
    closure = constraint_small_closure([[0., 0.]], [-3., -3.], [3., 3.])
    result = constraint_solve_small(closure, value)
    assert result.fluxes[0] == pytest.approx(value, abs=1e-7)
    assert -3. <= result.fluxes[1] <= 3.
    assert -result.primary_objective < 1e-12


def test_nearly_dependent_rows_preserve_original_balances():
    closure = constraint_small_closure([[1., -1., 0.], [1., -1., 1e-10]],
                            [0., 0., -1.], [3., 3., 1.])
    result = constraint_solve_small(closure, 2.)
    np.testing.assert_allclose(result.fluxes, [2., 2., 0.], atol=1e-7)


def test_false_optimizer_success_is_rejected(monkeypatch):
    from PharmaPy.Metabolic.closures import reconciled
    # A bounded target forces optimization instead of the exact-target fast path.
    closure = constraint_small_closure([[1., -1.]], [0., 0.], [1., 1.])
    def false_success(fun, x0, **kwargs):
        x = np.zeros_like(x0)
        return SimpleNamespace(x=x, fun=fun(x), success=True, status=0, message='false success')
    monkeypatch.setattr(reconciled, 'minimize', false_success)
    with pytest.raises(MetabolicNumericalError, match='stationarity'):
        constraint_solve_small(closure, 2.)


def test_declared_scaling_through_builder_and_copy():
    from copy import deepcopy
    case, config, thermo = constraint_floor_case(1000.)
    declaration = {'value': 1e-6, 'basis': {
        'amount_unit': 'mmol', 'normalization': '10^6 cell', 'time_unit': 'day'}}
    config['model']['reconciliation']['reconciliation_scaling'] = {
        'target_floor': declaration, 'internal_scale': declaration,
        'target_floors': {'ua': {'value': 2e-6, 'basis': declaration['basis']}}}
    mechanism = build_bioreactor(case, config, thermo).mechanism
    assert mechanism.target_floor == pytest.approx(.001)
    assert mechanism.target_floors['ua'] == pytest.approx(.002)
    assert mechanism.internal_scale == pytest.approx(.001)
    targets = {n: config['parameters']['rates'][n] for n in CONSTRAINT_IDS}
    np.testing.assert_allclose(deepcopy(mechanism)._solve(targets).fluxes,
                               mechanism._solve(targets).fluxes, atol=1e-10)


def test_tiny_all_fixed_inconsistency_is_not_roundoff():
    closure = constraint_small_closure([[1., -1.]], [2e-12, 3e-12], [2e-12, 3e-12])
    with pytest.raises(MetabolicInfeasibleError):
        constraint_solve_small(closure, 1e-12, 1e-12)


@pytest.mark.parametrize('bad', [
    {'unknown': 1}, {'target_floors': {'missing_reaction': {'value': 1., 'basis': None}}},
    {'target_floor': {'value': 0., 'basis': None}},
])
def test_invalid_reconciliation_policy_scales_fail_at_construction(bad):
    case, config, thermo = constraint_load('relative')
    config['model']['reconciliation']['reconciliation_scaling'] = bad
    with pytest.raises(ValueError):
        build_bioreactor(case, config, thermo)


def test_mixed_basis_scales_require_compatible_declarations():
    case, config, thermo = constraint_load('relative')
    config['model']['kinetics']['flux_basis']['growth'] = {
        'amount_unit': '1', 'normalization': 'none', 'time_unit': 'day'}
    config['rate_provider']['output_units']['g'] = '1/day'
    # The common molecular floor cannot be used for a specific growth target.
    with pytest.raises(ValueError, match='not interchangeable'):
        build_bioreactor(case, config, thermo)
    policy = config['model']['reconciliation']['reconciliation_scaling']
    policy['target_floors'] = {'g': {'value': 1e-6, 'basis': {
        'amount_unit': '1', 'normalization': 'none', 'time_unit': 'day'}}}
    with pytest.raises(ValueError, match='mixed flux coordinates'):
        build_bioreactor(case, config, thermo)
    policy['internal_scale'] = dict(policy['target_floor'], value=1.)
    mechanism = build_bioreactor(case, config, thermo).mechanism
    result = mechanism._solve({n: config['parameters']['rates'][n] for n in CONSTRAINT_IDS})
    assert np.isfinite(result.fluxes).all()


def test_null_space_retry_preserves_nonzero_fixed_balance(monkeypatch):
    from PharmaPy.Metabolic.closures import reconciled
    closure = constraint_small_closure([[1., -1., -1.]], [0., 0., 1.], [3., 3., 1.])
    original = reconciled.minimize
    calls = []
    def stagnate(fun, x0, **kwargs):
        calls.append(len(x0))
        if len(calls) <= 2:
            return SimpleNamespace(x=x0, fun=fun(x0), success=False, status=9, message='stagnation')
        return original(fun, x0, **kwargs)
    monkeypatch.setattr(reconciled, 'minimize', stagnate)
    result = constraint_solve_small(closure, 2.)
    np.testing.assert_allclose(result.fluxes, [2., 1., 1.], atol=1e-8)
    assert calls == [2, 2, 1]


def test_null_space_roundoff_restores_declared_bounds(monkeypatch):
    from PharmaPy.Metabolic.closures import reconciled
    closure = constraint_small_closure([[1., -1.]], [0., 0.], [3., 3.])
    original = reconciled.minimize
    count = 0
    def perturb(fun, x0, **kwargs):
        nonlocal count
        count += 1
        if count <= 2:
            return SimpleNamespace(x=x0, fun=fun(x0), success=False, status=9, message='stagnation')
        result = original(fun, x0, **kwargs)
        result.x -= 1e-14 / kwargs['constraints'][0].A[0, 0]
        return result
    monkeypatch.setattr(reconciled, 'minimize', perturb)
    result = constraint_solve_small(closure, -1.)
    assert np.min(result.fluxes) >= 0.
    assert result.mass_balance_residual_inf < 1e-12
    assert -result.primary_objective == pytest.approx(1.)


def test_flux_variability_retains_analytical_intervals():
    from PharmaPy.Metabolic.closures.base import MetabolicEnvironment
    from PharmaPy.Metabolic.closures.reconciled import ReconciliationTarget
    closure = constraint_small_closure([[1., -1.]], [0., 0.], [3., 3.], internal=('r0',))
    result = closure.solve(MetabolicEnvironment({}, {}),
                           targets=(ReconciliationTarget('r0', 2., 1.),), internal_scale=1.,
                           fva_reactions=('r0',), fva_fraction=.1)
    # J=x^2+(x-2)^2=2+2(x-1)^2; the 10% ceiling is 2.2.
    np.testing.assert_allclose(result.fva_intervals['r0'],
                               [1.-np.sqrt(.1), 1.+np.sqrt(.1)], atol=1e-7)


# Operation qualification

OPERATION_BASE = Path(__file__).parents[2]/'examples/bioreactors/generic_fed_batch/reconciliation/level5'
OPERATION_CASES = sorted(p.name for p in (OPERATION_BASE/'inputs').iterdir() if p.is_dir())
OPERATION_SCALES = np.array([1.,1.,1.,.1,1.,1.,1.,.0001])


def operation_load(name, backend='fixed-step', step=.001):
    folder = OPERATION_BASE/'inputs'/name
    case = json.loads((folder/'case.json').read_text())
    config = json.loads((folder/'mechanism.json').read_text())
    case['numerics']['step']['value'] = step
    if backend == 'scipy':
        case['numerics'].update(backend='scipy', relative_tolerance=1e-10,
            absolute_tolerance=1e-15, maximum_step={'value':.002,'unit':'day'},
            jacobian_relative_step=1e-5, jacobian_state_scale=1e-6)
    return case, config, folder/'thermo.json'


def operation_optimum(config, limits=(np.inf,np.inf)):
    """Enumerate the unconstrained minimum, edges and vertices of a 2-D QP."""
    growth = config['parameters']['rates']['g'] > 0.
    b = .2 if growth else 0.
    mapping = np.array([[1.,1.],[0.,1.],[1.-b,0.],[0.,2.-b],[b,b]])
    target = np.array([config['parameters']['rates'][n] for n in ('ua','ub','p','q','g')])
    relative = config['model']['reconciliation']['reconciliation_policy'] == 'relative-regularized'
    weights = 1./np.maximum(abs(target),1e-6) if relative else np.ones(5)
    rows = weights[:,None]*mapping
    hessian = rows.T@rows + (np.eye(2)/max(abs(sum(target)),1e-6)**2 if relative else 0.)
    linear = rows.T@(weights*target)
    constraints = np.vstack((-np.eye(2),mapping,[1.,1.],[0.,1.]))
    bounds = np.r_[0.,0.,np.full(5,10.),limits]
    if relative:
        constraints = np.vstack((constraints,mapping,-mapping))
        bounds = np.r_[bounds,1.5*target,-.5*target]
    keep = np.isfinite(bounds) & (np.linalg.norm(constraints,axis=1)>0.)
    constraints,bounds = constraints[keep],bounds[keep]
    candidates = [np.linalg.solve(hessian,linear)]
    for row,bound in zip(constraints,bounds):
        system = np.block([[hessian,row[:,None]],[row[None,:],np.zeros((1,1))]])
        candidates.append(np.linalg.solve(system,np.r_[linear,bound])[:2])
    for i,j in combinations(range(len(bounds)),2):
        if abs(np.linalg.det(constraints[[i,j]])) > 1e-12:
            candidates.append(np.linalg.solve(constraints[[i,j]],bounds[[i,j]]))
    candidates = [z for z in candidates if np.max(constraints@z-bounds) <= 1e-11]
    if not candidates:
        raise ValueError('independent policy infeasible')
    z = min(candidates,key=lambda z:z@hessian@z-2*linear@z)
    return mapping@z


def operation_initial(config):
    state = config['state']
    return np.array([state['species']['amounts'][n] for n in ('nutrient_a','nutrient_b','byproduct')]
        +[state['product_g'],state['viable_cells_million'],state['dead_cells_million'],1.,state['integral']])


def operation_reference(case, config, histories, continuous=False, precision=1.):
    """Piecewise exact inventories/jumps; independent tight ODE only for growth."""
    state, inlets, answer = operation_initial(config), {}, []
    native = case['operation']['volume_policy'] == 'native'
    growth = config['parameters']['rates']['g'] > 0.
    for history in histories:
        times = history.time/86400.
        for event in config['recipes']['benchmark']:
            if abs(event['time']['value']-times[0]) > 1e-12:
                continue
            kind = event['event_type']
            if kind == 'flow':
                flow = event['volume_flow']['value']
                conc = event['concentrations']['values']
                inlets[event.get('inlet','inlet')] = (flow,flow*np.array([conc.get(n,0.) for n in ('nutrient_a','nutrient_b')]))
            elif kind == 'sample':
                volume = event['volume']['value']
                state[:6] *= 1.-volume/state[6]
                state[6] -= volume
            else:
                volume = event['volume']['value']
                state[:2] += volume*np.array([event['concentrations']['values'].get(n,0.) for n in ('nutrient_a','nutrient_b')])
                state[6] += volume
        flow = sum(v[0] for v in inlets.values())
        supply = sum((v[1] for v in inlets.values()),start=np.zeros(2))
        if growth and continuous:
            death = config['parameters']['rates']['death']
            def rhs(t,y):
                limits = np.where(y[:2] <= 1e-12,supply/y[4],np.inf)
                ua,ub,p,q,g = operation_optimum(config,limits)
                net = supply-y[4]*np.array([ua,ub])
                net[abs(net)<1e-13] = 0.
                return [*net,q*y[4],.1*p*y[4],
                        (g/2.-death)*y[4],death*y[4],flow,y[4]/(1000.*y[6])]
            solution = solve_ivp(rhs,(times[0],times[-1]),state,t_eval=times,
                                 method='DOP853',rtol=3e-13,atol=3e-15*precision,max_step=.001*precision)
            assert solution.success
            rows = solution.y.T
        else:
            rows = [state.copy()]
            for dt in np.diff(times):
                remaining = dt
                while remaining > 1e-14:
                    limits = (np.where(state[:2] <= 1e-12,supply/state[4],np.inf) if continuous
                              else (np.maximum(state[:2],0.)/remaining+supply)/state[4])
                    ua,ub,p,q,g = operation_optimum(config,limits)
                    death = config['parameters']['rates']['death']
                    if growth:
                        capacity = np.maximum(state[:2],0.)/remaining+supply
                        def exposure_balance(average):
                            growth_flux = operation_optimum(config,capacity/average)[4]
                            return average-.5*state[4]*(1.+np.exp((growth_flux/2.-death)*remaining))
                        average = brentq(exposure_balance,.5*state[4],
                            .5*state[4]*(1.+np.exp(5.*remaining)),xtol=1e-14)
                        ua,ub,p,q,g = operation_optimum(config,capacity/average)
                    end_viable = state[4]*np.exp((g/2.-death)*remaining) if growth else state[4]
                    exposure = .5*(state[4]+end_viable)
                    derivative = np.r_[supply-exposure*np.array([ua,ub]),q*exposure,.1*p*exposure]
                    duration = remaining
                    if continuous:
                        consuming = (derivative[:2] < -1e-12) & (state[:2] > 1e-12)
                        duration = min(duration,np.min(state[:2][consuming]/-derivative[:2][consuming],initial=np.inf))
                    volume_rate = flow-(.0001*(p+g)*exposure if native else 0.)
                    if continuous and abs(volume_rate)>1e-15:
                        state[7] += exposure/(1000.*volume_rate)*np.log1p(volume_rate*duration/state[6])
                    else:
                        state[7] += duration*exposure/(1000.*state[6])
                    state[:4] += duration*derivative
                    state[6] += volume_rate*duration
                    if growth:
                        state[5] += state[4]*(1.-np.exp(-death*duration))
                        state[4] = end_viable
                    remaining -= duration
                rows.append(state.copy())
            rows = np.asarray(rows)
        answer.append(rows)
        state = rows[-1].copy()
    return answer


def operation_run(case, config, thermo):
    assembly = build_bioreactor(case,config,thermo)
    report = dict(closure_calls=0,max_scaled_feasibility=0.)
    solve = assembly.mechanism.closure.solve
    def checked(*args,**kwargs):
        solution = solve(*args,**kwargs)
        report['closure_calls'] += 1
        scale = max(1.,max(abs(solution.fluxes)))
        operation_error = max(solution.mass_balance_residual_inf,solution.bound_violation_inf)/scale
        availability = kwargs.get('availability')
        if availability is not None:
            operation_error = max(operation_error,-np.min(availability.residual(solution.fluxes)))
        report['max_scaled_feasibility'] = max(report['max_scaled_feasibility'],float(operation_error))
        assert operation_error <= 1e-8
        return solution
    assembly.mechanism.closure.solve = checked
    previous = signal.getsignal(signal.SIGALRM)
    def timeout(*args):
        raise TimeoutError('Level 5 small-system run exceeded the 60-second campaign guard')
    signal.signal(signal.SIGALRM,timeout)
    signal.alarm(60)
    start = time.perf_counter()
    try:
        histories = assembly.solve()
    finally:
        report['runtime_s'] = time.perf_counter()-start
        signal.alarm(0)
        signal.signal(signal.SIGALRM,previous)
    return assembly,histories,report


def operation_error(assembly,histories,expected):
    return max(float(np.max(abs(network_observed(assembly,h)-r)/OPERATION_SCALES)) for h,r in zip(histories,expected))


def operation_record_refinement(name, values):
    if os.environ.get('LEVEL5_RECORD') == '1':
        path = OPERATION_BASE/'outputs/refinement.json'
        report = json.loads(path.read_text()) if path.exists() else {}
        report[name] = values
        path.write_text(json.dumps(report,indent=2)+'\n')


@pytest.mark.parametrize('name',OPERATION_CASES)
@pytest.mark.parametrize('backend',['fixed-step','scipy'])
def test_operation_campaign(name,backend):
    case,config,thermo = operation_load(name,backend)
    assembly,histories,report = operation_run(case,config,thermo)
    expected = operation_reference(case,config,histories,continuous=backend=='scipy')
    report['max_scaled_trajectory_error'] = operation_error(assembly,histories,expected)
    assert report['max_scaled_trajectory_error'] < 1e-6
    for h in histories:
        assert np.isfinite(network_observed(assembly,h)).all()
        assert np.min(h.mass_j_liquid0/np.array(assembly.phase.mw)*1e6) >= -1e-9
    for before,after in zip(histories,histories[1:]):
        assert before.time[-1] == after.time[0]
    if name in ('ordered','reversed'):
        assert network_observed(assembly,histories[-1])[-1,0] == pytest.approx(1.65 if name=='ordered' else 1.70,abs=1e-10)
    if name in ('growth','growth_supply'):
        tighter = operation_reference(case,config,histories,continuous=True,precision=.25)
        nominal = operation_reference(case,config,histories,continuous=True)
        assert max(np.max(abs(a-b)/OPERATION_SCALES) for a,b in zip(tighter,nominal)) < 2e-7
    if os.environ.get('LEVEL5_RECORD')=='1':
        folder = OPERATION_BASE/'outputs'/name/backend
        folder.mkdir(parents=True,exist_ok=True)
        times = np.concatenate([h.time/86400. for h in histories])
        header = ','.join(['time_day',*NETWORK_FIELDS])
        for file,values in [('trajectories',np.vstack([network_observed(assembly,h) for h in histories])),('reference',np.vstack(expected))]:
            np.savetxt(folder/(file+'.csv'),np.column_stack((times,values)),delimiter=',',header=header,comments='')
        report.update(case=name,backend=backend,passed=True,reference='continuous' if backend=='scipy' else 'discrete')
        (folder/'verification.json').write_text(json.dumps(report,indent=2)+'\n')


@pytest.mark.parametrize('name',['competing','schedule','combined','growth','growth_supply'])
def test_operation_fixed_refinement(name):
    errors=[]
    for step in (.002,.001,.0005):
        case,config,thermo=operation_load(name,step=step)
        assembly,histories,_=operation_run(case,config,thermo)
        assert operation_error(assembly,histories,operation_reference(case,config,histories)) < 1e-6
        errors.append(operation_error(assembly,histories,operation_reference(case,config,histories,continuous=True)))
    assert errors[2] < 1e-3
    assert max(errors) < 1e-6 or errors[2] < errors[1] < errors[0]
    operation_record_refinement('fixed_'+name,dict(steps_day=[.002,.001,.0005],scaled_errors=errors))


@pytest.mark.parametrize('change',['tolerance','maximum_step'])
@pytest.mark.parametrize('name',['competing','combined','growth','growth_supply'])
def test_operation_adaptive_refinement(name,change):
    case,config,thermo=operation_load(name,'scipy')
    if change=='tolerance':
        case['numerics']['relative_tolerance']/=10.
        case['numerics']['absolute_tolerance']/=10.
    else:
        case['numerics']['maximum_step']['value']/=2.
    a,h,_=operation_run(case,config,thermo)
    maximum = operation_error(a,h,operation_reference(case,config,h,continuous=True))
    assert maximum < 1e-6
    operation_record_refinement('adaptive_'+name+'_'+change,dict(scaled_error=maximum,numerics=case['numerics']))


@pytest.mark.parametrize('backend',['fixed-step','scipy'])
def test_operation_off_grid_transition_brackets(backend):
    case,config,thermo=operation_load('competing',backend,step=.0005)
    a,h,_=operation_run(case,config,thermo)
    rows=network_observed(a,h[0]);times=h[0].time/86400.
    # B is exhausted at .0113/.5. Thereafter x=1.25 with y=0.
    for column,exact in ((1,.0226),(0,.0226+(.037-1.5*.0226)/1.25)):
        first=np.flatnonzero(rows[:,column] <= 1e-9)[0]
        assert times[first]-times[first-1] <= .0005+1e-12
        assert times[first-1]-1e-10 <= exact <= times[first]+1e-10


@pytest.mark.parametrize('name',['endpoints','combined'])
@pytest.mark.parametrize('backend',['fixed-step','scipy'])
def test_operation_reset_rebuild(name,backend):
    case,config,thermo=operation_load(name,backend)
    a,h,_=operation_run(case,config,thermo)
    a.reset();repeated=a.solve()
    b,fresh,_=operation_run(case,config,thermo)
    for x,y,z in zip(h,repeated,fresh):
        np.testing.assert_allclose(network_observed(a,x),network_observed(a,y),atol=1e-12,rtol=1e-9)
        np.testing.assert_allclose(network_observed(a,x),network_observed(b,z),atol=1e-12,rtol=1e-9)


@pytest.mark.parametrize('backend',['fixed-step','scipy'])
def test_operation_native_volume(backend):
    case,config,thermo=operation_load('combined',backend)
    case['operation']['volume_policy']='native'
    a,h,_=operation_run(case,config,thermo)
    assert operation_error(a,h,operation_reference(case,config,h,continuous=backend=='scipy')) < 1e-6


@pytest.mark.parametrize('bad',[('sample',-1.),('sample',1.),('sample',2.),('flow',-1.)])
def test_operation_invalid_operations_are_atomic(bad):
    case,config,thermo=operation_load('combined')
    kind,value=bad
    event=deepcopy(next(e for e in config['recipes']['benchmark'] if e['event_type']==kind))
    event['time']['value']=0.
    event['volume' if kind=='sample' else 'volume_flow']['value']=value
    config['recipes']['benchmark']=[event]
    a=build_bioreactor(case,config,thermo)
    initial_mass=a.phase.mass_j.copy()
    with pytest.raises(ValueError):
        a.solve()
    np.testing.assert_array_equal(a.phase.mass_j,initial_mass)
    assert not a._inlets


def test_operation_relative_starvation_is_infeasible():
    case,config,thermo=operation_load('relative')
    config['recipes']['benchmark']=[]
    with pytest.raises(MetabolicInfeasibleError):
        operation_run(case,config,thermo)


@pytest.mark.parametrize('backend',['fixed-step','scipy'])
@pytest.mark.parametrize('name',['combined','relative'])
def test_operation_equivalent_units_and_names(name,backend,tmp_path):
    case,config,thermo=operation_load(name,backend)
    original,histories,_=operation_run(case,config,thermo)
    case['operation']['runtime']={'value':.06*24.,'unit':'h'}
    case['operation']['initial_volume']={'value':1000.,'unit':'mL'}
    case['numerics']['step']={'value':.001*24.,'unit':'h'}
    for event in config['recipes']['benchmark']:
        event['time']={'value':event['time']['value']*24.,'unit':'h'}
        if 'volume' in event:
            event['volume']={'value':event['volume']['value']*1000.,'unit':'mL'}
        if 'volume_flow' in event:
            event['volume_flow']={'value':event['volume_flow']['value']*1000./24.,'unit':'mL/h'}
        if 'concentrations' in event:
            event['concentrations']['unit']='umol/L'
            event['concentrations']['values']={n:v*1000. for n,v in event['concentrations']['values'].items()}
    config['state']['species']['unit']='umol'
    config['state']['species']['amounts']={n:v*1000. for n,v in config['state']['species']['amounts'].items()}
    for name in ('ua','ub','p','q','g'):
        config['parameters']['rates'][name]*=1000.
        config['rate_provider']['output_units'][name]='umol/(10^6 cell day)'
    for basis in config['model']['kinetics']['flux_basis'].values():
        basis['amount_unit']='umol'
        if 'amount_per_million_cells' in basis:
            basis['amount_per_million_cells']*=1000.
    for reaction in config['network']['reactions']:
        reaction['lower']*=1000.;reaction['upper']*=1000.
    names=['nutrient_a','nutrient_b','byproduct']
    identifiers=names+config['network']['internal_metabolites']+[r['identifier'] for r in config['network']['reactions']]+['first','second']
    rename={name:'renamed_'+str(i) for i,name in enumerate(identifiers)}
    def transform(value):
        if isinstance(value,dict):
            return {rename.get(k,k):transform(v) for k,v in reversed(list(value.items()))}
        if isinstance(value,list):
            return [transform(v) for v in value]
        return rename.get(value,value) if isinstance(value,str) else value
    config=transform(config)
    for rule in config['model']['kinetics']['rule_graph']:
        rule['expression']['parameter']='.'.join(rename.get(x,x) for x in rule['expression']['parameter'].split('.'))
    for key in ('reactions','internal_metabolites','exchange_reactions'):
        config['network'][key].reverse()
    renamed_thermo=tmp_path/'thermo.json'
    renamed_thermo.write_text(json.dumps(transform(json.loads(thermo.read_text()))))
    changed,results,_=operation_run(case,config,renamed_thermo)
    assert len(histories)==len(results)
    for a,b in zip(histories,results):
        np.testing.assert_allclose(a.time,b.time,atol=1e-10,rtol=0.)
        assert np.max(abs(network_observed(original,a)-network_observed(changed,b,tuple(rename[n] for n in names)))/OPERATION_SCALES) < 1e-6


def test_operation_continuation_does_not_replay_old_events():
    case,config,thermo=operation_load('combined')
    whole,all_history,_=operation_run(case,config,thermo)
    part=build_bioreactor(case,config,thermo)
    grid=part.time_grid.copy();split=.03*86400.
    part.time_grid=grid[grid<=split]
    first=part.solve()
    part.time_grid=grid[grid>=split]
    second=part.solve()
    np.testing.assert_allclose(network_observed(part,second[-1])[-1],network_observed(whole,all_history[-1])[-1],atol=1e-10,rtol=1e-8)
    assert first[-1].time[-1]==second[0].time[0]


@pytest.mark.parametrize('backend',['fixed-step','scipy'])
def test_operation_pathway_receives_native_continuous_supply(backend):
    folder=OPERATION_BASE.parents[1]/'inputs'
    case=json.loads((folder/'case.json').read_text())
    config=json.loads((folder/'mechanism.json').read_text())
    case['operation']['runtime']={'value':.06,'unit':'h'}
    case['operation']['volume_policy']='working-volume'
    case['numerics']=operation_load('combined',backend)[0]['numerics']
    case['numerics']['step']={'value':.001,'unit':'h'}
    config['model']['pathways']['pathway_bound_rules']=[]
    config['state']['species']['amounts']['nutrient']=0.
    config['recipes']['fed_batch']=[dict(time={'value':0.,'unit':'h'},event_type='flow',
        volume_flow={'value':.1,'unit':'L/h'},concentrations={'unit':'mmol/L','values':{'nutrient':1.}},composition='supplied')]
    a=build_bioreactor(case,config,folder/'thermo.json')
    h=a.solve()[0]
    hours=h.time/3600.
    np.testing.assert_allclose(h.biomass_kg_liquid0,1e-4+1e-5*hours,atol=1e-12,rtol=0.)
    index=list(a.phase.name_species).index('nutrient')
    np.testing.assert_allclose(h.mass_j_liquid0[:,index],0.,atol=1e-12,rtol=0.)
    np.testing.assert_allclose(np.asarray(h.vessel_vol).reshape(-1)*1000.,1.+.1*hours,atol=1e-9,rtol=0.)


@pytest.mark.parametrize('reverse',[False,True])
def test_operation_coincident_mixed_units_keep_declared_order(reverse):
    case,config,thermo=operation_load('reversed' if reverse else 'ordered')
    events=config['recipes']['benchmark']
    events[0]['time']={'value':.003*24.,'unit':'h'}
    events[1]['time']={'value':.003,'unit':'day'}
    a,h,_=operation_run(case,config,thermo)
    assert len(h)==2
    assert network_observed(a,h[-1])[-1,0]==pytest.approx(1.70 if reverse else 1.65,abs=1e-10)


def test_operation_final_endpoint_in_another_unit_is_not_applied():
    case,config,thermo=operation_load('endpoints')
    case['operation']['runtime']={'value':.003,'unit':'day'}
    config['recipes']['benchmark'][-1]['time']={'value':.003*24.,'unit':'h'}
    a,h,_=operation_run(case,config,thermo)
    assert len(h)==1
    assert network_observed(a,h[0])[-1,6]==pytest.approx(1.02,abs=1e-12)


@pytest.mark.parametrize('bad',['negative_concentration','unknown_species','empty_identifier'])
def test_operation_invalid_inlet_preserves_existing_supply(bad):
    case,config,thermo=operation_load('multiple')
    a=build_bioreactor(case,config,thermo)
    a._set_inlet(a.events[0])
    existing=a._inlets.copy()
    event=deepcopy(a.events[1])
    if bad=='negative_concentration':
        event['concentrations_mmol_l']['nutrient_b']=-1.
    elif bad=='unknown_species':
        event['concentrations_mmol_l']['missing']=1.
    else:
        event['inlet']=''
    with pytest.raises(ValueError):
        a._set_inlet(event)
    assert a._inlets==existing
    assert len(a.unit.inlet_connections)==1


# Coupled qualification

@pytest.mark.parametrize('name,shape,nullity',[
    ('coupled04',(10,16),6),('coupled10',(22,36),14),
    ('coupled20',(42,68),26),('coupled20_scaled',(42,68),26)])
def test_coupled_configuration_and_connected_coordinates(name,shape,nullity):
    case,config,thermo=coupled_load(name)
    assembly=build_bioreactor(case,config,thermo)
    assert assembly.input_audit()['feed_composition']=='COMPLETE'
    assert assembly.unit.result is None
    matrix=network_matrix(config)
    assert matrix.shape==shape
    assert np.linalg.matrix_rank(matrix)==shape[0]
    reference=Reference(config)
    assert reference.mapping.shape[1]==nullity
    np.testing.assert_allclose(matrix@reference.mapping,0.,atol=1e-12,rtol=0.)
    # Every reaction belongs to the same metabolite/reaction graph component.
    reached={0}
    adjacency=abs(matrix).T@abs(matrix)>0.
    while True:
        grown=reached|set(np.flatnonzero(adjacency[list(reached)].any(axis=0)))
        if grown==reached:break
        reached=grown
    assert len(reached)==shape[1]
    # All exchanges and biological bases are explicit and finite.
    assert len(config['model']['kinetics']['exchange_mappings'])==reference.n+1
    assert config['model']['kinetics']['flux_basis']['growth']['amount_per_million_cells']==1.


def test_coupled_reference_qp_against_closed_form():
    h=np.diag([2.,4.]);c=np.array([4.,4.])
    a=np.array([[-1.,0.],[0.,-1.],[1.,1.]])
    x,proof=quadratic_program(h,c,a,np.array([0.,0.,1.]))
    np.testing.assert_allclose(x,[2./3.,1./3.],atol=1e-12,rtol=0.)
    assert proof['objective_gap_bound']<1e-10
    with pytest.raises(ValueError,match='infeasible'):
        quadratic_program(h,c,np.array([[1.,0.],[-1.,0.]]),np.array([0.,-1.]))


@pytest.mark.parametrize('name',COUPLED_CASES)
def test_coupled_independent_reference_static_certificate(name):
    _,config,_=coupled_load(name);r=Reference(config);state=r.initial()
    for caps in (None,np.zeros(r.n),np.full(r.n,.01)):
        flux,proof=r.solve(state,caps)
        np.testing.assert_allclose(network_matrix(config)@flux,0.,atol=1e-10,rtol=0.)
        assert proof['objective_gap_bound']<2e-9*max(1.,proof['objective'])
        assert proof['stationarity']<1e-9
        if caps is not None:
            assert max(flux[r.selected[:r.n]]-caps)<1e-10
        if caps is not None and not caps.any():
            np.testing.assert_allclose(flux[r.selected],0.,atol=1e-10)
            assert proof['envelope']==1.


def test_coupled_reference_row_scaling_equivalence():
    references=[Reference(coupled_load(n)[1]) for n in ('coupled20','coupled20_scaled')]
    answers=[r.solve(r.initial()) for r in references]
    np.testing.assert_allclose(answers[0][0],answers[1][0],atol=1e-10,rtol=1e-9)
    assert answers[0][1]['objective']==pytest.approx(answers[1][1]['objective'],abs=1e-10)


@pytest.mark.parametrize('name',COUPLED_CASES)
def test_coupled_declared_kinetics_match_independent_formulas(name):
    case,config,thermo=coupled_load(name)
    assembly=build_bioreactor(case,config,thermo)
    reference=Reference(config)
    for integral in (0.,.00022,.0005):
        state=reference.initial();state[-1]=integral
        state[:reference.n]*=.2
        snapshot=CultureSnapshot(dict(zip(reference.species,state[:reference.n+1]/state[-2])),
            state[-2],state[reference.n+1],state[reference.n+2],state[reference.n+3],state[-1])
        rates=assembly.mechanism.rate_provider.evaluate(snapshot,assembly.mechanism.conditions).rates
        np.testing.assert_allclose([rates[r] for r in reference.exchanges],reference.targets(state),
                                   atol=1e-14,rtol=1e-12)
    assert assembly.unit.result is None


def test_coupled_formal_inventory_identity_and_sampling():
    _,config,_=coupled_load('coupled10');r=Reference(config);y=r.initial()
    supply=np.linspace(.001,.01,r.n+1)
    rates=r.rhs(y,supply,.05)
    # Formal mmol: liquid pools + one mmol/million live/dead cells + P/0.1 g/mmol.
    weights=np.r_[np.ones(r.n+3),10.,0.,0.]
    assert weights@rates==pytest.approx(sum(supply),abs=1e-10)
    before=weights@y;exposure=y[-1];fraction=1.-.02/y[-2]
    after=y.copy();after[:r.n+4]*=fraction;after[-2]-=.02
    assert weights@after==pytest.approx(before*fraction,abs=1e-12)
    assert after[-1]==exposure


def test_coupled_reference_constant_growth_against_analytic_solution():
    case,config,_=coupled_load('coupled04');config=deepcopy(config)
    config['recipes']['benchmark']=[]
    config['state']['species']['amounts']={s:10. for s in config['model']['kinetics']['extracellular_species']}
    case['operation']['runtime']['value']=.004
    r=Reference(config);y=r.initial();flux,_=r.solve(y)
    growth=flux[r.growth_index];death=config['parameters']['rates']['death'];net=growth-death
    times,states=r.trajectory(case)[0]
    exposure=np.expm1(net*times)/net
    expected=np.repeat(y[None,:],len(times),axis=0)
    expected[:,:r.n]-=exposure[:,None]*flux[r.selected[:r.n]]
    expected[:,r.n]+=exposure*flux[r.ids.index('w')]
    expected[:,r.n+1]=np.exp(net*times)
    expected[:,r.n+2]=death*exposure
    expected[:,r.n+3]=.1*flux[r.ids.index('p')]*exposure
    expected[:,-1]=exposure/1000.
    np.testing.assert_allclose(states,expected,atol=1e-11,rtol=1e-10)
    refined=r.trajectory(case,refinement=.5)[0][1]
    np.testing.assert_allclose(states,refined,atol=1e-12,rtol=1e-11)


def test_coupled_reference_finite_exposure_bound():
    _,config,_=coupled_load('coupled04');r=Reference(config);state=r.initial()
    flux,proof=r.fixed_step(state,np.zeros(r.n+1),.002)
    unconstrained,_=r.solve(state)
    np.testing.assert_allclose(flux,unconstrained,atol=1e-9,rtol=1e-8)
    assert proof['objective_gap_bound']<1e-9*max(1.,proof['objective'])


def test_coupled_reference_constrained_exposure_refinement():
    _,config,_=coupled_load('coupled04');r=Reference(config);state=r.initial()
    state[:r.n]=np.array([config['parameters']['rates'][name] for name in r.exchanges[:r.n]])*.0002
    answers=[r.fixed_step(state,np.zeros(r.n+1),.002,gap_tolerance=t) for t in (1e-9,1e-10)]
    for flux,proof in answers:
        exposure=.001*state[r.n+1]*(1.+np.exp((flux[r.growth_index]-.03)*.002))
        assert np.max(exposure*flux[r.selected[:r.n]]-state[:r.n])<1e-12
        assert proof['objective_gap_bound']<1e-9*max(1.,proof['objective'])
    np.testing.assert_allclose(answers[0][0][r.selected],answers[1][0][r.selected],atol=2e-7,rtol=0.)
def test_reference_comparison_preserves_event_sides_and_requires_coverage():
    reference=np.array([[0.,0.,1.],[0.,1.,2.],[1.,1.,7.],[1.,2.,8.]])
    rows=reference.copy()
    rows[2,2]+=.25
    assert compare(rows,reference,np.array([1.]),require_all=True)==.25
    with pytest.raises(ValueError,match='every saved time'):
        compare(np.array([[0.,.5,1.5]]),reference,np.array([1.]),require_all=True)
