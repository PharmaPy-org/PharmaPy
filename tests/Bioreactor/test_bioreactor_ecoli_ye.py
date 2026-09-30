"""Independent balances verify the configured native fed-batch demonstration."""

import json
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy.integrate import solve_ivp

ROOT = Path(__file__).parents[2]
EXAMPLE = ROOT / 'examples/bioreactors/ecoli_ye_fed_batch'


def independent_solution(case, config, times):
    """Source equations in concentrations, independent of native rule evaluation."""
    p = config['parameters']['coefficients']
    species = ['biomass', 'glucose', 'acetate', 'fraction_a', 'fraction_b']
    thermo = json.loads((EXAMPLE / 'inputs/thermo.json').read_text())
    volume = case['operation']['initial_volume']['value']
    y = np.array([config['state']['species']['amounts'][n] * thermo[n]['mw'] / volume
                  for n in species] + [volume])
    events = config['recipes'][case['recipe']['name']]
    end = case['operation']['runtime']['value']
    boundaries = sorted({0., end, *(e['time'] for e in events)})
    prediction = np.full((len(times), 6), np.nan)
    flow = 0.
    feed = np.zeros(5)
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        for e in (e for e in events if e['time'] == start):
            if e['event_type'] == 'flow':
                flow = e['volume_l_h']
                feed = np.array([e['concentrations_g_l'].get(n, 0.) for n in species])
            elif e['event_type'] == 'sample':
                y[-1] -= e['volume_l']
            else:
                added = e['volume_l']
                stock = np.array([e['concentrations_g_l'].get(n, 0.) for n in species])
                y[:5] = (y[:5]*y[-1] + stock*added)/(y[-1]+added)
                y[-1] += added

        def rhs(t, y):
            x, s, a, ya, yb = np.maximum(y[:5], 0.)
            qs = p['qS_max']*s/(s+p['K_S'])
            qa = p['qYEF_A_max']*ya/(ya+p['K_YEF_A'])
            qb = p['qYEF_B_max']*yb/(yb+p['K_YEF_B'])
            ox = qs/(1+qa/p['Ki_YEA_qSox']+qb/p['Ki_YEB_qSox']) * p['K_qSox']/(p['K_qSox']+qs)
            ac = p['qAc_max']*a/(a+p['K_A'])/(1+s/p['Ki_AS'])
            mu = ((ox-p['qm'])*p['Y_XSem']+qa*p['Y_XYEF_A']
                  +qb*p['Y_XYEF_B']+ac*p['Y_XA'])
            reaction = x*np.array([mu, -qs, (qs-ox)*p['Y_AS']-ac, -qa, -qb])
            return np.r_[reaction+flow/y[-1]*(feed-y[:5]), flow]

        solved = solve_ivp(rhs, (start, stop), y, method='Radau', rtol=1e-10,
                           atol=1e-12, max_step=.05, dense_output=True)
        assert solved.success
        mask = (times >= start) & (times < stop)
        if mask.any():
            prediction[mask] = solved.sol(times[mask]).T
        y = solved.y[:, -1]
    prediction[times == end] = y
    return prediction


@pytest.fixture(scope='module')
def native_runs(tmp_path_factory):
    output = tmp_path_factory.mktemp('ecoli_ye')
    namespace = {'__name__': '__notebook__'}
    notebook = json.loads((EXAMPLE / 'workflow.ipynb').read_text())
    for step, cell in enumerate((c for c in notebook['cells'] if c['cell_type'] == 'code'), 1):
        exec(compile(''.join(cell['source']), f'workflow:step-{step}', 'exec'), namespace)
        if step == 1:
            namespace['EXPORT_DIR'] = output
    yield namespace
    plt.close('all')


@pytest.mark.parametrize('run', ['F10', 'F11'])
def test_native_feed_inventory_and_independent_kinetics(native_runs, run):
    case, config = next((c, m) for c, m in native_runs['configurations'] if c['recipe']['name'] == run)
    rows = native_runs['tables'][run]
    times = np.array([r['time_h'] for r in rows])
    keys = ['biomass_g_l', 'glucose_g_l', 'acetate_g_l', 'fraction_a_g_l', 'fraction_b_g_l', 'volume_l']
    values = np.array([[r[k] for k in keys] for r in rows])
    assert np.isfinite(values).all() and values.min() >= -1e-10
    events = config['recipes'][run]
    # Ignore discontinuity sides in the continuous comparison; test those exactly below.
    smooth = np.ones(len(times), dtype=bool)
    for e in events:
        smooth &= ~np.isclose(times, e['time'], atol=1e-8, rtol=0.)
    expected = independent_solution(case, config, times[smooth])
    np.testing.assert_allclose(values[smooth], expected, atol=2e-5, rtol=2e-5)
    volume = case['operation']['initial_volume']['value']
    flow = 0.
    previous = 0.
    for e in events:
        volume += flow*(e['time']-previous)
        previous = e['time']
        if e['event_type'] == 'flow':
            flow = e['volume_l_h']
        else:
            volume += e['volume_l'] * (-1 if e['event_type'] == 'sample' else 1)
        hits = np.flatnonzero(np.isclose(times, e['time'], atol=1e-9, rtol=0.))
        if len(hits) == 2 and e['event_type'] in {'sample', 'feed'}:
            before, after = rows[hits[0]], rows[hits[1]]
            for n in ['biomass', 'glucose', 'acetate', 'fraction_a', 'fraction_b']:
                if e['event_type'] == 'sample':
                    np.testing.assert_allclose(after[n+'_g_l'], before[n+'_g_l'], atol=1e-10)
                else:
                    addition = e['volume_l']*e['concentrations_g_l'].get(n, 0.)
                    np.testing.assert_allclose(after[n+'_g']-before[n+'_g'], addition, atol=1e-10)
    volume += flow*(case['operation']['runtime']['value']-previous)
    assert rows[-1]['volume_l'] == pytest.approx(volume, abs=1e-11)
    assert (native_runs['EXPORT_DIR'] / run / 'trajectories.csv').exists()


def test_notebook_adapter_accepts_renamed_species_and_new_recipe(tmp_path):
    notebook = json.loads((EXAMPLE / 'workflow.ipynb').read_text())
    cells = [c for c in notebook['cells'] if c['cell_type'] == 'code']
    namespace = {}
    for cell in cells[:2]:
        exec(''.join(cell['source']), namespace)
    names = list(namespace['thermo'])
    renamed = {n: f'component_{i}' for i, n in enumerate(names)}

    def rename(value):
        if isinstance(value, dict):
            return {renamed.get(k, k): rename(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [rename(v) for v in value]
        return renamed.get(value, value) if isinstance(value, str) else value

    case, config = rename(namespace['configurations'][0])
    case['operation']['runtime']['value'] = .4
    config['recipes'][case['recipe']['name']] = [
        dict(time=.1, event_type='flow', volume_l_h=.1,
             concentrations_g_l={renamed['glucose']: 1.}, density_kg_l=1.),
        dict(time=.2, event_type='feed', volume_l=.1, concentrations_g_l={}, density_kg_l=1., volume_basis='solution'),
        dict(time=.3, event_type='sample', volume_l=.05),
    ]
    thermo = rename(namespace['thermo'])
    path = tmp_path / 'thermo.json'
    path.write_text(json.dumps(thermo))
    namespace.update(configurations=[(case, config)], thermo=thermo, THERMO=path)
    for cell in cells[2:4]:
        exec(''.join(cell['source']), namespace)
    phase, history = namespace['solutions'][case['recipe']['name']]
    assert set(phase.name_species) == set(renamed.values())
    assert history[-1].vessel_vol[-1] == pytest.approx(.00108, abs=1e-12)
    assert np.isfinite(history[-1].mass_j_liquid0).all()


def test_alternative_units_preserve_full_forward_trajectories(native_runs):
    notebook = json.loads((EXAMPLE / 'workflow.ipynb').read_text())
    cells = [c for c in notebook['cells'] if c['cell_type'] == 'code']
    namespace = {}
    for cell in cells[:2]:
        exec(''.join(cell['source']), namespace)
    configurations = deepcopy(namespace['configurations'])
    thermo = namespace['thermo']
    for case, config in configurations:
        op, numerics = case['operation'], case['numerics']
        op['initial_volume'] = dict(unit='mL', value=op['initial_volume']['value']*1000.)
        op['runtime'] = dict(unit='min', value=op['runtime']['value']*60.)
        numerics['step'] = dict(unit='s', value=numerics['step']['value']*3600.)
        numerics['maximum_step'] = dict(unit='min', value=numerics['maximum_step']['value']/60.)
        state = config['state']['species']
        state['amounts'] = {n: v*thermo[n]['mw'] for n, v in state['amounts'].items()}
        state['unit'] = 'g'
        for event in config['recipes'][case['recipe']['name']]:
            event['time'] = dict(unit='min', value=event['time']*60.)
            if 'volume_l_h' in event:
                event['volume_flow'] = dict(unit='mL/min', value=event.pop('volume_l_h')*1000./60.)
            if 'volume_l' in event:
                event['volume'] = dict(unit='mL', value=event.pop('volume_l')*1000.)
            if 'concentrations_g_l' in event:
                event['concentrations'] = dict(unit='mmol/L', values={
                    n: v*1000./thermo[n]['mw'] for n, v in event.pop('concentrations_g_l').items()})
            if 'density_kg_l' in event:
                event['density'] = dict(unit='kg/m3', value=event.pop('density_kg_l')*1000.)
        kinetics = config['model']['kinetics']
        for species, output in list(kinetics['species_mass_rates'].items()):
            name = output + '_converted'
            kinetics['rule_graph'].append(dict(identifier=name, expression={
                'op': 'multiply', 'args': [{'ref': output}, 1000./(60.*thermo[species]['mw'])]}))
            kinetics['species_mass_rates'][species] = name
        kinetics['rate_unit'] = 'mol/(m3 min)'
    namespace['configurations'] = configurations
    for cell in cells[2:5]:
        exec(''.join(cell['source']), namespace)
    for run, rows in namespace['tables'].items():
        reference = native_runs['tables'][run]
        # Compare both sides of every event as well as the continuous trajectories.
        np.testing.assert_allclose([list(r.values()) for r in rows],
                                   [list(r.values()) for r in reference], rtol=2e-6, atol=2e-6)


def test_builder_rejects_conflicting_quantity_declarations(native_runs):
    from PharmaPy.Bioreactors import build_bioreactor
    case, config = deepcopy(native_runs['configurations'][0])
    config['recipes'][case['recipe']['name']] = [
        dict(time=0., event_type='sample', volume_l=1., volume=dict(value=2., unit='L'))]
    with pytest.raises(ValueError, match='once'):
        build_bioreactor(case, config, str(EXAMPLE / 'inputs/thermo.json'))


@pytest.mark.parametrize('backend', ['scipy', 'fixed-step'])
@pytest.mark.parametrize('policy', ['working-volume', 'native'])
def test_configured_multiple_inlets_stop_and_sampling(backend, policy):
    from PharmaPy.Bioreactors import build_bioreactor
    case, config = [json.loads((EXAMPLE / 'inputs' / f).read_text())
                    for f in ('case.json', 'mechanism.json')]
    case['operation'].update(runtime=dict(value=1., unit='h'), volume_policy=policy)
    case['numerics']['backend'] = backend
    if backend == 'fixed-step':
        case['numerics'] = {k: v if k in {'backend', 'step'} else None
                            for k, v in case['numerics'].items()}
    for rule in config['model']['kinetics']['rule_graph']:
        rule['expression'] = 0.
    # Solvent-only inlets have the same native and declared density in this fixture.
    config['recipes'][case['recipe']['name']] = [
        dict(time=0., event_type='flow', inlet='first', volume_l_h=.1,
             concentrations_mmol_l={}, composition='nutrient-free', density_kg_l=1.),
        dict(time=0., event_type='flow', inlet='second', volume_l_h=.2,
             concentrations_mmol_l={}, composition='nutrient-free', density_kg_l=1.),
        dict(time=.5, event_type='flow', inlet='first', volume_l_h=0.),
        dict(time=.5, event_type='sample', volume_l=.115),
        dict(time=.75, event_type='flow', inlet='second', volume_l_h=0.),
    ]
    case['recipe']['require_complete_inputs'] = True
    assembly = build_bioreactor(case, config, str(EXAMPLE / 'inputs/thermo.json'))
    initial = assembly.phase.mass_j.copy()
    result = assembly.solve()
    carrier = list(assembly.phase.name_species).index(assembly.carrier_species)
    expected = initial.copy()
    expected[carrier] += .15
    expected *= .9
    expected[carrier] += .05
    np.testing.assert_allclose(result[-1].mass_j_liquid0[-1], expected, rtol=1e-9, atol=1e-12)
    assert result[-1].vessel_vol[-1] == pytest.approx(.001085, abs=1e-12)
    assert assembly.unit.inlet_connections == []
    assert assembly.diagnostics(result)['optimization'] is None


def test_batch_rejects_continuous_inlet():
    from PharmaPy.Bioreactors import build_bioreactor
    case, config = [json.loads((EXAMPLE / 'inputs' / f).read_text())
                    for f in ('case.json', 'mechanism.json')]
    case['operation']['mode'] = 'batch'
    config['recipes'][case['recipe']['name']] = [dict(time=0., event_type='flow', volume_l_h=.1)]
    with pytest.raises(ValueError, match='fed-batch'):
        build_bioreactor(case, config, str(EXAMPLE / 'inputs/thermo.json')).solve()


def test_configured_kinetics_accepts_shared_blank_template_sections():
    from PharmaPy.Bioreactors import build_bioreactor
    case, config = [json.loads((EXAMPLE / 'inputs' / f).read_text())
                    for f in ('case.json', 'mechanism.json')]
    template = json.loads((ROOT / 'doc/online_docs/bioreactors/templates/mechanism.json').read_text())
    template['model']['kinetics'].update(config['model']['kinetics'])
    config['model'] = template['model']
    original = deepcopy((case, config))
    assembly = build_bioreactor(case, config, str(EXAMPLE / 'inputs/thermo.json'))
    assert type(assembly.mechanism).__name__ == 'ConfiguredRates'
    assert (case, config) == original


def test_configured_batch_uses_the_same_builder_and_balances():
    from PharmaPy.Bioreactors import build_bioreactor
    case, config = [json.loads((EXAMPLE / 'inputs' / f).read_text())
                    for f in ('case.json', 'mechanism.json')]
    case['operation'].update(mode='batch', runtime=dict(value=2., unit='h'))
    config['recipes'][case['recipe']['name']] = []
    assembly = build_bioreactor(case, config, str(EXAMPLE / 'inputs/thermo.json'))
    result, = assembly.solve()
    expected = independent_solution(case, config, result.time/3600.)
    names = list(assembly.phase.name_species)
    indices = [names.index(n) for n in ['biomass', 'glucose', 'acetate', 'fraction_a', 'fraction_b']]
    concentrations = result.mass_j_liquid0[:, indices] / np.asarray(result.vessel_vol).reshape(-1, 1)
    np.testing.assert_allclose(concentrations, expected[:, :5], atol=1e-7, rtol=1e-6)
