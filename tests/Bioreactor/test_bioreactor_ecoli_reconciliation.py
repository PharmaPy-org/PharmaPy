"""Frozen experimental model through native biomass-based reconciliation."""

import json
import os
from hashlib import sha256
from copy import deepcopy
from pathlib import Path
from time import monotonic

import matplotlib.pyplot as plt
import numpy as np
import pytest

from PharmaPy.Bioreactors import build_bioreactor, BioreactorUnitConverter as Units
from PharmaPy.Bioreactors.culture import CultureSnapshot
from test_bioreactor_ecoli_ye import EXAMPLE, independent_solution, native_runs


FOLDER = EXAMPLE / 'reconciliation'
SPECIES = ['biomass', 'glucose', 'acetate', 'fraction_a', 'fraction_b']


@pytest.fixture(scope='module')
def reconciled_runs(tmp_path_factory):
    namespace = {}
    output = Path(os.environ.get('BIOREACTOR_RECONCILIATION_OUTPUT',
                                tmp_path_factory.mktemp('ecoli_reconciliation')))
    notebook = json.loads((FOLDER / 'workflow.ipynb').read_text())
    start = monotonic()
    for step, cell in enumerate((c for c in notebook['cells'] if c['cell_type'] == 'code'), 1):
        exec(compile(''.join(cell['source']), f'reconciliation:step-{step}', 'exec'), namespace)
        if step == 1:
            namespace['EXPORT_DIR'] = output
    namespace['elapsed_seconds'] = monotonic() - start
    yield namespace
    plt.close('all')


def test_frozen_forward_campaign(reconciled_runs, native_runs):
    report = {'scope': 'experimental reproduction through reconciliation; no refitting',
              'elapsed_seconds': reconciled_runs['elapsed_seconds'], 'runs': {}}
    report['input_sha256'] = {str(p.relative_to(EXAMPLE)): sha256(p.read_bytes()).hexdigest()
                             for p in [*sorted((FOLDER/'inputs').rglob('*.json')),
                                       EXAMPLE/'inputs/thermo.json',
                                       EXAMPLE/'reference/F10.json', EXAMPLE/'reference/F11.json']}
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    for row_index, run in enumerate(['F10', 'F11']):
        rows = reconciled_runs['tables'][run]
        direct = native_runs['tables'][run]
        case, config = next((c, m) for c, m in reconciled_runs['configurations']
                            if c['recipe']['name'] == run)
        source_case, source = next((c, m) for c, m in native_runs['configurations']
                                  if c['recipe']['name'] == run)
        assert config['parameters'] == source['parameters']
        assert config['state']['species'] == source['state']['species']
        assert config['recipes'] == source['recipes']
        assert case['numerics'] == source_case['numerics']
        assert case['operation'] == source_case['operation']
        keys = [n+'_g_l' for n in SPECIES] + ['volume_l']
        times = np.array([r['time_h'] for r in rows])
        actual = np.array([[r[k] for k in keys] for r in rows])
        expected = np.array([[r[k] for k in keys] for r in direct])
        np.testing.assert_allclose(times, [r['time_h'] for r in direct], atol=1e-10, rtol=0)
        np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)
        assert np.isfinite(actual).all() and actual.min() >= -1e-10
        smooth = np.ones(len(times), dtype=bool)
        for event in config['recipes'][run]:
            smooth &= ~np.isclose(times, event['time'], atol=1e-8, rtol=0)
        oracle = independent_solution(source_case, source, times[smooth])
        np.testing.assert_allclose(actual[smooth], oracle, atol=2e-5, rtol=2e-5)
        phase, histories = reconciled_runs['solutions'][run]
        biomass = phase.name_species.index('biomass')
        for history in histories:
            np.testing.assert_allclose(history.biomass_kg_liquid0.reshape(-1),
                                       history.mass_j_liquid0[:, biomass], rtol=2e-6, atol=1e-12)
        mechanism = reconciled_runs['assemblies'][run].mechanism
        maximum_target_error = maximum_balance_error = 0.
        thermo = reconciled_runs['thermo']
        # Accepted output states, including both event sides, rather than rejected solver trials.
        for row in rows:
            snapshot = CultureSnapshot(
                {n: row[n+'_g_l']*1000/thermo[n]['mw'] for n in SPECIES},
                row['volume_l'], 0., 0., 0., 0., row['biomass_g']/1000)
            rates = mechanism.rate_provider.evaluate(snapshot, mechanism.conditions).rates
            targets = {r: rates[o] for r, o in mechanism.definition.kinetic_outputs['exchange_fluxes'].items()}
            solved = mechanism._solve(targets)
            flux = dict(zip(solved.reaction_ids, solved.fluxes))
            maximum_target_error = max(maximum_target_error, max(
                abs(flux[r]-value)/max(1., abs(value)) for r, value in targets.items()))
            maximum_balance_error = max(maximum_balance_error, solved.mass_balance_residual_inf)
        assert maximum_target_error < 2e-6
        assert maximum_balance_error < 2e-7
        measurements = json.loads((EXAMPLE / 'reference' / (run+'.json')).read_text())['measurements']
        metrics = {}
        for col, name in enumerate(SPECIES[:3]):
            data = measurements[name]
            prediction = np.interp(data['time_h'], times, actual[:, col])
            baseline = np.interp(data['time_h'], times, expected[:, col])
            rmse = float(np.sqrt(np.mean((prediction-data['g_l'])**2)))
            baseline_rmse = float(np.sqrt(np.mean((baseline-data['g_l'])**2)))
            assert abs(rmse-baseline_rmse) < 2e-4
            metrics[name] = {'rmse_g_l': rmse, 'direct_rmse_g_l': baseline_rmse,
                             'measurements': len(data['g_l'])}
            ax = axes[row_index, col]
            ax.plot(times, expected[:, col], '--', label='Direct kinetics')
            ax.plot(times, actual[:, col], label='Reconciliation')
            ax.scatter(data['time_h'], data['g_l'], color='black', s=14, label='Experiment')
            ax.set(xlabel='Time (h)', ylabel=name+' (g/L)', title=run)
            ax.grid(alpha=.2)
        report['runs'][run] = dict(
            maximum_absolute_trajectory_difference=float(np.max(np.abs(actual-expected))),
            maximum_absolute_oracle_difference=float(np.max(np.abs(actual[smooth]-oracle))),
            maximum_scaled_target_difference=maximum_target_error,
            maximum_balance_residual=maximum_balance_error,
            final_volume_l=rows[-1]['volume_l'], experimental=metrics)
    axes[0, 0].legend()
    output = reconciled_runs['EXPORT_DIR']
    fig.savefig(output/'comparison.png', dpi=160)
    fig.savefig(output/'comparison.svg')
    report['passed'] = True
    (output/'assessment.json').write_text(json.dumps(report, indent=2)+'\n')


def test_biomass_basis_units_and_sampling(tmp_path):
    case = json.loads((FOLDER/'inputs/case.json').read_text())
    config = json.loads((FOLDER/'inputs/mechanism.json').read_text())
    case['operation']['runtime']['value'] = .3
    config['recipes']['F10'] = [dict(time=.1, event_type='sample', volume_l=.1),
                              dict(time=.2, event_type='feed', volume_l=.1,
                                   concentrations_g_l={}, volume_basis='solution')]
    a = build_bioreactor(case, config, str(EXAMPLE/'inputs/thermo.json'))
    reference = a.solve(verbose=False)
    alternative = deepcopy(config)
    k = alternative['model']['kinetics']
    # mmol/gDW/h -> mol/kgDW/h is numerically identical; population stays kgDW.
    k['flux_basis']['exchange'].update(amount_unit='mol', normalization='kgDW')
    k['flux_basis']['product']['normalization'] = 'kgDW'
    for rule, unit in alternative['rate_provider']['output_units'].items():
        alternative['rate_provider']['output_units'][rule] = unit.replace('mmol/(gDW h)', 'mol/(kgDW h)').replace('g/(gDW h)', 'g/(kgDW h)')
    # Both declaration bases convert to exactly the same molecular source.
    b = build_bioreactor(case, alternative, str(EXAMPLE/'inputs/thermo.json'))
    assert Units.from_inputs(case, config).exchange_scale == pytest.approx(24000)
    assert b.mechanism.unit_converter.exchange_scale == a.mechanism.unit_converter.exchange_scale
    result = b.solve(verbose=False)
    for actual, expected in zip(result, reference):
        np.testing.assert_allclose(actual.mass_j_liquid0, expected.mass_j_liquid0, rtol=2e-6, atol=1e-12)
        np.testing.assert_allclose(actual.biomass_kg_liquid0, expected.biomass_kg_liquid0, rtol=2e-6, atol=1e-12)
    assert result[-1].vessel_vol[-1] == pytest.approx(.001, abs=1e-12)
    assert result[1].biomass_kg_liquid0[0] == pytest.approx(result[0].biomass_kg_liquid0[-1]*.9)
    bad = deepcopy(config)
    bad['state']['viable_cells_million'] = 1.
    with pytest.raises(ValueError, match='inactive'):
        build_bioreactor(case, bad, str(EXAMPLE/'inputs/thermo.json'))
    bad = deepcopy(config)
    bad['model']['kinetics']['rule_graph'][0]['expression'] = {'state': 'viable_cells_million'}
    with pytest.raises(ValueError, match='cell-count rules'):
        build_bioreactor(case, bad, str(EXAMPLE/'inputs/thermo.json'))
    thermo = json.loads((EXAMPLE/'inputs/thermo.json').read_text())
    names = {name: f'species_{i}' for i, name in enumerate(thermo)}
    def rename(value):
        if isinstance(value, dict):
            return {names.get(k, k): rename(v) for k, v in value.items()}
        if isinstance(value, list):
            return [rename(v) for v in value]
        return names.get(value, value) if isinstance(value, str) else value
    path = tmp_path/'thermo.json'
    path.write_text(json.dumps(rename(thermo)))
    renamed = build_bioreactor(rename(case), rename(config), str(path)).solve(verbose=False)
    for actual, expected in zip(renamed, reference):
        np.testing.assert_allclose(actual.mass_j_liquid0, expected.mass_j_liquid0, rtol=2e-6, atol=1e-12)


@pytest.mark.parametrize('refinement', ['tolerance', 'step'])
def test_full_forward_refinement(reconciled_runs, refinement):
    case, config = deepcopy(reconciled_runs['configurations'][0])
    if refinement == 'tolerance':
        case['numerics']['relative_tolerance'] /= 10
        case['numerics']['absolute_tolerance'] /= 10
    else:
        case['numerics']['maximum_step']['value'] /= 2
    assembly = build_bioreactor(case, config, str(EXAMPLE/'inputs/thermo.json'))
    actual = assembly.solve(verbose=False)
    reference = reconciled_runs['solutions']['F10'][1]
    largest = 0.
    for a, b in zip(actual, reference):
        concentration = a.mass_j_liquid0 / a.vessel_vol.reshape(-1, 1)
        baseline = b.mass_j_liquid0 / b.vessel_vol.reshape(-1, 1)
        # Use the original trajectory gate; compare refined accuracy to the
        # independent solution, since the original run is not an exact oracle.
        np.testing.assert_allclose(concentration, baseline, atol=2e-5, rtol=2e-5)
        largest = max(largest, float(np.max(abs(concentration-baseline))))
    times = np.concatenate([h.time for h in actual])/3600.
    concentrations = np.concatenate([h.mass_j_liquid0/h.vessel_vol.reshape(-1, 1) for h in actual])
    columns = [assembly.phase.name_species.index(n) for n in SPECIES]
    mask = np.ones(len(times), dtype=bool)
    for event in config['recipes']['F10']:
        mask &= ~np.isclose(times, event['time'], atol=1e-8, rtol=0)
    source = json.loads((EXAMPLE/'inputs/mechanism.json').read_text())
    oracle = independent_solution(case, source, times[mask])[:, :5]
    error = float(np.max(abs(concentrations[mask][:, columns]-oracle)))
    # The tighter run must establish accuracy below one fifth of the original
    # absolute trajectory allowance, not merely agree with its predecessor.
    assert error < (2e-6 if refinement == 'tolerance' else 2e-5)
    output = reconciled_runs['EXPORT_DIR']/'assessment.json'
    if output.exists():
        report = json.loads(output.read_text())
        report.setdefault('refinements', {})[refinement] = dict(
            maximum_concentration_change_g_l=largest, maximum_oracle_error_g_l=error)
        output.write_text(json.dumps(report, indent=2)+'\n')


def test_compatible_targets_avoid_optimization_but_conflicts_do_not(monkeypatch):
    from PharmaPy.Metabolic.closures import reconciled
    from PharmaPy.Metabolic.closures.base import MetabolicEnvironment
    from PharmaPy.Metabolic.network import MetabolicNetworkDefinition, ReactionDefinition
    network = MetabolicNetworkDefinition('transfer', '1', ['pool'], [
        ReactionDefinition('input', {'pool': 1.}, 0., 2.),
        ReactionDefinition('output', {'pool': -1.}, 0., 2.)], {'output': 1.})
    closure = reconciled.RateReconciledMFAClosure(network, [])
    original = reconciled.minimize
    calls = []
    def count(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(reconciled, 'minimize', count)
    target = reconciled.ReconciliationTarget
    solved = closure.solve(MetabolicEnvironment({}), targets=[target('input', 1., 1.)], internal_scale=1.)
    np.testing.assert_array_equal(solved.fluxes, [1., 1.])
    assert not calls
    solved = closure.solve(MetabolicEnvironment({}), targets=[
        target('input', .5, 1.), target('output', 1.5, 1.)], internal_scale=1.)
    np.testing.assert_allclose(solved.fluxes, [1., 1.], atol=1e-9)
    assert calls
