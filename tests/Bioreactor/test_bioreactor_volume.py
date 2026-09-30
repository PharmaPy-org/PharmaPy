"""Volume hooks preserve inventory balances, feed dilution, and native replay."""

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from PharmaPy.Bioreactors.construction import build_bioreactor, _apply_material_event

EXAMPLES = Path(__file__).parents[2] / 'examples/bioreactors'


def build(name='generic_batch', policy='working-volume', mode=None, backend=None):
    p = (Path(__file__).parent / 'fixtures' if name == 'fed_batch_cho' else EXAMPLES) / name / 'inputs'
    case, definition = [json.loads((p / f).read_text()) for f in ('case.json', 'mechanism.json')]
    case['operation']['volume_policy'] = policy
    if mode:
        case['operation']['mode'] = mode
    if backend:
        case['numerics'].update(backend=backend, relative_tolerance=None,
                                absolute_tolerance=None, maximum_step=None,
                                jacobian_relative_step=None, jacobian_state_scale=None)
        if backend == 'scipy':
            case['numerics'].update(relative_tolerance=1e-8, absolute_tolerance=1e-11)
    return build_bioreactor(case, definition, p / 'thermo.json')


@pytest.mark.parametrize('name', ['generic_batch', 'generic_fed_batch', 'batch_ecoli_dfba', 'fed_batch_cho'])
@pytest.mark.parametrize('mode', ['batch', 'fed-batch'])
@pytest.mark.parametrize('backend', ['scipy', 'fixed-step'])
def test_volume_replay_reset_and_copy(name, mode, backend):
    a = build(name, mode=mode, backend=backend)
    initial_volume = a.phase.vol
    initial_mass = a.phase.mass_j.copy()
    a.events = ()
    a.time_grid = np.array([0., 1., 2.])
    result, = a.solve(verbose=False)
    np.testing.assert_allclose(result.vessel_vol, initial_volume, rtol=0, atol=1e-15)
    np.testing.assert_allclose(result.working_volume_liquid0, initial_volume, rtol=0, atol=1e-15)
    np.testing.assert_allclose(a.phase.mass_conc, a.phase.mass_j / initial_volume)
    np.testing.assert_allclose(a.phase.mole_conc, a.phase.mass_j / a.phase.mw / initial_volume)
    copied = deepcopy(a.phase)
    copied.vol = initial_volume * 2.
    assert a.phase.vol == initial_volume
    np.testing.assert_allclose(copied.mass_conc, a.phase.mass_conc / 2.)
    a.unit.reset()
    np.testing.assert_allclose(a.phase.mass_j, initial_mass)
    assert a.phase.vol == initial_volume


@pytest.mark.parametrize('basis, jump', [('carrier', .1005), ('solution', .1)])
def test_feed_mass_volume_and_observation_sides(basis, jump):
    a = build('generic_fed_batch')
    a.events[0]['volume_basis'] = basis
    before, after = a.solve(verbose=False)
    np.testing.assert_allclose(before.vessel_vol, .001)
    np.testing.assert_allclose(after.vessel_vol, .001 + jump / 1000.)
    np.testing.assert_allclose(after.mass_j_liquid0[0].sum() - before.mass_j_liquid0[-1].sum(), jump)
    np.testing.assert_allclose(after.biomass_kg_liquid0[0], before.biomass_kg_liquid0[-1])
    pre = before.biomass_kg_liquid0[-1] / before.vessel_vol[-1]
    post = after.biomass_kg_liquid0[0] / after.vessel_vol[0]
    np.testing.assert_allclose(post / pre, 1. / (1. + jump))
    # The idealized stoichiometry conserves liquid plus separately tracked biomass.
    for result, addition in [(before, 0.), (after, jump)]:
        np.testing.assert_allclose(result.mass_j_liquid0.sum(axis=1)
                                   + np.asarray(result.biomass_kg_liquid0).reshape(-1),
                                   1.0001 + addition, atol=1e-10)


def test_stock_target_and_solute_only_addition():
    a = build()
    # Change liquid mass without changing working volume, exposing mass/density shortcuts.
    mass = a.phase.mass_j.copy(); mass[-1] *= .9
    a.phase.updatePhase(mass_j=mass)
    event = dict(time=0., event_type='stock_to_target', target_species='nutrient',
                 target_concentration_mmol_l=20., concentrations_mmol_l={'nutrient': 100.})
    _apply_material_event(a.phase, event, 1., 'water')
    assert a.phase.vol == pytest.approx(.001125)
    assert a.phase.mole_conc[0] * 1000. == pytest.approx(20.)
    mass = a.phase.mass_j.copy()
    _apply_material_event(a.phase, event, 1., 'water')
    np.testing.assert_allclose(a.phase.mass_j, mass, atol=1e-14)
    event = dict(time=0., event_type='target_concentration', target_species='nutrient',
                 target_concentration_mmol_l=30.)
    _apply_material_event(a.phase, event, 1., 'water')
    assert a.phase.vol == pytest.approx(.001125)
    assert a.phase.mole_conc[0] * 1000. == pytest.approx(30.)


def test_property_setters_and_rejected_inputs_are_consistent():
    a = build()
    mass = a.phase.mass_j.copy()
    a.phase.vol *= 2.
    np.testing.assert_array_equal(a.phase.mass_j, mass)
    conc = a.phase.mole_conc.copy()
    a.phase.updatePhase(mole_conc=conc)
    np.testing.assert_allclose(a.phase.mass_j, mass)
    a.phase.updatePhase(mass_conc=mass / a.phase.vol)
    np.testing.assert_allclose(a.phase.mass_j, mass)
    for volume in (0., -1., float('nan')):
        with pytest.raises(ValueError, match='working volume'):
            a.phase.vol = volume
    with pytest.raises(ValueError, match='every phase species'):
        a.phase.mass_conc = [1.]
    with pytest.raises(ValueError, match='volume_policy'):
        build(policy='constant-ish')
    event = dict(time=0., event_type='feed', volume_l=.1, volume_basis='solution',
                 concentrations_mmol_l={'nutrient': 1e9})
    with pytest.raises(ValueError, match='solute mass'):
        _apply_material_event(a.phase, event, 1., 'water')
    np.testing.assert_allclose(a.phase.mass_j, mass)


def test_working_volume_rejects_unconnected_flow_physics():
    from types import SimpleNamespace
    from PharmaPy.Bioreactors.mechanisms import WorkingVolume
    a = build()
    volume = a.phase.get_mechanism(WorkingVolume)
    unit = SimpleNamespace(inlet_connections=[object()], outlet_connections=[], phase_connections=[])
    with pytest.raises(ValueError, match='declared inlet flows'):
        volume.update_state({}, unit=unit)


@pytest.mark.parametrize('name', ['generic_batch', 'fed_batch_cho'])
@pytest.mark.parametrize('policy', ['native', 'working-volume'])
def test_sampling_preserves_concentrations_and_removes_extensive_biology(name, policy):
    a = build(name, policy=policy)
    mass, volume = a.phase.mass_j.copy(), a.phase.vol
    states = {key: np.array(getattr(a.mechanism, key), copy=True)
              for key in a.mechanism.extensive_states}
    integral = getattr(a.mechanism, 'ivcd_million_cell_day_per_ml', None)
    _apply_material_event(a.phase, dict(time=0., event_type='sample', volume_l=volume*100.),
                          a.carrier_density_kg_l, a.carrier_species)
    np.testing.assert_allclose(a.phase.mass_j, mass*.9, rtol=1e-14)
    assert a.phase.vol == pytest.approx(volume*.9, rel=1e-14)
    np.testing.assert_allclose(a.phase.mass_conc, mass/volume, rtol=1e-14)
    for key, value in states.items():
        np.testing.assert_allclose(getattr(a.mechanism, key), value*.9, rtol=1e-14)
    assert getattr(a.mechanism, 'ivcd_million_cell_day_per_ml', None) == integral
    a.events = ()
    a.time_grid = np.array([0., .001])
    result, = a.solve()
    for key, value in states.items():
        np.testing.assert_allclose(getattr(result, key + '_liquid0')[0], value*.9, rtol=1e-14)


@pytest.mark.parametrize('name', ['generic_fed_batch', 'fed_batch_cho'])
@pytest.mark.parametrize('backend', ['scipy', 'fixed-step'])
def test_metabolic_models_share_configured_flow_and_sampling(name, backend):
    p = (Path(__file__).parent / 'fixtures' if name == 'fed_batch_cho' else EXAMPLES) / name / 'inputs'
    case, definition = [json.loads((p / f).read_text()) for f in ('case.json', 'mechanism.json')]
    case['operation'].update(mode='fed-batch', volume_policy='working-volume',
                             runtime=dict(value=2., unit='s'))
    case['numerics']['step'] = dict(value=.25, unit='s')
    case['numerics']['backend'] = backend
    if backend == 'scipy':
        case['numerics'].update(relative_tolerance=1e-8, absolute_tolerance=1e-12)
    if backend == 'fixed-step':
        case['numerics'] = {key: value if key in {'backend', 'step'} else None
                            for key, value in case['numerics'].items()}
    definition['recipes'][case['recipe']['name']] = [
        dict(time=0., event_type='flow', volume_flow=dict(value=1., unit='mL/s'),
             concentrations=dict(unit='g/L', values={}), composition='nutrient-free'),
        dict(time=1., event_type='sample', volume=dict(value=1., unit='mL')),
    ]
    a = build_bioreactor(case, definition, p / 'thermo.json')
    initial_volume = a.phase.vol
    before, after = a.solve()
    fraction = 1. - 1e-6 / before.vessel_vol[-1]
    np.testing.assert_allclose(after.mass_j_liquid0[0], before.mass_j_liquid0[-1]*fraction,
                               rtol=1e-12, atol=1e-14)
    for key in a.mechanism.extensive_states:
        np.testing.assert_allclose(getattr(after, key+'_liquid0')[0],
                                   getattr(before, key+'_liquid0')[-1]*fraction, rtol=1e-12)
    assert after.vessel_vol[-1] == pytest.approx(initial_volume + 1e-6, abs=1e-12)


@pytest.mark.parametrize('name', ['generic_batch', 'generic_fed_batch'])
def test_generic_notebook_with_working_volume(name, monkeypatch):
    import matplotlib.pyplot as plt
    root = EXAMPLES.parents[1]
    monkeypatch.chdir(root)
    notebook = json.loads((EXAMPLES / name / 'workflow.ipynb').read_text())
    namespace = {'__name__': '__notebook__'}
    try:
        for cell in notebook['cells']:
            if cell['cell_type'] != 'code':
                continue
            source = ''.join(cell['source'])
            exec(compile(source, name, 'exec'), namespace)
            if source.startswith('# STEP 1:'):
                namespace['WRITE_ARTIFACTS'] = False
            if source.startswith('# STEP 2:'):
                namespace['case']['operation']['volume_policy'] = 'working-volume'
                if name == 'generic_fed_batch':
                    namespace['config']['recipes']['fed_batch'][0]['volume_basis'] = 'solution'
        assert all(np.isfinite(list(row.values())).all() for row in namespace['rows'])
        assert namespace['rows'][-1]['liquid_volume_l'] == pytest.approx(
            1. if name == 'generic_batch' else 1.1)
    finally:
        plt.close('all')


@pytest.mark.parametrize('event_type', ['feed', 'stock_to_target'])
def test_user_supplied_solution_density(event_type):
    a = build()
    event = dict(time=0., event_type=event_type, density_kg_l=1.2,
                 concentrations_mmol_l={'nutrient': 100.})
    if event_type == 'feed':
        event.update(volume_l=.125, volume_basis='solution')
    else:
        event.update(target_species='nutrient', target_concentration_mmol_l=20.)
    mass = a.phase.mass_j.copy()
    _apply_material_event(a.phase, event, 1., 'water')
    assert a.phase.vol == pytest.approx(.001125)
    assert a.phase.mass_j.sum() - mass.sum() == pytest.approx(.125 * 1.2)
    for density in (0., -1., float('nan')):
        event['density_kg_l'] = density
        with pytest.raises(ValueError, match='density_kg_l'):
            _apply_material_event(a.phase, event, 1., 'water')


def test_native_stock_rejects_inconsistent_dilution_before_mutation():
    a = build(policy='native')
    mass = a.phase.mass_j.copy()
    event = dict(time=0., event_type='stock_to_target', density_kg_l=1.2,
                 concentrations_mmol_l={'nutrient': 100.}, target_species='nutrient',
                 target_concentration_mmol_l=20.)
    with pytest.raises(ValueError, match='native additive volume'):
        _apply_material_event(a.phase, event, 1., 'water')
    np.testing.assert_array_equal(a.phase.mass_j, mass)

