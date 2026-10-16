"""Native chemostat balances, with analytical washout and Monod equilibria.

Synthetic 1 L culture: mu_max=0.4/h, Ks=1 mmol/L, yield=0.1 kgDW/mol,
sterile feed=10 mmol/L. No maintenance, death, gas transfer or retention.
Core uses SciPy; generated thermo inputs and records stay in tmp_path.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from PharmaPy.Bioreactors import build_bioreactor
from PharmaPy.Reactors_Refactored import ContinuousReactor
from .test_bioreactor_mechanism_interchange import _configured, _trajectory

pytestmark = pytest.mark.integration

ROOT = Path(__file__).parents[2]
INPUTS = ROOT / 'examples/bioreactors/generic_batch/inputs'


def continuous_inputs(tmp_path, dilution=0.1, closure='pathway-lp', growth=True):
    """Construct matched synthetic continuous cases through existing input forms.

    Parameters
    ----------
    tmp_path : Path
        Destination of temporary configured-route thermo inputs.
    dilution : float
        Feed dilution rate [1/h] for a 1 L working volume.
    closure : str
        Pathway or configured kinetic implementation.
    growth : bool
        False selects zero growth for an analytical dilution check.

    Returns
    -------
    tuple
        Case, mechanism and thermo path, in their declared input units.
    """
    case, model, thermo = [json.loads((INPUTS / f'{name}.json').read_text())
                           for name in ('case', 'mechanism', 'thermo')]
    case['operation'].update(mode='continuous', volume_policy='working-volume')
    case['operation']['runtime']['value'] = 80.  # [h]; eight residence times at D=0.1/h
    case['numerics']['step']['value'] = 0.5  # [h]; reporting only
    model['parameters']['rates']['uptake']['nutrient'] = 4. if growth else 0.  # [mol/(kgDW h)]
    model['recipes']['batch'] = [{'event_type': 'flow', 'time': 0., 'volume_l_h': dilution,
                                 'concentrations_mmol_l': {'nutrient': 10.}, 'density_kg_l': 1.}]
    if closure == 'configured-kinetics':
        model, path = _configured(case, model, thermo, 1000., tmp_path)
        if growth:
            for rule in model['model']['kinetics']['rule_graph'][1:]:
                rule['expression'] = {'op': 'multiply', 'args': [rule['expression'],
                    {'op': 'divide', 'args': [{'concentration': 'nutrient'},
                        {'op': 'add', 'args': [{'constant': 'half_saturation'},
                                             {'concentration': 'nutrient'}]}]}]}
            model['model']['kinetics']['constants']['half_saturation'] = 1.  # [mmol/L]
    else:
        path = INPUTS / 'thermo.json'
        if growth:
            model['parameters']['affinities']['nutrient'] = 1.  # [mol/m3] = [mmol/L]
            model['model']['pathways']['uptake_constraints'][0].update(
                type='monod', half_saturation_parameter='affinities.nutrient')
    return case, model, path


@pytest.mark.parametrize('closure', ['pathway-lp', 'configured-kinetics'])
def test_continuous_sterile_feed_washes_out_biomass(tmp_path, closure):
    """Each representation obeys X=X0 exp(-Dt), with fixed liquid volume.

    Parameters
    ----------
    tmp_path : Path
        Isolated thermo destination.
    closure : str
        Biological state representation under test.
    """
    case, model, path = continuous_inputs(tmp_path, closure=closure, growth=False)
    model['state']['species']['amounts']['nutrient'] = 0.002  # [mol]; asymmetric inlet/charge
    assembly = build_bioreactor(case, model, path)
    assert isinstance(assembly.unit, ContinuousReactor)
    histories = assembly.solve()
    time, nutrient, biomass = _trajectory(histories, assembly.phase, closure=='configured-kinetics')
    # [kgDW], [mol]; independent constant-volume dilution solutions.
    np.testing.assert_allclose(biomass, 1e-4*np.exp(-0.1*time), rtol=3e-5, atol=1e-12)
    np.testing.assert_allclose(nutrient, 0.01+(0.002-0.01)*np.exp(-0.1*time), rtol=3e-6, atol=1e-10)
    for history in histories:
        np.testing.assert_allclose(history.vessel_vol, 0.001, rtol=1e-9, atol=1e-14)
    assembly.reset()
    repeated = assembly.solve()
    np.testing.assert_allclose(repeated[-1].mass_j_liquid0, histories[-1].mass_j_liquid0, rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize('closure', ['pathway-lp', 'configured-kinetics'])
def test_monod_continuous_equilibrium_and_mass_yield(tmp_path, closure):
    """Both kinetic routes reach the independently derived positive equilibrium.

    Parameters
    ----------
    tmp_path : Path
        Temporary inputs.
    closure : str
        Kinetic closure to qualify.
    """
    folder = ROOT/'examples/bioreactors/generic_chemostat/inputs'
    if closure == 'configured-kinetics':
        folder = folder/'kinetics'
    case, model = [json.loads((folder/f'{name}.json').read_text()) for name in ('case', 'mechanism')]
    assembly = build_bioreactor(case, model, folder/'thermo.json')
    time, nutrient, biomass = _trajectory(assembly.solve(), assembly.phase, closure=='configured-kinetics')
    # S*=Ks D/(mu_max-D)=1/3 mmol/L; X*=Y(Sin-S*)=0.9667 g/L.
    assert nutrient[-1] == pytest.approx(1e-3/3., rel=2e-3)
    assert biomass[-1] == pytest.approx(0.1*(0.01-1e-3/3.), rel=2e-3)
    # The equivalent nutrient inventory follows dilution even during growth.
    total = nutrient+biomass/0.1  # [mol]; nutrient plus biomass/yield
    np.testing.assert_allclose(total, 0.01+(0.011-0.01)*np.exp(-0.1*time), rtol=3e-6, atol=1e-10)


@pytest.mark.parametrize('closure', ['pathway-lp', 'configured-kinetics'])
def test_monod_washout_above_growth_threshold(tmp_path, closure):
    """D=0.6/h exceeds even mu_max=0.4/h; nutrient tends to inlet strength.

    Parameters
    ----------
    tmp_path : Path
        Temporary inputs.
    closure : str
        Biological implementation.
    """
    case, model, path = continuous_inputs(tmp_path, dilution=0.6, closure=closure)
    assembly = build_bioreactor(case, model, path)
    time, nutrient, biomass = _trajectory(assembly.solve(), assembly.phase, closure=='configured-kinetics')
    assert biomass[-1] < 1e-10  # [kgDW]; six orders below initial inoculum
    assert nutrient[-1] == pytest.approx(0.01, abs=2e-9)  # [mol]
    assert np.all(biomass >= 0.)
    assert np.all(np.diff(biomass) <= 1e-12)  # [kgDW]; solver absolute tolerance


def test_pathway_sterile_initial_state_has_no_spontaneous_growth(tmp_path):
    """Zero initial cells remain zero despite nutrient feed.

    Parameters
    ----------
    tmp_path : Path
        Temporary inputs.
    """
    case, model, path = continuous_inputs(tmp_path)
    model['state']['biomass_kg'] = 0.  # [kgDW]
    case['operation']['runtime']['value'] = 1.  # [h]
    assembly = build_bioreactor(case, model, path)
    history = assembly.solve()[0]
    np.testing.assert_array_equal(history.biomass_kg_liquid0, 0.)
    assert assembly.mechanism.last_solution is None


def test_fixed_step_reserves_substrate_for_outlet_before_pathway_uptake(tmp_path):
    """LP inventory includes the same withdrawal that appears in the balance.

    Parameters
    ----------
    tmp_path : Path
        Temporary inputs.
    """
    case, model, path = continuous_inputs(tmp_path, growth=False)
    case['operation']['runtime']['value'] = 1.  # [h]; one Euler step
    case['numerics'] = dict.fromkeys(case['numerics'])
    case['numerics'].update(backend='fixed-step', step={'unit': 'h', 'value': 1.})
    model['state']['species']['amounts']['nutrient'] = 1e-4  # [mol]
    model['parameters']['rates']['uptake']['nutrient'] = 100.  # [mol/(kgDW h)]; availability binds
    assembly = build_bioreactor(case, model, path)
    history = assembly.solve()[0]
    nutrient_index = assembly.phase.name_species.index('nutrient')
    np.testing.assert_allclose(history.vessel_vol, .001, atol=1e-14, rtol=1e-9)
    # Initially .0001 mol + .001 mol fed - .00001 mol withdrawn = .00109 mol for growth.
    assert history.mass_j_liquid0[-1, nutrient_index] == pytest.approx(0., abs=1e-12)
    assert np.asarray(history.biomass_kg_liquid0[-1]).item() == pytest.approx(
        1e-4 + .1*.00109 - .1*1e-4, abs=1e-12)


def test_flow_changes_stop_and_multiple_inlets_follow_native_volume_balance(tmp_path):
    """Scheduled flows change D; stopping both inlets stops withdrawal.

    Parameters
    ----------
    tmp_path : Path
        Temporary inputs.
    """
    case, model, path = continuous_inputs(tmp_path, growth=False)
    case['operation']['runtime']['value'] = 6.  # [h]
    events = model['recipes']['batch']
    events[0]['inlet'] = 'a'
    events.extend([dict(events[0], time=1., inlet='b', volume_l_h=.2),
                   {'event_type': 'flow', 'time': 3., 'inlet': 'a', 'volume_l_h': 0.},
                   {'event_type': 'flow', 'time': 3., 'inlet': 'b', 'volume_l_h': 0.}])
    assembly = build_bioreactor(case, model, path)
    histories = assembly.solve()
    time, _, biomass = _trajectory(histories, assembly.phase, False)
    exposure = .1*np.minimum(time,1.)+.3*np.clip(time-1.,0.,2.)  # [-]; integral D dt
    np.testing.assert_allclose(biomass, 1e-4*np.exp(-exposure), rtol=3e-6, atol=1e-12)
    for history in histories:
        np.testing.assert_allclose(history.vessel_vol, .001, rtol=1e-9, atol=1e-14)


def test_native_rhs_callback_does_not_alias_previous_derivative(tmp_path):
    """SciPy must retain its previous derivative when the vessel reuses buffers.

    Parameters
    ----------
    tmp_path : Path
        Temporary inputs.
    """
    from PharmaPy.IntegratorBackends import _inventory_rates
    case, model, path = continuous_inputs(tmp_path, growth=False)
    assembly = build_bioreactor(case, model, path)
    assembly._set_inlet(assembly.events[0])
    unit = assembly.unit
    unit.compile_structure()
    states = unit.create_solver_init_states()
    settings = dict(constrained=np.array([],dtype=int), boundary_tolerance=np.zeros_like(states),
                    unit=unit, rate_tolerance=np.zeros_like(states))
    first = _inventory_rates(0., states, **settings)
    saved = first.copy()
    changed = states.copy()
    changed[2] *= .5  # [kgDW]; change only separate biomass inventory
    second = _inventory_rates(0., changed, **settings)
    assert not np.shares_memory(first, second)
    np.testing.assert_array_equal(first, saved)
    assert second[2] != first[2]




@pytest.mark.assimulo
def test_continuous_native_assimulo_preserves_dilution(tmp_path):
    """Exercise the existing optional native backend on the same public model.

    Parameters
    ----------
    tmp_path : Path
        Temporary inputs.
    """
    pytest.importorskip('assimulo.solvers.sundials')
    from PharmaPy.IntegratorBackends import AssimuloBackend
    case,model,path=continuous_inputs(tmp_path,growth=False)
    case['operation']['runtime']['value']=2.  # [h]
    a=build_bioreactor(case,model,path)
    a.unit.integrator=AssimuloBackend({'rtol':1e-9,'atol':1e-12})
    history=a.solve()[0]
    np.testing.assert_allclose(np.asarray(history.biomass_kg_liquid0).reshape(-1),
                               1e-4*np.exp(-.1*history.time/3600.),rtol=3e-6,atol=1e-12)


def test_native_volume_continuous_mode_keeps_native_density_policy(tmp_path):
    """An inert constant-density case also works without WorkingVolume hooks.

    Parameters
    ----------
    tmp_path : Path
        Temporary inputs.
    """
    case,model,path=continuous_inputs(tmp_path,growth=False)
    case['operation'].update(volume_policy='native',runtime={'unit':'h','value':2.})
    a=build_bioreactor(case,model,path)
    history=a.solve()[0]
    np.testing.assert_allclose(history.vessel_vol,.001,rtol=1e-9,atol=1e-14)
    np.testing.assert_allclose(np.asarray(history.biomass_kg_liquid0).reshape(-1),
                               1e-4*np.exp(-.1*history.time/3600.),rtol=3e-6,atol=1e-12)


@pytest.mark.parametrize('density', [0.8, 1.2])
def test_declared_feed_density_preserves_volume_and_mass_balance(tmp_path, density):
    """Different solution densities preserve Q and obey dM/dt=Q rho_in-D M.

    Parameters
    ----------
    tmp_path : Path
        Temporary inputs.
    density : float
        Synthetic feed density [kg/L], different from the initial 1 kg/L.
    """
    case, model, path = continuous_inputs(tmp_path, growth=False)
    case['operation']['runtime']['value'] = 2.  # [h]
    model['recipes']['batch'][0]['density_kg_l'] = density
    a = build_bioreactor(case, model, path)
    history = a.solve()[0]
    expected_mass = density+(1.-density)*np.exp(-.1*history.time/3600.)  # [kg]
    np.testing.assert_allclose(history.mass_j_liquid0.sum(axis=1), expected_mass,
                               rtol=1e-7, atol=1e-10)
    np.testing.assert_allclose(history.vessel_vol, .001, rtol=1e-9, atol=1e-14)


def test_continuous_tolerance_and_maximum_step_refinement(tmp_path):
    """Monod transient and equilibrium converge under tighter integration settings.

    Parameters
    ----------
    tmp_path : Path
        Temporary inputs.
    """
    case, model, path = continuous_inputs(tmp_path)
    baseline = build_bioreactor(case, model, path)
    _, nutrient, biomass = _trajectory(baseline.solve(), baseline.phase, False)
    case['numerics']['relative_tolerance'] = 1e-10  # [-]; tenfold refinement
    case['numerics']['absolute_tolerance'] = 1e-13  # [native solver units]
    case['numerics']['maximum_step']['value'] = 150.  # [s]; half the original maximum
    refined = build_bioreactor(case, model, path)
    _, other_nutrient, other_biomass = _trajectory(refined.solve(), refined.phase, False)
    np.testing.assert_allclose(other_nutrient, nutrient, rtol=2e-6, atol=1e-10)
    np.testing.assert_allclose(other_biomass, biomass, rtol=2e-6, atol=1e-12)


def test_fixed_step_overdraw_reports_infeasibility(tmp_path):
    """A step exceeding the available biomass fails rather than clipping cells.

    Parameters
    ----------
    tmp_path : Path
        Temporary inputs.
    """
    case, model, path = continuous_inputs(tmp_path, dilution=2., growth=False)
    case['operation']['runtime']['value'] = 1.  # [h]; D dt=2 exceeds the available cells
    case['numerics'] = dict.fromkeys(case['numerics'])
    case['numerics'].update(backend='fixed-step', step={'unit': 'h', 'value': 1.})
    a = build_bioreactor(case, model, path)
    with pytest.raises(RuntimeError, match='Inventory depletion cannot be corrected'):
        a.solve()


@pytest.mark.parametrize('closure', ['pathway-lp', 'configured-kinetics'])
def test_washout_threshold_uses_growth_at_feed_concentration(tmp_path, closure):
    """Growth declines at D=0.38/h, below mu_max but above mu(Sin).

    Parameters
    ----------
    tmp_path : Path
        Temporary inputs.
    closure : str
        Biological implementation.
    """
    case, model, path = continuous_inputs(tmp_path, dilution=.38, closure=closure)
    a = build_bioreactor(case, model, path)
    a._set_inlet(a.events[0])
    a.unit.compile_structure()
    states = a.unit.create_solver_init_states()
    rates = a.unit.unit_model(0., states, limiter_dt=0.).copy()
    if closure == 'pathway-lp':
        key = next(key for key in a.mechanism.solver_state_keys if key.name == 'biomass_kg')
        index = a.unit.solver_state_collection.material_slices[key].start
    else:
        key = next(key for key in a.unit.solver_state_collection.material_slices if key.name == 'mass_j')
        index = a.unit.solver_state_collection.material_slices[key].start+a.phase.name_species.index('biomass')
    expected = (.4*10./11.-.38)*1e-4/3600.  # [kgDW/s]; mu(Sin)-D times initial biomass
    assert rates[index] == pytest.approx(expected, rel=1e-9, abs=1e-15)
    assert rates[index] < 0.


def test_fixed_step_dilution_converges_with_step_refinement(tmp_path):
    """Halving the Euler step reduces its error against analytical dilution.

    Parameters
    ----------
    tmp_path : Path
        Temporary inputs.
    """
    errors = []  # [kgDW]
    for step in (.25, .125):  # [h]; stable steps with D dt < 1
        case, model, path = continuous_inputs(tmp_path, growth=False)
        case['operation']['runtime']['value'] = 4.  # [h]
        case['numerics'] = dict.fromkeys(case['numerics'])
        case['numerics'].update(backend='fixed-step', step={'unit': 'h', 'value': step})
        a = build_bioreactor(case, model, path)
        final = np.asarray(a.solve()[0].biomass_kg_liquid0[-1]).item()  # [kgDW]
        assert final == pytest.approx(1e-4*(1.-.1*step)**round(4./step), rel=1e-9)
        errors.append(abs(final-1e-4*np.exp(-.4)))
    assert errors[1] < .51*errors[0]  # [-]; first-order Euler convergence


def test_reconciliation_reserves_outlet_before_inventory_projection():
    """One finite step reserves 10% of a limiting nutrient for withdrawal.

    The synthetic network has excess kinetic demand for nutrient A. With no A
    in the inlet, 0.0001 mmol initially and D dt=0.1, precisely 0.00009 mmol
    remains available to metabolism. Cell growth is zero in this fixture.
    """
    folder = Path(__file__).parent/'fixtures/reconciliation/operating_transitions/inputs/single'
    case, model = [json.loads((folder/f'{name}.json').read_text()) for name in ('case', 'mechanism')]
    case['operation'].update(mode='continuous', runtime={'unit':'day','value':.001})
    model['state']['species']['amounts']['nutrient_a'] = .0001  # [mmol]
    model['recipes']['benchmark'] = [{'event_type':'flow','time':{'unit':'day','value':0.},
        'volume_flow':{'unit':'L/day','value':100.}, 'concentrations':{'unit':'mmol/L','values':{}},
        'density':{'unit':'kg/L','value':1.}, 'composition':'nutrient-free'}]
    a = build_bioreactor(case, model, folder/'thermo.json')
    a._set_inlet(a.events[0])
    a.unit.compile_structure()
    states = a.unit.create_solver_init_states()
    a.mechanism.prepare_step(86.4)  # [s]; 0.001 day
    rates = a.unit.unit_model(0., states, limiter_dt=86.4).copy()
    fluxes = dict(zip(a.mechanism.last_solution.reaction_ids, a.mechanism.last_solution.fluxes))
    assert fluxes['ua'] == pytest.approx(.09, rel=1e-7, abs=1e-10)  # [mmol/(10^6 cell day)]
    key = next(key for key in a.unit.solver_state_collection.material_slices if key.name == 'mass_j')
    index = a.unit.solver_state_collection.material_slices[key].start+a.phase.name_species.index('nutrient_a')
    assert states[index]+86.4*rates[index] == pytest.approx(0., abs=1e-12)  # [kg]
    volume_key = next(key for key in a.unit.solver_state_collection.material_slices if key.name == 'working_volume')
    assert rates[a.unit.solver_state_collection.material_slices[volume_key]].item() == pytest.approx(0., abs=1e-15)


def test_reconciliation_continuous_inventory_and_product_solution():
    """A constant-flux synthetic culture follows exact dilution/source balances.

    The fixture's compatible targets give ua=1.5, ub=0.5, p=q=1 mmol per
    million cells per day and zero growth. With D=2.4/day, cell inventory is
    exp(-Dt), product is 0.1 t exp(-Dt) g and A is (10-1.5 t) exp(-Dt) mmol.
    """
    folder = Path(__file__).parent/'fixtures/reconciliation/operating_transitions/inputs/single'
    case, model = [json.loads((folder/f'{name}.json').read_text()) for name in ('case', 'mechanism')]
    case['operation'].update(mode='continuous', runtime={'unit':'h','value':2.})
    case['numerics'].update(backend='scipy', relative_tolerance=1e-9, absolute_tolerance=1e-12,
                           maximum_step={'unit':'s','value':300.})
    model['state']['species']['amounts'].update(nutrient_a=10., nutrient_b=20.)  # [mmol]
    model['recipes']['benchmark'] = [{'event_type':'flow','time':{'unit':'h','value':0.},
        'volume_flow':{'unit':'L/h','value':.1}, 'concentrations':{'unit':'mmol/L','values':{}},
        'density':{'unit':'kg/L','value':1.}, 'composition':'nutrient-free'}]
    a = build_bioreactor(case, model, folder/'thermo.json')
    history = a.solve()[0]
    time = history.time/86400.  # [day]
    decay = np.exp(-2.4*time)  # [-]
    np.testing.assert_allclose(np.asarray(history.viable_cells_million_liquid0).reshape(-1), decay,
                               rtol=2e-6, atol=1e-10)
    np.testing.assert_allclose(np.asarray(history.product_g_liquid0).reshape(-1), .1*time*decay,
                               rtol=2e-6, atol=1e-10)
    index = a.phase.name_species.index('nutrient_a')
    amount = history.mass_j_liquid0[:,index]/a.phase.mw[index]*1e6  # [mmol]
    np.testing.assert_allclose(amount, (10.-1.5*time)*decay, rtol=2e-6, atol=1e-9)
    np.testing.assert_allclose(history.vessel_vol, .001, rtol=1e-9, atol=1e-14)
