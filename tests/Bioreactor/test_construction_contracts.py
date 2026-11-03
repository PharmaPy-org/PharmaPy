"""Physical initialization, endpoint operations and declared reaction invariants.

Real native reactors and property fixtures provide independent mass/volume/heat
oracles. Optional CVode cases exercise the same public JSON path.
"""

import json
import numpy as np
import pytest

from PharmaPy.Bioreactors import build_bioreactor
from tests.Bioreactor.test_bioreactor_adaptive_inventory import configured_case, EXAMPLE
from tests.Bioreactor.test_bioreactor_thermal import thermal_case
from tests.Bioreactor.test_forward_audit import native_case


# Explicit synthetic compositions isolate property and event contracts.
def concentrated_case(tmp_path):
    """Create 0.1 kg solute at 2000 kg/m3 and 0.9 kg carrier at 1000 kg/m3.

    Parameters
    ----------
    tmp_path : Path
        Temporary thermodynamic fixture location.

    Returns
    -------
    tuple
        Process JSON and thermodynamic path; ideal additive volume is 0.95 L.
    """
    document = configured_case('scipy')
    document['operation']['volume_policy'] = 'native'
    document['state']['species']['amounts']['biomass'] = 100.  # [g]
    document['operation']['initial_volume']['value'] = .95  # [L]; 0.1/2 + 0.9/1
    thermo = json.loads((EXAMPLE/'thermo.json').read_text())
    thermo['biomass']['rho_liq'] = 2000.  # [kg/m3]; deliberately distinct solute density
    path = tmp_path/'thermo.json'
    path.write_text(json.dumps(thermo))
    return document, path


@pytest.mark.integration
def test_native_density_uses_mixture_composition(tmp_path):
    document, path = concentrated_case(tmp_path)
    document['operation']['density'] = {'mode': 'native-phase'}
    assembly = build_bioreactor(document, thermo_path=path)
    assert assembly.phase.vol == pytest.approx(.00095)  # [m3]
    report = assembly.input_audit()['construction']
    assert report['density']['initial_check_density_kg_l'] == pytest.approx(1./.95)
    history, = assembly.solve()
    # Equal-mass conversion turns dense biomass into substrate of density 1000.
    expected = .001 - .00005*np.exp(-4.*history.time)  # [m3]; additive native volumes
    np.testing.assert_allclose(history.vessel_vol, expected, rtol=1e-6, atol=1e-10)


@pytest.mark.integration
@pytest.mark.parametrize('unit,value', [('kg/L', 1./.95), ('kg/m3', 1000./.95)])
def test_constant_mixture_density_is_distinct_from_carrier(tmp_path, unit, value):
    document, path = concentrated_case(tmp_path)
    document['operation']['density'] = {'mode': 'constant-mixture', 'unit': unit, 'value': value}
    assembly = build_bioreactor(document, thermo_path=path)
    history, = assembly.solve()
    np.testing.assert_allclose(history.vessel_vol, .00095, rtol=0., atol=1e-12)  # [m3]; explicitly constant mixture density
    np.testing.assert_allclose(history.mass_j_liquid0[:, 0], .1*np.exp(-4.*history.time), rtol=1e-6, atol=1e-10)  # [kg]; volume and kinetics must share one concentration basis
    np.testing.assert_allclose(assembly.phase.mass_conc*assembly.phase.vol, assembly.phase.mass_j, rtol=1e-14)
    np.testing.assert_allclose(assembly.phase.mole_conc*assembly.phase.mw, assembly.phase.mass_conc, rtol=1e-14)
    assert assembly.carrier_density_kg_l == 1.  # [kg/L]; independent pure-carrier input
    audit = assembly.input_audit()
    audit['construction']['density']['mode'] = 'changed copy'
    assert assembly.input_audit()['construction']['density']['mode'] == 'constant-mixture'
    assembly.reset()
    repeated, = assembly.solve()
    np.testing.assert_allclose(repeated.mass_j_liquid0, history.mass_j_liquid0, rtol=1e-10, atol=1e-12)


@pytest.mark.unit
def test_density_conflicts_are_not_silently_reconciled(tmp_path):
    document, path = concentrated_case(tmp_path)
    with pytest.raises(ValueError, match='initial_mass_kg must agree'):
        build_bioreactor(document, thermo_path=path)
    document['operation']['density'] = {'mode': 'native-phase'}
    document['operation']['volume_policy'] = 'working-volume'
    with pytest.raises(ValueError, match='independently prescribed'):
        build_bioreactor(document, thermo_path=path)


@pytest.mark.integration
@pytest.mark.parametrize('backend', ['scipy', pytest.param('assimulo', marks=pytest.mark.assimulo)])
def test_terminal_feed_is_applied_once_and_replays_after_reset(backend):
    if backend == 'assimulo':
        pytest.importorskip('assimulo')
    document = configured_case(backend)
    document['recipes']['batch'] = [{'time': 2., 'event_type': 'feed', 'volume_l': .5,
                                    'composition': 'nutrient-free', 'concentrations_mmol_l': {}}]  # [s], [L]
    assembly = build_bioreactor(document, thermo_path=EXAMPLE/'thermo.json')
    before, after = assembly.solve()
    assert before.time[-1] == pytest.approx(2.)  # [s]
    np.testing.assert_array_equal(after.time, [2.])
    assert before.mass_j_liquid0[-1].sum() == pytest.approx(1.)  # [kg]
    assert after.mass_j_liquid0[0].sum() == pytest.approx(1.5)  # [kg]; 0.5 L carrier, 1 kg/L
    assembly.reset()
    again = assembly.solve()
    np.testing.assert_allclose(again[-1].mass_j_liquid0, after.mass_j_liquid0, rtol=1e-10, atol=1e-12)


@pytest.mark.integration
@pytest.mark.parametrize('density_mode', ['legacy', 'constant-mixture', 'native-phase'])
def test_terminal_cold_feed_uses_native_enthalpy_mixing(thermal_case, density_mode):
    document, path = thermal_case
    if density_mode != 'legacy':
        document['operation']['volume_policy'] = 'native'
        document['operation']['density'] = {'mode': density_mode}
        if density_mode == 'constant-mixture':
            document['operation']['density'].update(value=1., unit='kg/L')  # [kg/L]; same synthetic liquid property
    document['recipes']['continuous'] = [{'time': 100., 'event_type': 'feed', 'volume_l': .5,
        'composition': 'nutrient-free', 'concentrations_mmol_l': {}, 'temperature_k': 270.}]  # [s], [L], [K]
    assembly = build_bioreactor(document, thermo_path=path)
    before, after = assembly.solve()
    expected = (float(np.asarray(before.global_temp).reshape(-1)[-1]) + .5*270.)/1.5  # [K]; equal Cp, 1 kg + 0.5 kg
    assert float(np.asarray(after.global_temp).reshape(-1)[0]) == pytest.approx(expected, abs=1e-7)


@pytest.mark.assimulo
@pytest.mark.integration
def test_assimulo_preserves_every_requested_time_near_terminal_feed(thermal_case):
    pytest.importorskip('assimulo')
    document, path = thermal_case
    document['operation']['volume_policy'] = 'native'
    document['operation']['density'] = {'mode': 'constant-mixture', 'value': 1., 'unit': 'kg/L'}
    document['numerical'].update(backend='assimulo', relative_tolerance=1e-6, absolute_tolerance=1e-10,
                                  maximum_step={'unit': 's', 'value': 5.})  # [-], [native kg,K,m3], [s]; regression where CVode omits 97.5 s without native continuous reporting
    document['recipes']['continuous'] = [{'time': 100., 'event_type': 'feed', 'volume_l': .5,
        'composition': 'nutrient-free', 'concentrations_mmol_l': {}, 'temperature_k': 270.}]  # [s], [L], [K]
    times = np.linspace(0., 100., 41)  # [s]; 2.5 s spacing includes the formerly missing 97.5 s row
    before, after = build_bioreactor(document, thermo_path=path).solve(report_times=times)
    np.testing.assert_array_equal(before.time, times)
    np.testing.assert_array_equal(after.time, [100.])


@pytest.mark.integration
def test_feed_density_does_not_overwrite_mixture_density(tmp_path):
    document, path = concentrated_case(tmp_path)
    document['operation']['density'] = {'mode': 'constant-mixture', 'value': 1./.95, 'unit': 'kg/L'}
    document['recipes']['batch'] = [{'time': 2., 'event_type': 'feed', 'volume_l': .1,
        'density_kg_l': 1.2, 'composition': 'nutrient-free', 'concentrations_mmol_l': {}}]  # [s], [L], [kg/L]; 0.12 kg added
    assembly = build_bioreactor(document, thermo_path=path)
    before, after = assembly.solve()
    assert after.mass_j_liquid0[0].sum() == pytest.approx(1.12)  # [kg]
    assert after.vessel_vol[0] == pytest.approx(.001064)  # [m3]; 1.12 kg / fixed (1/0.95 kg/L)


@pytest.mark.integration
@pytest.mark.parametrize('mode', ['fed-batch', 'continuous'])
@pytest.mark.parametrize('backend', ['scipy', pytest.param('assimulo', marks=pytest.mark.assimulo)])
def test_fixed_density_flow_modes_match_inventory_balances(tmp_path, mode, backend):
    if backend == 'assimulo':
        pytest.importorskip('assimulo')
    document, path = concentrated_case(tmp_path)
    document['numerical']['backend'] = backend
    document['operation']['mode'] = mode
    document['operation']['density'] = {'mode': 'constant-mixture', 'value': 1./.95, 'unit': 'kg/L'}
    document['recipes']['batch'] = [{'time': 0., 'event_type': 'flow', 'volume_l_h': 360.,
        'density_kg_l': 1./.95, 'composition': 'nutrient-free', 'concentrations_mmol_l': {}}]  # [s], [L/h], [kg/L]; 0.1 L/s at reactor density
    document['recipes']['batch'].append({'time': 2., 'event_type': 'flow', 'volume_l_h': 0.})  # [s], [L/h]; terminal shutoff must add no interval
    assembly = build_bioreactor(document, thermo_path=path)
    history, stopped = assembly.solve()
    np.testing.assert_array_equal(stopped.mass_j_liquid0[0], history.mass_j_liquid0[-1])
    assert not assembly.unit.inlet_connections
    survival = np.exp(-(.1/.95)*history.time) if mode == 'continuous' else np.ones_like(history.time)  # [-]; CSTR dilution, or no withdrawal
    expected_biomass = .1*np.exp(-4.*history.time)*survival  # [kg]; reaction plus possible washout
    expected_substrate = .1*(1.-np.exp(-4.*history.time))*survival  # [kg]; independent integrated S formation and washout
    total = np.ones_like(history.time) if mode == 'continuous' else 1.+(.1/.95)*history.time  # [kg]
    np.testing.assert_allclose(history.mass_j_liquid0[:, 0], expected_biomass, rtol=1e-6, atol=1e-10)
    np.testing.assert_allclose(history.mass_j_liquid0[:, 1], expected_substrate, rtol=1e-6, atol=1e-10)
    np.testing.assert_allclose(history.mass_j_liquid0.sum(axis=1), total, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(history.vessel_vol, total*.00095, rtol=1e-8, atol=1e-12)  # [m3]


@pytest.mark.integration
@pytest.mark.parametrize('time', [0., .5, 2.])  # [s]; initial, interior and terminal delivery
def test_feed_time_has_independent_piecewise_first_order_solution(time):
    document = configured_case('scipy')
    document['recipes']['batch'] = [{'time': time, 'event_type': 'feed', 'volume_l': .5,
        'composition': 'supplied', 'concentrations_mmol_l': {'biomass': 2.}}]  # [s], [L], [mmol/L]; adds exactly 1 g biomass
    assembly = build_bioreactor(document, thermo_path=EXAMPLE/'thermo.json')
    histories = assembly.solve()
    expected = .001*np.exp(-8.) + .001*np.exp(-4.*(2.-time))  # [kg]; independently superposed first-order trajectories
    assert histories[-1].mass_j_liquid0[-1, 0] == pytest.approx(expected, rel=1e-6, abs=1e-11)


@pytest.mark.integration
@pytest.mark.parametrize('sample_first,retained', [(True, .75), (False, 5./6.)])  # [-]; removal from 1 L or 1.5 L
def test_simultaneous_terminal_order_is_physical(sample_first, retained):
    document = configured_case('scipy')
    sample = {'time': 2., 'event_type': 'sample', 'volume_l': .25}  # [s], [L]
    feed = {'time': 2., 'event_type': 'feed', 'volume_l': .5,
            'composition': 'nutrient-free', 'concentrations_mmol_l': {}}  # [s], [L]
    document['recipes']['batch'] = [sample, feed] if sample_first else [feed, sample]
    assembly = build_bioreactor(document, thermo_path=EXAMPLE/'thermo.json')
    before, after = assembly.solve()
    assert after.mass_j_liquid0[0, 0] == pytest.approx(retained*.001*np.exp(-8.), rel=1e-6, abs=1e-11)  # [kg]
    assert after.mass_j_liquid0[0].sum() == pytest.approx(1.25)  # [kg]


@pytest.mark.integration
def test_audit_and_solver_share_terminal_roundoff_convention():
    document = configured_case('scipy')
    document['recipes']['batch'] = [{'time': np.nextafter(2., np.inf), 'event_type': 'feed',
        'volume_l': .5, 'composition': 'unknown', 'concentrations_mmol_l': {}}]  # [s], [L]; one representable step beyond endpoint
    assembly = build_bioreactor(document, thermo_path=EXAMPLE/'thermo.json')
    assert assembly.input_audit()['feed_composition'] == 'INCOMPLETE'
    with pytest.raises(ValueError, match='complete feed compositions are required'):
        assembly.solve()


@pytest.mark.integration
def test_public_balance_report_requires_independent_sources_and_jumps():
    assembly = native_case(flow=True)
    histories = assembly.solve()
    names = list(assembly.phase.name_species)
    feed = np.array([.0001 if name == 'nutrient' else .0999 for name in names])/3600.  # [kg/s]; declared 0.1 L/h, 1 g/L nutrient
    initial = np.array([.001 if name == 'nutrient' else .999 for name in names])  # [kg]; declared 1 L charge
    jump = -(initial + feed*900.)*(.05/1.025)  # [kg]; sample after 0.25 h, before continuing flow
    ledger = {kind: [np.zeros_like(h.mass_j_liquid0) for h in histories]
              for kind in ('inlet', 'outlet', 'transfer', 'biological')}
    ledger['inlet'] = [np.tile(feed, (len(h.time), 1)) for h in histories]
    budget = np.full(len(names), 1e-10)  # [kg]; explicit budget above ODE roundoff, far below fixture inventories
    evidence = dict(ledger=ledger, jumps_kg=[jump], budget_kg=budget,
        quadrature_error_kg=np.zeros(len(names)), invariants={'mass': dict.fromkeys(names, 1.)},
        invariant_budget={'mass': 1e-10})  # [kg]; constant-rate quadrature is exact
    assert assembly.balance_report(histories, **evidence)['status'] == 'passed'
    evidence['jumps_kg'] = [np.zeros_like(jump)]
    assert assembly.balance_report(histories, **evidence)['status'] == 'failed'
    assert assembly.balance_report(histories)['status'] == 'not_assessable'


@pytest.mark.unit
def test_declared_conservation_checks_are_scoped_and_order_independent():
    document = configured_case('scipy')
    coefficients = {'biomass': 1., 'substrate': 1., 'product': 1., 'water': 1.}  # [kg/kg]; total represented phase mass
    declaration = {'basis': 'kg/extent', 'reactions': {'conversion': {'substrate': 1., 'biomass': -1.}},
        'invariants': {'mass': {'unit': 'kg', 'coefficients': coefficients, 'absolute_tolerance': 1e-12}}}  # [kg/extent]; roundoff budget
    document['operation']['conservation'] = declaration
    assembly = build_bioreactor(document, thermo_path=EXAMPLE/'thermo.json')
    assert assembly.input_audit()['construction']['conservation']['status'] == 'passed'
    declaration['invariants']['mass']['coefficients'] = dict(reversed(list(coefficients.items())))
    reordered = build_bioreactor(document, thermo_path=EXAMPLE/'thermo.json')
    assert reordered.input_audit()['construction']['conservation'] == assembly.input_audit()['construction']['conservation']
    declaration['reactions']['conversion']['substrate'] = .9  # [kg/extent]; deliberately missing mass destination
    with pytest.raises(ValueError, match='invariant mass violated'):
        build_bioreactor(document, thermo_path=EXAMPLE/'thermo.json')
    declaration['invariants']['mass']['coefficients'].pop('water')
    partial = build_bioreactor(document, thermo_path=EXAMPLE/'thermo.json')
    assert partial.input_audit()['construction']['conservation']['status'] == 'not_assessable'


@pytest.mark.unit
@pytest.mark.parametrize('value', [0., -1., float('inf'), float('nan'), [1., 2.]])
def test_invalid_mixture_density_fails_before_solving(tmp_path, value):
    document, path = concentrated_case(tmp_path)
    document['operation']['density'] = {'mode': 'constant-mixture', 'value': value, 'unit': 'kg/L'}
    with pytest.raises(ValueError, match='finite'):
        build_bioreactor(document, thermo_path=path)


@pytest.mark.unit
def test_molar_conservation_uses_species_weights_and_order(tmp_path):
    document = configured_case('scipy')
    # Formal mass-equivalence test only; this is not a chemical reaction claim.
    document['operation']['conservation'] = {'basis': 'mol/extent',
        'reactions': {'formal': {'biomass': -1., 'water': 1000./18.015}},  # [mol/extent]; both sides represent 1 kg
        'invariants': {'mass': {'unit': 'kg', 'coefficients': dict.fromkeys(('water', 'product', 'substrate', 'biomass'), 1.),
                                'absolute_tolerance': 1e-12}}}  # [kg/extent]; roundoff budget, not measurement uncertainty
    thermo = json.loads((EXAMPLE/'thermo.json').read_text())
    for ordering in (thermo, dict(reversed(list(thermo.items())))):
        path = tmp_path/'reordered.json'
        path.write_text(json.dumps(ordering))
        assembly = build_bioreactor(document, thermo_path=path)
        result = assembly.input_audit()['construction']['conservation']
        assert result['status'] == 'passed'
        assert result['invariants']['mass']['residuals']['formal'] == pytest.approx(0., abs=1e-12)
    document['operation']['conservation']['reactions']['formal']['water'] = 1.  # [mol/extent]; wrong molecular conversion
    with pytest.raises(ValueError, match='invariant mass violated'):
        build_bioreactor(document, thermo_path=path)
    document['operation']['conservation']['reactions']['formal'] = {
        'biomass': [-1.], 'water': [1.], 'substrate': [0.], 'product': [0.]}  # [mol/extent]; column-shaped input must not broadcast molecular weights into a false balance
    with pytest.raises(ValueError, match='finite scalars'):
        build_bioreactor(document, thermo_path=path)
