"""Independent physical conversions and native integration of declared units."""

import json
from pathlib import Path

import numpy as np
import pytest

from PharmaPy.Bioreactors import BioreactorUnitConverter as Units, build_bioreactor


EXAMPLES = Path(__file__).parents[2] / 'examples/bioreactors'


@pytest.mark.parametrize('value, source, target, expected, properties', [
    (2., 'h', 's', 7200., {}),
    (750., 'mL', 'm3', .00075, {}),
    (1., 'pg', 'g', 1e-12, {}),
    (3., 'pmol', 'nmol', .003, {}),
    (2., '10^6 cell', 'cell', 2e6, {}),
    (1., 'mM', 'mol/m3', 1., {}),
    (2., 'g/L', 'kg/m3', 2., {}),
    (60., 'mL/min', 'm3/s', 1e-6, {}),
    (3.6, 'g/(L h)', 'kg/(m3 s)', .001, {}),
    (1., 'mmol/(mL min)', 'kg/(m3 s)', 3., {'molecular_weight_g_mol': 180.}),
    (1e6, 'cell/mL', '10^6 cell/L', 1000., {}),
    (37., 'degC', 'K', 310.15, {}),
    (32., 'degF', 'degC', 0., {}),
    (2., 'mmol', 'g', .36, {'molecular_weight_g_mol': 180.}),
    (2., 'mmol-C', 'g', .04508, {'mass_per_cmol_g': 22.54}),
    (1., 'mol', 'mol-C', 6., {'carbon_atoms': 6.}),
    (6., 'mol-C', 'mol', 1., {'molecular_weight_g_mol': 180., 'mass_per_cmol_g': 30.}),
    (3., '10^6 cell', 'g', .6, {'biomass_g_per_million_cells': .2}),
    (1., 'mmol/L', 'mg/L', 180., {'molecular_weight_g_mol': 180.}),
])
def test_physical_conversions_and_round_trip(value, source, target, expected, properties):
    actual = Units.convert(value, source, target, **properties)
    assert actual == pytest.approx(expected)
    assert Units.convert(actual, target, source, **properties) == pytest.approx(value)


@pytest.mark.parametrize('source, target, properties', [
    ('mol', 'g', {}), ('mol-C', 'g', {}), ('cell', 'g', {}),
    ('mol', 'mol-C', {}), ('g', 's', {}), ('unknown', 'unknown', {}),
    ('mol', 'g', {'molecular_weight_g_mol': -1.}),
    ('mol-C', 'g', {'mass_per_cmol_g': float('nan')}),
    ('cell', 'kg', {'biomass_g_per_million_cells': 0.}),
    ('mol/h', 'g/s', {'molecular_weight_g_mol': 180.}),
    ('mol/(L h)', 'kg/(m3 s)', {}),
    ('g/(L fortnight)', 'kg/(m3 s)', {}),
    ('L/h', 'kg/s', {}),
    ('g/(L h)', 'kg/m3', {}),
    (None, 'g', {}),
])
def test_missing_properties_and_incompatible_units_fail(source, target, properties):
    with pytest.raises(ValueError):
        Units.convert(1., source, target, **properties)


def test_arrays_native_phase_properties_and_invalid_values():
    p = EXAMPLES / 'generic_batch/inputs'
    assembly = build_bioreactor(json.loads((p/'case.json').read_text()),
                                json.loads((p/'mechanism.json').read_text()), p/'thermo.json')
    phase = assembly.phase
    concentration = np.ones(len(phase.mw)) * .01
    original = concentration.copy()
    actual = Units.convert(concentration, 'mol/L', 'kg/m3', molecular_weight_g_mol=phase.mw)
    np.testing.assert_allclose(actual, phase.conc_to_conc(mole_conc=concentration), rtol=1e-14)
    np.testing.assert_array_equal(concentration, original)
    result = Units.convert(concentration, 'mol', 'mol')
    result[0] = 100.
    np.testing.assert_array_equal(concentration, original)
    for bad in (float('inf'), float('nan')):
        with pytest.raises(ValueError):
            Units.convert(bad, 'mol', 'mol')
    with pytest.raises(ValueError):
        Units.convert(-274., 'degC', 'K')


@pytest.mark.parametrize('name', ['generic_batch', 'generic_fed_batch', 'batch_ecoli_dfba', 'fed_batch_cho'])
def test_json_paths_and_mappings_share_native_converter(name):
    p = (Path(__file__).parent / 'fixtures' if name == 'fed_batch_cho' else EXAMPLES) / name / 'inputs'
    case, mechanism = [json.loads((p/file).read_text()) for file in ('case.json', 'mechanism.json')]
    before = json.dumps([case, mechanism], sort_keys=True)
    paths = Units.from_inputs(p/'case.json', p/'mechanism.json')
    mappings = Units.from_inputs(case, mechanism)
    native = build_bioreactor(case, mechanism, p/'thermo.json').mechanism.unit_converter
    fields = ('exchange_scale', 'growth_scale', 'product_mass', 'product_scale') if name == 'fed_batch_cho' else ('per_second',)
    for field in fields:
        assert getattr(paths, field) == getattr(mappings, field) == getattr(native, field)
    assert json.dumps([case, mechanism], sort_keys=True) == before


def test_additional_json_volume_and_amount_units_preserve_initial_state():
    p = EXAMPLES / 'generic_batch/inputs'
    case, definition = [json.loads((p/file).read_text()) for file in ('case.json', 'mechanism.json')]
    original = build_bioreactor(case, definition, p/'thermo.json')
    case['operation']['initial_volume'] = {'unit': 'mL', 'value': 1000.}
    definition['state']['species']['unit'] = 'umol'
    definition['state']['species']['amounts'] = {key: val * 1e6 for key, val in definition['state']['species']['amounts'].items()}
    converted = build_bioreactor(case, definition, p/'thermo.json')
    np.testing.assert_allclose(converted.phase.mass_j, original.phase.mass_j, rtol=1e-14)
    np.testing.assert_array_equal(converted.time_grid, original.time_grid)


def test_rate_normalization_minutes_and_overflow():
    basis = dict(amount_unit='pmol', normalization='gDW', time_unit='min',
                 biomass_g_per_million_cells=.2)
    factor, label = Units.rate_basis(basis)
    assert factor == pytest.approx(288.)
    assert label == 'pmol/(gDW min)'
    basis['biomass_g_per_million_cells'] = 1e308
    with pytest.raises(ValueError, match='finite'):
        Units.rate_basis(basis)


@pytest.mark.parametrize('amount, normalization, time, content, scale', [
    ('mol', 'kgDW', 'h', None, 1.),
    ('mmol', 'gDW', 'h', None, 1.),
    ('umol', 'gDW', 'h', None, .001),
    ('nmol', '10^6 cell', 'h', .2, 5e-6),
    ('pmol', 'cell', 'h', .2, .005),
    ('mmol', 'gDW', 'day', None, 1.),
    ('mol', 'kgDW', 'min', None, 1.),
])
def test_equivalent_pathway_bases_preserve_native_simulation(amount, normalization, time, content, scale):
    p = EXAMPLES / 'generic_batch/inputs'
    case, definition = [json.loads((p/file).read_text()) for file in ('case.json', 'mechanism.json')]
    case['operation']['runtime']['value'] = .2
    reference = build_bioreactor(case, definition, p/'thermo.json').solve(verbose=False)[0]
    exchange = dict(amount_unit=amount, normalization=normalization, time_unit=time)
    if content is not None:
        exchange['biomass_g_per_million_cells'] = content
    definition['model']['pathways']['flux_basis'] = {
        'exchange': exchange, 'growth': dict(amount_unit='1', normalization='none', time_unit=time)}
    definition['flux_time_unit'] = '1/' + time
    definition['model']['pathways']['exchange_matrix'] = [[-10. / scale]]
    definition['parameters']['rates']['uptake']['nutrient'] = 2. * Units.TIME[time] / 3600. / scale
    assembly = build_bioreactor(case, definition, p/'thermo.json')
    actual = assembly.solve(verbose=False)[0]
    assert Units.from_inputs(case, definition).exchange_scale == pytest.approx(scale)
    assert assembly.mechanism.audit_conservation()['flux_basis'] == 'DECLARED'
    for field in ('mass_j_liquid0', 'biomass_kg_liquid0'):
        np.testing.assert_allclose(getattr(actual, field), getattr(reference, field), rtol=1e-9, atol=1e-13)


def test_cell_and_dry_mass_bases_require_supplied_cell_mass_only_when_crossing():
    basis = dict(amount_unit='mmol', normalization='gDW', time_unit='h')
    assert Units.rate_basis(basis, normalization='kgDW', time_unit='h')[0] == 1000.
    with pytest.raises(ValueError, match='biomass_g_per_million_cells'):
        Units.rate_basis(basis)
    basis = dict(amount_unit='nmol', normalization='cell', time_unit='h')
    with pytest.raises(ValueError, match='biomass_g_per_million_cells'):
        Units.rate_basis(basis, normalization='kgDW')
    basis['biomass_g_per_million_cells'] = .2
    assert Units.rate_basis(basis, normalization='kgDW', time_unit='h')[0] == pytest.approx(5e9)


@pytest.mark.parametrize('exchange, growth', [
    (dict(amount_unit='nmol', normalization='cell', time_unit='h'),
     dict(amount_unit='1', normalization='none', time_unit='h')),
    (dict(amount_unit='mol-C', normalization='kgDW', time_unit='h'),
     dict(amount_unit='1', normalization='none', time_unit='h')),
    (dict(amount_unit='mol', normalization='kgDW', time_unit='day'),
     dict(amount_unit='1', normalization='none', time_unit='h')),
    (dict(amount_unit='mol', normalization='kgDW', time_unit='h'),
     dict(amount_unit='mol-C', normalization='kgDW', time_unit='h')),
])
def test_ambiguous_or_incompatible_pathway_bases_fail(exchange, growth):
    with pytest.raises(ValueError):
        Units(flux_time_unit='1/h', flux_basis=dict(exchange=exchange, growth=growth))


def test_pathway_transfer_cap_uses_declared_exchange_basis():
    p = EXAMPLES / 'batch_ecoli_dfba/inputs'
    case, definition = [json.loads((p/file).read_text()) for file in ('case.json', 'mechanism.json')]
    original = build_bioreactor(case, definition, p/'thermo.json').mechanism.model
    model = definition['model']['pathways']
    model['flux_basis'] = {
        'exchange': dict(amount_unit='mmol', normalization='kgDW', time_unit='h'),
        'growth': dict(amount_unit='1', normalization='none', time_unit='h')}
    model['exchange_matrix'] = (np.asarray(model['exchange_matrix']) * 1000.).tolist()
    uptake = definition['parameters']['rates']['uptake']
    for name in uptake:
        uptake[name] *= 1000.
    converted = build_bioreactor(case, definition, p/'thermo.json').mechanism.model
    for biomass in (.001, 1.):
        first = original.solve_fluxes([10.8, .4, .21], biomass)
        second = converted.solve_fluxes([10.8, .4, .21], biomass)
        np.testing.assert_allclose(second.fluxes, first.fluxes, atol=1e-10)
        np.testing.assert_allclose(second.extracellular_rates * converted.exchange_scale,
                                   first.extracellular_rates, atol=1e-9)


@pytest.mark.parametrize('source, target, value, expected', [
    (('mmol', '10^6 cell', 'day'), ('umol', '10^6 cell', 'day'), 1e-6, .001),
    (('mmol', '10^6 cell', 'day'), ('mol', 'cell', 'h'), 24., 1e-9),
    (('1', 'none', 'day'), ('1', 'none', 'h'), 24., 1.),
    (('g', 'gDW', 'h'), ('mg', 'kgDW', 'day'), 1., 24e6),
    (('mmol-C', '10^6 cell', 'day'), ('umol-C', '10^6 cell', 'day'), 1., 1000.),
])
def test_reconciliation_rate_scale_conversion(source, target, value, expected):
    keys = ('amount_unit', 'normalization', 'time_unit')
    source, target = dict(zip(keys, source)), dict(zip(keys, target))
    result = Units.rate_quantity({'value': value, 'basis': source}, target)
    assert result == pytest.approx(expected)
    assert Units.rate_quantity({'value': result, 'basis': target}, source) == pytest.approx(value)


@pytest.mark.parametrize('value', [0., -1., float('inf'), float('nan')])
def test_reconciliation_invalid_scale(value):
    with pytest.raises(ValueError, match='positive'):
        Units.rate_quantity({'value': value, 'basis': None}, None)


@pytest.mark.parametrize('source_unit, target_unit', [('1', 'mmol'), ('g', 'mmol'), ('mmol-C', 'mmol')])
def test_reconciliation_does_not_invent_biological_conversion(source_unit, target_unit):
    source = dict(amount_unit=source_unit, normalization='10^6 cell', time_unit='day')
    target = dict(source, amount_unit=target_unit)
    with pytest.raises(ValueError):
        Units.rate_quantity({'value': 1., 'basis': source}, target)


def test_reconciliation_explicit_legacy_source_scale():
    assert Units.rate_quantity({'value': 1e-6, 'basis': None}, None) == 1e-6


def test_rate_scale_population_basis_requires_supplied_mass():
    source = dict(amount_unit='mmol', normalization='gDW', time_unit='day')
    target = dict(amount_unit='mmol', normalization='10^6 cell', time_unit='day')
    with pytest.raises(ValueError, match='biomass_g_per_million_cells'):
        Units.rate_quantity({'value': 1., 'basis': source}, target)
    source['biomass_g_per_million_cells'] = .2
    assert Units.rate_quantity({'value': 1., 'basis': source}, target) == pytest.approx(.2)
