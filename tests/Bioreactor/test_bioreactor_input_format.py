"""The shared forward template preserves native model inputs and behavior."""

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from PharmaPy.Bioreactors import build_bioreactor
from PharmaPy.Bioreactors.input_format import normalize_forward_inputs
from PharmaPy.Bioreactors.construction import _apply_material_event


ROOT = Path(__file__).parents[2]
EXAMPLES = ROOT / "examples/bioreactors"
NAMES = ("batch_ecoli_dfba", "fed_batch_cho", "generic_batch", "generic_fed_batch")


def test_reported_temperature_and_stock_feed_material_balance():
    case, definition, thermo = _load("fed_batch_cho")
    assembly = build_bioreactor(case, definition, Path(__file__).parent / "fixtures/fed_batch_cho/inputs/thermo.json")
    phase = assembly.phase
    assert phase.temp == 310.15
    event = next(e for e in assembly.events if e["event_type"] == "stock_to_target")
    before = np.asarray(phase.mass_j).copy()
    names = list(phase.name_species)
    glucose, water = names.index("GLC"), names.index("water")
    initial_volume = phase.vol * 1000
    # Independent calculation in grams and liters using the paper's 9/450 g/L.
    addition = (9 * initial_volume - before[glucose] * 1000) / (450 - 9)
    _apply_material_event(phase, event, 1., "water")
    np.testing.assert_allclose(phase.vol * 1000, initial_volume + addition, rtol=1e-12)
    np.testing.assert_allclose(phase.mass_j[glucose] * 1000 / (phase.vol * 1000), 9.)
    np.testing.assert_allclose(np.sum(phase.mass_j) - np.sum(before), addition)
    for i in range(len(names)):
        if i not in (glucose, water):
            np.testing.assert_allclose(phase.mass_j[i], before[i], rtol=1e-14, atol=0.)
    after = np.asarray(phase.mass_j).copy()
    _apply_material_event(phase, event, 1., "water")
    np.testing.assert_allclose(phase.mass_j, after, atol=1e-14)


@pytest.mark.parametrize("stock", [0., 1., -1., float("nan"), 1e9])
def test_invalid_stock_is_rejected_without_changing_inventory(stock):
    case, definition, _ = _load("fed_batch_cho")
    assembly = build_bioreactor(case, definition, Path(__file__).parent / "fixtures/fed_batch_cho/inputs/thermo.json")
    event = deepcopy(next(e for e in assembly.events if e["event_type"] == "stock_to_target"))
    event["concentrations_mmol_l"]["GLC"] = stock
    before = np.asarray(assembly.phase.mass_j).copy()
    with pytest.raises(ValueError):
        _apply_material_event(assembly.phase, event, 1., "water")
    np.testing.assert_array_equal(assembly.phase.mass_j, before)


def _load(name):
    folder = (Path(__file__).parent / 'fixtures' if name == 'fed_batch_cho' else EXAMPLES) / name / "inputs"
    return [json.loads((folder / file).read_text())
            for file in ("case.json", "mechanism.json", "thermo.json")]


def test_all_examples_use_the_same_ordered_forward_template():
    template = ROOT / "doc/online_docs/bioreactors/templates"
    case, model, thermo = [json.loads((template / file).read_text())
                           for file in ("case.json", "mechanism.json", "thermo.json")]
    for name in NAMES:
        configured_case, configured_model, properties = _load(name)
        assert list(configured_case) == list(case)
        for field in ("biology", "numerics", "operation", "recipe"):
            assert list(configured_case[field]) == list(case[field])
        assert list(configured_model) == list(model)
        for field in ("model", "state"):
            assert list(configured_model[field]) == list(model[field])
        for species in properties.values():
            assert list(species) == list(next(iter(thermo.values())))


@pytest.mark.parametrize("name", NAMES)
def test_standard_inputs_keep_legacy_construction_and_do_not_mutate_inputs(name):
    case, definition, _ = _load(name)
    before = deepcopy((case, definition))
    legacy_case, legacy_definition = normalize_forward_inputs(case, definition)
    thermo = (Path(__file__).parent / 'fixtures' if name == 'fed_batch_cho' else EXAMPLES) / name / "inputs/thermo.json"
    current = build_bioreactor(case, definition, thermo)
    legacy = build_bioreactor(legacy_case, legacy_definition, thermo)
    assert (case, definition) == before
    assert type(current.unit) is type(legacy.unit)
    np.testing.assert_array_equal(current.phase.mass_j, legacy.phase.mass_j)
    np.testing.assert_array_equal(current.time_grid, legacy.time_grid)
    assert current.events == legacy.events


@pytest.mark.parametrize("change", ["version", "inactive", "unknown", "missing"])
def test_standard_inputs_reject_unsupported_or_inactive_configuration(change):
    case, definition, _ = _load("generic_batch")
    if change == "version":
        definition["schema_version"] += 1
    elif change == "inactive":
        definition["model"]["reconciliation"]["reconciliation_relative_bound"] = 2.5
    elif change == "unknown":
        definition["model"]["misspelled_parameter"] = 1.0
    else:
        del definition["state"]["species"]
    with pytest.raises(ValueError):
        normalize_forward_inputs(case, definition)


def test_species_pathway_and_parameter_names_remain_user_defined(tmp_path):
    case, definition, thermo = _load("generic_batch")
    definition["parameters"]["rates"] = {"user_capacity": 2.0}
    definition["model"]["pathways"]["state_ids"] = ["user_substrate"]
    definition["model"]["pathways"]["pathway_ids"] = ["user_pathway"]
    rule = definition["model"]["pathways"]["uptake_constraints"][0]
    rule.update(identifier="user_uptake", species="user_substrate",
                maximum_parameter="rates.user_capacity")
    definition["model"]["pathways"]["pathway_bound_rules"][0].update(
        species="user_substrate", pathways=["user_pathway"])
    definition["state"]["species"]["amounts"] = {"user_substrate": 0.01}
    thermo["user_substrate"] = thermo.pop("nutrient")
    path = tmp_path / "thermo.json"
    path.write_text(json.dumps(thermo))
    assembly = build_bioreactor(case, definition, path)
    result = assembly.solve(verbose=False)[-1]
    expected = 0.0001 * np.exp(2.0 / 10.0 * 6.0)
    assert result.biomass_kg_liquid0[-1] == pytest.approx(expected, rel=1e-6)






@pytest.mark.parametrize('status, values, complete', [
    ('supplied', {'nutrient': 2.}, True), ('nutrient-free', {}, True),
    ('unknown', {}, False), ('unknown', {'nutrient': 2.}, False),
    ('unspecified', {}, False),
])
def test_feed_completeness_and_strict_mode(status, values, complete):
    case, definition, _ = _load('generic_fed_batch')
    event = next(iter(definition['recipes'].values()))[0]
    event['composition'] = status
    event['concentrations_mmol_l'] = values
    case['recipe']['require_complete_inputs'] = True
    assembly = build_bioreactor(case, definition, EXAMPLES / 'generic_fed_batch/inputs/thermo.json')
    assert (assembly.input_audit()['feed_composition'] == 'COMPLETE') is complete
    if not complete:
        before = assembly.phase.mass_j.copy()
        with pytest.raises(ValueError, match='complete feed compositions'):
            assembly.solve()
        np.testing.assert_array_equal(assembly.phase.mass_j, before)


@pytest.mark.parametrize('status, values', [
    ('invalid', {}), ('supplied', {}), ('nutrient-free', {'nutrient': 1.}),
])
def test_inconsistent_feed_declarations_rejected(status, values):
    case, definition, _ = _load('generic_fed_batch')
    event = next(iter(definition['recipes'].values()))[0]
    event.update(composition=status, concentrations_mmol_l=values)
    with pytest.raises(ValueError):
        build_bioreactor(case, definition, EXAMPLES / 'generic_fed_batch/inputs/thermo.json')


def test_legacy_unknown_feed_preserves_exact_material_increment():
    case, definition, _ = _load('generic_fed_batch')
    assembly = build_bioreactor(case, definition, EXAMPLES / 'generic_fed_batch/inputs/thermo.json')
    event = deepcopy(assembly.events[0])
    event.pop('composition', None)
    initial = assembly.phase.mass_j.copy()
    _apply_material_event(assembly.phase, event, 1., 'water')
    expected = assembly.phase.mass_j.copy()
    assembly.phase.updatePhase(mass_j=initial)
    event['composition'] = 'unknown'
    _apply_material_event(assembly.phase, event, 1., 'water')
    np.testing.assert_array_equal(assembly.phase.mass_j, expected)


def test_depletion_report_does_not_label_initial_zero_as_depletion():
    from types import SimpleNamespace
    case, definition, _ = _load('generic_batch')
    assembly = build_bioreactor(case, definition, EXAMPLES / 'generic_batch/inputs/thermo.json')
    initial = assembly.phase.mass_j.copy()
    depleted = initial.copy()
    i = list(assembly.phase.name_species).index('nutrient')
    depleted[i] = 0.
    history = SimpleNamespace(time=np.array([0., 1., 2., 3.]),
                              mass_j_liquid0=np.array([depleted, initial, depleted, initial]))
    report = assembly.diagnostics((history,))
    assert report['depletions'] == [dict(species='nutrient', first_depletion_time_s=2., amount_mmol=0.)]
    np.testing.assert_array_equal(assembly.phase.mass_j, initial)
    with pytest.raises(ValueError):
        assembly.diagnostics((history,), depletion_tolerance_mmol=-1.)


def test_strict_completeness_ignores_additions_after_the_simulation():
    case, definition, _ = _load('generic_fed_batch')
    case['recipe']['require_complete_inputs'] = True
    event = next(iter(definition['recipes'].values()))[0]
    event.update(time=100., composition='unknown')
    assembly = build_bioreactor(case, definition, EXAMPLES / 'generic_fed_batch/inputs/thermo.json')
    assert assembly.input_audit()['feed_composition'] == 'COMPLETE'


def test_strict_supplied_feed_preserves_forward_solution():
    case, definition, _ = _load('generic_fed_batch')
    thermo = EXAMPLES / 'generic_fed_batch/inputs/thermo.json'
    expected = build_bioreactor(case, definition, thermo).solve(verbose=False)
    case['recipe']['require_complete_inputs'] = True
    actual = build_bioreactor(case, definition, thermo).solve(verbose=False)
    for reference, result in zip(expected, actual):
        np.testing.assert_array_equal(result.mass_j_liquid0, reference.mass_j_liquid0)
        np.testing.assert_array_equal(result.biomass_kg_liquid0, reference.biomass_kg_liquid0)


def _explicit_culture_inputs():
    case, definition, _ = _load('fed_batch_cho')
    kinetics = definition['model']['kinetics']
    kinetics['flux_basis'] = {
        'exchange': dict(amount_unit='nmol', normalization='10^6 cell', time_unit='day'),
        'growth': dict(amount_unit='1', normalization='none', time_unit='day'),
        'product': dict(amount_unit='nmol', normalization='10^6 cell', time_unit='day')}
    for key in ('cell_flux_to_mmol', 'growth_flux_to_per_day', 'cell_product_scale'):
        kinetics['constants'].pop(key, None)
    units = definition['rate_provider']['output_units']
    for reaction, output in kinetics['kinetic_outputs']['exchange_fluxes'].items():
        units[output] = '1/day' if reaction == kinetics['kinetic_outputs']['growth_reaction'] else 'nmol/(10^6 cell day)'
    case['operation']['runtime']['value'] = .025
    return case, definition


def test_explicit_flux_basis_preserves_equivalent_legacy_sources():
    from PharmaPy.Bioreactors import BioreactorUnitConverter
    case, explicit, _ = _load('fed_batch_cho')
    case['operation']['runtime']['value'] = .025
    units = BioreactorUnitConverter.from_inputs(case, explicit)
    # Legacy coefficients must describe the same basis as the current declaration.
    legacy = deepcopy(explicit)
    legacy['model']['kinetics']['flux_basis'] = {}
    legacy['model']['kinetics']['constants'].update(
        cell_flux_to_mmol=units.exchange_scale, growth_flux_to_per_day=units.growth_scale,
        cell_product_scale=units.product_scale)
    legacy['model']['kinetics']['product_mapping']['molecular_weight_g_mol'] = units.product_mass
    thermo = Path(__file__).parent / 'fixtures/fed_batch_cho/inputs/thermo.json'
    old = build_bioreactor(case, legacy, thermo).solve(verbose=False)[0]
    assembly = build_bioreactor(case, explicit, thermo)
    new = assembly.solve(verbose=False)[0]
    for name in ('mass_j_liquid0', 'viable_cells_million_liquid0', 'product_g_liquid0'):
        np.testing.assert_allclose(getattr(new, name), getattr(old, name), rtol=1e-8, atol=1e-12)
    assert assembly.mechanism.audit_conservation()['flux_basis'] == 'DECLARED'


@pytest.mark.parametrize('amount, mass_factor', [
    ('nmol', 145755.29), ('nmol-C', 22.54), ('g', 1e9),
    ('pmol', 145.75529), ('pmol-C', .02254), ('pg', .001)])
def test_product_basis_changes_only_the_declared_mass_conversion(amount, mass_factor):
    case, definition = _explicit_culture_inputs()
    thermo = Path(__file__).parent / 'fixtures/fed_batch_cho/inputs/thermo.json'
    reference = build_bioreactor(case, definition, thermo).solve(verbose=False)[0]
    definition['model']['kinetics']['flux_basis']['product']['amount_unit'] = amount
    definition['rate_provider']['output_units']['product_flux'] = f'{amount}/(10^6 cell day)'
    assembly = build_bioreactor(case, definition, thermo)
    result = assembly.solve(verbose=False)[0]
    np.testing.assert_array_equal(result.mass_j_liquid0, reference.mass_j_liquid0)
    np.testing.assert_allclose(result.product_g_liquid0, reference.product_g_liquid0 * mass_factor / 145755.29, rtol=1e-12)


@pytest.mark.parametrize('change', ['missing_mass', 'missing_content', 'wrong_provider_units', 'conflicting_scale', 'mixed_time', 'mixed_normalization'])
def test_invalid_flux_basis_is_rejected(change):
    case, definition = _explicit_culture_inputs()
    kinetics = definition['model']['kinetics']
    if change == 'missing_mass':
        kinetics['flux_basis']['product']['amount_unit'] = 'mol-C'
        kinetics['product_mapping'].pop('mass_per_cmol_g')
    elif change == 'missing_content':
        for name in ('exchange', 'product'):
            kinetics['flux_basis'][name]['normalization'] = 'gDW'
    elif change == 'wrong_provider_units':
        definition['rate_provider']['output_units']['product_flux'] = 'mol-C/(10^6 cell day)'
    elif change == 'conflicting_scale':
        kinetics['constants']['cell_flux_to_mmol'] = 1000.
    elif change == 'mixed_time':
        kinetics['flux_basis']['product']['time_unit'] = 'h'
    else:
        kinetics['flux_basis']['product']['normalization'] = 'cell'
    with pytest.raises(ValueError):
        build_bioreactor(case, definition, Path(__file__).parent / 'fixtures/fed_batch_cho/inputs/thermo.json').solve(verbose=False)


def test_rate_basis_normalization_and_growth_content():
    from PharmaPy.Bioreactors import BioreactorUnitConverter
    _rate_basis = BioreactorUnitConverter.rate_basis
    # 1 unit/(gDW h) with 0.2 gDW per million cells = 4.8 units/(million cells day).
    declaration = dict(amount_unit='mmol', normalization='gDW', time_unit='h', biomass_g_per_million_cells=.2)
    assert _rate_basis(declaration)[0] == pytest.approx(4.8)
    # 1 nmol-C/(million cells day) divided by 2 nmol-C/million cells = 0.5/day.
    declaration = dict(amount_unit='nmol-C', normalization='10^6 cell', time_unit='day', amount_per_million_cells=2.)
    assert _rate_basis(declaration, growth=True)[0] == .5
    assert _rate_basis(dict(amount_unit='1', normalization='none', time_unit='h'), growth=True)[0] == 24.
    assert _rate_basis(dict(amount_unit='mol', normalization='cell', time_unit='s'))[0] == 86400e6
