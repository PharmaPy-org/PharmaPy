"""Declared rate units, biological normalizers and native source handoff.

Independent closed-form values test the finite equation catalog. Synthetic
1000 g/mol pools let the public reactor path expose time-unit mistakes exactly.
"""

from copy import deepcopy
import numpy as np
import pytest

from PharmaPy.Bioreactors import build_bioreactor
from PharmaPy.Bioreactors.kinetic_units import unit_signature
from PharmaPy.Bioreactors.relationships import expand_relationships
from tests.Bioreactor.test_bioreactor_relationships import evaluate, relation
from tests.Bioreactor.test_bioreactor_adaptive_inventory import configured_case, EXAMPLE


# Explicit dimensions are also the provenance of these algebraic test fixtures.
CASES = [
    ('monod', {'maximum': 2., 'substrate': 2., 'half_saturation': .002},
     {'maximum': '1/h', 'substrate': 'mmol/L', 'half_saturation': 'mol/L', 'output': '1/h'}, 1.),  # [1/h]; half saturation
    ('haldane', {'maximum': 3., 'substrate': 2., 'half_saturation': 2., 'inhibition_constant': 2.},
     {'maximum': '1/h', 'substrate': 'g/L', 'half_saturation': 'g/L', 'inhibition_constant': 'g/L', 'output': '1/h'}, 1.),  # [1/h]; three equal denominator contributions
    ('first-order-death', {'coefficient': .5, 'population': 2.},
     {'coefficient': '1/h', 'population': 'gDW/L', 'output': 'gDW/(L h)'}, 1.),  # [gDW/(L h)]
    ('additive-inhibition', {'rate': 6., 'inhibitors': [{'concentration': 2., 'constant': 2.}, {'concentration': 1., 'constant': 1.}]},
     {'rate': '1/h', 'inhibitors': [{'concentration': 'g/L', 'constant': 'g/L'}, {'concentration': 'mmol/L', 'constant': 'mmol/L'}], 'output': '1/h'}, 2.),  # [1/h]; divisor 1+1+1
    ('yield-sum', {'terms': [{'rate': 1., 'yield': 2.}, {'rate': 2., 'yield': .5}]},
     {'terms': [{'rate': 'g/(L h)', 'yield': 'gDW/g'}, {'rate': 'g/(L h)', 'yield': 'gDW/g'}], 'output': 'gDW/(L h)'}, 3.),  # [gDW/(L h)]; 2+1
    ('capacity-limited-rate', {'rate': 2., 'capacity': 2.},
     {'rate': 'g/(gDW h)', 'capacity': 'g/(gDW h)', 'output': 'g/(gDW h)'}, 1.),  # [g/(gDW h)]; half capacity
    ('net-production', {'yield': .5, 'production': 4., 'consumption': 1.},
     {'yield': 'gDW/g', 'production': 'g/(L h)', 'consumption': 'gDW/(L h)', 'output': 'gDW/(L h)'}, 1.),  # [gDW/(L h)]; 2-1
    ('maintenance-uptake', {'growth': .4, 'yield': .5, 'maintenance': 2.4},
     {'growth': '1/h', 'yield': 'gDW/g', 'maintenance': 'g/(gDW day)', 'output': 'g/(gDW h)'}, .9),  # [g/(gDW h)]; 0.4/0.5+2.4/24
    ('linear-transfer', {'coefficient': 2., 'equilibrium': 1., 'concentration': 3.},
     {'coefficient': '1/h', 'equilibrium': 'mmol/L', 'concentration': 'mmol/L', 'output': 'mmol/(L h)'}, -4.),  # [mmol/(L h)]; retains negative transfer
]


@pytest.mark.unit
@pytest.mark.parametrize('name,inputs,units,expected', CASES)
def test_every_named_form_has_independent_dimensional_oracle(name, inputs, units, expected):
    expression = relation(name, **inputs)
    expression['units'] = units
    assert evaluate(expression) == pytest.approx(expected, rel=1e-14)


@pytest.mark.unit
def test_dry_wet_and_cell_normalizations_cannot_be_interchanged():
    name, inputs, units, _ = CASES[-2]
    expression = relation(name, **inputs)
    expression['units'] = deepcopy(units)
    for basis in ('gWet', 'cell', 'g'):
        expression['units']['yield'] = basis+'/g'
        with pytest.raises(ValueError, match='biological bases'):
            evaluate(expression)


@pytest.mark.unit
def test_known_native_concentration_binding_cannot_be_relabeled():
    expression = relation('monod', maximum=1., substrate={'concentration': 'biomass'}, half_saturation=1.)
    expression['units'] = {'maximum': '1/h', 'substrate': 'g/L', 'half_saturation': 'g/L', 'output': '1/h'}
    with pytest.raises(ValueError, match='binding returns mmol/L'):
        evaluate(expression)


@pytest.mark.unit
@pytest.mark.parametrize('unit', ['g/', 'kg/(L hour)', 'g/(gUnknown h)', 'g//L', ''])
def test_unknown_or_incomplete_units_fail_explicitly(unit):
    with pytest.raises(ValueError, match='unit'):
        unit_signature(unit)


@pytest.mark.integration
@pytest.mark.parametrize('decay', [1., 4.])  # [1/s]; distinct rate bindings, not fitted values
def test_typed_output_reaches_native_source_in_correct_time_basis(decay):
    document = configured_case('scipy')
    document['parameters']['rates']['decay'] = decay
    expression = relation('first-order-death', coefficient={'parameter': 'rates.decay'},
        population={'op': 'multiply', 'args': [1., {'concentration': 'biomass'}]})  # [g/L]; 1000 g/mol * 1e-3 mol/mmol
    expression['units'] = {'coefficient': '1/s', 'population': 'g/L', 'output': 'g/(L h)'}
    document['model']['kinetics']['rule_graph'][0]['expression'] = expression
    negative = relation('net-production', production=0., consumption={'ref': 'loss'}, **{'yield': 1.})
    negative['units'] = {'yield': '-', 'production': 'g/(L h)', 'consumption': 'g/(L h)', 'output': 'g/(L h)'}
    document['model']['kinetics']['rule_graph'][1]['expression'] = negative
    document['model']['kinetics']['reported_rates'] = {'raw_loss': {'rule': 'loss', 'unit': 'g/(L h)', 'basis': 'liquid volume'}}
    document['model']['kinetics']['rate_unit'] = 'g/(L h)'
    assembly = build_bioreactor(document, thermo_path=EXAMPLE/'thermo.json')
    history, = assembly.solve()
    np.testing.assert_allclose(history.mass_j_liquid0[:, 0], .001*np.exp(-decay*history.time), rtol=1e-6, atol=1e-11)  # [kg]
    np.testing.assert_allclose(np.asarray(history.raw_loss_liquid0).reshape(-1), 3600.*decay*np.exp(-decay*history.time), rtol=1e-6, atol=1e-5)  # [g/(L h)]; reported/reference rule retains its original units
    assert assembly.input_audit()['construction']['kinetic_units']['declared_relationship_contracts'] == 2
    incompatible = deepcopy(document)
    incompatible['model']['kinetics'].pop('reported_rates')
    incompatible['model']['kinetics']['rate_unit'] = 'kg/(m3 s)'
    with pytest.raises(ValueError, match='output unit must match rate_unit'):
        build_bioreactor(incompatible, thermo_path=EXAMPLE/'thermo.json')
    expression['units'] = {'coefficient': '1/s', 'population': 'cell', 'output': 'cell/h'}
    with pytest.raises(ValueError, match='incompatible biological or physical basis'):
        build_bioreactor(document, thermo_path=EXAMPLE/'thermo.json')


@pytest.mark.unit
def test_legacy_explicit_graph_remains_unchanged():
    model = {'rule_graph': [{'identifier': 'rate', 'expression': {'op': 'multiply', 'args': [2., {'parameter': 'p'}]}}]}
    assert expand_relationships(model, {'p': 3.}) == model
