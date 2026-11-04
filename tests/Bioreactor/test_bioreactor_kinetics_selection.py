"""Familiar kinetic names preserve graphs, native trajectories and estimation.

Reuse the qualified Doyle fixtures and independent mass balances. No new
solver, model defaults or dependency is introduced by selecting an alias.
"""

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

import PharmaPy.Bioreactors as bioreactors
from PharmaPy.Bioreactors.input_format import normalize_forward_inputs, split_forward_inputs
from .test_bioreactor_relationships import evaluate, relation
from .test_bioreactor_named_kinetics import named_configuration
from .test_bioreactor_estimation_handoff import callback_inputs, SAMPLE_H, HOUR_S, CONCENTRATION_ATOL

TEMPLATE = Path(__file__).parents[2] / 'doc/online_docs/bioreactors/templates/kinetics-selection'


# Public-path regression and failure checks.
@pytest.mark.unit
@pytest.mark.parametrize('alias,canonical,inputs,expected', [
    ('monod', 'saturating-uptake', dict(maximum=2., substrate=3., half_saturation=1.), 1.5),
    ('haldane', 'substrate-inhibited-growth', dict(maximum=1., substrate=2., half_saturation=1., inhibition_constant=4.), .5),
    ('andrews', 'substrate-inhibited-growth', dict(maximum=1., substrate=2., half_saturation=1., inhibition_constant=4.), .5),
    ('pirt', 'maintenance-uptake', dict(growth=.2, yield_=.5, maintenance=.1), .5),
])
def test_aliases_match_independent_equations(alias, canonical, inputs, expected):
    """Check independently evaluated small rational examples.

    Parameters
    ----------
    alias, canonical : str
        Alternate and established names for the same equation.
    inputs : dict
        Synthetic rates [1/h] and concentrations [g/L]. Pirt uses growth
        [1/h], yield [gDW/gS] and maintenance [gS/(gDW h)].
    expected : float
        Hand-calculated output [1/h], or [gS/(gDW h)] for Pirt.
    """
    if 'yield_' in inputs:
        inputs = dict(inputs, **{'yield': inputs['yield_']})
        del inputs['yield_']
    assert evaluate(relation(alias, **inputs)) == pytest.approx(expected)
    assert evaluate(relation(alias, **inputs)) == evaluate(relation(canonical, **inputs))


@pytest.mark.unit
def test_catalog_is_complete_json_serializable_and_independent():
    """Keep discovery complete and prevent caller mutation of the registry."""
    catalog = bioreactors.relationship_catalog()
    assert len(catalog) == 9  # [-], all nine established forms remain discoverable
    assert catalog['saturating-uptake']['aliases'] == ['monod']
    assert catalog['substrate-inhibited-growth']['positive_scalar_inputs'] == ['half_saturation', 'inhibition_constant']
    assert catalog['first-order-death']['nonnegative_scalar_inputs'] == ['coefficient']
    assert all(entry['equation'] and entry['units'] for entry in catalog.values())
    assert json.loads(json.dumps(catalog)) == catalog
    catalog['saturating-uptake']['expression_template']['inputs']['maximum'] = 'changed'
    assert bioreactors.relationship_catalog()['saturating-uptake']['expression_template']['inputs']['maximum'] is None


@pytest.mark.unit
def test_unknown_name_lists_choices_and_missing_slots_are_actionable():
    """Report available forms and the missing inhibition concentration [g/L]."""
    with pytest.raises(ValueError, match='choose one of:.*monod'):
        evaluate(relation('monood', maximum=1., substrate=1., half_saturation=1.))
    with pytest.raises(ValueError, match="missing.*inhibition_constant"):
        evaluate(relation('haldane', maximum=1., substrate=1., half_saturation=1.))


@pytest.mark.integration
@pytest.mark.parametrize('alias,inhibited', [('monod', False), ('haldane', True), ('andrews', True)])
@pytest.mark.parametrize('legacy', [False, True])
def test_aliases_preserve_native_predictions_and_inputs(alias, inhibited, legacy):
    """Check both input schemas through native observations.

    Parameters
    ----------
    alias : str
        Familiar growth-form name.
    inhibited : bool
        Include the synthetic substrate-inhibition term.
    legacy : bool
        Exercise the separate case/mechanism API instead of combined inputs.
    """
    canonical = named_configuration(inhibited)
    selected = deepcopy(canonical)
    graph = selected['model']['kinetics']['rule_graph']
    next(rule for rule in graph if rule['identifier'] == 'mu_mechanistic')['expression']['relationship'] = alias
    original = deepcopy(selected)
    assert normalize_forward_inputs(selected) == normalize_forward_inputs(canonical)
    arguments = callback_inputs(selected, ('rates.mu_max',))
    if legacy:
        arguments['case'], arguments['definition'] = split_forward_inputs(selected)
    times = SAMPLE_H * HOUR_S  # [s], qualified 8 h fixture
    actual = bioreactors.simulate_measurements([.2], times, **arguments)  # [mmol/L, gDW/L]
    expected = bioreactors.simulate_measurements([.2], times,
        **callback_inputs(canonical, ('rates.mu_max',)))  # [mmol/L, gDW/L]
    np.testing.assert_allclose(actual, expected, rtol=0., atol=CONCENTRATION_ATOL)
    assert selected == original


@pytest.mark.integration
@pytest.mark.parametrize('inhibited', [False, True])
def test_runnable_selection_template_matches_qualified_native_model(inhibited):
    """Check the documented template against qualified trajectories.

    Parameters
    ----------
    inhibited : bool
        Supply the additional Ki [g/L] and select Haldane growth.
    """
    model = json.loads((TEMPLATE/'mechanism.json').read_text())
    if inhibited:
        model['parameters']['affinities']['Ki'] = 4.  # [g/L], existing synthetic peak at 2 g/L
        rule = next(rule for rule in model['model']['kinetics']['rule_graph'] if rule['identifier'] == 'mu')
        rule['expression']['relationship'] = 'haldane'
        rule['expression']['inputs']['inhibition_constant'] = {'parameter': 'affinities.Ki'}
    before = deepcopy(model)
    kwargs = callback_inputs(model, ('rates.mu_max',))
    kwargs['thermo_path'] = TEMPLATE/'thermo.json'
    times = SAMPLE_H * HOUR_S  # [s], existing independent-balance qualification window
    actual = bioreactors.simulate_measurements([.2], times, **kwargs)  # [mmol/L, gDW/L]
    expected = bioreactors.simulate_measurements([.2], times,
        **callback_inputs(named_configuration(inhibited), ('rates.mu_max',)))  # [mmol/L, gDW/L]
    np.testing.assert_allclose(actual, expected, rtol=0., atol=CONCENTRATION_ATOL)
    assert model == before
    if not inhibited:
        np.testing.assert_allclose(actual, np.tile([1., 4.5], (len(times), 1)),
                                   rtol=0., atol=CONCENTRATION_ATOL)


@pytest.mark.unit
@pytest.mark.parametrize('value', [0., -1., float('nan'), float('inf'), True, [], {}])
def test_alias_keeps_positive_parameter_validation(value):
    """Preserve rejection of invalid affinity bindings.

    Parameters
    ----------
    value : object
        Invalid half-saturation concentration [g/L] or malformed binding.
    """
    with pytest.raises(ValueError, match='finite positive scalar'):
        evaluate(relation('monod', maximum=1., substrate=1., half_saturation=value))


@pytest.mark.unit
def test_nested_aliases_keep_parameter_references_and_input_order():
    """Combine Monod growth [1/h] with Pirt uptake [gS/(gDW h)]."""
    inputs = dict(growth=relation('monod', maximum={'parameter': 'maximum'},
        substrate=3., half_saturation=1.), maintenance=.1, **{'yield': .5})
    original = deepcopy(inputs)
    assert evaluate(relation('pirt', **inputs), {'maximum': .2}) == pytest.approx(.4)
    assert inputs == original


@pytest.mark.unit
@pytest.mark.parametrize('alias', ['monod', 'haldane', 'andrews'])
@pytest.mark.parametrize('factor', [1e-3, 1., 1e3])
@pytest.mark.parametrize('depleted', [False, True])
def test_alias_concentration_rescaling_and_depletion(alias, factor, depleted):
    """Preserve half-maximum reference cases under concentration rescaling.

    Parameters
    ----------
    alias : str
        Selected growth form.
    factor : float
        Shared concentration rescaling [-]; tests three adjacent SI prefixes.
    depleted : bool
        Set substrate to zero to check the depletion boundary.

    Notes
    -----
    Monod at S=Ks is half maximum. Haldane at S=2, Ks=1, Ki=4 has
    S/(Ks+S+S**2/Ki)=1/2. Both vanish at zero substrate.
    """
    substrate = 1. if alias == 'monod' else 2.  # [g/L], independent reference above
    inputs = dict(maximum=1., substrate=(0. if depleted else substrate)*factor,
                  half_saturation=factor)  # [1/h, rescaled concentration, rescaled concentration]
    if alias != 'monod':
        inputs['inhibition_constant'] = 4.*factor  # [rescaled concentration], peak at S=2*factor
    expected = 0. if depleted else .5  # [1/h], analytic references above
    assert evaluate(relation(alias, **inputs)) == pytest.approx(expected)


@pytest.mark.unit
def test_malformed_python_input_keys_raise_actionable_value_error():
    """Retain validation errors for malformed mappings supplied through Python."""
    expression = {'relationship': 'monod', 'inputs': {None: 1., 2: 1.}}
    with pytest.raises(ValueError, match='requires inputs.*missing.*half_saturation'):
        evaluate(expression)
