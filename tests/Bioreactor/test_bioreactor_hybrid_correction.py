"""Generic correction expansion, validation and real native hybrid trajectories.

Synthetic algebraic rates have independently evaluated outputs. Native checks
compare the configured convenience with the previous explicit expression and
exercise parameter substitution through the existing measurement callback.
"""

from copy import deepcopy

import numpy as np
import pytest

from PharmaPy.Bioreactors import simulate_measurements
from .test_bioreactor_relationships import evaluate, relation
from .test_bioreactor_estimation_handoff import configuration, callback_inputs, SAMPLE_H, HOUR_S


# At S=S_ref=1 g/L and theta=2 ln(2), the multiplier is exactly two.
THETA = 2. * np.log(2.)  # [-], independently constructed algebraic test case
CORRECTION = {
    'type': 'exponential-saturation',
    'parameters': {'theta': THETA, 'reference_concentration': 1.},  # [-], [g/L]
    'concentration_unit': 'g/L',
    'input': 1.,  # [g/L]
}


# Public-path regression and failure checks.
@pytest.mark.unit
@pytest.mark.parametrize('name,inputs,expected', [
    ('monod', dict(maximum=2., substrate=3., half_saturation=1.), 1.5),
    ('haldane', dict(maximum=1., substrate=2., half_saturation=1., inhibition_constant=4.), .5),
    ('first-order-death', dict(coefficient=.2, population=5.), 1.),
    ('additive-inhibition', dict(rate=4., inhibitors=[dict(concentration=3., constant=1.)]), 1.),
    ('yield-sum', dict(terms=[{'rate': 2., 'yield': 3.}]), 6.),
    ('capacity-limited-rate', dict(rate=3., capacity=2.), 1.2),
    ('net-production', {'production': 3., 'consumption': 2., 'yield': .5}, -.5),
    ('pirt', {'growth': .4, 'maintenance': .1, 'yield': .5}, .9),
    ('linear-transfer', dict(coefficient=2., equilibrium=1., concentration=3.), -4.),
])
def test_correction_wraps_every_relationship_without_changing_rate_sign(name, inputs, expected):
    """Double independent base outputs; retain each relationship's declared basis.

    Parameters
    ----------
    name : str
        Relationship or alias.
    inputs : dict
        Synthetic concentration [g/L], time [h] and population [gDW] quantities;
        yield factors convert the relevant input rate to the output basis.
    expected : float
        Independently calculated base output in that relationship's rate units.
    """
    expression = relation(name, **inputs)
    expression['correction'] = deepcopy(CORRECTION)
    assert evaluate(expression) == pytest.approx(2. * expected)
    expression['correction']['parameters']['theta'] = 0.  # [-], neutral correction
    assert evaluate(expression) == pytest.approx(expected)


@pytest.mark.unit
@pytest.mark.parametrize('field,value,message', [
    ('type', 'unknown', 'correction type'),
    ('concentration_unit', 'kg/s', 'concentration_unit'),
    ('concentration_unit', None, 'concentration_unit'),
    ('input', '', 'correction input'),
    ('input', True, 'correction input'),
    ('input', float('nan'), 'correction input'),
    ('extra', 1., 'correction requires'),
])
def test_correction_rejects_malformed_declarations(field, value, message):
    """Reject unsupported fields and malformed concentration declarations.

    Parameters
    ----------
    field : str
        Correction field to invalidate.
    value : object
        Deliberately invalid declaration; scientific inputs otherwise use g/L.
    message : str
        Required actionable diagnostic fragment.
    """
    expression = relation('first-order-death', coefficient=1., population=1.)
    expression['correction'] = deepcopy(CORRECTION)
    expression['correction'][field] = value
    with pytest.raises(ValueError, match=message):
        evaluate(expression)


@pytest.mark.unit
@pytest.mark.parametrize('parameter,value', [
    ('theta', True), ('theta', float('nan')), ('theta', float('inf')),
    ('theta', {'parameter': 'absent'}), ('theta', [1.]),
    ('reference_concentration', 0.), ('reference_concentration', -1.),
    ('reference_concentration', True), ('reference_concentration', float('inf')),
])
def test_correction_validates_scalar_parameters(parameter, value):
    """Reject invalid theta [-] and nonpositive/nonfinite concentration scales.

    Parameters
    ----------
    parameter : str
        Correction parameter name.
    value : object
        Invalid value; theta [-], reference concentration [g/L].
    """
    expression = relation('first-order-death', coefficient=1., population=1.)
    expression['correction'] = deepcopy(CORRECTION)
    expression['correction']['parameters'][parameter] = value
    with pytest.raises(ValueError, match='correction'):
        evaluate(expression)


@pytest.mark.unit
def test_correction_references_keep_parameter_dependence_and_nested_composition():
    """Compose corrections; changing theta [-] changes the actual graph output."""
    expression = relation('first-order-death', coefficient=1., population=1.)
    expression['correction'] = deepcopy(CORRECTION)
    expression['correction']['parameters']['theta'] = {'parameter': 'theta'}
    expression['correction']['parameters']['reference_concentration'] = {'parameter': 'scale'}
    nested = {'op': 'multiply', 'args': [expression, 3.], 'correction': deepcopy(CORRECTION)}
    assert evaluate(nested, {'theta': THETA, 'scale': 1.}) == pytest.approx(12.)
    assert evaluate(nested, {'theta': -THETA, 'scale': 1.}) == pytest.approx(3.)


@pytest.mark.integration
@pytest.mark.parametrize('theta', [0., .2, -.2])
def test_native_convenience_matches_previous_explicit_hybrid_and_preserves_inputs(theta):
    """Compare full accepted trajectories through the public estimator callback.

    Parameters
    ----------
    theta : float
        Log correction [-], interior to the example's declared validity domain.
    """
    model = configuration(True)
    assert 'correction' in next(rule for rule in model['model']['kinetics']['rule_graph']
                                if rule['identifier'] == 'mu')['expression']
    original = deepcopy(model)
    explicit = deepcopy(model)
    graph = explicit['model']['kinetics']['rule_graph']
    next(rule for rule in graph if rule['identifier'] == 'mu')['expression'] = {
        'op': 'multiply', 'args': [{'ref': 'mu_mechanistic'},
        {'op': 'exp', 'args': [{'op': 'multiply', 'args': [{'ref': 'theta'}, {'ref': 'feature'}]}]}]}
    times = SAMPLE_H * HOUR_S  # [s], existing qualified transient sampling grid
    actual = simulate_measurements([theta], times, **callback_inputs(model))
    reference = simulate_measurements([theta], times, **callback_inputs(explicit))
    np.testing.assert_allclose(actual, reference, rtol=1e-10, atol=1e-12)
    # Budget is below existing integration tolerance: same physical equations.
    assert model == original


@pytest.mark.unit
@pytest.mark.parametrize('correction', [None, {}, {'type': 'exponential-saturation'}])
def test_correction_requires_complete_configuration(correction):
    """Require explicit correction fields rather than guessing scientific inputs.

    Parameters
    ----------
    correction : object
        Incomplete correction declaration.
    """
    expression = relation('first-order-death', coefficient=1., population=1.)
    expression['correction'] = correction
    with pytest.raises(ValueError, match='correction requires'):
        evaluate(expression)


@pytest.mark.unit
def test_correction_concentration_rescaling_preserves_dimensionless_multiplier():
    """Convert both concentrations from g/L to mg/L with the exact factor 1000."""
    expression = relation('first-order-death', coefficient=1., population=1.)
    expression['correction'] = deepcopy(CORRECTION)
    expected = evaluate(expression)
    expression['correction']['input'] = 1000.  # [mg/L], 1 g/L
    expression['correction']['parameters']['reference_concentration'] = 1000.  # [mg/L]
    expression['correction']['concentration_unit'] = 'mg/L'
    assert evaluate(expression) == pytest.approx(expected)
    expression['correction']['input'] = 0.  # [mg/L], zero feature implies no correction
    assert evaluate(expression) == pytest.approx(1.)


@pytest.mark.unit
def test_correction_discovery_and_invalid_rule_reference():
    """Expose the optional syntax and retain existing graph reference validation."""
    from PharmaPy.Bioreactors import relationship_catalog
    catalog = relationship_catalog()
    assert all('exponential-saturation' in entry['corrections'] for entry in catalog.values())
    template = catalog['saturating-uptake']['corrections']['exponential-saturation']['template']
    assert template['parameters']['theta'] == 0.
    template['parameters']['theta'] = 2.  # [-], mutate only the caller's copy
    assert relationship_catalog()['saturating-uptake']['corrections']['exponential-saturation']['template']['parameters']['theta'] == 0.
    expression = relation('monod', maximum=1., substrate=1., half_saturation=1.)
    expression['correction'] = deepcopy(CORRECTION)
    expression['correction']['input'] = 'missing_concentration_rule'
    with pytest.raises(ValueError, match='reference|rule'):
        evaluate(expression)
