"""Named kinetics through real construction, observations and native estimation.

Small synthetic fixtures qualify mathematical forms, not organism parameters.
The existing Doyle hybrid supplies explicit mass conversions and domain policy.
"""

from copy import deepcopy
import json

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from PharmaPy.Bioreactors import build_bioreactor, simulate_measurements
from PharmaPy.Bioreactors.observations import predict_measurements
from PharmaPy.Bioreactors.input_format import split_forward_inputs
from PharmaPy.ParamEstim import Experiment, Measurement, ParameterEstimation
from .test_bioreactor_estimation_handoff import (
    configuration, callback_inputs, EXAMPLE, HOUR_S, SAMPLE_H,
    CONCENTRATION_ATOL, FD_STEP,
)
from .test_bioreactor_relationships import evaluate, relation


# Independent synthetic inhibition/death cases; no empirical coefficients.
INHIBITION = 4.  # [g/L], places the inhibition peak at sqrt(Ks*Ki)=2 g/L
DEATH = .03  # [1/h], gives a resolvable loss over the existing 8 h fixture
FIT_ATOL = 1e-5  # [1/h], synthetic recovery target: 0.005% of the textbook maximum
REFERENCE_RTOL = 1e-11  # [-], tighter independent integration than native 1e-9
REFERENCE_ATOL = 1e-12  # [g/L], independent concentration integration tolerance


# Real fixture construction and independent qualification helpers.
def named_configuration(inhibited=False):
    """Select named growth in the existing mass-basis hybrid configuration.

    Parameters
    ----------
    inhibited : bool, optional
        Select substrate inhibition instead of Monod saturation.

    Returns
    -------
    dict
        Private schema-3 declaration; concentrations [g/L], rates [1/h] before
        multiplication by biomass, original hybrid correction fixed at zero.
    """
    model = configuration(True)
    inputs = dict(maximum={'parameter': 'rates.mu_max'},
                  substrate={'ref': 'substrate_g_l'},
                  half_saturation={'parameter': 'affinities.Ks'})
    family = 'saturating-uptake'
    if inhibited:
        family = 'substrate-inhibited-growth'
        model['parameters']['affinities']['Ki'] = INHIBITION  # [g/L]
        inputs['inhibition_constant'] = {'parameter': 'affinities.Ki'}
    graph = model['model']['kinetics']['rule_graph']
    next(rule for rule in graph if rule['identifier'] == 'mu_mechanistic')[
        'expression'] = relation(family, **inputs)
    return model


def inhibited_balances(time_h, concentrations, inhibition):
    """Evaluate independent concentration balances for the synthetic culture.

    Parameters
    ----------
    time_h : float
        Time [h]; autonomous case.
    concentrations : ndarray
        Biomass then substrate [g/L].
    inhibition : bool
        Include the synthetic Ki=4 g/L inhibition term.

    Returns
    -------
    list of float
        Biomass and substrate derivatives [g/(L h)].

    Notes
    -----
    Doyle supplies mu_max=0.2/h, Ks=1 g/L, Y=0.5 g/g, feed=10 g/L,
    D=0.1/h. Ki is a synthetic design assumption, not a textbook value.
    """
    biomass, substrate = concentrations  # [g/L]
    denominator = 1. + substrate  # [g/L], Ks + substrate
    if inhibition:
        denominator += substrate ** 2 / INHIBITION  # [g/L]
    growth = .2 * substrate / denominator * biomass  # [gDW/(L h)]
    return [growth - .1 * biomass, .1 * (10. - substrate) - growth / .5]


# Public-path regressions, independent balances, and failure qualification.
@pytest.mark.unit
@pytest.mark.parametrize('substrate,expected', [(-1e-12, 0.), (0., 0.),
                                              (1., 4./9.), (2., .5), (4., 4./9.)])
def test_substrate_inhibition_known_values_and_peak(substrate, expected):
    """Check equal rates on either side of the analytically known peak.

    Parameters
    ----------
    substrate : float
        Synthetic substrate [g/L]; -1e-12 exercises the documented inherited
        roundoff extension, not a valid negative-inventory biological case.
    expected : float
        Independent specific rate [1/h], for maximum=1/h, Ks=1 g/L, Ki=4 g/L.
    """
    expression = relation('substrate-inhibited-growth', maximum=1., substrate=substrate,
                          half_saturation=1., inhibition_constant=INHIBITION)
    assert evaluate(expression) == pytest.approx(expected)


@pytest.mark.unit
@pytest.mark.parametrize('population', [0., 2., 7.])
def test_first_order_death_is_positive_loss_magnitude(population):
    """Check loss per hour without implicitly activating a population balance.

    Parameters
    ----------
    population : float
        Synthetic viable biomass concentration [gDW/L].
    """
    loss = evaluate(relation('first-order-death', coefficient=DEATH,
                             population=population))  # [gDW/(L h)]
    assert loss == pytest.approx(DEATH * population)


@pytest.mark.unit
@pytest.mark.parametrize('family,inputs,slot', [
    ('substrate-inhibited-growth', dict(maximum=1., substrate=1., half_saturation=1.,
                                       inhibition_constant=INHIBITION), 'inhibition_constant'),
    ('first-order-death', dict(coefficient=DEATH, population=2.), 'coefficient'),
])
@pytest.mark.parametrize('bad', [-1., float('nan'), float('inf'), True,
                                [1.], {'parameter': 'absent'}])
def test_named_coefficients_reject_invalid_trials(family, inputs, slot, bad):
    """Reject invalid scalar bindings before any native solve.

    Parameters
    ----------
    family : str
        Canonical relationship identifier.
    inputs : dict
        Slots in the units documented for the family.
    slot : str
        Constrained coefficient to invalidate.
    bad : object
        Invalid scalar or unresolved parameter reference.
    """
    inputs = dict(inputs, **{slot: bad})
    with pytest.raises(ValueError, match='finite|unknown relationship parameter'):
        evaluate(relation(family, **inputs))


@pytest.mark.integration
@pytest.mark.parametrize('inhibited', [False, True])
@pytest.mark.parametrize('legacy', [False, True])
def test_named_and_explicit_public_builder_paths_match(inhibited, legacy, tmp_path):
    """Compare complete native trajectories with unequal formal species weights.

    Parameters
    ----------
    inhibited : bool
        Select Monod or Haldane–Andrews growth.
    legacy : bool
        Exercise separate legacy inputs instead of combined schema 3.
    tmp_path : pathlib.Path
        Destination for a private thermodynamic fixture.
    """
    named = named_configuration(inhibited)
    thermo = json.loads((EXAMPLE / 'inputs/thermo.json').read_text())
    thermo['substrate']['mw'] /= 2.  # [g/mol], exposes a hidden molar/mass substitution
    thermo['biomass']['mw'] *= 2.  # [g/mol], independently asymmetric biomass basis
    constants = named['model']['kinetics']['constants']
    for species in ('substrate', 'biomass'):
        constants[species + '_g_per_mmol'] = thermo[species]['mw'] / 1000.  # [g/mmol]
    path = tmp_path / 'thermo.json'
    path.write_text(json.dumps(thermo))
    explicit = configuration(True)
    explicit['model']['kinetics']['constants'] = deepcopy(constants)
    # Independent original rational expression; do not expand the named model
    # to generate its own expected graph. Concentrations are explicitly [g/L].
    explicit_rule = next(rule for rule in explicit['model']['kinetics']['rule_graph']
                         if rule['identifier'] == 'mu_mechanistic')
    explicit_rule['expression'] = {'op': 'divide', 'args': [
        {'op': 'multiply', 'args': [{'parameter': 'rates.mu_max'}, {'ref': 'substrate_g_l'}]},
        {'op': 'add', 'args': [{'parameter': 'affinities.Ks'}, {'ref': 'substrate_g_l'}]}]}
    if inhibited:
        explicit['parameters']['affinities']['Ki'] = INHIBITION  # [g/L]
        graph = explicit['model']['kinetics']['rule_graph']
        expression = next(rule for rule in graph if rule['identifier'] == 'mu_mechanistic')['expression']
        expression['args'][1]['args'].append({'op': 'divide', 'args': [
            {'op': 'multiply', 'args': [{'ref': 'substrate_g_l'}, {'ref': 'substrate_g_l'}]},
            {'parameter': 'affinities.Ki'}]})
    original = deepcopy(named)
    times_s = SAMPLE_H * HOUR_S  # [s]
    kwargs = callback_inputs(named, ('rates.mu_max',))
    kwargs['thermo_path'] = path
    if legacy:
        kwargs['case'], kwargs['definition'] = split_forward_inputs(named)
    prediction = simulate_measurements([.2], times_s, **kwargs)  # [mmol/L, gDW/L]
    reference_kwargs = dict(kwargs, case=explicit)
    reference_kwargs.pop('definition', None)
    reference = simulate_measurements([.2], times_s, **reference_kwargs)  # [mmol/L, gDW/L]
    np.testing.assert_allclose(prediction, reference, rtol=0., atol=CONCENTRATION_ATOL)
    assert named == original
    if inhibited:
        assert prediction[-1, 0] != pytest.approx(prediction[0, 0])


@pytest.mark.integration
@pytest.mark.parametrize('inhibited', [False, True])
@pytest.mark.parametrize('use_alias', [False, True])
def test_named_growth_fits_native_estimator_against_independent_balances(inhibited, use_alias):
    """Recover only maximum growth, keeping affinity, yield and correction fixed.

    Parameters
    ----------
    inhibited : bool
        Use Monod or Haldane–Andrews through the same native callback.
    use_alias : bool
        Select the familiar name rather than its existing canonical identifier.
    """
    model = named_configuration(inhibited)
    if use_alias:
        rule = next(rule for rule in model['model']['kinetics']['rule_graph']
                    if rule['identifier'] == 'mu_mechanistic')
        rule['expression']['relationship'] = 'haldane' if inhibited else 'monod'
    before = deepcopy(model)
    times_s = SAMPLE_H * HOUR_S  # [s]
    reference = solve_ivp(inhibited_balances, (0., SAMPLE_H[-1]), [4.5, 1.],
        args=(inhibited,), t_eval=SAMPLE_H, method='DOP853',
        rtol=REFERENCE_RTOL, atol=REFERENCE_ATOL)
    assert reference.success, reference.message
    kwargs = callback_inputs(model, ('rates.mu_max',))
    kwargs['declaration']['fields']['S']['unit'] = 'g/L'
    scale = .001  # [g/L], numerical residual scale, not empirical assay uncertainty
    experiment = Experiment({
        'X': Measurement('X', times_s[1::2], reference.y[0, 1::2], uncertainty=scale, units='g/L'),
        'S': Measurement('S', times_s, reference.y[1], uncertainty=scale, units='g/L')}, kwargs=kwargs)
    seed = np.array([.15])  # [1/h], synthetic recovery seed 25% below the truth
    estimator = ParameterEstimation(simulate_measurements, seed, {'growth': experiment},
        output_names=('S', 'X'), name_params=['rates.mu_max'], dx_finitediff=FD_STEP)
    # This physical interval contains the textbook truth and limits the recovery exercise.
    fitted, _, info = estimator.optimize_fn(method='least_squares', bounds=([.1], [.3]), verbose=False)
    assert info['success']
    np.testing.assert_allclose(fitted, [.2], rtol=0., atol=FIT_ATOL)
    np.testing.assert_allclose(simulate_measurements(fitted, times_s, **kwargs),
                               reference.y[::-1].T, rtol=0., atol=CONCENTRATION_ATOL)
    assert model == before


@pytest.mark.unit
def test_existing_pirt_and_luedeking_piret_compositions_use_gross_growth():
    """Verify existing families suffice without adding synonymous catalog entries.

    Returns
    -------
    None
        Checks specific uptake [gS/(gDW h)] and product [gP/(gDW h)].

    Notes
    -----
    Synthetic mu=0.2/h, Y=0.5 gDW/gS, m=0.1 gS/(gDW h),
    alpha=0.3 gP/gDW, beta=0.02 gP/(gDW h). Gross synthesis remains
    positive at chemostat steady state even though net dX/dt=0.
    """
    uptake = relation('maintenance-uptake', growth=.2, maintenance=.1, **{'yield': .5})
    assert evaluate(uptake) == pytest.approx(.5)  # [gS/(gDW h)]
    product = relation('yield-sum', terms=[{'rate': .2, 'yield': .3},
                                          {'rate': .02, 'yield': 1.}])
    assert evaluate(product) == pytest.approx(.08)  # [gP/(gDW h)]
    uptake['inputs']['maintenance'] = 0.  # [gS/(gDW h)], growth-only demand
    assert evaluate(uptake) == pytest.approx(.4)


@pytest.mark.integration
@pytest.mark.parametrize('invalid', ['affinity', 'units', 'domain', 'slots'])
def test_named_public_path_keeps_existing_validation(invalid):
    """Reject invalid coefficients, dimensions, supported domain and declarations.

    Parameters
    ----------
    invalid : str
        Select a distinct invalid public input, without replacing collaborators.
    """
    model = named_configuration(True)
    if invalid == 'affinity':
        model['parameters']['affinities']['Ki'] = 0.  # [g/L], forbidden zero denominator scale
        message = 'inhibition_constant requires a finite positive'
    elif invalid == 'units':
        model['model']['kinetics']['rate_unit'] = 'g/L'
        message = 'unit|convert|Unsupported'
    elif invalid == 'domain':
        model['rate_provider']['validity_domain']['domain_substrate'] = [0., .5]  # [g/L], excludes initial 1 g/L
        message = 'outside validity domain'
    else:
        graph = model['model']['kinetics']['rule_graph']
        next(rule for rule in graph if rule['identifier'] == 'mu_mechanistic')[
            'expression']['inputs']['unexpected'] = 1.  # [-], conflicting unsupported declaration
        message = 'requires inputs'
    with pytest.raises(ValueError, match=message):
        simulate_measurements([.2], [0.], **callback_inputs(model, ('rates.mu_max',)))


@pytest.mark.integration
def test_death_recovery_and_forward_reconstruction_use_same_named_graph():
    """Fit one identifiable loss coefficient to an independent exponential.

    Returns
    -------
    None
        Assertions qualify units, accepted parameters, fixed inputs and replay.
    """
    model = named_configuration()
    model['operation']['mode'] = 'batch'
    model['recipes']['continuous'] = []
    model['parameters']['rates']['death'] = DEATH  # [1/h]
    model['model']['kinetics']['rule_graph'] = [
        {'identifier': 'loss', 'expression': relation('first-order-death',
            coefficient={'parameter': 'rates.death'}, population={'op': 'multiply',
                'args': [{'concentration': 'biomass'}, {'constant': 'biomass_g_per_mmol'}]})},
        {'identifier': 'source', 'expression': {'op': 'negate', 'args': [{'ref': 'loss'}]}}]
    model['model']['kinetics']['species_mass_rates'] = {'biomass': 'source'}
    model['rate_provider'] = None
    before = deepcopy(model)
    times_s = SAMPLE_H * HOUR_S  # [s]
    truth = 4.5 * np.exp(-DEATH * SAMPLE_H)  # [gDW/L], X'=-kd X, X0=4.5 gDW/L
    kwargs = callback_inputs(model, ('rates.death',))
    kwargs['declaration']['fields'].pop('S')
    sigma = .001  # [gDW/L], declared numerical residual scale for noiseless qualification
    experiment = Experiment({'biomass': Measurement('X', times_s, truth,
        units='g/L', uncertainty=sigma)}, kwargs=kwargs)
    seed = np.array([DEATH / 2.])  # [1/h], deliberately separated from the synthetic truth
    estimator = ParameterEstimation(simulate_measurements, seed,
        {'decay': experiment}, output_names=('X',), name_params=['rates.death'],
        dx_finitediff=FD_STEP)
    # [0, 1]/h is an assumed numerical recovery interval (up to one e-fold/h),
    # containing this fixture's seed and truth; it is not a validated biological range.
    fitted, _, info = estimator.optimize_fn(method='least_squares', bounds=([0.], [1.]),
                                           verbose=False)
    assert info['success']
    np.testing.assert_allclose(fitted, [DEATH], rtol=0., atol=FIT_ATOL)
    np.testing.assert_array_equal(info['x'], fitted)
    np.testing.assert_array_equal(experiment.measurements['biomass'].values, truth)
    np.testing.assert_allclose(simulate_measurements(fitted, times_s, **kwargs)[:, 0],
                               truth, rtol=0., atol=CONCENTRATION_ATOL)
    restored = deepcopy(model)
    restored['parameters']['rates']['death'] = float(fitted[0])  # [1/h]
    assembly = build_bioreactor(restored, thermo_path=EXAMPLE / 'inputs/thermo.json')
    histories = assembly.solve(report_times=times_s)
    observations = {'time_s': {'X': times_s}}
    replay = predict_measurements(assembly, histories, kwargs['declaration'], observations)
    np.testing.assert_allclose(replay['values']['X'], truth,
                               rtol=0., atol=CONCENTRATION_ATOL)
    assert model == before
