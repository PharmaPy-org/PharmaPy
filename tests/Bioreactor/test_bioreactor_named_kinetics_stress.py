"""Targeted named-kinetics stress tests with real native solves and estimators.

Synthetic ratios span inhibition regimes, not organism validity ranges. Independent
concentration balances and exponential death solutions supply the oracles. The
coupled ledger accounts for omitted death/maintenance sinks, not elemental closure.
"""

from copy import deepcopy
from decimal import Decimal

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from PharmaPy.Bioreactors import simulate_measurements
from PharmaPy.Bioreactors.input_format import split_forward_inputs
from PharmaPy.ParamEstim import Experiment, Measurement, ParameterEstimation
from .test_bioreactor_relationships import evaluate, relation
from .test_bioreactor_named_kinetics import named_configuration, DEATH
from .test_bioreactor_estimation_handoff import (
    callback_inputs, HOUR_S, SAMPLE_H, CONCENTRATION_ATOL,
)


MU_MAX = .2  # [1/h], Doyle Exercise 4.15
AFFINITY = 1.  # [gS/L], Doyle Exercise 4.15
YIELD = .5  # [gDW/gS], Doyle Exercise 4.15
FEED = 10.  # [gS/L], Doyle Exercise 4.15
MAINTENANCE = .002  # [gS/(gDW h)], synthetic small positive sink over 8 h
ALPHA = .1  # [gP/gDW], synthetic growth-associated production coefficient
BETA = .003  # [gP/(gDW h)], synthetic non-growth-associated production coefficient
PRODUCT_YIELD = 1.  # [gP/gS], assumed unit substrate-to-product conversion for ledger
REFERENCE_RTOL = 1e-11  # [-], independent integration tighter than native 1e-9
REFERENCE_ATOL = 1e-12  # [g/L; gDW h/L for integral], independent absolute budgets
LEDGER_ATOL = 5. * CONCENTRATION_ATOL  # [gS/L], five scaled concentration/integral terms
ROUNDING_RTOL = 32. * np.finfo(float).eps  # [-], allowance for a short scalar operator graph
FIT_ATOL = 1e-6  # [1/h], synthetic death-recovery target, 1/30000 of nominal hazard
HAZARD_UPPER = 2. * DEATH  # [1/h], deliberately bounded recovery fixture, not biological range
FD_STEPS = (1e-4, 5e-5, 2.5e-5)  # [1/h], halving sequence for derivative refinement


# Real fixture construction and independent qualification helpers.
def stress_configuration(inhibition=4., death=0., maintenance=0., alpha=0., beta=0.,
                         dilution=0., substrate=4., biomass=.5, theta=0.):
    """Compose named rates in the existing native three-species configuration.

    Parameters
    ----------
    inhibition : float, optional
        Positive substrate inhibition constant [gS/L].
    death : float, optional
        Viable biomass hazard [1/h]; no dead-pool recycling is assumed.
    maintenance : float, optional
        Specific substrate maintenance demand [gS/(gDW h)].
    alpha : float, optional
        Growth-associated product coefficient [gP/gDW].
    beta : float, optional
        Non-growth-associated specific production [gP/(gDW h)].
    dilution : float, optional
        Sterile-feed dilution [1/h]; zero selects batch.
    substrate, biomass : float, optional
        Initial concentrations [gS/L, gDW/L] in the existing 1 L fixture.
    theta : float, optional
        Existing synthetic hybrid log correction [-].

    Returns
    -------
    dict
        Private native schema-3 input; all species source rates [g/(L h)].
        Product demand is charged to substrate using PRODUCT_YIELD.
    """
    model = named_configuration(True)
    # The coupled/high-substrate fixtures need tighter solves for the existing
    # absolute 1e-7 g/L comparison budget; defaults are not a global error guarantee.
    model['numerical']['relative_tolerance'] = 1e-10  # [-], tenfold refinement
    model['numerical']['absolute_tolerance'] = 1e-13  # [native extensive state units], tenfold refinement
    model['numerical']['maximum_step']['value'] /= 2.  # [h], halve the step cap
    parameters = model['parameters']
    parameters['affinities']['Ki'] = inhibition  # [gS/L]
    parameters['rates'].update(death=death, maintenance=maintenance, beta=beta)  # [1/h, gS/(gDW h), gP/(gDW h)]
    parameters['yields'].update(alpha=alpha, product=PRODUCT_YIELD)  # [gP/gDW, gP/gS]
    parameters['corrections']['theta'] = theta  # [-]
    model['state']['species']['amounts'].update(substrate=substrate, biomass=biomass)  # [g], 1 L
    model['operation']['mode'] = 'continuous' if dilution else 'batch'
    model['recipes']['continuous'] = ([dict(event_type='flow', time=0., volume_l_h=dilution,
        concentrations_g_l={'substrate': FEED}, density_kg_l=1., composition='supplied')]
        if dilution else [])  # [h, L/h, gS/L, kg/L], existing 1 L/density conventions
    kinetics = model['model']['kinetics']
    graph = [rule for rule in kinetics['rule_graph'] if rule['identifier'] != 'consumption']
    graph.extend([
        dict(identifier='death_loss', expression=relation('first-order-death',
            coefficient={'parameter': 'rates.death'}, population={'ref': 'biomass_g_l'})),
        dict(identifier='net_growth', expression={'op': 'subtract',
             'args': [{'ref': 'growth'}, {'ref': 'death_loss'}]}),
        dict(identifier='specific_product', expression=relation('yield-sum', terms=[
            {'rate': {'ref': 'mu'}, 'yield': {'parameter': 'yields.alpha'}},
            {'rate': {'parameter': 'rates.beta'}, 'yield': 1.}])),
        dict(identifier='product_source', expression={'op': 'multiply',
             'args': [{'ref': 'specific_product'}, {'ref': 'biomass_g_l'}]}),
        dict(identifier='specific_uptake', expression=relation('maintenance-uptake',
            growth={'ref': 'mu'}, maintenance={'parameter': 'rates.maintenance'},
            **{'yield': {'parameter': 'yields.biomass'}})),
        dict(identifier='substrate_demand', expression=relation('yield-sum', terms=[
            {'rate': {'op': 'multiply', 'args': [{'ref': 'specific_uptake'},
                                                {'ref': 'biomass_g_l'}]}, 'yield': 1.},
            {'rate': {'ref': 'product_source'}, 'yield': 1. / PRODUCT_YIELD}])),
        dict(identifier='consumption', expression={'op': 'negate', 'args': [{'ref': 'substrate_demand'}]}),
    ])
    kinetics['rule_graph'] = graph
    kinetics['species_mass_rates'] = dict(biomass='net_growth', substrate='consumption', product='product_source')
    model['rate_provider']['output_units'] = {name: 'g/(L h)' for name in
                                             ('net_growth', 'consumption', 'product_source')}
    return model


def stress_kwargs(model, names=('rates.death',)):
    """Bind three mass-concentration outputs in deliberately asymmetric order.

    Parameters
    ----------
    model : dict
        Native input declaration.
    names : tuple of str, optional
        Selected scalar paths in callback order, in their input units.

    Returns
    -------
    dict
        Callback arguments with columns product, substrate, biomass [g/L].
    """
    kwargs = callback_inputs(model, names)
    fields = kwargs['declaration']['fields']
    fields['S']['unit'] = 'g/L'
    fields['P'] = dict(fields['X'], species='product')
    kwargs['declaration']['fields'] = {name: fields[name] for name in ('P', 'S', 'X')}
    return kwargs


def reference_rates(time_h, state, inhibition, death, maintenance, alpha, beta, dilution, theta):
    """Evaluate independent balances including cumulative viable biomass exposure.

    Parameters
    ----------
    time_h : float
        Time [h]; autonomous balances.
    state : ndarray
        Substrate [gS/L], biomass [gDW/L], product [gP/L], exposure [gDW h/L].
    inhibition : float
        Inhibition constant [gS/L].
    death : float
        Hazard [1/h].
    maintenance : float
        Maintenance [gS/(gDW h)].
    alpha : float
        Product growth coefficient [gP/gDW].
    beta : float
        Product non-growth coefficient [gP/(gDW h)].
    dilution : float
        Dilution [1/h].
    theta : float
        Existing synthetic hybrid correction [-].

    Returns
    -------
    list of float
        Derivatives [gS/(L h), gDW/(L h), gP/(L h), gDW/L].
    """
    substrate, biomass, product, _ = state  # [gS/L, gDW/L, gP/L, gDW h/L]
    growth = MU_MAX * substrate / (AFFINITY + substrate + substrate ** 2 / inhibition)  # [1/h]
    growth *= np.exp(theta * substrate / (AFFINITY + substrate))  # [1/h]
    production = (alpha * growth + beta) * biomass  # [gP/(L h)]
    demand = (growth / YIELD + maintenance) * biomass + production / PRODUCT_YIELD  # [gS/(L h)]
    return [dilution * (FEED - substrate) - demand,
            (growth - death - dilution) * biomass, production - dilution * product, biomass]


def reference_solution(substrate, biomass, args, times_h=SAMPLE_H):
    """Integrate independent equations with two distinct numerical methods.

    Parameters
    ----------
    substrate, biomass : float
        Initial concentrations [gS/L, gDW/L].
    args : tuple
        Physical arguments to reference_rates, in its documented order/units.
    times_h : ndarray, optional
        Unique increasing reporting times [h].

    Returns
    -------
    ndarray
        Independent solution columns substrate, biomass, product, exposure;
        units [gS/L, gDW/L, gP/L, gDW h/L].
    """
    solutions = [solve_ivp(reference_rates, (0., times_h[-1]), [substrate, biomass, 0., 0.],
        args=args, t_eval=times_h, method=method, rtol=REFERENCE_RTOL, atol=REFERENCE_ATOL)
        for method in ('DOP853', 'Radau')]
    for solution in solutions:
        assert solution.success, solution.message
    np.testing.assert_allclose(solutions[0].y, solutions[1].y, rtol=0., atol=CONCENTRATION_ATOL)
    return solutions[0].y.T


def death_configuration(hazard=DEATH, biomass=4.5):
    """Select a death-only model while preserving native construction and units.

    Parameters
    ----------
    hazard : float, optional
        Loss coefficient [1/h].
    biomass : float, optional
        Initial biomass [gDW/L], in the existing 1 L working volume.

    Returns
    -------
    dict
        Private batch input with only biomass loss; no dead-pool closure claimed.
    """
    model = stress_configuration(death=hazard, biomass=biomass)
    model['model']['kinetics']['rule_graph'] = [
        dict(identifier='X', expression={'op': 'multiply', 'args': [
            {'concentration': 'biomass'}, {'constant': 'biomass_g_per_mmol'}]}),
        dict(identifier='loss', expression=relation('first-order-death',
            coefficient={'parameter': 'rates.death'}, population={'ref': 'X'})),
        dict(identifier='source', expression={'op': 'negate', 'args': [{'ref': 'loss'}]})]
    model['model']['kinetics']['species_mass_rates'] = {'biomass': 'source'}
    model['rate_provider'] = None
    return model


def inhibition_tangent_rates(time_h, state):
    """Integrate independent batch balances and exact Ki tangent equations.

    Parameters
    ----------
    time_h : float
        Time [h]; autonomous fixture.
    state : ndarray
        S [gS/L], X [gDW/L], dS/dKi [-], dX/dKi [gDW/gS].

    Returns
    -------
    list of float
        Derivatives [gS/(L h), gDW/(L h), 1/h, gDW/(gS h)].

    Notes
    -----
    Ki=4 gS/L is the nominal synthetic inhibition case. Differentiating
    mu=a*S/F gives mu_S=a*(Ks-S²/Ki)/F² and mu_Ki=a*S³/(Ki²*F²).
    """
    substrate, biomass, substrate_tangent, biomass_tangent = state  # [gS/L, gDW/L, -, gDW/gS]
    inhibition = 4.  # [gS/L], same synthetic Ki as the coupled fixture
    denominator = AFFINITY + substrate + substrate ** 2 / inhibition  # [gS/L]
    growth = MU_MAX * substrate / denominator  # [1/h]
    growth_substrate = MU_MAX * (AFFINITY - substrate ** 2 / inhibition) / denominator ** 2  # [L/(gS h)]
    growth_inhibition = MU_MAX * substrate ** 3 / (inhibition ** 2 * denominator ** 2)  # [L/(gS h)]
    synthesis = growth * biomass  # [gDW/(L h)]
    synthesis_tangent = (growth * biomass_tangent + biomass *
                         (growth_substrate * substrate_tangent + growth_inhibition))  # [gDW/(gS h)]
    return [-synthesis / YIELD, synthesis, -synthesis_tangent / YIELD, synthesis_tangent]


# Public-path regressions, independent balances, and failure qualification.
@pytest.mark.unit
@pytest.mark.parametrize('substrate_ratio', [0., 1e-12, 1e-6, .01, 1., 1e2, 1e6])
@pytest.mark.parametrize('inhibition_ratio', [1e-6, .01, 1., 1e2, 1e6])
@pytest.mark.parametrize('unit_factor', [1e-6, 1., 1e6])
def test_inhibition_ratio_sweep_matches_high_precision_and_peak(substrate_ratio, inhibition_ratio, unit_factor):
    """Sweep dimensionless regimes and concentration rescalings.

    Parameters
    ----------
    substrate_ratio, inhibition_ratio : float
        S/Ks and Ki/Ks [-], logarithmic numerical stress cases, not organism ranges.
    unit_factor : float
        Concentration-coordinate scale [-]; all concentration inputs convert together.
    """
    expression = relation('substrate-inhibited-growth', maximum=MU_MAX,
        substrate=substrate_ratio * unit_factor, half_saturation=unit_factor,
        inhibition_constant=inhibition_ratio * unit_factor)
    substrate_decimal = Decimal(str(substrate_ratio))  # [-], S/Ks
    inhibition_decimal = Decimal(str(inhibition_ratio))  # [-], Ki/Ks
    expected = float(Decimal(str(MU_MAX)) * substrate_decimal /
        (1 + substrate_decimal + substrate_decimal ** 2 / inhibition_decimal))  # [1/h]
    actual = evaluate(expression)  # [1/h]
    np.testing.assert_allclose(actual, expected, rtol=ROUNDING_RTOL, atol=0.)
    peak = MU_MAX / (1. + 2. / np.sqrt(inhibition_ratio))  # [1/h], derivative vanishes at sqrt(Ks Ki)
    assert 0. <= actual <= peak * (1. + ROUNDING_RTOL)


@pytest.mark.integration
@pytest.mark.parametrize('inhibition', [.001, 1., 1e3])
@pytest.mark.parametrize('substrate', [.001, 1., 8.])
@pytest.mark.parametrize('dilution_ratio', [0., .99, 1.01])
def test_dynamic_inhibition_near_peak_growth_and_washout(inhibition, substrate, dilution_ratio):
    """Compare native trajectories on both sides of the maximum growth threshold.

    Parameters
    ----------
    inhibition : float
        Synthetic Ki [gS/L], from strong to weak inhibition.
    substrate : float
        Initial S [gS/L], from scarcity to high substrate within declared domain.
    dilution_ratio : float
        Dilution / attainable maximum growth [-]. This checks transient dynamics,
        not a claim of steady-state stability near an inhibitory bifurcation.
    """
    peak = MU_MAX / (1. + 2. * np.sqrt(AFFINITY / inhibition))  # [1/h]
    dilution = dilution_ratio * peak  # [1/h]
    biomass = .1  # [gDW/L], low inoculum avoids exhausting substrate in 8 h
    model = stress_configuration(inhibition=inhibition, dilution=dilution,
                                 substrate=substrate, biomass=biomass)
    before = deepcopy(model)
    args = (inhibition, 0., 0., 0., 0., dilution, 0.)  # [gS/L, 1/h, gS/(gDW h), gP/gDW, gP/(gDW h), 1/h, -]
    reference = reference_solution(substrate, biomass, args)
    actual = simulate_measurements([0.], SAMPLE_H * HOUR_S, **stress_kwargs(model))  # [gP/L, gS/L, gDW/L]
    np.testing.assert_allclose(actual, reference[:, [2, 0, 1]], rtol=0., atol=CONCENTRATION_ATOL)
    # Exact sterile-feed substrate-equivalent identity when there are no sinks/products.
    equivalent = FEED + (substrate + biomass / YIELD - FEED) * np.exp(-dilution * SAMPLE_H)  # [gS/L]
    np.testing.assert_allclose(actual[:, 1] + actual[:, 2] / YIELD, equivalent,
                               rtol=0., atol=LEDGER_ATOL)
    assert np.all(actual >= -CONCENTRATION_ATOL)
    if dilution_ratio > 1.:
        bound = biomass * np.exp((peak - dilution) * SAMPLE_H)  # [gDW/L], growth never exceeds peak
        assert np.all(actual[:, 2] <= bound + CONCENTRATION_ATOL)
    assert model == before


@pytest.mark.integration
@pytest.mark.parametrize('theta', [-np.log(2.), 0., np.log(2.)])
@pytest.mark.parametrize('death', [0., DEATH])
def test_coupled_named_rates_preserve_gross_growth_and_sink_ledger(theta, death):
    """Check product/maintenance coupling with independent balances and a ledger.

    Parameters
    ----------
    theta : float
        Synthetic hybrid correction [-], including both declared endpoints.
    death : float
        Synthetic viable loss [1/h].
    """
    model = stress_configuration(death=death, maintenance=MAINTENANCE,
                                 alpha=ALPHA, beta=BETA, theta=theta)
    args = (4., death, MAINTENANCE, ALPHA, BETA, 0., theta)  # [gS/L, 1/h, gS/(gDW h), gP/gDW, gP/(gDW h), 1/h, -]
    reference = reference_solution(4., .5, args)
    actual = simulate_measurements([death], SAMPLE_H * HOUR_S, **stress_kwargs(model))  # [gP/L, gS/L, gDW/L]
    np.testing.assert_allclose(actual, reference[:, [2, 0, 1]], rtol=0., atol=CONCENTRATION_ATOL)
    # d(S+X/Y+P/Yp)/dt = -(kd/Y+m)X; exposure is independently integrated.
    accounted = (actual[:, 1] + actual[:, 2] / YIELD + actual[:, 0] / PRODUCT_YIELD
                 + (death / YIELD + MAINTENANCE) * reference[:, 3])  # [gS/L]
    np.testing.assert_allclose(accounted, 4. + .5 / YIELD, rtol=0., atol=LEDGER_ATOL)
    assert np.all(np.diff(actual[:, 0]) > 0.)


@pytest.mark.integration
@pytest.mark.parametrize('legacy', [False, True])
@pytest.mark.parametrize('reverse', [False, True])
def test_coupled_trial_ordering_failure_recovery_and_repeatability(legacy, reverse):
    """Exercise heterogeneous parameter paths and three output fields after failure.

    Parameters
    ----------
    legacy : bool
        Select separate legacy inputs instead of combined schema 3.
    reverse : bool
        Reverse parameter bindings and field insertion order together with values.
    """
    model = stress_configuration()
    before = deepcopy(model)
    names = ('rates.death', 'rates.maintenance', 'rates.beta', 'yields.alpha', 'corrections.theta')
    values = np.array([DEATH, MAINTENANCE, BETA, ALPHA, 0.])  # [1/h, gS/(gDW h), gP/(gDW h), gP/gDW, -]
    if reverse:
        names, values = names[::-1], values[::-1]
    kwargs = stress_kwargs(model, names)
    columns = [2, 0, 1]  # bookkeeping: P, S, X from independent S, X, P, exposure
    if reverse:
        fields = kwargs['declaration']['fields']
        kwargs['declaration']['fields'] = dict(reversed(list(fields.items())))
        columns = columns[::-1]
    if legacy:
        kwargs['case'], kwargs['definition'] = split_forward_inputs(model)
    first = simulate_measurements(values, SAMPLE_H * HOUR_S, **kwargs)  # [field concentration units g/L]
    other = values.copy()  # [same per-parameter units as values]
    other[names.index('rates.death')] *= 2.  # [1/h], distinct intervening physical trial
    different = simulate_measurements(other, SAMPLE_H * HOUR_S, **kwargs)  # [field concentration units g/L]
    invalid = values.copy()  # [same per-parameter units as values]
    invalid[names.index('corrections.theta')] = 2.  # [-], deliberately beyond declared ln(2) bound
    with pytest.raises(ValueError, match='outside validity domain'):
        simulate_measurements(invalid, SAMPLE_H * HOUR_S, **kwargs)
    repeated = simulate_measurements(values, SAMPLE_H * HOUR_S, **kwargs)  # [field concentration units g/L]
    np.testing.assert_array_equal(first, repeated)
    assert np.max(np.abs(first - different)) > CONCENTRATION_ATOL
    args = (4., DEATH, MAINTENANCE, ALPHA, BETA, 0., 0.)  # [gS/L, 1/h, gS/(gDW h), gP/gDW, gP/(gDW h), 1/h, -]
    expected = reference_solution(4., .5, args)[:, columns]  # [field concentration units g/L]
    np.testing.assert_allclose(first, expected, rtol=0., atol=CONCENTRATION_ATOL)
    assert model == before


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.parametrize('substrate', [.5, 4., 6., 9.5])
def test_inhibitory_chemostat_distinguishes_attraction_basins_over_long_horizon(substrate):
    """Resolve the productive and washout basins of an inhibitory chemostat.

    Parameters
    ----------
    substrate : float
        Initial substrate [gS/L]. Initial X=Y*(Sin-S) fixes the exact invariant.

    Notes
    -----
    Ki=4 gS/L and D=0.08/h give substrate equilibria 3±sqrt(5) gS/L;
    the higher root is unstable on the invariant. Initial states on either
    side reach different attractors. The 400 h horizon supplies >9 feed-end
    net-decay e-folds: (D-mu(Sin))*400 > 9. This is numerical qualification.
    """
    dilution = .08  # [1/h], chosen below the 0.1/h inhibited peak and above mu(feed)
    biomass = YIELD * (FEED - substrate)  # [gDW/L], exact sterile-feed invariant
    times_h = np.linspace(0., 400., 81)  # [h], 5 h reporting over the basin-separation horizon
    model = stress_configuration(dilution=dilution, substrate=substrate, biomass=biomass)
    model['operation']['runtime']['value'] = times_h[-1]  # [h]
    model['numerical']['relative_tolerance'] = 1e-11  # [-], long-horizon reference comparison
    args = (4., 0., 0., 0., 0., dilution, 0.)  # [gS/L, 1/h, gS/(gDW h), gP/gDW, gP/(gDW h), 1/h, -]
    reference = reference_solution(substrate, biomass, args, times_h)
    actual = simulate_measurements([0.], times_h * HOUR_S, **stress_kwargs(model))  # [gP/L, gS/L, gDW/L]
    np.testing.assert_allclose(actual, reference[:, [2, 0, 1]], rtol=0., atol=CONCENTRATION_ATOL)
    np.testing.assert_allclose(actual[:, 1] + actual[:, 2] / YIELD, FEED, rtol=0., atol=LEDGER_ATOL)
    low_root, high_root = 3. - np.sqrt(5.), 3. + np.sqrt(5.)  # [gS/L], exact roots of S²-6S+4=0
    if substrate < high_root:
        np.testing.assert_allclose(actual[-1, 1], low_root, rtol=0., atol=CONCENTRATION_ATOL)
    else:
        assert actual[-1, 1] > substrate
        assert actual[-1, 2] < biomass


@pytest.mark.integration
@pytest.mark.parametrize('substrate', [.2, 2., 8.])
def test_inhibition_parameter_sensitivity_against_independent_tangent_equations(substrate):
    """Check native Ki differences below, at and above the initial inhibition peak.

    Parameters
    ----------
    substrate : float
        Initial S [gS/L]; initial peak is sqrt(Ks*Ki)=2 gS/L.
    """
    model = stress_configuration(substrate=substrate, biomass=.1)
    model['numerical']['relative_tolerance'] = 1e-12  # [-], avoid amplifying integration error in parameter differences
    model['numerical']['absolute_tolerance'] = 1e-15  # [native extensive state units]
    kwargs = stress_kwargs(model, ('affinities.Ki',))
    reference = solve_ivp(inhibition_tangent_rates, (0., SAMPLE_H[-1]),
        [substrate, .1, 0., 0.], t_eval=SAMPLE_H, method='DOP853',
        rtol=REFERENCE_RTOL, atol=REFERENCE_ATOL)
    assert reference.success, reference.message
    expected = reference.y[[2, 3]].T  # [-, gDW/gS], S then X sensitivity to Ki
    derivatives = []  # [-, gDW/gS], S and X derivatives with respect to Ki
    for step in (1e-3, 5e-4, 2.5e-4):  # [gS/L], halving central-difference sequence around Ki=4
        plus = simulate_measurements([4. + step], SAMPLE_H * HOUR_S, **kwargs)[:, 1:]  # [gS/L, gDW/L]
        minus = simulate_measurements([4. - step], SAMPLE_H * HOUR_S, **kwargs)[:, 1:]  # [gS/L, gDW/L]
        derivative = (plus - minus) / (2. * step)  # [-, gDW/gS]
        derivatives.append(derivative)
        # 1e-7 is a declared sensitivity budget, far below changes from the
        # substrate-inhibition term; separate tests retain the concentration budget.
        np.testing.assert_allclose(derivative, expected, rtol=0., atol=1e-7)
    np.testing.assert_allclose(derivatives[0], derivatives[-1], rtol=0., atol=1e-7)


@pytest.mark.integration
@pytest.mark.parametrize('hazard', [0., 1e-12, DEATH, 1.])
@pytest.mark.parametrize('biomass', [0., .5, 4.5])
def test_death_zero_inventory_zero_hazard_and_fast_loss(hazard, biomass):
    """Exercise exponential loss limits across zero and near-zero cases.

    Parameters
    ----------
    hazard : float
        Synthetic loss coefficient [1/h]; 1/h gives eight e-folds over the fixture.
    biomass : float
        Initial biomass [gDW/L], including an empty pool.
    """
    model = death_configuration(hazard, biomass)
    actual = simulate_measurements([hazard], SAMPLE_H * HOUR_S, **stress_kwargs(model))  # [gP/L, gS/L, gDW/L]
    expected = biomass * np.exp(-hazard * SAMPLE_H)  # [gDW/L], exact constant-hazard solution
    np.testing.assert_allclose(actual[:, 2], expected, rtol=0., atol=CONCENTRATION_ATOL)
    np.testing.assert_allclose(actual[:, :2], np.tile([0., 4.], (len(SAMPLE_H), 1)),
                               rtol=0., atol=CONCENTRATION_ATOL)


@pytest.mark.integration
@pytest.mark.parametrize('truth', [0., DEATH, 3. * DEATH])
@pytest.mark.parametrize('seed', [0., DEATH / 2., HAZARD_UPPER])
def test_death_fit_zero_seed_and_active_bounds_report_constrained_optimum(truth, seed):
    """Fit noiseless decay with inactive, lower-active and upper-active bounds.

    Parameters
    ----------
    truth : float
        Generating hazard [1/h]; 3*DEATH intentionally lies above the fit bound.
    seed : float
        Trial seed [1/h], including both zero and the upper bound.
    """
    model = death_configuration()
    before = deepcopy(model)
    kwargs = stress_kwargs(model)
    values = 4.5 * np.exp(-truth * SAMPLE_H)  # [gDW/L], independent observations
    scale = .001  # [gDW/L], numerical residual scale, not empirical uncertainty
    experiment = Experiment({'X': Measurement('X', SAMPLE_H * HOUR_S,
        values, uncertainty=scale, units='g/L')}, kwargs=kwargs)
    estimator = ParameterEstimation(simulate_measurements, np.array([seed]),
        {'decay': experiment}, output_names=('P', 'S', 'X'), name_params=['rates.death'],
        dx_finitediff=FD_STEPS[-1])
    # Disable relative-cost termination near a zero bound; require gradient/step
    # termination. 128 evaluations allow trust-region growth from SciPy's interiorized zero.
    fitted, _, info = estimator.optimize_fn(method='least_squares', verbose=False,
        bounds=([0.], [HAZARD_UPPER]), optim_options=dict(ftol=None, xtol=1e-12,
                                                       gtol=1e-10, max_nfev=128))
    assert info['success'], info
    expected = min(truth, HAZARD_UPPER)  # [1/h], monotone exponential residual gives this exact constrained optimum
    np.testing.assert_allclose(fitted, [expected], rtol=0., atol=FIT_ATOL)
    np.testing.assert_array_equal(info['x'], fitted)
    if truth > HAZARD_UPPER:
        assert info['scipy_result'].active_mask[0] == 1
        assert np.linalg.norm(info['fun']) > 1.  # [-], bounded optimum cannot fit out-of-range truth
    np.testing.assert_array_equal(experiment.measurements['X'].values, values)
    assert model == before


@pytest.mark.integration
@pytest.mark.parametrize('hazard', [0., DEATH])
def test_death_parameter_derivative_refines_at_zero_and_interior(hazard):
    """Check forward numerical derivatives against exact exponential sensitivity.

    Parameters
    ----------
    hazard : float
        Evaluation coefficient [1/h]; zero is a one-sided physical boundary.
    """
    model = death_configuration(hazard)
    model['numerical']['relative_tolerance'] = 1e-12  # [-], derivative amplification needs three-order refinement
    model['numerical']['absolute_tolerance'] = 1e-15  # [native extensive state units], three-order refinement
    model['numerical']['maximum_step']['value'] /= 2.  # [h], independent refinement of step cap
    kwargs = stress_kwargs(model)
    base = simulate_measurements([hazard], SAMPLE_H * HOUR_S, **kwargs)[:, 2]  # [gDW/L]
    exact = -SAMPLE_H * 4.5 * np.exp(-hazard * SAMPLE_H)  # [gDW h/L]
    errors = []  # [gDW h/L], maximum absolute death-parameter derivative errors
    for step in FD_STEPS:
        forward = simulate_measurements([hazard + step], SAMPLE_H * HOUR_S, **kwargs)[:, 2]  # [gDW/L]
        derivative = (forward - base) / step  # [gDW h/L]
        errors.append(np.max(np.abs(derivative - exact)))  # [gDW h/L]
        # Taylor bound |error| <= h*t²*X/2, plus tighter-native concentration roundoff.
        budget = step * np.max(SAMPLE_H ** 2 * 4.5) / 2. + 2. * REFERENCE_ATOL / step  # [gDW h/L]
        assert errors[-1] <= budget
    assert errors[-1] < errors[0]
