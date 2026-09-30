"""Coupled resource accounting across both biological optimization methods."""

import json
from pathlib import Path

import numpy as np
import pytest

from PharmaPy.Bioreactors import PathwayMetabolism, build_bioreactor
from PharmaPy.Bioreactors.inventory import LinearInventoryAvailability
from PharmaPy.DataClasses import IntraPhaseProcess, PhaseRef
from PharmaPy.IntegratorBackends import FixedStepBackend, SciPyBackend
from PharmaPy.Metabolic import MetabolicNetworkDefinition, ReactionDefinition
from PharmaPy.Metabolic.closures.base import FluxSolution, MetabolicEnvironment, MetabolicInfeasibleError
from PharmaPy.Metabolic.closures.reconciled import RateReconciledMFAClosure, ReconciliationTarget
from PharmaPy.Metabolic.pathways import ConfiguredPathwayModel, PathwayModelDefinition
from PharmaPy.MultiPhaseVessel import MultiPhaseVessel
from PharmaPy.Phases_Refactored import LiquidPhase


EXAMPLES = Path(__file__).parents[2] / 'examples/bioreactors'


def _model(matrix=((-1.,), (-1.,), (0.,))):
    return PathwayModelDefinition(
        state_ids=('glucose', 'acetate', 'oxygen'), pathway_ids=('growth',),
        exchange_matrix=matrix, growth_coefficients=[1.], objective_coefficients=[1.],
        parameters={'maximum': 1.}, uptake_constraints=(
            dict(identifier='glucose', type='constant', species='glucose',
                 maximum_parameter='maximum'),))


@pytest.mark.parametrize('backend', ['scipy', 'fixed-step'])
@pytest.mark.parametrize('acetate', [0., 1e-5])
def test_missing_uptake_rule_cannot_create_growth_from_empty_resource(backend, acetate):
    thermo = EXAMPLES / 'batch_ecoli_dfba/inputs/thermo.json'
    solute = np.array([.001, acetate, 0.]) * np.array([180.156, 60.052, 31.998]) / 1000.
    phase = LiquidPhase(thermo, mass=1., mass_frac=np.r_[solute, 1.-solute.sum()], verbose=False)
    mechanism = PathwayMetabolism(_model(), 1e-4, flux_time_unit='1/s')
    phase.mechanisms = mechanism
    integrator = (FixedStepBackend() if backend == 'fixed-step' else
                  SciPyBackend({'rtol': 1e-9, 'atol': 1e-13, 'max_step': .02}))
    unit = MultiPhaseVessel(integrator=integrator, isothermal=True)
    unit.Phases = phase
    unit.intraphase_processes = [IntraPhaseProcess(PhaseRef('liquid', 0), mechanism)]
    unit.solve_unit(time_grid=np.linspace(0., 1., 6), verbose=False)
    amounts = unit.result.mass_j_liquid0[:, :3] / np.asarray(phase.mw[:3]) * 1000.
    growth = unit.result.biomass_kg_liquid0 - 1e-4
    np.testing.assert_allclose(growth, acetate - amounts[:, 1], atol=2e-12, rtol=1e-7)
    np.testing.assert_allclose(growth, .001 - amounts[:, 0], atol=2e-12, rtol=1e-7)
    assert growth[-1] == pytest.approx(acetate, abs=2e-12)
    assert np.min(unit.result.mass_j_liquid0) >= 0.


def test_standalone_pathway_closes_all_exhausted_pools():
    result = ConfiguredPathwayModel(_model()).solve_fluxes([1., 0., 0.], 1.)
    assert result.growth_rate == 0.
    assert isinstance(result, FluxSolution)
    assert result.status == result.solver_status
    assert result.message == result.solver_message


def test_scheduled_feed_restarts_growth_after_depletion():
    p = EXAMPLES / 'generic_fed_batch/inputs'
    case, definition = [json.loads((p / name).read_text()) for name in ('case.json', 'mechanism.json')]
    definition['state']['species']['amounts']['nutrient'] = 0.
    assembly = build_bioreactor(case, definition, p / 'thermo.json')
    before, after = assembly.solve(verbose=False)
    np.testing.assert_allclose(before.biomass_kg_liquid0, 1e-4, atol=1e-12)
    assert after.biomass_kg_liquid0[-1] == pytest.approx(1e-4 * np.exp(.2 * 3.), rel=1e-6)
    nutrient = assembly.phase.name_species.index('nutrient')
    consumed = (after.mass_j_liquid0[0, nutrient] - after.mass_j_liquid0[-1, nutrient])
    consumed *= 1000. / assembly.phase.mw[nutrient]
    assert after.biomass_kg_liquid0[-1] - 1e-4 == pytest.approx(consumed / 10., rel=1e-6)


@pytest.mark.parametrize('scale', [1., 1e-12, 1e12])
def test_both_closures_use_actual_supply_and_same_inventory_interface(scale):
    available = LinearInventoryAvailability.boundary(
        [0.], [[-scale]], [scale * .25])
    pathway = ConfiguredPathwayModel(_model())
    p = pathway.solve_fluxes([1., 1., 1.], 1., availability=available)
    assert p.growth_rate == pytest.approx(.25)
    network = MetabolicNetworkDefinition('test', '1', ('A',),
        [ReactionDefinition('uptake', {'A': 1.}, 0., 10.),
         ReactionDefinition('growth', {'A': -1.}, 0., 10.)], {'growth': 1.}, ('uptake', 'growth'))
    closure = RateReconciledMFAClosure(network, ())
    available = LinearInventoryAvailability.boundary([0.], [[-scale, 0.]], [scale * .25])
    r = closure.solve(MetabolicEnvironment({}),
        targets=[ReconciliationTarget('uptake', 1., 1.)], internal_scale=1., availability=available)
    np.testing.assert_allclose(r.fluxes, [.25, .25], atol=1e-7)
    assert p.diagnostics().keys() == r.diagnostics().keys()
    assert p.mass_balance_residual_inf is None
    assert r.mass_balance_residual_inf < 1e-7
    assert r.inventory_violation_inf < 1e-7


def test_impossible_external_drain_is_reported_as_infeasible():
    availability = LinearInventoryAvailability([0.], [[-1.]], [-1.])
    with pytest.raises(MetabolicInfeasibleError):
        ConfiguredPathwayModel(_model()).solve_fluxes([1., 1., 1.], 1., availability=availability)


def test_transport_capacity_cannot_disagree_with_actual_supply():
    p = EXAMPLES / 'batch_ecoli_dfba/inputs'
    case, definition = [json.loads((p / name).read_text()) for name in ('case.json', 'mechanism.json')]
    definition['transport']['kla_per_s'] = 0.
    with pytest.raises(ValueError, match='actual transport'):
        build_bioreactor(case, definition, p / 'thermo.json')


def test_culture_adaptive_boundary_constrains_growth_and_product_together():
    p = Path(__file__).parent / 'fixtures/fed_batch_cho/inputs'
    case, definition = [json.loads((p / name).read_text()) for name in ('case.json', 'mechanism.json')]
    definition['state']['species']['amounts']['TRP'] = 0.
    assembly = build_bioreactor(case, definition, p / 'thermo.json')
    unit = assembly.unit
    unit.compile_structure()
    unit.unit_model(0., unit.create_solver_init_states(), limiter_dt=0.)
    mechanism = assembly.mechanism
    solution = mechanism.last_solution
    growth = mechanism.definition.kinetic_outputs['growth_reaction']
    product = mechanism.definition.product_mapping['reaction']
    for name in (growth, product):
        assert abs(solution.fluxes[solution.reaction_ids.index(name)]) < 1e-7
    assert solution.inventory_violation_inf < 1e-7


def test_lexicographic_selection_is_invariant_to_pathway_column_order():
    rates = []
    for ids, matrix in [(('a', 'b'), [[-1., -1.], [1., 2.]]),
                        (('b', 'a'), [[-1., -1.], [2., 1.]])]:
        model = ConfiguredPathwayModel(PathwayModelDefinition(
            state_ids=('substrate', 'product'), pathway_ids=ids, exchange_matrix=matrix,
            growth_coefficients=[1., 1.], objective_coefficients=[1., 1.], parameters={'maximum': 1.},
            uptake_constraints=(dict(identifier='cap', type='constant', species='substrate',
                                     maximum_parameter='maximum'),)))
        rates.append(model.solve_fluxes([1., 0.], 1.).extracellular_rates)
    np.testing.assert_array_equal(*rates)


def test_closed_uptake_screen_flags_free_growth_without_claiming_elemental_balance():
    assert ConfiguredPathwayModel(_model()).audit_closed_uptake()['status'] == 'PASS'
    mechanism = PathwayMetabolism(_model(((0.,), (0.,), (0.,))), 1.)
    report = mechanism.audit_conservation()
    assert report['closed_uptake']['status'] == 'FAIL'
    assert report['elemental_balance'] == 'NOT_ASSESSED'


@pytest.mark.parametrize('capacity, expected', [(10., 5.), (3., 3.)])
def test_unweighted_culture_matches_analytic_projection(capacity, expected):
    p = Path(__file__).parent / 'fixtures/khare_cho_fed_batch/inputs'
    case = json.loads((p / 'case.json').read_text())
    definition = json.loads((p / 'mechanism.json').read_text())
    mechanism = build_bioreactor(case, definition, p / 'thermo.json').mechanism
    assert mechanism.closure.internal_reaction_ids == ()
    network = MetabolicNetworkDefinition('projection', '1', ('A',),
        [ReactionDefinition('uptake', {'A': 1.}, 0., 10.),
         ReactionDefinition('growth', {'A': -1.}, 0., 10.)],
        {'growth': 1.}, ('uptake', 'growth'))
    mechanism.closure = RateReconciledMFAClosure(network, ())
    availability = LinearInventoryAvailability.boundary([0.], [[-1., 0.]], [capacity])
    # min (x-1)^2 + (x-9)^2 has x=5, or the physical capacity if smaller.
    # Relative target bounds would incorrectly exclude this optimum.
    result = mechanism._solve({'uptake': 1., 'growth': 9.}, availability)
    np.testing.assert_allclose(result.fluxes, expected, atol=1e-7)
    assert result.mass_balance_residual_inf < 1e-8
    assert result.inventory_violation_inf < 1e-8
    assert mechanism.last_reconciliation_relative_bound is None


@pytest.mark.parametrize('policy, bound', [('unknown', None), ('unweighted', 2.5)])
def test_culture_rejects_ambiguous_reconciliation_policy(policy, bound):
    p = Path(__file__).parent / 'fixtures/khare_cho_fed_batch/inputs'
    case = json.loads((p / 'case.json').read_text())
    definition = json.loads((p / 'mechanism.json').read_text())
    definition['model']['reconciliation'].update(
        reconciliation_policy=policy, reconciliation_relative_bound=bound)
    with pytest.raises(ValueError, match='reconciliation|relative target bounds'):
        build_bioreactor(case, definition, p / 'thermo.json')


@pytest.mark.parametrize('magnitude', [0., .01, 1., 100., 10000.])
def test_unweighted_small_feasible_targets_retain_exchange_accuracy(magnitude):
    p = Path(__file__).parent / 'fixtures/khare_cho_fed_batch/inputs'
    mechanism = build_bioreactor(json.loads((p / 'case.json').read_text()),
        json.loads((p / 'mechanism.json').read_text()), p / 'thermo.json').mechanism
    network = MetabolicNetworkDefinition('small-targets', '1', ('A', 'B', 'C'), [
        ReactionDefinition('uptake', {'A': 1.}, 0., 1.),
        ReactionDefinition('conversion', {'A': -1., 'B': 2.}, 0., 1.),
        ReactionDefinition('sink_a', {'A': -1.}, 0., 1.),
        ReactionDefinition('sink_b', {'B': -1.}, 0., 1.),
        ReactionDefinition('exchange', {'B': -1.}, -1., 1.),
        ReactionDefinition('source', {'C': 1.}, 0., 1.),
        ReactionDefinition('product', {'C': -1.}, 0., 1.)],
        {'product': 1.}, ('uptake', 'exchange', 'product'))
    mechanism.closure = RateReconciledMFAClosure(network, ())
    targets = dict(zip(('uptake', 'exchange', 'product'), magnitude * np.array([1e-7, -1e-8, 1e-5])))
    solution = mechanism._solve(targets)
    actual = [solution.fluxes[solution.reaction_ids.index(name)] for name in targets]
    # Bound error in the objective's common coordinate, including zero targets.
    scale = max(max(abs(value) for value in targets.values()), mechanism.target_floor)
    # SLSQP stops on a squared, scaled residual objective (ftol=1e-15).
    # A rate error is therefore bounded in square-root objective coordinates;
    # demanding 1e-8 here exceeded that accuracy contract across SciPy builds.
    np.testing.assert_allclose(actual, list(targets.values()), rtol=0.,
                               atol=np.sqrt(1e-15) * scale)
    assert solution.mass_balance_residual_inf < 1e-10
