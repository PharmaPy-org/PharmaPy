import copy
import json
from pathlib import Path

import numpy as np

from PharmaPy.Bioreactors import PathwayMetabolism, build_bioreactor
from PharmaPy.DataClasses import (IntraPhaseProcess, PhaseRef, StateKey,
                                  StateVariable, TransferResult)
from PharmaPy.IntegratorBackends import AssimuloBackend, SciPyBackend
from PharmaPy.Mechanisms import Mechanism
from PharmaPy.MultiPhaseVessel import MultiPhaseVessel
from PharmaPy.Phases_Refactored import LiquidPhase
from PharmaPy.Reactors_Refactored import BatchReactor, SemiBatchReactor


def _definition():
    return {
        "state_ids": ["glucose", "acetate", "oxygen"],
        "pathway_ids": ["v1", "v2", "v3", "v4"],
        "exchange_matrix": [
            [0, -9.46, -9.84, -19.23],
            [-39.43, 0, 1.24, 12.12],
            [-35, -12.92, -12.73, 0],
        ],
        "growth_coefficients": [1, 1, 1, 1],
        "objective_coefficients": [1, 1, 1, 1],
        "parameters": {
            "glucose_half_saturation": 0.015,
            "glucose_uptake_max": 10,
            "oxygen_mass_transfer": 7.5,
            "oxygen_saturation": 0.21,
            "oxygen_uptake_max": 15,
        },
        "uptake_constraints": [
            {
                "identifier": "glucose_uptake",
                "type": "monod",
                "species": "glucose",
                "maximum_parameter": "glucose_uptake_max",
                "half_saturation_parameter": "glucose_half_saturation",
            },
            {
                "identifier": "oxygen_uptake",
                "type": "transfer-cap",
                "species": "oxygen",
                "maximum_parameter": "oxygen_uptake_max",
                "mass_transfer_parameter": "oxygen_mass_transfer",
                "saturation_parameter": "oxygen_saturation",
            },
        ],
        "pathway_bound_rules": [
            {
                "type": "disable-at-depletion",
                "species": "acetate",
                "pathways": ["v1"],
            }
        ],
        "depletion_tolerance": 1e-8,
        "problem_name": "test pathway allocation"
    }


def test_pathway_mechanism_is_dimension_independent():
    payload = _definition()
    payload["pathway_ids"] = ["growth"]
    payload["exchange_matrix"] = [[-1], [0], [-1]]
    payload["growth_coefficients"] = [1]
    payload["objective_coefficients"] = [1]
    payload["pathway_bound_rules"] = []
    mechanism = PathwayMetabolism(payload, 1e-6)
    assert mechanism.model.pathway_ids == ("growth",)


def test_flagship_inputs_construct_native_reactor_specializations():
    root = Path(__file__).parents[2] / "examples/bioreactors"
    expected = {
        "batch_ecoli_dfba": BatchReactor,
        "fed_batch_cho": SemiBatchReactor,
    }
    for folder_name, reactor_type in expected.items():
        folder = (Path(__file__).parent / 'fixtures' if folder_name == 'fed_batch_cho' else root) / folder_name
        case = json.loads((folder / "inputs/case.json").read_text())
        definition = json.loads((folder / "inputs/mechanism.json").read_text())
        assembly = build_bioreactor(case, definition, folder / "inputs/thermo.json")
        assert type(assembly.unit) is reactor_type
        assert assembly.unit.Phases.Liquids[0] is assembly.phase
        assert assembly.unit.intraphase_processes[0].mechanism is assembly.mechanism


def test_inactive_biology_metadata_is_rejected():
    folder = Path(__file__).parents[2] / "examples/bioreactors/batch_ecoli_dfba"
    case = json.loads((folder / "inputs/case.json").read_text())
    definition = json.loads((folder / "inputs/mechanism.json").read_text())
    renamed = copy.deepcopy(case)
    renamed["biology"]["system"] = {"organism": "descriptive metadata only"}
    with np.testing.assert_raises_regex(ValueError, "unsupported.*system"):
        build_bioreactor(renamed, definition, folder / "inputs/thermo.json")


def test_pathway_mechanism_runs_inside_native_multiphase_vessel():
    thermo = (
        Path(__file__).parents[2]
        / "examples/bioreactors/batch_ecoli_dfba/inputs/thermo.json"
    )
    molecular_weights = np.array([180.156, 60.052, 31.998])
    species_moles = np.array([0.0108, 0.0004, 0.00021])
    solute_mass = species_moles * molecular_weights / 1000.0
    mass_j = np.r_[solute_mass, 1.0 - solute_mass.sum()]
    phase = LiquidPhase(thermo, mass=1.0, mass_frac=mass_j, verbose=False)
    mechanism = PathwayMetabolism(_definition(), 1e-6)
    phase.mechanisms = mechanism

    vessel = MultiPhaseVessel(integrator=AssimuloBackend({"maxh": 1.0}), isothermal=True)
    vessel.Phases = phase
    phase_ref = PhaseRef("liquid", 0)
    vessel.intraphase_processes = [IntraPhaseProcess(phase_ref, mechanism)]
    vessel.compile_structure()

    state = vessel.create_solver_init_states()
    rates = vessel.unit_model(0.0, state)
    assert rates.shape == state.shape
    assert np.isfinite(rates).all()
    biomass_slice = vessel.solver_state_collection.slices[StateKey("biomass_kg", phase_ref)]
    assert rates[biomass_slice][0] > 0.0
    assert mechanism.last_provider_result.provider_id == "pathway-optimization"
    assert mechanism.last_provider_result.units["growth_rate"] == "1/h"


def test_adaptive_backend_enforces_depletion_as_a_hard_boundary():
    thermo = Path(__file__).parents[2] / "examples/bioreactors/batch_ecoli_dfba/inputs/thermo.json"
    payload = _definition()
    payload["pathway_ids"] = ["growth"]
    payload["exchange_matrix"] = [[-1.0], [0.0], [0.0]]
    payload["growth_coefficients"] = [1.0]
    payload["objective_coefficients"] = [1.0]
    payload["uptake_constraints"] = [
        {
            "identifier": "glucose_uptake",
            "type": "constant",
            "species": "glucose",
            "maximum_parameter": "glucose_uptake_max",
        }
    ]
    payload["pathway_bound_rules"] = []
    solute_mass = np.array([1e-9, 0.0, 0.0])
    phase = LiquidPhase(thermo, mass=1.0,
                        mass_frac=np.r_[solute_mass, 1.0 - solute_mass.sum()],
                        verbose=False)
    mechanism = PathwayMetabolism(payload, 1e-6, flux_time_unit="1/s")
    phase.mechanisms = mechanism
    unit = MultiPhaseVessel(integrator=SciPyBackend({"max_step": 0.1}), isothermal=True)
    unit.Phases = phase
    ref = PhaseRef("liquid", 0)
    unit.intraphase_processes = [IntraPhaseProcess(ref, mechanism)]
    unit.solve_unit(time_grid=np.linspace(0.0, 2.0, 21), verbose=False)
    glucose = np.asarray(unit.result.mass_j_liquid0)[:, 0]
    assert np.all(glucose >= 0.0)
    assert glucose[-1] == 0.0


class _AlgebraicTestMechanism(Mechanism):
    def __init__(self):
        super().__init__()
        self.constraint = 2.0
        self.solver_states = (
            StateVariable("constraint", 1, "-", state_type="alg"),
        )
        self._update_exposed_attributes(); self._timers = {}

    def get_overrides(self, name=None): return {} if name is None else None

    def get_solver_state_rates(self, process, phase, **kwargs):
        return TransferResult({StateKey("mass_j", process.phaseref):
                               np.zeros(phase.num_species)}, {}, 0.0)

    def get_solver_state_residuals(self, completed_state, **kwargs):
        key = self.solver_state_keys[0]
        return {key: completed_state[key] - 2.0}


def test_native_vessel_assembles_differential_algebraic_residual():
    thermo = Path(__file__).parents[2] / "examples/bioreactors/batch_ecoli_dfba/inputs/thermo.json"
    phase = LiquidPhase(thermo, mass=1.0, mass_frac=np.full(4, 0.25), verbose=False)
    mechanism = _AlgebraicTestMechanism(); phase.mechanisms = mechanism
    unit = MultiPhaseVessel(isothermal=True)
    unit.Phases = phase
    ref = PhaseRef("liquid", 0)
    unit.intraphase_processes = [IntraPhaseProcess(ref, mechanism)]
    unit.compile_structure()
    states = unit.create_solver_init_states()
    derivatives = unit.create_solver_init_derivatives(states)
    assert unit.solver_state_collection.differential_mask.tolist() == [True] * 4 + [False]
    assert np.allclose(unit.dae_residual(0.0, states, derivatives), 0.0)


def test_network_conservation_audit_distinguishes_internal_and_boundary_reactions():
    from PharmaPy.Metabolic import MetabolicNetworkDefinition, ReactionDefinition

    def network(stoichiometry, exchanges=()):
        return MetabolicNetworkDefinition(
            'audit', '1', ('A', 'B'),
            [ReactionDefinition('convert', stoichiometry, 0., 10.),
             ReactionDefinition('feed', {'A': 1.}, 0., 10.)],
            {'convert': 1.}, exchanges)

    balanced = network({'A': -1., 'B': 2.}, ('feed',))
    before = balanced.stoichiometric_matrix.copy()
    assert balanced.audit_conservation() == {
        'stoichiometric_consistency': 'PASS', 'elemental_balance': 'NOT_ASSESSED'}
    np.testing.assert_array_equal(before, balanced.stoichiometric_matrix)
    # A source disguised as an internal reaction cannot conserve positive mass.
    assert network({'A': -1., 'B': 2.}).audit_conservation()[
        'stoichiometric_consistency'] == 'FAIL'
    for scale in (1., 1e-12, 1e12):
        assert network({'A': scale, 'B': scale}, ('feed',)).audit_conservation()[
            'stoichiometric_consistency'] == 'FAIL'
    assert network({'A': -1., 'B': 2.}, ('feed', 'convert')).audit_conservation()[
        'stoichiometric_consistency'] == 'NOT_ASSESSED'


def test_culture_audit_exposes_untracked_exchanges_without_changing_bounds():
    folder = Path(__file__).parents[2] / 'tests/Bioreactor/fixtures/fed_batch_cho'
    case = json.loads((folder / 'inputs/case.json').read_text())
    definition = json.loads((folder / 'inputs/mechanism.json').read_text())
    mechanism = build_bioreactor(case, definition, folder / 'inputs/thermo.json').mechanism
    lower, upper = mechanism.network.bounds()
    report = mechanism.audit_conservation()
    assert {item['reaction'] for item in report['untracked_exchanges']} == {
        'R041', 'R042', 'R043', 'R044', 'R045'}
    assert report['elemental_balance'] == 'NOT_ASSESSED'
    assert report['growth_coupled_to_flux'] is True
    assert report['inventory_constraints'] == 'BOUNDARY'
    mechanism.prepare_step(60.)
    assert mechanism.audit_conservation()['inventory_constraints'] == 'FIXED_STEP'
    np.testing.assert_array_equal(mechanism.network.lower_bounds, lower)
    np.testing.assert_array_equal(mechanism.network.upper_bounds, upper)
    for field, value in [('species', 'missing'), ('reaction', 'missing'),
                         ('internal_coefficient', float('nan'))]:
        invalid = copy.deepcopy(definition)
        invalid['model']['kinetics']['exchange_mappings'][0][field] = value
        with np.testing.assert_raises(ValueError):
            build_bioreactor(case, invalid, folder / 'inputs/thermo.json')


def test_inventory_constraint_couples_growth_to_resource_exposure():
    from PharmaPy.Bioreactors.inventory import InventoryAvailability

    availability = InventoryAvailability(
        [1.], [[-1., 0.]], [0.], viable=1., step_day=1.,
        cell_flux_scale=1., growth=0., death=0., growth_index=1)
    # The same uptake becomes infeasible when a larger population consumes it.
    assert availability.residual(np.array([0.75, 0.]))[0] > 0.
    assert availability.residual(np.array([0.75, 1.]))[0] < 0.
    flux = np.array([0.75, 0.4])
    delta = np.eye(2) * 1e-6
    numerical = np.column_stack([
        (availability.residual(flux + step) - availability.residual(flux - step)) / 2e-6
        for step in delta])
    np.testing.assert_allclose(availability.jacobian(flux), numerical, rtol=1e-8)


def test_closed_uptake_audit_detects_resource_free_production():
    from PharmaPy.Metabolic import MetabolicNetworkDefinition, ReactionDefinition
    feed = ReactionDefinition('feed', {'A': 1.}, 0., 10.)
    export = ReactionDefinition('export', {'B': -1.}, 0., 10.)
    balanced = ReactionDefinition('convert', {'A': -1., 'B': 1.}, 0., 10.)
    source = ReactionDefinition('source', {'B': 1.}, 0., 10.)
    def network(reactions, **kwargs):
        return MetabolicNetworkDefinition('test', '1', ('A', 'B'), reactions,
                                          {'export': 1.}, ('feed', 'export'), **kwargs)
    assert network([feed, export, balanced]).audit_closed_uptake()['status'] == 'PASS'
    audit = network([feed, export, balanced, source]).audit_closed_uptake()
    assert audit['status'] == 'FAIL'
    assert audit['maximum_exports']['export'] == 10.
    for roles in ({'missing': 'tracked'}, {'feed': 'invented'}):
        with np.testing.assert_raises(ValueError):
            network([feed, export, balanced], exchange_roles=roles)
    with np.testing.assert_raises(ValueError):
        network([feed, export, balanced], representation='invented')


def test_exchange_role_must_agree_with_inventory_mapping():
    folder = Path(__file__).parents[2] / 'tests/Bioreactor/fixtures/fed_batch_cho/inputs'
    case = json.loads((folder / 'case.json').read_text())
    definition = json.loads((folder / 'mechanism.json').read_text())
    definition['network']['exchange_roles']['R043'] = 'tracked'
    with np.testing.assert_raises_regex(ValueError, 'inventory mapping'):
        build_bioreactor(case, definition, folder / 'thermo.json')


def test_closed_uptake_audit_does_not_pass_an_infeasible_closed_model():
    from PharmaPy.Metabolic import MetabolicNetworkDefinition, ReactionDefinition
    n = MetabolicNetworkDefinition('test', '1', ('A',), [
        ReactionDefinition('feed', {'A': 1.}, 1., 10.),
        ReactionDefinition('export', {'A': -1.}, 0., 10.)],
        {'export': 1.}, ('feed', 'export'))
    assert n.audit_closed_uptake()['status'] == 'NOT_ASSESSED'
