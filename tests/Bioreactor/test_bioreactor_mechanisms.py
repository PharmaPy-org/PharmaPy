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
        folder = root / folder_name
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
