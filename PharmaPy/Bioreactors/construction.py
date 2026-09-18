"""Construct input-declared biological processes with native PharmaPy objects.

``build_bioreactor`` converts generic mappings into a ``LiquidPhase``, a
biological ``Mechanism``, and a native ``BatchReactor`` or
``SemiBatchReactor``. Scheduled material operations are applied between
ordinary native ``solve_unit`` calls.
"""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from PharmaPy.DataClasses import IntraPhaseProcess, PhaseRef
from PharmaPy.IntegratorBackends import FixedStepBackend, SciPyBackend
from PharmaPy.Metabolic import MetabolicNetworkDefinition
from PharmaPy.Phases_Refactored import LiquidPhase
from PharmaPy.Reactors_Refactored import BatchReactor, SemiBatchReactor

from .mechanisms import PathwayMetabolism, RateReconciledCulture


_TIME_FACTORS = {"s": 1.0, "min": 60.0, "h": 3600.0, "day": 86400.0}
_REACTORS = {
    "batch": BatchReactor,
    "fed-batch": SemiBatchReactor,
    "semibatch": SemiBatchReactor,
}


def _require_keys(mapping, required, name, optional=()):
    """Reject missing and inactive configuration fields at each input boundary."""
    if not isinstance(mapping, Mapping):
        raise ValueError(f"{name} must be a mapping")
    keys = set(mapping)
    required = set(required)
    allowed = required | set(optional)
    missing = required - keys
    unknown = keys - allowed
    if missing or unknown:
        details = []
        if missing:
            details.append(f"missing {sorted(missing)}")
        if unknown:
            details.append(f"unsupported {sorted(unknown)}")
        raise ValueError(f"{name} has " + " and ".join(details))


def _quantity_seconds(quantity, name):
    if not isinstance(quantity, Mapping) or set(quantity) != {"unit", "value"}:
        raise ValueError(f"{name} must contain exactly unit and value")
    unit = quantity["unit"]
    if unit not in _TIME_FACTORS:
        raise ValueError(f"unsupported {name} time unit {unit!r}")
    value = float(quantity["value"])
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return value * _TIME_FACTORS[unit]


def _quantity_liters(quantity, name):
    if not isinstance(quantity, Mapping) or set(quantity) != {"unit", "value"}:
        raise ValueError(f"{name} must contain exactly unit and value")
    factors = {"L": 1.0, "m3": 1000.0}
    if quantity["unit"] not in factors:
        raise ValueError(f"unsupported {name} volume unit {quantity['unit']!r}")
    value = float(quantity["value"]) * factors[quantity["unit"]]
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _phase(thermo_path, mass_kg, species_amounts, amount_scale, carrier_species):
    if not np.isfinite(mass_kg) or mass_kg <= 0.0:
        raise ValueError("initial_mass_kg must be finite and positive")
    with Path(thermo_path).open() as stream:
        thermo_names = tuple(json.load(stream))
    probe = LiquidPhase(
        thermo_path, mass=1.0,
        mass_frac=np.full(len(thermo_names), 1.0 / len(thermo_names)),
        verbose=False,
    )
    names = tuple(probe.name_species)
    if carrier_species not in names:
        raise ValueError(f"carrier species {carrier_species!r} is absent from phase")
    solute_names = tuple(name for name in names if name != carrier_species)
    if set(species_amounts) != set(solute_names):
        raise ValueError(
            "initial species identifiers must exactly match non-solvent phase species"
        )
    amounts = np.asarray([species_amounts[name] for name in solute_names], dtype=float)
    if not np.isfinite(amounts).all() or np.any(amounts < 0.0):
        raise ValueError("initial species amounts must be finite and nonnegative")
    molecular_weights = np.asarray(
        [probe.mw[names.index(name)] for name in solute_names], dtype=float
    )
    solute_mass = amounts * molecular_weights * amount_scale
    solvent_mass = mass_kg - float(solute_mass.sum())
    if solvent_mass < 0.0:
        raise ValueError("declared solute mass exceeds initial phase mass")
    mass = np.zeros(len(names), dtype=float)
    for name, value in zip(solute_names, solute_mass):
        mass[names.index(name)] = value
    mass[names.index(carrier_species)] = solvent_mass
    phase = LiquidPhase(thermo_path, mass=mass_kg,
                        mass_frac=mass / mass_kg, verbose=False)
    return phase


def _pathway_mechanism(definition, phase, operation):
    model = definition["model"]
    state = definition["state"]
    phase_species = set(phase.name_species) - {operation["carrier_species"]}
    if set(model["state_ids"]) != phase_species:
        raise ValueError("pathway state_ids must exactly match phase solutes")
    transport = definition.get("transport")
    transfers = ()
    if transport is not None:
        required = {"species", "kla_per_s", "saturation_mol_m3"}
        if not required <= set(transport):
            raise ValueError(f"transport must declare {sorted(required)}")
        transfers = ({key: transport[key] for key in required},)
    flux_unit = definition.get("flux_time_unit")
    if flux_unit is None:
        raise ValueError("pathway mechanism must explicitly declare flux_time_unit")
    mechanism = PathwayMetabolism(
        model, state["biomass_kg"], flux_time_unit=flux_unit,
        mass_transfer=transfers,
    )
    return mechanism


def _reconciled_mechanism(definition, conditions):
    model = dict(definition["model"])
    model["parameters"] = definition["parameters"]
    state = definition["state"]
    network = MetabolicNetworkDefinition.from_mapping(definition["network"])
    for name, limits in model.get("condition_bounds", {}).items():
        if name not in conditions:
            raise ValueError(f"required operating condition {name!r} is absent")
        value = float(conditions[name])
        lower, upper = map(float, limits)
        if not np.isfinite(value) or not lower <= value <= upper:
            raise ValueError(
                f"operating condition {name!r} lies outside its declared bounds"
            )
    required_state = {
        "viable_cells_million": state["viable_cells_million"],
        "dead_cells_million": state["dead_cells_million"],
        "product_g": state["product_g"],
        "ivcd_million_cell_day_per_ml": state["integral"],
    }
    return RateReconciledCulture(
        model, network, required_state, conditions=conditions,
        rate_provider=definition["rate_provider"],
    )


def _time_grid(operation, numerics):
    runtime = _quantity_seconds(operation["runtime"], "runtime")
    step = _quantity_seconds(numerics["step"], "step")
    count = int(np.floor(runtime / step))
    grid = np.arange(count + 1, dtype=float) * step
    if grid[-1] < runtime:
        grid = np.r_[grid, runtime]
    else:
        grid[-1] = runtime
    return grid


def _integrator(numerics):
    backend = numerics["backend"]
    if backend == "fixed-step":
        _require_keys(numerics, {"backend", "step"}, "fixed-step numerics")
        return FixedStepBackend()
    if backend == "scipy":
        _require_keys(
            numerics,
            {"backend", "step", "relative_tolerance", "absolute_tolerance"},
            "scipy numerics",
            {"maximum_step", "jacobian_relative_step", "jacobian_state_scale"},
        )
        options = {
            "rtol": float(numerics["relative_tolerance"]),
            "atol": float(numerics["absolute_tolerance"]),
        }
        if "maximum_step" in numerics:
            options["max_step"] = _quantity_seconds(
                numerics["maximum_step"], "maximum_step"
            )
        for name in ("jacobian_relative_step", "jacobian_state_scale"):
            if name in numerics:
                options[name] = numerics[name]
        return SciPyBackend(options)
    raise ValueError(f"unsupported native bioreactor backend {backend!r}")


def _event_time_seconds(event, time_unit):
    if "time" in event:
        value = float(event["time"])
    elif time_unit in event:
        value = float(event[time_unit])
    else:
        raise ValueError(f"recipe event must declare time or {time_unit!r}")
    if not np.isfinite(value) or value < 0.0:
        raise ValueError("recipe event time must be finite and nonnegative")
    return value * _TIME_FACTORS[time_unit]


def _apply_material_event(phase, event, carrier_density_kg_l, carrier_species):
    mass = np.asarray(phase.mass_j, dtype=float).copy()
    names = tuple(phase.name_species)
    event_type = event["event_type"]
    if event_type == "feed":
        _require_keys(
            event, {"time", "event_type", "volume_l", "concentrations_mmol_l"},
            "feed event",
        )
        volume_l = float(event["volume_l"])
        if not np.isfinite(volume_l) or volume_l < 0.0:
            raise ValueError("feed volume_l must be finite and nonnegative")
        concentrations = event["concentrations_mmol_l"]
        unknown = set(concentrations) - set(names)
        if unknown:
            raise ValueError(f"feed references unknown species {sorted(unknown)}")
        for species, concentration in concentrations.items():
            concentration = float(concentration)
            if not np.isfinite(concentration) or concentration < 0.0:
                raise ValueError("feed concentrations must be finite and nonnegative")
            index = names.index(species)
            mass[index] += concentration * volume_l * phase.mw[index] * 1e-6
        mass[names.index(carrier_species)] += volume_l * carrier_density_kg_l
    elif event_type == "target_concentration":
        _require_keys(
            event,
            {"time", "event_type", "target_species",
             "target_concentration_mmol_l"},
            "target-concentration event",
        )
        species = event["target_species"]
        if species not in names:
            raise ValueError(f"target references unknown species {species!r}")
        target = float(event["target_concentration_mmol_l"])
        if not np.isfinite(target) or target < 0.0:
            raise ValueError("target concentration must be finite and nonnegative")
        index = names.index(species)
        volume_l = float(phase.vol) * 1000.0
        current_mmol = mass[index] / phase.mw[index] * 1e6
        addition_mmol = target * volume_l - current_mmol
        if addition_mmol < 0.0:
            raise ValueError(
                f"target concentration for {species} cannot be reached by addition"
            )
        mass[index] += addition_mmol * phase.mw[index] * 1e-6
    else:
        raise ValueError(f"unsupported material event {event_type!r}")
    phase.updatePhase(mass_j=mass)


@dataclass
class NativeBioreactorAssembly:
    """Hold a native reactor and the input-declared schedule used to solve it."""

    unit: object
    phase: LiquidPhase
    mechanism: object
    time_grid: np.ndarray
    events: tuple
    event_time_unit: str
    carrier_density_kg_l: float
    carrier_species: str

    def solve(self, *, verbose=False):
        """Execute scheduled operations between native reactor solves."""
        if not self.events:
            self.unit.solve_unit(time_grid=self.time_grid, verbose=verbose)
            return (self.unit.result,)
        scheduled = sorted(
            ((_event_time_seconds(event, self.event_time_unit), event)
             for event in self.events), key=lambda item: item[0]
        )
        final_time = float(self.time_grid[-1])
        scheduled = [(time, event) for time, event in scheduled
                     if time <= final_time]
        if not scheduled:
            self.unit.solve_unit(time_grid=self.time_grid, verbose=verbose)
            return (self.unit.result,)
        boundaries = sorted({float(self.time_grid[0]), final_time,
                             *(time for time, _ in scheduled)})
        histories = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            for time, event in scheduled:
                if time == start:
                    _apply_material_event(
                        self.phase, event, self.carrier_density_kg_l,
                        self.carrier_species,
                    )
            grid = self.time_grid[
                (self.time_grid >= start) & (self.time_grid <= end)
            ]
            if grid.size == 0 or grid[0] != start:
                grid = np.r_[start, grid]
            if grid[-1] != end:
                grid = np.r_[grid, end]
            self.unit.solve_unit(time_grid=grid, verbose=verbose)
            histories.append(self.unit.result)
        return tuple(histories)


def build_bioreactor(case, definition, thermo_path):
    """Build one input-declared process as an ordinary native reactor object."""
    _require_keys(case, {"biology", "numerics", "operation", "recipe"}, "case")
    _require_keys(case["biology"], {"conditions", "mechanism"}, "biology")
    _require_keys(
        case["operation"],
        {"carrier_density_kg_l", "carrier_species", "initial_mass_kg",
         "initial_volume", "mode", "runtime"},
        "operation",
    )
    _require_keys(case["recipe"], {"name"}, "recipe")
    if case["biology"]["mechanism"] != definition["mechanism"]:
        raise ValueError("case and mechanism declarations do not agree")
    operation = case["operation"]
    mode = operation["mode"]
    if mode not in _REACTORS:
        raise ValueError(f"unsupported native reactor mode {mode!r}")
    mass_kg = float(operation["initial_mass_kg"])
    density = float(operation["carrier_density_kg_l"])
    if not np.isfinite(density) or density <= 0.0:
        raise ValueError("carrier_density_kg_l must be finite and positive")
    initial_volume_l = _quantity_liters(
        operation["initial_volume"], "initial_volume"
    )
    if not np.isclose(mass_kg, initial_volume_l * density, rtol=0.0, atol=1e-12):
        raise ValueError(
            "initial_mass_kg must agree with initial_volume and carrier density"
        )
    state = definition["state"]
    carrier_species = str(operation["carrier_species"])
    mechanism_name = definition["mechanism"]
    if mechanism_name == "pathway-lp":
        _require_keys(
            definition,
            {"mechanism", "transport", "flux_time_unit", "model", "recipes",
             "state"},
            "pathway mechanism input",
        )
        if not np.isclose(
            float(state["liquid_volume_m3"]) * 1000.0, initial_volume_l,
            rtol=0.0, atol=1e-12,
        ):
            raise ValueError("pathway state volume disagrees with case initial_volume")
        phase = _phase(
            thermo_path, mass_kg, state["species_mol"], 1e-3,
            carrier_species,
        )
    elif mechanism_name == "rate-reconciled-culture":
        _require_keys(
            definition,
            {"mechanism", "model", "network", "parameters", "rate_provider",
             "recipes", "state"},
            "rate-reconciled mechanism input",
        )
        if not np.isclose(
            float(state["volume_l"]), initial_volume_l,
            rtol=0.0, atol=1e-12,
        ):
            raise ValueError("culture state volume disagrees with case initial_volume")
        phase = _phase(
            thermo_path, mass_kg, state["species_mmol"], 1e-6,
            carrier_species,
        )
    else:
        raise ValueError(f"unsupported biological mechanism {mechanism_name!r}")

    conditions = dict(case["biology"]["conditions"])
    if mechanism_name == "pathway-lp":
        mechanism = _pathway_mechanism(definition, phase, operation)
    else:
        mechanism = _reconciled_mechanism(definition, conditions)
    phase.mechanisms = mechanism
    unit = _REACTORS[mode](integrator=_integrator(case["numerics"]),
                           isothermal=True)
    unit.Phases = phase
    unit.intraphase_processes = [
        IntraPhaseProcess(PhaseRef("liquid", 0), mechanism)
    ]

    recipe_name = case["recipe"]["name"]
    if recipe_name not in definition["recipes"]:
        raise ValueError(f"mechanism input has no recipe {recipe_name!r}")
    events = tuple(definition["recipes"][recipe_name])
    time_unit = case["numerics"]["step"]["unit"]
    return NativeBioreactorAssembly(
        unit, phase, mechanism, _time_grid(operation, case["numerics"]),
        events, time_unit, density, carrier_species,
    )
