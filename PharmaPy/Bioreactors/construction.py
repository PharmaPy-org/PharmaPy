"""Construct input-declared biological processes with native PharmaPy objects.

``build_bioreactor`` converts generic mappings into a ``LiquidPhase``, a
biological ``Mechanism``, and a native ``BatchReactor`` or
``SemiBatchReactor``. Scheduled material operations are applied between
ordinary native ``solve_unit`` calls.
"""

import json
from copy import deepcopy
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from PharmaPy.DataClasses import IntraPhaseProcess, PhaseRef
from PharmaPy.IntegratorBackends import FixedStepBackend, SciPyBackend
from PharmaPy.Metabolic import MetabolicNetworkDefinition
from PharmaPy.Phases_Refactored import LiquidPhase
from PharmaPy.Reactors_Refactored import BatchReactor, SemiBatchReactor
from PharmaPy.Streams_Refactored import LiquidStream

from .mechanisms import ConfiguredRates, PathwayMetabolism, RateReconciledCulture, WorkingVolume
from .input_format import normalize_forward_inputs
from .BioreactorUnitConverter import BioreactorUnitConverter


_TIME_FACTORS = BioreactorUnitConverter.TIME
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
    return BioreactorUnitConverter.convert(value, unit, "s")


def _quantity_liters(quantity, name):
    if not isinstance(quantity, Mapping) or set(quantity) != {"unit", "value"}:
        raise ValueError(f"{name} must contain exactly unit and value")
    factors = BioreactorUnitConverter.VOLUME
    if quantity["unit"] not in factors:
        raise ValueError(f"unsupported {name} volume unit {quantity['unit']!r}")
    value = BioreactorUnitConverter.convert(quantity["value"], quantity["unit"], "L")
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _phase(thermo_path, mass_kg, species_amounts, amount_scale, carrier_species,
           temperature_k=298.15, amount_unit=None):
    if not np.isfinite(mass_kg) or mass_kg <= 0.0:
        raise ValueError("initial_mass_kg must be finite and positive")
    with Path(thermo_path).open() as stream:
        thermo_names = tuple(json.load(stream))
    probe = LiquidPhase(
        thermo_path, mass=1.0, temp=temperature_k,
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
    solute_mass = (amounts * molecular_weights * amount_scale if amount_unit is None else
                   BioreactorUnitConverter.convert(amounts, amount_unit, 'kg',
                                                  molecular_weight_g_mol=molecular_weights))
    solvent_mass = mass_kg - float(solute_mass.sum())
    if solvent_mass < 0.0:
        raise ValueError("declared solute mass exceeds initial phase mass")
    mass = np.zeros(len(names), dtype=float)
    for name, value in zip(solute_names, solute_mass):
        mass[names.index(name)] = value
    mass[names.index(carrier_species)] = solvent_mass
    phase = LiquidPhase(thermo_path, mass=mass_kg, temp=temperature_k,
                        mass_frac=mass / mass_kg, verbose=False)
    return phase


def _pathway_mechanism(definition, phase, operation, conditions):
    model = dict(definition["model"])
    flux_basis = model.pop('flux_basis', None)
    rule_graph = model.pop('rule_graph', ())
    constants = model.pop('constants', {})
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
        mass_transfer=transfers, flux_basis=flux_basis,
        rate_provider=definition.get('rate_provider'), rule_graph=rule_graph,
        constants=constants, conditions=conditions,
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
    required_state = ({'biomass_kg': state['biomass_kg'], 'product_g': state['product_g']}
                      if state.get('biomass_kg') is not None else {
        "viable_cells_million": state["viable_cells_million"],
        "dead_cells_million": state["dead_cells_million"],
        "product_g": state["product_g"],
        "ivcd_million_cell_day_per_ml": state["integral"],
    })
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
        quantity = event['time']
        if isinstance(quantity, Mapping):
            _require_keys(quantity, {'unit', 'value'}, 'event time')
            time_unit, quantity = quantity['unit'], quantity['value']
        value = float(quantity)
    elif time_unit in event:
        value = float(event[time_unit])
    else:
        raise ValueError(f"recipe event must declare time or {time_unit!r}")
    if not np.isfinite(value) or value < 0.0:
        raise ValueError("recipe event time must be finite and nonnegative")
    return BioreactorUnitConverter.convert(value, time_unit, 's')


def _normalize_event(event, phase):
    """Convert explicit recipe quantities once; retain legacy unit-suffixed inputs."""
    event = {key: value for key, value in event.items() if value is not None}
    for name, legacy, target in (('volume', 'volume_l', 'L'),
                                 ('volume_flow', 'volume_l_h', 'L/h'),
                                 ('density', 'density_kg_l', 'kg/L')):
        if name in event:
            if legacy in event:
                raise ValueError(f'declare {name} once')
            quantity = event.pop(name)
            _require_keys(quantity, {'unit', 'value'}, name)
            event[legacy] = BioreactorUnitConverter.convert(quantity['value'], quantity['unit'], target)
    concentration_keys = {'concentrations', 'concentrations_g_l', 'concentrations_mmol_l'} & event.keys()
    if len(concentration_keys) > 1:
        raise ValueError('declare feed concentrations once')
    if concentration_keys - {'concentrations_mmol_l'}:
        if 'concentrations' in event:
            declaration = event.pop('concentrations')
            _require_keys(declaration, {'unit', 'values'}, 'feed concentrations')
            unit, values = declaration['unit'], declaration['values']
        else:
            unit, values = 'g/L', event.pop('concentrations_g_l')
        names = list(phase.name_species)
        if not isinstance(values, Mapping) or set(values) - set(names):
            raise ValueError('feed concentrations must name phase species')
        event['concentrations_mmol_l'] = {
            name: BioreactorUnitConverter.convert(value, unit, 'mmol/L',
                                                  molecular_weight_g_mol=phase.mw[names.index(name)])
            for name, value in values.items()}
    return event


def _apply_material_event(phase, event, carrier_density_kg_l, carrier_species):
    _feed_composition(event)
    carrier_density_kg_l = float(event.get("density_kg_l", carrier_density_kg_l))
    if not np.isfinite(carrier_density_kg_l) or carrier_density_kg_l <= 0.:
        raise ValueError("feed density_kg_l must be finite and positive")
    mass = np.asarray(phase.mass_j, dtype=float).copy()
    volume_increment_l = 0.
    names = tuple(phase.name_species)
    event_type = event["event_type"]
    if event_type == 'sample':
        _require_keys(event, {'time', 'event_type', 'volume_l'}, 'sample event')
        withdrawn = BioreactorUnitConverter.convert(event['volume_l'], 'L', 'm3')
        if withdrawn < 0. or withdrawn >= phase.vol:
            raise ValueError('sample volume must be nonnegative and smaller than liquid volume')
        fraction = 1. - withdrawn / phase.vol
        mass *= fraction
        for mechanism in phase.mechanisms:
            for name in getattr(mechanism, 'extensive_states', ()):
                setattr(mechanism, name, getattr(mechanism, name) * fraction)
        volume_increment_l = -float(event['volume_l'])
    elif event_type == "feed":
        _require_keys(
            event, {"time", "event_type", "volume_l", "concentrations_mmol_l"},
            "feed event",
            optional={"composition", "volume_basis", "density_kg_l"},
        )
        volume_l = float(event["volume_l"])
        if not np.isfinite(volume_l) or volume_l < 0.0:
            raise ValueError("feed volume_l must be finite and nonnegative")
        concentrations = event["concentrations_mmol_l"]
        unknown = set(concentrations) - set(names)
        if unknown:
            raise ValueError(f"feed references unknown species {sorted(unknown)}")
        solute_mass = 0.
        for species, concentration in concentrations.items():
            concentration = float(concentration)
            if not np.isfinite(concentration) or concentration < 0.0:
                raise ValueError("feed concentrations must be finite and nonnegative")
            index = names.index(species)
            addition = concentration * volume_l * phase.mw[index] * 1e-6
            mass[index] += addition
            solute_mass += addition
        basis = event.get("volume_basis", "carrier")
        if basis not in {"carrier", "solution"}:
            raise ValueError("feed volume_basis must be carrier or solution")
        carrier_mass = volume_l * carrier_density_kg_l
        if basis == "solution":
            if carrier_species in concentrations:
                raise ValueError("solution feed concentrations must exclude the carrier")
            carrier_mass -= solute_mass
            if carrier_mass < 0.:
                raise ValueError("feed solute mass exceeds declared solution density")
        mass[names.index(carrier_species)] += carrier_mass
        volume_increment_l = volume_l
        if basis == "carrier":
            volume_increment_l += solute_mass / carrier_density_kg_l
    elif event_type == "stock_to_target":
        _require_keys(
            event, {"time", "event_type", "concentrations_mmol_l",
                    "target_species", "target_concentration_mmol_l"},
            "stock-to-target event",
            optional={"composition", "density_kg_l"},
        )
        concentrations = event["concentrations_mmol_l"]
        if set(concentrations) - (set(names) - {carrier_species}):
            raise ValueError("stock must reference known non-carrier species")
        species = event["target_species"]
        if species not in concentrations:
            raise ValueError("target species must be present in stock")
        stock = np.zeros(len(names))
        for name, value in concentrations.items():
            stock[names.index(name)] = float(value)
        target = float(event["target_concentration_mmol_l"])
        index = names.index(species)
        if (not np.isfinite(stock).all() or np.any(stock < 0.0)
                or not np.isfinite(target) or target < 0.0
                or stock[index] <= target):
            raise ValueError("stock concentrations must be nonnegative and exceed the target")
        stock_mass = stock * np.asarray(phase.mw) * 1e-6
        carrier_mass = carrier_density_kg_l - stock_mass.sum()
        if carrier_mass < 0.0:
            raise ValueError("stock solute mass exceeds declared solution density")
        stock_mass[names.index(carrier_species)] = carrier_mass
        # Constant-density, additive-volume mixing: n + Cs*dV = Ct*(V + dV).
        deficit = target * float(phase.vol) * 1000.0 - mass[index] / phase.mw[index] * 1e6
        volume_l = max(0.0, deficit / (stock[index] - target))
        mass += stock_mass * volume_l
        volume_increment_l = volume_l
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
    volume = phase.get_mechanism(WorkingVolume)
    if volume is None and event_type == "stock_to_target" and volume_increment_l > 0.:
        trial = deepcopy(phase)
        trial.updatePhase(mass_j=mass)
        expected_volume = phase.vol + volume_increment_l / 1000.
        if not np.isclose(trial.vol, expected_volume, rtol=1e-10, atol=1e-15):
            raise ValueError("stock density disagrees with native additive volume; supply consistent properties or select working-volume")
    if volume is not None:
        volume.set_volume(volume.get_volume() + volume_increment_l / 1000.)
    phase.updatePhase(mass_j=mass)


def _feed_composition(event):
    """Validate completeness metadata without inventing unreported nutrients."""
    status = event.get("composition", "unspecified")
    if status not in {"supplied", "nutrient-free", "unknown", "unspecified"}:
        raise ValueError("feed composition must be supplied, nutrient-free, or unknown")
    if "composition" in event and event["event_type"] not in {"feed", "stock_to_target", "flow"}:
        raise ValueError("composition applies only to feed and stock additions")
    concentrations = event.get("concentrations_mmol_l", {})
    if not isinstance(concentrations, Mapping) or any(
            not np.isfinite(float(x)) or float(x) < 0. for x in concentrations.values()):
        raise ValueError("feed concentrations must be a mapping of finite nonnegative values")
    if status == "nutrient-free" and any(float(x) != 0. for x in concentrations.values()):
        raise ValueError("nutrient-free additions cannot contain declared solutes")
    if status == "supplied" and not concentrations:
        raise ValueError("supplied composition requires concentrations; use nutrient-free for solvent")
    return status


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
    require_complete_inputs: bool = False
    thermo_path: str = ''
    _inlets: dict = field(default_factory=dict, repr=False)

    def reset(self):
        """Restore initial inventories and clear the assembly's scheduled inlets."""
        self._inlets.clear()
        self.unit.inlet_connections = []
        volume = self.phase.get_mechanism(WorkingVolume)
        if volume is not None:
            volume.inlet_connections = ()
            volume.volume_rate = 0.
        self.unit.reset()

    def _set_inlet(self, event):
        _require_keys(event, {'time', 'event_type', 'volume_l_h'}, 'flow event',
                      {'inlet', 'concentrations_mmol_l', 'density_kg_l', 'composition'})
        if not isinstance(self.unit, SemiBatchReactor):
            raise ValueError('continuous feed requires fed-batch operation')
        flow = BioreactorUnitConverter.convert(event['volume_l_h'], 'L/h', 'm3/s')
        if flow < 0.:
            raise ValueError('inlet flow must be nonnegative')
        name = event.get('inlet', 'inlet')
        if not isinstance(name, str) or not name:
            raise ValueError('inlet identifier must be a nonempty string')
        if flow == 0.:
            self._inlets.pop(name, None)
        else:
            names = list(self.phase.name_species)
            concentrations = event.get('concentrations_mmol_l', {})
            if set(concentrations) - (set(names) - {self.carrier_species}):
                raise ValueError('inlet solutes must name non-carrier phase species')
            density = BioreactorUnitConverter.convert(
                event.get('density_kg_l', self.carrier_density_kg_l), 'kg/L', 'kg/m3')
            conc = BioreactorUnitConverter.convert(
                [concentrations.get(n, 0.) for n in names], 'mmol/L', 'kg/m3',
                molecular_weight_g_mol=self.phase.mw)
            if density <= 0. or np.any(conc < 0.) or conc.sum() > density:
                raise ValueError('invalid inlet concentrations or solution density')
            conc[names.index(self.carrier_species)] = density - conc.sum()
            stream = LiquidStream(self.thermo_path, mass_flow=density*flow,
                                  mass_frac=conc/density, temp=self.phase.temp)
            if (self.phase.get_mechanism(WorkingVolume) is None
                    and not np.isclose(stream.vol_flow, flow, rtol=1e-10, atol=1e-15)):
                raise ValueError('inlet density disagrees with native volume; supply consistent properties or select working-volume')
            self._inlets[name] = (stream, flow)
        self.unit.inlet_connections = []
        if self._inlets:
            self.unit.Inlet = [stream for stream, _ in self._inlets.values()]
        volume = self.phase.get_mechanism(WorkingVolume)
        if volume is not None:
            volume.inlet_connections = tuple(self.unit.inlet_connections)
            volume.volume_rate = sum(flow for _, flow in self._inlets.values())

    def input_audit(self):
        """Report missing feed composition without changing legacy additions."""
        incomplete = [dict(time_s=_event_time_seconds(event, self.event_time_unit),
                           composition=_feed_composition(event))
                      for event in self.events
                      if event["event_type"] in {"feed", "stock_to_target", "flow"}
                      and not (event['event_type'] == 'flow' and event['volume_l_h'] == 0.)
                      and self.time_grid[0] <= _event_time_seconds(event, self.event_time_unit) < self.time_grid[-1]
                      and _feed_composition(event) in {"unknown", "unspecified"}]
        return {"feed_composition": "INCOMPLETE" if incomplete else "COMPLETE",
                "incomplete_additions": incomplete}

    def diagnostics(self, histories, *, depletion_tolerance_mmol=1e-9):
        """Inspect saved inventories; no additional reactor solves or rate changes."""
        if not np.isfinite(depletion_tolerance_mmol) or depletion_tolerance_mmol < 0.:
            raise ValueError("depletion tolerance must be finite and nonnegative")
        report = self.input_audit()
        names = tuple(self.phase.name_species)
        seen_positive, depleted = set(), {}
        for history in histories:
            amounts = np.asarray(history.mass_j_liquid0) / np.asarray(self.phase.mw) * 1e6
            for time, row in zip(history.time, amounts):
                if not np.isfinite(row).all():
                    raise ValueError("cannot audit nonfinite inventory history")
                for name, amount in zip(names, row):
                    if name == self.carrier_species:
                        continue
                    if amount > depletion_tolerance_mmol:
                        seen_positive.add(name)
                    elif name in seen_positive and name not in depleted:
                        depleted[name] = dict(species=name, first_depletion_time_s=float(time),
                                              amount_mmol=float(amount))
        report["conservation"] = self.mechanism.audit_conservation()
        solution = self.mechanism.last_solution
        report['optimization'] = None if solution is None else solution.diagnostics()
        report['optimization_scope'] = 'Last evaluated closure only; not a trajectory-wide feasibility audit.'
        report['volume_basis'] = ('Working volume: declared inlet flows and scheduled material operations.'
                                  if self.phase.get_mechanism(WorkingVolume) is not None else
                                  'Native phase mass and density; biological state inventories are separate.')
        if isinstance(self.mechanism, RateReconciledCulture):
            network = self.mechanism.network
            outputs = self.mechanism.definition.kinetic_outputs
            report['population_rates'] = self.mechanism.last_population_rates
            report['population_rates_scope'] = 'Last rate evaluation only, not a trajectory-wide audit.'
            provider = self.mechanism.last_provider_result
            report['kinetics_validity'] = (None if provider is None else dict(
                provider=provider.provider_id, inside_validity_domain=provider.inside_validity_domain))
            mapped = {item['reaction'] for item in self.mechanism.definition.exchange_mappings}
            mapped.update((self.mechanism.definition.product_mapping['reaction'],
                           outputs.get('growth_reaction')))
            report['tie_break_reactor_reactions'] = ([] if solution is None or solution.tie_break is None else
                [name for name in solution.tie_break['changed_reactions'] if name in mapped])
            for name, item in depleted.items():
                exchanges = [mapping["reaction"] for mapping in self.mechanism.definition.exchange_mappings
                             if mapping["species"] == name]
                metabolites = {metabolite for reaction in network.reactions
                               if reaction.reaction_id in exchanges
                               for metabolite in reaction.stoichiometry}
                item["mapped_exchanges"] = exchanges
                item["potential_consuming_reactions"] = [
                    reaction.reaction_id for reaction in network.reactions
                    if reaction.reaction_id not in network.exchange_reaction_ids
                    and any((coefficient < 0. and reaction.upper_bound > 0.)
                            or (coefficient > 0. and reaction.lower_bound < 0.)
                            for metabolite, coefficient in reaction.stoichiometry.items()
                            if metabolite in metabolites)]
        report["depletions"] = list(depleted.values())
        report["depletion_scope"] = (
            "First recorded depletion after a positive inventory; times have output-grid resolution. "
            "Reaction dependencies are structural possibilities, not proof of active flux limitation.")
        return report

    def solve(self, *, verbose=False):
        """Execute scheduled operations between native reactor solves."""
        audit = self.input_audit()
        if self.require_complete_inputs and audit["incomplete_additions"]:
            raise ValueError("complete feed compositions are required; unknown or unspecified additions remain")
        if verbose and audit["incomplete_additions"]:
            print("Incomplete feed composition: only declared solutes will be added.")
        if not self.unit.integrator._compiled:
            self.unit.compile_integrator(verbose=False)
        if not self.events:
            self.unit.solve_unit(time_grid=self.time_grid, verbose=verbose)
            return (self.unit.result,)
        scheduled = sorted(
            ((_event_time_seconds(event, self.event_time_unit), index, event)
             for index, event in enumerate(self.events)), key=lambda item: item[0])
        # Unit conversion may separate the same instant by a few floating-point
        # ulps. Preserve recipe order within that numerical instant.
        canonical = None
        for index, (time, order, event) in enumerate(scheduled):
            tolerance = 8. * np.finfo(float).eps * max(1., abs(time))
            for endpoint in (self.time_grid[0], self.time_grid[-1]):
                if abs(time - endpoint) <= tolerance:
                    time = float(endpoint)
            if canonical is None or time - canonical > tolerance:
                canonical = time
            scheduled[index] = (canonical, order, event)
        scheduled = [(time, event) for time, _, event in sorted(scheduled)]
        final_time = float(self.time_grid[-1])
        scheduled = [(time, event) for time, event in scheduled
                     if self.time_grid[0] <= time <= final_time]
        if not scheduled:
            self.unit.solve_unit(time_grid=self.time_grid, verbose=verbose)
            return (self.unit.result,)
        boundaries = sorted({float(self.time_grid[0]), final_time,
                             *(time for time, _ in scheduled)})
        histories = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            for time, event in scheduled:
                if time == start:
                    if event['event_type'] == 'flow':
                        self._set_inlet(event)
                    else:
                        _apply_material_event(
                            self.phase, event, self.carrier_density_kg_l,
                            self.carrier_species,
                        )
            tolerance = 8. * np.finfo(float).eps * max(1., abs(start), abs(end))
            grid = np.r_[start, self.time_grid[
                (self.time_grid > start + tolerance) &
                (self.time_grid < end - tolerance)], end]
            self.unit.solve_unit(time_grid=grid, verbose=verbose)
            histories.append(self.unit.result)
        return tuple(histories)


def build_bioreactor(case, definition, thermo_path):
    """Build one input-declared process as an ordinary native reactor object."""
    case, definition = normalize_forward_inputs(case, definition)
    _require_keys(case, {"biology", "numerics", "operation", "recipe"}, "case")
    _require_keys(case["biology"], {"conditions", "mechanism"}, "biology")
    _require_keys(
        case["operation"],
        {"carrier_density_kg_l", "carrier_species", "initial_mass_kg",
         "initial_volume", "mode", "runtime"},
        "operation", optional={"temperature_k", "volume_policy"},
    )
    _require_keys(case["recipe"], {"name"}, "recipe", {"require_complete_inputs", "time_unit"})
    complete = case["recipe"].get("require_complete_inputs", False)
    if not isinstance(complete, bool):
        raise ValueError("require_complete_inputs must be boolean")
    if case["biology"]["mechanism"] != definition["mechanism"]:
        raise ValueError("case and mechanism declarations do not agree")
    operation = case["operation"]
    temperature_k = operation.get("temperature_k")
    if temperature_k is None:
        temperature_k = 298.15
    temperature_k = float(temperature_k)
    if not np.isfinite(temperature_k) or temperature_k <= 0.0:
        raise ValueError("temperature_k must be finite and positive")
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
            {'rate_provider'},
        )
        if not np.isclose(
            float(state["liquid_volume_m3"]) * 1000.0, initial_volume_l,
            rtol=0.0, atol=1e-12,
        ):
            raise ValueError("pathway state volume disagrees with case initial_volume")
        phase = _phase(
            thermo_path, mass_kg, state["species_mol"], 1e-3,
            carrier_species, temperature_k,
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
            carrier_species, temperature_k,
        )
    elif mechanism_name == 'configured-kinetics':
        _require_keys(definition, {'mechanism', 'model', 'parameters', 'recipes', 'state'},
                      'configured kinetic input', {'rate_provider'})
        _require_keys(definition['model'], {'rule_graph', 'species_mass_rates', 'rate_unit'},
                      'configured kinetics', {'constants'})
        phase = _phase(thermo_path, mass_kg, state['species']['amounts'], None,
                       carrier_species, temperature_k, amount_unit=state['species']['unit'])
    else:
        raise ValueError(f"unsupported biological mechanism {mechanism_name!r}")

    conditions = dict(case["biology"]["conditions"])
    if mechanism_name == "pathway-lp":
        mechanism = _pathway_mechanism(definition, phase, operation, conditions)
    elif mechanism_name == 'configured-kinetics':
        mechanism = ConfiguredRates(phase, definition['model'], definition['parameters'], conditions,
                                    rate_provider=definition.get('rate_provider'))
    else:
        mechanism = _reconciled_mechanism(definition, conditions)
    policy = operation.get("volume_policy", "native")
    if policy not in {"native", "working-volume"}:
        raise ValueError("volume_policy must be native or working-volume")
    mechanisms = [mechanism]
    if policy == "working-volume":
        mechanisms.append(WorkingVolume(phase, initial_volume_l / 1000.))
    phase.mechanisms = mechanisms
    unit = _REACTORS[mode](integrator=_integrator(case["numerics"]),
                           isothermal=True)
    unit.Phases = phase
    unit.intraphase_processes = [
        IntraPhaseProcess(PhaseRef("liquid", 0), item) for item in mechanisms
    ]

    recipe_name = case["recipe"]["name"]
    if recipe_name not in definition["recipes"]:
        raise ValueError(f"mechanism input has no recipe {recipe_name!r}")
    events = tuple(_normalize_event(event, phase) for event in definition["recipes"][recipe_name])
    for event in events:
        _feed_composition(event)
    time_unit = case['recipe'].get('time_unit', case["numerics"]["step"]["unit"])
    if time_unit not in _TIME_FACTORS:
        raise ValueError('unsupported recipe time_unit')
    return NativeBioreactorAssembly(
        unit, phase, mechanism, _time_grid(operation, case["numerics"]),
        events, time_unit, density, carrier_species, complete, str(thermo_path),
    )
