"""Generic biological mechanisms for native PharmaPy vessel balances.

The mechanisms translate input-declared biological closures into native
``Mechanism`` rate contributions without embedding organism, pathway-count,
species, or fitted-parameter assumptions.
"""

from __future__ import annotations

from collections.abc import Mapping
import numpy as np

from PharmaPy.DataClasses import StateKey, StateVariable, TransferResult
from PharmaPy.Mechanisms import Mechanism
from PharmaPy.Metabolic.closures.base import (
    MetabolicEnvironment,
    MetabolicInfeasibleError,
)
from PharmaPy.Metabolic.closures.reconciled import (
    RateReconciledMFAClosure,
    ReconciliationTarget,
)
from PharmaPy.Metabolic.pathways import (
    ConfiguredPathwayModel,
    PathwayModelDefinition,
)

from .culture import CultureModelDefinition, CultureSnapshot
from .inventory import InventoryAvailability
from .rate_providers import RateProviderResult, build_rate_provider


class PathwayMetabolism(Mechanism):
    """Couple an arbitrary input-declared pathway LP to a native liquid phase."""

    def __init__(self, definition, biomass_kg, *, flux_time_unit="1/h",
                 mass_transfer=()):
        super().__init__()
        if isinstance(definition, Mapping):
            definition = PathwayModelDefinition.from_mapping(definition)
        if not isinstance(definition, PathwayModelDefinition):
            raise TypeError("definition must be a PathwayModelDefinition or mapping")
        if flux_time_unit not in {"1/h", "1/s"}:
            raise ValueError("flux_time_unit must be '1/h' or '1/s'")
        biomass_kg = float(biomass_kg)
        if not np.isfinite(biomass_kg) or biomass_kg <= 0.0:
            raise ValueError("biomass_kg must be finite and positive")

        self.model = ConfiguredPathwayModel(definition)
        self.biomass_kg = biomass_kg
        self.flux_time_unit = flux_time_unit
        self.mass_transfer = tuple(dict(item) for item in mass_transfer)
        self.last_solution = None
        self.last_provider_result = None
        self.solver_states = (
            StateVariable("biomass_kg", 1, "kgDW", state_type="diff"),
        )
        self._update_exposed_attributes()
        self._timers = {}

    def get_overrides(self, name=None):
        return {} if name is None else None

    def __deepcopy__(self, memo):
        """Copy runtime state while sharing the immutable pathway definition."""
        duplicate = object.__new__(type(self))
        memo[id(self)] = duplicate
        duplicate.__dict__ = self.__dict__.copy()
        duplicate._timers = {}
        return duplicate

    def get_solver_state_rates(self, *, phase, completed_state, **kwargs):
        """Optimize pathway use and return native species-mass and biomass rates."""
        phase_ref = self.solver_state_keys[0].phaseref
        biomass = float(completed_state[StateKey("biomass_kg", phase_ref)])
        volume = float(phase.vol)
        if volume <= 0.0:
            raise ValueError("bioreactor liquid volume must be positive")

        # Native mass inventory -> mol/m3, aligned by declared species name.
        concentrations_all = {
            name: float(mass / mw * 1000.0 / volume)
            for name, mass, mw in zip(phase.name_species, phase.mass_j, phase.mw)
        }
        try:
            concentrations = {
                name: concentrations_all[name] for name in self.model.state_names
            }
        except KeyError as error:
            raise ValueError(
                f"pathway state {error.args[0]!r} is absent from the owning phase"
            ) from error

        solution = self.model.solve_fluxes(concentrations, biomass / volume)
        self.last_solution = solution
        flux_time = "h" if self.flux_time_unit == "1/h" else "s"
        provider_result = RateProviderResult(
            rates={**dict(zip(self.model.state_names,
                              solution.extracellular_rates)),
                   "growth_rate": solution.growth_rate},
            units={**{name: f"mol/(kgDW {flux_time})"
                      for name in self.model.state_names},
                   "growth_rate": f"1/{flux_time}"},
            provider_id="pathway-optimization",
            version="1",
            inside_validity_domain=True,
            diagnostics={"provider_type": "constraint-based",
                         "solver_status": solution.status,
                         "solver_message": solution.message},
        )
        self.last_provider_result = provider_result
        per_second = 1.0 / 3600.0 if self.flux_time_unit == "1/h" else 1.0

        molar_sources = dict(zip(self.model.state_names, solution.extracellular_rates))
        species_mass_rate = np.zeros(phase.num_species)
        for index, (name, molecular_weight) in enumerate(
            zip(phase.name_species, phase.mw)
        ):
            # mol/(kgDW time) * kgDW -> mol/time -> kg/s
            species_mass_rate[index] = (
                molar_sources.get(name, 0.0) * biomass * per_second
                * molecular_weight / 1000.0
            )
        for transfer in self.mass_transfer:
            name = transfer["species"]
            index = phase.name_species.index(name)
            mol_per_second = (float(transfer["kla_per_s"])
                * (float(transfer["saturation_mol_m3"]) - concentrations_all[name])
                * volume)
            species_mass_rate[index] += mol_per_second * phase.mw[index] / 1000.0
        biomass_rate = solution.growth_rate * biomass * per_second

        return TransferResult(
            state_rates={
                StateKey("mass_j", phase_ref): species_mass_rate,
                StateKey("biomass_kg", phase_ref): biomass_rate,
            },
            aux={"pathway_solution": solution, "rate_provider": provider_result},
            net_mass_rate=float(species_mass_rate.sum() + biomass_rate),
        )


class RateReconciledCulture(Mechanism):
    """Reconcile an arbitrary kinetic rule graph with an input-declared network."""

    def __init__(self, definition, network, initial_state, *, conditions,
                 rate_provider, target_floor=1e-6):
        super().__init__()
        if isinstance(definition, Mapping):
            definition = CultureModelDefinition.from_mapping(definition)
        self.definition = definition
        self.network = network
        self.closure = RateReconciledMFAClosure(
            network, definition.internal_reactions, tolerance=2e-7
        )
        if not isinstance(conditions, Mapping) or not conditions:
            raise ValueError("conditions must be supplied explicitly")
        self.conditions = dict(conditions)
        outputs = definition.kinetic_outputs
        growth_reaction = outputs.get("growth_reaction")
        if growth_reaction is not None:
            if growth_reaction not in dict(outputs["exchange_fluxes"]):
                raise ValueError("growth_reaction must name a declared exchange target")
            conversion = definition.constants["growth_flux_to_per_day"]
            if not np.isfinite(conversion) or conversion <= 0.0:
                raise ValueError("growth_flux_to_per_day must be finite and positive")
        required_outputs = tuple(dict.fromkeys((
            outputs["growth"], outputs["death"],
            *dict(outputs["exchange_fluxes"]).values(),
            *(item["rate_output"] for item in definition.degradation_mappings),
        )))
        self.rate_provider = build_rate_provider(
            rate_provider, definition, required_outputs
        )
        self.target_floor = float(target_floor)
        self.previous_solution = None
        self.last_rates = None
        self.last_solution = None
        self.last_provider_result = None
        self._step_s = None

        state_names = (
            ("viable_cells_million", "10^6 cell"),
            ("dead_cells_million", "10^6 cell"),
            ("product_g", "g"),
            ("ivcd_million_cell_day_per_ml", "10^6 cell day/mL"),
        )
        self.solver_states = tuple(
            StateVariable(name, 1, units, state_type="diff")
            for name, units in state_names
        )
        for name, _ in state_names:
            value = float(initial_state[name])
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"initial {name} must be finite and nonnegative")
            setattr(self, name, value)
        self._update_exposed_attributes()
        self._timers = {}

    def get_overrides(self, name=None):
        return {} if name is None else None

    def __deepcopy__(self, memo):
        """Copy runtime state while sharing immutable network declarations."""
        duplicate = object.__new__(type(self))
        memo[id(self)] = duplicate
        duplicate.__dict__ = self.__dict__.copy()
        duplicate.closure = RateReconciledMFAClosure(
            self.network, self.definition.internal_reactions, tolerance=2e-7
        )
        duplicate._timers = {}
        return duplicate

    def reset(self):
        self.previous_solution = None

    def prepare_step(self, step_s):
        """Declare the fixed propagation interval for exact population updates."""
        self._step_s = float(step_s)

    def _targets(self, exchange_fluxes):
        targets = []
        for reaction, value in exchange_fluxes.items():
            scale = abs(value)
            if scale <= self.target_floor:
                scale = self.target_floor
            targets.append(ReconciliationTarget(reaction, value, scale))
        return tuple(targets)

    def _solve(self, exchange_fluxes, availability=None):
        targets = self._targets(exchange_fluxes)
        exclusions = set(self.definition.normalization_exclusions)
        internal_scale = abs(sum(value for reaction, value in exchange_fluxes.items()
                                 if reaction not in exclusions))
        internal_scale = max(internal_scale, self.target_floor)
        maximum = self.definition.reconciliation_relative_bound
        increment = self.definition.reconciliation_bound_increment
        fractions = np.arange(0.0, maximum - 0.5 + 0.5 * increment, increment)
        last_error = None
        network_lower, network_upper = self.network.bounds({})
        positions = {name: i for i, name in enumerate(self.network.reaction_ids)}
        for fraction in fractions:
            bounds = {}
            for target in targets:
                ends = ((0.5 - fraction) * target.value,
                        (1.5 + fraction) * target.value)
                i = positions[target.reaction_id]
                bounds[target.reaction_id] = (max(min(ends), network_lower[i]),
                                              min(max(ends), network_upper[i]))
            if any(low > high for low, high in bounds.values()):
                last_error = MetabolicInfeasibleError(
                    'target interval conflicts with declared reaction bounds',
                    network_id=self.network.network_id, lower_bounds=network_lower,
                    upper_bounds=network_upper, status='conflicting-bounds')
                continue
            environment = MetabolicEnvironment(bounds, self.conditions)
            try:
                solution = self.closure.solve(
                    environment, self.previous_solution, targets=targets,
                    internal_scale=internal_scale, availability=availability,
                )
                self.last_reconciliation_relative_bound = 0.5 + float(fraction)
                return solution
            except MetabolicInfeasibleError as error:
                last_error = error
        raise MetabolicInfeasibleError(
            "rate targets are infeasible within the explicitly declared "
            f"maximum relative reconciliation bound {maximum:g}",
            network_id=self.network.network_id,
            lower_bounds=last_error.lower_bounds,
            upper_bounds=last_error.upper_bounds,
            status=last_error.status,
            environment_context=self.conditions,
        ) from last_error

    def _availability(self, phase, rules, values):
        """Map all declared exchanges and degradation into joint inventory constraints."""
        if self._step_s is None or self._step_s <= 0.0:
            return None
        definition = self.definition
        species = list(phase.name_species)
        reactions = list(self.network.reaction_ids)
        amounts = np.asarray(phase.mass_j) / np.asarray(phase.mw) * 1e6
        matrix = np.zeros((len(species), len(reactions)))
        for mapping in definition.exchange_mappings:
            matrix[species.index(mapping['species']), reactions.index(mapping['reaction'])] += (
                -float(mapping['internal_coefficient']))
        step_day = self._step_s / definition.constants['seconds_per_day']
        other = np.zeros(len(species))
        for mapping in definition.degradation_mappings:
            source = species.index(mapping['source'])
            loss = rules[mapping['rate_output']] * amounts[source] * step_day
            other[source] -= loss
            for product, coefficient in mapping['products'].items():
                other[species.index(product)] += float(coefficient) * loss
        growth_reaction = definition.kinetic_outputs.get('growth_reaction')
        return InventoryAvailability(
            amounts, matrix, other, viable=values['viable_cells_million'],
            step_day=step_day, cell_flux_scale=definition.constants['cell_flux_to_mmol'],
            growth=rules[definition.kinetic_outputs['growth']],
            death=rules[definition.kinetic_outputs['death']],
            growth_index=None if growth_reaction is None else reactions.index(growth_reaction),
            growth_conversion=(1.0 if growth_reaction is None else
                               definition.constants['growth_flux_to_per_day']))

    def get_solver_state_rates(self, *, phase, completed_state, **kwargs):
        """Evaluate kinetics, reconcile fluxes, and return native culture balances."""
        phase_ref = self.solver_state_keys[0].phaseref
        values = {
            key.name: float(completed_state[key])
            for key in self.solver_state_keys
        }
        volume_l = float(phase.vol) * 1000.0
        concentrations = {
            name: float(mass / mw * 1e6 / volume_l)
            for name, mass, mw in zip(phase.name_species, phase.mass_j, phase.mw)
            if name in self.definition.extracellular_species
        }
        snapshot = CultureSnapshot(concentrations, volume_l, **values)
        provider_result = self.rate_provider.evaluate(snapshot, self.conditions)
        self.last_provider_result = provider_result
        rules = provider_result.rates
        outputs = self.definition.kinetic_outputs
        exchange_fluxes = {
            reaction: rules[rule]
            for reaction, rule in dict(outputs["exchange_fluxes"]).items()
        }
        solution = self._solve(exchange_fluxes, self._availability(phase, rules, values))
        self.previous_solution = solution
        self.last_solution = solution
        self.last_rates = rules
        fluxes = dict(zip(solution.reaction_ids, solution.fluxes))

        viable = values["viable_cells_million"]
        growth = rules[outputs["growth"]]
        if outputs.get("growth_reaction") is not None:
            growth = (fluxes[outputs["growth_reaction"]]
                      * self.definition.constants["growth_flux_to_per_day"])
        death = rules[outputs["death"]]
        seconds_per_day = self.definition.constants["seconds_per_day"]
        step_s = self._step_s
        if step_s is None or step_s <= 0.0:
            viable_after = viable
            average_viable = viable
            viable_rate = viable * (growth - death) / seconds_per_day
            dead_rate = viable * death / seconds_per_day
        else:
            step_day = step_s / seconds_per_day
            viable_after = viable * np.exp((growth - death) * step_day)
            average_viable = 0.5 * (viable + viable_after)
            viable_rate = (viable_after - viable) / step_s
            dead_rate = viable * (1.0 - np.exp(-death * step_day)) / step_s
        species_rate = np.zeros(phase.num_species)
        species_index = {name: i for i, name in enumerate(phase.name_species)}
        mmol_scale = self.definition.constants["cell_flux_to_mmol"]
        for mapping in self.definition.exchange_mappings:
            mmol_per_day = (-float(mapping["internal_coefficient"])
                            * fluxes[mapping["reaction"]] * average_viable * mmol_scale)
            index = species_index[mapping["species"]]
            species_rate[index] += mmol_per_day * phase.mw[index] * 1e-6 / seconds_per_day
        for mapping in self.definition.degradation_mappings:
            source = mapping["source"]
            rate = rules[mapping["rate_output"]] * concentrations[source] * volume_l
            source_index = species_index[source]
            species_rate[source_index] -= rate * phase.mw[source_index] * 1e-6 / seconds_per_day
            for product, coefficient in mapping["products"].items():
                product_index = species_index[product]
                species_rate[product_index] += (rate * float(coefficient)
                    * phase.mw[product_index] * 1e-6 / seconds_per_day)

        product = self.definition.product_mapping
        product_rate = (fluxes[product["reaction"]]
            * float(product["molecular_weight_g_mol"])
            / float(product["mass_per_cmol_g"])
            * float(product["mass_per_cmol_g"]) * average_viable
            * self.definition.constants["cell_product_scale"])
        if product_rate < 0.0:
            raise ValueError("the configured cumulative-product rate is negative")
        state_rates = {
            StateKey("mass_j", phase_ref): species_rate,
            StateKey("viable_cells_million", phase_ref): viable_rate,
            StateKey("dead_cells_million", phase_ref): dead_rate,
            StateKey("product_g", phase_ref): product_rate / seconds_per_day,
            StateKey("ivcd_million_cell_day_per_ml", phase_ref): (
                0.5 * (viable + viable_after) / (volume_l * 1000.0) / seconds_per_day
            ),
        }
        return TransferResult(
            state_rates,
            {"flux_solution": solution, "rate_provider": provider_result},
            float(species_rate.sum()),
        )
