"""Generic biological mechanisms for native PharmaPy vessel balances.

The mechanisms translate input-declared biological closures into native
``Mechanism`` rate contributions without embedding organism, pathway-count,
species, or fitted-parameter assumptions.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import SimpleNamespace
import numpy as np

from PharmaPy.DataClasses import StateKey, StateVariable, TransferResult
from PharmaPy.Mechanisms import Mechanism
from PharmaPy.Metabolic.closures.base import (
    MetabolicEnvironment,
    MetabolicInfeasibleError,
    MetabolicNumericalError,
)
from PharmaPy.Metabolic.closures.reconciled import (
    RateReconciledMFAClosure,
    ReconciliationTarget,
)
from PharmaPy.Metabolic.pathways import (
    ConfiguredPathwayModel,
    PathwayModelDefinition,
)

from .culture import (CultureModelDefinition, CultureSnapshot,
                      _validate_graph, _evaluate, _references)
from .BioreactorUnitConverter import BioreactorUnitConverter
from .inventory import InventoryAvailability, LinearInventoryAvailability
from .rate_providers import RateProviderResult, build_rate_provider


class WorkingVolume(Mechanism):
    """Additive liquid volume through native phase hooks and declared inlet flows.

    Replaces the phase's density-based volume with an integrated working-volume
    state, ``dV/dt = sum(inlet volume flows)``. Scheduled feeds and samples
    change the volume between solves. Assumes ideal additive mixing; outlets
    and phase connections are rejected because no withdrawal volume is modeled.

    Parameters
    ----------
    phase : LiquidPhase
        Owning native phase whose volume and concentration hooks are replaced.
    volume : float
        Initial working volume [m3].
    """

    def __init__(self, phase, volume):
        """Attach the working-volume state to a native phase.

        Parameters
        ----------
        phase : LiquidPhase
            Owning native phase.
        volume : float
            Initial working volume [m3].

        Raises
        ------
        ValueError
            ``volume`` is not finite and positive.
        """
        super().__init__()
        self.owning_phase = phase
        self.inlet_connections = ()
        self.volume_rate = 0.  # [m3/s]
        self.set_volume(volume)
        self.solver_states = (StateVariable("working_volume", 1, "m3", state_type="diff"),)
        self._update_exposed_attributes()

    def get_overrides(self):
        """Return the phase property hooks this mechanism replaces.

        Returns
        -------
        dict[str, Callable]
            Overrides for ``vol``, ``set_vol``, ``mass_conc``, ``mole_conc``,
            ``set_mass_conc`` and ``set_mole_conc``.
        """
        return {"vol": self.get_volume, "set_vol": self.set_volume,
                "mass_conc": self.get_mass_conc, "mole_conc": self.get_mole_conc,
                "set_mass_conc": self.set_mass_conc, "set_mole_conc": self.set_mole_conc}

    def get_volume(self):
        """Return the working volume.

        Returns
        -------
        float
            Liquid volume [m3].
        """
        return float(np.asarray(self.working_volume).item())

    def set_volume(self, value):
        """Set the working volume.

        Parameters
        ----------
        value : float or array_like
            New liquid volume [m3]; a one-element array is accepted.

        Raises
        ------
        ValueError
            ``value`` is not finite and positive.
        """
        value = float(np.asarray(value).item())
        if not np.isfinite(value) or value <= 0.:
            raise ValueError("working volume must be finite and positive")
        self.working_volume = value

    def get_mass_conc(self):
        """Return species mass concentrations on the working-volume basis.

        Returns
        -------
        ndarray, shape (n_species,)
            Mass concentrations [kg/m3].
        """
        return self.owning_phase.mass_j / self.get_volume()

    def get_mole_conc(self):
        """Return species molar concentrations on the working-volume basis.

        Returns
        -------
        ndarray, shape (n_species,)
            Molar concentrations [mol/L], equal to [kmol/m3]: [kg/m3] divided
            by ``mw`` [g/mol]. This matches the native ``mole_conc`` basis in
            ``ThermoModule.frac_to_conc``.
        """
        return self.get_mass_conc() / self.owning_phase.mw

    def set_mass_conc(self, value):
        """Set phase species masses from concentrations at the working volume.

        Parameters
        ----------
        value : array_like, shape (n_species,)
            Mass concentrations [kg/m3].

        Raises
        ------
        ValueError
            The shape is wrong, or values are nonfinite, negative or all zero.
        """
        value = np.asarray(value, dtype=float)
        if (value.shape != (self.owning_phase.num_species,)
                or not np.isfinite(value).all() or np.any(value < 0.) or not value.any()):
            raise ValueError("supply finite nonnegative concentrations for every phase species")
        self.owning_phase.mass_j = value * self.get_volume()

    def set_mole_conc(self, value):
        """Set phase species masses from molar concentrations at the working volume.

        Parameters
        ----------
        value : array_like, shape (n_species,)
            Molar concentrations [mol/L], the native ``mole_conc`` basis.

        Raises
        ------
        ValueError
            The shape is wrong, or the derived mass concentrations are invalid.
        """
        value = np.asarray(value, dtype=float)
        if value.shape != (self.owning_phase.num_species,):
            raise ValueError("supply concentrations for every phase species")
        self.set_mass_conc(value * self.owning_phase.mw)

    def update_state(self, completed_state, unit=None, **kwargs):
        """Accept a completed solver state after checking the unit's connections.

        Parameters
        ----------
        completed_state : Mapping
            Native completed state passed to ``Mechanism.update_state``.
        unit : MultiPhaseVessel, optional
            Owning unit, used to verify connections.
        **kwargs
            Forwarded to ``Mechanism.update_state``.

        Raises
        ------
        ValueError
            The unit's inlets differ from the declared flows, or it has outlet or
            phase connections, or the updated volume is invalid.
        """
        if unit is not None and (tuple(unit.inlet_connections) != self.inlet_connections or unit.outlet_connections
                                 or unit.phase_connections):
            raise ValueError("working-volume requires declared inlet flows and no outlet or phase connections")
        super().update_state(completed_state, **kwargs)
        self.set_volume(self.working_volume)

    def get_solver_state_rates(self, **kwargs):
        """Return the working-volume derivative.

        Parameters
        ----------
        **kwargs
            Native mechanism arguments; unused.

        Returns
        -------
        TransferResult
            ``d(working_volume)/dt`` [m3/s]; no species mass is transferred.
        """
        return TransferResult(state_rates={self.solver_state_keys[0]: self.volume_rate},
                              aux={}, net_mass_rate=0.)


class ConfiguredRates(Mechanism):
    """Map user-defined volumetric kinetic rules to native species mass balances.

    Each phase species named in ``species_mass_rates`` receives the volumetric
    rate of its rule, converted from ``rate_unit`` to [kg/(m3 s)] and
    multiplied by liquid volume to give [kg/s]. Unnamed species receive zero.
    No elemental balance is declared, so conservation is reported as not
    checked.

    Parameters
    ----------
    phase : LiquidPhase
        Owning native phase.
    model : Mapping
        ``rule_graph``, ``species_mass_rates`` (species -> rule) and
        ``rate_unit`` (volumetric, such as ``'mmol/(L h)'``); optional
        ``constants``.
    parameters : Mapping[str, float]
        Flattened named parameters.
    conditions : Mapping[str, float]
        Declared operating conditions.
    rate_provider : Mapping, optional
        Alternative provider declaration; it must return the same outputs in
        ``rate_unit``.
    """

    def __init__(self, phase, model, parameters, conditions, *, rate_provider=None):
        """Validate the rule graph and build the declared rate provider.

        Parameters
        ----------
        phase : LiquidPhase
            Owning native phase.
        model : Mapping
            Configured-kinetics model section.
        parameters : Mapping[str, float]
            Flattened named parameters.
        conditions : Mapping[str, float]
            Declared operating conditions.
        rate_provider : Mapping, optional
            Provider declaration; defaults to the built-in rule graph.

        Raises
        ------
        ValueError
            Outputs reference unknown species or rules, or the rate unit is not a
            supported volumetric rate.
        """
        super().__init__()
        self.owning_phase = phase
        self.conditions = dict(conditions)
        self.definition = SimpleNamespace(rule_graph=model['rule_graph'],
                                          parameters=parameters, constants=model.get('constants', {}))
        _validate_graph(self.definition, () if rate_provider is not None else model['species_mass_rates'].values())
        self.outputs = dict(model['species_mass_rates'])
        if not set(self.outputs) <= set(phase.name_species):
            raise ValueError('Rate outputs must map phase species to declared rules.')
        self.rate_scale = BioreactorUnitConverter.convert(  # [kg/(m3 s) per rate_unit]
            np.ones(len(phase.mw)), model['rate_unit'], 'kg/(m3 s)', molecular_weight_g_mol=phase.mw)
        self.last_solution = None

        units = {name: model['rate_unit'] for name in self.outputs.values()}
        declaration = rate_provider if rate_provider is not None else {
            'type': 'rule-graph', 'output_units': units}
        self.rate_provider = build_rate_provider(declaration, self.definition, tuple(units))
        self.expected_rate_units = units
        self.last_provider_result = None

    def get_overrides(self, name=None):
        """Declare that no phase property hooks are replaced.

        Parameters
        ----------
        name : str, optional
            Property name being queried.

        Returns
        -------
        dict or None
            Empty mapping for the full query; None for a named property.
        """
        return {} if name is None else None

    def audit_conservation(self):
        """Report that configured rules carry no elemental balance.

        Returns
        -------
        dict
            ``{'status': 'NOT_CHECKED', 'reason': ...}``.
        """
        return {'status': 'NOT_CHECKED', 'reason': 'Configured kinetic rules do not declare elemental balances.'}

    def get_solver_state_rates(self, phase, process, **kwargs):
        """Evaluate the provider and return native species mass rates.

        Parameters
        ----------
        phase : LiquidPhase
            Current phase; supplies ``mass_conc`` [kg/m3], ``mw`` [g/mol] and
            ``vol`` [m3].
        process : IntraPhaseProcess
            Native process holding the phase reference.
        **kwargs
            Native mechanism arguments; unused.

        Returns
        -------
        TransferResult
            Species mass rates [kg/s] and the provider result in ``aux``.

        Raises
        ------
        ValueError
            The provider returns units other than the configured rate unit.
        """
        snapshot = SimpleNamespace(concentrations_mmol_l=dict(zip(
            phase.name_species, BioreactorUnitConverter.convert(
                phase.mass_conc, 'kg/m3', 'mmol/L', molecular_weight_g_mol=phase.mw))),
            volume_l=BioreactorUnitConverter.convert(phase.vol, 'm3', 'L'))
        result = self.rate_provider.evaluate(snapshot, self.conditions)
        if dict(result.units) != self.expected_rate_units:
            raise ValueError('provider output units disagree with configured volumetric rates')
        self.last_provider_result = result
        rates = result.rates
        mass_rates = np.array([rates[self.outputs[n]] if n in self.outputs else 0.  # [kg/s]
                               for n in phase.name_species]) * self.rate_scale * phase.vol
        return TransferResult({StateKey('mass_j', process.phaseref): mass_rates},
                              {'rate_provider': result}, float(mass_rates.sum()))


class PathwayMetabolism(Mechanism):
    """Couple an arbitrary input-declared pathway LP to a native liquid phase.

    At each rate evaluation the pathway LP is solved for the current
    concentrations and biomass. Accepted pathway fluxes give species amount
    rates ``biomass * exchange_scale * per_second * S @ v`` [mol/s] and
    specific growth ``g @ v`` [1/flux time]. Inventory constraints prevent
    uptake from an exhausted pool beyond its actual supply.

    Parameters
    ----------
    definition : PathwayModelDefinition or Mapping
        Pathway LP declaration.
    biomass_kg : float
        Initial dry biomass [kgDW].
    flux_time_unit : str, optional
        Pathway flux time unit. Default ``'1/h'``.
    mass_transfer : Sequence[Mapping], optional
        Gas-liquid transfer terms with ``species``, ``kla_per_s`` [1/s] and
        ``saturation_mol_m3`` [mol/m3].
    flux_basis : Mapping, optional
        Declared ``exchange`` and ``growth`` bases.
    rate_provider : Mapping, optional
        Provider for ``provider``-type uptake limits.
    rule_graph : Sequence, optional
        Rules used by the uptake provider.
    constants : Mapping, optional
        Constants used by the uptake provider.
    conditions : Mapping, optional
        Operating conditions passed to the provider.
    """

    extensive_states = ('biomass_kg',)

    requires_inlet_rates = True

    def __init__(self, definition, biomass_kg, *, flux_time_unit="1/h",
                 mass_transfer=(), flux_basis=None, rate_provider=None, rule_graph=(),
                 constants=None, conditions=None):
        """Validate the pathway declaration, transport and optional uptake provider.

        Parameters
        ----------
        definition : PathwayModelDefinition or Mapping
            Pathway LP declaration.
        biomass_kg : float
            Initial dry biomass [kgDW].
        flux_time_unit : str, optional
            Pathway flux time unit. Default ``'1/h'``.
        mass_transfer : Sequence[Mapping], optional
            Transfer terms (``kla_per_s`` [1/s], ``saturation_mol_m3`` [mol/m3]).
        flux_basis : Mapping, optional
            Declared exchange and growth bases.
        rate_provider : Mapping, optional
            Uptake-limit provider declaration.
        rule_graph : Sequence, optional
            Provider rules.
        constants : Mapping, optional
            Provider constants.
        conditions : Mapping, optional
            Operating conditions.

        Raises
        ------
        TypeError
            ``definition`` has the wrong type.
        ValueError
            Biomass, transport coefficients, provider pairing or a transfer-cap
            declaration is invalid. A transfer cap must equal
            ``kLa * C_sat`` within relative 1e-12, which allows only roundoff
            between the two declarations of the same product.
        """
        super().__init__()
        if isinstance(definition, Mapping):
            definition = PathwayModelDefinition.from_mapping(definition)
        if not isinstance(definition, PathwayModelDefinition):
            raise TypeError("definition must be a PathwayModelDefinition or mapping")
        biomass_kg = float(biomass_kg)
        if not np.isfinite(biomass_kg) or biomass_kg <= 0.0:
            raise ValueError("biomass_kg must be finite and positive")

        self.unit_converter = BioreactorUnitConverter(flux_time_unit=flux_time_unit, flux_basis=flux_basis)
        self.model = ConfiguredPathwayModel(definition, exchange_scale=self.unit_converter.exchange_scale)
        outputs = tuple(dict.fromkeys(r['output'] for r in definition.uptake_constraints
                                      if r['type'] == 'provider'))
        if bool(outputs) != (rate_provider is not None):
            raise ValueError('provider uptake rules and rate_provider must be supplied together')
        self.conditions = dict(conditions or {})
        provider_definition = SimpleNamespace(rule_graph=rule_graph, constants=constants or {},
                                              parameters=definition.parameters)
        _validate_graph(provider_definition, ())
        self.uptake_provider = (build_rate_provider(rate_provider, provider_definition, outputs)
                                if outputs else None)
        self.uptake_outputs = outputs
        self.biomass_kg = biomass_kg
        self.flux_time_unit = flux_time_unit
        self.mass_transfer = tuple(dict(item) for item in mass_transfer)
        for transfer in self.mass_transfer:
            if transfer['species'] not in definition.state_ids:
                raise ValueError('transport species must be a declared pathway state')
            values = [transfer['kla_per_s'], transfer['saturation_mol_m3']]
            if not np.isfinite(values).all() or min(values) < 0.:
                raise ValueError('transport coefficients must be finite and nonnegative')
        for rule in definition.uptake_constraints:
            if rule['type'] != 'transfer-cap':
                continue
            transfers = [item for item in self.mass_transfer if item['species'] == rule['species']]
            if transfers:
                parameters = definition.parameters
                expected = (parameters[rule['mass_transfer_parameter']]
                            * parameters[rule['saturation_parameter']] * self.unit_converter.per_second)
                actual = sum(item['kla_per_s'] * item['saturation_mol_m3'] for item in transfers)
                if not np.isclose(expected, actual, rtol=1e-12, atol=0.):
                    raise ValueError('transfer-cap supply disagrees with actual transport; update both declarations')
        self._step_s = None
        self.last_solution = None
        self.last_provider_result = None
        self.solver_states = (
            StateVariable("biomass_kg", 1, "kgDW", state_type="diff"),
        )
        self._update_exposed_attributes()
        self._timers = {}

    def get_overrides(self, name=None):
        """Declare that no phase property hooks are replaced.

        Parameters
        ----------
        name : str, optional
            Property name being queried.

        Returns
        -------
        dict or None
            Empty mapping for the full query; None for a named property.
        """
        return {} if name is None else None

    def prepare_step(self, step_s):
        """Declare the fixed propagation interval used for end-inventory constraints.

        Parameters
        ----------
        step_s : float
            Next fixed step [s].
        """
        self._step_s = float(step_s)

    def reset(self):
        """Clear the fixed step and the last accepted solution."""
        self._step_s = None
        self.last_solution = None

    def audit_conservation(self):
        """Report the scope of conservation evidence for the reduced pathway map.

        Returns
        -------
        dict
            Status fields stating that stoichiometric and elemental balances are
            not assessed, the flux basis source, the closed-uptake screen, and
            whether fixed-step or boundary inventory constraints are active.
        """
        return dict(stoichiometric_consistency='NOT_ASSESSED', elemental_balance='NOT_ASSESSED',
                    flux_basis='DECLARED' if self.unit_converter.flux_basis else 'FIXED_CONTRACT', representation='reduced',
                    scope='Reduced exchange/growth map; intracellular and elemental balance are not verified.',
                    closed_uptake=self.model.audit_closed_uptake(),
                    inventory_constraints='FIXED_STEP' if self._step_s else 'BOUNDARY',
                    growth_coupled_to_flux=True, untracked_exchanges=None)

    def __deepcopy__(self, memo):
        """Copy runtime state while sharing the immutable pathway definition.

        Parameters
        ----------
        memo : dict
            ``copy.deepcopy`` memo dictionary.

        Returns
        -------
        PathwayMetabolism
            Shallow-copied mechanism with fresh timers.
        """
        duplicate = object.__new__(type(self))
        memo[id(self)] = duplicate
        duplicate.__dict__ = self.__dict__.copy()
        duplicate._timers = {}
        return duplicate

    def get_solver_state_rates(self, *, phase, completed_state, **kwargs):
        """Optimize pathway use and return native species-mass and biomass rates.

        Parameters
        ----------
        phase : LiquidPhase
            Current phase; supplies ``mass_j`` [kg], ``mw`` [g/mol] and ``vol``
            [m3].
        completed_state : Mapping
            Native completed state; supplies ``biomass_kg`` [kgDW].
        **kwargs
            ``inlet_rates`` maps the phase ``mass_j`` key to inlet mass rates
            [kg/s], counted as actual supply in inventory constraints.

        Returns
        -------
        TransferResult
            Species mass rates [kg/s], biomass rate [kgDW/s], and the pathway
            solution and rate-provider result in ``aux``.

        Raises
        ------
        ValueError
            The volume is not positive, a pathway state is absent from the phase,
            or provider uptake units disagree with mol/(kgDW time).
        MetabolicInfeasibleError
            The LP has no feasible flux.
        """
        phase_ref = self.solver_state_keys[0].phaseref
        biomass = float(completed_state[StateKey("biomass_kg", phase_ref)])  # [kgDW]
        volume = float(phase.vol)  # [m3]
        if volume <= 0.0:
            raise ValueError("bioreactor liquid volume must be positive")

        # Native mass inventory -> mol/m3, aligned by declared species name.
        concentrations_all = {  # [mol/m3]
            name: float(mass / mw * 1000.0 / volume)
            for name, mass, mw in zip(phase.name_species, phase.mass_j, phase.mw)
        }
        try:
            concentrations = {  # [mol/m3]
                name: concentrations_all[name] for name in self.model.state_names
            }
        except KeyError as error:
            raise ValueError(
                f"pathway state {error.args[0]!r} is absent from the owning phase"
            ) from error

        indices = [phase.name_species.index(name) for name in self.model.state_names]
        amounts = np.array([concentrations[name] * volume for name in self.model.state_names])  # [mol]
        transfer_rates = np.zeros(len(indices))  # [mol/s]
        for transfer in self.mass_transfer:
            index = self.model.state_names.index(transfer['species'])
            transfer_rates[index] += (transfer['kla_per_s']  # [mol/s]
                * (transfer['saturation_mol_m3'] - concentrations[transfer['species']]) * volume)
        inlet_mass = kwargs.get('inlet_rates', {}).get(StateKey('mass_j', phase_ref),
                                                      np.zeros(phase.num_species))
        supply = self.unit_converter.convert(np.asarray(inlet_mass)[indices], 'kg', 'mol',  # [mol/s]
                                             molecular_weight_g_mol=np.asarray(phase.mw)[indices])
        source_matrix = (self.model.source_matrix * self.unit_converter.exchange_scale  # [mol/s per flux unit]
                         * biomass * self.unit_converter.per_second)
        if self._step_s:
            availability = LinearInventoryAvailability(
                amounts, source_matrix * self._step_s, (transfer_rates + supply) * self._step_s)
        else:
            availability = LinearInventoryAvailability.boundary(amounts, source_matrix, transfer_rates + supply)
        uptake_result = None
        limits = None
        if self.uptake_provider is not None:
            snapshot = SimpleNamespace(concentrations_mmol_l=concentrations_all,
                                       volume_l=volume * 1000., biomass_kg=biomass)
            uptake_result = self.uptake_provider.evaluate(snapshot, self.conditions)
            expected_unit = f'mol/(kgDW {self.flux_time_unit[2:]})'
            if dict(uptake_result.units) != dict.fromkeys(self.uptake_outputs, expected_unit):
                raise ValueError('provider uptake units must be mol per kgDW per declared time unit')
            limits = {name: value / self.unit_converter.exchange_scale  # [flux unit]
                      for name, value in uptake_result.rates.items()}
        solution = self.model.solve_fluxes(concentrations, biomass / volume, availability=availability,
                                           uptake_limits=limits)
        self.last_solution = solution
        flux_time = self.flux_time_unit[2:]
        provider_result = RateProviderResult(
            rates={**dict(zip(self.model.state_names,
                              solution.extracellular_rates * self.unit_converter.exchange_scale)),
                   "growth_rate": solution.growth_rate},
            units={**{name: f"mol/(kgDW {flux_time})"
                      for name in self.model.state_names},
                   "growth_rate": f"1/{flux_time}"},
            provider_id="pathway-optimization",
            version="1",
            inside_validity_domain=(uptake_result.inside_validity_domain if uptake_result else True),
            diagnostics={"provider_type": "constraint-based",
                         "solver_status": solution.status,
                         "solver_message": solution.message, "uptake_provider": uptake_result},
        )
        self.last_provider_result = provider_result
        per_second = self.unit_converter.per_second  # [1/s per 1/flux time]

        species_mass_rate = np.zeros(phase.num_species)  # [kg/s]
        species_mass_rate[indices] = self.unit_converter.mass_rates(
            source_matrix @ solution.fluxes + transfer_rates, np.asarray(phase.mw)[indices],
            amount_unit='mol', time_unit='s')
        biomass_rate = solution.growth_rate * biomass * per_second  # [kgDW/s]

        return TransferResult(
            state_rates={
                StateKey("mass_j", phase_ref): species_mass_rate,
                StateKey("biomass_kg", phase_ref): biomass_rate,
            },
            aux={"pathway_solution": solution, "rate_provider": provider_result},
            net_mass_rate=float(species_mass_rate.sum() + biomass_rate),
        )


class RateReconciledCulture(Mechanism):
    """Reconcile an arbitrary kinetic rule graph with an input-declared network.

    Kinetic rules predict exchange fluxes; a steady-state network closure
    reconciles them subject to ``S v = 0``, bounds and inventory feasibility.
    Accepted fluxes, times the population and ``exchange_scale``, give species
    sources. The population is cell count [10^6 cell] or dry biomass [kgDW];
    unit comments write either basis as [population].
    On fixed steps the population follows ``X exp((mu - kd) dt)`` and sources
    use the trapezoidal mean population.

    Parameters
    ----------
    definition : CultureModelDefinition or Mapping
        Kinetic and reconciliation declaration.
    network : MetabolicNetworkDefinition
        Steady-state reaction network.
    initial_state : Mapping[str, float]
        Initial population, product [g] and, for cell counts, dead cells and
        IVCD.
    conditions : Mapping[str, float]
        Declared operating conditions; required and nonempty.
    rate_provider : Mapping
        Kinetic provider declaration.
    target_floor : float, optional
        Legacy residual floor in network coordinates.
    """

    requires_inlet_rates = True
    extensive_states = ('viable_cells_million', 'dead_cells_million', 'product_g')

    def __init__(self, definition, network, initial_state, *, conditions,
                 rate_provider, target_floor=None):
        """Validate mappings, compile flux bases and initialize population states.

        Parameters
        ----------
        definition : CultureModelDefinition or Mapping
            Kinetic and reconciliation declaration.
        network : MetabolicNetworkDefinition
            Steady-state reaction network.
        initial_state : Mapping[str, float]
            Initial states: ``viable_cells_million``, ``dead_cells_million`` [10^6
            cell], ``product_g`` [g], ``ivcd_million_cell_day_per_ml`` [10^6 cell
            day/mL]; or ``biomass_kg`` [kgDW] and ``product_g``.
        conditions : Mapping[str, float]
            Declared operating conditions.
        rate_provider : Mapping
            Kinetic provider declaration.
        target_floor : float, optional
            Legacy residual floor in network coordinates.

        Raises
        ------
        ValueError
            Mappings reference unknown species or reactions, exchange roles
            disagree with the mappings, conditions are missing, a death increment
            is malformed, population states are mixed, or an initial state is
            negative or nonfinite.

        Notes
        -----
        The closure tolerance ``2e-7`` [-] is the relative balance and bound
        tolerance used to accept a reconciled flux; it is a numerical setting of
        this mechanism, not a biological parameter.
        """
        super().__init__()
        if isinstance(definition, Mapping):
            definition = CultureModelDefinition.from_mapping(definition)
        self.definition = definition
        self.network = network
        if definition.kinetic_outputs.get('death_increment') is not None and not definition.flux_basis:
            raise ValueError('death_increment requires declared flux_basis')
        self.population_state = ('biomass_kg' if initial_state.get('biomass_kg') is not None
                                 else 'viable_cells_million')
        self.unit_converter = BioreactorUnitConverter(definition, population_basis=(
            'kgDW' if self.population_state == 'biomass_kg' else '10^6 cell'))
        reactions = set(network.reaction_ids)
        referenced = (set(definition.internal_reactions)
                      | set(definition.kinetic_outputs["exchange_fluxes"])
                      | {definition.product_mapping["reaction"]})
        for mapping in definition.exchange_mappings:
            referenced.add(mapping["reaction"])
            if mapping["species"] not in definition.extracellular_species:
                raise ValueError("exchange mapping references an unknown extracellular species")
            if not np.isfinite(float(mapping["internal_coefficient"])):
                raise ValueError("exchange mapping coefficient must be finite")
        if referenced - reactions:
            raise ValueError(f"culture mappings reference unknown reactions: {sorted(referenced - reactions)}")
        mapped = {item["reaction"] for item in definition.exchange_mappings}
        mapped.add(definition.product_mapping["reaction"])
        mapped.add(definition.kinetic_outputs.get("growth_reaction"))
        for reaction, role in network.exchange_roles.items():
            if (role == "tracked") != (reaction in mapped):
                raise ValueError("exchange role disagrees with reactor inventory mapping")
        self.closure = RateReconciledMFAClosure(
            network, (() if definition.reconciliation_policy != "relative-regularized"
                      else definition.internal_reactions), tolerance=2e-7
        )
        if not isinstance(conditions, Mapping) or not conditions:
            raise ValueError("conditions must be supplied explicitly")
        self.conditions = dict(conditions)
        outputs = definition.kinetic_outputs
        growth_reaction = outputs.get("growth_reaction")
        if growth_reaction is not None:
            if growth_reaction not in dict(outputs["exchange_fluxes"]):
                raise ValueError("growth_reaction must name a declared exchange target")
            conversion = self.unit_converter.growth_scale
            if not np.isfinite(conversion) or conversion <= 0.0:
                raise ValueError("growth_flux_to_per_day must be finite and positive")
        required_outputs = tuple(dict.fromkeys((
            outputs["growth"], outputs["death"],
            *((outputs["dead_removal"],) if outputs.get("dead_removal") else ()),
            *dict(outputs["exchange_fluxes"]).values(),
            *(item["rate_output"] for item in definition.degradation_mappings),
        )))
        self.death_increment = outputs.get('death_increment')
        if self.death_increment is not None:
            declaration = self.death_increment
            if (not isinstance(declaration, Mapping) or set(declaration) != {'unit', 'expression'}
                    or declaration['unit'] != '1/day' or not definition.flux_basis):
                raise ValueError('death_increment requires expression, unit 1/day and declared flux_basis')
            expression = declaration['expression']
            if _references(expression, 'accepted_flux') - set(outputs['exchange_fluxes']):
                raise ValueError('death_increment references an undeclared exchange flux')
            if _references(expression) - set(required_outputs):
                raise ValueError('death_increment references an unavailable provider output')
        self.rate_provider = build_rate_provider(
            rate_provider, definition, required_outputs
        )
        groups = {name: 'exchange' for name in outputs['exchange_fluxes']
                  if name not in definition.untargeted_exchanges}
        if definition.product_mapping['reaction'] in groups:
            groups[definition.product_mapping['reaction']] = 'product'
        if growth_reaction in groups:
            groups[growth_reaction] = 'growth'
        if definition.reconciliation_policy == 'unit-scaled':
            self.target_scales = self.unit_converter.reconciliation_target_scales(
                definition.reconciliation_scaling, groups)
        else:
            self.target_floors, self.target_floor, self.internal_scale = (
                self.unit_converter.reconciliation_scales(
                    definition.reconciliation_scaling, groups, target_floor))
        self.previous_solution = None
        self.last_rates = None
        self.last_solution = None
        self.last_provider_result = None
        self.last_population_rates = None
        self._step_s = None

        state_names = (
            ("viable_cells_million", "10^6 cell"),
            ("dead_cells_million", "10^6 cell"),
            ("product_g", "g"),
            ("ivcd_million_cell_day_per_ml", "10^6 cell day/mL"),
        )
        if self.population_state == 'biomass_kg':
            if any(initial_state.get(name) is not None for name in (
                    'viable_cells_million', 'dead_cells_million', 'ivcd_million_cell_day_per_ml')):
                raise ValueError('declare either biomass or cell-count population states')
            cell_states = {'viable_cells_million', 'dead_cells_million',
                           'ivcd_million_cell_day_per_ml', 'viable_cell_density_million_ml'}
            if any(_references(rule['expression'], 'state') & cell_states
                   for rule in definition.rule_graph):
                raise ValueError('cell-count rules require a cell-count population')
            if definition.kinetic_outputs.get('dead_removal'):
                raise ValueError('dead-cell removal requires a cell-count population')
            state_names = (('biomass_kg', 'kgDW'), ('product_g', 'g'))
            self.extensive_states = ('biomass_kg', 'product_g')
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

    def audit_conservation(self):
        """Report network consistency and exchanges without reactor inventories.

        Returns
        -------
        dict
            Network stoichiometric audit, flux-basis source, representation and
            scope statement, closed-uptake screen, boundary exchanges not mapped to
            reactor species, the active inventory-constraint mode, and whether
            growth and death are coupled to accepted fluxes.
        """
        report = self.network.audit_conservation()
        report["flux_basis"] = "DECLARED" if self.definition.flux_basis else "LEGACY_UNVERIFIED"
        report["representation"] = self.network.representation
        report["scope"] = ("Structural test of declared species, not complete chemical or elemental closure; "
                           "reduced networks can omit cofactor counterparts and other material pools.")
        report["closed_uptake"] = self.network.audit_closed_uptake()
        mapped = {item["reaction"] for item in self.definition.exchange_mappings}
        mapped.add(self.definition.product_mapping["reaction"])
        growth = self.definition.kinetic_outputs.get("growth_reaction")
        if growth is not None:
            mapped.add(growth)
        report["untracked_exchanges"] = [
            {"reaction": reaction.reaction_id,
             "stoichiometry": dict(reaction.stoichiometry),
             "role": self.network.exchange_roles.get(reaction.reaction_id, "unspecified"),
             "lower_bound": reaction.lower_bound,
             "upper_bound": reaction.upper_bound}
            for reaction in self.network.reactions
            if reaction.reaction_id in self.network.exchange_reaction_ids
            and reaction.reaction_id not in mapped]
        report["inventory_constraints"] = (
            "FIXED_STEP" if self._step_s is not None and self._step_s > 0.
            else "BOUNDARY")
        report["growth_coupled_to_flux"] = growth is not None
        report['post_reconciliation_death'] = self.death_increment is not None
        return report

    def get_overrides(self, name=None):
        """Declare that no phase property hooks are replaced.

        Parameters
        ----------
        name : str, optional
            Property name being queried.

        Returns
        -------
        dict or None
            Empty mapping for the full query; None for a named property.
        """
        return {} if name is None else None

    def __deepcopy__(self, memo):
        """Copy runtime state while sharing immutable network declarations.

        Parameters
        ----------
        memo : dict
            ``copy.deepcopy`` memo dictionary.

        Returns
        -------
        RateReconciledCulture
            Shallow-copied mechanism with its own closure and timers.
        """
        duplicate = object.__new__(type(self))
        memo[id(self)] = duplicate
        duplicate.__dict__ = self.__dict__.copy()
        duplicate.closure = RateReconciledMFAClosure(
            self.network, self.closure.internal_reaction_ids,
            tolerance=self.closure.tolerance,
        )
        duplicate._timers = {}
        return duplicate

    def reset(self):
        """Clear the warm start, last solution, population rates and fixed step."""
        self.previous_solution = None
        self.last_solution = None
        self.last_population_rates = None
        self._step_s = None

    def prepare_step(self, step_s):
        """Declare the fixed propagation interval for exact population updates.

        Parameters
        ----------
        step_s : float
            Next fixed step [s].
        """
        self._step_s = float(step_s)

    def _targets(self, exchange_fluxes):
        """Build relative-regularized reconciliation targets.

        Parameters
        ----------
        exchange_fluxes : Mapping[str, float]
            Kinetic target fluxes in network coordinates.

        Returns
        -------
        tuple of ReconciliationTarget
            Targets with scale ``max(|value|, floor)`` in network coordinates.
        """
        targets = []
        for reaction, value in exchange_fluxes.items():
            scale = max(abs(value), self.target_floors[reaction])  # [network flux]
            targets.append(ReconciliationTarget(reaction, value, scale))
        return tuple(targets)

    def _solve(self, exchange_fluxes, availability=None):
        """Reconcile kinetic targets under the declared policy.

        Parameters
        ----------
        exchange_fluxes : Mapping[str, float]
            Kinetic target fluxes in network coordinates; untargeted exchanges are
            removed.
        availability : LinearInventoryAvailability, optional
            Inventory feasibility constraint.

        Returns
        -------
        FluxSolution
            Accepted network fluxes.

        Raises
        ------
        MetabolicInfeasibleError
            No target interval up to the declared maximum relative bound is
            feasible.

        Notes
        -----
        The relative-regularized policy widens each target interval from
        ``[0.5 v, 1.5 v]`` by ``reconciliation_bound_increment`` until
        ``reconciliation_relative_bound``; 0.5 is the fixed starting half-width
        [-] of that declared search.
        """
        exchange_fluxes = {reaction: value for reaction, value in exchange_fluxes.items()
                           if reaction not in self.definition.untargeted_exchanges}
        if self.definition.reconciliation_policy == 'unit-scaled':
            common = max(abs(value) / self.target_scales[reaction]  # [-]
                         for reaction, value in exchange_fluxes.items()) or 1.
            targets = tuple(ReconciliationTarget(reaction, value, self.target_scales[reaction] * common)
                            for reaction, value in exchange_fluxes.items())
            self.last_reconciliation_relative_bound = None
            return self.closure.solve(
                MetabolicEnvironment({}, self.conditions), self.previous_solution,
                targets=targets, internal_scale=1., availability=availability)
        if self.definition.reconciliation_policy == "unweighted":
            # One common scale conditions the objective without changing its minimizer.
            scale = max(max(abs(value) for value in exchange_fluxes.values()), self.target_floor)  # [network flux]
            targets = tuple(ReconciliationTarget(reaction, value, scale)
                            for reaction, value in exchange_fluxes.items())
            self.last_reconciliation_relative_bound = None
            return self.closure.solve(
                MetabolicEnvironment({}, self.conditions), self.previous_solution,
                targets=targets, internal_scale=1., availability=availability)
        targets = self._targets(exchange_fluxes)
        exclusions = set(self.definition.normalization_exclusions)
        internal_scale = abs(sum(value for reaction, value in exchange_fluxes.items()  # [network flux]
                                 if reaction not in exclusions))
        internal_scale = (max(internal_scale, self.target_floor) if self.internal_scale is None
                          else self.internal_scale)
        maximum = self.definition.reconciliation_relative_bound
        increment = self.definition.reconciliation_bound_increment
        fractions = np.arange(0.0, maximum - 0.5 + 0.5 * increment, increment)  # [-]
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

    def _availability(self, phase, rules, values, matrix, other, supply):
        """Map all declared exchanges and degradation into joint inventory constraints.

        Parameters
        ----------
        phase : LiquidPhase
            Current phase; supplies ``mass_j`` [kg] and ``mw`` [g/mol].
        rules : Mapping[str, float]
            Provider rates, including growth and death [1/day] and degradation
            rates [1/day].
        values : Mapping[str, float]
            Current mechanism states; the population entry is used.
        matrix : ndarray, shape (n_species, n_reactions)
            Species change per unit network flux [-].
        other : ndarray, shape (n_species,)
            Degradation source rates [mmol/day] for boundary constraints.
        supply : ndarray, shape (n_species,)
            Inlet supply [mmol/day].

        Returns
        -------
        LinearInventoryAvailability, InventoryAvailability or None
            Boundary rate constraints without a fixed step; end-of-step inventory
            constraints with population exposure on a fixed step.
        """
        definition = self.definition
        reactions = list(self.network.reaction_ids)
        growth_reaction = definition.kinetic_outputs.get('growth_reaction')
        amounts = np.asarray(phase.mass_j) / np.asarray(phase.mw) * 1e6  # [mmol]
        if self._step_s is None or self._step_s <= 0.0:
            return LinearInventoryAvailability.boundary(
                amounts, matrix * values[self.population_state] * self.unit_converter.exchange_scale,
                other + supply, growth_index=None if growth_reaction is None else reactions.index(growth_reaction))
        step_day = self._step_s / definition.constants['seconds_per_day']  # [day]
        # Preserve the propagation scheme's established multiplication order;
        # near depletion, roundoff can select a different constrained optimum.
        increment = supply * step_day  # [mmol]
        species = list(phase.name_species)
        for mapping in definition.degradation_mappings:
            source = species.index(mapping['source'])
            loss = rules[mapping['rate_output']] * amounts[source] * step_day  # [mmol]
            increment[source] -= loss
            for product, coefficient in mapping['products'].items():
                increment[species.index(product)] += float(coefficient) * loss
        return InventoryAvailability(
            amounts, matrix, increment, viable=values[self.population_state],
            step_day=step_day, cell_flux_scale=self.unit_converter.exchange_scale,
            growth=rules[definition.kinetic_outputs['growth']],
            death=rules[definition.kinetic_outputs['death']],
            growth_index=None if growth_reaction is None else reactions.index(growth_reaction),
            growth_conversion=(1.0 if growth_reaction is None else
                               self.unit_converter.growth_scale))

    def get_solver_state_rates(self, *, phase, completed_state, **kwargs):
        """Evaluate kinetics, reconcile fluxes, and return native culture balances.

        Parameters
        ----------
        phase : LiquidPhase
            Current phase; supplies ``mass_j`` [kg], ``mw`` [g/mol] and ``vol``
            [m3].
        completed_state : Mapping
            Native completed state with the population, product and IVCD states.
        **kwargs
            ``inlet_rates`` maps the phase ``mass_j`` key to inlet mass rates
            [kg/s], counted as supply in inventory constraints.

        Returns
        -------
        TransferResult
            Species mass rates [kg/s]; population rates [10^6 cell/s] or
            [kgDW/s]; product rate [g/s]; IVCD rate [10^6 cell day/(mL s)]; and the
            flux solution and provider result in ``aux``.

        Raises
        ------
        ValueError
            Provider units disagree with the flux basis, a death increment,
            dead-cell removal or product rate is negative.
        MetabolicNumericalError
            A post-reconciliation death increment invalidates fixed-step
            inventories.
        MetabolicInfeasibleError
            Targets cannot be reconciled within the declared bounds.
        """
        phase_ref = self.solver_state_keys[0].phaseref
        values = {
            key.name: float(completed_state[key])
            for key in self.solver_state_keys
        }
        volume_l = float(phase.vol) * 1000.0  # [L]
        concentrations = {  # [mmol/L]
            name: float(mass / mw * 1e6 / volume_l)
            for name, mass, mw in zip(phase.name_species, phase.mass_j, phase.mw)
            if name in self.definition.extracellular_species
        }
        snapshot_values = values
        if self.population_state == 'biomass_kg':
            snapshot_values = dict(viable_cells_million=0., dead_cells_million=0.,
                                   ivcd_million_cell_day_per_ml=0., **values)
        snapshot = CultureSnapshot(concentrations, volume_l, **snapshot_values)
        provider_result = self.rate_provider.evaluate(snapshot, self.conditions)
        self.unit_converter.validate_rate_units(provider_result.units)
        self.last_provider_result = provider_result
        rules = provider_result.rates
        outputs = self.definition.kinetic_outputs
        exchange_fluxes = {
            reaction: rules[rule]
            for reaction, rule in dict(outputs["exchange_fluxes"]).items()
        }
        species_index = {name: i for i, name in enumerate(phase.name_species)}
        reaction_index = {name: i for i, name in enumerate(self.network.reaction_ids)}
        matrix = np.zeros((phase.num_species, len(reaction_index)))  # [-]
        for mapping in self.definition.exchange_mappings:
            matrix[species_index[mapping['species']], reaction_index[mapping['reaction']]] += (
                -float(mapping['internal_coefficient']))
        other = np.zeros(phase.num_species)  # [mmol/day]
        for mapping in self.definition.degradation_mappings:
            source = mapping['source']
            loss = rules[mapping['rate_output']] * concentrations[source] * volume_l  # [mmol/day]
            other[species_index[source]] -= loss
            for product, coefficient in mapping['products'].items():
                other[species_index[product]] += float(coefficient) * loss
        inlet_mass = kwargs.get('inlet_rates', {}).get(StateKey('mass_j', phase_ref),
                                                      np.zeros(phase.num_species))
        supply = self.unit_converter.convert(inlet_mass, 'kg', 'mmol',  # [mmol/day]
                                             molecular_weight_g_mol=phase.mw) * self.definition.constants['seconds_per_day']
        solution = self._solve(exchange_fluxes, self._availability(
            phase, rules, values, matrix, other, supply))
        self.previous_solution = solution
        self.last_solution = solution
        self.last_rates = rules
        fluxes = dict(zip(solution.reaction_ids, solution.fluxes))

        viable = values[self.population_state]  # [10^6 cell] or [kgDW]
        growth = rules[outputs["growth"]]  # [1/day]
        if outputs.get("growth_reaction") is not None:
            growth = (fluxes[outputs["growth_reaction"]]  # [1/day]
                      * self.unit_converter.growth_scale)
        death = rules[outputs["death"]]  # [1/day]
        extra_death = 0.0  # [1/day]
        if self.death_increment is not None:
            accepted = {name: fluxes[name] * self.unit_converter.exchange_scale
                        for name in outputs['exchange_fluxes']}
            product_reaction = self.definition.product_mapping['reaction']
            if product_reaction in accepted:
                accepted[product_reaction] = (fluxes[product_reaction]
                    * self.unit_converter.product_mass * self.unit_converter.product_scale)
            if outputs.get('growth_reaction') is not None:
                accepted[outputs['growth_reaction']] = growth
            extra_death = float(_evaluate(self.death_increment['expression'], rules,  # [1/day]
                self.definition, snapshot, self.conditions, accepted))
            if not np.isfinite(extra_death) or extra_death < 0.0:
                raise ValueError('death_increment must be finite and nonnegative')
            # Recheck the smaller exposure: less biological production can
            # matter when it offsets an independent degradation source term.
            death += extra_death
            if self._step_s is not None and self._step_s > 0.0:
                actual_rules = dict(rules, **{outputs['death']: death})
                availability = self._availability(phase, actual_rules, values, matrix, other, supply)
                if not self.closure._acceptable(solution.fluxes, solution.lower_bounds,
                                               solution.upper_bounds, availability):
                    raise MetabolicNumericalError('post-reconciliation death invalidates step inventories; '
                                                 'reduce the step or use adaptive integration')
        self.last_population_rates = dict(growth=growth, basal_death=rules[outputs['death']],
            death_increment=extra_death, death=death, net_growth=growth-death,
            growth_source='accepted_flux' if outputs.get('growth_reaction') else 'kinetics')
        dead_removal = (rules[outputs["dead_removal"]]  # [1/day]
                        if outputs.get("dead_removal") else 0.0)
        if dead_removal < 0.0:
            raise ValueError("the configured dead-cell removal rate is negative")
        seconds_per_day = self.definition.constants["seconds_per_day"]  # [s/day]
        step_s = self._step_s
        if step_s is None or step_s <= 0.0:
            viable_after = viable  # [population]
            average_viable = viable  # [population]
            viable_rate = viable * (growth - death) / seconds_per_day  # [population/s]
            dead_rate = (viable * death - values.get("dead_cells_million", 0.)  # [10^6 cell/s]
                         * dead_removal) / seconds_per_day
        else:
            step_day = step_s / seconds_per_day  # [day]
            viable_after = viable * np.exp((growth - death) * step_day)  # [population]
            average_viable = 0.5 * (viable + viable_after)  # [population]
            viable_rate = (viable_after - viable) / step_s  # [population/s]
            dead_rate = (viable * (1.0 - np.exp(-death * step_day))  # [10^6 cell/s]
                         - values.get("dead_cells_million", 0.)
                         * (1.0 - np.exp(-dead_removal * step_day))) / step_s
        mmol_scale = self.unit_converter.exchange_scale  # [mmol/(population day) per flux unit]
        mmol_per_day = (matrix @ solution.fluxes) * average_viable * mmol_scale  # [mmol/day]
        species_rate = self.unit_converter.mass_rates(  # [kg/s]
            mmol_per_day, phase.mw, amount_unit='mmol', time_unit='s') / seconds_per_day
        for mapping in self.definition.degradation_mappings:
            source = mapping['source']
            rate = rules[mapping['rate_output']] * concentrations[source] * volume_l  # [mmol/day]
            index = species_index[source]
            species_rate[index] -= self.unit_converter.mass_rates(
                rate, phase.mw[index], amount_unit='mmol', time_unit='s') / seconds_per_day
            for product, coefficient in mapping['products'].items():
                index = species_index[product]
                species_rate[index] += self.unit_converter.mass_rates(
                    rate * float(coefficient), phase.mw[index],
                    amount_unit='mmol', time_unit='s') / seconds_per_day

        product = self.definition.product_mapping
        product_rate = (fluxes[product["reaction"]]  # [g/day]
            * self.unit_converter.product_mass * average_viable * self.unit_converter.product_scale)
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
        if self.population_state == 'biomass_kg':
            state_rates = {key: value for key, value in state_rates.items()
                           if key.name in {'mass_j', 'product_g'}}
            state_rates[StateKey('biomass_kg', phase_ref)] = viable_rate
        return TransferResult(
            state_rates,
            {"flux_solution": solution, "rate_provider": provider_result},
            float(species_rate.sum()),
        )
