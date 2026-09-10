
"""
Created on Fri July 10 2026

@author: zhillma
Refactored the code by dcasasor
"""
from PharmaPy.Phases import classify_phases, SolidPhase, LiquidPhase, VaporPhase, BasePhase
from PharmaPy.Streams import LiquidStream, SolidStream, VaporStream
from PharmaPy.MixedPhases import Slurry, SlurryStream, MixedPhase, MixedStream

from PharmaPy.ProcessControl_Refactor import Controller
from PharmaPy.Results import DynamicResult



import copy
import numpy as np
import os
from PharmaPy.DataClasses import *
from time import perf_counter



eps = np.finfo(float).eps

class MultiPhaseVessel():
    def __init__(self,integrator=None,temp_ref=273.15,
     isothermal=False, reset_states=False, controller=Controller(), h_conv=0, 
      state_events={},
      adiabatic=False,jac_type="AD",Phases=None,
      basis='mass_j',ht_mode="jacket",diam=0,area_base=0):
      

        if isothermal and controller is not None:
            assert 'global_temp' not in controller.states and 'temp' not in controller.states, "Cannot change the temperature of an isothermal unit"

        self.basis = basis
        self.adiabatic = adiabatic
        self.isothermal = isothermal

        self.jac_type = jac_type
        

        self.controller = controller #TODO ZZ refactor analyze_controls to give a Controls dataclass, an empty one if controls None
        self.oper_mode = None #This is not called within the class, but is used by pharmapy to handle connections (either 'batch' or 'continuous', etc.)
        
        
        # Phase init
        self._phase_connections = []
        if Phases is not None:
            self.Phases = Phases
        self._intraphase_processes = []

        #heat transfer
        self.area_ht = None
        self._Utility = None
        self.ht_mode = ht_mode
        self.h_conv = h_conv
        self.diam = diam
        self.area_base = area_base
        self.temp_ref = temp_ref #enthalpy ref

        #State events
        if state_events is None:
            state_events = []
        self.state_event_list = state_events

        #state initialization, all types
        self._initialize_states(reset_states)
        
        #port initialization
        self._inlet_connections = []
        self._outlet_connections = []

        #Integrator
        self.integrator = integrator




    @property
    def Phases(self):
        return self._Phases
    
    @Phases.setter
    def Phases(self, phases):
        self._Phases = MixedPhase(phases)
        self._post_set_phases()
        
    

    def _basis_units(self):

        units = {
            "mass_j": "kg",
            "mass_conc": "kg/m3",
            "mole_j": "kmol",
            "mole_conc": "kmol/m3"
        }

        return units[self.basis]
    
    def _material_state_definition(self)->StateVariable:

        return StateVariable(
            name=self.basis,
            dim=self.Phases.num_species,
            units=self._basis_units(),
            index=self.Phases.name_species,
            
        )

    def _post_set_phases(self):
        self.define_material_states()
        self.initialize_defualt_states()
        self.configure_default_connections()
        self.nomenclature() 
    def configure_default_connections(self):
        """Hook for subclasses to create default outlet connections."""
        pass
    def initialize_defualt_states(self):
        self.default_diff_states_from_phases()
    def _initialize_state_collections(self):

        self.phase_states = PhaseStateCollection()

        # States exposed to solver
        self.solver_state_collection = StateCollection()

        # States exposed as outputs
        self.output_state_collection = StateCollection()

    @property
    def Inlet(self):
        raise AttributeError("Inlet is a convenience API for setting inlet_connections")


    @Inlet.setter
    def Inlet(self, inlet):

        inlet= inlet if isinstance(inlet,(list,tuple)) else [inlet]

        self._create_default_connections('inlet_connections',inlet)
    
    @property
    def inlet_connections(self)->list[StreamConnection]:
        return self._inlet_connections


    @inlet_connections.setter
    def inlet_connections(self, connections):

        if not isinstance(connections, (list,tuple)):
            raise TypeError(
                "inlet_connections should be a list or tuple"
            )

        if not all(
            isinstance(c, StreamConnection)
            for c in connections
        ):
            raise TypeError(
                "inlet_connections must contain StreamConnection objects")

        self._inlet_connections = connections
    def _create_default_connections(self,connection_attr, inlet_streams):

        connections = []

        for stream in inlet_streams:

            if not isinstance(stream, MixedStream):
                stream = MixedStream(stream)

            mappings = []

            counts = {}

            for phase in stream:

                phase_type = (phase.phase_family.lower())

                idx = counts.get(phase_type, 0)
                counts[phase_type] = idx + 1

                ref = PhaseRef(phase_type, idx)

                mappings.append(
                    PhaseMapping(
                        source_phaseref=ref,
                        sink_phaseref=ref
                    )
                )

            connections.append(
                StreamConnection(
                    stream=stream,
                    phase_mappings=mappings
                )
            )
        setattr(self,connection_attr,connections)

    @property
    def outlet_connections(self)->list[StreamConnection]:
        return self._outlet_connections
    
    @outlet_connections.setter
    def outlet_connections(self,connections):

        if not isinstance(connections, (list,tuple)):
            raise TypeError("outlet_connections should be a list or tuple")
        
        if not all(isinstance(c, StreamConnection) for c in connections):
            raise TypeError(
                "outlet_connections must contain StreamConnection objects")

        self._outlet_connections = connections

    @property
    def Outlet(self):

        if self.outlet_conditions is not None:

            if len(self.outlet_conditions.streams) == 1:
                return self.outlet_conditions.streams[0].stream

            raise AttributeError(
                "Multiple outlet streams exist."
            )

        # Backward compatibility before solve
        if len(self.outlet_connections) == 1:
            return self.outlet_connections[0].stream

        raise AttributeError(
            "Multiple outlet connections exist. "
            "Use outlet_connections instead.")
    
    @Outlet.setter
    def Outlet(self,outlet):
        outlet = outlet if isinstance(outlet,(list,tuple)) else [outlet]
        self._create_default_connections('outlet_connections',outlet)
    

    def define_material_states(self):


        material_state = self._material_state_definition()
        counts = {}
        for i, phase in enumerate(self.Phases):
            phase_type = phase.phase_family.lower()
            idx=counts.get(phase_type,0)
            counts[phase_type] = idx+1
            phase_ref = PhaseRef(
                phase_type=phase_type,
                index=idx
            )

            self.phase_states.add(
                phase_ref,
                copy.deepcopy(material_state)
            )

    def default_diff_states_from_phases(self):
        """
        Mark the default material state for every phase as differential.

        Unit operations that require different behavior should override this
        method or modify `phase_states` after phase initialization.
        """
        # do nothing if the phases are already marked diff
        if any(
            state.state_type == "diff"
            for phasestate in self.phase_states
            for state in self.phase_states[phasestate.phaseref].states.values()
        ):
            return
        for phasestatevar in self.phase_states:
            if phasestatevar.state.name==self.basis:
                phasestatevar.state.update_variable('state_type','diff')

    @property
    def phase_connections(self)->list[PhaseConnection]:
        return self._phase_connections
    
    @phase_connections.setter
    def phase_connections(self,connections:list):
        if not all([isinstance(c,PhaseConnection) for c in connections]):
            raise TypeError("phase_connections should all be PhaseConnection objects")
        if not isinstance(connections,list):
            raise TypeError("phase_connections is expected to be a list")
        self._phase_connections = connections
        self.nomenclature(overwrite=True)

    @property
    def intraphase_processes(self)->list[IntraPhaseProcess]:
        return self._intraphase_processes
    
    @intraphase_processes.setter
    def intraphase_processes(self, regions):

        if not isinstance(regions, list):
            raise TypeError(
                "reaction_regions is expected to be a list"
            )

        if not all(
            isinstance(r, IntraPhaseProcess)
            for r in regions
        ):
            raise TypeError(
                "reaction_regions should all be "
                "ReactionRegion objects"
            )

        self._intraphase_processes = regions

        self.nomenclature(overwrite=True)
    
    @property
    def Utility(self):
        return self._Utility

    @Utility.setter
    def Utility(self, utility):
        self.u_ht = 1 / (1 / self.h_conv + 1 / utility.h_conv)
        self._Utility = utility
        self.output_state_collection.add(
            StateVariable(
                name="q_ht",
                dim=1,
                units="W",
                state_type="post",
                compute_value=self.compute_qht_value
            ),overwrite=True
        )
        

    
    def complete_state(self, state:dict, time:float)->dict[StateKey]:

        completed = state.copy()

        controlled = self.controller.compute_states(
            time=time,
            completed_state=completed,
            unit=self,
        )

        completed.update(controlled)

        required_states = (
            self.solver_state_collection.states
            | self.output_state_collection.states
        )

        for statekey, variable in required_states.items():

            if statekey in completed:
                continue

            
            # Pull from phases/state sources
            if statekey == StateKey("global_temp"):
                value = self.get_default_temperature()
            else:
                value = self.get_state_value(statekey)

            if value is not None:
                completed[statekey] = value

        return completed
    def get_phase_ref(self, phase):
        for phase_ref in self.phase_states.phasestates:

            if self.Phases.get_phase_from_ref(phase_ref) is phase:
                return phase_ref

        raise ValueError(
            "The phase associated with the mechanism is not a phase "
            "belonging to this vessel."
        )
    def get_default_temperature(self):
        return self.Phases[0].temp
    
    def get_state_value(self, statekey):

        # phase-associated state
        if statekey.phaseref is not None:

            phase = self.phase_states.get_phase(statekey.phaseref)

            return getattr(phase, statekey.name, None)

        # unit operation state
        return getattr(self, statekey.name, None)
            
    def __getattr__(self, name):
        # For Backward compatability 
        # You should not use phase_# explicitly, 
        # everywhere should always iterate over all phases
        #Exception is setting the default phase_connections 
        # since those use the same assumptions as PharmaPy 1.0
        if name.startswith("Liquid_"):
            idx = int(name.split("_")[1]) - 1
            return self.Phases.Liquids[idx]

        if name.startswith("Solid_"):
            idx = int(name.split("_")[1]) - 1
            return self.Phases.Solids[idx]

        if name.startswith("Vapor_"):
            idx = int(name.split("_")[1]) - 1
            return self.Phases.Vapors[idx]

        raise AttributeError(name) #TODO Check if this raises unwanted errors
       
    def _initialize_states(self,reset=False):

        self.reset_states = reset

        self.state_variables = StateCollection()

        self.input_states = StateCollection()

        self.output_states = StateCollection()

        self.elapsed_time = 0

        self.result = None
        self._initial_solver_state = None
        self._initialize_state_collections()

        
        
    def reset(self):

        self.controller.reset()

        self.elapsed_time = 0
        self.result = None

        if self._initial_solver_state is not None:
            self.update_phases_from_state(self._initial_solver_state)

        for process in self.intraphase_processes:
            process.mechanism.reset()

        for connection in self.phase_connections:
            if connection.mechanism is not None:
                connection.mechanism.reset()
    
    @property
    def has_energy_balance(self):
        anytemp = any(StateKey('temp',phaseref) in self.controller.states for phaseref in self.phase_states.phasestates.keys())
        return (
            not self.isothermal
            and StateKey("global_temp") not in self.controller.states and not anytemp
        )
    @property
    def has_utility_balance(self):
        return (
            self.has_energy_balance
            and self.Utility is not None
            and not self.adiabatic
        )
    
    def define_solver_states(self,overwrite=False):

        
        for phase_state in self.phase_states:
            if phase_state.state.state_type!='diff': 
                continue
            state_copy = copy.deepcopy(phase_state.state)
            state_copy.phaseref = (phase_state.phaseref)
            self.solver_state_collection.add(state_copy,overwrite)
        
        if self.has_energy_balance:

            self.solver_state_collection.add(
                StateVariable(
                    name="global_temp",
                    dim=1,
                    units="K",
                    state_type='diff'
                ),overwrite
            )
            
        for connection in self.phase_connections:

            if connection.mechanism is not None:

                mechanism = connection.mechanism

                phase_ref = None

                if mechanism.owning_phase is not None:
                    phase_ref = self.get_phase_ref(mechanism.owning_phase)

                mechanism.add_solver_state_variables(
                    self.solver_state_collection,
                    overwrite=overwrite,
                    phase_ref=phase_ref,
                )

        for process in self.intraphase_processes:
            process.mechanism.add_solver_state_variables(
                self.solver_state_collection,overwrite
            )
    def define_output_states(self,overwrite=False):
        self.recorded_output_history ={}
        self.output_state_collection.add(
            StateVariable(
                name="Total_m_in_vessel",
                dim=1,
                units="kg/s",
                state_type="post",
                compute_value=self.compute_total_mass_in_vessel
            ),overwrite
        )

        if not self.has_energy_balance:
            self.output_state_collection.add(
                StateVariable(
                                name="global_temp",
                                dim=1,
                                units="K",
                                state_type='post',
                            ),overwrite
                        )
        
        for process in self.intraphase_processes:

            phase = self.Phases.get_phase_from_ref(process.phaseref)
            process.mechanism.add_output_state_variables(
                self.output_state_collection,
                overwrite=overwrite,
                process=process
            )
            self.output_state_collection.add(
                StateVariable(
                    name="mole_conc",
                    dim=phase.num_species,
                    units="kmol/m3",
                    state_type="post",
                    index=phase.name_species,
                    phaseref=process.phaseref,
                    compute_value=self.compute_mole_conc_value
                ),
                overwrite
            )

        

        for conn in self.phase_connections:

            if conn.mechanism is not None:
                conn.mechanism.add_output_state_variables(
                    self.output_state_collection,overwrite
                )

    def nomenclature(self,overwrite=False):

        self.define_solver_states(overwrite)
        self.define_output_states(overwrite)
        self.name_states = self.solver_state_collection.names()
        self.dim_states = self.solver_state_collection.dims()


    @staticmethod
    def compute_total_mass_in_vessel(
            state_var,
            time,
            completed_state,
            context,
            resolved_inlets=None,
            resolved_outlets=None,
            operating_conditions=None,
        ):
        total_mass = 0
        for phase in context.Phases:
            total_mass+= phase.mass
        return total_mass
        
    @staticmethod
    def compute_outlet_massflow_value(
            state_var,
            time,
            completed_state,
            context,
            resolved_inlets=None,
            resolved_outlets=None,
            operating_conditions=None,
        ):

        if resolved_outlets is None:
            return 0.0

        m_flow = 0.0

        for connection in resolved_outlets.streams:
            for transfer in connection:
                m_flow += transfer.species_flow.sum()

        return m_flow
    
        
    @staticmethod
    def compute_qht_value(
            state_var,
            time,
            completed_state,
            context,
            resolved_inlets=None,
            resolved_outlets=None,
            operating_conditions=None,
        ):
        temp = completed_state[StateKey("global_temp")]
        temp_ht = context.Utility.temp_in

        return context.get_heat_transfer_rate(
            temp,
            temp_ht,
        )
    
    @staticmethod
    def compute_mole_conc_value(
            state_var,
            time,
            completed_state,
            context,
            resolved_inlets=None,
            resolved_outlets=None,
            operating_conditions=None,
        ):

        phase_ref = state_var.phaseref

        mass_key = StateKey(
            context.basis,
            phase_ref,
        )

        mass = completed_state[mass_key]

        phase = context.phase_states.get_phase(
            phase_ref
        )

        # kmol/m3
        return mass / phase.mw / phase.vol
    def update_phases_from_state(self, completed_state):

        global_temp = completed_state.get(StateKey("global_temp"))

        for phase_ref,collection in self.phase_states.phasestates.items():

            phase = self.phase_states.get_phase(phase_ref)

            updates = {}

            for variable in collection.states.values():
                key = StateKey(variable.name, phase_ref)

                if key in completed_state:
                    updates[variable.name] = completed_state[key]

            if global_temp is not None and "temp" not in updates:
                updates["temp"] = global_temp
            t0 = perf_counter()
            phase.update_from_solver_state(
                updates,
                completed_state,
                unit=self
            )
            self._timers['update_phases_from_solver_state'] = self._timers.get('update_phases_from_solver_state',0)+perf_counter()-t0

    def pack_state_rates(self, material_rates, global_rates=None):

        buffer = self._solver_rate_buffer
        buffer.fill(0.0)

        material_slices = self.solver_state_collection.material_slices
        solver_slices = self.solver_state_collection.slices

        for key in self.solver_state_collection.material_keys:
            try:
                buffer[solver_slices[key]] = material_rates[material_slices[key]]
            except KeyError:
                try:
                    buffer[solver_slices[key]] = np.asarray(global_rates[key]).reshape(-1)
                except KeyError:
                    raise KeyError(f"StateKey {key} not found in material_rates or global_rates")

        return buffer

    def save_initial_solver_state(self, states=None, time=None):
        """
        Save the completed solver state corresponding to the initial condition.

        The state is saved only if an initial state has not already been
        established. This prevents subsequent solves from redefining the
        original state after the phases have mutated.
        """
        if hasattr(self, "_initial_solver_state") and self._initial_solver_state is not None:
            return self._initial_solver_state

        if states is None:
            states = self.create_solver_init_states()

        if time is None:
            time = self.elapsed_time

        unpacked_state = self.solver_state_collection.unpack(states)
        completed_state = self.complete_state(unpacked_state,time)

        self._initial_solver_state = completed_state

        return completed_state
    def unit_model(self, time, states, params=None, sw=None,
                    mat_bce=False, enrgy_bce=False,alg_bce=False, limiter_dt=None):
        if not hasattr(self, "model_call_count"):
            self.model_call_count = 0
            self._timers={}
        self.model_call_count += 1
        limiter_dt = limiter_dt if limiter_dt is not None else 1.0
        t0=perf_counter()
        unpacked_state = self.solver_state_collection.unpack(states)
        self._timers['unpack'] = self._timers.get('unpack',0)+perf_counter()-t0

        t0=perf_counter()
        completed_state = self.complete_state(unpacked_state,time)
        self._timers['complete_state'] = self._timers.get('complete_state',0)+perf_counter()-t0

        t0=perf_counter()
        self.update_phases_from_state(completed_state)
        self._timers['update_phases'] = self._timers.get('update_phases',0)+perf_counter()-t0
        # Balances
        t0=perf_counter()
        material_rates, material_buffer = self.material_balances(
            time,completed_state, limiter_dt=limiter_dt)
        self._timers['material_balances_total'] = self._timers.get('material_balances_total',0)+perf_counter()-t0

        if mat_bce:
            return self.pack_state_rates(material_rates)
        global_rates = {}
        t0 = perf_counter()
        energy_rates = self.energy_balances(time,completed_state, material_buffer)
        self._timers['energy_balances_total'] = self._timers.get('energy_balances_total',0)+perf_counter()-t0
        global_rates.update(energy_rates)

        # utility_rates = self.utility_energy_balance(
        #     time,completed_state)
        # global_rates.update(utility_rates)

        if enrgy_bce:
            return self.pack_state_rates(global_rates=global_rates)
        t0 = perf_counter()
        balances = self.pack_state_rates(material_rates=material_rates,
                                        global_rates=global_rates)
        self._timers['pack_state_rates'] = self._timers.get('pack_state_rates',0)+perf_counter()-t0
        assert len(balances) == len(states), (
            f"Returned {len(balances)} derivatives "
            f"for {len(states)} solver states."
        )

        self.derivatives = balances
        return balances
        # algebraic_residuals = self.algebraic_balances(
        #     time,
        #     completed_state,
        #     operating_conditions,
        # )

        # if alg_bce:
        #     return algebraic_residuals
        # if not algebraic_residuals:
        #     # legacy behavior
        #     return balances
        # else:
        #     return balances, algebraic_residuals


    def compile_structure(self):

        """Compiles the array-based representation of the unit operation's state variables and their relationships to the underlying phases and mechanisms.
        This method should be called after the unit operation's phases, mechanisms, and state variables have been defined, but before any simulation is run.
        It prepares the internal data structures for efficient numerical computation."""

        # State layouts
        self.solver_state_collection.compile()
        self.output_state_collection.compile()

        # Phase layouts
        self.phase_states.compile(self.Phases)

        # Persistent numerical buffers
        self._material_contributions = MaterialContributionBuffer(
            self.solver_state_collection.material_dim
        )

        self._solver_rate_buffer = np.empty(
            self.solver_state_collection.dim
        )

        # PhaseRef -> this phase's material inventory slice
        self._material_slice_by_phase = {}

        for key, material_slice in (
            self.solver_state_collection.material_slices.items()
        ):
            if key.name == self.basis:
                self._material_slice_by_phase[key.phaseref] = material_slice

        # Default cross-phase transfer mechanisms
        for connection in self._phase_connections:
            if connection.mechanism is None:
                connection.mechanism = DirectTransfer()

    def compile_integrator(self, **kwargs):
        self.compile_structure()
        return self.integrator.compile_integrator(
            self,
            **kwargs
        ) # TODO I don't think this gets called, though it may if the user wants to use fast solve and not solve_unit

    def configure_solver(self):
        pass

    def create_solver_init_states(self):
        return self.solver_state_collection.pack(
            self.complete_state({},0))

    def solve_unit(
        self,
        runtime=None,
        time_grid=None,
        **kwargs,
    ):
        self.compile_structure()
        return self.integrator.solve(
            self,
            runtime=runtime,
            time_grid=time_grid,
            **kwargs,
        )

    
    def initialize_rate_dictionary(self)->dict[StateKey,Any]:

        rates = {}

        for key, state in self.solver_state_collection.states.items():

            if state.state_type != 'diff':
                continue

            rates[key] = np.zeros(state.dim)

        return rates
    def get_operating_conditions(self,time:float,completed_state:dict[StateKey])->tuple[StreamConditions,StreamConditions,dict[OperatingKey]]:
        # --------------------------------------------
        # Controller sees current vessel state
        # --------------------------------------------

        self.controller.observe(time,completed_state,self)

        operating_conditions=self.controller.compute_operating_conditions(
                time,
                completed_state,
                self
            )
        # --------------------------------------------
        # Resolve inlet after possible inlet control
        # --------------------------------------------
        resolved_inlets = self.resolve_inlets(completed_state,operating_conditions)
        # --------------------------------------------
        # Controller observes actual inlet
        # --------------------------------------------
        self.controller.observe(time,completed_state,self,resolved_inlets=resolved_inlets)


        operating_conditions.update(
            self.controller.compute_operating_conditions(
                time,completed_state,self,resolved_inlets=resolved_inlets))


        # Re-resolve inlet in case controller changed it
        resolved_inlets = self.resolve_inlets(completed_state,operating_conditions)

        return resolved_inlets,operating_conditions

    def get_events(self):
        events = []

        events.extend(self.controller.get_events(self))

        for connection in self.phase_connections:
            if connection.mechanism is not None:
                events.extend(
                    connection.mechanism.get_events(self)
                )

        for process in self.intraphase_processes:
            events.extend(
                process.mechanism.get_events(self)
            )

        return events
    
    def material_balances(
        self,
        time:float,
        completed_state:dict[StateKey],
        limiter_dt=1.0
    ):

        resolved_inlets,operating_conditions = self.get_operating_conditions(time,completed_state)
        # --------------------------------------------
        # Material balances
        # --------------------------------------------
        buffer = self.limit_material_rates(time,completed_state,resolved_inlets,operating_conditions,limiter_dt=limiter_dt)


        rates = self.sum_material_contributions(buffer.contributions)

        return rates, buffer

    def check_negative_inventory(
            self,
            rates,
            completed_state,
            limiter_dt=1.0,
            inventory_atol=1e-12):

        violations = {}

        for state_key,material_slice in self.solver_state_collection.material_slices.items():
            if not self.solver_state_collection.states[state_key].limit_negative_inventory:
                continue

            inventory = completed_state[state_key]
            rate = rates[material_slice]

            mask = (inventory + rate * limiter_dt < -inventory_atol)

            if np.any(mask):
                violations[state_key] = mask

        return violations
    
    def calculate_scale(
            self,
            violations,
            completed_state,
            buffer,
            limiter_dt=1.0,
            inventory_atol=1e-12
        ):

        scalable_terms = [
            buffer.CROSS_PHASE,
            buffer.OUTLET,
        ]

        fixed_terms = [
            buffer.INLET,
            buffer.INTRAPHASE,
        ]

        scales = {}

        # Only phases that violated need correction
        for state_key in violations:

            material_slice = self.solver_state_collection.material_slices[state_key]

            inventory = completed_state[state_key]

            fixed = (
                buffer.contributions[buffer.INLET, material_slice]
                + buffer.contributions[buffer.INTRAPHASE, material_slice]
            )

            scalable = (
                buffer.contributions[buffer.CROSSPHASE, material_slice]
                + buffer.contributions[buffer.OUTLET, material_slice]
            )

            # How much inventory remains after unavoidable mechanisms
            allowable = inventory + fixed*limiter_dt


            violating = violations[state_key] & (scalable < 0)

            species_scales = np.ones_like(inventory, dtype=float)

            species_scales[violating] = (
                allowable[violating]
                / (-scalable[violating] * limiter_dt + eps)
            )

            phase_scale = np.min(species_scales)



            scales[state_key] = np.full_like(
                inventory,
                phase_scale,
                dtype=float
            )


        return scales,scalable_terms,fixed_terms
    
    def scale_phase_inventory(
            self,
            scales):

        for state_key, scale_vector in scales.items():

            phase = self.phase_states.get_phase(
                state_key.phaseref
            )

            new_mass = (
                phase.mass_j
                * scale_vector
            )

            phase.updatePhase(
                mass_j=new_mass
            )
                
    def restore_effective_inventory(
            self,
            original_inventory:dict[PhaseRef],
            scales,
        ):

        for phase_ref, mass in original_inventory.items():

            phase = self.phase_states.get_phase(
                phase_ref
            )

            phase.updatePhase(
                **{
                    self.basis:
                        mass * scales[phase_ref]
                }
            )
    
    def limit_material_rates(
                self,
                time,
                completed_state,
                resolved_inlets,
                operating_conditions,
                limiter_dt=1.0,
            ):

            resolved_outlets = self._resolve_outlets(
                completed_state,
                operating_conditions
            )

            buffer = self.calculate_material_contributions(
                time,
                completed_state,
                resolved_inlets,
                resolved_outlets,
            )

            rates = self.sum_material_contributions(buffer.contributions)

            buffer = self.apply_linear_rate_scaling(
                buffer,
                rates,
                completed_state,
                limiter_dt
            )

            return buffer
    def apply_linear_rate_scaling(
            self,
            buffer,
            rates,
            completed_state,
            limiter_dt=1.0,
        ):
        """
        Linearly scale consuming contributions to prevent negative inventory.

        Assumes dt is the expected integration step. This does not modify
        phases and does not regenerate nonlinear mechanisms.
        """

        violations = self.check_negative_inventory(
            rates,
            completed_state,
            limiter_dt=limiter_dt
        )

        if not violations:
            return buffer

        scales,scalable_terms,fixed_terms = self.calculate_scale(
            violations,
            completed_state,
            buffer,
            dt=limiter_dt
        )

        

        for term in scalable_terms:

            for state_key, rate in buffer.contributions[term].items():

                if state_key not in scales:
                    continue

                buffer.contributions[term][state_key] *= scales[state_key]
        for item in buffer.aux["outlet"]:
            phase_ref = item.mapping.sink_phaseref
            key = self.material_key(phase_ref)

            if key in scales:
                item.scale(scales[key][0], self.basis)
                assert np.allclose(
                    item.species_flow,
                    getattr(item.stream_phase, self.basis+"_flow"))
        return buffer

    def sum_material_contributions(self,contributions):
        return contributions.sum(axis=0)
    def material_key(self, phase_ref:PhaseRef):
        return StateKey(self.basis, phase_ref)
    
    def resolve_outlet_flows(
        self,
        operating_conditions:dict[OperatingKey,Any],
    ):

        flows = {}
        for i, connection in enumerate(self.outlet_connections):
            key = OperatingKey("vol_flow",connection=i,port='outlet')
            if key in operating_conditions:
                flows[i] = operating_conditions[key]

        return flows
    
    def get_total_inlet_vol_flow(self,resolved_inlets:StreamConditions)->float:

        total = 0.0
        for inlet in resolved_inlets.streams:
            total += inlet.stream.vol_flow

        return total
    
    def get_phase_operating_conditions(
        self,
        operating_conditions:dict[OperatingKey,Any],
        connection:int,
        phase_ref:PhaseRef,
        port:str
    )->dict[OperatingKey,Any]:

        updates = {}
        port = port.lower()

        for key, value in operating_conditions.items():

            if key.connection != connection:
                continue

            if (key.phaseref is not None
                and key.phaseref != phase_ref):
                continue

            if (key.port is not None
                and key.port!=port):
                continue

            updates[key.name] = value

        return updates

    def compute_requested_phase_outlet_flow(
            self,
            vessel_phase,
            total_outlet_flow,
            connection,
        ):
        """Compute desired outlet flow for a vessel phase.

        The default implementation distributes the specified outlet
        connection flow among the mapped phases in proportion to their
        current volume within the vessel.

        If no total outlet flow is specified, no outlet flow is requested.
        """

        if total_outlet_flow is None:
            return 0.0

        total_vessel_flow = sum(
            self.phase_states.get_phase(m.sink_phaseref).vol
            for m in connection.phase_mappings)

        if total_vessel_flow <= 0:
            return 0.0

        fraction = vessel_phase.vol / total_vessel_flow

        return fraction * total_outlet_flow
    
    def compute_actual_phase_outlet_flow(
        self,
        vessel_phase,
        requested_flow,
    ):
        """
        Compute the physically achievable outlet flow for a vessel phase.

        The default implementation assumes the requested flow is
        achievable. Subclasses may override this to enforce additional
        constraints (e.g. settling, phase disengagement, hydraulics).
        """

        return min(max(requested_flow, 0.0),vessel_phase.vol)
    
    def _resolve_outlets(
        self,
        completed_state:dict,
        operating_conditions:dict[OperatingKey],
    )->StreamConditions:

        resolved = []

        outlet_flows = self.resolve_outlet_flows(operating_conditions)

        for connection_num, connection in enumerate(self.outlet_connections):# iterate over outlet streams

            outlet_stream = copy.deepcopy(connection.stream)

            total_outlet_flow = outlet_flows.get(connection_num)
            transfers=[]
            for mapping in connection.phase_mappings: #iterate over each phase in that stream

                vessel_phase = self.phase_states.get_phase(mapping.sink_phaseref)
                outlet_phase = outlet_stream.get_phase_from_ref(mapping.source_phaseref)
                
                # Default outlet request
                requested_flow = self.compute_requested_phase_outlet_flow(vessel_phase,total_outlet_flow,connection)

                # Controller (or other operating conditions) may override the request
                ops =self.get_phase_operating_conditions(
                    operating_conditions,
                    connection_num,
                    mapping.source_phaseref,
                    "outlet",
                )
                requested_flow = ops.pop("vol_flow", requested_flow)

                # Apply physical limits once
                actual_flow = self.compute_actual_phase_outlet_flow(vessel_phase,requested_flow)
                updates = vessel_phase.state_dict

                # Amounts are determined from the resolved outlet flow
                for name in outlet_phase.amount_names:
                    updates.pop(name, None)

                # Keep only the preferred composition representation
                for name in outlet_phase.composition_names:
                    if name != outlet_phase.default_composition_name:
                        updates.pop(name, None)

                # Add any remaining operating-condition overrides
                updates.update(ops)

                # Physical limit always wins
                updates["vol_flow"] = actual_flow

                outlet_phase.updatePhase(**updates)
                species_flow = getattr(outlet_phase,self.basis+"_flow")
                transfers.append(
                    ResolvedPhaseTransfer(
                        connection=connection,
                        mapping=mapping,
                        vessel_phase=vessel_phase,
                        stream_phase=outlet_phase,
                        vol_flow=actual_flow,
                        species_flow=species_flow,
                        direction="outlet",
                    )
                )

            resolved.append(ResolvedStreamConnection(connection=connection,transfers=transfers))

        return StreamConditions(resolved)
        
    def resolve_inlets(self,completed_state,operating_conditions)->StreamConditions:

        resolved = []

        for connection_num, connection in enumerate(self.inlet_connections):

            inlet_stream = copy.deepcopy(connection.stream)
            transfers= []
            for mapping in connection.phase_mappings:
                stream_phase = inlet_stream.get_phase_from_ref(mapping.source_phaseref)

                vessel_phase = self.phase_states.get_phase(mapping.sink_phaseref)


                ops = self.get_phase_operating_conditions(
                    operating_conditions,
                    connection_num,
                    mapping.source_phaseref,
                    "inlet",
                )

                stream_phase.updatePhase(**ops)


                species_flow = getattr(stream_phase,self.basis+"_flow")


                transfers.append(
                    ResolvedPhaseTransfer(
                        connection=connection,
                        mapping=mapping,
                        vessel_phase=vessel_phase,
                        stream_phase=stream_phase,
                        vol_flow=stream_phase.vol_flow,
                        species_flow=species_flow,
                        direction="inlet",
                    )
                )
            resolved.append(ResolvedStreamConnection(connection=connection,
                                                     transfers=transfers))


        return StreamConditions(resolved)
    
    def calculate_material_contributions(
        self,
        time,
        completed_state,
        resolved_inlets,
        resolved_outlets,
    ):
        buffer = self._material_contributions
        buffer.reset()
        t0 = perf_counter()
        self.add_inlet_terms(
            buffer,
            time,
            completed_state,
            resolved_inlets,
            
        )
        self._timers['add_inlet_terms'] = self._timers.get('add_inlet_terms',0)+perf_counter()-t0

        t0 = perf_counter()
        self.add_intraphase_terms(
            buffer,
            time,
            completed_state,
        )
        self._timers['add_intraphase_terms'] = self._timers.get('add_intraphase_terms',0)+perf_counter()-t0
        t0 = perf_counter()
        self.add_crossphase_terms(
            buffer,
            time,
            completed_state,
        )
        self._timers['add_crossphase_terms'] = self._timers.get('add_crossphase_terms',0)+perf_counter()-t0

        t0 = perf_counter()
        self.add_outlet_terms(
            buffer,
            time,
            completed_state,
            resolved_outlets,
        )
        self._timers['add_outlet_terms'] = self._timers.get('add_outlet_terms',0)+perf_counter()-t0
        return buffer
    
    def add_inlet_terms(
            self,
            buffer:MaterialContributionBuffer,
            time,
            completed_state,
            resolved_inlets,
        ):
            for resolved_connection in resolved_inlets:
                for transfer in resolved_connection:

                    material_slice = self._material_slice_by_phase[
                        transfer.mapping.sink_phaseref
                    ]

                    buffer.contributions[
                        buffer.INLET,
                        material_slice,
                    ] += transfer.species_flow

                    for mechanism in transfer.vessel_phase.mechanisms:

                        state_rates = mechanism.get_inlet_contributions(
                            transfer.stream_phase,
                            transfer.vessel_phase,
                            transfer.vol_flow,
                            completed_state,
                        )

                        for state_key, value in state_rates.items():

                            state_slice = self.solver_state_collection.material_slices.get(state_key)

                            if state_slice is not None:
                                buffer.contributions[
                                    buffer.INLET,
                                    state_slice,
                                ] += value

                    buffer.aux[
                        buffer.INLET
                    ].append(transfer)

        

    def add_outlet_terms(
        self,
        buffer:MaterialContributionBuffer,
        time,
        completed_state,
        resolved_outlets,
    ):

        for resolved_connection in resolved_outlets:
            for transfer in resolved_connection:

                material_slice = self._material_slice_by_phase[
                    transfer.mapping.sink_phaseref
                ]

                buffer.contributions[
                    buffer.OUTLET,
                    material_slice,
                ] += transfer.species_flow

                for mechanism in transfer.vessel_phase.mechanisms:

                    state_rates = mechanism.get_outlet_contributions(
                        transfer.stream_phase,
                        transfer.vessel_phase,
                        transfer.vol_flow,
                        completed_state,
                    )

                    for state_key, value in state_rates.items():

                        state_slice = self.solver_state_collection.material_slices.get(state_key)

                        if state_slice is not None:
                            buffer.contributions[
                                buffer.OUTLET,
                                state_slice,
                            ] += value

                buffer.aux[
                    buffer.OUTLET
                ].append(transfer)

    
    def add_intraphase_terms(self,
            buffer:MaterialContributionBuffer,
            time,
            completed_state
        ):
        for process in self.intraphase_processes:
            phase = self.phase_states.get_phase(process.phaseref)

            intraphase_result = process.mechanism.get_solver_state_rates(
                process=process,
                phase=phase,
                time=time,
                completed_state=completed_state
            )

            for state_key,rate in intraphase_result.state_rates.items():
                state_slice = self.solver_state_collection.material_slices.get(state_key)
                if state_slice is not None:
                    buffer.contributions[
                        buffer.INTRAPHASE,
                        state_slice,
                    ] += rate

            buffer.aux[buffer.INTRAPHASE].append(intraphase_result.aux)

    def add_crossphase_terms(
            self,
            buffer:MaterialContributionBuffer,
            time:float,
            completed_state:dict[StateKey]
        ):

        # temp = completed_state["temp"]

        for connection in self.phase_connections:

            source_phase = self.phase_states.get_phase(connection.source_phaseref)

            sink_phase = self.phase_states.get_phase(connection.sink_phaseref)
            

            if not connection.active_condition(source_phase,sink_phase):
                continue

            t0 = perf_counter()
            crossphase_result = connection.mechanism.get_solver_state_rates(
                source_phase=source_phase,
                sink_phase=sink_phase,
                connection=connection,
                completed_state=completed_state,
                time=time
            )
            self._timers['crossphase_mechanism'] = self._timers.get('crossphase_mechanism',0)+perf_counter()-t0

            t0 = perf_counter()
            for state_key, rate in crossphase_result.state_rates.items():

                if isinstance(state_key.phaseref, BasePhase):
                    actual_phaseref = self.phase_states.get_ref(state_key.phaseref)

                    if actual_phaseref is None:
                        raise RuntimeError(
                            f"Could not find mechanism's phase: "
                            f"{state_key.phaseref} in vessel phases"
                        )

                    lookup_key = StateKey(state_key.name,actual_phaseref)
                else:
                    lookup_key = state_key

                material_slice = self.solver_state_collection.material_slices.get(lookup_key)

                if material_slice is not None:
                    buffer.contributions[
                        buffer.CROSSPHASE,
                        material_slice,
                    ] += rate
            self._timers['crossphase_contributions'] = self._timers.get('crossphase_contributions',0)+perf_counter()-t0
            
            buffer.aux[buffer.CROSSPHASE].append(crossphase_result.aux)
    
    def energy_balances(
            self,
            time,
            completed_state,
            material_buffer
        ):
        """
        Energy contributions are accumulated in SI units (joules).

        Positive contributions add energy to the vessel.
        Negative contributions remove energy from the vessel.
        """

        aux = material_buffer.aux
        contributions = {
            "inlet": 0,
            "intraphase": 0,
            "crossphase": 0,
            "outlet": 0,
            "utility": 0,
            "mixing":0,
            "shaftwork":0
        }
        t0 = perf_counter()
        self.add_inlet_energy_terms(
            contributions,
            aux[material_buffer.INLET],
            time,
            completed_state
        )
        self._timers['add_inlet_energy_terms'] = self._timers.get('add_inlet_energy_terms',0)+perf_counter()-t0

        t0 = perf_counter()
        self.add_intraphase_energy_terms(
            contributions,
            aux[material_buffer.INTRAPHASE],
            time,
            completed_state
        )
        self._timers['add_intraphase_energy_terms'] = self._timers.get('add_intraphase_energy_terms',0)+perf_counter()-t0

        t0 = perf_counter()
        self.add_crossphase_energy_terms(
            contributions,
            aux[material_buffer.CROSSPHASE],
            time,
            completed_state
        )
        self._timers['add_crossphase_energy_terms'] = self._timers.get('add_crossphase_energy_terms',0)+perf_counter()-t0

        t0 = perf_counter()
        self.add_outlet_energy_terms(
            contributions,
            aux[material_buffer.OUTLET],
            time,
            completed_state
        )
        self._timers['add_outlet_energy_terms'] = self._timers.get('add_outlet_energy_terms',0)+perf_counter()-t0

        t0 = perf_counter()
        self.add_utility_energy_terms(
            contributions,
            time,
            completed_state
        )
        self._timers['add_utility_energy_terms'] = self._timers.get('add_utility_energy_terms',0)+perf_counter()-t0

        t0 = perf_counter()
        self.add_mixing_energy_terms(
            contributions,
            time,
            completed_state
        )
        self._timers['add_mixing_energy_terms'] = self._timers.get('add_mixing_energy_terms',0)+perf_counter()-t0

        t0 = perf_counter()
        self.add_shaftwork_energy_terms(
            contributions,
            time,
            completed_state
        )
        self._timers['add_shaftwork_energy_terms'] = self._timers.get('add_shaftwork_energy_terms',0)+perf_counter()-t0

        t0 = perf_counter()
        qdot = sum(contributions.values())
        self._timers['sum_energy_contributions'] = self._timers.get('sum_energy_contributions',0)+perf_counter()-t0
        t0 = perf_counter()
        basis = 'mass' if self.basis=='mass_j' else self.basis
        heat_capacity = self.Phases.getCp(basis = basis)
        self._timers['get_heat_capacity'] = self._timers.get('get_heat_capacity',0)+perf_counter()-t0
        dtemp_dt = qdot / heat_capacity
        return {StateKey("global_temp"): dtemp_dt}

    def add_inlet_energy_terms(
            self,
            contributions,
            aux:list[ResolvedPhaseTransfer],
            time,
            completed_state
        ):
        """Computes enthalpy effects"""
        for inlet in aux:

            phase = inlet.stream_phase

            h_in = phase.getEnthalpy(
                phase.temp,
                temp_ref=self.temp_ref,
                total_h=True,
                basis='mass'
            )  # J/kg mixture
            vessel_phase = inlet.vessel_phase

            h_vessel = vessel_phase.getEnthalpy(
                vessel_phase.temp,
                temp_ref=self.temp_ref,
                total_h=True,
                basis="mass",
            )

            contributions["inlet"] += (
                inlet.species_flow * (h_in - h_vessel)
            ).sum()

    def add_crossphase_energy_terms(
            self,
            contributions,
            aux,
            time,
            completed_state
        ):
        
        for crossphase in aux:
            connection = crossphase['connection']
            contributions['crossphase'] += connection.mechanism.get_heat_generation(aux=crossphase,
                                                             completed_state=completed_state,
                                                             time=time)
    def add_intraphase_energy_terms(self,
                                  contributions,
                                  aux,
                                  time,
                                  completed_state
                                ):
        for process_aux in aux:
            process = process_aux['process']
            q = process.mechanism.get_heat_generation(aux=process_aux,
                                                    completed_state=completed_state,
                                                    time=time)
            contributions['intraphase'] += q

    def add_outlet_energy_terms(
            self,
            contributions,
            aux:list[ResolvedPhaseTransfer],
            time,
            completed_state
        ):
        """Computes enthalpy effects"""
        for outlet in aux:
            phase = outlet.stream_phase
            temp= phase.temp
            h_out = phase.getEnthalpy(
                temp,
                temp_ref=self.temp_ref,
                total_h=True,
                basis='mass'
            )

            contributions["outlet"] -= (outlet.species_flow * h_out).sum()

    def get_heat_transfer_temperature(self):
        """The temperature to use to determine heat transfer from the vessel. The default assumption is to use the temperature of the first liquid phase"""
        return self.Phases[0].temp
    
    def add_utility_energy_terms(
            self,
            contributions,
            time,
            completed_state
        ):

        if not self.has_utility_balance:
            return

        temp = self.get_heat_transfer_temperature()
        temp_ht = self.Utility.temp_in#completed_state[StateKey("temp_ht")]

        qdot = self.get_heat_transfer_rate(temp, temp_ht)
                                           
        self.Utility.temp_out = temp_ht +qdot/(self.Utility.mass_flow*self.Utility.cp)

        contributions["utility"] -= qdot

    def add_mixing_energy_terms(self,contributions,time,completed_state):
        "Used to add heat of mixing terms"
        #should be += self.Phases.getHeatOfMixing()
        contributions['mixing']+=0

    def add_shaftwork_energy_terms(self,contributions,time,completed_state):
        contributions['shaftwork']+=0

    def get_heat_transfer_rate(
            self,
            temp,
            temp_ht
        ):

        if self.ht_mode == "coil":
            raise NotImplementedError

        area = self.get_heat_transfer_area()

        return self.u_ht * area * (temp - temp_ht)
    
        
    def get_heat_transfer_area(self):

        liquid = self.phase_states.get_phase(
            PhaseRef("liquid",0)
        )
        if self.diam <= 0:
            raise ValueError(
                "Heat transfer requires reactor diameter > 0."
            )

        return 4 * liquid.vol / self.diam + self.area_base
    
    
    
    def build_solver_history(
            self,
            time,
            solver_states
        ):

        history = self.solver_state_collection.unpack_history(
            solver_states
        )

        history["time"] = np.asarray(time)

        return history
    
    def update_final_state(self, solver_history):

        final_state = {
            key: value[-1]
            for key, value in solver_history.items()
            if key != "time"
        }

        self.update_phases_from_state(final_state)
        return final_state
    
    def update_final_conditions(self,completed_state,time,solver_history,output_history):
        completed_state = self.complete_state(completed_state,time[-1])
        resolved_inlets,operating_conditions = self.get_operating_conditions(time,completed_state)
        resolved_outlets = self._resolve_outlets(completed_state,operating_conditions)
        self.outlet_conditions =resolved_outlets
        
        self.elapsed_time = time[-1]

        
    def retrieve_results(self, time, solver_states):

        solver_history = self.build_solver_history(time,solver_states)
        output_history = self.find_output_states_from_replay(time,solver_history)

        completed_state = self.update_final_state(solver_history)
        self.update_final_conditions(completed_state,time,solver_history,output_history)
        self.result = self.build_dynamic_result(time,
                    solver_history,
                    output_history)
        return self.result
    

    def build_dynamic_result(
        self,
        time,
        solver_history,
        output_history,
    ):

        data = {"time": np.asarray(time)}

        data.update(
            self.solver_state_collection.flatten(solver_history)
        )

        data.update(
            self.output_state_collection.flatten(output_history)
        )

        states_di = {}

        for key, state in self.solver_state_collection.states.items():

            states_di[
                self.solver_state_collection.format_key(key)
            ] = state.as_dict()

        fstates_di = {}

        for key, state in self.output_state_collection.states.items():

            fstates_di[
                self.output_state_collection.format_key(key)
            ] = state.as_dict()

        return DynamicResult(
            states_di,
            fstates_di,
            **data
        )
    @property
    def name_species(self):
        #Backward Compatibility
        return self.Phases.name_species

    @property
    def num_species(self):
        #backward compatibility
        return self.Phases.num_species
    def create_pseudo(self):
        """
        Create an independent vessel for replaying an accepted solver
        trajectory.

        The pseudo vessel contains the model structure, phases, mechanisms,
        connections, and controller, but does not contain the integrator or
        other runtime solver objects.
        """

        pseudo = object.__new__(type(self))

        # Make self-references inside copied objects resolve to the pseudo
        # rather than accidentally retaining a reference to the real vessel.
        memo = {id(self): pseudo}

        # Objects that are either non-copyable or represent runtime state
        # that should not be carried into replay.
        excluded = {
            "integrator",
            "result",
            "derivatives",
            "elapsed_time",
            "outlet_conditions",
        }

        for name, value in self.__dict__.items():

            if name in excluded:
                continue

            pseudo.__dict__[name] = copy.deepcopy(
                value,
                memo,
            )

        # The pseudo is not an integratable vessel.
        pseudo.integrator = None

        pseudo.reset()

        return pseudo
    
    def find_output_states_from_replay(
        self,
        time,
        solver_history
    ):
        """
        Reconstruct all output states by replaying the accepted solver history.

        Parameters
        ----------
        solver_states : ndarray
            Solver state history returned by the integrator.

        time : ndarray
            Accepted/output time points corresponding to solver_states.

        Returns
        -------
        dict
            Dictionary keyed by StateKey containing the reconstructed output
            histories.
        """

        pseudo = self.create_pseudo()


        output_history = {
            key: []
            for key in pseudo.output_state_collection.states
        }

        for i, t in enumerate(time):

            # -----------------------------
            # Recover solver state
            # -----------------------------
            completed_state = {
                key: value[i]
                for key, value in solver_history.items()
                if key != "time"
            }

            # -----------------------------
            # Reconstruct controller states
            # -----------------------------
            completed_state = pseudo.complete_state(
                completed_state,
                t,
            )

            # -----------------------------
            # Update vessel phases
            # -----------------------------
            pseudo.update_phases_from_state(completed_state)

            # -----------------------------
            # Reconstruct operating conditions
            # -----------------------------
            resolved_inlets, operating_conditions = (pseudo.get_operating_conditions(t,completed_state))

            resolved_outlets = pseudo._resolve_outlets(
                completed_state,
                operating_conditions,
            )

            # -----------------------------
            # Evaluate output variables
            # -----------------------------
            for key, state in (
                pseudo.output_state_collection.states.items()
            ):

                output_history[key].append(
                    state.compute_value(
                        state_var=state,
                        time=t,
                        completed_state=completed_state,
                        context=pseudo,
                        resolved_inlets=resolved_inlets,
                        resolved_outlets=resolved_outlets,
                        operating_conditions=operating_conditions,
                    )
                )

        # Convert lists to arrays
        for key in output_history:

            output_history[key] = np.asarray(
                output_history[key]
            )

        return output_history
