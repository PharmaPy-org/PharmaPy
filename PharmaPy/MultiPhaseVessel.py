
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

    # Declared on the class so a subclass can override it. Assigning it in
    # __init__ would shadow the subclass value with None on every instance,
    # and Connections and SimExec branch on it, so a shadowed value silently
    # takes a flowsheet down the wrong path. Spellings match the rest of
    # PharmaPy: 'Batch', 'Semibatch', 'Continuous'.
    oper_mode = None

    def __init__(self,integrator=None,temp_ref=273.15,
     isothermal=False, reset_states=False, controller=None, h_conv=0, 
      state_events={},
      adiabatic=False,Phases=None,
      basis='mass_j',ht_mode="jacket",diam=0,area_base=0,
      emit_events=True):

        # Built here rather than as a default argument: a default is
        # evaluated once at class-definition time, so every vessel in the
        # session would share one controller object, and a level controller's
        # target_volume would leak between unrelated vessels.
        if controller is None:
            controller = Controller()

        if isothermal and controller is not None:
            assert 'global_temp' not in controller.states and 'temp' not in controller.states, "Cannot change the temperature of an isothermal unit"

        self.basis = basis
        self.adiabatic = adiabatic
        self.isothermal = isothermal

        self.controller = controller #TODO ZZ refactor analyze_controls to give a Controls dataclass, an empty one if controls None

        
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

        # Whether mechanisms and the controller hand their switching
        # surfaces to the integrator. Root-finding pays off when a surface
        # is actually crossed and the step size is limited by that crossing;
        # it is pure overhead when the step size is limited by something
        # else, and the cost is per-step (and heavier under diffeqpy, where
        # every condition evaluation crosses into Python). Measure before
        # assuming it helps.
        self.emit_events = emit_events

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
        """
        The unit's single outlet stream.

        After a solve this is the resolved stream, carrying the flows and
        composition the vessel actually discharged. Before one it is the
        connection's stream, which still holds the configured template.

        AttributeError raised in a property is indistinguishable from a
        missing attribute, so Python falls through to __getattr__ and the
        caller sees a bare "Outlet" instead of the reason. Every message
        here therefore says what is actually wrong.
        """

        conditions = self.outlet_conditions

        if conditions is not None and conditions.streams:

            if len(conditions.streams) == 1:
                return conditions.streams[0].stream

            raise AttributeError(
                f"This unit resolved {len(conditions.streams)} outlet "
                "streams. Use outlet_conditions to pick one."
            )

        # Backward compatibility before solve
        if len(self.outlet_connections) == 1:
            return self.outlet_connections[0].stream

        if not self.outlet_connections:
            raise AttributeError(
                f"{type(self).__name__} has no outlet connections. "
                "Continuous units define one in "
                "configure_default_connections; batch units have none."
            )

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

            value = getattr(phase, statekey.name, None)

            if value is not None:
                return value

            # A phase delegates to the mechanisms in its own `mechanisms`
            # list, but a mechanism attached to an intraphase process is not
            # in that list, so its states have to be found directly.
            return self.get_mechanism_state_value(statekey)

        # unit operation state
        value = getattr(self, statekey.name, None)

        if value is not None:
            return value

        return self.get_mechanism_state_value(statekey)

    def get_mechanism_state_value(self, statekey):
        """Initial value for a solver state owned by a mechanism."""

        for mechanism in self.iter_mechanisms():

            if mechanism.owns_state(statekey):
                return getattr(mechanism, statekey.name, None)

        return None
            
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

        # A property whose getter raises AttributeError looks to Python
        # exactly like a missing attribute, so __getattr__ runs and the
        # explanation the getter raised is replaced by a bare name. Re-run
        # the getter so its own message reaches the caller.
        descriptor = getattr(type(self), name, None)

        if isinstance(descriptor, property) and descriptor.fget is not None:
            return descriptor.fget(self)

        raise AttributeError(
            f"{type(self).__name__!s} has no attribute {name!r}"
        )
       
    def _initialize_states(self,reset=False):

        self.reset_states = reset

        # Populated lazily by unit_model, but update_phases_from_state can run
        # before the first right-hand side evaluation (reset, replay).
        self._timers = {}

        self.state_variables = StateCollection()

        self.input_states = StateCollection()

        self.output_states = StateCollection()

        self.elapsed_time = 0

        self.result = None
        self._initial_solver_state = None

        # Resolved outlet streams, available once the unit has been solved.
        self.outlet_conditions = None

        # Memo for evaluate_events, keyed on the exact (time, states).
        self._event_cache_key = None
        self._event_cache_values = None

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
    def has_algebraic_balance(self):
        """
        True when some mechanism contributes a residual rather than a rate.

        Checked the way has_energy_balance is: when it is False the whole
        algebraic pass is skipped, so an ordinary ODE never pays for the
        machinery or allocates a vector of zeros to carry nothing.
        """
        return self.solver_state_collection.has_algebraic

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
            # An intraphase mechanism belongs to the phase its process runs
            # in, so that is the owner of any solver state it declares.
            process.mechanism.add_solver_state_variables(
                self.solver_state_collection,
                overwrite=overwrite,
                phase_ref=process.phaseref,
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

        for mechanism in getattr(self, "_workspace_mechanisms", ()):
            mechanism.update_state(completed_state, unit=self)

    def pack_state_rates(self, material_rates, global_rates=None,
                         algebraic_residuals=None):
        """
        Pack one vector the solver can consume.

        Differential slots carry the derivative; algebraic slots carry the
        residual g(y), which must be zero at a consistent solution. Keeping
        both in one vector of the same length means the backends differ only
        in how they interpret it (mass matrix, or an implicit residual), and
        unit_model's return shape never changes.
        """

        buffer = self._solver_rate_buffer
        buffer.fill(0.0)

        material_slices = self.solver_state_collection.material_slices
        solver_slices = self.solver_state_collection.slices

        for key in self.solver_state_collection.keys:

            if algebraic_residuals is not None and key in algebraic_residuals:
                buffer[solver_slices[key]] = np.asarray(
                    algebraic_residuals[key]
                ).reshape(-1)
                continue

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
        if self.has_energy_balance:
            t0 = perf_counter()
            energy_rates = self.energy_balances(time,completed_state, material_buffer)
            self._timers['energy_balances_total'] = self._timers.get('energy_balances_total',0)+perf_counter()-t0
            global_rates.update(energy_rates)

        # utility_rates = self.utility_energy_balance(
        #     time,completed_state)
        # global_rates.update(utility_rates)

        if enrgy_bce:
            return self.pack_state_rates(global_rates=global_rates)

        # Skipped entirely, like the energy balance, when no mechanism
        # declares an algebraic state: an ordinary ODE never builds or
        # carries a residual vector.
        algebraic_residuals = None

        if self.has_algebraic_balance:
            t0 = perf_counter()
            algebraic_residuals = self.algebraic_balances(time, completed_state)
            self._timers['algebraic_balances_total'] = self._timers.get('algebraic_balances_total',0)+perf_counter()-t0

            if alg_bce:
                return algebraic_residuals

        t0 = perf_counter()
        balances = self.pack_state_rates(material_rates=material_rates,
                                        global_rates=global_rates,
                                        algebraic_residuals=algebraic_residuals)
        self._timers['pack_state_rates'] = self._timers.get('pack_state_rates',0)+perf_counter()-t0
        assert len(balances) == len(states), (
            f"Returned {len(balances)} derivatives "
            f"for {len(states)} solver states."
        )

        self.derivatives = balances
        return balances

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

        self._compile_stream_workspaces()
        self._compile_positivity_layout()
        self.compile_events()

    def _compile_stream_workspaces(self):
        """
        Allocate one reusable stream per connection.

        Resolving a connection used to deep-copy its stream on every
        right-hand side evaluation. For a stream carrying a discretized
        phase that copy dominated the run time, so each connection now owns
        a single scratch stream that is overwritten in place instead. The
        copy still exists, so the user's connection stream is never mutated;
        it is just made once rather than tens of thousands of times.
        """

        self._inlet_workspaces = [
            copy.deepcopy(connection.stream)
            for connection in self.inlet_connections
        ]

        self._outlet_workspaces = [
            copy.deepcopy(connection.stream)
            for connection in self.outlet_connections
        ]

        # Which inlet fields a controller last overrode, per connection and
        # phase, so they can be put back when it stops overriding them.
        self._inlet_overrides_applied = {}

        # A stream phase shares its vessel phase's mechanisms by reference
        # (to_stream is a shallow copy), so the workspaces hold their own
        # copies and those copies have to be advanced alongside the vessel's.
        # Sharing the live mechanism instead would let resolving an outlet
        # write back into the vessel's own distribution.
        self._workspace_mechanisms = [
            mechanism
            for workspaces in (self._inlet_workspaces, self._outlet_workspaces)
            for workspace in workspaces
            for phase in workspace
            for mechanism in phase.mechanisms
        ]

    def _apply_inlet_overrides(
        self,
        stream_phase,
        template_phase,
        ops,
        workspace_key,
    ):
        """
        Apply controller overrides to a reused inlet stream.

        The workspace persists between calls, so a field the controller
        overrode on one evaluation and left alone on the next has to be
        restored from the connection's own stream. A fresh copy used to do
        that implicitly.
        """

        previous = self._inlet_overrides_applied.get(workspace_key)

        if previous:

            stale = previous - ops.keys()

            if stale:
                stream_phase.updatePhase(
                    **{
                        name: getattr(template_phase, name)
                        for name in stale
                    }
                )

        if ops:
            stream_phase.updatePhase(**ops)

        self._inlet_overrides_applied[workspace_key] = set(ops)

    def _get_inlet_workspace(self, connection_num, connection):

        workspaces = getattr(self, "_inlet_workspaces", None)

        if workspaces is None or connection_num >= len(workspaces):
            self._compile_stream_workspaces()
            workspaces = self._inlet_workspaces

        return workspaces[connection_num]

    def _get_outlet_workspace(self, connection_num, connection):

        workspaces = getattr(self, "_outlet_workspaces", None)

        if workspaces is None or connection_num >= len(workspaces):
            self._compile_stream_workspaces()
            workspaces = self._outlet_workspaces

        return workspaces[connection_num]

    def _compile_positivity_layout(self):
        """
        Precompute the flat layout used by the positivity limiter.

        The limiter runs on every right-hand side evaluation, so everything
        that depends on the state layout rather than on the state values is
        resolved once, here.
        """

        collection = self.solver_state_collection

        limited = np.zeros(collection.material_dim, dtype=bool)
        groups = []

        for key, material_slice in collection.material_slices.items():

            if not collection.states[key].limit_negative_inventory:
                continue

            limited[material_slice] = True
            groups.append((key, material_slice))

        self._positivity_limited = limited
        self._positivity_groups = tuple(groups)
        self._has_positivity_limits = bool(groups)

        self._inventory_buffer = np.zeros(collection.material_dim)
        self._scale_buffer = np.ones(collection.material_dim)

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

    def iter_mechanisms(self):
        """
        Every mechanism attached to this vessel, each yielded once.

        A phase mechanism is commonly also reachable through a phase
        connection wrapped in a MechanismView, so identity is tracked through
        the wrapper to avoid registering the same events or residuals twice.
        """

        seen = set()

        for phase in self.Phases:

            for mechanism in getattr(phase, "mechanisms", ()) or ():

                if id(mechanism) not in seen:
                    seen.add(id(mechanism))
                    yield mechanism

        for connection in self.phase_connections:

            mechanism = connection.mechanism

            if mechanism is None:
                continue

            # MechanismView delegates attribute access, so unwrap before
            # testing identity against the phase's own mechanism list.
            underlying = getattr(mechanism, "mechanism", mechanism)

            if id(underlying) not in seen:
                seen.add(id(underlying))
                yield mechanism

        for process in self.intraphase_processes:

            if id(process.mechanism) not in seen:
                seen.add(id(process.mechanism))
                yield process.mechanism

    def get_events(self):
        """
        Switching surfaces declared by the controller and the mechanisms.

        Handing these to the integrator lets it locate a regime change by
        root-finding and restart cleanly there, instead of discovering it by
        failing steps. That is what makes a small maxh unnecessary.
        """

        if not self.emit_events:
            return []

        events = list(self.controller.get_events(self))

        for mechanism in self.iter_mechanisms():
            events.extend(mechanism.get_events(self))

        # Events supplied directly to the constructor, when they use the
        # StateEvent contract rather than the legacy dictionary format that
        # Commons.eval_state_events consumes.
        for event in self.state_event_list or ():
            if isinstance(event, StateEvent):
                events.append(event)

        return events

    def compile_events(self):
        self._compiled_events = tuple(self.get_events())
        self._event_cache_key = None
        self._event_cache_values = None
        return self._compiled_events

    @property
    def compiled_events(self):
        events = getattr(self, "_compiled_events", None)

        if events is None:
            events = self.compile_events()

        return events

    @property
    def has_events(self):
        return len(self.compiled_events) > 0

    def evaluate_events(self, time, states):
        """
        Evaluate every event function at (time, states).

        The phases are brought up to date first, because event functions are
        written against the vessel (unit.Phases.vol, a mechanism's
        supersaturation) rather than against the raw solver vector. This runs
        once per solver step, not once per right-hand side evaluation.
        """

        events = self.compiled_events

        if not events:
            return np.zeros(0)

        states = np.asarray(states, dtype=float)

        # Every event is evaluated together and the result memoized against
        # the exact (time, state) it came from. SciML asks one callback at a
        # time, so without this a vessel with N events would rebuild the
        # phases N times per step to answer N questions about the same point.
        # Hashing the raw bytes is exact, and cheap next to complete_state.
        cache_key = (time, states.tobytes())

        if self._event_cache_key == cache_key:
            return self._event_cache_values

        unpacked = self.solver_state_collection.unpack(states)
        completed_state = self.complete_state(unpacked, time)
        self.update_phases_from_state(completed_state)

        values = np.empty(len(events))

        for index, event in enumerate(events):
            values[index] = event.function(time, completed_state, self)

        self._event_cache_key = cache_key
        self._event_cache_values = values

        return values

    def handle_event(self, time, event_indices):
        """
        Respond to located events.

        Non-terminal events need no action: the integrator has already
        stopped at the root and will restart there, which is the whole point.
        Returns True when the run should stop.
        """

        events = self.compiled_events

        return any(
            events[index].terminal
            for index in event_indices
            if 0 <= index < len(events)
        )

    
    def algebraic_balances(self, time, completed_state):
        """
        Residuals for solver states defined by a constraint, not a rate.

        Each mechanism returns {StateKey: residual}; a consistent solution is
        one where every residual is zero. Only called when
        has_algebraic_balance is True.
        """

        residuals = {}

        for mechanism in self.iter_mechanisms():

            contribution = mechanism.get_solver_state_residuals(
                time=time,
                completed_state=completed_state,
                unit=self,
            )

            if contribution:
                residuals.update(contribution)

        missing = [
            key for key in self.solver_state_collection.algebraic_keys
            if key not in residuals
        ]

        if missing:
            raise KeyError(
                "No mechanism supplied a residual for algebraic state(s) "
                f"{[str(key) for key in missing]}. Every state registered "
                "with state_type='alg' must be returned by some mechanism's "
                "get_solver_state_residuals."
            )

        return residuals

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

    def gather_inventory(self, completed_state):
        """Pack the current phase inventories into the material layout."""

        inventory = self._inventory_buffer

        for state_key, material_slice in self._positivity_groups:
            inventory[material_slice] = completed_state[state_key]

        return inventory

    @staticmethod
    def soft_saturate(ratio, width=0.1):
        """
        C1-continuous stand-in for ``min(ratio, 1)``.

        A hard minimum puts a kink in the right-hand side, so the integrator
        has to cut its step every time the limiter engages or disengages and
        the Jacobian it extrapolates from is wrong on one side of the
        switch. This version equals ``ratio`` below ``1 - width`` and ``1``
        above ``1 + width``, bridged by a quadratic whose value and slope
        match at both ends, so the derivative stays continuous.
        """

        lower = 1.0 - width
        upper = 1.0 + width

        blended = ratio - (ratio - lower) ** 2 / (4.0 * width)

        return np.where(
            ratio <= lower,
            ratio,
            np.where(ratio >= upper, 1.0, blended),
        )

    def calculate_scale(
            self,
            buffer,
            completed_state,
            limiter_dt=1.0,
        ):
        """
        Compute one throttling factor per phase inventory.

        Inlet and intraphase terms are treated as fixed; cross-phase and
        outlet terms compete for whatever inventory is left over
        ``limiter_dt`` and share a single factor per phase, which keeps an
        outlet composition consistent with the phase it drains.

        Returns ``None`` when nothing needs throttling, so the common case
        costs one vectorized comparison and no allocation of scale vectors.
        """

        contributions = buffer.contributions

        inventory = self.gather_inventory(completed_state)

        fixed = (
            contributions[buffer.INLET]
            + contributions[buffer.INTRAPHASE]
        )

        scalable = (
            contributions[buffer.CROSSPHASE]
            + contributions[buffer.OUTLET]
        )

        available = inventory + fixed * limiter_dt
        demand = -scalable * limiter_dt

        active = self._positivity_limited & (demand > 0.0)

        if not active.any():
            return None

        scales = self._scale_buffer
        scales.fill(1.0)

        scales[active] = self.soft_saturate(
            np.maximum(available[active], 0.0) / demand[active]
        )

        engaged = False

        for _, material_slice in self._positivity_groups:

            phase_scale = scales[material_slice].min()

            if phase_scale < 1.0:
                engaged = True

            scales[material_slice] = phase_scale

        if not engaged:
            return None

        return scales


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

            self.apply_rate_scaling(
                buffer,
                completed_state,
                limiter_dt
            )

            return buffer

    def apply_rate_scaling(
            self,
            buffer,
            completed_state,
            limiter_dt=1.0,
        ):
        """
        Throttle consuming contributions to keep phase inventories positive.

        ``limiter_dt`` is the horizon over which depletion is anticipated.
        Phases and nonlinear mechanisms are not re-evaluated: each mechanism
        that drains a throttled phase is scaled down at both ends, so the
        transfer stays closed even while it is being limited.

        Returns True when the limiter engaged.
        """

        scales = self.calculate_scale(
            buffer,
            completed_state,
            limiter_dt=limiter_dt,
        )

        if scales is None:
            return False

        self.scale_crossphase_transfers(buffer, scales)
        self.scale_outlet_transfers(buffer, scales)

        return True

    def _transfer_scale(self, scales, material_writes):
        """
        The factor a single mechanism may run at.

        A mechanism is held to the tightest limit among the phases it
        drains, so it can never take more from a phase than that phase's own
        factor allows.
        """

        factor = 1.0

        for material_slice, rate in material_writes:
            factor = min(factor, scales[material_slice.start])

        return factor

    def scale_crossphase_transfers(self, buffer, scales):
        """
        Scale each cross-phase mechanism down by a single factor.

        Both ends of a transfer move together, so throttling the source
        drain also throttles what arrives in the sink. Scaling the flat
        contribution rows independently would credit the sink with material
        the source never gave up.
        """

        row = buffer.contributions[buffer.CROSSPHASE]

        for aux in buffer.aux[buffer.CROSSPHASE]:

            material_writes = aux.get("material_writes")

            if not material_writes:
                continue

            factor = self._transfer_scale(scales, material_writes)

            if factor >= 1.0:
                continue

            for material_slice, rate in material_writes:
                row[material_slice] -= (1.0 - factor) * rate

    def scale_outlet_transfers(self, buffer, scales):
        """Scale outlet draws, and the streams that report them, together."""

        row = buffer.contributions[buffer.OUTLET]

        for transfer in buffer.aux[buffer.OUTLET]:

            material_slice = self._material_slice_by_phase.get(
                transfer.mapping.sink_phaseref
            )

            if material_slice is None:
                continue

            factor = scales[material_slice.start]

            if factor >= 1.0:
                continue

            row[material_slice] += (1.0 - factor) * transfer.species_flow

            for mechanism_slice, rate in transfer.material_writes:
                row[mechanism_slice] += (1.0 - factor) * rate

            transfer.scale(factor, self.basis)
            transfer.vol_flow *= factor

    def sum_material_contributions(self,contributions):
        return contributions.sum(axis=0)
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
    
    # Operating conditions that describe a whole connection rather than one
    # phase within it. A controller that asks for an outlet vol_flow means
    # the stream's total, which the caller distributes over the mapped
    # phases; handing the same number to every phase as its own flow would
    # multiply the draw by the number of phases.
    connection_level_operating_names = ("vol_flow",)

    def get_phase_operating_conditions(
        self,
        operating_conditions:dict[OperatingKey,Any],
        connection:int,
        phase_ref:PhaseRef,
        port:str,
        skip_connection_level=False,
    )->dict[OperatingKey,Any]:

        updates = {}
        port = port.lower()

        for key, value in operating_conditions.items():

            if key.connection != connection:
                continue

            if key.phaseref is None:
                if (skip_connection_level
                    and key.name in self.connection_level_operating_names):
                    continue
            elif key.phaseref != phase_ref:
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

        The default implementation only rejects backflow. Draining a phase
        faster than it can supply is left to the positivity limiter, which
        throttles smoothly; clamping here would need a volumetric flow
        (m3/s) to be compared against a volume (m3), and the hard corner it
        introduced cost the integrator a step every time it was reached.

        Subclasses may override this to enforce additional constraints
        (e.g. settling, phase disengagement, hydraulics).
        """

        return max(requested_flow, 0.0)
    
    def _resolve_outlets(
        self,
        completed_state:dict,
        operating_conditions:dict[OperatingKey],
    )->StreamConditions:

        resolved = []

        outlet_flows = self.resolve_outlet_flows(operating_conditions)

        for connection_num, connection in enumerate(self.outlet_connections):# iterate over outlet streams

            outlet_stream = self._get_outlet_workspace(connection_num, connection)

            total_outlet_flow = outlet_flows.get(connection_num)
            transfers=[]
            for mapping in connection.phase_mappings: #iterate over each phase in that stream

                vessel_phase = self.phase_states.get_phase(mapping.sink_phaseref)
                outlet_phase = outlet_stream.get_phase_from_ref(mapping.source_phaseref)

                # Default outlet request
                requested_flow = self.compute_requested_phase_outlet_flow(vessel_phase,total_outlet_flow,connection)

                # Controller (or other operating conditions) may override the
                # request, but only a phase-specific vol_flow does: the
                # connection's total has already been distributed above.
                ops =self.get_phase_operating_conditions(
                    operating_conditions,
                    connection_num,
                    mapping.source_phaseref,
                    "outlet",
                    skip_connection_level=True,
                )
                requested_flow = ops.pop("vol_flow", requested_flow)

                # Apply physical limits once
                actual_flow = self.compute_actual_phase_outlet_flow(vessel_phase,requested_flow)

                # Only the intensive state is inherited from the vessel
                # phase: the amount is set by the resolved outlet flow, and
                # one composition basis fixes the rest. Reading the vessel
                # phase's full state_dict here evaluated every composition
                # representation just to discard all but one of them.
                composition_name = outlet_phase.default_composition_name

                updates = {
                    "temp": vessel_phase.temp,
                    "pres": vessel_phase.pres,
                    composition_name: getattr(vessel_phase, composition_name),
                }

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

            resolved.append(ResolvedStreamConnection(connection=connection,
                                                     transfers=transfers,
                                                     stream=outlet_stream))

        return StreamConditions(resolved)
        
    def resolve_inlets(self,completed_state,operating_conditions)->StreamConditions:

        resolved = []

        for connection_num, connection in enumerate(self.inlet_connections):

            inlet_stream = self._get_inlet_workspace(connection_num, connection)
            transfers= []
            for mapping in connection.phase_mappings:
                stream_phase = inlet_stream.get_phase_from_ref(mapping.source_phaseref)

                vessel_phase = self.phase_states.get_phase(mapping.sink_phaseref)


                # As on the outlet side, a connection-level vol_flow
                # describes the whole stream. Handing it to each phase would
                # multiply a multi-phase feed by the number of phases.
                ops = self.get_phase_operating_conditions(
                    operating_conditions,
                    connection_num,
                    mapping.source_phaseref,
                    "inlet",
                    skip_connection_level=True,
                )

                self._apply_inlet_overrides(
                    stream_phase,
                    connection.stream.get_phase_from_ref(
                        mapping.source_phaseref
                    ),
                    ops,
                    (connection_num, mapping.source_phaseref),
                )


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
                                                     transfers=transfers,
                                                     stream=inlet_stream))


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

                material_slice = self._material_slice_by_phase.get(
                    transfer.mapping.sink_phaseref
                )
                if material_slice is not None:
                    buffer.contributions[
                        buffer.OUTLET,
                        material_slice,
                    ] -= transfer.species_flow

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
                            ] -= value

                            transfer.material_writes.append(
                                (state_slice, value)
                            )

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
            material_writes = []

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

                    material_writes.append((material_slice, rate))

            self._timers['crossphase_contributions'] = self._timers.get('crossphase_contributions',0)+perf_counter()-t0

            crossphase_result.aux["material_writes"] = material_writes

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

        # The resolved streams live in a workspace that later evaluations
        # overwrite, so the reported final condition takes its own copy.
        self.outlet_conditions = copy.deepcopy(resolved_outlets)

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
