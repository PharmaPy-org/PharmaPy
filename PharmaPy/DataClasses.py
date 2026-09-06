from __future__ import annotations
import copy
import string
import numpy as np
import os
from dataclasses import dataclass, field
from typing import Optional, Sequence, Any
from types import MethodType
from collections import OrderedDict
from collections.abc import Callable

import PharmaPy.Kinetics as pk
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PharmaPy.Mechanisms import Mechanism,CrossPhaseTransferMechanism,DirectTransfer
    from PharmaPy.Phases import BasePhase
    from PharmaPy.MixedPhases import MixedPhase,MixedStream
## Dataclasses
@dataclass(frozen=True)
class PhaseRef:
    phase_type: str
    index: int
    def __post_init__(self):
        object.__setattr__(self, "phase_type", str(self.phase_type).lower())
    def __eq__(self,otherPhaseRef):
        return self.phase_type==otherPhaseRef.phase_type and self.index==otherPhaseRef.index

@dataclass
class PhaseConnection:
    #TODO move this to connections when done
    #active_condition checks the source sink and temp and must return a boolean
    source_phaseref: PhaseRef
    sink_phaseref: PhaseRef
    kinetics:pk.CrystKinetics|pk.RxnKinetics
    species_weights: np.ndarray | None = None
    active_condition: callable=lambda source_phase,sink_phase:True
    mechanism:"CrossPhaseTransferMechanism | None" = None



@dataclass
class PhaseMapping:

    source_phaseref: PhaseRef

    sink_phaseref: PhaseRef

@dataclass
class StreamConnection:

    stream: "MixedStream"
    phase_mappings: list[PhaseMapping]
    split_fraction: float = 1.0

class StreamConditions:

    def __init__(self, streams):
        self.streams = streams

    def __iter__(self):
        return iter(self.streams)
@dataclass
class IntraPhaseProcess:
    phaseref:PhaseRef
    mechanism: "Mechanism"



@dataclass
class StateVariable:
    name: str
    dim: int
    units: str
    state_type: str = "post"
    index: Optional[Sequence] = None
    depends_on: tuple = ("time",)
    stream: Optional[str] = None
    phaseref: Optional[PhaseRef] = None

    compute_value: Callable[
        [
            Any,   # state_var
            float, # time
            dict,  # completed_state
            Any,   # context
        ],
        Any
    ] | None = None

    def __post_init__(self):
        if self.compute_value is None and self.state_type=='post':
            self.compute_value = self.default_compute_value

    def as_dict(self):
        """Backward compatibility."""
        out = {
            "dim": self.dim,
            "units": self.units,
            "type": self.state_type,
            "depends_on": list(self.depends_on),
        }

        if self.index is not None:
            out["index"] = self.index

        return out

    def update_variable(self, variable_name, new_value):
        setattr(self, variable_name, new_value)
    

    @staticmethod
    def default_compute_value(
        state_var,
        time,
        completed_state,
        context,
        resolved_inlets=None,
        resolved_outlets=None,
        operating_conditions=None,
    ):
        try:
            return completed_state[
                StateKey(state_var.name, state_var.phaseref)
            ]
        except KeyError:
            raise KeyError(
                f"'{state_var.name}' is not present in the completed state."
            )
    
@dataclass(frozen=True)
class StateKey:
    name: str
    phaseref: PhaseRef | None = None

@dataclass(frozen=True)
class OperatingKey:

    name: str

    connection: int | None = None

    phaseref: PhaseRef | None = None

    component: str | None = None

    port: str | None = None

    def __post_init__(self):
        object.__setattr__(self,'port',self.port.lower())



@dataclass
class ResolvedPhaseTransfer:
    connection: StreamConnection
    mapping: PhaseMapping

    vessel_phase: "BasePhase"
    stream_phase: "BasePhase"

    vol_flow: float
    species_flow: np.ndarray

    direction: str
    def scale(self, factor, basis):

        self.species_flow *= factor

        self.stream_phase.updatePhase(**{basis:self.species_flow})
        
@dataclass
class ResolvedStreamConnection:
    connection: StreamConnection
    transfers: list[ResolvedPhaseTransfer]

    def __iter__(self):
        return iter(self.transfers)

@dataclass
class TransferResult:
    state_rates: dict[StateKey]
    aux: dict
    net_mass_rate:float
@dataclass
class StateCollection:
    states: dict[StateKey, StateVariable] = field(default_factory=dict)

   
    def add(self, state: StateVariable,overwrite=False,error_on_conflict=False):
        key = StateKey(state.name,state.phaseref)
        existing = self.states.get(key)

        if existing is None:
            self.states[key] = state
            return

        same = state == existing

        if same:
            return

        if overwrite:
            self.states[key] = state
            return

        if error_on_conflict:
            raise ValueError(
                f"State {state.name} already exists "
                f"for phase {state.phaseref} and overwrite was False"
            )

    def names(self):
        return [k.name for k in self.states]

    def dims(self):
        return [state.dim for state in self.states.values()]

    def __contains__(self, name):

        if isinstance(name, str):
            return any(k.name == name for k in self.states)

        return name in self.states
    
    def unpack(self, y):

        states = {}

        start = 0

        for key,state in self.states.items():

            end = start + state.dim

            value = y[start:end]

            if state.dim == 1:
                value = value[0]

            states[key] = value

            start = end

        return states
    def pack(self, state_dict):

        values = []

        for key,state in self.states.items():

            value = np.asarray(
                state_dict[key]
            ).flatten()

            values.extend(value)

        return np.asarray(values)
    def unpack_history(self, y_history):
        """
        Parameters
        ----------
        y_history : ndarray
            Shape (num_times, num_solver_states)

        Returns
        -------
        dict
            state_name -> full time history
        """

        history = {}
        start = 0

        for key,state in self.states.items():

            end = start + state.dim
            values = y_history[:, start:end]
            if state.dim == 1:
                values = values[:, 0]

            history[key] = values
            start = end

        return history
    def flatten(self, state_dict):

        flat = {}

        for key, value in state_dict.items():
            if isinstance(key,str):
                flat[key]=value
                continue
            
            flat[self.format_key(key)] = value

        return flat
    @staticmethod
    def format_key(key):

        if key.phaseref is None:
            return key.name

        return (
            f"{key.name}_"
            f"{key.phaseref.phase_type}"
            f"{key.phaseref.index}"
        )
        

@dataclass
class PhaseStateVariable:
    phaseref: PhaseRef
    state: StateVariable

@dataclass
class PhaseStateCollection:
    phasestates: dict[PhaseRef, StateCollection] = field(default_factory=dict)

    def add(self, phase: PhaseRef, state: StateVariable):
        if phase not in self.phasestates:
            self.phasestates[phase] = StateCollection()

        self.phasestates[phase].add(state)

    def __getitem__(self, phase):
        return self.phasestates[phase]
    def __iter__(self):
        for phaseref, collection in self.phasestates.items():
            for state in collection.states.values():
                yield PhaseStateVariable(phaseref, state)


@dataclass
class StateEvent:

    name: str
    function: Callable
    direction: int = 0
    terminal: bool = False
    source: Any = None
    callbakc: Any = None