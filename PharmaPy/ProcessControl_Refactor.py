from PharmaPy.DataClasses import (PhaseConnection,PhaseMapping,PhaseRef,PhaseStateCollection,PhaseStateVariable,
                                  IntraPhaseProcess,StateCollection,StateKey,StateVariable,StreamConnection,
                                  OperatingKey, StateEvent)
from typing import Any
import numpy as np


class Controller:

    def __init__(self):
        self.states = {}
        self.operating_conditions = {}

    def reset(self):
        self.states.clear()
        self.operating_conditions.clear()

    def compute_states(
        self,
        time,
        completed_state,
        unit,
    ):

        self.states = {}

        self.update_state(
            time,
            completed_state,
            unit,
        )

        return self.states.copy()


    def observe(
        self,
        time,
        completed_state,
        unit,
        resolved_inlets=None,
        resolved_outlets=None,
    ):
        """
        Update controller measurements/internal state.

        This should not modify operating conditions.
        """

        pass


    def compute_operating_conditions(
        self,
        time,
        completed_state,
        unit,
        resolved_inlets=None,
        resolved_outlets=None,
    )->dict[OperatingKey,Any]:

        self.operating_conditions = {}

        self.actuate(
            time,
            completed_state,
            unit,
            resolved_inlets,
            resolved_outlets,
        )

        return self.operating_conditions.copy()


    def actuate(
        self,
        time,
        completed_state,
        unit,
        resolved_inlets=None,
        resolved_outlets=None,
    ):
        """
        Set manipulated variables through operating_conditions.
        """

        pass


    def update_state(
        self,
        time,
        completed_state,
        unit,
    ):
        pass
    def get_events(self,unit):
            return []
class SimpleTemperatureController(Controller):
    def __init__(self,temp_func,temp_0 =273.15):
        super().__init__()
        self.temp_func = temp_func
        self.temp_0 = temp_0
        self.states[StateKey('global_temp')]=temp_0
    def reset(self):

        super().reset()

        self.states[
            StateKey("global_temp")
        ] = self.temp_0
    def update_state(self, time, completed_state, unit):
        statekey = StateKey('global_temp')
        self.states[statekey] = self.temp_func(time)
        
    
class DefaultContinuousVesselVolume(Controller):

    def __init__(
        self,
        target_volume=None,
        K=1e4,
    ):

        super().__init__()

        self.target_volume = target_volume
        self.K = K


    def observe(
        self,
        time,
        completed_state,
        unit,
        resolved_inlets=None,
        resolved_outlets=None,
    ):

        if self.target_volume is None:
            self.target_volume = unit.Phases.vol
        

    def actuate(
        self,
        time,
        completed_state,
        unit,
        resolved_inlets=None,
        resolved_outlets=None,
    ):

        if resolved_inlets is None:
            return

        inlet_flow = sum(
            transfer.stream_phase.vol_flow
            for transfer in resolved_inlets.streams[0].transfers
        )

        volume_error = unit.Phases.vol - self.target_volume

        outlet_flow = inlet_flow + self.K * volume_error

        self.operating_conditions[
            OperatingKey(
                "vol_flow",
                connection=0,
                port='outlet'
            )
        ] = max(outlet_flow,0.0)
    def get_events(self, unit):

        return [
            StateEvent(
                name='outlet_flow',
                function=lambda t, state, unit: unit.Phases.vol - 2,
                direction=0,
                terminal=False,
            )
        ]

class TankLevelController(Controller):

    def update_operating_conditions(
            self,
            time,
            completed_state,
            unit,
        ):

        vessel_vol = unit.Phases.vol

        inlet_flow = sum(
            connection.stream.vol_flow
            for connection in unit.inlet_connections
        )

        outlet_flow = (
            0.0
            if vessel_vol < 2
            else inlet_flow
        )

        self.operating_conditions[
            OperatingKey(
                "vol_flow",
                connection=0,
            )
        ] = outlet_flow

class ComplexController(Controller):

    def update_operating_conditions(
            self,
            time,
            completed_state,
            unit,
        ):

        self.update_inlet_conditions(
            time,
            completed_state,
            unit,
        )

        self.update_outlet_conditions(
            time,
            completed_state,
            unit,
        )

    def update_inlet_conditions(self,time,completed_state,unit):
        ...

    def update_outlet_conditions(self,time,completed_state,unit):
        ...
        