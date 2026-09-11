from PharmaPy.DataClasses import (PhaseConnection,PhaseMapping,PhaseRef,PhaseStateCollection,PhaseStateVariable,
                                  IntraPhaseProcess,StateCollection,StateKey,StateVariable,StreamConnection,
                                  OperatingKey, StateEvent)
from typing import Any
import numpy as np

eps = np.finfo(float).eps


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
    """
    Hold vessel volume by trimming the outlet around the inlet flow.

    The outlet request is ``inlet + K * (V - V_target)``, so the volume
    error decays with time constant ``1 / K``. That time constant is the
    fastest mode in the model, and it is created entirely by this
    controller rather than by the process: pushing it far below the
    residence time buys no extra level control and costs the integrator
    proportionally many steps.

    ``tau`` therefore sets the gain from a settling time rather than
    exposing a bare gain. The default settles a level upset in a hundredth
    of a residence time, which is tight control and still ~1e4 times less
    stiff than a gain of 1e4 on a vessel of this size.
    """

    def __init__(
        self,
        target_volume=None,
        tau=None,
        K=None,
    ):

        super().__init__()

        self.target_volume = target_volume
        self.tau = tau
        self.K = K

        # Blend width, as a fraction of the inlet flow, over which the
        # no-backflow floor is applied. A hard max() puts a kink in the
        # right-hand side at the point the controller normally sits on.
        self.floor_width = 1e-3


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


    def get_gain(self, inlet_flow):
        """
        Proportional gain, in 1/s.

        An explicit K wins. Otherwise the gain is derived from the
        residence time, so the controller stays equally tight on vessels of
        any size instead of being ferociously stiff on small ones.
        """

        if self.K is not None:
            return self.K

        tau = self.tau

        if tau is None:

            if inlet_flow > 0 and self.target_volume:
                residence_time = self.target_volume / inlet_flow
            else:
                residence_time = 1.0

            tau = 0.01 * residence_time

        return 1.0 / tau

    @staticmethod
    def soft_floor(value, width):
        """
        C1-continuous stand-in for ``max(value, 0)``.

        Equals ``value`` above ``width`` and ``0`` below ``-width``, with a
        quadratic bridge whose slope matches at both ends. The controller
        sits near its floor whenever the vessel is at target, so a hard
        corner there is one the integrator keeps rediscovering.
        """

        if width <= 0:
            return max(value, 0.0)

        if value >= width:
            return value

        if value <= -width:
            return 0.0

        return (value + width) ** 2 / (4.0 * width)


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

        outlet_flow = inlet_flow + self.get_gain(inlet_flow) * volume_error

        self.operating_conditions[
            OperatingKey(
                "vol_flow",
                connection=0,
                port='outlet'
            )
        ] = self.soft_floor(
            outlet_flow,
            self.floor_width * max(inlet_flow, eps),
        )

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
        