from PharmaPy.MultiPhaseVessel import MultiPhaseVessel
from Mechanisms import ReactionMechanism
from PharmaPy.DataClasses import *
from ProcessControl_Refactor import Controller, DefaultContinuousVesselVolume

class _BaseReactor(MultiPhaseVessel):
    def __init__(self, integrator=None, temp_ref=273.15, isothermal=False, reset_states=False, controller=Controller(), h_conv=0, state_events={}, adiabatic=False, jac_type="AD", Phases=None, basis='mass_j', ht_mode="jacket", diam=0, area_base=0):
        super().__init__(integrator, temp_ref, isothermal, reset_states, controller, h_conv, state_events, adiabatic, jac_type, Phases, basis, ht_mode, diam, area_base)
        if self.isothermal:
                    assert self.adiabatic != 1, "Cannot be isothermal and adiabatic with a reaction present"
    @property
    def RxnKinetics(self):
        raise AttributeError(
            "RxnKinetics is a convenience initializer only. Modify self.reaction_regions directly instead"
            "Use reaction_regions[ind].kinetics if kinetics access are desired."
        )

    @RxnKinetics.setter
    def RxnKinetics(self, instance):
        """Conveniene initializer to instantiate reaction kinetics in their proper regimes"""
        if not isinstance(instance,list):
            instance = [instance]
        self._create_intraphaseProcesses_from_RxnKinetics(instance)
        self._post_RxnKinetics_setter(instance)

    def _create_intraphaseProcesses_from_RxnKinetics(self,RKs):
        '''Creates intraphase_processes from RxnKinetics, assuming everything is in the liquid phase corresponding to the index of the RxnKinetic
        This is because RxnKinetics handles multiple reactions, but assumes a single phase
        If reaction_regions is already defined, this step is ignored
        For reactions in other phases, the user must define reaction_regions directly
        '''
        if len(self.intraphase_processes) >0:
            raise RuntimeError(
                "reaction_regions already defined. "
                "Cannot use RxnKinetics convenience API."
            )
        processes = []
        for i,rk in enumerate(RKs):
            assert len(self.Phases.Liquids)>=i-1, "The number of reaction kinetics must match or be less than the number of liquid phases or you must specify reaction_regions manually"
            region = IntraPhaseProcess(phaseref=PhaseRef("liquid",i),
                                    mechanism=ReactionMechanism(rk,self.Phases.Liquids[i]))
            processes.append(region)
        self.intraphase_processes = processes

    def _post_RxnKinetics_setter(self,RKs):
        "Place holder in case future children need special behavior"
        self.nomenclature(overwrite=True)
class BatchReactor(_BaseReactor):

    oper_mode = "batch"

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

class SemiBatchReactor(_BaseReactor):

    oper_mode = "semibatch"

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

class ContinuousReactor(_BaseReactor):

    oper_mode = "continuous"

    def __init__(self, integrator=None, temp_ref=273.15, isothermal=False, reset_states=False, controller=DefaultContinuousVesselVolume(), h_conv=0, state_events={}, adiabatic=False, jac_type="AD", Phases=None, basis='mass_j', ht_mode="jacket", diam=0, area_base=0):
        super().__init__(integrator, temp_ref, isothermal, reset_states, controller, h_conv, state_events, adiabatic, jac_type, Phases, basis, ht_mode, diam, area_base)

    def configure_default_connections(self):

        # User already supplied outlets
        if len(self.outlet_connections) > 0:
            return

        stream = self.Phases.to_stream()

        mappings = []

        counts = {}

        for phase in self.Phases:

            phase_type = phase.phase_family.lower()

            idx = counts.get(phase_type, 0)
            counts[phase_type] = idx + 1

            ref = PhaseRef(phase_type, idx)

            mappings.append(
                PhaseMapping(
                    source_phaseref=ref,
                    sink_phaseref=ref,
                )
            )

        self.outlet_connections = [
            StreamConnection(
                stream=stream,
                phase_mappings=mappings,
            )
        ]
    