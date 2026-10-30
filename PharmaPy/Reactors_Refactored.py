from PharmaPy.MultiPhaseVessel import MultiPhaseVessel
from PharmaPy.Mechanisms import ReactionMechanism
from PharmaPy.DataClasses import *
from PharmaPy.ProcessControl_Refactored import Controller, DefaultContinuousVesselVolume

class _BaseReactor(MultiPhaseVessel):

    # Forwarded wholesale rather than re-declared: repeating the base
    # signature here and passing it on positionally meant every parameter
    # added to MultiPhaseVessel silently failed to reach a reactor.
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.isothermal:
                    assert self.adiabatic != 1, "Cannot be isothermal and adiabatic with a reaction present"
    @property
    def RxnKinetics(self):
        raise AttributeError(
            "RxnKinetics is a convenience initializer only. Modify "
            "self.intraphase_processes directly instead. Use "
            "self.intraphase_processes[ind].mechanism.kinetics if kinetics "
            "access is desired."
        )

    @RxnKinetics.setter
    def RxnKinetics(self, instance):
        """Convenience initializer to instantiate reaction kinetics in their proper regimes."""
        if not isinstance(instance,list):
            instance = [instance]
        self._create_intraphaseProcesses_from_RxnKinetics(instance)
        self._post_RxnKinetics_setter(instance)

    def _create_intraphaseProcesses_from_RxnKinetics(self,RKs):
        '''Creates intraphase_processes from RxnKinetics, assuming everything is in the liquid phase corresponding to the index of the RxnKinetic.
        This is because RxnKinetics handles multiple reactions, but assumes a single phase.
        For reactions in other phases, the user must define intraphase_processes directly.
        '''
        if len(self.intraphase_processes) >0:
            raise RuntimeError(
                "intraphase_processes already defined. "
                "Cannot use RxnKinetics convenience API."
            )
        liquid_phases = (
            self.Phases.phases_by_type.get("liquid", [])
            if self.Phases is not None
            else []
        )
        if len(liquid_phases) < len(RKs):
            raise RuntimeError(
                "The number of reaction kinetics must match or be less than "
                "the number of liquid phases in self.Phases, or specify "
                "self.intraphase_processes manually."
            )
        processes = []
        for i,rk in enumerate(RKs):
            region = IntraPhaseProcess(phaseref=PhaseRef("liquid",i),
                                    mechanism=ReactionMechanism(rk,self.Phases.Liquids[i]))
            processes.append(region)
        self.intraphase_processes = processes

    def _post_RxnKinetics_setter(self,RKs):
        "Place holder in case future children need special behavior"
        self.nomenclature(overwrite=True)
class BatchReactor(_BaseReactor):

    oper_mode = "Batch"

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

class SemiBatchReactor(_BaseReactor):

    oper_mode = "Semibatch"

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

class ContinuousReactor(_BaseReactor):

    oper_mode = "Continuous"

    def __init__(self, *args, controller=None, **kwargs):
        # Constructed per instance; a default argument would be shared by
        # every ContinuousReactor in the session.
        if controller is None:
            controller = DefaultContinuousVesselVolume()
        super().__init__(*args, controller=controller, **kwargs)

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
