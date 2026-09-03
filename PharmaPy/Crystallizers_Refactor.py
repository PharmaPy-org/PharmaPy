from PharmaPy.MultiPhaseVessel import MultiPhaseVessel
from PharmaPy.Mechanisms import *
from PharmaPy.DataClasses import *
from PharmaPy.ProcessControl_Refactor import DefaultContinuousVesselVolume

class _BaseCrystallizer(MultiPhaseVessel):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        
    @property
    def CrystKinetics(self):
        raise AttributeError(
            "CrystKinetics is a convenience initializer only. Modify self.phase_connections directly instead."
            "Use phase_connections[ind].kinetics if the kinetics are desired."
        )

    @CrystKinetics.setter
    def CrystKinetics(self, instance: pk.CrystKinetics|list):
        ''' CrystKinetics is only a convenience initializer ONLY that assumes transfer between liquid1 and solid1 for each Crystkinetic in the list
            and that those connections are always active. It also assumes that target_comp is the only species moving for that index
            THIS CANNOT BE SET BEFORE PHASES
        If any more complex behavior is needed, or if phases need to be set after this, the user should set phase_connections directly using a list of PhaseConnection objects'''
        assert self._Phases is not None, 'Phases must be set before using the crystkinetics convenienve API. Otherwise, you must set phase_connections directly'
        if not isinstance(instance,list) and not isinstance(instance,pk.CrystKinetics):
            raise TypeError("CrystKinetics must be set by either a CrystKinetics object or a list of CrystKinetics objects")
        
        if not isinstance(instance,list):
            instance = [instance]
        self._CrystKinetics = instance
        self._create_default_phase_connections()
        self.nomenclature(overwrite=True)
        self._post_CrystKinetics_setter()
    def _create_default_phase_connections(self):
        ''' Assumes everything occurs between the first liquid and the first solid phases
        Assumes that whatever the target index is for that Crystkinetic, the massfrac is 100% that compound and 0 everything else
        Only runs if phase_connections are not already set'''

        if len(self.phase_connections)>0:
            raise RuntimeError(
                "phase_connections already defined. "
                "Cannot use CrystKinetics convenience API."
            )
        connections = []
        
        for i,ck in enumerate(self._CrystKinetics):
            solidphase_ref = PhaseRef('solid',0)
            try:
                solidphase=self.Phases.get_phase_from_ref(solidphase_ref)
            except IndexError:
                raise IndexError("The solid phase cannot be found, did you initialize the vessel with a solid phase?")
            pbm = solidphase.get_mechanism(OneDFVMMechanism)
            pbm.liquid_phase_ref = PhaseRef("liquid",0)
            weights = pbm.fraction
            if pbm._mechanism_kinetics is None:
                try:
                    pbm.mechanism_kinetics=ck
                except AttributeError:
                    raise AttributeError("Your solid phase does not have kinetics in its mechanism. The solidphase you assign to the vessel must have a mechanism with kinetics or crystallization cannot occur")
            reversible = ReversibleTransferMechanism(source_mechanism=pbm)
            
            if ck.supports(['growth','nucl_prim','nucl_sec']):
                # liquid to solid because crystallization is valid
                connection = PhaseConnection(source_phase=PhaseRef("liquid",0),
                                             sink_phase=solidphase_ref,
                                             kinetics=ck,
                                             species_weights=weights,
                                             active_condition=lambda source,sink:True,
                                             mechanism=reversible.forward
                                             )
                connections.append(connection)
            if ck.supports('dissolution'):
                #solid to liquid because dissolution
                connection = PhaseConnection(source_phase=PhaseRef("solid",0),
                                             sink_phase=PhaseRef('liquid',0),
                                             kinetics=ck,
                                             species_weights=weights,
                                             active_condition=lambda source,sink:True,
                                             mechanism=reversible.reverse

                                             )
                connections.append(connection)
        self.phase_connections = connections

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
                if phasestatevar.state.name==self.basis and phasestatevar.phaseref == PhaseRef("liquid",0):
                    phasestatevar.state.update_variable('state_type','diff')
    def _post_CrystKinetics_setter(self):
        "Place holder in case future children need special behavior"
        pass
    def configure_solver(self):
        #Assimulo option, does nothing if not using assimulo backend
        self.integrator._solver.linear_solver = "SPGMR"
    def _post_set_phases(self):
        super()._post_set_phases()
        
        



class BatchCrystallizer(_BaseCrystallizer):

    oper_mode = "batch"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)



class SemiBatchCrystallizer(_BaseCrystallizer):

    oper_mode = "semibatch"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)



class ContinuousCrystallizer(_BaseCrystallizer):

    oper_mode = "continuous"

    def __init__(
        self,
        controller=DefaultContinuousVesselVolume(),
        **kwargs
    ):

        super().__init__(
            controller=controller,
            **kwargs
        )


    def configure_default_connections(self):

        if len(self.outlet_connections)>0:
            return


        stream = self.Phases.to_stream()

        mappings=[]

        counts={}

        for phase in self.Phases:

            phase_type = phase.phase_family.lower()

            idx = counts.get(
                phase_type,
                0
            )

            counts[phase_type]=idx+1


            ref = PhaseRef(
                phase_type,
                idx
            )

            mappings.append(
                PhaseMapping(
                    source_phase=ref,
                    sink_phase=ref
                )
            )


        self.outlet_connections=[
            StreamConnection(
                stream=stream,
                phase_mappings=mappings
            )
        ]