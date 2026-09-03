import numpy as np
import copy
from PharmaPy.DataClasses import (StateVariable,PhaseConnection,PhaseMapping,
                                  PhaseRef,PhaseStateCollection,PhaseStateVariable,
                                  StateKey,StateCollection,StreamConnection,IntraPhaseProcess,
                                  TransferResult
)
from PharmaPy.Phases import BasePhase

eps = np.finfo(float).eps

class Mechanism:
    """
    Base class for any mechanism that contributes material and/or energy.

    A mechanism may:
        * own internal differential states
        * own algebraic output states
        * compute material rates
        * compute heat generation

    The vessel knows nothing about the implementation.
    """

    solver_states = ()
    output_states = ()
    owning_phase = None

    def __init__(self):
        self.exposed_attributes = set()

    def _update_exposed_attributes(self):
        self.exposed_attributes.update(
            state.name for state in self.solver_states
        )

    def expose(self, *names):
        self.exposed_attributes.update(names)

    @property
    def solver_state_keys(self):
        return tuple(
            StateKey(state.name, state.phaseref)
            for state in self.solver_states
        )
    def owns_state(self, state_key):
        """
        Return True if this mechanism owns this transported state.
        """
        return state_key in self.solver_state_keys
    def reset(self):
        """Optional."""
        pass

    def update_state(self, completed_state,**kwargs):

        for variable in self.solver_states:

            key = StateKey(variable.name, variable.phaseref)

            if key in completed_state:
                setattr(
                    self,
                    variable.name,
                    completed_state[key]
                )

    def add_solver_state_variables(
        self,
        collection,
        overwrite=False,
        phase_ref=None
    ):
        for state in self.solver_states:
            if phase_ref is None and state.phaseref is None:
                raise ValueError(f"{type(self).__name__} defines solver states but has no owning phase reference")
            copiedState = copy.deepcopy(state)
            copiedState.phaseref = phase_ref
            collection.add(copiedState, overwrite)

    def add_output_state_variables(
            self,
            state_collection,
            overwrite=False,
            **kwargs
        ):

        for state in self.output_states:
            state_collection.add(
                copy.deepcopy(state),
                overwrite
            )
    def get_solver_state_rates(self, **kwargs)->TransferResult:
        raise NotImplementedError

    def get_heat_generation(
        self,
        aux,
        completed_state,
        time
    ):
        return 0.0
    def get_algebraic_residuals(
        self,
        **kwargs,
    ):
        """
        Return algebraic residuals keyed by StateKey.

        Differential-only mechanisms simply return {}.
        """
        return TransferResult({},{},0)
    def get_override(self,name):
        return None
    def get_inlet_contributions(
        self,
        stream_phase,
        vessel_phase,
        amount,
        completed_state,
    ):
        return {}

    def get_outlet_contributions(
        self,
        vessel_phase,
        outlet_phase,
        amount,
        completed_state,
    ):
        return {}
    def get_events(self,unit):
        return []
class CrossPhaseTransferMechanism(Mechanism):
    """
    Computes material exchanged between two phases.

    Returns
    -------
    transfer_rate : ndarray
        Species mass transfer rates.

    aux : dict
        Cached information required by get_heat_generation().
    """
    def __init__(self):
        super().__init__()
    def get_solver_state_rates(self,
            source_phase,
            sink_phase,
            connection,
            completed_state,
            time,
        )->TransferResult:

        raise NotImplementedError
    def update_outlet(self, outlet_phase, completed_state, operating_conditions):
        pass
    def update_inlet(self, inlet_phase, completed_state, operating_conditions):
        pass
class MechanismView:

    def __init__(self, mechanism, direction):
        self.mechanism = mechanism
        self.direction = direction

    def __getattr__(self, name):
        return getattr(self.mechanism, name)

    def get_solver_state_rates(
            self,
            *args,
            **kwargs,
        ):

        result = self.mechanism.get_solver_state_rates(
            *args,
            **kwargs,
        )

        if self.direction == "forward":
            if result.net_mass_rate < 0:
                return self.zero_out(result)

        else:
            if result.net_mass_rate > 0:
                return self.zero_out(result)

        return result
    def zero_out(self,result:TransferResult):
        return TransferResult(
            state_rates={},
            aux=result.aux,
            net_mass_rate=0.0,
        )
    def get_heat_generation(self, *args, **kwargs):

        if self.direction == "reverse":
            return getattr(self.mechanism.mechanism_kinetics,'heat_dissol',0.0)

        return self.mechanism.get_heat_generation(
            *args,
            **kwargs,
        )

    
class ReversibleTransferMechanism:

    def __init__(
        self,
        source_mechanism:CrossPhaseTransferMechanism,
    ):

        self.forward = MechanismView(source_mechanism,"forward")
        self.reverse  = MechanismView(source_mechanism,"reverse")

    
class ReactionMechanism(Mechanism):
    """
    Computes intraphase material generation or consumption.

    Returns
    -------
    species_mass_rates : ndarray
        Species generation/consumption rates.

    aux : dict
        Cached reaction information required by
        get_heat_generation().
    """
    def __init__(
        self,
        kinetics,
        owning_phase,
        molarity_in_L=True
    ):
        super().__init__()
        self.kinetics = kinetics
        self.molarity_in_L = molarity_in_L
        self.owning_phase=owning_phase
    def get_solver_state_rates(
            self,
            process:IntraPhaseProcess,
            phase,
            time,
            completed_state
        )->TransferResult:
        "aux must have process field that stores process"
        # mole_adjust = 1000 if self.molarity_in_L else 1
        

        temp = phase.temp

        mask = np.array([
            species in self.kinetics.partic_species
            for species in phase.name_species
        ])

        conc = np.maximum(phase.mole_conc,0.0) #sanitize input incase integrator gave slightly negative
        deltah_rxn = None

        if self.kinetics.keq_params is not None:
            deltah_rxn = phase.getHeatOfRxn(
                self.kinetics.stoich_matrix,
                temp,
                mask,
                self.kinetics.delta_hrxn,
                self.kinetics.tref_hrxn
            )


        reaction_rates,species_rates = self.kinetics.get_rxn_rates(
            conc[mask],
            temp,
            return_both=True,
            delta_hrxn = deltah_rxn
        )

        species_massPerVol_rates = np.zeros(phase.num_species)
        species_massPerVol_rates[mask] = species_rates
        species_massPerVol_rates *= phase.mw
        species_mass_rates = species_massPerVol_rates* phase.vol#*mole_adjust
        state_rates = {StateKey('mass_j',process.phase):species_mass_rates}
        aux = {
                "phase": phase,
                "rxn_rates": reaction_rates,
                "process":process
                }
        result = TransferResult(state_rates=state_rates,aux=aux,net_mass_rate=species_mass_rates.sum())
        return result
    
    def get_heat_generation(self, aux, completed_state, time):
        # mole_adjust = 1000 if self.molarity_in_L else 1
        phase = aux['phase']
        temp = phase.temp
        rk = self.kinetics
        mask = np.array([
            species in rk.partic_species
            for species in phase.name_species
        ])

        deltah_rxn = (
            phase.getHeatOfRxn(
                rk.stoich_matrix,
                temp,
                mask,
                rk.delta_hrxn,
                rk.tref_hrxn
            ))
        q =  -(deltah_rxn * aux["rxn_rates"]).sum()* phase.vol #*mole_adjust # molarity here is mol/L, but J/kg is expected for internal consistency
        return q
    
    


   
    

    
class DirectTransfer(CrossPhaseTransferMechanism):

    def get_solver_state_rates(
            self,
            source_phase,
            sink_phase,
            kinetics,
            **kwargs):

        return kinetics.get_rate(
            source_phase,
            sink_phase,
            **kwargs
        )
    
class PopulationBalanceMechanism(CrossPhaseTransferMechanism):
    """
    Base class for crystallization population balance mechanisms.

    Responsibilities
    ----------------
    * Evaluate crystallization kinetics.
    * Compute supersaturation/solubility.
    * Compute distribution moments.
    * Convert crystal growth into species transfer rates.
    * Define common output variables.

    Child classes are responsible only for solving the population
    balance equation (FVM, MOM, QMOM, etc.).
    """

    def __init__(
        self,
        owning_phase:BasePhase,
        target_components:str|list,
        solvent_name:str,
        kinetics=None,
        density=None,
        kv=1,
        fraction=None
    ):
        """
        owning_phase: BasePhase the instance of the phase that owns the population balance (e.g. the solid phase for a crystallization)
            the owning_phase.mass_frac should be the mass_frac of the resulting members. for multiple mass_frac
            you will need multiple mechanisms
        kinetics: the kinetics object that describes how the mechanism takes place. If it is not present, 
            the mechanism only exists to track state properties not dynamics
        kv: float the volumetric shape factor
            
        """
        super().__init__()
        self._mechanism_kinetics = kinetics
        self.owning_phase = owning_phase
        if isinstance(target_components, str):
            target_components = [target_components]
        self.target_components = target_components
        self.solvent_name =solvent_name
        
        self._density = density

        if self.target_components is not None:
            self.target_ind = []
            for tc in self.target_components:
                name_bool = [name == tc for name in self.owning_phase.name_species] #TODO check that it selects correctly
                self.target_ind.append(np.where(name_bool)[0][0])
        self.solvent_ind = self.owning_phase.name_species.index(self.solvent_name)

        
        self.kv = kv
        self.output_states=[StateVariable(name="supersat",dim=len(self.target_ind),units="-",state_type="post"),
                    StateVariable(name="solubility",dim=len(self.target_ind),units="kg/m3",state_type="post"),
                    StateVariable(name="mu_n",dim=4,index=[0, 1, 2, 3],units="various",state_type="post")]
        if fraction is None:
            fraction = np.zeros(self.owning_phase.num_species)
            fraction[self.target_ind] = np.full(len(self.target_components),1/len(self.target_components))
        self.fraction = fraction

    @property
    def mechanism_kinetics(self):
        if self._mechanism_kinetics is not None:
            return self._mechanism_kinetics
        raise AttributeError("No kinetics were specified")
    
    @mechanism_kinetics.setter
    def mechanism_kinetics(self,value):
        if value is not None:
            self._mechanism_kinetics = value
        value.target_idx = self.target_ind # this is needed for the Base PharmaPy CrystKinetics, but perhaps not for arbitrary user kinetics?

    def getDensity(self):
        return self.density
    
    @property
    def density(self):
        if self._density is not None:
            return self._density
        return self.owning_phase.density
    
    def get_override(self, name):

        overrides = {
            "mass": self.get_mass,
            "set_mass": self.set_mass,
            "getDensity": self.getDensity if self._density is not None else None,
        }

        return overrides.get(name)
    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def compute_moments(
        self,
        distrib,
        x_grid,
    ):
        """
        Compute moments from a number density distribution.
        """

        moments = np.zeros(4)

        for n in range(4):
            moments[n] = np.trapezoid(
                distrib * x_grid**n,
                x_grid,
            )

        return moments

    def compute_solubility(
        self,
        liquid,
    ):
        """
        Returns equilibrium solubility.

        Delegated entirely to the kinetics object.
        """

        return self.mechanism_kinetics.get_solubility(
            liquid.temp,
            liquid,
        )

    def compute_supersaturation(
        self,
        liquid,
        solubility,
    ):

        conc = liquid.mass_conc

        

        return conc[self.target_ind] - solubility

    # ------------------------------------------------------------------
    # Crystal -> liquid coupling
    # ------------------------------------------------------------------

    def compute_species_transfer(
        self,
        crystal_mass_rate,
        liquid,
    ):
        """
        Convert crystal growth (kg crystal/s)
        into species transfer.

        Positive crystal growth removes material
        from the liquid.
        """

        rates = np.zeros(liquid.num_species)

        mask = self.target_ind

        # Assume equal split unless overridden later.
        rates[mask] = -crystal_mass_rate/ len(mask)

        return rates
    def get_solid_mass(self):
        raise NotImplementedError
    # ------------------------------------------------------------------
    # API for child classes
    # ------------------------------------------------------------------

    def solve_population_balance(
        self,
        liquid:BasePhase,
        solid:BasePhase,
        completed_state:dict[StateKey],
        time:float,
    )->TransferResult:
        """
        Child classes implement this.

        Returns
        -------
        distribution_rate
        crystal_mass_rate
        aux
        """

        raise NotImplementedError

    # ------------------------------------------------------------------
    # Vessel interface
    # ------------------------------------------------------------------

    def get_solver_state_rates(
        self,
        source_phase:BasePhase,
        sink_phase:BasePhase,
        connection:PhaseConnection,
        completed_state:dict[StateKey],
        time:float,
    )->tuple[dict[StateKey],dict]:
        """
        Called by the vessel.

        The liquid phase is the source.

        The solid phase is the sink.
        """

        result= self.solve_population_balance(source_phase,sink_phase,completed_state,time,connection)
        if not hasattr(self,"liquid_phase_ref"):
            self.liquid_phase_ref = connection.source_phase

        result.aux.update(
            {
                "connection": connection,
                "state_rates": result.state_rates,
                "liquid_phase": source_phase,
                "solid_phase": sink_phase,
            }
        )
        return result

    # ------------------------------------------------------------------
    # Heat generation
    # ------------------------------------------------------------------

    def get_heat_generation(
        self,
        aux,
        completed_state,
        time,
    ):
        """
        Heat of crystallization.

        Default assumes none.
        """

        return 0.0
    def update_state(self, completed_state, unit=None):

        super().update_state(completed_state, unit=unit)

        if unit is None:
            return

        #we need the liquid volume since moments/csd are  per (m3_liquid)
        self.reference_vol = unit.Phases.get_phase_from_ref(self.liquid_phase_ref).vol
    
class OneDFVMMechanism(PopulationBalanceMechanism):

    def __init__(
        self,
        owning_phase:BasePhase,
        target_components:str|list,
        solvent_name:str,
        x_grid,
        distrib_init:np.ndarray|None,
        kinetics=None,
        density=None,
        kv=1,
        distribution_state_name="distrib",
        scale=1
    ):
        """
        Assumes an x_grid of constant dx
        """
        super().__init__(
            owning_phase=owning_phase,
            target_components=target_components,
            solvent_name=solvent_name,
            kinetics=kinetics,
            density=density,
            kv=kv)

        self.x_grid = np.asarray(x_grid)
        self.dx = self.x_grid[1] - self.x_grid[0]
        self.rad = self.x_grid[0]
        self.distribution_state_name = distribution_state_name
        assert len(distrib_init)==len(x_grid), "x_grid and distrib must be the same length"
        setattr(self,distribution_state_name,distrib_init)
        # self.x_grid is in microns and distrib is counts per micron-bin per m3
        self.solver_states = (
            StateVariable(
                name=distribution_state_name,
                phaseref=None,
                dim=len(self.x_grid),
                units="#/(micron m3)",
                state_type="diff",
            ),
        )

        self.scale =scale
        self._update_exposed_attributes()
        self.expose('x_grid')

    def get_mass(self):
        m3 = self.compute_moments(getattr(self,self.distribution_state_name),self.x_grid)[3]
        return self.getDensity()*self.kv*m3*self.reference_vol
    def set_mass(self, mass):

        if mass is None:
            return

        target_m3 = mass/ (self.getDensity()* self.kv* self.reference_vol)

        self.set_third_moment(target_m3)
    def set_third_moment(self, target_m3):

        distribution = getattr(self,self.distribution_state_name)

        current_m3 = self.compute_moments(distribution,self.x_grid)[3]

        if current_m3 <= 0:
            raise ValueError("Cannot scale a distribution with zero third moment.")

        scale = target_m3 / current_m3

        setattr(self,self.distribution_state_name,distribution * scale)

    def solve_population_balance(
        self,
        liquid:BasePhase,
        solid:BasePhase,
        completed_state:dict[StateKey],
        time:float,
        connection:PhaseConnection
    ) -> TransferResult:

        csd = getattr(self,self.distribution_state_name)

        moms = self.compute_moments(csd,self.x_grid)

        mu2 = moms[2] #total surface area

        solubility = self.compute_solubility(liquid)

        supersat = self.compute_supersaturation(
            liquid,
            solubility,
        )

        conc = liquid.mass_conc #TODO check if these are the units expected by CrystKin

        nucl, growth, dissol = (
            self.mechanism_kinetics.get_kinetics(
                conc,
                liquid.temp,
                self.kv,
                moms,
            )
        )
        nucl *= self.scale*liquid.vol
        impurity_factor = self.mechanism_kinetics.alpha_fn(conc) #TODO check if con is the right units
        growth *= impurity_factor

        gparams = self.mechanism_kinetics.params["growth"]

        boundary = nucl / np.maximum(growth, eps)
        f_aug = np.concatenate(([boundary, boundary],csd,[csd[-1]]))

        # Flux source terms
        f_diff = np.diff(f_aug)
        if growth > 0:
            theta = (f_diff[:-1]/ (f_diff[1:] + eps * 10))
        else:
            theta = (f_diff[1:]/ (f_diff[:-1] + eps * 10))

        #Van-Leer limiter
        limiter = np.zeros_like(f_diff)
        limiter[:-1] = ((np.abs(theta) + theta)/ (1 + np.abs(theta)))

        # Constant growth
        if len(gparams) == 3:

            growth_term = growth* (f_aug[1:-1]+ 0.5 * f_diff[1:] * limiter[:-1])
            dissol_term = dissol* (f_aug[2:]- 0.5 * f_diff[1:] * limiter[1:])
            mass_transfer = self.density* self.kv* (3 * (growth + dissol) * mu2+ nucl * self.rad**3)* 1e-6

        # Size-dependent growth
        else:

            alpha = gparams[3]
            beta = gparams[4]

            growth_dep = (growth* (1 + beta * self.x_grid) ** alpha)
            dissol_dep = dissol* np.ones_like(self.x_grid)
            growth_pad = np.append(growth_dep,growth_dep[-1],)
            dissol_pad = np.append(dissol_dep,dissol_dep[-1])
            growth_term = growth_pad* (f_aug[1:-1]+ 0.5 * f_diff[1:] * limiter[:-1])
            dissol_term = dissol_pad* (f_aug[2:]- 0.5 * f_diff[1:] * limiter[1:])
            r = self.x_grid
            growth_int = np.trapezoid(growth_dep * csd * r**2,r)
            dissol_int = np.trapezoid(dissol_dep * csd * r**2,r)
            mass_transfer = (self.density* self.kv* 3* 
                             (growth_int+ dissol_int+ nucl * self.rad**3)* 1e-18)

        flux = growth_term + dissol_term

        dcsd_dt = -np.diff(flux) / self.dx

        aux = {
            "supersaturation": supersat,
            "solubility": solubility,
            "moments": moms,
            "growth": growth,
            "dissolution": dissol,
            "nucleation": nucl,
            "flux": flux,
        }
        species_rates_out = self.compute_species_transfer(mass_transfer,liquid)

        state_rates = {StateKey(self.solver_states[0].name,self.solver_states[0].phaseref): dcsd_dt}
        state_rates.update({StateKey('mass_j',connection.source_phase):species_rates_out})
        result = TransferResult(state_rates=state_rates,aux=aux,net_mass_rate=mass_transfer)
        return result
    def get_solid_mass(self):

        csd = getattr(
            self,
            self.distribution_state_name,
            None
        )

        if csd is None:
            return 0.0

        mu3 = np.trapezoid(
            csd * self.x_grid**3,
            self.x_grid
        )

        return (
            self.density
            * self.kv
            * mu3
            * 1e-18
        )
   

class MomentsPopulationBalance(PopulationBalanceMechanism):
    
    def add_output_state_variables(self, outputs):
        super().add_output_state_variables(outputs)
        outputs.add(
            StateVariable(
                name="mu_n",
                dim=4,
                index=list(range(4)),
                units="m**n",
                state_type="post"
            )
        )







def method_of_moments(self, mu, conc, temp, params, rho_cry, vol=1):
        kv = self.Solid_1.kv # shape factor

        # Kinetics
        if self.basis == 'mass_frac':
            rho_liq = self.Liquid_1.getDensity()
            comp_kin = conc / rho_liq
        else:
            comp_kin = conc

        # Kinetic terms
        mu_susp = mu*(1e-6)**np.arange(self.num_distr) / vol  # m**n/m**3_susp
        nucl, growth, dissol = self.CrystKinetics.get_kinetics(comp_kin, temp, kv,
                                                          mu_susp)

        growth = growth * self.CrystKinetics.alpha_fn(conc)

        ind_mom = np.arange(1, len(mu))

        # Model
        dmu_zero_dt = np.atleast_1d(nucl * vol)
        dmu_1on_dt = ind_mom * (growth + dissol) * mu[:-1] + \
            nucl * self.rad**ind_mom
        dmu_dt = np.concatenate((dmu_zero_dt, dmu_1on_dt))

        # Material balance in kg_API/s --> G in um, u_2 in um**2 (or m**2/m**3)
        mass_transf = np.atleast_1d(rho_cry * kv * (
            3*(growth + dissol)*mu[2] + nucl*self.rad**3)) * (1e-6)**3

        return dmu_dt, mass_transf

def fvm_method(self, csd, moms, conc, temp, params, rho_cry,
                   output='dstates', vol=1):

    mu_2 = moms[2]
    #assumes solid1 is target
    kv_cry = self.Solid_1.kv # volumetric shape factor

    # Kinetic terms
    if self.basis == 'mass_frac':
        rho_liq = self.Liquid_1.getDensity()
        comp_kin = conc / rho_liq
    else:
        comp_kin = conc

    nucl, growth, dissol = self.CrystKinetics.get_kinetics(comp_kin, temp,
                                                        kv_cry, moms)

    nucl = nucl * self.scale * vol 

    impurity_factor = self.CrystKinetics.alpha_fn(conc)
    growth = growth * impurity_factor  # um/s 
    gparams = self.CrystKinetics.params['growth']
    

    # dissol = dissol  # um/s
    boundary_cond = nucl / np.maximum(growth, eps) # num/um or num/um/m**3 initial
    f_aug = np.concatenate(([boundary_cond]*2, csd, [csd[-1]])) # TODO adjust for reaction or handled by concentration? 

    # Flux source terms
    f_diff = np.diff(f_aug)
    
    # f_diff[f_diff == 0] = eps  # avoid division by zero for theta

    if growth > 0:
        theta = f_diff[:-1] / (f_diff[1:] + eps*10)
        # theta = f_diff[:-1] / (f_diff[1:] + eps)
        # theta = f_diff[:-1] / f_diff[1:]
    else:
        theta = f_diff[1:] / (f_diff[:-1] + eps*10)
        # theta = f_diff[:-1] / (f_diff[1:] + eps)
        # theta = f_diff[:-1] / f_diff[1:]
    # Van-Leer limiter
    limiter = np.zeros_like(f_diff)
    limiter[:-1] = (np.abs(theta) + theta) / (1 + np.abs(theta))
    if len(gparams)==3:
    
        growth_term = growth * (f_aug[1:-1] + 0.5 * f_diff[1:] * limiter[:-1])
        dissol_term = dissol * (f_aug[2:] - 0.5 * f_diff[1:] * limiter[1:])
    else:
        growth_dependent = growth * (1 + self.x_grid * gparams[4])**gparams[3]
        dissol_dependent = dissol * (1 + self.x_grid * 0) # TODO add size-dependent dissol params
        growth_pad = np.append(growth_dependent,growth_dependent[-1])
        dissol_pad = np.append(dissol_dependent, dissol_dependent[-1])
        growth_term = growth_pad * (f_aug[1:-1] + 0.5 * f_diff[1:] * limiter[:-1])
        dissol_term = dissol_pad * (f_aug[2:] - 0.5 * f_diff[1:] * limiter[1:])
    flux = growth_term + dissol_term

        
    if output == 'flux':
        return flux  # TODO: isn't it necessary to divide by dx?
    elif output=='dstates':
        dcsd_dt = -np.diff(flux) / self.dx

        # Material bce in kg_API/s --> G in um, mu_2 in m**2 (or m**2/m**3)
        # AKA R_v (rho_c*kv*d_mu3_d_t)
        # Handle stoich in material balance
        if len(gparams)==3:
            mass_transfer = rho_cry * kv_cry * (
                3*(growth + dissol)*mu_2 + nucl*self.rad**3) * (1e-6)
        else:
            r_m = self.x_grid
            mass_transfer_growth = np.trapezoid(growth_dependent*csd*r_m**2,r_m)
            mass_transfer_dissol = np.trapezoid(dissol_dependent*csd*r_m**2,r_m)
            mass_transfer_nucl = nucl*self.rad**3
            mass_transfer = rho_cry*kv_cry*3*(mass_transfer_dissol+mass_transfer_growth+mass_transfer_nucl)*1e-18
        return dcsd_dt, np.array(mass_transfer)