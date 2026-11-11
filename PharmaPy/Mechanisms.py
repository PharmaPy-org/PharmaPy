from __future__ import annotations
import warnings
import numpy as np
import copy
from PharmaPy.DataClasses import (StateVariable,PhaseConnection,PhaseMapping,
                                  PhaseRef,PhaseStateCollection,PhaseStateVariable,
                                  StateKey,StateCollection,StreamConnection,IntraPhaseProcess,
                                  TransferResult,StateEvent
)
from PharmaPy.Phases_Refactored import BasePhase
from PharmaPy.Errors import (PharmaPyNonImplementedError, PharmaPySpecificationError,
                             PharmaPyTypeError, PharmaPyValueError)
from PharmaPy.CrystallizationKernels import KernelConditions
from PharmaPy.MomentInversion import invert_moments
from time import perf_counter

eps = np.finfo(float).eps

# Reaction rates are in kmol/(m**3 s), i.e. mol/(L s), and reaction enthalpies
# in J/mol, so a rate times an enthalpy needs this exact factor to give W/m**3.
MOL_PER_KMOL = 1e3  # [mol/kmol]


def growth_size_factor(sizes, growth_params):
    """Size factor of the growth rate, G(L) = G0 * (1 + beta*L)**alpha.

    Parameters
    ----------
    sizes : float or numpy.ndarray
        Crystal sizes L [um].
    growth_params : array-like
        Growth parameters of ``CrystKinetics``; entries 3 and 4 are alpha
        [-] and beta [1/um].

    Returns
    -------
    float or numpy.ndarray
        ``(1 + beta*L)**alpha`` [-], shaped like ``sizes``.
    """
    alpha, beta = np.asarray(growth_params, dtype=float)[3:5]  # [-], [1/um]
    return (1 + beta * np.asarray(sizes, dtype=float)) ** alpha

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

    # Window over which a mechanism that guards positivity should anticipate
    # depletion, in seconds. MultiPhaseVessel.compile_structure overwrites
    # this with its own positivity_horizon; the class default keeps a
    # mechanism usable on its own, outside a vessel.
    positivity_horizon = 1.0

    solver_states = ()
    output_states = ()
    owning_phase = None
    _solver_state_keys=()
    

    def __init__(self):
        self.exposed_attributes = set()
        self._timers = {}

    def _update_exposed_attributes(self):
        self.exposed_attributes.update(state.name for state in self.solver_states)

    def expose(self, *names):
        self.exposed_attributes.update(names)

    @property
    def solver_state_keys(self):
        return self._solver_state_keys
    def owns_state(self, state_key):
        """
        Return True if this mechanism owns this transported state.
        """
        return state_key in self.solver_state_keys
    def reset(self):
        """Optional."""
        pass

    def update_state(self, completed_state,**kwargs):
        t0 = perf_counter()

        # solver_states carries the template definitions, whose phaseref is
        # still None; the vessel stamps the owning phase onto the copies it
        # registers and records the resulting keys in solver_state_keys.
        # Rebuilding a key from the template here looks up
        # StateKey(name, None), which is never in completed_state, so the
        # mechanism silently kept its initial value for the whole solve.
        keys = self.solver_state_keys

        if not keys:
            keys = tuple(
                StateKey(variable.name, variable.phaseref)
                for variable in self.solver_states
            )

        for variable, key in zip(self.solver_states, keys):

            if key in completed_state:
                setattr(
                    self,
                    variable.name,
                    completed_state[key]
                )

        self._timers['update_state'] = self._timers.get('update_state',0)+perf_counter()-t0
    def add_solver_state_variables(
        self,
        collection,
        overwrite=False,
        phase_ref=None
    ):
        keys = []

        for state in self.solver_states:

            # A caller-supplied owner wins, but a state that already names
            # its own phase keeps it. Stamping phase_ref unconditionally
            # overwrote that with None for mechanisms attached to an
            # intraphase process, which registered the state under a key no
            # lookup could ever match.
            owner = phase_ref if phase_ref is not None else state.phaseref

            if owner is None:
                raise ValueError(
                    f"{type(self).__name__} defines solver states but has no "
                    "owning phase reference. Give the StateVariable a "
                    "phaseref, or attach the mechanism to a phase."
                )

            copiedState = copy.deepcopy(state)
            copiedState.phaseref = owner
            collection.add(copiedState, overwrite)
            keys.append(StateKey(state.name, owner))

        self._solver_state_keys = tuple(keys)

    def compile_solver_state_keys(self):
        self._solver_state_keys = tuple(
            StateKey(state.name, state.phaseref)
            for state in self.solver_states
        )

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
    def get_solver_state_residuals(self, **kwargs):
        """
        Residuals for any solver state this mechanism declared as algebraic.

        Mirrors get_solver_state_rates, but returns {StateKey: residual}: the
        solver drives each residual to zero rather than integrating it. A
        mechanism with no algebraic states returns {}, and the vessel then
        skips the algebraic pass entirely.
        """
        return {}
    def get_overrides(self,name):
        return None
    def apply_inlet_state(self, name, value):
        """Set an inlet state this mechanism owns, from a trajectory.

        A crystal size distribution belongs to the mechanism rather than
        to the phase, so a feed carrying one cannot be applied through
        BasePhase.updatePhase. The vessel routes those states here.
        """
        setattr(self, name, np.asarray(value, dtype=float))

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

    def get_dilution_contributions(self, net_vol_flow, completed_state):
        """Rate of change of a per-volume state as the vessel volume changes.

        Parameters
        ----------
        net_vol_flow : float
            Rate of change of the vessel volume [m**3/s].
        completed_state : dict
            Solver and derived states keyed by ``StateKey``.

        Returns
        -------
        dict
            State rates keyed by ``StateKey``, in the state's units per
            second. Empty here: only a mechanism whose state is a density per
            unit volume needs the term.
        """
        return {}

    def get_events(self,unit):
        return []

    # -----------------------------------------------------------------
    # Conversion to and from the pre-refactor phase representation.
    #
    # Old PharmaPy kept this state on the phase itself; here it lives on
    # the mechanism. A mechanism that predates the split therefore has to
    # say how to spell its state the old way, and how to read it back.
    # The base returns None for "I do not know how", which the phase turns
    # into an explicit error rather than a silently incomplete conversion --
    # a mechanism written after the split has no obligation to be
    # backwards compatible, but it must not pretend to be.
    # -----------------------------------------------------------------

    def to_legacy_state(self):
        """Constructor kwargs for the equivalent old-stack phase, or None."""
        return None

    @classmethod
    def from_legacy_phase(cls, legacy_phase, owning_phase):
        """Build this mechanism from an old phase, or return None."""
        return None

    @classmethod
    def claims_legacy_phase(cls, legacy_phase):
        """Whether this mechanism is the one that owns that phase's state."""
        return False
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
        mechanism = object.__getattribute__(self, 'mechanism')
        return getattr(mechanism, name)
    def __deepcopy__(self, memo):
        cls = type(self)
        copied = cls.__new__(cls)
        memo[id(self)] = copied

        copied.mechanism = copy.deepcopy(self.mechanism, memo)
        copied.direction = self.direction

        return copied

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
        """Species mass rates from the reactions in ``phase``.

        Parameters
        ----------
        process : IntraPhaseProcess
            Process this mechanism serves; its ``phaseref`` keys the rates.
        phase : BasePhase
            Reacting phase, with ``mole_conc`` [kmol/m**3], ``mw``
            [kg/kmol] and ``vol`` [m**3].
        time : float
            Integration time [s], unused here.
        completed_state : dict
            Solver and derived states, unused here.

        Returns
        -------
        TransferResult
            Species mass rates [kg/s], shape ``(phase.num_species,)``, and an
            ``aux`` payload with the ``phase``, the per-reaction
            ``rxn_rates`` [kmol/(m**3 s)] and the ``process``, which
            ``get_heat_generation`` reads.
        """
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
            delta_hrxn = deltah_rxn,
            # The same window the vessel's own positivity limiter uses, so
            # the two guards agree on how far ahead they are looking.
            limiter_dt=self.positivity_horizon,
        )

        species_massPerVol_rates = np.zeros(phase.num_species)
        species_massPerVol_rates[mask] = species_rates
        species_massPerVol_rates *= phase.mw
        species_mass_rates = species_massPerVol_rates* phase.vol   # kmol/(m3 s) * kg/kmol * m3 = kg/s
        state_rates = {StateKey('mass_j',process.phaseref):species_mass_rates}
        aux = {
                "phase": phase,
                "rxn_rates": reaction_rates,
                "process":process
                }
        result = TransferResult(state_rates=state_rates,aux=aux,net_mass_rate=species_mass_rates.sum())
        return result
    
    def get_heat_generation(self, aux, completed_state, time):
        """Heat released by the reactions in the owning phase.

        Parameters
        ----------
        aux : dict
            Payload from ``get_solver_state_rates``: the ``phase`` and its
            per-reaction ``rxn_rates`` [kmol/(m**3 s)], shape ``(n_rxns,)``.
        completed_state : dict
            Solver and derived states, unused here.
        time : float
            Integration time [s], unused here.

        Returns
        -------
        float
            Heat generation rate [W], positive when the reactions release
            heat (negative reaction enthalpy).
        """
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
            ))  # [J/mol]
        # rxn_rates are per extent of each reaction divided by its first
        # reactant's coefficient, so the enthalpy, per reaction as written,
        # goes on that basis too, as the legacy reactors do (#22).
        deltah_rate_basis = (
            deltah_rxn / rk.stoich_normalization
        )  # [J/mol of normalized extent]
        heat_density = (
            -(deltah_rate_basis * aux["rxn_rates"]).sum() * MOL_PER_KMOL
        )  # [W/m**3]
        q = heat_density * phase.vol  # [W]
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
        self._bind_kinetics_target(kinetics)

        
        self.kv = kv
        self.output_states=[StateVariable(name="supersat",dim=len(self.target_ind),units="-",state_type="post", compute_value=self.compute_supersat_output),
                    StateVariable(name="solubility",dim=len(self.target_ind),units="kg/m3",state_type="post",compute_value=self.compute_solubility_output),
                    StateVariable(name="mu_n",dim=4,index=[0, 1, 2, 3],units="m**n",state_type="post", compute_value=self.compute_moments_output)]
        if fraction is None:
            fraction = np.zeros(self.owning_phase.num_species)
            fraction[self.target_ind] = np.full(len(self.target_components),1/len(self.target_components))
        self.fraction = fraction

        # Set on every right-hand side evaluation; see _slurry_volume.
        self.slurry_volume = None

        # The liquid this population is suspended in, set when the vessel
        # wires its phase connections. Needed to turn an intensive population
        # into an inventory at any point, not just after a solve.
        self.liquid_phase = None

        self._timers = {}

    def _bind_kinetics_target(self, kinetics):
        """Default ``kinetics.target_idx`` to the mechanism's first target index.

        Parameters
        ----------
        kinetics : CrystKinetics or None
            Kinetics instance being attached to the mechanism.
        """
        if (kinetics is not None
                and getattr(kinetics, "target_idx", None) is None
                and getattr(self, "target_ind", None)):
            kinetics.target_idx = self.target_ind[0]

    @property
    def mechanism_kinetics(self):
        if self._mechanism_kinetics is not None:
            return self._mechanism_kinetics
        raise AttributeError("No kinetics were specified")
    
    @mechanism_kinetics.setter
    def mechanism_kinetics(self,value):
        if value is not None:
            self._bind_kinetics_target(value)
            self._mechanism_kinetics = value
            

    def getDensity(self):
        return self.density
    
    @property
    def density(self):
        if self._density is not None:
            return self._density
        return self.owning_phase.density
    
    def get_overrides(self):
        overrides = {
            "mass": self.get_mass,
            "set_mass": self.set_mass,
        }

        if self._density is not None:
            overrides["getDensity"] = self.getDensity

        return overrides
    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    # x_grid is in microns. Moments are reported in m**n, one factor of this
    # per moment order, matching SolidPhase.getMoments.
    MOMENT_UNIT_FACTOR = 1e-6

    @staticmethod
    def _per_slurry_volume(population, vol_slurry):
        """An old total (or flow) population as a density per m3 of slurry.

        Parameters
        ----------
        population : array-like
            Population of an old solid: a total in the vessel for a phase, or
            a flow for a stream, in any per-size or per-moment unit.
        vol_slurry : float or None
            Slurry volume [m**3] of the phase, or slurry volumetric flow
            [m**3/s] of the stream.

        Returns
        -------
        numpy.ndarray
            The population per m**3 of slurry.

        Raises
        ------
        ValueError
            If the population is not empty and ``vol_slurry`` is missing or
            not positive.
        """
        population = np.asarray(population, dtype=float)

        if not np.any(population):
            return population

        if vol_slurry is None or not vol_slurry > 0:
            raise ValueError(
                "Converting an old solid needs the volume, or volumetric "
                "flow, of the slurry it belongs to: its crystals are a total "
                "or a flow, and this stack stores them per m3 of slurry. "
                "Convert the whole slurry with MixedPhase.from_legacy.")

        return population / vol_slurry

    def compute_moments(
        self,
        distrib,
        x_grid,
    ):
        """
        Compute moments from a number density distribution, in m**n.

        ``x_grid`` is in microns, so the raw integral is in micron**n. The
        conversion to metres is what SolidPhase.getMoments applies, and the
        original crystallizer relies on it: its kinetics and its
        ``vol_solid = mu_n[3] * kv`` both expect m**n. Returning the raw
        micron-based integral instead silently rescaled mu_1 by 1e6, mu_2 by
        1e12 and mu_3 by 1e18.

        Use compute_second_moment / compute_third_moment where a micron-based
        integral is wanted instead.
        """

        moments = np.zeros(4)

        for n in range(4):
            moments[n] = self.integrate_over_size(distrib * x_grid**n,
                                                  x_grid)

        return moments * self.MOMENT_UNIT_FACTOR ** np.arange(4)

    def integrate_over_size(self, integrand, x_grid=None):
        """Quadrature for a moment integral.

        A scheme that carries cell averages must integrate them the same way
        it advances them, or its moments do not obey its own conservation
        law. Subclasses override this; the generic rule is the trapezoid.
        """
        return np.trapezoid(integrand, self.x_grid if x_grid is None
                            else x_grid)

    
    def liquid_in_solubility_basis(self, liquid):
        """Liquid composition converted into the kinetics' declared basis.

        A solubility correlation is just numbers; which composition basis it
        was fitted in lives only in the person who fitted it. CrystKinetics
        now states it, and this converts the liquid to match, so the
        supersaturation and the solubility are always compared like for like.

        Previously this was hard-coded to mass per volume of *pure solvent*
        while original PharmaPy feeds solubility_temp a mass concentration on
        a *solution* basis -- a silent ~5% offset on concentration that the
        exponents amplified to ~13% on growth and ~31% on nucleation.
        """
        basis = getattr(self.mechanism_kinetics, 'solubility_basis',
                        'mass_per_volume_solution')

        if basis == 'mass_per_volume_solution':
            return liquid.mass_conc

        if basis == 'mole_per_volume_solution':
            return liquid.mole_conc

        if basis == 'mass_per_mass_solution':
            return liquid.mass_frac

        if basis == 'mole_frac':
            return liquid.mole_frac

        mass_solvent = liquid.mass_j[self.solvent_ind]

        if basis == 'mass_per_mass_solvent':
            return liquid.mass_j / mass_solvent

        if basis == 'mass_per_volume_solvent':
            solvent_weights = np.zeros(liquid.num_species, dtype=bool)
            solvent_weights[self.solvent_ind] = True
            rho_solvent = liquid.getDensityPure(
                'liquid', liquid.temp,
                weights=solvent_weights)[0][self.solvent_ind]  # [kg/m**3]
            return liquid.mass_j / mass_solvent * rho_solvent

        raise ValueError(
            f"Unknown solubility basis {basis!r}. Valid bases are "
            f"{list(getattr(self.mechanism_kinetics, 'SOLUBILITY_BASES', ()))}"
        )
    def compute_supersaturation(
        self,
        liquid,
    ):
        """Evaluate concentration, supersaturation driving force, and solubility.

        Parameters
        ----------
        liquid : BasePhase
            Liquid phase containing the crystallizing species and solvent.

        Returns
        -------
        conc : ndarray
            Species concentrations in the declared solubility basis.
        supersat : float or ndarray
            Supersaturation driving force on the declared ``sup_sat_type``
            basis (``c - c_sat`` for ``'absolute'``, ``(c - c_sat) / c_sat``
            for ``'relative'`` and deprecated ``'ratio'``).
        solubility : float or ndarray
            Equilibrium solubility in the declared solubility basis.
        """
        conc = self.liquid_in_solubility_basis(liquid)
        conc_target = conc.T[self.target_ind]
        solubility = self.mechanism_kinetics.get_solubility(liquid.temp, conc)
        supersat = self.mechanism_kinetics._driving_force(conc_target, solubility)

        return conc, supersat, solubility

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

    def transfer_volume_rate(self, crystal_mass_rate, species_rates, liquid):
        """Rate at which the transfer changes the slurry volume.

        Parameters
        ----------
        crystal_mass_rate : float
            Mass rate into the crystals [kg/s], negative for dissolution.
        species_rates : ndarray
            Mass rates of the liquid species [kg/s], shape
            ``(num_species,)``, as ``compute_species_transfer`` returns them.
        liquid : BasePhase
            Liquid the crystals are suspended in.

        Returns
        -------
        float
            Rate of change of the slurry volume [m**3/s]: the crystal volume
            gained at the crystal density plus the liquid volume lost at the
            pure-species liquid densities.

        Notes
        -----
        Exact for the ideal-volume mixing rule of
        ``ThermoModule.getDensityMix``, under which the liquid volume is
        ``sum(mass_i / rho_i)``, at constant temperature. The population is
        carried per m**3 of slurry, so this volume change dilutes or
        concentrates it; see ``get_dilution_contributions``. With the crystal
        density equal to the solute's liquid density the two terms cancel,
        which is why an equal-density fixture never showed the missing term.
        """
        rates = np.asarray(species_rates, dtype=float)  # [kg/s] per species
        active = rates != 0.0
        vol_rate_solid = crystal_mass_rate / self.density  # [m**3/s]
        if not np.any(active):
            return float(vol_rate_solid)

        rho_liquid_pure, _ = liquid.getDensityPure(
            'liquid', liquid.temp, weights=active)  # [kg/m**3] per species
        vol_rate_liquid = float(np.sum(
            rates[active] / rho_liquid_pure[active]))  # [m**3/s]

        return float(vol_rate_solid + vol_rate_liquid)

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

        Crystallization runs liquid -> solid, but dissolution is wired as a
        second connection running solid -> liquid, so source and sink swap
        roles between the two. The population balance is written in terms of
        liquid and solid, not source and sink, so identify them by phase
        family rather than by connection direction. Reading them positionally
        bound the solid to the liquid slot on the dissolution connection,
        where its all-zero mass_j produced 0/0 and poisoned the whole
        right-hand side with NaN.
        """
        liquid_phase, solid_phase = self._orient_phases(source_phase,
                                                        sink_phase)

        t0 = perf_counter()
        result= self.solve_population_balance(liquid_phase,solid_phase,completed_state,time,connection)
        self._timers['pop_balance_solve_total'] = self._timers.get('pop_balance_solve_total',0)+perf_counter()-t0
        if not hasattr(self,"liquid_phase_ref"):
            self.liquid_phase_ref = self._liquid_phaseref(connection)

        result.aux.update(
            {
                "connection": connection,
                "state_rates": result.state_rates,
                "liquid_phase": liquid_phase,
                "solid_phase": solid_phase,
            }
        )
        return result

    @staticmethod
    def _orient_phases(source_phase, sink_phase):
        """Return (liquid, solid) regardless of connection direction."""
        if getattr(source_phase, 'phase_family', None) == 'solid':
            return sink_phase, source_phase

        return source_phase, sink_phase

    def _liquid_phaseref(self, connection):
        """The PhaseRef naming the liquid side of this connection."""
        if connection.source_phaseref.phase_type == 'solid':
            return connection.sink_phaseref

        return connection.source_phaseref

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

    def get_events(self, unit):
        """
        Crystallization changes regime when supersaturation changes sign:
        growth and primary nucleation above zero, dissolution below. The
        kinetics branch there, so the right-hand side is only piecewise
        smooth, and the integrator has to be told where the corner is instead
        of hunting for it with short steps.
        """

        if self._mechanism_kinetics is None:
            return []

        liquid_ref = getattr(self, "liquid_phase_ref", None)

        if liquid_ref is None:
            return []

        def supersaturation_sign(time, completed_state, unit):
            liquid = unit.Phases.get_phase_from_ref(liquid_ref)
            supersat = np.atleast_1d(self.compute_supersaturation(liquid)[1])
            return float(supersat[0])

        return [
            StateEvent(
                name=f"supersaturation_sign[{type(self).__name__}]",
                function=supersaturation_sign,
                direction=0,
                terminal=False,
                source=self,
            )
        ]

    def compute_supersat_output(
        self,
        state_var,
        time,
        completed_state,
        context,
        resolved_inlets=None,
        resolved_outlets=None,
        operating_conditions=None,
    ):
        """
        Compute supersaturation from the current replay state.

        This is the StateVariable.compute_value interface. The mechanism
        itself is the context during replay.
        """

        liquid = context.Phases.get_phase_from_ref(
            self.liquid_phase_ref
        )


        return self.compute_supersaturation(liquid)[1]

    def compute_solubility_output(
        self,
        state_var,
        time,
        completed_state,
        context,
        resolved_inlets=None,
        resolved_outlets=None,
        operating_conditions=None,
    ):
        """
        Compute solubility from the current replay state.
        """

        liquid = context.Phases.get_phase_from_ref(
            self.liquid_phase_ref
        )

        return self.compute_supersaturation(liquid)[2]

    def compute_moments_output(
        self,
        state_var,
        time,
        completed_state,
        context,
        resolved_inlets=None,
        resolved_outlets=None,
        operating_conditions=None,
    ):
        distribution = completed_state[
            StateKey(
                self.distribution_state_name,
                self.owning_phase_ref,
            )
        ]

        return self.compute_moments(
            distribution,
            self.x_grid,
        )
    
    # ---------------- basis helpers ----------------
    #
    # The population state is a number DENSITY per m3 of slurry. The slurry
    # volume therefore appears at exactly two boundaries, where an intensive
    # quantity becomes an extensive one: the mass transfer leaving the
    # population balance, and the solid inventory. Nowhere else -- in
    # particular NOT on the nucleation rate, which is already #/(s m3 slurry).
    #
    # The old `scale` argument was meant as numerical conditioning, but it
    # multiplied the state without being divided back out of the moments, so
    # it acted as a nucleation multiplier and moved a batch answer by 1e8
    # between scale=1 and scale=1e-9. It is accepted and ignored now; solver
    # tolerances do this job.

    @staticmethod
    def _reject_scale(scale):
        if scale is None or scale == 1:
            return

        warnings.warn(
            'scale= is deprecated and ignored. It was meant as numerical '
            'conditioning but leaked into the physics, acting as a '
            'nucleation multiplier. The population balance is now written so '
            'that no conditioning factor is needed.',
            DeprecationWarning,
            stacklevel=3,
        )

    def true_state(self, state):
        """Kept so callers written against the scaled state still work."""
        return np.asarray(state, dtype=float)

    def slurry_volume_from(self, liquid, true_state):
        """Slurry volume implied by the liquid volume and the solid fraction.

        Under the intensive basis kv * mu_3 is a volume FRACTION (m3 of
        crystal per m3 of slurry), not a volume, so the old
        ``vol_liquid + kv*mu_3`` no longer makes sense dimensionally:
        vol_slurry = vol_liquid / (1 - phi).

        Takes the TRUE (unscaled) population state and asks
        solid_volume_fraction for phi, so there is one convention for the
        fraction rather than one helper wanting micron-based moments and
        another metre-based ones.
        """
        phi = min(max(self.solid_volume_fraction(true_state), 0.0),
                  1.0 - 1e-12)

        vol_slurry = float(liquid.vol) / (1.0 - phi)
        self.slurry_volume = vol_slurry

        return vol_slurry

    def _slurry_volume(self, true_state=None):
        """Slurry volume for the inventory accessors.

        The inventory is read DURING a right-hand side evaluation -- the
        solid's vol setter calls set_mass -- and that can happen before
        solve_population_balance has run and cached a volume. So compute it
        from the liquid phase when one is known, and only fall back to the
        cache. Raising here instead surfaced as an opaque
        'repeated recoverable right-hand side errors' from CVode.
        """
        liquid = getattr(self, 'liquid_phase', None)

        if liquid is not None and true_state is not None:
            try:
                return self.slurry_volume_from(liquid, true_state)
            except (AttributeError, TypeError, ValueError):
                pass

        if self.slurry_volume is not None:
            return self.slurry_volume

        return 0.0

    # ---------------- flow terms ----------------
    #
    # MultiPhaseVessel calls these for every resolved transfer whose
    # vessel-side phase owns this mechanism. Neither was implemented, so
    # both fell through to Mechanism's empty dict and crystals never
    # entered through a feed nor left with the product: a continuous
    # crystallizer was a batch crystallizer with liquid flowing past it.
    #
    # Reference is old MSMPR.material_balances (Crystallizers.py), which
    # writes the same expression for both discretisations:
    #
    #     tau_inv    = input_flow / vol
    #     flow_state = tau_inv * (input_state - state)
    #
    # for a per-slurry-volume population n, which is exactly the basis
    # these mechanisms carry, so it applies unchanged:
    #
    #     dn/dt = (Q / vol) * n_in  -  (Q / vol) * n
    #
    # the first term being the inlet half and the second the outlet half.
    #
    # SIGN CONVENTION: both hooks return a positive magnitude. The vessel
    # adds inlet contributions and SUBTRACTS outlet ones
    # (MultiPhaseVessel.add_outlet_terms does '-= value', matching how it
    # treats species_flow), so returning a negative rate here would feed
    # crystals back in rather than withdrawing them.
    #
    # Both terms are exactly zero for a vessel with no connections, which
    # keeps a batch crystallizer untouched. They live on the base class so
    # a resolved distribution and a set of moments behave identically --
    # the moments path having silently had no flow terms at all.

    # Populations are carried in micron-based units, so a third moment
    # needs this to become m3.
    VOLUME_UNIT_FACTOR = 1e-18

    @property
    def flow_state_name(self):
        """The population state that travels with the stream."""
        for attr in ('distribution_state_name', 'moments_state_name'):

            name = getattr(self, attr, None)

            if name:
                return name

        return None

    def _flow_state_key(self):
        return StateKey(self.flow_state_name, self.owning_phase_ref)

    def _validate_inlet_state(self, value):
        """Hook for a subclass to reject a feed it cannot accept."""
        return value

    def solid_volume_fraction(self, value):
        """Solid volume per m3 of slurry, from a per-slurry-volume state.

        Used to split a slurry feed's volumetric flow between its phases.
        """
        raise NotImplementedError

    def _inlet_population_density(self, stream_phase):
        """Feed population per m3 of slurry.

        Published in that basis by MultiPhaseVessel.outputs, which is what
        old PharmaPy moved between units -- the old crystallizer
        multiplied by vol_slurry again on the way back into a phase.
        """
        name = self.flow_state_name

        for mechanism in getattr(stream_phase, 'mechanisms', None) or ():

            if getattr(mechanism, 'flow_state_name', None) != name:
                continue

            value = getattr(mechanism, name, None)

            if value is None:
                continue

            return self._validate_inlet_state(
                np.asarray(value, dtype=float))

        return None

    def get_inlet_contributions(self, stream_phase, vessel_phase, amount,
                                completed_state):
        """Crystal inflow carried by a slurry feed.

        Parameters
        ----------
        stream_phase : BasePhase
            Feed phase whose population mechanism publishes the feed
            population per m**3 of feed slurry.
        vessel_phase : BasePhase
            Solid phase of the vessel, which owns this mechanism.
        amount : float
            Volumetric flow of the whole feed slurry [m**3/s].
        completed_state : dict
            Solver and derived states keyed by ``StateKey``.

        Returns
        -------
        dict
            ``{flow state key: (amount / V_slurry) * n_in}``, in the
            population's units per second; empty when there is no feed, no
            feed population or no slurry volume.
        """
        if not amount:
            return {}

        density = self._inlet_population_density(stream_phase)

        if density is None:
            return {}

        # 'amount' is the WHOLE slurry flow of the feed (MultiPhaseVessel.
        # add_inlet_terms), so it is divided by the vessel's SLURRY volume,
        # not by the volume of the solid phase that owns this mechanism (the
        # crystal volume). Dividing by the latter overstated the crystal
        # inflow by 1/phi, and gave none at all into a vessel without
        # crystals yet. Intensive state, so this is the MSMPR form directly:
        #     dn/dt = (Q/V_slurry) * (n_in - n)
        key = self._flow_state_key()
        state = completed_state.get(key)

        if state is None:
            state = getattr(self, self.flow_state_name)

        vol_slurry = self._slurry_volume(
            self.true_state(np.asarray(state, dtype=float)))

        if vol_slurry <= 0:
            return {}

        return {key: (amount / vol_slurry) * density}

    def get_outlet_contributions(self, stream_phase, vessel_phase, amount,
                                 completed_state):

        if not amount:
            return {}

        vol = getattr(vessel_phase, 'vol', 0.0) or 0.0

        if vol <= 0:
            # Nothing held, so nothing to withdraw. The population is zero
            # here in any case, so the term would vanish anyway; the guard
            # is only to keep amount/vol finite.
            return {}

        key = self._flow_state_key()
        state = completed_state.get(key)

        if state is None:
            state = getattr(self, self.flow_state_name)

        return {key: (amount / vol) * np.asarray(state, dtype=float)}

    def get_dilution_contributions(self, net_vol_flow, completed_state):
        """Dilution of the population as the slurry volume changes.

        Parameters
        ----------
        net_vol_flow : float
            Rate of change of the slurry volume [m**3/s]: the volumetric
            inflow minus outflow, plus the volume the cross-phase transfers
            create or consume (``transfer_volume_rate``).
        completed_state : dict
            Solver and derived states keyed by ``StateKey``.

        Returns
        -------
        dict
            ``{flow state key: -(net_vol_flow / V_slurry) * n}``, in the
            population's units per second; empty when the volume is constant
            or unknown.

        Notes
        -----
        The state n is per m3 of slurry, so from d(n V)/dt = Q_in n_in - Q_out n

            dn/dt = (Q_in n_in - Q_out n) / V  -  n (dV/dt) / V

        The inlet and outlet hooks give the first term. This one is the
        second, with dV/dt = Q_in - Q_out + dV_transfer/dt (``net_vol_flow``).
        It vanishes in an MSMPR whose volume is held, but in a semibatch
        crystallizer fed with liquid it is what keeps the growing volume
        from multiplying the crystal mass, and in a closed batch it is what
        keeps the crystal mass equal to the liquid's loss: solute leaving the
        liquid at its liquid density and arriving as crystals at the crystal
        density changes the slurry volume, and without this term the solid
        mass reconstructed from the per-volume population drifted from the
        liquid loss by ``phi / (1 - phi) * (1 - rho_crystal / rho_solute)``
        of the transfer rate.

        The flow part, Q_in - Q_out, holds while the slurry density does not
        change, so n V is conserved exactly for a feed of the vessel's own
        liquid. A feed of another density and a temperature change are not
        counted in dV/dt.
        """
        if not net_vol_flow:
            return {}

        key = self._flow_state_key()
        state = completed_state.get(key)

        if state is None:
            state = getattr(self, self.flow_state_name)

        state = np.asarray(state, dtype=float)
        vol_slurry = self._slurry_volume(self.true_state(state))

        if vol_slurry <= 0:
            return {}

        return {key: -(net_vol_flow / vol_slurry) * state}


class OneDFVMMechanism(PopulationBalanceMechanism):
    """Finite-volume population balance on a fixed one-dimensional size grid.

    Carries the number density n [#/(um m3 slurry)] as cell averages on
    ``x_grid``. Growth and dissolution move crystals between cells through a
    Van Leer limited flux, nuclei enter the first cell through the boundary
    flux, and moments and the crystal mass sum over the cells
    (``integrate_over_size``). Agglomeration and breakage kernels add the
    terms of ``FVMKernelSources`` through ``kernel_sources``.
    """

    # Absolute integration tolerance on the number density [#/(um m3 slurry)].
    # A numerical design choice, checked on the case-study crystallizer
    # (35 geometric cells over 1-1500 um, CVode with a dense linear solver):
    # crystal count and L4,3 agree to five significant figures for values from
    # 1e2 to 1e4 at rtol 1e-6 and 1e-8, and with SciPy LSODA, while 1e6 moves
    # the count by 0.04 % and 1e8 by 0.7 %. Over that whole grid it admits at
    # most 1.5e7 crystals per m3, against the 5e10 the batch makes. The
    # solver default of 1e-6 made CVode fail at nucleation onset. Lower it for
    # populations far more dilute than this.
    DEFAULT_ABS_TOL = 1e4

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
        scale=None,
        rad=None,
        abs_tol=None,
        agglomeration=None,
        breakage=None,
        fragments=None,
    ):
        """Finite-volume population balance on a fixed size grid.

        Parameters
        ----------
        owning_phase : BasePhase
            Solid phase whose crystals the distribution describes.
        target_components : str or list of str
            Species that crystallize.
        solvent_name : str
            Name of the solvent species.
        x_grid : array-like
            Size grid [um], shape ``(num_grid,)``, strictly increasing.
            Uniform and geometric grids are both supported.
        distrib_init : numpy.ndarray
            Initial number density [#/(um m3 slurry)], shape ``(num_grid,)``.
        kinetics : CrystKinetics, optional
            Crystallization kinetics, usually set later through the vessel's
            ``CrystKinetics`` property.
        density : float, optional
            Crystal density [kg/m3]. None uses the owning phase's density.
        kv : float, optional
            Volume shape factor [-]. Defaults to 1.
        distribution_state_name : str, optional
            Name of the distribution state. Defaults to ``'distrib'``.
        scale : float, optional
            Deprecated and ignored; see ``_reject_scale``.
        rad : float, optional
            Nucleus size [um]. Defaults to the first grid node, where the
            boundary condition injects nuclei.
        abs_tol : float or array-like, optional
            Absolute integration tolerance on the number density
            [#/(um m3 slurry)], a scalar or one value per size class.
            Defaults to ``DEFAULT_ABS_TOL``.
        agglomeration : callable, optional
            Agglomeration kernel ``beta(L_i, L_j, conditions)`` [m**3/s], as
            in ``PharmaPy.CrystallizationKernels``, evaluated on the grid
            nodes; see ``FVMKernelSources``. None, the default, means no
            agglomeration.
        breakage : callable, optional
            Selection function ``a(L, conditions)`` [1/s], the frequency at
            which a crystal of size L breaks, evaluated on the grid nodes.
            None, the default, means no breakage. Given with ``fragments``.
        fragments : object, optional
            Fragment distribution whose ``grid_counts(grid_sizes)`` gives the
            fragments [-] at each node of one parent at each node, as the
            built-in distributions of ``PharmaPy.CrystallizationKernels`` do.
            Given with ``breakage``.

        Raises
        ------
        ValueError
            If the grid has fewer than two nodes or is not strictly
            increasing.
        PharmaPyTypeError
            If ``agglomeration`` or ``breakage`` is neither None nor
            callable, or ``fragments`` has no ``grid_counts`` method.
        PharmaPyValueError
            If only one of ``breakage`` and ``fragments`` is given; if a
            kernel or selection function is infinite at zero size (its
            ``needs_finite_nuclei`` is true) and ``rad`` is not positive; or
            if the fragments on this grid fail the checks of
            ``FVMKernelSources``.
        """
        super().__init__(
            owning_phase=owning_phase,
            target_components=target_components,
            solvent_name=solvent_name,
            kinetics=kinetics,
            density=density,
            kv=kv)

        self.x_grid = np.asarray(x_grid)

        if len(self.x_grid) < 2:
            raise ValueError(
                'A size grid needs at least two nodes to have a cell width.')

        if np.any(np.diff(self.x_grid) <= 0):
            raise ValueError(
                'The size grid must be strictly increasing.')

        self.x_grid_sq = self.x_grid**2
        self.x_grid_cu = self.x_grid**3
        self.dx = self._cell_widths(self.x_grid)
        self._growth_size_params = None
        self._growth_size_factor = None
        self._update_growth_size_factor()
        # Nucleus size for the nucleation MASS term. This must equal the size
        # at which the boundary condition injects nuclei, which for this
        # discretisation is the first grid node: the flux at the left face is
        # G*n(0) = B, so the population gains B*x_grid[0]**3 of volume per
        # second. Accounting for it as B*rad_zero**3 with rad_zero = 0 -- as
        # the original does -- means the crystals gain mass the liquid never
        # gave up. Measured on an empty vessel, the population balance grew
        # mu_3 at 8.5e-5 kg/s while the mass transfer reported 6e-18 kg/s.
        #
        # Overridable for a genuinely different nucleus size, but it should
        # track the grid unless you know why it should not.
        self.rad = self.x_grid[0] if rad is None else rad
        FVMKernelSources.check_arguments(agglomeration, breakage, fragments, self.rad)
        self.distribution_state_name = distribution_state_name
        assert len(distrib_init)==len(x_grid), "x_grid and distrib must be the same length"
        setattr(self,distribution_state_name,distrib_init)
        # self.x_grid is in microns and distrib is counts per micron-bin per m3
        self.solver_states = (
            StateVariable(
                name=distribution_state_name,
                phaseref=None,
                dim=len(self.x_grid),
                units="#/(micron m3 slurry)",
                state_type="diff",
                limit_negative_inventory=False,
                abs_tol=self.DEFAULT_ABS_TOL if abs_tol is None else abs_tol,
            ),
        )

        self._reject_scale(scale)
        self.scale = 1
        self._update_exposed_attributes()
        self.expose('x_grid')

        self.kernel_sources = (
            FVMKernelSources(self, agglomeration, breakage, fragments)
            if agglomeration is not None or breakage is not None else None)

    def add_solver_state_variables(self, collection, overwrite=False,
                                   phase_ref=None):
        """Register the distribution, and forget any earlier kernel registration.

        Parameters
        ----------
        collection : StateCollection
            The vessel's solver states.
        overwrite : bool, optional
            Passed on to the base class.
        phase_ref : PhaseRef, optional
            Phase owning the distribution.

        Notes
        -----
        The vessel registers phase and connection mechanisms before its
        intraphase processes, so ``kernel_sources`` is marked attached again
        only if it is still one of them.
        """
        if self.kernel_sources is not None:
            self.kernel_sources.attached = False

        super().add_solver_state_variables(collection, overwrite, phase_ref)

    # ---------------- legacy conversion ----------------

    @classmethod
    def claims_legacy_phase(cls, legacy_phase):
        """An old SolidPhase built from a distribution belongs here."""
        return getattr(legacy_phase, 'distrib', None) is not None \
            and getattr(legacy_phase, 'x_distrib', None) is not None

    def to_legacy_state(self):
        """The distribution, spelled the way an old SolidPhase holds it.

        Returns
        -------
        dict
            Keyword arguments for ``PharmaPy.Phases.SolidPhase``:
            ``x_distrib``, the size grid [um], shape ``(num_grid,)``;
            ``distrib``, the total crystal count in the vessel per micron
            [#/um], same shape; and ``kv``, the volume shape factor [-].

        Raises
        ------
        ValueError
            If the population is not empty but no slurry volume is known, so
            the density cannot be turned into a count.

        Notes
        -----
        This mechanism stores a number density per m3 of slurry
        [#/(um m3)], but the old SolidPhase holds the total count in the
        vessel [#/um]: its mass is kv * rho * mu_3 with no volume factor.
        The slurry volume [m3] converts one into the other. Copying the array
        unchanged handed the old Filter a 0.06 m3 batch's crystals as
        1/0.06, about 17 times, their mass.

        The caller builds the old phase with ``mass=0``, which makes it take
        the array as a number distribution instead of rescaling it.
        """
        density = np.asarray(getattr(self, self.distribution_state_name),
                             dtype=float)  # [#/(um m3 slurry)]
        vol_slurry = self._slurry_volume(self.true_state(density))  # [m3]

        if vol_slurry <= 0 and np.any(density != 0):
            raise ValueError(
                'Cannot express the crystal distribution as a total count '
                'for the pre-refactor SolidPhase: the slurry volume is '
                'unknown. Attach the solid to a vessel with a liquid phase.')

        return {
            'x_distrib': np.asarray(self.x_grid),  # [um]
            'distrib': density * vol_slurry,  # [#/um], total in the vessel
            'kv': self.kv,  # [-]
        }

    @classmethod
    def from_legacy_phase(cls, legacy_phase, owning_phase, vol_slurry=None,
                          **kwargs):
        """Rebuild the mechanism from an old SolidPhase or SolidStream.

        Parameters
        ----------
        legacy_phase : SolidPhase or SolidStream
            Old solid. Its ``distrib`` is the total count in the vessel
            [#/um] for a phase, or the crystal flow [#/(um s)] for a stream.
        owning_phase : SolidPhase
            Refactored solid that will own the mechanism.
        vol_slurry : float, optional
            Slurry volume [m**3] of the phase, or slurry volumetric flow
            [m**3/s] of the stream, from
            ``MixedPhase.legacy_slurry_volume``. Needed unless there are no
            crystals.
        **kwargs
            Passed to the constructor, e.g. ``target_components``.

        Returns
        -------
        OneDFVMMechanism or None
            Mechanism holding the number density [#/(um m**3 slurry)], or
            None when the old solid carries no distribution.

        Raises
        ------
        ValueError
            If there are crystals and ``vol_slurry`` is missing.
        """
        if not cls.claims_legacy_phase(legacy_phase):
            return None

        density = cls._per_slurry_volume(
            legacy_phase.distrib, vol_slurry)  # [#/(um m**3)]

        return cls(
            owning_phase=owning_phase,
            x_grid=np.asarray(legacy_phase.x_distrib),
            distrib_init=density,
            kv=getattr(legacy_phase, 'kv', 1),
            **kwargs,
        )

    def integrate_over_size(self, integrand, x_grid=None):
        """Rectangle rule, because this scheme stores CELL AVERAGES.

        ``dcsd_dt = -diff(flux)/dx`` advances cell averages, so the moment
        consistent with that conservation law is ``sum(n * x**k * dx)``, with
        dx the per-cell width -- a scalar on a uniform grid, an array on a
        geometric one, broadcasting either way. The
        trapezoid rule gives the first and last nodes half weight -- and the
        nucleation boundary condition injects every nucleus into exactly the
        first cell, so the population balance gained mu_3 at twice the rate
        the mass transfer removed solute from the liquid. Measured as a clean
        0.5000 ratio over 400 samples before this changed.

        The original carries the same mismatch: SolidPhase.getMoments is a
        trapezoid over the same cell averages.
        """
        return np.sum(integrand * self.dx)

    def compute_second_moment(
            self,
            distrib,
        ):
            """
            Compute second moment from a number density distribution.
            """
    
            return self.integrate_over_size(distrib * self.x_grid_sq)
    def compute_third_moment(
            self,
            distrib,
        ):
        """
        Compute third moment from a number density distribution.
        """
        return self.integrate_over_size(distrib * self.x_grid_cu)
    # compute_third_moment integrates over x_grid in microns, so its result is
    # in micron**3 and this converts it to m**3. The solid inventory is then
    # density * kv * mu_3, with no volume factor: the original crystallizer
    # scales nucleation by the slurry volume, which makes the distribution an
    # absolute count rather than a per-volume density
    # (Crystallizers.py: "mu_3 is total, not by volume").
    VOLUME_UNIT_FACTOR = 1e-18

    def get_mass(self):
        # mu_3 is now m3 of crystal per m3 of slurry, so the inventory needs
        # the slurry volume to become a mass.
        distribution = self.true_state(
            getattr(self,self.distribution_state_name))
        m3 = self.compute_third_moment(distribution)
        return (
            self.getDensity()
            * self.kv
            * m3
            * self.VOLUME_UNIT_FACTOR
            * self._slurry_volume(distribution)
        )
    def set_mass(self, mass):

        if mass is None:
            return

        vol_slurry = self._slurry_volume(
            self.true_state(getattr(self,self.distribution_state_name)))

        if vol_slurry <= 0:
            if mass == 0:
                return
            raise ValueError(
                'Cannot set a crystal mass without knowing the slurry volume; '
                'the distribution is a number density per m3 of slurry.')

        target_m3 = mass / (
            self.getDensity()
            * self.kv
            * self.VOLUME_UNIT_FACTOR
            * vol_slurry
        )

        self.set_third_moment(target_m3)
    def _update_growth_size_factor(self, gparams=None):
        """Return and cache the size-dependent growth multiplier on ``x_grid``.

        Parameters
        ----------
        gparams : sequence of float, optional
            Growth parameter vector ``(k_g, E_g, g[, alpha, beta])``. Defaults
            to the attached kinetics' ``params['growth']``.

        Returns
        -------
        numpy.ndarray or None
            Size factor ``(1 + beta * x_grid) ** alpha`` [-], shape
            ``x_grid.shape``, when ``gparams`` has more than three entries;
            ``None`` otherwise.
        """
        if gparams is None:
            if self._mechanism_kinetics is None:
                self._growth_size_params = None
                self._growth_size_factor = None
                return None
            gparams = self._mechanism_kinetics.params['growth']

        if len(gparams) > 3:
            alpha = float(gparams[3])  # [-]
            beta = float(gparams[4])  # [1/um]
            key = (alpha, beta)
            if (getattr(self, '_growth_size_params', None) != key
                    or getattr(self, '_growth_size_factor', None) is None):
                self._growth_size_params = key
                self._growth_size_factor = growth_size_factor(
                    self.x_grid, gparams)  # [-]
        else:
            self._growth_size_params = None
            self._growth_size_factor = None

        return self._growth_size_factor

    @PopulationBalanceMechanism.mechanism_kinetics.setter
    def mechanism_kinetics(self,value):
        """Attach crystallization kinetics and tabulate the growth size factor.

        Parameters
        ----------
        value : CrystKinetics or None
            Kinetics to attach; None leaves the current ones. A growth law
            past three entries has the size factor of
            ``growth_size_factor``, cached on ``x_grid`` by
            ``_update_growth_size_factor``.
        """
        if value is not None:
            self._bind_kinetics_target(value)
            self._mechanism_kinetics = value
            if hasattr(self, 'x_grid'):
                self._update_growth_size_factor(value.params['growth'])
    def set_third_moment(self, target_m3):
        """Rescale the distribution to a given third moment, keeping its shape.

        Parameters
        ----------
        target_m3 : float
            Third moment to reach [um**3/m3 slurry].

        Raises
        ------
        ValueError
            If a nonzero target is asked of a distribution whose third moment
            is not positive, so there is no shape to rescale.

        Notes
        -----
        A zero target empties the population. An integrator holds a
        just-nucleating population only to its absolute tolerance, so the
        distribution can carry small negative densities whose third moment is
        negative; asking such a population for no crystals must give none,
        not an error.
        """
        distribution = getattr(self,self.distribution_state_name)

        if target_m3 == 0:
            setattr(self, self.distribution_state_name,
                    np.zeros_like(distribution))  # [#/(um m3 slurry)]
            return

        # target_m3 is a TRUE moment, so compare against the true state; the
        # ratio is then applied to the conditioned state, which is what is
        # actually stored.
        current_m3 = self.compute_third_moment(self.true_state(distribution))
        if current_m3 <= 0:
            raise ValueError("Cannot scale a distribution with zero third moment.")

        factor = target_m3 / current_m3

        setattr(self,self.distribution_state_name,distribution * factor)

    @staticmethod
    def _cell_widths(x_grid):
        """Width of the finite volume cell around each grid node.

        A uniform grid gives one scalar. A geometric grid -- which is the
        natural choice here, because it resolves the small sizes where
        nucleation happens -- gives an array, with cell faces at the
        geometric mean of adjacent nodes and the outer two extrapolated by
        the grid ratio.

        This mirrors Phases.getDistribution and the Slurry setup in
        MixedPhases, which is where the original computes exactly the same
        thing; the refactor had reduced it to ``x_grid[1] - x_grid[0]``, a
        single scalar applied to every cell. On geomspace(1, 1500, 35) that
        is 0.24 micron standing in for cells up to 323 micron wide -- a 1347x
        misweighting of the largest cells, in both the flux divergence and
        the moment integrals.

        Parameters
        ----------
        x_grid : numpy.ndarray
            Size grid [um], shape ``(num_grid,)``, strictly increasing.

        Returns
        -------
        float or numpy.ndarray
            Cell widths [um]: the node spacing on a uniform grid, otherwise
            the differences of ``_cell_faces``, shape ``(num_grid,)``.
        """
        widths = np.diff(x_grid)

        if np.allclose(widths, widths[0], rtol=OneDFVMMechanism.UNIFORM_GRID_RTOL):
            return widths[0]

        return np.diff(OneDFVMMechanism._cell_faces(x_grid))

    # A grid is uniform when its node spacings agree to this relative
    # tolerance. A design choice: far above the rounding of a linspace grid,
    # about 1e-15, and far below any spacing change a grid is built with.
    UNIFORM_GRID_RTOL = 1e-8  # [-]

    @staticmethod
    def _cell_faces(x_grid):
        """Faces of the finite volume cells around the grid nodes.

        Parameters
        ----------
        x_grid : numpy.ndarray
            Size grid [um], shape ``(num_grid,)``, strictly increasing.

        Returns
        -------
        numpy.ndarray
            Cell faces [um], shape ``(num_grid + 1,)``. On a uniform grid
            they lie midway between nodes, and half a spacing beyond the
            outer two; on any other grid, at the geometric mean of adjacent
            nodes, with the outer two extrapolated by the ratio of the first
            two nodes, as ``_cell_widths`` describes.
        """
        widths = np.diff(x_grid)

        if np.allclose(widths, widths[0], rtol=OneDFVMMechanism.UNIFORM_GRID_RTOL):
            half_width = widths[0] / 2  # [um]
            return np.append(x_grid - half_width, x_grid[-1] + half_width)

        ratio = x_grid[1] / x_grid[0]  # [-]
        interior = np.sqrt(x_grid[1:] * x_grid[:-1])  # [um]

        return np.concatenate(([interior[0] / ratio], interior, [interior[-1] * ratio]))

    # ---------------- flow terms ----------------
    #
    # The inlet and outlet halves live on PopulationBalanceMechanism so
    # this and the moments form behave identically. Only the two pieces
    # that genuinely differ are here.

    def _validate_inlet_state(self, value):
        """A feed distribution must be on this vessel's own size grid."""
        if value.shape != self.x_grid.shape:
            raise ValueError(
                'An inlet crystal size distribution has %d bins but this '
                'vessel discretises size into %d. Feeding a distribution '
                'between different grids would need interpolation, which '
                'is not implemented -- give both units the same x_grid.'
                % (value.size, self.x_grid.size))

        return value

    def solid_volume_fraction(self, value):
        """Solid volume per m3 of slurry, from a number density."""
        return float(self.kv
                     * self.compute_third_moment(value)
                     * self.VOLUME_UNIT_FACTOR)
    def solve_population_balance(
        self,
        liquid:BasePhase,
        solid:BasePhase,
        completed_state:dict[StateKey],
        time:float,
        connection:PhaseConnection
    ) -> TransferResult:
        """Distribution and liquid rates of growth, nucleation and dissolution.

        Parameters
        ----------
        liquid : BasePhase
            Liquid the crystals grow from.
        solid : BasePhase
            The solid, unused: the distribution is read from
            ``completed_state``.
        completed_state : dict
            Solver and derived states keyed by ``StateKey``.
        time : float
            Time [s], unused.
        connection : PhaseConnection
            Connection being solved; names the liquid.

        Returns
        -------
        TransferResult
            d(n)/dt [#/(um m3 slurry s)], shape ``(num_grid,)``, the liquid's
            species rates [kg/s], the crystallized mass rate [kg/s] and the
            slurry volume rate [m**3/s].

        Raises
        ------
        PharmaPySpecificationError
            If this population balance has an agglomeration or breakage
            kernel but its ``kernel_sources`` is not an intra-phase process
            of the vessel.
        """
        if self.kernel_sources is not None:
            self.kernel_sources.require_attached()

        t0=perf_counter()
        # The distribution lives on the phase that owns this mechanism (the
        # solid), not on the connection's sink. For crystallization the two
        # coincide, but the dissolution connection runs solid -> liquid, so
        # keying off the sink looked for 'distrib' on the liquid and raised
        # KeyError. Any crystallizer given dissolution kinetics hit this.
        statekey =  StateKey(self.distribution_state_name,self.owning_phase_ref)
        csd = completed_state[statekey]
        self._timers['statekey_construct'] = self._timers.get('statekey_construct',0)+perf_counter()-t0
        t0 = perf_counter()
        # Physics reads the true population, never the conditioned state.
        csd_true = self.true_state(csd)
        moms = self.compute_moments(csd_true,self.x_grid)
        self._timers['pop_balance_compute_moments'] = self._timers.get('pop_balance_compute_moments',0)+perf_counter()-t0

        # mu_2 is in m**2, matching the original's getMoments basis. The
        # size-dependent branch below needs the micron-based integral instead.
        mu2 = moms[2] #total surface area
        t0 = perf_counter()
        conc, supersat, solubility = self.compute_supersaturation(liquid)
        self._timers['pop_balance_compute_supersat'] = self._timers.get('pop_balance_compute_supersat',0)+perf_counter()-t0

        t0 = perf_counter()
        nucl, growth, dissol = (
            self.mechanism_kinetics.get_kinetics(
                conc,
                liquid.temp,
                self.kv,
                moms,
            )
        )
        self._timers['pop_balance_compute_kinetics'] = self._timers.get('pop_balance_compute_kinetics',0)+perf_counter()-t0
        # nucl is #/(s m3 slurry) and the distribution is a number density on
        # that same basis, so there is NO volume factor here. Multiplying by
        # vol_slurry made the distribution an absolute count, which in turn
        # made the moments fed back into get_kinetics extensive -- so
        # secondary nucleation, which depends on magma density, scaled with
        # vessel size. The volume returns once, on the mass transfer below.
        vol_slurry = self.slurry_volume_from(liquid, csd_true)
        nucl_intensive = nucl
        nucl = nucl_intensive
        impurity_factor = self.mechanism_kinetics.alpha_fn(conc) #TODO check if con is the right units
        growth *= impurity_factor
        t0 = perf_counter()
        gparams = self.mechanism_kinetics.params["growth"]

        boundary = nucl / np.maximum(growth, eps)
        f_aug = np.concatenate(([boundary, boundary],csd,[csd[-1]]))
        self._timers['pop_balance_compute_boundary'] = self._timers.get('pop_balance_compute_boundary',0)+perf_counter()-t0
        t0 = perf_counter()
        # Flux source terms
        f_diff = np.diff(f_aug)
        if growth > 0:
            theta = (f_diff[:-1]/ (f_diff[1:] + eps * 10))
        else:
            theta = (f_diff[1:]/ (f_diff[:-1] + eps * 10))

        #Van-Leer limiter
        limiter = np.zeros_like(f_diff)
        limiter[:-1] = ((np.abs(theta) + theta)/ (1 + np.abs(theta)))
        self._timers['pop_balance_compute_limiter'] = self._timers.get('pop_balance_compute_limiter',0)+perf_counter()-t0

        # t0 = perf_counter()
        # Constant growth
        if len(gparams) == 3:

            growth_term = growth* (f_aug[1:-1]+ 0.5 * f_diff[1:] * limiter[:-1])
            dissol_term = dissol* (f_aug[2:]- 0.5 * f_diff[1:] * limiter[1:])
            growth_int = growth * self.compute_second_moment(csd_true)

        # Size-dependent growth
        else:

            t0 = perf_counter()
            growth_dep = growth * self._update_growth_size_factor(gparams)
            self._timers['pop_balance_compute_growth_dep'] = self._timers.get('pop_balance_compute_growth_dep',0)+perf_counter()-t0
            t0 = perf_counter()
            growth_pad = np.append(growth_dep,growth_dep[-1],)
            self._timers['pop_balance_compute_growth_dissol_pad'] = self._timers.get('pop_balance_compute_growth_dissol_pad',0)+perf_counter()-t0
            t0 = perf_counter()
            growth_term = growth_pad* (f_aug[1:-1]+ 0.5 * f_diff[1:] * limiter[:-1])
            dissol_term = dissol* (f_aug[2:]- 0.5 * f_diff[1:] * limiter[1:])
            self._timers['pop_balance_compute_growth_dissol_term'] = self._timers.get('pop_balance_compute_growth_dissol_term',0)+perf_counter()-t0
            t0 = perf_counter()
            growth_int = self.integrate_over_size(
                growth_dep * csd_true * self.x_grid_sq)
            self._timers['pop_balance_compute_growth_dissol_int'] = self._timers.get('pop_balance_compute_growth_dissol_int',0)+perf_counter()-t0

        t0 = perf_counter()
        # d(mu_3)/dt = 3*G*mu_2 + B*rad**3, so the 3 belongs to the growth
        # and dissolution terms ONLY -- it used to distribute over the
        # nucleation term as well. Every term is micron-based here, so one
        # VOLUME_UNIT_FACTOR converts the lot (the original applied 1e-6 to a
        # metre-based mu_2 and to a micron-based rad**3 together, leaving the
        # nucleation term wrong by 1e12).
        #
        # This is a rate per m3 of slurry; vol_slurry makes it extensive.
        dissol_int = dissol * self.compute_second_moment(csd_true)

        mass_transfer = (
            self.density * self.kv
            * (3 * (growth_int + dissol_int) + nucl_intensive * self.rad**3)
            * self.VOLUME_UNIT_FACTOR
            * vol_slurry
        )
        self._timers['pop_balance_compute_mass_transfer'] = self._timers.get('pop_balance_compute_mass_transfer',0)+perf_counter()-t0
        # self._timers['pop_balance_handle_growth'] = self._timers.get('pop_balance_handle_growth',0)+perf_counter()-t0
        t0 = perf_counter()
        flux = growth_term + dissol_term
        self._timers['pop_balance_flux_sum'] = self._timers.get('pop_balance_flux_sum',0)+perf_counter()-t0
        t0 = perf_counter()
        dcsd_dt = -np.diff(flux) / self.dx
        self._timers['pop_balance_compute_dcsd_dt'] = self._timers.get('pop_balance_compute_dcsd_dt',0)+perf_counter()-t0
        t0 = perf_counter()
        aux = {
            "supersaturation": supersat,
            "solubility": solubility,
            "moments": moms,
            "growth": growth,
            "dissolution": dissol,
            "nucleation": nucl,
            "flux": flux,
        }
        self._timers['pop_balance_compute_flux'] = self._timers.get('pop_balance_compute_flux',0)+perf_counter()-t0
        t0 = perf_counter()
        species_rates_out = self.compute_species_transfer(mass_transfer,liquid)
        self._timers['pop_balance_compute_species_transfer'] = self._timers.get('pop_balance_compute_species_transfer',0)+perf_counter()-t0
        # The transfer changes the slurry volume, which dilutes the
        # per-volume population; the vessel adds it to the flow terms.
        vol_rate = self.transfer_volume_rate(
            mass_transfer, species_rates_out, liquid)  # [m**3/s]
        t0 = perf_counter()
        state_rates = {StateKey(self.solver_states[0].name,self.owning_phase): dcsd_dt} #if phaseref is a phase instead of a PhaseRef, the vessel will determine the phaseref
        # Species leave or enter the LIQUID, which is the connection's source
        # for crystallization but its sink for dissolution. Keying off the
        # source unconditionally would return dissolved mass to the solid.
        state_rates.update({StateKey('mass_j',self._liquid_phaseref(connection)):species_rates_out})
        result = TransferResult(state_rates=state_rates,aux=aux,net_mass_rate=mass_transfer,
                                net_vol_rate=vol_rate)
        self._timers['pop_balance_format'] = self._timers.get('pop_balance_format',0)+perf_counter()-t0
        return result

class MomentsPopulationBalance(PopulationBalanceMechanism):
    """Method of moments, ported from Crystallizers.py::method_of_moments.

    Tracks mu_0..mu_{n-1} rather than a resolved distribution. Moments are
    carried in micron**n as the original does; kinetics are handed them in
    m**n per m**3 of suspension, and the mass transfer carries the
    (1e-6)**3 conversion.

    Caveat inherited from the original: the moment form was only ever
    exercised on the volume basis in the batch crystallizer. It was still
    part-built for the reactive crystallizer, which was the first move to a
    mass basis, so the mass-basis path here is unproven.
    """

    # Secondary nucleation scaling as (kv*mu_3)**s_2 with s_2 below this
    # switches on and off at mu_3 = 0, and with nuclei born at zero size the
    # crystal count then follows the integrator's step pattern (#529). This is
    # what the case-study crystallizer showed over CVode step caps of 0.5 to
    # 30 s, not a general bound: up to 17-25 % for s_2 <= 0.01, at most
    # 0.001 % for s_2 >= 0.1 (29 Oct 2026).
    SWITCH_LIKE_SECONDARY_ORDER = 0.1  # [-]

    def __init__(
        self,
        owning_phase,
        target_components,
        solvent_name,
        moments_init,
        kinetics=None,
        density=None,
        kv=1,
        rad=0.0,
        scale=None,
        moments_state_name='mu_n',
        abs_tol=None,
    ):
        """Population balance carried as its leading moments.

        Parameters
        ----------
        owning_phase : BasePhase
            Solid phase whose crystals the moments describe.
        target_components : str or list of str
            Species that crystallize.
        solvent_name : str
            Name of the solvent species.
        moments_init : array-like
            Initial moments mu_0..mu_{n-1} [um**k/m3 slurry], shape
            ``(num_mom,)``.
        kinetics : CrystKinetics, optional
            Crystallization kinetics, usually set later through the vessel's
            ``CrystKinetics`` property. Growth must be size independent.
        density : float, optional
            Crystal density [kg/m3]. None uses the owning phase's density.
        kv : float, optional
            Volume shape factor [-]. Defaults to 1.
        rad : float, optional
            Nucleus size [um]. Defaults to 0.
        scale : float, optional
            Deprecated and ignored; see ``_reject_scale``.
        moments_state_name : str, optional
            Name of the moments state. Defaults to ``'mu_n'``.
        abs_tol : float or array-like, optional
            Absolute integration tolerance per moment [um**k/m3 slurry], a
            scalar or one value per moment. None, the default, keeps the
            integrator's own tolerance.

        Raises
        ------
        PharmaPyNonImplementedError
            If ``kinetics`` has size-dependent growth.

        Warns
        -----
        RuntimeWarning
            If ``kinetics`` can make the crystal count follow the integrator's
            step pattern; see ``_warn_if_onset_follows_steps``.

        Notes
        -----
        Unlike ``OneDFVMMechanism``, the moments declare no tolerance of
        their own. Secondary nucleation switches on once the third moment is
        positive, so the onset must be resolved near zero. A tolerance scaled
        to the final moments let CVode hold the third moment just below zero
        on a leaner case-study feed, and that batch never crystallized.
        """
        # The base class stores kinetics without going through the setter.
        if kinetics is not None:
            self._reject_size_dependent_growth(kinetics)

        super().__init__(
            owning_phase=owning_phase,
            target_components=target_components,
            solvent_name=solvent_name,
            kinetics=kinetics,
            density=density,
            kv=kv,
        )

        moments_init = np.atleast_1d(np.asarray(moments_init, dtype=float))

        self.moments_state_name = moments_state_name
        self.num_mom = len(moments_init)

        # rad is the nucleus size. It lived on the old unit operation as
        # rad_zero (default 0), never on the phase, so it must be supplied.
        self.rad = rad

        if kinetics is not None:
            self._warn_if_onset_follows_steps(kinetics)

        self._reject_scale(scale)
        self.scale = 1

        setattr(self, moments_state_name, moments_init)

        # PopulationBalanceMechanism registers a derived 'mu_n' output,
        # computed by integrating a resolved distribution. Here the
        # moments ARE the solver state, so that output both duplicates it
        # and collides with it by name -- the vessel then has one name
        # meaning two different things.
        self.output_states = [state for state in self.output_states
                              if state.name != moments_state_name]

        self.solver_states = (
            StateVariable(
                name=moments_state_name,
                dim=self.num_mom,
                index=list(range(self.num_mom)),
                units='micron**n',
                state_type='diff',
                limit_negative_inventory=False,
                abs_tol=abs_tol,
            ),
        )

        self._update_exposed_attributes()

    @PopulationBalanceMechanism.mechanism_kinetics.setter
    def mechanism_kinetics(self, value):
        """Attach crystallization kinetics the moment closure can carry.

        Parameters
        ----------
        value : CrystKinetics or None
            Kinetics to attach; None leaves the current ones.

        Raises
        ------
        PharmaPyNonImplementedError
            If ``value`` has size-dependent growth.

        Warns
        -----
        RuntimeWarning
            If ``value`` can make the crystal count follow the integrator's
            step pattern; see ``_warn_if_onset_follows_steps``.
        """
        if value is not None:
            self._reject_size_dependent_growth(value)
            self._bind_kinetics_target(value)
            self._mechanism_kinetics = value
            self._warn_if_onset_follows_steps(value)

    @staticmethod
    def _reject_size_dependent_growth(kinetics):
        """Refuse kinetics whose growth rate depends on crystal size.

        Parameters
        ----------
        kinetics : CrystKinetics
            Kinetics to check through ``has_size_dependent_growth``. Growth
            entries past the first three are the size factor
            ``(1 + beta*L)**alpha`` that ``OneDFVMMechanism`` applies:
            alpha [-], then beta [1/um]. An object without the predicate is
            accepted.

        Raises
        ------
        PharmaPyNonImplementedError
            If those entries are all nonzero, so the size factor is not
            identically one.

        Notes
        -----
        ``dmu_k/dt = k*G*mu_(k-1) + B*rad**k`` closes only for a growth rate G
        that does not depend on size; a size-dependent G brings in the
        integral of L**(k-1)*G(L)*n(L), which the moments do not carry. The
        moment form used to ignore alpha and beta without a warning. With
        either of them zero the factor is exactly one, so that growth is
        still size independent.
        """
        is_size_dependent = getattr(kinetics, 'has_size_dependent_growth',
                                    None)

        if is_size_dependent is not None and is_size_dependent():
            size_factor = np.asarray(kinetics.params['growth'],
                                     dtype=float)[3:5]  # alpha [-], beta [1/um]
            raise PharmaPyNonImplementedError(
                'MomentsPopulationBalance cannot carry size-dependent growth '
                '(alpha, beta = %s): the moment equations close only for a '
                'size-independent growth rate. Use QMOMPopulationBalance or '
                'OneDFVMMechanism for size-dependent growth, or give growth '
                'only its first three parameters.' % size_factor.tolist())

    def _warn_if_onset_follows_steps(self, kinetics):
        """Warn when the crystal count can follow the integrator's steps.

        Parameters
        ----------
        kinetics : CrystKinetics
            Kinetics being attached.

        Warns
        -----
        RuntimeWarning
            If nuclei are born at zero size (``rad`` = 0) and built-in
            secondary nucleation is active with an order ``s_2`` [-] on the
            crystal moment below ``SWITCH_LIKE_SECONDARY_ORDER``.

        Notes
        -----
        The factor ``(kv*max(0, mu_k))**s_2`` is then close to one for any
        positive moment and zero otherwise. Just after onset the moment grows
        from zero, far below the integrator's absolute tolerance, so
        integration error decides when secondary nucleation is on (#529).
        Nuclei of finite size made CVode fail outright at larger step caps
        instead, which is not silent, so they do not warn. Secondary
        nucleation counts as inactive with a zero prefactor or the default
        parameters, as in ``CrystKinetics.get_kinetics``.
        """
        if self.rad != 0 or 'nucl_sec' in getattr(kinetics, 'custom_mechanisms', {}):
            return

        params = np.atleast_1d(np.asarray(
            getattr(kinetics, 'params', {}).get('nucl_sec', ()), dtype=float))
        if params.size < 4:
            return

        reformulated = getattr(kinetics, 'reformulate_kin', False)
        default = np.zeros_like(params)
        if reformulated:
            default[0] = np.log(eps)  # log of a zero prefactor, as CrystKinetics stores it
        if np.allclose(params, default) or (not reformulated and params[0] == 0):
            return

        order = params[3]  # s_2 [-]
        if order < self.SWITCH_LIKE_SECONDARY_ORDER:
            warnings.warn(
                'Secondary nucleation scales as (kv*mu_%d)**s_2 with s_2 = %g, '
                'below %g, so it switches on and off at mu_%d = 0. With nuclei '
                'born at zero size (rad = 0), the crystal count of this moment '
                'form can then follow the integrator\'s step pattern at '
                'nucleation onset, by up to 25 %% on the case study (PharmaPy '
                'issue #529). Prefer OneDFVMMechanism, or check the result '
                'against a solve with another step cap.'
                % (getattr(kinetics, 'mu_sec_nucl', 3), order,
                   self.SWITCH_LIKE_SECONDARY_ORDER,
                   getattr(kinetics, 'mu_sec_nucl', 3)),
                RuntimeWarning,
                stacklevel=3,
            )

    # ---------------- legacy conversion ----------------

    @classmethod
    def claims_legacy_phase(cls, legacy_phase):
        """An old SolidPhase built from moments rather than a distribution."""
        return (getattr(legacy_phase, 'moments', None) is not None
                and getattr(legacy_phase, 'distrib', None) is None)

    def to_legacy_state(self):
        """The moments, spelled the way an old SolidPhase holds them.

        Returns
        -------
        dict
            Keyword arguments for ``PharmaPy.Phases.SolidPhase``:
            ``moments``, the total moments in the vessel [m**k], shape
            ``(num_mom,)``; ``num_mom``; and ``kv``, the volume shape
            factor [-].

        Raises
        ------
        ValueError
            If the population is not empty but no slurry volume is known, so
            the moments cannot be turned into totals.

        Notes
        -----
        This mechanism stores moments per m3 of slurry in microns
        [um**k/m3], but the old SolidPhase holds total moments in metres
        [m**k]: its mass is kv * rho * mu_3 with no volume factor. Copying
        the array unchanged misstated mu_3 by a factor of 1e18 / V_slurry.
        """
        moments = self.true_state(np.asarray(
            getattr(self, self.moments_state_name),
            dtype=float))  # [um**k/m3 slurry]
        vol_slurry = self._slurry_volume(moments)  # [m3]

        if vol_slurry <= 0 and np.any(moments != 0):
            raise ValueError(
                'Cannot express the crystal moments as totals for the '
                'pre-refactor SolidPhase: the slurry volume is unknown. '
                'Attach the solid to a vessel with a liquid phase.')

        orders = np.arange(len(moments))

        return {
            'moments': (moments * vol_slurry
                        * self.MOMENT_UNIT_FACTOR ** orders),  # [m**k]
            'num_mom': self.num_mom,
            'kv': self.kv,  # [-]
        }

    @classmethod
    def from_legacy_phase(cls, legacy_phase, owning_phase, vol_slurry=None,
                          **kwargs):
        """Rebuild the mechanism from an old solid that carries only moments.

        Parameters
        ----------
        legacy_phase : SolidPhase or SolidStream
            Old solid. Its ``moments`` are totals in the vessel [m**k] for a
            phase, or flows [m**k/s] for a stream.
        owning_phase : SolidPhase
            Refactored solid that will own the mechanism.
        vol_slurry : float, optional
            Slurry volume [m**3] of the phase, or slurry volumetric flow
            [m**3/s] of the stream, from
            ``MixedPhase.legacy_slurry_volume``. Needed unless there are no
            crystals.
        **kwargs
            Passed to the constructor, e.g. ``target_components``.

        Returns
        -------
        MomentsPopulationBalance or None
            Mechanism holding the moments [um**k/m**3 slurry], or None when
            the old solid does not carry moments alone.

        Raises
        ------
        ValueError
            If there are crystals and ``vol_slurry`` is missing.
        """
        if not cls.claims_legacy_phase(legacy_phase):
            return None

        moments = np.asarray(legacy_phase.moments, dtype=float)  # [m**k]
        orders = np.arange(len(moments))
        moments_um = moments / cls.MOMENT_UNIT_FACTOR ** orders  # [um**k]

        return cls(
            owning_phase=owning_phase,
            moments_init=cls._per_slurry_volume(
                moments_um, vol_slurry),  # [um**k/m**3]
            kv=getattr(legacy_phase, 'kv', 1),
            **kwargs,
        )

    # ---------------- population balance ----------------

    def get_mass(self):
        """Solid mass implied by the third moment."""
        moments = np.asarray(getattr(self, self.moments_state_name))

        if len(moments) < 4:
            return 0.0

        return (self.density * self.kv * self.true_state(moments)[3] * 1e-18
                * self._slurry_volume(self.true_state(moments)))

    def set_mass(self, value):
        """Rescale the moments to a crystal mass, keeping their ratios.

        Parameters
        ----------
        value : float
            Crystal mass in the vessel [kg].

        Raises
        ------
        ValueError
            If fewer than four moments are carried, if the slurry volume is
            unknown for a nonzero mass, or if a nonzero mass is asked of
            moments whose third moment is not positive.

        Notes
        -----
        With a slurry volume known, a zero mass empties the population, for
        the reason given in ``OneDFVMMechanism.set_third_moment``. Without
        one, a zero mass leaves the moments unchanged.
        """
        moments = np.asarray(getattr(self, self.moments_state_name),
                             dtype=float)

        if len(moments) < 4:
            raise ValueError(
                'Setting mass needs at least four moments (mu_0..mu_3)')

        vol_slurry = self._slurry_volume(self.true_state(moments))

        if vol_slurry <= 0:
            if value == 0:
                return
            raise ValueError(
                'Cannot set a crystal mass without knowing the slurry '
                'volume; the moments are per m3 of slurry.')

        target_mu3 = value / (self.density * self.kv * 1e-18 * vol_slurry)
        current = self.true_state(moments)[3]

        if target_mu3 == 0:
            setattr(self, self.moments_state_name,
                    np.zeros_like(moments))  # [um**k/m3 slurry]
            return

        if current <= 0:
            raise ValueError(
                'Cannot scale moments with a zero third moment.')

        setattr(self, self.moments_state_name,
                moments * (target_mu3 / current))  # ratio is scale-invariant

    # ---------------- flow terms ----------------
    #
    # Inherited from PopulationBalanceMechanism. Old
    # MSMPR.material_balances applies the identical expression to the
    # moment set that it applies to a resolved distribution:
    #
    #     input_distrib = u_inputs['Inlet']['mu_n'] * (1e6)**arange(n)
    #     flow_distrib  = tau_inv * (input_distrib - distrib)
    #
    # so only the volume fraction differs, mu_3 being available directly
    # rather than needing to be integrated.

    def _validate_inlet_state(self, value):
        """Accept a feed only if it carries this vessel's number of moments.

        Parameters
        ----------
        value : numpy.ndarray
            Feed moments [um**k/m3 of feed slurry].

        Returns
        -------
        numpy.ndarray
            ``value``, unchanged.

        Raises
        ------
        PharmaPyValueError
            If ``value`` does not have shape ``(num_mom,)``.
        """
        if value.shape != (self.num_mom,):
            raise PharmaPyValueError(
                'A feed carrying %d moments cannot enter this vessel, which '
                'carries %d; the missing moments cannot be recovered. Give '
                'both units the same number of moments.'
                % (value.size, self.num_mom))

        return value

    def solid_volume_fraction(self, value):
        """Solid volume per m3 of slurry, from the third moment."""
        moments = np.atleast_1d(np.asarray(value, dtype=float))

        if len(moments) < 4:
            return 0.0

        return float(self.kv * moments[3] * self.VOLUME_UNIT_FACTOR)

    def _growth_integrals(self, moments, orders):
        """Integrals of L**k * G(L)/G0 * n(L) over size, for the growth terms.

        Parameters
        ----------
        moments : numpy.ndarray
            Moments mu_0..mu_{n-1} [um**k/m3 slurry], shape ``(num_mom,)``.
        orders : numpy.ndarray of int
            Orders k wanted, each below ``num_mom``.

        Returns
        -------
        numpy.ndarray
            The integrals [um**k/m3 slurry], shaped like ``orders``. Growth
            is size independent here, G(L) = G0, so they are the moments
            mu_k themselves.
        """
        return moments[orders]

    def solve_population_balance(
        self,
        liquid,
        solid,
        completed_state,
        time,
        connection,
    ):
        """Moment rates and the solute transfer of crystallization.

        Parameters
        ----------
        liquid : BasePhase
            Liquid the crystals are suspended in.
        solid : BasePhase
            Solid phase owning the moments.
        completed_state : dict
            Solver and derived states keyed by ``StateKey``.
        time : float
            Time [s].
        connection : PhaseConnection
            Connection the transfer runs along.

        Returns
        -------
        TransferResult
            d(mu_k)/dt [um**k/(m3 slurry s)] of the kinetics alone, the
            liquid species rates [kg/s], the crystal mass rate [kg/s] and
            the slurry volume rate [m**3/s] of the transfer.

        Raises
        ------
        PharmaPyNonImplementedError
            If ``set_params`` has made the attached growth law size
            dependent since it was attached.

        Notes
        -----
        dmu_0/dt = B and, for k >= 1,
        dmu_k/dt = k (G0 I_(k-1) + D mu_(k-1)) + B rad**k, with
        I_k = integral of L**k G(L)/G0 n(L) dL from ``_growth_integrals``
        and the dissolution rate D size independent. The crystal mass rate
        is rho kv (3 (G0 I_2 + D mu_2) + B rad**3) per m**3 of slurry.
        """
        self._reject_size_dependent_growth(self.mechanism_kinetics)

        statekey = StateKey(self.moments_state_name, self.owning_phase_ref)
        mu = np.asarray(completed_state[statekey], dtype=float)

        conc, supersat, solubility = self.compute_supersaturation(liquid)

        # Intensive basis: mu is already per m3 of slurry, so the kinetics
        # need only the micron -> metre conversion, with no volume division.
        # The volume returns on the mass transfer below.
        mu_true = self.true_state(mu)
        mu_susp = mu_true * (1e-6) ** np.arange(len(mu_true))
        vol_slurry = self.slurry_volume_from(liquid, mu_true)

        nucl, growth, dissol = self.mechanism_kinetics.get_kinetics(
            conc, liquid.temp, self.kv, mu_susp)

        nucl_intensive = nucl
        nucl = nucl_intensive
        growth = growth * self.mechanism_kinetics.alpha_fn(conc)

        ind_mom = np.arange(1, len(mu))

        # Orders 0..n-2 for the moment rates and 2 for the mass transfer.
        orders = np.arange(max(len(mu_true) - 1, 3))
        growth_integrals = self._growth_integrals(mu_true, orders)  # [um**k/m3]

        dmu_zero_dt = np.atleast_1d(nucl)
        dmu_1on_dt = (ind_mom * (growth * growth_integrals[:len(mu) - 1]
                                 + dissol * mu_true[:-1])
                      + nucl * self.rad ** ind_mom)

        dmu_dt = np.concatenate((dmu_zero_dt, dmu_1on_dt))

        # Per m3 of slurry, then made extensive. mu_2 is micron**2 and
        # growth micron/s, so (1e-6)**3 converts to m3/s. The true moments
        # and the unscaled nucleation rate, not the conditioned state.
        mass_transfer = float(np.atleast_1d(
            self.density * self.kv
            * (3 * (growth * growth_integrals[2] + dissol * mu_true[2])
               + nucl_intensive * self.rad ** 3)
        )[0]) * (1e-6) ** 3 * vol_slurry

        species_rates_out = self.compute_species_transfer(mass_transfer,
                                                          liquid)

        # The transfer changes the slurry volume, which dilutes the
        # per-volume moments; the vessel adds it to the flow terms.
        vol_rate = self.transfer_volume_rate(
            mass_transfer, species_rates_out, liquid)  # [m**3/s]

        aux = {
            'supersat': supersat,
            'solubility': solubility,
            'growth': growth,
            'dissolution': dissol,
            'nucleation': nucl,
        }

        state_rates = {
            StateKey(self.moments_state_name, self.owning_phase): dmu_dt,
            StateKey('mass_j', self._liquid_phaseref(connection)):
                species_rates_out,
        }

        return TransferResult(state_rates=state_rates, aux=aux,
                              net_mass_rate=mass_transfer,
                              net_vol_rate=vol_rate)


class QMOMPopulationBalance(MomentsPopulationBalance):
    """Quadrature method of moments (QMOM), for size-dependent growth.

    Carries the same state as ``MomentsPopulationBalance``, the moments
    mu_0..mu_{2N-1} per m**3 of slurry [um**k/m3], with the same feed,
    outlet, dilution and legacy handling. Where the moment form needs a
    size-independent growth rate, QMOM evaluates the growth integrals with
    an N-node Gauss quadrature recovered from the moments at every
    right-hand side (McGraw, 1997, Aerosol Sci. Technol. 27:255-265):

        integral of L**k G(L) n(L) dL = G0 sum_i w_i L_i**k (1 + beta L_i)**alpha

    with the size factor of ``OneDFVMMechanism`` (``growth_size_factor``).
    The rule integrates polynomials of degree below 2N exactly, so linear
    growth (alpha = 1) is exact; other alpha are approximated, the more
    closely the more nodes. Growth without a size factor uses the moments
    themselves and reduces exactly to ``MomentsPopulationBalance``.

    The moment form's assumptions remain: dissolution is size independent
    with no disappearance of crystals at L = 0, and nuclei are born at size
    ``rad``. Agglomeration and breakage, when given their kernels, conserve
    crystal volume; their terms come from ``QMOMKernelSources``.

    A solve also reports ``quadrature_nodes`` [um] and ``quadrature_weights``
    [#/m3 slurry], shape ``(N,)`` per time, NaN where the moments hold fewer
    than N nodes, and the mean sizes ``L32`` = mu_3/mu_2 and, with N >= 3,
    ``L43`` = mu_4/mu_3 [um], NaN without crystals.
    """

    # Crystals from a pre-refactor unit would arrive as a plain moment form or
    # a distribution, without this vessel's model; Connection refuses them.
    refuses_legacy_crystals = True

    def __init__(
        self,
        owning_phase,
        target_components,
        solvent_name,
        moments_init,
        kinetics=None,
        density=None,
        kv=1,
        rad=0.0,
        scale=None,
        moments_state_name='mu_n',
        abs_tol=None,
        agglomeration=None,
        breakage=None,
        fragments=None,
    ):
        """Population balance carried as 2N moments, closed by quadrature.

        Parameters
        ----------
        owning_phase : BasePhase
            Solid phase whose crystals the moments describe.
        target_components : str or list of str
            Species that crystallize.
        solvent_name : str
            Name of the solvent species.
        moments_init : array-like
            Initial moments mu_0..mu_{2N-1} [um**k/m3 slurry], shape
            ``(2N,)`` with N >= 2 quadrature nodes.
        kinetics : CrystKinetics, optional
            Crystallization kinetics, usually set later through the vessel's
            ``CrystKinetics`` property. Growth may be size dependent.
        density : float, optional
            Crystal density [kg/m3]. None uses the owning phase's density.
        kv : float, optional
            Volume shape factor [-]. Defaults to 1.
        rad : float, optional
            Nucleus size [um]. Defaults to 0.
        scale : float, optional
            Deprecated and ignored; see ``_reject_scale``.
        moments_state_name : str, optional
            Name of the moments state. Defaults to ``'mu_n'``.
        abs_tol : float or array-like, optional
            Absolute integration tolerance per moment [um**k/m3 slurry]. None,
            the default, keeps the integrator's own tolerance.
        agglomeration : callable, optional
            Agglomeration kernel ``beta(L_i, L_j, conditions)`` [m**3/s], as
            in ``PharmaPy.CrystallizationKernels``. None, the default, means
            no agglomeration.
        breakage : callable, optional
            Selection function ``a(L, conditions)`` [1/s], the frequency at
            which a crystal of size L breaks. None, the default, means no
            breakage. Given with ``fragments``.
        fragments : object, optional
            Fragment distribution whose ``moments(parent_sizes, orders)``
            gives the moments [um**k] of the fragments of one parent, as in
            ``PharmaPy.CrystallizationKernels``. Given with ``breakage``.

        Raises
        ------
        PharmaPyValueError
            If ``moments_init`` does not hold an even number of moments, at
            least four: N nodes take 2N moments, and the crystal mass needs
            mu_3. If only one of ``breakage`` and ``fragments`` is given. If
            a kernel or selection function is infinite at zero size (its
            ``needs_finite_nuclei`` is true) and ``rad`` is not positive.
        PharmaPyTypeError
            If ``agglomeration`` or ``breakage`` is neither None nor
            callable, or ``fragments`` has no ``moments`` method.

        Warns
        -----
        RuntimeWarning
            As ``MomentsPopulationBalance``, if ``kinetics`` can make the
            crystal count follow the integrator's step pattern (#529).
        """
        num_mom = np.size(moments_init)

        if num_mom < 4 or num_mom % 2:
            raise PharmaPyValueError(
                'QMOMPopulationBalance needs an even number of moments, at '
                'least four (N >= 2 nodes take mu_0..mu_{2N-1}), got %d.'
                % num_mom)

        QMOMKernelSources.check_arguments(agglomeration, breakage, fragments, rad)

        super().__init__(
            owning_phase=owning_phase,
            target_components=target_components,
            solvent_name=solvent_name,
            moments_init=moments_init,
            kinetics=kinetics,
            density=density,
            kv=kv,
            rad=rad,
            scale=scale,
            moments_state_name=moments_state_name,
            abs_tol=abs_tol,
        )

        self.kernel_sources = (
            QMOMKernelSources(self, agglomeration, breakage, fragments)
            if agglomeration is not None or breakage is not None else None)

        num_nodes = self.num_mom // 2
        node_index = list(range(num_nodes))
        self.output_states = list(self.output_states) + [
            StateVariable(name='quadrature_nodes', dim=num_nodes, index=node_index,
                          units='um', state_type='post',
                          compute_value=self.compute_nodes_output),
            StateVariable(name='quadrature_weights', dim=num_nodes, index=node_index,
                          units='#/m**3', state_type='post',
                          compute_value=self.compute_weights_output),
            StateVariable(name='L32', dim=1, units='um', state_type='post',
                          compute_value=self.compute_l32_output),
        ]
        # mu_4 is carried from three nodes on.
        if self.num_mom > 4:
            self.output_states.append(
                StateVariable(name='L43', dim=1, units='um', state_type='post',
                              compute_value=self.compute_l43_output))

    def add_solver_state_variables(self, collection, overwrite=False,
                                   phase_ref=None):
        """Register the moments, and forget any earlier kernel registration.

        Parameters
        ----------
        collection : StateCollection
            The vessel's solver states.
        overwrite : bool, optional
            Passed on to the base class.
        phase_ref : PhaseRef, optional
            Phase owning the moments.

        Notes
        -----
        The vessel registers phase and connection mechanisms before its
        intraphase processes, so ``kernel_sources`` is marked attached again
        only if it is still one of them.
        """
        if self.kernel_sources is not None:
            self.kernel_sources.attached = False

        super().add_solver_state_variables(collection, overwrite, phase_ref)

    @staticmethod
    def _reject_size_dependent_growth(kinetics):
        """Accept any growth law: the quadrature carries the size factor.

        Parameters
        ----------
        kinetics : CrystKinetics
            Kinetics being attached.
        """

    def solve_population_balance(self, liquid, solid, completed_state, time,
                                 connection):
        """Moment rates of the transfer, once the kernel terms are wired.

        Parameters
        ----------
        liquid, solid, completed_state, time, connection
            As ``MomentsPopulationBalance.solve_population_balance``.

        Returns
        -------
        TransferResult
            As ``MomentsPopulationBalance.solve_population_balance``.

        Raises
        ------
        PharmaPySpecificationError
            If this population balance has a kernel but its
            ``kernel_sources`` is not an intra-phase process of the vessel,
            which would leave agglomeration or breakage out without a word.
        """
        if self.kernel_sources is not None:
            self.kernel_sources.require_attached()

        return super().solve_population_balance(liquid, solid, completed_state,
                                                time, connection)

    def _growth_integrals(self, moments, orders):
        """Integrals of L**k * G(L)/G0 * n(L) over size, by quadrature.

        Parameters
        ----------
        moments : numpy.ndarray
            Moments mu_0..mu_{2N-1} [um**k/m3 slurry], shape ``(2N,)``.
        orders : numpy.ndarray of int
            Orders k wanted.

        Returns
        -------
        numpy.ndarray
            ``sum_i w_i L_i**k (1 + beta L_i)**alpha`` [um**k/m3 slurry],
            shaped like ``orders``, over the N-node rule of
            ``invert_moments``. Without a size factor, the moments mu_k.
        """
        kinetics = self.mechanism_kinetics
        is_size_dependent = getattr(kinetics, 'has_size_dependent_growth',
                                    None)

        if is_size_dependent is None or not is_size_dependent():
            return super()._growth_integrals(moments, orders)

        rule = invert_moments(moments, self.num_mom // 2)  # [um], [#/m3]
        size_factor = growth_size_factor(rule.nodes,
                                         kinetics.params['growth'])  # [-]

        return np.sum(rule.weights[:, None] * size_factor[:, None]
                      * rule.nodes[:, None] ** orders, axis=0)

    def _inlet_population_density(self, stream_phase):
        """Feed moments per m3 of slurry, refusing any other feed population.

        Parameters
        ----------
        stream_phase : BasePhase
            Feed phase, whose population mechanism publishes the feed.

        Returns
        -------
        numpy.ndarray or None
            Feed moments [um**k/m3 of feed slurry], shape ``(2N,)``; None for
            a feed without crystals.

        Raises
        ------
        PharmaPySpecificationError
            If the feed carries its crystals as another state, such as a
            resolved distribution, which the base class would skip, losing
            those crystals without a word.
        """
        density = super()._inlet_population_density(stream_phase)

        if density is None:
            for mechanism in getattr(stream_phase, 'mechanisms', None) or ():
                state_name = getattr(mechanism, 'flow_state_name', None)

                if state_name not in (None, self.flow_state_name):
                    raise PharmaPySpecificationError(
                        'A feed carrying its crystals as %r cannot enter this '
                        'QMOMPopulationBalance, which takes moments as %r. '
                        'Use a QMOM or moment-form unit upstream.'
                        % (state_name, self.flow_state_name))

        return density

    def _moments_from(self, completed_state):
        """Read the moments from a replay state.

        Parameters
        ----------
        completed_state : dict
            Solver and derived states keyed by ``StateKey``.

        Returns
        -------
        numpy.ndarray
            Moments [um**k/m3 slurry], shape ``(2N,)``.
        """
        key = StateKey(self.moments_state_name, self.owning_phase_ref)
        return np.asarray(completed_state[key], dtype=float)

    def _padded_rule(self, completed_state):
        """Quadrature of a replay state, padded to the output arrays' shape.

        Parameters
        ----------
        completed_state : dict
            Solver and derived states keyed by ``StateKey``.

        Returns
        -------
        nodes, weights : numpy.ndarray
            Sizes [um] and weights [#/m3 slurry], shape ``(N,)``, ascending,
            with NaN where the moments hold fewer than N nodes.
        """
        num_nodes = self.num_mom // 2
        rule = invert_moments(self._moments_from(completed_state), num_nodes)
        nodes = np.full(num_nodes, np.nan)
        weights = np.full(num_nodes, np.nan)
        nodes[:len(rule.nodes)] = rule.nodes
        weights[:len(rule.weights)] = rule.weights
        return nodes, weights

    @staticmethod
    def _mean_size(numerator, denominator):
        """Moment-ratio mean size, undefined without crystals.

        Parameters
        ----------
        numerator : float
            Moment of order k+1 [um**(k+1)/m3 slurry].
        denominator : float
            Moment of order k [um**k/m3 slurry].

        Returns
        -------
        numpy.ndarray
            Their ratio [um], shape ``(1,)``; NaN unless ``denominator`` is
            positive.
        """
        return np.array([numerator / denominator if denominator > 0 else np.nan])

    def compute_nodes_output(self, state_var, time, completed_state, context,
                             **kwargs):
        """Quadrature sizes, on the ``StateVariable.compute_value`` interface.

        Parameters
        ----------
        state_var, time, context, **kwargs
            Replay arguments, unused.
        completed_state : dict
            Solver and derived states keyed by ``StateKey``.

        Returns
        -------
        numpy.ndarray
            Sizes [um], shape ``(N,)``; see ``_padded_rule``.
        """
        return self._padded_rule(completed_state)[0]

    def compute_weights_output(self, state_var, time, completed_state, context,
                               **kwargs):
        """Quadrature weights, on the ``StateVariable.compute_value`` interface.

        Parameters
        ----------
        state_var, time, context, **kwargs
            Replay arguments, unused.
        completed_state : dict
            Solver and derived states keyed by ``StateKey``.

        Returns
        -------
        numpy.ndarray
            Weights [#/m3 slurry], shape ``(N,)``; see ``_padded_rule``.
        """
        return self._padded_rule(completed_state)[1]

    def compute_l32_output(self, state_var, time, completed_state, context,
                           **kwargs):
        """Mean size mu_3/mu_2, on the ``StateVariable.compute_value`` interface.

        Parameters
        ----------
        state_var, time, context, **kwargs
            Replay arguments, unused.
        completed_state : dict
            Solver and derived states keyed by ``StateKey``.

        Returns
        -------
        numpy.ndarray
            L3,2 [um], shape ``(1,)``; see ``_mean_size``.
        """
        moments = self._moments_from(completed_state)
        return self._mean_size(moments[3], moments[2])

    def compute_l43_output(self, state_var, time, completed_state, context,
                           **kwargs):
        """Mean size mu_4/mu_3, on the ``StateVariable.compute_value`` interface.

        Parameters
        ----------
        state_var, time, context, **kwargs
            Replay arguments, unused.
        completed_state : dict
            Solver and derived states keyed by ``StateKey``.

        Returns
        -------
        numpy.ndarray
            L4,3 [um], shape ``(1,)``; see ``_mean_size``.
        """
        moments = self._moments_from(completed_state)
        return self._mean_size(moments[4], moments[3])


def _agglomeration_moment_rates(nodes, weights, kernel_values, num_mom):
    """Moment rates of volume-conserving agglomeration over a quadrature.

    Parameters
    ----------
    nodes : numpy.ndarray
        Quadrature sizes L_i [um], shape ``(n,)``.
    weights : numpy.ndarray
        Quadrature weights w_i [#/m3 slurry], shape ``(n,)``.
    kernel_values : numpy.ndarray
        beta(L_i, L_j) [m**3/s], shape ``(n, n)``.
    num_mom : int
        Number of moments.

    Returns
    -------
    numpy.ndarray
        d(mu_k)/dt [um**k/(m3 slurry s)], shape ``(num_mom,)``:
        1/2 sum_ij w_i w_j beta_ij ((L_i**3 + L_j**3)**(k/3) - L_i**k - L_j**k).

    Notes
    -----
    Each pair of crystals that agglomerates leaves one crystal of the summed
    volume, so mu_3 is conserved and mu_0 falls by one per event; the half
    counts each unordered pair once.
    """
    orders = np.arange(num_mom)
    summed_cubes = nodes[:, None] ** 3 + nodes[None, :] ** 3  # [um**3]
    pair_rates = 0.5 * weights[:, None] * weights[None, :] * kernel_values  # [#/(m3 s)]
    born = summed_cubes[..., None] ** (orders / 3)  # [um**k]
    lost = nodes[:, None, None] ** orders + nodes[None, :, None] ** orders  # [um**k]

    return np.einsum('ij,ijk->k', pair_rates, born - lost)


def _breakage_moment_rates(nodes, weights, selection_rates, fragment_moments):
    """Moment rates of breakage over a quadrature.

    Parameters
    ----------
    nodes : numpy.ndarray
        Quadrature sizes L_i [um], shape ``(n,)``.
    weights : numpy.ndarray
        Quadrature weights w_i [#/m3 slurry], shape ``(n,)``.
    selection_rates : numpy.ndarray
        Breakage frequencies a(L_i) [1/s], shape ``(n,)``.
    fragment_moments : numpy.ndarray
        Moments b_k(L_i) [um**k] of the fragments of one parent of each
        node, shape ``(n, num_mom)``.

    Returns
    -------
    numpy.ndarray
        d(mu_k)/dt [um**k/(m3 slurry s)], shape ``(num_mom,)``:
        sum_i w_i a(L_i) (b_k(L_i) - L_i**k), each parent replaced by its
        fragments.
    """
    orders = np.arange(fragment_moments.shape[1])
    parents = nodes[:, None] ** orders  # [um**k]

    return np.sum((weights * selection_rates)[:, None] * (fragment_moments - parents),
                  axis=0)


class KernelSources(Mechanism):
    """Agglomeration and breakage terms of a population balance, as an intra-phase process.

    Agglomeration and breakage move crystals within the solid whatever the
    liquid does, so they cannot ride on the liquid-solid transfer, which the
    crystallizer runs once per direction and drops against the sign of the
    mass rate. The crystallizer registers a population balance's
    ``kernel_sources`` as an ``IntraPhaseProcess`` of the solid instead, and
    the vessel adds its rates once per right-hand side. It declares no state
    of its own.

    This base holds what every form shares: the argument checks, the
    attachment record, the kernel conditions and the checked kernel and
    selection values. Each subclass turns them into rates of its own state.

    Parameters
    ----------
    population_balance : PopulationBalanceMechanism
        Population balance whose state the terms act on.
    agglomeration : callable, optional
        Kernel ``beta(L_i, L_j, conditions)`` [m**3/s].
    breakage : callable, optional
        Selection function ``a(L, conditions)`` [1/s], with ``fragments``.
    fragments : object, optional
        Fragment distribution with the method named by ``FRAGMENT_METHOD``.
    """

    # The fragment distribution's method a form needs, and its call as the
    # refusal spells it; set by each subclass.
    FRAGMENT_METHOD = None
    FRAGMENT_CALL = None

    # Fragments must hold their parent's volume (Marchisio et al., 2003, J.
    # Colloid Interface Sci. 258:322-334). A design tolerance: far above the
    # rounding of a size's cube, about 1e-16, yet small enough to catch a
    # distribution that loses or makes crystal volume.
    FRAGMENT_VOLUME_RTOL = 1e-9  # [-]

    def __init__(self, population_balance, agglomeration=None, breakage=None,
                 fragments=None):
        super().__init__()
        self.population_balance = population_balance
        self.agglomeration = agglomeration
        self.breakage = breakage
        self.fragments = fragments
        self.attached = False

    @classmethod
    def check_arguments(cls, agglomeration, breakage, fragments, rad):
        """Refuse kernel arguments the population balance cannot use.

        Parameters
        ----------
        agglomeration : callable or None
            Agglomeration kernel.
        breakage : callable or None
            Selection function.
        fragments : object or None
            Fragment distribution.
        rad : float
            Nucleus size of the population balance [um].

        Raises
        ------
        PharmaPyTypeError
            If ``agglomeration`` or ``breakage`` is neither None nor
            callable, or ``fragments`` lacks ``FRAGMENT_METHOD``.
        PharmaPyValueError
            If only one of ``breakage`` and ``fragments`` is given, or a rate
            law infinite at zero size (``needs_finite_nuclei``) meets a
            nucleus size that is not positive.
        """
        if agglomeration is not None and not callable(agglomeration):
            raise PharmaPyTypeError(
                'agglomeration must be a kernel callable '
                'beta(L_i, L_j, conditions), got %r.' % (agglomeration,))

        if (breakage is None) != (fragments is None):
            raise PharmaPyValueError(
                'Breakage needs both a selection function (breakage) and a '
                'fragment distribution (fragments); got only one of them.')

        if breakage is not None and not callable(breakage):
            raise PharmaPyTypeError(
                'breakage must be a selection function a(L, conditions), '
                'got %r.' % (breakage,))

        if fragments is not None and not callable(
                getattr(fragments, cls.FRAGMENT_METHOD, None)):
            raise PharmaPyTypeError(
                'fragments must have a %s method, got %r.'
                % (cls.FRAGMENT_CALL, fragments))

        for rate_law in (agglomeration, breakage):
            if getattr(rate_law, 'needs_finite_nuclei', False) and not rad > 0:
                raise PharmaPyValueError(
                    '%s is infinite at zero size, where nuclei born at rad = %r '
                    'sit. Give the population balance a nucleus size rad > 0 [um].'
                    % (type(rate_law).__name__, rad))

    def add_solver_state_variables(self, collection, overwrite=False,
                                   phase_ref=None):
        """Record that a vessel registered this process; adds no state.

        Parameters
        ----------
        collection : StateCollection
            The vessel's solver states.
        overwrite : bool, optional
            Passed on to the base class.
        phase_ref : PhaseRef, optional
            Phase the process runs in.
        """
        super().add_solver_state_variables(collection, overwrite, phase_ref)
        self.attached = True

    def require_attached(self):
        """Refuse to solve while the vessel does not run these terms.

        Raises
        ------
        PharmaPySpecificationError
            If no vessel registered this process since the population
            balance last registered its states, which would leave
            agglomeration or breakage out without a word.
        """
        if not self.attached:
            raise PharmaPySpecificationError(
                'This %s has agglomeration or breakage kernels, but its '
                'kernel_sources is not among the vessel\'s intraphase_processes. '
                'Set CrystKinetics through the crystallizer, or add '
                'IntraPhaseProcess(PhaseRef("solid", 0), mechanism.kernel_sources) '
                'yourself.' % type(self.population_balance).__name__)

    def _conditions(self, moments_si):
        """Kernel conditions from the live liquid and the kinetics.

        Parameters
        ----------
        moments_si : numpy.ndarray
            Moments mu_0..mu_k [m**k/m3 slurry], as the population balance
            hands them to ``CrystKinetics.get_kinetics``.

        Returns
        -------
        KernelConditions
            Temperature [K], supersaturation, growth rate G0 [um/s] and
            shape factor [-], as the transfer evaluates them.

        Raises
        ------
        PharmaPyValueError
            If the population balance has no liquid or no kinetics.
        """
        balance = self.population_balance
        liquid = getattr(balance, 'liquid_phase', None)
        kinetics = balance.mechanism_kinetics

        if liquid is None or kinetics is None:
            raise PharmaPyValueError(
                'Agglomeration and breakage kernels need the crystallizer\'s '
                'liquid and CrystKinetics; set CrystKinetics on the vessel first.')

        conc, supersat, _ = balance.compute_supersaturation(liquid)
        _, growth, _ = kinetics.get_kinetics(conc, liquid.temp, balance.kv, moments_si)
        growth = growth * kinetics.alpha_fn(conc)  # [um/s]

        # One crystallizing species: its single value, whether the kinetics
        # hand back a scalar or a one-element array.
        return KernelConditions(
            temp=float(np.ravel(liquid.temp)[0]),  # [K]
            supersaturation=float(np.ravel(supersat)[0]),
            growth_rate=float(np.ravel(growth)[0]),  # [um/s]
            shape_factor=float(balance.kv),  # [-]
        )

    def _kernel_values(self, sizes, conditions):
        """Evaluate and check the agglomeration kernel on every pair of nodes.

        Parameters
        ----------
        sizes : numpy.ndarray
            Node sizes the kernel sees [um], shape ``(n,)``.
        conditions : KernelConditions
            Passed to the kernel.

        Returns
        -------
        numpy.ndarray
            beta(L_i, L_j) [m**3/s], shape ``(n, n)``.

        Raises
        ------
        PharmaPyValueError
            If the rates have another shape, or are negative or not finite.
        """
        num_nodes = len(sizes)
        values = np.asarray(self.agglomeration(sizes[:, None], sizes[None, :],
                                               conditions), dtype=float)

        if values.shape != (num_nodes, num_nodes):
            raise PharmaPyValueError(
                'The agglomeration kernel must broadcast over its two size '
                'arrays to shape %s, got %s.' % ((num_nodes, num_nodes), values.shape))

        if not np.all(np.isfinite(values)) or np.any(values < 0):
            raise PharmaPyValueError(
                'The agglomeration kernel must return finite, non-negative '
                'rates [m**3/s], got %s.' % values.tolist())

        return values

    def _selection_rates(self, sizes, conditions):
        """Evaluate and check the breakage frequency of every node.

        Parameters
        ----------
        sizes : numpy.ndarray
            Node sizes the selection function sees [um], shape ``(n,)``.
        conditions : KernelConditions
            Passed to the selection function.

        Returns
        -------
        numpy.ndarray
            a(L_i) [1/s], shape ``(n,)``.

        Raises
        ------
        PharmaPyValueError
            If the frequencies have another shape, or are negative or not
            finite.
        """
        values = np.asarray(self.breakage(sizes, conditions), dtype=float)

        if values.shape != sizes.shape:
            raise PharmaPyValueError(
                'The breakage selection function must return one frequency per '
                'size, shape %s, got %s.' % (sizes.shape, values.shape))

        if not np.all(np.isfinite(values)) or np.any(values < 0):
            raise PharmaPyValueError(
                'The breakage selection function must return finite, '
                'non-negative frequencies [1/s], got %s.' % values.tolist())

        return values


class QMOMKernelSources(KernelSources):
    """Agglomeration and breakage terms of a QMOM population balance.

    Evaluates the kernels on the quadrature recovered from the moments and
    returns the moment rates of both terms (Marchisio et al., 2003, J.
    Colloid Interface Sci. 258:322-334).

    Parameters
    ----------
    population_balance : QMOMPopulationBalance
        Population balance whose moments the terms act on.
    agglomeration, breakage : callable, optional
        As ``KernelSources``.
    fragments : object, optional
        Fragment distribution with ``moments(parent_sizes, orders)``.
    """

    FRAGMENT_METHOD = 'moments'
    FRAGMENT_CALL = 'moments(parent_sizes, orders)'

    def get_solver_state_rates(self, process, phase, time, completed_state):
        """Moment rates of agglomeration and breakage at the current state.

        Parameters
        ----------
        process : IntraPhaseProcess
            Process this mechanism serves; its ``phaseref`` names the solid.
        phase : BasePhase
            The solid, unused: the moments are read from ``completed_state``.
        time : float
            Time [s], unused.
        completed_state : dict
            Solver and derived states keyed by ``StateKey``.

        Returns
        -------
        TransferResult
            d(mu_k)/dt [um**k/(m3 slurry s)], shape ``(2N,)``, keyed by the
            moments state, with no mass or volume transfer.

        Raises
        ------
        PharmaPyValueError
            If a kernel, selection function or fragment distribution returns
            values that fail the checks of ``_kernel_values``,
            ``_selection_rates`` or ``_fragment_moments``, or if the
            population balance has no liquid or kinetics to evaluate the
            kernel conditions with.
        """
        balance = self.population_balance
        key = StateKey(balance.moments_state_name, process.phaseref)
        moments = np.asarray(completed_state[key], dtype=float)  # [um**k/m3]
        rule = invert_moments(moments, balance.num_mom // 2)  # [um], [#/m3]

        rates = np.zeros(balance.num_mom)
        if len(rule.nodes):
            conditions = self._conditions(
                moments * balance.MOMENT_UNIT_FACTOR ** np.arange(len(moments)))  # [m**k/m3]
            # No crystal is smaller than a nucleus, so the rate laws see each
            # node at no less than rad. A node below it comes from rounding or
            # integration error near zero size, and would otherwise drive a
            # law that grows as 1/L, such as the Brownian kernel, without
            # bound. The fragments belong to the node itself.
            sizes = np.maximum(rule.nodes, balance.rad)  # [um]

            if self.agglomeration is not None:
                rates += _agglomeration_moment_rates(
                    rule.nodes, rule.weights, self._kernel_values(sizes, conditions),
                    balance.num_mom)

            if self.breakage is not None:
                rates += _breakage_moment_rates(
                    rule.nodes, rule.weights, self._selection_rates(sizes, conditions),
                    self._fragment_moments(rule.nodes, balance.num_mom))

        return TransferResult(state_rates={key: rates}, aux={'process': process},
                              net_mass_rate=0.0)

    def _fragment_moments(self, nodes, num_mom):
        """Evaluate and check the fragment moments of every node.

        Parameters
        ----------
        nodes : numpy.ndarray
            Parent sizes [um], shape ``(n,)``.
        num_mom : int
            Number of moments.

        Returns
        -------
        numpy.ndarray
            b_k(L_i) [um**k], shape ``(n, num_mom)``.

        Raises
        ------
        PharmaPyValueError
            If the moments have another shape or are not finite, or if the
            fragments do not hold their parent's volume, b_3 = L**3, to
            ``FRAGMENT_VOLUME_RTOL``.
        """
        values = np.asarray(self.fragments.moments(nodes, np.arange(num_mom)),
                            dtype=float)

        if values.shape != (len(nodes), num_mom):
            raise PharmaPyValueError(
                'The fragment distribution must return moments of shape %s '
                '(parents, orders), got %s.' % ((len(nodes), num_mom), values.shape))

        if not np.all(np.isfinite(values)):
            raise PharmaPyValueError(
                'The fragment distribution must return finite moments, got %s.'
                % values.tolist())

        parent_volume = nodes ** 3  # [um**3]
        if np.any(np.abs(values[:, 3] - parent_volume)
                  > self.FRAGMENT_VOLUME_RTOL * parent_volume):
            raise PharmaPyValueError(
                'The fragments must hold their parent\'s volume: b_3 = %s for '
                'parents of L**3 = %s.' % (values[:, 3].tolist(), parent_volume.tolist()))

        return values


class FVMKernelSources(KernelSources):
    """Agglomeration and breakage terms of a finite-volume population balance.

    Works on the crystal count of each cell, N_i = n_i dx_i [#/m3 slurry],
    every crystal of cell i having its node's volume v_i = L_i**3 [um**3];
    the shape factor scales every volume alike, so it cancels from the
    split. Each pair of cells i <= j agglomerates at beta_ij N_i N_j, halved
    for i = j. Following the cell average technique (Kumar et al., 2006,
    Chem. Eng. Sci. 61:3327-3342), the agglomerates are collected in the cell
    whose faces bracket their volume v_i + v_j, then split between that
    cell's node and its neighbour on the side of their mean volume, keeping
    both their number and their volume. The population balance's mu_3,
    sum N_i L_i**3, is therefore conserved to rounding.

    Agglomerates whose mean volume lies above the last node stay in the last
    cell with their volume: the cell gains their volume over v_last in
    crystals, so the crystal mass stays exact while their size is capped at
    the last node. After a solve, ``review_solution`` warns when that cell
    held more than ``TOP_CELL_VOLUME_SHARE`` of the crystal volume.

    Breakage follows the fixed pivot (Kumar and Ramkrishna, 1996, Chem. Eng.
    Sci. 51:1311-1332): the cells break at a(L_i) N_i, and the fragment
    distribution's ``grid_counts`` places their fragments on the nodes once
    for the grid, keeping their number and volume, so d(N)/dt =
    F (a N) - a N. Fragments below the first node keep their volume there.

    The kernel and selection function see each node at no less than the
    nucleus size ``rad``, as in ``QMOMKernelSources``.

    Parameters
    ----------
    population_balance : OneDFVMMechanism
        Population balance whose distribution the terms act on.
    agglomeration : callable, optional
        Kernel ``beta(L_i, L_j, conditions)`` [m**3/s].
    breakage : callable, optional
        Selection function ``a(L, conditions)`` [1/s], with ``fragments``.
    fragments : object, optional
        Fragment distribution with ``grid_counts(grid_sizes)``.

    Raises
    ------
    PharmaPyValueError
        If the fragments on the grid fail the checks of
        ``_checked_fragment_counts``.
    """

    FRAGMENT_METHOD = 'grid_counts'
    FRAGMENT_CALL = 'grid_counts(grid_sizes)'

    # Share of the crystal volume the last cell may hold before a solve
    # warns, a design choice (10 Nov 2026). A grid that reaches past the
    # population leaves that cell all but empty (below 1e-100 in the Scott
    # solves of test_fvm_kernels), while agglomerates kept there are capped
    # at the last node's size; at 0.1 % of the volume the grid should be
    # extended well before such crystals shape the distribution.
    TOP_CELL_VOLUME_SHARE = 1e-3  # [-]

    def __init__(self, population_balance, agglomeration=None, breakage=None,
                 fragments=None):
        super().__init__(population_balance, agglomeration, breakage, fragments)

        volumes = population_balance.x_grid ** 3  # [um**3]
        face_volumes = population_balance._cell_faces(population_balance.x_grid) ** 3  # [um**3]
        num_grid = len(volumes)

        # Every unordered pair of cells once, its agglomerate's volume, and
        # the cell whose faces bracket it. The agglomerate holds at least the
        # larger parent's volume, so the cell is never below that parent's;
        # one past the last face is kept in the last cell.
        self._left, self._right = np.triu_indices(num_grid)
        self._merged_volumes = volumes[self._left] + volumes[self._right]  # [um**3]
        self._pair_weights = np.where(self._left == self._right, 0.5, 1.0)  # [-]
        self._birth_cells = np.minimum(
            np.searchsorted(face_volumes, self._merged_volumes, side='right') - 1,
            num_grid - 1)
        self._volumes = volumes
        self._fragment_counts = (self._checked_fragment_counts()
                                 if fragments is not None else None)  # [-]

    def _checked_fragment_counts(self):
        """Place the fragments on the grid once, and check them.

        Returns
        -------
        numpy.ndarray
            F [-], shape ``(num_grid, num_grid)``: entry [i, k] is the
            fragments at node i of one parent at node k.

        Raises
        ------
        PharmaPyValueError
            If the counts have another shape or are negative or not finite,
            or if any parent's fragments do not hold its volume,
            sum_i F[i, k] v_i = v_k, to ``FRAGMENT_VOLUME_RTOL``.
        """
        grid = self.population_balance.x_grid  # [um]
        num_grid = len(grid)
        counts = np.asarray(self.fragments.grid_counts(grid), dtype=float)

        if counts.shape != (num_grid, num_grid):
            raise PharmaPyValueError(
                'The fragment distribution must return grid counts of shape %s '
                '(nodes, parents), got %s.' % ((num_grid, num_grid), counts.shape))

        if not np.all(np.isfinite(counts)) or np.any(counts < 0):
            raise PharmaPyValueError(
                'The fragment distribution must return finite, non-negative '
                'grid counts, got %s.' % counts.tolist())

        placed_volume = self._volumes @ counts  # [um**3], per parent
        if np.any(np.abs(placed_volume - self._volumes)
                  > self.FRAGMENT_VOLUME_RTOL * self._volumes):
            raise PharmaPyValueError(
                'The fragments must hold their parent\'s volume: they place %s '
                'for parents of L**3 = %s.' % (placed_volume.tolist(), self._volumes.tolist()))

        return counts

    def get_solver_state_rates(self, process, phase, time, completed_state):
        """Distribution rates of agglomeration and breakage at the current state.

        Parameters
        ----------
        process : IntraPhaseProcess
            Process this mechanism serves; its ``phaseref`` names the solid.
        phase : BasePhase
            The solid, unused: the distribution is read from
            ``completed_state``.
        time : float
            Time [s], unused.
        completed_state : dict
            Solver and derived states keyed by ``StateKey``.

        Returns
        -------
        TransferResult
            d(n)/dt [#/(um m3 slurry s)], shape ``(num_grid,)``, keyed by the
            distribution state, with no mass or volume transfer.

        Raises
        ------
        PharmaPyValueError
            If the kernel or selection function returns values that fail the
            checks of ``_kernel_values`` or ``_selection_rates``, or the
            population balance has no liquid or kinetics to evaluate the
            kernel conditions with.
        """
        balance = self.population_balance
        key = StateKey(balance.distribution_state_name, process.phaseref)
        density = balance.true_state(completed_state[key])  # [#/(um m3 slurry)]
        counts = density * balance.dx  # [#/m3 slurry]
        conditions = self._conditions(
            balance.compute_moments(density, balance.x_grid))  # moments [m**k/m3]
        sizes = np.maximum(balance.x_grid, balance.rad)  # [um]

        rates = np.zeros_like(counts)  # [#/(m3 slurry s)]
        if self.agglomeration is not None:
            rates += self._agglomeration_count_rates(
                counts, self._kernel_values(sizes, conditions))

        if self.breakage is not None:
            broken = self._selection_rates(sizes, conditions) * counts  # [#/(m3 s)]
            rates += self._fragment_counts @ broken - broken

        return TransferResult(state_rates={key: rates / balance.dx},
                              aux={'process': process}, net_mass_rate=0.0)

    def review_solution(self, process, solver_history):
        """Warn when the last size class held too much of the crystal volume.

        Parameters
        ----------
        process : IntraPhaseProcess
            Process this mechanism served; its ``phaseref`` names the solid.
        solver_history : dict
            Solved states over time keyed by ``StateKey``, with the times
            [s] under ``'time'``.

        Warns
        -----
        RuntimeWarning
            If at any reported time the last cell held more than
            ``TOP_CELL_VOLUME_SHARE`` of the crystal volume, naming the
            largest share and when it occurred.
        """
        balance = self.population_balance
        key = StateKey(balance.distribution_state_name, process.phaseref)
        density = np.atleast_2d(balance.true_state(solver_history[key]))  # [#/(um m3 slurry)]
        cell_volumes = density * balance.dx * self._volumes  # [um**3/m3 slurry], per cell
        total = cell_volumes.sum(axis=1)  # [um**3/m3 slurry]
        share = np.divide(cell_volumes[:, -1], total, out=np.zeros_like(total),
                          where=total > 0)  # [-]
        worst = int(np.argmax(share))

        if share[worst] > self.TOP_CELL_VOLUME_SHARE:
            warnings.warn(
                'The last size class held %.3g %% of the crystal volume at t = %.6g s, '
                'above the %.3g %% that FVMKernelSources.TOP_CELL_VOLUME_SHARE allows. '
                'Agglomerates past the last node, %.6g um, are kept there at its size; '
                'extend x_grid to larger sizes.'
                % (100 * share[worst], np.asarray(solver_history['time'])[worst],
                   100 * self.TOP_CELL_VOLUME_SHARE, balance.x_grid[-1]),
                RuntimeWarning, stacklevel=2)

    def _agglomeration_count_rates(self, counts, kernel_values):
        """Cell-count rates of cell-average agglomeration.

        Parameters
        ----------
        counts : numpy.ndarray
            Crystals in each cell, N_i [#/m3 slurry], shape ``(num_grid,)``.
        kernel_values : numpy.ndarray
            beta(L_i, L_j) [m**3/s], shape ``(num_grid, num_grid)``.

        Returns
        -------
        numpy.ndarray
            d(N_i)/dt [#/(m3 slurry s)], shape ``(num_grid,)``.

        Notes
        -----
        With B_i agglomerates collected in cell i and V_i their volume, the
        excess E_i = V_i - B_i v_i [um**3] places them: E_i / (v_{i+1} - v_i)
        move to node i + 1 when E_i >= 0, and -E_i / (v_i - v_{i-1}) to node
        i - 1 otherwise, which keeps their number and volume. The excess is
        linear in the counts' products, so the split never divides by B_i.
        The first cell has no deficit to move down: every agglomerate holds
        at least twice the first node's volume. Only integrator noise in the
        counts could make one, and its crystals then stay in the first cell.
        """
        num_grid = len(counts)
        volumes = self._volumes  # [um**3]
        events = (self._pair_weights * kernel_values[self._left, self._right]
                  * counts[self._left] * counts[self._right])  # [#/(m3 s)]

        born = np.bincount(self._birth_cells, events, num_grid)  # [#/(m3 s)]
        born_volume = np.bincount(self._birth_cells, events * self._merged_volumes,
                                  num_grid)  # [um**3/(m3 s)]
        excess = born_volume - born * volumes  # [um**3/(m3 s)]
        above = excess >= 0
        spacing = np.diff(volumes)  # [um**3], v_{i+1} - v_i

        moved_up = np.zeros(num_grid)  # [#/(m3 s)]
        moved_up[:-1] = np.where(above[:-1], excess[:-1] / spacing, 0.0)
        moved_down = np.zeros(num_grid)  # [#/(m3 s)]
        moved_down[1:] = np.where(above[1:], 0.0, -excess[1:] / spacing)

        rates = born - moved_up - moved_down
        rates[1:] += moved_up[:-1]
        rates[:-1] += moved_down[1:]
        # Above the last node: its volume stays in the last cell.
        if above[-1]:
            rates[-1] += excess[-1] / volumes[-1]

        lost = (np.bincount(self._left, events, num_grid)
                + np.bincount(self._right, events, num_grid))  # [#/(m3 s)]

        return rates - lost
