import numpy as np
from PharmaPy.ThermoModule import ThermoPhysicalManager
from PharmaPy.Commons import trapezoidal_rule
from scipy.optimize import newton
import copy
from functools import wraps

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from PharmaPy.Mechanisms import Mechanism
    from PharmaPy.DataClasses import PhaseRef,StateCollection,StateKey
import warnings
from time import perf_counter

eps = np.finfo(float).eps


def classify_phases(instance, names=None):

    phases = instance.Phases

    if names is None:
        solid_count = 1
        liquid_count = 1
        vapor_count = 1

        for phase in phases:
            if 'Liquid' in phase.__class__.__name__:
                phase_name = 'Liquid_{}'.format(liquid_count)
                liquid_count += 1

            elif 'Solid' in phase.__class__.__name__:
                phase_name = 'Solid_{}'.format(solid_count)
                solid_count += 1

            elif 'Vapor' in phase.__class__.__name__:
                phase_name = 'Vapor_{}'.format(solid_count)
                vapor_count += 1

            setattr(phase, 'name', phase_name)
            setattr(instance, phase_name, phase)
    else:
        for phase, name in zip(phases, names):
            setattr(phase, 'name', phase_name)
            setattr(instance, name, phase)


def getPropsPhaseMix(phases, basis='mass'):
    # Empty containers
    props_matrix = np.zeros((len(phases), 3))
    vfrac_phases = []
    props_matrix = []

    for ind, phase in enumerate(phases):
        all_props = phase.getProps(basis=basis)
        props_matrix.append(all_props[:3])

        if phase.__class__.__name__ == 'LiquidPhase':
            ind_liq = ind
        if phase.__class__.__name__ == 'SolidPhase':
            mom_solid = all_props[-2]
            conv_exp = np.arange(len(mom_solid))
            # num/m**3, m/m**3, m**2/m**3, ...
            mom_meters = mom_solid * (1e-6)**conv_exp

            vfrac_solid = mom_meters[-1] * phase.kv
            vfrac_phases.append(vfrac_solid)

    props_matrix = np.vstack(props_matrix)

    # Volume fraction and mass fractions of phases
    vfrac_rest = 1 - sum(vfrac_phases)
    vfrac_phases.insert(ind_liq, vfrac_rest)

    # to avoid internal casting next line
    vfrac_phases = np.array(vfrac_phases)

    mass_phases = vfrac_phases * props_matrix[:, 1]
    mfrac_phases = mass_phases / mass_phases.sum()

    cp, rho, enthalpy = props_matrix.T

    return cp, rho, enthalpy, vfrac_phases, mfrac_phases

def overridable(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        override = self._overrides.get(func.__name__)

        if override is not None:
            return override(*args, **kwargs)

        return func(self, *args, **kwargs)

    return wrapper

def overridable_property(func):
    return property(overridable(func))
def overridable_setter(func):

    @wraps(func)
    def wrapper(self, value):
        if value is None:
            return
        override = self._overrides.get("set_" + func.__name__)

        if override is not None:
            return override(value)

        return func(self, value)

    return wrapper
class BasePhase(ThermoPhysicalManager):
    is_stream= False
    phase_family = None
    stream_class = None
    amount_names = {'mass','moles','vol','mass_j'}
    composition_names = {'mass_j','mass_frac','mass_conc','mole_conc'}

    def __init__(
        self,
        path_thermo=None,
        temp=298.15,
        pressure=101325,
        mass=None,
        moles=None,
        vol=None,
        mass_j=None,
        mass_frac=None,
        mole_frac=None,
        mass_conc=None,
        mole_conc=None,
        check_input=True,
        verbose=True,
        **kwargs,
    ):

        super().__init__(path_thermo)
        self.path_thermo=path_thermo
        self.temp = float(temp)
        self.pres = pressure



        # ----------------------------------------------------------
        # Check extensive specification
        # ----------------------------------------------------------

        amount_count = (
            (mass is not None)
            + (vol is not None)
            + (moles is not None)
        )
        if amount_count > 1:
            raise ValueError("Only one amount update allowed")

        composition_count = (
            (mass_j is not None)
            + (mass_frac is not None)
            + (mole_frac is not None)
            + (mass_conc is not None)
            + (mole_conc is not None)
        )
        if composition_count > 1:
            raise ValueError(
                "Specify one or fewer composition basis: "
                "mass_j, mass_frac, mole_frac, mass_conc, or mole_conc"
            )

        self._mass_frac=None
        self._mass=None
        self._mechanisms = []
        self._overrides = {}
        # ----------------------------------------------------------
        # Initialize composition
        # ----------------------------------------------------------

        self.mass_frac = mass_frac
        self.mole_frac = mole_frac
        self.mass_conc = mass_conc
        self.mole_conc = mole_conc
        self.mass_j = mass_j


        # ----------------------------------------------------------
        # Initialize amount
        # ----------------------------------------------------------
        self.mass = mass
        self.vol = vol
        self.moles = moles

        self.y_upstream = None
        self._name = None
        self.transferred_from_uo = False
        if self.phase_family is None:
            raise TypeError("BasePhase cannot be instantiated directly")

        




    # ==============================================================
    # Composition truth
    # ==============================================================

    @overridable_property
    def mass_j(self):
        
        if self.mass is None:
            return None

        return self.mass * self.mass_frac


    @mass_j.setter
    @overridable_setter
    def mass_j(self, value):

        if value is None:
            return

        value = np.asarray(value, dtype=float)

        if value.shape[0] != self.num_species:
            raise ValueError("mass_j length does not match number of species")

        total_mass = value.sum()

        if total_mass > 0:

            self.mass = total_mass
            self.mass_frac = value / total_mass

        else:

            # Preserve composition information for zero inventory states
            self.mass = np.float64(0.0)

            if self.mass_frac is None:
                raise RuntimeError("Cannot assign zero mass_j without an existing composition")


    @overridable_property
    def mass(self):
        return self._mass
    
    @mass.setter
    @overridable_setter
    def mass(self, value):

        if value is None:
            return

        if self.mass_frac is None:
            raise RuntimeError("Cannot set mass before composition is defined")


        self._mass = np.float64(value)




    @overridable_property
    def mass_frac(self):
        return self._mass_frac


    @mass_frac.setter
    @overridable_setter
    def mass_frac(self,value):

        value=np.asarray(value,dtype=float)

        if not np.isclose(value.sum(),1):
            raise ValueError(
                "mass_frac must sum to one"
            )

        self._mass_frac=value.copy()

    @overridable_property
    def mole_frac(self):
        if self.mass_frac is None:
            return None
        
        return self.frac_to_frac(
            mass_frac=self.mass_frac
        )


    @mole_frac.setter
    @overridable_setter
    def mole_frac(self,value):
        mass_frac=self.frac_to_frac(
            mole_frac=np.asarray(value)
        )

        self.mass_frac=mass_frac



    @overridable_property
    def mole_conc(self):
        if self.mass_frac is None:
            return None
                
        return self.frac_to_conc(
            mass_frac=self.mass_frac,
            basis='mole'
        )


    @mole_conc.setter
    @overridable_setter
    def mole_conc(self,value):

        mass_frac,mole_frac=self.conc_to_frac(
            value
        )

        self.mass_frac=mass_frac



    @overridable_property
    def mass_conc(self):
        if self.mass_frac is None:
            return None
                
        return self.frac_to_conc(
            mass_frac=self.mass_frac,
            basis='mass'
        )


    @mass_conc.setter
    @overridable_setter
    def mass_conc(self,value):

        mass_frac=self.mass_conc_to_frac(value)

        self.mass_frac=mass_frac



    # ==============================================================
    # Extensive derived quantities
    # ==============================================================

    @overridable_property
    def moles(self):
        #keep in mind these are essentially kmol since mass is kg
        if self.mass is None:
            return None
                
        return self.mass / self.mw_av


    @moles.setter
    @overridable_setter
    def moles(self,value):
        self.mass = value*self.mw_av



    @overridable_property
    def vol(self):
        if self.mass is None:
            return None
        return self.mass/self.density


    @vol.setter
    @overridable_setter
    def vol(self,value):
        self.mass=value*self.density



    @overridable_property
    def mw_av(self):
        if self.mass_frac is None:
            return None
        return np.dot(
            self.mole_frac,
            self.mw
        )



    # ==============================================================
    # Backward Compatibility functions -- Deprecation Warning
    # ==============================================================

    @property
    def name(self):
        return self._name
    @name.setter
    def name(self,value):
        if value is not None:
            self._name = value
    def getBubblePoint(self, pres=None, mass_frac=None, mole_frac=None,
                        thermo_method='ideal', y_vap=False):

        if mass_frac is None and mole_frac is None:
            mole_frac = self.mole_frac

        elif mole_frac is None:
            mole_frac = self.frac_to_frac(mass_frac=mass_frac)

        if pres is None:
            pres = self.pres

        def bubble_fn(temp):
            k_vals = self.getKeqVLE(temp, pres, mole_frac,
                                    gamma_model=thermo_method)

            obj = np.dot(mole_frac, (k_vals - 1))

            return obj

        temp_pure = self.AntoineEquation(pres=pres)
        temp_seed = np.dot(mole_frac, temp_pure)
        temp_bubble = newton(bubble_fn, temp_seed, full_output=False)

        if y_vap:
            k_vals = self.getKeqVLE(temp_bubble, pres, mole_frac,
                                    gamma_model=thermo_method)

            y_frac = k_vals * mole_frac

            return temp_bubble, y_frac
        else:
            return temp_bubble

    def getBubblePressure(self, temp=None, mass_frac=None, mole_frac=None,
                            thermo_method='ideal', y_vap=False):

            if mass_frac is None and mole_frac is None:
                mole_frac = self.mole_frac

            elif mole_frac is None:
                mole_frac = self.frac_to_frac(mass_frac=mass_frac)

            if temp is None:
                temp = self.temp

            def bubble_fn(pr):
                k_vals = self.getKeqVLE(temp, pr, mole_frac,
                                        gamma_model=thermo_method)

                obj = np.dot(mole_frac, (k_vals - 1))

                return obj

            pres_pure = self.AntoineEquation(temp=temp)
            pres_seed = np.dot(mole_frac, pres_pure)
            pres_bubble = newton(bubble_fn, pres_seed, full_output=False)

            return pres_bubble
    @overridable
    def getProps(self, basis='mass'):
        cpmass, cpmole = self.getCpMix(self.temp, self.mass_frac)
        rhoMass, rhoMole = self.getDensityMix(self.mass_frac, temp=self.temp)
        hmass, hmole = self.getEnthalpy(self.temp, mass_frac=self.mass_frac)
        # viscosity = self.getViscosityMix(self.temp, self.mass_frac)
        if basis == 'mass':
            cp = cpmass
            enthalpy = hmass
            rho = rhoMass
        else:
            cp = cpmole
            enthalpy = hmole
            rho = rhoMole

        return cp, rho, enthalpy

    def getActivityCoeff(self, method='ideal', mole_frac=None, temp=None):

        if mole_frac is None:
            mole_frac = self.mole_frac

        if temp is None:
            temp = self.temp

        if method == 'ideal':
            gamma = np.ones_like(mole_frac)
        elif method == 'UNIQUAC':
            if 'qip' not in self.__dict__:
                self.qip = self.qi

            gamma = self.UNIQUAC(mole_frac, temp)

        else:
            gamma = self.UNIFAC_DMD(mole_frac, temp)

        return gamma

    def getViscosity(self, temp=None, mass_frac=None, mole_frac=None):
        viscosity = self.getViscosityMix(temp, mass_frac, mole_frac,
                                            phase='liquid')

        return viscosity

    def getSurfTensionPure(self, temp=None):
        surface_pure = self.surf_tension
        surface_pure[np.isnan(surface_pure)] = 0

        return surface_pure

    def getSurfTension(self, mass_frac=None, mole_frac=None, temp=None):

        if mass_frac is None:
            mass_frac = self.mass_frac

        if temp is None:
            temp = self.temp

        surfacePure = self.getSurfTensionPure(temp)
        surfaceMix = np.dot(mass_frac, surfacePure)

        return surfaceMix

    # ==============================================================
    # Used functions
    # ==============================================================

    def updatePhase(
        self,
        mole_conc=None,
        mass_conc=None,
        mass_frac=None,
        mole_frac=None,
        mass_j=None,
        mass=None,
        vol=None,
        moles=None,
        temp=None,
        pres=None,
        **kwargs
    ):

        if temp is not None:
            self.temp=temp

        if pres is not None:
            self.pres=pres


        # composition update

        amount_count = (
            (mass is not None)
            + (vol is not None)
            + (moles is not None)
        )
        if amount_count > 1:
            raise ValueError("Only one amount update allowed")

        composition_count = (
            (mass_j is not None)
            + (mass_frac is not None)
            + (mole_frac is not None)
            + (mass_conc is not None)
            + (mole_conc is not None)
        )
        if composition_count > 1:
            raise ValueError(
                "Specify one or fewer composition basis: "
                "mass_j, mass_frac, mole_frac, mass_conc, or mole_conc"
            )
        if mass_frac is not None:
            self.mass_frac = mass_frac
        elif mole_frac is not None:
            self.mole_frac = mole_frac
        elif mass_conc is not None:
            self.mass_conc = mass_conc
        elif mole_conc is not None:
            self.mole_conc = mole_conc
        elif mass_j is not None:
            self.mass_j = mass_j

        if mass is not None:
            self.mass = mass
        elif moles is not None:
            self.moles = moles
        elif vol is not None:
            self.vol = vol
    
    @overridable_property
    def default_composition_name(self):
        return 'mass_frac'
    @overridable_property
    def default_quantity_name(self):
        return 'mass'

    @property
    def state_dict(self):

        return {
            "mass":None if self.mass is None else self.mass.copy(),
            "vol":self.vol,
            "moles":self.moles,
            "mass_frac": None if self.mass_frac is None else self.mass_frac.copy(),
            "mass_conc":self.mass_conc,
            "mole_conc":self.mole_conc,
            "mass_j":self.mass_j,
            "temp":self.temp,
            "pres":self.pres,
        }

    @overridable
    def get_state_dict(self,state_collection):

        return {
            state.name:getattr(self,state.name)
            for state in state_collection.states.values()
        }


    @overridable
    def getEnthalpy(self,*args,**kwargs):

        if "mass_frac" not in kwargs:
            kwargs["mass_frac"]=self.mass_frac
        if 'phase' not in kwargs:
                    kwargs['phase'] = self.phase_family
        return super().getEnthalpy(
            *args,
            **kwargs
        )

    @overridable
    def getCp(self,*args,**kwargs):

        if "mass_frac" not in kwargs:
            kwargs["mass_frac"]=self.mass_frac
        if 'phase' not in kwargs:
                    kwargs['phase'] = self.phase_family
        if 'temp' not in kwargs:
            kwargs['temp'] = self.temp
        return super().getCpMix(
            *args,
            **kwargs
        )
    @overridable_property
    def density(self):
        return self.getDensity()

    @overridable
    def getDensity(self,*args,**kwargs):
        
        if "mass_frac" not in kwargs:
            kwargs["mass_frac"]=self.mass_frac
        if 'phase' not in kwargs:
            kwargs['phase'] = self.phase_family
        return super().getDensityMix(
            *args,
            **kwargs
        )


    def to_stream(self):
        stream = copy.copy(self)
        stream.__class__ = self.stream_class
        stream.is_stream = True
        stream._DynamicInlet = None
        stream.controllable = ('mass_flow', 'mole_flow', 'vol_flow', 'temp')
        stream.time_upstream = None
        stream.bipartite = None
        stream.composition_names.add('mass_j_flow')
        stream.amount_names.update({'mass_flow','mass_j_flow','vol_flow','mole_flow'})
        return stream

    ######## Mechanism logic
    @property
    def mechanisms(self)->list["Mechanism"]:

        return self._mechanisms
    @mechanisms.setter
    def mechanisms(self, value):
        if value is None:
            mechanisms = []
        elif isinstance(value, (list, tuple, set)):
            mechanisms = list(value)
        else:
            mechanisms = [value]

        self._mechanisms = mechanisms

        overrides = {}
        for mechanism in mechanisms:
            for name, override in mechanism.get_overrides().items():
                if override is None:
                    continue

                if name in overrides:
                    raise AttributeError(
                        f"Multiple mechanisms override {name!r}"
                    )

                overrides[name] = override

        self._overrides = overrides
    def get_mechanism(self,mechanismClass):
        for m in self.mechanisms:
            if isinstance(m,mechanismClass):
                return m
    def _call_override(self, name, *args, **kwargs):
        override = self._overrides.get(name)

        if override is not None:
            return override(*args, **kwargs)

        return None

    def _get_mechanism_attribute_owner(self, name):
        mechanisms = self.__dict__.get("_mechanisms", ())
        matches = []

        for mechanism in mechanisms:
            for attr in mechanism.exposed_attributes:
                if attr == name:
                    matches.append(mechanism)
                    break

        if len(matches) == 1:
            return matches[0]

        if len(matches) > 1:
            raise AttributeError(
                f"Ambiguous phase attribute {name!r}: "
                f"multiple mechanisms own an attribute with this name."
            )

        return None

    def __getattr__(self, name):

        mechanism = self._get_mechanism_attribute_owner(name)

        if mechanism is not None:
            return getattr(mechanism, name)

        raise AttributeError(
            f"{type(self).__name__} has no attribute {name!r}"
        )
    ##### Vessel API
    def update_from_solver_state(self,updates:dict,completed_state:dict["StateKey"],unit=None):
        t0 = perf_counter()
        if updates:
            self.updatePhase(**updates)
        unit._timers['updatePhase0'] = unit._timers.get('updatePhase0',0)+perf_counter()-t0
        t0 = perf_counter()
        for mech in self.mechanisms:
            mech.update_state(completed_state,unit=unit)
        unit._timers['updatePhaseMech'] = unit._timers.get('updatePhaseMech',0)+perf_counter()-t0

class LiquidPhase(BasePhase):
    def __init__(
            self,
            path_thermo=None,
            temp=298.15,
            pressure=101325,
            mass=None,
            moles=None,
            vol=None,
            mass_j=None,
            mass_frac=None,
            mole_frac=None,
            mass_conc=None,
            mole_conc=None,
            name_solv=None,
            check_input=True,
            verbose=True,
            **kwargs,
        ):
        self.phase_family = "liquid"
        from PharmaPy.Streams import LiquidStream
        self.stream_class = LiquidStream
        super().__init__(
            path_thermo=path_thermo,
            temp=temp,
            pressure=pressure,
            mass=mass,
            moles=moles,
            vol=vol,
            mass_j=mass_j,
            mass_frac=mass_frac,
            mole_frac=mole_frac,
            mass_conc=mass_conc,
            mole_conc=mole_conc,
            name_solv=name_solv,
            check_input=check_input,
            verbose=verbose,
            **kwargs,
        )
        self.cp_liq = np.atleast_2d(self.cp_liq)
        self.p_vap = np.atleast_2d(self.p_vap)

    


class VaporPhase(BasePhase):
    def __init__(
        self,
        path_thermo=None,
        temp=298.15,
        pressure=101325,
        mass=None,
        moles=None,
        vol=None,
        mass_j=None,
        mass_frac=None,
        mole_frac=None,
        mass_conc=None,
        mole_conc=None,
        name_solv=None,
        check_input=True,
        verbose=True,
        **kwargs,
    ):
        self.phase_family = 'vapor'
        from PharmaPy.Streams import VaporStream
        self.stream_class = VaporStream
        super().__init__(
            path_thermo=path_thermo,
            temp=temp,
            pressure=pressure,
            mass=mass,
            moles=moles,
            vol=vol,
            mass_j=mass_j,
            mass_frac=mass_frac,
            mole_frac=mole_frac,
            mass_conc=mass_conc,
            mole_conc=mole_conc,
            name_solv=name_solv,
            check_input=check_input,
            verbose=verbose,
            **kwargs,
        )

class SolidPhase(BasePhase):
    def __init__(
        self,
        path_thermo=None,
        temp=298.15,
        pressure=101325,
        mass=None,
        moles=None,
        vol=None,
        mass_j=None,
        mass_frac=None,
        mole_frac=None,
        mass_conc=None,
        mole_conc=None,
        name_solv=None,
        check_input=True,
        verbose=True,
        **kwargs,
    ):
        self.phase_family = 'solid'
        from PharmaPy.Streams import SolidStream
        self.stream_class = SolidStream
        super().__init__(
            path_thermo=path_thermo,
            temp=temp,
            pressure=pressure,
            mass=mass,
            moles=moles,
            vol=vol,
            mass_j=mass_j,
            mass_frac=mass_frac,
            mole_frac=mole_frac,
            mass_conc=mass_conc,
            mole_conc=mole_conc,
            name_solv=name_solv,
            check_input=check_input,
            verbose=verbose,
            **kwargs,
        )

    #============================================================
    # Compatibility API: Deprecation Warning
    # ===========================================================
    def getPorosity(self, distrib=None, diam_filter=1, AR=None,
                    sphericity=None):

        if distrib is None:
            distrib = self.distrib
            mom_zero = self.moments[0]
            mom_one = self.moments[1]
        else:
            mom_zero, mom_one = self.getMoments(mom_num=(0, 1))

        # mom_one *= 1e-6  # m
        x_dist = self.x_distrib * 1e-6  # m

        if AR is None:
            AR = 2

        if sphericity is None:
            sphericity = 0.7

        # Yu, Zou et al (1996) and Yu,Zou, Stnadish (1996) model
        kv = 0.524  # Volumetric shape coefficient
        ks = 3.142  # Surface shape coefficient

        del_x_dist = np.diff(x_dist)
        node_x_dist = (x_dist[:-1] + x_dist[1:]) / 2
        node_CSD = (distrib[:-1] + distrib[1:]) / 2

        # Volume of crystals in each bin
        vol_cry = node_CSD * del_x_dist * (kv * node_x_dist**3)
        frac_vol_cry = vol_cry / (np.sum(vol_cry) + eps)

        vol_particle = kv * node_x_dist**3
        d_part_sphere = (6 * vol_particle / np.pi)**(1/3)
        d_part_equiv_pack = d_part_sphere / (sphericity**2.785 *
                                                np.exp(2.946 * (1 - sphericity)))

        # Initial porosity
        D_mean = mom_one/(mom_zero + eps)
        E_0_Jeschar = 0.375 + 0.34 * D_mean/diam_filter  # average porosity of packing of uniform sized spheres [-]

        initial_porosity = E_0_Jeschar

        V = 1/(1 - initial_porosity) * np.ones_like(node_x_dist)  # Specific Volume for initial porosity

        # Evaluate specific volume using modified linear packing model
        num_x = len(node_x_dist)
        V_T_node = np.zeros(num_x)

        for i in range(num_x):

            r = d_part_equiv_pack[:i] / d_part_equiv_pack[i]
            g_r = (1 - r)**2 + 0.4*r*(1 - r)**3.7
            V_large_j = V[:i] - (V[:i] - 1) * g_r - V[i]
            sum_V_large_term = sum(V_large_j * frac_vol_cry[:i])

            r_inv = r = d_part_equiv_pack[i] / d_part_equiv_pack[i + 1:]
            f_r = (1 - r_inv)**3.3 + 2.8*r_inv*(1 - r_inv)**2.7
            V_small_j = V[i + 1:] * (1 - f_r) - V[i]
            sum_V_small_term = sum(V_small_j * frac_vol_cry[i + 1:])

            V_T_node[i] = V[i] + sum_V_large_term + sum_V_small_term

        V_T = max(V_T_node)

        porosity = 1 - 1/V_T

        return porosity