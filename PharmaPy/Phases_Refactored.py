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

eps = np.finfo(float).eps

# CODATA 2018: R = 8.314462618 J/(mol K), rounded to four significant figures
# to match the existing gas_ct = 8.314 in Kinetics, Crystallizers, Reactors,
# Drying_Model, Evaporators, and ThermoModule, so vapor amounts agree with
# the evaporator's own P V / (R T).
VAPOR_GAS_CONSTANT = 8.314  # [J/mol/K]

# Exponent n of the Watson latent-heat correlation,
# dh(T) = dh(Tref) * ((Tc - T) / (Tc - Tref))**n (K. M. Watson, Ind. Eng.
# Chem. 1943, 35, 398-406; Poling, Prausnitz and O'Connell, 5th ed., 2001,
# Section 7-11, Eq. (7-11.1), p. 7.24).
WATSON_EXPONENT = 0.38  # [-]

# Exact SI conversion factor between grams and kilograms.
GRAMS_PER_KG = 1000.0  # [g/kg]

# Exact SI conversion factor between liters and cubic meters.
LITERS_PER_M3 = 1000.0  # [L/m**3]


def _resolve_phase_family(phase) -> str:
    """Resolve the canonical phase family of a phase or stream object.

    Parameters
    ----------
    phase : object
        Phase or stream instance.

    Returns
    -------
    str
        ``'liquid'``, ``'solid'``, ``'vapor'``, or ``''`` when unrecognized.
        Explicit ``phase_family`` takes precedence over class-name substring
        matching so subclasses whose names contain another family word (such as
        ``LiquidCrystal(SolidPhase)``) are classified by their declared family.
    """
    family = getattr(phase, 'phase_family', None)
    family_lower = family.lower() if isinstance(family, str) else ''
    if family_lower in ('liquid', 'solid'):
        return family_lower
    if family_lower in ('vapor', 'gas'):
        return 'vapor'
    class_name = phase.__class__.__name__
    if 'Liquid' in class_name:
        return 'liquid'
    if 'Solid' in class_name:
        return 'solid'
    if 'Vapor' in class_name:
        return 'vapor'
    return ''


def classify_phases(instance, names=None):
    """Classify phases on an instance using legacy 1-indexed attribute names.

    Parameters
    ----------
    instance : object
        Instance holding ``Phases`` as an iterable of phase objects.
    names : Sequence[str], optional
        Explicit attribute names to assign to ``instance.Phases`` in order.
        When ``None``, 1-indexed family names (``Liquid_1``, ``Solid_1``,
        ``Vapor_1``, ...) are generated.

    Returns
    -------
    None
        Installs ``name`` on each phase and the corresponding attribute on
        ``instance`` in place.

    .. deprecated::
        Use :class:`~PharmaPy.MixedPhases_Refactored.MixedPhase` or
        :class:`~PharmaPy.DataClasses.PhaseStateCollection` instead.
    """
    warnings.warn(
        "classify_phases in PharmaPy.Phases_Refactored is deprecated; "
        "use MixedPhase or PhaseStateCollection phase indexing instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    phases = instance.Phases

    if names is None:
        solid_count = 1
        liquid_count = 1
        vapor_count = 1

        for phase in phases:
            resolved_family = _resolve_phase_family(phase)
            if resolved_family == 'liquid':
                phase_name = 'Liquid_{}'.format(liquid_count)
                liquid_count += 1

            elif resolved_family == 'solid':
                phase_name = 'Solid_{}'.format(solid_count)
                solid_count += 1

            elif resolved_family == 'vapor':
                phase_name = 'Vapor_{}'.format(vapor_count)
                vapor_count += 1
            else:
                phase_name = phase.__class__.__name__

            setattr(phase, 'name', phase_name)
            setattr(instance, phase_name, phase)
    else:
        for phase, name in zip(phases, names):
            setattr(phase, 'name', name)
            setattr(instance, name, phase)


def getPropsPhaseMix(phases, basis='mass'):
    """Aggregate phase properties using the legacy tuple contract.

    Parameters
    ----------
    phases : Sequence[BasePhase]
        Sequence of phases containing one liquid phase and optional solid
        phases.
    basis : {'mass', 'mole'}, optional
        Mixture basis for heat capacity, density, and enthalpy; default
        ``'mass'``.

    Returns
    -------
    cp : numpy.ndarray
        Heat capacities of each phase, [J/kg/K] for ``basis='mass'`` or
        [J/mol/K] for ``basis='mole'``, shape ``(num_phases,)``.
    rho : numpy.ndarray
        Densities of each phase, [kg/m**3] for ``basis='mass'`` or [mol/L]
        for ``basis='mole'``, shape ``(num_phases,)``.
    enthalpy : numpy.ndarray
        Enthalpies of each phase, [J/kg] for ``basis='mass'`` or [J/mol] for
        ``basis='mole'``, shape ``(num_phases,)``.
    vfrac_phases : numpy.ndarray
        Phase volume fractions [-], shape ``(num_phases,)``.
    mfrac_phases : numpy.ndarray
        Phase mass or molar fractions [-] matching ``basis``, shape
        ``(num_phases,)``.

    Raises
    ------
    ValueError
        If ``basis`` is invalid for the underlying phase property methods.

    .. deprecated::
        Use :class:`~PharmaPy.MixedPhases_Refactored.MixedPhase` property
        methods instead.
    """
    warnings.warn(
        "getPropsPhaseMix in PharmaPy.Phases_Refactored is deprecated; "
        "use MixedPhase property methods instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    vfrac_phases = []
    props_matrix = []
    ind_liq = 0
    total_vol = sum(
        float(getattr(p, 'vol', 0.0) or 0.0) for p in phases
    )  # [m**3] or [m**3/s]

    for ind, phase in enumerate(phases):
        all_props = (
            phase.getCp(basis=basis),
            phase.getDensity(basis=basis),
            phase.getEnthalpy(basis=basis),
        )
        props_matrix.append(all_props[:3])

        resolved_family = _resolve_phase_family(phase)
        if resolved_family == 'liquid':
            ind_liq = ind
        elif resolved_family == 'solid':
            if getattr(phase, 'moments', None) is not None and hasattr(phase, 'kv'):
                mom_solid = np.asarray(phase.moments, dtype=float)
                conv_exp = np.arange(len(mom_solid))
                mom_meters = mom_solid * (1e-6)**conv_exp  # [m**k/m**3]
                vfrac_solid = mom_meters[-1] * phase.kv  # [-]
            elif total_vol > 0 and getattr(phase, 'vol', None) is not None:
                vfrac_solid = float(phase.vol) / total_vol  # [-]
            else:
                vfrac_solid = 0.0  # [-]
            vfrac_phases.append(vfrac_solid)

    props_matrix = np.vstack(props_matrix)

    # Volume fraction and mass fractions of phases
    vfrac_rest = 1 - sum(vfrac_phases)  # [-]
    vfrac_phases.insert(ind_liq, vfrac_rest)

    # to avoid internal casting next line
    vfrac_phases = np.array(vfrac_phases)  # [-]

    mass_phases = vfrac_phases * props_matrix[:, 1]
    mfrac_phases = mass_phases / mass_phases.sum()  # [-]

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

    # Index of the solvent species, or None when the phase has no designated
    # solvent. Declared on the class so that phases built by paths which skip
    # __init__ -- copy.copy, to_stream -- still answer the attribute. A vessel
    # can also fill this in from a mechanism; see adopt_solvent_index.
    ind_solv = None
    name_solv = None

    # Set by PharmaPy.Connections when a connection is wired. Declared here
    # because reading it on a stream that was never connected would otherwise
    # fall through to __getattr__ and raise from inside mechanism delegation.
    y_inlet = None

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
        """Initialize a refactored phase state.

        Parameters
        ----------
        path_thermo : str or pathlib.Path, optional
            Path to the thermodynamic property database JSON file.
        temp : float or numpy.ndarray, optional
            Phase temperature [K]; default 298.15 K.
        pressure : float or numpy.ndarray, optional
            Phase pressure [Pa]; default 101325 Pa.
        mass : float or numpy.ndarray, optional
            Total phase mass [kg] (or mass flow rate [kg/s] on streams).
        moles : float or numpy.ndarray, optional
            Total molar amount [mol] (or molar flow rate [mol/s] on streams).
        vol : float or numpy.ndarray, optional
            Total phase volume [m**3] (or volumetric flow rate [m**3/s] on
            streams).
        mass_j : numpy.ndarray, optional
            Species masses [kg] (or species mass flow rates [kg/s] on streams),
            shape ``(num_species,)``.
        mass_frac, mole_frac : numpy.ndarray, optional
            Species mass or mole fractions [-], shape ``(num_species,)`` or
            ``(num_points, num_species)``.
        mass_conc : numpy.ndarray, optional
            Species mass concentrations [kg/m**3], shape ``(num_species,)`` or
            ``(num_points, num_species)``.
        mole_conc : numpy.ndarray, optional
            Species molar concentrations [mol/L], shape ``(num_species,)`` or
            ``(num_points, num_species)``.
        name_solv : str, optional
            Solvent species name used for concentration-to-fraction closure.
        check_input : bool, optional
            Validate composition inputs when ``True``.
        verbose : bool, optional
            Emit informational warnings when ``True``.
        **kwargs
            Optional keyword arguments ``temp_ref`` (reference temperature [K];
            default 298.15 K) and ``pres`` (legacy alias for ``pressure``
            [Pa]).

        Raises
        ------
        ValueError
            If ``name_solv`` is not in ``name_species``, more than one extensive
            quantity or composition basis is supplied, or unknown keyword
            arguments remain.
        TypeError
            If :class:`BasePhase` is instantiated directly without a concrete
            ``phase_family``.
        """
        super().__init__(path_thermo)
        self.path_thermo=path_thermo
        temp_ref = kwargs.pop("temp_ref", 298.15)  # [K]
        pres = kwargs.pop("pres", None)  # [Pa]
        if pres is not None:
            pressure = pres  # [Pa]
        self.temp = float(temp) if np.ndim(temp) == 0 else np.asarray(temp, dtype=float)  # [K]
        self.temp_ref = float(temp_ref)  # [K]
        self.pres = pressure  # [Pa]

        # Must come after super().__init__ (name_species) and after temp is
        # set, because closing the solvent balance evaluates a temperature
        # dependent pure density -- and before any composition is assigned
        # below, since the concentration setters consume it.
        self.name_solv = name_solv

        if name_solv is None:
            self.ind_solv = None
        else:
            try:
                self.ind_solv = list(self.name_species).index(name_solv)
            except ValueError:
                raise ValueError(
                    f"name_solv={name_solv!r} is not one of this phase's "
                    f"species {list(self.name_species)}"
                ) from None



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
        """Species molar concentrations [mol/L] (equivalently [kmol/m**3]).

        Returns
        -------
        numpy.ndarray or None
            Species molar concentrations [mol/L], shape ``(num_species,)`` or
            ``(num_points, num_species)``, evaluated from the phase's own mass
            density (liquid ideal-mixing, solid pure-component mixing, or
            vapor ideal-gas density). ``None`` when composition is unset.
        """
        if self.mass_frac is None:
            return None
        if self.phase_family in ("solid", "vapor"):
            rho_mass = self.getDensity(basis="mass")  # [kg/m**3]
            return (
                np.asarray(self.mass_frac, dtype=float)
                * np.expand_dims(rho_mass, axis=-1)
                / self.mw
            )  # [mol/L]

        return self.frac_to_conc(
            mass_frac=self.mass_frac,
            basis='mole'
        )  # [mol/L]


    @mole_conc.setter
    @overridable_setter
    def mole_conc(self,value):
        """Set species molar concentrations [mol/L] and reconcile composition [-].

        Parameters
        ----------
        value : array-like
            Species molar concentrations [mol/L], shape ``(num_species,)`` or
            ``(num_points, num_species)``.

        Raises
        ------
        ValueError
            If a non-vapor phase has no designated solvent species.
        """
        if self.phase_family == "vapor" and self.ind_solv is None:
            conc = np.asarray(value, dtype=float)  # [mol/L]
            total = conc.sum(axis=-1, keepdims=True)  # [mol/L]
            mole_frac = np.divide(conc, total, out=np.zeros_like(conc), where=total > 0)  # [-]
            self.mole_frac = mole_frac  # [-]
            return

        # Index rather than unpack: conc_to_frac returns 2 or 3 values
        # depending on solvent_ind, and its gate is a truthiness test while
        # mass_conc_to_frac's is an "is None" test. mass_frac is element 0 on
        # every branch of both, so this is correct regardless.
        # np.array(..., dtype=float) because conc_to_frac writes the closed
        # solvent value back into the array it is given, and silently
        # truncates it if that array is of integer dtype.
        out = self.conc_to_frac(
            np.array(value, dtype=float),
            solvent_ind=self._require_solvent_index('mole_conc'),
        )

        self.mass_frac = out[0]  # [-]



    @overridable_property
    def mass_conc(self):
        """Species mass concentrations [kg/m**3].

        Returns
        -------
        numpy.ndarray or None
            Species mass concentrations [kg/m**3], shape ``(num_species,)`` or
            ``(num_points, num_species)``, evaluated from the phase's own mass
            density. ``None`` when composition is unset.
        """
        if self.mass_frac is None:
            return None
        if self.phase_family in ("solid", "vapor"):
            rho_mass = self.getDensity(basis="mass")  # [kg/m**3]
            return (
                np.asarray(self.mass_frac, dtype=float)
                * np.expand_dims(rho_mass, axis=-1)
            )  # [kg/m**3]

        return self.frac_to_conc(
            mass_frac=self.mass_frac,
            basis='mass'
        )  # [kg/m**3]


    @mass_conc.setter
    @overridable_setter
    def mass_conc(self,value):
        """Set species mass concentrations [kg/m**3] and reconcile composition [-].

        Parameters
        ----------
        value : array-like
            Species mass concentrations [kg/m**3], shape ``(num_species,)`` or
            ``(num_points, num_species)``.

        Raises
        ------
        ValueError
            If a non-vapor phase has no designated solvent species.
        """
        if self.phase_family == "vapor" and self.ind_solv is None:
            conc = np.asarray(value, dtype=float)  # [kg/m**3]
            total = conc.sum(axis=-1, keepdims=True)  # [kg/m**3]
            mass_frac = np.divide(conc, total, out=np.zeros_like(conc), where=total > 0)  # [-]
            self.mass_frac = mass_frac  # [-]
            return

        # See mole_conc.setter. Unpacking here was an outright bug: this
        # helper returns a 3-tuple whenever a solvent index is supplied, so
        # mass_conc= always raised "mass_frac must sum to one".
        out = self.mass_conc_to_frac(
            np.array(value, dtype=float),
            solvent_ind=self._require_solvent_index('mass_conc'),
        )

        self.mass_frac = out[0]  # [-]


    def _require_solvent_index(self, basis):
        """Solvent index for a concentration -> fraction conversion.

        A concentration is only meaningful relative to a solvent: without
        knowing which species it is, the conversion cannot close the balance
        and silently renormalizes instead, discarding the absolute scale. That
        is how a liquid specified by mole_conc ended up with zero solvent.

        Rather than guess, refuse. The index can come from name_solv on the
        phase, or from a mechanism by way of adopt_solvent_index.
        """
        if self.ind_solv is not None:
            return self.ind_solv

        raise ValueError(
            f"Setting {basis} on this {type(self).__name__} needs to know "
            "which species is the solvent, and none is set. Either construct "
            "the phase with name_solv='<species>', or attach it to a vessel "
            "whose mechanism declares a solvent so the phase can adopt it."
        )


    def adopt_solvent_index(self, ind_solv, name_solv=None, source=''):
        """Take a solvent index supplied by something else, e.g. a mechanism.

        A crystallization mechanism already knows its solvent, but it belongs
        to the solid phase while the concentration lives on the liquid. This
        lets a vessel hand that index across. An explicitly configured phase
        wins -- a disagreement is an error rather than a silent overwrite,
        because the two would then describe different solutions.
        """
        if ind_solv is None:
            return

        if self.ind_solv is not None:
            if self.ind_solv != ind_solv:
                raise ValueError(
                    f"{type(self).__name__} was configured with solvent "
                    f"{self.name_solv!r} (index {self.ind_solv}), but "
                    f"{source or 'a mechanism'} declares index {ind_solv}"
                    f"{f' ({name_solv!r})' if name_solv else ''}. "
                    "They must agree."
                )
            return

        self.ind_solv = ind_solv
        self.name_solv = name_solv



    # ==============================================================
    # Extensive derived quantities
    # ==============================================================

    @overridable_property
    def moles(self):
        """Total molar amount of the phase [mol] (or molar flow [mol/s] on streams).

        Returns
        -------
        float, numpy.ndarray, or None
            Total molar amount [mol] (or [mol/s] for streams), related to
            ``self.mass`` [kg] and ``self.mw_av`` [g/mol] by
            ``mass * 1000 / mw_av`` (issue #156). ``None`` when ``self.mass``
            is ``None``.
        """
        if self.mass is None:
            return None

        return self.mass * GRAMS_PER_KG / self.mw_av  # [mol]


    @moles.setter
    @overridable_setter
    def moles(self,value):
        """Set total molar amount [mol] (or molar flow [mol/s] on streams) and update mass [kg].

        Parameters
        ----------
        value : float or numpy.ndarray
            Total molar amount [mol] (or molar flow rate [mol/s] for streams).
        """
        self.mass = value * self.mw_av / GRAMS_PER_KG  # [kg]



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
        """Return heat capacity, density, and enthalpy on the requested basis.

        Parameters
        ----------
        basis : {'mass', 'mole'}, optional
            Physical basis of the returned properties; default ``'mass'``.

        Returns
        -------
        cp : float or numpy.ndarray
            Heat capacity, [J/kg/K] for ``basis='mass'`` or [J/mol/K] for
            ``basis='mole'``.
        rho : float or numpy.ndarray
            Density, [kg/m**3] for ``basis='mass'`` or [mol/L] for
            ``basis='mole'``.
        enthalpy : float or numpy.ndarray
            Enthalpy, [J/kg] for ``basis='mass'`` or [J/mol] for
            ``basis='mole'``.

        Raises
        ------
        ValueError
            If ``basis`` is neither ``'mass'`` nor ``'mole'``.

        .. deprecated::
            Call :meth:`getCp`, :meth:`getDensity`, and :meth:`getEnthalpy`
            directly.
        """
        # stacklevel=3 steps past @overridable's wrapper frame to the caller.
        warnings.warn(
            "getProps is deprecated; call getCp(basis=...), "
            "getDensity(basis=...), and getEnthalpy(basis=...) directly.",
            DeprecationWarning,
            stacklevel=3,
        )
        if basis not in ('mass', 'mole'):
            raise ValueError("basis must be 'mass' or 'mole'")
        cp = self.getCp(basis=basis)  # [J/kg/K] or [J/mol/K]
        rho = self.getDensity(basis=basis)  # [kg/m**3] or [mol/L]
        enthalpy = self.getEnthalpy(basis=basis)  # [J/kg] or [J/mol]
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


    def _stored_fractions(self):
        """Return stored composition fractions, zeroing sentinel species lacking Cp data.

        Returns
        -------
        mass_frac, mole_frac : numpy.ndarray or None
            Stored mass and mole fractions [-]. A species whose stored mass
            fraction is at or below the zero sentinel ``eps`` in every row and
            that has no heat-capacity polynomial for the phase gets exact zeros
            in both arrays.
        """
        if self.mass_frac is None:
            return None, None
        prop_by_phase = {
            "liquid": "cp_liq",
            "solid": "cp_solid",
            "vapor": "cp_vapor",
        }
        prop_name = prop_by_phase.get(self.phase_family)
        if prop_name is None:
            return self.mass_frac, self.mole_frac
        mass_frac = np.asarray(self.mass_frac, dtype=float)  # [-]
        mole_frac = np.asarray(self.mole_frac, dtype=float)  # [-]
        rows = self._property_rows(prop_name)
        if mass_frac.shape[-1] != len(rows):
            return self.mass_frac, self.mole_frac
        sentinel = np.atleast_2d(mass_frac <= eps).all(axis=0)  # [-]
        absent = sentinel & np.isnan(rows).any(axis=1)
        if not absent.any():
            return self.mass_frac, self.mole_frac
        return (
            np.where(absent, 0.0, mass_frac),
            np.where(absent, 0.0, mole_frac),
        )

    def _classify_criticality(self, temp):
        """Validate temperatures and classify every species at each of them.

        Parameters
        ----------
        temp : float or array-like
            Scalar temperature or one-dimensional array of independent
            temperatures [K].

        Returns
        -------
        temp_values : ndarray
            Requested temperatures as floats [K], shape ``(num_temperatures,)``.
        is_supercritical : ndarray of bool
            Criticality mask [-], shape ``(num_temperatures, num_species)``.

        Raises
        ------
        ValueError
            If ``temp`` has more than one dimension, is empty, or contains
            non-finite values.
        """
        temp_values = np.asarray(temp, dtype=float)  # [K]
        if temp_values.ndim > 1:
            raise ValueError(
                "temp must be a scalar or a one-dimensional array of "
                f"independent temperatures [K]; got shape {temp_values.shape}"
            )
        temp_values = np.atleast_1d(temp_values)  # [K]
        if temp_values.size == 0:
            raise ValueError("temp must contain at least one temperature [K]")
        non_finite = np.flatnonzero(~np.isfinite(temp_values))
        if non_finite.size > 0:
            raise ValueError(
                "temp must be finite [K]; got "
                f"{temp_values[non_finite].tolist()} at positions "
                f"{non_finite.tolist()}"
            )
        is_supercritical = temp_values[:, np.newaxis] > self.t_crit  # [-]
        return temp_values, is_supercritical

    def getHeatVaporization(self, temp=None, basis='mass'):
        """Calculate the latent heat of vaporization of each species.

        Parameters
        ----------
        temp : float or array-like, optional
            Temperature at which the latent heat is evaluated [K]; defaults to
            ``self.temp``.
        basis : {'mass', 'mole'}, optional
            Basis of the returned latent heat. Default ``'mass'``.

        Returns
        -------
        ndarray
            Latent heat of vaporization per species, in [J/kg] for
            ``basis='mass'`` and [J/mol] for ``basis='mole'``.

        Raises
        ------
        ValueError
            If ``basis`` is invalid, ``temp`` is invalid, or the Watson
            reference properties of a subcritical species are invalid.
        """
        if basis not in ('mass', 'mole'):
            raise ValueError("basis must be 'mass' or 'mole'")
        if temp is None:
            temp = self.temp
        temp_values, is_supercritical = self._classify_criticality(temp)  # [K], [-]
        num_temp, num_comp = is_supercritical.shape

        has_critical_data = ~np.isnan(self.t_crit)  # [-]
        uses_watson = ~is_supercritical & has_critical_data  # [-]
        rows, cols = np.nonzero(uses_watson)

        missing_data = np.full(num_comp, np.nan)  # [K] or [J/mol]
        tref_hvap = np.asarray(getattr(self, 'tref_hvap', missing_data), dtype=float)  # [K]
        delta_hvap = np.asarray(getattr(self, 'delta_hvap', missing_data), dtype=float)  # [J/mol]

        watson_columns = np.unique(cols)
        t_crit_used = self.t_crit[watson_columns]  # [K]
        tref_used = tref_hvap[watson_columns]  # [K]
        delta_hvap_used = delta_hvap[watson_columns]  # [J/mol]
        valid_reference = (
            np.isfinite(t_crit_used)
            & np.isfinite(tref_used)
            & np.isfinite(delta_hvap_used)
            & (tref_used < t_crit_used)
        )  # [-]
        if not np.all(valid_reference):
            invalid = ~valid_reference  # [-]
            names = [self.name_species[ind] for ind in watson_columns[invalid]]
            raise ValueError(
                "The Watson latent-heat correlation needs finite t_crit, "
                "tref_hvap and delta_hvap with tref_hvap < t_crit for every "
                "species evaluated at or below its critical temperature; "
                f"check the property data of {names}: "
                f"t_crit = {t_crit_used[invalid].tolist()} K, "
                f"tref_hvap = {tref_used[invalid].tolist()} K, "
                f"delta_hvap = {delta_hvap_used[invalid].tolist()} J/mol"
            )

        watson_ratio = (
            (self.t_crit[cols] - temp_values[rows])
            / (self.t_crit[cols] - tref_hvap[cols])
        )  # [-]
        bad_ratio = ~(np.isfinite(watson_ratio) & (watson_ratio >= 0))  # [-]
        if np.any(bad_ratio):
            names = sorted({self.name_species[ind] for ind in cols[bad_ratio]})
            raise ValueError(
                "The Watson ratio (t_crit - temp) / (t_crit - tref_hvap) must "
                "be finite and non-negative; got "
                f"{watson_ratio[bad_ratio].tolist()} for species {names} at temp = "
                f"{temp_values[rows[bad_ratio]].tolist()} K"
            )
        watson = watson_ratio**WATSON_EXPONENT  # [-]

        deltahvap = np.zeros((num_temp, num_comp))  # [J/mol]
        deltahvap[rows, cols] = watson * delta_hvap[cols]  # [J/mol]

        if basis == 'mass':
            deltahvap[rows, cols] = (
                deltahvap[rows, cols] / self.mw[cols] * GRAMS_PER_KG
            )  # [J/kg]

        if num_temp == 1:
            deltahvap = deltahvap[0]  # [J/kg] or [J/mol]

        return deltahvap

    def _validate_fraction_width(self, fractions, name):
        """Check that supplied fractions carry one entry per species.

        Parameters
        ----------
        fractions : array-like
            Species mass or mole fractions [-], shape ``(num_species,)`` or
            ``(num_temperatures, num_species)``.
        name : str
            Public argument name used in the error message.

        Returns
        -------
        numpy.ndarray
            ``fractions`` as a float array [-] with unchanged shape and values.

        Raises
        ------
        ValueError
            If ``fractions`` is not one- or two-dimensional or its last axis
            does not have ``num_species`` entries.
        """
        fraction_values = np.asarray(fractions, dtype=float)  # [-]
        if (
            fraction_values.ndim not in (1, 2)
            or fraction_values.shape[-1] != self.num_species
        ):
            raise ValueError(
                f"{name} must have shape (num_species,) = "
                f"{(self.num_species,)} or (num_temperatures, num_species) "
                f"with num_species = {self.num_species} in the order "
                f"{self.name_species}; got shape {fraction_values.shape}"
            )
        return fraction_values

    def _get_vapor_enthalpy(
        self,
        temp=None,
        temp_ref=None,
        mass_frac=None,
        mole_frac=None,
        total_h=True,
        basis='mass',
        weights=None,
        **kwargs,
    ):
        """Calculate vapor-phase enthalpy relative to a liquid reference.

        Parameters
        ----------
        temp : float or array-like, optional
            Temperature at which to calculate enthalpy [K], scalar or shape
            ``(num_temperatures,)``. Defaults to ``self.temp`` when ``None``.
        temp_ref : float, optional
            Liquid-reference temperature for the sensible-enthalpy integral
            [K]. Defaults to ``self.temp_ref`` (298.15 K) when ``None``.
        mass_frac, mole_frac : numpy.ndarray, optional
            Vapor-phase species fractions [-], shape ``(num_species,)`` or
            ``(num_temperatures, num_species)``. ``mass_frac`` takes precedence
            over ``mole_frac`` when both are supplied; stored fractions are
            used when both are ``None``.
        total_h : bool, optional
            If ``True`` (default), return the fraction-weighted mixture
            enthalpy; if ``False``, return unweighted species enthalpies.
        basis : {'mass', 'mole'}, optional
            Physical basis of the returned enthalpy; default ``'mass'``.
        weights : array-like, optional
            Used when ``total_h=False``: species amounts or flow rates [any
            basis, e.g. kg/s or mol/s] by which the caller weights the returned
            species enthalpies, shape ``(num_species,)`` or
            ``(num_temperatures, num_species)``. When given, only species with a
            nonzero weight in some row require heat-capacity coefficients.

        Returns
        -------
        float or numpy.ndarray
            Vapor-phase enthalpy, [J/kg] for ``basis='mass'`` and [J/mol] for
            ``basis='mole'``. Scalar for one temperature with ``total_h=True``,
            shape ``(num_temperatures,)`` for multiple temperatures, and shape
            ``(num_temperatures, num_species)`` when ``total_h=False``.

        Raises
        ------
        ValueError
            If ``basis`` is invalid, ``temp`` is empty/non-finite/above 1D, or
            fraction or weight shapes are invalid.
        MissingPropertyError
            If required ``cp_liq`` or ``cp_vapor`` coefficients are missing.
        """
        if basis not in ('mass', 'mole'):
            raise ValueError("basis must be 'mass' or 'mole'")
        if mass_frac is None and mole_frac is None:
            mass_frac = self.mass_frac  # [-]
            mole_frac = self.mole_frac  # [-]
        elif mass_frac is None:
            mole_frac = self._validate_fraction_width(mole_frac, 'mole_frac')  # [-]
            mass_frac = self.frac_to_frac(mole_frac=mole_frac)  # [-]
        else:
            mass_frac = self._validate_fraction_width(mass_frac, 'mass_frac')  # [-]
            mole_frac = self.frac_to_frac(mass_frac)  # [-]

        if temp is None:
            temp = self.temp
        if temp_ref is None:
            temp_ref = getattr(self, 'temp_ref', 298.15)  # [K]

        temp_values, is_supercritical = self._classify_criticality(temp)  # [K], [-]
        num_temp, num_comp = is_supercritical.shape

        if total_h:
            frac = np.asarray(
                mass_frac if basis == 'mass' else mole_frac, dtype=float
            )  # [-]
            if frac.shape not in ((num_comp,), (num_temp, num_comp)):
                raise ValueError(
                    "The weighting composition must have shape "
                    f"(num_species,) = {(num_comp,)} for a fixed composition "
                    "or (num_temperatures, num_species) = "
                    f"{(num_temp, num_comp)} for a profile paired row by row "
                    f"with temp; got {frac.shape}"
                )
            is_used = np.broadcast_to(frac != 0, is_supercritical.shape)
        elif weights is not None:
            weights_arr = np.asarray(weights, dtype=float)
            if weights_arr.shape not in ((num_comp,), (num_temp, num_comp)):
                raise ValueError(
                    "weights must have shape "
                    f"(num_species,) = {(num_comp,)} for a fixed weighting "
                    "or (num_temperatures, num_species) = "
                    f"{(num_temp, num_comp)} for a profile paired row by row "
                    f"with temp; got {weights_arr.shape}"
                )
            is_used = np.broadcast_to(weights_arr != 0, is_supercritical.shape)
        else:
            is_used = np.ones_like(is_supercritical)

        subcritical_columns = np.flatnonzero(
            (~is_supercritical & is_used).any(axis=0)
        )
        supercritical_columns = np.flatnonzero(
            (is_supercritical & is_used).any(axis=0)
        )

        h_liquid = np.zeros((num_temp, num_comp))  # [J/kg] or [J/mol]
        if subcritical_columns.size > 0:
            self._check_species_rows(
                'cp_liq',
                self._property_rows('cp_liq')[subcritical_columns],
                subcritical_columns,
                np.ones(subcritical_columns.size, bool),
                'are at or below t_crit (or have no t_crit) at a requested '
                'temperature, so their enthalpy',
            )
            h_liquid[:, subcritical_columns] = super().getEnthalpy(
                temp_values,
                temp_ref,
                phase='liquid',
                basis=basis,
                idx=subcritical_columns,
                total_h=False,
            )  # [J/kg] or [J/mol]

        h_vapor = np.zeros((num_temp, num_comp))  # [J/kg] or [J/mol]
        if supercritical_columns.size > 0:
            self._check_species_rows(
                'cp_vapor',
                self._property_rows('cp_vapor')[supercritical_columns],
                supercritical_columns,
                np.ones(supercritical_columns.size, bool),
                'are supercritical (temp > t_crit) at a requested '
                'temperature, so their enthalpy',
                remedy=' or evaluate at or below t_crit',
            )
            h_vapor[:, supercritical_columns] = super().getEnthalpy(
                temp_values,
                temp_ref,
                phase='vapor',
                basis=basis,
                idx=supercritical_columns,
                total_h=False,
            )  # [J/kg] or [J/mol]

        h_sensible = np.where(is_supercritical, h_vapor, h_liquid)  # [J/kg] or [J/mol]
        latent_heat = np.reshape(
            self.getHeatVaporization(temp_values, basis=basis),
            (num_temp, num_comp),
        )  # [J/kg] or [J/mol]
        h_species = h_sensible + latent_heat  # [J/kg] or [J/mol]

        if not total_h:
            return h_species

        h_total = (h_species * frac).sum(axis=1)  # [J/kg] or [J/mol]
        if num_temp == 1:
            h_total = h_total[0]

        return h_total

    def getDewPoint(
        self,
        pres=None,
        mass_frac=None,
        mole_frac=None,
        thermo_method='ideal',
        x_liq=False,
    ):
        """Solve for the vapor dew-point temperature [K].

        Parameters
        ----------
        pres : float, optional
            System pressure [Pa]; defaults to ``self.pres`` when ``None``.
        mass_frac, mole_frac : numpy.ndarray, optional
            Vapor species fractions [-], shape ``(num_species,)``. Defaults to
            ``self.mole_frac`` when neither is supplied.
        thermo_method : {'ideal', 'UNIFAC', 'UNIQUAC'}, optional
            Liquid activity-coefficient model; default ``'ideal'``.
        x_liq : bool, optional
            If ``True``, return ``(temp_dew, x_liq)`` where ``x_liq`` is the
            equilibrium liquid mole fraction array [-]. Default ``False``.

        Returns
        -------
        float or tuple of (float, numpy.ndarray)
            Dew-point temperature [K], or ``(temp_dew, x_liq)`` when
            ``x_liq=True``.

        Raises
        ------
        RuntimeError
            If the Newton dew-point solver fails to converge.
        """
        if mass_frac is None and mole_frac is None:
            mole_frac = self.mole_frac  # [-]
        elif mole_frac is None:
            mole_frac = self.frac_to_frac(mass_frac=mass_frac)  # [-]
        if pres is None:
            pres = self.pres  # [Pa]

        def dew_fn(temp):
            k_vals = self._vle_ratios(
                temp, pres, mole_frac, thermo_method, check=False
            )  # [-]
            return np.dot(mole_frac, 1 / k_vals) - 1

        temp_seed = self._antoine_seed(mole_frac, pres=pres)  # [K]
        temp_dew = newton(dew_fn, temp_seed, full_output=False)
        k_vals = self.getKeqVLE(
            temp_dew, pres, mole_frac, gamma_model=thermo_method
        )  # [-]
        if x_liq:
            return temp_dew, mole_frac / k_vals
        return temp_dew

    def _get_vapor_density(
        self,
        pres_gas=None,
        temp_gas=None,
        phase='gas',
        basis='mass',
        *,
        mass_frac=None,
        mole_frac=None,
        temp=None,
        pres=None,
    ):
        """Return ideal-gas density on the requested basis.

        Parameters
        ----------
        pres_gas, pres : float or numpy.ndarray, optional
            Gas pressure [Pa]; mutually exclusive. Defaults to ``self.pres``.
        temp_gas, temp : float or numpy.ndarray, optional
            Gas temperature [K]; mutually exclusive. Defaults to ``self.temp``.
        phase : str, optional
            Legacy phase selector; accepted for signature compatibility.
        basis : {'mass', 'mole'}, optional
            Physical basis of the returned density; default ``'mass'``.
        mass_frac, mole_frac : numpy.ndarray, optional
            Species fractions [-], shape ``(num_species,)`` or
            ``(num_points, num_species)``. ``mole_frac`` takes precedence when
            both are supplied; defaults to ``self.mole_frac``.

        Returns
        -------
        float or numpy.ndarray
            Ideal-gas density, [kg/m**3] for ``basis='mass'`` or [mol/L]
            (equivalently [kmol/m**3]) for ``basis='mole'``.

        Raises
        ------
        ValueError
            If ``basis`` is invalid or both ``pres_gas`` and ``pres`` (or both
            ``temp_gas`` and ``temp``) are supplied.
        """
        if basis not in ('mass', 'mole'):
            raise ValueError("basis must be 'mass' or 'mole'")
        if pres_gas is not None and pres is not None:
            raise ValueError("Specify either 'pres_gas' or 'pres', not both")
        if temp_gas is not None and temp is not None:
            raise ValueError("Specify either 'temp_gas' or 'temp', not both")
        if temp is None:
            temp = self.temp if temp_gas is None else temp_gas  # [K]
        if pres is None:
            pres = self.pres if pres_gas is None else pres_gas  # [Pa]
        molar_density = pres / (VAPOR_GAS_CONSTANT * temp)  # [mol/m**3]
        if basis == 'mole':
            composition = mole_frac if mole_frac is not None else mass_frac  # [-]
            if composition is None:
                composition = self.mole_frac  # [-]
            if np.ndim(composition) > 1:
                return np.broadcast_to(
                    molar_density / LITERS_PER_M3, (len(composition),)
                )  # [mol/L]
            return molar_density / LITERS_PER_M3  # [mol/L]
        if mole_frac is None:
            if mass_frac is None:
                mole_frac = self.mole_frac  # [-]
            else:
                mole_frac = self.frac_to_frac(mass_frac=mass_frac)  # [-]
        mw_av = np.dot(mole_frac, self.mw)  # [g/mol]
        return molar_density * mw_av / GRAMS_PER_KG  # [kg/m**3]

    @overridable
    def getEnthalpy(self, *args, **kwargs):
        """Calculate phase enthalpy on the requested mixture or species basis.

        Parameters
        ----------
        *args, **kwargs
            Forwarded to :meth:`_get_vapor_enthalpy` for vapor phases or
            ``ThermoPhysicalManager.getEnthalpy`` for liquid and solid phases.
            ``temp`` [K] defaults to ``self.temp``, ``temp_ref`` [K] defaults
            to ``self.temp_ref`` (298.15 K unless set on construction), and
            ``mass_frac`` / ``mole_frac`` [-] default to
            :meth:`_stored_fractions` when neither is supplied (or both are
            ``None``).

        Returns
        -------
        float or numpy.ndarray
            Phase enthalpy, [J/kg] for ``basis='mass'`` and [J/mol] for
            ``basis='mole'``.

        Raises
        ------
        ValueError
            If ``basis`` is invalid or fraction shapes are incompatible.
        MissingPropertyError
            If required heat-capacity coefficients are missing.
        """
        target_phase = kwargs.get("phase") or (
            args[4]
            if len(args) > 4 and isinstance(args[4], str)
            else self.phase_family
        )
        if self.phase_family == "vapor" and target_phase == "vapor":
            return self._get_vapor_enthalpy(*args, **kwargs)

        args_list = list(args)
        pos_mass_frac = args_list[2] if len(args_list) > 2 else None  # [-]
        pos_mole_frac = args_list[3] if len(args_list) > 3 else None  # [-]
        kw_mass_frac = kwargs.get("mass_frac")  # [-]
        kw_mole_frac = kwargs.get("mole_frac")  # [-]

        if (
            pos_mass_frac is None
            and pos_mole_frac is None
            and kw_mass_frac is None
            and kw_mole_frac is None
        ):
            stored_mass_frac, stored_mole_frac = self._stored_fractions()  # [-], [-]
            if len(args_list) > 2:
                args_list[2] = stored_mass_frac
                if len(args_list) > 3 and stored_mole_frac is not None:
                    args_list[3] = stored_mole_frac
                elif stored_mole_frac is not None and "mole_frac" not in kwargs:
                    kwargs["mole_frac"] = stored_mole_frac
            else:
                kwargs["mass_frac"] = stored_mass_frac
                if stored_mole_frac is not None:
                    kwargs["mole_frac"] = stored_mole_frac

        if len(args_list) < 1:
            if kwargs.get("temp") is None:
                kwargs["temp"] = self.temp
        elif args_list[0] is None:
            args_list[0] = self.temp

        default_temp_ref = getattr(self, "temp_ref", 298.15)  # [K]
        if len(args_list) < 2:
            if kwargs.get("temp_ref") is None:
                kwargs["temp_ref"] = default_temp_ref
        elif args_list[1] is None:
            args_list[1] = default_temp_ref

        if len(args_list) < 5:
            if kwargs.get("phase") is None:
                kwargs["phase"] = self.phase_family
        elif args_list[4] is None:
            args_list[4] = self.phase_family

        return super().getEnthalpy(*args_list, **kwargs)

    @overridable
    def getCp(self, *args, **kwargs):
        """Return mixture heat capacity on the requested basis.

        Parameters
        ----------
        *args, **kwargs
            Forwarded to ``ThermoPhysicalManager.getCpMix``. ``temp`` [K]
            defaults to ``self.temp``, ``phase`` defaults to
            ``self.phase_family``, and ``mass_frac`` / ``mole_frac`` [-]
            default to :meth:`_stored_fractions` when neither is supplied (or
            both are ``None``).

        Returns
        -------
        float or numpy.ndarray
            Mixture heat capacity, [J/kg/K] for ``basis='mass'`` and [J/mol/K]
            for ``basis='mole'``.

        Raises
        ------
        ValueError
            If ``basis`` is invalid.
        MissingPropertyError
            If required heat-capacity coefficients are missing.
        """
        args_list = list(args)
        pos_mass_frac = args_list[1] if len(args_list) > 1 else None  # [-]
        pos_mole_frac = args_list[2] if len(args_list) > 2 else None  # [-]
        kw_mass_frac = kwargs.get("mass_frac")  # [-]
        kw_mole_frac = kwargs.get("mole_frac")  # [-]

        if (
            pos_mass_frac is None
            and pos_mole_frac is None
            and kw_mass_frac is None
            and kw_mole_frac is None
        ):
            stored_mass_frac, stored_mole_frac = self._stored_fractions()  # [-], [-]
            if len(args_list) > 1:
                args_list[1] = stored_mass_frac
                if len(args_list) > 2 and stored_mole_frac is not None:
                    args_list[2] = stored_mole_frac
                elif stored_mole_frac is not None and "mole_frac" not in kwargs:
                    kwargs["mole_frac"] = stored_mole_frac
            else:
                kwargs["mass_frac"] = stored_mass_frac
                if stored_mole_frac is not None:
                    kwargs["mole_frac"] = stored_mole_frac

        if len(args_list) < 1:
            if kwargs.get("temp") is None:
                kwargs["temp"] = self.temp
        elif args_list[0] is None:
            args_list[0] = self.temp

        if len(args_list) < 4:
            if kwargs.get("phase") is None:
                kwargs["phase"] = self.phase_family
        elif args_list[3] is None:
            args_list[3] = self.phase_family

        return super().getCpMix(*args_list, **kwargs)

    @overridable_property
    def density(self):
        return self.getDensity()

    @overridable
    def getDensity(self, *args, **kwargs):
        """Ideal-mixing or ideal-gas density at the phase's or requested composition.

        Parameters
        ----------
        *args, **kwargs
            Passed to ``ThermoPhysicalManager.getDensityMix`` (for liquid and
            solid phases) or :meth:`_get_vapor_density` (for vapor phases).
            ``mass_frac`` or ``mole_frac`` [-], shape ``(num_species,)`` or
            ``(num_points, num_species)``, selects the composition; with
            neither (or both ``None``), the phase's own fractions are used.
            ``phase`` defaults to the phase family.

        Returns
        -------
        float or numpy.ndarray
            Density [kg/m**3], or [kmol/m**3] ([mol/L]) with ``basis='mole'``,
            one value per composition row.

        Raises
        ------
        ValueError
            If ``basis`` is invalid or mutually exclusive overrides are
            supplied.
        """
        target_phase = kwargs.get("phase") or (
            args[2]
            if len(args) > 2 and isinstance(args[2], str)
            else self.phase_family
        )
        if self.phase_family == "vapor" and target_phase in ("vapor", "gas"):
            return self._get_vapor_density(*args, **kwargs)

        args_list = list(args)
        pos_mass_frac = args_list[0] if len(args_list) > 0 else None  # [-]
        pos_mole_frac = args_list[1] if len(args_list) > 1 else None  # [-]
        kw_mass_frac = kwargs.get("mass_frac")  # [-]
        kw_mole_frac = kwargs.get("mole_frac")  # [-]

        if (
            pos_mass_frac is None
            and pos_mole_frac is None
            and kw_mass_frac is None
            and kw_mole_frac is None
        ):
            if len(args_list) > 0:
                args_list[0] = self.mass_frac
                if len(args_list) > 1:
                    args_list[1] = self.mole_frac
            else:
                kwargs["mass_frac"] = self.mass_frac
                if "mole_frac" in kwargs:
                    kwargs["mole_frac"] = self.mole_frac

        if len(args_list) < 3:
            if kwargs.get("phase") is None:
                kwargs["phase"] = self.phase_family
        elif args_list[2] is None:
            args_list[2] = self.phase_family

        if len(args_list) < 4:
            if kwargs.get("temp") is None:
                kwargs["temp"] = self.temp
        elif args_list[3] is None:
            args_list[3] = self.temp

        return super().getDensityMix(*args_list, **kwargs)


    def to_stream(self):
        """Convert this phase into its refactored stream counterpart.

        Returns
        -------
        BaseStream
            Refactored stream of the matching phase family carrying the same
            thermodynamic state and mass/molar/volumetric magnitudes as flow
            rates.
        """
        stream_cls = self.stream_class
        if stream_cls is None:
            from PharmaPy.Streams_Refactored import (
                LiquidStream,
                SolidStream,
                VaporStream,
            )
            stream_cls = {
                "liquid": LiquidStream,
                "vapor": VaporStream,
                "solid": SolidStream,
            }[self.phase_family]
        stream = copy.copy(self)
        stream.__class__ = stream_cls
        stream.is_stream = True
        stream._DynamicInlet = None
        stream.controllable = ('mass_flow', 'mole_flow', 'vol_flow', 'temp')
        stream.time_upstream = None
        stream.bipartite = None

        # Per-instance copies: these are class attributes on BasePhase, so
        # updating them in place leaked the stream-only names onto every
        # phase in the process.
        stream.composition_names = set(self.composition_names) | {'mass_j_flow'}
        stream.amount_names = set(self.amount_names) | {
            'mass_flow', 'mass_j_flow', 'vol_flow', 'mole_flow'}

        return stream

    # =================================================================
    # Conversion to the pre-refactor stack.
    #
    # The old unit operations import PharmaPy.Phases and type-check
    # against it by module string and by isinstance, so handing them a
    # refactored phase does not work no matter how closely it duck-types.
    # These build a genuine old-stack object instead.
    #
    # State that used to live on the phase now lives on a mechanism, so
    # each attached mechanism is asked to spell its own state the old way.
    # A mechanism that cannot is named in the error rather than dropped.
    # =================================================================

    LEGACY_CLASS_NAME = None

    # Only the old LiquidPhase takes name_solv; SolidPhase and VaporPhase
    # have no such argument and would reject it.
    LEGACY_ACCEPTS_NAME_SOLV = False

    def _legacy_class(self):
        import PharmaPy.Phases as legacy

        if self.LEGACY_CLASS_NAME is None:
            raise TypeError(
                f"{type(self).__name__} has no pre-refactor equivalent")

        return getattr(legacy, self.LEGACY_CLASS_NAME)

    def _legacy_mechanism_state(self):
        """Merge what every attached mechanism contributes, or explain."""
        state = {}

        for mechanism in self.mechanisms:
            contribution = mechanism.to_legacy_state()

            if contribution is None:
                raise NotImplementedError(
                    f"{type(mechanism).__name__} holds state for this "
                    f"{type(self).__name__} but does not define "
                    "to_legacy_state(), so it cannot be expressed in the "
                    "pre-refactor representation. Implement that method on "
                    "the mechanism, or keep this phase on the refactored "
                    "side of the flowsheet."
                )

            state.update(contribution)

        return state

    def to_legacy(self):
        """Build an equivalent phase from the pre-refactor ``PharmaPy.Phases``.

        Returns
        -------
        object
            Instance of the corresponding class in :mod:`PharmaPy.Phases`
            preserving temperature [K], pressure [Pa], reference temperature
            [K], mass [kg], composition [-], solvent, and attached mechanism
            state.

        Raises
        ------
        TypeError
            If this phase class has no pre-refactor counterpart.
        NotImplementedError
            If an attached mechanism cannot express its state in the
            pre-refactor representation.
        """
        legacy_cls = self._legacy_class()

        kwargs = dict(
            path_thermo=self.path_thermo,
            temp=self.temp,
            pres=self.pres,
            mass=float(self.mass) if self.mass is not None else 0,
            mass_frac=np.asarray(self.mass_frac).copy(),
        )

        if self.phase_family == 'solid':
            kwargs['temp_ref'] = getattr(self, 'temp_ref', 298.15)  # [K]

        if self.LEGACY_ACCEPTS_NAME_SOLV and self.name_solv is not None:
            kwargs['name_solv'] = self.name_solv

        mechanism_state = self._legacy_mechanism_state()
        kwargs.update(mechanism_state)

        if 'distrib' in mechanism_state:
            # The old SolidPhase reinterprets a supplied distribution
            # against a supplied mass (getDistribution treats it as a
            # volume-percent curve and rescales). Given mass=0 it takes the
            # array verbatim as a number density and derives the mass from
            # the third moment instead, which is what we want: the
            # distribution is the source of truth on both sides.
            kwargs['mass'] = 0

        legacy_phase = legacy_cls(**kwargs)
        if hasattr(self, 'temp_ref') and not hasattr(legacy_phase, 'temp_ref'):
            legacy_phase.temp_ref = self.temp_ref  # [K]
        return legacy_phase

    @staticmethod
    def from_legacy(legacy_phase, **mechanism_kwargs):
        """Build a refactored phase from a pre-refactor one.

        The inverse of to_legacy, and not symmetric with it: the old stack
        kept particle state on the phase, so that state has to be moved into
        a mechanism here. Which mechanism is decided by asking each
        candidate whether it claims the phase -- a distribution goes to the
        finite-volume mechanism, bare moments to the method-of-moments one.

        A mechanism also needs things the old phase never carried, such as
        which species crystallizes; those come in as mechanism_kwargs from
        whoever is driving the conversion, and their absence is an error
        rather than a silently mechanism-less phase.

        Parameters
        ----------
        legacy_phase : object
            Phase or stream instance from :mod:`PharmaPy.Phases` or
            :mod:`PharmaPy.Streams`.
        **mechanism_kwargs
            Additional keyword arguments forwarded to
            ``candidate.from_legacy_phase``.

        Returns
        -------
        BasePhase
            Refactored phase preserving temperature [K], pressure [Pa],
            reference temperature [K], composition [-], mass [kg], and
            converted mechanism state.

        Raises
        ------
        NotImplementedError
            If ``legacy_phase`` has no refactored counterpart.
        """
        import PharmaPy.Mechanisms as mech

        name = type(legacy_phase).__name__
        target_cls = {
            'LiquidPhase': LiquidPhase,
            'SolidPhase': SolidPhase,
            'VaporPhase': VaporPhase,
            'LiquidStream': LiquidPhase,
            'SolidStream': SolidPhase,
            'VaporStream': VaporPhase,
        }.get(name)

        if target_cls is None:
            raise NotImplementedError(
                f"No refactored equivalent of {name} is known")

        kwargs = dict(
            path_thermo=getattr(legacy_phase, 'path_data', None),
            temp=legacy_phase.temp,
            pressure=getattr(legacy_phase, 'pres', 101325),
            temp_ref=getattr(legacy_phase, 'temp_ref', 298.15),
            mass_frac=np.asarray(legacy_phase.mass_frac).copy(),
        )

        ind_solv = getattr(legacy_phase, 'ind_solv', None)

        if ind_solv is not None and target_cls is LiquidPhase:
            kwargs['name_solv'] = legacy_phase.name_species[ind_solv]

        phase = target_cls(**kwargs)

        # Composition first, then amount: mass cannot be set before
        # mass_frac exists, which the constructor has just taken care of.
        mass = getattr(legacy_phase, 'mass', None)

        if mass:
            phase.mass = float(mass)

        candidates = (mech.OneDFVMMechanism, mech.MomentsPopulationBalance)

        for candidate in candidates:

            if not candidate.claims_legacy_phase(legacy_phase):
                continue

            built = candidate.from_legacy_phase(legacy_phase, phase,
                                                **mechanism_kwargs)

            if built is None:
                raise NotImplementedError(
                    f"{candidate.__name__} claims this {name} but could not "
                    "rebuild itself from it"
                )

            phase.mechanisms = built
            break

        return phase

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
        if updates:
            self.updatePhase(**updates)
        for mech in self.mechanisms:
            mech.update_state(completed_state,unit=unit)

class LiquidPhase(BasePhase):
    LEGACY_CLASS_NAME = 'LiquidPhase'
    LEGACY_ACCEPTS_NAME_SOLV = True
    phase_family = "liquid"
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
        from PharmaPy.Streams_Refactored import LiquidStream
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
    LEGACY_CLASS_NAME = 'VaporPhase'
    phase_family = 'vapor'
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
        from PharmaPy.Streams_Refactored import VaporStream
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
    LEGACY_CLASS_NAME = 'SolidPhase'
    phase_family = 'solid'
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
        from PharmaPy.Streams_Refactored import SolidStream
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
        """Estimate packed-bed porosity with the legacy linear packing model.

        Parameters
        ----------
        distrib : array-like, optional
            Number distribution [#/um], shape ``(num_sizes,)``. Defaults to
            ``self.distrib``.
        diam_filter : float, optional
            Filter diameter [m]; default 1 m.
        AR : float, optional
            Legacy unused aspect ratio [-].
        sphericity : float, optional
            Particle sphericity [-]; default 0.7.

        Returns
        -------
        float
            Estimated packed-bed void fraction [-].

        .. deprecated::
            Particle size distributions belong to the crystallization
            mechanism in the refactored architecture.
        """
        warnings.warn(
            "getPorosity on refactored SolidPhase is deprecated; particle "
            "size distributions belong to the crystallization mechanism.",
            DeprecationWarning,
            stacklevel=2,
        )

        if distrib is None:
            distrib = self.distrib
            mom_zero = self.moments[0]
            mom_one = self.moments[1]
        elif hasattr(self, "getMoments"):
            mom_zero, mom_one = self.getMoments(mom_num=(0, 1))
        else:
            mom_zero = trapezoidal_rule(self.x_distrib, distrib)  # [#]
            mom_one = (
                trapezoidal_rule(self.x_distrib, distrib * self.x_distrib)
                * 1e-6
            )  # [m], exact um-to-m conversion

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