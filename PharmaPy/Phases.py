# -*- coding: utf-8 -*-


# import numpy as np
# from autograd import numpy as np
from typing import Optional, Sequence, Union

import numpy as np
from numpy.typing import ArrayLike
from PharmaPy.ThermoModule import ThermoPhysicalManager
from PharmaPy.Commons import trapezoidal_rule
from scipy.optimize import newton

import warnings

eps = np.finfo(float).eps

# CODATA 2018: R = 8.314462618 J/(mol K), rounded to four significant figures
# to match the existing gas_ct = 8.314 in Kinetics, Crystallizers, Reactors,
# Drying_Model, Evaporators, and ThermoModule, so vapor amounts agree with
# the evaporator's own P V / (R T).
VAPOR_GAS_CONSTANT = 8.314  # [J/mol/K]


def _as_float_array(values: ArrayLike) -> np.ndarray:
    """Convert numeric array-like input to a float array without reshaping.

    Parameters
    ----------
    values : array-like
        Composition, distribution, moment, or temperature values. Shape and
        physical units are defined by the calling boundary and retained.

    Returns
    -------
    numpy.ndarray
        Float array with the input shape, values, units, and physical basis.
        An existing float array may be shared with the caller.

    Raises
    ------
    TypeError
        If ``values`` is None; supply numeric array-like input instead.
    """
    if values is None:
        raise TypeError("Provide numeric array-like input, not None")
    return np.asarray(values, dtype=float)


def classify_phases(instance: object, names: Optional[Sequence[str]] = None) -> None:
    """Name phases and install them as attributes of their container.

    Parameters
    ----------
    instance : object
        Container exposing its phase objects through ``Phases``.
    names : sequence of str, optional
        Explicit names paired with phases in order. If omitted, names are
        generated from the Liquid, Solid, or Vapor class and a per-type count.

    Notes
    -----
    Both each phase's ``name`` and the corresponding container attribute are
    assigned. Explicit names retain the existing ``zip`` pairing behavior.
    """
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
                phase_name = 'Vapor_{}'.format(vapor_count)
                vapor_count += 1

            setattr(phase, 'name', phase_name)
            setattr(instance, phase_name, phase)
    else:
        for phase, name in zip(phases, names):
            setattr(phase, 'name', name)
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


class LiquidPhase(ThermoPhysicalManager):
    """ Creates a LiquidPhase object.
    
    Parameters
    ----------
    mass_frac : array-like (optional)
        mass fractions of the constituents of the phase.
    mole_conc : array-like (optional)
        molar concentrations of the constituents of the phase, excluding
        the solvent
    ind_solv : int
        index of solvent components in the liquid phase. It must be
        only specified if 'mass_frac' or 'mole_frac' are not given.
    """
    def __init__(self, path_thermo=None, temp: float = 298.15, pres=101325,
                 mass=0, vol=0, moles=0,
                 mass_frac: Optional[ArrayLike] = None,
                 mole_frac: Optional[ArrayLike] = None,
                 mass_conc: Optional[ArrayLike] = None,
                 mole_conc: Optional[ArrayLike] = None,
                 name_solv=None, verbose=True, check_input=True) -> None:

        """Initialize liquid composition and reconcile its supplied amount.

        Parameters
        ----------
        path_thermo : str, optional
            Path to the species thermophysical-property JSON file.
        temp : float, optional
            Scalar temperature only [K], stored as a Python float; default
            298.15 K. Temperature profiles are accepted only by updatePhase.
        pres : float or array-like, optional
            Pressure [Pa], retained without coercion; default 101325 Pa.
        mass, vol, moles : float, optional
            Mass [kg], volume [m**3], and amount [mol]. The first positive
            value in that order controls; zero means no amount was supplied.
        mass_frac, mole_frac : array-like, optional
            Species mass or mole fractions [-], shape ``(num_species,)`` or
            ``(num_points, num_species)``. Stored as float arrays.
        mass_conc, mole_conc : array-like, optional
            Species mass [kg/m**3] or molar [mol/L] concentrations with the
            same shapes as fractions, stored as float arrays. Solvent
            completion supports only shape ``(num_species,)``.
        name_solv : str, optional
            Solvent species whose concentration is completed from the volume
            balance when concentrations are supplied; None normalizes directly.
        verbose : bool, optional
            Print diagnostics for fractions summing below the legacy 0.99
            threshold [-]. Defaults to True.
        check_input : bool, optional
            Warn if all amounts are zero. Defaults to True.

        Raises
        ------
        ValueError
            If no composition measure is supplied.
        RuntimeWarning
            If more than one composition measure is supplied.

        Notes
        -----
        Exactly one composition measure is required. Constructor composition
        inputs are copied before conversion. Derived concentrations retain
        the liquid volume basis. No physical basis changes during coercion.
        """

        super().__init__(path_thermo)

        self.cp_liq = np.atleast_2d(self.cp_liq)
        self.p_vap = np.atleast_2d(self.p_vap)

        if name_solv is None:
            ind_solv = name_solv
        else:
            ind_solv = self.name_species.index(name_solv)

        self.ind_solv = ind_solv

        self.temp = float(temp)
        self.pres = pres

        self.mass = mass
        self.vol = vol
        self.moles = moles

        unspec_num = (mass_frac is None) + (mass_conc is None) + \
            (mole_conc is None) + (mole_frac is None)

        if unspec_num == 4:
            raise ValueError("No measure of composition was provided")
        elif unspec_num < 3:
            raise RuntimeWarning("More than one measure of composition was "
                                 "provided")

        # Copy constructor compositions: conc_to_frac and mass_conc_to_frac
        # complete the solvent entry in place, so caller arrays must be isolated.
        if mass_frac is not None:
            self.mass_frac = np.array(mass_frac, dtype=float)  # [-]
            self.mass_conc = mass_conc

            self.mole_frac = mole_frac
            self.mole_conc = mole_conc

            if self.mass_frac.ndim == 1:
                sum_fracs = sum(self.mass_frac)
                less_than_one = sum_fracs < 0.99
            else:
                sum_fracs = self.mass_frac.sum(axis=1)
                less_than_one = any(sum_fracs < 0.99)

            if less_than_one:
                if verbose:
                    print()
                    print('PharmaPy Warning: '
                          'The sum of mass fractions is less than 0.99 '
                          '(sum(mass_frac) = %.4f) for %s object'
                          % (sum_fracs, self.__class__.__name__))
                    print()

            self.__calcComposition()

        elif mole_frac is not None:
            self.mass_frac = mass_frac
            self.mass_conc = mass_conc

            self.mole_frac = np.array(mole_frac, dtype=float)  # [-]
            self.mole_conc = mole_conc

            if self.mole_frac.ndim == 1:
                sum_fracs = sum(self.mole_frac)
                less_than_one = sum_fracs < 0.99
            else:
                sum_fracs = self.mole_frac.sum(axis=1)
                less_than_one = any(sum_fracs < 0.99)

            if less_than_one:
                if verbose:
                    print()
                    print('PharmaPy Warning: '
                          'The sum of mass fractions is less than 0.99 '
                          '(sum(mass_frac) = %.4f) for %s object'
                          % (sum_fracs, self.__class__.__name__))
                    print()

            self.__calcComposition()

        elif mass_conc is not None:
            self.mass_conc = np.array(mass_conc, dtype=float)  # [kg/m**3]
            self.mole_frac = mole_frac

            self.mass_frac = mass_frac
            self.mole_conc = mole_conc

            self.__calcComposition()

        elif mole_conc is not None:
            self.mole_conc = np.array(mole_conc, dtype=float)  # [mol/L]
            self.mole_frac = mole_frac

            self.mass_frac = mass_frac
            self.mass_conc = mass_conc

            self.__calcComposition()

        if (mass + vol + moles) == 0:
            if check_input:
                warnings.simplefilter("always")
                warnings.warn("'mass', 'moles' and 'vol' are all set to zero. "
                              "Model may not perform as intended.",
                              RuntimeWarning)

                warnings.simplefilter("ignore")

        self.y_upstream = None

        self._name = None
        self.transferred_from_uo = False

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        self._name = name

    def __set_amounts(self, mass, vol, moles, massfrac, molefrac,
                      conc, mass_conc):
        densMass = self.getDensityMix(massfrac)
        mw_av = np.dot(molefrac, self.mw)
        if mass > 0:
            self.mass = mass
            self.vol = mass / densMass
            self.moles = mass / mw_av * 1000

        elif vol > 0:
            self.vol = vol
            self.mass = vol * densMass
            self.moles = self.mass / mw_av * 1000

        elif moles > 0:
            self.moles = moles
            self.mass = moles * mw_av / 1000  # kg
            self.vol = self.mass / densMass

        self.mass_frac = massfrac
        self.mole_frac = molefrac
        self.mole_conc = conc
        self.mass_conc = mass_conc

        self.mw_av = mw_av

    def __calcComposition(self):

        if self.mole_conc is not None:
            frac_out = self.conc_to_frac(self.mole_conc,
                                         solvent_ind=self.ind_solv)
            if self.ind_solv is None:
                mass_frac, mole_frac = frac_out
                mole_conc = self.mole_conc
            else:
                mass_frac, mole_frac, mole_conc = frac_out

            mass_conc = mole_conc * self.mw  # kg / m3

        elif self.mass_conc is not None:
            frac_out = self.mass_conc_to_frac(self.mass_conc,
                                              solvent_ind=self.ind_solv)

            if self.ind_solv is None:
                mass_frac, mole_frac = frac_out
                mass_conc = self.mass_conc
            else:
                mass_frac, mole_frac, mass_conc = frac_out

            mole_conc = mass_conc / self.mw  # mol/L

        elif self.mass_frac is not None:
            mole_conc = self.frac_to_conc(self.mass_frac)
            mass_conc = mole_conc * self.mw

            mole_frac = self.frac_to_frac(self.mass_frac)
            mass_frac = self.mass_frac

        elif self.mole_frac is not None:
            mole_conc = self.frac_to_conc(mole_frac=self.mole_frac)
            mass_conc = mole_conc * self.mw

            mass_frac = self.frac_to_frac(mole_frac=self.mole_frac)
            mole_frac = self.mole_frac

        self.__set_amounts(self.mass, self.vol, self.moles,
                           mass_frac, mole_frac, mole_conc, mass_conc)

    def updatePhase(self, mole_conc: Optional[ArrayLike] = None,
                    mass_conc: Optional[ArrayLike] = None,
                    mass_frac: Optional[ArrayLike] = None,
                    mole_frac: Optional[ArrayLike] = None,
                    vol=0, mass=0, moles=0,
                    temp: Optional[ArrayLike] = None, pres=None) -> None:
        """Update the liquid composition, amount, and intensive state.

        Parameters
        ----------
        mole_conc : array-like, optional
            Species molar concentrations with shape ``(num_species,)``
            [mol/L]. When the phase declares a solvent, the solvent entry is
            back-calculated from the mixture volume balance and any value
            supplied at that index is ignored.
        mass_conc : array-like, optional
            Species mass concentrations with shape ``(num_species,)``
            [kg/m**3], handled like ``mole_conc`` but on a mass basis.
        mass_frac : array-like, optional
            Species mass fractions with shape ``(num_species,)`` [-].
        mole_frac : array-like, optional
            Species mole fractions with shape ``(num_species,)`` [-].
        vol : float, optional
            Liquid volume [m**3].
        mass : float, optional
            Liquid mass [kg].
        moles : float, optional
            Amount of liquid [mol].
        temp : float or array-like, optional
            Liquid temperature [K], stored as a Python float for scalar input.
            A spatial profile of shape ``(num_points,)`` is stored as a float
            array without changing its shape.
            The stored temperature is left unchanged when ``None``.
        pres : float, optional
            Liquid pressure [Pa]. The stored pressure is left unchanged when
            ``None``.

        Returns
        -------
        None
            The phase state is updated in place.

        Notes
        -----
        Supplied float arrays may be stored by reference without copying, so
        callers must not rely on isolation from later mutations.
        Supplied compositions are stored as float arrays with unchanged shape.
        Composition arguments are resolved in the order ``mole_conc``,
        ``mass_conc``, ``mass_frac``, ``mole_frac``; the first one supplied
        determines the new composition and the remaining ones are ignored.
        When none is supplied, the stored composition is retained and only
        the amount and intensive state are refreshed.

        The amount is set from the first positive value among ``mass``,
        ``vol``, and ``moles``, using the mixture mass density [kg/m**3] and
        the average molar mass [g/mol]. Without a positive explicit amount,
        composition or intensive-state changes conserve the stored mass [kg],
        the liquid inventory integrated by the mass balances, and recompute
        volume [m**3] and moles [mol] from the updated mixture properties.
        This retained-mass reconciliation applies only when the resolved
        composition is one-dimensional and stored mass is scalar. For
        two-dimensional compositions shaped ``(num_points, num_species)``
        or array-valued stored mass, profile inventories are not reconciled:
        mass, volume, and moles retain their values and scalar-versus-array
        behavior when no positive explicit amount is supplied.
        A no-argument update leaves all state unchanged. Zero amounts mean
        "not supplied" and cannot empty the phase through this method.

        Solvent handling matches the constructor: ``self.ind_solv`` is
        compared with ``None`` rather than tested for truth, so a solvent
        declared as the first species (index ``0``) is honored and the
        concentration converters' three-value return is unpacked correctly.
        """

        explicit_amount = any(amount > 0 for amount in (mass, vol, moles))
        if not explicit_amount:
            if all(value is None for value in
                   (mole_conc, mass_conc, mass_frac, mole_frac, temp, pres)):
                return

        if mole_conc is not None:
            mole_conc = _as_float_array(mole_conc)  # [mol/L]
            frac_out = self.conc_to_frac(mole_conc,
                                         solvent_ind=self.ind_solv)
            if self.ind_solv is not None:
                mass_frac, mole_frac, mole_conc = frac_out
            else:
                mass_frac, mole_frac = frac_out

            mass_conc = mole_conc * self.mw

        elif mass_conc is not None:
            mass_conc = _as_float_array(mass_conc)  # [kg/m**3]
            frac_out = self.mass_conc_to_frac(mass_conc,
                                              solvent_ind=self.ind_solv)

            if self.ind_solv is not None:
                mass_frac, mole_frac, mass_conc = frac_out
            else:
                mass_frac, mole_frac = frac_out

            mole_conc = mass_conc / self.mw

        elif mass_frac is not None:
            mass_frac = _as_float_array(mass_frac)  # [-]
            mole_conc = self.frac_to_conc(mass_frac)
            mass_conc = mole_conc * self.mw
            mole_frac = self.frac_to_frac(mass_frac)

        elif mole_frac is not None:
            mole_frac = _as_float_array(mole_frac)  # [-]
            mole_conc = self.frac_to_conc(mole_frac=mole_frac)
            mass_conc = mole_conc * self.mw
            mass_frac = self.frac_to_frac(mole_frac=mole_frac)

        else:
            mass_frac = self.mass_frac
            mole_frac = self.mole_frac
            mole_conc = self.mole_conc
            mass_conc = self.mass_conc

        if (not explicit_amount and np.ndim(mass_frac) == 1
                and np.ndim(self.mass) == 0):
            mass = self.mass  # [kg], authoritative retained liquid inventory

        if temp is not None:
            self.temp = (float(temp) if np.ndim(temp) == 0
                         else _as_float_array(temp))  # [K]

        if pres is not None:
            self.pres = pres

        self.__set_amounts(mass, vol, moles, mass_frac, mole_frac,
                           mole_conc, mass_conc)

    def getDensity(self, mass_frac=None, mole_frac=None, temp=None,
                   basis='mass'):

        if temp is None:
            temp = self.temp

        if mass_frac is None and mole_frac is None:
            mass_frac = self.mass_frac
            mole_frac = self.mole_frac

        rhoLiq = self.getDensityMix(mass_frac, mole_frac, phase='liquid',
                                    basis=basis, temp=temp)

        return rhoLiq

    def getCp(self, temp=None, mass_frac=None, mole_frac=None, basis='mole'):
        if temp is None:
            temp = self.temp

        if mass_frac is None and mole_frac is None:
            mass_frac = self.mass_frac
            mole_frac = self.mole_frac

        cpLiq = super().getCpMix(temp, mass_frac, mole_frac, basis=basis)

        return cpLiq

    def getEnthalpy(self, temp=None, temp_ref=298.15, mass_frac=None,
                    mole_frac=None, total_h=True, basis='mass'):

        if mass_frac is None and mole_frac is None:
            mass_frac = self.mass_frac
            mole_frac = self.mole_frac

        if temp is None:
            temp = self.temp

        hLiq = super().getEnthalpy(temp, temp_ref, mass_frac, mole_frac,
                                   phase='liquid', total_h=total_h,
                                   basis=basis)

        return hLiq

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


class VaporPhase(ThermoPhysicalManager):
    """Thermodynamic state of a homogeneous vapor mixture.

    Exactly one composition measure must define the mixture. Material amounts
    use mass, volume, or molar bases and are kept mutually consistent with the
    specified composition, temperature, and pressure using the ideal-gas
    equation of state. Density uses the same ideal-gas assumption.
    """

    def __init__(self, path_thermo: Optional[str] = None,
                 temp: float = 298.15, pres: float = 101325,
                 mass: float = 0, vol: float = 0, moles: float = 0,
                 mass_frac: Optional[ArrayLike] = None,
                 mole_frac: Optional[ArrayLike] = None,
                 mole_conc: Optional[ArrayLike] = None,
                 check_input: bool = True, verbose: bool = True) -> None:
        """Initialize a vapor-phase thermodynamic state.

        Parameters
        ----------
        path_thermo : str, optional
            Path to the species thermophysical-property JSON file.
        temp : float, optional
            Vapor temperature [K]; default is standard ambient temperature.
        pres : float, optional
            Vapor pressure [Pa]; default is one standard atmosphere.
        mass : float, optional
            Total vapor mass [kg].
        vol : float, optional
            Total vapor volume [m**3].
        moles : float, optional
            Total amount of vapor [mol].
        mass_frac : array-like, optional
            Species mass fractions with shape ``(num_species,)`` [-].
        mole_frac : array-like, optional
            Species mole fractions with shape ``(num_species,)`` [-].
        mole_conc : array-like, optional
            Species molar concentrations with shape ``(num_species,)``
            [mol/L].
        check_input : bool, optional
            If ``True``, warn when mass, volume, and moles are all zero [-].
        verbose : bool, optional
            If ``True``, print composition-normalization warnings [-].

        Raises
        ------
        ValueError
            If no composition measure is provided, an amount is negative, or
            positive mass or volume is requested with zero mixture molar mass.
        RuntimeWarning
            If more than one composition measure is provided.

        Warns
        -----
        RuntimeWarning
            If input checking is enabled and mass, volume, and moles are all
            zero.

        Notes
        -----
        Provide exactly one of ``mass_frac``, ``mole_frac``, or
        ``mole_conc``. Concentration inputs use PharmaPy's [mol/L] basis and
        are retained as supplied. Concentrations derived from fractions are
        the converters' liquid-basis values, not gas-EOS concentrations.
        The first positive amount in the order mass, volume, moles is
        authoritative; the other amounts follow from the ideal-gas EOS.
        If all amounts are zero, the stored amounts remain zero. A positive
        molar amount with zero mole fractions has zero mass and an EOS volume.
        """

        super().__init__(path_thermo)

        composition = {
            'mass_frac': mass_frac, 'mole_frac': mole_frac,
            'mole_conc': mole_conc,
        }  # mass/mole fractions [-]; mole_conc [mol/L]
        supplied = [name for name, value in composition.items()
                    if value is not None]
        if not supplied:
            raise ValueError("No measure of composition was provided")
        if len(supplied) > 1:
            raise RuntimeWarning("More than one measure of composition was "
                                 "provided")

        name = supplied[0]
        composition[name] = np.array(composition[name])  # [-] or [mol/L]
        if name in ('mass_frac', 'mole_frac'):
            fraction_sum = composition[name].sum(axis=-1)  # [-]
            # Preserve LiquidPhase's existing composition-warning threshold.
            warning_fraction_sum = 0.99  # [-], legacy normalization diagnostic
            if verbose and np.any(fraction_sum < warning_fraction_sum):
                print("PharmaPy Warning: The sum of fractions is less than "
                      "0.99 (sum = {}) for {} object".format(
                          fraction_sum, self.__class__.__name__))

        self.temp = float(temp)  # [K]
        self.pres = pres  # [Pa]
        self.mass = 0  # [kg]
        self.moles = 0  # [mol]
        self.vol = 0  # [m**3]
        # Use the phase implementation before a stream has its flow aliases.
        VaporPhase.updatePhase(self, mass=mass, vol=vol, moles=moles,
                               **composition)
        if mass == vol == moles == 0 and check_input:
            warnings.warn("'mass', 'moles' and 'vol' are all set to zero. "
                          "Model may not perform as intended.",
                          RuntimeWarning, stacklevel=2)

        self.y_upstream = None
        self._name = None

        self.transferred_from_uo = False

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        self._name = name

    def __set_amounts(self, mass: float, vol: float, moles: float,
                      massfrac: np.ndarray, molefrac: np.ndarray,
                      conc: np.ndarray, mass_conc: np.ndarray) -> None:
        """Store composition and reconcile positive amounts with the gas EOS.

        Parameters
        ----------
        mass, vol, moles : float
            Explicit amounts [kg], [m**3], and [mol], respectively. The first
            positive value in that order controls; zeros leave amounts alone.
        massfrac, molefrac : ndarray
            Species mass and mole fractions, shape ``(num_species,)`` [-].
        conc, mass_conc : ndarray
            Stored species concentrations, shape ``(num_species,)``, on molar
            [mol/L] and mass [kg/m**3] bases, respectively.

        Raises
        ------
        ValueError
            If positive mass or volume is requested with zero mixture molar mass.

        Notes
        -----
        Stream amount storage uses the corresponding per-second units.
        With no positive amount, no EOS or molecular-weight division occurs.
        Positive moles with zero composition give zero mass and an EOS volume.
        """
        mw_av = np.dot(molefrac, self.mw)  # [g/mol]
        if (mass > 0 or vol > 0) and mw_av == 0:
            raise ValueError("Positive mass or volume requires a nonzero "
                             "composition; zero mixture molar mass was given")

        self.mass_frac = massfrac  # [-]
        self.mole_frac = molefrac  # [-]
        self.mole_conc = conc  # [mol/L], retained converter basis
        self.mass_conc = mass_conc  # [kg/m**3], retained converter basis
        self.mw_av = mw_av  # [g/mol]

        if mass > 0 or vol > 0 or moles > 0:
            molar_volume = VAPOR_GAS_CONSTANT * self.temp / self.pres  # [m**3/mol]
            if mass > 0:
                self.moles = mass * 1000 / self.mw_av  # [mol], 1000 g/kg
            elif vol > 0:
                self.moles = vol / molar_volume  # [mol]
            else:
                self.moles = moles  # [mol]
            self.mass = self.moles * self.mw_av / 1000  # [kg], 1000 g/kg
            self.vol = self.moles * molar_volume  # [m**3]

    def updatePhase(self, mole_conc: Optional[ArrayLike] = None,
                    mass_conc: Optional[ArrayLike] = None,
                    mass_frac: Optional[ArrayLike] = None,
                    mole_frac: Optional[ArrayLike] = None,
                    vol: float = 0, mass: float = 0, moles: float = 0,
                    temp: Optional[float] = None,
                    pres: Optional[float] = None) -> None:
        """Update vapor composition, intensive state, and ideal-gas amounts.

        Parameters
        ----------
        mole_conc, mass_conc : array-like, optional
            Species concentrations, shape ``(num_species,)``, on molar
            [mol/L] and mass [kg/m**3] bases, respectively. No solvent is
            inferred. Supplied concentrations are retained on their basis.
        mass_frac, mole_frac : array-like, optional
            Species mass and mole fractions, shape ``(num_species,)`` [-].
        vol, mass, moles : float, optional
            Explicit volume [m**3], mass [kg], and amount [mol]. Zero means
            no amount was supplied. The first positive amount in the order
            mass, volume, moles controls the other two amounts.
        temp, pres : float, optional
            New temperature [K] and pressure [Pa]. ``None`` retains the
            corresponding stored value.

        Raises
        ------
        ValueError
            If an explicit amount is negative, or positive mass or volume is
            requested with zero mixture molar mass.

        Notes
        -----
        Supplied float arrays may be stored by reference without copying, so
        callers must not rely on isolation from later mutations.
        Supplied compositions are stored as float arrays with unchanged shape.
        Positive moles with zero composition give zero mass and an EOS volume.
        With no positive amount, a composition, temperature, or pressure
        update conserves stored moles and recomputes mass and gas volume.
        A no-argument update leaves all amounts unchanged. Composition inputs
        take precedence in the order mole_conc, mass_conc, mass_frac, mole_frac.
        Concentrations derived from fractions are the converters' liquid-basis
        values, not gas-EOS values. Stream amounts use per-second units.
        """
        for name, amount in (('mass', mass), ('vol', vol), ('moles', moles)):
            # amount uses [kg], [m**3], or [mol], respectively (rates on streams).
            if amount < 0:
                raise ValueError(f"{name} must be nonnegative")

        state_changed = any(value is not None for value in
                            (mole_conc, mass_conc, mass_frac, mole_frac,
                             temp, pres))
        if not state_changed and mass == vol == moles == 0:
            return
        if temp is not None:
            self.temp = float(temp)  # [K]
        if pres is not None:
            self.pres = pres  # [Pa]
        if mass == vol == moles == 0 and state_changed:
            moles = self.moles  # [mol], conserved inventory

        if mole_conc is not None:
            mole_conc = _as_float_array(mole_conc)  # [mol/L]
            mass_frac, mole_frac = self.conc_to_frac(mole_conc)  # [-]
            mass_conc = mole_conc * self.mw  # [kg/m**3], g/L equals kg/m**3
        elif mass_conc is not None:
            mass_conc = _as_float_array(mass_conc)  # [kg/m**3]
            mass_frac, mole_frac = self.mass_conc_to_frac(mass_conc)  # [-]
            mole_conc = mass_conc / self.mw  # [mol/L], kg/m**3 equals g/L
        elif mass_frac is not None:
            mass_frac = _as_float_array(mass_frac)  # [-]
            mole_conc = self.frac_to_conc(mass_frac)  # [mol/L]
            mass_conc = mole_conc * self.mw  # [kg/m**3], g/L equals kg/m**3
            mole_frac = self.frac_to_frac(mass_frac)  # [-]
        elif mole_frac is not None:
            mole_frac = _as_float_array(mole_frac)  # [-]
            if np.any(mole_frac):
                mole_conc = self.frac_to_conc(mole_frac=mole_frac)  # [mol/L]
                mass_frac = self.frac_to_frac(mole_frac=mole_frac)  # [-]
            else:
                # Empty evaporator placeholders have no normalized composition.
                mole_conc = np.zeros_like(mole_frac)  # [mol/L]
                mass_frac = np.zeros_like(mole_frac)  # [-]
            mass_conc = mole_conc * self.mw  # [kg/m**3], g/L equals kg/m**3
        else:
            mass_frac = self.mass_frac  # [-]
            mole_frac = self.mole_frac  # [-]
            mole_conc = self.mole_conc  # [mol/L]
            mass_conc = self.mass_conc  # [kg/m**3]

        self.__set_amounts(mass, vol, moles, mass_frac, mole_frac,
                           mole_conc, mass_conc)

    def getCp(self, temp, mass_frac=None, mole_frac=None, basis='mass'):
        if mass_frac is None and mole_frac is None:
            mass_frac = self.mass_frac

        cpMix = self.getCpMix(temp, mass_frac, mole_frac, phase='vapor',
                              basis=basis)

        return cpMix

    def getHeatVaporization(self, temp, basis='mass'):
        """Calculate the latent heat of vaporization of each species.

        Species that are supercritical at every requested temperature
        cannot condense. Their latent heat is reported as zero rather
        than omitted, so the component axis of the returned array stays
        aligned with the mass- or mole-fraction vectors that callers
        weight it with.

        Parameters
        ----------
        temp : float or array-like
            Temperature at which the latent heat is evaluated [K]. An
            array-like input is read as several independent
            temperatures, such as the spatial nodes of a distributed
            model, not as a per-species temperature.
        basis : {'mass', 'mole'}, optional
            Basis of the returned latent heat. The default is 'mass'.

        Returns
        -------
        ndarray
            Latent heat of vaporization per species, in [J/kg] for
            ``basis='mass'`` and [J/mol] for ``basis='mole'``. The shape
            is ``(num_species, )`` for a scalar or single-element
            ``temp`` and ``(num_temperatures, num_species)`` otherwise.
            Columns of species that are supercritical at every
            requested temperature are zero.

        Raises
        ------
        ValueError
            If the Watson temperature ratio is negative, either because a
            species is subcritical at one requested temperature and
            supercritical at another, or because the tabulated
            ``tref_hvap`` of a subcritical species lies above its
            ``t_crit``.

        Notes
        -----
        The latent heat is extrapolated from ``delta_hvap`` at
        ``tref_hvap`` with the Watson correlation,
        ``dh(T) = dh(Tref) * ((Tc - T) / (Tc - Tref))**0.38``.
        """

        temp = np.atleast_1d(temp)
        num_comp = len(self.t_crit)

        num_temp = len(temp)
        if num_temp > 1:
            temp = temp[..., np.newaxis]
            idx = np.unique(np.where(temp < self.t_crit)[1])
            delta_shape = (num_temp, num_comp)
        else:
            idx = np.where(temp < self.t_crit)[0]
            delta_shape = num_comp

        tref = self.tref_hvap[idx]

        watson = ((self.t_crit[idx] - temp) / (self.t_crit[idx] - tref))**0.38
        if np.isnan(watson.flatten()).any():
            raise ValueError("(self.t_crit[idx] - temp) / (self.t_crit[idx] - tref) was negative. Check property values")
        deltahvap = np.zeros(delta_shape)

        if num_temp > 1:
            deltahvap[:, idx] = (watson * self.delta_hvap[idx])  # [J/mol]
        else:
            deltahvap[idx] = (watson * self.delta_hvap[idx])  # [J/mol]

        if basis == 'mass':
            # Convert the populated subcritical entries in place so that
            # the species axis keeps its full width. Supercritical
            # species stay at zero latent heat instead of being dropped,
            # which would misalign the result with a fraction vector.
            if num_temp > 1:
                deltahvap[:, idx] = (deltahvap[:, idx] / self.mw[idx]
                                     * 1000)  # [J/kg]
            else:
                deltahvap[idx] = deltahvap[idx] / self.mw[idx] * 1000  # [J/kg]

        return deltahvap

    def getEnthalpy(self, temp=None, temp_ref=298.15, mass_frac=None,
                    mole_frac=None, total_h=True, basis='mass'):
        """ Calculate vapor phase enthalpy. It assumes that the reference state
        is a liquid at t_ref.

        Parameters
        ----------
        temp : float or array-like
            Temperature for enthalpy calculation in K.   
        temp_ref : float, optional
            Reference temperature for enthalpy calculation. The default is 298.15.
        mass_frac : array-like, optional
            Fraction of the species participating in the vapor phase in mass. The default is None.
        mole_frac : array-like, optional
            Fraction of the species participating in the vapor phase in mole. The default is None.
        total_h : bool, optional
            If True, the total enthalpy is returned. If False, an array
            of individual enthalpy for each species is returned.
            The default is True.
        basis : {'mass', 'mole'}, optional
            Basis for the returned enthalpy. The default is 'mass'.

        Returns
        -------
        hvapMass : J/kg
        hvapMole : J/mol

        """
        if mass_frac is None and mole_frac is None:
            mass_frac = self.mass_frac
            mole_frac = self.mole_frac
        elif mass_frac is None:
            mass_frac = self.frac_to_frac(mole_frac=mole_frac)
        else:
            mole_frac = self.frac_to_frac(mass_frac)

        if temp is None:
            temp = self.temp

        # Sensible heat
        if any(temp > self.t_crit):
            ind_super = np.where(temp > self.t_crit)[0]
            ind_sub = np.where(temp < self.t_crit)[0]

            ind_sort = np.argsort(np.concatenate((ind_super, ind_sub)))

            sensSuper = super().getEnthalpy(
                temp, temp_ref, mass_frac, mole_frac, total_h=total_h,
                idx=ind_super, phase='vapor', basis=basis)

            if len(ind_sub) > 0:
                sensSub = super().getEnthalpy(
                    temp, temp_ref, mass_frac, mole_frac, phase='liquid',
                    total_h=total_h, idx=ind_sub, basis=basis)

                if total_h:
                    hSens = sensSuper + sensSub
                else:
                    hSens = np.concatenate((sensSuper, sensSub))[ind_sort]

            else:
                hSens = sensSuper

        else:
            hSens = super().getEnthalpy(
                temp, temp_ref, mass_frac, mole_frac, phase='liquid',
                total_h=total_h, basis=basis)

        # Phase change
        deltaVap = self.getHeatVaporization(temp, basis=basis)

        # Collect terms
        if total_h:
            frac = mass_frac if basis == 'mass' else mole_frac
            hVap = hSens + np.dot(deltaVap, frac)

        else:
            hVap = hSens + deltaVap

        return hVap

    def AntoineEquation(self, temp=None, pres=None):
        a_ct, b_ct, c_ct = self.p_vap.T

        if pres is None:
            if isinstance(temp, np.ndarray):
                temp = temp[..., np.newaxis]

            vap_pressure = a_ct - b_ct / (temp + c_ct)

            return 10**(vap_pressure)

        else:
            if isinstance(pres, np.ndarray):
                pres = pres[..., np.newaxis]

            temp_sat = b_ct / (a_ct - np.log10(pres)) - c_ct

            return temp_sat

    def getDewPoint(self, pres=None, mass_frac=None, mole_frac=None,
                    thermo_method='ideal', x_liq=False):

        if mass_frac is None and mole_frac is None:
            mole_frac = self.mole_frac

        elif mole_frac is None:
            mole_frac = self.frac_to_frac(mass_frac=mass_frac)

        if pres is None:
            pres = self.pres

        def dew_fn(temp):
            k_vals = self.getKeqVLE(temp, pres, mole_frac,
                                    gamma_model=thermo_method)

            obj = np.dot(mole_frac, 1/k_vals) - 1

            return obj
        temp_pure = self.AntoineEquation(pres=pres)
        temp_seed = np.dot(mole_frac, temp_pure)
        temp_dew = newton(dew_fn, temp_seed, full_output=False)

        if x_liq:
            k_vals = self.getKeqVLE(temp_dew, pres, mole_frac,
                                    gamma_model=thermo_method)

            x_frac = mole_frac/k_vals

            return temp_dew, x_frac
        else:
            return temp_dew

    def getViscosity(self, temp=None, mass_frac=None, mole_frac=None):
        viscosity = self.getViscosityMix(temp, mass_frac, mole_frac,
                                         phase='vapor')

        return viscosity

    def getDensity(self, mass_frac: Optional[np.ndarray] = None,
                   mole_frac: Optional[np.ndarray] = None,
                   temp: Optional[float] = None, pres: Optional[float] = None,
                   basis: str = 'mass') -> Union[float, np.ndarray]:
        """Return ideal-gas density on the requested physical basis.

        Parameters
        ----------
        mass_frac, mole_frac : ndarray, optional
            Species fractions [-], shape ``(num_species,)`` or
            ``(num_points, num_species)``. Mole fractions take precedence
            when both are given; omitting both uses stored composition.
        temp, pres : float, optional
            Temperature [K] and pressure [Pa]; omitted values use stored state.
        basis : {'mass', 'mole'}, optional
            Mass density [kg/m**3] (default) or molar density [mol/L].

        Returns
        -------
        density : float or ndarray
            Density [kg/m**3] for ``'mass'`` or [mol/L] (= kmol/m**3) for
            ``'mole'``. Molar density is the SI value [mol/m**3] divided by
            1000 L/m**3. Both bases return shape ``(num_points,)`` for a
            two-dimensional composition, with one value per row. Molar density
            is composition-independent and is broadcast to the row count.

        Raises
        ------
        ValueError
            If ``basis`` is neither ``'mass'`` nor ``'mole'``.

        Notes
        -----
        Overrides do not change stored state. An empty vapor placeholder
        constructed with zero ``mole_frac`` has zero mass density; zero
        ``mass_frac`` or ``mole_conc`` construction is not supported.
        """
        if basis not in ('mass', 'mole'):
            raise ValueError("basis must be 'mass' or 'mole'")
        if temp is None:
            temp = self.temp  # [K]
        if pres is None:
            pres = self.pres  # [Pa]
        molar_density = pres / (VAPOR_GAS_CONSTANT * temp)  # [mol/m**3]
        if basis == 'mole':
            composition = mole_frac if mole_frac is not None else mass_frac  # [-]
            if composition is None:
                composition = self.mole_frac  # [-]
            if np.ndim(composition) > 1:
                return np.broadcast_to(molar_density / 1000,
                                       (len(composition),))  # [mol/L], 1000 L/m**3
            return molar_density / 1000  # [mol/L], 1000 L/m**3
        if mole_frac is None:
            if mass_frac is None:
                mole_frac = self.mole_frac  # [-]
            else:
                mole_frac = self.frac_to_frac(mass_frac=mass_frac)  # [-]
        mw_av = np.dot(mole_frac, self.mw)  # [g/mol]
        return molar_density * mw_av / 1000  # [kg/m**3], 1000 g/kg


class SolidPhase(ThermoPhysicalManager):
    """Represent a solid inventory with one mixture density across size bins.

    Distribution moments use a total-population basis: order n has units
    [m**n], with order zero a crystal count [-]. The volumetric shape factor
    ``kv`` converts the third moment to physical solid volume [m**3].

    ``distrib_type`` selects the documented mass- or volume-based bin weights.
    Under the uniform-density assumption, the normalized weights and resulting
    number distribution are identical for both bases. See ``__init__`` for
    the parameter reference and precedence of supplied moments.
    """

    def __init__(self, path_thermo, temp: ArrayLike = 298.15,
                 temp_ref: float = 298.15, pres=101325,
                 mass=0, mass_frac: Optional[ArrayLike] = None,
                 moments: Optional[ArrayLike] = None, num_mom=4,
                 distrib: Optional[ArrayLike] = None,
                 x_distrib: Optional[ArrayLike] = None, distrib_type='vol_perc',
                 moisture=0, porosity=0,
                 mole_conc: Optional[ArrayLike] = None, kv=1) -> None:
        """Initialize a solid inventory and its optional size distribution.

        Parameters
        ----------
        path_thermo : str
            Path to the thermodynamic property database.
        temp : float or array-like, optional
            Temperature [K]; default 298.15 K is the reference condition.
            Scalars are stored as Python floats. A spatial temperature profile
            of shape ``(num_points,)`` is stored as a float array without
            changing its shape; its axis is independent of the species axis.
        temp_ref : float, optional
            Enthalpy reference temperature [K], default 298.15 K, stored as a
            Python float.
        pres : float, optional
            Pressure [Pa], default one standard atmosphere (101325 Pa).
        mass : float, optional
            Solid mass [kg]. Zero derives inventory from the supplied moments
            or raw number distribution; positive mass scales bin weights.
        mass_frac : array-like
            Required species mass fractions [-], shape ``(num_species,)``.
            Copied before zero entries are replaced by machine epsilon.
        moments : array-like, optional
            Total-population moments, shape ``(num_moments,)``. Order n has
            units [m**n], with order zero a crystal count [-]. Takes precedence
            over ``distrib``; its length sets the stored moment count.
            When supplied, ``distrib`` and the values of ``x_distrib`` are
            stored as float arrays without scaling or conversion;
            ``distrib_type`` is unused for conversion. A supplied grid is
            stored as a float array and refreshes ``dx`` [um]; no grid leaves
            ``dx`` unset.
        num_mom : int, optional
            Number of moment orders [-]; default four covers orders zero
            through three, including the volume-related third moment.
        distrib : array-like, optional
            Shape ``(num_sizes,)``: number density [#/um] when mass is zero,
            otherwise bin weights [-] normalized on the ``distrib_type`` basis.
            If ``moments`` is supplied, stored as a float array without scaling
            or conversion, regardless of mass or ``distrib_type``.
        x_distrib : array-like, optional
            Crystal sizes [um], shape ``(num_sizes,)``. With ``moments``, values
            are stored unchanged as a float array and refresh ``dx`` [um].
        distrib_type : {'vol_perc', 'mass_frac'}, optional
            Basis of bin weights, default volume. One mixture density across
            all bins makes normalized mass and volume weights equivalent.
        moisture : float, optional
            Stored moisture content [-], default zero for dry solids.
        porosity : float, optional
            Stored pore volume fraction [-], default zero for nonporous solids.
        mole_conc : array-like, optional
            Reserved species molar concentrations [mol/L], shape
            ``(num_species,)``; currently unused.
        kv : float, optional
            Volumetric shape factor [-] in ``particle_volume = kv * size**3``;
            default one represents cubic particles.

        Raises
        ------
        ValueError
            If ``mass_frac`` is None, ``distrib_type`` is not 'vol_perc' or
            'mass_frac', or a supplied distribution grid has fewer than two
            points.
        RuntimeError
            If the species mass fractions sum to less than the existing
            composition threshold of 0.99 [-].
        """
        if mass_frac is None:
            raise ValueError("SolidPhase requires mass_frac; provide species "
                             "mass fractions with shape (num_species,)")
        if distrib_type not in ('vol_perc', 'mass_frac'):
            raise ValueError("distrib_type must be 'vol_perc' or 'mass_frac'; "
                             f"got {distrib_type!r}")

        super().__init__(path_thermo)
        self.kv = kv
        self.distrib_type = distrib_type
        self.num_mom = num_mom  # [-]

        self.cp_solid = np.atleast_2d(self.cp_solid)

        self.temp = (float(temp) if np.ndim(temp) == 0
                     else _as_float_array(temp))  # [K]
        self.temp_ref = float(temp_ref)  # [K]
        self.pres = pres

        self.mass = mass

        mass_frac = np.array(np.atleast_1d(mass_frac), dtype=float)  # [-]
        mass_frac[mass_frac == 0] = eps

        self.mass_frac = mass_frac
        self.mole_frac = self.frac_to_frac(mass_frac=self.mass_frac)

        solid_spec = False

        if moments is not None:
            self.num_mom = len(moments)
            self.moments = _as_float_array(moments)  # [m**n], order n

            self.x_distrib = None
            if x_distrib is not None:
                self.x_distrib = _as_float_array(x_distrib)  # [um]
                self.dx = self._get_grid_spacing(self.x_distrib)  # [um]
            # With moments, retain the supplied distribution basis.
            self.distrib = (None if distrib is None
                            else _as_float_array(distrib))  # [-] or [#/um]

            solid_spec = True

        elif distrib is not None:
            x_distrib = _as_float_array(x_distrib)  # [um]
            distrib = _as_float_array(distrib)  # [-] if mass > 0, else [#/um]

            self.x_distrib = x_distrib
            self.distrib = self.getDistribution(x_distrib, distrib)

            self.num_distrib = len(distrib)

            mom_idx = np.arange(self.num_mom)
            self.moments = self.getMoments(mom_num=mom_idx)

            solid_spec = True

        else:
            pass
            # print('Neither moment nor distribution data was '
            #       'provided for this SolidPhase object. Make sure to provide '
            #       'one of the two either when declaring this phase, or in a '
            #       'Slurry object to which this phase is aggregated')

        # Mass and volume
        dens = self.getDensity()

        if solid_spec:
            if self.mass == 0:
                self.vol = self.moments[3] * kv
                self.mass = self.vol * dens
            else:
                self.vol = self.mass / dens

        mw_av = np.dot(self.mole_frac, self.mw)  # [g/mol]
        self.mw_av = mw_av  # [g/mol]
        self._reconcile_moles_from_mass()

        sum_fracs = sum(mass_frac)  # [-]
        if sum_fracs < 0.99:
            raise RuntimeError(
                'The sum of mass fractions is less than 0.99')

        self.moisture = moisture
        self.porosity = porosity
        self.distribProf = None

        self._name = None
        self.transferred_from_uo = False

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        self._name = name

    def _reconcile_moles_from_mass(self) -> None:
        """Recalculate solid moles from the current mass.

        Notes
        -----
        Solid mass [kg] is converted using the exact ``1000 g/kg`` SI factor
        before division by the mixture-average molecular weight [g/mol]. The
        resulting amount is stored in ``moles`` [mol].
        """
        mass_grams = self.mass * 1000  # [g]
        self.moles = mass_grams / self.mw_av  # [mol]

    def updatePhase(self, x_distrib: Optional[ArrayLike] = None,
                    distrib: Optional[ArrayLike] = None,
                    mass: Optional[float] = None,
                    moments: Optional[ArrayLike] = None) -> None:
        """Update the solid size distribution, mass, or moments.

        Parameters
        ----------
        x_distrib : array-like, optional
            Crystal-size grid with shape ``(num_sizes,)`` [um]. When supplied,
            it replaces the stored grid before distribution moments are
            recalculated and refreshes the stored bin widths ``dx`` [um],
            using uniform spacing or geometric bin boundaries.
        distrib : array-like, optional
            Number-based crystal-size distribution on the total-population
            basis with shape ``(num_sizes,)`` [#/um]. It is assigned directly,
            without the constructor's mass-based normalization or conversion.
            Its third moment [m**3] is converted to physical solid volume with
            the volumetric shape factor ``kv``.
        mass : float, optional
            Solid mass [kg]. When supplied, it determines the stored volume
            from the solid mixture density [kg/m**3].
        moments : array-like, optional
            Crystal-size-distribution moments with shape ``(num_moments,)``.
            Moment order ``n`` has units [m**n] on the total-population basis;
            order zero is a crystal count [-].

        Notes
        -----
        Supplied float arrays may be stored by reference without copying, so
        callers must not rely on isolation from later mutations.
        Supplied grids, distributions, and moments are stored as float arrays
        without changing their shapes or bases.
        On the required total-population basis, the distribution-derived third
        moment ``mu_3`` has units [m**3], and volume follows
        ``V_solid = kv * mu_3`` [m**3]. By contrast, construction with
        ``mass > 0`` interprets ``distrib`` as normalized bin weights and
        converts them according to ``distrib_type`` before moments are taken.

        If ``mass`` is supplied in the same call, the explicit mass and its
        density-derived volume take precedence. If ``moments`` is also
        supplied, it replaces the recalculated moments without another mass or
        volume update. Distribution updates recalculate orders zero through
        ``self.num_mom - 1``, preserving the configured moment-state size. The
        mole amount is recalculated whenever a distribution or explicit
        mass changes the solid inventory. A moments-only update does not imply
        an amount change and therefore leaves mass, volume, and moles intact.
        """
        if x_distrib is not None:
            self.x_distrib = _as_float_array(x_distrib)  # [um]
            self.dx = self._get_grid_spacing(self.x_distrib)  # [um]

        if distrib is not None:
            self.distrib = _as_float_array(distrib)  # [#/um]
            moment_orders = np.arange(self.num_mom)  # [-]
            self.moments = self.getMoments(mom_num=moment_orders)
            self.num_distrib = len(self.distrib)

            self.vol = self.moments[3] * self.kv  # [m**3]
            self.mass = self.vol * self.getDensity()  # [kg]

        if mass is not None:
            self.mass = mass
            self.vol = mass / self.getDensity()

        if moments is not None:
            self.moments = _as_float_array(moments)  # [m**n], order n

        if distrib is not None or mass is not None:
            self._reconcile_moles_from_mass()

    def convert_distribution(self, x_distrib=None, num_distr=None,
                             vol_distr=None, mass=0):
        if x_distrib is None:
            x_distrib = self.x_distrib

        if num_distr is not None and vol_distr is not None:
            raise ValueError("Specify either 'num_distr' or 'vol_distr', "
                             "not both")
        elif num_distr is not None:  # convert to vol perc
            mom_three = self.getMoments(distrib=num_distr, mom_num=3)
            mom_three[mom_three == 0] = eps

            distrib_out = num_distr * self.dx * x_distrib**3 * self.kv / \
                mom_three / 1e18
        elif vol_distr is not None:
            if mass == 0:
                raise ValueError("'vol_perc' given, mass must be greater "
                                 "than zero.")
            dens = self.getDensity()
            distrib_out = (mass / dens) * vol_distr / self.kv / \
                x_distrib**3 / self.dx * 1e18  # number/um

        return distrib_out

    def _get_grid_spacing(self, x_distrib: np.ndarray) -> Union[float, np.ndarray]:
        """Calculate widths using the established crystal-grid convention.

        Parameters
        ----------
        x_distrib : numpy.ndarray
            Crystal sizes [um], shape ``(num_sizes,)``, with at least two
            points. Nonuniform grids are assumed to be geometric series.

        Returns
        -------
        float or numpy.ndarray
            Bin widths [um]: a scalar for uniform spacing, otherwise an array
            of shape ``(num_sizes,)``.

        Raises
        ------
        ValueError
            If fewer than two grid points are supplied, so bin widths cannot
            be determined.

        Notes
        -----
        For geometric grids, interior boundaries are geometric means of
        adjacent sizes. End boundaries extend that sequence by the grid ratio.
        Uniform-spacing detection retains NumPy's default isclose tolerances
        for compatibility with construction.
        """
        if len(x_distrib) < 2:
            raise ValueError("x_distrib must contain at least two grid points "
                             "to calculate bin widths")
        delta_x = np.diff(x_distrib)  # [um]
        equal = np.isclose(delta_x[1:], delta_x[:-1]).all()
        if equal:
            return delta_x[0]

        ratio = x_distrib[1] / x_distrib[0]  # [-]
        x_shifted = np.zeros(len(x_distrib) + 1)  # [um]
        x_gr = np.sqrt(x_distrib[1:] * x_distrib[:-1])  # [um]
        x_shifted[0] = x_gr[0] / ratio
        x_shifted[-1] = x_gr[-1] * ratio
        x_shifted[1:-1] = x_gr
        return np.diff(x_shifted)

    def getDistribution(self, x_distrib: np.ndarray,
                        distrib: np.ndarray) -> np.ndarray:
        """Convert initial bin weights to a total-population distribution.

        Parameters
        ----------
        x_distrib : numpy.ndarray
            Crystal-size grid [um], shape ``(num_sizes,)``; matches the stored
            grid used by ``convert_distribution``.
        distrib : numpy.ndarray
            Bin weights [-] when stored mass is positive, otherwise raw number
            density [#/um], with shape ``(num_sizes,)``.

        Returns
        -------
        numpy.ndarray
            Number distribution [#/um], shape ``(num_sizes,)``.

        Notes
        -----
        Refreshes ``dx`` [um]. Positive-mass inputs are normalized by their
        sum. With one mixture density across bins, mass and volume fractions
        coincide. The volume conversion divides solid mass by density [kg/m**3]
        and each bin's volume by ``kv * size**3`` and bin width to obtain [#/um].
        Zero-mass inputs are returned without normalization or conversion.
        """
        self.dx = self._get_grid_spacing(x_distrib)  # [um]
        distrib = np.asarray(distrib)  # [-] if mass > 0, otherwise [#/um]
        if self.mass > 0:
            bin_weights = distrib / distrib.sum()  # [-]
            distr = self.convert_distribution(
                vol_distr=bin_weights, mass=self.mass)  # [#/um]
        else:
            distr = distrib  # [#/um]

        return distr

    def getMoments(self, x_distrib=None, distrib=None, mom_num=None):
        if x_distrib is None:
            x_distrib = self.x_distrib

        if distrib is None:
            distrib = self.distrib

        if mom_num is None:
            mom_ind = range(4)
        elif isinstance(mom_num, int):
            mom_ind = [mom_num]
        else:
            mom_ind = mom_num

        if distrib.ndim == 1 or len(distrib) == 1:
            moments = np.zeros(len(mom_ind))
            for ind, exp in enumerate(mom_ind):
                integrand = distrib * x_distrib**exp
                moments[ind] = trapezoidal_rule(x_distrib, integrand.T)

            if len(mom_ind) == 1:
                moments = moments[0]

        else:
            moments = np.zeros((len(distrib), len(mom_ind)))
            for ind, exp in enumerate(mom_ind):
                integrand = distrib * x_distrib**exp
                moments[:, ind] = trapezoidal_rule(x_distrib, integrand.T)

        conv_factors = (1e-6)**np.array(mom_ind)
        moments *= conv_factors

        return moments

    def getDensity(self, mass_frac=None, mole_frac=None, temp=None,
                   basis='mass'):

        if temp is None:
            temp = self.temp

        if mass_frac is None and mole_frac is None:
            mass_frac = self.mass_frac
            # mole_frac = self.mole_frac

        densSolid = self.getDensityMix(mass_frac, mole_frac, phase='solid',
                                       temp=temp, basis=basis)

        return densSolid

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

    def getCp(self, temp=None, mass_frac=None, mole_frac=None, basis='mass'):
        if temp is None:
            temp = self.temp

        if mass_frac is None and mole_frac is None:
            mass_frac = self.mass_frac
            mole_frac = self.mole_frac

        cpSolid = super().getCpMix(temp, mass_frac, mole_frac, phase='solid',
                                   basis=basis)

        return cpSolid

    def getEnthalpy(self, temp=None, temp_ref=298.15, mass_frac=None,
                    mole_frac=None, total_h=True, basis='mass'):

        if mass_frac is None and mole_frac is None:
            mass_frac = self.mass_frac
            mole_frac = self.mole_frac

        if temp is None:
            temp = self.temp

        hSolid = super().getEnthalpy(temp, temp_ref, mass_frac, mole_frac,
                                     phase='solid', total_h=total_h,
                                     basis=basis)

        return hSolid
