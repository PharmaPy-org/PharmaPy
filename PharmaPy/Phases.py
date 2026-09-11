# -*- coding: utf-8 -*-


# import numpy as np
# from autograd import numpy as np
from typing import Optional, Union

import numpy as np
from PharmaPy.ThermoModule import ThermoPhysicalManager
from PharmaPy.Commons import trapezoidal_rule
from scipy.optimize import newton

import warnings

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
                phase_name = 'Vapor_{}'.format(vapor_count)
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
    def __init__(self, path_thermo=None, temp=298.15, pres=101325,
                 mass=0, vol=0, moles=0,
                 mass_frac=None, mole_frac=None,
                 mass_conc=None, mole_conc=None,
                 name_solv=None, verbose=True, check_input=True):

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

        if mass_frac is not None:
            self.mass_frac = np.array(mass_frac)
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

            self.mole_frac = np.array(mole_frac)
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
            self.mass_conc = np.array(mass_conc)
            self.mole_frac = mole_frac

            self.mass_frac = mass_frac
            self.mole_conc = mole_conc

            self.__calcComposition()

        elif mole_conc is not None:
            self.mole_conc = np.array(mole_conc)
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

    def updatePhase(self, mole_conc=None, mass_conc=None,
                    mass_frac=None, mole_frac=None,
                    vol=0, mass=0, moles=0, temp=None, pres=None):
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
        temp : float, optional
            Liquid temperature [K]. The stored temperature is left unchanged
            when ``None``.
        pres : float, optional
            Liquid pressure [Pa]. The stored pressure is left unchanged when
            ``None``.

        Returns
        -------
        None
            The phase state is updated in place.

        Notes
        -----
        Composition arguments are resolved in the order ``mole_conc``,
        ``mass_conc``, ``mass_frac``, ``mole_frac``; the first one supplied
        determines the new composition and the remaining ones are ignored.
        When none is supplied, the stored composition is retained and only
        the amount and intensive state are refreshed.

        The amount is set from the first positive value among ``mass``,
        ``vol``, and ``moles``, using the mixture mass density [kg/m**3] and
        the average molar mass [g/mol]. When all three are left at zero, the
        previously stored mass, volume, and moles are retained.

        Solvent handling matches the constructor: ``self.ind_solv`` is
        compared with ``None`` rather than tested for truth, so a solvent
        declared as the first species (index ``0``) is honored and the
        concentration converters' three-value return is unpacked correctly.
        """

        if mole_conc is not None:
            frac_out = self.conc_to_frac(mole_conc,
                                         solvent_ind=self.ind_solv)
            if self.ind_solv is not None:
                mass_frac, mole_frac, mole_conc = frac_out
            else:
                mass_frac, mole_frac = frac_out

            mass_conc = mole_conc * self.mw

        elif mass_conc is not None:
            frac_out = self.mass_conc_to_frac(mass_conc,
                                              solvent_ind=self.ind_solv)

            if self.ind_solv is not None:
                mass_frac, mole_frac, mass_conc = frac_out
            else:
                mass_frac, mole_frac = frac_out

            mole_conc = mass_conc / self.mw

        elif mass_frac is not None:
            mole_conc = self.frac_to_conc(mass_frac)
            mass_conc = mole_conc * self.mw
            mole_frac = self.frac_to_frac(mass_frac)

        elif mole_frac is not None:
            mole_conc = self.frac_to_conc(mole_frac=mole_frac)
            mass_conc = mole_conc * self.mw
            mass_frac = self.frac_to_frac(mole_frac=mole_frac)

        else:
            mass_frac = self.mass_frac
            mole_frac = self.mole_frac
            mole_conc = self.mole_conc
            mass_conc = self.mass_conc

        if temp is not None:
            self.temp = temp

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
    specified composition.
    """

    def __init__(self, path_thermo=None, temp=298.15, pres=101325,
                 mass=0, vol=0, moles=0,
                 mass_frac=None, mole_frac=None, mole_conc=None,
                 check_input=True, verbose=True):
        """Initialize a vapor-phase thermodynamic state.

        Parameters
        ----------
        path_thermo : str, optional
            Path to the species thermophysical-property JSON file.
        temp : float, optional
            Vapor temperature [K].
        pres : float, optional
            Vapor pressure [Pa].
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
            If no composition measure is provided.
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
        ``mole_conc``. Concentration inputs use PharmaPy's [mol/L] basis.
        """

        super().__init__(path_thermo)

        # Calculate amount of material and compositions using LiquidPhase
        props = LiquidPhase(
            path_thermo=path_thermo, temp=temp, pres=pres,
            mass=mass, vol=vol, moles=moles,
            mass_frac=mass_frac, mole_frac=mole_frac,
            mole_conc=mole_conc,
            check_input=check_input, verbose=verbose,
        )

        self.mass = props.mass
        self.moles = props.moles
        self.vol = props.vol

        self.mass_frac = props.mass_frac
        self.mole_frac = props.mole_frac
        self.mole_conc = props.mole_conc

        self.mw_av = props.mw_av

        self.temp = float(temp)
        self.pres = pres  # [Pa]

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

    def updatePhase(self, mole_conc=None, mass_conc=None,
                    mass_frac=None, mole_frac=None,
                    vol=0, mass=0, moles=0):

        if mole_conc is not None:
            frac_out = self.conc_to_frac(mole_conc,
                                         solv_ind=self.ind_solv)
            if self.ind_solv:
                mass_frac, mole_frac, mole_conc = frac_out
            else:
                mass_frac, mole_frac = frac_out

            mass_conc = mole_conc * self.mw

        elif mass_conc is not None:
            frac_out = self.mass_conc_to_frac(mass_conc,
                                              solv_ind=self.ind_solv)

            if self.ind_solv:
                mass_frac, mole_frac, mass_conc = frac_out
            else:
                mass_frac, mole_frac = frac_out

            mole_conc = mass_conc / self.mw

        elif mass_frac is not None:
            mole_conc = self.frac_to_conc(mass_frac)
            mass_conc = mole_conc * self.mw
            mole_frac = self.frac_to_frac(mass_frac)

        elif mole_frac is not None:
            mole_conc = self.frac_to_conc(mole_frac=mole_frac)
            mass_conc = mole_conc * self.mw
            mass_frac = self.frac_to_frac(mole_frac=mole_frac)

        else:
            mass_frac = self.mass_frac
            mole_frac = self.mole_frac
            mole_conc = self.mole_conc
            mass_conc = self.mass_conc

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

    def getEnthalpy(
            self,
            temp: Optional[Union[float, np.ndarray]] = None,
            temp_ref: float = 298.15,
            mass_frac: Optional[np.ndarray] = None,
            mole_frac: Optional[np.ndarray] = None,
            total_h: bool = True,
            basis: str = 'mass') -> Union[float, np.ndarray]:
        """Calculate vapor-phase enthalpy relative to a liquid reference.

        Parameters
        ----------
        temp : float or ndarray, optional
            Temperature at which to calculate enthalpy [K]. The phase
            temperature is used when ``temp`` is ``None``. The supported
            forms are a scalar or a one-element array.
        temp_ref : float, optional
            Liquid-reference temperature for the sensible-enthalpy integral
            [K]. The default is 298.15 K.
        mass_frac : ndarray, optional
            One-dimensional vapor-phase species mass fractions [-], ordered
            according to the phase's components. The phase composition is
            used when neither fraction vector is supplied.
        mole_frac : ndarray, optional
            One-dimensional vapor-phase species mole fractions [-], ordered
            according to the phase's components. The phase composition is
            used when neither fraction vector is supplied.
        total_h : bool, optional
            If ``True``, return fraction-weighted mixture enthalpy. If
            ``False``, return individual species enthalpies. The default is
            ``True``.
        basis : {'mass', 'mole'}, optional
            Physical basis of the returned enthalpy. The default is ``'mass'``.

        Returns
        -------
        float or ndarray
            Vapor-phase enthalpy on the selected basis: [J/kg] for
            ``basis='mass'`` and [J/mol] for ``basis='mole'``. For the
            supported single-temperature forms, when the requested
            temperature differs from every species critical temperature,
            mixture enthalpy is scalar and individual species enthalpy has
            shape ``(1, num_species)`` in the phase's component order.

        Raises
        ------
        ValueError
            If the Watson correlation used for a subcritical species receives
            a negative reduced-temperature ratio.

        Notes
        -----
        Supercritical species contribute vapor sensible heat and no latent
        heat. Subcritical species contribute liquid sensible heat plus latent
        heat of vaporization.

        Arrays containing more than one temperature are outside the supported
        public contract, but this boundary is not yet validated. The current
        implementation accepts an array only when its length equals the number
        of species; every other length raises from the elementwise
        ``temp > t_crit`` comparison. That comparison pairs a temperature
        index with a species index, so the split is meaningful only while each
        species keeps one classification over the requested temperatures.

        A temperature exactly equal to a species critical temperature is
        omitted from both strict comparison subsets. The broader
        temperature-axis validation and criticality classification contract is
        tracked in issue #178.
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
                    hSens = np.concatenate(
                        (sensSuper, sensSub), axis=1)[:, ind_sort]

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

    def getDensity(self, pres_gas=None, temp_gas=None, phase ='gas', basis='mole'):

        if pres_gas is None and temp_gas is None:

            pres_gas = self.pres_gas
            temp_gas = self.temp

        densGas = pres_gas/ (8.314 * temp_gas)

        return densGas


class SolidPhase(ThermoPhysicalManager):
    """    

    Parameters
    ----------
    path_thermo : string
        Directory of the physical properties .json file
    temp : float or array-like
        Temperature for enthalpy calculation in K.   
    temp_ref : float, optional
        Reference temperature for enthalpy calculation. The default is 298.15.
    pres : float, optional
        Pressure in atmosphere of the system in pascals. The default is 101325.
    mass : float, optional
        Mass of solids in kg. The default is 0.
    mass_frac : array-like, optional
        Fraction of the species participating in the solid phase in mass basis.
        The default is None.
    moments : array, optional
        Total-population distribution moments with shape ``(num_moments,)``.
        Moment order ``n`` has units [m**n], with order zero a crystal count
        [-]. The default is None.
    num_mom : integer, optional
        Number of moments describing the solid phase [-]. This sizes the
        classical moment-method state vector. The default is 4 (orders 0--3).
    x_distrib : array, optional
        Array of size N, containing the internal grid
        size coordinate of the solids [um]. The default is None
    distrib : array, optional
        Initial crystal-size distribution with shape ``(num_sizes,)``. When
        ``mass == 0``, this must be a number-based total-population
        distribution [#/um] and is stored directly. When ``mass > 0``, the
        values are normalized as dimensionless bin weights [-] on the
        ``distrib_type`` basis, then converted to [#/um] consistently with the
        specified solid mass. The default is None.
    distrib_type : string, optional
        Type of distribution of crystals. The option is 'mass_frac' 
        or 'vol_perc'. The default is 'vol_perc'.
    moisture : float, optional
        Initial moisture content of the solids. The default is 0.
    porosity : float, optional
        Volume-based pore fraction out of the packed solid beds. The default is 0.
    mole_conc : array-like, optional
        Concentration of the species participating in the solid phase in mole basis. The default is None.
    kv : float, optional
        Volumetric shape factor of the solids. The default is 1.



    Returns
    -------
    None.

    Notes
    -----
    Solid mass is stored in kilograms, while molecular weight is stored in
    grams per mole. Mole amounts are calculated from the finalized solid mass
    after converting it to grams during construction.

    ``moles`` is reconciled whenever construction or ``updatePhase`` changes
    the solid mass, so mass-, volume-, and mole-basis amounts describe the
    same physical inventory.

    """
    
    def __init__(self, path_thermo, temp=298.15, temp_ref=298.15, pres=101325,
                 mass=0, mass_frac=None,
                 moments=None, num_mom=4,
                 distrib=None, x_distrib=None, distrib_type='vol_perc',
                 moisture=0, porosity=0,
                 mole_conc=None, kv=1):
        
        super().__init__(path_thermo)
        self.kv = kv
        self.distrib_type = distrib_type
        self.num_mom = num_mom  # [-]

        self.cp_solid = np.atleast_2d(self.cp_solid)

        self.temp = temp
        self.temp_ref = temp_ref
        self.pres = pres

        self.mass = mass

        mass_frac = np.atleast_1d(mass_frac)
        mass_frac[mass_frac == 0] = eps

        self.mass_frac = mass_frac
        self.mole_frac = self.frac_to_frac(mass_frac=self.mass_frac)

        solid_spec = False

        if moments is not None:
            self.num_mom = len(moments)
            self.moments = moments

            self.x_distrib = x_distrib
            self.distrib = distrib

            solid_spec = True

        elif distrib is not None:
            x_distrib = np.asarray(x_distrib)

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

        if mass_frac is not None:
            sum_fracs = sum(mass_frac)
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

    def updatePhase(self, x_distrib: Optional[np.ndarray] = None,
                    distrib: Optional[np.ndarray] = None,
                    mass: Optional[float] = None,
                    moments: Optional[np.ndarray] = None) -> None:
        """Update the solid size distribution, mass, or moments.

        Parameters
        ----------
        x_distrib : numpy.ndarray, optional
            Crystal-size grid with shape ``(num_sizes,)`` [um]. When supplied,
            it replaces the stored grid before distribution moments are
            recalculated. It does not refresh the stored bin widths ``dx``;
            that pre-existing synchronization defect is tracked in issue #162.
        distrib : numpy.ndarray, optional
            Number-based crystal-size distribution on the total-population
            basis with shape ``(num_sizes,)`` [#/um]. It is assigned directly,
            without the constructor's mass-based normalization or conversion.
            Its third moment [m**3] is converted to physical solid volume with
            the volumetric shape factor ``kv``.
        mass : float, optional
            Solid mass [kg]. When supplied, it determines the stored volume
            from the solid mixture density [kg/m**3].
        moments : numpy.ndarray, optional
            Crystal-size-distribution moments with shape ``(num_moments,)``.
            Moment order ``n`` has units [m**n] on the total-population basis;
            order zero is a crystal count [-].

        Notes
        -----
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
            self.x_distrib = x_distrib

        if distrib is not None:
            self.distrib = distrib
            moment_orders = np.arange(self.num_mom)  # [-]
            self.moments = self.getMoments(mom_num=moment_orders)
            self.num_distrib = len(distrib)

            self.vol = self.moments[3] * self.kv  # [m**3]
            self.mass = self.vol * self.getDensity()  # [kg]

        if mass is not None:
            self.mass = mass
            self.vol = mass / self.getDensity()

        if moments is not None:
            self.moments = moments

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

    def getDistribution(self, x_distrib, distrib):
        dens = self.getDensity()

        # Crystal size dimension
        delta_x = np.diff(x_distrib)
        equal = np.isclose(delta_x[1:], delta_x[:-1]).all()
        if equal:
            self.dx = delta_x[0]
        else:  # assume geometric series and make adjustments
            ratio = x_distrib[1] / x_distrib[0]
            x_shifted = np.zeros(len(x_distrib) + 1)
            x_gr = np.sqrt(x_distrib[1:] * x_distrib[:-1])

            x_shifted[0] = x_gr[0] / ratio
            x_shifted[-1] = x_gr[-1] * ratio

            x_shifted[1:-1] = x_gr

            self.dx = np.diff(x_shifted)

        # Distribution
        distrib = np.asarray(distrib)
        if self.mass > 0:
            distrib = distrib / distrib.sum()
            if self.distrib_type == 'vol_perc':
                distr = self.convert_distribution(vol_distr=distrib,
                                                  mass=self.mass)
            elif self.distrib_type == 'mass_perc':
                distr = self.mass*distrib / x_distrib**3 / self.kv * 1e18

        else:
            distr = distrib

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
