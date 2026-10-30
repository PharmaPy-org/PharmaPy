# -*- coding: utf-8 -*-


# import numpy as np
# from autograd import numpy as np
from collections.abc import Mapping
from typing import Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike
from PharmaPy.ThermoModule import (MissingPropertyError,
                                  ThermoPhysicalManager,
                                  validate_activity_model)
from PharmaPy.Commons import trapezoidal_rule
from PharmaPy.Errors import PharmaPyValueError
from scipy.optimize import newton

import warnings

eps = np.finfo(float).eps

# CODATA 2018: R = 8.314462618 J/(mol K), rounded to four significant figures
# to match the existing gas_ct = 8.314 in Kinetics, Crystallizers, Reactors,
# Drying_Model, Evaporators, and ThermoModule, so vapor amounts agree with
# the evaporator's own P V / (R T).
VAPOR_GAS_CONSTANT = 8.314  # [J/mol/K]

# Exponent n of the Watson latent-heat correlation,
# dh(T) = dh(Tref) * ((Tc - T) / (Tc - Tref))**n (K. M. Watson, Ind. Eng.
# Chem. 1943, 35, 398-406). Poling, Prausnitz and O'Connell, The Properties
# of Gases and Liquids, 5th ed. (2001), Section 7-11, Eq. (7-11.1), p. 7.24,
# list n = 0.375 or 0.38 as the common choices; PharmaPy has always used 0.38.
# Valid for subcritical temperatures, T <= Tc, with Tref < Tc.
WATSON_EXPONENT = 0.38  # [-]


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


def _amount_supplied(amount: ArrayLike) -> bool:
    """Tell whether an amount argument was supplied.

    Parameters
    ----------
    amount : float or array-like
        Mass [kg], volume [m**3] or moles [mol] (per second for streams),
        scalar or per-point with shape ``(num_points,)``.

    Returns
    -------
    bool
        True if any entry is positive. Zero means "not supplied", so a
        per-point amount counts as supplied when at least one point holds
        material; its zero points then hold none.
    """
    return bool(np.any(np.asarray(amount, dtype=float) > 0))


def _amount_ratio(amount: ArrayLike, divisor: ArrayLike):
    """Divide an amount by a property, giving zero where the amount is zero.

    Parameters
    ----------
    amount : float or array-like
        Mass [kg], volume [m**3] or moles [mol] (per second for streams),
        scalar or per point.
    divisor : float or array-like
        Density or molar mass converting it, in matching units, scalar or
        per point.

    Returns
    -------
    numpy.float64 or numpy.ndarray
        ``amount / divisor`` with the broadcast shape; points without
        material (zero amount) give exactly zero, even where the divisor is
        zero or not finite because their composition is undefined.
    """
    amount = np.asarray(amount, dtype=float)
    divisor = np.asarray(divisor, dtype=float)
    shape = np.broadcast(amount, divisor).shape
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.divide(amount, divisor, out=np.zeros(shape),
                          where=amount != 0)
    return ratio[()] if ratio.ndim == 0 else ratio


def _amount_product(amount: ArrayLike, factor: ArrayLike):
    """Multiply an amount by a property, giving zero where the amount is zero.

    Parameters
    ----------
    amount : float or array-like
        Mass [kg], volume [m**3] or moles [mol] (per second for streams),
        scalar or per point.
    factor : float or array-like
        Density or molar mass converting it, in matching units, scalar or
        per point.

    Returns
    -------
    numpy.float64 or numpy.ndarray
        ``amount * factor`` with the broadcast shape; points without
        material give exactly zero, even where the factor is not finite.
    """
    amount = np.asarray(amount, dtype=float)
    factor = np.asarray(factor, dtype=float)
    shape = np.broadcast(amount, factor).shape
    with np.errstate(invalid='ignore'):
        product = np.multiply(amount, factor, out=np.zeros(shape),
                              where=amount != 0)
    return product[()] if product.ndim == 0 else product


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
    """Liquid holdup with a composition, an amount and an intensive state.

    The composition is given by exactly one of ``mass_frac`` or
    ``mole_frac`` [-], ``mass_conc`` [kg/m**3] or ``mole_conc`` [mol/L],
    each covering every species of the database, solvent included. The
    amount is given by ``mass`` [kg], ``vol`` [m**3] or ``moles`` [mol];
    see :meth:`__init__` for the constructor parameters.

    Notes
    -----
    The pure-species liquid densities ``rho_liq`` [kg/m**3] are temperature
    independent, and :meth:`getDensity` mixes them ideally (additive
    volumes): ``1 / rho = sum_i w_i / rho_i``.

    How an amount is reconciled depends on the composition basis:

    * Fractions carry no absolute packing, so the ideal-mixing density
      relates mass, volume and moles.
    * Concentrations with ``name_solv``: the solvent concentration is
      completed so that ``sum_i c_i v_i = 1``, with ``v_i = MW_i / rho_i``
      the pure molar volume [L/mol]. The concentrations then close ideal
      mixing, and the ideal-mixing density applies as for fractions.
    * Concentrations without ``name_solv`` are authoritative (issue #439):
      ``moles = sum_i c_i * vol * 1000`` [mol] and
      ``mass = sum_i mass_conc_i * vol`` [kg], so the inventory density is
      ``sum_i mass_conc_i`` [kg/m**3] and the ideal-mixing model is not
      used for the amounts. The supplied concentrations need not close
      ideal mixing; when they do, both rules agree to rounding.

    In every case the stored state satisfies
    ``mole_conc * vol * 1000 == mole_frac * moles`` and
    ``sum(mass_conc) * vol == mass`` to rounding. :attr:`concentration_basis`
    reports which rule relates the amounts, and :meth:`getInventoryDensity`
    returns the corresponding density; :meth:`getDensity` remains the
    ideal-mixing property model.
    """
    # Frames between the zero-amount warning and the code that constructed
    # the phase: ``__init__`` sits one frame below its caller. Subclasses
    # whose ``__init__`` delegates here add one frame each so Python's
    # per-location ``default`` filter reports every constructing line.
    _zero_amount_stacklevel = 2

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
            Solvent species whose concentration is completed from the
            ideal-mixing volume balance ``sum_i c_i v_i = 1`` when
            concentrations are supplied. With None (default), supplied
            concentrations are authoritative: fractions are normalized from
            them and the amounts follow ``sum(mole_conc) * vol * 1000``
            [mol] and ``sum(mass_conc) * vol`` [kg] (see the class Notes).
            Ignored for fraction inputs.
        verbose : bool, optional
            Print diagnostics for fractions summing below the legacy 0.99
            threshold [-]. Defaults to True.
        check_input : bool, optional
            Warn if all amounts are zero, subject to the caller's warning
            filters. Defaults to True.

        Raises
        ------
        ValueError
            If no composition measure is supplied.
        PharmaPyValueError
            A subclass of ``ValueError``. If a positive ``mass``, ``vol``
            or ``moles`` is supplied with authoritative concentrations (no
            ``name_solv``) whose total is zero, which leaves the fractions
            (and, for mass or moles, the volume) undefined.
        MissingPropertyError
            A subclass of ``AttributeError`` and ``ValueError``. If a species
            with a nonzero fraction or concentration has no ``rho_liq``
            [kg/m**3], which the ideal-mixing volume and concentrations
            need; with ``name_solv``, the solvent always needs it. Absent
            species need none.
        RuntimeWarning
            If more than one composition measure is supplied.

        Warnings
        --------
        RuntimeWarning
            If all amounts are zero and check_input is True. The caller's
            warning policy determines whether this is shown, ignored, or raised.
            The warning is attributed to the line that constructed the phase,
            so a per-location policy such as ``default`` reports each
            constructing line once.

        Notes
        -----
        Warning filters are left unchanged.
        Exactly one composition measure is required. Constructor composition
        inputs are copied before conversion. Derived concentrations retain
        the liquid volume basis. No physical basis changes during coercion.
        Amounts are reconciled on the concentration basis when ``mole_conc``
        or ``mass_conc`` is supplied without ``name_solv`` and on the
        ideal-mixing basis otherwise; :attr:`concentration_basis` records
        the choice. Earlier releases always used ideal mixing, so for
        concentrations that do not close ``sum_i c_i v_i = 1`` the mass,
        moles and fractions-times-amount now follow the concentrations
        (issue #439).
        """

        super().__init__(path_thermo)

        if name_solv is None:
            ind_solv = name_solv
        else:
            ind_solv = self.name_species.index(name_solv)

        self.ind_solv = ind_solv

        # Set after ThermoPhysicalManager.__init__, which replaces __dict__.
        # __calcComposition sets the basis of the supplied composition.
        self._concentration_basis = False
        self._solv_mass_frac = None

        self.temp = float(temp)
        self.pres = pres

        self.mass = mass
        self.vol = vol
        self.moles = moles

        # Support solute concentrations alongside a solvent-blend mass_frac mapping (#336, #338).
        has_single_conc = (mole_conc is not None) ^ (mass_conc is not None)
        if has_single_conc and isinstance(mass_frac, Mapping) and mole_frac is None:
            if name_solv is not None:
                raise ValueError(
                    "Specify either 'name_solv' or a solvent-blend 'mass_frac' "
                    "mapping with solute concentrations, not both."
                )
            conc_arg = mole_conc if mole_conc is not None else mass_conc
            if isinstance(conc_arg, Mapping):
                overlap = set(conc_arg.keys()) & set(mass_frac.keys())
                if overlap:
                    raise ValueError(
                        f"Species {sorted(overlap)} cannot be specified in both "
                        "solute concentrations and solvent mass_frac."
                    )
            self._solv_mass_frac = dict(mass_frac)
            mass_frac = None

        unspec_num = (mass_frac is None) + (mass_conc is None) + \
            (mole_conc is None) + (mole_frac is None)

        if unspec_num == 4:
            raise ValueError("No measure of composition was provided")
        elif unspec_num < 3:
            raise RuntimeWarning("More than one measure of composition was "
                                 "provided")

        mass_frac = self._coerce_composition_mapping(mass_frac, 'mass_frac')
        mole_frac = self._coerce_composition_mapping(mole_frac, 'mole_frac')
        mass_conc = self._coerce_composition_mapping(mass_conc, 'mass_conc')
        mole_conc = self._coerce_composition_mapping(mole_conc, 'mole_conc')

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
                warnings.warn("'mass', 'moles' and 'vol' are all set to zero. "
                              "Model may not perform as intended.",
                              RuntimeWarning,
                              stacklevel=self._zero_amount_stacklevel)

        self.y_upstream = None

        self._name = None
        self.transferred_from_uo = False

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        self._name = name

    @property
    def concentration_basis(self) -> bool:
        """Whether the amounts follow the stored concentrations (read-only).

        Returns
        -------
        bool
            True when the current composition was supplied as ``mole_conc``
            [mol/L] or ``mass_conc`` [kg/m**3] without a solvent
            (``name_solv`` None). Mass [kg], volume [m**3] and moles [mol]
            are then related by the inventory density
            ``sum(mass_conc)`` [kg/m**3] and ``sum(mole_conc)`` [mol/L]
            instead of ideal mixing (issue #439). False for fraction inputs
            and for every ``name_solv`` input.

        Notes
        -----
        The flag is set by the constructor and by :meth:`updatePhase`
        composition updates: concentrations without a solvent switch it on,
        fractions or a declared solvent switch it off, and amount-only or
        intensive-state updates keep it. It is read-only so that it cannot
        disagree with the stored state; consumers that convert this phase's
        own amount to a volume use :meth:`getInventoryDensity`.
        """
        return self._concentration_basis

    def __set_amounts(self, mass, vol, moles, massfrac, molefrac,
                      conc, mass_conc, concentration_basis=False):
        """Store a composition and reconcile mass, volume and moles.

        Parameters
        ----------
        mass : float or numpy.ndarray
            Liquid mass [kg] (streams: [kg/s]), scalar or per point with
            shape ``(num_points,)``; used when any entry is positive.
        vol : float or numpy.ndarray
            Liquid volume [m**3] (streams: [m**3/s]); used when ``mass`` is
            not positive and ``vol`` is.
        moles : float or numpy.ndarray
            Liquid amount [mol] (streams: [mol/s]); used when neither
            ``mass`` nor ``vol`` is positive.
        massfrac, molefrac : numpy.ndarray
            Species mass and mole fractions [-], shape ``(num_species,)``
            or ``(num_points, num_species)``.
        conc : numpy.ndarray
            Species molar concentrations [mol/L], same shape.
        mass_conc : numpy.ndarray
            Species mass concentrations [kg/m**3], same shape.
        concentration_basis : bool, optional
            If True, the concentrations are authoritative: the inventory
            mass density is ``sum(mass_conc)`` [kg/m**3] and the molar
            density ``sum(conc)`` [mol/L] along the species axis. If False
            (default), the ideal-mixing density of ``massfrac`` and the
            average molar mass relate the amounts.

        Raises
        ------
        PharmaPyValueError
            If ``concentration_basis`` is True, a positive amount controls,
            and the total concentration is zero: the fractions are then
            undefined and, for mass or moles, the volume infinite.

        Notes
        -----
        The first positive amount among ``mass``, ``vol`` and ``moles``
        controls; without one the stored amounts are left unchanged. A point
        of a profile without material (zero controlling amount) gets zero
        mass, volume and moles, never NaN, even though its fractions are
        undefined when its composition is zero. The
        factor 1000 converts [mol/L] times [m**3] to [mol] and [g/mol] to
        [kg/mol]. On the concentration basis, with ``mw_av = sum(mass_conc)
        / sum(conc)`` [g/mol], both rules coincide when the concentrations
        close ideal mixing.
        """
        mw_av = np.dot(molefrac, self.mw)  # [g/mol]
        if concentration_basis:
            mass_density = np.sum(mass_conc, axis=-1)  # [kg/m**3]
            molar_density = np.sum(conc, axis=-1)  # [mol/L] == [kmol/m**3]
            if _amount_supplied(mass):
                self.__reject_zero_total(mass_density, mass, 'mass')
                self.mass = mass  # [kg]
                self.vol = _amount_ratio(mass, mass_density)  # [m**3]
                self.moles = self.vol * molar_density * 1000  # [mol]

            elif _amount_supplied(vol):
                self.__reject_zero_total(molar_density, vol, 'vol')
                self.vol = vol  # [m**3]
                self.mass = vol * mass_density  # [kg]
                self.moles = vol * molar_density * 1000  # [mol]

            elif _amount_supplied(moles):
                self.__reject_zero_total(molar_density, moles, 'moles')
                self.moles = moles  # [mol]
                self.vol = _amount_ratio(moles, molar_density * 1000)  # [m**3]
                self.mass = self.vol * mass_density  # [kg]

        else:
            densMass = self.getDensityMix(massfrac)  # [kg/m**3], ideal mixing
            if _amount_supplied(mass):
                self.mass = mass  # [kg]
                self.vol = _amount_ratio(mass, densMass)  # [m**3]
                self.moles = _amount_ratio(mass, mw_av) * 1000  # [mol]

            elif _amount_supplied(vol):
                self.vol = vol  # [m**3]
                self.mass = _amount_product(vol, densMass)  # [kg]
                self.moles = _amount_ratio(self.mass, mw_av) * 1000  # [mol]

            elif _amount_supplied(moles):
                self.moles = moles  # [mol]
                self.mass = _amount_product(moles, mw_av) / 1000  # [kg]
                self.vol = _amount_ratio(self.mass, densMass)  # [m**3]

        self.mass_frac = massfrac
        self.mole_frac = molefrac
        self.mole_conc = conc
        self.mass_conc = mass_conc

        self.mw_av = mw_av
        self._concentration_basis = concentration_basis

    @staticmethod
    def __reject_zero_total(total_conc, amount, amount_name: str) -> None:
        """Reject an amount that authoritative zero concentrations cannot hold.

        Parameters
        ----------
        total_conc : float or numpy.ndarray
            Total concentration, ``sum(mass_conc)`` [kg/m**3] or
            ``sum(mole_conc)`` [mol/L], scalar or shape ``(num_points,)``.
        amount : float or numpy.ndarray
            Controlling amount, [kg], [m**3] or [mol], scalar or per point.
        amount_name : {'mass', 'vol', 'moles'}
            Name of the controlling positive amount, used in the message.

        Raises
        ------
        PharmaPyValueError
            If a point with a positive amount has an exactly zero total
            concentration.
        """
        if np.any((np.asarray(total_conc) == 0) & (np.asarray(amount) > 0)):
            raise PharmaPyValueError(
                f"Liquid concentrations sum to zero, so the positive "
                f"'{amount_name}' leaves the composition (and, for mass or "
                f"moles, the volume) undefined; supply nonzero "
                f"concentrations, or fractions with the amount.")

    def __calcComposition(self):
        """Derive the full composition from the constructor input.

        Notes
        -----
        Exactly one of ``mole_conc`` [mol/L], ``mass_conc`` [kg/m**3],
        ``mass_frac`` or ``mole_frac`` [-] is set on entry; the others are
        derived from it, completing the solvent concentration when
        ``ind_solv`` is set. The stored ``mass`` [kg], ``vol`` [m**3] and
        ``moles`` [mol] are then reconciled by ``__set_amounts`` on the
        concentration basis for concentrations without a solvent, and on
        the ideal-mixing basis otherwise.

        Raises
        ------
        PharmaPyValueError
            If authoritative concentrations total zero with a positive
            amount; see ``__set_amounts``.
        """
        has_solvent_closure = (
            self.ind_solv is not None or self._solv_mass_frac is not None
        )
        concentration_basis = (
            (self.mole_conc is not None or self.mass_conc is not None)
            and not has_solvent_closure)

        if self.mole_conc is not None:
            frac_out = self.conc_to_frac(
                self.mole_conc,
                solvent_ind=self.ind_solv,
                solv_mass_frac=self._solv_mass_frac,
            )
            if not has_solvent_closure:
                mass_frac, mole_frac = frac_out
                mole_conc = self.mole_conc
            else:
                mass_frac, mole_frac, mole_conc = frac_out

            mass_conc = mole_conc * self.mw  # kg / m3

        elif self.mass_conc is not None:
            frac_out = self.mass_conc_to_frac(
                self.mass_conc,
                solvent_ind=self.ind_solv,
                solv_mass_frac=self._solv_mass_frac,
            )

            if not has_solvent_closure:
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
                           mass_frac, mole_frac, mole_conc, mass_conc,
                           concentration_basis=concentration_basis)

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
            Every composition argument also accepts a profile shaped
            ``(num_points, num_species)``.
        vol : float or array-like, optional
            Liquid volume [m**3], scalar or per point ``(num_points,)``.
        mass : float or array-like, optional
            Liquid mass [kg], scalar or per point ``(num_points,)``.
        moles : float or array-like, optional
            Amount of liquid [mol], scalar or per point ``(num_points,)``.
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
        ``vol``, and ``moles``. On the concentration basis (``mole_conc``
        or ``mass_conc`` without a declared solvent) the inventory densities
        ``sum(mass_conc)`` [kg/m**3] and ``sum(mole_conc)`` [mol/L] relate
        them, so ``mole_conc * vol * 1000 == mole_frac * moles``; otherwise
        the ideal-mixing mass density [kg/m**3] and the average molar mass
        [g/mol] do (issue #439). A fraction update switches the phase to the
        ideal-mixing basis, a concentration update without a solvent to the
        concentration basis, and an update without a composition keeps the
        current basis (:attr:`concentration_basis`). Without a positive
        explicit amount, composition or intensive-state changes conserve
        the stored mass [kg], the liquid inventory integrated by the mass
        balances, and recompute volume [m**3] and moles [mol] from the
        updated composition on that basis.
        Profiles are reconciled point by point in the same way (issue
        #439): a two-dimensional composition shaped
        ``(num_points, num_species)`` uses the species axis, and amounts
        may be scalars, which apply to every point, or arrays of shape
        ``(num_points,)``. A composition update without a positive explicit
        amount retains the stored mass of each point, scalar or per point,
        and recomputes volume and moles per point, so the invariants above
        hold for every point. Earlier releases left profile amounts
        unchanged, contradicting the updated composition. An update of only
        the temperature or pressure of a profile leaves its amounts, already
        consistent with the unchanged composition, untouched.
        A no-argument update leaves all state unchanged. Zero amounts mean
        "not supplied" and cannot empty the phase through this method; a
        per-point amount is supplied when any point is positive, and its
        zero points, which hold no material, keep zero mass, volume and
        moles (their fractions are undefined if their composition is zero).

        Solvent handling matches the constructor: ``self.ind_solv`` is
        compared with ``None`` rather than tested for truth, so a solvent
        declared as the first species (index ``0``) is honored and the
        concentration converters' three-value return is unpacked correctly.

        Raises
        ------
        PharmaPyValueError
            If concentrations without a solvent total zero while a positive
            amount, explicit or retained, is reconciled; or if a per-point
            amount, explicit or retained, does not have one entry per point
            of a two-dimensional composition.
        MissingPropertyError
            A subclass of ``AttributeError`` and ``ValueError``. As in the
            constructor: if a species with a nonzero fraction or
            concentration in the updated composition has no ``rho_liq``
            [kg/m**3]; with a solvent closure, the solvent always needs it.
            Absent species need none.
        """

        explicit_amount = any(_amount_supplied(amount)
                              for amount in (mass, vol, moles))
        composition_supplied = any(value is not None for value in
                                   (mole_conc, mass_conc, mass_frac, mole_frac))
        if not explicit_amount:
            if not composition_supplied and temp is None and pres is None:
                return

        # Support solute concentrations alongside a solvent-blend mass_frac mapping (#336, #338).
        has_single_conc = (mole_conc is not None) ^ (mass_conc is not None)
        solv_mass_frac = None
        if has_single_conc and isinstance(mass_frac, Mapping) and mole_frac is None:
            conc_arg = mole_conc if mole_conc is not None else mass_conc
            if isinstance(conc_arg, Mapping):
                overlap = set(conc_arg.keys()) & set(mass_frac.keys())
                if overlap:
                    raise ValueError(
                        f"Species {sorted(overlap)} cannot be specified in both "
                        "solute concentrations and solvent mass_frac."
                    )
            solv_mass_frac = dict(mass_frac)
            self._solv_mass_frac = solv_mass_frac
            mass_frac = None

        mole_conc_was_mapping = isinstance(mole_conc, Mapping)
        mass_conc_was_mapping = isinstance(mass_conc, Mapping)

        mole_conc = self._coerce_composition_mapping(mole_conc, 'mole_conc')
        mass_conc = self._coerce_composition_mapping(mass_conc, 'mass_conc')
        mass_frac = self._coerce_composition_mapping(mass_frac, 'mass_frac')
        mole_frac = self._coerce_composition_mapping(mole_frac, 'mole_frac')

        if solv_mass_frac is None and getattr(self, '_solv_mass_frac', None) is not None:
            if (mole_conc_was_mapping or mass_conc_was_mapping) and np.ndim(
                mole_conc if mole_conc is not None else mass_conc
            ) == 1:
                conc_arr = mole_conc if mole_conc is not None else mass_conc
                solv_indices = [
                    self.name_species.index(sp)
                    for sp in self._solv_mass_frac.keys()
                    if sp in self.name_species
                ]
                if all(conc_arr[idx] == 0 for idx in solv_indices):
                    solv_mass_frac = self._solv_mass_frac

        has_solvent_closure = (
            self.ind_solv is not None or solv_mass_frac is not None
        )
        # Concentrations without a solvent are authoritative; fractions
        # and solvent-completed concentrations close ideal mixing.
        concentration_basis = not has_solvent_closure
        if mole_conc is not None:
            mole_conc = _as_float_array(mole_conc)  # [mol/L]
            frac_out = self.conc_to_frac(
                mole_conc,
                solvent_ind=self.ind_solv,
                solv_mass_frac=solv_mass_frac,
            )
            if has_solvent_closure:
                mass_frac, mole_frac, mole_conc = frac_out
            else:
                self._solv_mass_frac = None
                mass_frac, mole_frac = frac_out

            mass_conc = mole_conc * self.mw

        elif mass_conc is not None:
            mass_conc = _as_float_array(mass_conc)  # [kg/m**3]
            frac_out = self.mass_conc_to_frac(
                mass_conc,
                solvent_ind=self.ind_solv,
                solv_mass_frac=solv_mass_frac,
            )

            if has_solvent_closure:
                mass_frac, mole_frac, mass_conc = frac_out
            else:
                self._solv_mass_frac = None
                mass_frac, mole_frac = frac_out

            mole_conc = mass_conc / self.mw

        elif mass_frac is not None:
            self._solv_mass_frac = None
            concentration_basis = False
            mass_frac = _as_float_array(mass_frac)  # [-]
            mole_conc = self.frac_to_conc(mass_frac)
            mass_conc = mole_conc * self.mw
            mole_frac = self.frac_to_frac(mass_frac)

        elif mole_frac is not None:
            self._solv_mass_frac = None
            concentration_basis = False
            mole_frac = _as_float_array(mole_frac)  # [-]
            mole_conc = self.frac_to_conc(mole_frac=mole_frac)
            mass_conc = mole_conc * self.mw
            mass_frac = self.frac_to_frac(mole_frac=mole_frac)

        else:
            concentration_basis = self._concentration_basis
            mass_frac = self.mass_frac
            mole_frac = self.mole_frac
            mole_conc = self.mole_conc
            mass_conc = self.mass_conc

        profile = np.ndim(mass_frac) == 2 or np.ndim(self.mass) > 0
        if not explicit_amount and (composition_supplied or not profile):
            # [kg], authoritative retained liquid inventory, per point for
            # a profile (issue #439)
            mass = self.mass

        num_points = len(mass_frac) if np.ndim(mass_frac) == 2 else None
        for name, amount in (('mass', mass), ('vol', vol), ('moles', moles)):
            if (num_points is not None and np.ndim(amount) > 0
                    and np.shape(amount) != (num_points,)):
                raise PharmaPyValueError(
                    f"The liquid profile has {num_points} points but its "
                    f"'{name}' has shape {np.shape(amount)}; supply one "
                    f"amount per point or a scalar.")

        if temp is not None:
            self.temp = (float(temp) if np.ndim(temp) == 0
                         else _as_float_array(temp))  # [K]

        if pres is not None:
            self.pres = pres

        self.__set_amounts(mass, vol, moles, mass_frac, mole_frac,
                           mole_conc, mass_conc,
                           concentration_basis=concentration_basis)

    def getDensity(self, mass_frac: Optional[np.ndarray] = None,
                   mole_frac: Optional[np.ndarray] = None,
                   temp: Optional[float] = None, basis: str = 'mass',
                   weights: Optional[np.ndarray] = None
                   ) -> Union[float, np.ndarray]:
        """Return the ideal-mixing liquid density on the requested basis.

        Parameters
        ----------
        mass_frac, mole_frac : ndarray, optional
            Liquid species fractions [-], shape ``(num_species,)`` or
            ``(num_points, num_species)``. The stored phase composition is
            used when neither is supplied.
        temp : float, optional
            Temperature [K]; defaults to the phase temperature. Pure liquid
            densities are temperature independent, so it does not change the
            result.
        basis : {'mass', 'mole'}, optional
            Physical basis of the returned density; default mass.
        weights : ndarray, optional
            Presence weights [-], with exactly the shape of the fractions,
            that decide in their place which species need ``rho_liq``; see
            ``ThermoPhysicalManager.getDensityMix``.

        Returns
        -------
        float or ndarray
            Liquid density, [kg/m**3] for ``basis='mass'`` and [kmol/m**3]
            (equivalently [mol/L]) for ``basis='mole'``.

        Raises
        ------
        ValueError
            If ``basis`` is neither 'mass' nor 'mole', or if ``weights``
            does not have exactly the shape of the fractions, raised by
            ``ThermoPhysicalManager.getDensityMix``.
        MissingPropertyError
            A subclass of ``AttributeError`` and ``ValueError``. If a species
            with a nonzero fraction (or, with ``weights``, a nonzero weight)
            in some row has no ``rho_liq``; absent species need none and
            occupy no volume.

        Notes
        -----
        This is the ideal-mixing property model, a function of composition
        only; model equations and their Jacobians rely on that. It is not
        the density of the stored inventory when :attr:`concentration_basis`
        is True and the concentrations do not close ideal mixing; use
        :meth:`getInventoryDensity` to convert this phase's own amounts.
        """
        if temp is None:
            temp = self.temp

        if mass_frac is None and mole_frac is None:
            mass_frac = self.mass_frac
            mole_frac = self.mole_frac

        rhoLiq = self.getDensityMix(mass_frac, mole_frac, phase='liquid',
                                    basis=basis, temp=temp, weights=weights)

        return rhoLiq

    def getInventoryDensity(self, basis: str = 'mass'
                            ) -> Union[float, np.ndarray]:
        """Return the density that relates this phase's stored amounts.

        Parameters
        ----------
        basis : {'mass', 'mole'}, optional
            Physical basis of the returned density; default mass.

        Returns
        -------
        float or numpy.ndarray
            Inventory density, [kg/m**3] for ``basis='mass'`` and [mol/L]
            (equivalently [kmol/m**3]) for ``basis='mole'``. Scalar for a
            one-dimensional composition, shape ``(num_points,)`` for a
            profile.

        Raises
        ------
        ValueError
            If ``basis`` is neither 'mass' nor 'mole'.

        Notes
        -----
        On the concentration basis (:attr:`concentration_basis`) this is
        ``sum(mass_conc)`` [kg/m**3] or ``sum(mole_conc)`` [mol/L], so
        ``mass / vol`` and ``moles / (1000 * vol)`` reproduce it. Otherwise
        it is :meth:`getDensity` of the stored composition, bit for bit.
        Use it to convert this phase's (or stream's, per second) own mass or
        moles to a volume; use :meth:`getDensity` for a composition-only
        property model.
        """
        if basis not in ('mass', 'mole'):
            raise ValueError("basis must be 'mass' or 'mole'")

        if not self._concentration_basis:
            return self.getDensity(basis=basis)

        if basis == 'mass':
            return np.sum(self.mass_conc, axis=-1)  # [kg/m**3]

        return np.sum(self.mole_conc, axis=-1)  # [mol/L]

    def getCp(self, temp: Optional[ArrayLike] = None,
              mass_frac: Optional[np.ndarray] = None,
              mole_frac: Optional[np.ndarray] = None,
              basis: str = 'mass',
              weights: Optional[np.ndarray] = None
              ) -> Union[float, np.ndarray]:
        """Return liquid heat capacity on the requested mixture basis.

        Parameters
        ----------
        temp : float or array-like, optional
            Temperature [K], scalar or shape (num_temperatures,); defaults
            to the stored phase temperature.
        mass_frac, mole_frac : numpy.ndarray, optional
            Species fractions [-], shape (num_species,) or paired profiles
            (num_temperatures, num_species). Defaults to stored composition.
            Fractions matching the requested basis take precedence.
        basis : {'mass', 'mole'}, optional
            Mixture basis; defaults to mass, consistently with getEnthalpy.
            Callers requiring molar heat capacity must request 'mole'.
        weights : numpy.ndarray, optional
            Presence weights [-], with exactly the shape of the fractions on
            the requested basis, that decide in their place which species
            need ``cp_liq``; see ``ThermoPhysicalManager.getCpMix``.

        Returns
        -------
        float or numpy.ndarray
            Heat capacity [J/kg/K] for mass or [J/mol/K] for mole. Scalar
            temperature or a one-element temperature array with fixed
            composition returns a scalar. Two-dimensional composition profiles
            retain the temperature row axis, including one-row profiles.

        Raises
        ------
        ValueError
            If basis is neither 'mass' nor 'mole', or if ``weights`` does
            not have exactly the shape of the fractions on the requested
            basis, raised by ``ThermoPhysicalManager.getCpMix``.
        MissingPropertyError
            A subclass of ``AttributeError`` and ``ValueError``. If a species
            with a nonzero fraction (or, with ``weights``, a nonzero weight)
            in some row has no ``cp_liq``. A species whose fraction (weight)
            is zero in every row needs none and, without it, contributes
            exactly zero; with ``weights``, a zero-weight species that has
            ``cp_liq`` still contributes its fraction times its heat
            capacity.
        """
        if temp is None:
            temp = self.temp

        if mass_frac is None and mole_frac is None:
            mass_frac = self.mass_frac
            mole_frac = self.mole_frac

        cpLiq = super().getCpMix(temp, mass_frac, mole_frac, basis=basis,
                                 weights=weights)

        return cpLiq

    def getEnthalpy(self, temp: Optional[ArrayLike] = None,
                    temp_ref: float = 298.15,
                    mass_frac: Optional[np.ndarray] = None,
                    mole_frac: Optional[np.ndarray] = None,
                    total_h: bool = True, basis: str = 'mass',
                    weights: Optional[np.ndarray] = None
                    ) -> Union[float, np.ndarray]:
        """Calculate liquid sensible enthalpy relative to a reference.

        Parameters
        ----------
        temp : float or ndarray, optional
            Temperature [K], scalar or shape ``(num_temps,)``. The phase
            temperature is used when ``None``.
        temp_ref : float, optional
            Lower limit of the ``cp_liq`` integral [K]; default 298.15 K.
        mass_frac, mole_frac : ndarray, optional
            Liquid species fractions [-], shape ``(num_species,)`` or
            ``(num_rows, num_species)``. The stored phase composition is used
            when neither is supplied.
        total_h : bool, optional
            If ``True`` (default), return the fraction-weighted mixture
            enthalpy. If ``False``, return individual species enthalpies.
        basis : {'mass', 'mole'}, optional
            Physical basis of the returned enthalpy and of the weighting
            fractions; default mass.
        weights : ndarray, optional
            Amounts or presence flags [any basis] that decide which species
            need ``cp_liq``, shape ``(num_species,)`` or ``(num_rows,
            num_species)``, for both ``total_h`` modes; with
            ``total_h=True`` exactly the shape of the weighting fractions.
            See ``ThermoPhysicalManager.getEnthalpy``.

        Returns
        -------
        float or ndarray
            Liquid sensible enthalpy, [J/kg] for ``basis='mass'`` and [J/mol]
            for ``basis='mole'``. With ``total_h=True``, one value per
            temperature or composition row, broadcast as in
            ``ThermoPhysicalManager.getEnthalpy``: shape ``(num_temps,)`` for
            a fixed composition, ``(num_rows,)`` for a single temperature with
            a profile, and row-paired when ``num_temps == num_rows``. A scalar
            is returned only for a single temperature with a fixed composition
            or a one-row profile. With ``total_h=False``, shape
            ``(num_temps, num_species)``, including ``(1, num_species)`` for a
            scalar temperature.

        Raises
        ------
        ValueError
            If ``basis`` is neither 'mass' nor 'mole', for both ``total_h``
            modes, or, with ``total_h=True``, if ``weights`` does not have
            exactly the shape of the weighting fractions, raised by
            ``ThermoPhysicalManager.getEnthalpy``.
        MissingPropertyError
            A subclass of ``AttributeError`` and ``ValueError``.
            If a species that the result needs has no ``cp_liq`` data; see
            ``ThermoPhysicalManager.getEnthalpy``.
            Also raised, whatever the fractions or weights, if the property is
            stored as one scalar per species instead of a coefficient list (see
            ``_property_rows``).
        """
        if mass_frac is None and mole_frac is None:
            mass_frac = self.mass_frac
            mole_frac = self.mole_frac

        if temp is None:
            temp = self.temp

        hLiq = super().getEnthalpy(temp, temp_ref, mass_frac, mole_frac,
                                   phase='liquid', total_h=total_h,
                                   basis=basis, weights=weights)

        return hLiq

    def getBubblePoint(self, pres: Optional[float] = None,
                       mass_frac: Optional[np.ndarray] = None,
                       mole_frac: Optional[np.ndarray] = None,
                       thermo_method: str = 'ideal',
                       y_vap: bool = False,
                       weights: Optional[np.ndarray] = None
                       ) -> Union[float, tuple]:
        """Solve for the liquid bubble-point temperature.

        Parameters
        ----------
        pres : float, optional
            Pressure [Pa]; the phase pressure when ``None``.
        mass_frac, mole_frac : ndarray, optional
            Liquid fractions [-], shape ``(num_species,)``. Mole fractions
            are converted from mass fractions when only those are given; the
            phase composition is used when neither is.
        thermo_method : {'ideal', 'UNIFAC', 'UNIQUAC'}, optional
            Liquid activity model passed to ``getKeqVLE``.
        y_vap : bool, optional
            Also return the incipient vapor composition.
        weights : ndarray, optional
            Presence weights [-] with the shape of the mole fractions. When
            given, only species with a nonzero weight enter the bubble-point
            sum, its Antoine seed, the incipient vapor and a non-ideal
            activity model, and only they need VLE and activity data (see
            ``getKeqVLE``); the others count as absent, with exactly zero
            vapor fraction and the neutral activity coefficient 1.
            Solver-based units pass a structural presence mask here, so that
            round-off fractions of absent species need no data.

        Returns
        -------
        temp_bubble : float
            Bubble-point temperature [K].
        y_frac : ndarray
            Incipient vapor mole fractions [-], shape ``(num_species,)``;
            only when ``y_vap`` is True.

        Raises
        ------
        ValueError
            If ``weights`` does not have exactly the shape of the mole
            fractions, before any evaluation.
        MissingPropertyError
            A subclass of ``AttributeError`` and ``ValueError``.
            From ``getKeqVLE`` at the converged root, if a species with a
            nonzero fraction (or weight) has no ``p_vap`` coefficients at or
            below its critical temperature. Already from the Newton
            iterates, which evaluate the same ratios, if a species with a
            nonzero fraction (or weight) is above its critical temperature
            at an iterate without
            ``henry_constant`` [Pa], if no species supplies ``t_crit``, if no
            species supplies ``henry_constant`` while some species (with
            ``weights``, some weighted species) is supercritical at an
            iterate, or if ``p_vap`` is stored as one scalar per species
            (from the Antoine seed). With ``weights`` and
            ``thermo_method='UNIQUAC'``, also if a weighted species lacks
            ``ri``, ``qi`` or (when the database has the key) ``qip``; UNIFAC
            group counts are taken as given (see ``getKeqVLE``).
        RuntimeError
            If the Newton iteration does not converge.

        Notes
        -----
        The Newton seed is ``sum_i x_i * Tsat_i(pres)`` [K], with
        ``weights`` over the species with a nonzero weight only. The seed
        and the Newton iterates evaluate a species without Antoine data
        with zero coefficients, as releases before #414 did, so the
        convergence path is unchanged (see ``_antoine_seed``). The
        converged root is then checked with ``getKeqVLE``: a species with a
        nonzero fraction (or, with ``weights``, a nonzero weight) that is at
        or below its critical temperature there must have ``p_vap``.
        """
        if mass_frac is None and mole_frac is None:
            mole_frac = self.mole_frac

        elif mole_frac is None:
            mole_frac = self.frac_to_frac(mass_frac=mass_frac)

        mole_frac = np.asarray(mole_frac, dtype=float)  # [-], also from a list

        if pres is None:
            pres = self.pres

        # Absent (zero-weight) species leave the sum and the vapor exactly;
        # their ratios may be NaN placeholders (see getKeqVLE).
        if weights is None:
            used = np.ones(np.shape(mole_frac), dtype=bool)  # [-]
        else:
            weights = self._validate_weights_shape(weights, mole_frac,
                                                   'mole_frac')
            used = weights != 0  # [-]

        def bubble_fn(temp: float) -> float:
            """Evaluate the bubble-point residual at a Newton iterate.

            Parameters
            ----------
            temp : float
                Temperature iterate [K].

            Returns
            -------
            float
                ``sum_i x_i (K_i - 1)`` over the used species [-]; zero at
                the bubble point.
            """
            # [-], iterate only
            k_vals = self._vle_ratios(temp, pres, mole_frac, thermo_method,
                                      check=False, weights=weights)

            if weights is None:
                obj = np.dot(mole_frac, (k_vals - 1))
            else:
                obj = np.dot(mole_frac[used], (k_vals[used] - 1))

            return obj

        # Absent species are left out of the seed too: their saturation
        # temperature may be infinite, and 0 * inf is NaN.
        temp_seed = self._antoine_seed(
            mole_frac, pres=pres,
            used=None if weights is None else used)  # [K]
        temp_bubble = newton(bubble_fn, temp_seed, full_output=False)

        # Check the data the converged root needs.
        k_vals = self.getKeqVLE(temp_bubble, pres, mole_frac,
                                gamma_model=thermo_method,
                                weights=weights)  # [-]

        if y_vap:
            if weights is None:
                y_frac = k_vals * mole_frac
            else:
                y_frac = np.where(used, k_vals * mole_frac, 0.0)  # [-]

            return temp_bubble, y_frac
        else:
            return temp_bubble

    def getBubblePressure(self, temp: Optional[float] = None,
                          mass_frac: Optional[np.ndarray] = None,
                          mole_frac: Optional[np.ndarray] = None,
                          thermo_method: str = 'ideal',
                          y_vap: bool = False) -> float:
            """Solve for the liquid bubble-point pressure.

            Parameters
            ----------
            temp : float, optional
                Temperature [K]; the phase temperature when ``None``.
            mass_frac, mole_frac : ndarray, optional
                Liquid fractions [-], shape ``(num_species,)``, resolved as
                in ``getBubblePoint``.
            thermo_method : {'ideal', 'UNIFAC', 'UNIQUAC'}, optional
                Liquid activity model passed to ``getKeqVLE``.
            y_vap : bool, optional
                Accepted for signature symmetry; unused.

            Returns
            -------
            float
                Bubble-point pressure [Pa].

            Raises
            ------
            MissingPropertyError
                A subclass of ``AttributeError`` and ``ValueError``.
                From ``getKeqVLE`` at the converged root, if a species with
                a nonzero fraction has no ``p_vap`` coefficients at or below
                its critical temperature. Already from the Newton iterates,
                which evaluate the same ratios, if a species with a nonzero
                fraction is above its critical temperature at an iterate
                without ``henry_constant`` [Pa], if no species supplies
                ``t_crit``, if no species supplies ``henry_constant`` while
                some species is supercritical at an iterate, or if ``p_vap``
                is stored as one scalar per species (from the Antoine seed).
            RuntimeError
                If the Newton iteration does not converge.

            Notes
            -----
            The Newton seed is ``sum_i x_i * psat_i(temp)`` [Pa]. Seed and
            iterates treat missing Antoine data as in ``getBubblePoint``, and
            the converged root is checked with ``getKeqVLE``.
            """
            if mass_frac is None and mole_frac is None:
                mole_frac = self.mole_frac

            elif mole_frac is None:
                mole_frac = self.frac_to_frac(mass_frac=mass_frac)

            if temp is None:
                temp = self.temp

            def bubble_fn(pr):
                k_vals = self._vle_ratios(temp, pr, mole_frac, thermo_method,
                                          check=False)  # [-], iterate only

                obj = np.dot(mole_frac, (k_vals - 1))

                return obj

            pres_seed = self._antoine_seed(mole_frac, temp=temp)  # [Pa]
            pres_bubble = newton(bubble_fn, pres_seed, full_output=False)

            # Check the data the converged root needs.
            self.getKeqVLE(temp, pres_bubble, mole_frac,
                           gamma_model=thermo_method)

            return pres_bubble

    def getProps(self, basis: str = 'mass') -> tuple:
        """Return Cp, density, and enthalpy at the current phase state.

        Parameters
        ----------
        basis : {'mass', 'mole'}, optional
            Property basis; default mass.

        Returns
        -------
        tuple
            Cp, density, and enthalpy, in that order. Mass units are [J/kg/K],
            [kg/m**3], and [J/kg]. Molar units follow the phase providers:
            [J/mol/K], [kmol/m**3] (equivalently [mol/L]), and [J/mol].
            Scalars are returned for a scalar state; profile shapes follow
            the individual providers. Enthalpy uses their 298.15 K reference.

        Raises
        ------
        ValueError
            If basis is neither 'mass' nor 'mole'.
        """
        if basis not in ('mass', 'mole'):
            raise ValueError("basis must be 'mass' or 'mole'")
        cp = self.getCp(basis=basis)  # [J/kg/K] or [J/mol/K]
        rho = self.getDensity(basis=basis)  # [kg/m**3] or [kmol/m**3]
        enthalpy = self.getEnthalpy(basis=basis)  # [J/kg] or [J/mol]
        return cp, rho, enthalpy

    def getActivityCoeff(self, method: str = 'ideal',
                         mole_frac: Optional[np.ndarray] = None,
                         temp: Optional[Union[float, np.ndarray]] = None,
                         *, weights: Optional[np.ndarray] = None
                         ) -> np.ndarray:
        """Return liquid activity coefficients for the selected model.

        Parameters
        ----------
        method : {'ideal', 'UNIFAC', 'UNIQUAC'}, optional
            Case-sensitive activity model. Its property parameters must be
            present in the database; UNIQUAC uses ``qi`` when ``qip`` is absent.
        mole_frac : ndarray, optional
            Liquid mole fractions [-], shape (num_species,) or
            (num_points, num_species); defaults to the phase composition.
        temp : float or ndarray, optional
            Temperature [K], scalar or paired profile (num_points,);
            defaults to the phase temperature.
        weights : ndarray, optional
            Keyword only. Presence weights [-] with exactly the shape of the
            mole fractions. When given, the model is evaluated on the
            species with a nonzero weight only, as if the others were
            absent; those others need no activity parameters and get the
            neutral coefficient of exactly 1, so that a ratio of the
            coefficients of two liquids (the extractors' default
            distribution ratio) is exactly 1 for them. Solver-based units
            pass a structural presence mask here (through ``getKeqVLE``,
            or directly for the extractors). See
            ``ThermoPhysicalManager._activity_coefficients``.

        Returns
        -------
        ndarray
            Activity coefficients [-] in the supplied composition order
            and shape. The ideal model returns ones.

        Raises
        ------
        ValueError
            If ``method`` is not a supported selector. Public parameter names
            and defaults are unchanged; unknown names no longer select UNIFAC.
            Also if ``weights`` does not have the shape of the mole
            fractions.
        MissingPropertyError
            A subclass of ``AttributeError`` and ``ValueError``. With
            ``weights`` and UNIQUAC, if a weighted species lacks ``ri``,
            ``qi`` or (when the database has the key) ``qip``.
        """
        validate_activity_model(method, param_name='method')

        if mole_frac is None:
            mole_frac = self.mole_frac

        if temp is None:
            temp = self.temp

        if weights is not None:
            weights = self._validate_weights_shape(weights, mole_frac,
                                                   'mole_frac')

        return self._activity_coefficients(method, mole_frac, temp,
                                           weights)  # [-]

    def getViscosity(self, temp: Optional[ArrayLike] = None,
                     mass_frac: Optional[np.ndarray] = None,
                     mole_frac: Optional[np.ndarray] = None
                     ) -> Union[float, np.ndarray]:
        """Return the liquid mixture viscosity.

        Parameters
        ----------
        temp : float or array-like, optional
            Temperature [K], scalar or shape ``(num_temps,)``; the phase
            temperature when ``None``.
        mass_frac, mole_frac : numpy.ndarray, optional
            Fractions [-], shape ``(num_species,)`` or
            ``(num_points, num_species)``; mole fractions take precedence,
            and the phase mole fractions are used when neither is supplied.

        Returns
        -------
        float or numpy.ndarray
            Viscosity [Pa*s], shaped as in ``getViscosityMix``: a scalar for
            one temperature and one composition, otherwise one value per
            temperature or composition row, row-paired when both are arrays
            of equal length.

        Raises
        ------
        MissingPropertyError
            A subclass of ``AttributeError`` and ``ValueError``. If a
            species with a nonzero fraction in some row has no ``visc_liq``
            coefficients.
            Also raised, whatever the fractions or weights, if the property is
            stored as one scalar per species instead of a coefficient list (see
            ``_property_rows``).
        """
        viscosity = self.getViscosityMix(temp, mass_frac, mole_frac,
                                         phase='liquid')

        return viscosity

    def getSurfTensionPure(self, temp: Optional[float] = None) -> np.ndarray:
        """Return the pure-component surface tensions.

        Parameters
        ----------
        temp : float, optional
            Temperature [K]. Accepted for interface compatibility and
            ignored: the stored surface tensions are constants.

        Returns
        -------
        numpy.ndarray
            Surface tensions [N/m], shape ``(num_species,)``, in database
            order. A copy: modifying it leaves the stored data unchanged.

        Raises
        ------
        MissingPropertyError
            A subclass of ``AttributeError`` and ``ValueError``. If any
            species has no ``surf_tension``; every species is returned, so
            every species needs it.

        Notes
        -----
        Earlier releases returned 0 N/m for a species without data and
        wrote those zeros into the stored ``surf_tension`` array.
        """
        surface_pure = self._scalar_values('surf_tension')  # [N/m]
        num_sp = len(surface_pure)
        self._check_species_rows(
            'surf_tension', surface_pure, np.arange(num_sp),
            np.ones(num_sp, dtype=bool),
            'are evaluated as pure components, so their surface tension')

        return surface_pure.copy()  # [N/m]

    def getSurfTension(self, mass_frac: Optional[np.ndarray] = None,
                       mole_frac: Optional[np.ndarray] = None,
                       temp: Optional[float] = None
                       ) -> Union[float, np.ndarray]:
        """Return the mass-fraction-weighted liquid surface tension.

        Parameters
        ----------
        mass_frac : numpy.ndarray, optional
            Liquid mass fractions [-], shape ``(num_species,)`` or
            ``(num_points, num_species)``; the phase mass fractions when
            ``None``.
        mole_frac : numpy.ndarray, optional
            Unused; retained for interface compatibility.
        temp : float, optional
            Temperature [K]. Unused: the stored surface tensions are
            constants.

        Returns
        -------
        float or numpy.ndarray
            ``sum_i w_i * sigma_i`` [N/m]: a scalar for one composition,
            shape ``(num_points,)`` for a composition profile.

        Raises
        ------
        MissingPropertyError
            A subclass of ``AttributeError`` and ``ValueError``. If a
            species with a nonzero mass fraction in some row has no
            ``surf_tension``. A species whose fraction is zero in every row
            needs none and contributes exactly zero. Nonzero is bitwise: a
            round-off-level fraction counts as present.

        Notes
        -----
        Earlier releases weighted a missing surface tension as 0 N/m, also
        for present species.
        """
        if mass_frac is None:
            mass_frac = self.mass_frac

        surface_pure = self._scalar_values('surf_tension')  # [N/m]
        missing = self._check_species_rows(
            'surf_tension', surface_pure, np.arange(len(surface_pure)),
            self._needed_by_weights(mass_frac, len(surface_pure)),
            'carry a nonzero mass fraction, so the mixture surface tension')
        surface_used = np.where(missing, 0.0, surface_pure)  # [N/m]
        surfaceMix = np.dot(mass_frac, surface_used)  # [N/m]

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
                 mass_frac: Optional[Union[ArrayLike, Mapping[str, ArrayLike]]] = None,
                 mole_frac: Optional[Union[ArrayLike, Mapping[str, ArrayLike]]] = None,
                 mole_conc: Optional[Union[ArrayLike, Mapping[str, ArrayLike]]] = None,
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
        mass_frac : array-like or mapping of str to float/array-like, optional
            Species mass fractions [-], shape ``(num_species,)`` or
            ``(num_points, num_species)``, or a species-keyed mapping.
        mole_frac : array-like or mapping of str to float/array-like, optional
            Species mole fractions [-], shape ``(num_species,)`` or
            ``(num_points, num_species)``, or a species-keyed mapping.
        mole_conc : array-like or mapping of str to float/array-like, optional
            Species molar concentrations [mol/L], shape ``(num_species,)`` or
            ``(num_points, num_species)``, or a species-keyed mapping.
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
        authoritative for every composition row; the other amounts follow
        from the ideal-gas EOS. Composition-dependent amounts have one value
        per row; a shared amount independent of composition remains scalar.
        If all amounts are zero, the stored amounts remain zero. A positive
        molar amount with zero mole fractions has zero mass and an EOS volume.
        """

        super().__init__(path_thermo)

        mass_frac = self._coerce_composition_mapping(mass_frac, 'mass_frac')
        mole_frac = self._coerce_composition_mapping(mole_frac, 'mole_frac')
        mole_conc = self._coerce_composition_mapping(mole_conc, 'mole_conc')

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

    def __set_amounts(self, mass: float, vol: float,
                      moles: Union[float, np.ndarray],
                      massfrac: np.ndarray, molefrac: np.ndarray,
                      temp: float, pres: float) -> None:
        """Store composition and reconcile positive amounts with the gas EOS.

        Parameters
        ----------
        mass, vol : float
            Explicit mass [kg] and volume [m**3]. The first positive value in
            mass, volume, moles order controls; zeros leave amounts alone.
        moles : float or ndarray
            Molar amount [mol], scalar or conserved per-row inventory with
            shape ``(num_points,)`` for a composition profile.
        massfrac, molefrac : ndarray
            Species mass and mole fractions [-], shape ``(num_species,)`` or
            ``(num_points, num_species)``.
        temp, pres : float
            Proposed vapor temperature [K] and pressure [Pa], stored only
            after the amount/composition validation succeeds.

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
        if (mass > 0 or vol > 0) and np.any(mw_av == 0):
            raise ValueError("Positive mass or volume requires a nonzero "
                             "composition; zero mixture molar mass was given")

        self.temp = temp  # [K]
        self.pres = pres  # [Pa]
        self.mass_frac = massfrac  # [-]
        self.mole_frac = molefrac  # [-]
        molar_density = np.asarray(pres) / (VAPOR_GAS_CONSTANT * temp) / 1000  # [mol/L], 1000 L/m**3
        self.mole_conc = molefrac * molar_density[..., None]  # [mol/L], species on the last axis
        self.mass_conc = self.mole_conc * self.mw  # [kg/m**3], g/L equals kg/m**3
        self.mw_av = mw_av  # [g/mol]

        if mass > 0 or vol > 0 or np.any(moles > 0):
            molar_volume = VAPOR_GAS_CONSTANT * self.temp / self.pres  # [m**3/mol]
            if mass > 0:
                self.moles = mass * 1000 / self.mw_av  # [mol], 1000 g/kg
            elif vol > 0:
                self.moles = vol / molar_volume  # [mol]
            else:
                self.moles = moles  # [mol]
            self.mass = self.moles * self.mw_av / 1000  # [kg], 1000 g/kg
            self.vol = self.moles * molar_volume  # [m**3]

    def updatePhase(self, mole_conc: Optional[Union[ArrayLike, Mapping[str, ArrayLike]]] = None,
                    mass_conc: Optional[Union[ArrayLike, Mapping[str, ArrayLike]]] = None,
                    mass_frac: Optional[Union[ArrayLike, Mapping[str, ArrayLike]]] = None,
                    mole_frac: Optional[Union[ArrayLike, Mapping[str, ArrayLike]]] = None,
                    vol: float = 0, mass: float = 0, moles: float = 0,
                    temp: Optional[float] = None,
                    pres: Optional[float] = None) -> None:
        """Update vapor composition, intensive state, and ideal-gas amounts.

        Parameters
        ----------
        mole_conc, mass_conc : array-like or mapping of str to float/array-like, optional
            Species concentrations, shape ``(num_species,)`` or
            ``(num_points, num_species)``, on molar [mol/L] and mass
            [kg/m**3] bases, respectively, or species-keyed mappings. No
            solvent is inferred. Supplied concentrations determine normalized
            composition; their total is reconciled with the specified gas
            temperature and pressure.
        mass_frac, mole_frac : array-like or mapping of str to float/array-like, optional
            Species mass and mole fractions [-], shape ``(num_species,)`` or
            ``(num_points, num_species)``, or species-keyed mappings.
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
        Rejected negative amounts and zero-molar-mass requests leave the
        stored state unchanged. A no-argument update leaves all amounts
        unchanged. Composition inputs take precedence in the order mole_conc,
        mass_conc, mass_frac, mole_frac.
        All concentrations use the ideal-gas EOS, including zero-inventory
        phases. Stream amounts use per-second units.
        """
        for name, amount in (('mass', mass), ('vol', vol), ('moles', moles)):
            # amount uses [kg], [m**3], or [mol], respectively (rates on streams).
            if amount < 0:
                raise ValueError(f"{name} must be nonnegative")

        mole_conc = self._coerce_composition_mapping(mole_conc, 'mole_conc')
        mass_conc = self._coerce_composition_mapping(mass_conc, 'mass_conc')
        mass_frac = self._coerce_composition_mapping(mass_frac, 'mass_frac')
        mole_frac = self._coerce_composition_mapping(mole_frac, 'mole_frac')

        state_changed = any(value is not None for value in
                            (mole_conc, mass_conc, mass_frac, mole_frac,
                             temp, pres))
        if not state_changed and mass == vol == moles == 0:
            return
        temp = self.temp if temp is None else float(temp)  # [K]
        pres = self.pres if pres is None else pres  # [Pa]
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
            mole_frac = self.frac_to_frac(mass_frac)  # [-]
        elif mole_frac is not None:
            mole_frac = _as_float_array(mole_frac)  # [-]
            if np.any(mole_frac):
                mass_frac = self.frac_to_frac(mole_frac=mole_frac)  # [-]
            else:
                mass_frac = np.zeros_like(mole_frac)  # [-], empty placeholder
        else:
            mass_frac = self.mass_frac  # [-]
            mole_frac = self.mole_frac  # [-]
            mole_conc = self.mole_conc  # [mol/L]
            mass_conc = self.mass_conc  # [kg/m**3]

        self.__set_amounts(mass, vol, moles, mass_frac, mole_frac, temp, pres)

    def getCp(self, temp: ArrayLike,
              mass_frac: Optional[np.ndarray] = None,
              mole_frac: Optional[np.ndarray] = None,
              basis: str = 'mass',
              weights: Optional[np.ndarray] = None
              ) -> Union[float, np.ndarray]:
        """Return the vapor mixture heat capacity on the requested basis.

        Parameters
        ----------
        temp : float or array-like
            Temperature [K], scalar or shape ``(num_temps,)``.
        mass_frac, mole_frac : numpy.ndarray, optional
            Vapor fractions [-], shape ``(num_species,)`` or
            ``(num_points, num_species)``; the stored mass fractions when
            neither is given.
        basis : {'mass', 'mole'}, optional
            Basis of the result; default mass.
        weights : numpy.ndarray, optional
            Presence weights [-], with exactly the shape of the fractions on
            the requested basis, that decide in their place which species
            need ``cp_vapor``; see ``ThermoPhysicalManager.getCpMix``.

        Returns
        -------
        float or numpy.ndarray
            Mixture heat capacity from ``cp_vapor``, [J/kg/K] or [J/mol/K],
            shaped as in ``ThermoPhysicalManager.getCpMix``.

        Raises
        ------
        ValueError
            If ``basis`` is neither 'mass' nor 'mole', or if ``weights``
            does not have exactly the shape of the fractions on the
            requested basis.
        MissingPropertyError
            A subclass of ``AttributeError`` and ``ValueError``. If a species
            with a nonzero fraction (or, with ``weights``, a nonzero weight)
            in some row has no ``cp_vapor`` coefficients. Also raised,
            whatever the fractions, if ``cp_vapor`` is stored as one scalar
            per species instead of a coefficient list (see
            ``_property_rows``).
        """
        if mass_frac is None and mole_frac is None:
            mass_frac = self.mass_frac

        cpMix = self.getCpMix(temp, mass_frac, mole_frac, phase='vapor',
                              basis=basis, weights=weights)

        return cpMix

    def _classify_criticality(
            self, temp: ArrayLike) -> Tuple[np.ndarray, np.ndarray]:
        """Validate temperatures and classify every species at each of them.

        Parameters
        ----------
        temp : float or array-like
            Scalar temperature or one-dimensional array of independent
            temperatures [K], such as the spatial nodes of a distributed
            model. Entries are never paired with species.

        Returns
        -------
        temp_values : ndarray
            Requested temperatures as floats [K], shape
            ``(num_temperatures,)``; a scalar becomes one entry.
        is_supercritical : ndarray of bool
            Criticality mask [-], shape ``(num_temperatures, num_species)``.
            Entry ``[i, j]`` is ``True`` when ``temp_values[i]`` lies strictly
            above ``t_crit[j]``.

        Raises
        ------
        ValueError
            If ``temp`` has more than one dimension, is empty, or contains a
            non-finite value.
        MissingPropertyError
            A subclass of ``AttributeError`` and ``ValueError``. If no
            species supplies ``t_crit``; the error names every species.

        Notes
        -----
        Supercritical means strictly ``T > Tc``. Every other entry, including
        ``T == Tc``, is subcritical, so the two classes partition every
        ``(temperature, species)`` pair. Assigning equality to the subcritical
        side matches the earlier all-subcritical scalar path, which earlier
        releases used whenever no species was supercritical, and agrees with
        the ``temp > t_crit`` supercritical test of the evaporator models.
        Earlier releases mishandled ``T == Tc`` only when another species was
        supercritical at the same temperature. The
        Watson latent heat is exactly zero at ``Tc``, so the latent term is
        continuous there; the sensible term switches from liquid to vapor heat
        capacity at ``Tc``, so the model's species enthalpy jumps at ``Tc``
        whichever side the equality is assigned to.

        A species without critical data, whose ``t_crit`` is parsed as NaN,
        compares as not supercritical at every temperature, so it always uses
        liquid sensible heat. ``getHeatVaporization`` assigns it zero latent
        heat, the established treatment of such a non-condensing species.
        """
        temp_values = np.asarray(temp, dtype=float)  # [K]
        if temp_values.ndim > 1:
            raise ValueError(
                "temp must be a scalar or a one-dimensional array of "
                f"independent temperatures [K]; got shape {temp_values.shape}")
        temp_values = np.atleast_1d(temp_values)  # [K]
        if temp_values.size == 0:
            raise ValueError("temp must contain at least one temperature [K]")
        non_finite = np.flatnonzero(~np.isfinite(temp_values))
        if non_finite.size > 0:
            raise ValueError(
                "temp must be finite [K]; got "
                f"{temp_values[non_finite].tolist()} at positions "
                f"{non_finite.tolist()}")

        t_crit = self._require_property(
            't_crit', 'are classified as sub- or supercritical, so their '
            'enthalpy or latent heat')  # [K]
        is_supercritical = temp_values[:, np.newaxis] > t_crit  # [-]

        return temp_values, is_supercritical

    def getHeatVaporization(self, temp: ArrayLike,
                            basis: str = 'mass',
                            weights: Optional[np.ndarray] = None
                            ) -> np.ndarray:
        """Calculate the latent heat of vaporization of each species.

        Every requested temperature classifies every species separately. A
        species that is supercritical at a temperature cannot condense there,
        so its latent heat in that row is reported as zero rather than
        omitted. The species axis of the returned array therefore stays
        aligned with the mass- or mole-fraction vectors that callers weight
        it with, also when the requested temperatures straddle a critical
        temperature.

        Parameters
        ----------
        temp : float or array-like
            Temperature at which the latent heat is evaluated [K]: a scalar or
            a one-dimensional array of independent temperatures, such as the
            spatial nodes of a distributed model. Array entries are never
            paired with species, so any number of temperatures is accepted.
        basis : {'mass', 'mole'}, optional
            Basis of the returned latent heat. The default is 'mass'.
        weights : numpy.ndarray, optional
            Presence weights [any basis, e.g. 0/1], exactly of shape
            ``(num_species,)``, applied at every requested temperature. When
            given, only the species with a nonzero weight are evaluated: the
            others need no Watson data and their columns are exactly zero,
            whatever their data, as for a species that cannot condense.
            Solver-based units pass a structural presence mask here, so
            that a species absent from the condensed phase needs no data
            (issue #425).

        Returns
        -------
        ndarray
            Latent heat of vaporization per species, in [J/kg] for
            ``basis='mass'`` and [J/mol] for ``basis='mole'``, in the phase's
            component order. The shape is ``(num_species,)`` for a scalar or
            one-element ``temp`` and ``(num_temperatures, num_species)``
            otherwise. Entries whose temperature lies strictly above the
            species critical temperature are zero; at exactly ``t_crit`` the
            Watson value is itself zero. Columns of species without critical
            data (``t_crit`` is NaN) are zero at every temperature, and so
            are the columns of species with zero weight.

        Raises
        ------
        ValueError
            If ``basis`` is neither 'mass' nor 'mole'; if ``temp`` is empty,
            non-finite, or has more than one dimension; if ``weights`` does
            not have exactly the shape ``(num_species,)``; or, defensively,
            if an evaluated Watson ratio is not finite and non-negative.
        MissingPropertyError
            A subclass of ``AttributeError`` and ``ValueError``. If a species
            with critical data (and, with ``weights``, a nonzero weight) is
            evaluated at or below its critical temperature while its
            ``t_crit``, ``tref_hvap`` or ``delta_hvap`` is not finite or its
            ``tref_hvap`` is not strictly below its ``t_crit``, which leaves
            the Watson reference undefined; without ``weights`` every
            species is returned, so every species evaluated at or below its
            critical temperature needs valid Watson data, whatever its
            fraction in a mixture. Also raised, naming every species, if no
            species supplies ``t_crit``.

        Notes
        -----
        The latent heat is extrapolated from ``delta_hvap`` at ``tref_hvap``
        with the Watson correlation,
        ``dh(T) = dh(Tref) * ((Tc - T) / (Tc - Tref))**WATSON_EXPONENT``,
        which requires ``Tref < Tc`` and is applied for ``T <= Tc``.
        Supercritical means strictly ``T > Tc``, so ``T == Tc`` is subcritical
        with zero latent heat; ``_classify_criticality`` documents why. A
        species whose ``t_crit`` is NaN (no critical data in the property
        database) is treated as non-condensing: zero latent heat at every
        temperature, and its Watson inputs are neither used nor checked.
        A ``tref_hvap`` or ``delta_hvap`` field that every species omits is
        treated as missing for every species, so it is needed only when some
        entry uses the Watson correlation.
        """
        if basis not in ('mass', 'mole'):
            raise ValueError("basis must be 'mass' or 'mole'")
        if weights is not None:
            weights = self._validate_species_weights(weights)
        temp_values, is_supercritical = self._classify_criticality(temp)  # [K], [-]

        if weights is None:
            is_used = np.ones_like(is_supercritical)  # [-], every entry
        else:
            is_used = np.broadcast_to(weights != 0,
                                      is_supercritical.shape)  # [-]
        deltahvap = self._latent_heat(
            temp_values, is_supercritical, basis,
            is_used)  # [J/kg] or [J/mol], by basis

        if len(temp_values) == 1:
            deltahvap = deltahvap[0]  # [J/kg] or [J/mol], by basis

        return deltahvap

    def _latent_heat(self, temp_values: np.ndarray,
                     is_supercritical: np.ndarray, basis: str,
                     is_used: np.ndarray) -> np.ndarray:
        """Evaluate the Watson latent heat of the entries a result uses.

        Parameters
        ----------
        temp_values : ndarray
            Temperatures [K], shape ``(num_temperatures,)``, as returned by
            ``_classify_criticality``.
        is_supercritical : ndarray of bool
            Criticality mask [-], shape ``(num_temperatures, num_species)``.
        basis : {'mass', 'mole'}
            Basis of the returned latent heat; validated by the caller.
        is_used : ndarray of bool
            True for an entry whose latent heat the result uses, same shape
            as ``is_supercritical``. ``getHeatVaporization`` uses every
            entry, or with ``weights`` only the entries of species with a
            nonzero weight; a mixture enthalpy only those with a nonzero
            fraction.

        Returns
        -------
        ndarray
            Latent heat [J/kg] for ``basis='mass'`` or [J/mol] for
            ``basis='mole'``, shape ``(num_temperatures, num_species)``.
            Supercritical entries, species without critical data and
            unused entries are exactly zero.

        Raises
        ------
        MissingPropertyError
            A subclass of ``AttributeError`` and ``ValueError``.
            If a used entry at or below its species' critical temperature
            has an invalid Watson reference (see ``getHeatVaporization``).
        ValueError
            Defensively, if an evaluated Watson ratio is not finite and
            non-negative.
        """
        num_temp, num_comp = is_supercritical.shape

        # Species without critical data (NaN t_crit) never condense, so only
        # used subcritical entries of species with critical data use Watson.
        has_critical_data = ~np.isnan(self.t_crit)  # [-]
        uses_watson = ~is_supercritical & has_critical_data & is_used  # [-]
        rows, cols = np.nonzero(uses_watson)

        # ParseDatabase creates no attribute for a field that every species
        # omits; read such a field as missing (NaN) for every species, so a
        # Watson use raises the species-specific error below and a database
        # without latent-heat data is accepted when no entry needs it.
        missing_data = np.full(num_comp, np.nan)  # [K] or [J/mol] placeholder
        tref_hvap = np.asarray(getattr(self, 'tref_hvap', missing_data),
                               dtype=float)  # [K]
        delta_hvap = np.asarray(getattr(self, 'delta_hvap', missing_data),
                                dtype=float)  # [J/mol]

        watson_columns = np.unique(cols)
        t_crit_used = self.t_crit[watson_columns]  # [K]
        tref_used = tref_hvap[watson_columns]  # [K]
        delta_hvap_used = delta_hvap[watson_columns]  # [J/mol]
        valid_reference = (np.isfinite(t_crit_used) & np.isfinite(tref_used)
                           & np.isfinite(delta_hvap_used)
                           & (tref_used < t_crit_used))  # [-]
        if not np.all(valid_reference):
            invalid = ~valid_reference  # [-]
            names = [self.name_species[ind]
                     for ind in watson_columns[invalid]]
            raise MissingPropertyError(
                "The Watson latent-heat correlation needs finite t_crit, "
                "tref_hvap and delta_hvap with tref_hvap < t_crit for every "
                "species evaluated at or below its critical temperature; "
                f"check the property data of {names}: "
                f"t_crit = {t_crit_used[invalid].tolist()} K, "
                f"tref_hvap = {tref_used[invalid].tolist()} K, "
                f"delta_hvap = {delta_hvap_used[invalid].tolist()} J/mol")

        watson_ratio = ((self.t_crit[cols] - temp_values[rows])
                        / (self.t_crit[cols] - tref_hvap[cols]))  # [-]
        # Defensive: the checks above make the ratio finite and non-negative
        # unless the arithmetic overflows, so the fractional power is defined.
        bad_ratio = ~(np.isfinite(watson_ratio) & (watson_ratio >= 0))  # [-]
        if np.any(bad_ratio):
            names = sorted({self.name_species[ind] for ind in cols[bad_ratio]})
            raise ValueError(
                "The Watson ratio (t_crit - temp) / (t_crit - tref_hvap) must "
                "be finite and non-negative; got "
                f"{watson_ratio[bad_ratio].tolist()} for species {names} at temp = "
                f"{temp_values[rows[bad_ratio]].tolist()} K")
        watson = watson_ratio**WATSON_EXPONENT  # [-]

        deltahvap = np.zeros((num_temp, num_comp))  # [J/mol]
        deltahvap[rows, cols] = watson * delta_hvap[cols]  # [J/mol]

        if basis == 'mass':
            # Convert the populated subcritical entries in place so that the
            # species axis keeps its full width; supercritical entries stay
            # at zero latent heat instead of being dropped.
            deltahvap[rows, cols] = (deltahvap[rows, cols] / self.mw[cols]
                                     * 1000)  # [J/kg], 1000 g/kg

        return deltahvap

    def _validate_fraction_width(self, fractions: ArrayLike,
                                 name: str) -> np.ndarray:
        """Check that supplied fractions carry one entry per species.

        Parameters
        ----------
        fractions : array-like
            Species mass or mole fractions [-], shape ``(num_species,)`` or
            ``(num_rows, num_species)``.
        name : str
            Public argument name used in the error message.

        Returns
        -------
        ndarray
            ``fractions`` as a float array [-] with unchanged shape and
            values; an existing float array is returned without copying.

        Raises
        ------
        ValueError
            If ``fractions`` is not one- or two-dimensional or its last axis
            does not have ``num_species`` entries.
        """
        fraction_values = np.asarray(fractions, dtype=float)  # [-]
        if (fraction_values.ndim not in (1, 2)
                or fraction_values.shape[-1] != self.num_species):
            raise ValueError(
                f"{name} must have shape (num_species,) = "
                f"{(self.num_species,)} or (num_temperatures, num_species) "
                f"with num_species = {self.num_species} in the order "
                f"{self.name_species}; got shape {fraction_values.shape}")
        return fraction_values

    def getEnthalpy(
            self,
            temp: Optional[ArrayLike] = None,
            temp_ref: float = 298.15,
            mass_frac: Optional[np.ndarray] = None,
            mole_frac: Optional[np.ndarray] = None,
            total_h: bool = True,
            basis: str = 'mass',
            weights: Optional[np.ndarray] = None
            ) -> Union[float, np.ndarray]:
        """Calculate vapor-phase enthalpy relative to a liquid reference.

        Every requested temperature classifies every species separately.
        Entries strictly above the species critical temperature contribute
        vapor sensible heat and no latent heat. All other entries, including
        a temperature exactly equal to ``t_crit``, contribute liquid sensible
        heat plus the Watson latent heat of vaporization, which is zero at
        ``t_crit``. A species without critical data (``t_crit`` is NaN) is
        treated as non-condensing: liquid sensible heat and zero latent heat
        at every temperature.

        Parameters
        ----------
        temp : float or array-like, optional
            Temperature at which to calculate enthalpy [K]: a scalar or a
            one-dimensional array of independent temperatures, such as the
            spatial nodes of a distributed model. Array entries are never
            paired with species. The phase temperature is used when ``temp``
            is ``None``.
        temp_ref : float, optional
            Liquid-reference temperature for the sensible-enthalpy integral
            [K]. The default is 298.15 K.
        mass_frac : ndarray, optional
            Vapor-phase species mass fractions [-], ordered according to the
            phase's components: shape ``(num_species,)`` for a composition
            shared by every temperature, or ``(num_temperatures,
            num_species)`` for a profile whose row ``i`` is paired with
            temperature ``i``. Takes precedence over ``mole_frac``. The phase
            composition is used when neither fraction is supplied.
        mole_frac : ndarray, optional
            Vapor-phase species mole fractions [-], with the same ordering
            and shapes as ``mass_frac``. Used only when ``mass_frac`` is not
            supplied.
        total_h : bool, optional
            If ``True``, return the fraction-weighted mixture enthalpy. If
            ``False``, return unweighted individual species enthalpies; a
            supplied ``mass_frac`` or ``mole_frac`` must still have
            ``num_species`` entries on its last axis, but its row count is
            paired with ``temp`` only when ``total_h=True``. The default is
            ``True``.
        basis : {'mass', 'mole'}, optional
            Physical basis of the returned enthalpy and of the weighting
            fractions. The default is ``'mass'``.
        weights : ndarray, optional
            Only with ``total_h=True``: presence weights [-] with exactly
            the shape of the weighting fractions, ``(num_species,)`` or
            ``(num_temperatures, num_species)``, that
            replace the fractions in deciding which entries are used. An
            entry with zero weight adds exactly zero and needs no
            heat-capacity or latent-heat data, whatever its fraction; the
            fractions still weight the sum. Solver-based units pass a
            structural presence mask here, so that round-off fractions of
            absent species need no data.

        Returns
        -------
        float or ndarray
            Vapor-phase enthalpy on the selected basis: [J/kg] for
            ``basis='mass'`` and [J/mol] for ``basis='mole'``. For a scalar
            or one-element ``temp``, the mixture enthalpy is a scalar and the
            species enthalpy has shape ``(1, num_species)``. For several
            temperatures, the mixture enthalpy has shape
            ``(num_temperatures,)`` and the species enthalpy
            ``(num_temperatures, num_species)``. Species columns follow the
            phase's component order, and every row equals the evaluation at
            that row's temperature alone (with its composition row, for a
            paired profile).

        Raises
        ------
        ValueError
            If ``basis`` is neither 'mass' nor 'mole'; if ``temp`` is empty,
            non-finite, or has more than one dimension; if a supplied
            ``mass_frac`` or ``mole_frac`` is not one- or
            two-dimensional with ``num_species`` entries on its last axis;
            if, for ``total_h=True``, the weighting composition is neither
            ``(num_species,)`` nor ``(num_temperatures, num_species)`` (a
            scalar ``temp`` therefore accepts only a one-row profile); or if
            ``weights`` is given with ``total_h=False`` or does not have
            exactly the shape of the weighting composition.
        MissingPropertyError
            A subclass of ``AttributeError`` and ``ValueError``.
            If a used entry is supercritical but the property database
            supplies no ``cp_vapor`` data for its species, or is at or below
            its critical temperature without ``cp_liq`` data or with an
            invalid Watson reference (see ``getHeatVaporization``). Data
            missing for one species only (a NaN entry or row, see
            ``ParseDatabase``) count as missing. The error names the species
            and the property. Also raised if no species supplies ``t_crit``.
            Also raised, whatever the fractions or weights, if the property is
            stored as one scalar per species instead of a coefficient list (see
            ``_property_rows``).

        Notes
        -----
        Liquid heat capacities are integrated only for species that are
        subcritical at some requested temperature, and vapor heat capacities
        only for species that are supercritical at some requested
        temperature, so heat-capacity data that no entry uses need not exist.
        For ``total_h=True``, an entry whose weighting fraction (or, with
        ``weights``, whose weight, whatever its fraction) is zero adds
        nothing to the mixture and needs no heat-capacity data either; its
        sensible heat is taken as exactly zero entry by entry, so a species
        used in one row of a profile adds nothing in a row that leaves it
        unused, and never NaN times zero. Nonzero
        is bitwise: a round-off-level fraction counts as present, so a
        species carried as a solver state may need data at a nominal zero
        fraction. The same exemption covers the latent heat: a zero-fraction
        entry needs no ``delta_hvap`` or ``tref_hvap`` and adds exactly zero,
        so the mixture enthalpy equals the one computed without that species
        (issue #424). With ``total_h=False``, every entry is returned and
        needs its data, as does the public ``getHeatVaporization`` without
        ``weights``.
        The supercritical classification is strictly ``T > Tc``; see
        ``_classify_criticality`` for the equality rule. The mixture
        enthalpy is the row-wise species sum of species enthalpy times the
        fraction on the selected basis.
        """
        if basis not in ('mass', 'mole'):
            raise ValueError("basis must be 'mass' or 'mole'")
        if mass_frac is None and mole_frac is None:
            mass_frac = self.mass_frac
            mole_frac = self.mole_frac
        elif mass_frac is None:
            mole_frac = self._validate_fraction_width(mole_frac,
                                                      'mole_frac')  # [-]
            mass_frac = self.frac_to_frac(mole_frac=mole_frac)
        else:
            mass_frac = self._validate_fraction_width(mass_frac,
                                                      'mass_frac')  # [-]
            mole_frac = self.frac_to_frac(mass_frac)

        if temp is None:
            temp = self.temp

        temp_values, is_supercritical = self._classify_criticality(temp)  # [K], [-]
        num_temp, num_comp = is_supercritical.shape

        if total_h:
            frac = np.asarray(mass_frac if basis == 'mass' else mole_frac,
                              dtype=float)  # [-], on the requested basis
            if frac.shape not in ((num_comp,), (num_temp, num_comp)):
                raise ValueError(
                    "The weighting composition must have shape "
                    f"(num_species,) = {(num_comp,)} for a fixed composition "
                    "or (num_temperatures, num_species) = "
                    f"{(num_temp, num_comp)} for a profile paired row by row "
                    f"with temp; got {frac.shape}")
            # A zero-fraction (or zero-weight) entry adds nothing to the
            # mixture, so it needs no heat-capacity data and keeps zero
            # sensible heat below.
            if weights is None:
                presence = frac != 0  # [-]
            else:
                frac_name = 'mass_frac' if basis == 'mass' else 'mole_frac'
                presence = self._validate_weights_shape(
                    weights, frac, frac_name) != 0  # [-]
            is_used = np.broadcast_to(presence, is_supercritical.shape)
        else:
            if weights is not None:
                raise ValueError(
                    "weights applies to the mixture enthalpy (total_h=True); "
                    "species enthalpies (total_h=False) need every entry")
            # Every species enthalpy is returned, so every entry is used.
            is_used = np.ones_like(is_supercritical)

        subcritical_columns = np.flatnonzero(
            (~is_supercritical & is_used).any(axis=0))
        supercritical_columns = np.flatnonzero(
            (is_supercritical & is_used).any(axis=0))

        # Sensible heat from temp_ref on each candidate correlation; the
        # per-entry mask below selects one of them for every entry.
        h_liquid = np.zeros((num_temp, num_comp))  # [J/kg] or [J/mol], by basis
        if subcritical_columns.size > 0:
            self._check_species_rows(
                'cp_liq', self._property_rows('cp_liq')[subcritical_columns],
                subcritical_columns, np.ones(subcritical_columns.size, bool),
                'are at or below t_crit (or have no t_crit) at a requested '
                'temperature, so their enthalpy')
            h_liquid[:, subcritical_columns] = super().getEnthalpy(
                temp_values, temp_ref, phase='liquid', basis=basis,
                idx=subcritical_columns, total_h=False)  # [J/kg] or [J/mol]

        h_vapor = np.zeros((num_temp, num_comp))  # [J/kg] or [J/mol], by basis
        if supercritical_columns.size > 0:
            self._check_species_rows(
                'cp_vapor',
                self._property_rows('cp_vapor')[supercritical_columns],
                supercritical_columns,
                np.ones(supercritical_columns.size, bool),
                'are supercritical (temp > t_crit) at a requested '
                'temperature, so their enthalpy',
                remedy=' or evaluate at or below t_crit')
            h_vapor[:, supercritical_columns] = super().getEnthalpy(
                temp_values, temp_ref, phase='vapor', basis=basis,
                idx=supercritical_columns, total_h=False)  # [J/kg] or [J/mol]

        h_sensible = np.where(is_supercritical, h_vapor, h_liquid)  # [J/kg] or [J/mol]
        # A column is integrated if some row uses it, so an entry that its
        # own row leaves unused (zero weight) must be zeroed here, entry by
        # entry, before the fractions weight the sum.
        h_sensible = np.where(is_used, h_sensible, 0.0)  # [J/kg] or [J/mol]

        # Zero for supercritical and unused entries, Watson value otherwise.
        latent_heat = self._latent_heat(
            temp_values, is_supercritical, basis,
            is_used)  # [J/kg] or [J/mol], by basis

        h_species = h_sensible + latent_heat  # [J/kg] or [J/mol], by basis

        if not total_h:
            return h_species

        h_total = (h_species * frac).sum(axis=1)  # [J/kg] or [J/mol], by basis
        if num_temp == 1:
            h_total = h_total[0]  # [J/kg] or [J/mol], by basis

        return h_total

    def getDewPoint(self, pres: Optional[float] = None,
                    mass_frac: Optional[np.ndarray] = None,
                    mole_frac: Optional[np.ndarray] = None,
                    thermo_method: str = 'ideal',
                    x_liq: bool = False,
                    weights: Optional[np.ndarray] = None
                    ) -> Union[float, tuple]:
        """Solve for the vapor dew-point temperature.

        Parameters
        ----------
        pres : float, optional
            Pressure [Pa]; the phase pressure when ``None``.
        mass_frac, mole_frac : ndarray, optional
            Vapor fractions [-], shape ``(num_species,)``. Mole fractions
            are converted from mass fractions when only those are given; the
            phase composition is used when neither is.
        thermo_method : {'ideal', 'UNIFAC', 'UNIQUAC'}, optional
            Liquid activity model passed to ``getKeqVLE``.
        x_liq : bool, optional
            Also return the incipient liquid composition.
        weights : ndarray, optional
            Presence weights [-] with the shape of the mole fractions, as in
            ``LiquidPhase.getBubblePoint``. When given, only species with a
            nonzero weight enter the dew-point sum, its Antoine seed, the
            incipient liquid and a non-ideal activity model, and only they
            need VLE and activity data (see ``getKeqVLE``); the others count
            as absent, with exactly zero liquid fraction and the neutral
            activity coefficient 1. Unit operations pass the presence of
            the species they carry, so that an absent species needs no data.

        Returns
        -------
        temp_dew : float
            Dew-point temperature [K].
        x_frac : ndarray
            Incipient liquid mole fractions [-], shape ``(num_species,)``;
            only when ``x_liq`` is True.

        Raises
        ------
        ValueError
            If ``weights`` does not have exactly the shape of the mole
            fractions, before any evaluation.
        MissingPropertyError
            A subclass of ``AttributeError`` and ``ValueError``.
            From ``getKeqVLE`` at the converged root, if a species with a
            nonzero fraction (or weight) has no ``p_vap`` coefficients at or
            below its critical temperature. Already from the Newton
            iterates, which evaluate the same ratios, if a species with a
            nonzero fraction (or weight) is above its critical temperature
            at an iterate without
            ``henry_constant`` [Pa], if no species supplies ``t_crit``, if no
            species supplies ``henry_constant`` while some species (with
            ``weights``, some weighted species) is supercritical at an
            iterate, or if ``p_vap`` is stored as one scalar per species
            (from the Antoine seed). With ``weights`` and
            ``thermo_method='UNIQUAC'``, also if a weighted species lacks
            ``ri``, ``qi`` or (when the database has the key) ``qip``; UNIFAC
            group counts are taken as given (see ``getKeqVLE``).
        RuntimeError
            If the Newton iteration does not converge.

        Notes
        -----
        The Newton seed is ``sum_i y_i * Tsat_i(pres)`` [K], with
        ``weights`` over the species with a nonzero weight only. Seed and
        iterates treat missing Antoine data as in
        ``LiquidPhase.getBubblePoint``, and the converged root is checked
        with ``getKeqVLE`` weighted by the vapor composition (or by
        ``weights``).
        """
        if mass_frac is None and mole_frac is None:
            mole_frac = self.mole_frac

        elif mole_frac is None:
            mole_frac = self.frac_to_frac(mass_frac=mass_frac)

        mole_frac = np.asarray(mole_frac, dtype=float)  # [-], also from a list

        if pres is None:
            pres = self.pres

        # Absent (zero-weight) species leave the sum and the liquid exactly;
        # their ratios may be NaN placeholders (see getKeqVLE).
        if weights is not None:
            weights = self._validate_weights_shape(weights, mole_frac,
                                                   'mole_frac')
            used = weights != 0  # [-]

        def dew_fn(temp: float) -> float:
            """Evaluate the dew-point residual at a Newton iterate.

            Parameters
            ----------
            temp : float
                Temperature iterate [K].

            Returns
            -------
            float
                ``sum_i y_i / K_i - 1`` over the used species [-]; zero at
                the dew point.
            """
            k_vals = self._vle_ratios(temp, pres, mole_frac, thermo_method,
                                      check=False,
                                      weights=weights)  # [-], iterate only

            if weights is None:
                obj = np.dot(mole_frac, 1/k_vals) - 1
            else:
                obj = np.dot(mole_frac[used], 1/k_vals[used]) - 1

            return obj

        # Absent species are left out of the seed too: their saturation
        # temperature may be infinite, and 0 * inf is NaN.
        temp_seed = self._antoine_seed(
            mole_frac, pres=pres,
            used=None if weights is None else used)  # [K]
        temp_dew = newton(dew_fn, temp_seed, full_output=False)

        # Check the data the converged root needs.
        k_vals = self.getKeqVLE(temp_dew, pres, mole_frac,
                                gamma_model=thermo_method,
                                weights=weights)  # [-]

        if x_liq:
            if weights is None:
                x_frac = mole_frac/k_vals  # [-]
            else:
                x_frac = np.where(
                    used, mole_frac / np.where(used, k_vals, 1.0), 0.0)  # [-]

            return temp_dew, x_frac
        else:
            return temp_dew

    def getViscosity(self, temp: Optional[ArrayLike] = None,
                     mass_frac: Optional[np.ndarray] = None,
                     mole_frac: Optional[np.ndarray] = None,
                     weights: Optional[np.ndarray] = None
                     ) -> Union[float, np.ndarray]:
        """Return the vapor mixture viscosity from the Wilke rule.

        Parameters
        ----------
        temp : float or array-like, optional
            Temperature [K]; not used, the stored ``visc_gas`` values are
            constants.
        mass_frac, mole_frac : numpy.ndarray, optional
            Vapor fractions [-], shape ``(num_species,)`` or
            ``(num_points, num_species)``; mole fractions take precedence,
            and the phase mole fractions are used when neither is supplied.
        weights : numpy.ndarray, optional
            Presence weights [-], with exactly the shape of the fractions
            used; the Wilke sums then run over the species with a nonzero
            weight only, and the others need no ``visc_gas``. See
            ``ThermoPhysicalManager.getViscosityMix``.

        Returns
        -------
        float or numpy.ndarray
            Viscosity [Pa*s]: a scalar for one composition, shape
            ``(num_points,)`` for a composition profile.

        Raises
        ------
        ValueError
            If ``weights`` does not have exactly the shape of the fractions
            used.
        MissingPropertyError
            A subclass of ``AttributeError`` and ``ValueError``. If any
            species (with ``weights``, any species with a nonzero weight)
            has no ``visc_gas`` value [Pa*s].
        """
        viscosity = self.getViscosityMix(temp, mass_frac, mole_frac,
                                         phase='vapor', weights=weights)

        return viscosity

    def getDensity(self, pres_gas: Optional[Union[float, np.ndarray]] = None,
                   temp_gas: Optional[Union[float, np.ndarray]] = None,
                   phase: str = 'gas', basis: str = 'mass', *,
                   mass_frac: Optional[np.ndarray] = None,
                   mole_frac: Optional[np.ndarray] = None,
                   temp: Optional[Union[float, np.ndarray]] = None,
                   pres: Optional[Union[float, np.ndarray]] = None) -> Union[float, np.ndarray]:
        """Return ideal-gas density on the requested physical basis.

        Parameters
        ----------
        pres_gas, temp_gas : float or ndarray, optional
            Legacy pressure [Pa] and temperature [K] overrides, retaining
            their positional order. Scalar or one value per composition row.
        phase : str, optional
            Retained legacy argument, historically ignored. Density always
            uses the gas EOS regardless of this value.
        mass_frac, mole_frac : ndarray, optional
            Keyword-only species fractions [-], shape ``(num_species,)`` or
            ``(num_points, num_species)``. Mole fractions take precedence
            when both are given; omitting both uses stored composition.
        temp, pres : float or ndarray, optional
            Keyword-only temperature [K] and pressure [Pa] overrides. Scalar
            or one value per composition row. Omitted values use stored state.
            Do not combine an override with its legacy alias.
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
            If ``basis`` is neither ``'mass'`` nor ``'mole'``, or both names
            for the same override are supplied.

        Notes
        -----
        Legacy positional and keyword overrides retain their meaning. The
        corrected default is mass density [kg/m**3]; request ``basis='mole'``
        explicitly for molar density [mol/L], replacing the legacy mol/m**3
        result. Overrides do not change stored state. An empty vapor placeholder
        constructed with zero ``mole_frac`` has zero mass density; zero
        ``mass_frac`` or ``mole_conc`` construction is not supported.
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
                 mass=0,
                 mass_frac: Optional[Union[ArrayLike, Mapping[str, ArrayLike]]] = None,
                 moments: Optional[ArrayLike] = None, num_mom=4,
                 distrib: Optional[ArrayLike] = None,
                 x_distrib: Optional[ArrayLike] = None, distrib_type='vol_perc',
                 moisture=0, porosity=0,
                 mole_conc: Optional[ArrayLike] = None, kv: float = 1,
                 mole_frac: Optional[Union[ArrayLike, Mapping[str, ArrayLike]]] = None) -> None:
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
            Python float. ``getEnthalpy`` uses it when its own ``temp_ref`` is
            omitted; mixtures such as ``Slurry`` and ``Cake`` pass their own
            common reference instead.
        pres : float, optional
            Pressure [Pa], default one standard atmosphere (101325 Pa).
        mass : float, optional
            Solid mass [kg]. Zero derives inventory from the supplied moments
            or raw number distribution; positive mass scales bin weights.
        mass_frac : array-like or mapping of str to float/array-like, optional
            Species mass fractions [-], shape ``(num_species,)``, or a
            species-keyed mapping. Copied before zero entries are replaced by
            machine epsilon.
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
            Must be a finite positive scalar; default one represents cubic
            particles.
        mole_frac : array-like or mapping of str to float/array-like, optional
            Species mole fractions [-], shape ``(num_species,)``, or a
            species-keyed mapping, as an alternative to ``mass_frac``.

        Raises
        ------
        ValueError
            If neither ``mass_frac`` nor ``mole_frac`` is provided, both are
            provided, ``distrib_type`` is not 'vol_perc' or 'mass_frac', or a
            supplied distribution grid has fewer than two points, or ``kv`` is
            not a finite positive scalar.
        RuntimeError
            If the species fractions sum to less than the existing
            composition threshold of 0.99 [-].
        MissingPropertyError
            A subclass of ``AttributeError`` and ``ValueError``. If
            ``moments`` or ``distrib`` is given, so that the mixture density
            relates mass and volume, and a present species has no
            ``rho_solid`` [kg/m**3]; a species at the zero sentinel needs
            none (see ``getDensity``). Without them the density is not
            evaluated.
        """
        if mass_frac is None and mole_frac is None:
            raise ValueError("SolidPhase requires mass_frac; provide species "
                             "mass fractions with shape (num_species,)")
        if mass_frac is not None and mole_frac is not None:
            raise ValueError(
                "Specify either 'mass_frac' or 'mole_frac' for SolidPhase, "
                "not both."
            )
        if distrib_type not in ('vol_perc', 'mass_frac'):
            raise ValueError("distrib_type must be 'vol_perc' or 'mass_frac'; "
                             f"got {distrib_type!r}")

        if (np.ndim(kv) != 0 or not np.isrealobj(kv)
                or not np.issubdtype(np.asarray(kv).dtype, np.number)
                or not np.isfinite(kv) or kv <= 0):
            raise ValueError("kv must be a finite positive scalar")

        super().__init__(path_thermo)
        self.kv = kv  # [-], physical particle volume / size**3
        self.distrib_type = distrib_type
        self.num_mom = num_mom  # [-]

        self.temp = (float(temp) if np.ndim(temp) == 0
                     else _as_float_array(temp))  # [K]
        self.temp_ref = float(temp_ref)  # [K]
        self.pres = pres

        self.mass = mass

        if mole_frac is not None:
            mole_frac_arr = self._coerce_composition_mapping(
                mole_frac, 'mole_frac'
            )  # [-]
            mole_frac_arr = np.array(
                np.atleast_1d(mole_frac_arr), dtype=float
            )  # [-]
            if float(np.sum(mole_frac_arr)) < 0.99:
                raise RuntimeError(
                    'The sum of mole fractions is less than 0.99')
            mass_frac = self.frac_to_frac(mole_frac=mole_frac_arr)  # [-]
        else:
            mass_frac = self._coerce_composition_mapping(
                mass_frac, 'mass_frac'
            )  # [-]
            mass_frac = np.array(np.atleast_1d(mass_frac), dtype=float)  # [-]

        # Zero sentinel: exact zeros become eps (see _stored_fractions).
        mass_frac[mass_frac == 0] = eps

        self.mass_frac = mass_frac  # [-]
        self.mole_frac = self.frac_to_frac(mass_frac=self.mass_frac)  # [-]

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

        # Mass and volume; the density is needed only to relate them.
        if solid_spec:
            dens = self.getDensity()  # [kg/m**3]
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
                    moments: Optional[ArrayLike] = None,
                    mass_frac: Optional[Union[ArrayLike, Mapping[str, ArrayLike]]] = None,
                    mole_frac: Optional[Union[ArrayLike, Mapping[str, ArrayLike]]] = None) -> None:
        """Update the solid size distribution, composition, mass, or moments.

        Parameters
        ----------
        x_distrib : array-like, optional
            Crystal-size grid with shape ``(num_sizes,)`` [um]. When supplied,
            it is checked before replacing the stored grid and bin widths
            ``dx`` [um], using uniform spacing or geometric bin boundaries.
            Distribution moments are then recalculated if ``distrib`` is given.
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
        mass_frac : array-like or mapping of str to float/array-like, optional
            Updated species mass fractions [-], shape ``(num_species,)``, or a
            species-keyed mapping.
        mole_frac : array-like or mapping of str to float/array-like, optional
            Updated species mole fractions [-], shape ``(num_species,)``, or a
            species-keyed mapping.

        Raises
        ------
        ValueError
            If the supplied grid has fewer than two points, or both
            ``mass_frac`` and ``mole_frac`` are provided. The existing
            grid, widths, distribution, and inventory remain unchanged.
        RuntimeError
            If updated species fractions sum to less than 0.99 [-].
        MissingPropertyError
            A subclass of ``AttributeError`` and ``ValueError``. As in the
            constructor: if a ``distrib`` or ``mass`` update relates mass and
            volume through the density and a present species of the stored
            composition has no ``rho_solid`` [kg/m**3]; a species at the zero
            sentinel needs none.

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
        mole amount is recalculated whenever a distribution, composition, or
        explicit mass changes the solid inventory. A moments-only update does
        not imply an amount change and therefore leaves mass, volume, and
        moles intact.
        """
        if mass_frac is not None and mole_frac is not None:
            raise ValueError(
                "Specify either 'mass_frac' or 'mole_frac' for SolidPhase, "
                "not both."
            )

        composition_updated = False
        if mole_frac is not None:
            mole_frac_arr = self._coerce_composition_mapping(
                mole_frac, 'mole_frac'
            )  # [-]
            mole_frac_arr = np.array(
                np.atleast_1d(mole_frac_arr), dtype=float
            )  # [-]
            if float(np.sum(mole_frac_arr)) < 0.99:
                raise RuntimeError(
                    'The sum of mole fractions is less than 0.99')
            updated_mass_frac = self.frac_to_frac(mole_frac=mole_frac_arr)  # [-]
            updated_mass_frac[updated_mass_frac == 0] = eps
            self.mass_frac = updated_mass_frac  # [-]
            self.mole_frac = self.frac_to_frac(mass_frac=self.mass_frac)  # [-]
            self.mw_av = np.dot(self.mole_frac, self.mw)  # [g/mol]
            composition_updated = True
        elif mass_frac is not None:
            updated_mass_frac = self._coerce_composition_mapping(
                mass_frac, 'mass_frac'
            )  # [-]
            updated_mass_frac = np.array(
                np.atleast_1d(updated_mass_frac), dtype=float
            )  # [-]
            if float(np.sum(updated_mass_frac)) < 0.99:
                raise RuntimeError(
                    'The sum of mass fractions is less than 0.99')
            updated_mass_frac[updated_mass_frac == 0] = eps
            self.mass_frac = updated_mass_frac  # [-]
            self.mole_frac = self.frac_to_frac(mass_frac=self.mass_frac)  # [-]
            self.mw_av = np.dot(self.mole_frac, self.mw)  # [g/mol]
            composition_updated = True

        if x_distrib is not None:
            updated_grid = _as_float_array(x_distrib)  # [um]
            updated_widths = self._get_grid_spacing(updated_grid)  # [um]
            self.x_distrib = updated_grid  # [um]
            self.dx = updated_widths  # [um]

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

        if distrib is not None or mass is not None or composition_updated:
            self._reconcile_moles_from_mass()

    def convert_distribution(self, x_distrib: Optional[ArrayLike] = None,
                             num_distr: Optional[ArrayLike] = None,
                             vol_distr: Optional[ArrayLike] = None,
                             mass: float = 0) -> np.ndarray:
        """Convert nodal number densities and volume fractions consistently.

        Parameters
        ----------
        x_distrib : array-like, optional
            Strictly increasing, nonnegative sizes [um], shape (num_sizes,).
            Defaults to the attached grid. At least two nodes are required.
        num_distr : array-like, optional
            Number density [#/um], shape (..., num_sizes). Rows may represent
            total, volume-specific, or flow populations; normalization cancels
            their common inventory basis. Signed solver undershoots are
            preserved as diagnostics, not clipped. Mutually exclusive with
            vol_distr.
        vol_distr : array-like, optional
            Nonnegative nodal volume fractions [-], shape (..., num_sizes).
            Fractions allocate volume to trapezoidal node contributions and
            should sum to one for a complete population.
        mass : float, optional
            Positive solid inventory [kg] for volume-to-number conversion.

        Returns
        -------
        numpy.ndarray
            Volume fractions [-] for num_distr, or number density [#/um] for
            vol_distr, preserving input shape. Empty number populations yield
            zero fractions; their physical normalized distribution is undefined.

        Raises
        ------
        ValueError
            If the grid, population shape, finite values, nonnegative volume
            inputs, mutually exclusive inputs, or positive mass are invalid. Positive volume at
            zero particle size cannot define a finite number population.

        Notes
        -----
        Each interior quadrature weight is half the distance between its two
        neighbors; endpoint weights are half the adjacent interval. These are
        exactly the weights used by getMoments. A node's particle volume is
        kv * size**3. The shape factor cancels between node and total volumes
        in number-to-volume conversion, including on nonuniform grids. This
        corrects the legacy combination of bin-width numerators and trapezoid
        denominators; FVM grid widths themselves are not changed.
        """
        sizes = np.asarray(self.x_distrib if x_distrib is None else x_distrib,
                           dtype=float)  # [um]
        if (sizes.ndim != 1 or sizes.size < 2 or not np.isfinite(sizes).all()
                or np.any(sizes < 0) or np.any(np.diff(sizes) <= 0)):
            raise ValueError('x_distrib must contain at least two finite, '
                             'nonnegative, strictly increasing sizes [um]')
        if (num_distr is None) == (vol_distr is None):
            raise ValueError("Specify exactly one of 'num_distr' or 'vol_distr'")
        values = np.asarray(num_distr if num_distr is not None else vol_distr,
                            dtype=float)  # [#/um] or [-]
        if (values.ndim == 0 or values.shape[-1] != sizes.size
                or not np.isfinite(values).all()
                or (vol_distr is not None and np.any(values < 0))):
            raise ValueError('Distribution must be finite with one entry per size '
                             'on its last axis; volume fractions must be nonnegative')
        intervals = np.diff(sizes)  # [um]
        support = np.r_[intervals[0], intervals[:-1] + intervals[1:],
                        intervals[-1]] / 2  # [um], trapezoid node weights
        particle_volume = self.kv * (sizes * 1e-6)**3  # [m**3/particle]
        if num_distr is not None:
            contributions = values * support * particle_volume  # [m**3], input population basis
            total = contributions.sum(axis=-1, keepdims=True)  # [m**3], same basis
            return np.divide(contributions, total, out=np.zeros_like(contributions),
                             where=total != 0)
        if not np.isfinite(mass) or mass <= 0:
            raise ValueError("'vol_distr' given, mass must be finite and greater than zero [kg]")
        if np.any(values[..., sizes == 0] > 0):
            raise ValueError('Positive volume at zero particle size is undefined')
        solid_volume = mass / self.getDensity()  # [m**3]
        denominator = particle_volume * support  # [m**3*um/particle]
        return np.divide(solid_volume * values, denominator,
                         out=np.zeros_like(values), where=denominator > 0)

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
        and each node's volume by ``kv * size**3`` and its trapezoidal support
        width to obtain [#/um], consistent with getMoments.
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
        """Return the ideal-mixing solid density on the requested basis.

        Parameters
        ----------
        mass_frac, mole_frac : ndarray, optional
            Solid species fractions [-], shape ``(num_species,)`` or
            ``(num_points, num_species)``. The stored phase mass fractions are
            used, and converted for ``basis='mole'``, when neither is
            supplied.
        temp : float, optional
            Temperature [K]; defaults to the phase temperature. Pure solid
            densities are temperature independent, so it does not change the
            result.
        basis : {'mass', 'mole'}, optional
            Physical basis of the returned density; default mass.

        Returns
        -------
        float or ndarray
            Solid density, [kg/m**3] for ``basis='mass'`` and [kmol/m**3]
            (equivalently [mol/L]) for ``basis='mole'``.

        Raises
        ------
        ValueError
            If ``basis`` is neither 'mass' nor 'mole', raised by
            ``ThermoPhysicalManager.getDensityMix``.
        MissingPropertyError
            A subclass of ``AttributeError`` and ``ValueError``. If a species
            with a nonzero fraction has no ``rho_solid``; absent species need
            none and occupy no volume.

        Notes
        -----
        With the stored composition, a species at the zero sentinel ``eps``
        without ``rho_solid`` is absent, as in ``getCp`` (see
        ``_stored_fractions``); fractions passed explicitly keep the bitwise
        nonzero rule.
        """
        if temp is None:
            temp = self.temp

        if mass_frac is None and mole_frac is None:
            mass_frac, _ = self._stored_fractions('rho_solid')  # [-]

        densSolid = self.getDensityMix(mass_frac, mole_frac, phase='solid',
                                       temp=temp, basis=basis)

        return densSolid

    def getPorosity(self, distrib=None, diam_filter: float = 1, AR=None,
                    sphericity=None) -> float:
        """Estimate packed-bed porosity with the existing linear packing model.

        Parameters
        ----------
        distrib : array-like, optional
            Number distribution [#/um], shape (num_sizes,). Defaults to the
            attached distribution. Mean size uses the attached phase moments.
        diam_filter : float, optional
            Filter diameter [m]; default 1 m is the existing reference bed.
        AR : float, optional
            Legacy unused aspect ratio [-].
        sphericity : float, optional
            Particle sphericity [-]; default 0.7 is the existing model
            assumption. A positive value is required for packing diameters.

        Returns
        -------
        float
            Pore fraction of total packed-bed volume [-].

        Raises
        ------
        ValueError
            If the distribution represents zero, negative, or nonfinite
            particle volume and therefore cannot define normalized weights.

        Notes
        -----
        Repository commit 9b646f2 attributes this model and its empirical
        coefficients to Yu, Zou, and Standish (1996). It does not establish a
        calibrated range. The existing Jeschar initial-porosity relation is
        ``0.375 + 0.34 * mean_size / diam_filter``. The equivalent packing
        diameter uses the phase-owned volume factor ``kv`` [-]. Common
        diameter scaling cancels in pairwise size ratios, and ``kv`` does not
        affect normalized volume weights. A zero or invalid particle volume
        cannot define packing and raises ValueError.
        """

        if distrib is None:
            distrib = self.distrib
            mom_zero = self.moments[0]
            mom_one = self.moments[1]
        else:
            mom_zero, mom_one = self.getMoments(mom_num=(0, 1))

        x_dist = self.x_distrib * 1e-6  # [m], exact um-to-m conversion

        if sphericity is None:
            sphericity = 0.7  # [-], existing particle-shape assumption

        # Yu, Zou, and Standish (1996), attributed in commit 9b646f2.
        kv = self.kv  # [-], phase-owned volumetric shape coefficient

        del_x_dist = np.diff(x_dist)
        node_x_dist = (x_dist[:-1] + x_dist[1:]) / 2
        node_CSD = (distrib[:-1] + distrib[1:]) / 2

        # Volume of crystals in each bin
        vol_cry = node_CSD * del_x_dist * (kv * node_x_dist**3)  # [m**4/um], per-um CSD with meter grid; common factor cancels
        total_volume = np.sum(vol_cry)  # [m**4/um], same common scale as bin volumes
        if not np.isfinite(total_volume) or total_volume <= 0:
            raise ValueError("Porosity requires a finite positive particle volume")
        frac_vol_cry = vol_cry / total_volume  # [-], common kv cancels exactly

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

    def _stored_fractions(self, prop_name: str = 'cp_solid') -> tuple:
        """Return the stored composition for a mixture value of a property.

        Parameters
        ----------
        prop_name : {'cp_solid', 'rho_solid'}, optional
            Database key of the property that the mixture value weights:
            the ``cp_solid`` coefficient rows (default) or the scalar
            ``rho_solid`` [kg/m**3].

        Returns
        -------
        mass_frac, mole_frac : numpy.ndarray
            Stored mass and mole fractions [-], shape ``(num_species,)`` or
            ``(num_rows, num_species)``. A species whose stored mass fraction
            is at or below the zero sentinel ``eps`` in every row and that
            has no ``prop_name`` data gets exact zeros in both arrays.

        Notes
        -----
        The constructor writes ``eps`` = ``np.finfo(float).eps`` [-] for a
        zero fraction, so a stored mass fraction at or below ``eps`` is a
        declared zero, or a mixture or copy of one (such as a solid built
        from another solid's ``mass_frac``), and is physically zero at
        double precision. Such a species is absent for the decision whether
        ``prop_name`` data are needed, also when the mole-fraction basis is
        evaluated. With data it keeps its ``eps`` weight, so results are
        unchanged; without data its contribution is exactly zero. For
        ``cp_solid`` that is what the zero rows of releases before #414 gave;
        for ``rho_solid``, earlier releases returned a NaN density (or raised
        ``AttributeError`` when no species supplied ``rho_solid``). A stored
        fraction above ``eps`` is present however small, and fractions
        passed to a method keep the bitwise nonzero rule. A ``cp_solid``
        stored as one scalar per species raises ``MissingPropertyError``
        from ``_property_rows`` before any fraction is considered.
        """
        mass_frac = np.asarray(self.mass_frac, dtype=float)  # [-]
        mole_frac = np.asarray(self.mole_frac, dtype=float)  # [-]
        if prop_name == 'rho_solid':
            lacking = np.isnan(self._scalar_values(prop_name))  # [-]
        else:
            rows = self._property_rows(prop_name)  # [J/mol/K/K**k], ascending k
            lacking = np.isnan(rows).any(axis=1)  # [-]
        if mass_frac.shape[-1] != len(lacking):
            return self.mass_frac, self.mole_frac

        sentinel = np.atleast_2d(mass_frac <= eps).all(axis=0)  # [-]
        absent = sentinel & lacking
        if not absent.any():
            return self.mass_frac, self.mole_frac

        return (np.where(absent, 0.0, mass_frac),
                np.where(absent, 0.0, mole_frac))  # [-], [-]

    def getCp(self, temp=None, mass_frac=None, mole_frac=None, basis='mass'):
        """Return the solid heat capacity on the requested basis.

        Parameters
        ----------
        temp : float or array-like, optional
            Temperature [K]; the phase temperature when ``None``.
        mass_frac, mole_frac : numpy.ndarray, optional
            Species fractions [-], shape ``(num_species,)`` or
            ``(num_rows, num_species)``; the stored composition when neither
            is supplied, in which a species at the zero sentinel without
            ``cp_solid`` data is absent (see ``_stored_fractions``).
        basis : {'mass', 'mole'}, optional
            Basis of the result; default mass.

        Returns
        -------
        float or numpy.ndarray
            Heat capacity [J/kg/K] or [J/mol/K], shaped as in
            ``ThermoPhysicalManager.getCpMix``.

        Raises
        ------
        ValueError
            If ``basis`` is neither 'mass' nor 'mole'.
        MissingPropertyError
            A subclass of ``AttributeError`` and ``ValueError``. If a
            species with a nonzero fraction has no ``cp_solid`` data.
            Also raised, whatever the fractions or weights, if the property is
            stored as one scalar per species instead of a coefficient list (see
            ``_property_rows``).
        """
        if temp is None:
            temp = self.temp

        if mass_frac is None and mole_frac is None:
            mass_frac, mole_frac = self._stored_fractions()  # [-]

        cpSolid = super().getCpMix(temp, mass_frac, mole_frac, phase='solid',
                                   basis=basis)

        return cpSolid

    def getEnthalpy(
            self,
            temp: Optional[ArrayLike] = None,
            temp_ref: Optional[float] = None,
            mass_frac: Optional[np.ndarray] = None,
            mole_frac: Optional[np.ndarray] = None,
            total_h: bool = True,
            basis: str = 'mass') -> Union[float, np.ndarray]:
        """Calculate solid sensible enthalpy relative to a reference temperature.

        Parameters
        ----------
        temp : float or ndarray, optional
            Temperature at which enthalpy is evaluated [K], a scalar or a
            temperature profile of shape ``(num_temps,)``. The phase
            temperature ``self.temp`` is used when ``temp`` is ``None``.
        temp_ref : float, optional
            Lower limit of the heat-capacity integral [K]. An explicit value
            takes precedence over the phase state. When ``None``, the stored
            constructor reference ``self.temp_ref`` is used; its default is
            298.15 K.
        mass_frac : ndarray, optional
            Solid species mass fractions [-], shape ``(num_species,)`` or
            ``(num_rows, num_species)``. The phase composition is used when
            neither fraction vector is supplied.
        mole_frac : ndarray, optional
            Solid species mole fractions [-], with the same shapes as
            ``mass_frac``. The phase composition is used when neither
            fraction vector is supplied.
        total_h : bool, optional
            If ``True`` (default), return the fraction-weighted mixture
            enthalpy. If ``False``, return individual species enthalpies.
        basis : {'mass', 'mole'}, optional
            Physical basis of the returned enthalpy. The default is
            ``'mass'``.

        Returns
        -------
        float or ndarray
            Solid sensible enthalpy, [J/kg] for ``basis='mass'`` and [J/mol]
            for ``basis='mole'``. With ``total_h=True``, one value per
            temperature or composition row, broadcast as in
            ``ThermoPhysicalManager.getEnthalpy``; a scalar only for a single
            temperature with a fixed composition or a one-row profile. With
            ``total_h=False`` the shape is
            ``(num_temps, num_species)``, including ``(1, num_species)`` for a
            scalar temperature, in the phase's component order.

        Raises
        ------
        ValueError
            If ``basis`` is neither 'mass' nor 'mole', for both ``total_h``
            modes, raised by ``ThermoPhysicalManager.getEnthalpy``.
        MissingPropertyError
            A subclass of ``AttributeError`` and ``ValueError``. If a
            species that the result needs has no ``cp_solid`` data: every
            species for ``total_h=False``, species with a nonzero fraction
            otherwise. In the stored composition, a species at the zero
            sentinel ``eps`` is absent (see ``_stored_fractions``).
            Also raised, whatever the fractions or weights, if the property is
            stored as one scalar per species instead of a coefficient list (see
            ``_property_rows``).

        Notes
        -----
        Species enthalpy is the integral of the ``cp_solid`` polynomial
        [J/mol/K] from ``temp_ref`` to ``temp``. The mass basis divides it by
        the species molecular weight [kg/mol].

        Before issue #332, an omitted ``temp_ref`` always used 298.15 K and
        ignored a nondefault constructor reference. Callers that relied on
        that behavior should pass ``temp_ref=298.15`` explicitly. Mixture
        providers such as ``Slurry`` and ``Cake`` pass their own common
        reference to both phases, so the stored solid reference does not mix
        references inside a mixture balance.
        """

        if mass_frac is None and mole_frac is None:
            mass_frac, mole_frac = self._stored_fractions()  # [-]

        if temp is None:
            temp = self.temp

        if temp_ref is None:
            temp_ref = self.temp_ref  # [K]

        hSolid = super().getEnthalpy(temp, temp_ref, mass_frac, mole_frac,
                                     phase='solid', total_h=total_h,
                                     basis=basis)

        return hSolid
