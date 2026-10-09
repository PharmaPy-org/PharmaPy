# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 17:28:24 2020

@author: dcasasor
"""

import json
import pathlib
import warnings
from typing import Any, Dict, Optional, Sequence, Union

import numpy as np
import pandas as pd


_DatabasePath = Union[str, pathlib.Path]


VALID_ACTIVITY_MODELS = ('ideal', 'UNIFAC', 'UNIQUAC')

# Database key of the pure-component heat-capacity polynomial of each phase.
_CP_PROPERTY_BY_PHASE = {'liquid': 'cp_liq', 'solid': 'cp_solid',
                         'vapor': 'cp_vapor'}

# Meaning of the list-valued properties that the property methods evaluate,
# and the output unit and coefficient convention of each correlation, quoted
# in missing-data errors. Temperatures T are in [K].
_LIST_PROPERTY_DESCRIPTIONS = {
    'cp_liq': 'liquid heat-capacity coefficients',
    'cp_solid': 'solid heat-capacity coefficients',
    'cp_vapor': 'vapor heat-capacity coefficients',
    'visc_liq': 'liquid-viscosity coefficients',
    'p_vap': 'Antoine vapor-pressure coefficients',
    'diffusivity': 'diffusivities',
}
_LIST_PROPERTY_UNITS = {
    'cp_liq': '(cp [J/mol/K] = sum_k c_k * T**k)',
    'cp_solid': '(cp [J/mol/K] = sum_k c_k * T**k)',
    'cp_vapor': '(cp [J/mol/K] = sum_k c_k * T**k)',
    'visc_liq': '(log10(mu / [mPa*s]) = A + B/T + C*T + D*T**2)',
    'p_vap': '(log10(p / [Pa]) = A - B/(T + C))',
    'diffusivity': '([m**2/s], one column per reference species)',
}
# Coefficient count of each fixed-form correlation above (Antoine A, B, C;
# viscosity A, B, C, D), used to shape the all-NaN rows of a property that
# no species supplies. Polynomials and diffusivity rows take any length.
_CORRELATION_WIDTHS = {'p_vap': 3, 'visc_liq': 4}


class MissingPropertyError(AttributeError):
    """Property data that a result needs are missing for some species.

    Raised by the property methods when a needed species has no data for a
    property (a NaN row, see ``ParseDatabase``, or a property that the
    database lacks entirely). The message names the species and the
    property. It subclasses ``AttributeError``, the type raised for a
    missing property before #414, so existing handlers still catch it.
    Reactor solves (``Reactors._BaseReactor._simulate``) use the subclass to
    recognize it and re-raise it, chained from the CVode error, when the
    solver hides the original exception. In other solver-based unit
    operations such an error can still be lost behind the solver's error,
    or stall the solver, which issue #435 tracks; the exception-type policy
    across the property methods is issue #424.
    """


def validate_activity_model(model: str, param_name: str = 'gamma_model') -> None:
    """Validate a case-sensitive activity-coefficient model selector.

    Parameters
    ----------
    model : {'ideal', 'UNIFAC', 'UNIQUAC'}
        Activity model selector. Activity coefficients have molar basis [-].
    param_name : str, optional
        Public parameter name to include in the error message.

    Raises
    ------
    ValueError
        If the selector is not one of ``VALID_ACTIVITY_MODELS``.

    Notes
    -----
    Public names ``gamma_model``, ``gamma_method``, ``activity_model``, and
    ``method`` retain their existing names and defaults. Only the three exact
    spellings above are accepted; unknown names no longer fall through to a
    different model. This validates selection, not availability of the model's
    required property data. Extractors retain their public validator alias.
    """
    if model not in VALID_ACTIVITY_MODELS:
        raise ValueError(
            f"{param_name} must be one of {VALID_ACTIVITY_MODELS}, "
            f"got {model!r}")


def ParseDatabase(
        path_datafile: Union[_DatabasePath, Sequence[_DatabasePath]],
        to_arrays: bool = True) -> Dict[str, Any]:
    """Parse one or more component-property JSON databases.

    Parameters
    ----------
    path_datafile : str, pathlib.Path, or sequence of path-like
        JSON database path or ordered paths to merge. Property values retain
        the units and physical basis declared by their source database.
    to_arrays : bool, optional
        If True, homogeneous numeric properties are returned as float arrays;
        non-numeric properties retain their list representation.

    Returns
    -------
    dict
        Property names mapped to arrays or lists. Numeric arrays preserve the
        source property's units and basis; ``name_species`` contains component
        identifiers [-].

    Raises
    ------
    OSError
        If a database path cannot be opened.
    json.JSONDecodeError
        If a database does not contain valid JSON.
    KeyError
        If a structured property lacks its required ``value`` field.
    TypeError
        If a structured property value cannot be converted to a float array,
        or a species supplies a scalar for a property that another species
        supplies as a list; the message names the species and the property.
    OverflowError
        If a numeric property lies outside the representable float range.

    Notes
    -----
    Only ``ValueError`` indicates the established non-numeric-property case.
    Structural and range errors propagate so malformed scientific data are not
    silently returned on a different representation or basis.

    Missing data are marked with NaN, for scalar and list-valued properties
    alike. A scalar property such as ``t_crit`` or ``henry_constant`` is NaN
    for a species that omits it. A list-valued property, such as the
    correlation coefficients ``cp_liq``, ``cp_solid``, ``cp_vapor``,
    ``visc_liq`` or ``p_vap``, or ``diffusivity`` rows, becomes a float array
    of shape ``(num_species, num_coefficients)`` in species order, where
    ``num_coefficients`` is the length of the longest supplied list. A species
    that supplies no coefficients (key absent, ``null`` or an empty list)
    gets a row of NaN. A NaN row cannot be mistaken for coefficients, unlike
    the row of zeros that earlier releases stored, which evaluated as, for
    example, a zero heat capacity. The property methods of
    ``ThermoPhysicalManager`` raise ``MissingPropertyError`` (a subclass of
    ``AttributeError``) naming the species and the property when a result
    needs a NaN row.

    A property that no species supplies creates no key: an omitted key, and
    also a list-valued key (one that some species gives as a list, or one of
    ``cp_liq``, ``cp_solid``, ``cp_vapor``, ``visc_liq``, ``p_vap`` and
    ``diffusivity``) for which every species gives ``null`` or an empty
    list. The property methods then report the property as missing for every
    species that needs it.

    A shorter list is padded with trailing zeros. For the correlation
    coefficients ``cp_liq``, ``cp_solid``, ``cp_vapor``, ``visc_liq`` and
    ``p_vap`` this is exact, because a trailing zero is an absent term: of
    the ascending heat-capacity polynomial, of the four-term ``visc_liq``
    correlation, or ``C = 0`` of a two-term Antoine list. A short
    ``diffusivity`` row is padded the same way; that is unchanged legacy
    behavior, not a physical statement. Once one species supplies a list,
    every species that supplies the property must supply a list. Scalars for
    every species keep the scalar representation. With ``to_arrays=False``,
    the raw values, including ``None`` and empty lists, are returned.
    """
    if isinstance(path_datafile, (list, tuple)):
        with open(path_datafile[0]) as file:
            original_data = json.load(file)

        for path in path_datafile[1:]:
            with open(path) as file:
                data = json.load(file)

            original_data.update(data)
    else:
        with open(path_datafile) as file:
            original_data = json.load(file)

    entries = []

    for key, dat in original_data.items():
        entries.append(dat.keys())

    # Extract interaction data, if existent
    interac = original_data.pop('interaction', None)

    # Collect data with the same key
    entries = set().union(*entries)

    dd = {}

    for entry in entries:
        vals = []
        tref_hvap = []
        for val in original_data.values():
            item = val.get(entry)
            if isinstance(item, dict):
                item = item['value']

                if entry == 'delta_hvap':
                    tref_hvap.append(val.get(entry)['temp_ref'])

            vals.append(item)

        dd[entry] = vals

        if len(tref_hvap) > 0:
            dd['tref_hvap'] = tref_hvap

    # Convert to arrays
    if to_arrays:
        dd_arrays = {}
        names = list(original_data.keys())
        for key, vals in dd.items():

            islist = [type(val) is list for val in vals]
            lengths = [len(val) for val in vals if isinstance(val, list)]
            is_empty = [val is None or (isinstance(val, list) and not val)
                        for val in vals]

            if all(is_empty) and (any(islist)
                                  or key in _LIST_PROPERTY_DESCRIPTIONS):
                # No species supplies coefficients: same as an omitted key.
                continue

            if any(islist):  # there is multidimensional data
                length = max(lengths)  # [-], number of coefficients
                props = []
                for name, val, empty in zip(names, vals, is_empty):
                    if empty:
                        # Missing-data sentinel, as for scalar properties
                        props.append(np.full(length, np.nan))
                    elif not isinstance(val, list):
                        raise TypeError(
                            f"Property {key!r} of species {name!r} must be a "
                            "list, as other species supply a list for it; "
                            f"got {val!r}")
                    else:
                        val_array = np.zeros(length)
                        val_array[:len(val)] = val
                        props.append(val_array)

            else:
                props = vals
            try:
                props = np.array(props, dtype=float)
            except ValueError:
                pass  # non-numeric property (e.g. strings), keep as list

            dd_arrays[key] = props

        dd_arrays['name_species'] = list(original_data.keys())

        if interac is not None:
            # Convert interaction parameters
            interac['amk'] = np.array(interac.get('amk', None))
            interac['vk'] = np.array(interac.get('vk', None))

            if 'unifac_groups' in interac:  # Avoid pandas xs warning
                interac['unifac_groups'] = [
                    tuple(pair) for pair in interac['unifac_groups']]

            dd_arrays.update(interac)
        return dd_arrays
    else:
        dd['name_species'] = list(original_data.keys())
        return dd


class ThermoPhysicalManager:
    def __init__(self, path_data):

        props_dict = ParseDatabase(path_data)
        self.__dict__ = props_dict
        self.name_species = props_dict['name_species']

        self.num_species = len(self.name_species)

        self.path_data = path_data

        # UNIFAC
        if 'unifac_groups' in props_dict:
            rk, qk, a_mat, b_mat, c_mat = self.get_UNIFACParams()
            self.Rk = rk
            self.Qk = qk
            self.a_unifac = a_mat
            self.b_unifac = b_mat
            self.c_unifac = c_mat

    def selectProperties(self, names):
        mapping = dict((key, ind)
                       for ind, key in enumerate(self.compound_names))

        idx_selection = [mapping[key] for key in names]

        for key, val in self.__dict__.items():
            values = getattr(self, key)
            if type(val) is list:
                vals = [values[i] for i in idx_selection]
                setattr(self, key, vals)
            else:
                vals = values[idx_selection]
                setattr(self, key, vals)

    def set_object(self):
        self.cpLiqPure = self.getCpLiqPure(temp=5)

    def _property_rows(self, prop_name: str) -> np.ndarray:
        """Return a list-valued property as per-species coefficient rows.

        Parameters
        ----------
        prop_name : str
            Database key of the property, for example ``'cp_liq'``.

        Returns
        -------
        ndarray
            ``np.atleast_2d`` of the stored property, the representation that
            the property methods index by species, with the property's units:
            shape ``(num_species, num_coefficients)`` for parsed list-valued
            data. A NaN row marks a species without data (see
            ``ParseDatabase``). A property that every species omits gives
            NaN of shape ``(num_species, num_coefficients)``, with the
            coefficient count of ``_CORRELATION_WIDTHS`` (one for a
            polynomial or a diffusivity column), so it takes the same
            missing-data path and the correlations still unpack it.
        """
        if not hasattr(self, prop_name):
            width = _CORRELATION_WIDTHS.get(prop_name, 1)  # [-]
            return np.full((len(self.name_species), width), np.nan)

        return np.atleast_2d(getattr(self, prop_name))

    def _check_species_rows(self, prop_name: str, rows: np.ndarray,
                            species_idx: Sequence[int], needed: np.ndarray,
                            reason: str, remedy: str = '') -> np.ndarray:
        """Flag missing property rows and reject those a result needs.

        Parameters
        ----------
        prop_name : str
            Database key of the list-valued property, a key of
            ``_LIST_PROPERTY_DESCRIPTIONS``.
        rows : ndarray
            Property rows of the selected species, in ``species_idx`` order,
            shape ``(num_selected, num_coefficients)``; units of the property.
        species_idx : sequence of int
            Database index of each selected species, shape
            ``(num_selected,)``.
        needed : ndarray of bool
            True for a selected species whose data the result uses, shape
            ``(num_selected,)``.
        reason : str
            Clause completing "Species [...] <reason> needs <property>", for
            example ``'carry a nonzero fraction, so the mixture enthalpy'``.
        remedy : str, optional
            Alternative to supplying the data, appended to the message.

        Returns
        -------
        ndarray of bool
            True where a selected species has no data (its row contains NaN),
            shape ``(num_selected,)``. Callers give such species exactly zero
            weight when they are not needed, instead of letting NaN times a
            zero fraction turn the result into NaN.

        Raises
        ------
        MissingPropertyError
            If a needed species has no data. The message names the species
            and the property, as for a property that the database lacks
            entirely. It is an ``AttributeError``.
        """
        missing = np.isnan(rows).any(axis=1)
        lacking = np.flatnonzero(missing & needed)
        if lacking.size > 0:
            names = [self.name_species[species_idx[ind]] for ind in lacking]
            raise MissingPropertyError(
                f"Species {names} {reason} needs "
                f"{_LIST_PROPERTY_DESCRIPTIONS[prop_name]} '{prop_name}' "
                f"{_LIST_PROPERTY_UNITS[prop_name]}, which the property "
                f"database {self.path_data!r} does not supply for them; add "
                f"'{prop_name}' for these species{remedy}")

        return missing

    @staticmethod
    def _needed_by_weights(weights: Optional[np.ndarray],
                           num_species: int) -> np.ndarray:
        """Mark the species that a weighted sum needs.

        Parameters
        ----------
        weights : ndarray or None
            Weights of the per-species values [any basis], shape
            ``(num_species,)`` or ``(num_rows, num_species)``; ``None`` means
            every species is needed.
        num_species : int
            Number of species columns.

        Returns
        -------
        ndarray of bool
            True for a species whose weight is nonzero in some row, shape
            ``(num_species,)``. Nonzero is bitwise: a round-off-level weight
            counts as present, and NaN counts as nonzero.
        """
        if weights is None:
            return np.ones(num_species, dtype=bool)

        return np.atleast_2d(np.asarray(weights) != 0).any(axis=0)

    def _cp_rows(self, phase: str) -> tuple:
        """Return the heat-capacity property of a phase and its rows.

        Parameters
        ----------
        phase : {'liquid', 'solid', 'vapor'}
            Phase whose heat-capacity polynomial is requested.

        Returns
        -------
        prop_name : str
            ``'cp_liq'``, ``'cp_solid'`` or ``'cp_vapor'``.
        cp_cts : ndarray
            Ascending polynomial coefficients [J/mol/K/K**k], shape
            ``(num_species, num_coefficients)``; NaN rows mark species
            without data.

        Raises
        ------
        ValueError
            If ``phase`` is not one of the three phases.
        """
        if phase not in _CP_PROPERTY_BY_PHASE:
            raise ValueError(
                f"phase must be one of {tuple(_CP_PROPERTY_BY_PHASE)}, "
                f"got {phase!r}")

        prop_name = _CP_PROPERTY_BY_PHASE[phase]
        return prop_name, self._property_rows(prop_name)

    def getCpPure(self, temp: Union[float, np.ndarray],
                  phase: str = 'liquid',
                  weights: Optional[np.ndarray] = None) -> tuple:
        """Evaluate pure-component heat capacities.

        Parameters
        ----------
        temp : float or array-like
            Temperature [K], scalar or shape ``(num_temps,)``.
        phase : {'liquid', 'solid', 'vapor'}, optional
            Selects the ``cp_liq``, ``cp_solid`` or ``cp_vapor`` polynomial;
            default liquid.
        weights : ndarray, optional
            Amounts by which the caller weights the returned values [any
            basis, e.g. mol/L], shape ``(num_species,)`` or
            ``(num_rows, num_species)``. When given, only species with a
            nonzero weight in some row need coefficients.

        Returns
        -------
        cp_mass : ndarray
            Heat capacities [J/kg/K], shape ``(num_species,)`` for one
            temperature and ``(num_temps, num_species)`` otherwise.
        cp_mole : ndarray
            Heat capacities [J/mol/K], same shape as ``cp_mass``.

        Raises
        ------
        ValueError
            If ``phase`` is not one of the three phases.
        MissingPropertyError
            A subclass of ``AttributeError``.
            If a needed species has no coefficients for the phase. Without
            ``weights`` every species is returned, so every species is
            needed.

        Notes
        -----
        With ``weights``, a species without coefficients whose weight is zero
        in every row is returned as exactly 0, so that the caller's weighted
        sum is exact (never NaN times zero). That 0 is a placeholder for a
        zero-weight term, not a heat capacity. Nonzero is bitwise: a
        round-off-level weight counts as present.
        """
        prop_name, cp_cts = self._cp_rows(phase)
        num_sp = len(cp_cts)
        if weights is None:
            reason = 'are evaluated as pure components, so their heat capacity'
        else:
            reason = 'carry a nonzero weight, so their heat capacity'

        missing = self._check_species_rows(
            prop_name, cp_cts, np.arange(num_sp),
            self._needed_by_weights(weights, num_sp), reason)
        cp_mass, cp_mole = self._evaluate_cp_pure(temp, cp_cts)  # [J/kg/K], [J/mol/K]

        return (np.where(missing, 0.0, cp_mass),
                np.where(missing, 0.0, cp_mole))

    def _evaluate_cp_pure(self, temp: Union[float, np.ndarray],
                          cp_cts: np.ndarray) -> tuple:
        """Evaluate heat-capacity polynomials without checking for data.

        Parameters
        ----------
        temp : float or array-like
            Temperature [K], scalar or shape ``(num_temps,)``.
        cp_cts : ndarray
            Ascending polynomial coefficients [J/mol/K/K**k], shape
            ``(num_species, num_coefficients)``.

        Returns
        -------
        cp_mass : ndarray
            Heat capacities [J/kg/K], shape ``(num_species,)`` for one
            temperature and ``(num_temps, num_species)`` otherwise; NaN for a
            species with a NaN row.
        cp_mole : ndarray
            Heat capacities [J/mol/K], same shape as ``cp_mass``.
        """
        temp = np.atleast_1d(temp)
        num_temp = len(temp)

        num_sp = len(cp_cts)
        ind_poly = np.arange(cp_cts.shape[1])

        cpMole = np.zeros((num_temp, num_sp))
        for ind, val in enumerate(temp):
            cpMole[ind] = np.dot(cp_cts, val**ind_poly)  # J/mol_j

        cpMass = cpMole / self.mw * 1000  # J/kg/K

        if len(cpMole) == 1:
            cpMole = cpMole[0]
            cpMass = cpMass[0]

        return cpMass, cpMole

    def getCpMix(self, temp, mass_frac=None, mole_frac=None, phase='liquid',
                 basis: str = 'mass'):
        """Mix pure heat capacities over the species axis.

        Parameters
        ----------
        temp : float or array-like
            Temperature [K], scalar or shape (num_temperatures,).
        mass_frac, mole_frac : ndarray, optional
            Species fractions [-], shape (num_species,) for fixed composition
            or (num_temperatures, num_species) for paired profiles. The
            requested basis takes precedence; the other basis is converted
            when needed.
        phase : {'liquid', 'solid', 'vapor'}, optional
            Pure-component Cp correlation; default liquid.
        basis : {'mass', 'mole'}, optional
            Cp basis, default mass.

        Returns
        -------
        float or ndarray
            Mixture Cp [J/kg/K] for mass or [J/mol/K] for mole. A single
            temperature and fixed composition return a scalar; multiple
            temperatures return shape (num_temperatures,). Composition
            profiles retain their row axis, including a one-row profile.

        Raises
        ------
        ValueError
            If basis is neither 'mass' nor 'mole', or ``phase`` is unknown.
        MissingPropertyError
            A subclass of ``AttributeError``.
            If a species with a nonzero fraction in some composition row has
            no coefficients for the phase. A species whose fraction is zero
            in every row needs none and contributes exactly zero. Nonzero is
            bitwise: a round-off-level fraction counts as present.
        """
        if basis not in ('mass', 'mole'):
            raise ValueError("basis must be 'mass' or 'mole'")
        prop_name, cp_cts = self._cp_rows(phase)
        cp_mass, cp_mole = self._evaluate_cp_pure(temp, cp_cts)  # [J/kg/K], [J/mol/K]

        if basis == 'mass':
            if mass_frac is None:
                mass_frac = self.frac_to_frac(mole_frac=mole_frac)

            frac = mass_frac  # [-]
            cp_pure = cp_mass  # [J/kg/K]

        elif basis == 'mole':
            if mole_frac is None:
                mole_frac = self.frac_to_frac(mass_frac)

            frac = mole_frac  # [-]
            cp_pure = cp_mole  # [J/mol/K]

        # Only species present in some composition row need coefficients;
        # absent species without them contribute exactly zero, not NaN * 0.
        needed = np.atleast_2d(np.asarray(frac) != 0).any(axis=0)
        missing = self._check_species_rows(
            prop_name, cp_cts, np.arange(len(cp_cts)), needed,
            'carry a nonzero fraction, so the mixture heat capacity')
        cp_pure = np.where(missing, 0.0, cp_pure)  # [J/kg/K] or [J/mol/K]

        if frac.ndim == 1:
            cpMix = np.dot(cp_pure, frac)  # [J/kg/K] or [J/mol/K]
        elif frac.ndim == 2:
            cpMix = (frac * cp_pure).sum(axis=1)  # [J/kg/K] or [J/mol/K]

        return cpMix

    def getEnthalpy(self, temp: Union[float, np.ndarray],
                    temp_ref: float = 298.15,
                    mass_frac: Optional[np.ndarray] = None,
                    mole_frac: Optional[np.ndarray] = None,
                    phase: str = 'liquid', basis: str = 'mass',
                    idx: Optional[Sequence[int]] = None,
                    total_h: bool = True,
                    weights: Optional[np.ndarray] = None
                    ) -> Union[float, np.ndarray]:
        """Integrate pure-component heat capacities into sensible enthalpies.

        Parameters
        ----------
        temp : float or array-like
            Temperature [K], scalar or shape ``(num_temps,)``.
        temp_ref : float, optional
            Lower limit of the heat-capacity integral [K]; default 298.15 K.
        mass_frac, mole_frac : ndarray, optional
            Species fractions [-], shape ``(num_species,)`` for a fixed
            composition or ``(num_rows, num_species)`` for a composition
            profile. Used only when ``total_h`` is ``True``. The fraction
            matching ``basis`` is used when supplied; otherwise the other one
            is converted.
        phase : {'liquid', 'solid', 'vapor'}, optional
            Selects the ``cp_liq``, ``cp_solid`` or ``cp_vapor`` heat-capacity
            polynomial [J/mol/K]; default liquid.
        basis : {'mass', 'mole'}, optional
            Physical basis of the returned enthalpy and of the weighting
            fractions; default mass.
        idx : array-like of int, optional
            Species indices to evaluate, in the requested order. All species
            are evaluated when ``None``. The matching fraction columns weight
            the selected species without renormalization.
        total_h : bool, optional
            If ``True`` (default), return the fraction-weighted sum over the
            selected species. If ``False``, return the species enthalpies.
        weights : ndarray, optional
            Used only with ``total_h=False``: amounts by which the caller
            weights the returned species enthalpies [any basis, e.g. mol/L],
            shape ``(num_species,)`` or ``(num_rows, num_species)`` in
            database order. When given, only selected species with a nonzero
            weight in some row need coefficients, and the others are returned
            as exactly 0 if they have none.

        Returns
        -------
        float or ndarray
            Sensible enthalpy, [J/kg] for ``basis='mass'`` and [J/mol] for
            ``basis='mole'``. With ``total_h=True``, one value per
            temperature or composition row: the ``(num_temps,
            num_selected_species)`` species enthalpies broadcast against the
            fractions, giving shape ``(num_temps,)`` for a fixed composition,
            ``(num_rows,)`` for a single temperature with a profile, and
            row-paired ``(num_temps,)`` when ``num_temps == num_rows``. A
            scalar is returned only when that result has one entry, that is,
            a single temperature with a fixed composition or a one-row
            profile. With ``total_h=False``, shape ``(num_temps,
            num_selected_species)``, including ``(1, num_selected_species)``
            for a scalar temperature.

        Raises
        ------
        ValueError
            If ``basis`` is neither 'mass' nor 'mole'. The check precedes any
            computation and applies to both ``total_h`` modes. NumPy also
            raises it, for ``total_h=True``, when ``num_temps`` and
            ``num_rows`` differ and neither is one. Also raised for an
            unknown ``phase``.
        MissingPropertyError
            A subclass of ``AttributeError``.
            If a selected species that the result needs has no
            coefficients for the phase (a NaN row, see ``ParseDatabase``).
            With ``total_h=False`` every selected species is needed, or,
            with ``weights``, every one with a nonzero weight. With
            ``total_h=True`` only species with a nonzero fraction in some
            composition row are; the others contribute exactly zero.
            Nonzero is bitwise: a round-off-level fraction or weight counts
            as present, including round-off that a solver leaves in a
            fraction that should be zero; reactors therefore pass a
            structural presence mask as ``weights``.

        Notes
        -----
        With ascending coefficients ``cp = sum_k c_k * T**k`` [J/mol/K], the
        species molar enthalpy is
        ``sum_k c_k / (k + 1) * (temp**(k + 1) - temp_ref**(k + 1))`` [J/mol].
        The mass basis multiplies it by ``1000 / mw`` [g/kg / (g/mol)].
        """
        if basis not in ('mass', 'mole'):
            raise ValueError("basis must be 'mass' or 'mole'")

        temp = np.atleast_1d(temp)

        if idx is None:
            idx = np.arange(len(self.mw))

        prop_name, cp_cts = self._cp_rows(phase)
        cp_cts = cp_cts[idx]
        species_idx = np.asarray(idx)

        if not total_h:
            if weights is None:
                needed = np.ones(len(cp_cts), dtype=bool)
                reason = ('are evaluated individually (total_h=False), so '
                          'their enthalpy')
            else:
                needed = self._needed_by_weights(
                    np.asarray(weights)[..., species_idx], len(cp_cts))
                reason = 'carry a nonzero weight, so their enthalpy'

            missing = self._check_species_rows(
                prop_name, cp_cts, species_idx, needed, reason)

        ind_poly = np.arange(cp_cts.shape[1])
        exp = ind_poly + 1

        integral = []
        for ind, val in enumerate(temp):
            temp_term = val**exp - temp_ref**exp
            integral.append(np.dot(cp_cts / exp, temp_term))

        integral = np.vstack(integral)

        if total_h:
            if basis == 'mass':
                integralMass = integral * 1000 / self.mw[idx]  # J/kg_i
                if mass_frac is None:
                    mass_frac = self.frac_to_frac(mole_frac=mole_frac, ind=ind)

                if mass_frac.ndim == 1:
                    frac_selected = mass_frac[idx]  # [-]
                else:
                    frac_selected = mass_frac[:, idx]  # [-]

                h_selected = integralMass  # [J/kg]
            elif basis == 'mole':
                if mole_frac is None:
                    mole_frac = self.frac_to_frac(mass_frac, ind=ind)

                if mole_frac.ndim == 1:
                    frac_selected = mole_frac[idx]  # [-]
                else:
                    frac_selected = mole_frac[:, idx]  # [-]

                h_selected = integral  # [J/mol]

            # Only species present in some composition row need coefficients;
            # absent species without them contribute exactly zero.
            needed = np.atleast_2d(frac_selected != 0).any(axis=0)
            missing = self._check_species_rows(
                prop_name, cp_cts, species_idx, needed,
                'carry a nonzero fraction, so the mixture enthalpy')
            h_selected = np.where(missing, 0.0, h_selected)  # [J/kg] or [J/mol]

            enthalpyOut = (h_selected * frac_selected).sum(axis=1)  # [J/kg] or [J/mol]

            if len(enthalpyOut) == 1:
                enthalpyOut = enthalpyOut[0]

            return enthalpyOut
        else:
            if basis == 'mass':
                integralOut = integral * 1000 / self.mw[idx]  # J/kg_i
            else:
                integralOut = integral

            # Zero-weight species without data: exact 0 for weighted sums.
            return np.where(missing, 0.0, integralOut)  # [J/kg] or [J/mol]

    def getHeatOfRxn(self, stoich_matrix: np.ndarray,
                     temp: Union[float, np.ndarray], mask: np.ndarray,
                     heat_rxn_ref: Union[float, np.ndarray],
                     tref_hrxn: float) -> np.ndarray:
        """Correct reference heats of reaction to the requested temperature.

        Parameters
        ----------
        stoich_matrix : ndarray
            Stoichiometric coefficients [-], shape
            ``(num_reactions, num_participating_species)``, columns in the
            order of the species that ``mask`` selects.
        temp : float or array-like
            Temperature [K], scalar or shape ``(num_temps,)``.
        mask : ndarray of bool or int
            Selects the participating species from the database order, as an
            index into ``cp_liq``.
        heat_rxn_ref : float or ndarray
            Heats of reaction at ``tref_hrxn`` [J/mol of reaction as
            written], scalar or shape ``(num_reactions,)``.
        tref_hrxn : float
            Reference temperature of ``heat_rxn_ref`` [K].

        Returns
        -------
        ndarray
            Heats of reaction [J/mol of reaction as written],
            ``heat_rxn_ref + sum_j nu_rj * integral(cp_liq_j, tref_hrxn,
            temp)``: shape ``(num_reactions,)`` for one temperature and
            ``(num_temps, num_reactions)`` otherwise.

        Raises
        ------
        MissingPropertyError
            A subclass of ``AttributeError``.
            If a selected species with a nonzero stoichiometric coefficient
            in some reaction has no ``cp_liq`` coefficients. A selected
            species that no reaction involves needs none and contributes
            exactly zero.
        """
        temp = np.atleast_1d(temp)
        temp_ref = tref_hrxn

        n_temp = len(temp)
        n_rxns, n_species = stoich_matrix.shape

        cp_cts = self._property_rows('cp_liq')[mask]  # [J/mol/K/K**k]
        reacting = (np.asarray(stoich_matrix) != 0).any(axis=0)
        missing = self._check_species_rows(
            'cp_liq', cp_cts, np.arange(len(self.name_species))[mask],
            reacting, 'take part in a reaction (nonzero stoichiometric '
            'coefficient), so the heat of reaction')

        ind_poly = np.arange(cp_cts.shape[1])
        exp = ind_poly + 1

        integral = np.zeros((n_temp, n_species))
        for ind, val in enumerate(temp):
            temp_term = val**exp - temp_ref**exp
            integral[ind] = np.dot(cp_cts / exp, temp_term)  # J/mol_j

        # Species that no reaction involves contribute exactly zero.
        integral = np.where(missing, 0.0, integral)  # [J/mol]

        delta_cp = np.dot(integral, stoich_matrix.T)
        heat_of_rxn = heat_rxn_ref + delta_cp

        if len(heat_of_rxn) == 1:
            heat_of_rxn = heat_of_rxn[0]

        return heat_of_rxn  # J/mol

    def getDensityPure(self, phase='liquid', temp=None):
        if phase == 'liquid':
            rhoMass = self.rho_liq  # TODO: T-dependent rho
        elif phase == 'solid':
            rhoMass = self.rho_solid

        rhoMole = rhoMass / self.mw  # kmol/m**3  (mol/L)

        return rhoMass, rhoMole

    def getDensityMix(self, mass_frac: Optional[np.ndarray] = None,
                      mole_frac: Optional[np.ndarray] = None,
                      phase: str = 'liquid', temp: Optional[float] = None,
                      basis: str = 'mass') -> Union[float, np.ndarray]:
        """Mix pure-component densities assuming ideal (additive) volumes.

        Parameters
        ----------
        mass_frac, mole_frac : ndarray, optional
            Species fractions [-], shape ``(num_species,)`` or
            ``(num_points, num_species)``. The fraction matching ``basis`` is
            used when supplied; otherwise the other one is converted.
        phase : {'liquid', 'solid'}, optional
            Selects the ``rho_liq`` or ``rho_solid`` pure density
            [kg/m**3]; default liquid.
        temp : float, optional
            Temperature [K]; defaults to ``self.temp``. The pure densities
            are temperature independent, so it does not change the result.
        basis : {'mass', 'mole'}, optional
            Physical basis of the returned density; default mass.

        Returns
        -------
        float or ndarray
            Mixture density, [kg/m**3] for ``basis='mass'`` and [kmol/m**3]
            (equivalently [mol/L]) for ``basis='mole'``. Scalar for a
            one-dimensional composition, shape ``(num_points,)`` otherwise.

        Raises
        ------
        ValueError
            If ``basis`` is neither 'mass' nor 'mole'. The check precedes any
            computation.

        Notes
        -----
        ``1 / rho_mix = sum_i frac_i / rho_i`` on the selected basis, with
        pure molar densities ``rho_i / mw_i`` [kmol/m**3].
        """
        if basis not in ('mass', 'mole'):
            raise ValueError("basis must be 'mass' or 'mole'")

        if temp is None:
            temp = self.temp

        rhoMass, rhoMole = self.getDensityPure(phase, temp)

        if basis == 'mass':
            if mass_frac is None:
                mass_frac = self.frac_to_frac(mole_frac=mole_frac)

            rhoMix = 1 / np.dot(mass_frac, 1 / rhoMass)

        else:
            if mole_frac is None:
                mole_frac = self.frac_to_frac(mass_frac=mass_frac)

            rhoMix = 1 / np.dot(mole_frac, 1 / rhoMole)

        return rhoMix

    def getMolWeight(self, mole_frac=None, mass_frac=None):
        if mass_frac is None and mole_frac is None:
            mole_frac = self.mole_frac
        elif mass_frac is not None:
            mole_frac = self.frac_to_frac(mass_frac=mass_frac)

        mw_av = np.dot(self.mw, mole_frac.T)

        return mw_av

    def getViscosityPure(self, phase: str = 'liquid',
                         temp: Optional[Union[float, np.ndarray]] = None) -> np.ndarray:
        """Return pure-component dynamic viscosities.

        Parameters
        ----------
        phase : {'liquid', 'vapor'}, optional
            Phase whose pure viscosities are requested.
        temp : float or ndarray, optional
            Temperature [K]. The liquid branch accepts a scalar only and
            defaults to the phase temperature. The vapor branch ignores it.

        Returns
        -------
        ndarray
            Pure viscosities [Pa*s], shape (num_species,). Vapor values come
            directly from the database's ``visc_gas`` entries in species order.

        Raises
        ------
        ValueError
            If ``phase`` is unknown, or vapor ``visc_gas`` data [Pa*s] are
            absent or contain NaN entries for missing species.
        MissingPropertyError
            A subclass of ``AttributeError``.
            If any species has no liquid ``visc_liq`` coefficients; every
            species is returned, so every species needs them.

        Notes
        -----
        Vapor data are supplied constants [Pa*s], with no temperature
        correlation. The database provider must establish their applicable
        temperature and pressure range; shipped databases do not supply them.
        Liquid correlations retain their existing temperature dependence.
        """
        if phase not in ('liquid', 'vapor'):
            raise ValueError(
                f"phase must be one of ('liquid', 'vapor'), got {phase!r}")

        if phase == 'liquid':
            visc_cts = self._property_rows('visc_liq')  # [-], [K], [1/K], [1/K**2]
            num_sp = len(visc_cts)
            self._check_species_rows(
                'visc_liq', visc_cts, np.arange(num_sp),
                np.ones(num_sp, dtype=bool),
                'are evaluated as pure components, so their liquid viscosity')
            viscosity = self._evaluate_liquid_viscosity(temp, visc_cts)  # [Pa*s]

        elif phase == 'vapor':
            if (not hasattr(self, 'visc_gas')
                    or np.isnan(self.visc_gas).any()):
                raise ValueError(
                    "Vapor viscosity requires the missing 'visc_gas' property "
                    "[Pa*s]. Supply a pure-component value for every species "
                    "in the property JSON, valid at the requested conditions.")
            viscosity = self.visc_gas  # [Pa*s]

        return viscosity

    def _evaluate_liquid_viscosity(self, temp: Optional[float],
                                   visc_cts: np.ndarray) -> np.ndarray:
        """Evaluate the liquid-viscosity correlation without checking data.

        Parameters
        ----------
        temp : float or None
            Temperature [K], scalar; the phase temperature when ``None``.
        visc_cts : ndarray
            Coefficients ``A, B, C, D`` of
            ``log10(mu / [mPa*s]) = A + B/T + C*T + D*T**2``, with units
            [-], [K], [1/K] and [1/K**2]; shape ``(num_species, 4)``.

        Returns
        -------
        ndarray
            Pure liquid viscosities [Pa*s], shape ``(num_species,)``; NaN for
            a species with a NaN row.
        """
        if temp is None:
            temp = self.temp  # [K]

        temp_term = np.array([1, 1/temp, temp, temp**2])  # [-], [1/K], [K], [K**2]

        viscosity = 10**(np.dot(visc_cts, temp_term))/1000  # [Pa*s], 1000 mPa*s per Pa*s

        return viscosity

    def getViscosityMix(self, temp: Optional[Union[float, np.ndarray]] = None,
                        mass_frac: Optional[np.ndarray] = None,
                        mole_frac: Optional[np.ndarray] = None,
                        phase: str = 'liquid') -> Union[float, np.ndarray]:
        """Mix dynamic viscosities on the mole-fraction basis.

        Parameters
        ----------
        temp : float or ndarray, optional
            Temperature [K]. The liquid branch accepts a scalar only and
            defaults to the phase temperature. The vapor branch ignores it.
        mass_frac, mole_frac : ndarray, optional
            Mass or mole fractions [-], shape (num_species,) or
            (num_points, num_species). Mole fractions take precedence;
            if neither is supplied, use the phase mole fractions.
        phase : {'liquid', 'vapor'}, optional
            Select logarithmic liquid mixing or the dilute-gas Wilke rule.

        Returns
        -------
        float or ndarray
            Mixture viscosity [Pa*s], scalar for one composition or shape
            (num_points,) for a composition profile.

        Raises
        ------
        ValueError
            If ``phase`` is unknown, or required vapor ``visc_gas`` data
            [Pa*s] are absent or contain NaN entries for missing species.
        MissingPropertyError
            A subclass of ``AttributeError``.
            For the liquid, if a species with a nonzero mole fraction in some
            composition row has no ``visc_liq`` coefficients. A species whose
            fraction is zero in every row needs none and contributes exactly
            zero to the logarithmic mixing rule. Nonzero is bitwise: a
            round-off-level fraction counts as present.

        Notes
        -----
        Wilke, J. Chem. Phys. 18, 517 (1950), doi:10.1063/1.1747673;
        Poling, Prausnitz and O'Connell, The Properties of Gases and Liquids,
        5th ed., equations 9-5.13 and 9-5.14:
        ``mu = sum_i y_i*mu_i / sum_j y_j*phi_ij``, where
        ``phi_ij = (1 + sqrt(mu_i/mu_j)*(M_j/M_i)**0.25)**2
        / sqrt(8*(1 + M_i/M_j))``. The rule assumes a low-pressure gas;
        molar masses use the same units for every species.
        """
        if phase == 'liquid':
            visc_cts = self._property_rows('visc_liq')  # [-], [K], [1/K], [1/K**2]
            visc_comp = self._evaluate_liquid_viscosity(temp, visc_cts)  # [Pa*s]
        else:
            visc_comp = self.getViscosityPure(phase, temp)  # [Pa*s]

        if mass_frac is None and mole_frac is None:
            mole_frac = self.mole_frac
        elif mole_frac is None:
            mole_frac = self.frac_to_frac(mass_frac=mass_frac)

        # Mixing rules
        if phase == 'liquid':
            # Absent species without coefficients contribute exactly zero.
            needed = np.atleast_2d(np.asarray(mole_frac) != 0).any(axis=0)
            missing = self._check_species_rows(
                'visc_liq', visc_cts, np.arange(len(visc_cts)), needed,
                'carry a nonzero fraction, so the mixture viscosity')
            log_visc = np.where(missing, 0.0, np.log(visc_comp))  # [-], ln(mu/[Pa*s])
            viscMix = np.exp(
                np.dot(mole_frac, log_visc))

        elif phase == 'vapor':
            if visc_comp.ndim == 1:
                visc_term = np.outer(visc_comp, 1/visc_comp)  # [-]
                mw_term = np.outer(self.mw, 1/self.mw)  # [-], M_i/M_j

                phi_mix = (1 + visc_term**0.5 * (1/mw_term)**0.25)**2 / \
                    np.sqrt(8*(1 + mw_term))  # [-]

                interactions = np.dot(mole_frac, phi_mix.T)  # [-]

                viscMix = (visc_comp * mole_frac / interactions).sum(axis=-1)  # [Pa*s]

        return viscMix

    def getDiffusivityPure(self, wrt: int,
                           temp: Optional[float] = None) -> np.ndarray:
        """Return species diffusivities with respect to one species.

        Parameters
        ----------
        wrt : int
            Index of the reference species, usually the solvent; selects a
            column of the ``diffusivity`` rows.
        temp : float, optional
            Temperature [K]; unused, the stored values are constants.

        Returns
        -------
        ndarray
            Diffusivity of every species in ``wrt`` [m**2/s], shape
            ``(num_species,)``.

        Raises
        ------
        MissingPropertyError
            A subclass of ``AttributeError``.
            If any species has no ``diffusivity`` data in column ``wrt``;
            every species is returned, so every species needs it.
        """
        rows = self._property_rows('diffusivity')  # [m**2/s]
        if hasattr(self, 'diffusivity'):
            rows = rows[:, [wrt]]  # [m**2/s], the values returned below

        num_sp = len(rows)
        self._check_species_rows(
            'diffusivity', rows, np.arange(num_sp),
            np.ones(num_sp, dtype=bool),
            'are evaluated as pure components, so their diffusivity')

        diffusivity = self.diffusivity[:, wrt]  # [m**2/s]
        return diffusivity

    def frac_to_conc(self, mass_frac=None, mole_frac=None, basis='mole'):
        densMass, densMole = self.getDensityPure()
        if mass_frac is None:
            if mole_frac.ndim == 1:
                concentr = mole_frac / np.dot(mole_frac, 1 / densMole)
            else:
                concentr = mole_frac.T / np.dot(mole_frac, 1 / densMole)
                concentr = concentr.T
        else:
            if mass_frac.ndim == 1:
                concentr = (mass_frac / self.mw) / np.dot(mass_frac,
                                                          1 / densMass)
            else:
                concentr = (mass_frac / self.mw).T / np.dot(mass_frac,
                                                            1 / densMass)
                concentr = concentr.T

        if basis == 'mass':
            concentr *= self.mw

        return concentr  # mol/L (kmol/m**3) - kg/m**3

    def frac_to_frac(self, mass_frac=None, mole_frac=None, ind=None):
        if mole_frac is not None:
            if mole_frac.ndim == 1:
                mass_frac = mole_frac * self.mw / np.dot(mole_frac, self.mw)
                frac_out = mass_frac
            else:
                mass_frac = (mole_frac * self.mw).T / np.dot(mole_frac, self.mw)
                frac_out = mass_frac.T
        else:
            if mass_frac.ndim == 1:
                mole_frac = (mass_frac / self.mw) / np.dot(mass_frac, 1/self.mw)
                frac_out = mole_frac
            else:
                mole_frac = (mass_frac / self.mw).T / np.dot(mass_frac, 1/self.mw)
                frac_out = mole_frac.T

        return frac_out

    def conc_to_frac(self, conc, solvent_ind=None, basis=None):
        """Convert liquid molar concentrations into composition fractions.

        Parameters
        ----------
        conc : array-like
            Species molar concentrations [mol/L]. Shape ``(num_species,)``
            is always accepted; ``(num_points, num_species)`` is accepted
            only when ``solvent_ind`` is ``None``. When ``solvent_ind`` is
            given, the value at that index is ignored on input and replaced
            by the concentration that closes the mixture volume balance, and
            that back-calculation supports one-dimensional input only.
        solvent_ind : int, optional
            Positional index of the solvent species [-]. Index ``0`` is a
            valid solvent index, so this argument is compared with ``None``
            and never tested for truth. When ``None``, ``conc`` is taken as a
            complete composition and no solvent concentration is
            back-calculated.
        basis : {'mole', 'mass', None}, optional
            Basis of the returned fractions. ``'mole'`` returns mole
            fractions only, ``'mass'`` returns mass fractions only, and the
            default ``None`` returns both.

        Returns
        -------
        mass_frac : numpy.ndarray
            Species mass fractions [-]. Returned when ``basis`` is ``'mass'``
            or ``None``.
        mole_frac : numpy.ndarray
            Species mole fractions [-]. Returned when ``basis`` is ``'mole'``
            or ``None``.
        conc : numpy.ndarray
            Species molar concentrations with the solvent entry filled in
            [mol/L]. Appended to the returned tuple only when ``solvent_ind``
            is not ``None``.

        Raises
        ------
        IndexError
            If ``conc`` is two-dimensional and ``solvent_ind`` is not
            ``None``. The solvent mask is built with ``numpy.ones_like``, so
            for a two-dimensional argument it selects along the point axis
            instead of the species axis.

        Notes
        -----
        The solvent concentration closes the ideal-mixture volume balance
        ``sum_i c_i * v_i = 1``, in which ``v_i = 1 / rho_i`` is the
        pure-species molar volume [L/mol] taken from
        :meth:`getDensityPure`::

            c_solv = (1 - sum_{i != solv} c_i * v_i) / v_solv

        The length of the returned tuple therefore depends on whether
        ``solvent_ind`` was supplied, not on its value, so callers must
        branch on ``solvent_ind is not None``.

        Warnings
        --------
        ``conc`` is wrapped with ``numpy.asarray`` and the solvent entry is
        written in place, so a NumPy array supplied by the caller is modified
        by this call. A list or tuple is copied by ``asarray`` and is left
        untouched.
        """
        conc = np.asarray(conc)

        if solvent_ind is not None:
            _, densMole = self.getDensityPure(phase='liquid')
            molVol = 1 / densMole  # mol/L, kmol/m3

            mask_solv = np.ones_like(conc, dtype=bool)
            mask_solv[solvent_ind] = False

            conc_solv = (1 - np.dot(conc[mask_solv], molVol[mask_solv])) / \
                molVol[solvent_ind]

            conc[solvent_ind] = conc_solv

        if conc.ndim == 1:
            mole_frac = conc / conc.sum()
        else:
            mole_frac = (conc.T / conc.sum(axis=1)).T

        if basis == 'mole':
            if solvent_ind is not None:
                return mole_frac, conc
            else:
                return mole_frac

        elif basis == 'mass':
            mass_frac = self.frac_to_frac(mole_frac=mole_frac)
            if solvent_ind is not None:
                return mass_frac, conc
            else:
                return mass_frac
        else:
            mass_frac = self.frac_to_frac(mole_frac=mole_frac)
            if solvent_ind is not None:
                return mass_frac, mole_frac, conc
            else:
                return mass_frac, mole_frac

    def mass_conc_to_frac(self, conc, solvent_ind=None, basis=None):
        """Convert liquid mass concentrations into composition fractions.

        Parameters
        ----------
        conc : array-like
            Species mass concentrations [kg/m**3]. Shape ``(num_species,)``
            is always accepted; ``(num_points, num_species)`` is accepted
            only when ``solvent_ind`` is ``None``. When ``solvent_ind`` is
            given, the value at that index is ignored on input and replaced
            by the concentration that closes the mixture volume balance, and
            that back-calculation supports one-dimensional input only.
        solvent_ind : int, optional
            Positional index of the solvent species [-]. Index ``0`` is a
            valid solvent index, so this argument is compared with ``None``
            and never tested for truth. When ``None``, ``conc`` is taken as a
            complete composition and no solvent concentration is
            back-calculated.
        basis : {'mass', 'mole', None}, optional
            Basis of the returned fractions. ``'mass'`` returns mass
            fractions only, ``'mole'`` returns mole fractions only, and the
            default ``None`` returns both.

        Returns
        -------
        mass_frac : numpy.ndarray
            Species mass fractions [-]. Returned when ``basis`` is ``'mass'``
            or ``None``.
        mole_frac : numpy.ndarray
            Species mole fractions [-]. Returned when ``basis`` is ``'mole'``
            or ``None``.
        conc : numpy.ndarray
            Species mass concentrations with the solvent entry filled in
            [kg/m**3]. Appended to the returned tuple only when
            ``solvent_ind`` is not ``None``.

        Raises
        ------
        IndexError
            If ``conc`` is two-dimensional and ``solvent_ind`` is not
            ``None``, for the same reason as in :meth:`conc_to_frac`.

        Notes
        -----
        The solvent concentration closes the ideal-mixture volume balance
        ``sum_i c_i / rho_i = 1``, in which ``rho_i`` is the pure-species mass
        density [kg/m**3] taken from :meth:`getDensityPure`::

            c_solv = (1 - sum_{i != solv} c_i / rho_i) * rho_solv

        This is the mass-basis counterpart of :meth:`conc_to_frac`, and the
        length of the returned tuple likewise depends on whether
        ``solvent_ind`` was supplied, not on its value.

        Warnings
        --------
        ``conc`` is wrapped with ``numpy.asarray`` and the solvent entry is
        written in place, so a NumPy array supplied by the caller is modified
        by this call. A list or tuple is copied by ``asarray`` and is left
        untouched.
        """
        conc = np.asarray(conc)

        if solvent_ind is not None:
            dens_mass, _ = self.getDensityPure(phase='liquid')
            mask_solv = np.ones_like(conc, dtype=bool)
            mask_solv[solvent_ind] = False

            conc_solv = (
                1 - np.dot(conc[mask_solv], 1/dens_mass[mask_solv])) * \
                dens_mass[solvent_ind]

            conc[solvent_ind] = conc_solv

        if conc.ndim == 1:
            mass_frac = conc / conc.sum()
        else:
            mass_frac = (conc.T / conc.sum(axis=1)).T

        if basis == 'mass':
            if solvent_ind is not None:
                return mass_frac, conc
            else:
                return mass_frac

        elif basis == 'mole':
            mole_frac = self.frac_to_frac(mass_frac=mass_frac)
            if solvent_ind is not None:
                return mole_frac, conc
            else:
                return mole_frac

        else:
            mole_frac = self.frac_to_frac(mass_frac=mass_frac)
            if solvent_ind is not None:
                return mass_frac, mole_frac, conc
            else:
                return mass_frac, mole_frac

    def conc_to_conc(self, mass_conc=None, mole_conc=None):
        if mass_conc is not None:
            conv_conc = mass_conc / self.mw  # mol / L
        else:
            conv_conc = mole_conc * self.mw  # kg / m3

        return conv_conc

    def getMolVolMix(self, frac):
        molvolMix = np.dot(self.mol_vol, frac)

        return molvolMix

    def AntoineEquation(self, temp: Optional[Union[float, np.ndarray]] = None,
                        pres: Optional[Union[float, np.ndarray]] = None,
                        idx: Optional[Sequence[int]] = None) -> np.ndarray:
        """Evaluate the Antoine equation for saturation pressure or temperature.

        Parameters
        ----------
        temp : float or ndarray, optional
            Temperature [K], used when ``pres`` is ``None``. An array of any
            shape gains a trailing species axis.
        pres : float or ndarray, optional
            Pressure [Pa]. When given, the saturation temperature is returned
            instead; an array gains a trailing species axis.
        idx : sequence of int, optional
            Species indices to evaluate, in the requested order, as a list,
            tuple or one-dimensional integer array; all species when
            ``None``.

        Returns
        -------
        ndarray
            Saturation pressure [Pa] at ``temp``, or saturation temperature
            [K] at ``pres``, with the selected species on the last axis.

        Raises
        ------
        MissingPropertyError
            A subclass of ``AttributeError``.
            If a selected species has no ``p_vap`` coefficients; every
            selected species is returned, so every one needs them.
        TypeError
            If ``idx`` does not hold integers.

        Notes
        -----
        ``log10(p / [Pa]) = A - B / (T + C)`` with ``p_vap = [A, B, C]`` in
        [-], [K] and [K].
        """
        p_vap_cts = self._property_rows('p_vap')  # [-], [K], [K]
        species_idx = np.arange(len(p_vap_cts))
        if idx is not None:
            # A tuple would otherwise index several array axes.
            idx = np.asarray(idx).ravel()
            if idx.size == 0:
                idx = idx.astype(np.intp)
            elif idx.dtype.kind not in 'iu':
                raise TypeError(
                    f"idx must hold integer species indices, got {idx!r}")

            p_vap_cts = p_vap_cts[idx]  # [-], [K], [K]
            species_idx = species_idx[idx]

        self._check_species_rows(
            'p_vap', p_vap_cts, species_idx, np.ones(len(p_vap_cts), dtype=bool),
            'are evaluated as pure components, so their saturation pressure '
            'or temperature')

        return self._evaluate_antoine(p_vap_cts, temp, pres)

    def _evaluate_antoine(self, p_vap_cts: np.ndarray,
                          temp: Optional[Union[float, np.ndarray]] = None,
                          pres: Optional[Union[float, np.ndarray]] = None
                          ) -> np.ndarray:
        """Evaluate the Antoine equation without checking for data.

        Parameters
        ----------
        p_vap_cts : ndarray
            Antoine coefficients ``A`` [-], ``B`` [K], ``C`` [K] of the
            evaluated species, shape ``(num_selected, 3)``.
        temp, pres : float or ndarray, optional
            Temperature [K] or pressure [Pa], as in ``AntoineEquation``.

        Returns
        -------
        ndarray
            Saturation pressure [Pa] or temperature [K], species on the last
            axis; NaN for a species with a NaN row.
        """
        a_ct, b_ct, c_ct = p_vap_cts.T

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

    def _antoine_seed(self, mole_frac: np.ndarray,
                      temp: Optional[float] = None,
                      pres: Optional[float] = None) -> float:
        """Weight Antoine saturation values into a bubble or dew-point seed.

        Parameters
        ----------
        mole_frac : ndarray
            Mole fractions [-], shape ``(num_species,)``.
        temp : float, optional
            Temperature [K]; seeds a bubble pressure when ``pres`` is None.
        pres : float, optional
            Pressure [Pa]; seeds a bubble or dew temperature.

        Returns
        -------
        float
            ``sum_i x_i * psat_i`` [Pa] or ``sum_i x_i * Tsat_i`` [K].

        Notes
        -----
        Initial guess only; it matches the pre-#414 seed so convergence
        paths are unchanged. A species without ``p_vap`` data is evaluated
        with zero Antoine coefficients, exactly as the zero rows of earlier
        releases were: ``Tsat = -0`` K and ``psat = 1`` Pa. A seed is not a
        result; the converged root is checked with ``getKeqVLE``.
        """
        p_vap_cts = self._legacy_antoine_rows()  # [-], [K], [K]
        saturation = self._evaluate_antoine(p_vap_cts, temp, pres)  # [Pa] or [K]

        return np.dot(mole_frac, saturation)

    def _legacy_antoine_rows(self) -> np.ndarray:
        """Return Antoine rows with missing rows replaced by zeros.

        Returns
        -------
        ndarray
            Antoine ``A`` [-], ``B`` [K], ``C`` [K], shape
            ``(num_species, 3)``. A species without ``p_vap`` data gets zero
            coefficients, the rows of releases before #414, which evaluate to
            ``psat = 1`` Pa. Use them only for solver iterates and seeds and
            for zero-weight terms, never for a returned property of a needed
            species.
        """
        p_vap_cts = self._property_rows('p_vap')  # [-], [K], [K]
        return np.where(np.isnan(p_vap_cts), 0.0, p_vap_cts)

    def getKeqVLE(self, temp: Optional[Union[float, np.ndarray]] = None,
                  pres: Optional[float] = None,
                  x_liq: Optional[np.ndarray] = None,
                  y_vap: Optional[np.ndarray] = None,
                  gamma_model: str = 'ideal') -> np.ndarray:
        """Return vapor-liquid equilibrium ratios on a molar basis.

        Parameters
        ----------
        temp : float or ndarray, optional
            Temperature [K], scalar or shape (num_points,); defaults to
            the phase temperature.
        pres : float, optional
            Total pressure [Pa]; defaults to the phase pressure.
        x_liq : ndarray, optional
            Liquid mole fractions [-], shape (num_species,) or
            (num_points, num_species); defaults to the phase composition.
        y_vap : ndarray, optional
            Vapor mole fractions [-]. Retained for API compatibility; unused
            by the current ideal-vapor model.
        gamma_model : {'ideal', 'UNIFAC', 'UNIQUAC'}, optional
            Liquid activity model; its required parameters must be supplied
            in the property database.

        Returns
        -------
        ndarray
            Ratios ``y_i/x_i`` [-], with species on the last axis. A scalar
            temperature and composition vector give (num_species,); a
            temperature vector gives (num_points, num_species), including
            a one-row vector.

        Raises
        ------
        ValueError
            If ``gamma_model`` is not a supported, case-sensitive selector.
        MissingPropertyError
            A subclass of ``AttributeError``.
            If a species with a nonzero ``x_liq`` entry (in some row) that is
            at or below its critical temperature at some requested
            temperature, or has no ``t_crit``, lacks ``p_vap`` coefficients.
            A species above ``t_crit`` at every requested temperature uses
            its Henry constant and needs none.

        Notes
        -----
        ``K_i = gamma_i*p_i/pres``. For each temperature/species pair with
        ``temp > t_crit_i``, ``p_i`` is the Henry constant [Pa] on a liquid
        mole-fraction basis; otherwise it is Antoine saturation pressure [Pa].
        At the critical temperature itself, the Antoine branch is retained.
        The vector-temperature branch is validated for ``gamma_model='ideal'``.
        Non-ideal models require paired two-dimensional ``x_liq`` with shape
        (num_points, num_species). UNIQUAC with a temperature vector and a
        one-dimensional composition mis-indexes the temperature profile; this
        pre-existing limitation remains a follow-up, not a supported contract.

        ``x_liq`` is the composition the caller weights the ratios by, such
        as ``y = K * x`` or ``sum x * K``; a dew-point caller passes the
        vapor composition. A species whose entry is zero in every row needs
        no Antoine data: without data its ratio is computed from zero
        coefficients (``p = 1`` Pa), the value of releases before #414. That
        finite placeholder keeps zero-weight terms, including ``y / K``,
        exactly zero; it is not a vapor pressure. Nonzero is bitwise: a
        round-off-level fraction counts as present. A fraction integrated by
        a solver can carry round-off where it should be zero, so a unit
        operation other than a reactor needs data for every species it
        carries as a state.
        """
        return self._vle_ratios(temp, pres, x_liq, gamma_model, check=True)

    def _vle_ratios(self, temp: Optional[Union[float, np.ndarray]],
                    pres: Optional[float], x_liq: Optional[np.ndarray],
                    gamma_model: str, check: bool) -> np.ndarray:
        """Evaluate VLE ratios, optionally checking the Antoine data.

        Parameters
        ----------
        temp, pres, x_liq, gamma_model
            As in ``getKeqVLE``: temperature [K], pressure [Pa], weighting
            mole fractions [-] and activity model.
        check : bool
            If True, apply the missing-data rule of ``getKeqVLE``. If False,
            every species without Antoine data uses the zero-coefficient
            placeholder; for solver iterates only, whose converged root the
            caller then checks with ``getKeqVLE``.

        Returns
        -------
        ndarray
            Ratios ``y_i/x_i`` [-], shaped as in ``getKeqVLE``.

        Raises
        ------
        ValueError
            If ``gamma_model`` is not a supported selector.
        MissingPropertyError
            A subclass of ``AttributeError``.
            With ``check``, as in ``getKeqVLE``.
        """
        validate_activity_model(gamma_model)

        if temp is None:
            temp = self.temp

        if pres is None:
            pres = self.pres

        if x_liq is None:
            x_liq = self.mole_frac

        p_vap_cts = self._legacy_antoine_rows()  # [-], [K], [K]
        crit = isinstance(temp, np.ndarray) and temp.ndim == 1
        if crit:
            p_vap = self._evaluate_antoine(p_vap_cts, temp)  # [Pa]
            supercrit = temp[:, np.newaxis] > self.t_crit
            if np.any(supercrit):
                p_vap = np.where(supercrit, self.henry_constant, p_vap)  # [Pa]
        else:
            supercrit = temp > self.t_crit
            p_vap = self._evaluate_antoine(p_vap_cts, temp)  # [Pa]
            if any(supercrit):
                p_vap[supercrit] = self.henry_constant[supercrit]

        if check:
            # Antoine data are needed by weighted species at or below t_crit.
            num_sp = len(p_vap_cts)
            needed = (np.atleast_2d(~supercrit).any(axis=0)
                      & self._needed_by_weights(x_liq, num_sp))
            self._check_species_rows(
                'p_vap', self._property_rows('p_vap'), np.arange(num_sp),
                needed,
                'carry a nonzero fraction and are at or below t_crit (or '
                'have no t_crit) at a requested temperature, so their VLE '
                'ratio')

        if gamma_model == 'ideal':
            gamma = np.ones_like(x_liq)
        elif gamma_model == 'UNIFAC':
            gamma = self.UNIFAC_DMD(x_liq, temp)
        elif gamma_model == 'UNIQUAC':
            gamma = self.UNIQUAC(x_liq, temp)

        k_vals = p_vap * gamma / pres

        return k_vals

    def UNIQUAC(self, mole_frac: Optional[np.ndarray] = None,
                temp: Optional[Union[float, np.ndarray]] = None) -> np.ndarray:
        r""" Calculate activity coefficients :math:`\gamma_i` using UNIQUAC model.

        Parameters
        ------------
        mole_frac : ndarray
            Liquid mole fractions [-], shape (num_species,) or
            (num_points, num_species). Required by the model.
        temp : float or ndarray
            Temperature [K], scalar for one composition or a paired
            (num_points,) profile. Required by the model.
        amk : ndarray
            n x n array containing interaction parameters [J/mol]:
                component m (row) respect to component k (column):

            .. math::

               \begin{bmatrix}
                   a_{11}       & a_{12}          & \cdots    & a_{1 n_{comp}} \\
                   a_{21}       & a_{22}          & \cdots    & a_{2 n_{comp}} \\
                   \vdots       & \vdots          & \ddots    & \vdots \\
                   a_{n_{comp}} & a_{n_{comp} 2}  & \cdots    & a_{n_{comp} n_{comp}}
               \end{bmatrix}

        ri : ndarray
            Database molecular volume constants [-], shape (num_species,).
        qi : ndarray
            Database molecular surface area constants [-], shape (num_species,).
        qip : ndarray
            Database molecular surface area constants [-] for systems with
            water or alcohols, shape (num_species,). If absent, use ``qi`` locally
            and warn once per instance. This inherited fallback is an explicit
            assumption and may be unsuitable for water/alcohol mixtures.

        Returns
        -----------
        output : ndarray
            Activity coefficients [-], with the same shape as ``mole_frac``.

        Notes
        ---------------------------
        UNIQUAC model assumes that activity coefficient is the sum of a
        combinatorial and a residual part [1], [2], [3]:

        .. math::

            ln\gamma_i = ln\gamma_i^{comb} + ln\gamma_i^{res}, \, i = 1, \cdots, n

        with

        .. math::

            ln\gamma_i^{comb} = ln\frac{\phi_i (r_i, x_i)}{x_i} +
                                5q_i ln\frac{\theta_i(q_i, x_i)}{\phi_i} +
                                l_i (r_i, q_i) -
                                \frac{\phi_i}{x_i} \sum_j x_j l_j

        and

        .. math::

            ln\gamma_i^{res} =  q'_i \left[ 1 - ln \left( \sum_j \theta'_j (q'_j, x_j) \cdot \tau_{ji}(a_{ji}, T) \right) +
                                \sum_j \frac{\theta'_j\tau_{ij}}{\sum_k \theta'_k \tau_{kj}} \right]

        Temperature dependency is included in the term :math:`\tau_{ij}`:

        .. math::
            \tau_{ij} = exp \left[ \frac{-a_{mk}}{R T} \right]

        where :math:`a_{mk}` are *interaction* coefficients obtained by fitting
        experimental data.


        References
        ----------------
        [1] J. M. Smith, H. Van Ness and M. Abbott, Introduction to Chemical
        Engineering Thermodynamics, McGraw Hill, 7th Ed., 2004.

        [2] J. M. Prausnitz, R. N. Lichtenthaler and E. Gomes de Acevedo, Molecular
        thermodynamics of fluid-phase equilibria, Prentice Hall, New Jersey, 1999.

        [3] G. Kontogeorgis and G. Folas, Thermodynamic Models for Industrial
        Applications, John Willey & Sons, Inc., West Sussex, First Edit., 2010.

        """

        if not hasattr(self, 'qip') and not getattr(self, '_warned_qip_fallback', False):
            warnings.warn(
                "UNIQUAC qip is absent; assuming qip = qi. This fallback may "
                "be unsuitable for water or alcohol mixtures; supply validated "
                "qip values for those systems.", UserWarning, stacklevel=2)
            self._warned_qip_fallback = True

        # Rename
        ri = self.ri
        qi = self.qi
        qip = getattr(self, 'qip', self.qi)  # [-], local fallback leaves database fields unchanged
        amk = self.amk

        x_liq = mole_frac

        temp = np.asarray(temp)
        one_dimensional_output = False
    #    temp = temp[np.newaxis]

        if x_liq.ndim == 1:
            one_dimensional_output = True
            x_liq = x_liq[np.newaxis, :]
            temp = temp[np.newaxis]

        num_gammas, num_components = x_liq.shape

        # # Avoid log of zero
        # x_liq[x_liq == 0] = 1e-15

        gas_constant = 8.314  # J/mol/K

        # Avoid two-dimensional arrays
        ri = ri.ravel()
        qi = qi.ravel()
        qip = qip.ravel()

        # -------------------- Preliminary calculations
        tau = []
        for tp in temp:
            tau.append(np.exp(-amk / gas_constant / tp))

        phi = ri / (ri * x_liq).sum(axis=1)[:, np.newaxis]
        theta = qi / (qi * x_liq).sum(axis=1)[:, np.newaxis]

        li = 5*(ri - qi) - (ri - 1)

        # -------------------- Combinatorial term (same as non-modified UNIFAC)
        gamma_combinatorial = np.log(phi) + 5 * qi * np.log(theta / phi) +\
            li - phi * (li * x_liq).sum(axis=1)[:, np.newaxis]

        # -------------------- Residual term, with qip instead of qi
        theta_p = x_liq * qip / (qip * x_liq).sum(axis=1)[:, np.newaxis]

        tau_theta = np.zeros((num_gammas, num_components))
        for ind, row in enumerate(theta_p):
            tau_theta[ind] = (tau[ind].T * row).sum(axis=1)

        tau_theta_residual = np.zeros_like(tau_theta)
        for ind, row in enumerate(theta_p):
            tau_theta_residual[ind] = (tau[ind] * row / tau_theta[ind]).sum(axis=1)

        gamma_residual = qip*(1 - np.log(tau_theta) - tau_theta_residual)

        # -------------------- Unify terms
        log_gamma = gamma_residual + gamma_combinatorial

        gamma = np.exp(log_gamma)

        if one_dimensional_output:
            return np.squeeze(gamma)
        else:
            return gamma

    def get_UNIFACParams(self, dataframes=False):
        """ Get UNIFAC-Dortmund constants by specifying the indexes of the groups
            present in the mixture

            Parameters
            ----------
            group_idx : list of tuples
                list containing 2-tuples, which describe the main and secondary
                group index (without repetition) for each group in the mixture.

            Returns
            -------

            Example
            -------
                For a water ethanol mixture, group_idx would be:

                    >>> water_ethanol = [(7, 16), (1, 1), (1, 2), (5, 14)]

                The first tuple (7, 16) corresponds to water, and the next three
                conform ethanol: [CH3, CH2 and OH(p)]. For information on the
                group numbers, refer to [1]

            References
            ----------
            [1] Gmehling, J.; Li, J.; Schiller, M. Ind. Eng. Chem. Res. 1993,
            32 (1), 178–193.

        """
        group_tuples = self.unifac_groups
        group_idx = np.array(self.unifac_groups)

        # Import data
        root = str(pathlib.Path(__file__).parents[1]) + '/data/thermodynamics/'
        interac_path = root + 'unifac_interaction_params.csv'

        interac_data = pd.read_csv(interac_path, index_col=(0, 1))

        rq_path = root + 'unifac_rk_qk.csv'
        rk_qk = pd.read_csv(rq_path, index_col=(2, 0))

        # Create empty arrays
        num_groups = len(group_idx)
        a_matrix = np.zeros((num_groups, num_groups))
        b_matrix = np.zeros_like(a_matrix)
        c_matrix = np.zeros_like(a_matrix)

        main = group_idx[:, 0]

        # Create a grid with coordinates
        j_coord, i_coord = np.meshgrid(main, main)

        for m in range(num_groups):
            for n in range(num_groups):
                i, j = (i_coord[m, n], j_coord[m, n])

                if i == j:
                    pass
                elif i < j:
                    a_matrix[m, n] = interac_data.loc[i, j]['anm']
                    b_matrix[m, n] = interac_data.loc[i, j]['bnm']
                    c_matrix[m, n] = interac_data.loc[i, j]['cnm']
                else:
                    a_matrix[m, n] = interac_data.loc[j, i]['amn']
                    b_matrix[m, n] = interac_data.loc[j, i]['bmn']
                    c_matrix[m, n] = interac_data.loc[j, i]['cmn']

        r_k = []
        q_k = []
        for ind in group_tuples:
            r_k.append(rk_qk['Rk'].xs(ind))
            q_k.append(rk_qk['Qk'].xs(ind))

        r_k = np.array(r_k)
        q_k = np.array(q_k)

        if dataframes:
            a_matrix = pd.DataFrame(a_matrix, index=group_tuples,
                                    columns=group_tuples)
            b_matrix = pd.DataFrame(b_matrix, index=group_tuples,
                                    columns=group_tuples)
            c_matrix = pd.DataFrame(c_matrix, index=group_tuples,
                                    columns=group_tuples)

            r_k = pd.Series(r_k, index=group_tuples)
            q_k = pd.Series(q_k, index=group_tuples)

        return [r_k, q_k, a_matrix, b_matrix, c_matrix]

    def UNIFAC_DMD(self, x_i=None, temp=None):

        """ Calculate activity coefficients of a liquid mixture using the UNIFAC
        group-contribution method

        Parameters
        ----------
        x_i : array
            liquid molar fractions
        temp : float or array
            temperature (K)
        r_k : array
            surface area parameter for constituent groups 1,..., k,..., K
        q_k : array
            volume parameters for constituent groups 1,..., k,..., K
        a_inter : array
            K x K array with the interaction parameter between gropus m and n
            at the (m, n) position. Note that a_inter[k, k] = 0
        v_matrix : array
            N x K array. Each row contains the number of occurrences of the group k
            in the molecule i (i = 1,...,N)
        dmd : bool (default: False)
            if True, use the Dortmund variation of UNIFAC [1], which implies that
            b_inter and c_inter cannot be None (N x K arrays like a_inter)

        Returns
        -------

        gamma : array
            activity coefficients for the mixture(s). It has the same size as x_i

        Notes
        -----
        * This implementation is based on [2] for pure UNIFAC (dmd=False). The
          implemented thermodynamic models (pure UNIFAC and UNIFAC-Dortmund)
          are described in detail in [1] and [2]
        * If x_i is multi-dimensional with shape P x N, then temp must be a
          P-sized, 1-D array

        References
        ----------
        [1] (1) Gmehling, J.; Li, J.; Schiller, M. Ind. Eng. Chem. Res. 1993,
        32 (1), 178-193.

        [2] Fredenslund, A.; Jones, R. L.; Prausnitz, J. M. AIChE J. 1975, 21 (6),
        1086-1099.

        """

        if x_i is None:
            x_i = self.mole_frac

        if temp is None:
            temp = self.temp

        def get_gamma_log(psi_matrix, theta_vec, q_vec):
            psi_divided = psi_matrix / np.dot(psi_matrix.T, theta_vec)

            log_gamma = q_vec * (1 - np.log(np.dot(psi_matrix.T, theta_vec)) -
                                 np.dot(psi_divided, theta_vec))

            return log_gamma

        a_inter = self.a_unifac
        b_inter = self.b_unifac
        c_inter = self.c_unifac
        r_k = self.Rk
        q_k = self.Qk

        v_matrix = self.vk

        num_groups = len(a_inter)

        x_i = np.asarray(x_i)
        temp = np.asarray(temp)
        one_dimensional_output = False

        if x_i.ndim == 1:
            one_dimensional_output = True
            x_i = x_i[np.newaxis, :]
            temp = temp[np.newaxis]

        num_comp = x_i.shape[1]

        # --------------- Combinatorial term
        r_i = np.dot(v_matrix, r_k)
        q_i = np.dot(v_matrix, q_k)

        V = r_i / (r_i * x_i).sum(axis=1)[:, np.newaxis]
        V_prime = r_i**(3/4) / (x_i * r_i**(3/4)).sum(axis=1)[:, np.newaxis]
        F = q_i / (q_i * x_i).sum(axis=1)[:, np.newaxis]

        g_comb = 1 - V_prime + np.log(V_prime) - 5 * q_i * \
            (1 - V/F + np.log(V/F))

        # --------------- Residual term
        x_upper_i = v_matrix / v_matrix.sum(axis=1)[:, np.newaxis]
        theta_i = q_k * x_upper_i / np.dot(x_upper_i, q_k)[:, np.newaxis]

        g_resid = []
        for frac, tp in zip(x_i, temp):
            x_upper = np.dot(v_matrix.T, frac) / \
                np.sum(v_matrix * frac[:, np.newaxis])
            theta = q_k * x_upper / np.dot(q_k, x_upper)

            psi = np.exp(-(a_inter + b_inter * tp + c_inter * tp**2) / tp)

            ln_gamma = get_gamma_log(psi, theta, q_k)
            ln_gamma_i = np.zeros((num_comp, num_groups))

            for ind, row in enumerate(theta_i):
                ln_gamma_i[ind] = get_gamma_log(psi, row, q_k)

            g_res = v_matrix * (ln_gamma - ln_gamma_i)
            g_res = g_res.sum(axis=1)

            g_resid.append(g_res)

        g_resid = np.vstack(g_resid)
        g_total = g_comb + g_resid
        gamma = np.exp(g_total)

        if one_dimensional_output:
            return np.squeeze(gamma)
        else:
            return gamma
